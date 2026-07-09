#!/usr/bin/env python3
"""Confidence (for ECE) + latency pass against a vLLM-served model.

Sends the same prompts VLMEvalKit uses (dataset.build_prompt) with logprobs=true,
greedy decoding. First --latency-samples queries run sequentially (clean
single-stream latency); the rest run with --concurrency workers (throughput).

Output: JSONL per (model, dataset) in vlm_eval/results/ece/ with per-sample
prediction, token logprobs, confidence aggregates, and wall latency.
ECE/ANLS aggregation happens in eval/analyze.py.

Usage (server already running on --port):
  python inference/ece_latency.py --model Qwen3.5-0.8B --data DocVQA_VAL_SUB300 InfoVQA_VAL_SUB300
"""
import argparse
import base64
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("LMUData", "/content/drive/MyDrive/vlm_eval/LMUData")
import requests

OUT_DIR = Path("/content/drive/MyDrive/vlm_eval/results/ece")


def to_openai_messages(msgs):
    content = []
    for m in msgs:
        if m["type"] == "image":
            with open(m["value"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        else:
            content.append({"type": "text", "text": m["value"]})
    return [{"role": "user", "content": content}]


def query(sess, url, model, messages, max_tokens):
    t0 = time.perf_counter()
    r = sess.post(url, json={
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
    }, timeout=300)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    out = r.json()
    choice = out["choices"][0]
    lps = [t["logprob"] for t in (choice.get("logprobs") or {}).get("content", [])]
    return {
        "prediction": choice["message"]["content"],
        "latency_s": latency,
        "n_tokens": len(lps),
        "logprobs": lps,
        "conf_geo": math.exp(sum(lps) / len(lps)) if lps else None,   # geometric-mean token prob
        "conf_mean": (sum(math.exp(x) for x in lps) / len(lps)) if lps else None,
        "finish_reason": choice.get("finish_reason"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--latency-samples", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    # ImageVQADataset explicitly (NOT build_dataset: unknown names fall back to
    # CustomVQADataset, which lacks the "single word or phrase" prompt suffix
    # the main eval used: prompts must match exactly)
    from vlmeval.dataset.image_vqa import ImageVQADataset
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()

    for ds_name in args.data:
        dataset = ImageVQADataset(dataset=ds_name)
        assert dataset is not None, f"dataset {ds_name} not found"
        out_file = OUT_DIR / f"{args.model}_{ds_name}.jsonl"
        done = set()
        if out_file.exists():  # resume after disconnect
            with open(out_file) as f:
                done = {json.loads(l)["index"] for l in f if l.strip()}
        lines = [dataset.data.iloc[i] for i in range(len(dataset.data))]
        todo = [l for l in lines if str(l["index"]) not in done]
        print(f"[{ds_name}] {len(todo)} to run ({len(done)} already done)")

        fout = open(out_file, "a")

        def run_one(line, mode):
            msgs = dataset.build_prompt(line)
            try:
                res = query(sess, url, args.model, to_openai_messages(msgs), args.max_tokens)
            except Exception as e:
                res = {"error": str(e)}
            res.update({"index": str(line["index"]), "answer": str(line["answer"]),
                        "question": str(line["question"]), "mode": mode})
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            return res

        n_seq = max(0, min(args.latency_samples - sum(1 for _ in done), len(todo)))
        for i, line in enumerate(todo[:n_seq]):  # sequential slice: clean latency
            run_one(line, "sequential")
            if (i + 1) % 10 == 0:
                print(f"  seq {i+1}/{n_seq}")
        rest = todo[n_seq:]
        if rest:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                for i, _ in enumerate(ex.map(lambda l: run_one(l, "concurrent"), rest)):
                    if (i + 1) % 50 == 0:
                        print(f"  conc {i+1}/{len(rest)}")
        fout.close()
        print(f"[{ds_name}] done -> {out_file}")


if __name__ == "__main__":
    main()
