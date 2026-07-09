#!/usr/bin/env python3
"""Prompting-strategy eval: direct (non-CoT) vs chain-of-thought, same models,
same 300-sample subsets, same ANLS scoring path: so the only variable is the
prompt.

- direct : dataset.build_prompt(line) verbatim: the "Answer the question using a
           single word or phrase." prompt used by the main VLMEvalKit run. Parity
           with inference/full_eval_vllm.sh, so `direct` reproduces the headline numbers.
- cot    : same image + question, but instruct the model to reason step by step
           and end with `Answer: <short answer>`. The final answer is parsed back
           out for ANLS; confidence (for ECE) is taken over the answer-span tokens
           only, not the reasoning.

Assumes the model is already served on --port (see inference/run_prompting.sh).
Greedy decoding (temperature 0) in both modes. Resumable JSONL per
(model, dataset, mode) in vlm_eval/results/prompting/.

  python inference/prompt_eval.py --model Qwen3.5-0.8B \
      --data DocVQA_VAL_SUB300 InfoVQA_VAL_SUB300 --modes direct cot
"""
import argparse
import base64
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("LMUData", "/content/drive/MyDrive/vlm_eval/LMUData")
import requests

OUT_DIR = Path("/content/drive/MyDrive/vlm_eval/results/prompting")

COT_SUFFIX = (
    "\n\nReason step by step about what the document shows, then end your reply "
    "with the final answer on its own line, exactly in the form:\n"
    "Answer: <a single word or short phrase>"
)
ANS_RE = re.compile(r"(?i)\banswer\s*[:\-—]\s*")


def extract_answer(text):
    """Return (answer_string, char_offset_where_answer_starts_in_text)."""
    if not text:
        return "", 0
    ms = list(ANS_RE.finditer(text))
    if ms:
        start = ms[-1].end()
        ans = text[start:]
    else:  # model ignored the format: fall back to the last non-empty line
        lines = [l for l in text.strip().splitlines() if l.strip()]
        ans = lines[-1] if lines else text
        start = text.rfind(ans) if ans else 0
    ans = ans.strip().splitlines()[0].strip() if ans.strip() else ""
    ans = ans.strip().strip("*").strip().rstrip(".").strip()
    return ans, max(0, start)


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
    }, timeout=600)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    ch = r.json()["choices"][0]
    content = (ch.get("logprobs") or {}).get("content", []) or []
    return {"text": ch["message"]["content"], "latency_s": latency,
            "content": content, "finish_reason": ch.get("finish_reason")}


def geo(lps):
    return math.exp(sum(lps) / len(lps)) if lps else None


def build_messages(dataset, line, mode):
    if mode == "direct":
        return dataset.build_prompt(line)          # verbatim parity with main eval
    imgs = dataset.dump_image(line)                # CoT: same image, custom instruction
    if isinstance(imgs, str):
        imgs = [imgs]
    txt = str(line["question"]) + COT_SUFFIX
    return [{"type": "image", "value": p} for p in imgs] + [{"type": "text", "value": txt}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["direct", "cot"],
                    choices=["direct", "cot"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--latency-samples", type=int, default=30)
    ap.add_argument("--max-tokens-direct", type=int, default=256)
    ap.add_argument("--max-tokens-cot", type=int, default=768)
    args = ap.parse_args()

    # ImageVQADataset explicitly (build_dataset would route custom SUB300 names to
    # CustomVQADataset, dropping the standard prompt suffix: prompts must match).
    from vlmeval.dataset.image_vqa import ImageVQADataset
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()

    for ds_name in args.data:
        dataset = ImageVQADataset(dataset=ds_name)
        assert dataset is not None, f"dataset {ds_name} not found"
        lines = [dataset.data.iloc[i] for i in range(len(dataset.data))]

        for mode in args.modes:
            out_file = OUT_DIR / f"{args.model}_{ds_name}_{mode}.jsonl"
            done = set()
            if out_file.exists():                   # resume after a disconnect
                with open(out_file) as f:
                    done = {json.loads(l)["index"] for l in f if l.strip()}
            todo = [l for l in lines if str(l["index"]) not in done]
            print(f"[{ds_name}|{mode}] {len(todo)} to run ({len(done)} done)")
            fout = open(out_file, "a")
            max_tokens = args.max_tokens_cot if mode == "cot" else args.max_tokens_direct

            def run_one(line, sched):
                try:
                    msgs = build_messages(dataset, line, mode)
                    res = query(sess, url, args.model, to_openai_messages(msgs), max_tokens)
                except Exception as e:
                    rec = {"index": str(line["index"]), "mode": mode, "error": str(e)}
                    fout.write(json.dumps(rec) + "\n"); fout.flush()
                    return rec
                text, content = res["text"], res["content"]
                all_lps = [t["logprob"] for t in content]
                if mode == "cot":
                    pred, ans_start = extract_answer(text)
                    span, cur = [], 0
                    for t in content:               # confidence over answer-span tokens
                        seg = cur; cur += len(t.get("token", ""))
                        if seg >= ans_start:
                            span.append(t["logprob"])
                    conf = geo(span) if span else geo(all_lps)
                    n_ans = len(span)
                else:
                    pred = text.strip()
                    conf = geo(all_lps)
                    n_ans = len(all_lps)
                rec = {"index": str(line["index"]), "mode": mode,
                       "question": str(line["question"]), "answer": str(line["answer"]),
                       "prediction": pred, "raw": text if mode == "cot" else None,
                       "latency_s": res["latency_s"], "n_tokens": len(all_lps),
                       "n_ans_tokens": n_ans, "conf_geo": conf, "sched": sched,
                       "finish_reason": res["finish_reason"]}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                return rec

            n_seq = max(0, min(args.latency_samples - len(done), len(todo)))
            for i, line in enumerate(todo[:n_seq]):     # sequential slice = clean latency
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
            print(f"[{ds_name}|{mode}] done -> {out_file}")


if __name__ == "__main__":
    main()
