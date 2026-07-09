#!/usr/bin/env python3
"""Evaluate Donut (naver-clova-ix/donut-base-finetuned-docvqa) on the same
300-sample subsets. Donut is a task-specific OCR-free encoder-decoder: no
chat template, no VLMEvalKit/vLLM support: so this drives it directly via HF,
but emits the SAME JSONL schema as inference/ece_latency.py so eval/analyze.py can
aggregate it (model name "Donut-DocVQA").

Caveats (documented in the report): the checkpoint is fine-tuned on DocVQA
train (in-domain for DocVQA leg); InfoVQA is out-of-domain for it. Greedy
decoding; batch 1 => every sample is single-stream latency.

Resumable: appends to JSONL, skips done indices.
  python inference/donut_eval.py [--limit 5]   # optional quick check on 5 samples
"""
import argparse
import json
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("LMUData", "/content/drive/MyDrive/vlm_eval/LMUData")

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL = "naver-clova-ix/donut-base-finetuned-docvqa"
NAME = "Donut-DocVQA"
OUT_DIR = Path("/content/drive/MyDrive/vlm_eval/results")
DATASETS = ["DocVQA_VAL", "InfoVQA_VAL"]

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

from vlmeval.dataset.image_vqa import ImageVQADataset
from PIL import Image

processor = DonutProcessor.from_pretrained(MODEL)
model = VisionEncoderDecoderModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
tok = processor.tokenizer

(OUT_DIR / NAME).mkdir(parents=True, exist_ok=True)
for ds_name in DATASETS:
    dataset = ImageVQADataset(dataset=ds_name)
    out_file = OUT_DIR / NAME / f"{ds_name}_ece.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(l)["index"] for l in open(out_file) if l.strip()}
    rows = [dataset.data.iloc[i] for i in range(len(dataset.data))]
    if args.limit:
        rows = rows[:args.limit]
    todo = [r for r in rows if str(r["index"]) not in done]
    print(f"[{ds_name}] {len(todo)} to run ({len(done)} done)")

    with open(out_file, "a") as fout:
        for i, line in enumerate(todo):
            msgs = dataset.build_prompt(line)  # dumps image to disk
            img_path = next(m["value"] for m in msgs if m["type"] == "image")
            img = Image.open(img_path).convert("RGB")
            task = f"<s_docvqa><s_question>{line['question']}</s_question><s_answer>"

            t0 = time.perf_counter()
            pix = processor(img, return_tensors="pt").pixel_values.half().cuda()
            dec_ids = tok(task, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                out = model.generate(
                    pix, decoder_input_ids=dec_ids, max_length=512,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                    bad_words_ids=[[tok.unk_token_id]], use_cache=True,
                    return_dict_in_generate=True, output_scores=True,
                )
            latency = time.perf_counter() - t0

            gen = out.sequences[0][dec_ids.shape[1]:]
            lps = []
            for t, sc in zip(gen, out.scores):
                lp = torch.log_softmax(sc[0].float(), dim=-1)[t].item()
                if t not in (tok.pad_token_id,):
                    lps.append(lp)
            text = tok.decode(gen, skip_special_tokens=False)
            text = text.replace(tok.eos_token, "").replace(tok.pad_token, "")
            pred = re.sub(r"<.*?>", "", text).strip()

            fout.write(json.dumps({
                "index": str(line["index"]), "answer": str(line["answer"]),
                "question": str(line["question"]), "prediction": pred,
                "latency_s": latency, "n_tokens": len(lps), "logprobs": lps,
                "conf_geo": math.exp(sum(lps) / len(lps)) if lps else None,
                "conf_mean": (sum(math.exp(x) for x in lps) / len(lps)) if lps else None,
                "finish_reason": "stop", "mode": "sequential",
            }) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(todo)}")
    print(f"[{ds_name}] done -> {out_file}")
