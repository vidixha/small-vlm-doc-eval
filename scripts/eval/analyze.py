#!/usr/bin/env python3
"""Aggregate all results: ANLS (from VLMEvalKit acc.csv), ECE + latency (from
the inference/ece_latency.py JSONLs). Emits vlm_eval/results/summary.md + summary.csv.

ECE: confidence = geometric-mean token probability (conf_geo); correctness =
per-sample ANLS >= 0.5 (standard DocVQA threshold); 10 equal-width bins.
Latency: sequential-mode records only (single-stream); tokens/s = completion
tokens / latency.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LMUData", "/content/drive/MyDrive/vlm_eval/LMUData")
from vlmeval.dataset.utils.vqa_eval import hit_calculate, process_line

WORK = Path("/content/drive/MyDrive/eval_work")
ECE_DIR = Path("/content/drive/MyDrive/vlm_eval/results")
OUT = Path("/content/drive/MyDrive/vlm_eval/results")
MODELS = ["Qwen3.5-0.8B", "InternVL3-1B", "SmolVLM-500M", "Donut-DocVQA"]
DATASETS = ["DocVQA_VAL", "InfoVQA_VAL"]


def anls_of(answer, prediction):
    # process_line(anls) returns normalized Levenshtein DISTANCES per gt;
    # hit_calculate applies the ANLS threshold: 0 if (1-min_dist)<0.5 else 1-min_dist
    line = pd.Series({"answer": answer, "prediction": prediction})
    r = process_line(line, method="anls")
    return float(hit_calculate([r], "DocVQA")[0])


def ece_10bin(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return ece


rows = []
for model in MODELS:
    for ds in DATASETS:
        row = {"model": model, "dataset": ds}
        # 1) canonical ANLS from the main eval
        accs = sorted(glob.glob(str(WORK / model / f"{model}_{ds}_acc.csv")))
        if accs:
            row["anls_main"] = float(pd.read_csv(accs[-1])["Overall"].iloc[0])
        # 2) ECE + latency from the logprobs pass
        jf = ECE_DIR / model / f"{ds}_ece.jsonl"
        if jf.exists():
            recs = [json.loads(l) for l in open(jf) if l.strip()]
            ok = [r for r in recs if "error" not in r and r.get("conf_geo") is not None]
            per_anls = np.array([anls_of(r["answer"], r["prediction"]) for r in ok])
            conf = np.array([r["conf_geo"] for r in ok])
            correct = (per_anls >= 0.5).astype(float)
            row.update({
                "n_ece": len(ok),
                "anls_ece_pass": round(float(per_anls.mean()) * 100, 2),
                "acc@0.5": round(float(correct.mean()) * 100, 2),
                "mean_conf": round(float(conf.mean()) * 100, 2),
                "ECE": round(ece_10bin(conf, correct) * 100, 2),
                "overconfidence": round(float(conf.mean() - correct.mean()) * 100, 2),
            })
            seq = [r for r in ok if r.get("mode") == "sequential"]
            if seq:
                lat = np.array([r["latency_s"] for r in seq])
                tps = np.array([r["n_tokens"] / r["latency_s"] for r in seq if r["latency_s"] > 0])
                row.update({
                    "lat_mean_s": round(float(lat.mean()), 2),
                    "lat_p50_s": round(float(np.median(lat)), 2),
                    "tok_per_s": round(float(tps.mean()), 1),
                })
        rows.append(row)

df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "summary.csv", index=False)

md = ["# Sub-1B VLM Document Understanding: Results Summary\n",
      "300-sample fixed-seed (42) subsets, greedy decoding, vLLM-served on T4.\n",
      df.to_markdown(index=False)]
(OUT / "summary.md").write_text("\n".join(md))
print(df.to_string(index=False))
print(f"\nsaved -> {OUT}/summary.csv, {OUT}/summary.md")
