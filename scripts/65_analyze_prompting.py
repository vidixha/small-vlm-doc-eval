#!/usr/bin/env python3
"""Aggregate the CoT-vs-direct prompting eval (40_prompt_eval.py JSONLs) into a
side-by-side comparison. Same ANLS path as 60_analyze.py (process_line + the
DocVQA 0.5 threshold via hit_calculate), same 10-bin ECE.

Emits vlm_eval/results/prompting_summary.{csv,md}:
  - long CSV: one row per (model, dataset, mode)
  - markdown: ANLS direct vs cot with the delta, plus ECE / latency / answer length
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LMUData", "/content/drive/MyDrive/vlm_eval/LMUData")
from vlmeval.dataset.utils.vqa_eval import hit_calculate, process_line

PROMPT_DIR = Path("/content/drive/MyDrive/vlm_eval/results/prompting")
OUT = Path("/content/drive/MyDrive/vlm_eval/results")
MODELS = ["Qwen3.5-0.8B", "InternVL3-1B", "SmolVLM-500M"]
DATASETS = ["DocVQA_VAL_SUB300", "InfoVQA_VAL_SUB300"]
MODES = ["direct", "cot"]


def anls_of(answer, prediction):
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
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


rows = []
for model in MODELS:
    for ds in DATASETS:
        for mode in MODES:
            jf = PROMPT_DIR / f"{model}_{ds}_{mode}.jsonl"
            if not jf.exists():
                continue
            recs = [json.loads(l) for l in open(jf) if l.strip()]
            ok = [r for r in recs if "error" not in r and r.get("conf_geo") is not None]
            if not ok:
                continue
            per_anls = np.array([anls_of(r["answer"], r["prediction"]) for r in ok])
            conf = np.array([r["conf_geo"] for r in ok])
            correct = (per_anls >= 0.5).astype(float)
            seq = [r for r in ok if r.get("sched") == "sequential"]
            lat_p50 = float(np.median([r["latency_s"] for r in seq])) if seq else np.nan
            rows.append({
                "model": model, "dataset": ds, "mode": mode, "n": len(ok),
                "anls": round(float(per_anls.mean()) * 100, 2),
                "acc@0.5": round(float(correct.mean()) * 100, 2),
                "ECE": round(ece_10bin(conf, correct) * 100, 2),
                "overconf": round(float(conf.mean() - correct.mean()) * 100, 2),
                "lat_p50_s": round(lat_p50, 2),
                "mean_out_tokens": round(float(np.mean([r["n_tokens"] for r in ok])), 1),
            })

df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "prompting_summary.csv", index=False)

# ANLS pivot: direct vs cot + delta
lines = ["# CoT vs Direct Prompting — Sub-1B VLMs\n",
         "300-sample fixed-seed (42) subsets, greedy decoding, vLLM-served on T4. "
         "`direct` = the standard single-phrase prompt (reproduces the headline eval); "
         "`cot` = step-by-step reasoning with the final answer parsed back out for ANLS.\n"]
if not df.empty:
    piv = df.pivot_table(index=["model", "dataset"], columns="mode", values="anls")
    if {"direct", "cot"}.issubset(piv.columns):
        piv["Δ (cot−direct)"] = (piv["cot"] - piv["direct"]).round(2)
    lines.append("## ANLS\n")
    lines.append(piv.reset_index().to_markdown(index=False))
    lines.append("\n## Full metrics (per model × dataset × mode)\n")
    lines.append(df.to_markdown(index=False))
else:
    lines.append("_No prompting result JSONLs found in results/prompting/ yet._")

(OUT / "prompting_summary.md").write_text("\n".join(lines))
print(df.to_string(index=False) if not df.empty else "no results yet")
print(f"\nsaved -> {OUT}/prompting_summary.csv, {OUT}/prompting_summary.md")
