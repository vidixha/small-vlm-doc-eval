#!/usr/bin/env python3
"""Aggregate the custom-set prompting eval (CustomDocVQA, from inference/prompt_eval.py)
into an overall CoT-vs-direct table PLUS a per-failure-mode accuracy breakdown -
the point of the custom set: which degradation conditions each model chokes on.

Same ANLS path as the other analyzers (process_line + DocVQA 0.5 threshold).
Joins predictions to failure modes via results/custom_index_map.json.

Emits results/custom_summary.{csv,md}.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LMUData", "/content/drive/MyDrive/vlm_eval/LMUData")
from vlmeval.dataset.utils.vqa_eval import hit_calculate, process_line

BASE = Path("/content/drive/MyDrive/vlm_eval")
PROMPT_DIR = BASE / "results" / "prompting"
INDEX_MAP = BASE / "results" / "custom_index_map.json"
OUT = BASE / "results"
MODELS = ["Qwen3.5-0.8B", "InternVL3-1B", "SmolVLM-500M", "Donut-DocVQA"]
MODES = ["direct", "cot"]
DS = "CustomDocVQA"


def anls_of(answer, prediction):
    line = pd.Series({"answer": answer, "prediction": prediction})
    return float(hit_calculate([process_line(line, method="anls")], "DocVQA")[0])


imap = json.loads(INDEX_MAP.read_text()) if INDEX_MAP.exists() else {}

overall, by_mode_records = [], {}
for model in MODELS:
    for mode in MODES:
        jf = PROMPT_DIR / f"{model}_{DS}_{mode}.jsonl"
        if not jf.exists():
            continue
        recs = [json.loads(l) for l in open(jf) if l.strip()]
        ok = [r for r in recs if "error" not in r]
        if not ok:
            continue
        anls = np.array([anls_of(r["answer"], r["prediction"]) for r in ok])
        correct = (anls >= 0.5).astype(float)
        overall.append({"model": model, "mode": mode, "n": len(ok),
                        "anls": round(float(anls.mean()) * 100, 2),
                        "acc@0.5": round(float(correct.mean()) * 100, 2)})
        by_mode_records[(model, mode)] = list(zip([str(r["index"]) for r in ok], correct))

df = pd.DataFrame(overall)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "custom_summary.csv", index=False)

# ----- per-failure-mode accuracy breakdown -----
# failure modes are free-text; normalize (lower/strip) for grouping.
fm_rows = []
for (model, mode), recs in by_mode_records.items():
    bucket = defaultdict(list)
    for index, c in recs:
        meta = imap.get(index, {})
        for fm in (meta.get("failure_modes") or ["(none)"]):
            bucket[fm.strip().lower()].append(c)
    for fm, cs in bucket.items():
        fm_rows.append({"failure_mode": fm, "model": model, "mode": mode,
                        "n": len(cs), "acc@0.5": round(float(np.mean(cs)) * 100, 1)})
fm_df = pd.DataFrame(fm_rows)
if not fm_df.empty:
    fm_df.to_csv(OUT / "custom_failuremode_breakdown.csv", index=False)

# ----- markdown -----
lines = ["# Custom Document Set: Prompting Eval\n",
         f"Hand-annotated degraded-scan set ({len(imap)} QA pairs). "
         "`direct` vs `cot`, greedy, vLLM-served, ANLS (DocVQA 0.5 threshold).\n"]
if not df.empty:
    piv = df.pivot_table(index="model", columns="mode", values="anls")
    if {"direct", "cot"}.issubset(piv.columns):
        piv["Δ (cot-direct)"] = (piv["cot"] - piv["direct"]).round(2)
    lines += ["## Overall ANLS\n", piv.reset_index().to_markdown(index=False), "\n"]
    lines += ["## Full metrics\n", df.to_markdown(index=False), "\n"]
if not fm_df.empty:
    for mode in MODES:
        sub = fm_df[fm_df["mode"] == mode]
        if sub.empty:
            continue
        p = sub.pivot_table(index="failure_mode", columns="model", values="acc@0.5")
        p.insert(0, "n", sub.groupby("failure_mode")["n"].max())
        lines += [f"## acc@0.5 by failure mode: {mode}\n",
                  p.reset_index().to_markdown(index=False), "\n"]
else:
    lines.append("_No results yet: run scripts/inference/run_custom.sh first._")

(OUT / "custom_summary.md").write_text("\n".join(lines))
print(df.to_string(index=False) if not df.empty else "no results yet")
print(f"\nsaved -> {OUT}/custom_summary.md, custom_summary.csv, custom_failuremode_breakdown.csv")
