#!/usr/bin/env python3
"""Build fixed-seed evaluation subsets (same seed, same N, identical subset for
all models).

Current VLMEvalKit dropped `--limit`, so subsets are materialized as custom TSVs
in LMUData and passed via --data-config with class=ImageVQADataset. Names keep
the DocVQA/InfoVQA substring so evaluate_heuristic still routes to ANLS.

IMPORTANT: the `image` column may contain a pointer (another row's index) rather
than base64, deduplicating repeated document images. Subsetting naively breaks
those rows, so images are materialized before sampling.

Usage:
  python setup/make_subsets.py            # writes SUB300 TSVs + indices json
"""
import json
import pandas as pd

LMU = "/content/drive/MyDrive/vlm_eval/LMUData"
OUT_META = "/content/drive/MyDrive/vlm_eval/subset_indices.json"
SEED = 42
N = 300

meta = {"seed": SEED, "n": N, "datasets": {}}

for name in ["DocVQA_VAL", "InfoVQA_VAL"]:
    print(f"loading {name}.tsv ...")
    df = pd.read_csv(f"{LMU}/{name}.tsv", sep="\t", dtype=str)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # materialize image pointers (short values reference another row's index)
    img_map = dict(zip(df["index"], df["image"]))
    def resolve(v):
        if isinstance(v, str) and 0 < len(v) <= 64:
            return img_map[v]
        return v
    df["image"] = df["image"].map(resolve)
    assert df["image"].str.len().min() > 64, "unresolved image pointer remains"

    sub = df.sample(n=N, random_state=SEED).sort_index()
    sub.to_csv(f"{LMU}/{name}_SUB{N}.tsv", sep="\t", index=False)
    meta["datasets"][f"{name}_SUB{N}"] = [str(i) for i in sub["index"].tolist()]
    print(f"  wrote {name}_SUB{N}.tsv ({len(sub)} rows)")

with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=1)
print(f"subset indices saved to {OUT_META}")
