#!/usr/bin/env python3
"""Build the LoRA training set: InfoVQA_VAL rows DISJOINT from the 300-sample
eval subset (eval rows are never trained on). Decodes base64 images to disk and
writes train.jsonl. Fixed seed for reproducibility.

Output: /content/drive/MyDrive/vlm_eval/lora/{train.jsonl, images/}
"""
import base64
import json
import random
from pathlib import Path

import pandas as pd

LMU = "/content/drive/MyDrive/vlm_eval/LMUData"
OUT = Path("/content/drive/MyDrive/vlm_eval/lora")
SEED = 42
N_TRAIN = 800

sub_idx = set(json.load(open("/content/drive/MyDrive/vlm_eval/subset_indices.json"))
              ["datasets"]["InfoVQA_VAL_SUB300"])

df = pd.read_csv(f"{LMU}/InfoVQA_VAL.tsv", sep="\t", dtype=str)
img_map = dict(zip(df["index"], df["image"]))
df["image"] = df["image"].map(lambda v: img_map[v] if isinstance(v, str) and 0 < len(v) <= 64 else v)

pool = df[~df["index"].isin(sub_idx)]
print(f"pool: {len(pool)} rows (of {len(df)}, {len(sub_idx)} held out for eval)")
train = pool.sample(n=N_TRAIN, random_state=SEED)

img_dir = OUT / "images"
img_dir.mkdir(parents=True, exist_ok=True)
records = []
for _, row in train.iterrows():
    p = img_dir / f"{row['index']}.jpg"
    if not p.exists():
        p.write_bytes(base64.b64decode(row["image"]))
    ans = row["answer"]
    if isinstance(ans, str) and ans.startswith("["):  # stringified list -> first gt
        ans = eval(ans)[0]
    records.append({"index": str(row["index"]), "image": str(p),
                    "question": str(row["question"]), "answer": str(ans)})

with open(OUT / "train.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(records)} training records -> {OUT}/train.jsonl")
