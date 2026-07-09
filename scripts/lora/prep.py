#!/usr/bin/env python3
"""Build a larger LoRA training set: DocVQA_VAL + InfoVQA_VAL rows, each
DISJOINT from their own 300-sample eval subset. Decodes base64 images to disk
and writes train.jsonl. Fixed seed for reproducibility.

Bigger and mixed-domain vs the first PoC (800 InfoVQA-only rows), since that
run did not move the needle.

Output: /content/drive/MyDrive/vlm_eval/lora/{train.jsonl, images/}
"""
import base64
import json
from pathlib import Path

import pandas as pd

LMU = "/content/drive/MyDrive/vlm_eval/LMUData"
OUT = Path("/content/drive/MyDrive/vlm_eval/lora")
SEED = 42
N_PER_DATASET = {"DocVQA_VAL": 2500, "InfoVQA_VAL": 2000}

img_dir = OUT / "images"
img_dir.mkdir(parents=True, exist_ok=True)

records = []
for name, n in N_PER_DATASET.items():
    sub = pd.read_csv(f"{LMU}/{name}_SUB300.tsv", sep="\t", dtype=str, usecols=["index"])
    sub_idx = set(sub["index"])

    df = pd.read_csv(f"{LMU}/{name}.tsv", sep="\t", dtype=str)
    img_map = dict(zip(df["index"], df["image"]))
    df["image"] = df["image"].map(lambda v: img_map[v] if isinstance(v, str) and 0 < len(v) <= 64 else v)

    pool = df[~df["index"].isin(sub_idx)]
    print(f"{name}: pool {len(pool)} rows (of {len(df)}, {len(sub_idx)} held out for eval)")
    take = pool.sample(n=min(n, len(pool)), random_state=SEED)

    for _, row in take.iterrows():
        p = img_dir / f"{name}_{row['index']}.jpg"
        if not p.exists():
            p.write_bytes(base64.b64decode(row["image"]))
        ans = row["answer"]
        if isinstance(ans, str) and ans.startswith("["):  # stringified list -> first gt
            ans = eval(ans)[0]
        records.append({"index": f"{name}_{row['index']}", "image": str(p),
                        "question": str(row["question"]), "answer": str(ans)})

import random
random.Random(SEED).shuffle(records)

with open(OUT / "train.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(records)} training records -> {OUT}/train.jsonl")
