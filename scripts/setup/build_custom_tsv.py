#!/usr/bin/env python3
"""Convert the hand-annotated custom document set (custom_docs/annotations.json,
exported from the annotator artifact) into a VLMEvalKit-style TSV so the same
eval pipeline can score it.

- One TSV row per (document, question) pair.
- `image` column = base64 of the document (downscaled to <= MAX_DIM so the
  giant 4-18MP scans don't OOM the non-Qwen servers; text stays legible).
- `answer` = Python list-repr string (e.g. ['$3360.00']) so process_line's ANLS
  path eval()s it, exactly like the DocVQA/InfoVQA subsets.
- Dataset name keeps the `DocVQA` substring (-> ANLS routing) but is clearly
  custom: CustomDocVQA.
- Also writes results/custom_index_map.json mapping each row index back to its
  doc id + file + failure_modes, so eval/analyze_custom.py can break accuracy down
  by failure mode.

Run on Colab after annotations.json is in custom_docs/:
  python setup/build_custom_tsv.py
"""
import base64
import io
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image

BASE = Path("/content/drive/MyDrive/vlm_eval")
ANN = BASE / "custom_docs" / "annotations.json"
IMG_DIR = BASE / "custom_docs"
LMU = BASE / "LMUData"
OUT_TSV = LMU / "CustomDocVQA.tsv"
INDEX_MAP = BASE / "results" / "custom_index_map.json"
MAX_DIM = 2000          # cap the longest side; big enough for dense fine print
JPEG_Q = 90
INDEX_BASE = 900000     # keep indices clear of the DocVQA/InfoVQA ranges


def encode_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(1.0, MAX_DIM / max(w, h))
    if s < 1.0:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_Q)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ann = json.loads(ANN.read_text())
    records = ann.get("annotations", ann)
    LMU.mkdir(parents=True, exist_ok=True)
    INDEX_MAP.parent.mkdir(parents=True, exist_ok=True)

    rows, index_map = [], {}
    img_cache = {}
    idx = INDEX_BASE
    skipped = 0

    for doc in records:
        doc_id = str(doc["id"])
        fname = doc.get("file", f"{doc_id}.png")
        fpath = IMG_DIR / fname
        if not fpath.exists():
            # tolerate .png/.jpg mismatch
            alt = list(IMG_DIR.glob(f"{doc_id}.*"))
            if not alt:
                print(f"!! image for doc {doc_id} not found: skipping")
                continue
            fpath = alt[0]
        if fpath.name not in img_cache:
            img_cache[fpath.name] = encode_image(fpath)
        b64 = img_cache[fpath.name]
        fails = doc.get("failure_modes", doc.get("tags", []))

        for qa in doc.get("qas", []):
            q = (qa.get("question") or "").strip()
            answers = [a.strip() for a in (qa.get("answers") or []) if a.strip()]
            if not q or not answers:
                skipped += 1
                continue
            rows.append({
                "index": idx,
                "image": b64,
                "question": q,
                "answer": str(answers),          # list-repr string -> ANLS eval()
                "doc_id": doc_id,
                "file": fpath.name,
                "failure_modes": str(fails),
            })
            index_map[str(idx)] = {"doc_id": doc_id, "file": fpath.name,
                                   "question": q, "answers": answers,
                                   "failure_modes": fails}
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    INDEX_MAP.write_text(json.dumps(index_map, indent=1))

    n_docs = len({r["doc_id"] for r in rows})
    print(f"wrote {OUT_TSV}  ({len(df)} QA rows over {n_docs} docs, {skipped} empty skipped)")
    print(f"wrote {INDEX_MAP}")
    print("\nrun the eval with:")
    print("  bash scripts/inference/run_custom.sh            # direct + cot, all 3 models")
    print("  python scripts/eval/analyze_custom.py      # -> results/custom_summary.{md,csv}")


if __name__ == "__main__":
    main()
