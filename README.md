# Small VLMs for Document Understanding

Evaluation of sub-1B vision-language models (Qwen3.5-0.8B, InternVL3-1B, SmolVLM-500M, plus a Donut specialist baseline) on DocVQA, InfoVQA, and a custom set of 25 hand-annotated degraded documents. Metrics: ANLS, calibration (ECE), latency, and direct vs chain-of-thought prompting.

Full methodology, results, and discussion: **[report/Technical_Report.pdf](report/Technical_Report.pdf)** (source: [report/report.md](report/report.md)).

## Problem statement

Vision-language models under 1B parameters show strong general visual reasoning, which makes them attractive for on-device and edge deployment where compute and memory are tightly constrained. However, their performance on document understanding (extracting and reasoning over forms, invoices, reports, and infographics) is largely uncharacterized: published evaluations focus on models in the 3B+ range, and generic VQA accuracy does not capture what document pipelines actually need. Three capabilities matter in practice: dense text extraction, layout-aware reasoning, and knowing when an extraction is wrong, because a deployed pipeline must decide when to auto-accept a field value and when to defer to review.

This work addresses that gap in three parts:

1. A systematic evaluation of architecturally distinct sub-1B VLMs on document benchmarks, with metrics beyond accuracy (calibration and latency).
2. A custom, hand-annotated evaluation set ([custom_docs/](custom_docs/), 25 images, 70 QA pairs with failure-mode tags) probing robustness to real-world document degradation, which public benchmarks do not isolate.
3. An evidence-based, parameter-efficient improvement strategy targeted at the observed failure pattern.

## Repository layout

```
scripts/setup/       environments, model registration, benchmark subsets, custom-set TSV
scripts/inference/   model serving and prediction runs (main eval, prompting, ECE, Donut)
scripts/eval/        analysis: summary tables and failure-mode breakdowns
custom_docs/         the public custom dataset: 25 document images + annotations.json
results/             all result tables and per-sample records
report/              technical report (md + pdf)
subset_indices.json  exact benchmark sample indices (reproducibility)
```

## How to run

Prerequisites: a Google Colab session with a T4 GPU and Drive mounted. Scripts assume the project lives at `/content/drive/MyDrive/vlm_eval/`, so clone or copy the repository there. Benchmark TSVs (about 1.4 GB, not committed) are fetched automatically in step 2. Every stage writes to Drive and is resumable after a disconnect.

```bash
# 1. environments
bash scripts/setup/setup_env.sh           # miniconda + vlmeval env + VLMEvalKit
bash scripts/setup/setup_vllm_env.sh      # separate vllm env (pins its own torch)
python scripts/setup/register_models.py   # register the 3 models (T4 sdpa fix)

# 2. data + sanity check
python scripts/setup/make_subsets.py      # fixed-seed 300-sample DocVQA/InfoVQA subsets
bash scripts/inference/smoke_test.sh      # mandatory 5-sample run per model

# 3. main evaluation (vLLM-served, greedy)
bash scripts/inference/full_eval_vllm.sh  # headline ANLS, 3 models x 2 benchmarks
bash scripts/inference/run_ece.sh         # confidence (ECE) + latency pass
bash scripts/inference/run_prompting.sh   # direct vs chain-of-thought
python scripts/inference/donut_eval.py    # Donut baseline

# 4. custom document set
python scripts/setup/build_custom_tsv.py  # annotations.json -> CustomDocVQA.tsv
bash scripts/inference/run_custom.sh      # direct + cot on the custom set
python scripts/inference/donut_custom.py  # Donut on the custom set (direct only)

# 5. analysis -> results/
python scripts/eval/analyze.py            # main summary
python scripts/eval/analyze_prompting.py  # prompting comparison
python scripts/eval/analyze_custom.py     # custom-set summary + failure-mode breakdown
```

`scripts/inference/full_eval_hf.sh` is a plain HF-transformers fallback for step 3; note InternVL3-1B only runs via the vLLM path on transformers 5.x.
