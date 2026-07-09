# Small VLMs for Document Understanding

## Problem statement

Sub-1B vision-language models are attractive for on-device document processing, but their document understanding ability is largely uncharacterized: published evaluations focus on 3B+ models, and generic VQA accuracy misses what pipelines need (dense text extraction, layout reasoning, and knowing when an extraction is wrong). This work benchmarks small VLMs on document tasks with metrics beyond accuracy (ANLS, calibration, latency) and probes robustness on real degraded documents.

Full methodology and results: [report/Technical_Report.pdf](report/Technical_Report.pdf)

## Datasets

| Dataset | Why |
|---|---|
| DocVQA (val, 300-sample subset) | Canonical document VQA benchmark (forms, invoices, letters); comparable with published results |
| InfoVQA (val, 300-sample subset) | Infographics needing joint text + layout + graphics reasoning, beyond plain extraction |
| [Custom set](custom_docs/) (25 docs, 70 QA) | Hand-annotated degraded scans (skew, handwriting, fine print); tests real-world robustness the public benchmarks skip and cannot be memorized from pretraining |

## Models

| Model | Size | Why |
|---|---|---|
| Qwen3.5-0.8B | 0.8B | Newest sub-1B entrant; document understanding is an explicit design goal |
| InternVL3-1B | ~1B | Resolution-optimized encoder (dynamic tiling); tests if encoder resolution drives document performance |
| SmolVLM-500M-Instruct | 0.5B | Purpose-built edge-native design; the "designed for constraint" reference |
| Donut (docvqa-ft) | ~0.2B | Task-specific OCR-free specialist baseline, fine-tuned on DocVQA train |

## Code to reproduce

Colab session with a T4 GPU and Drive mounted; clone the repo to `/content/drive/MyDrive/vlm_eval/`. Benchmark TSVs (~1.4 GB, not committed) are fetched in step 2. Every stage writes to Drive and resumes after a disconnect.

```bash
# 1. environments
bash scripts/setup/setup_env.sh           # miniconda + vlmeval env + VLMEvalKit
bash scripts/setup/setup_vllm_env.sh      # separate vllm env (pins its own torch)
python scripts/setup/register_models.py   # register the 3 models (T4 sdpa fix)

# 2. data
python scripts/setup/make_subsets.py      # fixed-seed 300-sample DocVQA/InfoVQA subsets

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
