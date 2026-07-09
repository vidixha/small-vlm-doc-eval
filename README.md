# Adapting Small VLMs for Document Understanding

A reproducible evaluation of **sub-1B vision-language models** on document-understanding
benchmarks (DocVQA, InfoVQA), with metrics **beyond accuracy** — ANLS, calibration
(ECE), and single-stream latency — plus a **chain-of-thought vs. direct prompting**
comparison. Everything runs on a single **Google Colab free-tier T4** and persists to
Drive so it survives disconnects.

## Summary

Four models under one identical protocol: three sub-1B general-purpose VLMs
(**Qwen3.5-0.8B**, **InternVL3-1B**, **SmolVLM-500M-Instruct**) and a task-specific
specialist baseline (**Donut**, `donut-base-finetuned-docvqa`, ~200M).

**Qwen3.5-0.8B wins on every axis** — best accuracy on both benchmarks, best
calibration, and fastest inference. The shared weakness across all models is
**layout/graphical reasoning** (a 32–38 point ANLS drop from DocVQA to InfoVQA), and
that failure mode is **overconfident** — the dangerous kind for deployment.

### Headline results — ANLS (300-sample fixed-seed subsets, greedy, vLLM-served on T4)

| Model | DocVQA | InfoVQA | ECE (Doc / Info) | p50 latency |
|---|---|---|---|---|
| **Qwen3.5-0.8B** | **86.9** | **54.1** | 3.2 / 14.6 | **0.48 s** |
| InternVL3-1B | 83.2 | 50.9 | 5.3 / 24.9 | 1.61 s |
| SmolVLM-500M | 61.8 | 23.3 | 15.3 / 40.5 | 0.99 s |
| Donut-DocVQA (~200M) | 62.6 | 13.9 | 29.4 / 66.0 | 0.60 s |

Full numbers and the knowledge-gap analysis: [`results/summary.md`](results/summary.md),
[`results/gap_analysis.md`](results/gap_analysis.md). Technical write-up:
[`report/Technical_Report.pdf`](report/Technical_Report.pdf).

## Chain-of-thought vs. direct prompting

The three VLMs are also evaluated under two prompting regimes on the same subsets and
the same ANLS scoring path, so the **only variable is the prompt**:

- **`direct`** — the standard *"Answer the question using a single word or phrase."*
  prompt (reproduces the headline eval above).
- **`cot`** — step-by-step reasoning, with the final answer parsed back out of the
  reasoning (`Answer: <…>`) for ANLS; confidence for ECE is taken over the
  answer-span tokens only, not the reasoning.

```bash
# after setup + subsets (see below), with the vllm/vlmeval envs built:
bash scripts/45_run_prompting.sh                 # serves each model, runs direct + cot
python scripts/65_analyze_prompting.py           # -> results/prompting_summary.{md,csv}
```

## Custom annotated document set

A held-out set of 25 degraded real-world scans (invoices, receipts, cheques,
forms) was hand-annotated — up to 3 Q/A pairs per document plus free-text
**failure-mode** tags (faint print, handwriting, skewed/upside-down, dense fine
print, stylized logos) — using a self-contained browser annotator. The export
(`annotations.json`) is converted into the same TSV/ANLS format and scored under
both prompting regimes, with accuracy broken down **by failure mode** to show
which degradation conditions each model chokes on.

**The dataset is public in this repository:**

- **Documents (25 images):** [`custom_docs/`](https://github.com/vidixha/small-vlm-doc-eval/tree/main/custom_docs)
- **Annotations (70 QA pairs + failure-mode tags):** [`custom_docs/annotations.json`](https://github.com/vidixha/small-vlm-doc-eval/blob/main/custom_docs/annotations.json)
- **Row-to-document map used by the analysis:** [`results/custom_index_map.json`](results/custom_index_map.json)

`annotations.json` schema: one entry per document with `id`, `file` (image name in
`custom_docs/`), `failure_modes` (list of free-text tags), and `qas` (list of
`{question, answers}` pairs; `answers` is a list of acceptable ground-truth strings,
scored with ANLS exactly like DocVQA).

Results on this set: [`results/custom_summary.md`](results/custom_summary.md) and
[`results/custom_failuremode_breakdown.csv`](results/custom_failuremode_breakdown.csv);
discussion in the [technical report](report/Technical_Report.pdf).

## Repository layout

```
scripts/       pipeline (numbered by run order)
  00 / 05      env setup (miniconda + vlmeval env; separate vllm env)
  10 / 15      register models into VLMEvalKit; build fixed-seed 300-sample subsets
  20           5-sample smoke test
  30 / 35      full eval — HF path / vLLM-served path (headline numbers)
  40 / 45      CoT-vs-direct prompting driver + orchestrator
  47           run direct + cot on the custom annotated set
  50 / 55      confidence (ECE) + latency pass
  60 / 65 / 66 analysis — main summary / prompting comparison / custom-set + failure modes
  70 / 71 / 72 LoRA PoC (prep / train / eval) — prepared, execution descoped
  80 / 81      Donut specialist baseline — benchmark subsets / custom set
  90           custom-document annotation-page builder
  95           annotations.json -> CustomDocVQA TSV converter
custom_docs/   the public custom dataset: 25 document images + annotations.json
results/       summary.{csv,md}, gap_analysis.md, custom_summary.{csv,md},
               custom_failuremode_breakdown.csv, ece/ + prompting/ (per-sample JSONLs)
report/        report.md + Technical_Report.pdf
TASK.md        problem statement + methodology
WORKLOG.md     timestamped engineering log of the run
subset_indices.json   exact sampled indices (reproducibility)
```

## How to run the pipeline (Colab T4, from scratch)

The scripts are numbered by run order and every stage is resumable: results land on
Drive, completed (model, dataset) legs are skipped on rerun, and the setup scripts can
be rerun verbatim after any Colab disconnect.

**Prerequisites:** a Colab session with a T4 GPU and Drive mounted. All scripts assume
the project lives at `/content/drive/MyDrive/vlm_eval/`, so first copy or clone this
repository there. Benchmark TSVs go under `LMUData/` (not committed, about 1.4 GB;
fetched automatically by `15_make_subsets.py`).

### Stage 1 — Environments

```bash
bash scripts/00_setup_env.sh          # miniconda + `vlmeval` env + VLMEvalKit
bash scripts/05_setup_vllm_env.sh     # separate `vllm` env (vLLM pins its own torch)
python scripts/10_register_models.py  # register the 3 models; T4 flash-attn -> sdpa fix
```

### Stage 2 — Data and smoke test

```bash
python scripts/15_make_subsets.py      # downloads DocVQA/InfoVQA VAL, builds fixed-seed
                                       # 300-sample subsets (indices -> subset_indices.json)
bash   scripts/20_smoke_test.sh        # mandatory 5-sample sanity run per model
```

### Stage 3 — Main evaluation (vLLM-served, greedy)

```bash
bash   scripts/35_full_eval_vllm.sh    # headline ANLS: 3 models x 2 benchmarks
bash   scripts/55_ece_all.sh           # confidence (ECE) + latency pass
bash   scripts/45_run_prompting.sh     # CoT vs direct on the 300-sample subsets
python scripts/80_donut_eval.py        # Donut specialist baseline (HF driver)
python scripts/60_analyze.py           # -> results/summary.{md,csv} + gap analysis inputs
python scripts/65_analyze_prompting.py # -> results/prompting_summary.{md,csv}
```

### Stage 4 — Custom annotated document set

```bash
# (only if re-annotating from scratch; annotations.json is already committed)
python scripts/90_build_annotator.py     # builds the browser annotation page from custom_docs/

python scripts/95_build_custom_tsv.py    # annotations.json -> LMUData/CustomDocVQA.tsv
                                         #   + results/custom_index_map.json
bash   scripts/47_run_custom.sh          # serves each VLM, runs direct + cot on the custom set
python scripts/81_donut_custom.py        # Donut on the custom set (direct only; no chat/CoT)
python scripts/66_analyze_custom.py      # -> results/custom_summary.{md,csv}
                                         #   + results/custom_failuremode_breakdown.csv
```

### Stage 5 (optional) — LoRA adaptation PoC

```bash
python scripts/70_lora_prep.py         # 800 InfoVQA rows disjoint from the eval subset
python scripts/71_lora_train.py        # r=16 decoder-only LoRA (sanity-checked; ~8.1 GB VRAM)
bash   scripts/72_lora_eval.sh         # serve merged model, re-run both subsets
```

Prepared and sanity-checked; the training run itself was descoped, so no adapter is shipped.

**Fallback:** `scripts/30_full_eval.sh` is the HF-transformers (non-vLLM) path for the
main eval; note InternVL3-1B only works via the vLLM path on transformers 5.x.

## Notes & limitations

- **Hardware:** Google Colab free tier — Tesla T4 (15 GB VRAM), 12 GB system RAM. Serving
  uses `fp16`, `--mm-processor-cache-gb 0`, and 4 parallel clients to stay within RAM.
- **Subsets:** n=300 per benchmark (95% CI ≈ ±3–5 ANLS); observed gaps (21–38) far exceed it.
- **Cross-validation:** ANLS is computed twice per VLM (VLMEvalKit's evaluator and an
  independent logprobs pass) and agrees within 1–4 points on all legs.
- **LoRA PoC** (`scripts/70–72`) is prepared and sanity-checked but its training run was
  descoped; no adapter is shipped.
- **Not committed:** the `LMUData/` benchmark datasets and LoRA training data. The custom
  document set (25 images + `annotations.json`) **is** committed under `custom_docs/`.
