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

## Repository layout

```
scripts/       pipeline (numbered by run order)
  00 / 05      env setup (miniconda + vlmeval env; separate vllm env)
  10 / 15      register models into VLMEvalKit; build fixed-seed 300-sample subsets
  20           5-sample smoke test
  30 / 35      full eval — HF path / vLLM-served path (headline numbers)
  40 / 45      CoT-vs-direct prompting driver + orchestrator
  50 / 55      confidence (ECE) + latency pass
  60 / 65      analysis — main summary / prompting comparison
  70 / 71 / 72 LoRA PoC (prep / train / eval) — prepared, execution descoped
  80           Donut specialist baseline (custom HF driver)
  90           custom-document annotation-page builder
results/       summary.{csv,md}, gap_analysis.md, ece/ (per-sample logprob JSONLs)
report/        report.md + Technical_Report.pdf
TASK.md        problem statement + methodology
WORKLOG.md     timestamped engineering log of the run
subset_indices.json   exact sampled indices (reproducibility)
```

## Reproducing from scratch (Colab T4)

```bash
# 1. environments (re-runnable after any Colab disconnect)
bash scripts/00_setup_env.sh          # miniconda + `vlmeval` env + VLMEvalKit
bash scripts/05_setup_vllm_env.sh     # separate `vllm` env (pins its own torch)
python scripts/10_register_models.py  # register the 3 models; T4 flash-attn -> sdpa fix

# 2. data — fixed-seed 300-sample subsets of DocVQA/InfoVQA VAL
python scripts/15_make_subsets.py
bash   scripts/20_smoke_test.sh        # sanity: 5-sample DocVQA

# 3. evaluate (vLLM-served, greedy)
bash   scripts/35_full_eval_vllm.sh    # headline ANLS
bash   scripts/55_ece_all.sh           # ECE + latency
bash   scripts/45_run_prompting.sh     # CoT vs direct
python scripts/80_donut_eval.py        # specialist baseline

# 4. analyse
python scripts/60_analyze.py           # -> results/summary.{md,csv}
python scripts/65_analyze_prompting.py # -> results/prompting_summary.{md,csv}
```

All scripts assume the project lives at `/content/drive/MyDrive/vlm_eval/` and the
benchmark TSVs under `LMUData/` (not committed — 1.4 GB; fetched by `15_make_subsets.py`).

## Notes & limitations

- **Hardware:** Google Colab free tier — Tesla T4 (15 GB VRAM), 12 GB system RAM. Serving
  uses `fp16`, `--mm-processor-cache-gb 0`, and 4 parallel clients to stay within RAM.
- **Subsets:** n=300 per benchmark (95% CI ≈ ±3–5 ANLS); observed gaps (21–38) far exceed it.
- **Cross-validation:** ANLS is computed twice per VLM (VLMEvalKit's evaluator and an
  independent logprobs pass) and agrees within 1–4 points on all legs.
- **LoRA PoC** (`scripts/70–72`) is prepared and sanity-checked but its training run was
  descoped; no adapter is shipped.
- **Not committed:** the `LMUData/` benchmark datasets, LoRA training data, and the custom
  document images — code and result summaries only.
