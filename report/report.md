# Adapting Small VLMs for Document Understanding
## A Systematic Evaluation of Sub-1B Vision-Language Models on Document Benchmarks

**Author:** Akshata Bhat &nbsp;·&nbsp; **Date:** 2026-07-08 &nbsp;·&nbsp; **Hardware:** Google Colab free tier (Tesla T4, 15GB VRAM, 12GB system RAM)

---

## Executive Summary

Four models were evaluated on document-understanding benchmarks under an identical, reproducible protocol: three sub-1B general-purpose VLMs — **Qwen3.5-0.8B**, **InternVL3-1B**, **SmolVLM-500M-Instruct** — and a task-specific specialist baseline, **Donut** (donut-base-finetuned-docvqa, ~200M). Metrics go beyond accuracy: ANLS, calibration (ECE from token logprobs), and single-stream latency.

**Qwen3.5-0.8B wins on every axis**: best accuracy on both benchmarks (DocVQA 86.9 / InfoVQA 54.1 ANLS), best calibration (DocVQA ECE 3.2 — its confidence is actually trustworthy), and fastest inference (0.48 s/query median on a T4). The shared weakness of all models is **layout/graphical reasoning** (32–38 point drop from DocVQA to InfoVQA), and that failure mode is **overconfident** — the dangerous kind for deployment. The specialist baseline is dominated even on its home benchmark: Donut loses DocVQA to Qwen3.5-0.8B by ~24 ANLS despite being fine-tuned on DocVQA's training set.

---

## 1. Problem Statement

Vision-language models under 1B parameters show strong general visual reasoning, making them attractive for on-device and edge deployment. However, their performance on **document understanding** — extracting and reasoning over forms, invoices, reports, and infographics — is largely uncharacterized: published evaluations focus on 3B+ models, and generic VQA accuracy does not capture what document pipelines need: dense text extraction, layout-aware reasoning, and *knowing when the extraction is wrong*. This work (1) systematically evaluates architecturally distinct small VLMs on document benchmarks with metrics beyond accuracy, and (2) derives an evidence-based, parameter-efficient improvement strategy targeted at the observed failure pattern.

## 2. Methodology

### 2.1 Models

| Model | Params | Vision Encoder | Decoder | Why included |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | Native Qwen encoder (high-res tiling) | Qwen3.5 (hybrid linear/full attention) | Newest sub-1B entrant; document understanding is an explicit design goal |
| InternVL3-1B | ~1B | InternViT (dynamic tiling) | Qwen2.5-0.5B | Tests whether encoder-side resolution handling dominates document performance |
| SmolVLM-500M-Instruct | 0.5B | SigLIP | SmolLM2-360M | Purpose-built edge-native design — the "designed for constraint" reference |
| Donut (docvqa-ft) | ~0.2B | Swin (OCR-free) | BART | Task-specific specialist baseline; fine-tuned on DocVQA train (in-domain there) |

The three VLMs share no encoder or decoder family, so results cannot be attributed to a single component. Donut was added as a specialist-vs-generalist axis.

### 2.2 Benchmarks

- **DocVQA (VAL)** — canonical document VQA: forms, invoices, reports, letters; largely text-extraction-heavy.
- **InfoVQA (VAL)** — infographics requiring joint reasoning over text, layout, and graphical elements; isolates layout/visual reasoning beyond extraction.

**Subsets:** identical fixed-seed (42) 300-sample subsets per benchmark for every model (exact indices in `subset_indices.json`). At n=300 the 95% CI on accuracy-type metrics is roughly ±3–5 points; the observed gaps (21–38 points) far exceed it.

### 2.3 Metrics

- **ANLS** — standard for DocVQA/InfoVQA; tolerant of near-miss extractions (threshold 0.5).
- **ECE (10-bin)** — confidence = geometric-mean token probability from generation logprobs; correctness = per-sample ANLS ≥ 0.5. Measures whether the model knows when it's wrong — critical for on-device pipelines that must defer/flag uncertain extractions.
- **Latency** — single-stream (sequential queries): mean/median s/query and tokens/s on the T4.

### 2.4 Experimental Setup

- **Serving:** all three VLMs hosted with vLLM 0.24 (OpenAI-compatible API, fp16, identical flags), evaluated through VLMEvalKit's API mode with 4 parallel clients; greedy decoding (temperature 0), max 512 new tokens; Qwen pixel caps (256·28²–1280·28²) mirrored between backends. Donut has no vLLM/VLMEvalKit support and used a custom HF driver with the same subsets, prompt-equivalent task format, greedy decoding, and the same scoring path.
- **Prompt parity:** every model receives the dataset's standard prompt including the "Answer the question using a single word or phrase." suffix (Donut uses its native task-token format, which is its equivalent).
- **Cross-validation of the pipeline:** ANLS was computed twice per VLM — by VLMEvalKit's evaluator on the main run and independently from the logprobs-pass predictions. The two agree within 1–4 points on all six legs.
- **Software:** dedicated conda environments (`vlmeval`: Python 3.10, transformers 5.13; `vllm`: Python 3.12, vLLM 0.24). All scripts, configs, subset indices, and a timestamped work log are on Drive; every step is re-runnable after a Colab disconnect.

## 3. Results

| Model | DocVQA ANLS | InfoVQA ANLS | ECE Doc | ECE Info | Overconf. Info (conf−acc) | p50 latency (s) | tok/s |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-0.8B** | **86.9** | **54.1** | **3.2** | **14.6** | +12.4 | **0.48** | 11.2 |
| InternVL3-1B | 83.2 | 50.9 | 5.3 | 24.9 | +31.3 | 1.61 | 7.1 |
| SmolVLM-500M | 61.8 | 23.3 | 15.3 | 40.5 | +40.5 | 0.99 | 7.3 |
| Donut (docvqa-ft) | 62.6* | 13.9 | 29.4 | 66.0 | +66.0 | 0.60 | 10.5 |

\* Donut's ANLS from the per-sample scoring path (identical formula); for the VLMs this path matched the main evaluator within 1–4 points. Donut is *in-domain* on DocVQA (fine-tuned on its train split).

### Findings

1. **Qwen3.5-0.8B is the best sub-1B document model on every axis** — accuracy, calibration, and speed. Its DocVQA calibration is essentially honest (ECE 3.2, slightly *under*confident at −1.4), meaning confidence-thresholded auto-accept is actually viable for DocVQA-style inputs.
2. **The dominant capability gap is layout/graphical reasoning, not text extraction.** Every model drops 32–38 ANLS from DocVQA to InfoVQA under an identical protocol.
3. **The InfoVQA failure mode is wrong-and-confident.** Overconfidence grows monotonically as capability falls (+12 → +66). Naive confidence gating is unsafe for infographic-style inputs on all models.
4. **Edge-native design did not pay off on documents.** SmolVLM-500M trails the generalists by 21–31 ANLS at comparable latency and is the worst-calibrated VLM.
5. **The specialist is dominated by modern generalists.** Donut loses its home benchmark by ~24 ANLS to Qwen3.5-0.8B despite in-domain fine-tuning, collapses out-of-domain (13.9), has the worst calibration of the whole set (95% mean confidence on DocVQA at 66% accuracy), and isn't faster than Qwen. Pretraining scale/recipe beat task-specific architecture. *(Caveat: single public checkpoint, not retrained.)*
6. **Latency ordering is counter-intuitive:** the largest model (Qwen 0.8B) is the fastest (0.48s p50) — architecture and serving efficiency matter more than parameter count at this scale. All models are within interactive range on a free T4.

## 4. Knowledge Gap & Improvement Strategy

**Gap (best model, Qwen3.5-0.8B):** InfoVQA-style layout/graphical reasoning — 54.1 vs 86.9 ANLS — compounded by overconfidence (+12.4) on exactly those inputs, i.e. the model does not signal its own failures there.

**Proposed remediation — LoRA, prepared and sanity-checked (execution descoped by decision):** rank-16 LoRA on the decoder's attention/MLP projections (6.4M trainable params, 0.74%), vision encoder frozen; 800 InfoVQA training samples strictly disjoint from the eval subset; prompts identical to evaluation; loss on answer tokens only; fp16 + gradient checkpointing fits a T4 at 8.1GB. Post-training evaluation re-runs the identical InfoVQA subset (target: >54.1) plus DocVQA as a regression check (hold ≈86.9). All assets (train set, training script validated by a 1-step forward/backward check, serve-and-eval script) are committed and re-runnable.

**Expected outcome:** measurable ANLS gain on the weak subset without DocVQA regression, plus (secondary) reduced InfoVQA overconfidence, since the adapter trains on exactly the distribution where confidence is miscalibrated.

## 5. Limitations

- 300-sample subsets (±3–5 point CI); validation splits only.
- LoRA train pool shares the InfoVQA VAL distribution (official train split requires registration); disjoint at sample level — sufficient for a PoC claim, not a leaderboard claim.
- Donut comparison uses the single public checkpoint without retraining.
- Latency measured on one hardware target (T4, fp16, vLLM/HF); rankings may shift on NPUs/CPUs.
- ECE uses sequence-level geometric-mean token probability; other confidence estimators (e.g. verbalized confidence) were not tested.

## 6. Reproducibility & Engineering Notes

All code, data manifests, logs, and results live in `vlm_eval/` (scripts `00`–`80`, numbered by pipeline stage) with a timestamped `WORKLOG.md`. Notable engineering findings preserved there: T4 requires SDPA (flash-attn 2 unsupported); InternVL3's remote code is incompatible with transformers 5.x (vLLM's native implementation sidesteps it); vLLM's default 4GB multimodal cache plus concurrent infographic payloads OOM-kills the engine on 12GB-RAM Colab (`--mm-processor-cache-gb 0` fixes it); Qwen3.5's linear-attention Triton autotune costs ~12 min on first start (cached thereafter); VLMEvalKit's `process_line(anls)` returns distances, not similarities — `hit_calculate` applies the threshold.

## 7. Recommendation

For on-device document pipelines in this compute class, **Qwen3.5-0.8B is the clear foundation**: best accuracy, honest confidence on extraction-style documents (enabling automatic accept/defer routing), and the lowest latency. Infographic-style inputs should be routed to human review or a larger model regardless of reported confidence — for now. The prepared LoRA adaptation is the lowest-cost path to closing that gap and is ready to execute.
