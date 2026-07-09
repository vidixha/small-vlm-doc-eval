# Adapting Small VLMs for Document Understanding
## A Systematic Evaluation of Sub-1B Vision-Language Models on Document Benchmarks and a Custom Degraded-Document Set

**Author:** Akshata Bhat &nbsp;·&nbsp; **Date:** 2026-07-09 &nbsp;·&nbsp; **Hardware:** Google Colab free tier (Tesla T4, 15GB VRAM, 12GB system RAM)

---

## Executive Summary

Four models were evaluated on document understanding under an identical, reproducible protocol: three sub-1B general-purpose vision-language models (**Qwen3.5-0.8B**, **InternVL3-1B**, **SmolVLM-500M-Instruct**) and a task-specific specialist baseline, **Donut** (donut-base-finetuned-docvqa, about 200M parameters). Evaluation covered two public benchmarks (DocVQA, InfoVQA) and a hand-annotated custom set of 25 real documents (70 QA pairs) targeting degradation conditions the public benchmarks do not test. Metrics go beyond accuracy: ANLS, calibration (ECE from token logprobs), and latency.

Three headline findings:

1. **Qwen3.5-0.8B wins the public benchmarks on every axis**: best accuracy (DocVQA 86.9, InfoVQA 54.1 ANLS), best calibration (DocVQA ECE 3.2), fastest inference (0.48 s/query median).
2. **Public benchmark scores overstate real-world robustness by roughly 25 to 30 ANLS points.** On the custom degraded-document set every model drops sharply (Qwen 86.9 to 54.8; InternVL3 83.2 to 57.0), the ranking between the top two models flips into a tie, and confidence calibration degrades to the point where naive confidence gating would be unsafe.
3. **Chain-of-thought prompting hurts extractive document QA at this scale.** CoT cost InternVL3-1B 22 ANLS points and Qwen3.5-0.8B 8 points on the custom set relative to direct prompting.

## 1. Problem Statement

Vision-language models under 1B parameters show strong general visual reasoning, which makes them attractive for on-device and edge deployment where compute and memory are tightly constrained. However, their performance on document understanding (extracting and reasoning over forms, invoices, reports, and infographics) is largely uncharacterized: published evaluations focus on models in the 3B+ range, and generic VQA accuracy does not capture what document pipelines actually need. Three capabilities matter in practice: dense text extraction, layout-aware reasoning, and knowing when an extraction is wrong, because a deployed pipeline must decide when to auto-accept a field value and when to defer to review.

This work addresses the gap in three parts:

1. A systematic evaluation of architecturally distinct sub-1B VLMs on document benchmarks, with metrics beyond accuracy (calibration and latency).
2. A custom, hand-annotated evaluation set probing robustness to real-world document degradation, which public benchmarks do not isolate.
3. An evidence-based, parameter-efficient improvement strategy targeted at the observed failure pattern.

## 2. Model Choice

| Model | Params | Vision Encoder | Decoder | Why included |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | Native Qwen encoder (high-res tiling) | Qwen3.5 (hybrid linear/full attention) | Newest sub-1B entrant; document understanding is an explicit design goal of the release |
| InternVL3-1B | ~1B | InternViT (dynamic tiling) | Qwen2.5-0.5B | Tests whether encoder-side resolution handling dominates document performance |
| SmolVLM-500M-Instruct | 0.5B | SigLIP | SmolLM2-360M | Purpose-built edge-native design; the "designed for constraint" reference point |
| Donut (docvqa-ft) | ~0.2B | Swin (OCR-free) | BART | Task-specific specialist baseline, fine-tuned on the DocVQA training split (in-domain there) |

The three VLMs share no encoder or decoder family, so results cannot be attributed to a single shared component. The set gives a clean three-way contrast: newest general-purpose design (Qwen3.5) versus resolution-optimized encoder (InternVL3) versus edge-native design (SmolVLM). Donut adds a fourth axis: does a 2022-era task-specific document architecture still beat modern generalists of similar size on its home task?

## 3. Data Choice

### 3.1 Public benchmarks

- **DocVQA (validation split)**: the canonical document VQA benchmark. Real-world forms, invoices, reports, and letters; largely text-extraction-heavy. Chosen for comparability with published results.
- **InfoVQA (validation split)**: infographic documents requiring joint reasoning over text, layout, and graphical elements. Chosen to complement DocVQA: it isolates layout and visual reasoning beyond plain extraction, so the pair separates "can read text" from "can reason over structure".

**Subsets:** identical fixed-seed (seed 42) 300-sample subsets per benchmark for every model; `scripts/setup/make_subsets.py` regenerates the exact same samples deterministically. At n=300 the 95% confidence interval on accuracy-type metrics is roughly plus or minus 3 to 5 points; the gaps observed (21 to 38 points) far exceed it.

### 3.2 Why a custom dataset was needed

Neither DocVQA nor InfoVQA stress-tests the conditions that actually break production document pipelines. Their images are mostly clean, well-digitized documents. Real capture is not: phone photos and office scans arrive skewed, rotated, blurred, faded, and full of handwriting, fine print, and dense tables. A model can score 87 ANLS on DocVQA and still fail on the receipts a user actually photographs. There is also a familiarity concern: both benchmarks are public and widely used, so their images may overlap model pretraining data; a private set cannot be memorized.

A custom set of **25 real documents** was therefore built and annotated by hand, yielding **70 question-answer pairs**. Each document was tagged with the degradation conditions it exhibits (17 free-text failure-mode tags in total), the most frequent being handwritten text (8 docs), degraded image quality, scanned document, and degraded text quality (5 each), plus dense text, fine print, faded or faint print, vertical text, skewed capture, and an upside-down image. Annotation was done in a purpose-built, self-contained browser annotator, exported as JSON (`custom_docs/annotations.json`), and converted by `scripts/setup/build_custom_tsv.py` into a DocVQA-format TSV so the identical evaluation pipeline and ANLS scoring apply unchanged. The TSV build also emits an index map (`results/custom_index_map.json`, regenerated from the annotations, not committed) linking every scored row back to its document and failure-mode tags, enabling per-condition diagnosis instead of a single aggregate number.

The results justified the effort: the custom set produced a 25 to 32 point ANLS drop and a change in model ranking that are invisible from the public benchmarks alone (Section 7).

## 4. Metrics and Experimental Setup

- **ANLS** (Average Normalized Levenshtein Similarity): standard for DocVQA-style extractive QA; tolerant of near-miss extractions (threshold 0.5). acc@0.5 is the fraction of answers at or above the threshold.
- **ECE (10-bin)**: confidence is the geometric mean token probability from generation logprobs; correctness is per-sample ANLS at or above 0.5. This measures whether the model knows when it is wrong, which decides whether confidence-thresholded auto-accept is safe.
- **Latency**: single-stream seconds per query (mean and median) and tokens per second on the T4.

**Serving:** all three VLMs were hosted with vLLM 0.24 (OpenAI-compatible API, fp16, identical flags) and evaluated through VLMEvalKit's API mode; greedy decoding (temperature 0) everywhere. Donut has no vLLM or VLMEvalKit support and used a custom HF driver with the same subsets, a prompt-equivalent task format, greedy decoding, and the same scoring path. Every model received the dataset's standard prompt including the "Answer the question using a single word or phrase." suffix (Donut uses its native task-token format, which is its equivalent). The pipeline was cross-validated: ANLS was computed twice per VLM (VLMEvalKit's evaluator and an independent logprobs-pass path) and agreed within 1 to 4 points on all six legs.

**Prompting comparison (custom set):** each VLM was run in two modes with everything else held fixed. `direct` uses the standard single-word-or-phrase prompt. `cot` instructs the model to reason step by step and finish with `Answer: <short answer>`; the final answer is parsed out for scoring and confidence is measured over the answer span only, not the reasoning. Donut has no chat capability, so it contributes a direct-mode result only.

## 5. Hardware Requirements

Everything runs on a single Google Colab free-tier instance; no paid compute was used.

- **GPU:** Tesla T4, 15GB VRAM. Serving is fp16 via vLLM at 0.85 GPU memory utilization; the largest model (about 1B parameters) fits comfortably with an 8192-token context. The T4 is a Turing part (SM75), so flash-attention 2 is unsupported and attention runs through SDPA.
- **System RAM:** 12GB, and this is the binding constraint, not VRAM. vLLM's multimodal processor cache must be disabled (`--mm-processor-cache-gb 0`) and client concurrency capped at 4, otherwise concurrent multi-megabyte infographic payloads get the engine OOM-killed.
- **Disk:** about 1.4GB of benchmark TSVs plus model weights and two conda environments (several GB more). Datasets and all results live on mounted Drive so nothing is lost when Colab reclaims the session.
- **Runtime:** the first Qwen3.5 server start pays roughly 12 minutes of Triton kernel autotuning (cached afterwards). Each pipeline stage completes within about an hour, and every stage is resumable after a disconnect.

## 6. Results on Public Benchmarks

| Model | DocVQA ANLS | InfoVQA ANLS | ECE Doc | ECE Info | Overconf. Info (conf minus acc) | p50 latency (s) | tok/s |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-0.8B** | **86.9** | **54.1** | **3.2** | **14.6** | +12.4 | **0.48** | 11.2 |
| InternVL3-1B | 83.2 | 50.9 | 5.3 | 24.9 | +31.3 | 1.61 | 7.1 |
| SmolVLM-500M | 61.8 | 23.3 | 15.3 | 40.5 | +40.5 | 0.99 | 7.3 |
| Donut (docvqa-ft) | 62.6* | 13.9 | 29.4 | 66.0 | +66.0 | 0.60 | 10.5 |

\* Donut's ANLS comes from the per-sample scoring path (identical formula); for the VLMs this path matched the main evaluator within 1 to 4 points. Donut is in-domain on DocVQA (fine-tuned on its training split).

**Discussion.**

1. **Qwen3.5-0.8B is the best sub-1B document model on every axis**: accuracy, calibration, and speed. Its DocVQA calibration is essentially honest (ECE 3.2, and slightly underconfident at -1.4), which means confidence-thresholded auto-accept is genuinely viable for clean DocVQA-style inputs.
2. **The dominant capability gap is layout and graphical reasoning, not text extraction.** Every model drops 32 to 38 ANLS points from DocVQA to InfoVQA under an identical protocol. Qwen's DocVQA score is near published 3B-class results while its InfoVQA score lags far behind, so the failure is joint reasoning over charts, layouts, and graphics, exactly what InfoVQA isolates.
3. **The InfoVQA failure mode is wrong-and-confident, the dangerous kind.** Overconfidence grows monotonically as capability falls (+12 for Qwen up to +66 for Donut). SmolVLM answers infographic questions with 78% mean confidence at 38% accuracy. Naive confidence gating is unsafe for infographic-style inputs on all models.
4. **Edge-native design did not pay off on documents.** SmolVLM-500M trails the generalists by 21 to 31 ANLS at comparable latency and is the worst-calibrated VLM in the set.
5. **The specialist is dominated by modern generalists.** Donut loses its home benchmark to Qwen3.5-0.8B by about 24 ANLS despite in-domain fine-tuning, collapses out of domain (13.9 on InfoVQA), has the worst calibration in the set (95% mean confidence at 66% accuracy on DocVQA), and is not faster than Qwen. Pretraining scale and recipe beat task-specific architecture here. (Caveat: single public checkpoint, not retrained.)
6. **Latency ordering is counter-intuitive**: the largest model (Qwen, 0.8B) is the fastest at 0.48 s median. Architecture and serving efficiency matter more than parameter count at this scale, and all four models are within interactive range on a free T4.

## 7. Results on the Custom Degraded-Document Set

All numbers below are on the 25-document, 70-question custom set; ANLS with the standard 0.5 threshold; greedy decoding; vLLM-served (Donut via the HF driver).

### 7.1 Overall: the benchmark-to-reality gap

| Model | DocVQA ANLS (public) | Custom ANLS (direct) | Drop | Custom acc@0.5 | Mean conf | Overconfidence |
|---|---|---|---|---|---|---|
| InternVL3-1B | 83.2 | **57.0** | -26.2 | **62.9** | 86.3 | +23.5 |
| Qwen3.5-0.8B | 86.9 | 54.8 | -32.1 | **62.9** | 79.5 | +16.6 |
| SmolVLM-500M | 61.8 | 40.3 | -21.5 | 51.4 | 86.9 | +35.5 |
| Donut (docvqa-ft) | 62.6 | 38.8 | -23.8 | 47.1 | 90.6 | +43.4 |

Three observations:

1. **Every model loses 21 to 32 ANLS points relative to DocVQA.** These are the same kinds of documents (invoices, forms, receipts, letters), differing mainly in capture quality. This is the central argument for the custom set: DocVQA scores materially overstate how these models behave on real captured documents, and the degradation robustness gap is roughly as large as the DocVQA-to-InfoVQA reasoning gap.
2. **The public-benchmark ranking does not survive contact with degraded input.** Qwen led InternVL3 by 3.7 points on DocVQA; on the custom set InternVL3 edges Qwen on ANLS (57.0 versus 54.8) and they tie exactly on acc@0.5 (62.9). At n=70 this reversal is within noise, so the honest claim is parity, but parity itself is the finding: Qwen's clean-benchmark advantage disappears. The direction is consistent with InternVL3's dynamic-tiling encoder preserving more detail on degraded, fine-print scans.
3. **Calibration on real documents looks like InfoVQA, not DocVQA.** Qwen's honest DocVQA confidence (ECE 3.2, overconfidence -1.4) does not transfer: on the custom set it is +16.6 points overconfident, and the others range from +23.5 to +43.4. A confidence threshold tuned on DocVQA would silently auto-accept wrong extractions on degraded input. Confidence gating must be calibrated on in-domain, in-condition data.

### 7.2 Direct versus chain-of-thought

| Model | Direct ANLS | CoT ANLS | Delta (CoT minus direct) | Direct acc@0.5 | CoT acc@0.5 |
|---|---|---|---|---|---|
| InternVL3-1B | 57.0 | 34.9 | **-22.1** | 62.9 | 40.0 |
| Qwen3.5-0.8B | 54.8 | 46.7 | -8.2 | 62.9 | 52.9 |
| SmolVLM-500M | 40.3 | 44.5 | +4.2 | 51.4 | 55.7 |

Chain-of-thought prompting is a net negative for extractive document QA at this scale. InternVL3 loses 22 ANLS points, Qwen loses 8, and only SmolVLM gains slightly (from the lowest base). Inspection of the CoT transcripts shows the expected failure pattern: the models reason their way past a correct extraction, substitute a paraphrase or a "corrected" value for what is literally printed, or produce reasoning that contaminates the parsed answer span. Extraction rewards reading precisely, not deliberating, and sub-1B decoders are not strong enough reasoners for deliberation to pay for its added failure surface.

One nuance: CoT helped exactly where reading fails outright. On the geometric failure modes (skewed and upside-down captures) Qwen went from 0% direct to 33% with CoT, suggesting reasoning can partially compensate when the raw percept is disoriented. That does not change the deployment recommendation (direct prompting), but it hints that orientation handling, not prompting, is the real fix.

### 7.3 Failure-mode breakdown

Per-condition acc@0.5 in direct mode (buckets overlap because documents carry multiple tags; per-bucket n ranges from 3 to 23, so treat these as directional):

- **Catastrophic, all models: geometric transforms.** Skewed captures and the upside-down document scored 0% for every model in direct mode. None of the four models has any rotation robustness. This is the single clearest actionable finding: a trivial deskew/orientation-correction preprocessing step would recover more accuracy than any model swap.
- **Very weak, all models: fine print (0 to 17%) and dense text (22 to 33%).** Small fonts and dense tabular layouts overwhelm the vision encoders at these parameter budgets, InternVL3's high-resolution tiling notwithstanding.
- **Middling: handwriting (52 to 65%) and general scan degradation (46 to 87% depending on severity bucket).** SmolVLM is comparatively strongest on handwriting (65%), the one bright spot for the edge-native design.
- **Strong for the top two models: layout-adjacent conditions.** Stylized logos, horizontal banner text, and vertical text score 100% for Qwen and InternVL3 while Donut fails vertical text completely (0%), consistent with a rigid reading-order prior in the specialist.
- **Donut trails the generalists on nearly every condition**, confirming the public-benchmark conclusion on private data it cannot have memorized.

### 7.4 What the custom set changed

Without the custom set, the report would have concluded "use Qwen3.5-0.8B, trust its confidence on DocVQA-style documents". Both halves of that sentence needed qualification: on realistically degraded documents Qwen no longer clearly leads, and its confidence is no longer trustworthy. The custom set also localized the damage to specific, fixable conditions (orientation, fine print, density) rather than a diffuse quality drop, which converts an evaluation result into an engineering roadmap.

## 8. Knowledge Gap and Improvement Strategy

**Gap (best model, Qwen3.5-0.8B):** two distinct weaknesses were identified. First, InfoVQA-style layout and graphical reasoning (54.1 versus 86.9 ANLS), compounded by overconfidence on exactly those inputs. Second, robustness to real-world capture degradation (54.8 on the custom set), where the cheapest wins are preprocessing (deskew and orientation correction, which alone address the 0% geometric buckets) rather than model changes.

**Proposed remediation, prepared and sanity-checked (execution descoped by decision):** rank-16 LoRA on the decoder's attention and MLP projections (6.4M trainable parameters, 0.74%), vision encoder frozen; 800 InfoVQA training samples strictly disjoint from the evaluation subset; prompts identical to evaluation; loss on answer tokens only; fp16 with gradient checkpointing fits the T4 at 8.1GB. Post-training evaluation re-runs the identical InfoVQA subset (target: beat 54.1) plus DocVQA as a regression check (hold approximately 86.9). All assets (training set, training script validated by a one-step forward/backward check, serve-and-eval script) were prepared and remain archived on the project drive; they are not part of the repository since the run was descoped. The custom set now additionally provides a held-out robustness check for any future adaptation, and its failure-mode tags define what a degradation-focused fine-tuning set should contain (rotated/skewed augmentations, fine print, dense tables).

## 9. Limitations

- 300-sample public-benchmark subsets (roughly plus or minus 3 to 5 point CI); validation splits only.
- The custom set is small (25 documents, 70 QA pairs) and single-annotator; per-failure-mode buckets run from n=3 to n=23 and tags are free text (including some near-duplicate labels), so per-condition numbers are directional, not precise. Overall custom-set deltas (20+ points) far exceed plausible noise.
- The LoRA training pool shares the InfoVQA validation distribution (the official training split requires registration); disjoint at the sample level. Sufficient for a proof-of-concept claim, not a leaderboard claim.
- The Donut comparison uses the single public checkpoint without retraining.
- Latency was measured on one hardware target (T4, fp16, vLLM/HF); rankings may shift on NPUs or CPUs.
- ECE uses sequence-level geometric-mean token probability; other confidence estimators (for example verbalized confidence) were not tested.

## 10. Reproducibility and Engineering Notes

All code, data manifests, and results live in the repository, with scripts grouped by pipeline stage: `scripts/setup/` (environments, model registration, subset and custom-TSV builds), `scripts/inference/` (main eval, prompting, ECE/latency, Donut drivers), and `scripts/eval/` (analysis and summaries). The custom-set pipeline is fully re-runnable: `setup/build_custom_tsv.py`, `inference/run_custom.sh`, `inference/donut_custom.py`, `eval/analyze_custom.py`.

### Results directory

`results/` holds one folder per model with the raw per-sample records:

```
results/
  <model>/                           per-sample JSONL records for each model:
    DocVQA_VAL_SUB300_ece.jsonl      confidence + latency pass on DocVQA (one line per
                                     sample: prediction, answer, confidence, latency)
    InfoVQA_VAL_SUB300_ece.jsonl     same for InfoVQA
    CustomDocVQA_direct.jsonl        custom-set predictions, direct prompting
    CustomDocVQA_cot.jsonl           custom-set predictions, chain-of-thought (VLMs only;
                                     Donut has no chat capability)
```

The analysis scripts regenerate aggregate tables at runtime (`summary`, `custom_summary`, the failure-mode breakdown, and the row-to-document index map); those files are not committed, since every number they contain is reproduced in this report (Sections 6 and 7, and the Appendix in full).

### Engineering notes

Notable findings from the runs: the T4 requires SDPA (flash-attention 2 unsupported on Turing); InternVL3's remote code is incompatible with transformers 5.x (vLLM's native implementation sidesteps it); vLLM's default 4GB multimodal cache plus concurrent infographic payloads OOM-kills the engine on 12GB-RAM Colab (`--mm-processor-cache-gb 0` fixes it); Qwen3.5's linear-attention Triton autotune costs about 12 minutes on first start (cached thereafter); VLMEvalKit's `process_line(anls)` returns distances, not similarities (`hit_calculate` applies the threshold); and shell quoting of vLLM's `--mm-processor-kwargs` JSON must survive bash array expansion (an over-escaped form silently broke the Qwen server launch and was fixed in `inference/run_custom.sh`).

## 11. Recommendation

For on-device document pipelines in this compute class, **Qwen3.5-0.8B remains the best single-model foundation**, but with two qualifications the public benchmarks alone would have missed. First, add orientation and deskew preprocessing before inference; this is the highest-leverage fix available, recovering failure modes where every model currently scores zero. Second, do not reuse DocVQA-calibrated confidence thresholds on captured documents; calibration must be set per input condition, and infographic-style inputs should be routed to human review or a larger model regardless of reported confidence. Use direct prompting, not chain-of-thought, for extraction. The prepared LoRA adaptation remains the lowest-cost path to closing the layout-reasoning gap, and the custom set doubles as its robustness regression suite.

## 12. Appendix: Full Result Tables

These are the complete tables the analysis scripts emit; the body of the report quotes selected columns from them.

### A.1 Public benchmarks, all metrics

300-sample fixed-seed subsets, greedy decoding, vLLM-served on the T4. `ANLS (main)` is VLMEvalKit's evaluator; `ANLS (ECE pass)` is the independent per-sample scoring path used for calibration (the two agree within 1 to 4 points, cross-validating the pipeline). Confidence is the geometric mean token probability; overconfidence is mean confidence minus acc@0.5.

| Model | Dataset | ANLS (main) | ANLS (ECE pass) | acc@0.5 | Mean conf | ECE | Overconf. | Lat mean (s) | Lat p50 (s) | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | DocVQA | 86.89 | 86.71 | 89.00 | 87.63 | 3.22 | -1.37 | 0.51 | 0.48 | 11.2 |
| Qwen3.5-0.8B | InfoVQA | 54.10 | 54.62 | 59.33 | 71.69 | 14.56 | +12.36 | 0.48 | 0.45 | 9.3 |
| InternVL3-1B | DocVQA | 83.23 | 83.47 | 85.67 | 90.92 | 5.25 | +5.25 | 1.33 | 1.61 | 7.1 |
| InternVL3-1B | InfoVQA | 50.87 | 50.88 | 58.00 | 82.93 | 24.93 | +24.93 | 0.90 | 0.79 | 11.7 |
| SmolVLM-500M | DocVQA | 61.76 | 64.87 | 75.33 | 90.60 | 15.27 | +15.27 | 1.01 | 0.99 | 7.3 |
| SmolVLM-500M | InfoVQA | 23.33 | 27.57 | 38.00 | 78.46 | 40.46 | +40.46 | 0.78 | 0.71 | 8.3 |
| Donut-DocVQA | DocVQA | n/a* | 62.57 | 65.67 | 95.11 | 29.44 | +29.44 | 0.62 | 0.60 | 10.5 |
| Donut-DocVQA | InfoVQA | n/a* | 13.89 | 18.33 | 84.37 | 66.03 | +66.03 | 0.72 | 0.67 | 6.8 |

\* Donut runs outside VLMEvalKit (custom HF driver), so only the per-sample scoring path applies.

### A.2 Custom set, all metrics

25 documents, 70 QA pairs, ANLS with the standard 0.5 threshold, greedy decoding.

| Model | Mode | n | ANLS | acc@0.5 |
|---|---|---|---|---|
| Qwen3.5-0.8B | direct | 70 | 54.84 | 62.86 |
| Qwen3.5-0.8B | cot | 70 | 46.69 | 52.86 |
| InternVL3-1B | direct | 70 | 57.03 | 62.86 |
| InternVL3-1B | cot | 70 | 34.91 | 40.00 |
| SmolVLM-500M | direct | 70 | 40.30 | 51.43 |
| SmolVLM-500M | cot | 70 | 44.45 | 55.71 |
| Donut-DocVQA | direct | 70 | 38.84 | 47.14 |

### A.3 Custom set, acc@0.5 by failure mode (direct)

Failure-mode tags are free text from the annotation pass, so near-duplicate labels appear (for example "degraded image" vs "degraded image quality" and one misspelling); buckets overlap because documents carry multiple tags, and per-bucket n is small (3 to 23), so read these as directional.

| Failure mode | n | Donut-DocVQA | InternVL3-1B | Qwen3.5-0.8B | SmolVLM-500M |
|---|---|---|---|---|---|
| degarded image quality | 3 | 33.3 | 33.3 | 33.3 | 33.3 |
| degraded image | 12 | 50.0 | 75.0 | 75.0 | 58.3 |
| degraded image quality | 15 | 66.7 | 86.7 | 80.0 | 66.7 |
| degraded text quality | 11 | 45.5 | 63.6 | 63.6 | 45.5 |
| dense text | 9 | 33.3 | 22.2 | 33.3 | 33.3 |
| faded text | 3 | 33.3 | 33.3 | 66.7 | 66.7 |
| faint print | 3 | 33.3 | 66.7 | 33.3 | 100.0 |
| fine print | 6 | 16.7 | 0.0 | 16.7 | 16.7 |
| handwritten text | 23 | 52.2 | 60.9 | 60.9 | 65.2 |
| horizontal text | 8 | 62.5 | 100.0 | 100.0 | 50.0 |
| scanned document | 11 | 45.5 | 63.6 | 63.6 | 45.5 |
| skewed | 3 | 0.0 | 0.0 | 0.0 | 0.0 |
| striked out text | 3 | 66.7 | 66.7 | 66.7 | 66.7 |
| stylized logo | 8 | 62.5 | 100.0 | 100.0 | 50.0 |
| unclear text | 3 | 66.7 | 66.7 | 66.7 | 33.3 |
| upside down image | 3 | 0.0 | 0.0 | 0.0 | 0.0 |
| vertical text | 3 | 0.0 | 100.0 | 100.0 | 66.7 |

### A.4 Custom set, acc@0.5 by failure mode (chain-of-thought)

Donut has no chat capability, so no CoT column for it.

| Failure mode | n | InternVL3-1B | Qwen3.5-0.8B | SmolVLM-500M |
|---|---|---|---|---|
| degarded image quality | 3 | 33.3 | 33.3 | 33.3 |
| degraded image | 12 | 66.7 | 58.3 | 58.3 |
| degraded image quality | 15 | 40.0 | 66.7 | 80.0 |
| degraded text quality | 11 | 45.5 | 45.5 | 63.6 |
| dense text | 9 | 11.1 | 11.1 | 33.3 |
| faded text | 3 | 33.3 | 66.7 | 66.7 |
| faint print | 3 | 66.7 | 33.3 | 66.7 |
| fine print | 6 | 16.7 | 0.0 | 16.7 |
| handwritten text | 23 | 39.1 | 52.2 | 65.2 |
| horizontal text | 8 | 50.0 | 100.0 | 50.0 |
| scanned document | 11 | 45.5 | 45.5 | 63.6 |
| skewed | 3 | 0.0 | 33.3 | 0.0 |
| striked out text | 3 | 0.0 | 33.3 | 66.7 |
| stylized logo | 8 | 50.0 | 100.0 | 50.0 |
| unclear text | 3 | 66.7 | 66.7 | 33.3 |
| upside down image | 3 | 0.0 | 33.3 | 0.0 |
| vertical text | 3 | 66.7 | 66.7 | 66.7 |
