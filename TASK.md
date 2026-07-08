# Adapting Small VLMs for Document Understanding — Task Plan

## Problem Statement

Vision-language models under 1B parameters have shown strong general visual reasoning ability, making them attractive for on-device and edge deployment where compute and memory are tightly constrained. However, their performance on **document understanding** — extracting and reasoning over information in forms, invoices, reports, and infographics — remains largely uncharacterized, since most published evaluations focus on models in the 3B+ range.

This gap matters specifically for edge deployment use cases: document understanding is a common real-world workload (receipt scanning, form processing, on-device OCR-adjacent tasks), and if small VLMs are to serve as a viable foundation for this domain, their capabilities and failure modes need to be systematically characterized rather than assumed from general-purpose VQA benchmarks alone. Generic VQA accuracy does not capture what actually matters for document understanding — dense text extraction, layout-aware reasoning, and reliability under real-world image degradation — so evaluation needs to go beyond a single aggregate accuracy number.

This work addresses that gap in two parts: (1) a systematic evaluation of three architecturally distinct sub-1B VLMs — Qwen3.5-0.8B (newest general-purpose design), InternVL3-1B (resolution-optimized encoder), and SmolVLM-500M-Instruct (edge-native design) — across document-specific benchmarks (DocVQA, InfoVQA) and metrics beyond accuracy (ANLS, calibration via ECE, latency); and (2) an evidence-based improvement strategy targeting the specific knowledge gap of the best-performing model, using a parameter-efficient adaptation method grounded in the observed failure pattern.

## Part 1 — Evaluation

### 1. Model Selection

| Model | Params | Vision Encoder | Decoder | Why included |
|---|---|---|---|---|
| **Qwen3.5-0.8B** | 0.8B | Native Qwen vision encoder (high-res tiling, strong OCR-oriented pretraining) | Qwen3.5 LLM | Newest entrant in the sub-1B bracket; Alibaba explicitly targets document/text-in-image understanding as a design goal for this release, not just general VQA. Included to test whether "newest" translates to real document-task gains vs. older architectures. |
| **InternVL3-1B** | ~1B | InternViT (dynamic high-resolution tiling, pixel-shuffle downsampling) | Qwen2.5-0.5B | InternViT's tiling strategy is specifically built to preserve fine-grained detail in high-res inputs — directly relevant to dense text/small-font document images, where naive fixed-resolution encoders lose detail. Chosen to test whether encoder-side resolution handling is the dominant factor in document performance. |
| **SmolVLM-500M-Instruct** | 500M | SigLIP | SmolLM2-360M | Purpose-built for edge/on-device deployment — directly aligned with the on-device / edge deployment use case, unlike the other two which are general-purpose VLMs scaled down. Included as the "designed for constraint" reference point: does an architecture explicitly optimized for small-footprint deployment actually hold up on document tasks against larger-decoder competitors in the same parameter bracket? |

**Decoder-family note:** All three models use distinct decoder families (Qwen3.5, Qwen2.5-0.5B, SmolLM2-360M) and distinct encoders (native Qwen vision encoder, InternViT, SigLIP) — genuine architectural diversity across the set, so results can't be attributed to a single shared component. This also gives a clean three-way contrast: newest general-purpose design (Qwen3.5) vs. resolution-optimized encoder (InternVL3) vs. edge-native design (SmolVLM).

### 2. Benchmark Selection

- **DocVQA** (VLMEvalKit built-in) — canonical document VQA benchmark: real-world forms, invoices, reports, letters. Chosen for comparability with published literature/leaderboards.
- **InfoVQA** (VLMEvalKit built-in) — infographic-style documents requiring joint reasoning over text, layout, and graphical/visual elements together, not just extractable text. Chosen to complement DocVQA: DocVQA is largely text-extraction-heavy, InfoVQA forces layout+visual reasoning, giving benchmark diversity beyond a single dataset's biases.
- **(Optional, time-permitting) Custom 15-20 sample set** — targeting degraded scans / dense tabular layouts, motivated directly by production OCR evaluation experience. Justification: neither DocVQA nor InfoVQA explicitly stress-test *robustness to degradation*, which is the actual failure mode that matters for real document pipelines and is the gap the brief explicitly asks evaluation to probe "beyond generic VQA accuracy."

### 3. Evaluation Metrics

- **ANLS** (Average Normalized Levenshtein Similarity) — standard for DocVQA/InfoVQA; tolerant of near-matches in extracted text, appropriate for extractive QA.
- **ECE (Expected Calibration Error)** — measures whether model confidence tracks correctness. Relevant because a document-understanding system deployed on-device (the target edge use case) needs to know when to defer/flag uncertain extractions rather than silently hallucinating a wrong field value.
- **Robustness delta** (if custom set is built) — accuracy drop under synthetic degradation (blur, rotation, JPEG compression) vs. clean input. Relevant because real-world document capture (phone photos, scans) is rarely clean.
- **Latency** (tokens/sec or sec/query on the eval hardware) — directly relevant to edge/on-device deployment, which is the stated on-device / edge product context.

### 4. Experimental Setup & Reproducibility

- **Hardware:** Google Colab free tier (T4 GPU, 16GB VRAM) — chosen per task's explicit allowance for free GPU resources; consistent across all 3 models for fair comparison.
- **Inference config:** Same generation config across all models — greedy decoding (temperature=0), fixed max_new_tokens, fixed image preprocessing resolution/aspect handling per each model's documented defaults (no manual resolution overrides that would advantage one model).
- **Preprocessing:** Identical DocVQA/InfoVQA sample subset (same seed, same N samples) fed to all 3 models — no per-model dataset shuffling.
- **Reproducibility:** All scripts, configs, and the exact sample indices used are checked into the submitted repo; a fixed `--seed` flag ensures another engineer gets identical subset sampling.

### 5. Software Stack

- **VLMEvalKit** — chosen because it already implements DocVQA/InfoVQA loaders, ANLS scoring, and has registered configs for all 3 candidate models (Qwen3.5-0.8B, InternVL3-1B, SmolVLM-500M-Instruct), minimizing custom glue code within the time budget available.
- **transformers / accelerate** — standard HF inference stack, compatible with all 3 models without custom kernels.
- **timm** — required for InternVL3's InternViT encoder loading.

---

## Part 2 — Knowledge Gap & Improvement Strategy

### 1. Knowledge Gap Analysis (fill in after results)
- Identify which benchmark (DocVQA vs InfoVQA) shows the largest gap for the best model — text-extraction failure vs layout/visual-reasoning failure.
- Cross-reference against ECE: is the model *wrong and confident* (dangerous) or *wrong and uncertain* (safer failure mode)?

### 2. Improvement Strategy
- **Proposed approach:** LoRA (parameter-efficient fine-tuning) on the best-performing model, targeting the identified weak task type, using a small curated set of document-domain examples (few hundred to low-thousands) reflecting the specific failure pattern found.
- **Justification:** Full fine-tuning is infeasible in the time/compute budget and risks catastrophic forgetting of general VQA ability; LoRA preserves base capability while adapting to the narrow domain gap, and is the standard efficient-adaptation approach for sub-1B models per current literature.

### 3. Expected Outcomes
- Expected ANLS improvement on the specific weak subset (e.g. dense tables or degraded scans) post-LoRA, measured on a held-out portion of the same benchmark not used in fine-tuning, to confirm genuine improvement rather than overfitting to the tuning set.

---

## Colab Setup Instructions (for execution agent)

**Cell 1 — GPU check**
```python
!nvidia-smi
```
If no GPU shown: Runtime → Change runtime type → T4 GPU → Save, then rerun.

**Cell 2 — Mount Drive (survive disconnects)**
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/eval_work', exist_ok=True)
```

**Cell 3 — Clone + install VLMEvalKit**
```python
!git clone https://github.com/open-compass/VLMEvalKit.git
%cd VLMEvalKit
!pip install -e . -q
!pip install timm einops -q
```

**Cell 4 — Check exact registered model names**
Do not guess model name strings — VLMEvalKit's registered keys don't always match HF repo names.
```python
!grep -i "smolvlm\|internvl3\|qwen3" vlmeval/config.py
```
This returns the exact `--model` string to use for each of the 3 candidate models. If any of the 3 don't appear, VLMEvalKit doesn't have a built-in config for it yet — find the closest existing entry in `vlmeval/config.py` and adapt it (same file, add a new dict entry following the existing pattern for a similar model).

**Cell 5 — One-sample smoke test per model (mandatory before full run)**
```python
!python run.py --model <exact_name_from_config> --data DocVQA_VAL --limit 5 --work-dir /content/drive/MyDrive/eval_work
```
Run once per model with `--limit 5` first. Fix any crash here before proceeding — do not discover a broken config after a full-length run.

**Cell 6 — Full runs (DocVQA + InfoVQA, all 3 models)**
```python
for model in ["<model1>", "<model2>", "<model3>"]:
    !python run.py --model {model} --data DocVQA_VAL InfoVQA_VAL --work-dir /content/drive/MyDrive/eval_work
```
Runs all 6 (model × benchmark) combinations sequentially. Results are written to the Drive-mounted work-dir so a dropped Colab session doesn't lose completed results.

**Operational notes:**
- Free-tier T4 sessions can be reclaimed under load or hit the ~12hr session cap — `--work-dir` must point at the Drive path (not local `/content/`) so partial results survive a disconnect.
- If `InternVL3-1B` is not yet registered in `vlmeval/config.py` (newer release, configs can lag), manually add a config entry adapted from the closest existing InternVL entry before running Cell 5 for that model.
- Do not skip Cell 5 for any model to save time — a config or dependency error caught late costs more time than the smoke test itself.

---

## Timeline (today, deadline day)
1. Colab setup + VLMEvalKit install (30 min)
2. Run all 3 models on DocVQA + InfoVQA subset (core eval — prioritize this)
3. Build comparison table + write interpretation
4. LoRA PoC on best model (keep minimal — a working proof, not a full sweep)
5. Write technical report PDF from this task.md as the skeleton
