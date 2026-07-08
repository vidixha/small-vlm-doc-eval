# Part 2 §1 — Knowledge Gap Analysis

Full metric table: `summary.csv` / `summary.md` (300-sample fixed-seed subsets,
greedy, vLLM-served, T4). Cross-check: the independent ECE pass reproduces the
main eval's ANLS within 1–4 points on all 6 legs — pipelines agree.

| Model | DocVQA ANLS | InfoVQA ANLS | ECE Doc | ECE Info | conf−acc (Info) | p50 lat (s, Doc) | tok/s |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-0.8B** | **86.9** | **54.1** | **3.2** | **14.6** | +12.4 | **0.48** | 11.2 |
| InternVL3-1B | 83.2 | 50.9 | 5.3 | 24.9 | +31.3 | 1.61 | 7.1 |
| SmolVLM-500M | 61.8 | 23.3 | 15.3 | 40.5 | +40.5 | 0.99 | 7.3 |

## Findings

1. **Best model: Qwen3.5-0.8B, on every axis.** Highest ANLS on both benchmarks,
   best-calibrated (DocVQA ECE 3.2 — essentially honest; overconfidence −1.4,
   i.e. *slightly under*confident), and fastest single-stream (0.48 s/query).
   The "newest design" hypothesis holds: its OCR-oriented pretraining shows.

2. **The dominant gap is layout/visual reasoning, not text extraction.** All
   three models drop 32–38 ANLS points from DocVQA to InfoVQA on identical
   protocol. For Qwen the DocVQA ceiling (86.9) is near published 3B-class
   scores, while InfoVQA (54.1) lags far behind — the failure mode is joint
   reasoning over charts/layouts/graphics, exactly what InfoVQA isolates.

3. **Wrong-and-confident is the InfoVQA failure mode (the dangerous kind).**
   ECE degrades in lockstep with capability: every model is markedly
   overconfident on InfoVQA (confidence exceeds accuracy by +12 to +40 points).
   SmolVLM answers infographic questions with 78% mean confidence at 38%
   accuracy. For an on-device extraction pipeline this rules out naive
   confidence-thresholded auto-accept for infographic-style inputs — except for
   Qwen on DocVQA-style documents, where confidence is actually trustworthy.

4. **Edge-native ≠ document-capable.** SmolVLM-500M trails by 21–31 ANLS
   despite comparable latency, and is the worst-calibrated. Architecture
   designed for constraint did not hold up on document tasks against
   general-purpose designs in the same bracket.

5. **Specialist baseline (added later): Donut-DocVQA (~200M, OCR-free,
   DocVQA-train fine-tuned; custom HF driver, same subsets/scoring).**
   DocVQA ANLS 62.6 / acc@0.5 65.7 — despite its *in-domain training
   advantage*, it only matches SmolVLM and trails the modern generalists by
   ~24 points. InfoVQA (out-of-domain for it) collapses to 13.9. Calibration is
   the worst of the set: 95.1% mean confidence on DocVQA (ECE 29.4) and +66
   overconfidence on InfoVQA. Latency is good (0.60 s p50) but not better than
   Qwen3.5-0.8B. Takeaway: the 2022-era task-specific architecture is dominated
   on every axis by a 2026 sub-1B generalist — pretraining scale/recipe beat
   task-specific fine-tuning here. (Caveat: single public checkpoint, not
   retrained; a Donut retrained on both tasks would look better on InfoVQA but
   the DocVQA in-domain comparison is already conclusive.)

## §2 Improvement strategy (per plan): LoRA on Qwen3.5-0.8B targeting InfoVQA-style
layout/graphical reasoning — the largest, most dangerous (overconfident) gap.

- **Train data**: InfoVQA_VAL rows *disjoint* from the 300-sample eval subset
  (the eval subset is never trained on). Caveat (documented): same distribution
  as eval since the official train split's images require registration to
  download; acceptable for a PoC whose claim is "targeted adaptation moves the
  weak metric without regressing DocVQA".
- **Eval**: post-LoRA re-run of InfoVQA_SUB300 (target: beat 54.1) and
  DocVQA_SUB300 (regression check vs 86.9), identical protocol.
