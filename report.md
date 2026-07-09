# Adapting Small VLMs for Document Understanding
### A Systematic Evaluation of Sub-1B Vision-Language Models on Document Benchmarks and a Custom Dataset


## 1. Problem Statement

Vision-language models under 1B parameters show strong general visual reasoning, which makes them effective for on-device and edge deployment where compute and memory are tightly constrained. However, their performance on document understanding (extracting and reasoning over forms, invoices, reports, and infographics) is largely uncharacterized and published evaluations focus on models in the 3B+ range, and generic VQA accuracy does not capture what document pipelines actually fail on. 

This work addresses the gap in three parts:

1. A systematic evaluation of architecturally distinct sub-1B VLMs on document benchmarks, with metrics beyond accuracy (calibration and latency).
2. A custom, hand-annotated evaluation set probing robustness to real-world document degradation, which public benchmarks do not isolate.
3. An evidence-based, parameter-efficient improvement strategy targeted at the observed failure pattern.

Four models were evaluated on document understanding under an identical, reproducible protocol: three sub-1B general-purpose vision-language models (**Qwen3.5-0.8B**, **InternVL3-1B**, **SmolVLM-500M-Instruct**) and a task-specific specialist baseline, **Donut** (donut-base-finetuned-docvqa, about 200M parameters). Evaluation covered two public benchmarks (DocVQA, InfoVQA) and a hand-annotated custom set of 25 real documents (70 QA pairs) targeting degradation conditions the public benchmarks do not test. Metrics go beyond accuracy: ANLS, calibration (ECE from token logprobs), and latency.


## 2. Model Choice

| Model                                                                                          | Params | Vision Encoder                                      | Decoder      | Why do we evaluate this?                                                                                                                   |
| ---------------------------------------------------------------------------------------------- | -----: | --------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| [Qwen3.5-VL-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)                                          |   0.8B | Native Qwen vision encoder (high-resolution tiling) | Qwen3.5      | Represents the latest generation of sub-1B VLMs, with document understanding being one of its primary target applications.                |
| [InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B)                                          |    ~1B | InternViT (dynamic tiling)                          | Qwen2.5-0.5B | Included to evaluate the effect of a strong, high-resolution vision encoder on document understanding under a similar parameter budget.   |
| [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)            |   0.5B | SigLIP                                              | SmolLM2-360M | A lightweight VLM designed for resource-constrained environments, providing a representative small-model baseline.                        |
| [Donut (DocVQA fine-tuned)](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa) |  ~0.2B | Swin Transformer                                    | BART         | An OCR-free document understanding model specialized for DocVQA. It serves as a task-specific baseline rather than a general-purpose VLM. |

The three VLMs share no encoder or decoder family, so results cannot be attributed to a single shared component. We evaluate the newest general-purpose design (Qwen3.5) versus resolution-optimized encoder (InternVL3) versus edge-native design (SmolVLM). Donut is added to evaluate whether a task-specific document architecture can beat generalist models on a task it has specifically been finetuned for.

The main findings:

1. **Qwen3.5-0.8B wins the public benchmarks on every metric**: best accuracy (DocVQA 86.9, InfoVQA 54.1 ANLS), best calibration (DocVQA ECE 3.2), fastest inference (0.48 s/query median).
2. **Public benchmark scores overstate real-world robustness by roughly 25 to 30 ANLS points.** On the custom degraded-document set every model drops sharply (Qwen 86.9 to 54.8; InternVL3 83.2 to 57.0), the ranking between the top two models flips into a tie, and confidence calibration degrades to the point where naive confidence gating would be unsafe.
3. **Chain-of-thought prompting hurts extractive document QA at this scale.** CoT cost InternVL3-1B 22 ANLS points and Qwen3.5-0.8B 8 points on the custom set relative to direct prompting.


## 3. Data Choice

### 3.1 Public benchmarks

- **DocVQA (validation split)**: A widely used benchmark for document visual question answering, containing real-world forms, invoices, reports, and letters. It was chosen because it provides a diverse set of document types and enables direct comparison with prior work on document understanding models.
- **InfoVQA (validation split)**: A benchmark based on infographic documents that require reasoning over text, layout, and graphical elements. It complements DocVQA by evaluating a model's ability to interpret document structure and visual information in addition to reading text.

**Subsets:** identical fixed-seed (seed 42) 300-sample subsets per benchmark for every model; `scripts/setup/make_subsets.py` regenerates the exact same samples deterministically. At n=300 the 95% confidence interval on accuracy-type metrics is roughly plus or minus 3 to 5 points; the gaps observed (21 to 38 points) far exceed it.

### 3.2 Why a custom dataset was needed

Neither DocVQA nor InfoVQA stress-tests the conditions that actually break production document pipelines. Their images are mostly clean, well-digitized documents. Documents that are captured through phone photos and office scans arrive skewed, rotated, blurred, faded, and full of handwriting, fine print, and dense tables. A model can score 87 ANLS on DocVQA and still fail on the receipts a user actually photographs. There is also a familiarity concern: both benchmarks are public and widely used, so their images may overlap model pretraining data; a private set cannot be memorized.

This dataset was created based on a gap I have observed in SOTA Multimodal DocumentAI systems. In my earlier work on financial-document OCR (SAVIOR [1]), I showed that OCR pipelines and vision-language models with high scores on standard document benchmarks systematically underperform on document patterns that are operationally critical in real workflows: vertical or rotated text, logo-embedded vendor names, fine-print clauses, degraded scans, and complex multi-column layouts. These patterns are underrepresented in public datasets yet account for a substantial share of real-world failure cases.

Based on that experience, I built and hand-annotated a custom set of **25 real documents**, yielding **70 question-answer pairs**, and tagged every document with the degradation conditions it exhibits using the failure taxonomy from SAVIOR [1] (17 free-text failure-mode tags in total). The most frequent tags are handwritten text (8 docs), degraded image quality, scanned document, and degraded text quality (5 each), plus dense text, fine print, faded or faint print, vertical text, stylized logos, skewed capture, and an upside-down image; these map directly onto the high-impact patterns identified in the paper. Annotation was done in a purpose-built, self-contained browser annotator, exported as JSON (`custom_docs/annotations.json`), and converted by `scripts/setup/build_custom_tsv.py` into a DocVQA-format TSV so the identical evaluation pipeline and ANLS scoring apply unchanged. The TSV build also emits an index map (`results/custom_index_map.json`, regenerated from the annotations, not committed) linking every scored row back to its document and failure-mode tags, enabling per-condition diagnosis instead of a single aggregate number.

As expected, the custom set produced a 25 to 32 point ANLS drop and a change in model ranking that are invisible from the public benchmarks alone (Section 7).

## 4. Metrics and Experimental Setup

- **ANLS** (Average Normalized Levenshtein Similarity): standard metric for DocVQA-style extractive QA and tolerant of near-miss extractions (threshold 0.5). acc@0.5 is the fraction of answers at or above the threshold.
- **ECE (10-bin)**: confidence is the geometric mean token probability from generation logprobs; correctness is per-sample ANLS at or above 0.5. This measures whether the model knows when it is wrong, which decides whether confidence-thresholded auto-accept is safe.
- **Latency**: single-stream seconds per query (mean and median) and tokens per second on the T4.

**Serving and inference:** all three VLMs were hosted with vLLM 0.24 (OpenAI-compatible API, fp16, identical flags) and evaluated through VLMEvalKit's API mode; greedy decoding (temperature 0) everywhere. Donut has no vLLM or VLMEvalKit support and used a custom HF driver with the same subsets, a prompt-equivalent task format, greedy decoding, and the same scoring path. Every model received the dataset's standard prompt including the "Answer the question using a single word or phrase." suffix (Donut uses its native task-token format, which is its equivalent). The pipeline was cross-validated: ANLS was computed twice per VLM (VLMEvalKit's evaluator and an independent logprobs-pass path) and agreed within 1 to 4 points on all six legs.

**Prompting comparison (custom set):** To evaluate whether explicit reasoning benefits small VLMs, each chat-capable model was evaluated under two prompting strategies while keeping the model, decoding parameters, and inputs fixed. In the direct setting, the model was asked to provide only a short answer. In the CoT setting, the model was instructed to reason step by step before concluding with ```Answer: <short answer>```. For evaluation, only the final extracted answer was scored, and confidence was computed over the answer span rather than the reasoning text. Since Donut is not a chat-based model and does not support reasoning prompts, it was evaluated only in the direct setting.

## 5. Hardware Requirements


- **GPU:** Colab GPU ; Tesla T4, 15GB VRAM. Model serving is fp16 via vLLM at 0.85 GPU memory utilization; the largest model (about 1B parameters) fits comfortably with an 8192-token context.
- **System RAM:** 12GB, and this is the binding constraint, not VRAM. vLLM's multimodal processor cache must be disabled (`--mm-processor-cache-gb 0`) and client concurrency capped at 4, otherwise concurrent payloads cause Out of Memory issues.

## 6. Results on Public Benchmarks

| Model | DocVQA ANLS | InfoVQA ANLS | ECE Doc | ECE Info | Overconf. Info (conf minus acc) | p50 latency (s) | tok/s |
|---|---|---|---|---|---|---|---|
| **Qwen3.5-0.8B** | **86.9** | **54.1** | **3.2** | **14.6** | +12.4 | **0.48** | 11.2 |
| InternVL3-1B | 83.2 | 50.9 | 5.3 | 24.9 | +31.3 | 1.61 | 7.1 |
| SmolVLM-500M | 61.8 | 23.3 | 15.3 | 40.5 | +40.5 | 0.99 | 7.3 |
| Donut (docvqa-ft) | 62.6* | 13.9 | 29.4 | 66.0 | +66.0 | 0.60 | 10.5 |

\* Donut's ANLS comes from the per-sample scoring path (identical formula); for the VLMs this path matched the main evaluator within 1 to 4 points. Donut is in-domain on DocVQA (fine-tuned on its training split).

**Discussion and analysis**

1. **Qwen3.5-0.8B consistently performs the best.** It achieves the highest accuracy, has the most reliable confidence estimates, and is also the fastest model evaluated.
2. **All models find infographic documents much harder than standard documents.** Performance drops substantially on InfoVQA compared to DocVQA, showing that reasoning over layouts, charts, and graphics remains a major challenge.
3. **Confidence is not always reliable on difficult documents.** On InfoVQA, models often give incorrect answers with high confidence, making confidence alone a poor indicator of correctness.
4. **Smaller or document-specific models do not outperform modern general-purpose VLMs.** Both SmolVLM and Donut lag behind Qwen3.5-0.8B across most metrics, despite being designed for lightweight deployment or document understanding.
5. **Model architecture matters as much as model size.** Although Qwen3.5-0.8B is the largest model in this comparison, it is also the fastest, showing that efficient architectures can deliver both higher accuracy and lower latency.

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

1. **Performance drops noticeably on the custom dataset.** Every model scores much lower than on DocVQA, showing that real captured documents with noise, blur, and other degradations are substantially more challenging than standard benchmark documents.
2. **The benchmark ranking changes on real-world data.** While Qwen3.5-0.8B performs best on DocVQA, it performs similarly to InternVL3-1B on the custom dataset. This suggests that strong benchmark performance does not always translate to real document images.
3. **Confidence estimates become less reliable.** All models are more overconfident on the custom dataset than on DocVQA, indicating that confidence thresholds should be calibrated using data that closely matches the intended deployment setting.
   
### 7.2 Direct versus chain-of-thought

| Model | Direct ANLS | CoT ANLS | Delta (CoT minus direct) | Direct acc@0.5 | CoT acc@0.5 |
|---|---|---|---|---|---|
| InternVL3-1B | 57.0 | 34.9 | **-22.1** | 62.9 | 40.0 |
| Qwen3.5-0.8B | 54.8 | 46.7 | -8.2 | 62.9 | 52.9 |
| SmolVLM-500M | 40.3 | 44.5 | +4.2 | 51.4 | 55.7 |

Chain-of-thought prompting does not improve extractive document question answering for the models evaluated. Qwen3.5-0.8B and InternVL3-1B both perform worse with CoT prompting, while SmolVLM shows only a small improvement from a much lower baseline. In many cases, the additional reasoning causes models to modify or paraphrase information instead of copying the text exactly, leading to incorrect final answers.
One exception is heavily distorted images. For rotated or skewed documents, CoT occasionally helps the model recover the correct answer when direct prompting fails. However, these gains are limited and do not outweigh the overall drop in performance. For document extraction tasks, direct prompting remains the more reliable choice.

### 7.3 Failure-mode breakdown
Performance on the custom dataset varies considerably across different document conditions:

- **Geometric distortions are the most challenging.** All models fail on heavily skewed or upside-down documents, indicating that orientation correction or deskewing is an important preprocessing step.
- **Fine print and dense text remain difficult.** Small fonts and densely packed information consistently reduce performance across all models.
- **Handwriting and moderate scan noise are handled with mixed success.** Accuracy varies by model, with SmolVLM performing comparatively better on handwritten content than the other models.
- **Modern general-purpose VLMs handle varied layouts more effectively.** Qwen3.5-0.8B and InternVL3-1B perform well on documents containing logos, vertical text, and other non-standard layouts, while Donut struggles on several of these cases.
- **Overall, Qwen3.5-0.8B and InternVL3-1B are the most robust across different document conditions**, whereas Donut performs less consistently despite being designed specifically for document understanding.

### 7.4 What the custom set changed

The custom dataset provided insights that are not visible from standard benchmarks alone.
- **Performance on benchmark datasets does not fully reflect real-world performance.** While Qwen3.5-0.8B is the strongest model on DocVQA, its advantage becomes much smaller on degraded document images.
- **Confidence scores become less reliable on challenging documents.** Models that are well calibrated on benchmark datasets can become overconfident when evaluated on real captured documents.
- **The evaluation identifies specific weaknesses.** The largest performance drops occur on rotated documents, fine print, and dense layouts, suggesting that preprocessing and better handling of these conditions are likely to provide larger gains than simply switching models.
  
## 8. Knowledge Gap and Improvement Strategy

The evaluation highlights two main areas where the best-performing model, **Qwen3.5-0.8B**, can be improved. First, its performance drops considerably on infographic-style documents that require reasoning over layouts, charts, and graphics. Second, it is less robust to real-world document degradations such as rotation, skew, fine print, and dense layouts.

To address these limitations, LoRA fine-tuning can be applied to Qwen3.5-0.8B. The approach uses parameter-efficient fine-tuning with the vision encoder kept frozen and trains only a small fraction of the model parameters on a subset of the InfoVQA training data. The goal is to improve layout reasoning while maintaining performance on standard document question answering.

## 9. Code to reproduce results

All code, data manifests, and results are in the repository, with scripts grouped by pipeline stage: `scripts/setup/` (environments, model registration, subset and custom-TSV builds), `scripts/inference/` (main eval, prompting, ECE/latency, Donut drivers), and `scripts/eval/` (analysis and summaries). The custom-set pipeline is fully re-runnable: `setup/build_custom_tsv.py`, `inference/run_custom.sh`, `inference/donut_custom.py`, `eval/analyze_custom.py`.

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


## 10. Recommendation

For on-device document pipelines in this compute class, **Qwen3.5-0.8B remains the best single-model foundation**, but with two qualifications the public benchmarks alone would have missed. First, add orientation and deskew preprocessing before inference; this is the highest-leverage fix available, recovering failure modes where every model currently scores zero. Second, do not reuse DocVQA-calibrated confidence thresholds on captured documents; calibration must be set per input condition, and infographic-style inputs should be routed to human review or a larger model regardless of reported confidence. Use direct prompting, not chain-of-thought, for extraction. LoRA adaptation remains the lowest-cost path to closing the layout-reasoning gap, and the custom set doubles as its robustness regression suite.

## 11. Appendix: Full Result Tables

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

25 documents, 70 QA pairs (n=70 for every cell), ANLS with the standard 0.5 threshold, greedy decoding. Donut has no chat capability, so it has no cot column.

| Model | ANLS<br>direct | ANLS<br>cot | acc@0.5<br>direct | acc@0.5<br>cot |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 54.84 | 46.69 | 62.86 | 52.86 |
| InternVL3-1B | 57.03 | 34.91 | 62.86 | 40.00 |
| SmolVLM-500M | 40.30 | 44.45 | 51.43 | 55.71 |
| Donut-DocVQA | 38.84 | - | 47.14 | - |

### A.3 Custom set, acc@0.5 by failure mode

Each model column shows direct / cot. Donut has no chat capability, so it has a direct score only. Numbers recomputed from the per-sample records after merging near-duplicate tags; buckets overlap because documents carry multiple tags, and the skewed and upside-down tags sit on the same document, so the merged skewed bucket stays at n=3.

| Failure mode | n | Qwen3.5-0.8B<br>direct / cot | InternVL3-1B<br>direct / cot | SmolVLM-500M<br>direct / cot | Donut-DocVQA<br>direct |
|---|---:|---:|---:|---:|---:|
| Degraded image quality | 30 | 73.3 / 60.0 | 76.7 / 50.0 | 60.0 / 66.7 | 56.7 |
| Degraded text quality | 11 | 63.6 / 45.5 | 63.6 / 45.5 | 45.5 / 63.6 | 45.5 |
| Dense text | 9 | 33.3 / 11.1 | 22.2 / 11.1 | 33.3 / 33.3 | 33.3 |
| Faded text | 3 | 66.7 / 66.7 | 33.3 / 33.3 | 66.7 / 66.7 | 33.3 |
| Fine print | 9 | 22.2 / 11.1 | 22.2 / 33.3 | 44.4 / 33.3 | 22.2 |
| Handwritten text | 23 | 60.9 / 52.2 | 60.9 / 39.1 | 65.2 / 65.2 | 52.2 |
| Horizontal text | 8 | 100.0 / 100.0 | 100.0 / 50.0 | 50.0 / 50.0 | 62.5 |
| Scanned document | 11 | 63.6 / 45.5 | 63.6 / 45.5 | 45.5 / 63.6 | 45.5 |
| Skewed document | 3 | 0.0 / 33.3 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 |
| Struck-out text | 3 | 66.7 / 33.3 | 66.7 / 0.0 | 66.7 / 66.7 | 66.7 |
| Stylized logo | 8 | 100.0 / 100.0 | 100.0 / 50.0 | 50.0 / 50.0 | 62.5 |
| Unclear text | 3 | 66.7 / 66.7 | 66.7 / 66.7 | 33.3 / 33.3 | 66.7 |
| Vertical text | 3 | 100.0 / 66.7 | 100.0 / 66.7 | 66.7 / 66.7 | 0.0 |

## References

[1] Akshata A. Bhat, Sharath Naganna, Saiful Haq, Prashant Khatri, Krishna Chaitanya Reddy Tamataam, Neha Arun, Niyati Chhaya, and Pushpak Bhattacharyya. SAVIOR: Sample-efficient Adaptation of Vision-Language Models for OCR Representation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops, VisionDocs, 2026. https://openaccess.thecvf.com/content/WACV2026W/VisionDocs/papers/Bhat_SAVIOR_Sample-efficient_Adaptation_of_Vision-Language_Models_for_OCR_Representation_WACVW_2026_paper.pdf
