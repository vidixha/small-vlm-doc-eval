# Sub-1B VLM Document Understanding — Results Summary

300-sample fixed-seed (42) subsets, greedy decoding, vLLM-served on T4.

| model        | dataset            |   anls_main |   n_ece |   anls_ece_pass |   acc@0.5 |   mean_conf |   ECE |   overconfidence |   lat_mean_s |   lat_p50_s |   tok_per_s |
|:-------------|:-------------------|------------:|--------:|----------------:|----------:|------------:|------:|-----------------:|-------------:|------------:|------------:|
| Qwen3.5-0.8B | DocVQA_VAL_SUB300  |     86.8884 |     300 |           86.71 |     89    |       87.63 |  3.22 |            -1.37 |         0.51 |        0.48 |        11.2 |
| Qwen3.5-0.8B | InfoVQA_VAL_SUB300 |     54.1021 |     300 |           54.62 |     59.33 |       71.69 | 14.56 |            12.36 |         0.48 |        0.45 |         9.3 |
| InternVL3-1B | DocVQA_VAL_SUB300  |     83.2315 |     300 |           83.47 |     85.67 |       90.92 |  5.25 |             5.25 |         1.33 |        1.61 |         7.1 |
| InternVL3-1B | InfoVQA_VAL_SUB300 |     50.8692 |     300 |           50.88 |     58    |       82.93 | 24.93 |            24.93 |         0.9  |        0.79 |        11.7 |
| SmolVLM-500M | DocVQA_VAL_SUB300  |     61.7626 |     300 |           64.87 |     75.33 |       90.6  | 15.27 |            15.27 |         1.01 |        0.99 |         7.3 |
| SmolVLM-500M | InfoVQA_VAL_SUB300 |     23.3282 |     300 |           27.57 |     38    |       78.46 | 40.46 |            40.46 |         0.78 |        0.71 |         8.3 |
| Donut-DocVQA | DocVQA_VAL_SUB300  |    nan      |     300 |           62.57 |     65.67 |       95.11 | 29.44 |            29.44 |         0.62 |        0.6  |        10.5 |
| Donut-DocVQA | InfoVQA_VAL_SUB300 |    nan      |     300 |           13.89 |     18.33 |       84.37 | 66.03 |            66.03 |         0.72 |        0.67 |         6.8 |