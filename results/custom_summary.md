# Custom Document Set: Prompting Eval

Hand-annotated degraded-scan set (70 QA pairs). `direct` vs `cot`, greedy, vLLM-served, ANLS (DocVQA 0.5 threshold).

## Overall ANLS

| model        |    cot |   direct |   Δ (cot-direct) |
|:-------------|-------:|---------:|-----------------:|
| Donut-DocVQA | nan    |    38.84 |           nan    |
| InternVL3-1B |  34.91 |    57.03 |           -22.12 |
| Qwen3.5-0.8B |  46.69 |    54.84 |            -8.15 |
| SmolVLM-500M |  44.45 |    40.3  |             4.15 |


## Full metrics

| model        | mode   |   n |   anls |   acc@0.5 |
|:-------------|:-------|----:|-------:|----------:|
| Qwen3.5-0.8B | direct |  70 |  54.84 |     62.86 |
| Qwen3.5-0.8B | cot    |  70 |  46.69 |     52.86 |
| InternVL3-1B | direct |  70 |  57.03 |     62.86 |
| InternVL3-1B | cot    |  70 |  34.91 |     40    |
| SmolVLM-500M | direct |  70 |  40.3  |     51.43 |
| SmolVLM-500M | cot    |  70 |  44.45 |     55.71 |
| Donut-DocVQA | direct |  70 |  38.84 |     47.14 |


## acc@0.5 by failure mode: direct

| failure_mode           |   n |   Donut-DocVQA |   InternVL3-1B |   Qwen3.5-0.8B |   SmolVLM-500M |
|:-----------------------|----:|---------------:|---------------:|---------------:|---------------:|
| degarded image quality |   3 |           33.3 |           33.3 |           33.3 |           33.3 |
| degraded image         |  12 |           50   |           75   |           75   |           58.3 |
| degraded image quality |  15 |           66.7 |           86.7 |           80   |           66.7 |
| degraded text quality  |  11 |           45.5 |           63.6 |           63.6 |           45.5 |
| dense text             |   9 |           33.3 |           22.2 |           33.3 |           33.3 |
| faded text             |   3 |           33.3 |           33.3 |           66.7 |           66.7 |
| faint print            |   3 |           33.3 |           66.7 |           33.3 |          100   |
| fine print             |   6 |           16.7 |            0   |           16.7 |           16.7 |
| handwritten text       |  23 |           52.2 |           60.9 |           60.9 |           65.2 |
| horizontal text        |   8 |           62.5 |          100   |          100   |           50   |
| scanned document       |  11 |           45.5 |           63.6 |           63.6 |           45.5 |
| skewed                 |   3 |            0   |            0   |            0   |            0   |
| striked out text       |   3 |           66.7 |           66.7 |           66.7 |           66.7 |
| stylized logo          |   8 |           62.5 |          100   |          100   |           50   |
| unclear text           |   3 |           66.7 |           66.7 |           66.7 |           33.3 |
| upside down image      |   3 |            0   |            0   |            0   |            0   |
| vertical text          |   3 |            0   |          100   |          100   |           66.7 |


## acc@0.5 by failure mode: cot

| failure_mode           |   n |   InternVL3-1B |   Qwen3.5-0.8B |   SmolVLM-500M |
|:-----------------------|----:|---------------:|---------------:|---------------:|
| degarded image quality |   3 |           33.3 |           33.3 |           33.3 |
| degraded image         |  12 |           66.7 |           58.3 |           58.3 |
| degraded image quality |  15 |           40   |           66.7 |           80   |
| degraded text quality  |  11 |           45.5 |           45.5 |           63.6 |
| dense text             |   9 |           11.1 |           11.1 |           33.3 |
| faded text             |   3 |           33.3 |           66.7 |           66.7 |
| faint print            |   3 |           66.7 |           33.3 |           66.7 |
| fine print             |   6 |           16.7 |            0   |           16.7 |
| handwritten text       |  23 |           39.1 |           52.2 |           65.2 |
| horizontal text        |   8 |           50   |          100   |           50   |
| scanned document       |  11 |           45.5 |           45.5 |           63.6 |
| skewed                 |   3 |            0   |           33.3 |            0   |
| striked out text       |   3 |            0   |           33.3 |           66.7 |
| stylized logo          |   8 |           50   |          100   |           50   |
| unclear text           |   3 |           66.7 |           66.7 |           33.3 |
| upside down image      |   3 |            0   |           33.3 |            0   |
| vertical text          |   3 |           66.7 |           66.7 |           66.7 |

