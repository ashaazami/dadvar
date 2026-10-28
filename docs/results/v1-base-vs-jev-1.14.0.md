# Dadvar v1-base vs Jev 1.14.0

Generated 2026-10-28 19:49 UTC from 2 run(s): dadvar/v1-base, typesafe/jev-1.14.0.

## Accuracy

Scored on the rows every model answered: **53 excluded** because at least one run skipped them, a question being longer than its model's token budget. `--include-skipped` scores each run on every row it answered.

### By question type

| question type | questions | dadvar/v1-base | typesafe/jev-1.14.0 |
|---|---|---|---|
| choice | 14100 | 0.350 | **0.743** |
| noul | 1216 | 0.696 | **0.854** |
| score | 3035 | **0.569** | 0.526 |

## Metrics

### By question type

| question type | metric | dadvar/v1-base | typesafe/jev-1.14.0 |
|---|---|---|---|
| choice | accuracy | 0.350 | **0.743** |
|  | brier | 0.790 | **0.362** |
| noul | accuracy | 0.696 | **0.854** |
|  | brier | 0.272 | **0.114** |
| score | accuracy | **0.569** | 0.526 |
|  | brier | **0.450** | 0.583 |
|  | score MAE | **0.593** | 0.648 |

## By source

Accuracy and expected calibration error (ECE) per source, scored on every row each model answered.

| source | questions | accuracy, dadvar/v1-base | accuracy, typesafe/jev-1.14.0 | ECE, dadvar/v1-base | ECE, typesafe/jev-1.14.0 |
|---|---|---|---|---|---|
| banking77 | 500 | 0.712 | 0.782 | 0.083 | 0.082 |
| boolq | 500 | 0.606 | 0.938 | 0.077 | 0.036 |
| clinc_oos | 500 | 0.666 | 0.918 | 0.082 | 0.024 |
| go_emotions | 500 | 0.564 | 0.328 | 0.082 | 0.350 |
| helpsteer2 | 2235 | 0.553 | 0.452 | 0.035 | 0.193 |
| massive_intent | 6000 | 0.262 | 0.774 | 0.163 | 0.070 |
| massive_intent/ar | 500 | 0.088 | 0.704 | 0.248 | 0.093 |
| massive_intent/de | 500 | 0.310 | 0.806 | 0.140 | 0.063 |
| massive_intent/en | 500 | 0.748 | 0.814 | 0.051 | 0.074 |
| massive_intent/es | 500 | 0.380 | 0.772 | 0.100 | 0.076 |
| massive_intent/fr | 500 | 0.442 | 0.774 | 0.079 | 0.092 |
| massive_intent/hi | 500 | 0.068 | 0.792 | 0.262 | 0.063 |
| massive_intent/ja | 500 | 0.274 | 0.822 | 0.187 | 0.056 |
| massive_intent/ko | 500 | 0.054 | 0.794 | 0.300 | 0.070 |
| massive_intent/ru | 500 | 0.208 | 0.798 | 0.185 | 0.079 |
| massive_intent/sw | 500 | 0.072 | 0.638 | 0.173 | 0.078 |
| massive_intent/th | 500 | 0.048 | 0.770 | 0.212 | 0.085 |
| massive_intent/zh-CN | 500 | 0.456 | 0.808 | 0.080 | 0.064 |
| massive_scenario | 6000 | 0.338 | 0.731 | 0.230 | 0.087 |
| massive_scenario/ar | 500 | 0.114 | 0.706 | 0.337 | 0.096 |
| massive_scenario/de | 500 | 0.442 | 0.748 | 0.179 | 0.084 |
| massive_scenario/en | 500 | 0.830 | 0.756 | 0.048 | 0.093 |
| massive_scenario/es | 500 | 0.482 | 0.748 | 0.150 | 0.100 |
| massive_scenario/fr | 500 | 0.552 | 0.732 | 0.120 | 0.104 |
| massive_scenario/hi | 500 | 0.102 | 0.714 | 0.391 | 0.115 |
| massive_scenario/ja | 500 | 0.368 | 0.740 | 0.197 | 0.093 |
| massive_scenario/ko | 500 | 0.072 | 0.748 | 0.397 | 0.109 |
| massive_scenario/ru | 500 | 0.280 | 0.732 | 0.234 | 0.104 |
| massive_scenario/sw | 500 | 0.096 | 0.672 | 0.359 | 0.063 |
| massive_scenario/th | 500 | 0.072 | 0.722 | 0.355 | 0.120 |
| massive_scenario/zh-CN | 500 | 0.640 | 0.752 | 0.086 | 0.096 |
| prompt_injections | 116 | 0.853 | 0.724 | 0.055 | 0.193 |
| typed_decisions | 2000 | 0.653 | 0.739 | 0.105 | 0.053 |
| typed_decisions/agent_trace_observability | 500 | 0.584 | 0.612 | 0.086 | 0.140 |
| typed_decisions/customer_service | 500 | 0.706 | 0.838 | 0.127 | 0.036 |
| typed_decisions/invoice_processing | 500 | 0.680 | 0.784 | 0.137 | 0.055 |
| typed_decisions/security_incidents | 500 | 0.642 | 0.720 | 0.119 | 0.092 |

## Sources

| source | dataset | revision | split | rows | questions | types | in v1-base training | in jev-1.14.0 training |
|---|---|---|---|---|---|---|---|---|
| banking77 | [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) | [`18072d26`](https://huggingface.co/datasets/mteb/banking77/tree/18072d2685ea682290f7b8924d94c62acc19c0b2) | test | 500 | 500 | choice | yes | unknown |
| boolq | [`google/boolq`](https://huggingface.co/datasets/google/boolq) | [`35b264d0`](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5) | validation | 500 | 500 | noul | no | unknown |
| clinc_oos | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) | [`155b9c71`](https://huggingface.co/datasets/clinc/clinc_oos/tree/155b9c710419136e17307b80d0a13e68cd46b4ec) | test | 500 | 500 | choice | yes | unknown |
| go_emotions | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | [`add49224`](https://huggingface.co/datasets/google-research-datasets/go_emotions/tree/add492243ff905527e67aeb8b80c082af02207c3) | test | 500 | 500 | choice | yes | unknown |
| helpsteer2 | [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | [`990b2711`](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/990b2711a36180dd19d9c94b8627844866f8982a) | validation | 500 | 2235 | score | yes | unknown |
| massive_intent | [`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent) | [`940fd47a`](https://huggingface.co/datasets/mteb/amazon_massive_intent/tree/940fd47a81eaa7f2cc7b129674d945d618ac38c2) | test | 6000 | 6000 | choice | yes | unknown |
| massive_scenario | [`mteb/amazon_massive_scenario`](https://huggingface.co/datasets/mteb/amazon_massive_scenario) | [`58871793`](https://huggingface.co/datasets/mteb/amazon_massive_scenario/tree/58871793b91addb7c5f7afff26ccf08737fb6697) | test | 6000 | 6000 | choice | yes | unknown |
| prompt_injections | [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) | [`4f61ecb0`](https://huggingface.co/datasets/deepset/prompt-injections/tree/4f61ecb038e9c3fb77e21034b22511b523772cdd) | test | 116 | 116 | noul | yes | unknown |
| typed_decisions | [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | [`7f479ec3`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/7f479ec3eadb645c56fa441366d401ad96527980) | test | 400 | 2000 | choice, noul, score | yes | unknown |

Seed 13, prompt v1. Training columns: "yes" or "no" from the model's own record of what it trained on (a model trained here records it), "unknown" where the model states nothing.

---

Hosted model cost for these answers: typesafe/jev-1.14.0 $0.4159.

