# Benchmark datasets

Public datasets for evaluating typed decisions (`choice`, `score`, `noul`).
All are on Hugging Face and load with `datasets.load_dataset(...)`; files are
cached under `~/.cache/huggingface/`.

Sizes and licenses were checked on Hugging Face on 2026-09-21 and 2026-09-26. Each source's
license and what it may be used for (benchmark, training) are declared in its file under
`data/templates/` and enforced by the builder; three sources are evaluation only (BoolQ,
CommonsenseQA, SummEval). Datasets rejected are listed at the end with the reason. What is implemented, and each source's splits and policy, is in
[`data/`](../data/README.md).

## Typed decisions (System One format)

| Dataset | Config / split | Rows | Task | License |
|---|---|---:|---|---|
| [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | `all` or one of 4 workflows; `train` / `test` | 1,200 / 400 cases (×5 questions) | 5 mixed `noul` / `choice` / `score` questions per case, over agent traces, customer service, invoices, security incidents | apache-2.0 |

- Rows replay directly as a TypeSafe `POST /v1/systemone` body (`state` + `questions`).
- Synthetic and labeled by a ~4B teacher (mean of 3 samples), so scores measure agreement with the teacher. References: majority 0.520, factor ceiling 0.704, teacher self-agreement 0.735.
- Published scores on the test split: TypeSafe Jev 1.13.0 (zero-shot) 0.727; ModernBERT-base specialist 0.646; MiniLM-L6 specialist 0.587.
- The card distinguishes **specialist** (trained on these workflows) from **generalist** (zero-shot) scores and says the two aren't comparable. Always label which mode a number comes from.

## Rubric grading (described criteria)

| Dataset | Split | Rows | Maps to | Gold | License |
|---|---|---:|---|---|---|
| [`prometheus-eval/Feedback-Collection`](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection) | train (only split) | 99,952 | `score`: grade a response with the row's own rubric (criterion + 5 described levels) | GPT-4 scores 1–5, balanced | cc-by-4.0 |
| [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | validation | 1,038 | 5 `score` questions per row: helpfulness, correctness, coherence, complexity, verbosity (0–4) | human (Scale AI) | cc-by-4.0 |

## English and yes/no

| Dataset | Split used | Rows | Maps to | License | Used |
|---|---|---:|---|---|---|
| [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) | test 3,076 / train 9,993 | 3,076 | `choice` intent, 77 labels | mit | ✅ benchmark + train |
| [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) | test 116 / train 546 | 116 | `noul` injection | apache-2.0 | ✅ benchmark + train |
| [`google/boolq`](https://huggingface.co/datasets/google/boolq) | validation | 3,270 | `noul` over a passage | cc-by-sa-3.0 | ✅ benchmark only (share-alike) |

## Multilingual

| Dataset | Split used | Rows | Maps to | Languages | License | Used |
|---|---|---:|---|---|---|---|
| [`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent) | test | 2,974 per language | `choice` intent | 51 | apache-2.0 | ✅ benchmark + train |
| [`mteb/amazon_massive_scenario`](https://huggingface.co/datasets/mteb/amazon_massive_scenario) | test | 2,974 per language | `choice` scenario | 51 | apache-2.0 | ✅ benchmark + train |

## Added 2026-09-26: yes/no, safety, human quality ratings, legal

Chosen to fill gaps in the training mix: yes/no (`noul`) questions were 8% of items, and there
was no safety, legal or human quality-rating data. All human-labelled except where noted.
Licenses read from the Hugging Face API and the dataset cards on 2026-09-26. Each is
`default: false`, so `all` benchmark runs still cover the earlier set; training uses every source
licensed for it.

| Source | Dataset | Splits (eval / train) | Maps to | Gold | License |
|---|---|---|---|---|---|
| `paws` | [`google-research-datasets/paws`](https://huggingface.co/datasets/google-research-datasets/paws) `labeled_final` | test / train (49,401) | `noul`: do `sentence1` and `sentence2` mean the same thing? | human, 94.7% agreement | Google: "may be freely used for any purpose" |
| `amazon_counterfactual` | [`mteb/amazon_counterfactual`](https://huggingface.co/datasets/mteb/amazon_counterfactual) `en` | test / train (4,018) | `noul`: does `review` describe something that did not or could not happen? | human | cc-by-4.0 |
| `civil_comments` | [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) | test / train (1.8M) | 5 `noul`: toxic, insult, obscene, threat, identity attack | **soft**: share of annotators who said yes | cc0-1.0 |
| `aegis_safety` | [`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | test / train (30,007) | `noul` unsafe prompt; `noul` unsafe response where the row has one | prompt: human; response: human or LLM jury | cc-by-4.0; intended for moderation and safety |
| `oasst_quality` | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1), English assistant replies joined to their prompt | validation / train | 3 `score` (5 levels): quality, helpfulness, creativity; `noul`: fails the task | **soft**: reviewers' 0-1 average, split between the two nearest levels | apache-2.0 |
| `ledgar` | [`coastalcph/lex_glue`](https://huggingface.co/datasets/coastalcph/lex_glue) `ledgar` | test / train (60,000) | `choice`: contract provision type, 100 labels | label | cc-by-4.0 |
| `case_hold` | [`coastalcph/lex_glue`](https://huggingface.co/datasets/coastalcph/lex_glue) `case_hold` | test / train (45,000) | `choice`: which of 5 holdings fits `<HOLDING>`; options are the holdings' text | label | cc-by-4.0 |

- **civil_comments is mostly "no":** on a 200-row sample, 84% of answers are exactly 0 and 3% are
  a majority yes, the dataset's natural rate.
- **Aegis:** redacted prompts (text `REDACTED`) are dropped. About 38% of rows carry a labelled
  response, so they ask both questions.
- **Fit at 1,024 tokens** (ModernBERT tokenizer, 200 training rows each): 100% for all except
  Aegis, 99.0%.
- **Considered in the same search and left for later:** WANLI (cc-by-4.0, 103k NLI pairs; GPT-3
  wrote the pairs, humans labelled and revised them); UltraFeedback (MIT; GPT-4 ratings);
  WildGuardMix (ODC-By, gated; GPT-4 training labels); QASC (cc-by-4.0); bias_in_bios (MIT); jailbreak-classification (apache-2.0); SPML prompt
  injection (MIT). hh-rlhf and CommonsenseQA from this list were added later. Model-written text or labels may carry the model provider's terms. MMLU and
  BIG-bench have permissive licenses but are standard benchmarks, so training on them would
  contaminate later evaluations.

## Not used, and why

Permissive enough to evaluate on, but not to train a model that may be distributed
commercially, or with no declared license at all. SST-5 (no declared license) and MultiNLI (some
genres share-alike) were first rejected here and later added for research use (`docs/DATASET.md`); their licenses are recorded as such in the table below. Reconsider individually if a use case
needs them.

| Dataset | Reason |
|---|---|
| [`microsoft/ms_marco`](https://huggingface.co/datasets/microsoft/ms_marco) | Microsoft's terms: non-commercial research only |
| [`lmsys/toxic-chat`](https://huggingface.co/datasets/lmsys/toxic-chat) | cc-by-nc-4.0 (non-commercial) |
| [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | cc-by-nc-4.0 (non-commercial) |
| [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | no declared license; also in Laya's training mix |
| [`facebook/xnli`](https://huggingface.co/datasets/facebook/xnli) | no declared license |
| [`SetFit/enron_spam`](https://huggingface.co/datasets/SetFit/enron_spam) | no declared license |
| [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | license "other"; research-only terms |
| [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) | lgpl-3.0, unusual for data; unclear obligations |
| [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp) (cc-by-4.0) | license fine; not a typed-decision task on its own (Laya used it only to build a synthetic routing task) |
| [`prometheus-eval/BiGGen-Bench`](https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench) | rubrics only, no graded responses to score |
| [`prometheus-eval/Feedback-Bench`](https://huggingface.co/datasets/prometheus-eval/Feedback-Bench) | no declared license (its train-split sibling is used instead) |
| [`btzsc/btzsc`](https://huggingface.co/datasets/btzsc/btzsc), UCI SMS Spam | used by third-party Jev benchmarks; not needed here |
| [`stanfordnlp/snli`](https://huggingface.co/datasets/stanfordnlp/snli), [`fever/fever`](https://huggingface.co/datasets/fever/fever), [`tals/vitaminc`](https://huggingface.co/datasets/tals/vitaminc), [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc), [`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14), [`google-research-datasets/natural_questions`](https://huggingface.co/datasets/google-research-datasets/natural_questions) | share-alike (cc-by-sa) |
| [`facebook/anli`](https://huggingface.co/datasets/facebook/anli), [`allenai/scifact`](https://huggingface.co/datasets/allenai/scifact), [`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq), [`PKU-Alignment/BeaverTails`](https://huggingface.co/datasets/PKU-Alignment/BeaverTails), [`PatronusAI/HaluBench`](https://huggingface.co/datasets/PatronusAI/HaluBench), [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank) | non-commercial |
| [`lytang/LLM-AggreFact`](https://huggingface.co/datasets/lytang/LLM-AggreFact) | cc-by-nd-4.0 (no derivatives) |
| HellaSwag, WinoGrande, OpenBookQA, PIQA, Social IQa, IMDB, SST-2, Yelp reviews, TREC, Rotten Tomatoes, Quora duplicates, RAGTruth | no declared or "unknown"/"other" license on the Hugging Face card |
| [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | cdla-sharing-1.0 (share-alike for data) |

## Published comparison suites (code, not datasets)

| Repo | What it measures | Jev results |
|---|---|---|
| [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) | Jev vs GLiNER2.5 on BTZSC conditions, 100 held-out examples each; accuracy, Brier, NLL, ECE, selective risk, latency | AG News 0.910, Banking77 (72 labels) 0.870, Emotion 0.480 |
| [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark) | Jev vs 8 constrained LLMs and baselines: banking77 (77-way), SMS spam, a synthetic "code word" cardinality test, permuted option order; cost, latency, ECE | banking 76.3%, spam 93.0%, permuted 76.7%, 13% flip rate, ECE 0.246 |

Neither repo declares a license, so reuse their methods, not their code.

## Methods worth reproducing (no dataset needed)

- **Option-order robustness:** shuffle the options and count how often the chosen label changes.
- **Label-cardinality stress:** raise the number of options (e.g. 4 → 77 → 256+) and track accuracy and latency.
- **Calibration:** report Brier, log loss / NLL and ECE next to accuracy. Refit per-type temperatures on a held-out split and report before/after.
- **Held-out vs in-training:** mark which datasets a checkpoint saw in training. The Laya checkpoints saw AG News, BoolQ, support tickets, Enron spam, phishing and MS MARCO. Emotion, banking77 and toxic-chat were held out.

## Label-set size against the local token budget

> Measured on the Laya checkpoints, which this repository no longer runs. It is kept because the
> label-size limits it establishes still bound any model with a fixed option budget. A model trained
> here cannot clip an option: `dadvar/sequence.py` builds the whole sequence and the runner skips a
> question that does not fit, rather than cutting it.

A Laya checkpoint's options share a fixed `head_max_len` (192 on `laya`, 256 on
`laya-typed-decisions` and `laya-multilingual`). Past roughly 40 labels the option block
overruns it, every option is capped at 4 tokens, and the instruction is floored at 8 -- so the
model answers a different question from the one the dataset asks. Measured on those
checkpoints:

| Dataset | Options | Option block | laya (192) | typed-decisions (256) |
|---|---:|---:|---|---|
| `clinc_oos` | 151 | 459 tok | clipped, 1 collision | clipped, 1 collision |
| `banking77` | 77 | 342 tok | clipped, 3 collisions | clipped, 3 collisions |
| `massive_intent` | 60 | 229 tok | clipped, 2 collisions | **intact** |
| `go_emotions` | 28 | 58 tok | intact | intact |
| `massive_scenario` | 18 | 39 tok | intact | intact |

A *collision* is two labels that become the same tokens once capped, so the model cannot tell
them apart by text; `massive_intent`'s five `iot hue light*` intents collapse into one. Jev is
unaffected -- it receives the full label set.

This does not disqualify a dataset: a run skips a row its sequence builder would cut, and reports
score every model on the rows all of them answered (see [`benchmarks/`](../benchmarks/README.md)).
It does mean a high-cardinality source measures the token budget as much as the model, and that
the two local checkpoints do not see the same question on `massive_intent`.

## Every dataset found, and whether it is used

Generated from the sources (each one's file under `data/templates/`, collected as `SOURCES`) and the
rejected candidates above. **Used** is what a file's `use:` allows; *opt-in* marks `default: false`,
left out of an `all` benchmark run unless named. Licenses are as those files record them.

| Source | Dataset | Used | License |
|---|---|---|---|
| `aegis_safety` | [`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | ✅ benchmark + train *(opt-in)* | cc-by-4.0 |
| `amazon_counterfactual` | [`mteb/amazon_counterfactual`](https://huggingface.co/datasets/mteb/amazon_counterfactual) | ✅ benchmark + train *(opt-in)* | cc-by-4.0 |
| `amazon_reviews` | [`SetFit/amazon_reviews_multi_en`](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en) | ✅ benchmark + train *(opt-in)* | other (Multilingual Amazon Reviews Corpus terms, non-commercial; SetFit's copy is tagged apache-2.0) |
| `aqua_rat` | [`deepmind/aqua_rat`](https://huggingface.co/datasets/deepmind/aqua_rat) | ✅ benchmark + train *(opt-in)* | apache-2.0 |
| `arena_preference` | [`lmarena-ai/arena-human-preference-55k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) | ✅ benchmark + train *(opt-in)* | apache-2.0 |
| `banking77` | [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) | ✅ benchmark + train | mit |
| `boolq` | [`google/boolq`](https://huggingface.co/datasets/google/boolq) | ✅ benchmark only | cc-by-sa-3.0 |
| `case_hold` | [`coastalcph/lex_glue`](https://huggingface.co/datasets/coastalcph/lex_glue) · `case_hold` | ✅ benchmark + train *(opt-in)* | cc-by-4.0 |
| `civil_comments` | [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) | ✅ benchmark + train *(opt-in)* | cc0-1.0 |
| `clinc_oos` | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) | ✅ benchmark + train | cc-by-3.0 |
| `commonsense_qa` | [`tau/commonsense_qa`](https://huggingface.co/datasets/tau/commonsense_qa) | ✅ benchmark only *(opt-in)* | mit |
| `ethics` | [`hendrycks/ethics`](https://huggingface.co/datasets/hendrycks/ethics) · `commonsense`, `deontology`, `justice` | ✅ benchmark + train *(opt-in)* | mit |
| `feedback_collection` | [`prometheus-eval/Feedback-Collection`](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection) | ✅ benchmark + train *(opt-in)* | cc-by-4.0 |
| `go_emotions` | [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | ✅ benchmark + train | apache-2.0 |
| `gsm8k` | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | ✅ benchmark + train *(opt-in)* | mit |
| `hate_speech` | [`ucberkeley-dlab/measuring-hate-speech`](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | ✅ benchmark + train *(opt-in)* | cc-by-4.0 |
| `helpsteer2` | [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | ✅ benchmark + train | cc-by-4.0 |
| `helpsteer3` | [`nvidia/HelpSteer3`](https://huggingface.co/datasets/nvidia/HelpSteer3) | ✅ benchmark + train *(opt-in)* | cc-by-4.0 |
| `helpsteer_v1` | [`nvidia/HelpSteer`](https://huggingface.co/datasets/nvidia/HelpSteer) | ✅ benchmark + train *(opt-in)* | cc-by-4.0 |
| `hh_rlhf` | [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) | ✅ benchmark + train *(opt-in)* | mit |
| `ledgar` | [`coastalcph/lex_glue`](https://huggingface.co/datasets/coastalcph/lex_glue) · `ledgar` | ✅ benchmark + train *(opt-in)* | cc-by-4.0 |
| `massive_intent` | [`mteb/amazon_massive_intent`](https://huggingface.co/datasets/mteb/amazon_massive_intent) · 51 languages, 12 by default | ✅ benchmark + train | apache-2.0 |
| `massive_scenario` | [`mteb/amazon_massive_scenario`](https://huggingface.co/datasets/mteb/amazon_massive_scenario) · 51 languages, 12 by default | ✅ benchmark + train | apache-2.0 |
| `math_qa` | [`allenai/math_qa`](https://huggingface.co/datasets/allenai/math_qa) | ✅ benchmark + train *(opt-in)* | apache-2.0 |
| `multi_nli` | [`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli) | ✅ benchmark + train *(opt-in)* | per genre: cc-by-3.0, cc-by-sa-3.0, mit, other (OANC) |
| `oasst_quality` | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) | ✅ benchmark + train *(opt-in)* | apache-2.0 |
| `paws` | [`google-research-datasets/paws`](https://huggingface.co/datasets/google-research-datasets/paws) | ✅ benchmark + train *(opt-in)* | paws (free for any purpose) |
| `prompt_injections` | [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) | ✅ benchmark + train | apache-2.0 |
| `reward_bench` | [`allenai/reward-bench`](https://huggingface.co/datasets/allenai/reward-bench) | ✅ benchmark + train *(opt-in)* | odc-by |
| `ruletaker` | [`tasksource/ruletaker`](https://huggingface.co/datasets/tasksource/ruletaker) | ✅ benchmark + train *(opt-in)* | apache-2.0 |
| `squad_v2` | [`rajpurkar/squad_v2`](https://huggingface.co/datasets/rajpurkar/squad_v2) | ✅ benchmark + train *(opt-in)* | cc-by-sa-4.0 |
| `sst5` | [`SetFit/sst5`](https://huggingface.co/datasets/SetFit/sst5) | ✅ benchmark + train *(opt-in)* | unknown (Stanford Sentiment Treebank) |
| `stsb` | [`sentence-transformers/stsb`](https://huggingface.co/datasets/sentence-transformers/stsb) | ✅ benchmark + train *(opt-in)* | unknown (STS Benchmark; GLUE lists it as other) |
| `summeval` | [`mteb/summeval`](https://huggingface.co/datasets/mteb/summeval) | ✅ benchmark only *(opt-in)* | mit |
| `tab_fact` | [`wenhu/tab_fact`](https://huggingface.co/datasets/wenhu/tab_fact) | ✅ benchmark + train *(opt-in)* | cc-by-4.0 |
| `typed_decisions` | [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | ✅ benchmark + train | apache-2.0 |
| — | [`microsoft/ms_marco`](https://huggingface.co/datasets/microsoft/ms_marco) | ❌ not used | Microsoft terms: non-commercial research |
| — | [`lmsys/toxic-chat`](https://huggingface.co/datasets/lmsys/toxic-chat) | ❌ not used | cc-by-nc-4.0 — non-commercial |
| — | [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | ❌ not used | cc-by-nc-4.0 — non-commercial |
| — | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | ❌ not used | none declared — also in Laya's training mix |
| — | [`facebook/xnli`](https://huggingface.co/datasets/facebook/xnli) | ❌ not used | none declared |
| — | [`SetFit/enron_spam`](https://huggingface.co/datasets/SetFit/enron_spam) | ❌ not used | none declared |
| — | [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | ❌ not used | other — research-only terms |
| — | [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) | ❌ not used | lgpl-3.0 — unusual for data; unclear obligations |
| — | [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp) | ❌ not used | cc-by-4.0 — licence fine; not a typed-decision task on its own |
| — | [`prometheus-eval/BiGGen-Bench`](https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench) | ❌ not used | — — rubrics only, no graded responses to score |
| — | [`prometheus-eval/Feedback-Bench`](https://huggingface.co/datasets/prometheus-eval/Feedback-Bench) | ❌ not used | none declared — its train-split sibling is used instead |
| — | [`btzsc/btzsc`](https://huggingface.co/datasets/btzsc/btzsc) | ❌ not used | — — used by third-party Jev benchmarks; not needed here |
| — | UCI SMS Spam | ❌ not used | — — used by third-party Jev benchmarks; not needed here |
| — | [`stanfordnlp/snli`](https://huggingface.co/datasets/stanfordnlp/snli) | ❌ not used | cc-by-sa — share-alike |
| — | [`fever/fever`](https://huggingface.co/datasets/fever/fever) | ❌ not used | cc-by-sa — share-alike |
| — | [`tals/vitaminc`](https://huggingface.co/datasets/tals/vitaminc) | ❌ not used | cc-by-sa — share-alike |
| — | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | ❌ not used | cc-by-sa — share-alike |
| — | [`fancyzhx/dbpedia_14`](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | ❌ not used | cc-by-sa — share-alike |
| — | [`google-research-datasets/natural_questions`](https://huggingface.co/datasets/google-research-datasets/natural_questions) | ❌ not used | cc-by-sa — share-alike |
| — | [`facebook/anli`](https://huggingface.co/datasets/facebook/anli) | ❌ not used | non-commercial |
| — | [`allenai/scifact`](https://huggingface.co/datasets/allenai/scifact) | ❌ not used | non-commercial |
| — | [`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) | ❌ not used | non-commercial |
| — | [`PKU-Alignment/BeaverTails`](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | ❌ not used | non-commercial |
| — | [`PatronusAI/HaluBench`](https://huggingface.co/datasets/PatronusAI/HaluBench) | ❌ not used | non-commercial |
| — | [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank) | ❌ not used | non-commercial |
| — | [`lytang/LLM-AggreFact`](https://huggingface.co/datasets/lytang/LLM-AggreFact) | ❌ not used | cc-by-nd-4.0 — no derivatives |
| — | [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | ❌ not used | cdla-sharing-1.0 — share-alike for data |
| — | HellaSwag, WinoGrande, OpenBookQA, PIQA, Social IQa, IMDB, SST-2, Yelp reviews, TREC, Rotten Tomatoes, Quora duplicates, RAGTruth | ❌ not used | none declared / unknown / other |
| — | WANLI | ❌ not used | cc-by-4.0 — left for later: GPT-3 wrote the pairs, humans labelled them |
| — | UltraFeedback | ❌ not used | mit — left for later: GPT-4 ratings |
| — | [`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix) | ❌ not used | odc-by — left for later: gated; GPT-4 training labels |
| — | QASC | ❌ not used | cc-by-4.0 — left for later |
| — | bias_in_bios | ❌ not used | mit — left for later |
| — | [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | ❌ not used | apache-2.0 — left for later |
| — | [`reshabhs/SPML_Chatbot_Prompt_Injection`](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection) | ❌ not used | mit — left for later |
| — | MMLU, BIG-bench | ❌ not used | permissive — standard benchmarks: training on them would contaminate evaluation |

36 registry sources (all evaluated, 33 also licensed for training) and 36 datasets considered and rejected.

