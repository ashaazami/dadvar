# Data

Public Hugging Face datasets converted to the
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) schema,
with TypeSafe questions and answers. Datasets are downloaded from the Hugging Face Hub at run
time, at a pinned revision; none of their content is stored in this repository.

```python
from data import Split, build

ds, info = build("banking77")                            # evaluation rows (default)
ds, info = build("banking77", Split.TRAIN, n=2000)       # training rows
ds, info = build("massive_intent", configs=["en", "de"]) # per-language configs
```

Rows carry the typed-decisions columns plus `source`: `id`, `source`, `split`, `state`,
`questions`, `gold`. The last three are JSON strings; `questions` are TypeSafe `Choice` / `Noul` /
`Score`, and `gold` holds `ChoiceAnswer` / `NoulAnswer` / `ScoreAnswer`, one-hot where the dataset
has hard labels. `target_distribution(question, answer)` turns any answer into probabilities over
the question's outcomes; scoring and training targets both use it.

## Sources

Only datasets whose licenses allow the listed use are included. Each source name links to its
dataset on the Hugging Face Hub; `data/sources.py` holds the dataset ids and the conversions.

| Source | Questions | Eval split | Train split | License |
|---|---|---|---|---|
| [`typed_decisions`][td] | 5 per row: choice, noul and score, soft gold | test (400) | train (1,200) | Apache-2.0 |
| [`banking77`][b77] | `choice`, 77 intents | test | train | MIT |
| [`massive_intent`][mi] | `choice`, 60 intents, per language | test | train | Apache-2.0 |
| [`massive_scenario`][ms] | `choice`, 18 scenarios, per language | test | train | Apache-2.0 |
| [`prompt_injections`][pi] | `noul` | test (116) | train (546) | Apache-2.0 |
| [`helpsteer2`][hs2] | 5 `score` questions per row, human gold | validation | train | CC-BY-4.0 |
| [`feedback_collection`][fc] | `score`, a different rubric every row, GPT-4 gold | train¹ | train¹ | CC-BY-4.0 |
| [`boolq`][bq] | `noul` over a passage | validation | **not allowed**² | CC-BY-SA-3.0 |

[td]: https://huggingface.co/datasets/LocalLLaMA/typed-decisions
[b77]: https://huggingface.co/datasets/mteb/banking77
[mi]: https://huggingface.co/datasets/mteb/amazon_massive_intent
[ms]: https://huggingface.co/datasets/mteb/amazon_massive_scenario
[pi]: https://huggingface.co/datasets/deepset/prompt-injections
[hs2]: https://huggingface.co/datasets/nvidia/HelpSteer2
[fc]: https://huggingface.co/datasets/prometheus-eval/Feedback-Collection
[bq]: https://huggingface.co/datasets/google/boolq

¹ Only one split exists, so eval and training rows are carved out of it disjointly.
² Share-alike would attach to a redistributed adaptation and it is unsettled whether model weights
count, so this source is evaluation-only. Laya trained on BoolQ, so it is a retention check, not a
zero-shot measurement.

MASSIVE defaults to 12 languages (en de fr es ru ar hi zh-CN ja ko th sw); pass others explicitly.
Non-English MASSIVE needs the multilingual checkpoint.

## Guarantees

- **Policy is enforced in code.** `build(..., Split.TRAIN)` raises for a source whose `use` policy
  or splits disallow training.
- **Train and eval never overlap.** Separate Hugging Face splits give this by construction; a
  source with one split holds out the evaluation sample first and draws training rows from the rest.
- **Pinned revisions.** Each build resolves the dataset's commit SHA and loads exactly that version.
  The SHA, the sampled row indices, the seed and the prompt version are recorded with the build.
- **Seeded samples**, never "the first N".
- **Versioned wording.** Question text is part of the measurement; bump `PROMPT_VERSION` in
  `sources.py` when instructions or criteria change.
- **Validated.** Every row must parse as TypeSafe questions and answers, gold must be among the
  options, and probabilities must sum to 1.
- **Nothing is redistributed.** Builds are cached under `data/cache/` (git-ignored); datasets are
  downloaded from their source.

## Adding a source

Write one `<source>(row, index, split, config, labels)` function in `sources.py` that returns a
row (the `row()`, `gold_choice()`, `gold_noul()` and `gold_score()` helpers build one), then add a
registry entry with its `fn`, dataset id, configs, splits, license, `use` policy and `seen_by`.
Bump `PROMPT_VERSION` if you change the wording of an existing source.
