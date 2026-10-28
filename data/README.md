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

Only datasets whose licenses allow the listed use are included. There are 36 sources; the table
below is the default `all` set plus the opt-in preference and rubric sources, and
[`docs/BENCHMARK_DATASETS.md`](../docs/BENCHMARK_DATASETS.md) lists every one with its license.
Each source name links to its dataset on the Hugging Face Hub; each source's `data/templates/` file
holds its dataset id and settings, and `data/sources.py` the conversions they name.

| Source | Questions | Eval | Train | License |
|---|---|---|---|---|
| **mixed** | | | | |
| [`typed_decisions`][td] | triage an agent trace, ticket, invoice or alert; soft gold | 2,000 | 6,000 | Apache-2.0 |
| **choice** | | | | |
| [`banking77`][b77] | banking intents | 3,076 | 9,993 | MIT |
| [`massive_intent`][mi] | assistant intents, per language | 2,974 per lang | 11,514 per lang | Apache-2.0 |
| [`massive_scenario`][ms] | assistant scenarios, per language | 2,974 per lang | 11,514 per lang | Apache-2.0 |
| [`clinc_oos`][clinc] | assistant intents, plus out-of-scope | 5,500 | 15,250 | CC-BY-3.0 |
| [`go_emotions`][ge] | emotions³ | 4,590 | 36,308 | Apache-2.0 |
| ✗ [`reward_bench`][rb] | better of two responses¹ ⁴ | 2,985 shared | | ODC-BY |
| ✗ [`arena_preference`][arena] | better of two responses, **or a tie**¹ ⁵ | 57,477 shared | | Apache-2.0 |
| **noul** | | | | |
| [`prompt_injections`][pi] | is this a prompt injection | 116 | 546 | Apache-2.0 |
| [`boolq`][bq] | yes/no about a passage | 3,270 | **not allowed**² | CC-BY-SA-3.0 |
| **score** | | | | |
| [`helpsteer2`][hs2] | helpfulness, correctness, coherence, complexity, verbosity; human gold | 5,190 | 101,620 | CC-BY-4.0 |
| ✗ [`feedback_collection`][fc] | grade against the row's own rubric, GPT-4 gold¹ | 99,952 shared | | CC-BY-4.0 |
| ✗ [`helpsteer3`][hs3] | graded preference, by domain⁶ | 2,017 | 38,459 | CC-BY-4.0 |

Eval and train are the **questions** in each split; "shared" marks a single-split source and
"per lang" a per-language count (MASSIVE has 51 languages; `--langs` picks which run). `--n` samples **rows**,
and typed_decisions and helpsteer2 ask five questions per row. Which split each source uses is in
each source's template file. Sources marked ✗ are excluded from `all` runs (`default: false`) but run
when named.

[td]: https://huggingface.co/datasets/LocalLLaMA/typed-decisions
[b77]: https://huggingface.co/datasets/mteb/banking77
[mi]: https://huggingface.co/datasets/mteb/amazon_massive_intent
[ms]: https://huggingface.co/datasets/mteb/amazon_massive_scenario
[clinc]: https://huggingface.co/datasets/clinc/clinc_oos
[ge]: https://huggingface.co/datasets/google-research-datasets/go_emotions
[rb]: https://huggingface.co/datasets/allenai/reward-bench
[arena]: https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k
[pi]: https://huggingface.co/datasets/deepset/prompt-injections
[bq]: https://huggingface.co/datasets/google/boolq
[hs2]: https://huggingface.co/datasets/nvidia/HelpSteer2
[fc]: https://huggingface.co/datasets/prometheus-eval/Feedback-Collection
[hs3]: https://huggingface.co/datasets/nvidia/HelpSteer3

¹ Only one split exists, so eval and training rows are carved out of it disjointly.
² Share-alike would attach to a redistributed adaptation and it is unsettled whether model weights
count, so this source is evaluation-only. Another model may have trained on BoolQ, which a run records
when the model reports its own training sources.

³ GoEmotions is multi-label; the rows carrying exactly one emotion are kept (85% of them), so the
question stays a single choice over all 28. Counts above are after that filter.
⁴ The preferred response takes position `a` on even rows and `b` on odd ones, so a model that
always picks the first cannot score above chance.
⁵ Real head-to-head votes from LMArena. The dataset's own ordering is kept: the votes belong to the
positions the voters actually saw.
⁶ Long states: 85% of rows exceed a 512-token window (median 1,249 tokens), and most exceed the
1,024-token budget of the models trained here, which skip such rows rather than cut them. Rows
carry their domain (code, general, multilingual, stem), so results break down by domain.

MASSIVE defaults to 12 languages; `--langs en` runs one, `--langs all` all 51. A non-English run
needs a model that reads them. Every language has the same rows, so scores compare directly.

## Guarantees

- **Policy is enforced in code.** `build(..., Split.TRAIN)` raises for a source whose `use` policy
  or splits disallow training.
- **Train and eval never overlap.** Separate Hugging Face splits give this by construction; a
  source with one split holds out the evaluation sample first and draws training rows from the rest.
- **Pinned revisions.** Each build resolves the dataset's commit SHA and loads exactly that version.
  The SHA, the sampled row indices, the seed and the prompt version are recorded with the build.
- **Seeded samples**, never "the first N".
- **Versioned wording.** Question text is part of the measurement; bump `PROMPT_VERSION` in
  `sources.py` when instructions or criteria change. Builds and reports key on it, so runs made at
  different versions are never compared.
- **Validated.** Every row must parse as TypeSafe questions and answers, gold must be among the
  options, and probabilities must sum to 1.
- **Nothing is redistributed.** Builds are cached under `data/cache/` (git-ignored); datasets are
  downloaded from their source.

## Adding a source

Write one YAML file under `data/templates/` (see `data/templates.py` for the format, and the folder's
README for the questions). There is no registry to edit: the file **is** the source, and `SOURCES` is
every such file, read. Bump `PROMPT_VERSION` if you change the wording of an existing source.

The source-level keys of that file:

| Key | Meaning |
|---|---|
| `source` | the source's name; several files may share it, one per `config` |
| `dataset` | the Hugging Face dataset id |
| `configs` | which configs to load; `[default]` for a dataset with one, or e.g. the languages of MASSIVE |
| `splits` | `{eval: <split>, train: <split> or null}`; naming the same split for both makes the builder carve disjoint samples |
| `license` | recorded with every run and reported |
| `use` | `{benchmark: bool, train: bool}`, enforced by the builder |
| `commercial` | whether the source may train a model that is sold or served commercially; required. `training.prepare_data --commercial` builds only these |
| `labels` | optional, where the full label set lives (below) |
| `keep` | optional, a row filter named in `sources.py` (`KEEPS`); rows it rejects are dropped **before** sampling |
| `prepare` | optional, reshapes the whole split first (`PREPARES`) |
| `oversample` | optional, `{fn: <name>, share: <s>}` (`OVERSAMPLES`) |
| `parquet` | optional, read the Parquet conversion rather than the loading script |
| `default` | optional, `false` keeps a source out of `all` runs; it still runs when named |

Whether a model trained on a source is a fact about the model, not the dataset, so it lives with the
models: a model trained here reads it from its own data provenance (`trained_on` in
`benchmarks/models.py`); for any other model it is unknown unless its `Model` subclass sets it.

`labels` is read once, in full, before sampling, so every row offers the same options:

```yaml
labels: [test, label_text]        # the distinct values of a column, sorted (banking77)
labels: [train, label_text, en]   # the same, from a named config (MASSIVE: labels come from en)
labels: [features, intent]        # the names of a ClassLabel column, in the dataset's own order
```

The `features` form covers datasets that store an integer plus the names in the schema, including
when the column is a *list* of labels:

```python
# clinc_oos      row {'text': 'how would you say fly in italian', 'intent': 61}
#                features['intent'] = ClassLabel(names=[...151...]) -> names[61] = 'translate'
# go_emotions    row {'text': "I'm really sorry ...", 'labels': [25]}
#                features['labels'] = List(ClassLabel(names=[...28...])) -> names[25] = 'sadness'
```

That order is the dataset's own, not sorted, because the order options are shown in changes a
model's answer.

`keep` is for rows a source cannot turn into a question. GoEmotions is multi-label, so rows with
more than one emotion are dropped rather than forced into a single choice:

```yaml
keep: single_label                # in data/sources.py: len(r["labels"]) == 1
```

Filtering before sampling means `--n 500` gives 500 usable rows, and the recorded row indices
refer to the filtered dataset.
