# Benchmarks

Runs models over the rows built by [`data/`](../data/README.md) and scores their answers. Every
model answers the same TypeSafe request, so the runner and the metrics never learn which model
replied:

    model.system_one(state, questions) -> {"model": ..., "answers": {...}, "usage": {...}}

```bash
pip install -e '.[bench]'

MODEL="--model training/runs/<run>/model"                    # a model trained here

python -m benchmarks.run banking77 $MODEL                     # 500 seeded rows
python -m benchmarks.run typed_decisions boolq --n 200 $MODEL
python -m benchmarks.run massive_intent --langs en de ja $MODEL
python -m benchmarks.run all --n 500 helpsteer2=100 $MODEL    # per-source sample sizes
TYPESAFE_API_KEY=... python -m benchmarks.run banking77 --backend jev

python -m benchmarks.report                                   # the two most recent runs
python -m benchmarks.report --runs RUN_A RUN_B --group "mine=banking77,boolq" --out report.md
```

## Models

| Backend | What it runs | Needs |
|---|---|---|
| `dadvar-model` (default) | a model trained by `training/finetune.py`: `--model` is its `model/` folder | the folder; `--device` cpu, mps or cuda |
| `jev` | TypeSafe's hosted model, pinned to `jev-1.14.0` (`--model` picks another version) | `TYPESAFE_API_KEY`; billed per input token |

Adding one is a subclass of `Model` in `models.py` with a `system_one` method, plus `@register`.
Set `model_id`, `model_version` and `settings` correctly: they are part of the cache key, so a
different model, version or dtype never reuses another's answers. A model that knows which sources it
trained on sets `trained_on`; the run then records, per source, whether it was seen in training.
Laya, for example, would be run from its own package through such a subclass.

## Cache

Answers are stored in `benchmarks/cache/predictions.sqlite`, keyed by the request (a sha256 of the
canonical `{state, questions}`) and the model. A rerun calls the model only for rows it has not
seen, so re-scoring, new metrics and new reports cost nothing and a hosted model is never paid for
twice. Failed calls are not cached. `--no-cache` bypasses it, which is what latency measurements
need.

```bash
python -m benchmarks.cache          # rows, tokens and cost per model
```

Latency in a run comes only from calls actually made; if every row was a cache hit, the stored
timings are reported and flagged `from_cache`.

## Metrics

Per question, aggregated overall, per source and per question type. Reports print accuracy, Brier
and score MAE; `--all-metrics` adds the rest.

| Metric | Meaning |
|---|---|
| accuracy | the predicted label matches gold (noul: P(true) ≥ 0.5; score: argmax level) |
| brier | Σ (p − g)² over the outcomes; 0 is perfect |
| score_mae | \|expected score − gold score\|, score questions only |
| soft_accuracy | Σ p·g: probability placed on the gold distribution |
| log_loss | −Σ g log p against the gold distribution |
| kl | Σ g log(g / p) |
| tv | ½ Σ \|p − g\| |
| ece | top-label expected calibration error, 15 bins |
| within_one | share of score questions within one level of gold |

A call that raises counts as wrong with a uniform distribution; it is not skipped.

## The sequence, and skipped rows

A local model answers the token sequence built from the question, not the question itself
(`dadvar/sequence.py`): `[CLS] instructions [SEP] options [SEP] state [SEP]`, with one `[MASK]`
marker per option. The whole sequence is built and nothing is cut, so a question longer than the
model's `max_len` is skipped instead.

A run never answers a cut input: a cut option list can make two options the same tokens, and a
cut state can lose the part the question is about, so such a score would measure the budget rather
than the model. A skipped row is not sent to the model and not scored; `predictions.jsonl` records
it with the reason (`"skipped"`). Reports drop a row any compared run skipped from **every** run,
so all of them are scored on the identical set; `--include-skipped` scores each on every row it
answered. A hosted model receives the question as sent and never skips.

## Output

Each run writes `benchmarks/results/<timestamp>-<model>/`:

- `run.json`: settings, dataset revisions and sampled indices, every metric, latency, tokens, cost,
  cache hits, and whether the model's publisher says it trained on each source.
- `predictions.jsonl`: per row, the answers as TypeSafe types and the whole per-question score, or
  `skipped` with the reason. No dataset text; the full request and raw response live in the cache. The score is stored whole rather than as a readable subset because a report
  that drops rows has to re-aggregate from it.

Reports are Markdown, built from `run.json` and `predictions.jsonl`, and state each dataset's
revision, sample, seed and prompt version. `--group NAME=SRC,SRC` pools sources into a named suite.
`benchmarks/results/` and `benchmarks/cache/` are git-ignored.

## Diagnostics

`python -m benchmarks.diagnostics --backend <backend> [--model ...]` runs five small suites that the
benchmark's per-source scores cannot show: whether the model reads the question (the same texts
asked different yes/no questions), a ladder of reasoning levels L1-L6 with one-fact twins, reference
following (two texts in one state), option-order stability, and the hand-made out-of-domain set.
Everything is seeded, so every model gets the same questions. Results go to
`benchmarks/results/diagnostics/`. Levels and suites are described in `docs/DATASET.md`.

## Option counts

Choice questions are evaluated with their full label sets by default. `--options 5` cuts every
choice question to 5 options (the gold label plus similar and random distractors, fixed per row),
recorded in each source's run info and in the run folder name (`-k5`); reports refuse to compare
runs with different settings.
