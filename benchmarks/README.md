# Benchmarks

Runs models over the rows built by [`data/`](../data/README.md) and scores their answers. Every
model answers the same TypeSafe request, so the runner and the metrics never learn which model
replied:

    model.system_one(state, questions) -> {"model": ..., "answers": {...}, "usage": {...}}

```bash
pip install -e '.[bench]'

python -m benchmarks.run banking77                            # 500 seeded rows, a local checkpoint
python -m benchmarks.run typed_decisions boolq --n 200
python -m benchmarks.run massive_intent --langs en de ja --subfolder multilingual
python -m benchmarks.run all --n 500 helpsteer2=100           # per-source sample sizes
TYPESAFE_API_KEY=... python -m benchmarks.run banking77 --backend jev

python -m benchmarks.report                                   # the two most recent runs
python -m benchmarks.report --runs RUN_A RUN_B --group "mine=banking77,boolq" --out report.md
```

## Models

| Backend | What it runs | Needs |
|---|---|---|
| `dadvar` | a local checkpoint: a hub id or a directory, `--subfolder` for one inside a repo | the weights, downloaded on first use |
| `jev` | TypeSafe's hosted model, pinned to `jev-1.13.0` | `TYPESAFE_API_KEY`; billed per input token |

Adding one is a subclass of `Model` in `models.py` with a `system_one` method, plus `@register`.
Set `model_id`, `model_version` and `settings` correctly: they are part of the cache key, so a
different model, version or dtype never reuses another's answers.

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

## Output

Each run writes `benchmarks/results/<timestamp>-<model>/`:

- `run.json`: settings, dataset revisions and sampled indices, every metric, latency, tokens, cost,
  cache hits, and whether the model's publisher says it trained on each source.
- `predictions.jsonl`: per row, the answers as TypeSafe types plus per-question scores. No dataset
  text; the full request and raw response live in the cache.

Reports are Markdown, built from `run.json` alone, and state each dataset's revision, sample, seed
and prompt version. `--group NAME=SRC,SRC` pools sources into a named suite.
`benchmarks/results/` and `benchmarks/cache/` are git-ignored.
