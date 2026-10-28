# Dadvar

*Dadvar* (دادور) — Persian for "judge", literally "bearer of justice".

Training data, training and evaluation for **System One** decision models: models that answer a
structured question about a structured state in one pass, with a calibrated probability over the
answers. Requests and answers use the [TypeSafe System One](https://docs.typesafe.ai/concepts/system-one)
schema, with three question types:

| Type | Asks | Answer |
|---|---|---|
| `choice` | which of these options applies | a probability per option |
| `score` | where on an ordered scale of levels | a probability per level, and the expected level |
| `noul` | is this statement true | P(true) |

The repository has three parts:

- **Dataset** (`data/`): 36 public Hugging Face datasets converted into System One rows: a state, its
  questions and gold answers. One YAML file per source declares what to load and the questions to ask.
  A training recipe (dataset v1) adds varied wordings, opposite questions, and augmentations that teach
  a model to answer from the part of the state a question references.
- **Training** (`training/`, `dadvar/`): fine-tunes a decision model from a pretrained encoder, with
  soft targets over the options and a calibration temperature per question type.
- **Evaluation** (`benchmarks/`): runs any model — one trained here or TypeSafe's hosted Jev — on each
  source's evaluation split, and scores accuracy, calibration and latency on identical rows. Diagnostic
  suites test what per-source scores cannot show: whether a model reads the question, follows
  references into the state, and ignores option order.

## Install

```bash
git clone https://github.com/ashaazami/dadvar.git
cd dadvar
python -m venv .venv && source .venv/bin/activate
pip install -e '.[bench]'
```

Python 3.10+. Datasets and encoder weights are downloaded on first use and cached under
`~/.cache/huggingface/`.

## Quick start

```bash
TOK="--tokenizer answerdotai/ModernBERT-base --max-len 1024"

# 1. Dataset: training items for the v1 mix (training splits only)
python -m training.prepare_data --out training/data/v1_full1024.pt v1 --langs en $TOK \
    --n 12000 --variants 3 --p-opposite 0.5 --per-source 30000

# 2. Training
python -m training.finetune training/runs/v1-base --items training/data/v1_full1024.pt \
    --epochs 1 --micro-batch 16 --grad-accum 2 --balance type --epoch-size 115000

# 3. Evaluation, on the evaluation splits
MODEL="--model training/runs/v1-base/model"
python -m benchmarks.run all $MODEL
python -m benchmarks.diagnostics $MODEL
TYPESAFE_API_KEY=... python -m benchmarks.run all --backend jev     # a reference model, billed
python -m benchmarks.report --runs <run> <run>
```

See a source's questions exactly as a model receives them:

```bash
python -m data.preview banking77
```

## Dataset

Every source pins its dataset revision, samples with a seed, and declares its license and what it may
be used for; the builder enforces it. Training rows never come from a split the benchmark evaluates
on, and three sources (BoolQ, CommonsenseQA, SummEval) are evaluation-only. Of the rest, 29 are in the
training mix; multi-step math and multi-hop sources stay in the benchmark only, as a check on what a
one-pass model should be uncertain about.

- [data/README.md](data/README.md): the builder, the sources and the train/eval guarantees.
- [data/templates/README.md](data/templates/README.md): the source file format.
- [docs/DATASET.md](docs/DATASET.md): the v1 training recipe: scope, reference augmentations, levels.
- [docs/BENCHMARK_DATASETS.md](docs/BENCHMARK_DATASETS.md): every dataset considered, with its license.

Adding a source means one YAML file under `data/templates/`, plus any conversion function it needs in
`data/sources.py`.

## Training

`DadvarModel` (`dadvar/model.py`) is a pretrained encoder with a question-type embedding, two
transformer layers and a scorer: each option gets a `[MASK]` marker in the input, and the vector at
each marker is scored to one logit. The input for one question is built whole and never cut
(`dadvar/sequence.py`): a question longer than the token budget is skipped, in training and in
evaluation.

- [training/README.md](training/README.md): the recipe — loss, balancing, validation, calibration,
  continuing a run.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): what each run used and scored.

A run saves its model, tokenizer and config to `training/runs/<run>/model`:

```python
from dadvar.model import DadvarModel

net, cfg = DadvarModel.load("training/runs/v1-base/model")
cfg["max_len"], cfg["temperature"], cfg["items_provenance"]
```

## Evaluation

Every model answers the same System One request, so the runner and the metrics never learn which model
replied. Answers are cached by request and model, so re-scoring costs nothing and a hosted model is
never paid for twice. Each run records the dataset revisions, the sample, the prompt version, and
whether the model trained on each source.

- [benchmarks/README.md](benchmarks/README.md): the runner, the metrics, the cache, the diagnostics and
  reports.

Adding a model means one subclass of `Model` with a `system_one(state, questions)` method, in
`benchmarks/models.py`; `DadvarCheckpoint` there is the reference for turning a state and its
questions into answers with a model trained here.

## Development

```bash
pip install -e '.[bench,dev]'
black dadvar benchmarks data training         # formatting, line length 120 (pyproject.toml)
ruff check dadvar benchmarks data training
```

CI runs both, an import check on Python 3.10–3.13, and a package build.

## Background

The model's design and training recipe take their ideas from
[Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations: an encoder with a question-type
embedding and a few transformer layers on top, one `[MASK]` marker per option scored to one logit,
soft targets over the options, and a temperature per question type fitted after training. The model,
input construction and calibration code are adapted from Laya's (Apache-2.0); each adapted file says so
at the top, and [NOTICE](NOTICE) lists them. The dataset layer, training mix and benchmark are this
project's.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). No dataset content and no model weights are
stored here: datasets are downloaded at run time and used under their own licenses.
