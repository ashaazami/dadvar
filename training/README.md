# Training

The recipe follows the one Laya ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)) uses
to fine-tune its decision model: one sequence per question with a `[MASK]` marker per option, a soft
target over the options, and a temperature per question type fitted afterwards. Parts of the code
are adapted from Laya's; see [NOTICE](../NOTICE).

Two scripts: `prepare_data.py` turns the sources in [`data/`](../data/README.md) into training items,
and `finetune.py` trains a [`DadvarModel`](../dadvar/model.py) from a pretrained encoder.

```bash
pip install -e '.[bench]'

TOK="--tokenizer answerdotai/ModernBERT-base --max-len 1024"

# 1. Items from any allowed source's TRAIN split, one per question, whole sequences (dataset v1, `docs/DATASET.md`)
python -m training.prepare_data --out training/data/v1_full1024.pt v1 --langs en $TOK \
    --n 12000 --variants 3 --p-opposite 0.5 --per-source 30000
python -m training.prepare_data --out /tmp/items.pt typed_decisions --n 4 $TOK   # smoke test
# add --commercial for a model to sell: only sources whose file says `commercial: true`

# 2. Train the encoder and the decision head, balanced by question type
python -m training.finetune training/runs/base --items training/data/v1_full1024.pt \
    --epochs 1 --micro-batch 16 --grad-accum 2 --balance type --epoch-size 115000

# 3. Evaluate the result on the benchmark's evaluation split
python -m benchmarks.run typed_decisions --backend dadvar-model --model training/runs/base/model
```

Sources come from `data/`, which refuses one whose license does not allow training and guarantees
training rows never come from a benchmark's evaluation split. Each prepared file gets a
`*_provenance.json` recording the tokenizer, budget, sources, dataset revisions and counts; the
trained model records that provenance in its own config, which is how a benchmark run knows what the
model was trained on.

**Commercial use.** The default mix is for research: some sources are non-commercial, share-alike, have
no declared license, or were written by commercial models. Each source's file says whether it may train a
model that is sold or served (`commercial:`). Build with `--commercial` to use only those; the items'
provenance, and so the trained model's config, then records `"commercial": true`.

## Recipe

- **Everything from the first step,** with a small encoder learning rate (2e-5), a larger one for
  the new layers (1e-4), a warmup and cosine decay. `--head-epochs N` adds a first phase that trains
  only the new layers on a frozen encoder; it is off by default because on a frozen encoder they did
  not learn the task (`docs/EXPERIMENTS.md`).
- **Loss:** soft cross-entropy between the option probabilities and the gold distribution.
- **Balance:** `--balance source` (the default) draws at most `--per-source` items per source each
  epoch, a new sample each epoch, so a source with many items does not outweigh the rest and still
  contributes most of its items over several epochs. `--balance type --epoch-size N` draws a third of
  each epoch per question type, spread evenly over each type's sources; the command above uses it.
- **Validation:** `--val-per-source` items per source are held out before training. They pick the
  best checkpoint by validation loss, and the calibration temperatures (one per question type) are
  fitted on them afterwards, never on training items.
- **Continuing:** each phase leaves its last weights in `after-head/` or `after-full/`, and `best/`
  holds the lowest validation loss so far. `--init-from` starts a new run from any of them, with the
  same items file and seed so the validation split is unchanged.

Every item holds a whole sequence: a question longer than `--max-len` is skipped rather than cut, so
the model never trains on an input with something missing.

Results per run are in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).
`training/data/` and `training/runs/` are git-ignored.
