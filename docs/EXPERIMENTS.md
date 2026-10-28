# Training experiments

One section per trained model, newest first. Each records the base model, the data, the
hyperparameters and the results as run, so a write-up can cite them without re-deriving
anything. Runs live in `training/runs/<name>/`; every saved model's `dadvar_model.json` holds its
full arguments, data provenance, validation results and history.

Terms used throughout: an *item* is one question about one state (a dataset row with five
questions gives five items); *validation* is the held-out set `training/finetune.py` takes from the
training items, not a benchmark evaluation split.

Dataset v1 is described in [`DATASET.md`](DATASET.md) and the recipe in
[`training/README.md`](../training/README.md). These are early, single runs: the numbers come from one
seed and one sample of rows, the code may still have bugs, and the generated questions and gold answers
have not been checked by hand.

## v1-base (2026-10-28)

- **Model:** `answerdotai/ModernBERT-base` encoder (149.0M parameters) with 2 head layers and the scorer
  (14.8M), 163.8M in all.
- **Data:** `training/data/v1_full1024.pt`, dataset v1 (`PROMPT_VERSION` 1), the 29-source research mix
  (`commercial: false`), 733,869 items at a 1,024-token budget; 731,032 for training and 1,484 (about 50
  per source) held out for validation.
- **Training:** `python -m training.finetune training/runs/v1-base --items training/data/v1_full1024.pt
  --epochs 1 --micro-batch 16 --grad-accum 2 --balance type --epoch-size 115000 --save-every 100`:
  3,595 updates of 32 items, encoder learning rate 2e-5, head 1e-4, 5% warmup. Apple M5 Pro (MPS),
  Python 3.14.7, torch 2.14.0, transformers 5.17.0; 3.1 s per update, 3.2 hours.
- **Validation:** loss 0.716, accuracy 0.671. By type: choice 0.727, noul 0.764, score 0.470. By grade:
  G1 0.717, G2 0.869, G3 0.540, G4 0.622, G5 0.818 (22 items). Lowest sources: helpsteer3 0.30, paws 0.36
  (a yes/no task, so below chance: to investigate), arena_preference 0.36, hh_rlhf 0.48, stsb 0.47. The
  fitted temperatures (choice, score, noul) are 1.06, 1.02 and 1.02. With about 50 items per source,
  each source's figure is uncertain by several points.
- **Benchmark:** runs `20261028-122233-v1-base` and `20261028-122312-jev-1.14.0` (`benchmarks.run all`,
  seed 13, 500 rows per source and per MASSIVE language), compared on identical rows in
  [`results/v1-base-vs-jev-1.14.0.md`](results/v1-base-vs-jev-1.14.0.md). In English, v1-base scored
  higher than Jev 1.14.0 on helpsteer2 (0.553 against 0.452), go_emotions (0.564 against 0.328),
  prompt_injections (0.853 against 0.724, 116 rows) and MASSIVE scenario (0.830 against 0.756), and lower
  on MASSIVE intent, banking77, clinc_oos, typed_decisions and boolq (0.606 against 0.938; not in the
  training mix). Outside English it is far behind (0.30 against 0.75 over MASSIVE's twelve languages),
  as expected from English-only training. Median latency 27 ms per row locally against 118 ms for Jev.
- **Diagnostics:** not run yet.

For each run, record:

- **Model:** encoder and commit, head layers, parameter counts.
- **Data:** the items file, its `*_provenance.json` (sources, revisions, counts), and `PROMPT_VERSION`.
- **Training:** the `training.finetune` command, hardware, software versions, time per update.
- **Validation:** loss and accuracy overall and by type, grade, reference form and source; the fitted
  temperatures.
- **Benchmark:** the `benchmarks.run` run id, and comparisons on identical rows (`benchmarks.report`).
- **Diagnostics:** the `benchmarks.diagnostics` result file.

Lessons from earlier runs, before this restart, that the recipe keeps:

- **Train the encoder from the first step.** With a frozen ModernBERT-base encoder the new layers did not
  learn the task at any learning rate tried (banking77: 0.06 accuracy after 150 updates, against 0.70
  with the encoder trained at 3e-5), so `--head-epochs` defaults to 0.
- **Questions must vary within a source,** or the model learns to ignore them: a model trained on sources
  that each asked one fixed question gave the same P(yes) to opposite questions about the same text.
  Hence wordings, opposites and several questions per state.
- **Reference following does not come for free.** It stayed near chance (0.44–0.63) on every path form
  after one epoch with the reference augmentations, while Jev 1.13.0 answered every form; it is the
  headline diagnostic to watch.
