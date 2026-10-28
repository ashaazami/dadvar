# Training experiments

One section per trained model, newest first. Each records the base model, the data, the
hyperparameters and the results as run, so a write-up can cite them without re-deriving
anything. Runs live in `training/runs/<name>/`; every saved model's `dadvar_model.json` holds its
full arguments, data provenance, validation results and history.

Terms used throughout: an *item* is one question about one state (a dataset row with five
questions gives five items); *validation* is the held-out set `training/finetune.py` takes from the
training items, not a benchmark evaluation split.

No runs yet. Dataset v1 is described in [`DATASET.md`](DATASET.md) and the recipe in
[`training/README.md`](../training/README.md).

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
