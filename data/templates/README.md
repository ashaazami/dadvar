# Sources

One file per source: what to load (dataset, configs, splits, license, what it may be used for), the
state (which dataset column becomes which state key), and every question, with its wordings,
opposites, criteria, levels and gold function.

```
data/templates/<dataset>/<source>.yaml      one source, e.g. ethics/deontology.yaml;
                                            a dataset with one source: <dataset>/default.yaml
data/templates/<dataset>/_<name>.yaml       questions shared by others, pulled in with `include:`
```

**These files are the registry.** There is no source list in code: `data/templates.py` reads the
folder, and each file's source-level keys are that source's entry in `SOURCES`. A file may name a
Python function for the parts that have to be code — `record:`, `prepare:`, `keep:`, `oversample:`,
`rows:`, and a question's `gold:` — and those live in `data/sources.py`.

The format and the functions a file can name are documented in `data/templates.py`. Placeholders are
Jinja2 (`{{ field }}`, a row's field); a misspelled one fails the build.

Checked when a file loads: every backtick reference names a declared state key or one of the
question's own options; every named function exists; a state `role:` is known and, for a two-part
source, comes as a `context` + `reply` pair; the files of a multi-config source agree on their
source-level keys. Checked per row: every mapped column exists.

See any source as the model gets it:

    python -m data.preview amazon_reviews
    python -m data.preview ethics --config deontology --n 1
    python -m data.preview banking77 --split train --variant 1 --n 4

Every source is here: 38 files for 36 sources (ethics has one file per config), plus 2 shared files,
in one folder per dataset.

A file also declares what the training augmentations may do with the source, each next to what it
describes rather than in a block of its own:

- `train: false` and `train_keep:` at the top, beside the other source-level keys, and `commercial:`
  (required), whether the source may train a model that is sold or served commercially.
- a `role:` on a `state:` entry: `text` (the one text a reference copy re-places in a larger state),
  or `context` + `reply` (a two-part source, where a mismatched copy takes another row's reply).
- on a question: `on_mismatch: lowest | affirm`, what a mismatched copy answers instead; and
  `asked: mismatch`, a question that only an augmentation adds — the `relation` question, which is
  declared like any other, with its own wordings, grade and gold, and is never asked of a plain row.

Training rows are always dataset v1's (`docs/DATASET.md`); `data/build.py` holds the shares and layouts.

One source is not templated: typed_decisions carries its own questions in the data, so its file
declares only levels and grades, and a Python row builder (`rows: typed_decisions`) does the rest.
Its questions' `form` is read off each wording, since it has no `form:` to declare.
