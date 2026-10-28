"""Dadvar: a decision model with calibrated probabilities over a question's options.

    model.py      the architecture: one logit per option, trained by training/finetune.py
    sequence.py   the model input for one question, built whole and never cut
    text.py       what the model is shown: the state and a question's options as text
    batch.py      padding encoded questions into one batch

Import from the module that holds what you need, e.g. `from dadvar.model import DadvarModel`. The
package exports nothing of its own, so importing it costs no torch import time.

The distribution's version lives in pyproject.toml; read it with
`importlib.metadata.version("dadvar")`.
"""
