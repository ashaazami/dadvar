# Derived from Laya (https://github.com/NandhaKishorM/laya) at 573e5b6, Apache-2.0.
# Modified for Dadvar; see NOTICE.
"""Dadvar: Fast, non-autoregressive System 1 decision engine with calibrated probabilities."""

from .agent import Agent, RLAgent, load
from .common import (
    QTYPES,
    QTYPE_NAMES,
    confidence_from_probs,
    ece_score,
    proper_reward,
    render_options,
    td_lambda_targets,
)

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "proper_reward",
    "td_lambda_targets",
    "ece_score",
    "confidence_from_probs",
    "render_options",
    "QTYPES",
    "QTYPE_NAMES",
    "__version__",
]
