"""Score TypeSafe answers against gold answers.

Per question, a prediction and its gold are both TypeSafe answers (`ChoiceAnswer`, `NoulAnswer`,
`ScoreAnswer`). Both are turned into a distribution over the same outcomes:

    choice  the labels                noul  [false, true]           score  levels 0..k-1

Metrics, averaged over questions:

    accuracy   predicted label (argmax; noul: P(true) >= 0.5) == gold label (same rule)
    brier      sum over outcomes of (p - g)^2; 0 is perfect, 2 is maximally wrong
    log_loss   -sum g * log p (cross-entropy against the gold distribution)
    ece        top-label expected calibration error, 15 equal-width bins
    soft_accuracy   sum p * g: probability mass placed on the gold distribution
    kl         sum g * log(g / p): KL divergence from the gold distribution
    tv         0.5 * sum |p - g|: total variation distance
    score_mae  |predicted expected score - gold expected score|, score questions only
    within_one share of score questions with |predicted - gold expected score| <= 1

A failed prediction (an exception from the model) counts as wrong, with a uniform distribution.
"""

import math
from collections import defaultdict

import numpy as np

from data.targets import outcomes as outcomes_of
from data.targets import target_distribution

EPS = 1e-12


def distribution(question, answer):
    """The probabilities of a TypeSafe answer over the question's outcomes."""
    return np.array(target_distribution(question, answer.model_dump(mode="json")))


def score_question(question, gold, prediction):
    """Metrics for one question. `prediction` is a TypeSafe answer, or None if the call failed."""
    outcomes = outcomes_of(question)
    g = distribution(question, gold)
    p = distribution(question, prediction) if prediction is not None else np.full(len(outcomes), 1.0 / len(outcomes))
    correct = prediction is not None and int(p.argmax()) == int(g.argmax())
    out = {
        "type": question["type"],
        "correct": bool(correct),
        "confidence": float(p.max()),
        "brier": float(((p - g) ** 2).sum()),
        "log_loss": float(-(g * np.log(np.clip(p, EPS, 1.0))).sum()),
        "soft_accuracy": float((p * g).sum()),
        "kl": float((g * np.log(np.clip(g, EPS, 1.0) / np.clip(p, EPS, 1.0))).sum()),
        "tv": float(0.5 * np.abs(p - g).sum()),
        "failed": prediction is None,
    }
    if question["type"] == "score":
        levels = np.arange(len(outcomes))
        out["score_error"] = abs(float((levels * p).sum()) - float(gold.score))
    return out


def ece(confidence, correct, bins: int = 15) -> float:
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (confidence > lo) & (confidence <= hi) if i else (confidence >= lo) & (confidence <= hi)
        if m.any():
            total += m.mean() * abs(confidence[m].mean() - correct[m].mean())
    return float(total)


def summarize(scored):
    """Aggregate a list of per-question metric dicts."""
    if not scored:
        return {}
    conf = [s["confidence"] for s in scored]
    corr = [s["correct"] for s in scored]
    out = {
        "questions": len(scored),
        "accuracy": float(np.mean(corr)),
        "brier": float(np.mean([s["brier"] for s in scored])),
        "log_loss": float(np.mean([s["log_loss"] for s in scored])),
        "ece": ece(conf, corr),
        "soft_accuracy": float(np.mean([s["soft_accuracy"] for s in scored])),
        "kl": float(np.mean([s["kl"] for s in scored])),
        "tv": float(np.mean([s["tv"] for s in scored])),
        "failed": int(sum(s["failed"] for s in scored)),
    }
    errors = [s["score_error"] for s in scored if "score_error" in s]
    if errors:
        out["score_mae"] = float(np.mean(errors))
        out["within_one"] = float(np.mean([e <= 1.0 for e in errors]))
    return out


def summarize_by(scored, key):
    groups = defaultdict(list)
    for s in scored:
        groups[s[key]].append(s)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def percentile(values, q):
    return float(np.percentile(values, q)) if values else math.nan
