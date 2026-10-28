import json

import pytest
from pydantic import TypeAdapter
from typesafe_sdk import Answer

from benchmarks.metrics import ece, score_question, summarize
from data.targets import target_distribution

ANSWER = TypeAdapter(Answer)
CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}
SCORE = {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]}
NOUL = {"type": "noul", "instructions": "Is it?"}


def answer(d):
    return ANSWER.validate_json(json.dumps(d))


def choice(probs):
    return answer(
        {
            "type": "choice",
            "choice": max(probs, key=probs.get),
            "confidence": max(probs.values()),
            "probabilities": probs,
        }
    )


def test_target_distribution_per_type():
    assert target_distribution(NOUL, {"type": "noul", "noul": 0.8}) == pytest.approx([0.2, 0.8])
    assert target_distribution(CHOICE, {"type": "choice", "probabilities": {"b": 3, "a": 1}}) == [0.25, 0.75]
    assert target_distribution(SCORE, {"type": "score", "probabilities": {"2": 1.0}}) == [0.0, 0.0, 1.0]
    assert target_distribution(SCORE, {"type": "score"}) == pytest.approx([1 / 3] * 3)


def test_a_perfect_prediction_scores_perfectly():
    gold = choice({"a": 1.0, "b": 0.0})
    s = score_question(CHOICE, gold, gold)
    assert s["correct"] and s["brier"] == 0.0 and s["tv"] == 0.0 and s["soft_accuracy"] == 1.0


def test_a_failed_call_is_wrong_and_uniform():
    s = score_question(CHOICE, choice({"a": 1.0, "b": 0.0}), None)
    assert not s["correct"] and s["failed"] and s["confidence"] == 0.5


def test_score_error_is_on_the_expected_level():
    legend = {"0": "low", "1": "mid", "2": "high"}
    gold = answer(
        {"type": "score", "score": 2.0, "confidence": 1.0, "legend": legend, "probabilities": {"0": 0, "1": 0, "2": 1}}
    )
    pred = answer(
        {
            "type": "score",
            "score": 1.0,
            "confidence": 0.5,
            "legend": legend,
            "probabilities": {"0": 0.5, "1": 0, "2": 0.5},
        }
    )
    assert score_question(SCORE, gold, pred)["score_error"] == pytest.approx(1.0)


def test_ece_is_zero_when_confidence_matches_accuracy():
    assert ece([1.0, 1.0, 0.5, 0.5], [1, 1, 1, 0]) == pytest.approx(0.0)
    assert ece([0.9, 0.9], [0, 0]) == pytest.approx(0.9)


def test_summarize_counts_and_averages():
    gold = choice({"a": 1.0, "b": 0.0})
    scored = [score_question(CHOICE, gold, gold), score_question(CHOICE, gold, choice({"a": 0.0, "b": 1.0}))]
    out = summarize(scored)
    assert out["questions"] == 2 and out["accuracy"] == 0.5 and out["failed"] == 0
