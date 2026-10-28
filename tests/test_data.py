"""The source registry and the builder's offline pieces: no dataset is downloaded."""

import json
import random

import pytest

from data import SOURCES, Split
from data.build import limit_options, row_parts, sample_indices, validate
from data.templates import grade, train_sources, trains


def test_every_source_declares_its_policy():
    assert len(SOURCES) == 36
    for name, spec in SOURCES.items():
        assert isinstance(spec["commercial"], bool), name
        assert set(spec["use"]) == {"benchmark", "train"}, name
        assert spec["license"], name


def test_the_training_mix():
    mix = set(train_sources())
    assert len(mix) == 29
    assert {"boolq", "commonsense_qa", "summeval"}.isdisjoint(mix)  # evaluation only
    assert {"gsm8k", "math_qa", "aqua_rat", "tab_fact"}.isdisjoint(mix)  # out of scope, benchmark only
    assert all(trains(n) for n in mix)


def test_every_templated_question_has_a_grade():
    for name in train_sources():
        for t in SOURCES[name].get("templates", ()):
            if t.asked == "always":
                assert grade(name, t.id) in (1, 2, 3, 4, 5), (name, t.id)


def test_shared_split_samples_never_overlap():
    evaluation = set(sample_indices(1000, 100, 13, Split.EVAL, shared=True, eval_n=100))
    training = set(sample_indices(1000, 500, 13, Split.TRAIN, shared=True, eval_n=100))
    assert len(evaluation) == 100 and len(training) == 500 and not evaluation & training


def test_limit_options_keeps_the_gold_label_and_renormalises():
    labels = ["label %d" % i for i in range(20)]
    questions = {"q": {"type": "choice", "instructions": "?", "criteria": {l: None for l in labels}}}
    gold = {
        "q": {
            "type": "choice",
            "choice": "label 7",
            "confidence": 1.0,
            "probabilities": {l: float(l == "label 7") for l in labels},
        }
    }
    limit_options(questions, gold, random.Random(0), k=5)
    kept = list(questions["q"]["criteria"])
    assert len(kept) == 5 and "label 7" in kept
    assert sum(gold["q"]["probabilities"].values()) == pytest.approx(1.0)


def test_row_parts_reads_back_augmented_ids():
    assert row_parts("src/train/4") == ["src/train/4"]
    assert row_parts("src/train/4~sibling~src/train/9") == ["src/train/4", "src/train/9"]


def test_validate_rejects_a_gold_label_that_is_not_an_option():
    row = {
        "id": "x",
        "questions": json.dumps({"q": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}}}),
        "gold": json.dumps(
            {"q": {"type": "choice", "choice": "c", "confidence": 1.0, "probabilities": {"a": 0.0, "b": 0.0, "c": 1.0}}}
        ),
    }
    with pytest.raises(ValueError):
        validate(row)
