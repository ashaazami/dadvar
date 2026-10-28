import pytest

from dadvar.text import NOUL_FALSE, NOUL_TRUE, QTYPE_NAMES, QTYPES, option_texts, state_json, state_paths


def test_choice_options_keep_label_order_and_descriptions():
    q = {"t": "choice", "ins": "", "crit": {"refund": "money back", "other": None, 3: ""}}
    assert option_texts(q) == ["refund: money back", "other", "3"]


def test_structured_criteria_render_as_json():
    q = {"t": "choice", "ins": "", "crit": {"a": {"what": "x", "n": 1}}}
    assert option_texts(q) == ['a: {"what": "x", "n": 1}']


def test_score_levels_are_numbered_from_zero():
    q = {"t": "score", "ins": "", "crit": ["low", "high"]}
    assert option_texts(q) == ["level 0: low", "level 1: high"]


def test_noul_defaults_and_custom_labels():
    assert option_texts({"t": "noul", "ins": "", "crit": None}) == [
        "false: " + NOUL_FALSE,
        "true: " + NOUL_TRUE,
    ]
    q = {"t": "noul", "ins": "", "crit": {"true": "it holds"}, "labels": {"false": " no ", "true": "yes"}}
    assert option_texts(q) == ["no: " + NOUL_FALSE, "yes: it holds"]


@pytest.mark.parametrize(
    "labels",
    [{"false": "a", "true": "a"}, {"false": "a"}, {"false": "a", "true": " "}, {"false": "a", "true": 1}, ["a", "b"]],
)
def test_bad_noul_labels_are_refused(labels):
    with pytest.raises(ValueError):
        option_texts({"t": "noul", "ins": "", "crit": None, "labels": labels})


def test_labels_only_for_noul():
    with pytest.raises(ValueError):
        option_texts({"t": "choice", "ins": "", "crit": {"a": None}, "labels": {"false": "n", "true": "y"}})


def test_state_json_is_compact_and_keeps_unicode():
    assert state_json({"a": "é", "b": [1]}) == '{"a": "é", "b": [1]}'
    assert state_json("plain") == "plain"


def test_state_paths_writes_one_line_per_leaf():
    state = {"ticket": {"messages": [{"text": "Thanks!"}], "tags": [], "meta": {}}, "n": 2}
    assert state_paths(state).split("\n") == [
        "ticket.messages[0].text: Thanks!",
        "ticket.tags: []",
        "ticket.meta: {}",
        "n: 2",
    ]


def test_question_type_ids_round_trip():
    assert {QTYPE_NAMES[i]: i for i in QTYPES.values()} == QTYPES
