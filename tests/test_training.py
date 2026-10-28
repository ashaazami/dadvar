import random

import torch

from training.finetune import epoch_items, fit_one_temp
from training.prepare_data import build_training_item, reference_form


def _item(source, row, qtype, variant=1):
    return {"source": source, "row": row, "variant": variant, "qtype": qtype}


def test_build_training_item_targets_and_markers(tok):
    q = {"type": "choice", "instructions": "Which intent?", "criteria": {"refund": None, "card": None}}
    gold = {"type": "choice", "choice": "card", "confidence": 1.0, "probabilities": {"refund": 0.0, "card": 1.0}}
    it = build_training_item(tok, 64, {"message": "charged twice"}, q, gold)
    assert it["target"] == [0.0, 1.0] and it["label"] == 1 and len(it["markers"]) == 2
    assert build_training_item(tok, 5, {"message": "charged twice"}, q, gold) is None  # too long: skipped, not cut


def test_reference_form():
    assert reference_form("Is it spam?") == "none"
    assert reference_form("Is `message` spam?") == "top"
    assert reference_form({"question": "Does `a` answer `b`?"}) == "pair"


def test_balance_by_source_caps_each_source():
    items = [_item("big", r, 0) for r in range(100)] + [_item("small", r, 0) for r in range(5)]
    sample = epoch_items(items, per_source=10, seed=0)
    assert sum(it["source"] == "big" for it in sample) == 10
    assert sum(it["source"] == "small" for it in sample) == 5


def test_balance_by_type_gives_each_type_a_share():
    items = [_item("s%d" % t, r, t) for t in range(3) for r in range(100)]
    sample = epoch_items(items, per_source=0, seed=0, balance="type", epoch_size=60)
    assert sorted(sum(it["qtype"] == t for it in sample) for t in range(3)) == [20, 20, 20]


def test_a_rows_questions_stay_together():
    items = [_item("s", r, 0) for r in range(50) for _ in range(3)]
    sample = epoch_items(items, per_source=30, seed=1)
    rows = {}
    for it in sample:
        rows[it["row"]] = rows.get(it["row"], 0) + 1
    assert set(rows.values()) == {3}


def test_temperature_softens_overconfident_logits():
    rng = random.Random(0)
    sel = []
    for _ in range(200):  # logits say 0.99, the target says 0.7
        sel.append(([4.6, 0.0], [0.7, 0.3]) if rng.random() < 0.5 else ([0.0, 4.6], [0.3, 0.7]))
    t = fit_one_temp(sel)
    assert t > 1.5
    p = torch.softmax(torch.tensor([4.6, 0.0]) / t, -1)[0].item()
    assert abs(p - 0.7) < 0.05
    assert fit_one_temp(sel[:5]) == 1.0  # too few to fit
