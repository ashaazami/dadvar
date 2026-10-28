import pytest
import torch

from dadvar.batch import collate
from dadvar.sequence import SPECIAL_TOKENS, assemble, encode_segments, fits, lengths

QUESTION = {"t": "choice", "ins": "which intent in `message` ?", "crit": {"refund": None, "card": None}}
STATE = {"message": "my card was charged twice [MASK]"}


def test_one_marker_per_option_in_layout_order(tok):
    ids, markers = assemble(tok, encode_segments(tok, STATE, QUESTION))
    assert len(markers) == 2
    assert all(ids[m] == tok.mask_token_id for m in markers)
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    assert ids.count(tok.sep_token_id) == SPECIAL_TOKENS - 1


def test_a_literal_mask_in_the_text_is_blanked(tok):
    ids, markers = assemble(tok, encode_segments(tok, STATE, QUESTION))
    assert ids.count(tok.mask_token_id) == len(markers)


def test_option_order_moves_the_options(tok):
    default = encode_segments(tok, STATE, QUESTION)["options"]
    swapped = encode_segments(tok, STATE, QUESTION, option_order=[1, 0])["options"]
    assert sorted(default) == sorted(swapped) and default != swapped


def test_lengths_and_fits_match_the_assembled_sequence(tok):
    segments = encode_segments(tok, STATE, QUESTION)
    total = lengths(segments)["total"]
    assert total == len(assemble(tok, segments)[0])
    assert fits(segments, total) and not fits(segments, total - 1)


def test_collate_pads_and_masks(tok):
    a = {"ids": [1, 5, 3, 6, 2], "markers": [2], "qtype": 2, "target": [1.0], "label": 0, "source": "x"}
    b = {"ids": [1, 3, 7, 3, 8, 2, 9], "markers": [1, 3], "qtype": 0, "target": [0.25, 0.75], "label": 1}
    out = collate([a, b], pad_id=0)
    assert out["input_ids"].shape == (2, 7) and out["marker_pos"].shape == (2, 2)
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert out["marker_mask"].tolist() == [[True, False], [True, True]]
    assert torch.allclose(out["target"][1], torch.tensor([0.25, 0.75]))
    assert out["meta"][0] == {"qtype": 2, "label": 0, "source": "x"}
    assert collate([], pad_id=0) is None


def test_collate_refuses_a_target_longer_than_the_options():
    bad = {"ids": [1, 3, 2], "markers": [1], "qtype": 0, "target": [0.5, 0.5]}
    with pytest.raises(ValueError):
        collate([bad], pad_id=0)
