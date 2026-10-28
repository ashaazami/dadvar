import torch

from dadvar.batch import collate
from dadvar.model import DadvarModel, parameter_groups


def _batch():
    items = [
        {"ids": [1, 5, 3, 6, 3, 7, 2], "markers": [2, 4], "qtype": 0},
        {"ids": [1, 8, 3, 9, 3, 10, 3, 11, 2], "markers": [2, 4, 6], "qtype": 1},
    ]
    b = collate(items, pad_id=0)
    return [b[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]


def test_one_logit_per_option_and_padding_masked(tiny_encoder):
    model = DadvarModel(tiny_encoder).eval()
    with torch.no_grad():
        logits = model(*_batch())
    assert logits.shape == (2, 3)
    assert logits[0, 2].item() == -1e4  # the first row has two options
    assert torch.isfinite(logits[1]).all()


def test_save_and_load_round_trip(tiny_encoder, tmp_path):
    model = DadvarModel(tiny_encoder, head_layers=1).eval()
    model.save(str(tmp_path), extra={"max_len": 64, "temperature": [1.0, 1.0, 1.0]})
    loaded, cfg = DadvarModel.load(str(tmp_path))
    loaded.eval()
    assert cfg["head_layers"] == 1 and cfg["max_len"] == 64
    with torch.no_grad():
        assert torch.allclose(model(*_batch()), loaded(*_batch()))


def test_frozen_encoder_is_left_out_of_the_optimizer(tiny_encoder):
    model = DadvarModel(tiny_encoder)
    model.freeze_encoder(True)
    groups = parameter_groups(model, lr_encoder=1e-5, lr_head=1e-4)
    assert groups[0]["params"] == [] and groups[1]["params"]
    assert all(p.requires_grad for p in groups[1]["params"])
