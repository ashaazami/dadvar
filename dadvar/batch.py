# Adapted from Laya (https://github.com/NandhaKishorM/laya) at 4066d5d, Apache-2.0: laya/common.py
# (collate_items). Modified for Dadvar: takes a flat list of items; restructured. See NOTICE.
"""Padding a list of encoded questions into one batch of tensors.

An item is {"ids", "markers", "qtype"} plus, in training, "target" (one probability per option) and
"label"; any other key is passed through in `meta`.
"""

from typing import Dict, List, Optional

import torch

_TENSOR_KEYS = ("ids", "markers", "target")


def collate(items: List[Dict], pad_id: int) -> Optional[Dict]:
    """Pad `items` to the longest sequence and the most options. None for an empty list.

    Returns input_ids / attention_mask [B, L], marker_pos / marker_mask [B, K], qtype and label [B]
    (label -1 where an item has none), meta, and target [B, K] when any item has one.
    """
    if not items:
        return None
    width = max(len(it["ids"]) for it in items)
    slots = max(len(it["markers"]) for it in items)
    out = {
        "input_ids": torch.full((len(items), width), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros((len(items), width), dtype=torch.long),
        "marker_pos": torch.zeros((len(items), slots), dtype=torch.long),
        "marker_mask": torch.zeros((len(items), slots), dtype=torch.bool),
    }
    targets = any("target" in it for it in items)
    if targets:
        out["target"] = torch.zeros((len(items), slots), dtype=torch.float32)

    for row, it in enumerate(items):
        n, k = len(it["ids"]), len(it["markers"])
        out["input_ids"][row, :n] = torch.tensor(it["ids"])
        out["attention_mask"][row, :n] = 1
        out["marker_pos"][row, :k] = torch.tensor(it["markers"])
        out["marker_mask"][row, :k] = True
        if "target" in it:
            if len(it["target"]) > k:
                raise ValueError("item %d: %d target entries for %d options" % (row, len(it["target"]), k))
            out["target"][row, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)

    out["qtype"] = torch.tensor([it["qtype"] for it in items])
    out["label"] = torch.tensor([it.get("label", -1) for it in items])
    out["meta"] = [{k: v for k, v in it.items() if k not in _TENSOR_KEYS} for it in items]
    return out
