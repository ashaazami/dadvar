# Adapted from Laya (https://github.com/NandhaKishorM/laya) at 4066d5d, Apache-2.0: laya/common.py
# (DecisionModel). Modified for Dadvar: act head and temperature buffer removed; save/load,
# encoder freezing and optimizer parameter groups added. See NOTICE.
"""A decision model to train from a pretrained encoder: one logit per option, nothing else.

The design follows Laya's decision model (https://github.com/NandhaKishorM/laya) without its act head:

    input_ids ──► encoder ──► + type embedding ──► head layers ──► gather at [MASK] ──► scorer ──► logits
                  (BERT)       (per question type)  (2 transformer  (one vector per     (one number
                                                     layers)         option)            per option)

What a caller does with the logits -- temperature, softmax, choice / expected score / P(true), and
whether to act on the answer -- stays outside the model.
"""

import json
import os
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

CONFIG = "dadvar_model.json"  # what `save` writes next to the weights
WEIGHTS = "model.safetensors"


class DadvarModel(nn.Module):
    """Encoder + question-type embedding + head layers + scorer. `forward` returns option logits."""

    def __init__(
        self, encoder: nn.Module, head_layers: int = 2, dropout: float = 0.1, encoder_name: Optional[str] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.head_layers, self.dropout, self.encoder_name = head_layers, dropout, encoder_name
        d = encoder.config.hidden_size
        # the new layers, with PyTorch's default initialisation
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)  # choice, score, noul
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.checkpointing = False
        self.encoder_frozen = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        """Logits `[B, K]`: one per option, -1e4 in the slots of a row with fewer than K options.

        input_ids, attention_mask  [B, L]  tokens, 1 = real token / 0 = padding
        marker_pos, marker_mask    [B, K]  each option's [MASK] position; True = a real option
        qtype                      [B]     0 choice, 1 score, 2 noul
        """
        if self.encoder_frozen:
            # nothing upstream of the head trains: skip building the encoder's autograd graph
            with torch.no_grad():
                h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()  # PyTorch's layers take True = padding
            for layer in self.head.layers:
                if self.checkpointing and self.training and torch.is_grad_enabled():
                    h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        options = torch.gather(h, 1, idx)  # [B, K, d]: the vector at each [MASK]
        logits = self.scorer(options).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)

    def enable_checkpointing(self, on: bool = True) -> None:
        """Trade compute for memory in training: recompute activations in the backward pass."""
        self.checkpointing = on
        if on:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.encoder.gradient_checkpointing_disable()

    def freeze_encoder(self, frozen: bool = True) -> None:
        """Train only the added layers (type embedding, head, scorer) while `frozen`."""
        self.encoder.requires_grad_(not frozen)
        self.encoder_frozen = frozen

    @classmethod
    def from_encoder(cls, encoder: str, head_layers: int = 2, dropout: float = 0.1, revision: Optional[str] = None):
        """A pretrained encoder from the Hub (e.g. answerdotai/ModernBERT-base) with new, untrained layers."""
        from transformers import AutoModel

        kw = {"attn_implementation": "sdpa"}
        if revision:
            kw["revision"] = revision
        return cls(AutoModel.from_pretrained(encoder, **kw), head_layers, dropout, encoder_name=encoder)

    def save(self, out_dir: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Weights, the encoder's config and this model's settings. `extra` is stored with the
        settings (training budgets, provenance, fitted temperatures); the tokenizer is the caller's."""
        from safetensors.torch import save_file

        os.makedirs(out_dir, exist_ok=True)
        save_file(
            {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}, os.path.join(out_dir, WEIGHTS)
        )
        self.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
        with open(os.path.join(out_dir, CONFIG), "w") as f:
            json.dump(
                {
                    "encoder": self.encoder_name,
                    "head_layers": self.head_layers,
                    "dropout": self.dropout,
                    **(extra or {}),
                },
                f,
                indent=1,
            )

    @classmethod
    def load(cls, model_dir: str):
        """A model `save` wrote. Returns (model, config)."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        with open(os.path.join(model_dir, CONFIG)) as f:
            cfg = json.load(f)
        ecfg = AutoConfig.from_pretrained(os.path.join(model_dir, "encoder"))
        encoder = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        model = cls(encoder, cfg["head_layers"], cfg["dropout"], encoder_name=cfg.get("encoder"))
        model.load_state_dict(load_file(os.path.join(model_dir, WEIGHTS)), strict=True)  # every weight
        return model, cfg


def parameter_groups(
    model: DadvarModel, lr_encoder: float, lr_head: float, weight_decay: float = 0.01
) -> Iterable[dict]:
    """AdamW groups: a small learning rate for the pretrained encoder, a larger one for the new layers.
    Frozen parameters are left out."""
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    new = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    return [
        {"params": enc, "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": new, "lr": lr_head, "weight_decay": weight_decay},
    ]
