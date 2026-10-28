"""Offline fixtures: a word-level tokenizer and a tiny encoder, so no test downloads anything."""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest  # noqa: E402


class WordTokenizer:
    """Just enough of a Hugging Face tokenizer: one id per whitespace-separated word."""

    pad_token_id, cls_token_id, sep_token_id, mask_token_id = 0, 1, 2, 3
    mask_token = "[MASK]"

    def __init__(self):
        self.vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}

    def __call__(self, text, add_special_tokens=True, **_):
        ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return {"input_ids": ids}


@pytest.fixture
def tok():
    return WordTokenizer()


@pytest.fixture
def tiny_encoder():
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return BertModel(config)
