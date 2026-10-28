# Adapted from Laya (https://github.com/NandhaKishorM/laya) at 4066d5d, Apache-2.0: laya/common.py
# (build_sequence). Modified for Dadvar: segments are encoded in full and never truncated; a
# question that does not fit is skipped by the caller. See NOTICE.
"""The model input for one question, encoded segment by segment and never cut.

`encode_segments` tokenizes every segment in full and returns them by name; `lengths` and `fits`
say whether the whole sequence fits a budget as-is; `assemble` joins the segments:

    [CLS] instructions [SEP] options [SEP] state [SEP]

where `instructions` is "<type> question: <instructions>" and `options` is every option in turn,
each a [MASK] followed by its rendered text. A question that does not fit the budget is skipped by
the caller, never cut, so nothing the model is shown is silently lost.
"""

from typing import Dict, List, Optional, Union

from .text import encode, option_texts, state_json

SPECIAL_TOKENS = 4  # [CLS], and the [SEP] closing each of the three segments


def encode_segments(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    option_order: Optional[List[int]] = None,
    state_ids: Optional[List[int]] = None,
) -> Dict[str, list]:
    """Tokenize every segment of the sequence in full, with no budget applied.

    Returns {"instructions": [ids], "options": [ids], "state": [ids]}. `options` is one run,
    `[MASK] opt0 [MASK] opt1 ...`, in `option_order` (default: label order); each [MASK] is an
    option's marker, and no other [MASK] can occur because a literal one in the text is blanked.
    `q` is the internal form ({"t", "ins", "crit"}); `state_ids`
    reuses a state already tokenized for another question.
    """
    mask_tok = tok.mask_token
    opts = option_texts(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    instructions = encode(tok, "%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    options = []
    for i in order:
        options.append(tok.mask_token_id)
        options.extend(encode(tok, " " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"])
    if state_ids is None:
        state_ids = encode(tok, state_json(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    return {"instructions": instructions, "options": options, "state": list(state_ids)}


def lengths(segments: Dict[str, list]) -> Dict[str, int]:
    """Token count per segment, the special tokens, and the `total` the assembled sequence has.

    `options` includes each option's [MASK]; `head` is everything before the state.
    """
    n = {
        "instructions": len(segments["instructions"]),
        "options": len(segments["options"]),
        "state": len(segments["state"]),
        "special": SPECIAL_TOKENS,
    }
    n["head"] = n["instructions"] + n["options"] + 3  # [CLS] ... [SEP] ... [SEP]
    n["total"] = n["head"] + n["state"] + 1
    return n


def fits(segments: Dict[str, list], max_len: int) -> bool:
    """Does the whole sequence, nothing cut, fit in `max_len` tokens?"""
    return lengths(segments)["total"] <= max_len


def assemble(tok, segments: Dict[str, list]):
    """Join the segments into (ids, markers) in the layout above, cutting nothing.

    `markers` holds the position of each option's [MASK], in option order. The result can be
    longer than any budget; check `fits` first.
    """
    ids = [tok.cls_token_id] + segments["instructions"] + [tok.sep_token_id]
    markers = [len(ids) + j for j, t in enumerate(segments["options"]) if t == tok.mask_token_id]
    ids.extend(segments["options"])
    ids.append(tok.sep_token_id)
    ids.extend(segments["state"])
    ids.append(tok.sep_token_id)
    return ids, markers
