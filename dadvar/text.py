# Adapted from Laya (https://github.com/NandhaKishorM/laya) at 4066d5d, Apache-2.0: laya/common.py
# (encode_text, serialize_state, render_criterion, _resolve_noul_labels, render_options).
# Modified for Dadvar: restructured and renamed; state_paths added. See NOTICE.
"""What the model reads: the state and a question's options as text.

`option_texts` gives one string per option, in label order; `state_json` and `state_paths` give the
state; `encode` tokenizes under a lock, because a fast tokenizer changes its own settings when asked
to pad or truncate and so cannot be shared by threads encoding at the same time.
"""

import json
import threading
from typing import Dict, List, Union

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {i: name for name, i in QTYPES.items()}

NOUL_FALSE = "no, the statement does not hold"
NOUL_TRUE = "yes, the statement holds"

_encode_lock = threading.RLock()


def encode(tok, text, **kwargs):
    """`tok(text, **kwargs)`, one call at a time."""
    with _encode_lock:
        return tok(text, **kwargs)


def _json(value, **kwargs) -> str:
    return json.dumps(value, ensure_ascii=False, **kwargs)


def state_json(state: Union[str, dict, list]) -> str:
    """The state as compact JSON; a string state as it is."""
    return state if isinstance(state, str) else _json(state)


def state_paths(state: Union[str, dict, list]) -> str:
    """The state one value per line, each after its full path (`ticket.messages[1].text: Thanks!`),
    so a question that names a path finds it written out. A string state is returned as it is."""
    if isinstance(state, str):
        return state
    lines: List[str] = []

    def walk(value, path):
        if isinstance(value, (dict, list)):
            if not value:
                if path:
                    lines.append("%s: %s" % (path, "{}" if isinstance(value, dict) else "[]"))
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, "%s.%s" % (path, key) if path else str(key))
            else:
                for i, child in enumerate(value):
                    walk(child, "%s[%d]" % (path, i))
            return
        leaf = value if isinstance(value, str) else _json(value)
        lines.append("%s: %s" % (path, leaf) if path else leaf)

    walk(state, "")
    return "\n".join(lines)


def criterion_text(value) -> str:
    """A criterion as text: a string as it is, anything else as JSON."""
    return value if isinstance(value, str) else _json(value, separators=(", ", ": "), default=str)


def _described(value) -> bool:
    return value is not None and value != ""


def noul_labels(labels=None):
    """(false, true) label names for a noul question: its `labels` if given, else false/true."""
    if labels is None:
        return "false", "true"
    names = (labels.get("false"), labels.get("true")) if isinstance(labels, dict) else (None, None)
    if (
        not isinstance(labels, dict)
        or set(labels) != {"false", "true"}
        or not all(isinstance(n, str) and n.strip() for n in names)
        or names[0].strip() == names[1].strip()
    ):
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return names[0].strip(), names[1].strip()


def option_texts(q: Dict) -> List[str]:
    """One text per option, in label order. `q` is {"t", "ins", "crit"} (and `labels` for noul);
    a noul question's options are always [false, true]."""
    kind, crit = q["t"], q.get("crit")
    if "labels" in q and kind != "noul":
        raise ValueError("labels is only supported for noul questions")
    if kind == "choice":
        return ["%s: %s" % (label, criterion_text(d)) if _described(d) else str(label) for label, d in crit.items()]
    if kind == "score":
        return ["level %d: %s" % (i, criterion_text(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_name, true_name = noul_labels(q.get("labels"))
    return [
        "%s: %s" % (name, criterion_text(crit[side]) if _described(crit.get(side)) else default)
        for name, side, default in ((false_name, "false", NOUL_FALSE), (true_name, "true", NOUL_TRUE))
    ]
