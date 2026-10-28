"""The interface a benchmark run evaluates, and the models that implement it.

A model answers TypeSafe requests:

    model.system_one(state, questions) -> {"model": ..., "answers": {...}, "usage": {...}}

That is all the runner and the metrics use, so they never learn which model answered. To add
one, subclass `Model`, implement `system_one`, and register it:

    @register
    class MlxModel(Model):
        backend = "mlx"
        def __init__(self, model=None, **kw): ...
        def system_one(self, state, questions): ...

Implemented here:

    dadvar-model  a model trained with training/finetune.py, from its model/ folder
    jev           TypeSafe's hosted model, through the official SDK (needs TYPESAFE_API_KEY)
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional


class Model(ABC):
    """One model under test."""

    backend: ClassVar[str]
    #: warm-up calls before timing; a hosted model pays per call, so it keeps this small
    warmup: ClassVar[int] = 3
    #: USD per input token, or None when running the model is free
    cost_per_input_token: ClassVar[Optional[float]] = None

    #: what answered, recorded with every run
    model_id: str
    model_version: str
    device: str = "unknown"
    settings: Dict[str, Any] = {}

    @abstractmethod
    def system_one(self, state, questions) -> Dict[str, Any]:
        """Answer every question about `state`, in the TypeSafe response shape."""

    def skip_reason(self, state, questions) -> Optional[str]:
        """Why this model cannot be asked this row as it stands, or None when it can.

        A skipped row is not sent to the model and not scored; the runner records it with this
        reason, and a report comparing runs drops it from all of them. The default is never to
        skip.
        """
        return None

    def close(self) -> None:  # noqa: B027 -- an optional hook, not abstract
        """Release anything the model holds (connections, memory). Optional."""

    def usage_cost(self, input_tokens: int) -> Optional[float]:
        if self.cost_per_input_token is None:
            return None
        return input_tokens * self.cost_per_input_token


MODELS: Dict[str, type] = {}


def register(cls):
    """Add a Model subclass to the registry under its `backend` name."""
    MODELS[cls.backend] = cls
    return cls


@register
class DadvarCheckpoint(Model):
    """A model trained with training/finetune.py (`dadvar.model.DadvarModel`), from its `model/` folder.

    It is asked exactly as it was trained: the whole sequence for each question (`dadvar.sequence`),
    and a row with any question longer than `max_len` skipped, never cut.
    Probabilities are softmax(logits / T), with the temperature fitted per question type after
    training. `model_version` is a hash of the weights, so a retrained model never reuses answers.
    """

    backend = "dadvar-model"

    def __init__(self, model=None, device=None, state_format="json", **_):
        import hashlib
        import torch
        from transformers import AutoTokenizer
        from dadvar.model import DadvarModel, WEIGHTS

        if not model or not os.path.isdir(model):
            raise ValueError("--model must be the folder a training run saved, e.g. training/runs/<run>/model")
        self.torch = torch
        self.net, self.cfg = DadvarModel.load(model)
        self.dev = (
            torch.device(device)
            if device
            else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        )
        self.net.to(self.dev).eval()
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model, "tokenizer"))
        self.max_len = self.cfg["max_len"]
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        h = hashlib.sha256()
        with open(os.path.join(model, WEIGHTS), "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        run = os.path.basename(os.path.dirname(os.path.abspath(model)))
        self.model_id = "dadvar/" + (
            run if os.path.basename(os.path.abspath(model)) == "model" else os.path.basename(os.path.abspath(model))
        )
        self.model_version = h.hexdigest()[:16]
        self.device = str(self.dev)
        self.state_format = state_format
        self.settings = {
            "device": self.device,
            "max_len": self.max_len,
            "temperature": [round(t, 4) for t in self.temperature],
        }
        if state_format != "json":  # a different input: never share cached answers
            self.settings["state_format"] = state_format
        self._question_tokens = {}
        # the sources whose training splits it was trained on, from its data provenance
        self.trained_on = set((self.cfg.get("items_provenance") or {}).get("sources", {}))

    @staticmethod
    def _internal(qdef):
        # exactly as training/prepare_data.py built its items (object instructions read as JSON)
        ins = qdef["instructions"]
        return {
            "t": qdef["type"],
            "ins": ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False),
            "crit": qdef.get("criteria"),
        }

    def _state_ids(self, state):
        from dadvar.text import encode, state_json, state_paths

        text = state_paths(state) if self.state_format == "paths" else state_json(state)
        return encode(self.tok, text.replace(self.tok.mask_token, " "), add_special_tokens=False)["input_ids"]

    def skip_reason(self, state, questions):
        """A row with any question longer than `max_len` whole: the model was never trained on a cut input."""
        from dadvar.sequence import encode_segments, lengths

        n_state = len(self._state_ids(state))
        for qid, qdef in questions.items():
            key = json.dumps(qdef, sort_keys=True)
            if key not in self._question_tokens:
                self._question_tokens[key] = lengths(encode_segments(self.tok, "", self._internal(qdef)))["total"]
            need = self._question_tokens[key] + n_state
            if need > self.max_len:
                return "question %r needs %d tokens; max_len is %d" % (qid, need, self.max_len)
        return None

    def system_one(self, state, questions):
        import numpy as np
        from dadvar.batch import collate
        from dadvar.text import QTYPES
        from dadvar.sequence import assemble, encode_segments

        torch, ids = self.torch, list(questions)
        state_ids = self._state_ids(state)
        items = []
        for qid in ids:
            seq, markers = assemble(
                self.tok, encode_segments(self.tok, state, self._internal(questions[qid]), state_ids=state_ids)
            )
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[questions[qid]["type"]]})
        b = collate(items, self.tok.pad_token_id)
        with torch.no_grad():
            logits = self.net(
                *[b[k].to(self.dev) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
            )
        logits = logits.float().cpu().numpy()
        answers = {}
        for r, qid in enumerate(ids):
            qdef, k = questions[qid], len(items[r]["markers"])
            z = logits[r, :k] / self.temperature[items[r]["qtype"]]
            p = np.exp(z - z.max())
            p = p / p.sum()
            if qdef["type"] == "choice":
                crit = qdef["criteria"]
                labels = list(crit) if isinstance(crit, (dict, list)) else []
                answers[qid] = {
                    "type": "choice",
                    "choice": labels[int(p.argmax())],
                    "confidence": float(p.max()),
                    "probabilities": {lab: float(v) for lab, v in zip(labels, p)},
                }
            elif qdef["type"] == "score":
                answers[qid] = {
                    "type": "score",
                    "score": float((np.arange(k) * p).sum()),
                    "confidence": float(p.max()),
                    "legend": {str(i): c for i, c in enumerate(qdef["criteria"])},
                    "probabilities": {str(i): float(v) for i, v in enumerate(p)},
                }
            else:
                answers[qid] = {"type": "noul", "noul": float(p[1])}
        return {
            "model": self.model_version,
            "answers": answers,
            "usage": {"input_tokens": int(sum(len(it["ids"]) for it in items)), "output_tokens": 0},
        }


@register
class JevModel(Model):
    """TypeSafe's hosted model. Requests go to api.typesafe.ai and are billed per input token."""

    backend = "jev"
    warmup = 1
    cost_per_input_token = 0.042 / 1e6  # https://docs.typesafe.ai/models
    default_model = "jev-1.14.0"  # pinned: `jev-latest` can move to another version

    def __init__(self, model=None, **_):
        from typesafe_sdk import TypeSafeClient

        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is not set; needed for the jev backend")
        self.client = TypeSafeClient()
        self.model_version = model or self.default_model
        self.model_id = "typesafe/" + self.model_version
        self.device = "api"
        self.settings = {"endpoint": "api.typesafe.ai"}

    def system_one(self, state, questions):
        response = self.client.system_one(state=state, questions=questions, model=self.model_version)
        return response.model_dump(mode="json")

    def close(self):
        self.client.close()


def load_model(backend: str, model=None, **kwargs) -> Model:
    if backend not in MODELS:
        raise ValueError("unknown backend %r (choose from %s)" % (backend, ", ".join(sorted(MODELS))))
    return MODELS[backend](model=model, **kwargs)
