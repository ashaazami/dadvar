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

    dadvar   a local checkpoint (hub id or directory), run with dadvar.Agent
    jev      TypeSafe's hosted model, through the official SDK (needs TYPESAFE_API_KEY)
"""

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

    def close(self) -> None:
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
class DadvarModel(Model):
    """A local checkpoint. `model_version` is the weights' Hugging Face revision when known."""

    backend = "dadvar"

    def __init__(self, model="convaiinnovations/laya", subfolder=None, device=None, **_):
        import dadvar

        model = model or "convaiinnovations/laya"
        self.agent = dadvar.load(model, device=device, subfolder=subfolder)
        self.model_id = model + ("/" + subfolder if subfolder else "")
        self.model_version = _hub_revision(model) or dadvar.__version__
        self.device = str(self.agent.device)
        self.settings = {"device": self.device, "dadvar": dadvar.__version__}

    def system_one(self, state, questions):
        return self.agent.system_one(state, questions)


@register
class JevModel(Model):
    """TypeSafe's hosted model. Requests go to api.typesafe.ai and are billed per input token."""

    backend = "jev"
    warmup = 1
    cost_per_input_token = 0.042 / 1e6      # https://docs.typesafe.ai/models
    default_model = "jev-1.13.0"            # pinned: `jev-latest` can move to another version

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


def _hub_revision(model):
    """The Hugging Face commit the weights come from, if this is a hub model rather than a path."""
    if os.path.isdir(model):
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model).sha
    except Exception:
        return None


def load_model(backend: str, model=None, **kwargs) -> Model:
    if backend not in MODELS:
        raise ValueError("unknown backend %r (choose from %s)" % (backend, ", ".join(sorted(MODELS))))
    return MODELS[backend](model=model, **kwargs)
