"""Build a set of rows from a source: pinned revision, seeded sample, converted, validated.

The result is a Hugging Face `Dataset` in the typed-decisions schema (see `sources.py`), cached
under `data/cache/` (git-ignored, so no dataset content is committed).

    ds, info = build("banking77")                       # evaluation rows (the default)
    ds, info = build("banking77", Split.TRAIN, n=2000)  # training rows, never the eval split

Guarantees:
  * a source may only be used as its `use` policy allows (`train=False` raises);
  * train and eval rows never overlap. When a source names different Hugging Face splits for
    the two, that holds by construction. When both name the same split (a dataset that ships
    only one), eval rows are drawn first and training rows come from the remaining indices.
"""

import json
import os
import random
from enum import Enum

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi
from pydantic import TypeAdapter
from typesafe_sdk import Answer, Choice, Noul, Score

from .sources import PROMPT_VERSION, SOURCES

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEFAULT_N = 500
DEFAULT_SEED = 13

_QUESTION = {"choice": Choice, "noul": Noul, "score": Score}
_ANSWER = TypeAdapter(Answer)


class Split(str, Enum):
    """What the rows are for. The source decides which Hugging Face split each one maps to."""

    EVAL = "eval"
    TRAIN = "train"


def dataset_revision(repo: str) -> str:
    return HfApi().dataset_info(repo).sha


def label_set(spec, repo, config, revision):
    if "labels" not in spec:
        return None
    split, column, *label_config = spec["labels"]
    d = load_dataset(repo, label_config[0] if label_config else config, split=split, revision=revision)
    return sorted(set(d[column]))


def validate(r):
    """Every question must parse as a TypeSafe question and every gold as a TypeSafe answer."""
    questions, gold = json.loads(r["questions"]), json.loads(r["gold"])
    if set(questions) != set(gold):
        raise ValueError("%s: questions %s != gold %s" % (r["id"], sorted(questions), sorted(gold)))
    for qid, q in questions.items():
        _QUESTION[q["type"]].model_validate(q)
        a = _ANSWER.validate_json(json.dumps(gold[qid]))
        if a.type != q["type"]:
            raise ValueError("%s/%s: question is %s, gold is %s" % (r["id"], qid, q["type"], a.type))
        if a.type == "choice" and a.choice not in q["criteria"]:
            raise ValueError("%s/%s: gold %r is not an option" % (r["id"], qid, a.choice))
        if a.type in ("choice", "score") and abs(sum(a.probabilities.values()) - 1) > 1e-3:
            raise ValueError("%s/%s: gold probabilities do not sum to 1" % (r["id"], qid))


def resolve_split(spec, split: Split, name: str):
    """The Hugging Face split for `split`, and whether it is shared with the other one."""
    hf_split = spec["splits"].get(split.value)
    if hf_split is None:
        raise ValueError("source %r may not be used for %s (splits=%s)" % (name, split.value, spec["splits"]))
    if not spec["use"].get(split.value if split is Split.TRAIN else "benchmark", False):
        raise ValueError("source %r is not allowed for %s (license %s, use=%s)"
                         % (name, split.value, spec["license"], spec["use"]))
    return hf_split, hf_split == spec["splits"].get(Split.EVAL.value if split is Split.TRAIN else Split.TRAIN.value)


def sample_indices(total: int, n: int, seed: int, split: Split, shared: bool, eval_n: int):
    """Indices for `split`. When the two splits share one Hugging Face split, keep them disjoint:
    eval takes its seeded sample first, training draws from what is left."""
    if split is Split.EVAL or not shared:
        return sorted(random.Random(seed).sample(range(total), min(n, total)))
    held_out = set(random.Random(seed).sample(range(total), min(eval_n, total)))   # what eval takes
    rest = [i for i in range(total) if i not in held_out]
    return sorted(random.Random(seed + 1).sample(rest, min(n, len(rest))))


def build(name: str, split: Split = Split.EVAL, n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
          configs=None, eval_n: int = DEFAULT_N, refresh: bool = False):
    """Return (Dataset, info) for source `name`: `n` seeded random rows per config.

    `eval_n` is the evaluation sample size to hold out when a source's train and eval rows come
    from the same Hugging Face split; it must match the `n` used for evaluation runs.
    """
    spec = SOURCES[name]
    split = Split(split)
    hf_split, shared = resolve_split(spec, split, name)
    configs = tuple(configs or spec["configs"])
    revision = dataset_revision(spec["dataset"])
    key = "%s-%s-%s-n%d-s%d-p%d-%s" % (name, split.value, hf_split, n, seed, PROMPT_VERSION, "_".join(configs))
    path = os.path.join(CACHE_DIR, key)
    info_path = os.path.join(path, "benchmark_info.json")
    if not refresh and os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if info["revision"] == revision and info.get("eval_n") == eval_n:
            return load_from_disk(path), info

    parts, sampled = [], {}
    labels = None
    for config in configs:
        raw = load_dataset(spec["dataset"], None if config == "default" else config,
                           split=hf_split, revision=revision)
        if labels is None:
            labels = label_set(spec, spec["dataset"], None if config == "default" else config, revision)
        indices = sample_indices(len(raw), n, seed, split, shared, eval_n)
        sampled[config] = indices
        converted = raw.select(indices).map(
            lambda r, i: spec["fn"](r, indices[i], hf_split, config, labels),
            with_indices=True, remove_columns=raw.column_names)
        parts.append(converted)

    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    for r in ds:
        validate(r)

    info = {"source": name, "dataset": spec["dataset"], "revision": revision, "split": hf_split,
            "role": split.value, "shared_split": shared, "configs": list(configs), "n": n, "seed": seed,
            "eval_n": eval_n, "rows": len(ds), "sampled_indices": sampled,
            "prompt_version": PROMPT_VERSION, "license": spec["license"], "use": spec["use"]}
    ds.save_to_disk(path)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=1)
    return Dataset.load_from_disk(path), info
