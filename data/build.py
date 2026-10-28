"""Build a set of rows from a source: pinned revision, seeded sample, converted, validated.

The result is a Hugging Face `Dataset` in the typed-decisions schema (see `sources.py`), cached
under `data/cache/` (git-ignored, so no dataset content is committed).

    ds, info = build("banking77")                       # evaluation rows (the default)
    ds, info = build("banking77", Split.TRAIN, n=2000)  # training rows, never the eval split

Guarantees:
  * a source may only be used as its `use` policy allows (`use: {train: false}` refuses training);
  * train and eval rows never overlap. When a source names different Hugging Face splits for
    the two, that holds by construction. When both name the same split (a dataset that ships
    only one), eval rows are drawn first and training rows come from the remaining indices.
"""

import json
import os
import random
import re
from enum import Enum

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi
from pydantic import TypeAdapter
from typesafe_sdk import Answer, Choice, Noul, Score

from .sources import PROMPT_VERSION, SOURCES, row as make_row
from .templates import opposite, render

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DEFAULT_N = 500
DEFAULT_SEED = 13
# The shape of a row's `meta`. Bumped when it changes, so cached rows written by an older build are
# rebuilt rather than read back in the old shape. Question text is versioned separately, by
# PROMPT_VERSION, which is what benchmark results are compared on.
META_VERSION = 1

# Shares of eligible training rows (variants 1, 2, ...) that get each augmentation, and, among
# reference rows, how the text is placed. Everything else about an augmentation is per source, and
# declared in the source's template file (`state:` roles, a question's `asked` / `on_mismatch`).
SHARES = {"reference": 0.6, "mismatch": 0.3, "structured": 0.15}
LAYOUTS = (("sibling", 0.30), ("nested", 0.30), ("array", 0.25), ("metadata", 0.15))

_QUESTION = {"choice": Choice, "noul": Noul, "score": Score}
_ANSWER = TypeAdapter(Answer)


class Split(str, Enum):
    """What the rows are for. The source decides which Hugging Face split each one maps to."""

    EVAL = "eval"
    TRAIN = "train"


def dataset_revision(repo: str, parquet: bool = False) -> str:
    """The commit to load. A dataset published only as a loading script (which `datasets` no
    longer runs) is read from Hugging Face's automatic Parquet conversion, pinned to its commit."""
    if parquet:
        refs = HfApi().list_repo_refs(repo, repo_type="dataset")
        return next(c.target_commit for c in refs.converts if c.ref == "refs/convert/parquet")
    return HfApi().dataset_info(repo).sha


def load_split(spec, config, split, revision):
    """One split of one config. A Parquet-converted dataset is read file by file from the pinned
    conversion commit, since `datasets` does not see its folders as configs."""
    if spec.get("parquet"):
        path = "hf://datasets/%s@%s/%s/%s/*.parquet" % (spec["dataset"], revision, config, split)
        return load_dataset("parquet", data_files={split: path}, split=split)
    return load_dataset(spec["dataset"], None if config == "default" else config, split=split, revision=revision)


def label_set(spec, config, revision):
    """The full label set a source's questions offer, read before sampling.

    ("features", column): the names of a ClassLabel column, in the dataset's own order.
    (split, column[, config]): the distinct values of a column, sorted.
    """
    if "labels" not in spec:
        return None
    where, column, *label_config = spec["labels"]
    cfg = label_config[0] if label_config else config
    if where == "features":
        d = load_split(spec, cfg, spec["splits"]["eval"], revision)
        feature = d.features[column]
        feature = getattr(feature, "feature", feature)  # a sequence of labels holds its own
        return list(feature.names)
    d = load_split(spec, cfg, where, revision)
    return sorted(set(d[column]))


# How many options a training variant's choice question keeps: (share, sizes). Most questions
# get 5; the rest keep a mid-size or the full label set, so large option lists stay in training.
OPTION_MIX = ((0.70, (5, 5)), (0.15, (10, 30)), (0.15, None))
_WORD = re.compile(r"[a-z0-9]+")


def _words(label, desc):
    return set(_WORD.findall(("%s %s" % (label, desc or "")).lower()))


def limit_options(questions, gold, rng, k=None, shuffle=False):
    """Cut every choice question with more than `k` options down to `k`: the gold label plus
    distractors, half the most similar to it (shared words) and half at random. k=None draws the
    size from OPTION_MIX. `shuffle` reorders the options (the gold answer is keyed by label, so it
    is unaffected). Changes `questions` and `gold` in place."""
    for qid, q in questions.items():
        if q["type"] != "choice":
            continue
        crit = q["criteria"]
        labels = list(crit) if isinstance(crit, dict) else list(crit)
        desc = crit if isinstance(crit, dict) else {l: None for l in labels}
        size = k
        if size is None:
            r, acc = rng.random(), 0.0
            for share, sizes in OPTION_MIX:
                acc += share
                if r < acc:
                    size = len(labels) if sizes is None else rng.randint(*sizes)
                    break
        keep = labels
        if size and len(labels) > size:
            g = gold[qid]
            gold_label = g["choice"] if "choice" in g else max(g["probabilities"], key=g["probabilities"].get)
            others = [l for l in labels if l != gold_label]
            gw = _words(gold_label, desc[gold_label])
            ranked = sorted(others, key=lambda l: (-len(gw & _words(l, desc[l])), rng.random()))
            n_similar = (size - 1) // 2
            similar = ranked[:n_similar]
            rest = [l for l in others if l not in similar]
            keep = [gold_label] + similar + rng.sample(rest, size - 1 - n_similar)
            probs = {l: float(g.get("probabilities", {}).get(l, 0.0)) for l in keep}
            total = sum(probs.values()) or 1.0
            gold[qid] = dict(g, probabilities={l: v / total for l, v in probs.items()})
        order = [l for l in labels if l in set(keep)]
        if shuffle:
            rng.shuffle(order)
        q["criteria"] = {l: desc[l] for l in order} if isinstance(crit, dict) else order


FIELD_NAMES = ("text", "comment", "message", "post", "reply", "note", "review", "statement", "remark", "entry")


# --------------------------------------------------------------------- augmentations
#
# An augmented row's id is "<row>~<kind>~<other row>~...": every source row whose text it holds,
# so the training split can keep all of them on one side (`row_parts`).


def question_form(instructions):
    """ "question" or "statement": how a wording that came with the data asks (templates declare `form`)."""
    text = instructions if isinstance(instructions, str) else instructions.get("instructions", "")
    return "question" if text.rstrip().endswith("?") else "statement"


def row_parts(row_id):
    """The source rows an augmented row was built from (the "~" ids the augmentations write)."""
    if "~" in row_id:
        xs = row_id.split("~")
        return [xs[0]] + xs[2:]
    return [row_id]


def _answer_class(a):
    if a["type"] == "noul":
        return a["noul"] >= 0.5
    if a["type"] == "choice":
        return a["choice"]
    return max(a["probabilities"], key=a["probabilities"].get)


def _key(r):
    """(question id, answer class) of the row's first original question: what partners are matched on."""
    g, m = json.loads(r["gold"]), json.loads(r["meta"])
    qids = sorted(q for q in g if m.get(q, {}).get("kind", "original") == "original")
    return (qids[0], _answer_class(g[qids[0]])) if qids else (None, None)


_PEOPLE = ("Alex", "Sam", "Priya", "Jordan", "Mei", "Omar", "Lena", "Diego", "Chris", "Aisha", "Tom", "Yuki")
_SUBJECTS = ("Follow-up", "Question about my account", "Re: your request", "Update", "Feedback", "Issue report")
_HEADINGS = ("Background", "Details", "Summary", "Notes", "Discussion", "Comments")
_RECORD_KEYS = ("description", "notes", "summary", "details", "comment", "body")
_ARRAY_KEYS = ("items", "messages", "entries", "texts", "comments", "posts", "records")


def _layout(kind, texts, rng):
    """(state, path of each text) for texts placed by `kind`."""
    n = len(texts)
    if kind == "sibling":
        names = rng.sample(FIELD_NAMES, n)
        order = list(range(n))
        rng.shuffle(order)
        return {names[i]: texts[i] for i in order}, names
    if kind == "array":
        key = rng.choice(_ARRAY_KEYS)
        return {key: list(texts)}, ["%s[%d]" % (key, i) for i in range(n)]
    if kind == "metadata":
        name = rng.choice(FIELD_NAMES)
        meta = {
            "id": "%s-%d" % (rng.choice("ABCDEFGHJK"), rng.randint(1000, 99999)),
            "created_at": "2025-%02d-%02dT%02d:%02d:00Z"
            % (rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59)),
            "author": rng.choice(_PEOPLE),
            "channel": rng.choice(("web", "email", "app", "api", "chat")),
            "status": rng.choice(("open", "new", "pending", "closed")),
        }
        if rng.random() < 0.5:
            state = dict(meta)
            state[name] = texts[0]
            keys = list(state)
            rng.shuffle(keys)
            return {k: state[k] for k in keys}, [name]
        return {"meta": meta, "content": {name: texts[0]}}, ["content.%s" % name]
    # nested: a container from everyday software
    c = rng.choice(("ticket", "email", "chat", "record", "document"))
    if c == "ticket":
        msgs = [{"from": rng.choice(("customer", "agent")), "text": t} for t in texts]
        return (
            {"ticket": {"id": "T-%d" % rng.randint(1000, 99999), "subject": rng.choice(_SUBJECTS), "messages": msgs}},
            ["ticket.messages[%d].text" % i for i in range(n)],
        )
    if c == "email":
        thread = [
            {"from": "%s@example.com" % rng.choice(_PEOPLE).lower(), "subject": rng.choice(_SUBJECTS), "body": t}
            for t in texts
        ]
        return {"thread": thread}, ["thread[%d].body" % i for i in range(n)]
    if c == "chat":
        return {"chat": [{"user": rng.choice(_PEOPLE), "content": t} for t in texts]}, [
            "chat[%d].content" % i for i in range(n)
        ]
    if c == "record":
        keys = rng.sample(_RECORD_KEYS, n)
        return (
            {
                "record": {
                    "id": "R-%d" % rng.randint(1000, 99999),
                    "status": rng.choice(("open", "closed")),
                    "fields": dict(zip(keys, texts)),
                }
            },
            ["record.fields.%s" % k for k in keys],
        )
    heads = rng.sample(_HEADINGS, n)
    return (
        {
            "document": {
                "title": rng.choice(_SUBJECTS),
                "sections": [{"heading": h, "text": t} for h, t in zip(heads, texts)],
            }
        },
        ["document.sections[%d].text" % i for i in range(n)],
    )


def _point(q, field, path):
    """The question with every reference to `field` pointed at `path` instead."""
    return json.loads(json.dumps(q, ensure_ascii=False).replace("`%s`" % field, "`%s`" % path))


def add_reference_rows(rows, field, rng, share, layouts, max_chars=2500):
    """For about `share` of the rows whose state is one text under `field`, a copy with that text
    placed in a larger state (next to other rows' texts, inside a ticket, email, chat, record or
    document, in an array, or among metadata), its questions pointed at the text's path. Partner
    texts come from rows with a different answer to the same question (for yes/no, half the time
    the same), and that question is also asked about each partner, with the partner's gold."""
    usable = [r for r in rows if list(json.loads(r["state"])) == [field]]
    index = {}
    for r in usable:
        q, cls = _key(r)
        if q is not None:
            index.setdefault(q, {}).setdefault(cls, []).append(r)
    kinds, weights = zip(*layouts)
    new = []
    for a in usable:
        if rng.random() >= share:
            continue
        q, cls = _key(a)
        if q is None:
            continue
        kind = rng.choices(kinds, weights)[0]
        n_partners = {"sibling": 1, "nested": rng.randint(1, 2), "array": rng.randint(2, 3), "metadata": 0}[kind]
        partners, classes = [], index[q]
        for _ in range(n_partners):
            same = isinstance(cls, bool) and rng.random() < 0.5
            pool_cls = [cls] if same else [c for c in classes if c != cls]
            if not pool_cls:
                break
            pool, taken = classes[rng.choice(pool_cls)], {p["id"] for p in partners} | {a["id"]}
            for _ in range(5):  # a few draws, not a scan
                cand = rng.choice(pool)
                if cand["id"] not in taken:
                    partners.append(cand)
                    break
        if kind != "metadata" and not partners:
            continue
        group = [a] + partners
        texts = [json.loads(r["state"])[field] for r in group]
        if sum(len(t) for t in texts) > max_chars:
            if len(texts[0]) > max_chars:
                continue
            kind, group, texts = "metadata", [a], texts[:1]
        slots = list(range(len(group)))
        rng.shuffle(slots)  # where each text goes
        placed = [None] * len(group)
        for g, s in enumerate(slots):
            placed[s] = texts[g]
        state, paths = _layout(kind, placed, rng)
        questions, gold, meta = {}, {}, {}
        for g, r in enumerate(group):
            path = paths[slots[g]]
            qs, gs, ms = json.loads(r["questions"]), json.loads(r["gold"]), json.loads(r["meta"])
            for qid, qq in qs.items():
                if g and qid != q:
                    continue  # partners: the key question only
                new_id = qid if g == 0 else "%s_p%d" % (qid, g)
                questions[new_id], gold[new_id] = _point(qq, field, path), gs[qid]
                meta[new_id] = dict(ms.get(qid, {}), kind=kind if g == 0 else kind + "_other", ref=kind)
        new.append(
            dict(
                a,
                id="~".join([a["id"], kind] + [p["id"] for p in group[1:]]),
                state=json.dumps(state, ensure_ascii=False),
                questions=json.dumps(questions, ensure_ascii=False),
                gold=json.dumps(gold, ensure_ascii=False),
                meta=json.dumps(meta, sort_keys=True),
            )
        )
    return new


def _lowest(q):
    k = len(q["criteria"])
    return {
        "type": "score",
        "score": 0.0,
        "confidence": 1.0,
        "legend": {str(i): c for i, c in enumerate(q["criteria"])},
        "probabilities": {str(i): float(i == 0) for i in range(k)},
    }


def add_mismatch_rows(rows, role, relation, templates, rng, share):
    """For about `share` of the two-part rows, a copy whose `reply` comes from another row, so it
    no longer answers `context`. The `relation` question (the template the file marks `asked:
    mismatch`) is added to the original with its own gold and to the copy with the complement; the
    copy also gets each `on_mismatch: lowest` score at its lowest level and each `on_mismatch:
    affirm` question as a yes. Returns (rows with the relation question added, the copies)."""
    ctx, rep = role["context"], role["reply"]
    usable = [i for i, r in enumerate(rows) if {ctx, rep} <= set(json.loads(r["state"]))]
    yes = relation.gold({})  # `asked: mismatch` golds take nothing from the row
    rel_meta = {"template": relation.id, "level": relation.level, "form": relation.form}
    on_mismatch = [(t.id, t.on_mismatch) for kind in ("lowest", "affirm") for t in templates if t.on_mismatch == kind]
    out, new = list(rows), []
    for i in usable:
        if rng.random() >= share:
            continue
        a = rows[i]
        b = rows[rng.choice(usable)]
        if b["id"] == a["id"]:
            continue
        sa, sb = json.loads(a["state"]), json.loads(b["state"])
        if sa[rep] == sb[rep] or sa[ctx] == sb[ctx]:
            continue
        qa, ga, ma = json.loads(a["questions"]), json.loads(a["gold"]), json.loads(a["meta"])
        wording = rng.choice(list(relation.wordings))
        qa[relation.id] = {"type": relation.type, "instructions": wording}
        ga[relation.id] = yes
        ma[relation.id] = dict(rel_meta, kind="original", wording=relation.wordings.index(wording))
        out[i] = dict(
            a,
            questions=json.dumps(qa, ensure_ascii=False),
            gold=json.dumps(ga, ensure_ascii=False),
            meta=json.dumps(ma, sort_keys=True),
        )
        wording = rng.choice(list(relation.wordings))
        questions = {relation.id: {"type": relation.type, "instructions": wording}}
        gold = {relation.id: opposite(yes)}
        meta = {relation.id: dict(rel_meta, kind="mismatch", ref="mismatch", wording=relation.wordings.index(wording))}
        for qid, kind in on_mismatch:
            if qid not in qa:
                continue
            questions[qid] = qa[qid]
            gold[qid] = _lowest(qa[qid]) if kind == "lowest" else {"type": "noul", "noul": 1.0}
            meta[qid] = dict(ma.get(qid, {}), kind="mismatch", ref="mismatch")
        new.append(
            dict(
                a,
                id="~".join([a["id"], "mismatch", b["id"]]),
                state=json.dumps(dict(sa, **{rep: sb[rep]}), ensure_ascii=False),
                questions=json.dumps(questions, ensure_ascii=False),
                gold=json.dumps(gold, ensure_ascii=False),
                meta=json.dumps(meta, sort_keys=True),
            )
        )
    return out, new


_REF = re.compile(r"`([^`]+)`")


def structure_instructions(r, rng, share):
    """About `share` of the row's questions get their instructions as an object, as the docs
    allow: {"question": ...}, and half of those also {"inspect": <the first referenced path>}."""
    qs, ms = json.loads(r["questions"]), json.loads(r["meta"])
    changed = False
    for qid, q in qs.items():
        if isinstance(q["instructions"], str) and rng.random() < share:
            obj = {"question": q["instructions"]}
            refs = _REF.findall(q["instructions"])
            if refs and rng.random() < 0.5:
                obj["inspect"] = refs[0]
            q["instructions"] = obj
            ms[qid] = dict(ms.get(qid, {}), instructions="object")
            changed = True
    if not changed:
        return r
    return dict(r, questions=json.dumps(qs, ensure_ascii=False), meta=json.dumps(ms, sort_keys=True))


def augment(name, rows, seed, variant):
    """The training augmentations for one source's rows of one variant (docs/DATASET.md). What each
    source offers is declared in its template file: the state's `role`s say which augmentations can
    act on it, and a question's `asked` / `on_mismatch` say what a mismatched copy is asked."""
    from .templates import roles, templates_by

    role = roles(name)
    rng = random.Random("v1|%d|%s|%d" % (seed, name, variant))
    rows = list(rows)
    if role.get("text"):
        rows += add_reference_rows(rows, role["text"], rng, SHARES["reference"], LAYOUTS)
    if role.get("context"):
        relation = templates_by(name, asked="mismatch")
        rows, extra = add_mismatch_rows(rows, role, relation[0], templates_by(name), rng, SHARES["mismatch"])
        rows += extra
    return [structure_instructions(r, rng, SHARES["structured"]) for r in rows]


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
        raise ValueError(
            "source %r is not allowed for %s (license %s, use=%s)" % (name, split.value, spec["license"], spec["use"])
        )
    return hf_split, hf_split == spec["splits"].get(Split.EVAL.value if split is Split.TRAIN else Split.TRAIN.value)


def sample_indices(total: int, n: int, seed: int, split: Split, shared: bool, eval_n: int):
    """Indices for `split`. When the two splits share one Hugging Face split, keep them disjoint:
    eval takes its seeded sample first, training draws from what is left."""
    if split is Split.EVAL or not shared:
        return sorted(random.Random(seed).sample(range(total), min(n, total)))
    held_out = set(random.Random(seed).sample(range(total), min(eval_n, total)))  # what eval takes
    rest = [i for i in range(total) if i not in held_out]
    return sorted(random.Random(seed + 1).sample(rest, min(n, len(rest))))


def build(
    name: str,
    split: Split = Split.EVAL,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    configs=None,
    eval_n: int = DEFAULT_N,
    refresh: bool = False,
    variant: int = 0,
    p_opposite: float = 0.5,
    options=None,
):
    """Return (Dataset, info) for source `name`: `n` seeded random rows per config.

    `eval_n` is the evaluation sample size to hold out when a source's train and eval rows come
    from the same Hugging Face split; it must match the `n` used for evaluation runs.

    `variant` 0 asks the canonical questions. 1, 2, ... are training variants: a templated source
    (see templates.py) draws wordings and yes/no opposites, and every choice question gets its
    options cut per OPTION_MIX and shuffled. `options=k` instead cuts every choice question to a
    fixed k (a seeded, reproducible draw), for evaluating a many-label source at k options.
    Every row carries `meta`: per question, its template, level, form and kind.

    Training rows are dataset v1's (docs/DATASET.md): each source's template file declares its
    `train_keep:` filter, and, through its state roles and question keys, the augmentations its
    training variants get.
    """
    spec = SOURCES[name]
    split = Split(split)
    hf_split, shared = resolve_split(spec, split, name)
    configs = tuple(configs or spec["configs"])
    revision = dataset_revision(spec["dataset"], spec.get("parquet", False))
    templated = "templates" in spec
    key = "%s-%s-%s-n%d-s%d-p%d-%s" % (name, split.value, hf_split, n, seed, PROMPT_VERSION, "_".join(configs))
    if variant:
        key += "-v%d-o%g" % (variant, p_opposite)
    if options:
        key += "-k%d" % options
    if split is Split.TRAIN:
        key += "-r1"  # the training recipe is part of the key
    train_rows = split is Split.TRAIN
    path = os.path.join(CACHE_DIR, key)
    info_path = os.path.join(path, "benchmark_info.json")
    if not refresh and os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if info["revision"] == revision and info.get("eval_n") == eval_n and info.get("meta_version") == META_VERSION:
            return load_from_disk(path), info

    parts, sampled = [], {}
    labels = None
    for config in configs:
        raw = load_split(spec, config, hf_split, revision)
        if labels is None:
            labels = label_set(spec, config, revision)
        if spec.get("prepare"):  # reshape the split first, e.g. join a message to its parent
            raw = spec["prepare"](raw)
        if spec.get("keep"):  # drop rows a source cannot turn into a question, before sampling
            raw = raw.filter(spec["keep"])
        if train_rows:
            from .templates import train_keep

            row_filter = train_keep(name)
            if row_filter:
                raw = raw.filter(row_filter)
        indices = sample_indices(len(raw), n, seed, split, shared, eval_n)
        sampled[config] = indices
        if templated:

            def convert(r, i, config=config, indices=indices, labels=labels):
                rec = spec["record"](r, indices[i], hf_split, config, labels)
                source = rec.pop("source", name)
                row_id = "%s/%s/%d" % (source, hf_split, indices[i])
                state, questions, gold, meta = render(spec["templates"], rec, row_id, variant, seed, p_opposite)
                out = make_row(source, hf_split, indices[i], state, questions, gold)
                out["meta"] = json.dumps(meta, sort_keys=True)
                return reshape(out)

        else:
            levels = spec.get("levels", {})

            def convert(r, i, config=config, indices=indices, labels=labels, levels=levels):
                out = spec["fn"](r, indices[i], hf_split, config, labels)
                questions = json.loads(out["questions"])
                # the questions come with the data, so the form is read off the wording
                out["meta"] = json.dumps(
                    {
                        q: {
                            "template": q,
                            "level": levels.get(q),
                            "form": question_form(qq["instructions"]),
                            "kind": "original",
                        }
                        for q, qq in questions.items()
                    },
                    sort_keys=True,
                )
                return reshape(out)

        def reshape(out):
            if not (variant or options):
                return out
            questions, gold = json.loads(out["questions"]), json.loads(out["gold"])
            rng = random.Random("options|%d|%s|%d|%s" % (seed, out["id"], variant, options))
            limit_options(questions, gold, rng, k=options, shuffle=bool(variant))
            # not sorted: a row's dict keys are otherwise sorted, which would put the options back in
            # alphabetical order and undo the shuffle
            out["questions"], out["gold"] = json.dumps(questions, ensure_ascii=False), json.dumps(
                gold, ensure_ascii=False
            )
            return out

        converted = raw.select(indices).map(convert, with_indices=True, remove_columns=raw.column_names)
        parts.append(converted)

    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    if variant and train_rows:
        rows = augment(name, ds, seed, variant)
        ds = Dataset.from_list(rows, features=ds.features)
    for r in ds:
        validate(r)

    info = {
        "source": name,
        "dataset": spec["dataset"],
        "revision": revision,
        "split": hf_split,
        "role": split.value,
        "shared_split": shared,
        "configs": list(configs),
        "n": n,
        "seed": seed,
        "eval_n": eval_n,
        "rows": len(ds),
        "sampled_indices": sampled,
        "variant": variant,
        "options": options,
        "prompt_version": PROMPT_VERSION,
        "meta_version": META_VERSION,
        "license": spec["license"],
        "use": spec["use"],
        "recipe": "v1",
    }
    ds.save_to_disk(path)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=1)
    return Dataset.load_from_disk(path), info
