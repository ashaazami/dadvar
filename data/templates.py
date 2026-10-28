"""Every source as one YAML file: what to load, what the model sees, and the questions to ask.

    data/templates/<dataset>/<variant>.yaml     one source, e.g. ethics/deontology.yaml;
                                                a dataset with one source: <dataset>/default.yaml
    data/templates/<dataset>/_<name>.yaml       questions shared by others, pulled in with `include:`

There is no registry in code: `registry()` reads these files, and the source-level keys of each are
that source's entry in `SOURCES`. A file declares:

    source: helpsteer_v1           # the source's name; several files may share it, one per config
    dataset: nvidia/HelpSteer      # the Hugging Face dataset
    config: commonsense            # this file's config, when a source has one file per config
    configs: [default]             # the configs to load
    splits: {eval: validation, train: train}      # which HF split each kind of row comes from; a
                                   #   null train split refuses training. When both name the same
                                   #   split, the builder carves disjoint seeded slices out of it.
    labels: [test, label_text]     # optional: which split and column hold the full label set, read
                                   #   before sampling so every row sees every option
    license: cc-by-4.0
    use: {benchmark: true, train: true}           # what the source may be used for; enforced
    commercial: true               # whether it may train a model that is sold or served commercially;
                                   #   `prepare_data --commercial` builds only such sources
    default: false                 # optional: left out of an `all` run unless named
    parquet: true                  # optional: read the Parquet conversion, not the loading script
    keep: not_redacted             # optional: a row filter (KEEPS in sources.py), applied before
                                   #   sampling, so a row is either usable or absent
    prepare: squad_prepare         # optional: reshape the whole split first (PREPARES)
    oversample: {fn: out_of_scope, share: 0.15}   # optional: repeat matching training items
    row_source: ethics/deontology  # optional: row id prefix, when a source's variants share an entry;
                                   #   may use fields and the dataset config: massive_intent/{{ hf_config }}
    include: _attributes.yaml      # optional: questions from shared files (a name or a list)
    record: <name>                 # optional: a Python function (RECORDS in sources.py) that computes
                                   #   fields from a row, e.g. options from the label set; or
                                   #   {fn: <name>, <parameter>: <value>}
    fields:                        # optional: names for columns, e.g. statement: question
    train: false                   # optional: keep the source out of the training mix (benchmark only)
    train_keep: {<column>: [...]}  # optional: training rows only, keep rows whose column is one of these
    state:                         # what the model sees: state key -> field (a column, or a record
      prompt: prompt               #   or `fields` name); the dict form takes
      comment: {from: text, role: text}    #   optional: true, to leave the key out when the field is
                                   #   empty, and `role`, which says what the augmentations may do with
                                   #   it: `text` (the one text a reference augmentation re-places in a
                                   #   larger state), or `context` + `reply` (a two-part source, where a
                                   #   mismatched copy takes another row's reply)
    questions:
      <question id>:
        type: choice | noul | score
        level: 3                   # reasoning level, L1-L6 (docs/DATASET.md)
        grade: 3                   # grade, G1-G5 (docs/DATASET.md)
        form: question             # optional: question | statement | A | B
        canonical: true            # optional: asked in evaluation
        asked: mismatch            # optional: not asked of a plain row; the named augmentation adds it
        on_mismatch: lowest        # optional: what a mismatched copy answers instead — `lowest`
                                   #   (a score, at its lowest level) or `affirm` (a yes)
        wordings: [...]            # the first is the canonical wording; {{ field }} fills a field
        opposites: [...]           # yes/no only: exact complements, gold 1 - p
        criteria: [...] or {...}   # score levels (low to high) or choice options; or
        criteria_from: <field>     #   the options differ per row: a field holds them
        option_refs: [oos]         # option names the wording refers to in backticks (with criteria_from)
        applies_if: <field>        # optional: asked only when the field is set (not null or false)
        state_extra: {claim: statement}   # optional: state keys this question adds
        gold: <name> or {fn: <name>, <parameter>: <value>}   # see GOLD below

A source whose questions come with the data (typed_decisions) keeps a Python row builder instead:
`rows: <name>` (ROWS in sources.py), with `levels:` and `grades:` per question id ("*": the rest).

Wording is data, logic is code: gold functions, record functions and row builders stay in Python.
Checked when a file is loaded: every backtick reference names a declared state key, a question's
own option, or an `option_refs` entry; the files of a multi-config source agree on their
source-level keys. When a row is converted: every field the state maps exists.

A templated source splits conversion in two:

    record(r, index, split, config, labels) -> {"state": {...}, **fields}
        what the row says: the state the model sees, plus any fields templates need
    templates: [Template, ...]
        the questions that can be asked about a record, each with its own gold function

Evaluation (variant 0) asks every template marked `canonical`, with its first wording and no
opposites, so benchmark questions are stable. Training variants (1, 2, ...) draw per row, seeded
by (seed, row id, variant): which wording, and whether to add a yes/no question's opposite as a
second question on the same state. Each question's template, level, form and kind are returned as
`meta`, so metrics can be split by them.

A wording names a row's fields with Jinja2's "{{ statement }}"; a misspelled one fails the build.
"""

import glob
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_REF = re.compile(r"`([^`]+)`")
_JINJA = SandboxedEnvironment(undefined=StrictUndefined, keep_trailing_newline=True)


def fill(text, rec):
    """A wording with {{ field }} placeholders filled from the row; text without any is unchanged, and
    an unknown field is an error."""
    if "{{" not in text and "{%" not in text:
        return text
    return _JINJA.from_string(text).render(**rec)


def _fn(spec):
    """(name, parameters) of `name` or {fn: name, ...}."""
    if isinstance(spec, str):
        return spec, {}
    return spec["fn"], {k: v for k, v in spec.items() if k != "fn"}


# ------------------------------------------------------ a question, and the questions about one row


@dataclass
class Template:
    id: str
    type: str  # choice | noul | score
    level: int  # 1..6, see docs/DATASET.md
    wordings: Sequence[str]  # instructions; the first is the canonical one
    gold: Callable[[Dict], Dict]  # record -> TypeSafe answer (gold_* helpers)
    criteria: Any = None  # fixed criteria, or record -> criteria
    form: str = "question"  # question | statement | A (claim is the question) | B (fixed question about a claim)
    opposites: Sequence[str] = ()  # yes/no only: exact complements, gold 1 - p
    applies: Callable[[Dict], bool] = lambda rec: True
    state_extra: Optional[Callable[[Dict], Dict]] = None  # fields this template adds to the state
    canonical: bool = True  # asked in evaluation
    asked: str = "always"  # always, or the augmentation that adds it ("mismatch")
    on_mismatch: str = ""  # what a mismatched copy answers: lowest | affirm


def opposite(answer: Dict) -> Dict:
    """The complement of a yes/no answer: gold 1 - p."""
    return {"type": "noul", "noul": 1.0 - float(answer["noul"])}


def render(templates: List[Template], rec: Dict, row_id: str, variant: int, seed: int, p_opposite: float = 0.5):
    """(state, questions, gold, meta) for one record.

    variant 0: canonical templates, first wording, no opposites.
    variant k: every applicable template, a seeded wording, and for yes/no templates with
    opposites, the complement added as a second question with probability `p_opposite`.

    A template with `asked` other than "always" is left out: an augmentation adds it to the rows it
    builds (`build.py`), so it is never asked of a plain row.
    """
    rng = random.Random("%d|%s|%d" % (seed, row_id, variant))
    state = dict(rec["state"])
    questions, gold, meta = {}, {}, {}
    for t in templates:
        if t.asked != "always" or not t.applies(rec) or (variant == 0 and not t.canonical):
            continue
        if t.state_extra:
            state.update(t.state_extra(rec))
        crit = t.criteria(rec) if callable(t.criteria) else t.criteria
        wording = t.wordings[0] if variant == 0 else rng.choice(list(t.wordings))
        q = {"type": t.type, "instructions": fill(wording, rec)}
        if crit is not None:
            q["criteria"] = crit
        answer = t.gold(rec)
        questions[t.id], gold[t.id] = q, answer
        meta[t.id] = {
            "template": t.id,
            "level": t.level,
            "form": t.form,
            "kind": "original",
            "wording": t.wordings.index(wording),
        }
        if variant and t.type == "noul" and t.opposites and rng.random() < p_opposite:
            opp = rng.choice(list(t.opposites))
            qid = t.id + "_opposite"
            questions[qid] = {"type": "noul", "instructions": fill(opp, rec)}
            gold[qid] = opposite(answer)
            meta[qid] = {
                "template": t.id,
                "level": t.level,
                "form": t.form,
                "kind": "opposite",
                "wording": list(t.opposites).index(opp),
            }
    return state, questions, gold, meta


# ----------------------------------------------------------------------- gold functions
#
# Each takes (question id, parameters, criteria) and returns rec -> TypeSafe answer; `criteria(rec)`
# gives the question's criteria (fixed, or the row's own with `criteria_from`). `column` defaults to
# the question id.


def _gold():
    from . import sources as S

    def noul(qid, p, crit):
        col = p.get("column", qid)
        return lambda rec: S.gold_noul(rec[col])

    def flag(qid, p, crit):
        col, eq = p["column"], p.get("equals", True)
        return lambda rec: S.gold_noul(type(eq)(rec[col]) == eq)

    def constant(qid, p, crit):
        return lambda rec: S.gold_noul(p["value"])

    def share(qid, p, crit):
        col = p.get("column", qid)
        return lambda rec: S.gold_noul_share(rec[col])

    def share_by_value(qid, p, crit):
        col, values = p["column"], p["values"]
        return lambda rec: S.gold_noul_share(values[int(rec[col])])

    def choice(qid, p, crit):
        col, labels = p.get("column", qid), p.get("labels")
        if labels:
            return lambda rec: S.gold_choice(labels[int(rec[col])], list(crit(rec)))
        return lambda rec: S.gold_choice(rec[col], list(crit(rec)))

    def rating(qid, p, crit):
        col, offset = p.get("column", qid), p.get("offset", 0)
        return lambda rec: S.gold_score(int(rec[col]) + offset, crit(rec))

    def score_share(qid, p, crit):
        col, scale = p.get("column", qid), p.get("scale")
        if scale:
            lo, hi = scale
            return lambda rec: S.gold_score_share((rec[col] - lo) / (hi - lo), crit(rec))
        return lambda rec: S.gold_score_share(rec[col], crit(rec))

    def votes(qid, p, crit):
        col = p.get("column", qid)
        return lambda rec: S.gold_score_votes(rec[col], crit(rec))

    def python(qid, p, crit):
        return S.GOLD_FUNCTIONS[p["name"]]

    return {
        "noul": noul,
        "flag": flag,
        "constant": constant,
        "share": share,
        "share_by_value": share_by_value,
        "choice": choice,
        "rating": rating,
        "score_share": score_share,
        "votes": votes,
        "python": python,
    }


# ------------------------------------------------------------------------------ loading


def _read(path, seen=()):
    """The file's declarations, with `include`d questions first (the file's own win on a clash)."""
    if path in seen:
        raise ValueError("include cycle: %s" % " -> ".join(seen + (path,)))
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    questions = {}
    inc = spec.get("include") or []
    for name in [inc] if isinstance(inc, str) else inc:
        questions.update(_read(os.path.join(os.path.dirname(path), name), seen + (path,)).get("questions", {}))
    questions.update(spec.get("questions") or {})
    spec["questions"] = questions
    return spec


def _state_field(v):
    """(field, optional, role) for one `state:` entry, in either the plain or the dict form."""
    if isinstance(v, str):
        return v, False, ""
    return v["from"], bool(v.get("optional")), v.get("role", "")


ROLES = ("text", "context", "reply")
ON_MISMATCH = ("lowest", "affirm")
AUGMENTATIONS = ("mismatch",)


def _check_roles(path, spec):
    """The `role:` a state entry claims, checked and returned as {role: state key}."""
    rel, roles = os.path.relpath(path, TEMPLATE_DIR), {}
    for key, v in (spec.get("state") or {}).items():
        role = _state_field(v)[2]
        if not role:
            continue
        if role not in ROLES:
            raise ValueError("%s: state key %r claims role %r (known: %s)" % (rel, key, role, ", ".join(ROLES)))
        if role in roles:
            raise ValueError("%s: state keys %r and %r both claim role %r" % (rel, roles[role], key, role))
        roles[role] = key
    if bool(roles.get("context")) != bool(roles.get("reply")):
        raise ValueError(
            "%s: a two-part source needs both `context` and `reply` roles, it has %s"
            % (rel, ", ".join(sorted(roles)) or "neither")
        )
    if "text" in roles and "context" in roles:
        raise ValueError("%s: a source is either one text or two parts, not both" % rel)
    return roles


def _check_question_keys(path, spec, roles):
    """`asked:` and `on_mismatch:`: known values, and only where the augmentation can act on them."""
    rel = os.path.relpath(path, TEMPLATE_DIR)
    for qid, q in spec["questions"].items():
        asked, on_mismatch = q.get("asked", "always"), q.get("on_mismatch", "")
        if asked != "always" and asked not in AUGMENTATIONS:
            raise ValueError(
                "%s: question %r is asked %r (known: always, %s)" % (rel, qid, asked, ", ".join(AUGMENTATIONS))
            )
        if on_mismatch and on_mismatch not in ON_MISMATCH:
            raise ValueError(
                "%s: question %r has on_mismatch %r (known: %s)" % (rel, qid, on_mismatch, ", ".join(ON_MISMATCH))
            )
        if on_mismatch == "lowest" and q["type"] != "score":
            raise ValueError("%s: question %r is on_mismatch lowest but is a %s, not a score" % (rel, qid, q["type"]))
        if on_mismatch == "affirm" and q["type"] != "noul":
            raise ValueError("%s: question %r is on_mismatch affirm but is a %s, not a noul" % (rel, qid, q["type"]))
        if (on_mismatch or asked == "mismatch") and "context" not in roles:
            raise ValueError(
                "%s: question %r is about mismatched copies, but the state declares no "
                "`context` and `reply` roles to build one from" % (rel, qid)
            )
        if asked == "mismatch":
            for text in q["wordings"]:
                if "{{" in text or "{%" in text:
                    raise ValueError(
                        "%s: question %r is added by an augmentation, after the row is built, so "
                        "its wordings cannot use {{ field }} placeholders" % (rel, qid)
                    )


def _check_references(path, spec):
    state = set(spec.get("state") or {})
    for qid, q in spec["questions"].items():
        crit = q.get("criteria")
        allowed = state | set(q.get("state_extra") or {}) | set(q.get("option_refs") or [])
        if isinstance(crit, dict):
            allowed |= set(crit)
        texts = list(q.get("wordings", [])) + list(q.get("opposites", []))
        if isinstance(crit, dict):
            texts += [v for v in crit.values() if isinstance(v, str)]
        elif isinstance(crit, list):
            texts += [c for c in crit if isinstance(c, str)]
        for text in texts:
            for ref in _REF.findall(text):
                if re.split(r"[.\[]", ref, maxsplit=1)[0] not in allowed and ref not in allowed:
                    raise ValueError(
                        "%s: question %r refers to `%s`, which is neither a state key (%s) nor one of "
                        "its options" % (os.path.relpath(path, TEMPLATE_DIR), qid, ref, ", ".join(sorted(state)))
                    )


def _template(qid, q, applies, gold_fns):
    if "criteria_from" in q:
        field = q["criteria_from"]
        criteria, crit = (lambda rec: rec[field]), (lambda rec: rec[field])
    else:
        criteria = q.get("criteria")
        crit = lambda rec: criteria
    name, params = _fn(q["gold"])
    if name not in gold_fns:
        raise ValueError("question %r: unknown gold function %r (known: %s)" % (qid, name, ", ".join(gold_fns)))
    if "applies_if" in q:
        field, base = q["applies_if"], applies
        applies = lambda rec: base(rec) and rec.get(field) is not None and rec.get(field) is not False
    extra = q.get("state_extra")
    state_extra = (lambda rec: {k: rec[f] for k, f in extra.items()}) if extra else None
    return Template(
        qid,
        q["type"],
        q["level"],
        wordings=q["wordings"],
        gold=gold_fns[name](qid, params, crit),
        criteria=criteria,
        form=q.get("form", "question"),
        opposites=q.get("opposites", ()),
        applies=applies,
        state_extra=state_extra,
        canonical=q.get("canonical", True),
        asked=q.get("asked", "always"),
        on_mismatch=q.get("on_mismatch", ""),
    )


def load(files):
    """What a source's template file(s) say about its questions: one path, or {config: path} when
    each config of the dataset is its own variant. The source-level metadata in the same files is
    read by `registry()`."""
    from . import sources as S

    by_config = files if isinstance(files, dict) else {None: files}
    specs, templates, grades, levels, out, roles = {}, [], {}, {}, {}, {}
    gold_fns = _gold()
    for config, rel in by_config.items():
        path = os.path.join(TEMPLATE_DIR, rel)
        spec = _read(path)
        _check_references(path, spec)
        roles.update(_check_roles(path, spec))
        _check_question_keys(path, spec, roles)
        if spec.get("train") is False:
            out["train"] = False
        if spec.get("train_keep"):
            out["train_keep"] = spec["train_keep"]
        specs[config] = (rel, spec)
        if spec.get("rows"):  # questions come with the data
            out["fn"] = S.ROWS[spec["rows"]]
            levels.update(spec.get("levels") or {})
            grades.update(spec.get("grades") or {})
            continue
        applies = (lambda c: lambda rec: rec["_config"] == c)(config)
        for qid, q in spec["questions"].items():
            templates.append(_template(qid, q, applies, gold_fns))
            grades[qid] = q.get("grade")

    def record(r, index, split, config, labels):
        rel, spec = specs[config] if config in specs else specs[None]
        rec = dict(r)
        if spec.get("record"):
            fn, params = _fn(spec["record"])
            rec.update(S.RECORDS[fn](r, index, split, config, labels, **params))
        for key, col in (spec.get("fields") or {}).items():
            rec[key] = rec[col]
        state = {}
        for key, v in spec["state"].items():
            field, optional, _ = _state_field(v)
            if field not in rec:
                raise KeyError(
                    "%s maps state key %r to %r, which the row does not have (it has: %s)"
                    % (rel, key, field, ", ".join(sorted(rec)))
                )
            if optional and rec[field] is None:
                continue
            state[key] = rec[field]
        rec.update(state=state, _config=config if config in specs else None)
        if spec.get("row_source"):  # may name fields, and the dataset config as hf_config
            rec["source"] = fill(spec["row_source"], dict(rec, hf_config=config))
        return rec

    if templates:
        out.update(record=record, templates=templates)
    if levels:
        out["levels"] = levels
    out["grades"] = grades
    out["roles"] = roles
    return out


def attach(sources):
    """Give every registry entry with `template_files` what its files declare."""
    for spec in sources.values():
        if "template_files" in spec:
            spec.update(load(spec["template_files"]))


# ---------------------------------------------------------------------------------- the registry
#
# There is no registry in code: every source is one template file (or one per config), and its
# metadata is the source-level keys of that file. `registry()` collects them; the callables a file
# names (`keep`, `prepare`, `oversample`, `record`, `rows`) are looked up in `sources.py`.

SOURCE_KEYS = (
    "source",
    "dataset",
    "config",
    "configs",
    "splits",
    "labels",
    "license",
    "use",
    "commercial",
    "default",
    "parquet",
    "keep",
    "prepare",
    "oversample",
)
# everything else a file may declare is about its questions, not its source
QUESTION_KEYS = (
    "row_source",
    "include",
    "record",
    "rows",
    "fields",
    "train",
    "train_keep",
    "state",
    "questions",
    "levels",
    "grades",
)


def _source_entry(rel, spec):
    """The registry fields one file declares, with its callables resolved."""
    from . import sources as S

    if not isinstance(spec.get("commercial"), bool):
        raise ValueError("%s: `commercial` must be true or false (may it train a model sold commercially?)" % rel)
    out = {
        "dataset": spec["dataset"],
        "configs": tuple(spec["configs"]),
        "splits": spec["splits"],
        "license": spec["license"],
        "use": spec["use"],
        "commercial": spec["commercial"],
    }
    if spec.get("labels"):
        out["labels"] = tuple(spec["labels"])
    for key, table in (("keep", S.KEEPS), ("prepare", S.PREPARES)):
        if spec.get(key):
            if spec[key] not in table:
                raise ValueError("%s: unknown %s %r (known: %s)" % (rel, key, spec[key], ", ".join(sorted(table))))
            out[key] = table[spec[key]]
    if spec.get("oversample"):
        fn, share = spec["oversample"]["fn"], spec["oversample"]["share"]
        if fn not in S.OVERSAMPLES:
            raise ValueError("%s: unknown oversample %r (known: %s)" % (rel, fn, ", ".join(sorted(S.OVERSAMPLES))))
        out["oversample"] = (S.OVERSAMPLES[fn], share)
    for key in ("default", "parquet"):
        if key in spec:
            out[key] = spec[key]
    return out


def registry():
    """Every source, read from its template file(s), by source name. A file whose name starts with
    "_" is shared by others and declares no source of its own."""
    files, entries = {}, {}
    for path in sorted(glob.glob(os.path.join(TEMPLATE_DIR, "*", "*.yaml"))):
        if os.path.basename(path).startswith("_"):
            continue
        rel = os.path.relpath(path, TEMPLATE_DIR)
        with open(path) as f:
            spec = yaml.safe_load(f) or {}
        unknown = [k for k in spec if k not in SOURCE_KEYS + QUESTION_KEYS]
        if unknown:
            raise ValueError("%s: unknown key(s) %s" % (rel, ", ".join(unknown)))
        name = spec.get("source")
        if not name:
            raise ValueError("%s declares no `source`" % rel)
        entry = _source_entry(rel, spec)
        if name in entries and entries[name] != entry:
            differ = [k for k in set(entry) | set(entries[name]) if entry.get(k) != entries[name].get(k)]
            raise ValueError(
                "%s and the other file(s) of source %r disagree on %s" % (rel, name, ", ".join(sorted(differ)))
            )
        entries[name] = entry
        if "config" in spec:  # one file per config of the same source
            files.setdefault(name, {})[spec["config"]] = rel
        elif name in files:
            raise ValueError(
                "%s: source %r is already declared by %s, and neither names a `config`" % (rel, name, files[name])
            )
        else:
            files[name] = rel
    sources = {name: dict(entries[name], template_files=files[name]) for name in sorted(entries)}
    attach(sources)
    return sources


# ------------------------------------------------------------- what a source declares, by question
#
# The registry is the single place these are read from, so nothing else has to know about the files.


def roles(source):
    """{role: state key} for the roles the source's state claims: `text`, or `context` + `reply`."""
    from .sources import SOURCES

    return SOURCES[source].get("roles") or {}


def templates_by(source, **match):
    """The source's templates whose attributes all match, e.g. templates_by(name, asked="mismatch")."""
    from .sources import SOURCES

    return [t for t in SOURCES[source].get("templates", ()) if all(getattr(t, k) == v for k, v in match.items())]


def trains(source):
    """Whether the source is in the training mix: licensed for training, and not `train: false`."""
    from .sources import SOURCES

    spec = SOURCES[source]
    return bool(spec["use"].get("train")) and spec.get("train", True)


def train_sources():
    """Every source in the training mix, in registry order."""
    from .sources import SOURCES

    return tuple(n for n in SOURCES if trains(n))


def train_keep(source):
    """The training-only row filter the file declares (`train_keep:`), or None."""
    from .sources import SOURCES

    spec = SOURCES[source].get("train_keep")
    if not spec:
        return None
    return lambda r: all(r[col] in values for col, values in spec.items())


def grade(source, template):
    """The grade declared for a question, by question or template id ("*" for the rest)."""
    from .sources import SOURCES

    declared = SOURCES[source]["grades"]
    return declared.get(template, declared.get("*"))
