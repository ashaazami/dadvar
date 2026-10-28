# The row schema follows the LocalLLaMA/typed-decisions dataset (Apache-2.0). See NOTICE.
"""Sources: public Hugging Face datasets converted to the typed-decisions schema.

Every source maps one dataset row to a row with the columns of `LocalLLaMA/typed-decisions`
plus `source`:

    id         stable row id, "<source>/<split>/<original row index>"
    source     "typed_decisions/<workflow>", "banking77", "massive_intent/de", ...
    split      dataset split the row came from
    state      JSON string: TypeSafe state
    questions  JSON string: {question_id: Choice | Noul | Score}
    gold       JSON string: {question_id: ChoiceAnswer | NoulAnswer | ScoreAnswer}

Every source is one YAML file under data/templates/ — what to load, what the model sees, and the
questions to ask (see data/templates.py). This module keeps only what has to be code, and what a
file names by name: the functions that compute a row's fields (`record:`), reshape a split
(`prepare:`), drop unusable rows (`keep:`), oversample (`oversample:`), build rows that carry their
own questions (`rows:`), and the gold helpers. `SOURCES` at the bottom is those files, read.

Question wording is part of the benchmark: bump PROMPT_VERSION whenever instructions or criteria
change, so results produced with different wording are never compared.

Each file declares its source's license and what it may be used for (benchmark, training), and the
builder enforces it; a source licensed for evaluation only is never built for training.
"""

import json
import random
import re

from .templates import registry

PROMPT_VERSION = 1

# MASSIVE ships 51 parallel languages, each with the same 2,974 test and 11,514 train rows.
# `--langs` picks which to run; a non-English run needs a model that reads those languages.
MASSIVE_DEFAULT_LANGS = ("en", "de", "fr", "es", "ru", "ar", "hi", "zh-CN", "ja", "ko", "th", "sw")


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def gold_choice(label, options):
    return {
        "type": "choice",
        "choice": label,
        "confidence": 1.0,
        "probabilities": {o: float(o == label) for o in options},
    }


def gold_noul(value: bool):
    return {"type": "noul", "noul": float(bool(value))}


def gold_score(level: int, criteria):
    return {
        "type": "score",
        "score": float(level),
        "confidence": 1.0,
        "legend": {str(i): c for i, c in enumerate(criteria)},
        "probabilities": {str(i): float(i == level) for i in range(len(criteria))},
    }


def gold_noul_share(share: float):
    """Soft yes/no gold: the share of annotators who answered yes."""
    return {"type": "noul", "noul": min(1.0, max(0.0, float(share)))}


def gold_score_share(value: float, criteria):
    """Soft score gold from a 0-1 average rating: the value placed on the level scale (0..k-1) and
    split between the two nearest levels, so the expected level equals the average rating."""
    x = min(1.0, max(0.0, float(value))) * (len(criteria) - 1)
    lo = int(x)
    hi = min(lo + 1, len(criteria) - 1)
    w = x - lo
    probs = {str(i): 0.0 for i in range(len(criteria))}
    probs[str(lo)] += 1.0 - w
    probs[str(hi)] += w
    return {
        "type": "score",
        "score": x,
        "confidence": max(probs.values()),
        "legend": {str(i): c for i, c in enumerate(criteria)},
        "probabilities": probs,
    }


def row(source, split, index, state, questions, gold):
    return {
        "id": "%s/%s/%d" % (source, split, index),
        "source": source,
        "split": split,
        "state": dumps(state),
        "questions": dumps(questions),
        "gold": dumps(gold),
    }


def readable(label: str) -> str:
    return label.replace("_", " ")


_readable = readable


def gold_score_votes(votes, criteria):
    """Soft score gold from annotators' votes: the distribution of votes over the levels."""
    k = len(criteria)
    counts = [0.0] * k
    for v in votes:
        counts[min(k - 1, max(0, int(round(v))))] += 1
    total = sum(counts)
    probs = {str(i): c / total for i, c in enumerate(counts)}
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs.values())),
        "confidence": max(probs.values()),
        "legend": {str(i): c for i, c in enumerate(criteria)},
        "probabilities": probs,
    }


def _options(pairs):
    return {k: v.strip() for k, v in pairs}


# ------------------------------------------------------------------ typed-decisions row builder


def typed_decisions(r, index, split, config, labels):
    """Already in the target schema; only `gold` is renamed to the SDK answer fields."""
    questions = json.loads(r["questions"])
    gold = {}
    for qid, g in json.loads(r["gold"]).items():
        if g["type"] == "choice":
            gold[qid] = {
                "type": "choice",
                "choice": g["label"],
                "confidence": g["confidence"],
                "probabilities": g["probabilities"],
            }
        elif g["type"] == "score":
            gold[qid] = {
                "type": "score",
                "score": g["score"],
                "confidence": g["confidence"],
                "legend": {str(i): c for i, c in enumerate(questions[qid]["criteria"])},
                "probabilities": g["probabilities"],
            }
        else:
            gold[qid] = {"type": "noul", "noul": g["noul"]}
    out = row("typed_decisions/" + r["workflow"], split, index, json.loads(r["state"]), questions, gold)
    out["id"] = "typed_decisions/%s/%s" % (split, r["id"])
    return out


# ----------------------------------------------------------------------- prepare functions
#
# Reshape a whole split before sampling (the registry's `prepare`).

# Columns the prepare functions group per comment (hate_speech) and per summary (summeval).
HATE_COLUMNS = (
    "sentiment",
    "respect",
    "attack_defend",
    "status",
    "insult",
    "humiliate",
    "dehumanize",
    "violence",
    "genocide",
)
SUMMEVAL_COLUMNS = ("coherence", "consistency", "fluency", "relevance")
# The attributes civil_comments keeps a comment for: someone flagged at least one.
CIVIL_COLUMNS = ("toxicity", "insult", "obscene", "threat", "identity_attack")


def oasst_conversation_prepare(ds):
    """English assistant replies that reviewers rated, each with the whole conversation before it
    (every earlier turn, oldest first), not only the message it answers."""
    by_id = {m: (p, role, t) for m, p, role, t in zip(ds["message_id"], ds["parent_id"], ds["role"], ds["text"])}

    def history(parent):
        turns = []
        while parent in by_id:
            p, role, t = by_id[parent]
            turns.append({"role": "user" if role == "prompter" else "assistant", "content": t})
            parent = p
        return turns[::-1]

    ds = ds.filter(
        lambda r: r["role"] == "assistant"
        and r["lang"] == "en"
        and r["parent_id"] in by_id
        and {"quality", "helpfulness", "creativity", "fails_task"} <= set((r["labels"] or {}).get("name") or [])
    )
    return ds.map(lambda r: {"conversation_json": json.dumps(history(r["parent_id"]), ensure_ascii=False)})


def hate_speech_prepare(ds):
    from datasets import Dataset

    by = {}
    for r in ds:
        c = by.setdefault(r["comment_id"], {"text": r["text"], **{k: [] for k in list(HATE_COLUMNS) + ["hatespeech"]}})
        for k in list(HATE_COLUMNS) + ["hatespeech"]:
            if r[k] is not None:
                c[k].append(r[k])
    return Dataset.from_list([{"comment_id": k, **v} for k, v in sorted(by.items())])


def summeval_prepare(ds):
    """One row per machine summary (16 per article), each with its four expert ratings (1-5)."""
    from datasets import Dataset

    out = []
    for r in ds:
        for i, summary in enumerate(r["machine_summaries"]):
            out.append(
                {
                    "article": r["text"],
                    "summary": summary,
                    "article_id": r["id"],
                    "k": i,
                    **{m: r[m][i] for m in SUMMEVAL_COLUMNS},
                }
            )
    return Dataset.from_list(out)


def squad_prepare(ds):
    by_context = {}
    for c, a in zip(ds["context"], ds["answers"]):
        if a["text"]:
            by_context.setdefault(c, set()).add(a["text"][0].strip())
    return ds.map(lambda r: {"context_answers": json.dumps(sorted(by_context.get(r["context"], ())))})


# --------------------------------------------------------------------------- record functions
#
# What a template file's `record:` names: row -> fields its state, wordings and gold read. They
# compute; the wording and the state keys are in the template files. Signature:
# (row, row index, split, config, full label set, **parameters from the file).


def label_options(r, index, split, config, labels, label=None, label_index=None, readable=True):
    """Options from the source's whole label set, and the row's label as `answer`. The label is a
    column's text (`label`) or an index into the label set (`label_index`; the first, for a list)."""
    show = _readable if readable else (lambda x: x)
    if label is not None:
        raw = r[label]
    else:
        i = r[label_index]
        raw = labels[i[0] if isinstance(i, list) else i]
    return {"options": {show(o): None for o in labels}, "answer": show(raw)}


def rubric(r, index, split, config, labels):
    """Feedback-Collection: the row's own five score descriptions."""
    return {"rubric": [r["orig_score%d_description" % i] for i in range(1, 6)]}


def pair_by_parity(r, index, split, config, labels, chosen, rejected):
    """The preferred response is response_a on even rows, response_b on odd."""
    first = index % 2 == 0
    a, b = (r[chosen], r[rejected]) if first else (r[rejected], r[chosen])
    return {"response_a": a, "response_b": b, "better": "a" if first else "b"}


def arena(r, index, split, config, labels):
    """Prompts and responses are JSON lists, one entry per turn, joined into one text. A turn a model
    left unanswered is null (161 of 57,477 rows); it is left out of the join rather than the row
    dropped, since the vote was cast on what was shown."""
    turns = lambda v: (
        " ".join(t for t in json.loads(v) if t is not None) if isinstance(v, str) and v.startswith("[") else v
    )
    return {
        "prompt_text": turns(r["prompt"]),
        "response_a_text": turns(r["response_a"]),
        "response_b_text": turns(r["response_b"]),
        "winner": "a" if r["winner_model_a"] else "b" if r["winner_model_b"] else "tie",
    }


CASE_HOLD_OPTIONS = ("a", "b", "c", "d", "e")


def case_hold(r, index, split, config, labels):
    return {"options": dict(zip(CASE_HOLD_OPTIONS, r["endings"])), "answer": CASE_HOLD_OPTIONS[int(r["label"])]}


def aegis(r, index, split, config, labels):
    """The response is shown, and asked about, only when the row has a labelled one."""
    rec = {"shown_response": None, "unsafe_prompt": r["prompt_label"] == "unsafe", "has_response": False}
    if r["response"] and r["response"] != "None" and r["response_label"] in ("safe", "unsafe"):
        rec.update(shown_response=r["response"], has_response=True, unsafe_response=r["response_label"] == "unsafe")
    return rec


def oasst(r, index, split, config, labels):
    """The conversation before the reply, and the reviewers' ratings by name."""
    return {"conversation": json.loads(r["conversation_json"]), **dict(zip(r["labels"]["name"], r["labels"]["value"]))}


def tab_fact(r, index, split, config, labels):
    """The table as caption, columns and rows."""
    lines = [l for l in r["table_text"].split("\n") if l.strip()]
    rows = [l.split("#") for l in lines]
    return {"table": {"caption": r["table_caption"], "columns": rows[0], "rows": rows[1:]}}


def aqua_rat(r, index, split, config, labels):
    pairs = [(o.split(")", 1)[0].strip(), o.split(")", 1)[1]) for o in r["options"]]
    return {"options": _options(pairs), "answer": r["correct"].strip()}


def math_qa(r, index, split, config, labels):
    parts = re.split(r"(?:^|,\s*)([a-e])\s*\)\s*", r["options"])
    pairs = [(parts[i].upper(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    return {"options": _options(pairs), "answer": r["correct"].strip().upper()}


def gsm8k(r, index, split, config, labels):
    """The final answer, and a wrong one derived from it by a seeded, plausible slip (an off-by
    amount or a wrong operation)."""
    answer = r["answer"].split("####")[-1].strip().replace(",", "")
    n = float(answer)
    rng = random.Random("gsm8k|%s|%d" % (split, index))
    slips = [n + rng.choice([1, 2, 5, 10]), n - rng.choice([1, 2, 5, 10]), n * 2, n / 2 if n % 2 == 0 else n + 3]
    wrong = rng.choice([w for w in slips if w != n and w >= 0] or [n + 1])
    fmt = lambda x: str(int(x)) if float(x).is_integer() else ("%.2f" % x).rstrip("0").rstrip(".")
    return {"final_answer": fmt(n), "wrong_answer": fmt(wrong)}


def hh_rlhf(r, index, split, config, labels):
    """The shared conversation and the two last replies; the preferred one is reply_a on even rows."""
    cut = r["chosen"].rfind("\n\nAssistant:")
    conversation = r["chosen"][:cut].strip()
    chosen = r["chosen"][cut + len("\n\nAssistant:") :].strip()
    rejected = r["rejected"][r["rejected"].rfind("\n\nAssistant:") + len("\n\nAssistant:") :].strip()
    first = index % 2 == 0
    a, b = (chosen, rejected) if first else (rejected, chosen)
    return {"conversation": conversation, "reply_a": a, "reply_b": b, "better": "a" if first else "b"}


NOT_STATED = "not stated"


def squad(r, index, split, config, labels):
    """The extraction options: the answer, the answers to other questions on the same passage
    (none that contains or is contained in the answer), and "not stated"; no options when the
    passage has fewer than two other answers."""
    gold = r["answers"]["text"][0].strip() if r["answers"]["text"] else None
    close = lambda x: gold is not None and (x.lower() in gold.lower() or gold.lower() in x.lower())
    others = [x for x in json.loads(r["context_answers"]) if x and x != NOT_STATED and not close(x)]
    random.Random("squad|%s" % r["id"]).shuffle(others)
    options = None
    if len(others) >= 2:
        labels_ = ([gold] if gold else []) + others[: 4 if gold is None else 3] + [NOT_STATED]
        options = {l: ("`passage` does not give the answer" if l == NOT_STATED else None) for l in labels_}
    return {"answerable": gold is not None, "answer": gold or NOT_STATED, "options": options}


def commonsense_qa(r, index, split, config, labels):
    return {"options": dict(zip(r["choices"]["label"], r["choices"]["text"])), "answer": r["answerKey"]}


RECORDS = {
    f.__name__: f
    for f in (
        label_options,
        rubric,
        pair_by_parity,
        arena,
        case_hold,
        aegis,
        oasst,
        tab_fact,
        aqua_rat,
        math_qa,
        gsm8k,
        hh_rlhf,
        squad,
        commonsense_qa,
    )
}

# gold functions a template file names with {fn: python, name: <name>}: rec -> answer
GOLD_FUNCTIONS = {
    # the annotators' mean of 0/1/2 (no, unsure, yes), as a share of yes
    "hate_speech_share": lambda rec: gold_noul_share(sum(v / 2 for v in rec["hatespeech"]) / len(rec["hatespeech"])),
}

# row builders, for a source whose questions come with the data (template file `rows:`)
ROWS = {"typed_decisions": typed_decisions}


# ------------------------------------------------------------------------------- row filters
#
# A template file names one with `keep:`; it drops rows the source cannot turn into a question,
# before sampling, so a row is either usable or absent (never a hole in the sample).


def single_label(r):
    """go_emotions: comments one emotion was assigned, so the choice has one answer."""
    return len(r["labels"]) == 1


def any_attribute_flagged(r):
    """civil_comments: someone flagged at least one attribute."""
    return any(r[c] > 0 for c in CIVIL_COLUMNS)


def not_redacted(r):
    """Aegis: the prompt's text was withheld."""
    return r["prompt"] != "REDACTED"


def five_options(r):
    """MathQA: the five options parse as a) .. e)."""
    return len(re.findall(r"(?:^|,\s*)[a-e]\s*\)", r["options"])) == 5


def two_assistant_turns(r):
    """hh-rlhf: both sides have an assistant turn, and they differ."""
    return "\n\nAssistant:" in r["chosen"] and "\n\nAssistant:" in r["rejected"] and r["chosen"] != r["rejected"]


def labelled_nli(r):
    """MultiNLI: one of the three labels (-1 marks an unlabelled row)."""
    return r["label"] in (0, 1, 2)


KEEPS = {
    f.__name__: f
    for f in (single_label, any_attribute_flagged, not_redacted, five_options, two_assistant_turns, labelled_nli)
}


# ----------------------------------------------------------------------------- oversampling
#
# `oversample: {fn: <name>, share: <s>}`: repeat training items whose gold matches, up to `share`.


def out_of_scope(gold):
    """CLINC150: the request no intent covers."""
    return gold["intent"]["choice"] == "oos"


OVERSAMPLES = {f.__name__: f for f in (out_of_scope,)}

# `prepare: <name>`: reshape a whole split before sampling
PREPARES = {f.__name__: f for f in (oasst_conversation_prepare, hate_speech_prepare, summeval_prepare, squad_prepare)}


# -------------------------------------------------------------------------------- the registry
#
# There is no registry here: every source is one template file under data/templates/ (or one per
# config), and that file declares its own dataset, configs, splits, license, use, commercial and the names of
# the functions above it needs. `registry()` reads them all.

SOURCES = registry()
