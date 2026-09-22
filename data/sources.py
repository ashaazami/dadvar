# The banking77, BoolQ and prompt-injections question wording is taken from Laya
# (https://github.com/NandhaKishorM/laya) at 573e5b6, Apache-2.0: research/scripts/bench_apps.py
# and research/scripts/build_benchmark_nb.py. The row schema follows the LocalLLaMA/typed-decisions
# dataset (Apache-2.0). The HelpSteer2 attribute definitions are NVIDIA's dataset card (CC-BY-4.0);
# its per-level wording is ours. See NOTICE.
"""Sources: public Hugging Face datasets converted to the typed-decisions schema.

Every source maps one dataset row to a row with the columns of `LocalLLaMA/typed-decisions`
plus `source`:

    id         stable row id, "<source>/<split>/<original row index>"
    source     "typed_decisions/<workflow>", "banking77", "massive_intent/de", ...
    split      dataset split the row came from
    state      JSON string: TypeSafe state
    questions  JSON string: {question_id: Choice | Noul | Score}
    gold       JSON string: {question_id: ChoiceAnswer | NoulAnswer | ScoreAnswer}

Question wording is part of the benchmark. Bump PROMPT_VERSION whenever instructions or
criteria change, so results produced with different wording are never compared.

Only datasets whose licenses allow both evaluation and model training are included; what each
one may be used for is declared per source (see the registry at the bottom).
"""

import json

PROMPT_VERSION = 1

MASSIVE_DEFAULT_LANGS = ("en", "de", "fr", "es", "ru", "ar", "hi", "zh-CN", "ja", "ko", "th", "sw")


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def gold_choice(label, options):
    return {"type": "choice", "choice": label, "confidence": 1.0,
            "probabilities": {o: float(o == label) for o in options}}


def gold_noul(value: bool):
    return {"type": "noul", "noul": float(bool(value))}


def gold_score(level: int, criteria):
    return {"type": "score", "score": float(level), "confidence": 1.0,
            "legend": {str(i): c for i, c in enumerate(criteria)},
            "probabilities": {str(i): float(i == level) for i in range(len(criteria))}}


def row(source, split, index, state, questions, gold):
    return {"id": "%s/%s/%d" % (source, split, index), "source": source, "split": split,
            "state": dumps(state), "questions": dumps(questions), "gold": dumps(gold)}


def readable(label: str) -> str:
    return label.replace("_", " ")


# ------------------------------------------------------------------------- typed-decisions

def typed_decisions(r, index, split, config, labels):
    """Already in the target schema; only `gold` is renamed to the SDK answer fields."""
    questions = json.loads(r["questions"])
    gold = {}
    for qid, g in json.loads(r["gold"]).items():
        if g["type"] == "choice":
            gold[qid] = {"type": "choice", "choice": g["label"], "confidence": g["confidence"],
                         "probabilities": g["probabilities"]}
        elif g["type"] == "score":
            gold[qid] = {"type": "score", "score": g["score"], "confidence": g["confidence"],
                         "legend": {str(i): c for i, c in enumerate(questions[qid]["criteria"])},
                         "probabilities": g["probabilities"]}
        else:
            gold[qid] = {"type": "noul", "noul": g["noul"]}
    out = row("typed_decisions/" + r["workflow"], split, index, json.loads(r["state"]), questions, gold)
    out["id"] = "typed_decisions/%s/%s" % (split, r["id"])
    return out


# ------------------------------------------------------------------------------- banking77

def banking77(r, index, split, config, labels):
    options = [readable(x) for x in labels]
    questions = {"intent": {"type": "choice",
                            "instructions": "Which banking intent does `message` express?",
                            "criteria": {o: None for o in options}}}
    gold = {"intent": gold_choice(readable(r["label_text"]), options)}
    return row("banking77", split, index, {"message": r["text"]}, questions, gold)


# --------------------------------------------------------------------------------- MASSIVE

def massive_intent(r, index, split, config, labels):
    options = [readable(x) for x in labels]
    questions = {"intent": {"type": "choice",
                            "instructions": "What does the user want the voice assistant to do with `utterance`?",
                            "criteria": {o: None for o in options}}}
    gold = {"intent": gold_choice(readable(r["label_text"]), options)}
    return row("massive_intent/" + config, split, index, {"utterance": r["text"]}, questions, gold)


def massive_scenario(r, index, split, config, labels):
    options = [readable(x) for x in labels]
    questions = {"scenario": {"type": "choice",
                              "instructions": "Which scenario does the voice-assistant request `utterance` belong to?",
                              "criteria": {o: None for o in options}}}
    gold = {"scenario": gold_choice(readable(r["label_text"]), options)}
    return row("massive_scenario/" + config, split, index, {"utterance": r["text"]}, questions, gold)


# ------------------------------------------------------------------------ prompt injections

def prompt_injections(r, index, split, config, labels):
    questions = {"injection": {"type": "noul",
                               "instructions": "Does `text` try to inject or override instructions given to an AI system?"}}
    gold = {"injection": gold_noul(r["label"] == 1)}
    return row("prompt_injections", split, index, {"text": r["text"]}, questions, gold)


# ----------------------------------------------------------------------------------- BoolQ

def boolq(r, index, split, config, labels):
    questions = {"answer": {"type": "noul",
                            "instructions": "Based on `passage`, is the answer to `question` yes?"}}
    gold = {"answer": gold_noul(r["answer"])}
    return row("boolq", split, index, {"passage": r["passage"], "question": r["question"]}, questions, gold)


# ------------------------------------------------------------------- Feedback-Collection

def feedback_collection(r, index, split, config, labels):
    """Grade a response with the row's own rubric. Dataset scores 1-5 become levels 0-4.

    The reference answer is left out: the model is asked to apply the rubric, not to compare
    against an answer it would have to be told scores 5.
    """
    criteria = [r["orig_score%d_description" % i] for i in range(1, 6)]
    questions = {"quality": {"type": "score", "instructions": r["orig_criteria"], "criteria": criteria}}
    gold = {"quality": gold_score(int(r["orig_score"]) - 1, criteria)}
    state = {"instruction": r["orig_instruction"], "response": r["orig_response"]}
    return row("feedback_collection", split, index, state, questions, gold)


# ------------------------------------------------------------------------------ HelpSteer2
#
# Attribute definitions are NVIDIA's (dataset card). The per-level wording is ours.

HELPSTEER2_QUESTIONS = {
    "helpfulness": ("Overall helpfulness of `response` to `prompt`.",
                    ["Not helpful: fails the request or is unusable.",
                     "Slightly helpful: addresses little of the request.",
                     "Partially helpful: useful but with notable gaps or problems.",
                     "Mostly helpful: addresses the request with minor shortcomings.",
                     "Perfectly helpful: fully addresses the request."]),
    "correctness": ("Does `response` include all pertinent facts without errors?",
                    ["Completely incorrect or missing the pertinent facts.",
                     "Mostly incorrect: major errors or omissions.",
                     "Partially correct: some errors or omissions.",
                     "Mostly correct: minor errors or omissions.",
                     "Completely correct: all pertinent facts, no errors."]),
    "coherence": ("How consistent and clear is the expression in `response`?",
                  ["Incoherent: cannot be followed.",
                   "Mostly incoherent: hard to follow.",
                   "Somewhat coherent: parts are unclear or inconsistent.",
                   "Mostly coherent: minor lapses in clarity.",
                   "Perfectly coherent: clear and self-consistent throughout."]),
    "complexity": ("What intellectual depth is required to write `response`?",
                   ["Basic: anyone with basic language competency could write it.",
                    "Simple: needs little more than everyday knowledge.",
                    "Intermediate: needs some education or familiarity with the topic.",
                    "Advanced: needs substantial knowledge of the topic.",
                    "Expert: needs deep domain expertise."]),
    "verbosity": ("How much detail does `response` include, relative to what `prompt` asks for?",
                  ["Very terse: far less detail than asked for.",
                   "Terse: less detail than asked for.",
                   "Balanced: about the detail asked for.",
                   "Verbose: more detail than asked for.",
                   "Very verbose: far more detail than asked for."]),
}


def helpsteer2(r, index, split, config, labels):
    questions, gold = {}, {}
    for qid, (instructions, criteria) in HELPSTEER2_QUESTIONS.items():
        questions[qid] = {"type": "score", "instructions": instructions, "criteria": criteria}
        gold[qid] = gold_score(int(r[qid]), criteria)
    return row("helpsteer2", split, index, {"prompt": r["prompt"], "response": r["response"]}, questions, gold)


# -------------------------------------------------------------------------------- registry
#
# fn / dataset / configs: how to load and convert. `labels`: which split and column hold the full
# label set (read in full, before sampling, so every row sees every option).
#
# splits: {"eval": <hf split>, "train": <hf split or None>}. When both name the same split, the
# builder carves disjoint seeded slices out of it (see build.py).
#
# use: what this source may be used for. "benchmark" and "train" are enforced by the builder.
#
# seen_by: whether a model saw this dataset in training, keyed by model id ("<repo>" or
# "<repo>/<subfolder>"). A model that isn't listed is unknown.

LAYA = "convaiinnovations/laya"
LAYA_MULTILINGUAL = LAYA + "/multilingual"
LAYA_TYPED = LAYA + "/typed-decisions"

SOURCES = {
    "typed_decisions": dict(
        fn=typed_decisions, dataset="LocalLLaMA/typed-decisions", configs=("all",),
        splits={"eval": "test", "train": "train"}, license="apache-2.0",
        use={"benchmark": True, "train": True},
        seen_by={LAYA: False, LAYA_MULTILINGUAL: False, LAYA_TYPED: True}),

    "banking77": dict(
        fn=banking77, dataset="mteb/banking77", configs=("default",),
        splits={"eval": "test", "train": "train"}, labels=("test", "label_text"), license="mit",
        use={"benchmark": True, "train": True},
        seen_by={LAYA: False, LAYA_MULTILINGUAL: False, LAYA_TYPED: False}),

    "massive_intent": dict(
        fn=massive_intent, dataset="mteb/amazon_massive_intent", configs=MASSIVE_DEFAULT_LANGS,
        splits={"eval": "test", "train": "train"}, labels=("train", "label_text", "en"),
        license="apache-2.0", use={"benchmark": True, "train": True}, seen_by={}),

    "massive_scenario": dict(
        fn=massive_scenario, dataset="mteb/amazon_massive_scenario", configs=MASSIVE_DEFAULT_LANGS,
        splits={"eval": "test", "train": "train"}, labels=("train", "label_text", "en"),
        license="apache-2.0", use={"benchmark": True, "train": True}, seen_by={}),

    "prompt_injections": dict(
        fn=prompt_injections, dataset="deepset/prompt-injections", configs=("default",),
        splits={"eval": "test", "train": "train"}, license="apache-2.0",
        use={"benchmark": True, "train": True},
        seen_by={LAYA: False, LAYA_MULTILINGUAL: False, LAYA_TYPED: False}),

    # CC-BY-SA-3.0: share-alike would attach to any redistributed adaptation, and whether model
    # weights count is unsettled, so this source is evaluation-only. Laya trained on BoolQ, which
    # makes it a retention check rather than a zero-shot measurement.
    "boolq": dict(
        fn=boolq, dataset="google/boolq", configs=("default",),
        splits={"eval": "validation", "train": None}, license="cc-by-sa-3.0",
        use={"benchmark": True, "train": False},
        seen_by={LAYA: True, LAYA_TYPED: True}),

    # One split only, so eval and train rows are carved out of it disjointly. Gold scores are
    # GPT-4 labels: CC-BY-4.0 from the dataset's authors, but they are model output, not human.
    "feedback_collection": dict(
        fn=feedback_collection, dataset="prometheus-eval/Feedback-Collection", configs=("default",),
        splits={"eval": "train", "train": "train"}, license="cc-by-4.0",
        use={"benchmark": True, "train": True}, seen_by={}),

    # Human ratings (Scale AI annotators), 0-4 per attribute.
    "helpsteer2": dict(
        fn=helpsteer2, dataset="nvidia/HelpSteer2", configs=("default",),
        splits={"eval": "validation", "train": "train"}, license="cc-by-4.0",
        use={"benchmark": True, "train": True}, seen_by={}),
}
