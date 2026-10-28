# Dataset v1: one-step judgments, grounded in the referenced state

The training data recipe. Training rows are always built this way: each source declares what it
contributes in its own template file (its state's `role`s, its questions' `asked` / `on_mismatch`
keys, `train:` and `train_keep:`), and `data/build.py` holds the shares and layouts.

    python -m training.prepare_data --out training/data/v1_full1024.pt v1 \
        --tokenizer answerdotai/ModernBERT-base --max-len 1024 --n 12000 --variants 3 --p-opposite 0.5 \
        --per-source 30000 --seed 13 --eval-n 500 --langs en

## Scope: what a System One model is for, and what it is not

The mix trains what TypeSafe documents as System One's job, and leaves out what it documents as out of
scope ([System One](https://docs.typesafe.ai/concepts/system-one), [Primitives](https://docs.typesafe.ai/primitives),
[Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)).

**In scope, "one snap judgment per question":**

| Decision shape (use-case map) | Example |
|---|---|
| Classification / routing | intent, department, topic, entity type; with an `other` / `none` option |
| Detection | spam, injection, toxicity, refund request, PII: a yes/no probability |
| Scoring | severity, frustration, quality, relevance, on levels that describe situations |
| Relevance / retrieval | does `passage` answer `query`; which candidate is relevant |
| Verification | does `source` support `claim`; does `response` follow `policy`; does `reply` answer `message` |
| Preference / ranking | which of `response_a`, `response_b` is better |
| Extraction as choice | which candidate is the `field` value in `source_text`, or "not stated" |

**Out of scope (docs: "keep it in code"):** arithmetic and counting, date ordering and arithmetic, multi-hop
indirection ("a property of a property"), double negatives beyond one plain negation, generation. Sources
that need these (math_qa, aqua_rat, gsm8k, tab_fact, ruletaker depth ≥ 2) stay in the benchmark as a
boundary check, where the right answer is an uncertain one, and are not trained on (`train: false`).

## The core skill: answer from the referenced part of the state

The Primitives page: *"When a question is about one of those parts, name it in the `instructions` with a
dot-and-index path to its key, including the backticks."* This is the central skill, trained with
contrast: the same state, with questions about different parts whose answers differ.

| Reference form | Example |
|---|---|
| top-level key | `` `message` `` |
| two keys, one relation | `` Does `response` answer `prompt`? `` |
| nested path | `` `ticket.subject` ``, `` `order.customer.note` `` |
| array index | `` `messages[2].text` ``, `` `items[0]` `` |
| no reference (plain string state) | `"My card was charged twice."` |

### Reference augmentations (source-agnostic, training variants only)

Each builds new rows from existing ones, so every source contributes. Gold always comes from the row whose
text the question references.

1. **Sibling.** Two texts from different rows under random names, for every single-text source and every
   question type. For choice and score, the partner row has a different gold answer, so the unreferenced
   text would give the wrong one; for yes/no, it agrees half the time. Questions are asked about both texts.
2. **Nested containers.** The text is placed inside a realistic structure, and the question uses its path:
   a support ticket (`ticket.messages[i].text`, `ticket.subject`), an email (`email.body`, `email.from`),
   a chat log (`chat[i].content`), a record (`record.fields.description`), a document with sections
   (`document.sections[i].text`). The other slots hold texts from other rows with different answers.
3. **Arrays of items.** `items: [t0, t1, t2, …]` from different rows, one question per index:
   `` Is `items[1]` a complaint? ``. This teaches indexing, and repeats the contrast within one request.
4. **Mismatched pairs.** For two-part sources (prompt/response, conversation/reply, query/passage), swap in
   a part from another row. The relation question then has a known answer: `` Does `response` address
   `prompt`? `` no, and the source's helpfulness at its lowest level.
5. **Irrelevant fields.** Metadata keys (ids, timestamps, names, status) around the referenced text, at
   moderate size: the docs name large irrelevant state as a weakness. The question never needs them.

Shares are in `data/build.py` (SHARES): 60% of single-text rows get a reference copy, 30% of two-part rows
a mismatched copy.

### Question forms

- **Wordings and opposites.** A source's question has several wordings, and each yes/no question 2–3
  hand-written exact complements ("Is `comment` insulting?" / "`comment` contains no insult.", gold 1 − p),
  worded with several constructions so negation words do not become a shortcut.
- **Instructions** as a string (most), and as an object in about 15% of questions: `{"question": "..."}`,
  half of those with `"inspect": "<the referenced path>"`.
- **Choice options** with descriptions or bare labels, as each source has them, shuffled in every training
  variant. A choice question with more than 5 options keeps 5 in 70% of variants, 10–30 in 15% and the full
  set in 15% (`OPTION_MIX`); a cut keeps the gold label plus distractors, half the most similar to it and
  half at random. Evaluation keeps the full label sets unless `--options` is given.
- **Score levels** always low → high, as the API specifies. Levels describe situations. No reversed scales.
- **Not generated yet:** rubric-object options (`{"what", "not_for", "examples"}`) and Noul `criteria`.

## Levels and grades

Every template question declares a reasoning **level** (L1–L6) and a **grade** (G1–G5).

The level is what the answer requires; the diagnostics ladder (`benchmarks/diagnostics.py`) is built on it.
L5 and multi-hop L6 are out of scope for training (above) and are kept to measure that boundary.

| Level | The answer requires | Example |
|---|---|---|
| L1 | a property of one text | "Is `comment` insulting?" |
| L2 | its intent or meaning | "Does the customer want their money back?" |
| L3 | relating two texts | "Do `sentence1` and `sentence2` mean the same thing?" |
| L4 | checking a claim against evidence | "The study found the drug lowered blood pressure." |
| L5 | computing over evidence | "The order was delivered within 30 days of `today`." |
| L6 | applying a rule or rubric to facts | "`refund_policy` allows a refund without approval." |

The grade is what the answer relates, and is how validation is reported:

| Grade | What the answer depends on | Examples |
|---|---|---|
| G1 | one property of one referenced text | toxicity, sentiment, injection, is-a-question |
| G2 | one text against a taxonomy of options | intent, department, provision type, emotion |
| G3 | two referenced parts, one relation | relevance, answers-the-question, paraphrase, preference, entailment |
| G4 | one text against a stated rule, policy or rubric | violates `policy`, meets the rubric level, excuse justifies refusal |
| G5 | pick the candidate that fills a field | extraction as choice, with "not stated" |

Every grade is crossed with the reference forms above, so "difficulty" also means locating the right part
of a larger state.

## Sources

**In training:** banking77, clinc_oos, massive_intent, massive_scenario, go_emotions, ledgar,
prompt_injections, civil_comments, aegis_safety, amazon_counterfactual, hate_speech, helpsteer_v1,
helpsteer2, helpsteer3, feedback_collection, oasst_quality, hh_rlhf, reward_bench, arena_preference, paws,
ethics, case_hold, typed_decisions, squad_v2, multi_nli, stsb, amazon_reviews, sst5; **ruletaker depth-0
and depth-1 only.**

**Benchmark only:** math_qa, aqua_rat, gsm8k, tab_fact (out of scope, above); BoolQ, CommonsenseQA and
SummEval (licensed for evaluation only, and held out as zero-shot measures).

Licenses are recorded in each source's template file and in every build's provenance; the newer sources
(SQuAD 2.0, MultiNLI, STS-B, Amazon reviews, SST-5) are included for research use
(`docs/BENCHMARK_DATASETS.md`).

## Split and balance

- **Validation by text, not by row.** Every row an augmentation builds records the source rows it used; a
  source row and every row built from it go to the same side. No validation text appears in training.
- **Whole units.** The per-source cap and the per-epoch sampler draw a state's questions in one variant
  together, so a question and its opposite, or the texts of a sibling pair, are trained together.
- **Balance:** `training.finetune --balance type` draws a third of each epoch per question type and, within
  a type, an even share per source.

**Gap:** validation holds no nested, array or sibling items. Those rows combine several source rows, and a
held-out row's partners are almost never held out too, so they are dropped. Reference-following is
measured by `benchmarks/diagnostics.py` (references, `by_form`) instead; a validation set of reference rows
built from the benchmark split would close the gap.

## Evaluation

- The benchmark: each source's evaluation split, canonical questions.
- `benchmarks/diagnostics.py`: question sensitivity, the L1–L6 ladder, reference following by path form,
  option-order stability and the hand-made out-of-domain set. Mismatched-pair relevance is not in it yet.
- Out-of-scope boundary: math and multi-hop sources, scored on calibration (uncertainty), not accuracy.

## Implementation

| Piece | Where |
|---|---|
| Per source: in the mix (`train:`), row filter (`train_keep:`), which augmentations apply (state `role:`), what a mismatched copy is asked (`asked:`, `on_mismatch:`), levels and grades | each source's template file in `data/templates/` |
| Shares, layouts, option mix | `data/build.py` (SHARES, LAYOUTS, OPTION_MIX) |
| Augmentations: reference layouts, mismatched pairs, structured instructions | `data/build.py` (`augment`), applied to every training variant |
| Item fields `grade`, `ref`, `instructions`, `parts` | `training/prepare_data.py` |
| Split by source rows; validation by type, grade, reference form and source | `training/finetune.py` |
| Loader and checks, preview | `data/templates.py`, `python -m data.preview` |

An augmented row's id lists every source row it holds (`<row>~<kind>~<other row>~...`); `row_parts`
reads it back.
