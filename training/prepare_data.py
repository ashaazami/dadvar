# Adapted in part from Laya (https://github.com/NandhaKishorM/laya) at 4066d5d, Apache-2.0:
# notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb (the training item format and build_training_item).
# Modified for Dadvar: whole sequences only, many sources, variants and item metadata. See NOTICE.
"""Build fine-tuning items from one or more sources in `data/`.

One item per question: token ids, option marker positions, question type, and the gold
probability distribution as a soft target.

    TOK="--tokenizer answerdotai/ModernBERT-base --max-len 1024"

    python -m training.prepare_data --out training/data/items.pt typed_decisions $TOK
    python -m training.prepare_data --out /tmp/items.pt typed_decisions --n 4 $TOK      # smoke test
    python -m training.prepare_data --out training/data/v1_full1024.pt v1 --langs en $TOK \\
        --n 12000 --variants 3 --p-opposite 0.5 --per-source 30000      # dataset v1 (docs/DATASET.md)

Rows come from each source's training split, never from the split its benchmark runs on, and a
source whose license does not allow training is refused (its template file's `use:`).

`--commercial` builds only sources whose template file says `commercial: true`, for a model that will
be sold or served commercially: 'all' and 'v1' are narrowed to them, a named source that is not
cleared is refused. The provenance records `"commercial": true` whenever every source built is cleared.

Every item holds the whole sequence: a question longer than `--max-len` is skipped, never cut, so
the model never trains on an input with something missing.
"""

import argparse
import json
import os
import random
import re

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from dadvar.text import QTYPES, option_texts  # noqa: E402
from dadvar.sequence import assemble, encode_segments, fits  # noqa: E402
from data import DEFAULT_N, DEFAULT_SEED, SOURCES, Split, build, target_distribution  # noqa: E402
from data.build import row_parts  # noqa: E402
from data.templates import grade, roles, train_sources  # noqa: E402


def as_text(ins):
    """Instructions may be an object or an array (TypeSafe allows it); the model reads them as JSON."""
    return ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)


def build_training_item(tok, max_len, state, q, gold_q):
    """One training sequence: ids, markers, question type, soft target, argmax label.

    None when the question cannot be used: its whole sequence is longer than `max_len`, or it has
    fewer markers than options.
    """
    internal = {"t": q["type"], "ins": as_text(q["instructions"]), "crit": q.get("criteria")}
    target = target_distribution(q, gold_q)
    segments = encode_segments(tok, state, internal)
    if not fits(segments, max_len):
        return None
    seq, markers = assemble(tok, segments)
    if len(markers) != len(option_texts(internal)):
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[q["type"]],
        "target": target,
        "label": target.index(max(target)),
    }


def reference_form(instructions):
    """How a question points into the state: none, one field (top), or two or more (pair)."""
    n = len(set(re.findall(r"`([^`]+)`", as_text(instructions))))
    return "none" if n == 0 else "top" if n == 1 else "pair"


def varies(name):
    """Whether training variants of this source differ at all: a second wording, an opposite to add,
    or an augmentation the variant seeds (its state claims a `text` or `context` role). A source whose
    questions come with the data (`rows:`) and whose state claims no role has nothing to vary."""
    if any(len(t.wordings) > 1 or t.opposites for t in SOURCES[name].get("templates", ())):
        return True
    role = roles(name)
    return bool(role.get("text") or role.get("context"))


def many_options(name, configs, args):
    """Whether a source asks choice questions with more options than training usually keeps (5)."""
    ds, _ = build(name, Split.TRAIN, n=min(args.n, 20), seed=args.seed, configs=configs, eval_n=args.eval_n)
    return any(
        q["type"] == "choice" and len(q["criteria"]) > 5 for r in ds for q in json.loads(r["questions"]).values()
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "sources",
        nargs="+",
        help="source names, 'all' for every source licensed for training, or 'v1' for the "
        "training mix (every source not marked `train: false`): " + ", ".join(SOURCES),
    )
    ap.add_argument("--out", required=True, help="output .pt file")
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="hub id or directory of the tokenizer to encode with, e.g. answerdotai/ModernBERT-base",
    )
    ap.add_argument("--revision", help="hub commit, branch or tag of that tokenizer (default: main)")
    ap.add_argument("--max-len", type=int, required=True, help="token budget; a longer question is skipped")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="rows per source (per language for MASSIVE)")
    ap.add_argument(
        "--variants",
        type=int,
        default=1,
        help="templated sources: build this many training variants (seeded wordings and yes/no opposites) "
        "and keep all of them; other sources are built once",
    )
    ap.add_argument("--p-opposite", type=float, default=0.5, help="chance a yes/no question also gets its opposite")
    ap.add_argument(
        "--per-source",
        type=int,
        help="keep about this many items per source, a seeded sample of whole rows, and drop the rest. "
        "Usually leave it out and balance in training instead (training.finetune --per-source), "
        "which draws a new sample each epoch and so drops nothing",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--eval-n",
        type=int,
        default=DEFAULT_N,
        help="evaluation sample size to hold out where train and eval share a split",
    )
    ap.add_argument("--langs", nargs="+", help="MASSIVE languages (default: the source's own list)")
    ap.add_argument(
        "--commercial",
        action="store_true",
        help="only sources cleared for a commercially sold model (`commercial: true`)",
    )
    args = ap.parse_args(argv)

    names = (
        [n for n in SOURCES if SOURCES[n]["use"].get("train")]
        if args.sources == ["all"]
        else list(train_sources()) if args.sources == ["v1"] else args.sources
    )
    unknown = [s for s in names if s not in SOURCES]
    if unknown:
        ap.error("unknown source(s): %s" % ", ".join(unknown))
    if args.commercial:
        not_cleared = [n for n in names if not SOURCES[n]["commercial"]]
        if args.sources in (["all"], ["v1"]):
            names = [n for n in names if SOURCES[n]["commercial"]]
            print("--commercial: leaving out %s" % ", ".join(not_cleared))
        elif not_cleared:
            ap.error("not cleared for commercial use (`commercial: false`): %s" % ", ".join(not_cleared))

    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)

    items, provenance = [], {}
    for name in names:
        configs = args.langs if (name.startswith("massive_") and args.langs) else None
        # Variants differ where there is something to vary: a source's wordings and opposites, or a
        # large option set's subset. Other sources are built once, as variant 1 (options shuffled),
        # so they are not simply repeated.
        variants = range(1, args.variants + 1) if (varies(name) or many_options(name, configs, args)) else [1]
        before, skipped = len(items), 0
        for variant in variants:
            ds, info = build(
                name,
                Split.TRAIN,
                n=args.n,
                seed=args.seed,
                configs=configs,
                eval_n=args.eval_n,
                variant=variant,
                p_opposite=args.p_opposite,
            )
            for row in ds:
                state = json.loads(row["state"])
                questions = json.loads(row["questions"])
                gold = json.loads(row["gold"])
                meta = json.loads(row["meta"])
                match = bool(SOURCES[name].get("oversample") and SOURCES[name]["oversample"][0](gold))
                for qid, q in questions.items():
                    it = build_training_item(tok, args.max_len, state, q, gold[qid])
                    if it:
                        m = meta.get(qid, {})
                        # source: for balancing and validation per source; the rest for metrics
                        it.update(
                            source=name,
                            row=row["id"],
                            question=qid,
                            variant=variant,
                            oversample_match=match,
                            level=m.get("level"),
                            kind=m.get("kind", "original"),
                            template=m.get("template", qid),
                            form=m.get("form", "question"),
                        )
                        # the grade, how the question points into the state, and every source row
                        # whose text the state holds (the training split keeps them together)
                        it.update(
                            grade=grade(name, it["template"]),
                            ref=m.get("ref") or reference_form(q["instructions"]),
                            instructions="object" if isinstance(q["instructions"], dict) else "string",
                            parts=row_parts(row["id"]),
                        )
                        items.append(it)
                    else:
                        skipped += 1
        built = len(items) - before
        if SOURCES[name].get("oversample") and built:
            # repeat the items of rows matching the predicate until they make up `share` of this
            # source's items (training only; the benchmark keeps the natural rate)
            share = SOURCES[name]["oversample"][1]
            match = [it for it in items[before:] if it.get("oversample_match")]
            rng = random.Random(args.seed)
            while (
                match
                and sum(it.get("oversample_match", False) for it in items[before:]) / (len(items) - before) < share
            ):
                items.append(dict(rng.choice(match), kind="oversampled"))
        built = len(items) - before
        if args.per_source and built > args.per_source:
            # keep whole (row, variant) groups: a state's questions, their opposites and the
            # partner texts an augmentation placed beside them stay together, so the contrast
            # between them survives the cap
            groups = {}
            for i in range(before, len(items)):
                groups.setdefault((items[i]["row"], items[i]["variant"]), []).append(i)
            order = list(groups.values())
            random.Random(args.seed).shuffle(order)
            keep, total = [], 0
            for g in order:
                if total >= args.per_source:
                    break
                keep += g
                total += len(g)
            items[before:] = [items[i] for i in sorted(keep)]
        provenance[name] = {
            "rows": len(ds),
            "items": len(items) - before,
            "built": built,
            "skipped": skipped,
            "dataset": info["dataset"],
            "revision": info["revision"],
            "split": info["split"],
            "configs": info["configs"],
        }
        print(
            "%-20s %5d rows -> %5d items (%d skipped%s)"
            % (
                name,
                len(ds),
                built,
                skipped,
                ", kept %d" % (len(items) - before) if built != len(items) - before else "",
            )
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(items, args.out)
    with open(os.path.splitext(args.out)[0] + "_provenance.json", "w") as f:
        json.dump(
            {
                "tokenizer": args.tokenizer,
                "revision": args.revision,
                "recipe": "v1",
                "n": args.n,
                "per_source": args.per_source,
                "variants": args.variants,
                "p_opposite": args.p_opposite,
                "seed": args.seed,
                "eval_n": args.eval_n,
                "max_len": args.max_len,
                "commercial": all(SOURCES[n]["commercial"] for n in names),
                "items": len(items),
                "sources": provenance,
            },
            f,
            indent=1,
        )
    print("%d items -> %s" % (len(items), args.out))


if __name__ == "__main__":
    main()
