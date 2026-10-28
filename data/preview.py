"""Print a source's questions as the model gets them: real rows, the full request, and the options as
rendered into the model's input.

    python -m data.preview amazon_reviews                       # 2 benchmark rows, canonical questions
    python -m data.preview ethics --n 1 --config deontology     # one config of a dataset
    python -m data.preview banking77 --split train --variant 1 --n 4   # training rows, augmented

Rows are built as the benchmark or training builder builds them (a small seeded sample, cached like
any other build).
"""

import argparse
import json

from dadvar.text import option_texts

from . import SOURCES, Split, build


def show(row, gold_too=True):
    state, questions = json.loads(row["state"]), json.loads(row["questions"])
    gold, meta = json.loads(row["gold"]), json.loads(row.get("meta") or "{}")
    print("=" * 100)
    print(row["id"])
    print("state:")
    print(json.dumps(state, indent=2, ensure_ascii=False))
    for qid, q in questions.items():
        m = meta.get(qid, {})
        tags = ", ".join("%s %s" % (k, m[k]) for k in ("kind", "level", "ref") if m.get(k) not in (None, "original"))
        print("\n  %s  (%s%s)" % (qid, q["type"], ", " + tags if tags else ""))
        ins = q["instructions"]
        print("    instructions: %s" % (ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)))
        internal = {"t": q["type"], "ins": "", "crit": q.get("criteria")}
        for opt in option_texts(internal):
            print("      [MASK] %s" % opt)
        if gold_too:
            g = gold[qid]
            if g["type"] == "noul":
                print("    gold: noul %.2f" % g["noul"])
            elif g["type"] == "choice":
                print("    gold: %s" % g["choice"])
            else:
                probs = {k: round(v, 2) for k, v in g["probabilities"].items() if v}
                print("    gold: level %s  %s" % (g["score"], probs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=sorted(SOURCES))
    ap.add_argument("--n", type=int, default=2, help="rows to show")
    ap.add_argument("--split", choices=["eval", "train"], default="eval")
    ap.add_argument("--variant", type=int, default=0, help="0: the canonical questions; 1, 2, ...: training variants")
    ap.add_argument("--config", help="only this config of the dataset (e.g. ethics' deontology)")
    ap.add_argument("--no-gold", action="store_true")
    args = ap.parse_args(argv)

    configs = [args.config] if args.config else None
    ds, info = build(args.source, Split(args.split), n=max(args.n, 20), configs=configs, variant=args.variant)
    rows = list(ds)
    # augmented rows come after the plain ones; show some of each
    plain = [r for r in rows if "~" not in r["id"]]
    extra = [r for r in rows if r not in plain]
    pick = plain[: args.n] if not extra else plain[: (args.n + 1) // 2] + extra[: args.n // 2]
    print(
        "%s: %s @ %s, %s split, variant %d"
        % (args.source, info["dataset"], info["revision"][:8], info["split"], args.variant)
    )
    for r in pick:
        show(r, not args.no_gold)


if __name__ == "__main__":
    main()
