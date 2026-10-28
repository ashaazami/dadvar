"""Compare finished runs and write a Markdown report.

    python -m benchmarks.report                                   # the two most recent runs
    python -m benchmarks.report --runs 20260922-1228-jev 20260922-1229-dadvar --source typed_decisions
    python -m benchmarks.report --out jev-vs-dadvar.md

Reads each run's run.json and predictions.jsonl, so it never calls a model. Runs are only
comparable when they cover the same source at the same dataset revision, sample size, seed, prompt
version, configs and option count; the report refuses runs that differ, and states the settings.

A local model skips a row with a question longer than its token budget rather than cut it, so no
answer comes from a cut input. By default a row any run under comparison skipped is dropped from every run, so all of them are
scored on the identical set. `--include-skipped` instead scores each run on every row it
answered.
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

from . import metrics as _metrics
from .models import MODELS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# (key, label, format, higher_is_better, in the default set)
METRICS = [
    ("accuracy", "accuracy", "%.3f", True, True),
    ("brier", "brier", "%.3f", False, True),
    ("score_mae", "score MAE", "%.3f", False, True),
    ("soft_accuracy", "soft acc", "%.3f", True, False),
    ("log_loss", "log loss", "%.3f", False, False),
    ("kl", "KL", "%.3f", False, False),
    ("ece", "ECE", "%.3f", False, False),
    ("tv", "TV", "%.3f", False, False),
    ("within_one", "within 1", "%.3f", True, False),
]


def metrics_for(all_metrics):
    return [m for m in METRICS if all_metrics or m[4]]


def load_runs(names):
    runs = []
    for name in names:
        path = name if os.path.isdir(name) else os.path.join(RESULTS_DIR, name)
        with open(os.path.join(path, "run.json")) as f:
            run = json.load(f)
        run["_dir"] = os.path.basename(path.rstrip("/"))
        runs.append(run)
    return runs


HF_DATASET = "https://huggingface.co/datasets/%s"


def dataset_link(dataset):
    return "[`%s`](%s)" % (dataset, HF_DATASET % dataset)


def revision_link(dataset, revision):
    return "[`%s`](%s/tree/%s)" % (revision[:8], HF_DATASET % dataset, revision)


def rate_of(run):
    """USD per input token for the model a run used, or None when running it is free."""
    if run.get("cost_per_input_token") is not None:
        return run["cost_per_input_token"]
    if run.get("cost_usd") and run.get("billed_tokens"):
        return run["cost_usd"] / run["billed_tokens"]
    model = MODELS.get(run["backend"])
    return getattr(model, "cost_per_input_token", None)


def answer_cost(run):
    rate = rate_of(run)
    return "free" if not rate else "$%.4f" % (run["input_tokens"] * rate)


def cell(value, fmt="%.3f"):
    return "-" if value is None else fmt % value


def best(values, higher_is_better):
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return None
    return (max if higher_is_better else min)(present)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def read_predictions(run):
    path = os.path.join(RESULTS_DIR, run["_dir"], "predictions.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def restrict_to_common_rows(runs):
    """Re-score every run over the rows none of them skipped. Returns (runs, dropped).

    The run's own totals cover every row it attempted, so they cannot be filtered after the
    fact -- the per-question scores in predictions.jsonl are re-aggregated instead, which is
    why a run has to carry the whole score dict rather than a readable subset of it.
    """
    rows = {r["_dir"]: read_predictions(r) for r in runs}
    dropped = {row["id"] for rs in rows.values() for row in rs if row.get("skipped")}
    if not dropped:
        return runs, dropped

    for run in runs:
        scored = {}
        for row in rows[run["_dir"]]:
            if row["id"] in dropped:
                continue
            for qid, s in row["scores"].items():
                scored.setdefault(row["source"].split("/")[0], []).append(dict(s, source=row["source"], question=qid))
        for name, part in scored.items():
            res = run["results"][name]
            res["overall"] = _metrics.summarize(part)
            res["by_type"] = _metrics.summarize_by(part, "type")
            res["by_source"] = _metrics.summarize_by(part, "source")
        for name in list(run["results"]):
            if name not in scored:  # every row of this source was skipped
                run["results"][name]["overall"] = {"questions": 0}
                run["results"][name]["by_type"] = {}
    return runs, dropped


COMPARED = ("revision", "n", "seed", "prompt_version", "configs", "options")


def mismatch(runs, source):
    """The settings the runs disagree on for `source` (dataset revision, sample, seed, prompt
    version, configs, option count); empty when they are comparable."""
    infos = [r["results"][source]["info"] for r in runs]
    return [k for k in COMPARED if len({repr(i.get(k)) for i in infos}) > 1]


def report(runs, sources, title=None, all_metrics=False, groups_spec=None, by_model=False, dropped=()):
    chosen = metrics_for(all_metrics)
    # one column per run; two runs of the same model are numbered so their columns stay apart
    ids = [r["model_id"] for r in runs]
    names = [m + (" (%d)" % (ids[:i].count(m) + 1) if ids.count(m) > 1 else "") for i, m in enumerate(ids)]

    def pooled(pick):
        """Per-model metrics over the (question count, metrics) parts `pick` returns for a run."""
        out = []
        for r in runs:
            parts = [(n, m) for n, m in pick(r)]
            out.append(
                {
                    key: sum(n * m[key] for n, m in parts if m.get(key) is not None)
                    / sum(n for n, m in parts if m.get(key) is not None)
                    for key, _, _, _, _ in chosen
                    if any(m.get(key) is not None for _, m in parts)
                }
            )
            out[-1]["_n"] = sum(n for n, _ in parts)
        return out

    def by_suite(srcs):
        return pooled(lambda r: [(r["results"][s]["overall"]["questions"], r["results"][s]["overall"]) for s in srcs])

    def by_type(t, srcs=None):
        srcs = srcs if srcs is not None else sources
        return pooled(
            lambda r: [
                (r["results"][s]["by_type"][t]["questions"], r["results"][s]["by_type"][t])
                for s in srcs
                if t in r["results"][s]["by_type"]
            ]
        )

    types = sorted({t for r in runs for s in sources for t in r["results"][s]["by_type"]})
    groups = {}
    if groups_spec:
        groups["suite"] = [(name, by_suite(srcs)) for name, srcs in groups_spec.items()]
        rows = []
        for name, srcs in groups_spec.items():
            for t in types:
                if any(t in r["results"][s]["by_type"] for r in runs for s in srcs):
                    rows.append(("%s / %s" % (name, t), by_type(t, srcs)))
        groups["suite and type"] = rows
    elif types:
        groups["question type"] = [(t, by_type(t)) for t in types]

    def model_tables():
        """One table per suite: model down the side, split into a row per question type."""
        out = []
        for g, srcs in groups_spec.items():
            present = [t for t in types if any(t in r["results"][s]["by_type"] for r in runs for s in srcs)]
            # a suite spanning several types leads with its overall row; one type needs no total
            rows = [("**overall**", by_suite(srcs), True)] if len(present) > 1 else []
            rows += [(t, by_type(t, srcs), False) for t in present]
            body = []
            for i, model in enumerate(names):
                for j, (label, per, overall) in enumerate(rows):
                    values = [m.get("accuracy") for m in per]
                    top = best(values, True)
                    bold = values[i] is not None and (overall or values[i] == top)  # summary row always bold
                    body.append(
                        [
                            "**%s**" % model if j == 0 else "",
                            label,
                            "%d" % per[i]["_n"],
                            ("**%s**" if bold else "%s") % cell(values[i]),
                        ]
                    )
            out += ["### %s" % g, "", table(body, ["model", "type", "questions", "accuracy"]), ""]
        return out

    def accuracy_table(label, rows):
        # with --include-skipped the runs can have answered different numbers of questions; then
        # each cell carries its own count, and a run that answered none shows "-"
        body = []
        for name, per in rows:
            counts = {m["_n"] for m in per}
            top = best([m.get("accuracy") for m in per], True)
            cells = []
            for m in per:
                v = cell(m.get("accuracy"))
                v = "**%s**" % v if top is not None and m.get("accuracy") == top else v
                cells.append(v if len(counts) == 1 else "%s (%d)" % (v, m["_n"]))
            body.append([name, "%d" % per[0]["_n"] if len(counts) == 1 else "varies"] + cells)
        return table(body, [label, "questions"] + names)

    def metric_table(label, rows):
        body = []
        for name, per in rows:
            first = True  # the group's name on its first metric row only
            for key, metric_label, fmt, higher, _ in chosen:
                values = [m.get(key) for m in per]
                if all(v is None for v in values):
                    continue
                top = best(values, higher)
                body.append(
                    [name if first else "", metric_label]
                    + ["**%s**" % cell(v, fmt) if top is not None and v == top else cell(v, fmt) for v in values]
                )
                first = False
        return table(body, [label, "metric"] + names)

    lines = [
        "# %s" % (title or "Benchmark comparison"),
        "",
        "Generated %s from %d run(s): %s."
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(runs), ", ".join(names)),
        "",
        "## Accuracy",
        "",
    ]
    if dropped:
        # the kept rows are the intersection of what every model could be shown in full, so
        # each one is scored on the identical set rather than on its own best case
        lines += [
            "Scored on the rows every model answered: **%d excluded** because at least one "
            "run skipped them, a question being longer than its model's token budget. "
            "`--include-skipped` scores each run on every row it answered." % len(dropped),
            "",
        ]
    if by_model and groups_spec:
        lines += model_tables()
    else:
        for label, rows in groups.items():
            lines += ["### By %s" % label, "", accuracy_table(label, rows), ""]
    lines += ["## Metrics", ""]
    for label, rows in groups.items():
        lines += ["### By %s" % label, "", metric_table(label, rows), ""]
    pooled_specs = {n: v for n, v in (groups_spec or {}).items() if v != [n]}
    if pooled_specs:
        lines += ["Suites: " + "; ".join("**%s** = %s" % (n, ", ".join(v)) for n, v in pooled_specs.items()) + ".", ""]

    seen = {True: "yes", False: "no", None: "unknown"}
    lines += [
        "## Sources",
        "",
        table(
            [
                [
                    src,
                    dataset_link(runs[0]["results"][src]["info"]["dataset"]),
                    revision_link(
                        runs[0]["results"][src]["info"]["dataset"], runs[0]["results"][src]["info"]["revision"]
                    ),
                    runs[0]["results"][src]["info"]["split"],
                    "%d" % runs[0]["results"][src]["info"]["rows"],
                    "%d" % runs[0]["results"][src]["overall"]["questions"],
                    ", ".join(sorted({t for r in runs for t in r["results"][src]["by_type"]})),
                ]
                + [seen[r["results"][src].get("seen_in_training")] for r in runs]
                for src in sources
            ],
            ["source", "dataset", "revision", "split", "rows", "questions", "types"]
            + ["in %s training" % n.split("/", 1)[-1] for n in names],
        ),
        "",
        'Seed %d, prompt v%d. Training columns: "yes" or "no" from the model\'s own record of '
        'what it trained on (a model trained here records it), "unknown" where the model states nothing.'
        % (runs[0]["results"][sources[0]]["info"]["seed"], runs[0]["results"][sources[0]]["info"]["prompt_version"]),
        "",
    ]

    jev = [r for r in runs if rate_of(r)]
    if jev:
        lines += [
            "---",
            "",
            "Hosted model cost for these answers: %s."
            % ", ".join("%s %s" % (r["model_id"], answer_cost(r)) for r in jev),
            "",
        ]

    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", help="run directories (default: the two most recent)")
    ap.add_argument("--source", nargs="+", help="sources to report (default: those every run shares)")
    ap.add_argument("--title")
    ap.add_argument(
        "--group",
        action="append",
        metavar="NAME=SRC,SRC",
        help="pool sources into a named suite, e.g. composite=massive_scenario,helpsteer2",
    )
    ap.add_argument(
        "--all-metrics", action="store_true", help="also report soft accuracy, log loss, KL, TV, ECE, within-1"
    )
    ap.add_argument(
        "--by-model",
        action="store_true",
        help="transpose: model down the side, one row per question type, suites across the top",
    )
    ap.add_argument(
        "--include-skipped",
        action="store_true",
        help="score each run on every row it answered (default: drop a row any run skipped " "from all of them)",
    )
    ap.add_argument("--out", help="write Markdown here (default: print)")
    args = ap.parse_args(argv)

    names = args.runs or [os.path.basename(p) for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "*")))[-2:]]
    runs = load_runs(names)
    if not runs:
        ap.error("no runs found in %s" % RESULTS_DIR)
    shared = set(runs[0]["results"])
    for r in runs[1:]:
        shared &= set(r["results"])
    sources = args.source or sorted(shared)
    missing = [s for s in sources if s not in shared]
    if missing:
        ap.error("not every run covers: %s" % ", ".join(missing))
    differ = {s: mismatch(runs, s) for s in sources}
    differ = {s: d for s, d in differ.items() if d}
    if differ:
        ap.error(
            "runs are not comparable: " + "; ".join("%s differs in %s" % (s, ", ".join(d)) for s, d in differ.items())
        )

    groups_spec = {}
    for spec in args.group or []:
        name, _, srcs = spec.partition("=")
        groups_spec[name] = [x for x in srcs.split(",") if x]
        unknown = [x for x in groups_spec[name] if x not in sources]
        if unknown:
            ap.error("--group %s: not in the reported sources: %s" % (name, ", ".join(unknown)))
    dropped = set()
    if not args.include_skipped:
        runs, dropped = restrict_to_common_rows(runs)
    text = report(runs, sources, args.title, args.all_metrics, groups_spec, args.by_model, dropped)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print("wrote %s" % os.path.relpath(args.out))
    else:
        print(text)


if __name__ == "__main__":
    main()
