"""Compare finished runs and write a Markdown report.

    python -m benchmarks.report                                   # the two most recent runs
    python -m benchmarks.report --runs 20260922-1228-jev 20260922-1229-laya --source typed_decisions
    python -m benchmarks.report --out benchmarks/reports/jev-vs-dadvar.md

Reads each run's run.json, so it never calls a model. Runs are only comparable when they cover
the same source at the same dataset revision, sample size and seed; the report states those and
flags any mismatch.
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

from .models import MODELS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# (key, label, format, higher_is_better, in the default set)
METRICS = [("accuracy", "accuracy", "%.3f", True, True),
           ("brier", "brier", "%.3f", False, True),
           ("score_mae", "score MAE", "%.3f", False, True),
           ("soft_accuracy", "soft acc", "%.3f", True, False),
           ("log_loss", "log loss", "%.3f", False, False),
           ("kl", "KL", "%.3f", False, False),
           ("ece", "ECE", "%.3f", False, False),
           ("tv", "TV", "%.3f", False, False),
           ("within_one", "within 1", "%.3f", True, False)]


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


def billed_cost(run):
    rate = rate_of(run)
    if not rate:
        return "free"
    return "$%.4f" % (run.get("cost_usd") or 0.0)


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


def comparable(runs, source):
    infos = [r["results"][source]["info"] for r in runs]
    keys = {(i["revision"], i["n"], i["seed"], i["prompt_version"], tuple(i["configs"])) for i in infos}
    return len(keys) == 1, infos[0]


def report(runs, sources, title=None, all_metrics=False, groups_spec=None):
    chosen = metrics_for(all_metrics)
    names = [r["model_id"] for r in runs]

    def pooled(pick):
        """Per-model metrics over the (question count, metrics) parts `pick` returns for a run."""
        out = []
        for r in runs:
            parts = [(n, m) for n, m in pick(r)]
            out.append({key: sum(n * m[key] for n, m in parts if m.get(key) is not None) /
                        sum(n for n, m in parts if m.get(key) is not None)
                        for key, _, _, _, _ in chosen
                        if any(m.get(key) is not None for _, m in parts)})
            out[-1]["_n"] = sum(n for n, _ in parts)
        return out

    def by_suite(srcs):
        return pooled(lambda r: [(r["results"][s]["overall"]["questions"], r["results"][s]["overall"]) for s in srcs])

    def by_type(t, srcs=None):
        srcs = srcs if srcs is not None else sources
        return pooled(lambda r: [(r["results"][s]["by_type"][t]["questions"], r["results"][s]["by_type"][t])
                                 for s in srcs if t in r["results"][s]["by_type"]])

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

    def accuracy_table(label, rows):
        body = [[name, "%d" % per[0]["_n"]] +
                ["**%.3f**" % m["accuracy"] if m["accuracy"] == best([x["accuracy"] for x in per], True)
                 else "%.3f" % m["accuracy"] for m in per]
                for name, per in rows]
        return table(body, [label, "questions"] + names)

    def metric_table(label, rows):
        body = []
        for name, per in rows:
            for key, metric_label, fmt, higher, _ in chosen:
                values = [m.get(key) for m in per]
                if all(v is None for v in values):
                    continue
                top = best(values, higher)
                body.append([name if not body or body[-1][0] != name else "", metric_label] +
                            ["**%s**" % cell(v, fmt) if top is not None and v == top else cell(v, fmt)
                             for v in values])
        return table(body, [label, "metric"] + names)

    lines = ["# %s" % (title or "Benchmark comparison"), "",
             "Generated %s from %d run(s): %s." %
             (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(runs), ", ".join(names)), "",
             "## Accuracy", ""]
    for label, rows in groups.items():
        lines += ["### By %s" % label, "", accuracy_table(label, rows), ""]
    if groups_spec:
        lines += ["Suites: " + "; ".join("**%s** = %s" % (n, ", ".join(v)) for n, v in groups_spec.items()) + ".", ""]

    seen = {True: "yes", False: "no (stated)", None: "unknown"}
    lines += ["## Sources", "", table(
        [[src, dataset_link(runs[0]["results"][src]["info"]["dataset"]),
          revision_link(runs[0]["results"][src]["info"]["dataset"], runs[0]["results"][src]["info"]["revision"]),
          runs[0]["results"][src]["info"]["split"],
          "%d" % runs[0]["results"][src]["info"]["rows"],
          "%d" % runs[0]["results"][src]["overall"]["questions"],
          ", ".join(sorted({t for r in runs for t in r["results"][src]["by_type"]}))] +
         [seen[r["results"][src].get("seen_in_training")] for r in runs] for src in sources],
        ["source", "dataset", "revision", "split", "rows", "questions", "types"] +
        ["in %s training" % n.split("/", 1)[-1] for n in names]), "",
        "Seed %d, prompt v%d. Training columns: \"yes\" where the training run is documented, "
        "\"no (stated)\" where the publisher says the dataset was held out but the training data is "
        "not published, so it cannot be checked, and \"unknown\" where nothing is stated at all."
        % (runs[0]["results"][sources[0]]["info"]["seed"],
           runs[0]["results"][sources[0]]["info"]["prompt_version"]), ""]

    jev = [r for r in runs if rate_of(r)]
    if jev:
        lines += ["---", "",
                  "Hosted model cost for these answers: %s." %
                  ", ".join("%s %s" % (r["model_id"], answer_cost(r)) for r in jev), ""]

    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", help="run directories (default: the two most recent)")
    ap.add_argument("--source", nargs="+", help="sources to report (default: those every run shares)")
    ap.add_argument("--title")
    ap.add_argument("--group", action="append", metavar="NAME=SRC,SRC",
                    help="pool sources into a named suite, e.g. composite=massive_scenario,helpsteer2")
    ap.add_argument("--all-metrics", action="store_true",
                    help="also report soft accuracy, log loss, KL, TV, ECE, within-1")
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

    groups_spec = {}
    for spec in args.group or []:
        name, _, srcs = spec.partition("=")
        groups_spec[name] = [x for x in srcs.split(",") if x]
        unknown = [x for x in groups_spec[name] if x not in sources]
        if unknown:
            ap.error("--group %s: not in the reported sources: %s" % (name, ", ".join(unknown)))
    text = report(runs, sources, args.title, args.all_metrics, groups_spec)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print("wrote %s" % os.path.relpath(args.out))
    else:
        print(text)


if __name__ == "__main__":
    main()
