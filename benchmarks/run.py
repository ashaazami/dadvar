"""Run benchmark sources through a model and score the answers.

    python -m benchmarks.run banking77
    python -m benchmarks.run typed_decisions boolq --n 200
    python -m benchmarks.run massive_intent --langs en de ja --subfolder multilingual
    python -m benchmarks.run all
    python -m benchmarks.run banking77 --backend jev          # hosted TypeSafe model, billed

Answers are cached in benchmarks/cache/predictions.sqlite, keyed by the request and the model,
so a rerun only calls the model for rows it has not seen (`--no-cache` to bypass, e.g. when
measuring latency).

Writes benchmarks/results/<timestamp>-<model>/run.json (settings, dataset revisions, metrics,
latency) and predictions.jsonl (per row: id, latency, answers, per-question metrics). Neither
file contains dataset text.
"""

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pydantic import TypeAdapter  # noqa: E402
from typesafe_sdk import Answer  # noqa: E402

from data.build import DEFAULT_N, DEFAULT_SEED, Split, build  # noqa: E402
from data.sources import MASSIVE_DEFAULT_LANGS, SOURCES  # noqa: E402

from . import cache, metrics  # noqa: E402
from .models import MODELS, load_model  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

_ANSWER = TypeAdapter(Answer)


def parse_answers(response):
    return {qid: _ANSWER.validate_json(json.dumps(a)) for qid, a in response["answers"].items()}


def seen_in_training(name, model_id):
    """True / False if known for this model, None if unknown."""
    return SOURCES[name]["seen_by"].get(model_id)


def run_source(model, name, args, out, db):
    configs = None
    if name.startswith("massive_") and args.langs:
        configs = _massive_langs(name) if args.langs == ["all"] else args.langs
    ds, info = build(name, Split.EVAL, n=args.n_by_source.get(name, args.n_default),
                     seed=args.seed, configs=configs)
    print("\n%s: %d rows (%s @ %s)" % (name, len(ds), info["dataset"], info["revision"][:8]), flush=True)

    first = ds[0]
    for _ in range(model.warmup if db is None else 0):   # a warm-up call would be served from cache
        model.system_one(json.loads(first["state"]), json.loads(first["questions"]))

    scored, latencies, input_tokens, billed_tokens, reported = [], [], 0, 0, set()
    hits, cached_latencies = 0, []
    for i, r in enumerate(ds):
        state, questions, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
        gold = {qid: _ANSWER.validate_json(json.dumps(g)) for qid, g in gold.items()}
        req_hash = cache.request_hash(state, questions)
        key = cache.prediction_key(model.model_id, model.model_version, model.settings, req_hash)
        stored = cache.get(db, key) if db is not None else None
        error, ms = None, None
        if stored is not None:
            hits += 1
            response = json.loads(stored["response_json"])
            answers = parse_answers(response)
            cached_latencies.append(stored["latency_ms"])
        else:
            t = time.perf_counter()
            try:
                response = model.system_one(state, questions)
                answers = parse_answers(response)
                ms = (time.perf_counter() - t) * 1000
                latencies.append(ms)
                if db is not None:
                    cache.put(db, key, req_hash, state, questions, model, response, ms)
            except Exception as e:  # a failure is scored as wrong, not skipped, and never cached
                answers, error, response = {}, "%s: %s" % (type(e).__name__, e), {}
        tokens = (response.get("usage") or {}).get("input_tokens") or 0
        input_tokens += tokens
        billed_tokens += 0 if stored is not None else tokens
        reported.add(response.get("model"))
        row_scores = {}
        for qid, q in questions.items():
            s = metrics.score_question(q, gold[qid], answers.get(qid))
            s.update(source=r["source"], question=qid)
            scored.append(s)
            row_scores[qid] = {k: s[k] for k in ("correct", "confidence", "brier", "log_loss")}
        out.write(json.dumps({"id": r["id"], "source": r["source"], "request_hash": req_hash,
                              "ms": round(ms, 2) if ms is not None else None, "cached": stored is not None,
                              "error": error,
                              "answers": {q: a.model_dump(mode="json", exclude={"legend"}) for q, a in answers.items()},
                              "scores": row_scores}) + "\n")
        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, len(ds)), flush=True)

    result = {"info": info, "seen_in_training": seen_in_training(name, model.model_id),
              "input_tokens": input_tokens, "billed_tokens": billed_tokens,
              "cost_usd": model.usage_cost(billed_tokens),
              "reported_model": sorted(x for x in reported if x),
              "overall": metrics.summarize(scored),
              "by_source": metrics.summarize_by(scored, "source"),
              "by_type": metrics.summarize_by(scored, "type"),
              "by_question": metrics.summarize_by(
                  [dict(s, key="%s/%s" % (s["source"], s["question"])) for s in scored], "key"),
              "cache": {"hits": hits, "calls": len(ds) - hits},
              "latency_ms": _latency(latencies, cached_latencies)}
    return result


def _latency(fresh, cached):
    """Timings of the calls actually made; falls back to the cached ones, flagged as such."""
    values = fresh or [v for v in cached if v is not None]
    if not values:
        return {"p50": math.nan, "p95": math.nan, "mean": math.nan, "rows": 0, "from_cache": bool(cached)}
    return {"p50": metrics.percentile(values, 50), "p95": metrics.percentile(values, 95),
            "mean": sum(values) / len(values), "rows": len(values), "from_cache": not fresh}


def _massive_langs(name):
    from datasets import get_dataset_config_names
    return [c for c in get_dataset_config_names(SOURCES[name]["dataset"]) if c != "default"]


def print_table(results):
    print("\n%-34s %6s %8s %7s %8s %6s %9s %8s  %s" % (
        "source", "qs", "accuracy", "brier", "log_loss", "ece", "score_mae", "p50 ms", "seen in training"))
    for name, res in results.items():
        seen = {True: "yes", False: "no", None: "unknown"}[res["seen_in_training"]]
        rows = [(name, res["overall"])]
        if len(res["by_source"]) > 1:
            rows += [("  " + k, v) for k, v in res["by_source"].items()]
        for label, m in rows:
            mae = "%.3f" % m["score_mae"] if "score_mae" in m else "-"
            top = label == name
            p50 = "%.1f" % res["latency_ms"]["p50"] if top else ""
            print("%-34s %6d %8.3f %7.3f %8.3f %6.3f %9s %8s  %s" % (
                label[:34], m["questions"], m["accuracy"], m["brier"], m["log_loss"], m["ece"], mae, p50,
                seen if top else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="source names, or 'all': " + ", ".join(SOURCES))
    ap.add_argument("--n", nargs="+", default=[str(DEFAULT_N)], metavar="N|SOURCE=N",
                    help="rows per source (per language for MASSIVE); per-source overrides like helpsteer2=100")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--langs", nargs="+", help="MASSIVE languages, or 'all' (default: %s)" % " ".join(MASSIVE_DEFAULT_LANGS))
    ap.add_argument("--backend", default="dadvar", choices=sorted(MODELS))
    ap.add_argument("--model", help="dadvar: hub id or directory (default convaiinnovations/laya); jev: model version")
    ap.add_argument("--subfolder", help="dadvar: checkpoint inside the model repo, e.g. multilingual")
    ap.add_argument("--device", help="dadvar: cpu, mps or cuda (default: best available)")
    ap.add_argument("--no-cache", action="store_true", help="always call the model; do not read or write the cache")
    args = ap.parse_args(argv)
    args.n_default, args.n_by_source = DEFAULT_N, {}
    for spec in args.n:
        source, _, count = spec.rpartition("=")
        if source and source not in SOURCES:
            ap.error("unknown source in --n: %s" % source)
        if source:
            args.n_by_source[source] = int(count)
        else:
            args.n_default = int(count)

    names = list(SOURCES) if args.sources == ["all"] else args.sources
    unknown = [s for s in names if s not in SOURCES]
    if unknown:
        ap.error("unknown source(s): %s" % ", ".join(unknown))

    import platform as _platform

    t = time.perf_counter()
    model = load_model(args.backend, args.model, subfolder=args.subfolder, device=args.device)
    load_s = time.perf_counter() - t
    model_label = model.model_id.split("/")[-1] + ("-" + args.subfolder if args.subfolder else "")
    run_dir = os.path.join(RESULTS_DIR, "%s-%s" % (datetime.now().strftime("%Y%m%d-%H%M%S"), model_label))
    os.makedirs(run_dir)

    db = None if args.no_cache else cache.open_db()
    results = {}
    try:
        with open(os.path.join(run_dir, "predictions.jsonl"), "w") as out:
            for name in names:
                results[name] = run_source(model, name, args, out, db)
    finally:
        model.close()
        if db is not None:
            db.close()

    total_tokens = sum(r["input_tokens"] for r in results.values())
    billed = sum(r["billed_tokens"] for r in results.values())
    run = {"created_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
           "backend": model.backend, "model_id": model.model_id, "model_version": model.model_version,
           "device": model.device, "settings": model.settings,
           "cost_per_input_token": model.cost_per_input_token,
           "load_seconds": round(load_s, 2),
           "input_tokens": total_tokens, "billed_tokens": billed, "cost_usd": model.usage_cost(billed),
           "cache": {"enabled": not args.no_cache,
                     "hits": sum(r["cache"]["hits"] for r in results.values()),
                     "calls": sum(r["cache"]["calls"] for r in results.values())},
           "environment": {"python": platform.python_version(), "machine": platform.machine(),
                           "platform": platform.platform()},
           "results": results}
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=1)
    print_table(results)
    c = run["cache"]
    print("\ncache: %d served, %d called%s" % (c["hits"], c["calls"], "" if c["enabled"] else " (disabled)"))
    if run["cost_usd"] is not None:
        print("%s tokens: %d (%d billed this run)  cost: $%.4f"
              % (model.model_id, total_tokens, billed, run["cost_usd"]))
    mismatch = {m for r in results.values() for m in r["reported_model"]} - {model.model_version, None}
    if mismatch and model.backend != "dadvar":
        print("WARNING: answers came from %s, not the pinned %s" % (sorted(mismatch), model.model_version))
    print("\nresults: %s" % os.path.relpath(run_dir))


if __name__ == "__main__":
    main()
