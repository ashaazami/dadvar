"""Store every model answer so no request is ever paid for twice.

A prediction is keyed by *what was asked* and *which model answered it*:

    request_hash    = sha256(canonical_json({"state": ..., "questions": ...}))
    prediction_key  = sha256(model_id + model_version + settings + request_hash)

So a rerun with a different sample size, seed or source mix still reuses every row it has
already seen, and a new model version never reuses another version's answers.

Canonical JSON normalizes only what the model never sees: whitespace and escaping. Key order
is preserved, because it *is* the option order and the state's field order, and both change the
model's input.

    conn = open_db()
    hit = get(conn, key)                       # the stored response, or None
    put(conn, key, request_hash, state, questions, model, response, latency_ms)

Errors are not cached, so a failed call is retried on the next run.
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DB_PATH = os.path.join(CACHE_DIR, "predictions.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_hash   TEXT PRIMARY KEY,
    state_json     TEXT NOT NULL,
    questions_json TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_key TEXT PRIMARY KEY,
    request_hash   TEXT NOT NULL REFERENCES requests(request_hash),
    model_id       TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    settings_json  TEXT NOT NULL,
    response_json  TEXT NOT NULL,
    reported_model TEXT,
    latency_ms     REAL,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    cost_usd       REAL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_by_model ON predictions (model_id, model_version);
CREATE INDEX IF NOT EXISTS predictions_by_request ON predictions (request_hash);
"""


def canonical_json(value) -> str:
    """Deterministic JSON: no whitespace, one escaping form, key order preserved."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def request_hash(state, questions) -> str:
    return hashlib.sha256(canonical_json({"state": state, "questions": questions}).encode("utf-8")).hexdigest()


def prediction_key(model_id: str, model_version: str, settings, req_hash: str) -> str:
    material = canonical_json([model_id, model_version, settings, req_hash])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def open_db(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def get(conn: sqlite3.Connection, key: str):
    """The stored row for `key`, or None. `response_json` is the full model response."""
    row = conn.execute("SELECT * FROM predictions WHERE prediction_key = ?", (key,)).fetchone()
    return dict(row) if row else None


def put(conn, key, req_hash, state, questions, model, response, latency_ms):
    """Store one answer, with the request it answers. Called only for successful calls."""
    now = datetime.now(timezone.utc).isoformat()
    usage = response.get("usage") or {}
    conn.execute("INSERT OR IGNORE INTO requests VALUES (?, ?, ?, ?)",
                 (req_hash, canonical_json(state), canonical_json(questions), now))
    conn.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (key, req_hash, model.model_id, model.model_version, canonical_json(model.settings),
                  canonical_json(response), response.get("model"), latency_ms,
                  usage.get("input_tokens"), usage.get("output_tokens"),
                  model.usage_cost(usage.get("input_tokens") or 0), now))
    conn.commit()


def stats(conn: sqlite3.Connection):
    """Rows and cost per (model, version), for `python -m benchmarks.cache`."""
    return [dict(r) for r in conn.execute(
        "SELECT model_id, model_version, COUNT(*) AS predictions, SUM(input_tokens) AS input_tokens,"
        " ROUND(SUM(cost_usd), 4) AS cost_usd, MIN(created_at) AS first, MAX(created_at) AS last"
        " FROM predictions GROUP BY model_id, model_version ORDER BY model_id")]


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Show what the prediction cache holds.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()
    conn = open_db(args.db)
    rows = stats(conn)
    if not rows:
        print("empty: %s" % args.db)
        return
    print("%-34s %-14s %10s %12s %9s" % ("model", "version", "answers", "input tokens", "cost"))
    for r in rows:
        print("%-34s %-14s %10d %12s %9s" % (
            r["model_id"], (r["model_version"] or "")[:14], r["predictions"],
            r["input_tokens"] or "-", ("$%.4f" % r["cost_usd"]) if r["cost_usd"] else "free"))
    size = os.path.getsize(args.db) / 1e6
    print("\n%s (%.1f MB), %d distinct requests" % (
        os.path.relpath(args.db), size, conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]))


if __name__ == "__main__":
    main()
