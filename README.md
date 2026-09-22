# Dadvar

*Dadvar* (دادور) — Persian for "judge", literally "bearer of justice".

A benchmark for typed-decision models. It converts public datasets into the TypeSafe System One
schema (`choice`, `score` and `noul` questions with gold answers), runs any model over them — a
local checkpoint or TypeSafe's hosted Jev — and reports accuracy, probability error and latency on
identical rows.

Every run pins each dataset's revision, samples with a seed, caches each request and response so no
answer is paid for twice, and records whether a model's publisher says it trained on that data.

## Install

```bash
git clone https://github.com/ashaazami/dadvar.git
cd dadvar
python -m venv .venv && source .venv/bin/activate
pip install -e '.[bench]'
```

Python 3.10+. Model weights and datasets are downloaded on first use and cached under
`~/.cache/huggingface/`.

## Run a benchmark

```bash
python -m benchmarks.run typed_decisions                  # a local checkpoint
TYPESAFE_API_KEY=... python -m benchmarks.run typed_decisions --backend jev
python -m benchmarks.report --group "all=typed_decisions"
```

`data/` builds the rows, `benchmarks/` runs and scores them. See [data/README.md](data/README.md)
for the sources, their licences and the train/eval guarantees, and
[benchmarks/README.md](benchmarks/README.md) for the runner, the cache and the metrics.

Adding a model means one subclass of `Model` with a `system_one(state, questions)` method
(`benchmarks/models.py`); adding a dataset means one conversion function and one registry entry
(`data/sources.py`).

## Run a checkpoint directly

`dadvar/` holds the three modules of Laya needed to load a checkpoint and answer questions with it.
Its router, language detection and shortlist modules are not included here.

```python
import dadvar

agent = dadvar.load("convaiinnovations/laya")   # English; picks MPS / CUDA / CPU automatically

state = {
    "from": "user@example.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today.",
}
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                     "sales": "pricing, new contracts", "other": "everything else"},
    },
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}

answers = agent.predict(state, questions)["answers"]
answers["department"]["choice"]   # 'billing'  (p = 0.955)
answers["urgency"]["score"]       # 1.45 on a 0–2 scale
answers["refund"]["noul"]         # 0.86 = P(yes)
```

| Checkpoint | Encoder | Params | Context | Use for |
|---|---|---:|---:|---|
| `convaiinnovations/laya` | ModernBERT-large | 421M | 512 | English |
| `convaiinnovations/laya` (`subfolder="multilingual"`) | mmBERT-base | 322M | 1024 | 100+ languages |
| `convaiinnovations/laya` (`subfolder="typed-decisions"`) | ModernBERT-large | 421M | 1024 | typed-decisions workflows |

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). `dadvar/` is derived from
[Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations, whose checkpoints this code
loads; the weights are Convai's and are downloaded separately. No dataset content is stored here.
