"""Turn a TypeSafe answer into a probability distribution over a question's outcomes.

Used for two things: scoring a prediction against gold (benchmarks) and building soft training
targets (training). Both must read an answer the same way, so the conversion lives here.

Outcome order is the question's own order, which is also the order the model scores markers in:

    choice  the criteria keys        noul  [false, true]        score  levels 0..k-1
"""

from typing import Any, Dict, List

Question = Dict[str, Any]
AnswerDict = Dict[str, Any]


def outcomes(question: Question) -> List[Any]:
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "score":
        return list(range(len(question["criteria"])))
    return list(question["criteria"])


def target_distribution(question: Question, answer: AnswerDict) -> List[float]:
    """Probabilities over `outcomes(question)`, normalized; uniform if the answer carries none."""
    keys = outcomes(question)
    if answer.get("type") == "noul" and "probabilities" not in answer:
        p = [1.0 - float(answer["noul"]), float(answer["noul"])]
    else:
        probs = answer.get("probabilities") or {}
        p = [float(probs.get(str(k), probs.get(k, 0.0))) for k in keys]
    total = sum(p)
    return [v / total for v in p] if total > 0 else [1.0 / len(keys)] * len(keys)
