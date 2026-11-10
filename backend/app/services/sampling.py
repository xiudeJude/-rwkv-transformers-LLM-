"""Pure sampling-distribution helpers for the Concept Lab.

These functions intentionally operate on a *candidate logit set* (the top-50
returned by an adapter) and mirror the browser-side calculations. They are not
used by the normal generation path, which keeps the existing sampling behavior
unchanged.
"""

import math
from typing import Iterable, List, Sequence


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    finite = [value for value in logits if math.isfinite(value)]
    max_value = max(finite) if finite else 0.0
    exps = [math.exp(value - max_value) if math.isfinite(value) else 0.0 for value in logits]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def apply_repetition_penalty(
    logits: Sequence[float], token_ids: Sequence[int], seen_ids: Iterable[int], penalty: float
) -> List[float]:
    result = list(logits)
    if penalty == 1.0:
        return result
    seen = set(seen_ids)
    for index, token_id in enumerate(token_ids):
        if token_id in seen:
            if result[index] > 0:
                result[index] /= penalty
            else:
                result[index] *= penalty
    return result


def apply_temperature(logits: Sequence[float], temperature: float) -> List[float]:
    value = max(temperature, 1e-4)
    return [logit / value for logit in logits]


def apply_top_k(logits: Sequence[float], top_k: int) -> List[float]:
    if top_k <= 0 or top_k >= len(logits):
        return list(logits)
    threshold = sorted(logits, reverse=True)[top_k - 1]
    return [logit if logit >= threshold else float("-inf") for logit in logits]


def apply_top_p(logits: Sequence[float], top_p: float) -> List[float]:
    if top_p >= 1.0 or not logits:
        return list(logits)
    probs = _softmax(logits)
    order = sorted(range(len(logits)), key=lambda index: probs[index], reverse=True)
    kept = set()
    cumulative = 0.0
    for index in order:
        kept.add(index)
        cumulative += probs[index]
        if cumulative >= top_p:
            break
    return [logit if index in kept else float("-inf") for index, logit in enumerate(logits)]


def compute_distribution(
    logits: Sequence[float],
    token_ids: Sequence[int],
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seen_ids: Iterable[int] = (),
) -> List[float]:
    """Repetition -> temperature -> top-k -> top-p -> softmax."""
    transformed = apply_repetition_penalty(
        logits, token_ids, seen_ids, repetition_penalty
    )
    transformed = apply_temperature(transformed, temperature)
    transformed = apply_top_k(transformed, top_k)
    transformed = apply_top_p(transformed, top_p)
    return _softmax(transformed)
