import json
import math
import os
import sys

import pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.services import concept_lab
from app.services.sampling import compute_distribution
from scripts.generate_sampling_fixtures import FIXTURE_PATH, build_fixture


def _softmax(values):
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _mamba_available():
    model_id = settings.DEFAULT_MAMBA
    if os.path.exists(model_id):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
        return isinstance(cached, str) and os.path.exists(cached)
    except Exception:
        return False


def test_sampling_transformations_are_composable():
    logits = [0.0, 1.0, 2.0]
    ids = [10, 20, 30]
    raw = compute_distribution(logits, ids, temperature=1.0, top_k=0, top_p=1.0,
                               repetition_penalty=1.0, seen_ids=[])
    assert abs(sum(raw) - 1.0) < 1e-9
    assert max(range(3), key=lambda i: raw[i]) == 2

    temperature = compute_distribution(logits, ids, temperature=0.5, top_k=0, top_p=1.0,
                                       repetition_penalty=1.0, seen_ids=[])
    expected_temperature = _softmax([0.0, 2.0, 4.0])
    assert all(abs(a - b) < 1e-9 for a, b in zip(temperature, expected_temperature))

    top_k = compute_distribution(logits, ids, temperature=1.0, top_k=1, top_p=1.0,
                                 repetition_penalty=1.0, seen_ids=[])
    assert top_k == [0.0, 0.0, 1.0]

    top_p = compute_distribution(logits, ids, temperature=1.0, top_k=0, top_p=0.6,
                                 repetition_penalty=1.0, seen_ids=[])
    assert abs(sum(top_p) - 1.0) < 1e-9
    assert top_p[2] > 0

    repetition = compute_distribution(logits, ids, temperature=1.0, top_k=0, top_p=1.0,
                                      repetition_penalty=2.0, seen_ids=[20])
    expected_repetition = _softmax([0.0, 0.5, 2.0])
    assert all(abs(a - b) < 1e-9 for a, b in zip(repetition, expected_repetition))

    combined = compute_distribution(logits, ids, temperature=0.7, top_k=2, top_p=0.8,
                                    repetition_penalty=1.2, seen_ids=[10])
    assert abs(sum(combined) - 1.0) < 1e-9


def test_sampling_fixture_matches_backend_reference():
    with open(FIXTURE_PATH, encoding="utf-8") as handle:
        committed = json.load(handle)
    assert committed == build_fixture()
    assert committed["source"] == "backend/app/services/sampling.py"
    assert committed["tolerance"] == 1e-6
    assert len(committed["cases"]) >= 10


def test_memory_series_description_is_derived_from_values():
    assert "基本持平" in concept_lab._describe_series([100, 100, 100])
    assert "整体上升" in concept_lab._describe_series([100, 200, 300])
    assert "整体下降" in concept_lab._describe_series([300, 200, 100])


def test_real_mamba_topk_logits_and_distribution():
    if not _mamba_available():
        pytest.skip(f"Mamba weights not available locally: {settings.DEFAULT_MAMBA}")

    result = concept_lab.sample_topk_logits("The capital of France is", "mamba")
    logits = result["topk_logits"]
    assert len(logits) == 50
    assert all(logits[i]["logit"] >= logits[i + 1]["logit"] for i in range(len(logits) - 1))
    assert logits[0]["id"] == result["sampled_token_id"]

    distribution = compute_distribution(
        [item["logit"] for item in logits],
        [item["id"] for item in logits],
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.2,
        seen_ids=result["prompt_token_ids"],
    )
    assert abs(sum(distribution) - 1.0) < 1e-9
    assert any(value > 0 for value in distribution)
