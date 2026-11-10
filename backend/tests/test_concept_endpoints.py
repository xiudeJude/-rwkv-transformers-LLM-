import os
import sys

import pytest
from fastapi.testclient import TestClient

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from main import app


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


def test_tokenize_endpoint_returns_three_tokenizers_without_model_weights():
    client = TestClient(app)
    response = client.post("/api/concept/tokenize", json={"text": "Hello 世界"})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["models"]) == 3
    assert {item["model_type"] for item in payload["models"]} == {"transformer", "rwkv", "mamba"}
    for item in payload["models"]:
        assert "tokens" in item
        assert "token_count" in item
        assert "byte_count" in item
        if item["available"]:
            assert item["token_count"] == len(item["tokens"])


def test_memory_growth_endpoint_returns_four_prefill_series():
    client = TestClient(app)
    response = client.post(
        "/api/concept/memory-growth",
        json={"prompt": "The capital of France is"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["models"]) == 4
    assert {item["model_type"] for item in payload["models"]} == {
        "transformer",
        "attention_baseline",
        "rwkv",
        "mamba",
    }
    assert {64, 128, 256, 512, 1024, 2048}.issubset(set(payload["requested_lengths"]))
    if not any(item["available"] for item in payload["models"]):
        pytest.skip("No model weights available for the memory-growth endpoint")
    for model in payload["models"]:
        if not model["available"]:
            continue
        assert model["measurements"]
        lengths = {item["length"] for item in model["measurements"]}
        assert {64, 128, 256, 512}.issubset(lengths)
        for item in model["measurements"]:
            assert item["cache_bytes"] == item["kv_bytes"] + item["linear_state_bytes"]
            assert item["dtype_label"]
        if model["model_type"] in {"transformer", "attention_baseline"}:
            assert {1024, 2048}.issubset(lengths)
            assert model["theoretical"]["available"]
            assert model["theoretical"]["matches"] is True


def test_sample_logits_endpoint_uses_real_adapter_logits():
    if not _mamba_available():
        pytest.skip(f"Mamba weights not available locally: {settings.DEFAULT_MAMBA}")
    client = TestClient(app)
    response = client.post(
        "/api/concept/sample-logits",
        json={"prompt": "The capital of France is", "model_type": "mamba"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_type"] == "mamba"
    assert len(payload["topk_logits"]) == 50
    assert payload["prompt_token_ids"]
