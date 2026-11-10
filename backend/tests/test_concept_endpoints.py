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


def test_memory_growth_endpoint_returns_three_series():
    client = TestClient(app)
    response = client.post(
        "/api/concept/memory-growth",
        json={"prompt": "The capital of France is", "max_tokens": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["models"]) == 3
    assert {item["model_type"] for item in payload["models"]} == {"transformer", "rwkv", "mamba"}
    if not any(item["available"] for item in payload["models"]):
        pytest.skip("No model weights available for the memory-growth endpoint")


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
