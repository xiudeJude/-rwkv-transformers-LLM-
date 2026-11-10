import os
import sys

import pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.models.adapter_registry import get_adapter_class
from app.models.attention_baseline import PureAttentionAdapter


def _model_available(model_id: str) -> bool:
    if os.path.exists(model_id):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
        return isinstance(cached, str) and os.path.exists(cached)
    except Exception:
        return False


def _kv_shape_from_config(config) -> tuple[int, int, int]:
    num_layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
    num_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 0)))
    kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(getattr(config, "hidden_size", 0)) // num_heads
    return num_layers, kv_heads, int(head_dim)


@pytest.fixture(scope="module")
def baseline_adapter():
    model_id = settings.DEFAULT_ATTENTION_BASELINE
    if not _model_available(model_id):
        pytest.skip(f"Baseline weights not available locally: {model_id}")
    adapter = PureAttentionAdapter()
    try:
        adapter.load_model(model_id, device=settings.DEVICE)
    except Exception as exc:
        adapter.unload()
        pytest.skip(f"Baseline model could not be loaded: {exc}")
    yield adapter
    adapter.unload()


def _measure(adapter, length: int):
    state = adapter.create_state()
    token_ids = [1000 + (index % 1000) for index in range(length)]
    adapter.prefill(
        state,
        "baseline measurement",
        target_layer=0,
        prompt_token_ids=token_ids,
        capture_attention=False,
    )
    metrics = adapter.cache_metrics(state)
    adapter.close_state(state)
    return metrics


def test_attention_baseline_is_registered_without_new_tab():
    assert get_adapter_class("attention_baseline") is PureAttentionAdapter


def test_attention_baseline_kv_bytes_match_config_exactly(baseline_adapter):
    adapter = baseline_adapter
    num_layers, kv_heads, head_dim = _kv_shape_from_config(adapter.model.config)
    metrics_by_length = {
        length: _measure(adapter, length) for length in (64, 128)
    }

    for length, metrics in metrics_by_length.items():
        element_bytes = int(metrics["kv_element_bytes"])
        assert element_bytes > 0
        expected_kv = 2 * num_layers * kv_heads * head_dim * length * element_bytes
        diff = int(metrics["kv_bytes"]) - expected_kv
        assert diff == 0, (
            f"length={length}: actual kv_bytes={metrics['kv_bytes']} "
            f"theoretical={expected_kv} diff={diff}"
        )
        assert metrics["linear_state_bytes"] == 0
        assert metrics["cache_bytes"] == metrics["kv_bytes"]
        assert metrics["full_attn_layers"] == list(range(num_layers))

    first = metrics_by_length[64]
    second = metrics_by_length[128]
    element_bytes = int(first["kv_element_bytes"])
    per_token_from_cache = (
        int(second["kv_bytes"]) - int(first["kv_bytes"])
    ) // (128 - 64)
    expected_per_token = 2 * num_layers * kv_heads * head_dim * element_bytes
    # 核对值：Qwen2.5-0.5B = 2 × 24 层 × 2 个 KV 头 × 64 头维度 × 2 B
    # = 12288 B/token。这里不写死 12288，而是由 config 与实际缓存 dtype 推出；
    # 如果实测值偏离该推导值，上面缓存差值的断言会直接失败。
    assert per_token_from_cache == expected_per_token
