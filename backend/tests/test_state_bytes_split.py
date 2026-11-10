import os
import sys

import pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.models.mamba_engine import MambaAdapter
from app.models.rwkv_engine import RWKVAdapter
from app.models.transformer import TransformerAdapter


def _model_available(model_id: str) -> bool:
    if os.path.exists(model_id):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
        return isinstance(cached, str) and os.path.exists(cached)
    except Exception:
        return False


def _load_or_skip(adapter_cls, model_id):
    if not _model_available(model_id):
        pytest.skip(f"Model weights not available locally: {model_id}")
    adapter = adapter_cls()
    try:
        adapter.load_model(model_id, device=settings.DEVICE)
    except Exception as exc:
        adapter.unload()
        pytest.skip(f"Model could not be loaded: {exc}")
    return adapter


def test_transformer_cache_and_visualization_bytes_are_separate():
    adapter = _load_or_skip(TransformerAdapter, settings.DEFAULT_TRANSFORMER)
    try:
        state = adapter.create_state()
        adapter.prefill(state, "Hello world", target_layer=3)
        prefilled = adapter.export_state(state, step=0, layer=3, head=0, include_matrix=False)
        assert prefilled.cache_bytes > 0
        assert prefilled.viz_history_bytes > 0
        assert prefilled.state_bytes == prefilled.cache_bytes + prefilled.viz_history_bytes

        adapter.step(state, temperature=0.0, top_k=1, repetition_penalty=1.0,
                     target_layer=3, target_head=0)
        stepped = adapter.export_state(state, step=1, layer=3, head=0, include_matrix=False)
        assert stepped.state_bytes == stepped.cache_bytes + stepped.viz_history_bytes
        assert stepped.viz_history_bytes >= prefilled.viz_history_bytes
    finally:
        adapter.unload()


def test_hybrid_kv_bytes_match_theory_and_linear_state_is_separate():
    adapter = _load_or_skip(TransformerAdapter, settings.DEFAULT_TRANSFORMER)
    try:
        config = adapter.model.config
        layer_types = getattr(config, "layer_types", None)
        full_layers = (
            [index for index, item in enumerate(layer_types) if item == "full_attention"]
            if layer_types
            else list(range(int(getattr(config, "num_hidden_layers", 0))))
        )
        assert adapter.full_attn_layers == full_layers

        length = 64
        state = adapter.create_state()
        adapter.prefill(
            state,
            "hybrid split check",
            target_layer=0,
            prompt_token_ids=[1000 + (index % 1000) for index in range(length)],
            capture_attention=False,
        )
        metrics = adapter.cache_metrics(state)
        adapter.close_state(state)

        num_heads = int(getattr(config, "num_attention_heads"))
        kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = int(getattr(config, "hidden_size")) // num_heads
        element_bytes = int(metrics["kv_element_bytes"])
        expected_kv = 2 * len(full_layers) * kv_heads * int(head_dim) * length * element_bytes

        diff = int(metrics["kv_bytes"]) - expected_kv
        assert diff == 0, (
            f"hybrid kv_bytes={metrics['kv_bytes']} theoretical={expected_kv} "
            f"diff={diff}; full_attn_layers={full_layers}"
        )
        assert metrics["linear_state_bytes"] > 0
        assert metrics["cache_bytes"] == metrics["kv_bytes"] + metrics["linear_state_bytes"]
    finally:
        adapter.unload()


def test_rwkv_cache_and_visualization_bytes_are_separate():
    adapter = _load_or_skip(RWKVAdapter, settings.DEFAULT_RWKV)
    try:
        state = adapter.create_state()
        adapter.prefill(state, "Hello world", target_layer=0)
        prefilled = adapter.export_state(state, step=0, layer=0, include_matrix=False)
        assert prefilled.cache_bytes > 0
        assert prefilled.state_bytes == prefilled.cache_bytes + prefilled.viz_history_bytes

        adapter.step(state, temperature=0.0, top_k=1, repetition_penalty=1.0,
                     target_layer=0)
        stepped = adapter.export_state(state, step=1, layer=0, include_matrix=False)
        assert stepped.cache_bytes == prefilled.cache_bytes
        assert stepped.viz_history_bytes > 0
        assert stepped.state_bytes == stepped.cache_bytes + stepped.viz_history_bytes
    finally:
        adapter.unload()


def test_mamba_cache_and_visualization_bytes_are_separate():
    adapter = _load_or_skip(MambaAdapter, settings.DEFAULT_MAMBA)
    try:
        state = adapter.create_state()
        adapter.prefill(state, "Hello world", target_layer=0)
        prefilled = adapter.export_state(state, step=0, layer=0, include_matrix=False)
        assert prefilled.cache_bytes > 0
        assert prefilled.viz_history_bytes == 0
        assert prefilled.state_bytes == prefilled.cache_bytes + prefilled.viz_history_bytes

        adapter.step(state, temperature=0.0, top_k=1, repetition_penalty=1.0,
                     target_layer=0)
        stepped = adapter.export_state(state, step=1, layer=0, include_matrix=False)
        assert stepped.cache_bytes == prefilled.cache_bytes
        assert stepped.state_bytes == stepped.cache_bytes + stepped.viz_history_bytes
    finally:
        adapter.unload()
