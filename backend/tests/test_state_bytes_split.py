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
