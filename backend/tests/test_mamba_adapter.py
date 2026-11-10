import os
import sys

import pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.models.adapter_registry import get_adapter_class
from app.models.mamba_engine import MambaAdapter


def _mamba_model_available() -> bool:
    model_id = settings.DEFAULT_MAMBA
    if os.path.exists(model_id):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        cached_config = try_to_load_from_cache(model_id, "config.json")
        return isinstance(cached_config, str) and os.path.exists(cached_config)
    except Exception:
        return False


@pytest.fixture(scope="module")
def mamba_adapter():
    if not _mamba_model_available():
        pytest.skip(f"Mamba weights not available locally: {settings.DEFAULT_MAMBA}")

    adapter = MambaAdapter()
    try:
        adapter.load_model(settings.DEFAULT_MAMBA, device=settings.DEVICE)
    except Exception as exc:
        pytest.skip(f"Mamba model could not be loaded: {exc}")
    yield adapter
    adapter.unload()


def _component_map(state):
    return {component["name"]: component for component in state.meta["components"]}


def test_mamba_prefill_step_export(mamba_adapter):
    assert get_adapter_class("mamba") is MambaAdapter
    metadata = mamba_adapter.get_metadata()
    assert metadata.model_type == "mamba"
    assert metadata.num_layers > 0
    assert metadata.num_heads == 0

    state = mamba_adapter.create_state()
    prefill = mamba_adapter.prefill(state, "Hello world", target_layer=0)
    assert prefill["seq_len"] > 0
    assert prefill["num_heads"] == 0

    prefilled = mamba_adapter.export_state(state, step=0, layer=0)
    assert prefilled.kind == "recurrent_state"
    assert prefilled.model_type == "mamba"
    assert prefilled.head is None
    assert prefilled.matrix is None
    assert prefilled.state_bytes > 0
    assert prefilled.cache_bytes > 0
    assert prefilled.viz_history_bytes == 0
    assert prefilled.state_bytes == prefilled.cache_bytes + prefilled.viz_history_bytes
    assert prefilled.num_layers > 0

    components = _component_map(prefilled)
    assert {"conv_state", "ssm_state"}.issubset(components)
    expected_conv_dtype = "float16" if str(mamba_adapter.device).startswith("cuda") else "float32"
    assert components["conv_state"]["dtype"] == expected_conv_dtype
    assert components["ssm_state"]["dtype"] == "float32"
    assert all(len(c["layer_norms"]) == prefilled.num_layers for c in components.values())
    assert sum(c["bytes"] for c in components.values()) == prefilled.state_bytes

    step = mamba_adapter.step(
        state,
        temperature=0.0,
        top_k=5,
        repetition_penalty=1.0,
        target_layer=0,
    )
    assert step["step"] == 1
    assert step["token_id"] >= 0

    stepped = mamba_adapter.export_state(state, step=1, layer=0)
    assert stepped.step == 1
    assert stepped.seq_len == prefilled.seq_len + 1
    assert stepped.state_bytes == prefilled.state_bytes

    # 历史快照的 seq_len 应保持为当时的值，而不是当前总长度。
    prefilled_again = mamba_adapter.export_state(state, step=0, layer=0)
    assert prefilled_again.seq_len == prefilled.seq_len


def test_mamba_state_bytes_independent_of_sequence_length(mamba_adapter):
    short_state = mamba_adapter.create_state()
    short_prefill = mamba_adapter.prefill(short_state, "Hi", target_layer=0)
    short_export = mamba_adapter.export_state(short_state, step=0, layer=0)

    long_state = mamba_adapter.create_state()
    long_prefill = mamba_adapter.prefill(
        long_state,
        "The capital of France is a city with a long history and many museums.",
        target_layer=0,
    )
    long_export = mamba_adapter.export_state(long_state, step=0, layer=0)

    assert short_prefill["seq_len"] != long_prefill["seq_len"]
    assert short_export.state_bytes == long_export.state_bytes
    assert short_export.state_bytes > 0

    mamba_adapter.step(
        short_state,
        temperature=0.0,
        top_k=5,
        repetition_penalty=1.0,
        target_layer=0,
    )
    stepped_export = mamba_adapter.export_state(short_state, step=1, layer=0)
    assert stepped_export.state_bytes == short_export.state_bytes
