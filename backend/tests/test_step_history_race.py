import os
import sys
import threading
from types import SimpleNamespace

import torch

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.models.transformer import TransformerAdapter, TransformerAdapterState
from app.models.rwkv_engine import RWKVAdapter, RWKVAdapterState


class _FakeTokenizer:
    eos_token_id = -1

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(f"t{int(token_id)}" for token_id in token_ids)


class _BlockingTransformerModel:
    """在 decode forward 内阻塞，稳定制造“计数/历史”观察窗口。"""

    def __init__(self, entered: threading.Event, release: threading.Event):
        self.entered = entered
        self.release = release

    def __call__(self, input_ids, past_key_values=None, use_cache=True, output_attentions=True):
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise RuntimeError("test release timeout")
        # input_ids 是单 token，prefill 长度为 2，因此当前 kv 长度为 3
        attn = torch.zeros(1, 1, 1, 3)
        attn[0, 0, 0, -1] = 1.0
        return SimpleNamespace(
            past_key_values=object(),
            logits=torch.tensor([[[0.0, 10.0, 0.0, 0.0]]]),
            attentions=(attn,),
        )


class _FakePipeline:
    def decode(self, token_ids):
        return "t" + str(token_ids)


class _BlockingRWKVModel:
    def __init__(self, entered: threading.Event, release: threading.Event):
        self.entered = entered
        self.release = release

    def forward(self, tokens_or_id, state=None):
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise RuntimeError("test release timeout")
        next_state = [
            torch.zeros(2, 2),
            torch.ones(1, 2, 2),
            torch.zeros(2, 2),
        ]
        return torch.tensor([0.0, 10.0, 0.0, 0.0]), next_state


def _make_transformer_adapter():
    entered = threading.Event()
    release = threading.Event()
    adapter = TransformerAdapter()
    adapter.model = _BlockingTransformerModel(entered, release)
    adapter.tokenizer = _FakeTokenizer()
    adapter.device = "cpu"
    adapter.num_layers = 1
    adapter.num_heads = 1
    adapter.full_attn_layers = [0]
    state = TransformerAdapterState(
        tokens=["t1", "t2"],
        token_ids=[1, 2],
        prefill_attentions=[torch.eye(2).unsqueeze(0)],
        past_key_values=object(),
        last_logits=torch.zeros(1, 4),
    )
    return adapter, state, entered, release


def _make_rwkv_adapter():
    entered = threading.Event()
    release = threading.Event()
    adapter = RWKVAdapter()
    adapter.model = _BlockingRWKVModel(entered, release)
    adapter.pipeline = _FakePipeline()
    adapter.device = "cpu"
    adapter.num_layers = 1
    adapter.num_heads = 1
    adapter.head_size = 2
    adapter.hidden_size = 2
    adapter.vocab_size = 4
    initial_state = [
        torch.zeros(2, 2),
        torch.ones(1, 2, 2),
        torch.zeros(2, 2),
    ]
    state = RWKVAdapterState(
        tokens=["t1", "t2"],
        token_ids=[1, 2],
        history_states=[initial_state],
        history_summaries={0: [[1.0]]},
        current_state=initial_state,
        last_logits=torch.tensor([0.0, 10.0, 0.0, 0.0]),
    )
    return adapter, state, entered, release


def test_transformer_counter_never_advances_before_attention_history():
    adapter, state, entered, release = _make_transformer_adapter()
    observed = {}

    def concurrent_reader():
        try:
            assert entered.wait(timeout=10), "step() did not enter model forward"
            observed["count"] = state.step_count
            observed["history_len"] = len(state.step_attentions)
            observed["detail"] = adapter.get_attention_detail(
                state, step=None, layer=0, head=0
            )
        except BaseException as exc:  # pragma: no cover - 失败时保留原始异常
            observed["error"] = exc
        finally:
            release.set()

    reader = threading.Thread(target=concurrent_reader, daemon=True)
    reader.start()
    result = adapter.step(
        state,
        temperature=0.0,
        top_k=2,
        repetition_penalty=1.0,
        target_layer=0,
        target_head=0,
    )
    reader.join(timeout=10)

    assert "error" not in observed, f"concurrent reader failed: {observed.get('error')!r}"
    assert observed["count"] == observed["history_len"], (
        "step_count advanced before attention history was written: "
        f"count={observed['count']}, history={observed['history_len']}"
    )
    assert observed["detail"]["step"] == 0
    assert result["step"] == 1
    assert state.step_count == 1
    assert len(state.step_attentions) == 1


def test_rwkv_counter_never_advances_before_state_history():
    adapter, state, entered, release = _make_rwkv_adapter()
    observed = {}

    def concurrent_reader():
        try:
            assert entered.wait(timeout=10), "step() did not enter model forward"
            observed["count"] = state.step_count
            observed["history_len"] = len(state.history_states)
            observed["detail"] = adapter.get_recurrent_state(
                state, step=None, layer=0, head=0
            )
        except BaseException as exc:  # pragma: no cover
            observed["error"] = exc
        finally:
            release.set()

    reader = threading.Thread(target=concurrent_reader, daemon=True)
    reader.start()
    result = adapter.step(
        state,
        temperature=0.0,
        top_k=2,
        repetition_penalty=1.0,
        target_layer=0,
    )
    reader.join(timeout=10)

    assert "error" not in observed, f"concurrent reader failed: {observed.get('error')!r}"
    # 历史中始终包含 prefill 状态，因此正常关系是 history_len == count + 1。
    assert observed["history_len"] == observed["count"] + 1, (
        "step_count advanced before RWKV state history was written: "
        f"count={observed['count']}, history={observed['history_len']}"
    )
    assert observed["detail"]["step"] == 0
    assert result["step"] == 1
    assert state.step_count == 1
    assert len(state.history_states) == 2
