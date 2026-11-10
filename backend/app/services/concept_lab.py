"""Backend data sources for the concept lab.

The lab deliberately loads one model at a time and unloads it before the next
experiment. This keeps the 8GB GPU usable while demonstrating three different
state/memory shapes.
"""

import gc
import threading
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from ..core.config import settings
from ..models.mamba_engine import MambaAdapter
from ..models.rwkv_engine import RWKVAdapter
from ..models.transformer import TransformerAdapter

MODEL_TYPES = ("transformer", "rwkv", "mamba")
MAX_MEMORY_TOKENS = 32
_MODEL_LOCK = threading.Lock()


def _model_spec(model_type: str):
    if model_type == "transformer":
        return TransformerAdapter, settings.DEFAULT_TRANSFORMER
    if model_type == "rwkv":
        return RWKVAdapter, settings.DEFAULT_RWKV
    if model_type == "mamba":
        return MambaAdapter, settings.DEFAULT_MAMBA
    raise ValueError(f"Unknown model type: {model_type}")


def _format_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} B"


def _describe_series(values: List[int]) -> str:
    if not values:
        return "没有可用的观测值。"
    start, end = values[0], values[-1]
    low, high = min(values), max(values)
    span = high - low
    base = max(1, high)
    if span <= base * 0.01:
        shape = "基本持平"
    elif end > start:
        shape = "整体上升"
    elif end < start:
        shape = "整体下降"
    else:
        shape = "有波动但首尾接近"
    return (
        f"cache_bytes 从 {_format_bytes(start)} 到 {_format_bytes(end)}，"
        f"区间 {_format_bytes(low)} ~ {_format_bytes(high)}，形态{shape}。"
    )


def _theoretical_kv(adapter, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(adapter, TransformerAdapter):
        return {
            "available": False,
            "reason": "该架构没有 KV Cache，KV 理论公式不适用；因此不绘制理论曲线。",
            "values": [],
        }

    config = adapter.model.config if adapter.model is not None else None
    if config is None:
        return {"available": False, "reason": "无法读取 Transformer config。", "values": []}

    layer_types = getattr(config, "layer_types", None)
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    if layer_types:
        kv_layers = sum(1 for item in layer_types if item == "full_attention")
        layer_source = "config.layer_types 中 full_attention 层数"
    else:
        kv_layers = num_layers
        layer_source = "config.num_hidden_layers（无 layer_types）"

    num_attention_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", None))
    kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads

    if not kv_layers or not kv_heads or not head_dim:
        return {
            "available": False,
            "reason": "config 缺少 KV 层数/头数/头维度，无法可靠计算理论曲线。",
            "values": [],
        }

    try:
        element_bytes = int(next(adapter.model.parameters()).element_size())
    except StopIteration:
        element_bytes = 2

    values = [
        int(2 * kv_layers * kv_heads * head_dim * sample["seq_len"] * element_bytes)
        for sample in samples
    ]
    return {
        "available": True,
        "values": values,
        "formula": "2 × KV层数 × KV头数 × 头维度 × 序列长度 × 每元素字节数",
        "params": {
            "kv_layers": int(kv_layers),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
            "element_bytes": element_bytes,
            "layer_source": layer_source,
        },
        "reason": "理论曲线只计算 KV Cache；混合架构的线性层递归状态不在该公式内。",
    }


def _collect_growth(model_type: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    adapter_cls, model_id = _model_spec(model_type)
    adapter = adapter_cls()
    try:
        adapter.load_model(model_id, device=settings.DEVICE)
        state = adapter.create_state()
        prefill = adapter.prefill(state, prompt, target_layer=0)

        samples: List[Dict[str, Any]] = []

        def record(step: int) -> None:
            exported = adapter.export_state(
                state, step=step, layer=0, include_matrix=False
            )
            samples.append({
                "step": exported.step,
                "seq_len": exported.seq_len,
                "cache_bytes": int(exported.cache_bytes),
                "viz_history_bytes": int(exported.viz_history_bytes),
                "state_bytes": int(exported.state_bytes),
            })

        record(0)
        for _ in range(max_tokens):
            result = adapter.step(
                state,
                temperature=0.0,
                top_k=1,
                repetition_penalty=1.0,
                target_layer=0,
            )
            record(int(result["step"]))
            if result["is_finished"]:
                break

        return {
            "model_type": model_type,
            "model_id": model_id,
            "available": True,
            "prompt_tokens": len(prefill["token_ids"]),
            "samples": samples,
            "cache_series_note": _describe_series([item["cache_bytes"] for item in samples]),
            "theoretical": _theoretical_kv(adapter, samples),
        }
    except Exception as exc:
        return {
            "model_type": model_type,
            "model_id": model_id,
            "available": False,
            "error": str(exc),
            "samples": [],
            "theoretical": {"available": False, "reason": "模型未成功加载。", "values": []},
        }
    finally:
        try:
            adapter.unload()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_memory_growth(prompt: str, max_tokens: int) -> Dict[str, Any]:
    if not prompt.strip():
        raise ValueError("prompt 不能为空。")
    if max_tokens < 1 or max_tokens > MAX_MEMORY_TOKENS:
        raise ValueError(f"max_tokens 必须在 1 ~ {MAX_MEMORY_TOKENS} 之间。")

    with _MODEL_LOCK:
        models = [
            _collect_growth(model_type, prompt, max_tokens)
            for model_type in MODEL_TYPES
        ]
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "models": models,
    }


def _tokenize_with_hf(model_type: str, model_id: str, text: str) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    tokens = []
    for token_id in ids:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        tokens.append({
            "text": token_text,
            "id": int(token_id),
            "bytes": len(token_text.encode("utf-8")),
        })
    return {
        "model_type": model_type,
        "model_id": model_id,
        "available": True,
        "tokenizer": type(tokenizer).__name__,
        "tokens": tokens,
        "token_count": len(tokens),
        "byte_count": sum(item["bytes"] for item in tokens),
    }


def _tokenize_with_rwkv(text: str) -> Dict[str, Any]:
    from rwkv.utils import PIPELINE

    pipeline = PIPELINE(None, "rwkv_vocab_v20230424")
    ids = pipeline.encode(text)
    tokens = []
    for token_id in ids:
        token_text = pipeline.decode([token_id])
        tokens.append({
            "text": token_text,
            "id": int(token_id),
            "bytes": len(token_text.encode("utf-8")),
        })
    return {
        "model_type": "rwkv",
        "model_id": "rwkv_vocab_v20230424",
        "available": True,
        "tokenizer": "rwkv PIPELINE / TRIE_TOKENIZER",
        "tokens": tokens,
        "token_count": len(tokens),
        "byte_count": sum(item["bytes"] for item in tokens),
    }


def tokenize_compare(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("text 不能为空。")
    results = []
    for model_type in MODEL_TYPES:
        try:
            if model_type == "rwkv":
                results.append(_tokenize_with_rwkv(text))
            else:
                _, model_id = _model_spec(model_type)
                results.append(_tokenize_with_hf(model_type, model_id, text))
        except Exception as exc:
            results.append({
                "model_type": model_type,
                "available": False,
                "error": str(exc),
                "tokens": [],
                "token_count": 0,
                "byte_count": 0,
            })
    return {"text": text, "models": results}


def sample_topk_logits(prompt: str, model_type: str) -> Dict[str, Any]:
    if not prompt.strip():
        raise ValueError("prompt 不能为空。")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")

    adapter_cls, model_id = _model_spec(model_type)
    adapter = adapter_cls()
    try:
        with _MODEL_LOCK:
            adapter.load_model(model_id, device=settings.DEVICE)
            state = adapter.create_state()
            prefill = adapter.prefill(state, prompt, target_layer=0)
            result = adapter.step(
                state,
                temperature=0.0,
                top_k=50,
                repetition_penalty=1.0,
                target_layer=0,
            )
            return {
                "model_type": model_type,
                "model_id": model_id,
                "prompt": prompt,
                "prompt_tokens": prefill["tokens"],
                "prompt_token_ids": prefill["token_ids"],
                "sampled_token_id": result["token_id"],
                "topk_logits": result.get("topk_logits", []),
            }
    finally:
        try:
            adapter.unload()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
