"""Backend data sources for the concept lab.

The lab deliberately loads one model at a time and unloads it before the next
experiment. This keeps the 8GB GPU usable while demonstrating different state
and memory shapes.
"""

import gc
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from ..core.config import settings
from ..models.attention_baseline import PureAttentionAdapter
from ..models.mamba_engine import MambaAdapter
from ..models.rwkv_engine import RWKVAdapter
from ..models.transformer import TransformerAdapter
from .model_gate import ModelBusyError, model_gate

MEMORY_MODEL_TYPES = ("transformer", "attention_baseline", "rwkv", "mamba")
TOKENIZER_MODEL_TYPES = ("transformer", "rwkv", "mamba")
SAMPLE_MODEL_TYPES = ("transformer", "attention_baseline", "rwkv", "mamba")

DEFAULT_MEASURE_LENGTHS = (64, 128, 256, 512)
ATTENTION_EXTRA_LENGTHS = (1024, 2048)
MAX_MEASURE_LENGTH = 2048


def _model_spec(model_type: str):
    if model_type == "transformer":
        return TransformerAdapter, settings.DEFAULT_TRANSFORMER
    if model_type == "attention_baseline":
        return PureAttentionAdapter, settings.DEFAULT_ATTENTION_BASELINE
    if model_type == "rwkv":
        return RWKVAdapter, settings.DEFAULT_RWKV
    if model_type == "mamba":
        return MambaAdapter, settings.DEFAULT_MAMBA
    raise ValueError(f"Unknown model type: {model_type}")


def _display_name(model_type: str) -> str:
    return {
        "transformer": "Qwen3.5-0.8B 混合架构",
        "attention_baseline": "Qwen2.5-0.5B 纯注意力（基座）",
        "rwkv": "RWKV-7 Goose 0.4B",
        "mamba": "Mamba-130M（基座）",
    }[model_type]


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
    if span <= base * 0.001:
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


def _theoretical_kv(adapter, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    measured = [item for item in measurements if item.get("measured", True)]
    # 以实际 KV 张量的 element_size 为准，避免权重与缓存 dtype 不同时把
    # 理论曲线算错；取不到时退回模型参数 dtype。
    element_bytes = next(
        (
            int(item["kv_element_bytes"])
            for item in measured
            if item.get("kv_element_bytes")
        ),
        0,
    )
    if element_bytes == 0:
        try:
            element_bytes = int(next(adapter.model.parameters()).element_size())
        except StopIteration:
            element_bytes = 2

    values = [
        int(2 * kv_layers * kv_heads * head_dim * item["length"] * element_bytes)
        for item in measured
    ]
    actual_kv = [int(item.get("kv_bytes", 0)) for item in measured]
    diffs = [actual - expected for actual, expected in zip(actual_kv, values)]
    max_abs_diff = max((abs(diff) for diff in diffs), default=0)
    matches = all(diff == 0 for diff in diffs)
    is_hybrid = len(getattr(adapter, "full_attn_layers", [])) != int(num_layers or 0)
    if is_hybrid:
        reason = (
            "理论曲线只计算 KV Cache；实际 cache_bytes 另含线性层递归状态，"
            "因此与理论逐点比较的是 kv_bytes。"
        )
    else:
        reason = "纯注意力模型：实际 kv_bytes 应逐点等于理论 KV。"
    return {
        "available": True,
        "values": values,
        "lengths": [int(item["length"]) for item in measured],
        "actual_kv_bytes": actual_kv,
        "diff_bytes": diffs,
        "max_abs_diff_bytes": max_abs_diff,
        "matches": matches,
        "formula": "2 × KV层数 × KV头数 × 头维度 × 序列长度 × 每元素字节数",
        "params": {
            "kv_layers": int(kv_layers),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
            "element_bytes": element_bytes,
            "layer_source": layer_source,
        },
        "reason": reason,
    }


def _source_ids(adapter, prompt: str, target_length: int) -> List[int]:
    ids = list(adapter.tokenize(prompt)["token_ids"])
    if not ids:
        ids = [0]
    while len(ids) < target_length:
        ids.extend(ids[:target_length - len(ids)])
    return ids[:target_length]


def _collect_growth(model_type: str, prompt: str, requested_lengths: List[int]) -> Dict[str, Any]:
    adapter_cls, model_id = _model_spec(model_type)
    adapter = adapter_cls()
    try:
        adapter.load_model(model_id, device=settings.DEVICE)
        is_attention = isinstance(adapter, TransformerAdapter)
        measure_lengths = sorted(set(
            length for length in requested_lengths
            if is_attention or length <= max(DEFAULT_MEASURE_LENGTHS)
        ))
        if is_attention:
            measure_lengths = sorted(set(measure_lengths + list(ATTENTION_EXTRA_LENGTHS)))
        projection_lengths = [] if is_attention else sorted(set(
            length for length in requested_lengths
            if length > max(DEFAULT_MEASURE_LENGTHS)
        ))

        measurements: List[Dict[str, Any]] = []
        last_metrics: Optional[Dict[str, Any]] = None
        for length in measure_lengths:
            state = adapter.create_state()
            ids = _source_ids(adapter, prompt, length)
            prefill_kwargs = {"prompt_token_ids": ids}
            if is_attention:
                prefill_kwargs["capture_attention"] = False
            prefill = adapter.prefill(
                state,
                prompt,
                target_layer=0,
                **prefill_kwargs,
            )
            metrics = adapter.cache_metrics(state)
            last_metrics = metrics
            measurements.append({
                "length": int(length),
                "actual_tokens": int(prefill["seq_len"]),
                "measured": True,
                "cache_bytes": int(metrics["cache_bytes"]),
                "kv_bytes": int(metrics.get("kv_bytes", 0)),
                "linear_state_bytes": int(metrics.get("linear_state_bytes", 0)),
                "kv_element_bytes": int(metrics.get("kv_element_bytes", 0)),
                "dtype_label": metrics.get("dtype_label", ""),
                "full_attn_layers": metrics.get("full_attn_layers", []),
                "linear_layers": metrics.get("linear_layers", []),
            })
            adapter.close_state(state)

        for length in projection_lengths:
            if last_metrics is None:
                break
            measurements.append({
                "length": int(length),
                "actual_tokens": int(length),
                "measured": False,
                "cache_bytes": int(last_metrics["cache_bytes"]),
                "kv_bytes": int(last_metrics.get("kv_bytes", 0)),
                "linear_state_bytes": int(last_metrics.get("linear_state_bytes", 0)),
                "kv_element_bytes": int(last_metrics.get("kv_element_bytes", 0)),
                "dtype_label": last_metrics.get("dtype_label", ""),
                "full_attn_layers": last_metrics.get("full_attn_layers", []),
                "linear_layers": last_metrics.get("linear_layers", []),
                "note": "结构上恒定，未实测",
            })

        measured_values = [
            item["cache_bytes"] for item in measurements if item.get("measured")
        ]
        return {
            "model_type": model_type,
            "display_name": _display_name(model_type),
            "model_id": model_id,
            "available": True,
            "is_base_model": model_type in {"attention_baseline", "mamba"},
            "measurements": measurements,
            "cache_series_note": _describe_series(measured_values),
            "theoretical": _theoretical_kv(adapter, measurements),
        }
    except Exception as exc:
        return {
            "model_type": model_type,
            "display_name": _display_name(model_type),
            "model_id": model_id,
            "available": False,
            "is_base_model": model_type in {"attention_baseline", "mamba"},
            "error": str(exc),
            "measurements": [],
            "theoretical": {"available": False, "reason": "模型未成功加载。", "values": []},
        }
    finally:
        try:
            adapter.unload()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_memory_growth(prompt: str, lengths: Optional[List[int]] = None) -> Dict[str, Any]:
    if not prompt.strip():
        raise ValueError("prompt 不能为空。")
    # 64/128/256/512 是四个模型的共同测量基线；注意力模型再自动补
    # 1024/2048。自定义 lengths 只做增补，不会把基线长度删掉。
    requested = sorted(set(
        list(lengths or [])
        + list(DEFAULT_MEASURE_LENGTHS)
        + list(ATTENTION_EXTRA_LENGTHS)
    ))
    if any(length < 1 or length > MAX_MEASURE_LENGTH for length in requested):
        raise ValueError(f"lengths 必须在 1 ~ {MAX_MEASURE_LENGTH} 之间。")

    with model_gate.hold():
        models = [
            _collect_growth(model_type, prompt, requested)
            for model_type in MEMORY_MODEL_TYPES
        ]
    return {
        "prompt": prompt,
        "requested_lengths": sorted(set(requested)),
        "models": models,
    }


def _tokenize_with_hf(model_type: str, model_id: str, text: str) -> Dict[str, Any]:
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    tokenizer_path = model_id
    if model_type == "transformer":
        # GGUF 目录需要显式传入 gguf_file，AutoTokenizer 才能找到 tokenizer。
        from ..models.transformer import resolve_model_path_and_gguf

        tokenizer_path, gguf_file = resolve_model_path_and_gguf(model_id)
        if gguf_file:
            load_kwargs["gguf_file"] = gguf_file

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **load_kwargs)
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
    for model_type in TOKENIZER_MODEL_TYPES:
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
    if model_type not in SAMPLE_MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")

    adapter_cls, model_id = _model_spec(model_type)
    adapter = adapter_cls()
    with model_gate.hold():
        try:
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
