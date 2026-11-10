from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter_base import ModelAdapter
from .adapter_registry import register_adapter
from ..core.schemas import InternalState, ModelMetadata

# 纯 PyTorch 参考路径在 Windows 上可用但明显慢于 CUDA kernel，
# 对交互式演示设置保守上限。
MAMBA_MAX_PROMPT_TOKENS = 512
MAMBA_MAX_NEW_TOKENS = 128


@dataclass
class MambaAdapterState:
    """单个 Mamba 生成会话的递归缓存与逐层摘要历史。"""

    tokens: List[str] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    cache: Any = None
    last_logits: Optional[torch.Tensor] = None
    step_count: int = 0
    is_finished: bool = False
    history_summaries: List[Dict[str, Any]] = field(default_factory=list)


@register_adapter("mamba")
class MambaAdapter(ModelAdapter):
    """通过 transformers 内置 MambaForCausalLM 读取 Mamba 状态。

    未安装 mamba-ssm / causal-conv1d 时，transformers 会自动使用纯 PyTorch
    reference 实现；本适配器不依赖任何额外加速包。
    """

    def __init__(self):
        super().__init__()
        self.num_heads = 0
        self.d_inner = 0
        self.state_size = 0
        self.conv_kernel = 0

    def load_model(self, model_name_or_path: str, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_name_or_path
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[MambaAdapter] Loading tokenizer from: {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        print(f"[MambaAdapter] Loading model into device {self.device} ({dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

        config = self.model.config
        self.num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
        self.hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", 0))
        self.vocab_size = getattr(config, "vocab_size", len(self.tokenizer))
        self.d_inner = getattr(config, "intermediate_size", getattr(config, "d_inner", 0))
        self.state_size = getattr(config, "state_size", 0)
        self.conv_kernel = getattr(config, "conv_kernel", 0)
        # Mamba 不暴露注意力头；不要伪造 head 维度。
        self.num_heads = 0
        self.full_attn_layers = []
        print(
            f"[MambaAdapter] Initialized: Layers={self.num_layers}, "
            f"d_inner={self.d_inner}, state_size={self.state_size}, conv_kernel={self.conv_kernel}"
        )

    def unload(self):
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_type="mamba",
            num_layers=self.num_layers,
            num_heads=0,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            device=str(self.device),
            full_attn_layers=[],
        )

    def tokenize(self, text: str) -> Dict[str, Any]:
        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"]
        token_ids = input_ids[0].tolist()
        tokens = [self.decode_token(tid) for tid in token_ids]
        return {
            "input_ids": input_ids,
            "token_ids": token_ids,
            "tokens": tokens,
            "attention_mask": encoding.get("attention_mask", None),
        }

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def create_state(self) -> MambaAdapterState:
        return MambaAdapterState()

    def _get_cache(self, outputs: Any) -> Any:
        cache = getattr(outputs, "cache_params", None)
        if cache is None:
            cache = getattr(outputs, "past_key_values", None)
        if cache is None:
            raise RuntimeError("Mamba model returned no cache state.")
        return cache

    def _component_summaries(self, cache: Any) -> List[Dict[str, Any]]:
        """把 Mamba 的 conv_state / ssm_state 转成统一面板可消费的组件摘要。"""
        layers = getattr(cache, "layers", None)
        if not layers:
            return []

        components: List[Dict[str, Any]] = []
        for name, attr in (("conv_state", "conv_states"), ("ssm_state", "recurrent_states")):
            tensors: List[torch.Tensor] = []
            for layer in layers:
                value = getattr(layer, attr, None)
                if isinstance(value, dict):
                    tensors.extend(v for v in value.values() if isinstance(v, torch.Tensor))
                elif isinstance(value, (list, tuple)):
                    tensors.extend(v for v in value if isinstance(v, torch.Tensor))
                elif isinstance(value, torch.Tensor):
                    tensors.append(value)
            if not tensors:
                continue

            first = tensors[0]
            total_bytes = sum(int(t.numel() * t.element_size()) for t in tensors)
            layer_norms = [
                round(float(torch.norm(t.float(), p="fro").item()), 6)
                for t in tensors
            ]
            components.append({
                "name": name,
                "shape": list(first.shape),
                "dtype": str(first.dtype).replace("torch.", ""),
                "bytes": total_bytes,
                "layer_norms": layer_norms,
                "layer_count": len(tensors),
            })
        return components

    def _snapshot(self, cache: Any, seq_len: int) -> Dict[str, Any]:
        components = self._component_summaries(cache)
        total_bytes = sum(int(c["bytes"]) for c in components)

        max_layers = max((len(c["layer_norms"]) for c in components), default=0)
        combined = []
        for layer_idx in range(max_layers):
            value = sum(
                c["layer_norms"][layer_idx]
                for c in components
                if layer_idx < len(c["layer_norms"])
            )
            combined.append(round(float(value), 6))
        return {
            "components": components,
            "state_bytes": total_bytes,
            "summary": combined,
            "seq_len": seq_len,
        }

    def prefill(self, state: MambaAdapterState, prompt: str, target_layer: int = 0, target_head: int = 0) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded in MambaAdapter.")

        tokenized = self.tokenize(prompt)
        token_ids = tokenized["token_ids"]
        if len(token_ids) > MAMBA_MAX_PROMPT_TOKENS:
            raise ValueError(
                f"Mamba prompt 过长：{len(token_ids)} tokens，"
                f"当前纯 PyTorch 参考路径上限为 {MAMBA_MAX_PROMPT_TOKENS} tokens。"
            )

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        state.tokens = tokenized["tokens"]
        state.token_ids = token_ids
        state.cache = self._get_cache(outputs)
        state.last_logits = outputs.logits[:, -1, :].detach()
        state.step_count = 0
        state.is_finished = False
        state.history_summaries = [self._snapshot(state.cache, len(token_ids))]

        target_layer = min(max(0, target_layer), max(0, self.num_layers - 1))
        return {
            "session_id": None,
            "prompt": prompt,
            "tokens": state.tokens,
            "token_ids": state.token_ids,
            "seq_len": len(state.token_ids),
            "layer": target_layer,
            "num_layers": self.num_layers,
            "num_heads": 0,
        }

    def step(
        self,
        state: MambaAdapterState,
        temperature: float = 0.7,
        top_k: int = 10,
        repetition_penalty: float = 1.2,
        target_layer: int = 0,
        target_head: int = 0,
    ) -> Dict[str, Any]:
        if state.is_finished or state.step_count >= MAMBA_MAX_NEW_TOKENS:
            state.is_finished = True
            return {
                "step": state.step_count,
                "token": "",
                "token_id": -1,
                "token_prob": 0.0,
                "topk_candidates": [],
                "is_finished": True,
            }

        if state.cache is None or state.last_logits is None:
            raise RuntimeError("Session has not executed prefill stage. Call prefill() first.")

        logits = state.last_logits[0].clone().float()
        if repetition_penalty != 1.0 and len(state.token_ids) > 0:
            for tid in set(state.token_ids):
                if 0 <= tid < len(logits):
                    if logits[tid] > 0:
                        logits[tid] = logits[tid] / repetition_penalty
                    else:
                        logits[tid] = logits[tid] * repetition_penalty

        softmax_probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(softmax_probs, k=min(top_k, len(logits)))
        candidates = [
            {
                "token": self.decode_token(idx),
                "prob": round(float(p), 4),
                "id": int(idx),
            }
            for p, idx in zip(topk_probs.tolist(), topk_indices.tolist())
        ]

        if temperature <= 1e-4:
            next_token_id = int(torch.argmax(logits).item())
            sampled_prob = float(softmax_probs[next_token_id].item())
        else:
            scaled_logits = logits / max(temperature, 1e-4)
            filtered_probs = torch.softmax(scaled_logits, dim=-1)
            topk_vals, topk_idxs = torch.topk(filtered_probs, k=min(top_k, len(filtered_probs)))
            topk_vals = topk_vals / topk_vals.sum()
            sample_idx = torch.multinomial(topk_vals, num_samples=1).item()
            next_token_id = int(topk_idxs[sample_idx].item())
            sampled_prob = float(softmax_probs[next_token_id].item())

        sampled_token = self.decode_token(next_token_id)
        with torch.no_grad():
            outputs = self.model(
                input_ids=torch.tensor([[next_token_id]], device=self.device, dtype=torch.long),
                cache_params=state.cache,
                use_cache=True,
            )

        state.token_ids.append(next_token_id)
        state.tokens.append(sampled_token)
        state.cache = self._get_cache(outputs)
        state.last_logits = outputs.logits[:, -1, :].detach()
        state.history_summaries.append(self._snapshot(state.cache, len(state.token_ids)))
        state.step_count += 1
        if next_token_id == getattr(self.tokenizer, "eos_token_id", None):
            state.is_finished = True

        return {
            "step": state.step_count,
            "token": sampled_token,
            "token_id": next_token_id,
            "token_prob": round(sampled_prob, 4),
            "topk_candidates": candidates,
            "is_finished": state.is_finished,
        }

    def export_state(
        self,
        state: MambaAdapterState,
        step: Optional[int] = None,
        layer: int = 0,
        head: Optional[int] = None,
        include_matrix: bool = True,
    ) -> InternalState:
        if not state.history_summaries:
            if state.cache is None:
                raise RuntimeError("Session has no Mamba state. Run prefill() first.")
            state.history_summaries = [self._snapshot(state.cache, len(state.tokens))]

        if step is None or step >= len(state.history_summaries):
            target_step = len(state.history_summaries) - 1
        else:
            target_step = max(0, step)

        snapshot = state.history_summaries[target_step]
        layer = min(max(0, layer), max(0, self.num_layers - 1))
        summary = snapshot["summary"]
        if target_step > 0:
            previous = state.history_summaries[target_step - 1]["summary"]
            delta = [
                round(float(curr) - float(prev), 6)
                for curr, prev in zip(summary, previous)
            ]
        else:
            delta = [0.0] * len(summary)

        components = snapshot["components"]
        seq_len = int(snapshot.get("seq_len", len(state.tokens)))
        cache_bytes = int(snapshot["state_bytes"])
        return InternalState(
            kind="recurrent_state",
            model_type="mamba",
            layer=layer,
            head=None,
            num_layers=self.num_layers,
            num_heads=0,
            seq_len=seq_len,
            step=target_step,
            state_bytes=cache_bytes,
            cache_bytes=cache_bytes,
            viz_history_bytes=0,
            tokens=state.tokens[:seq_len],
            token_ids=state.token_ids[:seq_len],
            shape=[self.num_layers, len(components)],
            matrix=None,
            summary=summary,
            delta=delta,
            head_norms=None,
            meta={
                "model_id": self.model_id,
                "d_inner": self.d_inner,
                "state_size": self.state_size,
                "conv_kernel": self.conv_kernel,
                "components": components,
                "prompt_limit": MAMBA_MAX_PROMPT_TOKENS,
                "max_new_tokens": MAMBA_MAX_NEW_TOKENS,
            },
        )

    def get_recurrent_state(self, state: MambaAdapterState, step: Optional[int] = None, layer: int = 0, head: Optional[int] = None) -> Dict[str, Any]:
        """兼容通用 InferenceSession.get_rwkv_state 的字典形式。"""
        return self.export_state(state, step=step, layer=layer, head=head).model_dump()

    def close_state(self, state: MambaAdapterState) -> None:
        state.cache = None
        state.last_logits = None
        state.history_summaries.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
