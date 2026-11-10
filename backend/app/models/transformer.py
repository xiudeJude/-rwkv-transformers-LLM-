import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from .adapter_base import ModelAdapter, estimate_state_bytes
from .adapter_registry import register_adapter
from ..core.schemas import (
    ModelMetadata,
    AttentionData,
    TokenCandidate,
    SingleStepInspectResponse,
    InternalState,
)
from ..hooks.base import HookManager
from ..hooks.transformer import TransformerHookExtractor

def resolve_model_path_and_gguf(model_id: str) -> Tuple[str, Optional[str]]:
    actual_path = model_id
    gguf_file = None

    if os.path.exists(model_id):
        if os.path.isfile(model_id) and model_id.endswith(".gguf"):
            gguf_file = os.path.basename(model_id)
            actual_path = os.path.dirname(model_id)
        elif os.path.isdir(model_id):
            candidates = [f for f in os.listdir(model_id) if f.endswith(".gguf") and not f.startswith("mmproj")]
            if candidates:
                gguf_file = candidates[0]
                actual_path = model_id
        return actual_path, gguf_file

    try:
        from modelscope import snapshot_download
        print(f"[ModelLoader] Resolving '{model_id}' via ModelScope...")
        downloaded = snapshot_download(model_id)
        return downloaded, None
    except Exception as e:
        print(f"[ModelLoader] ModelScope resolution failed: {e}. Using raw model_id.")
        return model_id, None


@dataclass
class TransformerAdapterState:
    """单个 Transformer 生成会话的增量状态（KV Cache + 注意力历史）。"""
    past_key_values: Any = None
    tokens: List[str] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    prefill_attentions: Optional[List[torch.Tensor]] = None
    step_attentions: List[List[torch.Tensor]] = field(default_factory=list)
    last_logits: Optional[torch.Tensor] = None
    step_count: int = 0
    is_finished: bool = False


@register_adapter("transformer")
class TransformerAdapter(ModelAdapter):
    """Transformer 因果模型的统一适配器，使用 PyTorch forward hook 提取注意力。"""

    def __init__(self):
        super().__init__()
        self.hook_manager = HookManager()
        self.hook_extractor = TransformerHookExtractor(self.hook_manager)

    def load_model(self, model_name_or_path: str, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_name_or_path
        actual_path, gguf_file = resolve_model_path_and_gguf(model_name_or_path)

        # Determine dtype - explicitly require bfloat16 on CUDA as specified
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        if gguf_file:
            load_kwargs["gguf_file"] = gguf_file

        print(f"[TransformerEngine] Loading tokenizer from: {actual_path} (gguf: {gguf_file})...")
        self.tokenizer = AutoTokenizer.from_pretrained(actual_path, **load_kwargs)

        print(f"[TransformerEngine] Loading model into device {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            actual_path,
            dtype=dtype,
            device_map=self.device,
            attn_implementation="eager",
            **load_kwargs
        )

        self.model.config.output_attentions = True
        self.model.eval()

        # Cache config metadata
        config = self.model.config
        self.num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
        self.num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 0))
        self.hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 0))
        self.vocab_size = getattr(config, "vocab_size", len(self.tokenizer))

        # Check hybrid layer types (e.g. Qwen 3.5)
        layer_types = getattr(config, "layer_types", None)
        if layer_types:
            self.full_attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        else:
            self.full_attn_layers = list(range(self.num_layers))

        # Register extraction hooks
        self.hook_manager.remove_all()
        self.hook_extractor.register_attention_hooks(self.model)
        print(f"[TransformerEngine] Initialized: Layers={self.num_layers}, FullAttnLayers={self.full_attn_layers}, Heads={self.num_heads}")
        self.hook_extractor.register_attention_hooks(self.model)

    def unload(self):
        self.hook_manager.remove_all()
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_type="transformer",
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            device=str(self.device),
            full_attn_layers=self.full_attn_layers
        )

    def tokenize(self, text: str) -> Dict[str, Any]:
        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"]
        token_ids = input_ids[0].tolist()

        # Convert IDs to readable token representations
        tokens = []
        for tid in token_ids:
            decoded = self.tokenizer.decode([tid], skip_special_tokens=False)
            tokens.append(decoded)

        return {
            "input_ids": input_ids,
            "token_ids": token_ids,
            "tokens": tokens,
            "attention_mask": encoding.get("attention_mask", None)
        }

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def create_state(self) -> TransformerAdapterState:
        return TransformerAdapterState()

    def prefill(self, state: TransformerAdapterState, prompt: str, target_layer: int = 3, target_head: int = 0) -> Dict[str, Any]:
        """Runs the initial prefill stage on the complete prompt, initializing past_key_values."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded in TransformerEngine.")

        # 1. Tokenize input prompt
        tokenized = self.tokenize(prompt)
        input_ids = tokenized["input_ids"].to(self.device)
        state.token_ids = tokenized["token_ids"]
        state.tokens = tokenized["tokens"]
        seq_len = len(state.token_ids)

        # 2. Forward pass with use_cache=True and output_attentions=True
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
                output_attentions=True
            )

        # 3. Store past_key_values and save logits of last position
        state.past_key_values = outputs.past_key_values
        state.last_logits = outputs.logits[:, -1, :].detach()

        # 4. Extract attention tensors across layers
        # outputs.attentions is a tuple of [batch_size, num_heads, seq_len, seq_len]
        if outputs.attentions is None or len(outputs.attentions) == 0:
            raise RuntimeError("Model returned no attention tensors. Ensure attn_implementation='eager' is enabled.")

        # Store all layer attentions in CPU Float16 to keep GPU VRAM footprint minimal
        state.prefill_attentions = [attn[0].detach().cpu().to(torch.float16) for attn in outputs.attentions]


        # Determine effective layer index (mapping to full attention layers if hybrid)
        full_layers = self.full_attn_layers
        if full_layers and target_layer in full_layers:
            attn_layer_idx = full_layers.index(target_layer)
        elif full_layers:
            closest_layer = min(full_layers, key=lambda x: abs(x - target_layer))
            attn_layer_idx = full_layers.index(closest_layer)
            target_layer = closest_layer
        else:
            attn_layer_idx = min(target_layer, len(outputs.attentions) - 1)

        # Target attention tensor: [seq_len, seq_len]
        raw_matrix_tensor = outputs.attentions[attn_layer_idx][0, target_head].detach().float().cpu()
        matrix = [[round(val, 6) for val in row] for row in raw_matrix_tensor.tolist()]

        # Compute validation statistics
        min_val = float(raw_matrix_tensor.min().item())
        max_val = float(raw_matrix_tensor.max().item())
        row_sums = [round(float(s), 4) for s in raw_matrix_tensor.sum(dim=-1).tolist()]

        # Verify causal lower triangular mask: strictly 0 above diagonal
        upper_triangle = torch.triu(raw_matrix_tensor, diagonal=1)
        is_causal = bool((upper_triangle == 0).all().item())

        return {
            "session_id": None,
            "prompt": prompt,
            "tokens": state.tokens,
            "token_ids": state.token_ids,
            "seq_len": seq_len,
            "layer": target_layer,
            "head": target_head,
            "matrix_shape": list(outputs.attentions[attn_layer_idx].shape),  # [B, H, L, L]
            "min_val": min_val,
            "max_val": max_val,
            "row_sums": row_sums,
            "is_causal": is_causal,
            "matrix": matrix,
            "past_key_values_type": type(state.past_key_values).__name__
        }

    def step(
        self,
        state: TransformerAdapterState,
        temperature: float = 0.7,
        top_k: int = 10,
        repetition_penalty: float = 1.2,
        target_layer: int = 3,
        target_head: int = 0
    ) -> Dict[str, Any]:
        """Runs a single decode step using past_key_values and input_ids strictly shaped [1, 1]."""
        if state.is_finished:
            return {
                "step": state.step_count,
                "token": "",
                "token_id": -1,
                "token_prob": 0.0,
                "topk_candidates": [],
                "attention_slice": None,
                "is_finished": True
            }

        if state.last_logits is None or state.past_key_values is None:
            raise RuntimeError("Session has not executed prefill stage. Call prefill() first.")

        # 1. Sample next token from last_logits [1, vocab_size]
        logits = state.last_logits[0].clone().float()

        # Apply repetition penalty to already seen token IDs to prevent repetitive loops
        if repetition_penalty != 1.0 and len(state.token_ids) > 0:
            seen_token_ids = set(state.token_ids)
            for tid in seen_token_ids:
                if 0 <= tid < len(logits):
                    if logits[tid] > 0:
                        logits[tid] = logits[tid] / repetition_penalty
                    else:
                        logits[tid] = logits[tid] * repetition_penalty

        softmax_probs = torch.softmax(logits, dim=-1)

        # Compute Top-K candidate probabilities for UI visualization
        topk_probs, topk_indices = torch.topk(softmax_probs, k=min(top_k, len(logits)))
        candidates = []
        for p, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            candidates.append({
                "token": self.decode_token(idx),
                "prob": round(float(p), 4),
                "id": int(idx)
            })

        # Token Sampling
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
        state.tokens.append(sampled_token)
        state.token_ids.append(next_token_id)

        # Check EOS token
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if next_token_id == eos_id or next_token_id in [248046, 248044]:
            state.is_finished = True

        # 2. Decode forward pass: input_ids strictly [1, 1]
        step_input_ids = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)

        with torch.no_grad():
            outputs = self.model(
                input_ids=step_input_ids,
                past_key_values=state.past_key_values,
                use_cache=True,
                output_attentions=True
            )

        # 3. Update state in session
        state.past_key_values = outputs.past_key_values
        state.last_logits = outputs.logits[:, -1, :].detach()

        # 4. Store step attentions for all full attention layers in CPU Float16
        # 先落盘历史，再更新 step_count；避免并发 Level 2 读取看到“计数已增加、
        # 但对应注意力历史尚不存在”的中间状态。
        step_attns_cpu = [attn[0].detach().cpu().to(torch.float16) for attn in outputs.attentions]
        state.step_attentions.append(step_attns_cpu)
        state.step_count += 1

        # 5. Extract incremental attention slice for target layer and head
        full_layers = self.full_attn_layers
        if full_layers and target_layer in full_layers:
            attn_layer_idx = full_layers.index(target_layer)
        elif full_layers:
            closest_layer = min(full_layers, key=lambda x: abs(x - target_layer))
            attn_layer_idx = full_layers.index(closest_layer)
            target_layer = closest_layer
        else:
            attn_layer_idx = min(target_layer, len(outputs.attentions) - 1)

        # Target attention tensor: [1, L_prompt + t]
        raw_slice = outputs.attentions[attn_layer_idx][0, target_head, 0, :].detach().float().cpu()
        slice_weights = [round(val, 6) for val in raw_slice.tolist()]

        return {
            "step": state.step_count,
            "token": sampled_token,
            "token_id": next_token_id,
            "token_prob": round(sampled_prob, 4),
            "topk_candidates": candidates,
            "attention_slice": {
                "layer": target_layer,
                "head": target_head,
                "shape": list(outputs.attentions[attn_layer_idx].shape),
                "weights": slice_weights,
                "slice_len": len(slice_weights)
            },
            "is_finished": state.is_finished
        }

    def _resolve_attention_layer(self, target_layer: int) -> int:
        full_layers = self.full_attn_layers
        if full_layers and target_layer not in full_layers:
            return min(full_layers, key=lambda x: abs(x - target_layer))
        return target_layer

    def get_attention_detail(self, state: TransformerAdapterState, step: Optional[int] = None, layer: int = 3, head: int = 0) -> Dict[str, Any]:
        """Level 2 On-demand pull: returns full attention matrix and row for the specified step."""
        full_layers = self.full_attn_layers
        if full_layers and layer not in full_layers:
            raise ValueError(f"该层 (Layer {layer}) 为 linear_attn，暂不支持注意力矩阵可视化。仅支持 Full Attention 层：{full_layers}")

        if state.prefill_attentions is None:
            raise RuntimeError("Session has no prefill attention data.")

        attn_layer_idx = full_layers.index(layer) if full_layers else min(layer, len(state.prefill_attentions) - 1)

        # Determine target step
        if step is None or step > state.step_count:
            target_step = state.step_count
        else:
            target_step = max(0, step)

        L_prompt = len(state.prefill_attentions[attn_layer_idx][head])
        L_total = L_prompt + target_step
        tokens_slice = state.tokens[:L_total]

        # Reconstruct [L_total, L_total] causal attention matrix
        matrix: List[List[float]] = []

        # 1. Prefill rows: length L_prompt padded with zeros up to L_total
        prefill_tensor = state.prefill_attentions[attn_layer_idx][head].float()
        for i in range(L_prompt):
            row = prefill_tensor[i].tolist()
            if target_step > 0:
                row = row + [0.0] * target_step
            matrix.append([round(v, 6) for v in row])

        # 2. Decode step rows
        for s_idx in range(target_step):
            step_row = state.step_attentions[s_idx][attn_layer_idx][head, 0, :].float().tolist()
            # If step row is shorter than L_total (e.g. earlier step), pad with 0
            if len(step_row) < L_total:
                step_row = step_row + [0.0] * (L_total - len(step_row))
            matrix.append([round(v, 6) for v in step_row[:L_total]])

        # 3. Current step attention row
        if target_step == 0:
            attn_row = matrix[L_prompt - 1]
        else:
            attn_row = matrix[L_prompt + target_step - 1]

        return {
            "session_id": None,
            "step": target_step,
            "layer": layer,
            "head": head,
            "tokens": tokens_slice,
            "seq_len": L_total,
            "attention_row": attn_row,
            "full_matrix": matrix
        }

    def _cache_bytes(self, state: TransformerAdapterState) -> int:
        """推理必需的 KV Cache（混合架构中可能还含线性层状态）实际字节数。"""
        return estimate_state_bytes(state.past_key_values)

    def _viz_history_bytes(self, state: TransformerAdapterState) -> int:
        """仅为可视化额外保存的 CPU 注意力历史字节数。"""
        return (
            estimate_state_bytes(state.prefill_attentions)
            + estimate_state_bytes(state.step_attentions)
        )

    def _state_bytes(self, state: TransformerAdapterState) -> int:
        return self._cache_bytes(state) + self._viz_history_bytes(state)

    def export_state(
        self,
        state: TransformerAdapterState,
        step: Optional[int] = None,
        layer: int = 3,
        head: Optional[int] = None,
        include_matrix: bool = True,
    ) -> InternalState:
        if state.prefill_attentions is None:
            raise RuntimeError("Session has no prefill attention data.")

        target_head = 0 if head is None else head
        effective_layer = self._resolve_attention_layer(layer)

        if include_matrix:
            # prefill/step 对 hybrid 模型会把线性层映射到最近的 full attention 层，
            # 统一状态导出沿用同样的映射；Level 2 显式查询线性层仍由
            # get_attention_detail 抛出 ValueError。
            detail = self.get_attention_detail(state, step=step, layer=effective_layer, head=target_head)
            matrix = detail["full_matrix"]
            row = detail["attention_row"]
            target_step = detail["step"]
            tokens = detail["tokens"]
            seq_len = detail["seq_len"]
            shape = [seq_len, seq_len]
        else:
            # Level 1 只导出当前行，避免在流式消息中携带 O(L^2) 矩阵
            if state.prefill_attentions is None:
                raise RuntimeError("Session has no prefill attention data.")
            full_layers = self.full_attn_layers
            attn_layer_idx = full_layers.index(layer) if (full_layers and layer in full_layers) else (
                full_layers.index(effective_layer) if full_layers else min(effective_layer, len(state.prefill_attentions) - 1)
            )
            if step is None or step > state.step_count:
                target_step = state.step_count
            else:
                target_step = max(0, step)
            L_prompt = len(state.prefill_attentions[attn_layer_idx][target_head])
            L_total = L_prompt + target_step
            if target_step == 0:
                row = state.prefill_attentions[attn_layer_idx][target_head][L_prompt - 1].float().tolist()
            else:
                step_row = state.step_attentions[target_step - 1][attn_layer_idx][target_head, 0, :].float().tolist()
                if len(step_row) < L_total:
                    step_row = step_row + [0.0] * (L_total - len(step_row))
                row = step_row[:L_total]
            row = [round(v, 6) for v in row]
            matrix = None
            tokens = state.tokens[:L_total]
            seq_len = L_total
            shape = [seq_len, seq_len]

        return InternalState(
            kind="attention",
            model_type="transformer",
            layer=effective_layer,
            head=target_head,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            seq_len=seq_len,
            step=target_step,
            state_bytes=self._state_bytes(state),
            cache_bytes=self._cache_bytes(state),
            viz_history_bytes=self._viz_history_bytes(state),
            tokens=tokens,
            token_ids=state.token_ids[:seq_len],
            shape=shape,
            matrix=matrix,
            row=row,
            meta={"model_id": self.model_id, "full_attn_layers": self.full_attn_layers},
        )

    def close_state(self, state: TransformerAdapterState) -> None:
        """Explicitly release past_key_values and free GPU VRAM."""
        if state.past_key_values is not None:
            del state.past_key_values
            state.past_key_values = None
        state.prefill_attentions = None
        state.step_attentions = []
        state.last_logits = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def inspect_single_step(
        self,
        prompt: str,
        layer: int = 0,
        head: int = 0,
        top_k: int = 10,
        temperature: float = 0.7,
    ) -> SingleStepInspectResponse:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been loaded. Call load_model() first.")

        # Validate layer and head bounds
        layer = max(0, min(layer, self.num_layers - 1))
        head = max(0, min(head, self.num_heads - 1))

        # Tokenize
        tokenized = self.tokenize(prompt)
        input_ids = tokenized["input_ids"].to(self.device)
        token_ids = tokenized["token_ids"]
        tokens = tokenized["tokens"]

        # Clear previous hook data
        self.hook_manager.clear_data()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)

        # In hybrid models (like Qwen 3.5), only full_attention layers output softmax attention matrices
        effective_layer = layer
        if self.full_attn_layers and layer not in self.full_attn_layers:
            effective_layer = min(self.full_attn_layers, key=lambda x: abs(x - layer))

        # Retrieve attention tensor from HookManager
        attn_tensor = self.hook_extractor.get_attention_matrix(effective_layer, head)
        if attn_tensor is None and outputs.attentions is not None:
            if self.full_attn_layers and effective_layer in self.full_attn_layers:
                attn_idx = self.full_attn_layers.index(effective_layer)
                if attn_idx < len(outputs.attentions):
                    attn_tensor = outputs.attentions[attn_idx][0, head].detach().float().cpu()
            elif effective_layer < len(outputs.attentions):
                attn_tensor = outputs.attentions[effective_layer][0, head].detach().float().cpu()

        if attn_tensor is None:
            raise ValueError(f"Failed to capture attention matrix for layer {effective_layer}, head {head}")

        # [L, L] matrix converted to Python list of floats with reasonable precision
        matrix_list = [[round(val, 4) for val in row] for row in attn_tensor.tolist()]

        # Compute next-token prediction logits at final position
        last_logits = outputs.logits[0, -1, :].float()
        scaled_logits = last_logits / max(temperature, 1e-4)
        probs = F.softmax(scaled_logits, dim=-1)

        # Top-K candidate extraction
        topk_probs, topk_indices = torch.topk(probs, k=min(top_k, self.vocab_size))
        candidates: List[TokenCandidate] = []
        for p, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            candidates.append(TokenCandidate(
                token=self.decode_token(idx),
                prob=round(p, 4),
                id=idx
            ))

        next_best = candidates[0] if candidates else TokenCandidate(token="", prob=0.0, id=0)

        row_sums = [round(sum(row), 4) for row in matrix_list]
        state = InternalState(
            kind="attention",
            model_type="transformer",
            layer=effective_layer,
            head=head,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            seq_len=len(tokens),
            step=0,
            state_bytes=estimate_state_bytes(matrix_list),
            cache_bytes=0,
            viz_history_bytes=estimate_state_bytes(matrix_list),
            tokens=tokens,
            token_ids=token_ids,
            shape=[len(matrix_list), len(matrix_list[0]) if matrix_list else 0],
            matrix=matrix_list,
            row=matrix_list[-1] if matrix_list else [],
            meta={
                "min_val": float(attn_tensor.min().item()),
                "max_val": float(attn_tensor.max().item()),
                "row_sums": row_sums,
            },
        )

        return SingleStepInspectResponse(
            status="success",
            prompt=prompt,
            model_meta=self.get_metadata(),
            attention=AttentionData(
                layer=effective_layer,
                head=head,
                matrix=matrix_list,
                tokens=tokens,
                token_ids=token_ids
            ),
            next_token=next_best,
            top_k_candidates=candidates,
            state=state,
        )


# Backward-compatible alias: existing imports/tests continue to use TransformerEngine.
TransformerEngine = TransformerAdapter
