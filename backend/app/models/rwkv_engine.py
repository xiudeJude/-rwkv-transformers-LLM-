import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import torch

from .adapter_base import ModelAdapter, estimate_state_bytes
from .adapter_registry import register_adapter
from ..core.schemas import ModelMetadata, InternalState

# Explicitly enable RWKV-7 architecture support
os.environ["RWKV_V7_ON"] = "1"


@dataclass
class RWKVAdapterState:
    """单个 RWKV 生成会话的递归状态与历史快照。"""
    tokens: List[str] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    history_states: List[List[torch.Tensor]] = field(default_factory=list)
    history_summaries: Dict[int, List[List[float]]] = field(default_factory=dict)
    current_state: Optional[List[torch.Tensor]] = None
    last_logits: Optional[torch.Tensor] = None
    step_count: int = 0
    is_finished: bool = False


@register_adapter("rwkv")
class RWKVAdapter(ModelAdapter):
    """Inference engine implementation for RWKV-7 (Goose) models using the official rwkv package."""

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.num_layers = 24
        self.num_heads = 16
        self.head_size = 64
        self.hidden_size = 1024
        self.vocab_size = 65536

    def load_model(self, model_name_or_path: str, device: Optional[str] = None):
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_name_or_path

        # rwkv package automatically appends '.pth' if omitted, or strips if present
        clean_path = model_name_or_path
        if clean_path.endswith(".pth"):
            clean_path = clean_path[:-4]

        strategy = "cuda fp16" if self.device == "cuda" and torch.cuda.is_available() else "cpu fp32"
        print(f"[RWKVEngine] Loading RWKV-7 model: {clean_path} with strategy '{strategy}'...")
        self.model = RWKV(model=clean_path, strategy=strategy)

        print("[RWKVEngine] Initializing PIPELINE tokenizer (rwkv_vocab_v20230424)...")
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")

        # Extract config metadata
        self.num_layers = getattr(self.model.args, "n_layer", 24)
        self.hidden_size = getattr(self.model.args, "n_embd", 1024)
        self.head_size = getattr(self.model.args, "head_size", 64)
        self.num_heads = self.hidden_size // self.head_size
        self.vocab_size = getattr(self.model.args, "vocab_size", 65536)
        print(f"[RWKVEngine] Model ready: {self.num_layers} layers, {self.num_heads} heads ({self.head_size}d), hidden_size={self.hidden_size}.")

    def unload(self):
        self.model = None
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode(self, prompt: str) -> List[int]:
        if self.pipeline is None:
            raise RuntimeError("RWKVEngine pipeline is not loaded.")
        return self.pipeline.encode(prompt)

    def decode_token(self, token_id: int) -> str:
        if self.pipeline is None:
            raise RuntimeError("RWKVEngine pipeline is not loaded.")
        return self.pipeline.decode([token_id])

    def decode(self, token_ids: List[int]) -> str:
        if self.pipeline is None:
            raise RuntimeError("RWKVEngine pipeline is not loaded.")
        return self.pipeline.decode(token_ids)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.decode(token_ids)

    def forward(self, tokens_or_id, state: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.model is None:
            raise RuntimeError("RWKVEngine model is not loaded.")
        out, next_state = self.model.forward(tokens_or_id, state)
        return out, next_state

    def tokenize(self, text: str) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("RWKVEngine pipeline is not loaded.")
        token_ids = self.pipeline.encode(text)
        tokens = [self.decode_token(tid) for tid in token_ids]
        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "input_ids": torch.tensor([token_ids], device=self.device)
        }

    def create_state(self) -> RWKVAdapterState:
        return RWKVAdapterState()

    def prefill(self, state: RWKVAdapterState, prompt: str, target_layer: int = 0, target_head: int = 0) -> Dict[str, Any]:
        """Runs the initial prefill stage on the complete prompt, initializing RWKV state."""
        if self.model is None or self.pipeline is None:
            raise RuntimeError("Model is not loaded in RWKVEngine.")

        # 1. Encode prompt
        prompt_token_ids = self.encode(prompt)
        if not prompt_token_ids:
            prompt_token_ids = [187]  # newline fallback if prompt is empty

        state.token_ids = list(prompt_token_ids)
        state.tokens = [self.decode_token(tid) for tid in prompt_token_ids]
        seq_len = len(state.token_ids)

        # 2. Forward pass with initial state None
        with torch.no_grad():
            outputs, s0 = self.forward(state.token_ids, None)

        state.last_logits = outputs.detach().float()
        state.current_state = s0
        # Strategy A: store s0 directly on GPU in history
        state.history_states = [s0]

        # 3. Compute 16-dimensional Frobenius norm summary for target layer
        target_layer = min(max(0, target_layer), self.num_layers - 1)
        summary_0 = self._compute_layer_summary(s0, target_layer)
        state.history_summaries[target_layer] = [summary_0]

        delta_0 = [0.0] * len(summary_0)

        return {
            "session_id": None,
            "prompt": prompt,
            "tokens": state.tokens,
            "token_ids": state.token_ids,
            "seq_len": seq_len,
            "layer": target_layer,
            "state_summary": summary_0,
            "delta": delta_0,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads
        }

    def step(
        self,
        state: RWKVAdapterState,
        temperature: float = 0.7,
        top_k: int = 10,
        repetition_penalty: float = 1.2,
        target_layer: int = 0,
        target_head: int = 0
    ) -> Dict[str, Any]:
        """Executes a single decode step in the RWKV autoregressive cycle."""
        if state.is_finished:
            return {
                "step": state.step_count,
                "token": "",
                "token_id": -1,
                "token_prob": 0.0,
                "topk_candidates": [],
                "topk_logits": [],
                "state_summary": [0.0] * 16,
                "delta": [0.0] * 16,
                "is_finished": True
            }

        if state.last_logits is None or state.current_state is None:
            raise RuntimeError("Session has not executed prefill stage. Call prefill() first.")

        # 1. Sample next token from last_logits [vocab_size]
        logits = state.last_logits.clone().float()
        topk_logits = self.top_logits_payload(logits, k=50)

        # Apply repetition penalty to seen token IDs
        if repetition_penalty != 1.0 and len(state.token_ids) > 0:
            seen_ids = set(state.token_ids)
            for tid in seen_ids:
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

        # Check EOS token (0 is <|endoftext|> in RWKV)
        if next_token_id == 0:
            state.is_finished = True

        # 2. Decode forward pass: single token id
        with torch.no_grad():
            out, next_state = self.forward(next_token_id, state.current_state)

        state.last_logits = out.detach().float()
        state.current_state = next_state
        # Strategy A: save state directly to history list
        state.history_states.append(next_state)

        # 3. Compute 16D summary and Delta for target layer
        target_layer = min(max(0, target_layer), self.num_layers - 1)
        curr_summary = self._compute_layer_summary(next_state, target_layer)

        # Retrieve or compute previous step summary for this layer
        if target_layer not in state.history_summaries:
            # First time requesting this layer: compute previous step summary from history_states[-2]
            prev_summary = self._compute_layer_summary(state.history_states[-2], target_layer)
            state.history_summaries[target_layer] = [prev_summary]
        else:
            prev_summary = state.history_summaries[target_layer][-1]

        delta = [round(c - p, 6) for c, p in zip(curr_summary, prev_summary)]
        state.history_summaries[target_layer].append(curr_summary)

        # 先落盘状态/摘要历史，再更新 step_count；避免并发读取看到计数超前。
        state.step_count += 1

        return {
            "step": state.step_count,
            "token": sampled_token,
            "token_id": next_token_id,
            "token_prob": round(sampled_prob, 4),
            "topk_candidates": candidates,
            "topk_logits": topk_logits,
            "state_summary": curr_summary,
            "delta": delta,
            "layer": target_layer,
            "is_finished": state.is_finished
        }

    def _compute_layer_summary(self, state: List[torch.Tensor], layer: int) -> List[float]:
        """Computes the 16-dimensional Frobenius norm summary vector for the specified layer's state matrix."""
        # Layer wkv_state is located at index layer * 3 + 1, shape: [16, 64, 64]
        state_idx = layer * 3 + 1
        if state_idx >= len(state):
            raise ValueError(f"Layer index {layer} out of range (max layer: {len(state)//3 - 1})")
        wkv_state = state[state_idx]
        # Frobenius norm for each of the 16 heads
        head_norms = torch.norm(wkv_state, p='fro', dim=(-2, -1))
        return [round(float(val), 6) for val in head_norms.tolist()]

    def get_recurrent_state(self, state: RWKVAdapterState, step: Optional[int] = None, layer: int = 0, head: Optional[int] = None) -> Dict[str, Any]:
        """Level 2 On-demand pull: returns the state matrix for the specified step, layer, and optional head."""
        if not state.history_states:
            raise RuntimeError("Session has no state history. Run prefill() first.")

        layer = min(max(0, layer), self.num_layers - 1)

        # Determine target step
        if step is None or step >= len(state.history_states):
            target_step = len(state.history_states) - 1
        else:
            target_step = max(0, step)

        target_state = state.history_states[target_step]
        wkv_state_tensor = target_state[layer * 3 + 1].detach().float().cpu()

        # Compute head norms (16 heads)
        head_norms = torch.norm(wkv_state_tensor, p='fro', dim=(-2, -1))
        head_norms_list = [round(float(v), 6) for v in head_norms.tolist()]

        if head is not None:
            head = min(max(0, head), self.num_heads - 1)
            matrix = wkv_state_tensor[head].tolist()
            shape = list(wkv_state_tensor[head].shape)  # [64, 64]
        else:
            matrix = wkv_state_tensor.tolist()
            shape = list(wkv_state_tensor.shape)  # [16, 64, 64]

        return {
            "session_id": None,
            "step": target_step,
            "layer": layer,
            "head": head,
            "tokens": state.tokens,
            "shape": shape,
            "head_norms": head_norms_list,
            "matrix": matrix
        }

    def export_state(
        self,
        state: RWKVAdapterState,
        step: Optional[int] = None,
        layer: int = 0,
        head: Optional[int] = None,
        include_matrix: bool = True,
    ) -> InternalState:
        if not state.history_states:
            raise RuntimeError("Session has no state history. Run prefill() first.")

        layer = min(max(0, layer), self.num_layers - 1)
        if step is None or step >= len(state.history_states):
            target_step = len(state.history_states) - 1
        else:
            target_step = max(0, step)

        wkv_state_tensor = state.history_states[target_step][layer * 3 + 1].detach().float().cpu()
        head_norms = torch.norm(wkv_state_tensor, p='fro', dim=(-2, -1))
        head_norms_list = [round(float(v), 6) for v in head_norms.tolist()]

        if head is not None:
            head = min(max(0, head), self.num_heads - 1)
            selected = wkv_state_tensor[head]
            matrix = selected.tolist() if include_matrix else None
            shape = list(selected.shape)  # [64, 64]
        else:
            selected = wkv_state_tensor
            matrix = selected.tolist() if include_matrix else None
            shape = list(selected.shape)  # [16, 64, 64]

        cache_bytes = estimate_state_bytes(state.current_state)
        history_bytes = estimate_state_bytes(state.history_states)
        viz_history_bytes = max(0, history_bytes - cache_bytes)
        state_bytes = cache_bytes + viz_history_bytes

        if target_step > 0:
            prev_summary = self._compute_layer_summary(state.history_states[target_step - 1], layer)
            delta = [round(c - p, 6) for c, p in zip(head_norms_list, prev_summary)]
        else:
            delta = [0.0] * len(head_norms_list)

        return InternalState(
            kind="recurrent_state",
            model_type="rwkv",
            layer=layer,
            head=head,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            seq_len=len(state.tokens),
            step=target_step,
            state_bytes=state_bytes,
            cache_bytes=cache_bytes,
            viz_history_bytes=viz_history_bytes,
            tokens=state.tokens,
            token_ids=state.token_ids,
            shape=shape,
            matrix=matrix,
            summary=head_norms_list,
            delta=delta,
            head_norms=head_norms_list,
            meta={"model_id": self.model_id, "head_size": self.head_size, "history_steps": len(state.history_states)},
        )

    def close_state(self, state: RWKVAdapterState) -> None:
        """Explicitly release all state tensors and free GPU VRAM."""
        state.history_states.clear()
        state.history_summaries.clear()
        state.current_state = None
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
    ) -> Any:
        from ..core.schemas import SingleStepInspectResponse, AttentionData, TokenCandidate
        tokenized = self.tokenize(prompt)
        token_ids = tokenized["token_ids"]
        out, state = self.forward(token_ids, None)
        logits = out.float().cpu()
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=min(top_k, len(logits)))

        candidates = [
            TokenCandidate(token=self.decode_token(idx), prob=round(float(p), 4), id=int(idx))
            for p, idx in zip(topk_probs.tolist(), topk_indices.tolist())
        ]
        next_tok = candidates[0] if candidates else TokenCandidate(token="", prob=0.0, id=-1)

        # Frobenius norm for layer head
        wkv_state = state[layer * 3 + 1]
        norm = float(torch.norm(wkv_state[head], p='fro').item())
        cache_bytes = estimate_state_bytes(state)

        state_export = InternalState(
            kind="recurrent_state",
            model_type="rwkv",
            layer=layer,
            head=head,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            seq_len=len(token_ids),
            step=0,
            state_bytes=cache_bytes,
            cache_bytes=cache_bytes,
            viz_history_bytes=0,
            tokens=tokenized["tokens"],
            token_ids=token_ids,
            shape=[1, 1],
            matrix=[[norm]],
            summary=[norm],
            delta=[0.0],
            head_norms=[norm],
            meta={"model_id": self.model_id, "head_size": self.head_size},
        )

        return SingleStepInspectResponse(
            status="success",
            prompt=prompt,
            model_meta=self.get_metadata(),
            attention=AttentionData(
                layer=layer,
                head=head,
                matrix=[[norm]],
                tokens=tokenized["tokens"],
                token_ids=token_ids
            ),
            next_token=next_tok,
            top_k_candidates=candidates,
            state=state_export,
        )

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_type="rwkv",
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            device=self.device,
            full_attn_layers=[]
        )


# Backward-compatible alias: existing imports/tests continue to use RWKVEngine.
RWKVEngine = RWKVAdapter
