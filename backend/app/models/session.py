import uuid
import time
from typing import Dict, Any, List, Optional
import torch
from .transformer import TransformerEngine

class InferenceSession:
    """Manages the full lifecycle of an incremental inference session with KV Cache."""

    def __init__(self, session_id: str, engine: TransformerEngine):
        self.session_id: str = session_id
        self.engine: TransformerEngine = engine
        self.past_key_values = None
        self.tokens: List[str] = []
        self.token_ids: List[int] = []
        self.prefill_attentions: Optional[List[torch.Tensor]] = None
        self.step_attentions: List[List[torch.Tensor]] = []
        self.last_logits: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.is_finished: bool = False
        self.created_at: float = time.time()

    def prefill(self, prompt: str, target_layer: int = 3, target_head: int = 0) -> Dict[str, Any]:
        """Runs the initial prefill stage on the complete prompt, initializing past_key_values."""
        if self.engine.model is None or self.engine.tokenizer is None:
            raise RuntimeError("Model is not loaded in TransformerEngine.")

        # 1. Tokenize input prompt
        tokenized = self.engine.tokenize(prompt)
        input_ids = tokenized["input_ids"].to(self.engine.device)
        self.token_ids = tokenized["token_ids"]
        self.tokens = tokenized["tokens"]
        seq_len = len(self.token_ids)

        # 2. Forward pass with use_cache=True and output_attentions=True
        with torch.no_grad():
            outputs = self.engine.model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
                output_attentions=True
            )

        # 3. Store past_key_values and save logits of last position
        self.past_key_values = outputs.past_key_values
        self.last_logits = outputs.logits[:, -1, :].detach()

        # 4. Extract attention tensors across layers
        # outputs.attentions is a tuple of [batch_size, num_heads, seq_len, seq_len]
        if outputs.attentions is None or len(outputs.attentions) == 0:
            raise RuntimeError("Model returned no attention tensors. Ensure attn_implementation='eager' is enabled.")

        # Store all layer attentions in CPU Float16 to keep GPU VRAM footprint minimal
        self.prefill_attentions = [attn[0].detach().cpu().to(torch.float16) for attn in outputs.attentions]


        # Determine effective layer index (mapping to full attention layers if hybrid)
        full_layers = self.engine.full_attn_layers
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
            "session_id": self.session_id,
            "prompt": prompt,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "seq_len": seq_len,
            "layer": target_layer,
            "head": target_head,
            "matrix_shape": list(outputs.attentions[attn_layer_idx].shape),  # [B, H, L, L]
            "min_val": min_val,
            "max_val": max_val,
            "row_sums": row_sums,
            "is_causal": is_causal,
            "matrix": matrix,
            "past_key_values_type": type(self.past_key_values).__name__
        }

    def step(
        self,
        temperature: float = 0.7,
        top_k: int = 10,
        target_layer: int = 3,
        target_head: int = 0
    ) -> Dict[str, Any]:
        """Runs a single decode step using past_key_values and input_ids strictly shaped [1, 1]."""
        if self.is_finished:
            return {
                "step": self.step_count,
                "token": "",
                "token_id": -1,
                "token_prob": 0.0,
                "topk_candidates": [],
                "attention_slice": None,
                "is_finished": True
            }

        if self.last_logits is None or self.past_key_values is None:
            raise RuntimeError("Session has not executed prefill stage. Call prefill() first.")

        # 1. Sample next token from last_logits [1, vocab_size]
        logits = self.last_logits[0].float()
        softmax_probs = torch.softmax(logits, dim=-1)

        # Compute Top-K candidate probabilities for UI visualization
        topk_probs, topk_indices = torch.topk(softmax_probs, k=min(top_k, len(logits)))
        candidates = []
        for p, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            candidates.append({
                "token": self.engine.decode_token(idx),
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

        sampled_token = self.engine.decode_token(next_token_id)
        self.tokens.append(sampled_token)
        self.token_ids.append(next_token_id)
        self.step_count += 1

        # Check EOS token
        eos_id = getattr(self.engine.tokenizer, "eos_token_id", None)
        if next_token_id == eos_id or next_token_id in [248046, 248044]:
            self.is_finished = True

        # 2. Decode forward pass: input_ids strictly [1, 1]
        step_input_ids = torch.tensor([[next_token_id]], device=self.engine.device, dtype=torch.long)

        with torch.no_grad():
            outputs = self.engine.model(
                input_ids=step_input_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
                output_attentions=True
            )

        # 3. Update state in session
        self.past_key_values = outputs.past_key_values
        self.last_logits = outputs.logits[:, -1, :].detach()

        # 4. Store step attentions for all full attention layers in CPU Float16
        step_attns_cpu = [attn[0].detach().cpu().to(torch.float16) for attn in outputs.attentions]
        self.step_attentions.append(step_attns_cpu)

        # 5. Extract incremental attention slice for target layer and head
        full_layers = self.engine.full_attn_layers
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
            "step": self.step_count,
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
            "is_finished": self.is_finished
        }

    def get_attention_detail(self, step: Optional[int] = None, layer: int = 3, head: int = 0) -> Dict[str, Any]:
        """Level 2 On-demand pull: returns full attention matrix and row for the specified step."""
        full_layers = self.engine.full_attn_layers
        if full_layers and layer not in full_layers:
            raise ValueError(f"该层 (Layer {layer}) 为 linear_attn，暂不支持注意力矩阵可视化。仅支持 Full Attention 层：{full_layers}")

        if self.prefill_attentions is None:
            raise RuntimeError("Session has no prefill attention data.")

        attn_layer_idx = full_layers.index(layer) if full_layers else min(layer, len(self.prefill_attentions) - 1)

        # Determine target step
        if step is None or step > self.step_count:
            target_step = self.step_count
        else:
            target_step = max(0, step)

        L_prompt = len(self.prefill_attentions[attn_layer_idx][head])
        L_total = L_prompt + target_step
        tokens_slice = self.tokens[:L_total]

        # Reconstruct [L_total, L_total] causal attention matrix
        matrix: List[List[float]] = []

        # 1. Prefill rows: length L_prompt padded with zeros up to L_total
        prefill_tensor = self.prefill_attentions[attn_layer_idx][head].float()
        for i in range(L_prompt):
            row = prefill_tensor[i].tolist()
            if target_step > 0:
                row = row + [0.0] * target_step
            matrix.append([round(v, 6) for v in row])

        # 2. Decode step rows
        for s_idx in range(target_step):
            step_row = self.step_attentions[s_idx][attn_layer_idx][head, 0, :].float().tolist()
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
            "session_id": self.session_id,
            "step": target_step,
            "layer": layer,
            "head": head,
            "tokens": tokens_slice,
            "seq_len": L_total,
            "attention_row": attn_row,
            "full_matrix": matrix
        }

    def close(self):
        """Explicitly release past_key_values and free GPU VRAM."""
        if self.past_key_values is not None:
            del self.past_key_values
            self.past_key_values = None
        self.prefill_attentions = None
        self.step_attentions = []
        self.last_logits = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SessionManager:
    """Thread-safe registry managing active InferenceSessions."""

    def __init__(self):
        self._sessions: Dict[str, InferenceSession] = {}

    def create_session(self, engine: TransformerEngine) -> InferenceSession:
        session_id = uuid.uuid4().hex[:12]
        session = InferenceSession(session_id, engine)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[InferenceSession]:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
            return True
        return False

    def close_all(self):
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()

session_manager = SessionManager()
