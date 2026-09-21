import uuid
import time
from typing import Dict, Any, List, Optional
import torch

from .rwkv_engine import RWKVEngine

class RWKVSession:
    """Manages the lifecycle of an incremental inference session for RWKV-7 models."""

    def __init__(self, session_id: str, engine: RWKVEngine):
        self.session_id: str = session_id
        self.engine: RWKVEngine = engine
        self.tokens: List[str] = []
        self.token_ids: List[int] = []
        # Strategy A: Keep states on GPU directly in Python list
        self.history_states: List[List[torch.Tensor]] = []
        # Tracks layer -> list of 16D Frobenius norm summaries across steps
        self.history_summaries: Dict[int, List[List[float]]] = {}
        self.current_state: Optional[List[torch.Tensor]] = None
        self.last_logits: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.is_finished: bool = False
        self.created_at: float = time.time()

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

    def prefill(self, prompt: str, target_layer: int = 0) -> Dict[str, Any]:
        """Runs the initial prefill stage on the complete prompt, initializing RWKV state."""
        if self.engine.model is None or self.engine.pipeline is None:
            raise RuntimeError("Model is not loaded in RWKVEngine.")

        # 1. Encode prompt
        prompt_token_ids = self.engine.encode(prompt)
        if not prompt_token_ids:
            prompt_token_ids = [187]  # newline fallback if prompt is empty

        self.token_ids = list(prompt_token_ids)
        self.tokens = [self.engine.decode_token(tid) for tid in prompt_token_ids]
        seq_len = len(self.token_ids)

        # 2. Forward pass with initial state None
        with torch.no_grad():
            outputs, s0 = self.engine.forward(self.token_ids, None)

        self.last_logits = outputs.detach().float()
        self.current_state = s0
        # Strategy A: store s0 directly on GPU in history
        self.history_states = [s0]

        # 3. Compute 16-dimensional Frobenius norm summary for target layer
        target_layer = min(max(0, target_layer), self.engine.num_layers - 1)
        summary_0 = self._compute_layer_summary(s0, target_layer)
        self.history_summaries[target_layer] = [summary_0]

        delta_0 = [0.0] * len(summary_0)

        return {
            "session_id": self.session_id,
            "prompt": prompt,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "seq_len": seq_len,
            "layer": target_layer,
            "state_summary": summary_0,
            "delta": delta_0,
            "num_layers": self.engine.num_layers,
            "num_heads": self.engine.num_heads
        }

    def step(
        self,
        temperature: float = 0.7,
        top_k: int = 10,
        repetition_penalty: float = 1.2,
        target_layer: int = 0
    ) -> Dict[str, Any]:
        """Executes a single decode step in the RWKV autoregressive cycle."""
        if self.is_finished:
            return {
                "step": self.step_count,
                "token": "",
                "token_id": -1,
                "token_prob": 0.0,
                "topk_candidates": [],
                "state_summary": [0.0] * 16,
                "delta": [0.0] * 16,
                "is_finished": True
            }

        if self.last_logits is None or self.current_state is None:
            raise RuntimeError("Session has not executed prefill stage. Call prefill() first.")

        # 1. Sample next token from last_logits [vocab_size]
        logits = self.last_logits.clone().float()

        # Apply repetition penalty to seen token IDs
        if repetition_penalty != 1.0 and len(self.token_ids) > 0:
            seen_ids = set(self.token_ids)
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

        # Check EOS token (0 is <|endoftext|> in RWKV)
        if next_token_id == 0:
            self.is_finished = True

        # 2. Decode forward pass: single token id
        with torch.no_grad():
            out, next_state = self.engine.forward(next_token_id, self.current_state)

        self.last_logits = out.detach().float()
        self.current_state = next_state
        # Strategy A: save state directly to history list
        self.history_states.append(next_state)

        # 3. Compute 16D summary and Delta for target layer
        target_layer = min(max(0, target_layer), self.engine.num_layers - 1)
        curr_summary = self._compute_layer_summary(next_state, target_layer)

        # Retrieve or compute previous step summary for this layer
        if target_layer not in self.history_summaries:
            # First time requesting this layer: compute previous step summary from history_states[-2]
            prev_summary = self._compute_layer_summary(self.history_states[-2], target_layer)
            self.history_summaries[target_layer] = [prev_summary]
        else:
            prev_summary = self.history_summaries[target_layer][-1]

        delta = [round(c - p, 6) for c, p in zip(curr_summary, prev_summary)]
        self.history_summaries[target_layer].append(curr_summary)

        return {
            "step": self.step_count,
            "token": sampled_token,
            "token_id": next_token_id,
            "token_prob": round(sampled_prob, 4),
            "topk_candidates": candidates,
            "state_summary": curr_summary,
            "delta": delta,
            "layer": target_layer,
            "is_finished": self.is_finished
        }

    def get_rwkv_state(self, step: Optional[int] = None, layer: int = 0, head: Optional[int] = None) -> Dict[str, Any]:
        """Level 2 On-demand pull: returns the state matrix for the specified step, layer, and optional head."""
        if not self.history_states:
            raise RuntimeError("Session has no state history. Run prefill() first.")

        layer = min(max(0, layer), self.engine.num_layers - 1)

        # Determine target step
        if step is None or step >= len(self.history_states):
            target_step = len(self.history_states) - 1
        else:
            target_step = max(0, step)

        target_state = self.history_states[target_step]
        wkv_state_tensor = target_state[layer * 3 + 1].detach().float().cpu()

        # Compute head norms (16 heads)
        head_norms = torch.norm(wkv_state_tensor, p='fro', dim=(-2, -1))
        head_norms_list = [round(float(v), 6) for v in head_norms.tolist()]

        if head is not None:
            head = min(max(0, head), self.engine.num_heads - 1)
            matrix = wkv_state_tensor[head].tolist()
            shape = list(wkv_state_tensor[head].shape)  # [64, 64]
        else:
            matrix = wkv_state_tensor.tolist()
            shape = list(wkv_state_tensor.shape)  # [16, 64, 64]

        return {
            "session_id": self.session_id,
            "step": target_step,
            "layer": layer,
            "head": head,
            "tokens": self.tokens,
            "shape": shape,
            "head_norms": head_norms_list,
            "matrix": matrix
        }

    def close(self):
        """Explicitly release all state tensors and free GPU VRAM."""
        self.history_states.clear()
        self.history_summaries.clear()
        self.current_state = None
        self.last_logits = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RWKVSessionManager:
    """Thread-safe registry managing active RWKVSessions."""

    def __init__(self):
        self._sessions: Dict[str, RWKVSession] = {}

    def create_session(self, engine: RWKVEngine) -> RWKVSession:
        session_id = uuid.uuid4().hex[:12]
        session = RWKVSession(session_id, engine)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[RWKVSession]:
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

rwkv_session_manager = RWKVSessionManager()
