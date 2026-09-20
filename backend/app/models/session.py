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
        self.prefill_attentions = [attn.detach().cpu().to(torch.float16) for attn in outputs.attentions]

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

    def close(self):
        """Explicitly release past_key_values and free GPU VRAM."""
        if self.past_key_values is not None:
            del self.past_key_values
            self.past_key_values = None
        self.prefill_attentions = None
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
