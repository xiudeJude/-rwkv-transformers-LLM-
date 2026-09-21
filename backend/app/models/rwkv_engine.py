import os
from typing import Dict, Any, List, Optional, Tuple
import torch

from .base import BaseInferenceEngine
from ..core.schemas import ModelMetadata

# Explicitly enable RWKV-7 architecture support
os.environ["RWKV_V7_ON"] = "1"

class RWKVEngine(BaseInferenceEngine):
    """Inference engine implementation for RWKV-7 (Goose) models using the official rwkv package."""

    def __init__(self):
        self.model = None
        self.pipeline = None
        self.device = "cpu"
        self.model_id = ""
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
            top_k_candidates=candidates
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
