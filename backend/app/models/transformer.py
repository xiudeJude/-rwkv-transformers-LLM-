import os
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseInferenceEngine
from ..core.schemas import ModelMetadata, AttentionData, TokenCandidate, SingleStepInspectResponse
from ..hooks.base import HookManager
from ..hooks.transformer import TransformerHookExtractor

def resolve_model_path(model_id: str) -> str:
    if os.path.exists(model_id):
        return model_id
    try:
        from modelscope import snapshot_download
        print(f"[ModelLoader] Resolving '{model_id}' via ModelScope...")
        return snapshot_download(model_id)
    except Exception as e:
        print(f"[ModelLoader] ModelScope resolution failed: {e}. Using raw model_id.")
        return model_id

class TransformerEngine(BaseInferenceEngine):
    """Engine implementation for Transformer-based causal models using PyTorch forward hooks."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_id = ""
        self.hook_manager = HookManager()
        self.hook_extractor = TransformerHookExtractor(self.hook_manager)
        self.num_layers = 0
        self.num_heads = 0
        self.hidden_size = 0
        self.vocab_size = 0

    def load_model(self, model_name_or_path: str, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_name_or_path
        actual_path = resolve_model_path(model_name_or_path)

        # Determine dtype
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_path,
            trust_remote_code=True
        )

        # Force eager attention implementation to expose attention weights
        self.model = AutoModelForCausalLM.from_pretrained(
            actual_path,
            dtype=dtype,
            device_map=self.device,
            attn_implementation="eager",
            trust_remote_code=True
        )

        self.model.config.output_attentions = True
        self.model.eval()

        # Cache config metadata
        config = self.model.config
        self.num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
        self.num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 0))
        self.hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 0))
        self.vocab_size = getattr(config, "vocab_size", len(self.tokenizer))

        # Register extraction hooks on all layers
        self.hook_manager.remove_all()
        self.hook_extractor.register_attention_hooks(self.model)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.model_id,
            model_type="transformer",
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            device=str(self.device)
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

        # Retrieve attention tensor from HookManager
        attn_tensor = self.hook_extractor.get_attention_matrix(layer, head)
        if attn_tensor is None:
            # Fallback to output.attentions if hook intercepted differently
            if outputs.attentions is not None and len(outputs.attentions) > layer:
                attn_tensor = outputs.attentions[layer][0, head].detach().float().cpu()
            else:
                raise ValueError(f"Failed to capture attention matrix for layer {layer}, head {head}")

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

        return SingleStepInspectResponse(
            status="success",
            prompt=prompt,
            model_meta=self.get_metadata(),
            attention=AttentionData(
                layer=layer,
                head=head,
                matrix=matrix_list,
                tokens=tokens,
                token_ids=token_ids
            ),
            next_token=next_best,
            top_k_candidates=candidates
        )
