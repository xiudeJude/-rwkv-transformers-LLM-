from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from .base import HookManager, HookHandleWrapper

class TransformerHookExtractor:
    """Manages attention extraction hooks for Transformer decoder layers."""
    def __init__(self, hook_manager: HookManager):
        self.hook_manager = hook_manager

    def register_attention_hooks(self, model: nn.Module):
        """Finds decoder layers and attaches attention-capturing forward hooks."""
        # Locate decoder layers (compatible with Qwen2, LLaMA, Mistral, GPT-2)
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        elif hasattr(model, "layers"):
            layers = model.layers

        if layers is None:
            raise ValueError("Unable to locate Transformer decoder layers in the provided model.")

        for layer_idx, layer_module in enumerate(layers):
            attn_module = None
            if hasattr(layer_module, "self_attn"):
                attn_module = layer_module.self_attn
            elif hasattr(layer_module, "attn"):
                attn_module = layer_module.attn

            if attn_module is not None:
                hook_name = f"attn_layer_{layer_idx}"
                # Define hook capturing closure
                wrapper = None
                def make_hook(target_name: str):
                    def hook_fn(module, input_args, output):
                        # output can be tuple: (attn_output, attn_weights, ...)
                        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                            target_wrapper = self.hook_manager._hooks.get(target_name)
                            if target_wrapper:
                                target_wrapper.captured_data = output[1].detach().float().cpu()
                    return hook_fn

                self.hook_manager.register_forward_hook(hook_name, attn_module, make_hook(hook_name))

    def get_attention_matrix(self, layer_idx: int, head_idx: int) -> Optional[torch.Tensor]:
        """Returns [seq_len, seq_len] tensor for the specified layer and head."""
        data = self.hook_manager.get_data(f"attn_layer_{layer_idx}")
        if data is not None and isinstance(data, torch.Tensor):
            # Shape is [batch, heads, q_len, kv_len]
            # Squeeze batch 0
            if data.dim() == 4:
                return data[0, head_idx]
            elif data.dim() == 3:
                return data[head_idx]
        return None
