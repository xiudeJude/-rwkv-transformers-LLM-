from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn

class HookHandleWrapper:
    """Encapsulates a PyTorch Hook handle with safe removal and storage."""
    def __init__(self, name: str, module: nn.Module, hook_fn: Callable):
        self.name = name
        self.module = module
        self.handle = module.register_forward_hook(hook_fn)
        self.captured_data: Optional[Any] = None

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

class HookManager:
    """Manages forward and backward hooks across arbitrary model architectures."""
    def __init__(self):
        self._hooks: Dict[str, HookHandleWrapper] = {}

    def register_forward_hook(self, name: str, module: nn.Module, hook_fn: Callable) -> HookHandleWrapper:
        # Remove existing if already registered under same name
        if name in self._hooks:
            self._hooks[name].remove()

        wrapper = HookHandleWrapper(name, module, hook_fn)
        self._hooks[name] = wrapper
        return wrapper

    def get_data(self, name: str) -> Optional[Any]:
        if name in self._hooks:
            return self._hooks[name].captured_data
        return None

    def clear_data(self):
        for wrapper in self._hooks.values():
            wrapper.captured_data = None

    def remove_hook(self, name: str):
        if name in self._hooks:
            self._hooks[name].remove()
            del self._hooks[name]

    def remove_all(self):
        for wrapper in self._hooks.values():
            wrapper.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_data()
