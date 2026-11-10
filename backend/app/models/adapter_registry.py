from typing import Dict, List, Type
import threading

from .adapter_base import ModelAdapter


class AdapterRegistry:
    """模型类型名 -> 适配器类 的注册表。"""

    def __init__(self):
        self._adapters: Dict[str, Type[ModelAdapter]] = {}
        self._lock = threading.Lock()

    def register(self, model_type: str, adapter_cls: Type[ModelAdapter]) -> None:
        if not isinstance(adapter_cls, type) or not issubclass(adapter_cls, ModelAdapter):
            raise TypeError(f"{adapter_cls!r} must be a ModelAdapter subclass")
        if not model_type:
            raise ValueError("model_type must not be empty")
        with self._lock:
            self._adapters[model_type] = adapter_cls

    def get(self, model_type: str) -> Type[ModelAdapter]:
        with self._lock:
            adapter_cls = self._adapters.get(model_type)
            available = ", ".join(sorted(self._adapters)) or "<empty>"
        if adapter_cls is None:
            raise ValueError(f"Unknown model type '{model_type}'. Registered adapters: {available}")
        return adapter_cls

    def create(self, model_type: str) -> ModelAdapter:
        return self.get(model_type)()

    def list(self) -> List[str]:
        with self._lock:
            return sorted(self._adapters)


adapter_registry = AdapterRegistry()


def register_adapter(model_type: str):
    """类装饰器：@register_adapter("my-model")。"""
    def decorator(adapter_cls: Type[ModelAdapter]) -> Type[ModelAdapter]:
        adapter_registry.register(model_type, adapter_cls)
        adapter_cls.model_type = model_type
        return adapter_cls
    return decorator


def get_adapter_class(model_type: str) -> Type[ModelAdapter]:
    return adapter_registry.get(model_type)


def create_adapter(model_type: str) -> ModelAdapter:
    return adapter_registry.create(model_type)


def list_adapters() -> List[str]:
    return adapter_registry.list()
