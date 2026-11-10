from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

import torch

from ..core.schemas import InternalState, ModelMetadata


_CACHE_FIELD_NAMES = (
    "key_cache",
    "value_cache",
    "keys",
    "values",
    "cache",
    "layers",
    "states",
    "past_key_values",
)
_CACHE_METADATA_FIELDS = {
    "config",
    "generation_config",
    "layer_class_to_replicate",
    "dtype",
    "device",
}
_MAX_CACHE_DEPTH = 8


def estimate_state_bytes(value: Any) -> int:
    """估算张量/容器/缓存对象实际占用的字节数，供统一状态元信息使用。

    除了张量和 list/tuple/dict，递归访问缓存对象的公开字段，以兼容不同
    transformers 版本的缓存布局（例如 DynamicCache.layers[i].keys/values）。
    对 nn.Module、类对象和元数据字段做剪枝，避免误遍历模型计算图。
    """
    return _estimate_state_bytes(value, set(), 0)


def _estimate_state_bytes(value: Any, seen: set, depth: int) -> int:
    if value is None:
        return 0

    if isinstance(value, torch.Tensor):
        oid = id(value)
        if oid in seen:
            return 0
        seen.add(oid)
        return int(value.numel() * value.element_size())

    if isinstance(value, (str, bytes, int, float, bool)):
        return 0

    if depth >= _MAX_CACHE_DEPTH:
        return 0

    if isinstance(value, (list, tuple)):
        oid = id(value)
        if oid in seen:
            return 0
        seen.add(oid)
        return sum(_estimate_state_bytes(item, seen, depth + 1) for item in value)

    if isinstance(value, dict):
        oid = id(value)
        if oid in seen:
            return 0
        seen.add(oid)
        return sum(_estimate_state_bytes(item, seen, depth + 1) for item in value.values())

    if isinstance(value, type) or isinstance(value, torch.nn.Module):
        return 0

    oid = id(value)
    if oid in seen:
        return 0
    seen.add(oid)

    total = 0
    # 新版缓存（如 DynamicCache）通常把张量放在实例 __dict__ 的 layers 等字段中
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        for field, child in attributes.items():
            if field.startswith("_") or field in _CACHE_METADATA_FIELDS:
                continue
            if isinstance(child, type) or callable(child):
                continue
            total += _estimate_state_bytes(child, seen, depth + 1)

    # 兼容使用 __slots__ 或属性代理的缓存对象
    for field in _CACHE_FIELD_NAMES:
        if not hasattr(value, field):
            continue
        child = getattr(value, field)
        total += _estimate_state_bytes(child, seen, depth + 1)
    return total


class ModelAdapter(ABC):
    """统一模型适配层基类。

    一个适配器实例对应一份已加载的模型；每个生成会话通过
    create_state() 获得独立的、适配器私有的增量状态对象。这样同一份
    权重可以服务多个互不干扰的 KV Cache / 递归状态会话。
    """

    model_type: ClassVar[str] = "unknown"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_id = ""
        self.num_layers = 0
        self.num_heads = 0
        self.hidden_size = 0
        self.vocab_size = 0
        self.full_attn_layers: List[int] = []

    # ------------------------------------------------------------------
    # 模型生命周期
    # ------------------------------------------------------------------
    @abstractmethod
    def load_model(self, model_name_or_path: str, device: Optional[str] = None):
        """加载模型与分词器。"""

    @abstractmethod
    def unload(self):
        """卸载模型并释放显存/句柄。"""

    def is_loaded(self) -> bool:
        return self.model is not None

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """返回模型结构元信息。"""

    # ------------------------------------------------------------------
    # 分词 / 反分词
    # ------------------------------------------------------------------
    @abstractmethod
    def tokenize(self, text: str) -> Dict[str, Any]:
        """将文本编码为 token 列表和模型输入张量。"""

    @abstractmethod
    def decode_token(self, token_id: int) -> str:
        """将单个 token id 解码为文本。"""

    def detokenize(self, token_ids: List[int]) -> str:
        """将 token id 序列解码为文本；默认逐 token 拼接。"""
        return "".join(self.decode_token(token_id) for token_id in token_ids)

    def top_logits_payload(self, logits: torch.Tensor, k: int = 50) -> List[Dict[str, Any]]:
        """返回当前步原始 logits 的 top-k 统一格式，供概念实验室本地重算。"""
        flat = logits.reshape(-1).float()
        k = min(k, int(flat.numel()))
        values, indices = torch.topk(flat, k=k)
        return [
            {
                "id": int(token_id),
                "token": self.decode_token(int(token_id)),
                "logit": round(float(logit), 6),
            }
            for logit, token_id in zip(values.tolist(), indices.tolist())
        ]

    # ------------------------------------------------------------------
    # 会话状态与逐步生成
    # ------------------------------------------------------------------
    @abstractmethod
    def create_state(self) -> Any:
        """创建一个新的会话增量状态对象（KV Cache / 递归状态等）。"""

    @abstractmethod
    def prefill(self, state: Any, prompt: str, target_layer: int = 0, target_head: int = 0) -> Dict[str, Any]:
        """执行 prefill，初始化会话状态并返回与旧接口兼容的结果。"""

    @abstractmethod
    def step(
        self,
        state: Any,
        temperature: float = 0.7,
        top_k: int = 10,
        repetition_penalty: float = 1.2,
        target_layer: int = 0,
        target_head: int = 0,
    ) -> Dict[str, Any]:
        """执行一步自回归生成（包含采样与状态更新）。"""

    @abstractmethod
    def export_state(
        self,
        state: Any,
        step: Optional[int] = None,
        layer: int = 0,
        head: Optional[int] = None,
        include_matrix: bool = True,
    ) -> InternalState:
        """导出统一结构的内部状态。"""

    @abstractmethod
    def close_state(self, state: Any) -> None:
        """释放单个会话状态占用的资源。"""
