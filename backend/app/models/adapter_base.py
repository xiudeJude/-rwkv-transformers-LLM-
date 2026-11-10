from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

import torch

from ..core.schemas import InternalState, ModelMetadata


def estimate_state_bytes(value: Any) -> int:
    """估算张量/容器实际占用的字节数，供统一状态元信息使用。

    只递归已知的安全容器和张量，遇到模型对象等复杂对象时返回 0，避免
    在导出状态时误遍历整个计算图。
    """
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, (list, tuple)):
        return sum(estimate_state_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(estimate_state_bytes(item) for item in value.values())

    # HuggingFace DynamicCache 等缓存对象：只读取明确的 key/value 缓存字段
    total = 0
    for attr in ("key_cache", "value_cache"):
        cache = getattr(value, attr, None)
        if isinstance(cache, (list, tuple)):
            total += estimate_state_bytes(cache)
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
