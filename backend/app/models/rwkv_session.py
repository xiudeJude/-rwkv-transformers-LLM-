from typing import List

import torch

from .session import InferenceSession, SessionManager


class RWKVSession(InferenceSession):
    """RWKV 会话兼容包装器；递推状态与历史由 RWKVAdapter 维护。"""

    def _compute_layer_summary(self, state: List[torch.Tensor], layer: int) -> List[float]:
        """兼容旧脚本：计算指定层 WKV 状态的头范数摘要。"""
        return self.adapter._compute_layer_summary(state, layer)


class RWKVSessionManager(SessionManager):
    """Thread-safe registry managing active RWKVSessions with TTL and LRU capacity eviction."""

    session_cls = RWKVSession


rwkv_session_manager = RWKVSessionManager()
