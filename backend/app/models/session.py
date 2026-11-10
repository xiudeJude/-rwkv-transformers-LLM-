import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Type
import torch

from .adapter_base import ModelAdapter
from ..core.config import settings


class InferenceSession:
    """通用推理会话包装器。

    所有模型相关的增量状态（KV Cache、递归状态、注意力历史等）都由
    ModelAdapter.create_state() 创建并由适配器维护；本类只负责会话生命周期、
    时间戳以及与旧接口保持兼容的属性转发。
    """

    _LOCAL_ATTRS = {
        "session_id",
        "adapter",
        "engine",
        "_state",
        "created_at",
        "last_accessed_at",
    }

    def __init__(self, session_id: str, adapter: ModelAdapter):
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "adapter", adapter)
        # Backward-compatible attribute: old code referenced session.engine.model etc.
        object.__setattr__(self, "engine", adapter)
        object.__setattr__(self, "_state", adapter.create_state())
        object.__setattr__(self, "created_at", time.time())
        object.__setattr__(self, "last_accessed_at", self.created_at)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        state = self.__dict__.get("_state")
        if state is not None and hasattr(state, name):
            return getattr(state, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value):
        # 方法/类属性（例如测试中 monkeypatch 的 close）必须留在会话对象上，
        # 其余状态字段转发给适配器状态对象。
        if (
            name in self._LOCAL_ATTRS
            or "_state" not in self.__dict__
            or hasattr(type(self), name)
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(self._state, name, value)

    def prefill(self, prompt: str, target_layer: int = 3, target_head: int = 0) -> Dict[str, Any]:
        self.last_accessed_at = time.time()
        result = self.adapter.prefill(
            self._state,
            prompt,
            target_layer=target_layer,
            target_head=target_head,
        )
        result["session_id"] = self.session_id
        return result

    def step(
        self,
        temperature: float = 0.7,
        top_k: int = 10,
        repetition_penalty: float = 1.2,
        target_layer: int = 3,
        target_head: int = 0,
    ) -> Dict[str, Any]:
        self.last_accessed_at = time.time()
        return self.adapter.step(
            self._state,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            target_layer=target_layer,
            target_head=target_head,
        )

    def get_attention_detail(self, step: Optional[int] = None, layer: int = 3, head: int = 0) -> Dict[str, Any]:
        self.last_accessed_at = time.time()
        result = self.adapter.get_attention_detail(
            self._state,
            step=step,
            layer=layer,
            head=head,
        )
        result["session_id"] = self.session_id
        return result

    def get_rwkv_state(self, step: Optional[int] = None, layer: int = 0, head: Optional[int] = None) -> Dict[str, Any]:
        self.last_accessed_at = time.time()
        result = self.adapter.get_recurrent_state(
            self._state,
            step=step,
            layer=layer,
            head=head,
        )
        result["session_id"] = self.session_id
        return result

    def export_state(
        self,
        step: Optional[int] = None,
        layer: int = 0,
        head: Optional[int] = None,
        include_matrix: bool = True,
    ):
        return self.adapter.export_state(
            self._state,
            step=step,
            layer=layer,
            head=head,
            include_matrix=include_matrix,
        )

    def close(self):
        """Explicitly release the adapter-owned incremental state and free GPU VRAM."""
        self.adapter.close_state(self._state)


class SessionManager:
    """Thread-safe registry managing active InferenceSessions with TTL and LRU capacity eviction."""

    session_cls: Type[InferenceSession] = InferenceSession

    def __init__(self, ttl_seconds: Optional[int] = None, max_sessions: Optional[int] = None):
        self._sessions: Dict[str, InferenceSession] = {}
        self._lock = threading.Lock()
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.SESSION_TTL_SECONDS
        self.max_sessions = max_sessions if max_sessions is not None else settings.MAX_SESSIONS

    def _cleanup_expired_locked(self, now: float):
        """Evicts sessions that have exceeded the TTL limit."""
        expired_ids = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_accessed_at) > self.ttl_seconds
        ]
        for sid in expired_ids:
            sess = self._sessions.pop(sid, None)
            if sess:
                sess.close()

    def _evict_lru_locked(self, max_allowed: int):
        """Evicts the least recently accessed sessions when the session count exceeds max_allowed."""
        if len(self._sessions) > max_allowed:
            sorted_sessions = sorted(self._sessions.items(), key=lambda item: item[1].last_accessed_at)
            num_to_evict = len(self._sessions) - max_allowed
            for sid, sess in sorted_sessions[:num_to_evict]:
                self._sessions.pop(sid, None)
                sess.close()

    def create_session(self, engine: ModelAdapter) -> InferenceSession:
        now = time.time()
        with self._lock:
            # 1. Clean up expired sessions first
            self._cleanup_expired_locked(now)
            # 2. Evict LRU session if capacity limit is reached
            if len(self._sessions) >= self.max_sessions:
                self._evict_lru_locked(self.max_sessions - 1)

            session_id = uuid.uuid4().hex[:12]
            session = self.session_cls(session_id, engine)
            session.last_accessed_at = now
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[InferenceSession]:
        now = time.time()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if (now - session.last_accessed_at) > self.ttl_seconds:
                self._sessions.pop(session_id, None)
                session.close()
                return None
            session.last_accessed_at = now
            return session

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.close()
                return True
            return False

    def close_all(self):
        with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


session_manager = SessionManager()
