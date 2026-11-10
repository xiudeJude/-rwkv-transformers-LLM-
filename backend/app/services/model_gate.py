import threading
from contextlib import contextmanager


class ModelBusyError(RuntimeError):
    """另一个模型加载/实验任务正在运行。"""


class ModelGate:
    """进程内非阻塞互斥锁，保护会加载或卸载模型的接口。"""

    def __init__(self):
        self._lock = threading.Lock()

    @contextmanager
    def hold(self):
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("模型正在运行其他加载/实验任务，请稍后重试。")
        try:
            yield
        finally:
            self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


model_gate = ModelGate()
