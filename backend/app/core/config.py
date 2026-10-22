import os
from typing import Optional
from dotenv import load_dotenv
import torch

# 定位 backend 根目录并加载 .env 配置文件
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(BACKEND_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

class Settings:
    APP_NAME: str = "LLM Interpretability Workshop API"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    
    # 默认模型相对路径 (位于 backend/models/ 下)
    _DEFAULT_TRANSFORMER_PATH = os.path.join(BACKEND_DIR, "models", "Qwen3.5-0.8B-GGUF")
    _DEFAULT_RWKV_PATH = os.path.join(BACKEND_DIR, "models", "rwkv7-g1d-0.4b-20260210-ctx8192.pth")

    @property
    def DEFAULT_TRANSFORMER(self) -> str:
        """Transformer 权重路径，优先从 DEFAULT_TRANSFORMER 环境变量获取，支持相对路径与绝对路径"""
        path = os.getenv("DEFAULT_TRANSFORMER", self._DEFAULT_TRANSFORMER_PATH)
        if not os.path.isabs(path) and not path.startswith("http"):
            resolved = os.path.abspath(os.path.join(BACKEND_DIR, path))
            if os.path.exists(resolved):
                return resolved
        return path

    @property
    def DEFAULT_RWKV(self) -> str:
        """RWKV 权重路径，优先从 DEFAULT_RWKV 环境变量获取，支持相对路径与绝对路径"""
        path = os.getenv("DEFAULT_RWKV", self._DEFAULT_RWKV_PATH)
        if not os.path.isabs(path) and not path.startswith("http"):
            resolved = os.path.abspath(os.path.join(BACKEND_DIR, path))
            if os.path.exists(resolved) or os.path.exists(resolved + ".pth"):
                return resolved
        return path

    # HF Mirror fallback
    HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    
    @property
    def DEVICE(self) -> str:
        override = os.getenv("DEVICE", "")
        if override:
            return override
        return "cuda" if torch.cuda.is_available() else "cpu"

settings = Settings()

# Setup HF endpoint if not explicitly configured in environment
if "HF_ENDPOINT" not in os.environ and settings.HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT
