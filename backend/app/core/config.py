import os
from typing import Optional
import torch

class Settings:
    APP_NAME: str = "LLM Interpretability Workshop API"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Models
    DEFAULT_TRANSFORMER: str = os.getenv("DEFAULT_TRANSFORMER", "Qwen/Qwen2.5-0.5B-Instruct")
    DEFAULT_RWKV: str = os.getenv("DEFAULT_RWKV", "RWKV/v5-Eagle-0.4B-HF")
    
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
