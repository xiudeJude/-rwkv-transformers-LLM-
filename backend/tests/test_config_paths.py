import os
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core import config as config_module
from app.core.config import Settings


def test_relative_model_paths_resolve_under_backend_root(monkeypatch, tmp_path):
    """相对路径应基于 backend 根目录解析，不依赖开发机的 D:/ 绝对路径。"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    transformer_path = models_dir / "transformer-model"
    transformer_path.mkdir()
    rwkv_path = models_dir / "rwkv-model"
    rwkv_path.with_suffix(".pth").write_bytes(b"")
    mamba_path = models_dir / "mamba-model"
    mamba_path.mkdir()

    monkeypatch.setattr(config_module, "BACKEND_DIR", str(tmp_path))
    monkeypatch.setenv("DEFAULT_TRANSFORMER", "models/transformer-model")
    monkeypatch.setenv("DEFAULT_RWKV", "models/rwkv-model")
    monkeypatch.setenv("DEFAULT_MAMBA", "models/mamba-model")

    settings = Settings()
    assert settings.DEFAULT_TRANSFORMER == str(transformer_path)
    assert settings.DEFAULT_RWKV == str(rwkv_path)
    assert settings.DEFAULT_MAMBA == str(mamba_path)
    assert not settings.DEFAULT_TRANSFORMER.startswith(("D:/", "D:\\"))


def test_absolute_model_paths_still_supported(monkeypatch, tmp_path):
    absolute_model = tmp_path / "absolute-model"
    absolute_model.mkdir()
    monkeypatch.setenv("DEFAULT_TRANSFORMER", str(absolute_model))

    settings = Settings()
    assert settings.DEFAULT_TRANSFORMER == str(absolute_model)
