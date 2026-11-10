import os
import sys
import threading

import pytest
from fastapi.testclient import TestClient

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.api import concept_routes, routes
from app.services import concept_lab
from app.services.model_gate import model_gate
from main import app


def test_concept_memory_growth_rejects_concurrent_request(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def fake_collect(model_type, prompt, requested_lengths):
        started.set()
        assert release.wait(timeout=5), "test did not release the first experiment"
        return {
            "model_type": model_type,
            "available": False,
            "error": "fake",
            "measurements": [],
            "theoretical": {"available": False, "values": []},
        }

    monkeypatch.setattr(concept_lab, "_collect_growth", fake_collect)
    monkeypatch.setattr(concept_lab, "MEMORY_MODEL_TYPES", ("transformer",))

    thread = threading.Thread(
        target=concept_lab.run_memory_growth,
        args=("concurrency audit",),
    )
    thread.start()
    try:
        assert started.wait(timeout=2), "first experiment did not start"
        with pytest.raises(concept_lab.ModelBusyError, match="正在运行"):
            concept_lab.run_memory_growth("concurrency audit")
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert not model_gate.locked()


def test_memory_growth_busy_returns_409(monkeypatch):
    monkeypatch.setattr(routes, "release_global_engines", lambda: None)

    def busy(prompt, lengths):
        raise concept_lab.ModelBusyError("概念实验室正在运行其他模型任务，请稍后重试。")

    monkeypatch.setattr(concept_routes.concept_lab, "run_memory_growth", busy)

    client = TestClient(app)
    response = client.post(
        "/api/concept/memory-growth",
        json={"prompt": "concurrency audit"},
    )

    assert response.status_code == 409
    assert "正在运行" in response.json()["detail"]


def test_model_info_busy_returns_409(monkeypatch):
    monkeypatch.setattr(routes, "_engine", None)
    client = TestClient(app)

    with model_gate.hold():
        response = client.get("/api/model/info")

    assert response.status_code == 409
    assert "稍后重试" in response.json()["detail"]
