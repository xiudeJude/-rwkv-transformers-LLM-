import asyncio
import os
import sys
import time

import pytest

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.api import concept_routes, routes
from app.api.concept_routes import MemoryGrowthRequest
from app.core.schemas import StepRequest


class _FakeState:
    def model_dump(self):
        return {"kind": "attention"}


class _SlowSession:
    step_count = 1

    def step(self, **kwargs):
        time.sleep(0.4)
        return {
            "step": 1,
            "token": "x",
            "token_id": 1,
            "token_prob": 0.5,
            "is_finished": False,
        }

    def export_state(self, **kwargs):
        return _FakeState()


class _FakeEngine:
    def get_metadata(self):
        return {"model_type": "transformer"}


async def _assert_loop_stays_responsive(coro_factory, threshold=0.2):
    start = time.perf_counter()
    task = asyncio.create_task(coro_factory())
    await asyncio.sleep(0.05)
    elapsed = time.perf_counter() - start
    assert elapsed < threshold, f"event loop was blocked for {elapsed:.3f}s"
    assert not task.done(), "blocking work should still be running in a worker thread"
    return await task


@pytest.mark.asyncio
async def test_session_step_offloads_inference(monkeypatch):
    monkeypatch.setattr(
        routes.session_manager, "get_session", lambda session_id: _SlowSession()
    )

    result = await _assert_loop_stays_responsive(
        lambda: routes.session_step("session-1", StepRequest())
    )

    assert result["session_id"] == "session-1"
    assert result["step"] == 1


@pytest.mark.asyncio
async def test_model_info_offloads_model_load(monkeypatch):
    def slow_get_engine():
        time.sleep(0.4)
        return _FakeEngine()

    monkeypatch.setattr(routes, "get_engine", slow_get_engine)

    result = await _assert_loop_stays_responsive(
        lambda: routes.model_info("transformer")
    )

    assert result == {"model_type": "transformer"}


@pytest.mark.asyncio
async def test_concept_memory_growth_offloads_engine_release(monkeypatch):
    def slow_release():
        time.sleep(0.4)

    monkeypatch.setattr(routes, "release_global_engines", slow_release)
    monkeypatch.setattr(
        concept_routes.concept_lab,
        "run_memory_growth",
        lambda prompt, lengths: {"models": []},
    )

    result = await _assert_loop_stays_responsive(
        lambda: concept_routes.memory_growth(MemoryGrowthRequest(prompt="audit"))
    )

    assert result == {"models": []}
