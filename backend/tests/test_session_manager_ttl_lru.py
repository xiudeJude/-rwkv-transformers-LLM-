import os
import sys
import time
from unittest.mock import MagicMock

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from main import app
from app.models.session import SessionManager, InferenceSession
from app.models.rwkv_session import RWKVSessionManager, RWKVSession


def test_transformer_session_ttl_and_lru():
    print("\n--- [Test 1] Transformer SessionManager TTL & LRU Eviction ---", flush=True)
    engine_mock = MagicMock()
    # Manager configured with TTL=1.0s and MAX_SESSIONS=2
    mgr = SessionManager(ttl_seconds=1, max_sessions=2)

    # 1. Create two sessions
    s1 = mgr.create_session(engine_mock)
    s2 = mgr.create_session(engine_mock)
    assert mgr.active_count == 2
    assert mgr.get_session(s1.session_id) is not None
    assert mgr.get_session(s2.session_id) is not None
    print("  -> Created 2 sessions within capacity: OK", flush=True)

    # 2. Test LRU eviction when exceeding max_sessions=2
    # Refresh s1 access time so s2 becomes the least recently used
    time.sleep(0.05)
    _ = mgr.get_session(s1.session_id)
    time.sleep(0.05)

    # Mock close method to verify GPU VRAM cleanup is called
    s2_close_mock = MagicMock(wraps=s2.close)
    s2.close = s2_close_mock

    # Create 3rd session -> should evict s2
    s3 = mgr.create_session(engine_mock)
    assert mgr.active_count == 2, f"Active count should remain <= 2, got {mgr.active_count}"
    assert mgr.get_session(s2.session_id) is None, "s2 should have been evicted by LRU"
    assert mgr.get_session(s1.session_id) is not None, "s1 should still be active"
    assert mgr.get_session(s3.session_id) is not None, "s3 should be active"
    assert s2_close_mock.called, "s2.close() must be called to release GPU VRAM"
    print("  -> LRU eviction properly evicted oldest session (s2) and invoked close(): OK", flush=True)

    # 3. Test TTL expiration
    s3_close_mock = MagicMock(wraps=s3.close)
    s3.close = s3_close_mock
    # Advance time artificially by modifying last_accessed_at past TTL
    s3.last_accessed_at = time.time() - 10.0

    assert mgr.get_session(s3.session_id) is None, "Expired session should return None"
    assert s3_close_mock.called, "Expired session must have close() called upon detection"
    assert mgr.active_count == 1, "Active count should decrease after TTL eviction"
    print("  -> TTL expiration properly cleaned up expired session (s3) and invoked close(): OK", flush=True)


def test_rwkv_session_ttl_and_lru():
    print("\n--- [Test 2] RWKVSessionManager TTL & LRU Eviction ---", flush=True)
    engine_mock = MagicMock()
    # Manager configured with TTL=1.0s and MAX_SESSIONS=2
    mgr = RWKVSessionManager(ttl_seconds=1, max_sessions=2)

    # 1. Create two sessions
    r1 = mgr.create_session(engine_mock)
    r2 = mgr.create_session(engine_mock)
    assert mgr.active_count == 2
    assert mgr.get_session(r1.session_id) is not None
    assert mgr.get_session(r2.session_id) is not None
    print("  -> Created 2 RWKV sessions within capacity: OK", flush=True)

    # 2. Test LRU eviction
    time.sleep(0.05)
    _ = mgr.get_session(r1.session_id)
    time.sleep(0.05)

    r2_close_mock = MagicMock(wraps=r2.close)
    r2.close = r2_close_mock

    # Create r3 -> evicts r2
    r3 = mgr.create_session(engine_mock)
    assert mgr.active_count == 2
    assert mgr.get_session(r2.session_id) is None, "r2 should have been evicted by LRU"
    assert mgr.get_session(r1.session_id) is not None
    assert mgr.get_session(r3.session_id) is not None
    assert r2_close_mock.called, "r2.close() must be called to release GPU VRAM"
    print("  -> RWKV LRU eviction properly evicted oldest session (r2) and invoked close(): OK", flush=True)

    # 3. Test TTL expiration
    r3_close_mock = MagicMock(wraps=r3.close)
    r3.close = r3_close_mock
    r3.last_accessed_at = time.time() - 10.0

    assert mgr.get_session(r3.session_id) is None
    assert r3_close_mock.called, "r3.close() must be called to release GPU VRAM"
    assert mgr.active_count == 1
    print("  -> RWKV TTL expiration properly cleaned up expired session (r3) and invoked close(): OK", flush=True)


def test_api_expired_session_error_message():
    print("\n--- [Test 3] API Error Detail 'not found or expired' Verification ---", flush=True)
    client = TestClient(app)

    # 1. Non-existent / expired Transformer session
    resp = client.post("/api/session/non_existent_123/step", json={"layer": 3, "head": 0})
    assert resp.status_code == 404
    detail = resp.json().get("detail", "")
    assert "not found or expired" in detail, f"Expected 'not found or expired' in detail, got: {detail}"
    print(f"  -> Transformer 404 message: '{detail}' (verified)", flush=True)

    # 2. Non-existent / expired RWKV session
    resp_rwkv = client.post("/api/rwkv/session/non_existent_456/step", json={"layer": 0, "head": 0})
    assert resp_rwkv.status_code == 404
    detail_rwkv = resp_rwkv.json().get("detail", "")
    assert "not found or expired" in detail_rwkv, f"Expected 'not found or expired' in detail, got: {detail_rwkv}"
    print(f"  -> RWKV 404 message: '{detail_rwkv}' (verified)", flush=True)


if __name__ == "__main__":
    test_transformer_session_ttl_and_lru()
    test_rwkv_session_ttl_and_lru()
    test_api_expired_session_error_message()
    print("\nALL SESSION TTL & LRU TESTS PASSED!")
