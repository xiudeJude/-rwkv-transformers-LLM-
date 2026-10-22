import os
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from main import app

GHOST_ID = "00000000-0000-0000-0000-000000000000"


def test_transformer_close_is_idempotent():
    """Closing an already-released session must return 200, not 404.

    Real-world trigger: the WebSocket handler releases the session on disconnect,
    then the frontend calls POST /close as a safety net. Before this fix the
    second call surfaced a 404 in the console on every architecture switch.
    """
    print("[1] Transformer close on unknown session...", flush=True)
    client = TestClient(app)

    resp = client.post(f"/api/session/{GHOST_ID}/close")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "closed"
    assert body["session_id"] == GHOST_ID
    assert body["already_closed"] is True
    print("  -> 200 with already_closed=true", flush=True)

    # A second call must behave identically (retry-safe).
    resp2 = client.post(f"/api/session/{GHOST_ID}/close")
    assert resp2.status_code == 200
    assert resp2.json()["already_closed"] is True
    print("  -> repeat call also 200 (retry-safe)", flush=True)


def test_rwkv_close_is_idempotent():
    print("[2] RWKV close on unknown session...", flush=True)
    client = TestClient(app)

    resp = client.post(f"/api/rwkv/session/{GHOST_ID}/close")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["status"] == "closed"
    assert body["already_closed"] is True
    print("  -> 200 with already_closed=true", flush=True)


def test_close_does_not_break_live_session_contract():
    """The response schema must stay superset-compatible with the old one."""
    print("[3] Response schema compatibility...", flush=True)
    client = TestClient(app)
    body = client.post(f"/api/session/{GHOST_ID}/close").json()
    assert set(body.keys()) >= {"status", "session_id"}
    assert body["status"] == "closed"
    print("  -> legacy keys (status, session_id) preserved", flush=True)


if __name__ == "__main__":
    test_transformer_close_is_idempotent()
    test_rwkv_close_is_idempotent()
    test_close_does_not_break_live_session_contract()
    print("\nAll idempotent-close tests passed.", flush=True)
