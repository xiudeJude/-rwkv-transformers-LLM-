import sys
import os
import time

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from main import app
from app.models.session import session_manager
from app.models.rwkv_session import rwkv_session_manager

def test_transformer_websocket_stop():
    print("\n--- [Test 1] Transformer WebSocket Stop Signal Handling (asyncio.to_thread) ---", flush=True)
    client = TestClient(app)
    prompt = "测试异步WebSocket"

    create_resp = client.post("/api/session/create", json={"prompt": prompt, "layer": 3, "head": 0})
    assert create_resp.status_code == 200, f"Create session failed: {create_resp.text}"
    session_id = create_resp.json()["session_id"]
    print(f"Created Transformer session: {session_id}", flush=True)

    tokens_received = 0
    with client.websocket_connect(f"/ws/session/{session_id}/stream") as ws:
        # Request 30 tokens
        ws.send_json({
            "action": "start",
            "temperature": 0.7,
            "top_k": 10,
            "max_new_tokens": 30,
            "layer": 3,
            "head": 0
        })

        # Receive 1 token then send stop
        while tokens_received < 1:
            msg = ws.receive_json()
            if msg.get("type") == "token_step":
                tokens_received += 1
                print(f"  Received step {msg.get('step')}: '{msg.get('token')}'", flush=True)

        print("  Sending stop signal during active stream...", flush=True)
        start_stop = time.time()
        ws.send_json({"action": "stop"})

        # Collect messages until closed
        remaining = 0
        while True:
            try:
                msg = ws.receive_json()
                if msg.get("type") == "token_step":
                    remaining += 1
                    print(f"  Remaining step after stop: {msg.get('step')}", flush=True)
            except Exception as e:
                print(f"  WebSocket closed gracefully: {type(e).__name__}", flush=True)
                break

        stop_elapsed = time.time() - start_stop
        print(f"  Stream stopped in {stop_elapsed:.3f}s, tokens received after stop: {remaining}", flush=True)
        total = tokens_received + remaining
        assert total < 30, f"Expected generation to stop well before 30 tokens, but received {total}"
        print(f"  [PASS] Early termination verified! Total tokens: {total} < 30", flush=True)

    time.sleep(0.1)
    session_after = session_manager.get_session(session_id)
    assert session_after is None, "Expected session to be closed and cleaned up from session_manager"
    print("  [PASS] Session cleanup verified", flush=True)

def test_rwkv_websocket_stop():
    print("\n--- [Test 2] RWKV WebSocket Stop Signal Handling (asyncio.to_thread) ---", flush=True)
    client = TestClient(app)
    prompt = "测试RWKV异步流式停止"

    create_resp = client.post("/api/rwkv/session/create", json={"prompt": prompt, "layer": 0})
    assert create_resp.status_code == 200, f"Create RWKV session failed: {create_resp.text}"
    session_id = create_resp.json()["session_id"]
    print(f"Created RWKV session: {session_id}", flush=True)

    tokens_received = 0
    with client.websocket_connect(f"/ws/rwkv/session/{session_id}/stream") as ws:
        ws.send_json({
            "action": "start",
            "temperature": 0.7,
            "top_k": 10,
            "max_new_tokens": 30,
            "layer": 0
        })

        while tokens_received < 1:
            msg = ws.receive_json()
            if msg.get("type") == "token_step":
                tokens_received += 1
                print(f"  Received RWKV step {msg.get('step')}: '{msg.get('token')}'", flush=True)

        print("  Sending stop signal during RWKV stream...", flush=True)
        start_stop = time.time()
        ws.send_json({"action": "stop"})

        remaining = 0
        while True:
            try:
                msg = ws.receive_json()
                if msg.get("type") == "token_step":
                    remaining += 1
                    print(f"  Remaining RWKV step after stop: {msg.get('step')}", flush=True)
            except Exception as e:
                print(f"  RWKV WebSocket closed gracefully: {type(e).__name__}", flush=True)
                break

        stop_elapsed = time.time() - start_stop
        print(f"  RWKV Stream stopped in {stop_elapsed:.3f}s, tokens received after stop: {remaining}", flush=True)
        total = tokens_received + remaining
        assert total < 30, f"Expected generation to stop well before 30 tokens, got {total}"
        print(f"  [PASS] RWKV Early termination verified! Total tokens: {total} < 30", flush=True)

    time.sleep(0.1)
    session_after = rwkv_session_manager.get_session(session_id)
    assert session_after is None, "Expected RWKV session to be closed and cleaned up"
    print("  [PASS] RWKV Session cleanup verified", flush=True)

if __name__ == "__main__":
    test_transformer_websocket_stop()
    test_rwkv_websocket_stop()
    print("\n[ALL TESTS PASSED] WebSocket asyncio.to_thread and stop handling verified successfully!", flush=True)
