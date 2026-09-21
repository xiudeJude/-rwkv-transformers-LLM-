import sys
import os
import time
import json
import torch

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from main import app
from app.models.session import session_manager

def run_suite():
    print("=" * 80)
    print("  P0-Step3: WebSocket 流式推流与 rAF 渲染队列三项边界验证")
    print("=" * 80)

    client = TestClient(app)

    # -------------------------------------------------------------
    # 验证 1: 50 步 Token 流式生成与到达时间间隔 / 渲染队列节奏实测
    # -------------------------------------------------------------
    print("\n>>> [验证 1] 运行 50 步 Token 流式推流并采集时间间隔数据...")
    prompt = "注意力机制让大语言模型能够精确捕捉长距离语义依赖关系"

    create_resp = client.post("/api/session/create", json={"prompt": prompt, "layer": 3, "head": 0})
    assert create_resp.status_code == 200, f"Create session failed: {create_resp.text}"
    session_data = create_resp.json()
    session_id = session_data["session_id"]
    print(f"  [OK] 会话创建成功: session_id={session_id}")

    arrival_records = []
    tokens_text = []

    # Connect WebSocket
    with client.websocket_connect(f"/ws/session/{session_id}/stream") as ws:
        # Send start command
        start_payload = {
            "action": "start",
            "temperature": 0.55,
            "top_k": 10,
            "max_new_tokens": 50,
            "layer": 3,
            "head": 0
        }
        ws.send_json(start_payload)

        prev_arrival_time = None
        step_idx = 0

        # Simulated queue state
        simulated_queue = []
        last_consumer_time = time.perf_counter() * 1000

        while True:
            msg = ws.receive_json()
            now_ms = time.perf_counter() * 1000

            if prev_arrival_time is None:
                delta_arrive = 0.0
            else:
                delta_arrive = now_ms - prev_arrival_time
            prev_arrival_time = now_ms

            if msg.get("type") == "error":
                raise RuntimeError(f"WebSocket received error: {msg}")

            step = msg.get("step")
            tok = msg.get("token", "")
            is_fin = msg.get("is_finished", False)

            # Strict Level 1 check: NO heavy matrices
            assert "matrix" not in msg, "CRITICAL: Level 1 message leaked matrix!"
            assert "full_matrix" not in msg, "CRITICAL: Level 1 message leaked full_matrix!"
            assert "attention_row" not in msg, "CRITICAL: Level 1 message leaked attention_row!"

            tokens_text.append(tok)
            simulated_queue.append((step, tok, now_ms))
            backlog = len(simulated_queue)

            # Adaptive pace check: if backlog > 5, drain target is 15ms, else 40ms
            target_pace = 15 if backlog > 5 else 40

            # Simulate draining if time elapsed
            elapsed_consumer = now_ms - last_consumer_time
            drained_count = 0
            while simulated_queue and (elapsed_consumer >= target_pace):
                simulated_queue.pop(0)
                elapsed_consumer -= target_pace
                drained_count += 1
                last_consumer_time = now_ms

            arrival_records.append({
                "step": step,
                "token": tok,
                "delta_arrive_ms": round(delta_arrive, 2),
                "queue_backlog": backlog,
                "target_pace_ms": target_pace,
                "is_finished": is_fin
            })

            step_idx += 1
            if is_fin or step_idx >= 50:
                break

        # Send close action
        try:
            ws.send_json({"action": "close"})
        except Exception:
            pass

    # Print summary statistics
    print(f"\n  [实测生成文本]: {''.join(tokens_text)}")
    print(f"  [生成 Token 总数]: {len(arrival_records)} 步")

    deltas = [r["delta_arrive_ms"] for r in arrival_records[1:]]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    min_delta = min(deltas) if deltas else 0.0
    max_delta = max(deltas) if deltas else 0.0
    max_backlog = max(r["queue_backlog"] for r in arrival_records)

    print(f"  [Token 到达间隔统计]: 平均={avg_delta:.2f}ms, 最小={min_delta:.2f}ms, 最大={max_delta:.2f}ms")
    print(f"  [最大队列积压]: {max_backlog} tokens")

    print("\n  [部分步数时间数据采样 (前 10 步与后 5 步)]:")
    sample_records = arrival_records[:10] + arrival_records[-5:]
    print(f"  {'Step':<6} | {'Token':<10} | {'Δt 到达(ms)':<12} | {'队列积压':<8} | {'调度目标(ms)':<10} | {'Finished':<8}")
    print("  " + "-" * 65)
    for r in sample_records:
        tok_repr = repr(r['token'])
        print(f"  {r['step']:<6} | {tok_repr:<10} | {r['delta_arrive_ms']:<12.2f} | {r['queue_backlog']:<8} | {r['target_pace_ms']:<10} | {str(r['is_finished']):<8}")

    assert len(arrival_records) == 50, f"Expected 50 tokens, got {len(arrival_records)}"
    print("  --> [验证 1 通过]: 50 步流式生成完整，Level 1 隔离无泄漏，时间间隔采集正常！")

    # -------------------------------------------------------------
    # 验证 2: 模拟中途断开 WebSocket，验证 Session 销毁与显存释放
    # -------------------------------------------------------------
    print("\n>>> [验证 2] 模拟中途断开 WebSocket，验证 GPU 显存回收与 Session 资源销毁...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        base_vram = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  [当前基准 GPU 显存]: {base_vram:.2f} MB")
    else:
        base_vram = 0.0
        print("  [当前环境]: CPU 模式")

    # Create new session
    resp2 = client.post("/api/session/create", json={"prompt": "测试中途断开连接的显存释放逻辑", "layer": 3, "head": 0})
    s2_id = resp2.json()["session_id"]
    assert session_manager.get_session(s2_id) is not None, "Session 2 should be in manager"

    if torch.cuda.is_available():
        vram_after_create = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  [创建 Session 2 后显存]: {vram_after_create:.2f} MB (增加了 {vram_after_create - base_vram:.2f} MB)")

    # Connect WebSocket, receive 5 tokens, then forcibly disconnect
    with client.websocket_connect(f"/ws/session/{s2_id}/stream") as ws2:
        ws2.send_json({"action": "start", "max_new_tokens": 100, "layer": 3, "head": 0})
        for _ in range(5):
            msg = ws2.receive_json()
        print("  [模拟断开]: 接收 5 个 token 后主动强制关闭 WebSocket...")
        # Exiting context manager closes WebSocket cleanly from client side

    # Give backend a moment to process disconnect and close session
    time.sleep(0.3)

    session_after_disconnect = session_manager.get_session(s2_id)
    print(f"  [SessionManager 检查]: session_manager.get_session('{s2_id}') = {session_after_disconnect}")
    assert session_after_disconnect is None, "Session should be removed from manager after disconnect!"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_after_disconnect = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  [断开后 GPU 显存]: {vram_after_disconnect:.2f} MB (净增量: {vram_after_disconnect - base_vram:+.2f} MB)")
        assert (vram_after_disconnect - base_vram) < 5.0, "VRAM should be reclaimed after disconnect!"

    print("  --> [验证 2 通过]: 客户端断开连接时，Session 成功被销毁，显存安全释放！")

    # -------------------------------------------------------------
    # 验证 3: 异常边界与错误链路验证
    # -------------------------------------------------------------
    print("\n>>> [验证 3] 异常边界与错误链路验证...")

    # 3A: 非法参数 max_new_tokens = -5
    print("  [3A: 非法 max_new_tokens = -5 测试]...")
    resp3 = client.post("/api/session/create", json={"prompt": "测试非法参数", "layer": 3, "head": 0})
    s3_id = resp3.json()["session_id"]

    with client.websocket_connect(f"/ws/session/{s3_id}/stream") as ws3:
        ws3.send_json({"action": "start", "max_new_tokens": -5})
        err_msg = ws3.receive_json()
        print(f"    收到服务端错误消息: {err_msg}")
        assert err_msg.get("type") == "error", f"Expected type 'error', got {err_msg}"
        assert "非法参数" in err_msg.get("message", ""), f"Unexpected error message: {err_msg}"

    # 3B: Level 2 按需拉取 linear_attn 层 (Layer 1) 的 attention 矩阵
    print("\n  [3B: Level 2 按需拉取 linear_attn 层 (Layer 1) 校验]...")
    resp4 = client.post("/api/session/create", json={"prompt": "测试混合架构 Layer 校验", "layer": 3, "head": 0})
    s4_id = resp4.json()["session_id"]

    # Request linear_attn layer (Layer 1)
    attn_err_resp = client.get(f"/api/session/{s4_id}/attention?layer=1&head=0")
    print(f"    请求 Layer 1 (linear_attn) 状态码: {attn_err_resp.status_code}")
    print(f"    响应内容: {attn_err_resp.json()}")
    assert attn_err_resp.status_code == 400, f"Expected 400, got {attn_err_resp.status_code}"
    assert "linear_attn" in attn_err_resp.json()["detail"], "Expected linear_attn warning"

    # Request valid full_attn layer (Layer 3)
    attn_ok_resp = client.get(f"/api/session/{s4_id}/attention?layer=3&head=0")
    print(f"    请求 Layer 3 (full_attn) 状态码: {attn_ok_resp.status_code}")
    assert attn_ok_resp.status_code == 200, f"Expected 200, got {attn_ok_resp.status_code}"
    attn_data = attn_ok_resp.json()
    assert "full_matrix" in attn_data, "Should return full_matrix for Level 2"
    assert "attention_row" in attn_data, "Should return attention_row for Level 2"
    print(f"    成功拉取 Level 2 Attention: seq_len={attn_data['seq_len']}, matrix_shape=[{len(attn_data['full_matrix'])}, {len(attn_data['full_matrix'][0])}]")

    # Clean up session 4
    client.post(f"/api/session/{s4_id}/close")

    print("  --> [验证 3 通过]: 非法参数与 linear_attn 层错误提示链路完全符合预期！")

    print("  [SUCCESS] 三项边界验证全部 100% 通过！")


if __name__ == "__main__":
    run_suite()
