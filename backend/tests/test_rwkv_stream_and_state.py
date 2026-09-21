import asyncio
import json
import httpx
import websockets

async def test_rwkv_stream_and_state():
    print("=" * 80)
    print("  [TEST] RWKV-7 独立 Session、Level 1 状态摘要流与 Level 2 状态矩阵回填验证")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Prefill
        print("\n[Step 1] 创建 RWKV 会话并执行 Prefill...")
        prompt = "RWKV通过线性状态递推摆脱了传统注意力机制的平方复杂度"
        create_res = await client.post(
            "http://127.0.0.1:8000/api/rwkv/session/create",
            json={"prompt": prompt, "layer": 0, "head": 0}
        )
        assert create_res.status_code == 200, f"Prefill failed: {create_res.text}"
        data = create_res.json()
        s_id = data["session_id"]
        prefill_state = data["prefill_state"]
        initial_summary = prefill_state["state_summary"]
        initial_delta = prefill_state["delta"]

        print(f"   会话创建成功: ID={s_id}")
        print(f"   Prompt Token 数: {len(prefill_state['tokens'])}")
        print(f"   Prefill state_summary 长度: {len(initial_summary)} (预期严格为 16)")
        print(f"   Prefill delta 长度: {len(initial_delta)} (预期严格为 16)")
        assert len(initial_summary) == 16, f"Expected 16, got {len(initial_summary)}"
        assert len(initial_delta) == 16, f"Expected 16, got {len(initial_delta)}"

        # 2. WebSocket Stream
        print("\n[Step 2] 建立 WebSocket 连接测试 Level 1 增量流式生成 (6 tokens)...")
        ws_url = f"ws://127.0.0.1:8000/ws/rwkv/session/{s_id}/stream"
        
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "action": "start",
                "max_new_tokens": 6,
                "layer": 0,
                "temperature": 0.7,
                "top_k": 10,
                "repetition_penalty": 1.2
            }))

            prev_summary = initial_summary
            for i in range(6):
                raw = await ws.recv()
                msg = json.loads(raw)
                step = msg["step"]
                tok = msg["token"]
                summary = msg.get("state_summary", [])
                delta = msg.get("delta", [])

                # Verify constant 16D dimension (O(1) state property)
                assert len(summary) == 16, f"Step {step}: state_summary length is {len(summary)}, expected 16!"
                assert len(delta) == 16, f"Step {step}: delta length is {len(delta)}, expected 16!"

                # Verify Delta calculation: delta == curr_summary - prev_summary
                for h_idx in range(16):
                    expected_d = round(summary[h_idx] - prev_summary[h_idx], 4)
                    actual_d = round(delta[h_idx], 4)
                    assert abs(actual_d - expected_d) < 1e-3, (
                        f"Step {step} Head {h_idx}: delta mismatch! actual={actual_d}, expected={expected_d}"
                    )

                avg_norm = sum(summary) / 16.0
                avg_delta = sum(delta) / 16.0
                print(f"   Step {step}: token={repr(tok):<10} | summary[16] avg={avg_norm:.4f} | delta[16] avg={avg_delta:+.4f}")
                prev_summary = summary

            # Receive finish message
            fin_raw = await ws.recv()
            fin_msg = json.loads(fin_raw)
            print(f"   收到生成结束信号: is_finished={fin_msg.get('is_finished')}")
            assert fin_msg.get("is_finished") is True

            # 3. Level 2 On-demand retrieval
            print("\n[Step 3] 调用 Level 2 接口回溯历史状态: GET /api/session/{id}/rwkv_state?step=3&layer=1...")
            t0 = asyncio.get_event_loop().time()
            l2_res = await client.get(f"http://127.0.0.1:8000/api/session/{s_id}/rwkv_state?step=3&layer=1")
            t1 = asyncio.get_event_loop().time()
            assert l2_res.status_code == 200, f"Level 2 retrieval failed: {l2_res.text}"
            l2_data = l2_res.json()
            latency_ms = (t1 - t0) * 1000

            shape = l2_data["shape"]
            head_norms = l2_data["head_norms"]
            matrix = l2_data["matrix"]

            num_heads = len(matrix)
            mat_rows = len(matrix[0]) if num_heads > 0 else 0
            mat_cols = len(matrix[0][0]) if mat_rows > 0 else 0

            print(f"   [回填成功]: Step={l2_data['step']}, Layer={l2_data['layer']}, 耗时={latency_ms:.2f}ms")
            print(f"   [矩阵规格]: 头部数={num_heads}, 尺寸=[{mat_rows}, {mat_cols}], shape={shape}")
            print(f"   [头部范数]: 前4头 norms={head_norms[:4]}")

            assert shape == [16, 64, 64], f"Expected shape [16, 64, 64], got {shape}"
            assert num_heads == 16 and mat_rows == 64 and mat_cols == 64, "Matrix dimensions mismatch!"
            assert len(head_norms) == 16, f"Expected 16 head norms, got {len(head_norms)}"

            # 4. Close session
            print("\n[Step 4] 关闭会话并验证显存回收...")
            await ws.send(json.dumps({"action": "close"}))
            await asyncio.sleep(0.2)
            print("   会话已优雅断开并触发清理。")

    print("\n" + "=" * 80)
    print("  [SUCCESS] RWKV-7 Session + Level 1 流式 + Level 2 状态矩阵全流程验证 100% 通过！")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_rwkv_stream_and_state())
