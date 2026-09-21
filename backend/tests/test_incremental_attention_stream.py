import asyncio
import json
import httpx
import websockets

async def test_stream_and_refetch():
    print("=" * 80)
    print("  测试增量单行 Attention 推流与生成后 Level 2 层切换回填")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Prefill
        create_res = await client.post(
            "http://127.0.0.1:8000/api/session/create",
            json={"prompt": "注意力机制让大语言模型精确捕捉语义关系", "layer": 3, "head": 0}
        )
        assert create_res.status_code == 200
        data = create_res.json()
        s_id = data["session_id"]
        L_prompt = len(data["prefill_attention"]["tokens"])
        print(f"1. 会话建立成功: {s_id}, L_prompt={L_prompt}")

        # 2. Stream 6 tokens
        async with websockets.connect(f"ws://127.0.0.1:8000/ws/session/{s_id}/stream") as ws:
            await ws.send(json.dumps({
                "action": "start",
                "max_new_tokens": 6,
                "layer": 3,
                "head": 0
            }))

            for i in range(6):
                raw = await ws.recv()
                msg = json.loads(raw)
                step = msg["step"]
                tok = msg["token"]
                attn_row = msg.get("attention_row", [])
                expected_len = L_prompt + step
                print(f"   Step {step}: token={repr(tok):<10} | attention_row 长度={len(attn_row)} (预期 {expected_len})")
                assert len(attn_row) == expected_len, f"Expected {expected_len}, got {len(attn_row)}"

            # Wait for finish message
            fin_msg = json.loads(await ws.recv())
            print(f"   收到完成消息: is_finished={fin_msg.get('is_finished')}")
            assert fin_msg.get("is_finished") is True

            # 3. Post-generation Level 2 refetch on Layer 7 Head 2
            print("\n2. 生成结束后切换 Layer 7 / Head 2，调用 Level 2 回填接口...")
            t0 = asyncio.get_event_loop().time()
            l2_res = await client.get(f"http://127.0.0.1:8000/api/session/{s_id}/attention?layer=7&head=2")
            t1 = asyncio.get_event_loop().time()
            assert l2_res.status_code == 200
            l2_data = l2_res.json()
            matrix = l2_data["full_matrix"]
            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0
            latency_ms = (t1 - t0) * 1000
            print(f"   [回填成功]: Layer {l2_data['layer']} Head {l2_data['head']}")
            print(f"   [矩阵规格]: [{rows}, {cols}], 耗时: {latency_ms:.2f}ms")
            assert rows == L_prompt + 6
            assert cols == L_prompt + 6

            # Close session
            await ws.send(json.dumps({"action": "close"}))
            print("\n3. 会话关闭完成，显存安全释放。")

    print("=" * 80)
    print("  [SUCCESS] 增量推流与事后层切换回填验证 100% 通过！")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_stream_and_refetch())
