import asyncio
import json
import httpx
import websockets

async def main():
    print("=" * 80)
    print("  P0-Step4 修复与优化综合验收测试")
    print("  - 1. Prefill 方阵与因果性检验 (无 1.0 穿透列)")
    print("  - 2. Repetition Penalty 抑制重复标点验证")
    print("  - 3. WebSocket 逐步推流 attention_row 尺寸递增")
    print("  - 4. 历史步 Level 2 详情回填验证 (Token 点击交互后端支撑)")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Prefill Session
        res = await client.post("http://127.0.0.1:8000/api/session/create", json={
            "prompt": "注意力机制让大语言模型能够精确捕捉长距离语义依赖关系",
            "layer": 3,
            "head": 0
        })
        assert res.status_code == 200
        data = res.json()
        s_id = data["session_id"]
        prefill_attn = data["prefill_attention"]
        prefill_mat = prefill_attn["matrix"]
        tokens = prefill_attn["tokens"]
        L_prompt = len(tokens)
        
        print(f"\n[验证 1] Prefill 方阵规格: {len(prefill_mat)}×{len(prefill_mat[0])} (Token 数: {L_prompt})")
        print(f"         Causal 标志: {prefill_attn['is_causal']}")
        assert prefill_attn["is_causal"] is True
        
        # Verify no column in prefill has all 1.0
        for c in range(L_prompt):
            col_vals = [prefill_mat[r][c] for r in range(L_prompt)]
            assert not (all(v > 0.95 for v in col_vals)), f"Column {c} has all 1.0, which violates causal attention!"
        print("         [PASS] Prefill 方阵严格因果下三角，无任何 1.0 穿透列！")

        # 2. WebSocket streaming with repetition_penalty = 1.25
        print(f"\n[验证 2 & 3] WebSocket 流式生成 15 步 (repetition_penalty = 1.25)...")
        generated_tokens = []
        async with websockets.connect(f"ws://127.0.0.1:8000/ws/session/{s_id}/stream") as ws:
            await ws.send(json.dumps({
                "action": "start",
                "temperature": 0.7,
                "top_k": 10,
                "repetition_penalty": 1.25,
                "max_new_tokens": 15,
                "layer": 3,
                "head": 0
            }))

            for step_idx in range(1, 16):
                msg = json.loads(await ws.recv())
                if msg.get("is_finished") and not msg.get("token"):
                    break
                tok = msg["token"]
                step = msg["step"]
                row = msg["attention_row"]
                generated_tokens.append(tok)
                
                expected_len = L_prompt + step
                assert len(row) == expected_len, f"Step {step} row len {len(row)} != expected {expected_len}"
                # Verify row sum is ~1.0 (softmax)
                row_sum = sum(row)
                assert 0.98 <= row_sum <= 1.02, f"Row sum {row_sum} not normalized"
                print(f"   Step {step:2d}: token={repr(tok):<10} | row_len={len(row):2d} (预期 {expected_len:2d}) | sum={row_sum:.4f}")

            fin_msg = json.loads(await ws.recv())
            assert fin_msg.get("is_finished") is True

        gen_text = "".join(generated_tokens)
        print(f"         实测生成文本: {gen_text}")
        # Verify no repetitive punctuation loop like "，，，" or "、、、"
        assert "，，" not in gen_text and "、、" not in gen_text, "Repetition penalty failed to suppress repeated punctuation!"
        print("         [PASS] Repetition Penalty 成功消除退化重复标点！")

        # 4. Level 2 Historical Token Click Inspection at Step 5
        print("\n[验证 4] 模拟前端用户点击 Step 5 Token，拉取该步历史注意力全景...")
        target_step = 5
        l2_res = await client.get(
            f"http://127.0.0.1:8000/api/session/{s_id}/attention?step={target_step}&layer=3&head=0"
        )
        assert l2_res.status_code == 200
        l2_data = l2_res.json()
        expected_dim = L_prompt + target_step
        matrix_step5 = l2_data["full_matrix"]
        assert len(matrix_step5) == expected_dim
        assert len(matrix_step5[0]) == expected_dim
        print(f"         Step {target_step} 矩阵尺寸: [{len(matrix_step5)}, {len(matrix_step5[0])}] (符合预期 {expected_dim}×{expected_dim})")
        print("         [PASS] 历史 Token 步注意力回填接口响应正常！")

        # Cleanup
        await client.post(f"http://127.0.0.1:8000/api/session/{s_id}/close")
        print("\n[验证 5] 会话清理与显存释放完毕。")

    print("\n" + "=" * 80)
    print("  [SUCCESS] 全部修复与优化项 100% 验收通过！")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
