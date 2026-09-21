import asyncio
import time
import json
import httpx
import websockets

BASE_HTTP = "http://127.0.0.1:8000"
BASE_WS = "ws://127.0.0.1:8000"

async def test_concurrency():
    print("=" * 80)
    print("  并发场景基准测试：Level 2 矩阵拉取 vs WebSocket 流式生成资源竞争实测")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        prompt = "注意力机制让大语言模型能够精确捕捉长距离语义依赖关系"

        # -------------------------------------------------------------
        # 1. 空闲态基线测量：在未推流时连续拉取 Level 2 Attention 矩阵
        # -------------------------------------------------------------
        print("\n>>> [阶段 1] 测量 Session 空闲态下的 Level 2 Attention 响应延迟基线...")
        create_resp = await http_client.post(
            f"{BASE_HTTP}/api/session/create",
            json={"prompt": prompt, "layer": 3, "head": 0}
        )
        assert create_resp.status_code == 200, f"Create failed: {create_resp.text}"
        session_id_idle = create_resp.json()["session_id"]

        idle_latencies = []
        for i in range(5):
            t0 = time.perf_counter()
            resp = await http_client.get(
                f"{BASE_HTTP}/api/session/{session_id_idle}/attention",
                params={"layer": 3, "head": 0}
            )
            t1 = time.perf_counter()
            assert resp.status_code == 200
            latency_ms = (t1 - t0) * 1000
            idle_latencies.append(latency_ms)

        avg_idle_latency = sum(idle_latencies) / len(idle_latencies)
        min_idle = min(idle_latencies)
        max_idle = max(idle_latencies)
        print(f"  [空闲基线结果]: 5 次采样延迟平均={avg_idle_latency:.2f}ms (最小={min_idle:.2f}ms, 最大={max_idle:.2f}ms)")

        await http_client.post(f"{BASE_HTTP}/api/session/{session_id_idle}/close")

        # -------------------------------------------------------------
        # 2. 并发生成中测量：在推流到第 20 步时并发拉取 Level 2 Attention
        # -------------------------------------------------------------
        print("\n>>> [阶段 2] 启动 50 Token 流式生成，并在第 20 步并发发起 Level 2 请求...")
        create_resp2 = await http_client.post(
            f"{BASE_HTTP}/api/session/create",
            json={"prompt": prompt, "layer": 3, "head": 0}
        )
        assert create_resp2.status_code == 200
        session_id_stream = create_resp2.json()["session_id"]

        ws_url = f"{BASE_WS}/ws/session/{session_id_stream}/stream"
        token_records = []
        concurrent_req_info = {}

        async with websockets.connect(ws_url) as ws:
            # Send start action
            await ws.send(json.dumps({
                "action": "start",
                "temperature": 0.55,
                "top_k": 10,
                "max_new_tokens": 50,
                "layer": 3,
                "head": 0
            }))

            prev_time = None
            step_count = 0
            concurrent_fired = False

            while True:
                msg_text = await ws.recv()
                now = time.perf_counter()
                delta_ms = (now - prev_time) * 1000 if prev_time else 0.0
                prev_time = now

                msg = json.loads(msg_text)
                if msg.get("type") == "error":
                    raise RuntimeError(f"WS error: {msg}")

                step = msg.get("step", 0)
                tok = msg.get("token", "")
                is_fin = msg.get("is_finished", False)

                token_records.append({
                    "step": step,
                    "token": tok,
                    "delta_ms": round(delta_ms, 2),
                    "timestamp": now
                })
                step_count += 1

                # Trigger concurrent HTTP request at Step 20
                if step_count == 20 and not concurrent_fired:
                    concurrent_fired = True
                    print(f"  [事件触发]: 流式已到达第 {step_count} 步，正在异步并发发起 GET /api/session/.../attention 请求...")

                    async def fire_concurrent_request():
                        t_req_start = time.perf_counter()
                        c_resp = await http_client.get(
                            f"{BASE_HTTP}/api/session/{session_id_stream}/attention",
                            params={"layer": 3, "head": 0}
                        )
                        t_req_end = time.perf_counter()
                        latency = (t_req_end - t_req_start) * 1000
                        concurrent_req_info["status_code"] = c_resp.status_code
                        concurrent_req_info["latency_ms"] = round(latency, 2)
                        data = c_resp.json()
                        concurrent_req_info["seq_len"] = data.get("seq_len", 0)
                        concurrent_req_info["matrix_rows"] = len(data.get("full_matrix", []))
                        print(f"  [并发响应完成]: Level 2 响应耗时 = {latency:.2f}ms (HTTP {c_resp.status_code}, 矩阵尺寸=[{len(data.get('full_matrix', []))}, {len(data.get('full_matrix', []))}])")

                    # Launch without blocking WS loop
                    asyncio.create_task(fire_concurrent_request())

                if is_fin or step_count >= 50:
                    break

            try:
                await ws.send(json.dumps({"action": "close"}))
            except Exception:
                pass

        # -------------------------------------------------------------
        # 3. 数据分析与对比
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print("  并发竞争实测数据对比报告")
        print("=" * 80)

        # Token delta breakdown
        pre_deltas = [r["delta_ms"] for r in token_records[1:20]]
        avg_pre = sum(pre_deltas) / len(pre_deltas) if pre_deltas else 0

        # Steps around the concurrent request (steps 20, 21, 22)
        concurrent_deltas = [r["delta_ms"] for r in token_records[20:23]]
        post_deltas = [r["delta_ms"] for r in token_records[23:]]
        avg_post = sum(post_deltas) / len(post_deltas) if post_deltas else 0

        print(f"1. Level 2 请求响应延迟对比:")
        print(f"   - 空闲基线延迟: 平均 {avg_idle_latency:.2f} ms (范围: {min_idle:.2f} ~ {max_idle:.2f} ms)")
        print(f"   - 并发生成中延迟: {concurrent_req_info.get('latency_ms', 0):.2f} ms")
        delta_latency = concurrent_req_info.get('latency_ms', 0) - avg_idle_latency
        print(f"   - 延迟增幅: {delta_latency:+.2f} ms")

        print(f"\n2. WebSocket 流式生成 Token 到达间隔变化:")
        print(f"   - 并发前 (Step 2 ~ 19) 平均到达间隔: {avg_pre:.2f} ms")
        print(f"   - 并发期间 (Step 20 ~ 22) 实际到达间隔: {concurrent_deltas}")
        print(f"   - 并发后 (Step 23 ~ 50) 平均到达间隔: {avg_post:.2f} ms")

        print("\n3. 步骤时间采样 (Step 17 ~ 25):")
        print(f"   {'Step':<6} | {'Token':<10} | {'Δt 到达(ms)':<12} | {'事件标记'}")
        print("   " + "-" * 50)
        for r in token_records:
            if 17 <= r["step"] <= 25:
                tag = ""
                if r["step"] == 20:
                    tag = "<-- 并发 Level 2 请求发出"
                elif r["step"] == 21:
                    tag = "<-- 并发处理期间"
                elif r["step"] == 22:
                    tag = "<-- 并发处理结束"
                print(f"   {r['step']:<6} | {repr(r['token']):<10} | {r['delta_ms']:<12.2f} | {tag}")

        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_concurrency())
