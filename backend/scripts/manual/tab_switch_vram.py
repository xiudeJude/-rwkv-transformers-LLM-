import requests
import json
import time
import subprocess

def get_gpu_memory():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        used, free, total = [int(x.strip()) for x in res.strip().split(",")]
        return used, free, total
    except Exception as e:
        return 0, 0, 0

def run_test():
    u0, f0, t0 = get_gpu_memory()
    print(f"=== 基线状态 ===")
    print(f"初始 GPU 显存: {u0} MB / {t0} MB (剩余 {f0} MB)")

    created_transformer_sessions = []
    created_rwkv_sessions = []

    print("\n=== 模拟用户连续切换 Tab 10 次（各生成 30 步，不显式关闭 Session） ===")
    for i in range(5):
        # 1. User on Transformer Tab: creates session & generates 30 tokens
        t_res = requests.post("http://127.0.0.1:8000/api/session/create", json={
            "prompt": f"第 {i+1} 次 Transformer 测试长上下文语义依存关系",
            "layer": 3,
            "head": 0
        })
        t_sid = t_res.json()["session_id"]
        created_transformer_sessions.append(t_sid)
        for _ in range(30):
            requests.post(f"http://127.0.0.1:8000/api/session/{t_sid}/step", json={
                "temperature": 0.7, "top_k": 10, "layer": 3, "head": 0
            })
        
        # User abruptly switches to RWKV Tab: creates session & generates 30 tokens
        r_res = requests.post("http://127.0.0.1:8000/api/rwkv/session/create", json={
            "prompt": f"第 {i+1} 次 RWKV 状态演化测试",
            "layer": 0,
            "head": 0
        })
        r_sid = r_res.json()["session_id"]
        created_rwkv_sessions.append(r_sid)
        for _ in range(30):
            requests.post(f"http://127.0.0.1:8000/api/rwkv/session/{r_sid}/step", json={
                "temperature": 0.7, "top_k": 10, "layer": 0, "head": 0
            })
        
        u_curr, _, _ = get_gpu_memory()
        print(f"  轮次 {i+1}/5 完成 | 累计遗留未关闭 Session: {len(created_transformer_sessions) + len(created_rwkv_sessions)} | 显存已用: {u_curr} MB (增长 +{u_curr - u0} MB)")

    u_leak, _, _ = get_gpu_memory()
    print(f"\n未释放 Session 状态下的显存总增长: +{u_leak - u0} MB")

    print("\n=== 执行显式 Session 清理 (调用 /close 接口) ===")
    for sid in created_transformer_sessions:
        requests.post(f"http://127.0.0.1:8000/api/session/{sid}/close")
    for sid in created_rwkv_sessions:
        requests.post(f"http://127.0.0.1:8000/api/rwkv/session/{sid}/close")
    
    time.sleep(1)
    u_after, _, _ = get_gpu_memory()
    print(f"显式清理后 GPU 显存: {u_after} MB (相对基线回落: {u_after - u0} MB)")

if __name__ == "__main__":
    run_test()
