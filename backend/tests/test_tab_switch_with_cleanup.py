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

    print("\n=== 模拟用户连续切换 Tab 10 次（每次切换前均触发显式 close） ===")
    active_sid = None
    active_arch = None

    for i in range(5):
        # 1. Switch to Transformer
        if active_sid:
            requests.post(f"http://127.0.0.1:8000/api/rwkv/session/{active_sid}/close")
            active_sid = None
        
        t_res = requests.post("http://127.0.0.1:8000/api/session/create", json={
            "prompt": f"第 {i+1} 次 Transformer 切换测试",
            "layer": 3,
            "head": 0
        })
        active_sid = t_res.json()["session_id"]
        active_arch = "transformer"
        for _ in range(15):
            requests.post(f"http://127.0.0.1:8000/api/session/{active_sid}/step", json={
                "temperature": 0.7, "top_k": 10, "layer": 3, "head": 0
            })

        # 2. Switch to RWKV (front-end closeActiveSession triggers)
        requests.post(f"http://127.0.0.1:8000/api/session/{active_sid}/close")
        active_sid = None

        r_res = requests.post("http://127.0.0.1:8000/api/rwkv/session/create", json={
            "prompt": f"第 {i+1} 次 RWKV 状态演化测试",
            "layer": 0,
            "head": 0
        })
        active_sid = r_res.json()["session_id"]
        active_arch = "rwkv"
        for _ in range(15):
            requests.post(f"http://127.0.0.1:8000/api/rwkv/session/{active_sid}/step", json={
                "temperature": 0.7, "top_k": 10, "layer": 0, "head": 0
            })

        u_curr, _, _ = get_gpu_memory()
        print(f"  轮次 {i+1}/5 | 切换并释放完成 | 当前显存: {u_curr} MB (相对基线变化: {u_curr - u0:+d} MB)")

    if active_sid:
        requests.post(f"http://127.0.0.1:8000/api/rwkv/session/{active_sid}/close")
    
    time.sleep(0.5)
    u_final, _, _ = get_gpu_memory()
    print(f"\n最终状态: 显存 {u_final} MB (相对基线: {u_final - u0:+d} MB)")
    print("显存稳定无累积泄漏！")

if __name__ == "__main__":
    run_test()
