import os
import sys
import numpy as np
import torch

os.environ["RWKV_V7_ON"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.rwkv_engine import RWKVEngine
from app.models.rwkv_session import RWKVSession
from app.core.config import settings

def run_calibration():
    print("=" * 80)
    print("  [CALIBRATION] RWKV-7 双色带固定色阶实测数值分布校准")
    print("=" * 80)

    engine = RWKVEngine()
    engine.load_model(settings.DEFAULT_RWKV, device="cuda" if torch.cuda.is_available() else "cpu")

    prompts = [
        ("技术分析 (中)", "RWKV通过线性状态递推摆脱了传统注意力机制的平方复杂度，在保持RNN推理高效的同时"),
        ("文学叙事 (短)", "夜幕低垂，窗外的细雨淅淅沥沥地打在玻璃上，屋内的灯光渐渐暗了下来，"),
        ("逻辑推理 (中长)", "如果所有的苹果都是水果，并且所有的水果都含有种子，那么我们可以严密地推理出以下结论："),
        ("代码生成 (短)", "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n")
    ]

    all_summaries = []
    all_deltas = []
    per_layer_summaries = {l: [] for l in range(engine.num_layers)}
    per_layer_deltas = {l: [] for l in range(engine.num_layers)}

    total_steps_run = 0

    for p_idx, (p_name, prompt) in enumerate(prompts):
        print(f"\n[Prompt {p_idx + 1}/4] {p_name}: {prompt[:30]}...")
        session = RWKVSession(f"calib_{p_idx}", engine)
        prefill_res = session.prefill(prompt, target_layer=0)

        # Track previous step summaries for ALL layers
        # session.current_state contains all layers
        prev_all_layer_summaries = []
        for l in range(engine.num_layers):
            s_l = session._compute_layer_summary(session.current_state, l)
            prev_all_layer_summaries.append(s_l)
            all_summaries.extend(s_l)
            per_layer_summaries[l].extend(s_l)

        # Generate 30 decode steps
        steps_target = 30
        for step_i in range(steps_target):
            step_res = session.step(
                temperature=0.7,
                top_k=10,
                repetition_penalty=1.2,
                target_layer=0
            )
            total_steps_run += 1

            # Compute summary and delta across ALL 24 layers
            curr_state = session.current_state
            for l in range(engine.num_layers):
                curr_s_l = session._compute_layer_summary(curr_state, l)
                prev_s_l = prev_all_layer_summaries[l]

                delta_l = [c - p for c, p in zip(curr_s_l, prev_s_l)]

                all_summaries.extend(curr_s_l)
                all_deltas.extend(delta_l)
                per_layer_summaries[l].extend(curr_s_l)
                per_layer_deltas[l].extend(delta_l)

                prev_all_layer_summaries[l] = curr_s_l

            if session.is_finished:
                print(f"   Prompt {p_idx + 1} EOS reached at step {step_i + 1}")
                break

        session.close()

    all_summaries_arr = np.array(all_summaries)
    all_deltas_arr = np.array(all_deltas)
    abs_deltas_arr = np.abs(all_deltas_arr)

    print("\n" + "=" * 80)
    print(f"  校准样本统计: 总步数={total_steps_run}, 总数据点数={len(all_summaries_arr)} (24层 x 16头)")
    print("=" * 80)

    # 1. state_summary 统计
    s_min = float(np.min(all_summaries_arr))
    s_max = float(np.max(all_summaries_arr))
    s_mean = float(np.mean(all_summaries_arr))
    s_median = float(np.median(all_summaries_arr))
    s_p90 = float(np.percentile(all_summaries_arr, 90))
    s_p95 = float(np.percentile(all_summaries_arr, 95))
    s_p99 = float(np.percentile(all_summaries_arr, 99))

    print("\n[1. 色带 A: state_summary 绝对强度分布统计]")
    print(f"   最小值 (Min)      : {s_min:.4f}")
    print(f"   最大值 (Max)      : {s_max:.4f}")
    print(f"   均值 (Mean)       : {s_mean:.4f}")
    print(f"   中位数 (Median)   : {s_median:.4f}")
    print(f"   90 分位数 (P90)   : {s_p90:.4f}")
    print(f"   95 分位数 (P95)   : {s_p95:.4f}")
    print(f"   99 分位数 (P99)   : {s_p99:.4f}")

    # 2. delta 统计
    d_min = float(np.min(all_deltas_arr))
    d_max = float(np.max(all_deltas_arr))
    d_mean = float(np.mean(all_deltas_arr))
    d_abs_mean = float(np.mean(abs_deltas_arr))
    d_abs_p90 = float(np.percentile(abs_deltas_arr, 90))
    d_abs_p95 = float(np.percentile(abs_deltas_arr, 95))
    d_abs_p99 = float(np.percentile(abs_deltas_arr, 99))

    print("\n[2. 色带 B: delta 变化量分布统计]")
    print(f"   实际数值范围      : [{d_min:+.6f}, {d_max:+.6f}]")
    print(f"   算术均值          : {d_mean:+.6f} (验证是否对称归零)")
    print(f"   绝对值均值 (Mean|Δ|): {d_abs_mean:.6f}")
    print(f"   |Δ| 90 分位数 (P90): {d_abs_p90:.6f}")
    print(f"   |Δ| 95 分位数 (P95): {d_abs_p95:.6f}")
    print(f"   |Δ| 99 分位数 (P99): {d_abs_p99:.6f}")

    # 层间差异抽样
    print("\n[3. 代表性层间差异抽样 (Layer 0, 7, 15, 23)]")
    for sample_l in [0, 7, 15, 23]:
        s_arr = np.array(per_layer_summaries[sample_l])
        d_arr = np.abs(np.array(per_layer_deltas[sample_l]))
        print(f"   Layer {sample_l:2d}: summary 均值={np.mean(s_arr):.4f} (P99={np.percentile(s_arr, 99):.4f}) | |delta| 均值={np.mean(d_arr):.6f} (P99={np.percentile(d_arr, 99):.6f})")

    # 推荐锁定常数
    # state_summary: 取 P99 并向上取整到合适的小数位 (留余量)
    recommended_summary_max = round(float(np.ceil(s_p99 * 1.15 * 10) / 10), 2)
    # delta: 取 P99 并向上微调 (留余量)
    recommended_delta_bound = round(float(np.ceil(d_abs_p99 * 1.2 * 100) / 100), 3)

    print("\n" + "=" * 80)
    print("  [RECOMMENDATION] 推荐锁定的固定色阶范围:")
    print(f"  - 色带 A (state_summary): [0.0, {recommended_summary_max:.2f}] (覆盖 99% 数据，超出饱和截断)")
    print(f"  - 色带 B (delta)        : [{-recommended_delta_bound:.3f}, {recommended_delta_bound:+.3f}] (对称覆盖 99% 数据，超出饱和截断)")
    print("=" * 80)

    return {
        "summary": {
            "min": s_min, "max": s_max, "mean": s_mean, "median": s_median,
            "p90": s_p90, "p95": s_p95, "p99": s_p99, "recommended_max": recommended_summary_max
        },
        "delta": {
            "min": d_min, "max": d_max, "mean": d_mean, "abs_mean": d_abs_mean,
            "abs_p90": d_abs_p90, "abs_p95": d_abs_p95, "abs_p99": d_abs_p99,
            "recommended_bound": recommended_delta_bound
        }
    }

if __name__ == "__main__":
    run_calibration()
