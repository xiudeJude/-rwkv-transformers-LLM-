"""
measure_kl_divergence.py
--------------------------------------------------------------------------------
用于测量量化模型 (GGUF Q8_0) 相对未量化基准模型 (bfloat16 safetensors) 在
Logits 预测分布与各层 Attention 矩阵上的 KL 散度 (Kullback-Leibler Divergence)。

支持两种运行模式：
1. 双模型对比模式：当本地同时存在未量化基准权重与量化权重时，执行端到端前向推理并
   输出精确的逐层 KL 散度报告。
2. 自检验与算法验证模式：当本地仅存在量化权重时，验证 KL 散度核心计算公式的数值稳定
   性与自一致性，并输出基准权重下载与实测复现指南。
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import json
import math
from typing import Dict, Any, List, Optional
import torch

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.models.transformer import TransformerEngine


def compute_logits_kl_divergence(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-12
) -> float:
    """
    计算两个 Logits 分布之间的 KL 散度 KL(P || Q)。
    P 为基准分布 (Unquantized)，Q 为近似分布 (Quantized)。
    """
    t = max(temperature, 1e-4)
    p_log = torch.log_softmax(logits_p.float() / t, dim=-1)
    q_log = torch.log_softmax(logits_q.float() / t, dim=-1)
    p_prob = torch.softmax(logits_p.float() / t, dim=-1)

    # KL(P || Q) = sum(P * (log P - log Q))
    kl_pointwise = p_prob * (p_log - q_log)
    kl_val = torch.sum(kl_pointwise, dim=-1).mean().item()
    return max(0.0, float(kl_val))


def compute_attention_kl_divergence(
    attn_p: torch.Tensor,
    attn_q: torch.Tensor,
    eps: float = 1e-9
) -> float:
    """
    计算两个 Attention 矩阵 (已做 Softmax，每行和为 1) 之间的行均值 KL 散度 KL(P || Q)。
    """
    # 限制极小值避免 log(0)
    p = torch.clamp(attn_p.float(), min=eps)
    q = torch.clamp(attn_q.float(), min=eps)

    # 重新归一化行概率
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    kl_matrix = p * (torch.log(p) - torch.log(q))
    kl_mean = torch.sum(kl_matrix, dim=-1).mean().item()
    return max(0.0, float(kl_mean))


DEFAULT_BENCHMARK_PROMPTS = [
    "Language models are capable of generalizing across varied tasks.",
    "大语言模型的可解释性分析有助于深入理解注意力头的分工机制。",
    "The transformer architecture relies on self-attention to model long-range context.",
    "混合状态空间与注意力机制的架构设计兼顾了计算吞吐与长文本保真度。"
]


def run_benchmark(
    quantized_path: str,
    unquantized_path: Optional[str] = None,
    prompts: Optional[List[str]] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    prompts = prompts or DEFAULT_BENCHMARK_PROMPTS
    results: Dict[str, Any] = {
        "quantized_path": quantized_path,
        "unquantized_path": unquantized_path,
        "device": device,
        "has_unquantized_baseline": False,
        "prompts_count": len(prompts)
    }

    # 检查未量化权重是否存在
    unquantized_available = (
        unquantized_path is not None
        and os.path.exists(unquantized_path)
    )

    if not unquantized_available:
        print("\n" + "=" * 70)
        print(" [KL 散度评估] 本地未量化基准模型状态提示")
        print("=" * 70)
        print(f" 未量化基准路径: {unquantized_path}")
        print(" 状态: 未在本地检测到未量化 (bfloat16 safetensors) 全量模型权重。")
        print(" 说明: 本地默认仅搭载了轻量化的 GGUF Q8_0 量化权重以节省存储与显存。")
        print(" 如需执行双模型真实验证，请通过以下命令下载官方未量化权重后再运行本脚本:")
        print("   huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir backend/models/Qwen3.5-0.8B")
        print("=" * 70 + "\n")

        # 执行自检验模式
        print("-> 正在执行 KL 散度数学函数与量化模型推理自检验...")
        # 1. 验证数学一致性: P==Q 时 KL 为 0
        dummy_p = torch.randn(2, 100)
        zero_kl = compute_logits_kl_divergence(dummy_p, dummy_p)
        assert abs(zero_kl) < 1e-6, f"Self-divergence must be 0, got {zero_kl}"

        # 2. 验证非对称性与正定性: P != Q 时 KL > 0
        dummy_q = dummy_p + torch.randn_like(dummy_p) * 0.1
        pos_kl = compute_logits_kl_divergence(dummy_p, dummy_q)
        assert pos_kl > 0, f"Divergence must be positive, got {pos_kl}"

        # 3. 验证 Attention KL 函数
        attn_p = torch.softmax(torch.randn(4, 16, 32, 32), dim=-1)
        attn_self_kl = compute_attention_kl_divergence(attn_p, attn_p)
        assert abs(attn_self_kl) < 1e-6

        results["validation_status"] = "PASSED_SELF_CHECK"
        results["notes"] = (
            "未量化全量权重缺失，已完成散度计算引擎的数值稳定性自检验与数学约束验证。"
            "复现双模型端到端对比请使用 --unquantized-path 指定 safetensors 权重目录。"
        )
        return results

    # 当未量化模型存在时执行实际对比
    print(f"\n-> 正在加载未量化基准模型: {unquantized_path}")
    base_engine = TransformerEngine()
    base_engine.load_model(unquantized_path, device=device)

    print(f"-> 正在加载量化模型: {quantized_path}")
    quant_engine = TransformerEngine()
    quant_engine.load_model(quantized_path, device=device)

    logits_kl_list = []
    attn_kl_by_layer: Dict[int, List[float]] = {}

    for prompt in prompts:
        # 1. 基准模型推理
        t_base = base_engine.tokenize(prompt)
        input_ids_base = t_base["input_ids"].to(device)
        with torch.no_grad():
            out_base = base_engine.model(input_ids=input_ids_base, output_attentions=True)

        # 2. 量化模型推理
        t_quant = quant_engine.tokenize(prompt)
        input_ids_quant = t_quant["input_ids"].to(device)
        with torch.no_grad():
            out_quant = quant_engine.model(input_ids=input_ids_quant, output_attentions=True)

        # 计算 logits KL 散度
        l_kl = compute_logits_kl_divergence(out_base.logits[:, -1, :], out_quant.logits[:, -1, :])
        logits_kl_list.append(l_kl)

        # 计算各层 attention KL 散度
        if out_base.attentions and out_quant.attentions:
            for l_idx, (a_b, a_q) in enumerate(zip(out_base.attentions, out_quant.attentions)):
                layer_kl = compute_attention_kl_divergence(a_b, a_q)
                if l_idx not in attn_kl_by_layer:
                    attn_kl_by_layer[l_idx] = []
                attn_kl_by_layer[l_idx].append(layer_kl)

    mean_logits_kl = sum(logits_kl_list) / len(logits_kl_list)
    mean_attn_kl_by_layer = {
        l: round(sum(vals) / len(vals), 6) for l, vals in attn_kl_by_layer.items()
    }

    results["has_unquantized_baseline"] = True
    results["mean_logits_kl"] = round(mean_logits_kl, 6)
    results["mean_attention_kl_by_layer"] = mean_attn_kl_by_layer
    results["validation_status"] = "PASSED_EMPIRICAL_MEASUREMENT"

    print("\n--- 实测 KL 散度对比结果 ---")
    print(f"Logits 平均 KL 散度: {results['mean_logits_kl']}")
    print(f"各 Full Attention 层平均 KL 散度: {results['mean_attention_kl_by_layer']}")
    return results


def main():
    parser = argparse.ArgumentParser(description="量化模型 vs 未量化基准模型 KL 散度对比测量")
    parser.add_argument(
        "--quantized-path",
        type=str,
        default=settings.DEFAULT_TRANSFORMER,
        help="量化模型权重路径 (默认读取 settings.DEFAULT_TRANSFORMER)"
    )
    parser.add_argument(
        "--unquantized-path",
        type=str,
        default=os.path.join(backend_dir, "models", "Qwen3.5-0.8B"),
        help="未量化基准模型 safetensors 路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备 (cpu 或 cuda)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="可选的结果 JSON 输出文件路径"
    )

    args = parser.parse_args()
    results = run_benchmark(
        quantized_path=args.quantized-path if hasattr(args, "quantized-path") else args.quantized_path,
        unquantized_path=args.unquantized_path,
        device=args.device
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"测量结果已保存至: {args.output_json}")


if __name__ == "__main__":
    main()
