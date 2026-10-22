import os
import sys
import torch
import subprocess

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from tests.measure_kl_divergence import (
    compute_logits_kl_divergence,
    compute_attention_kl_divergence,
    run_benchmark,
)


def test_compute_logits_kl_divergence():
    print("\n--- [Test 1] Logits KL Divergence Mathematical Correctness ---", flush=True)

    # 1. Self divergence must be strictly 0
    p = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.2]])
    kl_self = compute_logits_kl_divergence(p, p)
    assert abs(kl_self) < 1e-6, f"Self KL divergence must be 0, got {kl_self}"
    print("  -> Self divergence KL(P || P) == 0.0: OK", flush=True)

    # 2. Non-identical distributions must have strictly positive KL divergence
    q = torch.tensor([[3.0, 1.0, 0.0], [-1.0, 2.0, 0.5]])
    kl_diff = compute_logits_kl_divergence(p, q)
    assert kl_diff > 0.0, f"Different distributions must have KL > 0, got {kl_diff}"
    print(f"  -> Positive definiteness KL(P || Q) = {kl_diff:.4f} > 0: OK", flush=True)

    # 3. Scale invariance under softmax temperature
    kl_t2 = compute_logits_kl_divergence(p, q, temperature=2.0)
    assert kl_t2 > 0.0
    print("  -> Temperature scaling behavior: OK", flush=True)


def test_compute_attention_kl_divergence():
    print("\n--- [Test 2] Attention KL Divergence Matrix Computation ---", flush=True)

    # Shape: [batch, heads, seq, seq]
    attn_base = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    kl_attn_self = compute_attention_kl_divergence(attn_base, attn_base)
    assert abs(kl_attn_self) < 1e-6, f"Self attention KL must be 0, got {kl_attn_self}"
    print("  -> Self attention KL(A || A) == 0.0: OK", flush=True)

    attn_other = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    kl_attn_diff = compute_attention_kl_divergence(attn_base, attn_other)
    assert kl_attn_diff > 0.0
    print(f"  -> Divergence between distinct attention matrices = {kl_attn_diff:.4f} > 0: OK", flush=True)


def test_benchmark_runner_self_check():
    print("\n--- [Test 3] Benchmark Runner in Self-Check Mode (Absence of Unquantized Weights) ---", flush=True)
    res = run_benchmark(
        quantized_path="dummy_path.gguf",
        unquantized_path="non_existent_unquantized_model_dir",
        device="cpu"
    )
    assert res["has_unquantized_baseline"] is False
    assert res["validation_status"] == "PASSED_SELF_CHECK"
    print("  -> Benchmark runner handled missing baseline gracefully with transparent guidance: OK", flush=True)


def test_measure_script_cli():
    print("\n--- [Test 4] measure_kl_divergence.py CLI Execution ---", flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable,
        os.path.join(backend_dir, "tests", "measure_kl_divergence.py"),
        "--device", "cpu"
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    assert proc.returncode == 0, f"Script failed with code {proc.returncode}:\n{proc.stderr}"
    assert "未在本地检测到未量化" in proc.stdout or "Qwen/Qwen3.5-0.8B" in proc.stdout
    print("  -> CLI execution exited 0 with complete guidance printed: OK", flush=True)


if __name__ == "__main__":
    test_compute_logits_kl_divergence()
    test_compute_attention_kl_divergence()
    test_benchmark_runner_self_check()
    test_measure_script_cli()
    print("\nALL KL DIVERGENCE MEASUREMENT TESTS PASSED!")
