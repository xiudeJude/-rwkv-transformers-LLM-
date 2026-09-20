import sys
import os
import time

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from app.models.transformer import TransformerEngine
from app.models.session import session_manager
from app.core.config import settings

def run_boundary_tests():
    print("=" * 70)
    print("【KV Cache 增量推理核心边界条件专项测试】")
    print(f"Device: {settings.DEVICE}")
    print(f"Model: {settings.DEFAULT_TRANSFORMER}")
    
    # Load model
    engine = TransformerEngine()
    engine.load_model(settings.DEFAULT_TRANSFORMER, device=settings.DEVICE)
    print("模型已成功加载至 GPU (attn_implementation='eager', bfloat16)\n")

    test_prompt = "注意力机制与线性RNN模型的核心差异在于"
    
    # -------------------------------------------------------------
    # 边界测试 A & B: 连续 12 步 Decode，验证增量切片形状 [B,H,1,L+t] 与显存稳定性
    # -------------------------------------------------------------
    print(">>> [测试 A & B] 连续 12 步增量递推验证 (切片形状严格递增 + 显存平稳监测)")
    session_a = session_manager.create_session(engine)
    prefill_res = session_a.prefill(test_prompt, target_layer=3, target_head=0)
    L_prompt = prefill_res["seq_len"]
    print(f"  Prompt Prefill 完成: L_prompt = {L_prompt} tokens")
    
    initial_vram = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    print(f"  Prefill 后基准已分配显存: {initial_vram:.2f} MB")

    slice_shapes = []
    vram_samples = []
    generated_tokens_a = []

    for step_idx in range(1, 13):
        res = session_a.step(temperature=0.7, top_k=10, target_layer=3, target_head=0)
        curr_vram = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        vram_samples.append(curr_vram)
        
        slice_info = res["attention_slice"]
        slice_shape = slice_info["shape"] # expected [1, 8, 1, L_prompt + step_idx]
        slice_shapes.append(slice_shape)
        generated_tokens_a.append(res["token"])
        
        expected_len = L_prompt + step_idx
        actual_len = slice_info["slice_len"]
        print(f"  Step {step_idx:2d}: Token='{res['token']:<4}' | SliceShape={slice_shape} (Len={actual_len}/{expected_len}) | VRAM={curr_vram:.2f} MB")
        
        # Verification A: Shape must strictly equal [1, 8, 1, L_prompt + t]
        assert slice_shape == [1, 8, 1, expected_len], f"Shape mismatch at step {step_idx}: expected [1, 8, 1, {expected_len}], got {slice_shape}"
        assert actual_len == expected_len, f"Slice length mismatch: {actual_len} != {expected_len}"

    final_vram = vram_samples[-1]
    vram_delta = final_vram - initial_vram
    print(f"\n  [测试 A 结果] 连续 12 步注意力切片形状验证: 全部 100% 符合严格递增 [1, 8, 1, L_prompt+t]！")
    print(f"  [测试 B 结果] 初始显存 {initial_vram:.2f} MB -> 最终显存 {final_vram:.2f} MB (净变化: {vram_delta:+.2f} MB)")
    
    # In KV cache with small tokens, VRAM growth is negligible (a few KB for KV tensors, not duplicate models)
    assert abs(vram_delta) < 30.0, f"VRAM leaked significantly! Delta: {vram_delta:.2f} MB"
    print(f"  [测试 B 结论] 显存曲线平稳收敛，未发生模型重载或历史 Tensor 冗余堆叠泄漏！")

    session_manager.close_session(session_a.session_id)
    post_close_vram = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    print(f"  会话销毁释放后显存: {post_close_vram:.2f} MB\n")

    # -------------------------------------------------------------
    # 边界测试 C: 连续两次生成同一个prompt（两个独立session，贪婪解码 temperature=0）
    # -------------------------------------------------------------
    print(">>> [测试 C] 两个完全独立的 Session 在 temperature=0 下贪婪解码一致性验证")
    
    # Run Session 1
    session_1 = session_manager.create_session(engine)
    session_1.prefill(test_prompt, target_layer=3, target_head=0)
    tokens_run_1 = []
    for _ in range(8):
        step_out = session_1.step(temperature=0.0, top_k=1, target_layer=3, target_head=0)
        tokens_run_1.append((step_out["token"], step_out["token_id"], step_out["token_prob"]))
    session_manager.close_session(session_1.session_id)
    
    # Run Session 2 (Independent session immediately after)
    session_2 = session_manager.create_session(engine)
    session_2.prefill(test_prompt, target_layer=3, target_head=0)
    tokens_run_2 = []
    for _ in range(8):
        step_out = session_2.step(temperature=0.0, top_k=1, target_layer=3, target_head=0)
        tokens_run_2.append((step_out["token"], step_out["token_id"], step_out["token_prob"]))
    session_manager.close_session(session_2.session_id)

    print(f"  Session 1 生成序列: {[t[0] for t in tokens_run_1]}")
    print(f"  Session 2 生成序列: {[t[0] for t in tokens_run_2]}")

    tokens_match = [t1[0] == t2[0] for t1, t2 in zip(tokens_run_1, tokens_run_2)]
    ids_match = [t1[1] == t2[1] for t1, t2 in zip(tokens_run_1, tokens_run_2)]
    probs_match = [abs(t1[2] - t2[2]) < 1e-4 for t1, t2 in zip(tokens_run_1, tokens_run_2)]

    print(f"  Token 序列逐字比对: {tokens_match}")
    print(f"  Token ID 严格一致: {all(ids_match)}")
    print(f"  概率值浮点严格一致: {all(probs_match)}")

    assert all(ids_match), "Two greedy runs produced different token IDs! KV cache is dirty or leaking state!"
    assert all(probs_match), "Two greedy runs produced different probabilities!"
    
    print("\n=== [测试 C 结论] 独立会话状态隔离完美，零状态污染，确定性输出 100% 吻合！===")
    print("=" * 70)

if __name__ == "__main__":
    run_boundary_tests()
