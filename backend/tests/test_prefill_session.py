import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from app.models.transformer import TransformerEngine
from app.models.session import session_manager
from app.core.config import settings

def test_prefill_session():
    print("=" * 60)
    print("【Step 2·Prefill 阶段验证测试】")
    print(f"Device: {settings.DEVICE} (CUDA available: {torch.cuda.is_available()})")
    print(f"Model: {settings.DEFAULT_TRANSFORMER}")
    
    # 1. Load engine
    engine = TransformerEngine()
    engine.load_model(settings.DEFAULT_TRANSFORMER, device=settings.DEVICE)
    print("Engine loaded successfully with eager attention & bfloat16!")
    print(f"Metadata: Layers={engine.num_layers}, FullAttnLayers={engine.full_attn_layers}, Heads={engine.num_heads}")

    # 2. Create session
    session = session_manager.create_session(engine)
    print(f"\n[Session Created] ID: {session.session_id}")

    # 3. Test prefill
    prompt = "注意力机制让语言模型能够精确捕捉长距离上下文语义"
    print(f"[Prefill Prompt]: '{prompt}'")
    target_layer = 3  # Full attention layer in Qwen3.5
    target_head = 0
    res = session.prefill(prompt, target_layer=target_layer, target_head=target_head)

    # 4. Inspect outputs
    tokens = res["tokens"]
    matrix = res["matrix"]
    seq_len = res["seq_len"]

    print("\n--- Prefill 结果分析 ---")
    print(f"Tokens 序列 ({seq_len} tokens): {tokens}")
    print(f"Token IDs: {res['token_ids']}")
    print(f"目标注意力层: Layer {res['layer']}, Head {res['head']}")
    print(f"Attention 张量全局形状 [B, H, L, L]: {res['matrix_shape']}")
    print(f"当前 Head 矩阵尺寸: {len(matrix)} 行 x {len(matrix[0])} 列")
    print(f"数值极值范围: Min = {res['min_val']:.8f}, Max = {res['max_val']:.8f}")
    print(f"因果下三角掩码验证 (上三角严格为0): {'通过 (True)' if res['is_causal'] else '未通过 (False)'}")
    print(f"各行权重归一化和 (Softmax Sum): {res['row_sums']}")
    print(f"past_key_values 类型: {res['past_key_values_type']}")
    
    # 5. Print a sample slice of the matrix
    print("\nAttention 矩阵切片 (前 4 行):")
    for i in range(min(4, seq_len)):
        row_str = " ".join([f"{v:6.4f}" for v in matrix[i][:i+1]])
        print(f"  Row {i} ({tokens[i]:>6}): [{row_str}]")

    # 6. Verify assertions
    assert res['matrix_shape'][2] == seq_len and res['matrix_shape'][3] == seq_len, "Matrix dimensions must match seq_len"
    assert res['is_causal'], "Causal attention matrix must be strictly lower-triangular"
    assert all(0.99 <= s <= 1.01 for s in res['row_sums']), "Each attention row must sum to 1.0"
    assert session.past_key_values is not None, "past_key_values must be initialized"
    print("\n=== 全部断言验证通过：Prefill 逻辑与 KV Cache 初始化完全符合预期 ===")

    # 7. Close session
    session_manager.close_session(session.session_id)
    print(f"[Session Closed] ID: {session.session_id}, past_key_values 引用已清除，显存已清理。")
    print("=" * 60)

if __name__ == "__main__":
    test_prefill_session()
