import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.transformer import TransformerEngine
from app.core.config import settings

def test_engine_hook():
    print(f"Testing TransformerEngine on device: {settings.DEVICE}...")
    engine = TransformerEngine()
    
    # We can test with Qwen2.5-0.5B-Instruct
    model_id = settings.DEFAULT_TRANSFORMER
    print(f"Loading model: {model_id}...")
    
    # Ensure CPU device if CUDA is not available
    device = "cpu"
    engine.load_model(model_id, device=device)
    
    meta = engine.get_metadata()
    print("Model loaded successfully!")
    print(f"Metadata: Layers={meta.num_layers}, Heads={meta.num_heads}, HiddenSize={meta.hidden_size}")
    
    prompt = "注意力机制让语言模型能够理解上下文关系"
    print(f"Running single step inspection for prompt: '{prompt}'...")
    res = engine.inspect_single_step(prompt=prompt, layer=0, head=0, top_k=5)
    
    print(f"Inspection Result Status: {res.status}")
    print(f"Tokens ({len(res.attention.tokens)}): {res.attention.tokens}")
    matrix = res.attention.matrix
    print(f"Attention Matrix Shape: {len(matrix)} x {len(matrix[0])}")
    print("Layer 0 Head 0 Attention Matrix row 0:", matrix[0])
    print(f"ArgMax Next Token: '{res.next_token.token}' (Prob: {res.next_token.prob})")
    print("Top-K Candidates:")
    for cand in res.top_k_candidates:
        print(f"  - '{cand.token}' (p={cand.prob})")
        
    assert len(matrix) == len(res.attention.tokens), "Matrix rows must match token count"
    assert len(matrix[0]) == len(res.attention.tokens), "Matrix cols must match token count"
    print("=== TEST PASSED: Hook accurately extracted attention tensor! ===")

if __name__ == "__main__":
    test_engine_hook()
