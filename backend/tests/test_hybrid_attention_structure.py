import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def inspect_hybrid_attentions():
    print("=" * 70)
    print("【Qwen3.5 混合架构 outputs.attentions 结构专项实测】")
    
    model_path = "D:/rwkv/models/unsloth/Qwen3.5-0.8B-GGUF"
    gguf_file = "Qwen3.5-0.8B-Q8_0.gguf"
    
    print(f"Loading {model_path} ({gguf_file})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, gguf_file=gguf_file, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        gguf_file=gguf_file,
        device_map="cuda",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.config.output_attentions = True
    model.eval()

    prompt = "测试Qwen3.5混合架构注意力输出"
    enc = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)

    attentions = outputs.attentions
    print("\n" + "-" * 50)
    print("1. outputs.attentions 的基础属性:")
    print(f"   type(outputs.attentions) = {type(attentions)}")
    print(f"   len(outputs.attentions)  = {len(attentions) if attentions is not None else None}")
    print(f"   总层数 (num_hidden_layers) = {model.config.num_hidden_layers}")
    
    layer_types = getattr(model.config, "layer_types", [])
    print(f"\n2. model.config.layer_types 全局分布 (共 {len(layer_types)} 层):")
    full_indices = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    linear_indices = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    print(f"   full_attention   层索引: {full_indices} (共 {len(full_indices)} 层)")
    print(f"   linear_attention 层索引: {linear_indices} (共 {len(linear_indices)} 层)")

    print("\n3. outputs.attentions 内部每个元素的类型与形状:")
    if attentions is not None:
        for idx, item in enumerate(attentions):
            if item is None:
                print(f"   tuple[{idx}]: None")
            elif isinstance(item, torch.Tensor):
                print(f"   tuple[{idx}]: Tensor, shape = {list(item.shape)}, dtype = {item.dtype}, min = {item.min().item():.4f}, max = {item.max().item():.4f}")
            else:
                print(f"   tuple[{idx}]: {type(item)}")

    print("\n4. 逐层探测: model.model.layers[i] 子模块类型与返回关联:")
    for layer_i in range(min(8, len(model.model.layers))):  # inspect first 8 layers
        layer_mod = model.model.layers[layer_i]
        has_self = hasattr(layer_mod, "self_attn")
        has_linear = hasattr(layer_mod, "linear_attn")
        attn_name = "self_attn (Full)" if has_self else ("linear_attn (Linear)" if has_linear else "Unknown")
        print(f"   Layer {layer_i:2d}: 架构类型={layer_types[layer_i]:<16} | 子模块={attn_name}")

    print("\n5. 映射关系结论:")
    if attentions is not None and len(attentions) == len(full_indices):
        print(f"   ★ 关键发现: len(outputs.attentions) == {len(attentions)}, 仅包含 {len(full_indices)} 个 full_attention 层！")
        print("   ★ 映射对照表 (UI层号 -> attentions tuple 索引):")
        for tuple_idx, layer_num in enumerate(full_indices):
            print(f"      UI Layer {layer_num:2d} (Full Attention)   -->   outputs.attentions[{tuple_idx}]")
    elif attentions is not None and len(attentions) == 24:
        print("   ★ len(outputs.attentions) == 24, 各层保持 1:1 对齐。")

    print("=" * 70)

if __name__ == "__main__":
    inspect_hybrid_attentions()
