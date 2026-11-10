# Qwen3.5-0.8B GGUF Q8_0 vs bf16 KL 实测

## 测量命令

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe backend\scripts\manual\measure_kl_divergence.py `
  --device cuda `
  --output-json C:\Users\xiude\AppData\Local\Temp\opencode\kl_real_result.json
```

## 模型与设备

| 项目 | 值 |
| :--- | :--- |
| 量化模型 | `D:/rwkv/models/unsloth/Qwen3.5-0.8B-GGUF`，`Qwen3.5-0.8B-Q8_0.gguf` |
| 未量化基线 | `backend/models/Qwen3.5-0.8B`，`model.safetensors-00001-of-00001.safetensors` |
| 基线仓库 | `Qwen/Qwen3.5-0.8B`（GGUF 元数据 `general.base_model.0.repo_url` 指向该仓库） |
| 设备 | CUDA，RTX 4060 8GB |
| 模型 dtype | GGUF Q8_0 / 基线 bf16 |
| 词表 | 248,320，logits 使用全词表，不是 top-k 截断 |

## 方法

- 使用未量化模型的 tokenizer 生成 token id，两个模型喂入同一组 `input_ids`；
  4 条 prompt 的 `token_ids_match` 均为 `true`。
- 每个 prompt 只看最后一个 prompt token 的 next-token logits 分布。
- Logits KL：`KL(P_bf16 || Q_Q8_0)`，对整个 248,320 词表求和。
- Attention KL：对 6 个 Full Attention 层（`[3, 7, 11, 15, 19, 23]`）
  的完整因果注意力矩阵、全部 8 个 head 求平均。
- 每项测量使用一次 forward，无采样随机性。

## 每个 prompt 的结果

| # | Prompt | Prompt tokens | Logits KL | 与 bf16 tokenizer 一致 |
| ---: | :--- | ---: | ---: | :---: |
| 1 | Language models are capable of generalizing across varied tasks. | 11 | 0.001725 | 是 |
| 2 | The transformer architecture relies on self-attention to model long-range context. | 14 | 0.001361 | 是 |
| 3 | Quantization reduces memory usage but can change the model's output distribution. | 14 | 0.001819 | 是 |
| 4 | A fair comparison uses the same tokenized prompt and evaluates the full vocabulary. | 15 | 0.002205 | 是 |

## 汇总

```text
prompt 数量: 4
prompt 长度: 11 / 14 / 14 / 15 tokens
logits 全词表: true (vocab_size=248320)

logits KL 均值: 0.001777
logits KL 最大: 0.002205
logits KL 最小: 0.001361
```

6 个 Full Attention 层的平均 Attention KL：

| 层（缓存内 full-attention 序号） | 实际层号 | 平均 Attention KL |
| ---: | ---: | ---: |
| 0 | 3 | 0.000228 |
| 1 | 7 | 0.000282 |
| 2 | 11 | 0.000354 |
| 3 | 15 | 0.000330 |
| 4 | 19 | 0.000637 |
| 5 | 23 | 0.000324 |

JSON 中的状态为：

```json
{
  "validation_status": "REAL_KL_MEASURED",
  "has_unquantized_baseline": true
}
```

## 结论与限制

- Q8_0 与 bf16 的分布差异很小，但不是“0”或“约等于 0”：本批 4 条英文
  prompt 的 logits KL 均值为 **0.001777**，最大值为 **0.002205**。
- 该结果只覆盖 4 条短英文 prompt 的最后一个 prompt token；不代表长文本生成、
  多轮对话或所有语言的全局误差。
- Attention KL 只统计 Qwen3.5 的 6 个 Full Attention 层；18 个 Gated DeltaNet
  线性递归层没有传统 Softmax 注意力矩阵，未参与该指标。
- 测量脚本在缺少未量化权重时会返回 `SELF_CHECK_ONLY_NO_BASELINE`，只表示
  散度计算函数通过自检，不表示 KL 已验证。
