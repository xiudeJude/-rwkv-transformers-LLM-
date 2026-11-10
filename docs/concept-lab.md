# 概念实验室（Concept Lab）

概念实验室是顶部与 Transformer / RWKV / Mamba 三个架构 Tab 并列的独立视图。
它不修改三个架构页面原有的生成、热力图、瀑布流与状态面板分支，只消费后端
新增的实验接口。

入口：顶部 `概念实验室` 按钮。实验卡片的数据来源如下。

## 1. 概念 A：状态与内存增长对比

### 数据来源

接口：

```text
POST /api/concept/memory-growth
{ "prompt": "The capital of France is", "max_tokens": 8 }
```

实现：`backend/app/services/concept_lab.py::run_memory_growth()`。

后端依次加载三个模型并**逐个卸载**，不会在 RTX 4060 8GB 上同时驻留：

1. 创建适配器；
2. `prefill` 同一段英文 prompt；
3. 记录 prefill 与每一步 decode 的 `InternalState.cache_bytes`；
4. 立即 `unload()` 并清理 CUDA cache；
5. 返回三条实际曲线与理论 KV 信息。

`cache_bytes` 来自 `InternalState` 的拆分字段：

- `cache_bytes`：推理本身必需的缓存（Transformer KV Cache / 混合线性层状态，
  RWKV 当前递归状态，Mamba conv+ssm 状态）；
- `viz_history_bytes`：仅为可视化保存的注意力历史或 RWKV `history_states`；
- `state_bytes = cache_bytes + viz_history_bytes` 保持兼容。

### 理论 KV 曲线

只有 Transformer 有 KV Cache 理论公式：

```text
2 × KV层数 × KV头数 × 头维度 × 序列长度 × 每元素字节数
```

参数从加载后的模型 config 读取：

- `layer_types` 存在时，KV 层数取 `full_attention` 层数；
- `num_key_value_heads` / `num_attention_heads` 取 KV 头数；
- `head_dim` 或 `hidden_size // num_attention_heads` 取头维度；
- 每元素字节数取模型参数 dtype。

Qwen3.5 可能是混合结构，部分层没有传统 KV Cache；理论曲线只计算 KV 部分，
实际 `cache_bytes` 还包含线性层递归状态，因此二者不应强行假设线性对应。
RWKV / Mamba 没有 KV Cache，接口返回 `available=false` 并说明原因，图中不绘制
它们的理论线。

### 图表说明

前端用 SVG 折线图绘制实际 `cache_bytes`，虚线绘制理论 KV 值。图下方文案由
观测值计算（首尾、区间、形态），不预设“哪条曲线必然增长”的结论。

## 2. 概念 B：分词对比

### 数据来源

接口：

```text
POST /api/concept/tokenize
{ "text": "Hello 世界" }
```

实现：`concept_lab.tokenize_compare()`。

只加载 tokenizer，不加载模型权重：

- Transformer：Qwen tokenizer（GGUF 目录通过 `gguf_file` 显式解析）；
- RWKV：`PIPELINE(None, "rwkv_vocab_v20230424")`，只使用官方 TRIE tokenizer；
- Mamba：`AutoTokenizer`（GPTNeoX tokenizer）。

每个 token 返回文本、token id、UTF-8 字节数；每个 tokenizer 返回 token 总数与
总字节数。前端用彩色 token 块并排展示，支持中文与英文示例。

## 3. 概念 C：采样如何改变概率分布

### 数据来源

接口：

```text
POST /api/concept/sample-logits
{ "prompt": "The capital of France is", "model_type": "mamba" }
```

实现：`concept_lab.sample_topk_logits()`。一次只加载一个模型，执行 prefill 与
一步适配器 `step`，返回该步原始 top-50 logits：

```json
{
  "topk_logits": [
    { "id": 275, "token": " in", "logit": 47.4375 }
  ],
  "prompt_token_ids": [510, 5347, ...]
}
```

`ModelAdapter.top_logits_payload()` 负责统一下降序 top-50 格式；三个适配器的
`step()` 都增加了 `topk_logits` 字段，但正常流式接口的现有字段保持不变。

前端 `frontend/src/utils/sampling.ts` 在浏览器本地计算：

- 原始 softmax；
- Temperature；
- Top-K；
- Top-P；
- 重复惩罚；
- 组合结果（重复惩罚 → Temperature → Top-K → Top-P → softmax）。

`backend/app/services/sampling.py` 是与前端公式对应的后端参考实现；
`backend/tests/test_concept_sampling.py` 用真实 Mamba top-50 logits 验证排序、
采样 id 与分布归一化。重复惩罚只作用于返回的 top-50 候选中出现过的已见 token，
用于解释机制，不等同于全词表采样；卡片内会明确提示。

右侧现有的 Temperature / Top-K / 重复惩罚控件保持不变，本卡片是解释性实验。

## 4. 测试与截图

测试：

```bash
pytest backend/tests/test_concept_sampling.py \
       backend/tests/test_concept_endpoints.py -q
```

截图：

- `docs/screenshots/05-concept-memory.png`
- `docs/screenshots/06-concept-tokenize.png`
- `docs/screenshots/07-concept-sampling.png`

所有模型相关测试在权重缺失时会自动 skip；概念 B 的 tokenizer 接口不加载模型权重。
