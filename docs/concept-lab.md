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
{ "prompt": "The capital of France is", "lengths": [64, 128, 256, 512] }
```

实现：`backend/app/services/concept_lab.py::run_memory_growth()`。

后端依次加载四个模型并**逐个卸载**，不会在 RTX 4060 8GB 上同时驻留：

| 模型 | 适配器 | 说明 |
| :--- | :--- | :--- |
| Qwen3.5-0.8B | `TransformerAdapter` | 混合架构，KV Cache + 线性层递归状态 |
| Qwen2.5-0.5B | `PureAttentionAdapter` | 纯注意力对照，全部 24 层为 `full_attention` |
| RWKV-7 Goose 0.4B | `RWKVAdapter` | WKV 递归状态 |
| Mamba-130M | `MambaAdapter` | 基座模型，`conv_state` + `ssm_state` |

Qwen2.5-0.5B 与 Mamba-130M 在界面中明确标注为**基座模型，不是对话模型**。
`PureAttentionAdapter` 只覆盖模型类型标识，加载、prefill、step、状态导出全部
复用 `TransformerAdapter`，不复制 Qwen3.5 的混合架构逻辑，也不新增顶层 Tab。

测量方式是在每个 prefill 长度上各做一次前向，不再逐 token decode：

1. 创建适配器；
2. 用重复到目标长度的同一段英文 token 做一次 prefill；
3. 记录 `cache_bytes` / `kv_bytes` / `linear_state_bytes` / `dtype`；
4. 立即 `unload()` 并清理 CUDA cache；
5. 四个模型都按 64/128/256/512 测量；Transformer 类额外测 1024/2048；
6. RWKV/Mamba 超过 512 的点不实测，图中用虚线并标注“结构上恒定，未实测”。

`cache_bytes` 等字段来自 `InternalState` 的拆分定义：

- `cache_bytes`：推理本身必需的缓存；
- `kv_bytes`：其中真正的 KV Cache 字节数（RWKV/Mamba 为 0）；
- `linear_state_bytes`：缓存中的非 KV 部分（Qwen3.5 线性层递归状态、
  RWKV WKV 状态、Mamba conv+ssm 状态）；
- `cache_bytes = kv_bytes + linear_state_bytes`；
- `viz_history_bytes`：仅为可视化保存的历史，不计入上面的推理缓存；
- `state_bytes = cache_bytes + viz_history_bytes` 保持兼容。

Transformer 的层编号不写死：`load_model()` 从 `config.layer_types` 生成
`full_attn_layers`，`cache_breakdown()` 再按这些层号遍历实际缓存对象。本机
Qwen3.5 的 config 当前给出 `[3, 7, 11, 15, 19, 23]` 六个 `full_attention` 层，
其余 18 层为线性层；Qwen2.5-0.5B 没有 `layer_types`，因此 24 层全部视为
`full_attention`。

### 理论 KV 曲线

只有 Transformer 有 KV Cache 理论公式：

```text
2 × KV层数 × KV头数 × 头维度 × 序列长度 × 每元素字节数
```

参数从加载后的模型 config 和实际 KV 张量读取：

- `layer_types` 存在时，KV 层数取 `full_attention` 层数；
- `num_key_value_heads` / `num_attention_heads` 取 KV 头数；
- `head_dim` 或 `hidden_size // num_attention_heads` 取头维度；
- 每元素字节数取实际 KV 张量的 `element_size()`，取不到时才回退模型参数 dtype。

理论曲线只和**实际 `kv_bytes`** 比较，不和混合架构的总 `cache_bytes` 比较。
接口额外返回 `actual_kv_bytes`、`diff_bytes`、`max_abs_diff_bytes` 与 `matches`；
当前实测两个注意力模型均为逐点相等（最大偏差 0 B）。Qwen3.5 的线性层状态
单独计入 `linear_state_bytes`，不会混进理论 KV 的比较。RWKV / Mamba 没有
KV Cache，接口返回 `available=false` 并说明原因，图中不绘制理论线。

### 图表说明

前端用 SVG 折线图绘制：

- 粗实线：实际总 `cache_bytes`；
- 细实线：混合架构的实际 `kv_bytes`（仅存在线性层状态时显示）；
- 点线：理论 KV；
- 虚线：RWKV/Mamba 超过 512 的“结构恒定，未实测”延伸。

图例标注每个模型缓存的 dtype（例如 Qwen3.5 为 `bfloat16 + float32`，Qwen2.5
为 `bfloat16`）。图下方文案由实际观测值计算（首尾、区间、形态），不预设
“哪条曲线必然增长”的结论；坐标轴支持线性/对数切换。

### 实测原始数字

以下为 2026-11-10 在 RTX 4060 8GB、`transformers 5.17.0` 下的实测值（单位 B）。
`measured=false` 的行为是结构恒定外推，不是实测。

| 模型 | dtype | 长度 | measured | cache_bytes | kv_bytes | linear_state_bytes |
| :--- | :--- | ---: | :---: | ---: | ---: | ---: |
| Qwen3.5-0.8B 混合 | bfloat16 + float32 | 64 | 是 | 20,545,536 | 786,432 | 19,759,104 |
| Qwen3.5-0.8B 混合 | bfloat16 + float32 | 128 | 是 | 21,331,968 | 1,572,864 | 19,759,104 |
| Qwen3.5-0.8B 混合 | bfloat16 + float32 | 256 | 是 | 22,904,832 | 3,145,728 | 19,759,104 |
| Qwen3.5-0.8B 混合 | bfloat16 + float32 | 512 | 是 | 26,050,560 | 6,291,456 | 19,759,104 |
| Qwen3.5-0.8B 混合 | bfloat16 + float32 | 1024 | 是 | 32,342,016 | 12,582,912 | 19,759,104 |
| Qwen3.5-0.8B 混合 | bfloat16 + float32 | 2048 | 是 | 44,924,928 | 25,165,824 | 19,759,104 |
| Qwen2.5-0.5B 纯注意力 | bfloat16 | 64 | 是 | 786,432 | 786,432 | 0 |
| Qwen2.5-0.5B 纯注意力 | bfloat16 | 128 | 是 | 1,572,864 | 1,572,864 | 0 |
| Qwen2.5-0.5B 纯注意力 | bfloat16 | 256 | 是 | 3,145,728 | 3,145,728 | 0 |
| Qwen2.5-0.5B 纯注意力 | bfloat16 | 512 | 是 | 6,291,456 | 6,291,456 | 0 |
| Qwen2.5-0.5B 纯注意力 | bfloat16 | 1024 | 是 | 12,582,912 | 12,582,912 | 0 |
| Qwen2.5-0.5B 纯注意力 | bfloat16 | 2048 | 是 | 25,165,824 | 25,165,824 | 0 |
| RWKV-7 Goose 0.4B | float16 + float32 | 64/128/256/512 | 是 | 6,389,760 | 0 | 6,389,760 |
| RWKV-7 Goose 0.4B | float16 + float32 | 1024/2048 | 否 | 6,389,760 | 0 | 6,389,760 |
| Mamba-130M 基座 | float16 + float32 | 64/128/256/512 | 是 | 2,654,208 | 0 | 2,654,208 |
| Mamba-130M 基座 | float16 + float32 | 1024/2048 | 否 | 2,654,208 | 0 | 2,654,208 |

核对：Qwen2.5-0.5B 的斜率是 `2 × 24 × 2 × 64 × 2 = 12,288 B/token`；
Mamba 的常数缓存 `2,654,208 B ÷ 12,288 B/token = 216 token`。也就是说，
注意力 KV 在约 **216 token** 处与 Mamba 递归状态交叉。Qwen3.5 的 KV 斜率是
`2 × 6 × 2 × 256 × 2 = 12,288 B/token`，与 Qwen2.5 数值相同但来自不同的
层数/头维度组合。

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

`ModelAdapter.top_logits_payload()` 负责统一下降序 top-50 格式；四个适配器的
`step()` 都增加了 `topk_logits` 字段，但正常流式接口的现有字段保持不变。

前端 `frontend/src/utils/sampling.ts` 在浏览器本地计算：

- 原始 softmax；
- Temperature；
- Top-K；
- Top-P；
- 重复惩罚；
- 组合结果（重复惩罚 → Temperature → Top-K → Top-P → softmax）。

`backend/app/services/sampling.py` 是与前端公式对应的后端参考实现。重复惩罚
只作用于返回的 top-50 候选中出现过的已见 token，用于解释机制，不等同于全词表
采样；卡片内会明确提示。

### 前后端一致性核对

`backend/scripts/generate_sampling_fixtures.py` 用后端的 `sampling.py` 生成
`frontend/tests/fixtures/sampling_cases.json`，覆盖 11 组用例：

- 原始 softmax；
- 普通 Temperature 与极小 Temperature（验证 `1e-4` 下限）；
- Top-K、Top-K 大于候选数；
- Top-P；
- 正/负 logits 的重复惩罚；
- 多阶段组合、只保留一个候选、空候选集。

`frontend/scripts/crosscheck_sampling.mjs` 使用已有的 `typescript` 开发依赖，
在内存中编译 UI 实际使用的 `src/utils/sampling.ts`，逐值比较后端 fixture，
容差为 `1e-6`，不新增任何 npm 依赖。

## 4. 测试与截图

后端测试：

```bash
pytest backend/tests/test_concept_sampling.py \
       backend/tests/test_concept_endpoints.py \
       backend/tests/test_attention_baseline.py \
       backend/tests/test_state_bytes_split.py -q
```

前后端采样一致性：

```bash
.venv/Scripts/python.exe backend/scripts/generate_sampling_fixtures.py
cd frontend
npm run test:sampling
```

截图：

- `docs/screenshots/05-concept-memory.png`
- `docs/screenshots/06-concept-tokenize.png`
- `docs/screenshots/07-concept-sampling.png`

所有模型相关测试在权重缺失时会自动 skip；概念 B 的 tokenizer 接口不加载模型
权重；Qwen2.5-0.5B 权重缺失时，基线适配器与理论一致性测试也会自动 skip。
