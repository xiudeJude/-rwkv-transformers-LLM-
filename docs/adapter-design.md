# 统一模型适配层设计（ModelAdapter）

本文档说明 `backend/app/models/adapter_base.py` 定义的统一模型适配层，以及
Transformer / RWKV 两条现有链路的迁移方式。目标是让可解释性可视化不再绑定
某一种架构：只要实现适配器并注册模型类型名，同一套会话、REST/WebSocket
接口和前端可视化都能复用。

## 1. 分层结构

```
API 路由 / WebSocket
        │
        ▼
InferenceSession（通用会话包装器）
        │  create_state / prefill / step / export_state
        ▼
ModelAdapter（抽象基类）
        ├── TransformerAdapter（transformer）
        │     └── PureAttentionAdapter（attention_baseline，仅概念实验室）
        ├── RWKVAdapter（rwkv）
        └── MambaAdapter（mamba）
```

- `ModelAdapter`：模型生命周期、分词/反分词、prefill、逐步生成、内部状态导出。
- `InferenceSession`：只负责会话 id、时间戳、TTL/LRU 所需的访问时间，以及把
  `prefill/step/export_state` 转发给适配器。模型相关的 KV Cache、递归状态、
  注意力历史全部由适配器通过 `create_state()` 创建并维护。
- `SessionManager`：通用 TTL/LRU 会话管理；`RWKVSessionManager` 只是指定
  `session_cls = RWKVSession` 的兼容子类。
- `AdapterRegistry`：模型类型名（字符串）到适配器类的映射。

旧类名 `TransformerEngine` / `RWKVEngine` 分别保留为 `TransformerAdapter` /
`RWKVAdapter` 的别名，现有 imports、脚本和测试不需要改动。

## 2. ModelAdapter 接口

定义位置：`backend/app/models/adapter_base.py`。

### 2.1 生命周期

```python
load_model(model_name_or_path: str, device: Optional[str] = None)
unload()
is_loaded() -> bool
get_metadata() -> ModelMetadata
```

- `unload()` 释放模型权重、hook、分词器和显存缓存；它不负责清理单个会话状态。
- 单个会话状态由 `create_state()` / `close_state(state)` 管理。

### 2.2 分词 / 反分词

```python
tokenize(text: str) -> Dict[str, Any]
decode_token(token_id: int) -> str
detokenize(token_ids: List[int]) -> str
```

`tokenize` 返回的 dict 至少包含 `tokens`、`token_ids`，并可按模型需要附带
`input_ids`、`attention_mask`。`detokenize` 默认逐 token 拼接，RWKV 适配器
覆盖为官方 `PIPELINE.decode`。

### 2.3 逐步生成

```python
create_state() -> Any
prefill(state, prompt, target_layer=0, target_head=0) -> Dict[str, Any]
step(state, temperature=0.7, top_k=10,
     repetition_penalty=1.2, target_layer=0, target_head=0) -> Dict[str, Any]
```

- `state` 是适配器私有的不透明对象。Transformer 中是 KV Cache + 注意力历史；
  RWKV 中是逐层 WKV 状态 + 历史快照。调用方不需要知道其内部结构。
- `prefill` / `step` 返回的 dict 保持原会话接口的字段形状（例如
  `attention_slice`、`state_summary`、`delta`），因此现有 REST/WS 响应和
  `backend/tests` 不受影响。
- 采样、重复惩罚、EOS 判断都在适配器内部完成；新增模型只需实现自己的
  `prefill/step`，不需要再写一套 Session 类。
- `prefill` 可选接收 `prompt_token_ids` / `capture_attention=False`，供概念实验室
  做精确长度、只关心缓存字节数的测量；默认参数保持原有行为。
- `cache_metrics(state)` 返回概念实验室使用的轻量拆分：`cache_bytes`、
  `kv_bytes`、`linear_state_bytes`、缓存 dtype 与层号。默认实现会报错，四个
  已注册适配器均已覆盖。

### 2.4 状态导出

```python
export_state(state, step=None, layer=0, head=None,
             include_matrix=True) -> InternalState
close_state(state) -> None
```

`include_matrix=False` 用于 Level 1 流式消息：只导出当前行/摘要，不携带
O(L²) 注意力矩阵。Level 2 按需查询使用默认值 `True`。

## 3. 统一内部状态 InternalState

定义位置：`backend/app/core/schemas.py`。适配器导出的所有状态都使用这一结构，
由 `kind` 区分类型。

| 字段 | 类型 | 含义 |
| :--- | :--- | :--- |
| `kind` | `str` | `"attention"` 或 `"recurrent_state"` |
| `model_type` | `str` | 适配器注册名，如 `"transformer"` / `"attention_baseline"` / `"rwkv"` / `"mamba"` |
| `layer` | `int` | 目标层 |
| `head` | `int \| null` | 目标头；为 null 表示整层/全部头 |
| `num_layers` | `int` | 模型总层数 |
| `num_heads` | `int` | 模型总头数 |
| `seq_len` | `int` | 当前序列长度 |
| `step` | `int` | 0 表示 prefill，1..N 表示 decode 步 |
| `state_bytes` | `int` | 兼容字段，等于 `cache_bytes + viz_history_bytes` |
| `cache_bytes` | `int` | 推理本身必需的缓存字节数（KV Cache / 递归状态 / Mamba 状态） |
| `viz_history_bytes` | `int` | 仅为可视化额外保存的历史字节数 |
| `tokens` | `List[str]` | token 文本 |
| `token_ids` | `List[int]` | token id |
| `shape` | `List[int]` | 状态张量形状 |
| `matrix` | `Any \| null` | 注意力矩阵或递归状态矩阵；Level 1 可为 null |
| `row` | `List[float] \| null` | 注意力当前行（`kind="attention"`） |
| `summary` | `List[float] \| null` | 各头状态摘要（`kind="recurrent_state"`） |
| `delta` | `List[float] \| null` | 相对上一步的变化（`kind="recurrent_state"`） |
| `head_norms` | `List[float] \| null` | 各头 Frobenius 范数（RWKV 兼容字段） |
| `meta` | `Dict[str, Any]` | 模型相关补充信息 |

### 3.1 state_bytes 的语义

`state_bytes` 保持向后兼容，等于 `cache_bytes + viz_history_bytes`：

- Transformer：
  - `cache_bytes` = 实际 `past_key_values` 字节数（Qwen 混合架构中可能包含
    线性层递归状态，以实测为准）；
  - `viz_history_bytes` = prefill 注意力历史 + decode 注意力历史字节数。
- RWKV：
  - `cache_bytes` = 当前完整递归状态 `current_state` 字节数；
  - `viz_history_bytes` = `history_states` 总字节数减去当前状态字节数，
    避免与推理缓存重复计数。
- Mamba：
  - `cache_bytes` = conv_state + ssm_state 张量字节数；
  - `viz_history_bytes` 当前为 0（只保存轻量摘要，不保存张量历史）。
- 计算函数 `estimate_state_bytes()` 递归张量、list/tuple/dict，并遍历缓存对象的
  公开字段（兼容 `DynamicCache.layers[i].keys/values`、`key_cache`、`value_cache`
  等不同版本布局）；遇到 `nn.Module`、类对象和配置元数据时剪枝，不会遍历模型计算图。

概念实验室的 A 卡片直接使用 `cache_bytes` 绘制四个模型的 prefill 缓存曲线；
Transformer 还会把缓存拆成 `kv_bytes` 与 `linear_state_bytes`，理论 KV 只与
`kv_bytes` 比较。字段拆分与数据来源见 `docs/concept-lab.md`。

### 3.2 前端消费方式

REST/WS 响应在保留全部旧字段的同时新增 `state` 字段：

- `POST /api/inspect/single-step` → `state`
- `POST /api/session/create` / `POST /api/rwkv/session/create` → `state`
- `POST /api/session/{id}/step` / `POST /api/rwkv/session/{id}/step` → `state`
- `GET /api/session/{id}/attention` → `state`
- `GET /api/session/{id}/rwkv_state` → `state`
- WebSocket `token_step` 消息 → `state`（不含完整矩阵）

`frontend/src/types/schema.ts` 定义了 `InternalState`；`App.tsx` 优先读取
`state`，旧字段作为回退。因此前端只依赖统一结构，同时旧服务端/旧客户端
仍可工作。

## 4. 注册表

定义位置：`backend/app/models/adapter_registry.py`。

```python
@register_adapter("transformer")
class TransformerAdapter(ModelAdapter):
    ...

@register_adapter("rwkv")
class RWKVAdapter(ModelAdapter):
    ...

@register_adapter("attention_baseline")
class PureAttentionAdapter(TransformerAdapter):
    # 只覆盖 model_type，加载/prefill/step/导出全部复用 TransformerAdapter
    ...

adapter = create_adapter("rwkv")   # 按注册名创建实例
```

配置项：

```dotenv
MODEL_TYPE=transformer                       # /api/model/info 默认引擎、Transformer 路由使用
RWKV_MODEL_TYPE=rwkv                         # RWKV 路由使用
MAMBA_MODEL_TYPE=mamba                       # Mamba 路由使用
ATTENTION_BASELINE_MODEL_TYPE=attention_baseline  # 概念实验室纯注意力对照
DEFAULT_ATTENTION_BASELINE=Qwen/Qwen2.5-0.5B
```

路由在启动时通过 `create_adapter(settings.MODEL_TYPE)` /
`create_adapter(settings.RWKV_MODEL_TYPE)` / `create_adapter(settings.MAMBA_MODEL_TYPE)`
获取适配器，而不是硬编码类名。`PureAttentionAdapter` 只由概念实验室直接创建，
不新增顶层架构 Tab，也不改变 Qwen3.5 的 GGUF 加载与推理行为。
注意：适配器模块必须被导入后才会执行 `@register_adapter`，`routes.py` 已经
在顶部导入四个适配器模块。

## 5. 新增一个模型（通用步骤）

下面用最小骨架说明注册步骤；Mamba 的完整真实案例见第 6 节。

1. 新建适配器模块，例如 `backend/app/models/my_adapter.py`：

```python
from .adapter_base import ModelAdapter
from .adapter_registry import register_adapter
from ..core.schemas import InternalState, ModelMetadata


@register_adapter("my-model")
class MyAdapter(ModelAdapter):
    def load_model(self, model_name_or_path, device=None):
        ...

    def unload(self):
        ...

    def get_metadata(self):
        return ModelMetadata(...)

    def tokenize(self, text):
        ...

    def decode_token(self, token_id):
        ...

    def create_state(self):
        return MySessionState()

    def prefill(self, state, prompt, target_layer=0, target_head=0):
        ...

    def step(self, state, temperature=0.7, top_k=10,
             repetition_penalty=1.2, target_layer=0, target_head=0):
        ...

    def export_state(self, state, step=None, layer=0, head=None,
                     include_matrix=True):
        return InternalState(
            kind="recurrent_state",
            model_type="my-model",
            layer=layer,
            head=head,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            seq_len=...,
            step=...,
            state_bytes=...,
            tokens=...,
            token_ids=...,
            shape=...,
            matrix=None,
            summary=...,
            delta=...,
        )

    def close_state(self, state):
        ...
```

2. 在 `backend/app/api/routes.py` 顶部导入该模块（触发注册），或把导入放入
   `backend/app/models/__init__.py`。
3. 在 `.env` 中为对应槽位配置模型类型与权重路径，例如
   `MODEL_TYPE=my-model`、`RWKV_MODEL_TYPE=my-model` 或
   `MAMBA_MODEL_TYPE=my-model`，并同步配置对应的 `DEFAULT_*` 权重路径。
4. 若前端需要新的展示形式，只需消费 `InternalState.kind` 与 `meta`/组件 payload；
   不再新增后端专用响应字段。

## 6. 新增递归类模型：Mamba 案例

`backend/app/models/mamba_engine.py` 已经实现并注册了 `MambaAdapter`，
模型类型名为 `"mamba"`。这个案例验证了 `kind="recurrent_state"` 抽象不止适用于
RWKV：Mamba 没有注意力 head，状态形状也与 RWKV 的 `[16, 64, 64]` 完全不同，
但复用同一个 `ModelAdapter`、`InferenceSession` 和前端状态面板。

### 6.1 Mamba 状态映射

- `MambaAdapter` 不伪造注意力：`num_heads=0`、`head=None`、`matrix=None`。
- `transformers` 的 Mamba 缓存是 `DynamicCache`，逐层包含：
  - `conv_states[0]`：卷积状态，形状 `[1, 1536, 4]`，fp16；
  - `recurrent_states[0]`：SSM 状态，形状 `[1, 1536, 16]`，float32。
- 适配器把这些状态放进 `InternalState.meta["components"]`，每个组件包含
  `name`、`shape`、`dtype`、`bytes`、`layer_norms`（逐层 Frobenius 范数摘要）：

```json
{
  "kind": "recurrent_state",
  "model_type": "mamba",
  "layer": 0,
  "head": null,
  "num_layers": 24,
  "num_heads": 0,
  "seq_len": 5,
  "step": 0,
  "state_bytes": 2654208,
  "matrix": null,
  "meta": {
    "components": [
      {
        "name": "conv_state",
        "shape": [1, 1536, 4],
        "dtype": "float16",
        "bytes": 294912,
        "layer_norms": ["...24 个浮点数..."]
      },
      {
        "name": "ssm_state",
        "shape": [1, 1536, 16],
        "dtype": "float32",
        "bytes": 2359296,
        "layer_norms": ["...24 个浮点数..."]
      }
    ]
  }
}
```

`state_bytes` 是各组件的张量字节数之和，与 prompt 长度、已生成 token 数无关。
前端 `MambaStatePanel` 只读取 `InternalState.kind` 和
`meta.components`，按组件通用渲染，不写死 Mamba 的层数、shape 或 dtype。

### 6.2 配置与限制

```dotenv
DEFAULT_MAMBA=state-spaces/mamba-130m-hf
MAMBA_MODEL_TYPE=mamba
```

`state-spaces/mamba-130m-hf` 是**基座模型，不是对话/指令微调模型**；它适合
观察状态和生成续写，不应按聊天助手效果评估。

当前机器未安装 `mamba-ssm` / `causal-conv1d` 时，`transformers` 会回退到纯
PyTorch reference 实现。为了保持交互可用，Mamba 路由限制：

- prompt 最多 512 tokens；
- 单次 WebSocket 生成最多 128 个新 token。

### 6.3 新增模型验证清单

1. 适配器能通过 `load_model` 加载 fp16 权重；
2. `prefill` 与 `step` 能返回 logits 和模型私有 cache；
3. `export_state` 返回统一 `InternalState`，并正确填充通用元信息；
4. `state_bytes` 在不同 prompt 长度下保持一致；
5. 前端只消费统一结构，未写死模型名称。

## 7. 纯注意力对照：PureAttentionAdapter

`backend/app/models/attention_baseline.py` 注册了 `"attention_baseline"`，用于
概念实验室 A 卡片的纯注意力基线。它继承 `TransformerAdapter`，只覆盖
`model_type`；没有复制加载、分词、prefill、step 或状态导出代码。

- 默认权重：`Qwen/Qwen2.5-0.5B`，非 gated，fp16/bf16 参数约 988 MB；
- config 没有 `layer_types`，因此 24 层全部是 `full_attention`；
- 每个 token 的 KV 字节数为 `2 × 24 × 2 × 64 × 2 = 12,288 B`；
- `cache_metrics()` 返回 `kv_bytes == cache_bytes`、`linear_state_bytes == 0`；
- 测试 `test_attention_baseline.py` 从 config 推导每 token 字节数，并逐点核对
  实际 `kv_bytes` 与理论值完全相等；
- 该适配器只在概念实验室内部创建，不注册顶层 Tab。

Qwen3.5 的混合缓存拆分走同一个 `TransformerAdapter.cache_breakdown()`：层号来自
`config.layer_types`，不写死；全注意力层的 `keys/values` 计入 `kv_bytes`，
其余缓存（线性层 `conv_states` / `recurrent_states`）计入 `linear_state_bytes`。
`test_state_bytes_split.py` 同时断言 `cache_bytes = kv_bytes + linear_state_bytes`
且 `kv_bytes` 与理论公式逐点相等。

## 8. 兼容性与行为保证

- `TransformerEngine` / `RWKVEngine` 保留为适配器别名。
- `InferenceSession` / `RWKVSession` 保留原类名与公开方法：
  `prefill`、`step`、`get_attention_detail`、`get_rwkv_state`、`close`；
  旧属性（`tokens`、`token_ids`、`past_key_values`、`current_state`、
  `history_states`、`step_count`、`is_finished` 等）通过会话包装器转发到
  适配器状态。
- REST/WS 原有字段全部保留；`state` 是增量字段。
- WebSocket 推理和状态导出继续使用 `asyncio.to_thread`，避免同步重计算阻塞
  事件循环。
- `backend/tests` 中的现有测试不需要修改。
