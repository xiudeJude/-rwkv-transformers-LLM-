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
        └── RWKVAdapter（rwkv）
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
| `model_type` | `str` | 适配器注册名，如 `"transformer"` / `"rwkv"` |
| `layer` | `int` | 目标层 |
| `head` | `int \| null` | 目标头；为 null 表示整层/全部头 |
| `num_layers` | `int` | 模型总层数 |
| `num_heads` | `int` | 模型总头数 |
| `seq_len` | `int` | 当前序列长度 |
| `step` | `int` | 0 表示 prefill，1..N 表示 decode 步 |
| `state_bytes` | `int` | 导出状态实际占用字节数 |
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

- Transformer：KV Cache、prefill 注意力历史、decode 注意力历史的字节数之和。
- RWKV：当前导出层/头的 WKV 状态张量字节数（请求头时只算该头）。
- 计算函数 `estimate_state_bytes()` 只递归张量、list/tuple/dict 和
  `key_cache` / `value_cache` 字段，不会遍历模型计算图。

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

adapter = create_adapter("rwkv")   # 按注册名创建实例
```

配置项：

```dotenv
MODEL_TYPE=transformer       # /api/model/info 默认引擎、Transformer 路由使用
RWKV_MODEL_TYPE=rwkv         # RWKV 路由使用
```

路由在启动时通过 `create_adapter(settings.MODEL_TYPE)` /
`create_adapter(settings.RWKV_MODEL_TYPE)` 获取适配器，而不是硬编码类名。
注意：适配器模块必须被导入后才会执行 `@register_adapter`，`routes.py` 已经
在顶部导入两个适配器模块。

## 5. 新增一个模型

以新增 Mamba 为例：

1. 新建 `backend/app/models/mamba_adapter.py`：

```python
from .adapter_base import ModelAdapter
from .adapter_registry import register_adapter
from ..core.schemas import InternalState, ModelMetadata


@register_adapter("mamba")
class MambaAdapter(ModelAdapter):
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
        return MambaSessionState()

    def prefill(self, state, prompt, target_layer=0, target_head=0):
        ...

    def step(self, state, temperature=0.7, top_k=10,
             repetition_penalty=1.2, target_layer=0, target_head=0):
        ...

    def export_state(self, state, step=None, layer=0, head=None,
                     include_matrix=True):
        return InternalState(
            kind="recurrent_state",
            model_type="mamba",
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
            matrix=...,
            summary=...,
            delta=...,
        )

    def close_state(self, state):
        ...
```

2. 在 `backend/app/api/routes.py` 顶部导入该模块（触发注册），或把导入放入
   `backend/app/models/__init__.py`。
3. 在 `.env` 中把对应槽位的 `MODEL_TYPE` / `RWKV_MODEL_TYPE` 改成 `mamba`，
   并把 `DEFAULT_TRANSFORMER` / `DEFAULT_RWKV` 指向该模型权重。
4. 若前端需要新的展示形式，只需消费 `InternalState.kind` 与对应 payload；
   不再新增后端专用响应字段。

## 6. 兼容性与行为保证

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
