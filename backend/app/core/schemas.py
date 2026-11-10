from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class TokenCandidate(BaseModel):
    token: str
    prob: float
    id: int

class ModelMetadata(BaseModel):
    model_id: str
    model_type: str  # 'transformer' | 'rwkv'
    num_layers: int
    num_heads: int
    hidden_size: int
    vocab_size: int
    device: str
    full_attn_layers: List[int] = Field(default_factory=list)

class AttentionData(BaseModel):
    layer: int
    head: int
    matrix: List[List[float]]  # [L, L] Attention weights
    tokens: List[str]          # token str strings for axis labels
    token_ids: List[int]

class InternalState(BaseModel):
    """统一内部状态导出结构。

    kind 用于区分状态类型：
      - "attention": Transformer 注意力权重
      - "recurrent_state": RWKV 递归状态

    所有适配器都必须填充通用元信息（层数、头数、序列长度、状态字节数），
    matrix/row/summary/delta 等字段按 kind 选用。
    """
    kind: str
    model_type: str = ""
    layer: int = 0
    head: Optional[int] = None
    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    step: int = 0
    # state_bytes 保持兼容，等于 cache_bytes + viz_history_bytes。
    state_bytes: int = 0
    # 推理本身必需的缓存（KV Cache、递归状态、Mamba conv/ssm 状态）。
    cache_bytes: int = 0
    # 仅为了可视化额外保存的历史（注意力历史、RWKV history_states 等）。
    viz_history_bytes: int = 0
    tokens: List[str] = Field(default_factory=list)
    token_ids: List[int] = Field(default_factory=list)
    shape: List[int] = Field(default_factory=list)
    matrix: Optional[Any] = None
    row: Optional[List[float]] = None
    summary: Optional[List[float]] = None
    delta: Optional[List[float]] = None
    head_norms: Optional[List[float]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class CreateSessionRequest(BaseModel):
    prompt: str = Field(..., description="Initial prompt to prefill")
    layer: int = Field(3, description="Target layer index to monitor")
    head: int = Field(0, description="Target head index to monitor")

class StepRequest(BaseModel):
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(10, ge=1, le=100)
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0, description="Repetition penalty factor")
    layer: int = Field(3, description="Target layer index")
    head: int = Field(0, description="Target head index")

class SingleStepInspectRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to inspect")
    model_name: Optional[str] = Field(None, description="HuggingFace model ID or local path (defaults to configured model)")
    layer: int = Field(3, description="Target layer index to extract")
    head: int = Field(0, description="Target attention head index to extract")
    top_k: int = Field(10, description="Number of next-token candidates to return")
    temperature: float = Field(0.7, ge=0.01, le=2.0)

class SingleStepInspectResponse(BaseModel):
    status: str = "success"
    prompt: str
    model_meta: ModelMetadata
    attention: AttentionData
    next_token: TokenCandidate
    top_k_candidates: List[TokenCandidate]
    state: Optional[InternalState] = None
