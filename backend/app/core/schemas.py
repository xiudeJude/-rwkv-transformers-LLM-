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
