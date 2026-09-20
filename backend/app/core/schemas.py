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

class AttentionData(BaseModel):
    layer: int
    head: int
    matrix: List[List[float]]  # [L, L] Attention weights
    tokens: List[str]          # token str strings for axis labels
    token_ids: List[int]

class SingleStepInspectRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to inspect")
    model_name: Optional[str] = Field("Qwen/Qwen2.5-0.5B-Instruct", description="HuggingFace model ID or local path")
    layer: int = Field(0, description="Target layer index to extract")
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
