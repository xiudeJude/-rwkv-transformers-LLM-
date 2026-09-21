export interface TokenCandidate {
  token: string;
  prob: number;
  id: number;
}

export interface ModelMetadata {
  model_id: string;
  model_type: 'transformer' | 'rwkv';
  num_layers: number;
  num_heads: number;
  hidden_size: number;
  vocab_size: number;
  device: string;
  full_attn_layers?: number[];
}

export interface AttentionData {
  layer: number;
  head: number;
  matrix: number[][]; // [L, L] attention weights
  tokens: string[];
  token_ids: number[];
}

export interface SingleStepInspectRequest {
  prompt: string;
  model_name?: string;
  layer: number;
  head: number;
  top_k: number;
  temperature: number;
}

export interface SingleStepInspectResponse {
  status: string;
  prompt: string;
  model_meta: ModelMetadata;
  attention: AttentionData;
  next_token: TokenCandidate;
  top_k_candidates: TokenCandidate[];
}

export interface TokenStepPayload {
  type: 'token_step' | 'error';
  step: number;
  token: string;
  token_id: number;
  token_prob: number;
  topk_candidates: TokenCandidate[];
  attention_row?: number[];
  is_finished: boolean;
  finish_reason?: string;
  message?: string;
}


export interface SessionAttentionResponse {
  status: string;
  session_id: string;
  step: number;
  layer: number;
  head: number;
  tokens: string[];
  seq_len: number;
  attention_row: number[];
  full_matrix: number[][];
}

