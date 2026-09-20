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
