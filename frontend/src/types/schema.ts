export interface TokenCandidate {
  token: string;
  prob: number;
  id: number;
}

export interface ModelMetadata {
  model_id: string;
  model_type: 'transformer' | 'rwkv' | 'mamba';
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

export interface RecurrentStateComponent {
  name: string;
  shape: number[];
  dtype: string;
  bytes: number;
  layer_norms: number[];
  layer_count?: number;
}

/**
 * 后端统一的内部状态结构。
 * kind === 'attention' 时 matrix 为 [L, L]，row 为当前步注意力行；
 * kind === 'recurrent_state' 时由适配器定义 matrix/summary 等载荷；
 * Mamba 等模型的递归状态组件放在 meta.components，面板按组件通用渲染。
 */
export interface InternalState {
  kind: 'attention' | 'recurrent_state';
  model_type: string;
  layer: number;
  head?: number | null;
  num_layers: number;
  num_heads: number;
  seq_len: number;
  step: number;
  state_bytes: number;
  tokens: string[];
  token_ids: number[];
  shape: number[];
  matrix?: number[][] | number[][][] | null;
  row?: number[] | null;
  summary?: number[] | null;
  delta?: number[] | null;
  head_norms?: number[] | null;
  meta?: {
    components?: RecurrentStateComponent[];
    model_id?: string;
    d_inner?: number;
    state_size?: number;
    conv_kernel?: number;
    prompt_limit?: number;
    max_new_tokens?: number;
    [key: string]: unknown;
  };
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
  state?: InternalState;
}

export interface TokenStepPayload {
  type: 'token_step' | 'error';
  step: number;
  token: string;
  token_id: number;
  token_prob: number;
  topk_candidates: TokenCandidate[];
  attention_row?: number[];
  state?: InternalState;
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
  state?: InternalState;
}

export interface RWKVTokenStepPayload extends TokenStepPayload {
  state_summary?: number[]; // length 16
  delta?: number[];         // length 16
  layer?: number;
}

export interface RWKVStateResponse {
  status: string;
  session_id: string;
  step: number;
  layer: number;
  head?: number | null;
  tokens: string[];
  shape: number[];
  head_norms: number[];
  matrix: number[][] | number[][][]; // [64, 64] if head specified, else [16, 64, 64]
  state?: InternalState;
}


