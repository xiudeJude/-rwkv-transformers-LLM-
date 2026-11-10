export interface LogitItem {
  id: number;
  token: string;
  logit: number;
}

export interface SamplingParams {
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  seenTokenIds: number[];
}

const softmax = (logits: number[]): number[] => {
  if (logits.length === 0) return [];
  const finite = logits.filter((value) => Number.isFinite(value));
  const maxValue = finite.length > 0 ? Math.max(...finite) : 0;
  const exps = logits.map((value) => (Number.isFinite(value) ? Math.exp(value - maxValue) : 0));
  const total = exps.reduce((sum, value) => sum + value, 0) || 1;
  return exps.map((value) => value / total);
};

const applyRepetition = (
  logits: number[],
  tokenIds: number[],
  seenIds: number[],
  penalty: number
): number[] => {
  const result = [...logits];
  if (penalty === 1) return result;
  const seen = new Set(seenIds);
  tokenIds.forEach((tokenId, index) => {
    if (!seen.has(tokenId)) return;
    result[index] = result[index] > 0 ? result[index] / penalty : result[index] * penalty;
  });
  return result;
};

const applyTemperature = (logits: number[], temperature: number): number[] => {
  const value = Math.max(temperature, 1e-4);
  return logits.map((logit) => logit / value);
};

const applyTopK = (logits: number[], topK: number): number[] => {
  if (topK <= 0 || topK >= logits.length) return [...logits];
  const threshold = [...logits].sort((a, b) => b - a)[topK - 1];
  return logits.map((logit) => (logit >= threshold ? logit : -Infinity));
};

const applyTopP = (logits: number[], topP: number): number[] => {
  if (topP >= 1 || logits.length === 0) return [...logits];
  const probs = softmax(logits);
  const order = probs
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value);
  const kept = new Set<number>();
  let cumulative = 0;
  for (const item of order) {
    kept.add(item.index);
    cumulative += item.value;
    if (cumulative >= topP) break;
  }
  return logits.map((logit, index) => (kept.has(index) ? logit : -Infinity));
};

export type SamplingMode = 'raw' | 'temperature' | 'topK' | 'topP' | 'repetition' | 'combined';

export const samplingModeLabels: Record<SamplingMode, string> = {
  raw: '原始',
  temperature: 'Temperature',
  topK: 'Top-K',
  topP: 'Top-P',
  repetition: '重复惩罚',
  combined: '组合结果',
};

export const computeDistribution = (
  items: LogitItem[],
  params: SamplingParams,
  mode: SamplingMode
): number[] => {
  const tokenIds = items.map((item) => item.id);
  let transformed = items.map((item) => item.logit);
  if (mode === 'repetition' || mode === 'combined') {
    transformed = applyRepetition(
      transformed,
      tokenIds,
      params.seenTokenIds,
      params.repetitionPenalty
    );
  }
  if (mode === 'temperature' || mode === 'combined') {
    transformed = applyTemperature(transformed, params.temperature);
  }
  if (mode === 'topK' || mode === 'combined') {
    transformed = applyTopK(transformed, params.topK);
  }
  if (mode === 'topP' || mode === 'combined') {
    transformed = applyTopP(transformed, params.topP);
  }
  return softmax(transformed);
};
