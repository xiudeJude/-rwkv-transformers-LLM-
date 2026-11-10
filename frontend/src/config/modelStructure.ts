/**
 * Static structural metadata used before any model weights are loaded.
 * Qwen3.5-0.8B config.layer_types has full_attention at these six layers.
 */
export const STATIC_MODEL_STRUCTURE = {
  transformer: {
    numLayers: 24,
    numHeads: 8,
    fullAttnLayers: [3, 7, 11, 15, 19, 23],
  },
  rwkv: {
    numLayers: 24,
    numHeads: 16,
  },
  mamba: {
    numLayers: 24,
    numHeads: 0,
  },
} as const;

export type StaticModelArch = keyof typeof STATIC_MODEL_STRUCTURE;
