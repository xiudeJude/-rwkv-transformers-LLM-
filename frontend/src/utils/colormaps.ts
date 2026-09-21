// Log-scale mapping for attention weights, keeping long tail weak values clearly discernible
export function toLogScale(val: number, c: number = 99): number {
  if (val <= 0) return 0;
  return Math.log10(1 + c * Math.min(Math.max(val, 0), 1)) / Math.log10(1 + c);
}

// Scientific colormaps for attention matrices: Viridis
export function getViridisColor(val: number, useLogScale: boolean = true): [number, number, number] {
  // Apply log-scale transform if enabled
  const scaled = useLogScale ? toLogScale(val) : val;
  // Clamp to [0, 1]
  const t = Math.max(0, Math.min(1, scaled));

  // High quality 5-point Viridis approximation
  // 0.0: #440154 (68, 1, 84)
  // 0.25: #3b528b (59, 82, 139)
  // 0.5: #21918c (33, 145, 140)
  // 0.75: #5ec962 (94, 201, 98)
  // 1.0: #fde725 (253, 231, 37)
  if (t < 0.25) {
    const s = t / 0.25;
    return [
      Math.round(68 + s * (59 - 68)),
      Math.round(1 + s * (82 - 1)),
      Math.round(84 + s * (139 - 84))
    ];
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return [
      Math.round(59 + s * (33 - 59)),
      Math.round(82 + s * (145 - 82)),
      Math.round(139 + s * (140 - 139))
    ];
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return [
      Math.round(33 + s * (94 - 33)),
      Math.round(145 + s * (201 - 145)),
      Math.round(140 + s * (98 - 140))
    ];
  } else {
    const s = (t - 0.75) / 0.25;
    return [
      Math.round(94 + s * (253 - 94)),
      Math.round(201 + s * (231 - 201)),
      Math.round(98 + s * (37 - 98))
    ];
  }
}

export function valToRgbaString(val: number, alpha: number = 1.0): string {
  const [r, g, b] = getViridisColor(val);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ==============================================================================
// RWKV-7 Per-Layer Calibrated Empirical Color Scales
// Derived from 47,616 real measurement samples across 4 diverse benchmark prompts
// (P99 + 15~20% headroom, fixed per layer to prevent temporal drift)
// ==============================================================================

export interface LayerScaleConfig {
  summaryMax: number;
  deltaBound: number;
}

export const RWKV_LAYER_SCALES: Record<number, LayerScaleConfig> = {
  0:  { summaryMax: 4.7,  deltaBound: 0.10 },
  1:  { summaryMax: 3.6,  deltaBound: 0.51 },
  2:  { summaryMax: 4.5,  deltaBound: 0.42 },
  3:  { summaryMax: 8.9,  deltaBound: 0.55 },
  4:  { summaryMax: 7.9,  deltaBound: 0.35 },
  5:  { summaryMax: 8.6,  deltaBound: 0.72 },
  6:  { summaryMax: 11.9, deltaBound: 0.59 },
  7:  { summaryMax: 11.2, deltaBound: 0.49 },
  8:  { summaryMax: 13.0, deltaBound: 3.17 },
  9:  { summaryMax: 12.1, deltaBound: 0.44 },
  10: { summaryMax: 8.7,  deltaBound: 0.40 },
  11: { summaryMax: 13.4, deltaBound: 0.59 },
  12: { summaryMax: 38.1, deltaBound: 5.13 },
  13: { summaryMax: 11.5, deltaBound: 1.09 },
  14: { summaryMax: 19.3, deltaBound: 1.91 },
  15: { summaryMax: 24.8, deltaBound: 2.13 },
  16: { summaryMax: 19.6, deltaBound: 0.90 },
  17: { summaryMax: 19.9, deltaBound: 1.11 },
  18: { summaryMax: 18.1, deltaBound: 4.92 },
  19: { summaryMax: 16.6, deltaBound: 5.68 },
  20: { summaryMax: 26.9, deltaBound: 2.53 },
  21: { summaryMax: 23.2, deltaBound: 3.90 },
  22: { summaryMax: 30.8, deltaBound: 3.68 },
  23: { summaryMax: 51.3, deltaBound: 7.62 },
};

export function getRwkvLayerScale(layer: number): LayerScaleConfig {
  return RWKV_LAYER_SCALES[layer] ?? { summaryMax: 15.0, deltaBound: 1.0 };
}

// Diverging Colormap (Coolwarm): -1.0 Blue -> 0.0 Neutral Grey/White -> +1.0 Red
export function getCoolwarmColor(delta: number, deltaBound: number): [number, number, number] {
  const norm = deltaBound > 0 ? Math.max(-1.0, Math.min(1.0, delta / deltaBound)) : 0;

  // Blue: [59, 76, 192] (#3b4cc0)
  // Neutral: [240, 240, 245] (#f0f0f5)
  // Red: [180, 4, 38] (#b40426)
  if (norm < 0) {
    const s = norm + 1.0; // 0 (deep blue) -> 1 (neutral)
    return [
      Math.round(59 + s * (240 - 59)),
      Math.round(76 + s * (240 - 76)),
      Math.round(192 + s * (245 - 192))
    ];
  } else {
    const s = norm; // 0 (neutral) -> 1 (deep red)
    return [
      Math.round(240 + s * (180 - 240)),
      Math.round(240 + s * (4 - 240)),
      Math.round(245 + s * (38 - 245))
    ];
  }
}

export function deltaToRgbaString(delta: number, deltaBound: number, alpha: number = 1.0): string {
  const [r, g, b] = getCoolwarmColor(delta, deltaBound);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Linear Viridis mapping for state_summary (no log-scale)
export function summaryToRgbaString(val: number, summaryMax: number, alpha: number = 1.0): string {
  const norm = summaryMax > 0 ? Math.max(0.0, Math.min(1.0, val / summaryMax)) : 0;
  const [r, g, b] = getViridisColor(norm, false); // false = linear, no log-scale
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
