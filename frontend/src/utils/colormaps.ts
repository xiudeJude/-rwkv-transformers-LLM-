// Scientific colormaps for attention matrices: Viridis and Plasma
export function getViridisColor(val: number): [number, number, number] {
  // Clamp to [0, 1]
  const t = Math.max(0, Math.min(1, val));
  
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
