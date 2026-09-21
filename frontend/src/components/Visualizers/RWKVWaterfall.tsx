import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  getRwkvLayerScale,
  summaryToRgbaString,
  deltaToRgbaString,
} from '../../utils/colormaps';
import { RefreshCw, ArrowDown, Activity, Columns, Rows } from 'lucide-react';

export interface RWKVIncrementalRowPayload {
  step: number;
  token: string;
  state_summary: number[]; // 16 heads
  delta: number[];         // 16 heads
}

interface RWKVWaterfallProps {
  layer: number;
  head: number;
  tokens: string[];
  initialSummaries?: number[][]; // [N, 16]
  initialDeltas?: number[][];    // [N, 16]
  incrementalRow?: RWKVIncrementalRowPayload | null;
  isStreaming?: boolean;
  isLoading?: boolean;
  loadingMessage?: string;
  onStepClick?: (step: number) => void;
  inspectedStep?: number | null;
}

interface HoverInfo {
  type: 'summary' | 'delta';
  step: number;
  head: number;
  val: number;
  token: string;
  bound: number;
  x: number;
  y: number;
}

export const RWKVWaterfall: React.FC<RWKVWaterfallProps> = ({
  layer,
  head,
  tokens,
  initialSummaries,
  initialDeltas,
  incrementalRow,
  isStreaming = false,
  isLoading = false,
  loadingMessage,
  onStepClick,
  inspectedStep = null,
}) => {
  const visibleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenSummaryRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenDeltaRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Local state for accumulated history
  const localSummariesRef = useRef<number[][]>([]);
  const localDeltasRef = useRef<number[][]>([]);
  const localTokensRef = useRef<string[]>([]);
  const lastSummariesRef = useRef<number[][] | null | undefined>(null);
  const lastLayerRef = useRef<number>(-1);

  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [layoutMode, setLayoutMode] = useState<'side-by-side' | 'stacked'>('side-by-side');
  const [currentStepCount, setCurrentStepCount] = useState(0);

  // Per-layer calibrated color scale bounds
  const scaleConfig = getRwkvLayerScale(layer);
  const { summaryMax, deltaBound } = scaleConfig;

  // Layout geometry constants (Grid is 16 heads * 20px = 320px)
  const NUM_HEADS = 16;
  const CELL_WIDTH = 20;
  const ROW_HEIGHT = 20;
  const GRID_WIDTH = NUM_HEADS * CELL_WIDTH; // 320px
  const LABEL_LEFT_WIDTH = 82;
  const CB_WIDTH = 12;

  // Side-by-side coordinate offsets
  const S_ZONE_A_X = LABEL_LEFT_WIDTH; // 82
  const S_CB_A_X = S_ZONE_A_X + GRID_WIDTH + 8; // 410
  const S_DIVIDER_X = S_CB_A_X + CB_WIDTH + 26; // 448
  const S_ZONE_B_X = S_DIVIDER_X + 20; // 468
  const S_CB_B_X = S_ZONE_B_X + GRID_WIDTH + 8; // 796
  const S_CANVAS_WIDTH = S_CB_B_X + CB_WIDTH + 52; // 860px
  const S_HEADER_Y = 56;

  // Stacked coordinate offsets
  const T_ZONE_X = LABEL_LEFT_WIDTH;
  const T_CB_X = T_ZONE_X + GRID_WIDTH + 14;
  const T_CANVAS_WIDTH = T_CB_X + CB_WIDTH + 56;
  const T_HEADER_A_Y = 52;
  const T_SECTION_GAP = 54;

  // Initialize offscreen canvases with ample headroom (200 rows = 4000px)
  if (!offscreenSummaryRef.current && typeof document !== 'undefined') {
    const c = document.createElement('canvas');
    c.width = GRID_WIDTH;
    c.height = 4000;
    offscreenSummaryRef.current = c;
  }
  if (!offscreenDeltaRef.current && typeof document !== 'undefined') {
    const c = document.createElement('canvas');
    c.width = GRID_WIDTH;
    c.height = 4000;
    offscreenDeltaRef.current = c;
  }

  // Draw a single row into an offscreen canvas
  const drawRowToOffscreen = useCallback(
    (
      canvas: HTMLCanvasElement,
      rowIndex: number,
      values: number[],
      colorFn: (v: number) => string
    ) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const y = rowIndex * ROW_HEIGHT;
      for (let h = 0; h < NUM_HEADS; h++) {
        const val = values[h] ?? 0;
        const x = h * CELL_WIDTH;
        ctx.fillStyle = colorFn(val);
        ctx.fillRect(x, y, CELL_WIDTH, ROW_HEIGHT);

        // Subtle cell grid lines
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.45)';
        ctx.lineWidth = 0.8;
        ctx.strokeRect(x, y, CELL_WIDTH, ROW_HEIGHT);
      }
    },
    []
  );

  // Full redraw of offscreen canvases (strictly called on layer switch or explicit data update)
  const fullRedrawOffscreen = useCallback(
    (summaries: number[][], deltas: number[][]) => {
      const offSummary = offscreenSummaryRef.current;
      const offDelta = offscreenDeltaRef.current;
      if (!offSummary || !offDelta) return;

      const totalSteps = summaries.length;
      const neededHeight = Math.max(totalSteps * ROW_HEIGHT + 1000, 4000);

      if (offSummary.width !== GRID_WIDTH || offSummary.height < neededHeight) {
        offSummary.width = GRID_WIDTH;
        offSummary.height = neededHeight;
      }
      if (offDelta.width !== GRID_WIDTH || offDelta.height < neededHeight) {
        offDelta.width = GRID_WIDTH;
        offDelta.height = neededHeight;
      }

      const ctxS = offSummary.getContext('2d');
      const ctxD = offDelta.getContext('2d');
      if (!ctxS || !ctxD) return;

      ctxS.clearRect(0, 0, GRID_WIDTH, offSummary.height);
      ctxD.clearRect(0, 0, GRID_WIDTH, offDelta.height);

      for (let s = 0; s < totalSteps; s++) {
        drawRowToOffscreen(offSummary, s, summaries[s] || [], (v) =>
          summaryToRgbaString(v, summaryMax)
        );
        drawRowToOffscreen(offDelta, s, deltas[s] || [], (v) =>
          deltaToRgbaString(v, deltaBound)
        );
      }
    },
    [summaryMax, deltaBound, drawRowToOffscreen]
  );

  // Composite offscreen buffers onto visible canvas
  const renderVisibleCanvas = useCallback(
    (stepCount: number, currentTokens: string[], hover: HoverInfo | null) => {
      const visibleCanvas = visibleCanvasRef.current;
      const offSummary = offscreenSummaryRef.current;
      const offDelta = offscreenDeltaRef.current;
      if (!visibleCanvas || !offSummary || !offDelta) return;

      const ctx = visibleCanvas.getContext('2d');
      if (!ctx) return;

      const N = Math.max(stepCount, 1);
      const zoneHeight = N * ROW_HEIGHT;
      const dpr = window.devicePixelRatio || 1;

      // ==========================================
      // Layout Branch A: Side-by-Side (Default)
      // ==========================================
      if (layoutMode === 'side-by-side') {
        const canvasWidth = S_CANVAS_WIDTH;
        const canvasHeight = S_HEADER_Y + zoneHeight + 35;

        visibleCanvas.width = canvasWidth * dpr;
        visibleCanvas.height = canvasHeight * dpr;
        visibleCanvas.style.width = `${canvasWidth}px`;
        visibleCanvas.style.height = `${canvasHeight}px`;

        ctx.scale(dpr, dpr);

        // Slate-950 background
        ctx.fillStyle = '#020617';
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);

        // 1. Zone A Title & Subtitle
        ctx.fillStyle = '#e2e8f0';
        ctx.font = 'bold 12px ui-sans-serif, system-ui';
        ctx.fillText(`色带 A: 隐状态绝对强度 S_t`, S_ZONE_A_X, 20);

        ctx.fillStyle = '#64748b';
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillText(`Viridis [0.0 ~ ${summaryMax.toFixed(1)}] · 记忆充盈度`, S_ZONE_A_X, 35);

        // Heads H0..H15 headers for Zone A
        for (let h = 0; h < NUM_HEADS; h++) {
          const x = S_ZONE_A_X + h * CELL_WIDTH;
          ctx.fillStyle = h === head ? '#38bdf8' : '#64748b';
          ctx.fillText(`H${h}`, x + 3, S_HEADER_Y - 5);
        }

        // Blit Zone A from offscreen buffer
        if (offSummary.width > 0 && offSummary.height > 0) {
          ctx.drawImage(offSummary, 0, 0, GRID_WIDTH, zoneHeight, S_ZONE_A_X, S_HEADER_Y, GRID_WIDTH, zoneHeight);
        }

        // Colorbar A (Viridis)
        const cbHeightA = Math.min(zoneHeight, 130);
        const gradA = ctx.createLinearGradient(0, S_HEADER_Y + cbHeightA, 0, S_HEADER_Y);
        gradA.addColorStop(0.0, 'rgb(68, 1, 84)');
        gradA.addColorStop(0.25, 'rgb(59, 82, 139)');
        gradA.addColorStop(0.5, 'rgb(33, 145, 140)');
        gradA.addColorStop(0.75, 'rgb(94, 201, 98)');
        gradA.addColorStop(1.0, 'rgb(253, 231, 37)');

        ctx.fillStyle = gradA;
        ctx.fillRect(S_CB_A_X, S_HEADER_Y, CB_WIDTH, cbHeightA);
        ctx.strokeStyle = '#334155';
        ctx.strokeRect(S_CB_A_X, S_HEADER_Y, CB_WIDTH, cbHeightA);

        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px ui-monospace, monospace';
        ctx.fillText(`${summaryMax.toFixed(1)}`, S_CB_A_X + CB_WIDTH + 4, S_HEADER_Y + 8);
        ctx.fillText('0.0', S_CB_A_X + CB_WIDTH + 4, S_HEADER_Y + cbHeightA);

        // Subtle vertical divider between Zone A and Zone B
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(S_DIVIDER_X, 10);
        ctx.lineTo(S_DIVIDER_X, canvasHeight - 10);
        ctx.stroke();

        // 2. Zone B Title & Subtitle
        ctx.fillStyle = '#e2e8f0';
        ctx.font = 'bold 12px ui-sans-serif, system-ui';
        ctx.fillText(`色带 B: 单步状态变化量 Δ_t`, S_ZONE_B_X, 20);

        ctx.fillStyle = '#64748b';
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillText(`Coolwarm [±${deltaBound.toFixed(2)}] · 红注能 / 蓝衰减`, S_ZONE_B_X, 35);

        // Heads H0..H15 headers for Zone B
        for (let h = 0; h < NUM_HEADS; h++) {
          const x = S_ZONE_B_X + h * CELL_WIDTH;
          ctx.fillStyle = h === head ? '#38bdf8' : '#64748b';
          ctx.fillText(`H${h}`, x + 3, S_HEADER_Y - 5);
        }

        // Blit Zone B from offscreen buffer
        if (offDelta.width > 0 && offDelta.height > 0) {
          ctx.drawImage(offDelta, 0, 0, GRID_WIDTH, zoneHeight, S_ZONE_B_X, S_HEADER_Y, GRID_WIDTH, zoneHeight);
        }

        // Colorbar B (Coolwarm)
        const cbHeightB = Math.min(zoneHeight, 130);
        const gradB = ctx.createLinearGradient(0, S_HEADER_Y + cbHeightB, 0, S_HEADER_Y);
        gradB.addColorStop(0.0, 'rgb(59, 76, 192)');    // Blue (-deltaBound)
        gradB.addColorStop(0.5, 'rgb(240, 240, 245)');  // Neutral (0)
        gradB.addColorStop(1.0, 'rgb(180, 4, 38)');     // Red (+deltaBound)

        ctx.fillStyle = gradB;
        ctx.fillRect(S_CB_B_X, S_HEADER_Y, CB_WIDTH, cbHeightB);
        ctx.strokeStyle = '#334155';
        ctx.strokeRect(S_CB_B_X, S_HEADER_Y, CB_WIDTH, cbHeightB);

        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px ui-monospace, monospace';
        ctx.fillText(`+${deltaBound.toFixed(2)}`, S_CB_B_X + CB_WIDTH + 4, S_HEADER_Y + 8);
        ctx.fillText('0.00', S_CB_B_X + CB_WIDTH + 4, S_HEADER_Y + cbHeightB / 2 + 3);
        ctx.fillText(`-${deltaBound.toFixed(2)}`, S_CB_B_X + CB_WIDTH + 4, S_HEADER_Y + cbHeightB);

        // 3. Row Labels (Left gutter, sync across both zones)
        ctx.font = '10px ui-monospace, monospace';
        for (let s = 0; s < stepCount; s++) {
          const tok = currentTokens[s] ?? '';
          const y = S_HEADER_Y + s * ROW_HEIGHT + 14;

          const isHovered = hover?.step === s;
          const isInspected = inspectedStep === s;
          ctx.fillStyle = isInspected ? '#f59e0b' : isHovered ? '#38bdf8' : '#94a3b8';
          const label = s === 0 ? 'Prefill' : `Step ${s}`;
          ctx.fillText(`${label} ${tok.slice(0, 3)}`, 6, y);

          // Highlight row outline on both Zone A and Zone B
          if (isInspected || isHovered) {
            ctx.strokeStyle = isInspected ? 'rgba(245, 158, 11, 0.8)' : 'rgba(56, 189, 248, 0.4)';
            ctx.lineWidth = 1;
            ctx.strokeRect(S_ZONE_A_X, S_HEADER_Y + s * ROW_HEIGHT, GRID_WIDTH, ROW_HEIGHT);
            ctx.strokeRect(S_ZONE_B_X, S_HEADER_Y + s * ROW_HEIGHT, GRID_WIDTH, ROW_HEIGHT);
          }
        }

        // 4. Highlight specific hovered cell
        if (hover) {
          const { step, head: h, type } = hover;
          const targetX = type === 'summary' ? S_ZONE_A_X + h * CELL_WIDTH : S_ZONE_B_X + h * CELL_WIDTH;
          const targetY = S_HEADER_Y + step * ROW_HEIGHT;
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 2;
          ctx.strokeRect(targetX, targetY, CELL_WIDTH, ROW_HEIGHT);
        }
      } else {
        // ==========================================
        // Layout Branch B: Stacked (Vertical)
        // ==========================================
        const headerBY = T_HEADER_A_Y + zoneHeight + T_SECTION_GAP;
        const canvasWidth = T_CANVAS_WIDTH;
        const canvasHeight = headerBY + zoneHeight + 40;

        visibleCanvas.width = canvasWidth * dpr;
        visibleCanvas.height = canvasHeight * dpr;
        visibleCanvas.style.width = `${canvasWidth}px`;
        visibleCanvas.style.height = `${canvasHeight}px`;

        ctx.scale(dpr, dpr);

        ctx.fillStyle = '#020617';
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);

        // Zone A Title
        ctx.fillStyle = '#e2e8f0';
        ctx.font = 'bold 12px ui-sans-serif, system-ui';
        ctx.fillText(`色带 A: 隐状态绝对强度 S_t (Layer ${layer})`, T_ZONE_X, 20);

        ctx.fillStyle = '#64748b';
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillText(`Viridis [0.0 ~ ${summaryMax.toFixed(1)}] · 16 Heads 记忆充盈度`, T_ZONE_X, 35);

        for (let h = 0; h < NUM_HEADS; h++) {
          const x = T_ZONE_X + h * CELL_WIDTH;
          ctx.fillStyle = h === head ? '#38bdf8' : '#64748b';
          ctx.fillText(`H${h}`, x + 3, T_HEADER_A_Y - 5);
        }

        if (offSummary.width > 0 && offSummary.height > 0) {
          ctx.drawImage(offSummary, 0, 0, GRID_WIDTH, zoneHeight, T_ZONE_X, T_HEADER_A_Y, GRID_WIDTH, zoneHeight);
        }

        // Colorbar A
        const cbHeightA = Math.min(zoneHeight, 140);
        const gradA = ctx.createLinearGradient(0, T_HEADER_A_Y + cbHeightA, 0, T_HEADER_A_Y);
        gradA.addColorStop(0.0, 'rgb(68, 1, 84)');
        gradA.addColorStop(0.25, 'rgb(59, 82, 139)');
        gradA.addColorStop(0.5, 'rgb(33, 145, 140)');
        gradA.addColorStop(0.75, 'rgb(94, 201, 98)');
        gradA.addColorStop(1.0, 'rgb(253, 231, 37)');

        ctx.fillStyle = gradA;
        ctx.fillRect(T_CB_X, T_HEADER_A_Y, CB_WIDTH, cbHeightA);
        ctx.strokeStyle = '#334155';
        ctx.strokeRect(T_CB_X, T_HEADER_A_Y, CB_WIDTH, cbHeightA);

        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px ui-monospace, monospace';
        ctx.fillText(`${summaryMax.toFixed(1)}`, T_CB_X + CB_WIDTH + 4, T_HEADER_A_Y + 8);
        ctx.fillText('0.0', T_CB_X + CB_WIDTH + 4, T_HEADER_A_Y + cbHeightA);

        // Zone B Title
        ctx.fillStyle = '#e2e8f0';
        ctx.font = 'bold 12px ui-sans-serif, system-ui';
        ctx.fillText(`色带 B: 单步状态变化量 Δ_t (Layer ${layer})`, T_ZONE_X, headerBY - 28);

        ctx.fillStyle = '#64748b';
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillText(`Coolwarm [±${deltaBound.toFixed(2)}] · 红注能 / 蓝衰减`, T_ZONE_X, headerBY - 14);

        for (let h = 0; h < NUM_HEADS; h++) {
          const x = T_ZONE_X + h * CELL_WIDTH;
          ctx.fillStyle = h === head ? '#38bdf8' : '#64748b';
          ctx.fillText(`H${h}`, x + 3, headerBY - 4);
        }

        if (offDelta.width > 0 && offDelta.height > 0) {
          ctx.drawImage(offDelta, 0, 0, GRID_WIDTH, zoneHeight, T_ZONE_X, headerBY, GRID_WIDTH, zoneHeight);
        }

        // Colorbar B
        const cbHeightB = Math.min(zoneHeight, 140);
        const gradB = ctx.createLinearGradient(0, headerBY + cbHeightB, 0, headerBY);
        gradB.addColorStop(0.0, 'rgb(59, 76, 192)');
        gradB.addColorStop(0.5, 'rgb(240, 240, 245)');
        gradB.addColorStop(1.0, 'rgb(180, 4, 38)');

        ctx.fillStyle = gradB;
        ctx.fillRect(T_CB_X, headerBY, CB_WIDTH, cbHeightB);
        ctx.strokeStyle = '#334155';
        ctx.strokeRect(T_CB_X, headerBY, CB_WIDTH, cbHeightB);

        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px ui-monospace, monospace';
        ctx.fillText(`+${deltaBound.toFixed(2)}`, T_CB_X + CB_WIDTH + 4, headerBY + 8);
        ctx.fillText('0.00', T_CB_X + CB_WIDTH + 4, headerBY + cbHeightB / 2 + 3);
        ctx.fillText(`-${deltaBound.toFixed(2)}`, T_CB_X + CB_WIDTH + 4, headerBY + cbHeightB);

        // Labels
        ctx.font = '10px ui-monospace, monospace';
        for (let s = 0; s < stepCount; s++) {
          const tok = currentTokens[s] ?? '';
          const yA = T_HEADER_A_Y + s * ROW_HEIGHT + 14;
          const yB = headerBY + s * ROW_HEIGHT + 14;
          const label = s === 0 ? 'Prefill' : `Step ${s}`;

          const isHovered = hover?.step === s;
          const isInspected = inspectedStep === s;
          ctx.fillStyle = isInspected ? '#f59e0b' : isHovered ? '#38bdf8' : '#94a3b8';
          ctx.fillText(`${label} ${tok.slice(0, 3)}`, 6, yA);
          ctx.fillText(`${label} ${tok.slice(0, 3)}`, 6, yB);

          if (isInspected || isHovered) {
            ctx.strokeStyle = isInspected ? 'rgba(245, 158, 11, 0.8)' : 'rgba(56, 189, 248, 0.4)';
            ctx.lineWidth = 1;
            ctx.strokeRect(T_ZONE_X, T_HEADER_A_Y + s * ROW_HEIGHT, GRID_WIDTH, ROW_HEIGHT);
            ctx.strokeRect(T_ZONE_X, headerBY + s * ROW_HEIGHT, GRID_WIDTH, ROW_HEIGHT);
          }
        }
      }
    },
    [layoutMode, head, summaryMax, deltaBound, inspectedStep, layer]
  );

  // Full redraw: strictly guards against full redraw during streaming!
  useEffect(() => {
    const summariesChanged = initialSummaries && initialSummaries !== lastSummariesRef.current;
    const layerChanged = layer !== lastLayerRef.current;

    if (summariesChanged || layerChanged) {
      lastSummariesRef.current = initialSummaries;
      lastLayerRef.current = layer;

      const initialS = (initialSummaries || []).map((row) => [...row]);
      const initialD = (initialDeltas || []).map((row) => [...row]);
      localSummariesRef.current = initialS;
      localDeltasRef.current = initialD;
      localTokensRef.current = [...tokens];
      setCurrentStepCount(initialS.length);
      setHoverInfo(null);

      fullRedrawOffscreen(initialS, initialD);
      renderVisibleCanvas(initialS.length, localTokensRef.current, null);
    }
  }, [layer, initialSummaries, initialDeltas, fullRedrawOffscreen, renderVisibleCanvas]);

  // Handle incremental row streaming from WebSocket rAF pipeline
  useEffect(() => {
    if (!incrementalRow) return;

    const { step, token, state_summary, delta } = incrementalRow;
    const offSummary = offscreenSummaryRef.current;
    const offDelta = offscreenDeltaRef.current;
    if (!offSummary || !offDelta) return;

    // Check buffer height and expand if needed (preserving existing pixels)
    const requiredHeight = (step + 1) * ROW_HEIGHT;
    if (offSummary.height < requiredHeight || offDelta.height < requiredHeight) {
      const newHeight = Math.max(requiredHeight + 1000, offSummary.height * 2);

      const tempSummary = document.createElement('canvas');
      tempSummary.width = GRID_WIDTH;
      tempSummary.height = offSummary.height;
      tempSummary.getContext('2d')?.drawImage(offSummary, 0, 0);

      const tempDelta = document.createElement('canvas');
      tempDelta.width = GRID_WIDTH;
      tempDelta.height = offDelta.height;
      tempDelta.getContext('2d')?.drawImage(offDelta, 0, 0);

      offSummary.height = newHeight;
      offSummary.getContext('2d')?.drawImage(tempSummary, 0, 0);

      offDelta.height = newHeight;
      offDelta.getContext('2d')?.drawImage(tempDelta, 0, 0);
    }

    // ONLY draw the new row into offscreen buffers
    drawRowToOffscreen(offSummary, step, state_summary, (v) =>
      summaryToRgbaString(v, summaryMax)
    );
    drawRowToOffscreen(offDelta, step, delta, (v) =>
      deltaToRgbaString(v, deltaBound)
    );

    // Update local history
    localSummariesRef.current[step] = state_summary;
    localDeltasRef.current[step] = delta;
    if (localTokensRef.current.length <= step) {
      localTokensRef.current.push(token);
    } else {
      localTokensRef.current[step] = token;
    }

    const nextCount = step + 1;
    setCurrentStepCount(nextCount);
    renderVisibleCanvas(nextCount, localTokensRef.current, hoverInfo);

    // Auto-scroll
    if (autoScroll && containerRef.current && step > 6) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [
    incrementalRow,
    summaryMax,
    deltaBound,
    autoScroll,
    hoverInfo,
    drawRowToOffscreen,
    renderVisibleCanvas,
  ]);

  // Mouse interaction: Hover tooltip
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = visibleCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const stepCount = currentStepCount;
    if (stepCount === 0) return;

    const zoneHeight = stepCount * ROW_HEIGHT;

    if (layoutMode === 'side-by-side') {
      const inY = y >= S_HEADER_Y && y < S_HEADER_Y + zoneHeight;
      if (!inY) {
        if (hoverInfo !== null) {
          setHoverInfo(null);
          renderVisibleCanvas(stepCount, localTokensRef.current, null);
        }
        return;
      }

      const s = Math.floor((y - S_HEADER_Y) / ROW_HEIGHT);

      // Check Zone A
      if (x >= S_ZONE_A_X && x < S_ZONE_A_X + GRID_WIDTH) {
        const h = Math.floor((x - S_ZONE_A_X) / CELL_WIDTH);
        const val = localSummariesRef.current[s]?.[h] ?? 0;
        const tok = localTokensRef.current[s] ?? '';
        const newHover: HoverInfo = {
          type: 'summary',
          step: s,
          head: h,
          val,
          token: tok,
          bound: summaryMax,
          x: e.clientX,
          y: e.clientY,
        };
        setHoverInfo(newHover);
        renderVisibleCanvas(stepCount, localTokensRef.current, newHover);
        return;
      }

      // Check Zone B
      if (x >= S_ZONE_B_X && x < S_ZONE_B_X + GRID_WIDTH) {
        const h = Math.floor((x - S_ZONE_B_X) / CELL_WIDTH);
        const val = localDeltasRef.current[s]?.[h] ?? 0;
        const tok = localTokensRef.current[s] ?? '';
        const newHover: HoverInfo = {
          type: 'delta',
          step: s,
          head: h,
          val,
          token: tok,
          bound: deltaBound,
          x: e.clientX,
          y: e.clientY,
        };
        setHoverInfo(newHover);
        renderVisibleCanvas(stepCount, localTokensRef.current, newHover);
        return;
      }
    } else {
      const headerBY = T_HEADER_A_Y + zoneHeight + T_SECTION_GAP;
      const inGridX = x >= T_ZONE_X && x < T_ZONE_X + GRID_WIDTH;

      // Check Zone A
      if (inGridX && y >= T_HEADER_A_Y && y < T_HEADER_A_Y + zoneHeight) {
        const s = Math.floor((y - T_HEADER_A_Y) / ROW_HEIGHT);
        const h = Math.floor((x - T_ZONE_X) / CELL_WIDTH);
        const val = localSummariesRef.current[s]?.[h] ?? 0;
        const tok = localTokensRef.current[s] ?? '';
        const newHover: HoverInfo = {
          type: 'summary',
          step: s,
          head: h,
          val,
          token: tok,
          bound: summaryMax,
          x: e.clientX,
          y: e.clientY,
        };
        setHoverInfo(newHover);
        renderVisibleCanvas(stepCount, localTokensRef.current, newHover);
        return;
      }

      // Check Zone B
      if (inGridX && y >= headerBY && y < headerBY + zoneHeight) {
        const s = Math.floor((y - headerBY) / ROW_HEIGHT);
        const h = Math.floor((x - T_ZONE_X) / CELL_WIDTH);
        const val = localDeltasRef.current[s]?.[h] ?? 0;
        const tok = localTokensRef.current[s] ?? '';
        const newHover: HoverInfo = {
          type: 'delta',
          step: s,
          head: h,
          val,
          token: tok,
          bound: deltaBound,
          x: e.clientX,
          y: e.clientY,
        };
        setHoverInfo(newHover);
        renderVisibleCanvas(stepCount, localTokensRef.current, newHover);
        return;
      }
    }

    if (hoverInfo !== null) {
      setHoverInfo(null);
      renderVisibleCanvas(stepCount, localTokensRef.current, null);
    }
  };

  const handleMouseLeave = () => {
    if (hoverInfo !== null) {
      setHoverInfo(null);
      renderVisibleCanvas(currentStepCount, localTokensRef.current, null);
    }
  };

  // Step click inspection
  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = visibleCanvasRef.current;
    if (!canvas || !onStepClick) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const stepCount = currentStepCount;
    if (stepCount === 0) return;

    const zoneHeight = stepCount * ROW_HEIGHT;

    if (layoutMode === 'side-by-side') {
      const inY = y >= S_HEADER_Y && y < S_HEADER_Y + zoneHeight;
      const inX =
        (x >= S_ZONE_A_X && x < S_ZONE_A_X + GRID_WIDTH) ||
        (x >= S_ZONE_B_X && x < S_ZONE_B_X + GRID_WIDTH);

      if (inY && inX) {
        const s = Math.floor((y - S_HEADER_Y) / ROW_HEIGHT);
        onStepClick(s);
      }
    } else {
      const headerBY = T_HEADER_A_Y + zoneHeight + T_SECTION_GAP;
      const inGridX = x >= T_ZONE_X && x < T_ZONE_X + GRID_WIDTH;

      if (inGridX) {
        if (y >= T_HEADER_A_Y && y < T_HEADER_A_Y + zoneHeight) {
          const s = Math.floor((y - T_HEADER_A_Y) / ROW_HEIGHT);
          onStepClick(s);
        } else if (y >= headerBY && y < headerBY + zoneHeight) {
          const s = Math.floor((y - headerBY) / ROW_HEIGHT);
          onStepClick(s);
        }
      }
    }
  };

  return (
    <div className="relative flex flex-col h-full bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
      {/* Top Header Bar */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-900/90 border-b border-slate-800/80 backdrop-blur z-10 shrink-0">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-emerald-400" />
          <span className="text-sm font-semibold text-slate-100">
            RWKV-7 状态演化双色带瀑布图
          </span>
          <span className="text-xs px-2 py-0.5 rounded font-mono bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
            Layer {layer} (固定标定)
          </span>
        </div>

        <div className="flex items-center gap-2 text-xs">
          {isStreaming && (
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono text-[11px] border border-emerald-500/30 animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span>逐行增量绘制中</span>
            </div>
          )}

          <div className="flex items-center gap-1.5 font-mono text-slate-400 bg-slate-800/80 px-2 py-1 rounded border border-slate-700">
            <span>步数:</span>
            <span className="text-emerald-300 font-bold">{currentStepCount}</span>
          </div>

          {/* Layout Mode Switcher */}
          <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800">
            <button
              type="button"
              onClick={() => {
                setLayoutMode('side-by-side');
                renderVisibleCanvas(currentStepCount, localTokensRef.current, null);
              }}
              className={`flex items-center gap-1 px-2 py-1 rounded text-[11px] transition-all cursor-pointer ${
                layoutMode === 'side-by-side'
                  ? 'bg-emerald-600 text-white font-medium shadow-xs'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="左右分栏同屏对比（推荐）：色带 A 与色带 B 并行下推"
            >
              <Columns className="w-3 h-3" />
              <span>左右对照</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setLayoutMode('stacked');
                renderVisibleCanvas(currentStepCount, localTokensRef.current, null);
              }}
              className={`flex items-center gap-1 px-2 py-1 rounded text-[11px] transition-all cursor-pointer ${
                layoutMode === 'stacked'
                  ? 'bg-emerald-600 text-white font-medium shadow-xs'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              title="上下堆叠模式"
            >
              <Rows className="w-3 h-3" />
              <span>上下堆叠</span>
            </button>
          </div>

          <button
            type="button"
            onClick={() => setAutoScroll((prev) => !prev)}
            className={`flex items-center gap-1 px-2 py-1 rounded transition-colors border cursor-pointer ${
              autoScroll
                ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30 font-medium'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
          >
            <ArrowDown className="w-3.5 h-3.5" />
            <span>跟踪</span>
          </button>
        </div>
      </div>

      {/* Main Canvas Scroll Area */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto p-4 custom-scrollbar relative"
        style={{ minHeight: '480px' }}
      >
        <canvas
          ref={visibleCanvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          className="cursor-pointer block shadow-lg rounded"
        />

        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-xs flex flex-col items-center justify-center gap-3 z-20">
            <RefreshCw className="w-7 h-7 text-emerald-400 animate-spin" />
            <span className="text-sm text-slate-200 font-medium tracking-wide">
              {loadingMessage || 'RWKV 状态数据加载中...'}
            </span>
          </div>
        )}
      </div>

      {/* Floating Hover Tooltip */}
      {hoverInfo && (
        <div
          className="fixed pointer-events-none z-50 bg-slate-950/95 border border-slate-700/80 rounded-lg p-2.5 shadow-2xl text-xs backdrop-blur font-mono space-y-1 transform -translate-x-1/2 -translate-y-full -mt-2"
          style={{ left: hoverInfo.x, top: hoverInfo.y }}
        >
          <div className="flex items-center justify-between gap-4 border-b border-slate-800 pb-1">
            <span className="text-slate-400 font-bold">
              {hoverInfo.step === 0 ? 'Prefill 初始' : `Step ${hoverInfo.step}`}
            </span>
            <span className="text-emerald-400 font-sans font-medium">
              &quot;{hoverInfo.token}&quot;
            </span>
          </div>

          <div className="flex items-center justify-between gap-4">
            <span className="text-slate-400">Head:</span>
            <span className="text-slate-200 font-bold">H{hoverInfo.head}</span>
          </div>

          {hoverInfo.type === 'summary' ? (
            <div className="flex items-center justify-between gap-4">
              <span className="text-slate-400">绝对强度 S_t:</span>
              <span className="text-yellow-400 font-bold">
                {hoverInfo.val.toFixed(4)}
              </span>
            </div>
          ) : (
            <div className="flex items-center justify-between gap-4">
              <span className="text-slate-400">单步变化量 Δ_t:</span>
              <span
                className={`font-bold ${
                  hoverInfo.val >= 0 ? 'text-rose-400' : 'text-blue-400'
                }`}
              >
                {hoverInfo.val >= 0 ? `+${hoverInfo.val.toFixed(4)}` : hoverInfo.val.toFixed(4)}
              </span>
            </div>
          )}

          <div className="flex items-center justify-between gap-4 text-[10px] text-slate-500 border-t border-slate-800/80 pt-1">
            <span>当前量程:</span>
            <span>
              {hoverInfo.type === 'summary'
                ? `[0.0, ${hoverInfo.bound.toFixed(1)}]`
                : `[±${hoverInfo.bound.toFixed(2)}]`}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
