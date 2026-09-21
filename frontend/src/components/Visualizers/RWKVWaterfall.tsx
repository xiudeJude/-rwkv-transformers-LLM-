import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  getRwkvLayerScale,
  summaryToRgbaString,
  deltaToRgbaString,
} from '../../utils/colormaps';
import { RefreshCw, ArrowDown, Activity } from 'lucide-react';

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
  const lastInitialRef = useRef<string>('');

  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [currentStepCount, setCurrentStepCount] = useState(0);

  // Per-layer calibrated color scale bounds
  const scaleConfig = getRwkvLayerScale(layer);
  const { summaryMax, deltaBound } = scaleConfig;

  // Layout constants
  const NUM_HEADS = 16;
  const CELL_WIDTH = 26; // 16 heads * 26 = 416px width
  const ROW_HEIGHT = 20;
  const GRID_WIDTH = NUM_HEADS * CELL_WIDTH;
  const LABEL_MARGIN_LEFT = 95;
  const HEADER_A_Y = 48;
  const SECTION_GAP = 56;
  const COLORBAR_WIDTH = 16;
  const COLORBAR_MARGIN = 28;

  // Initialize offscreen canvases
  if (!offscreenSummaryRef.current && typeof document !== 'undefined') {
    offscreenSummaryRef.current = document.createElement('canvas');
  }
  if (!offscreenDeltaRef.current && typeof document !== 'undefined') {
    offscreenDeltaRef.current = document.createElement('canvas');
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

        // Thin cell borders
        ctx.strokeStyle = 'rgba(15, 23, 42, 0.4)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, CELL_WIDTH, ROW_HEIGHT);
      }
    },
    []
  );

  // Full redraw of offscreen canvases (e.g. on layer switch, prefill reset)
  const fullRedrawOffscreen = useCallback(
    (summaries: number[][], deltas: number[][]) => {
      const offSummary = offscreenSummaryRef.current;
      const offDelta = offscreenDeltaRef.current;
      if (!offSummary || !offDelta) return;

      const totalSteps = summaries.length;
      const bufferHeight = Math.max(totalSteps * ROW_HEIGHT, ROW_HEIGHT);

      offSummary.width = GRID_WIDTH;
      offSummary.height = bufferHeight;
      offDelta.width = GRID_WIDTH;
      offDelta.height = bufferHeight;

      const ctxS = offSummary.getContext('2d');
      const ctxD = offDelta.getContext('2d');
      if (!ctxS || !ctxD) return;

      ctxS.clearRect(0, 0, GRID_WIDTH, bufferHeight);
      ctxD.clearRect(0, 0, GRID_WIDTH, bufferHeight);

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
      const headerBY = HEADER_A_Y + zoneHeight + SECTION_GAP;
      const canvasWidth = LABEL_MARGIN_LEFT + GRID_WIDTH + COLORBAR_MARGIN + COLORBAR_WIDTH + 64;
      const canvasHeight = headerBY + zoneHeight + 40;

      const dpr = window.devicePixelRatio || 1;
      visibleCanvas.width = canvasWidth * dpr;
      visibleCanvas.height = canvasHeight * dpr;
      visibleCanvas.style.width = `${canvasWidth}px`;
      visibleCanvas.style.height = `${canvasHeight}px`;

      ctx.scale(dpr, dpr);

      // Background
      ctx.fillStyle = '#0f172a'; // slate-900
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);

      // ==========================================
      // 1. Zone A: state_summary (Absolute Intensity)
      // ==========================================
      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 12px ui-sans-serif, system-ui';
      ctx.fillText(`色带 A: 隐状态绝对强度 S_t (Layer ${layer}, Frobenius 范数)`, LABEL_MARGIN_LEFT, 20);

      ctx.fillStyle = '#64748b';
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillText(`Viridis 线性映射 [0.0, ${summaryMax.toFixed(1)}] | 16 Heads 记忆充盈度`, LABEL_MARGIN_LEFT, 35);

      // Column (Head) headers for Zone A
      ctx.font = '10px ui-monospace, monospace';
      for (let h = 0; h < NUM_HEADS; h++) {
        const x = LABEL_MARGIN_LEFT + h * CELL_WIDTH;
        ctx.fillStyle = h === head ? '#38bdf8' : '#64748b';
        ctx.fillText(`H${h}`, x + 5, HEADER_A_Y - 5);
      }

      // Blit Zone A from offscreen buffer
      if (offSummary.width > 0 && offSummary.height > 0) {
        ctx.drawImage(offSummary, 0, 0, GRID_WIDTH, zoneHeight, LABEL_MARGIN_LEFT, HEADER_A_Y, GRID_WIDTH, zoneHeight);
      }

      // ==========================================
      // 2. Zone B: delta (Single-step Change)
      // ==========================================
      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 12px ui-sans-serif, system-ui';
      ctx.fillText(`色带 B: 单步状态变化量 Δ_t (Layer ${layer}, S_t - S_{t-1})`, LABEL_MARGIN_LEFT, headerBY - 28);

      ctx.fillStyle = '#64748b';
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillText(`Coolwarm 对称发散 [±${deltaBound.toFixed(2)}] | 红色注能增益 · 蓝色衰减遗忘`, LABEL_MARGIN_LEFT, headerBY - 14);

      // Column (Head) headers for Zone B
      ctx.font = '10px ui-monospace, monospace';
      for (let h = 0; h < NUM_HEADS; h++) {
        const x = LABEL_MARGIN_LEFT + h * CELL_WIDTH;
        ctx.fillStyle = h === head ? '#38bdf8' : '#64748b';
        ctx.fillText(`H${h}`, x + 5, headerBY - 4);
      }

      // Blit Zone B from offscreen buffer
      if (offDelta.width > 0 && offDelta.height > 0) {
        ctx.drawImage(offDelta, 0, 0, GRID_WIDTH, zoneHeight, LABEL_MARGIN_LEFT, headerBY, GRID_WIDTH, zoneHeight);
      }

      // ==========================================
      // 3. Row Labels (Tokens & Steps)
      // ==========================================
      ctx.font = '10px ui-monospace, monospace';
      for (let s = 0; s < stepCount; s++) {
        const tok = currentTokens[s] ?? '';
        const yA = HEADER_A_Y + s * ROW_HEIGHT + 14;
        const yB = headerBY + s * ROW_HEIGHT + 14;

        // Label on Zone A
        const isHovered = hover?.step === s;
        const isInspected = inspectedStep === s;
        ctx.fillStyle = isInspected ? '#f59e0b' : isHovered ? '#38bdf8' : '#94a3b8';
        const labelText = s === 0 ? `Prefill` : `Step ${s}`;
        ctx.fillText(`${labelText} ${tok.slice(0, 4)}`, 10, yA);

        // Label on Zone B
        ctx.fillText(`${labelText} ${tok.slice(0, 4)}`, 10, yB);

        // Row border highlight if inspected or hovered
        if (isInspected || isHovered) {
          ctx.strokeStyle = isInspected ? 'rgba(245, 158, 11, 0.8)' : 'rgba(56, 189, 248, 0.6)';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(LABEL_MARGIN_LEFT, HEADER_A_Y + s * ROW_HEIGHT, GRID_WIDTH, ROW_HEIGHT);
          ctx.strokeRect(LABEL_MARGIN_LEFT, headerBY + s * ROW_HEIGHT, GRID_WIDTH, ROW_HEIGHT);
        }
      }

      // ==========================================
      // 4. Colorbars on the Right
      // ==========================================
      const cbX = LABEL_MARGIN_LEFT + GRID_WIDTH + COLORBAR_MARGIN;

      // Colorbar A: Viridis
      const cbHeightA = Math.min(zoneHeight, 140);
      const gradA = ctx.createLinearGradient(0, HEADER_A_Y + cbHeightA, 0, HEADER_A_Y);
      gradA.addColorStop(0.0, 'rgb(68, 1, 84)');
      gradA.addColorStop(0.25, 'rgb(59, 82, 139)');
      gradA.addColorStop(0.5, 'rgb(33, 145, 140)');
      gradA.addColorStop(0.75, 'rgb(94, 201, 98)');
      gradA.addColorStop(1.0, 'rgb(253, 231, 37)');

      ctx.fillStyle = gradA;
      ctx.fillRect(cbX, HEADER_A_Y, COLORBAR_WIDTH, cbHeightA);
      ctx.strokeStyle = '#334155';
      ctx.strokeRect(cbX, HEADER_A_Y, COLORBAR_WIDTH, cbHeightA);

      ctx.fillStyle = '#94a3b8';
      ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(`${summaryMax.toFixed(1)}`, cbX + COLORBAR_WIDTH + 6, HEADER_A_Y + 9);
      ctx.fillText('0.0', cbX + COLORBAR_WIDTH + 6, HEADER_A_Y + cbHeightA);

      // Colorbar B: Coolwarm
      const cbHeightB = Math.min(zoneHeight, 140);
      const gradB = ctx.createLinearGradient(0, headerBY + cbHeightB, 0, headerBY);
      gradB.addColorStop(0.0, 'rgb(59, 76, 192)');    // Blue (-deltaBound)
      gradB.addColorStop(0.5, 'rgb(240, 240, 245)');  // Neutral (0)
      gradB.addColorStop(1.0, 'rgb(180, 4, 38)');     // Red (+deltaBound)

      ctx.fillStyle = gradB;
      ctx.fillRect(cbX, headerBY, COLORBAR_WIDTH, cbHeightB);
      ctx.strokeStyle = '#334155';
      ctx.strokeRect(cbX, headerBY, COLORBAR_WIDTH, cbHeightB);

      ctx.fillStyle = '#94a3b8';
      ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(`+${deltaBound.toFixed(2)}`, cbX + COLORBAR_WIDTH + 6, headerBY + 9);
      ctx.fillText('0.00', cbX + COLORBAR_WIDTH + 6, headerBY + cbHeightB / 2 + 3);
      ctx.fillText(`-${deltaBound.toFixed(2)}`, cbX + COLORBAR_WIDTH + 6, headerBY + cbHeightB);
    },
    [layer, head, summaryMax, deltaBound, inspectedStep]
  );

  // Initialize from initialSummaries or tokens
  useEffect(() => {
    const key = `${layer}_${initialSummaries?.length || 0}_${tokens.length}`;
    if (key !== lastInitialRef.current) {
      lastInitialRef.current = key;
      const initialS = initialSummaries || [];
      const initialD = initialDeltas || [];
      localSummariesRef.current = [...initialS];
      localDeltasRef.current = [...initialD];
      localTokensRef.current = [...tokens];
      setCurrentStepCount(initialS.length);
      fullRedrawOffscreen(initialS, initialD);
      renderVisibleCanvas(initialS.length, tokens, null);
    }
  }, [layer, initialSummaries, initialDeltas, tokens, fullRedrawOffscreen, renderVisibleCanvas]);

  // Incremental append when receiving WebSocket message
  useEffect(() => {
    if (!incrementalRow) return;

    const { step, token, state_summary, delta } = incrementalRow;
    const offSummary = offscreenSummaryRef.current;
    const offDelta = offscreenDeltaRef.current;
    if (!offSummary || !offDelta) return;

    // Check if step already handled
    const currentLen = localSummariesRef.current.length;
    if (step < currentLen) return;

    // Expand buffer height if needed
    const requiredHeight = (step + 1) * ROW_HEIGHT;
    if (offSummary.height < requiredHeight) {
      const newHeight = Math.max(requiredHeight, offSummary.height * 2);

      // Preserve existing pixels during resize
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

    // Incremental draw row
    drawRowToOffscreen(offSummary, step, state_summary, (v) =>
      summaryToRgbaString(v, summaryMax)
    );
    drawRowToOffscreen(offDelta, step, delta, (v) =>
      deltaToRgbaString(v, deltaBound)
    );

    // Update refs
    localSummariesRef.current[step] = state_summary;
    localDeltasRef.current[step] = delta;
    if (step >= localTokensRef.current.length) {
      localTokensRef.current.push(token);
    }

    const nextCount = step + 1;
    setCurrentStepCount(nextCount);
    renderVisibleCanvas(nextCount, localTokensRef.current, hoverInfo);

    // Auto-scroll
    if (autoScroll && containerRef.current) {
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

  // Mouse interaction: Hover tooltip & Step click
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = visibleCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const stepCount = currentStepCount;
    if (stepCount === 0) return;

    const zoneHeight = stepCount * ROW_HEIGHT;
    const headerBY = HEADER_A_Y + zoneHeight + SECTION_GAP;

    const inGridX = x >= LABEL_MARGIN_LEFT && x < LABEL_MARGIN_LEFT + GRID_WIDTH;

    // Check Zone A (state_summary)
    if (inGridX && y >= HEADER_A_Y && y < HEADER_A_Y + zoneHeight) {
      const s = Math.floor((y - HEADER_A_Y) / ROW_HEIGHT);
      const h = Math.floor((x - LABEL_MARGIN_LEFT) / CELL_WIDTH);
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

    // Check Zone B (delta)
    if (inGridX && y >= headerBY && y < headerBY + zoneHeight) {
      const s = Math.floor((y - headerBY) / ROW_HEIGHT);
      const h = Math.floor((x - LABEL_MARGIN_LEFT) / CELL_WIDTH);
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

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = visibleCanvasRef.current;
    if (!canvas || !onStepClick) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const stepCount = currentStepCount;
    if (stepCount === 0) return;

    const zoneHeight = stepCount * ROW_HEIGHT;
    const headerBY = HEADER_A_Y + zoneHeight + SECTION_GAP;
    const inGridX = x >= LABEL_MARGIN_LEFT && x < LABEL_MARGIN_LEFT + GRID_WIDTH;

    if (inGridX) {
      if (y >= HEADER_A_Y && y < HEADER_A_Y + zoneHeight) {
        const s = Math.floor((y - HEADER_A_Y) / ROW_HEIGHT);
        onStepClick(s);
      } else if (y >= headerBY && y < headerBY + zoneHeight) {
        const s = Math.floor((y - headerBY) / ROW_HEIGHT);
        onStepClick(s);
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
            Layer {layer} (固定色阶标定)
          </span>
        </div>

        <div className="flex items-center gap-3 text-xs">
          {isStreaming && (
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono text-[11px] border border-emerald-500/30 animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <span>增量绘制中</span>
            </div>
          )}

          <div className="flex items-center gap-1.5 font-mono text-slate-400 bg-slate-800/80 px-2 py-1 rounded border border-slate-700">
            <span>步数:</span>
            <span className="text-emerald-300 font-bold">{currentStepCount}</span>
          </div>

          <button
            type="button"
            onClick={() => setAutoScroll((prev) => !prev)}
            className={`flex items-center gap-1 px-2 py-1 rounded transition-colors border ${
              autoScroll
                ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30 font-medium'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
          >
            <ArrowDown className="w-3.5 h-3.5" />
            <span>自动跟踪</span>
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
              <span className="text-slate-400">变化量 Δ_t:</span>
              <span
                className={`font-bold ${
                  hoverInfo.val >= 0 ? 'text-rose-400' : 'text-blue-400'
                }`}
              >
                {hoverInfo.val >= 0 ? `+${hoverInfo.val.toFixed(4)}` : hoverInfo.val.toFixed(4)}
              </span>
            </div>
          )}

          <div className="flex items-center justify-between gap-4 text-[10px] text-slate-500 pt-0.5 border-t border-slate-800/80">
            <span>层级标定量程:</span>
            <span>
              {hoverInfo.type === 'summary'
                ? `[0, ${hoverInfo.bound.toFixed(1)}]`
                : `[±${hoverInfo.bound.toFixed(2)}]`}
            </span>
          </div>
          <div className="text-[9px] text-slate-500 text-center pt-0.5">
            点击可下钻查看第 {hoverInfo.step} 步完整状态矩阵
          </div>
        </div>
      )}
    </div>
  );
};
