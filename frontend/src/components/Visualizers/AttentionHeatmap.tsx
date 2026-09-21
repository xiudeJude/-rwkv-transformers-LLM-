import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { getViridisColor, toLogScale } from '../../utils/colormaps';
import { RefreshCw, ArrowDown, Eye, Layers } from 'lucide-react';

export interface IncrementalRowPayload {
  rowIndex: number;
  token: string;
  weights: number[]; // length L_prompt + t
}

interface AttentionHeatmapProps {
  matrix?: number[][]; // [L, L] full matrix (prefill, static inspect, or post-generation layer change)
  tokens: string[];
  layer: number;
  head: number;
  incrementalRow?: IncrementalRowPayload | null;
  isStreaming?: boolean;
  isLoading?: boolean;
  loadingMessage?: string;
}

interface HoverInfo {
  row: number;
  col: number;
  val: number;
  qToken: string;
  kToken: string;
  x: number;
  y: number;
}

export const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({
  matrix,
  tokens,
  layer,
  head,
  incrementalRow,
  isStreaming = false,
  isLoading = false,
  loadingMessage,
}) => {
  const visibleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Local state keeping the accumulated matrix & tokens
  const localMatrixRef = useRef<number[][]>([]);
  const localTokensRef = useRef<string[]>([]);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Layout constants
  const LABEL_MARGIN_TOP = 85;
  const LABEL_MARGIN_LEFT = 85;
  const COLORBAR_WIDTH = 20;
  const COLORBAR_MARGIN = 20;

  // Initialize offscreen canvas once
  if (!offscreenCanvasRef.current) {
    offscreenCanvasRef.current = document.createElement('canvas');
  }

  // Determine current matrix dimensions
  const L = localMatrixRef.current.length || (matrix ? matrix.length : 0);

  // Dynamic cell size
  const cellSize = useMemo(() => {
    if (L <= 20) return 26;
    if (L <= 45) return 20;
    return 16;
  }, [L]);

  const gridWidth = L * cellSize;
  const gridHeight = L * cellSize;
  const canvasWidth = LABEL_MARGIN_LEFT + gridWidth + COLORBAR_MARGIN + COLORBAR_WIDTH + 40;
  const canvasHeight = LABEL_MARGIN_TOP + gridHeight + 30;

  // Blit offscreen buffer to visible canvas and render axes/labels
  const renderVisibleCanvas = useCallback((currentTokens: string[], hover: HoverInfo | null) => {
    const visibleCanvas = visibleCanvasRef.current;
    const offscreen = offscreenCanvasRef.current;
    if (!visibleCanvas || !offscreen) return;

    const ctx = visibleCanvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    visibleCanvas.width = canvasWidth * dpr;
    visibleCanvas.height = canvasHeight * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#0f172a'; // slate-900
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    const totalL = currentTokens.length;

    // 1. Draw Axis Headings
    ctx.fillStyle = '#94a3b8'; // slate-400
    ctx.font = '11px ui-sans-serif, system-ui';
    ctx.fillText('Key Tokens (Attended To) →', LABEL_MARGIN_LEFT, 20);

    ctx.save();
    ctx.translate(20, LABEL_MARGIN_TOP + (totalL * cellSize) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('← Query Tokens (Attending)', 0, 0);
    ctx.restore();

    // 2. Draw Key Token Labels (Top, Rotated 45 deg)
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let j = 0; j < totalL; j++) {
      const rawTok = currentTokens[j] || '';
      const displayTok = rawTok.replace(/\n/g, '\\n').replace(/ /g, '␣');
      const cx = LABEL_MARGIN_LEFT + j * cellSize + cellSize / 2;
      const cy = LABEL_MARGIN_TOP - 8;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = hover && hover.col === j ? '#38bdf8' : '#cbd5e1';
      ctx.font = hover && hover.col === j ? 'bold 11px ui-monospace, monospace' : '10px ui-monospace, monospace';
      ctx.fillText(displayTok, 0, 0);
      ctx.restore();
    }

    // 3. Draw Query Token Labels (Left)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < totalL; i++) {
      const rawTok = currentTokens[i] || '';
      const displayTok = rawTok.replace(/\n/g, '\\n').replace(/ /g, '␣');
      const cx = LABEL_MARGIN_LEFT - 8;
      const cy = LABEL_MARGIN_TOP + i * cellSize + cellSize / 2;

      ctx.fillStyle = hover && hover.row === i ? '#38bdf8' : '#cbd5e1';
      ctx.font = hover && hover.row === i ? 'bold 11px ui-monospace, monospace' : '10px ui-monospace, monospace';
      ctx.fillText(displayTok, cx, cy);
    }

    // 4. Blit the offscreen pixel buffer (Single drawImage call: < 0.2ms!)
    if (offscreen.width > 0 && offscreen.height > 0) {
      ctx.drawImage(offscreen, LABEL_MARGIN_LEFT, LABEL_MARGIN_TOP);
    }

    // 5. Draw Hover Highlight Box & Crosshair
    if (hover) {
      const hx = LABEL_MARGIN_LEFT + hover.col * cellSize;
      const hy = LABEL_MARGIN_TOP + hover.row * cellSize;
      const curGridW = totalL * cellSize;
      const curGridH = totalL * cellSize;

      ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
      ctx.lineWidth = 1;
      ctx.strokeRect(LABEL_MARGIN_LEFT, hy, curGridW, cellSize);
      ctx.strokeRect(hx, LABEL_MARGIN_TOP, cellSize, curGridH);

      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.strokeRect(hx, hy, cellSize, cellSize);
    }

    // 6. Draw Colorbar Legend (Log-Scale Viridis)
    const cbX = LABEL_MARGIN_LEFT + totalL * cellSize + COLORBAR_MARGIN;
    const cbY = LABEL_MARGIN_TOP;
    const cbH = Math.max(120, Math.min(totalL * cellSize, 320));

    const gradient = ctx.createLinearGradient(cbX, cbY + cbH, cbX, cbY);
    for (let s = 0; s <= 20; s++) {
      const stopVal = s / 20;
      // Use log scale viridis
      const [cr, cg, cb] = getViridisColor(stopVal, true);
      gradient.addColorStop(stopVal, `rgb(${cr}, ${cg}, ${cb})`);
    }

    ctx.fillStyle = gradient;
    ctx.fillRect(cbX, cbY, COLORBAR_WIDTH, cbH);
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1;
    ctx.strokeRect(cbX, cbY, COLORBAR_WIDTH, cbH);

    // Colorbar Ticks
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('1.0', cbX + COLORBAR_WIDTH + 6, cbY);
    ctx.fillText('0.1', cbX + COLORBAR_WIDTH + 6, cbY + cbH * 0.48);
    ctx.fillText('0.01', cbX + COLORBAR_WIDTH + 6, cbY + cbH * 0.78);
    ctx.fillText('0.0', cbX + COLORBAR_WIDTH + 6, cbY + cbH);

    ctx.fillStyle = '#64748b';
    ctx.font = '9px ui-sans-serif, system-ui';
    ctx.fillText('(Log-Scale)', cbX, cbY + cbH + 15);
  }, [canvasWidth, canvasHeight, cellSize]);

  // Full redraw of offscreen canvas (for prefill, single-step, or post-generation layer change)
  const redrawFullOffscreen = useCallback((mat: number[][], toks: string[]) => {
    const offscreen = offscreenCanvasRef.current;
    if (!offscreen || mat.length === 0) return;

    const numRows = mat.length;
    const curCellSize = numRows <= 20 ? 26 : numRows <= 45 ? 20 : 16;
    const reqW = numRows * curCellSize;
    const reqH = numRows * curCellSize;

    offscreen.width = reqW;
    offscreen.height = reqH;

    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, reqW, reqH);

    // Render each cell with log-scale Viridis
    for (let i = 0; i < numRows; i++) {
      const rowLen = mat[i] ? mat[i].length : 0;
      for (let j = 0; j < rowLen; j++) {
        const val = mat[i][j] ?? 0;
        const [r, g, b] = getViridisColor(val, true);
        const x = j * curCellSize;
        const y = i * curCellSize;

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x, y, curCellSize, curCellSize);

        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, curCellSize, curCellSize);
      }
    }

    renderVisibleCanvas(toks, null);
  }, [renderVisibleCanvas]);

  // Append a single row to the offscreen buffer (Incremental Decode step)
  const appendRowToOffscreen = useCallback((rowIdx: number, weights: number[], toks: string[]) => {
    const offscreen = offscreenCanvasRef.current;
    if (!offscreen) return;

    const numCols = weights.length;
    const curCellSize = numCols <= 20 ? 26 : numCols <= 45 ? 20 : 16;
    const newWidth = Math.max(offscreen.width, numCols * curCellSize);
    const newHeight = (rowIdx + 1) * curCellSize;

    // Expand buffer if needed, preserving existing pixels
    if (offscreen.width < newWidth || offscreen.height < newHeight) {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = offscreen.width;
      tempCanvas.height = offscreen.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (tempCtx && offscreen.width > 0 && offscreen.height > 0) {
        tempCtx.drawImage(offscreen, 0, 0);
      }

      offscreen.width = newWidth;
      offscreen.height = newHeight;

      const ctx = offscreen.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, newWidth, newHeight);
        if (tempCanvas.width > 0 && tempCanvas.height > 0) {
          ctx.drawImage(tempCanvas, 0, 0);
        }
      }
    }

    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    // ONLY draw the new row (rowIdx)
    const y = rowIdx * curCellSize;
    for (let j = 0; j < numCols; j++) {
      const val = weights[j] ?? 0;
      const [r, g, b] = getViridisColor(val, true);
      const x = j * curCellSize;

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x, y, curCellSize, curCellSize);

      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x, y, curCellSize, curCellSize);
    }

    // Blit to visible canvas
    renderVisibleCanvas(toks, null);

    // Auto-scroll to bottom if enabled
    if (autoScroll && containerRef.current && rowIdx > 20) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [autoScroll, renderVisibleCanvas]);

  // Handle full matrix updates (prefill or static inspect)
  useEffect(() => {
    if (matrix && matrix.length > 0) {
      localMatrixRef.current = matrix.map((row) => [...row]);
      localTokensRef.current = [...tokens];
      setHoverInfo(null);
      redrawFullOffscreen(localMatrixRef.current, localTokensRef.current);
    }
  }, [matrix, tokens, redrawFullOffscreen]);

  // Handle incremental row streaming
  useEffect(() => {
    if (incrementalRow) {
      const { rowIndex, token, weights } = incrementalRow;
      localMatrixRef.current[rowIndex] = weights;
      if (localTokensRef.current.length <= rowIndex) {
        localTokensRef.current.push(token);
      } else {
        localTokensRef.current[rowIndex] = token;
      }
      appendRowToOffscreen(rowIndex, weights, localTokensRef.current);
    }
  }, [incrementalRow, appendRowToOffscreen]);

  // Mouse move handler for interactive cell hover inspection
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const visibleCanvas = visibleCanvasRef.current;
    const currentTokens = localTokensRef.current;
    const currentMatrix = localMatrixRef.current;
    const totalL = currentTokens.length;
    if (!visibleCanvas || totalL === 0) return;

    const rect = visibleCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const col = Math.floor((x - LABEL_MARGIN_LEFT) / cellSize);
    const row = Math.floor((y - LABEL_MARGIN_TOP) / cellSize);

    if (row >= 0 && row < totalL && col >= 0 && col <= row) {
      const val = currentMatrix[row]?.[col] ?? 0;
      const qToken = currentTokens[row] || '';
      const kToken = currentTokens[col] || '';

      const newHover: HoverInfo = { row, col, val, qToken, kToken, x: e.clientX, y: e.clientY };
      setHoverInfo(newHover);
      renderVisibleCanvas(currentTokens, newHover);
    } else if (hoverInfo) {
      setHoverInfo(null);
      renderVisibleCanvas(currentTokens, null);
    }
  };

  const handleMouseLeave = () => {
    if (hoverInfo) {
      setHoverInfo(null);
      renderVisibleCanvas(localTokensRef.current, null);
    }
  };

  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-4 shadow-xl space-y-3 relative overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-sky-400" />
          <h3 className="text-sm font-semibold text-slate-200">
            注意力权重热力图 (Attention Heatmap)
          </h3>
          <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-sky-300 font-mono border border-slate-700">
            Layer {layer} / Head {head}
          </span>
          <span className="text-[11px] px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-mono">
            {L}×{L} Causal
          </span>
          {isStreaming && (
            <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
              <span>动态生长中</span>
            </span>
          )}
        </div>


        <div className="flex items-center gap-3 text-xs">
          {/* Auto scroll toggle button */}
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`flex items-center gap-1 px-2.5 py-1 rounded font-medium transition-colors border ${
              autoScroll
                ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/20'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700'
            }`}
            title="新行产生时自动滚动到底部"
          >
            <ArrowDown className={`w-3.5 h-3.5 ${autoScroll ? 'text-emerald-400' : ''}`} />
            <span>自动滚动: {autoScroll ? '开' : '关'}</span>
          </button>

          <div className="flex items-center gap-1.5 text-slate-400">
            <Eye className="w-3.5 h-3.5 text-sky-400" />
            <span>悬浮查看 4 位浮点</span>
          </div>
        </div>
      </div>

      {/* Loading Overlay (Unified Loading Backdrop during Level 2 refetch) */}
      {isLoading && (
        <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3 z-30 transition-all">
          <RefreshCw className="w-8 h-8 text-sky-400 animate-spin" />
          <p className="text-sm font-semibold text-slate-200">
            {loadingMessage || `正在重新抓取 Layer ${layer} / Head ${head} 全量注意力矩阵...`}
          </p>
          <p className="text-xs text-slate-500">已启用 Level 2 增量矩阵重建，预计约 10ms 完成并重绘</p>
        </div>
      )}

      {/* Canvas Scroll Viewport */}
      <div
        ref={containerRef}
        className="overflow-auto max-h-[560px] border border-slate-800/80 rounded-lg bg-slate-950/60 scrollbar-thin scrollbar-thumb-slate-800 relative"
      >
        <canvas
          ref={visibleCanvasRef}
          style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px` }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="cursor-crosshair block"
        />

        {/* Interactive Hover Tooltip */}
        {hoverInfo && (
          <div
            className="pointer-events-none fixed z-50 bg-slate-900/95 border border-sky-500/60 rounded-lg p-2.5 shadow-2xl text-xs font-mono space-y-1 backdrop-blur"
            style={{
              left: `${hoverInfo.x + 15}px`,
              top: `${hoverInfo.y + 15}px`,
            }}
          >
            <div className="flex items-center justify-between gap-4 border-b border-slate-800 pb-1 text-slate-400">
              <span>Cell [{hoverInfo.row}, {hoverInfo.col}]</span>
              <span className="text-sky-400 font-bold">L{layer} H{head}</span>
            </div>
            <div className="text-slate-300">
              Query: <span className="text-emerald-300 font-bold">"{hoverInfo.qToken}"</span> (#{hoverInfo.row})
            </div>
            <div className="text-slate-300">
              Key: <span className="text-sky-300 font-bold">"{hoverInfo.kToken}"</span> (#{hoverInfo.col})
            </div>
            <div className="flex items-center justify-between gap-3 pt-1 border-t border-slate-800">
              <span className="text-slate-400">Weight:</span>
              <span className="text-amber-300 font-bold text-sm">
                {hoverInfo.val.toFixed(4)}
              </span>
            </div>
            <div className="flex items-center justify-between text-[10px] text-slate-500">
              <span>Log-scaled:</span>
              <span>{(toLogScale(hoverInfo.val) * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
