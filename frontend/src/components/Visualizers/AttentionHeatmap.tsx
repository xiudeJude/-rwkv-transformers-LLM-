import React, { useRef, useEffect, useState, useMemo } from 'react';
import { getViridisColor } from '../../utils/colormaps';

interface AttentionHeatmapProps {
  matrix: number[][]; // [L, L]
  tokens: string[];
  layer: number;
  head: number;
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
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);

  const L = matrix.length;

  // Layout parameters
  const LABEL_MARGIN_TOP = 90;  // Space for Key tokens (top)
  const LABEL_MARGIN_LEFT = 90; // Space for Query tokens (left)
  const COLORBAR_WIDTH = 24;
  const COLORBAR_MARGIN = 20;

  // Compute cell size dynamically based on container or minimum readable cell size
  const cellSize = useMemo(() => {
    if (L === 0) return 30;
    // Keep cells readable: between 22px and 45px
    return Math.max(22, Math.min(45, Math.floor(520 / Math.max(L, 1))));
  }, [L]);

  const gridWidth = L * cellSize;
  const gridHeight = L * cellSize;
  const canvasWidth = LABEL_MARGIN_LEFT + gridWidth + COLORBAR_MARGIN + COLORBAR_WIDTH + 40;
  const canvasHeight = LABEL_MARGIN_TOP + gridHeight + 30;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || L === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = canvasHeight * dpr;
    ctx.scale(dpr, dpr);

    // Clear background
    ctx.fillStyle = '#0f172a'; // slate-900
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // 1. Draw Axis Headings
    ctx.fillStyle = '#94a3b8'; // slate-400
    ctx.font = '11px ui-sans-serif, system-ui';
    ctx.fillText('Key Tokens (Attended To) →', LABEL_MARGIN_LEFT, 20);

    ctx.save();
    ctx.translate(20, LABEL_MARGIN_TOP + gridHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('← Query Tokens (Attending)', 0, 0);
    ctx.restore();

    // 2. Draw Key Token Labels (Top, Rotated)
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let j = 0; j < L; j++) {
      const rawTok = tokens[j] || '';
      const displayTok = rawTok.replace(/\n/g, '\\n').replace(/ /g, '␣');
      const cx = LABEL_MARGIN_LEFT + j * cellSize + cellSize / 2;
      const cy = LABEL_MARGIN_TOP - 8;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = hoverInfo && hoverInfo.col === j ? '#38bdf8' : '#cbd5e1';
      ctx.font = hoverInfo && hoverInfo.col === j ? 'bold 11px ui-monospace, monospace' : '10px ui-monospace, monospace';
      ctx.fillText(displayTok, 0, 0);
      ctx.restore();
    }

    // 3. Draw Query Token Labels (Left)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < L; i++) {
      const rawTok = tokens[i] || '';
      const displayTok = rawTok.replace(/\n/g, '\\n').replace(/ /g, '␣');
      const cx = LABEL_MARGIN_LEFT - 8;
      const cy = LABEL_MARGIN_TOP + i * cellSize + cellSize / 2;

      ctx.fillStyle = hoverInfo && hoverInfo.row === i ? '#38bdf8' : '#cbd5e1';
      ctx.font = hoverInfo && hoverInfo.row === i ? 'bold 11px ui-monospace, monospace' : '10px ui-monospace, monospace';
      ctx.fillText(displayTok, cx, cy);
    }

    // 4. Draw Attention Matrix Grid Cells
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < L; j++) {
        const val = matrix[i]?.[j] ?? 0;
        const [r, g, b] = getViridisColor(val);

        const x = LABEL_MARGIN_LEFT + j * cellSize;
        const y = LABEL_MARGIN_TOP + i * cellSize;

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x, y, cellSize, cellSize);

        // Subtle cell border
        ctx.strokeStyle = '#1e293b'; // slate-800
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, cellSize, cellSize);

        // Optionally print numbers if cell is large enough
        if (cellSize >= 34) {
          ctx.fillStyle = val > 0.5 ? '#000000' : '#ffffff';
          ctx.font = '9px ui-monospace, monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(val.toFixed(2), x + cellSize / 2, y + cellSize / 2);
        }
      }
    }

    // 5. Draw Hover Highlight Box & Crosshair
    if (hoverInfo) {
      const hx = LABEL_MARGIN_LEFT + hoverInfo.col * cellSize;
      const hy = LABEL_MARGIN_TOP + hoverInfo.row * cellSize;

      // Crosshair lines
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)'; // sky-400
      ctx.lineWidth = 1;
      // Horizontal crosshair
      ctx.strokeRect(LABEL_MARGIN_LEFT, hy, gridWidth, cellSize);
      // Vertical crosshair
      ctx.strokeRect(hx, LABEL_MARGIN_TOP, cellSize, gridHeight);

      // Selected cell highlight
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.strokeRect(hx, hy, cellSize, cellSize);
    }

    // 6. Draw Colorbar Legend
    const cbX = LABEL_MARGIN_LEFT + gridWidth + COLORBAR_MARGIN;
    const cbY = LABEL_MARGIN_TOP;
    const cbH = gridHeight;

    const gradient = ctx.createLinearGradient(cbX, cbY + cbH, cbX, cbY);
    for (let step = 0; step <= 10; step++) {
      const stopVal = step / 10;
      const [cr, cg, cb] = getViridisColor(stopVal);
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
    ctx.fillText('0.5', cbX + COLORBAR_WIDTH + 6, cbY + cbH / 2);
    ctx.fillText('0.0', cbX + COLORBAR_WIDTH + 6, cbY + cbH);

  }, [matrix, tokens, cellSize, hoverInfo, canvasWidth, canvasHeight, L]);

  // Mouse Move Handler for Interactive Inspection
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || L === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const col = Math.floor((x - LABEL_MARGIN_LEFT) / cellSize);
    const row = Math.floor((y - LABEL_MARGIN_TOP) / cellSize);

    if (col >= 0 && col < L && row >= 0 && row < L) {
      const val = matrix[row]?.[col] ?? 0;
      setHoverInfo({
        row,
        col,
        val,
        qToken: tokens[row] || '',
        kToken: tokens[col] || '',
        x: e.clientX,
        y: e.clientY,
      });
    } else {
      setHoverInfo(null);
    }
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  if (L === 0) {
    return (
      <div className="flex items-center justify-center h-72 border border-dashed border-slate-800 rounded-xl text-slate-500">
        无注意力矩阵数据，请点击"开始分析"
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative overflow-auto max-w-full p-4 bg-slate-900/60 rounded-xl border border-slate-800/80 shadow-2xl">
      <div className="flex items-center justify-between mb-2 px-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            Layer {layer}
          </span>
          <span className="text-xs font-semibold px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30">
            Head {head}
          </span>
          <span className="text-xs text-slate-400">
            ({L} × {L} Attention Matrix)
          </span>
        </div>
        <div className="text-xs text-slate-400">
          Viridis 色标 • 悬停查看数值与 Token 对应关系
        </div>
      </div>

      <canvas
        ref={canvasRef}
        style={{ width: canvasWidth, height: canvasHeight }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="cursor-crosshair rounded"
      />

      {/* Hover Floating Tooltip */}
      {hoverInfo && (
        <div
          className="pointer-events-none fixed z-50 transform -translate-x-1/2 -translate-y-full mb-3 px-3 py-2 bg-slate-950/95 border border-sky-500/50 rounded-lg shadow-2xl backdrop-blur-sm text-xs text-slate-200"
          style={{ left: hoverInfo.x, top: hoverInfo.y - 10 }}
        >
          <div className="font-semibold text-sky-400 mb-1 flex items-center gap-2">
            <span>Query [{hoverInfo.row}]:</span>
            <code className="bg-slate-800 px-1.5 py-0.5 rounded text-white">
              {hoverInfo.qToken}
            </code>
            <span>→ Key [{hoverInfo.col}]:</span>
            <code className="bg-slate-800 px-1.5 py-0.5 rounded text-white">
              {hoverInfo.kToken}
            </code>
          </div>
          <div className="flex items-center justify-between gap-4 text-slate-300">
            <span>注意力权重:</span>
            <span className="font-mono font-bold text-amber-300">
              {hoverInfo.val.toFixed(6)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
