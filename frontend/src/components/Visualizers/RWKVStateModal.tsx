import React, { useRef, useEffect } from 'react';
import { X, Cpu } from 'lucide-react';
import { getViridisColor } from '../../utils/colormaps';

interface RWKVStateModalProps {
  isOpen: boolean;
  onClose: () => void;
  step: number;
  layer: number;
  head: number;
  token?: string;
  matrix: number[][]; // [64, 64]
  headNorm?: number;
}

export const RWKVStateModal: React.FC<RWKVStateModalProps> = ({
  isOpen,
  onClose,
  step,
  layer,
  head,
  token,
  matrix,
  headNorm,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!isOpen || !matrix || matrix.length === 0) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = matrix.length;
    const cols = matrix[0]?.length || 0;
    if (rows === 0 || cols === 0) return;

    // Find min and max for normalization
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = matrix[r][c];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
    const range = Math.max(maxVal - minVal, 1e-6);

    const CELL = 6;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = cols * CELL * dpr;
    canvas.height = rows * CELL * dpr;
    canvas.style.width = `${cols * CELL}px`;
    canvas.style.height = `${rows * CELL}px`;

    ctx.scale(dpr, dpr);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const norm = (matrix[r][c] - minVal) / range;
        const [red, green, blue] = getViridisColor(norm, false);
        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
      }
    }
  }, [isOpen, matrix]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl shadow-2xl max-w-xl w-full p-5 space-y-4 animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2.5">
            <Cpu className="w-5 h-5 text-emerald-400" />
            <div>
              <h3 className="text-sm font-bold text-slate-100">
                RWKV-7 隐状态快照 (Head {head})
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                Step {step} {token ? `("${token}")` : ''} · Layer {layer}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Badges */}
        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="bg-slate-800/80 px-2.5 py-1 rounded border border-slate-700/60 text-slate-300">
            矩阵规格: <span className="text-emerald-400 font-bold">64 × 64 (4,096 参数)</span>
          </div>
          {headNorm !== undefined && (
            <div className="bg-slate-800/80 px-2.5 py-1 rounded border border-slate-700/60 text-slate-300">
              Frobenius 范数: <span className="text-yellow-400 font-bold">{headNorm.toFixed(4)}</span>
            </div>
          )}
        </div>

        {/* Canvas Display */}
        <div className="flex flex-col items-center justify-center p-3 bg-slate-950 rounded-xl border border-slate-800">
          <canvas ref={canvasRef} className="rounded shadow-md block" />
          <div className="w-full flex items-center justify-between text-[10px] text-slate-500 font-mono mt-2 px-1">
            <span>[64, 64] 内部递推状态张量</span>
            <span>色标: Viridis (线性自适应)</span>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end pt-1">
          <button
            type="button"
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors"
          >
            关闭详情
          </button>
        </div>
      </div>
    </div>
  );
};
