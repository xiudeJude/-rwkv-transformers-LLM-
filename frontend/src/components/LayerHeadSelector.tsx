import React from 'react';
import { Lock } from 'lucide-react';

interface LayerHeadSelectorProps {
  numLayers: number;
  numHeads: number;
  currentLayer: number;
  currentHead: number;
  onLayerChange: (layer: number) => void;
  onHeadChange: (head: number) => void;
  temperature: number;
  onTemperatureChange: (temp: number) => void;
  topK: number;
  onTopKChange: (k: number) => void;
  repetitionPenalty?: number;
  onRepetitionPenaltyChange?: (penalty: number) => void;
  fullAttnLayers?: number[];
  disabled?: boolean;
  isStreaming?: boolean;
  modelType?: 'transformer' | 'rwkv' | 'mamba';
  showHead?: boolean;
}

export const LayerHeadSelector: React.FC<LayerHeadSelectorProps> = ({
  numLayers,
  numHeads,
  currentLayer,
  currentHead,
  onLayerChange,
  onHeadChange,
  temperature,
  onTemperatureChange,
  topK,
  onTopKChange,
  repetitionPenalty = 1.2,
  onRepetitionPenaltyChange,
  fullAttnLayers = [],
  disabled = false,
  isStreaming = false,
  modelType = 'transformer',
  showHead = true,
}) => {
  const isHybrid = fullAttnLayers.length > 0 && fullAttnLayers.length < numLayers;
  const isFullAttn = !isHybrid || fullAttnLayers.includes(currentLayer);
  const selectableLayers = isHybrid
    ? fullAttnLayers
    : Array.from({ length: numLayers }, (_, index) => index);
  const sliderIndex = Math.max(0, selectableLayers.indexOf(currentLayer));
  const sliderMax = Math.max(0, selectableLayers.length - 1);
  const firstSelectable = selectableLayers[0] ?? 0;
  const middleSelectable = selectableLayers[Math.floor(selectableLayers.length / 2)] ?? 0;
  const lastSelectable = selectableLayers[selectableLayers.length - 1] ?? 0;

  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-4 space-y-4 shadow-xl">
      <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 flex items-center justify-between">
        <span>观察维度与采样参数</span>
        {isHybrid && (
          <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-medium border ${
            isFullAttn 
              ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' 
              : 'bg-amber-500/10 text-amber-300 border-amber-500/30'
          }`}>
            {isFullAttn ? 'Full Attention (Softmax)' : 'Linear Attention (DeltaNet)'}
          </span>
        )}
      </h3>

      {/* Locked notice during active generation */}
      {disabled && isStreaming && (
        <div className="p-2.5 bg-amber-500/10 border border-amber-500/30 rounded-lg text-xs text-amber-300 flex items-center gap-2 animate-pulse">
          <Lock className="w-4 h-4 text-amber-400 shrink-0" />
          <span>生成中已锁定观察 Layer {currentLayer} / Head {currentHead}，完成后可切换查看其他层</span>
        </div>
      )}


      {/* Layer selector slider & buttons */}
      <div>
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="text-slate-400">
            {modelType === 'rwkv'
              ? '选择 RWKV 状态递推层 (Layer):'
              : modelType === 'mamba'
                ? '选择 Mamba 递归层 (Layer):'
                : '选择 Transformer 层 (Layer):'}
          </span>
          <span className="font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
            Layer {currentLayer} / {Math.max(0, numLayers - 1)}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={sliderMax}
          value={sliderIndex}
          onChange={(e) => {
            const next = selectableLayers[parseInt(e.target.value)];
            if (next !== undefined) onLayerChange(next);
          }}
          disabled={disabled || selectableLayers.length <= 1}
          className="w-full accent-indigo-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
        />
        {isHybrid && (
          <p className="mt-1 text-[10px] text-slate-500">
            滑块只经过 {fullAttnLayers.length} 个有 Softmax 注意力矩阵的 Full Attention 层。
          </p>
        )}

        {isHybrid && !isFullAttn && (
          <div className="mt-2 p-2 rounded-lg border border-amber-500/30 bg-amber-500/10 text-[11px] text-amber-300">
            当前 Layer {currentLayer} 是线性注意力层，没有 Softmax 注意力矩阵。请选择上方任一 Full Attention 层。
          </div>
        )}

        {/* If hybrid architecture, show quick buttons for Full Attention layers */}
        {isHybrid && (
          <div className="mt-2 flex items-center gap-1.5 flex-wrap">
            <span className="text-[10px] text-slate-400">完整注意力层:</span>
            {fullAttnLayers.map((l) => (
              <button
                key={l}
                type="button"
                onClick={() => onLayerChange(l)}
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
                  currentLayer === l
                    ? 'bg-emerald-500 text-white font-bold border-emerald-400 shadow-sm'
                    : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
                }`}
              >
                L{l}
              </button>
            ))}
          </div>
        )}

        <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
          <span>L{firstSelectable}</span>
          <span>L{middleSelectable}</span>
          <span>L{lastSelectable}</span>
        </div>
      </div>

      {/* Head selector: Mamba has no attention/RWKV head semantics. */}
      {showHead && (
        <div>
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="text-slate-400">
              {modelType === 'rwkv' ? '选择状态观察头 (Head):' : '选择注意力头 (Head):'}
            </span>
            <span className="font-mono font-bold text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-500/20">
              Head {currentHead} / {Math.max(0, numHeads - 1)}
            </span>
          </div>
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-1.5">
            {Array.from({ length: numHeads || 8 }).map((_, idx) => (
              <button
                key={idx}
                type="button"
                disabled={disabled}
                onClick={() => onHeadChange(idx)}
                className={`py-1 text-xs font-mono rounded border transition-all ${
                  currentHead === idx
                    ? 'bg-sky-500 text-white font-bold border-sky-400 shadow-md shadow-sky-500/20'
                    : 'bg-slate-800/80 text-slate-400 border-slate-700 hover:bg-slate-700 hover:text-slate-200'
                }`}
              >
                H{idx}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Temperature and Top-K controls */}
      <div className="pt-2 border-t border-slate-800/80 space-y-3">
        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-slate-400">Temperature (采样温度):</span>
            <span className="font-mono text-amber-400">{temperature.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={2.0}
            step={0.05}
            value={temperature}
            onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full accent-amber-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
          />
        </div>

        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-slate-400">Top-K 候选数量:</span>
            <span className="font-mono text-slate-300">{topK}</span>
          </div>
          <div className="flex gap-2">
            {[5, 10, 15, 20].map((k) => (
              <button
                key={k}
                type="button"
                disabled={disabled}
                onClick={() => onTopKChange(k)}
                className={`flex-1 py-1 text-xs font-mono rounded border transition-all ${
                  topK === k
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 font-bold'
                    : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700'
                }`}
              >
                {k}
              </button>
            ))}
          </div>
        </div>

        {onRepetitionPenaltyChange && (
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-slate-400">Repetition Penalty (重复抑制):</span>
              <span className="font-mono text-emerald-400">{repetitionPenalty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={2.0}
              step={0.05}
              value={repetitionPenalty}
              onChange={(e) => onRepetitionPenaltyChange(parseFloat(e.target.value))}
              disabled={disabled}
              className="w-full accent-emerald-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
            />
          </div>
        )}
      </div>
    </div>
  );
};
