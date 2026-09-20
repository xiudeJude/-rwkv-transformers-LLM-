import React from 'react';

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
  disabled?: boolean;
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
  disabled = false,
}) => {
  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-4 space-y-4 shadow-xl">
      <h3 className="text-sm font-semibold text-slate-200 border-b border-slate-800 pb-2 flex items-center justify-between">
        <span>观察维度与采样参数</span>
      </h3>

      {/* Layer selector slider & buttons */}
      <div>
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="text-slate-400">选择 Transformer 层 (Layer):</span>
          <span className="font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
            Layer {currentLayer} / {Math.max(0, numLayers - 1)}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(0, numLayers - 1)}
          value={currentLayer}
          onChange={(e) => onLayerChange(parseInt(e.target.value))}
          disabled={disabled || numLayers <= 1}
          className="w-full accent-indigo-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
        />
        <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
          <span>0 (底层词法)</span>
          <span>{Math.floor(numLayers / 2)} (中层句法)</span>
          <span>{Math.max(0, numLayers - 1)} (高层语义)</span>
        </div>
      </div>

      {/* Head selector */}
      <div>
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="text-slate-400">选择注意力头 (Head):</span>
          <span className="font-mono font-bold text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-500/20">
            Head {currentHead} / {Math.max(0, numHeads - 1)}
          </span>
        </div>
        <div className="grid grid-cols-7 gap-1">
          {Array.from({ length: numHeads || 14 }).map((_, idx) => (
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
      </div>
    </div>
  );
};
