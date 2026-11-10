import React from 'react';
import { Activity, HardDrive, Layers } from 'lucide-react';
import type { InternalState, RecurrentStateComponent } from '../../types/schema';

interface MambaStatePanelProps {
  state: InternalState | null;
  states: InternalState[];
  currentLayer: number;
  inspectedStep: number | null;
  onSelectStep: (step: number) => void;
  isLoading: boolean;
  loadingMessage: string;
}

const formatBytes = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
};

const formatShape = (shape: number[]): string =>
  Array.isArray(shape) && shape.length > 0 ? `[${shape.join(' × ')}]` : '—';

const dtypeClass = (dtype: string): string => {
  const value = dtype.toLowerCase();
  if (value.includes('float16') || value.includes('half')) return 'text-amber-300 border-amber-500/30 bg-amber-500/10';
  if (value.includes('float32') || value.includes('float')) return 'text-sky-300 border-sky-500/30 bg-sky-500/10';
  return 'text-slate-300 border-slate-600 bg-slate-800';
};

const ComponentCard: React.FC<{
  component: RecurrentStateComponent;
  currentLayer: number;
}> = ({ component, currentLayer }) => {
  const norms = component.layer_norms ?? [];
  const maxNorm = Math.max(...norms, 1e-9);

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3 space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-bold text-slate-200">{component.name}</span>
          <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${dtypeClass(component.dtype)}`}>
            {component.dtype}
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono text-slate-400">
          <span>shape {formatShape(component.shape)}</span>
          <span>{formatBytes(component.bytes)}</span>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
          <span>逐层 Frobenius 范数摘要</span>
          <span>{norms.length} layers</span>
        </div>
        <div
          className="grid items-end gap-[2px] h-24 px-1 py-1 rounded bg-slate-900/80 border border-slate-800/80"
          style={{ gridTemplateColumns: `repeat(${Math.max(norms.length, 1)}, minmax(0, 1fr))` }}
        >
          {norms.map((norm, index) => (
            <div
              key={index}
              className={`min-h-[2px] rounded-sm transition-colors ${
                index === currentLayer ? 'bg-sky-400 ring-1 ring-sky-200' : 'bg-emerald-500/70'
              }`}
              style={{ height: `${Math.max(2, (norm / maxNorm) * 100)}%` }}
              title={`Layer ${index}: ${norm.toFixed(6)}`}
            />
          ))}
        </div>
        <div className="flex justify-between text-[9px] text-slate-500 font-mono mt-1">
          <span>L0</span>
          <span>L{Math.max(0, Math.floor(norms.length / 2))}</span>
          <span>L{Math.max(0, norms.length - 1)}</span>
        </div>
      </div>
    </div>
  );
};

export const MambaStatePanel: React.FC<MambaStatePanelProps> = ({
  state,
  states,
  currentLayer,
  inspectedStep,
  onSelectStep,
  isLoading,
  loadingMessage,
}) => {
  const components = state?.meta?.components ?? [];
  const stepOptions = states.length > 0 ? states : state ? [state] : [];

  if (!state) {
    return (
      <div className="h-96 rounded-xl border border-dashed border-slate-800 bg-slate-900/40 flex flex-col items-center justify-center text-slate-500 space-y-3 p-6 text-center">
        <div className="w-12 h-12 rounded-full bg-slate-900 flex items-center justify-center text-cyan-400/80 border border-slate-800">
          <Activity className="w-6 h-6" />
        </div>
        <div>
          <p className="text-sm text-slate-300 font-medium">Mamba 递归状态面板就绪</p>
          <p className="text-xs text-slate-500 mt-1">
            点击“流式生成 (WS)”，这里会按 InternalState 的 components 展示 conv_state 与 ssm_state。
          </p>
          <p className="text-[10px] text-slate-600 mt-2">
            纯 PyTorch 参考路径：无 mamba-ssm / causal-conv1d 加速，生成速度较慢。
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-4 shadow-xl relative">
      <div className="flex items-start justify-between gap-3 flex-wrap border-b border-slate-800 pb-3">
        <div className="space-y-1">
          <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span>Mamba 递归状态面板</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-cyan-500/30 bg-cyan-500/10 text-cyan-300">
              kind=recurrent_state
            </span>
          </h3>
          <p className="text-[10px] text-slate-500">
            由统一 InternalState 驱动，按 meta.components 通用渲染；展示的是状态范数摘要，不是注意力权重或 RWKV head。
          </p>
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
          <span className="px-2 py-1 rounded border border-slate-700 bg-slate-950">Layer {currentLayer}</span>
          <span className="px-2 py-1 rounded border border-slate-700 bg-slate-950">Step {state.step}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
          <div className="text-[10px] text-slate-500 flex items-center gap-1"><Layers className="w-3 h-3" />层数</div>
          <div className="text-sm font-mono font-bold text-slate-100 mt-1">{state.num_layers}</div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
          <div className="text-[10px] text-slate-500 flex items-center gap-1"><HardDrive className="w-3 h-3" />state_bytes</div>
          <div className="text-sm font-mono font-bold text-emerald-300 mt-1">{formatBytes(state.state_bytes)}</div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
          <div className="text-[10px] text-slate-500">序列长度</div>
          <div className="text-sm font-mono font-bold text-sky-300 mt-1">{state.seq_len}</div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-2.5">
          <div className="text-[10px] text-slate-500">组件</div>
          <div className="text-sm font-mono font-bold text-amber-300 mt-1">{components.length}</div>
        </div>
      </div>

      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-[10px] text-slate-500">历史快照:</span>
        {stepOptions.map((item) => (
          <button
            key={item.step}
            type="button"
            onClick={() => onSelectStep(item.step)}
            className={`text-[10px] font-mono px-2 py-1 rounded border transition-colors ${
              (inspectedStep ?? state.step) === item.step
                ? 'bg-cyan-500 text-white border-cyan-400'
                : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
            }`}
          >
            {item.step === 0 ? 'Prefill' : `Step ${item.step}`}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {components.map((component) => (
          <ComponentCard key={component.name} component={component} currentLayer={currentLayer} />
        ))}
      </div>

      <div className="text-[10px] text-slate-500 border-t border-slate-800 pt-2 leading-relaxed">
        conv_state 使用 fp16，ssm_state 使用 float32；state_bytes 由各组件张量字节数求和，
        与 prompt / 生成序列长度无关。当前纯 PyTorch 参考路径无 mamba-ssm / causal-conv1d 加速。
      </div>

      {isLoading && (
        <div className="absolute inset-0 rounded-xl bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center gap-2 z-20">
          <Activity className="w-6 h-6 text-cyan-400 animate-spin" />
          <span className="text-xs text-slate-200">{loadingMessage || '正在加载 Mamba 状态...'}</span>
        </div>
      )}
    </div>
  );
};
