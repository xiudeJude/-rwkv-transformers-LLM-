import React, { useState } from 'react';
import { BookOpen, FlaskConical, Play, Loader2 } from 'lucide-react';

interface ConceptLabProps {
  apiBase: string;
}

interface MemorySample {
  step: number;
  seq_len: number;
  cache_bytes: number;
  viz_history_bytes: number;
  state_bytes: number;
}

interface TheoreticalCurve {
  available: boolean;
  values: number[];
  formula?: string;
  reason?: string;
  params?: {
    kv_layers?: number;
    kv_heads?: number;
    head_dim?: number;
    element_bytes?: number;
    layer_source?: string;
  };
}

interface GrowModel {
  model_type: 'transformer' | 'rwkv' | 'mamba';
  model_id: string;
  available: boolean;
  error?: string;
  samples: MemorySample[];
  cache_series_note?: string;
  theoretical?: TheoreticalCurve;
}

interface GrowthResponse {
  prompt: string;
  max_tokens: number;
  models: GrowModel[];
}

const formatBytes = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
};

const MODEL_COLORS: Record<string, string> = {
  transformer: '#818cf8',
  rwkv: '#34d399',
  mamba: '#22d3ee',
};

const ConceptCard: React.FC<{
  index: string;
  title: string;
  explanation: string;
  children: React.ReactNode;
  mapping: string;
}> = ({ index, title, explanation, children, mapping }) => (
  <section className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-5 shadow-xl space-y-4">
    <div>
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-indigo-500/30 bg-indigo-500/10 text-indigo-300">
          概念 {index}
        </span>
        <h2 className="text-sm font-semibold text-slate-100">{title}</h2>
      </div>
      <p className="text-xs text-slate-400 mt-2 leading-relaxed">{explanation}</p>
    </div>
    {children}
    <div className="border-t border-slate-800 pt-3 text-[11px] text-slate-500 leading-relaxed">
      <span className="text-slate-400 font-medium">在三种架构里分别对应什么：</span>
      {mapping}
    </div>
  </section>
);

const MemoryChart: React.FC<{ models: GrowModel[] }> = ({ models }) => {
  const available = models.filter((model) => model.available && model.samples.length > 0);
  if (available.length === 0) {
    return <div className="h-56 rounded-lg border border-dashed border-slate-800 flex items-center justify-center text-xs text-slate-500">暂无可用曲线</div>;
  }

  const width = 780;
  const height = 300;
  const padLeft = 66;
  const padRight = 18;
  const padTop = 24;
  const padBottom = 40;
  const maxStep = Math.max(...available.map((model) => Math.max(...model.samples.map((sample) => sample.step))), 1);
  const allValues = available.flatMap((model) => [
    ...model.samples.map((sample) => sample.cache_bytes),
    ...(model.theoretical?.available ? model.theoretical.values : []),
  ]);
  const maxValue = Math.max(...allValues, 1);

  const x = (step: number) => padLeft + (step / maxStep) * (width - padLeft - padRight);
  const y = (value: number) => height - padBottom - (value / maxValue) * (height - padTop - padBottom);
  const line = (values: number[], steps: number[]) =>
    values.map((value, index) => `${index === 0 ? 'M' : 'L'} ${x(steps[index])} ${y(value)}`).join(' ');

  return (
    <div className="space-y-2">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto rounded-lg bg-slate-950/80 border border-slate-800">
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const value = maxValue * (1 - ratio);
          return (
            <g key={ratio}>
              <line x1={padLeft} x2={width - padRight} y1={y(value)} y2={y(value)} stroke="#1e293b" strokeWidth="1" />
              <text x={padLeft - 8} y={y(value) + 3} textAnchor="end" fill="#64748b" fontSize="9" fontFamily="ui-monospace, monospace">
                {formatBytes(value)}
              </text>
            </g>
          );
        })}
        <line x1={padLeft} x2={width - padRight} y1={height - padBottom} y2={height - padBottom} stroke="#475569" />
        {[0, Math.floor(maxStep / 2), maxStep].map((step) => (
          <text key={step} x={x(step)} y={height - padBottom + 16} textAnchor="middle" fill="#64748b" fontSize="9" fontFamily="ui-monospace, monospace">
            {step}
          </text>
        ))}
        {available.map((model) => {
          const steps = model.samples.map((sample) => sample.step);
          const values = model.samples.map((sample) => sample.cache_bytes);
          return (
            <g key={model.model_type}>
              <path d={line(values, steps)} fill="none" stroke={MODEL_COLORS[model.model_type]} strokeWidth="2.5" />
              {model.samples.map((sample, index) => (
                <circle key={index} cx={x(sample.step)} cy={y(sample.cache_bytes)} r="2.5" fill={MODEL_COLORS[model.model_type]} />
              ))}
              {model.theoretical?.available && model.theoretical.values.length === steps.length && (
                <path
                  d={line(model.theoretical.values, steps)}
                  fill="none"
                  stroke={MODEL_COLORS[model.model_type]}
                  strokeWidth="1.5"
                  strokeDasharray="5 4"
                  opacity="0.75"
                />
              )}
            </g>
          );
        })}
        <text x={width / 2} y={height - 6} textAnchor="middle" fill="#64748b" fontSize="10">decode step</text>
        <text x={12} y={padTop} fill="#64748b" fontSize="10" transform={`rotate(-90 12 ${padTop})`}>cache_bytes</text>
      </svg>
      <div className="flex items-center gap-4 flex-wrap text-[10px] text-slate-400 font-mono">
        {available.map((model) => (
          <span key={model.model_type} className="flex items-center gap-1.5">
            <span className="w-4 h-0.5 rounded" style={{ backgroundColor: MODEL_COLORS[model.model_type] }} />
            {model.model_type}
          </span>
        ))}
        <span className="flex items-center gap-1.5 text-slate-500">
          <span className="w-4 h-0.5 border-t border-dashed border-slate-400" />
          理论 KV（虚线）
        </span>
      </div>
    </div>
  );
};

const ConceptA: React.FC<{ apiBase: string }> = ({ apiBase }) => {
  const [prompt, setPrompt] = useState('The capital of France is');
  const [maxTokens, setMaxTokens] = useState(8);
  const [data, setData] = useState<GrowthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBase}/api/concept/memory-growth`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: maxTokens }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `实验失败 (${response.status})`);
      }
      setData(await response.json());
    } catch (err: any) {
      setError(err.message || '实验请求失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ConceptCard
      index="A"
      title="状态与内存增长对比"
      explanation="同一段英文 prompt 下，模型每生成一个 token 必须保留多少推理缓存？三个模型会逐个加载、测量、卸载，避免同时占用显存。"
      mapping="Transformer：KV Cache（混合层还含线性递归状态）；RWKV：WKV 递归状态；Mamba：conv_state + ssm_state。"
    >
      <div className="space-y-3">
        <div className="flex gap-2 flex-wrap items-center">
          <input
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            disabled={loading}
            className="flex-1 min-w-[260px] bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 disabled:opacity-60"
            placeholder="English prompt"
          />
          <label className="flex items-center gap-2 text-[11px] text-slate-400">
            生成上限
            <input
              type="range"
              min={1}
              max={32}
              value={maxTokens}
              onChange={(event) => setMaxTokens(Number(event.target.value))}
              disabled={loading}
              className="w-28 accent-indigo-500"
            />
            <span className="font-mono text-indigo-300">{maxTokens}</span>
          </label>
          <button
            type="button"
            onClick={run}
            disabled={loading || !prompt.trim()}
            className="px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white flex items-center gap-1.5"
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            {loading ? '逐个加载模型中...' : '运行对比'}
          </button>
        </div>
        {error && <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded px-3 py-2">{error}</div>}
        {data && (
          <div className="space-y-3">
            <MemoryChart models={data.models} />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              {data.models.map((model) => (
                <div key={model.model_type} className="rounded-lg border border-slate-800 bg-slate-950/70 p-3 text-[11px] text-slate-400">
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-slate-200">{model.model_type}</span>
                    {!model.available && <span className="text-rose-300">不可用</span>}
                  </div>
                  {model.available ? (
                    <>
                      <p className="mt-1 leading-relaxed">{model.cache_series_note}</p>
                      <p className="mt-1 text-slate-500">{model.theoretical?.available ? '包含理论 KV 虚线' : model.theoretical?.reason}</p>
                    </>
                  ) : (
                    <p className="mt-1 text-rose-300/80">{model.error}</p>
                  )}
                </div>
              ))}
            </div>
            <div className="text-[10px] text-slate-500 leading-relaxed">
              虚线为理论 KV Cache：2 × KV层数 × KV头数 × 头维度 × 序列长度 × 每元素字节数，参数从模型 config 读取；RWKV/Mamba 没有 KV Cache，不绘制理论线。
              理论值只计算 KV；Qwen 混合架构的线性层递归状态和实际缓存布局以实测为准。
            </div>
          </div>
        )}
      </div>
    </ConceptCard>
  );
};

export const ConceptLab: React.FC<ConceptLabProps> = ({ apiBase }) => {
  return (
    <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
      <header className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-rose-500 flex items-center justify-center shadow-lg shadow-amber-500/20">
          <FlaskConical className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-white flex items-center gap-2">
            概念实验室
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-amber-500/30 bg-amber-500/10 text-amber-300">
              Concept Lab
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            用可交互的小实验解释三个架构共享或差异化的概念。数据来自真实适配器、真实 tokenizer 与真实 logits。
          </p>
        </div>
      </header>

      <ConceptA apiBase={apiBase} />

      <ConceptCard
        index="B"
        title="分词对比"
        explanation="同一句话进入不同 tokenizer 后，会变成完全不同的 token 序列；这会影响上下文长度、显存与生成成本。"
        mapping="Transformer：Qwen tokenizer；RWKV：rwkv_vocab_v20230424；Mamba：GPTNeoX tokenizer。"
      >
        <div className="text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg p-4">
          分词对比模块将在下一阶段接入（只加载 tokenizer，不加载模型权重）。
        </div>
      </ConceptCard>

      <ConceptCard
        index="C"
        title="采样如何改变概率分布"
        explanation="同一组原始 logits 在 Temperature、Top-K、Top-P、重复惩罚下会变成不同的概率分布。"
        mapping="采样层是三个架构共享的后处理步骤；原始 logits 来自各自的适配器 step。"
      >
        <div className="text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg p-4">
          采样分布模块将在下一阶段接入（适配器 step 返回 top-50 原始 logits）。
        </div>
      </ConceptCard>

      <div className="flex items-center gap-2 text-[10px] text-slate-600">
        <BookOpen className="w-3.5 h-3.5" />
        <span>每个概念卡片的最后一行标注它在三种架构中的对应物；不改变现有 Transformer / RWKV / Mamba 页面。</span>
      </div>
    </main>
  );
};
