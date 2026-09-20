import { useState, useEffect } from 'react';
import {
  Sparkles,
  Layers,
  Cpu,
  RefreshCw,
  Zap,
  Info,
  Sliders,
  ChevronRight
} from 'lucide-react';
import { AttentionHeatmap } from './components/Visualizers/AttentionHeatmap';
import { TopKBarChart } from './components/Visualizers/TopKBarChart';
import { LayerHeadSelector } from './components/LayerHeadSelector';
import type { SingleStepInspectResponse, ModelMetadata } from './types/schema';

const API_BASE = 'http://127.0.0.1:8000';

export function App() {
  const [prompt, setPrompt] = useState('注意力机制让大语言模型能够精确捕捉长距离语义依赖关系');
  const [currentLayer, setCurrentLayer] = useState(0);
  const [currentHead, setCurrentHead] = useState(0);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(10);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inspectData, setInspectData] = useState<SingleStepInspectResponse | null>(null);
  const [modelMeta, setModelMeta] = useState<ModelMetadata | null>(null);

  // Quick preset prompts
  const presets = [
    '注意力机制让大语言模型能够精确捕捉长距离语义依赖关系',
    'RWKV通过线性状态递推摆脱了传统注意力机制的平方复杂度',
    'The quick brown fox jumps over the lazy dog in deep forest'
  ];

  // Fetch model metadata on initial load
  useEffect(() => {
    fetch(`${API_BASE}/api/model/info`)
      .then((res) => {
        if (!res.ok) throw new Error(`后端未就绪 (${res.status})`);
        return res.json();
      })
      .then((data: ModelMetadata) => {
        setModelMeta(data);
      })
      .catch((err) => {
        console.warn('Backend not ready yet:', err.message);
      });
  }, []);

  const handleRunInspect = async (layer = currentLayer, head = currentHead) => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/inspect/single-step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          layer,
          head,
          top_k: topK,
          temperature,
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `请求失败 (${res.status})`);
      }

      const data: SingleStepInspectResponse = await res.json();
      setInspectData(data);
      if (data.model_meta) {
        setModelMeta(data.model_meta);
      }
    } catch (err: any) {
      setError(err.message || '网络连接错误，请确认后端已启动');
    } finally {
      setLoading(false);
    }
  };

  const handleLayerChange = (newLayer: number) => {
    setCurrentLayer(newLayer);
    if (inspectData) {
      handleRunInspect(newLayer, currentHead);
    }
  };

  const handleHeadChange = (newHead: number) => {
    setCurrentHead(newHead);
    if (inspectData) {
      handleRunInspect(currentLayer, newHead);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-40 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-indigo-500 via-purple-500 to-sky-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-white flex items-center gap-2">
              LLM 可解释性交互工坊
              <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                P0 最小闭环骨架
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">
              基于 forward hook 实时抓取 Transformer 注意力分布与 RWKV 状态机
            </p>
          </div>
        </div>

        {/* Model Meta Badge */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-slate-400">当前引擎:</span>
            <span className="font-mono text-sky-300 font-semibold">
              {modelMeta?.model_id || 'Qwen2.5-0.5B'}
            </span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
            <div className={`w-2 h-2 rounded-full ${modelMeta ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
            <span className="text-slate-300 font-mono text-[11px]">
              {modelMeta ? `${modelMeta.device.toUpperCase()}` : '等待后端...'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* Prompt Input Card */}
        <section className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-5 shadow-xl space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              <span>输入测试 Prompt</span>
            </label>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>快捷预设:</span>
              {presets.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => setPrompt(preset)}
                  className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors truncate max-w-[140px]"
                  title={preset}
                >
                  预设 {idx + 1}
                </button>
              ))}
            </div>
          </div>

          <div className="flex gap-3">
            <textarea
              rows={2}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="请输入一段文本，模型将进行 Forward 并抓取各层 Attention 权重..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors resize-none font-sans"
            />
            <button
              onClick={() => handleRunInspect(currentLayer, currentHead)}
              disabled={loading || !prompt.trim()}
              className="px-6 rounded-lg bg-gradient-to-r from-indigo-600 to-sky-600 hover:from-indigo-500 hover:to-sky-500 font-semibold text-white shadow-lg shadow-indigo-600/25 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Hooking...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>开始分析</span>
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-xs text-rose-300 flex items-center gap-2">
              <Info className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* Token Chips Sequence (if inspected) */}
        {inspectData && (
          <section className="bg-slate-900/40 rounded-xl border border-slate-800 p-4">
            <div className="text-xs text-slate-400 mb-2 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              <span>Token 分词序列 ({inspectData.attention.tokens.length} tokens):</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {inspectData.attention.tokens.map((tok, idx) => (
                <div
                  key={idx}
                  className="px-2 py-1 rounded bg-slate-800/90 border border-slate-700/80 text-xs font-mono flex items-center gap-1.5 group hover:border-sky-500/50 transition-colors"
                >
                  <span className="text-slate-500 text-[10px]">{idx}</span>
                  <span className="text-sky-200 font-semibold">
                    {tok.replace(/\n/g, '\\n').replace(/ /g, '␣')}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    #{inspectData.attention.token_ids[idx]}
                  </span>
                </div>
              ))}
              <div className="flex items-center text-slate-500 text-xs px-1">
                <ChevronRight className="w-4 h-4" />
              </div>
              <div className="px-2 py-1 rounded bg-emerald-500/20 border border-emerald-500/40 text-xs font-mono flex items-center gap-1.5 shadow-sm">
                <span className="text-emerald-400 text-[10px]">NEXT</span>
                <span className="text-emerald-300 font-bold">
                  {inspectData.next_token.token.replace(/\n/g, '\\n').replace(/ /g, '␣')}
                </span>
                <span className="text-[10px] text-emerald-400">
                  {(inspectData.next_token.prob * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </section>
        )}

        {/* Two Columns: Visualizer and Controls */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left 2 Cols: Attention Heatmap */}
          <div className="lg:col-span-2 space-y-4">
            {inspectData ? (
              <AttentionHeatmap
                matrix={inspectData.attention.matrix}
                tokens={inspectData.attention.tokens}
                layer={inspectData.attention.layer}
                head={inspectData.attention.head}
              />
            ) : (
              <div className="h-96 rounded-xl border border-dashed border-slate-800 flex flex-col items-center justify-center text-slate-500 space-y-3 p-6 text-center">
                <div className="w-12 h-12 rounded-full bg-slate-900 flex items-center justify-center text-slate-600 border border-slate-800">
                  <Layers className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm text-slate-300 font-medium">尚未获取 Attention 数据</p>
                  <p className="text-xs text-slate-500 mt-1">
                    点击上方"开始分析"按钮，后端将通过 PyTorch Forward Hook 实时拦截提取指定层与注意头的权重矩阵
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Right 1 Col: Selector & Top-K Bar Chart */}
          <div className="space-y-4">
            <LayerHeadSelector
              numLayers={modelMeta?.num_layers || 24}
              numHeads={modelMeta?.num_heads || 14}
              currentLayer={currentLayer}
              currentHead={currentHead}
              onLayerChange={handleLayerChange}
              onHeadChange={handleHeadChange}
              temperature={temperature}
              onTemperatureChange={setTemperature}
              topK={topK}
              onTopKChange={setTopK}
              disabled={loading}
            />

            {inspectData && (
              <TopKBarChart
                candidates={inspectData.top_k_candidates}
                nextToken={inspectData.next_token}
              />
            )}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-900/40 py-3 text-center text-xs text-slate-500">
        传智杯 Vibe Coding 挑战赛 • LLM 可解释性交互工坊 • 模块解耦架构
      </footer>
    </div>
  );
}

export default App;
