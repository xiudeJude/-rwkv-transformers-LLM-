import { useState, useEffect, useRef } from 'react';
import {
  Sparkles,
  Layers,
  Cpu,
  RefreshCw,
  Zap,
  Sliders,
  Play,
  Square,
  Activity,
  Gauge,
  Clock,
  AlertTriangle,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { AttentionHeatmap, type IncrementalRowPayload } from './components/Visualizers/AttentionHeatmap';
import { RWKVWaterfall, type RWKVIncrementalRowPayload } from './components/Visualizers/RWKVWaterfall';
import { RWKVStateModal } from './components/Visualizers/RWKVStateModal';
import { TopKBarChart } from './components/Visualizers/TopKBarChart';
import { LayerHeadSelector } from './components/LayerHeadSelector';
import { SmoothRenderPipeline, type TokenTimingMetric, type AnyTokenStepPayload } from './utils/renderPipeline';
import type {
  SingleStepInspectResponse,
  ModelMetadata,
  TokenCandidate,
  RWKVTokenStepPayload,
  RWKVStateResponse,
  SessionAttentionResponse
} from './types/schema';

const API_BASE = 'http://127.0.0.1:8000';

type StreamStatus = 'idle' | 'creating_session' | 'connecting' | 'streaming' | 'finished' | 'stopped' | 'error';

export function App() {
  const [prompt, setPrompt] = useState('注意力机制让大语言模型能够精确捕捉长距离语义依赖关系');
  const [modelArch, setModelArch] = useState<'transformer' | 'rwkv'>('transformer');
  const [currentLayer, setCurrentLayer] = useState(3);
  const [currentHead, setCurrentHead] = useState(0);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(10);
  const [maxNewTokens, setMaxNewTokens] = useState(50);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.2);

  // RWKV waterfall state
  const [rwkvSummaries, setRwkvSummaries] = useState<number[][]>([]);
  const [rwkvDeltas, setRwkvDeltas] = useState<number[][]>([]);
  const [rwkvIncrementalRow, setRwkvIncrementalRow] = useState<RWKVIncrementalRowPayload | null>(null);
  const [rwkvTokens, setRwkvTokens] = useState<string[]>([]);
  const [rwkvModalMatrix, setRwkvModalMatrix] = useState<number[][] | null>(null);
  const [rwkvModalNorm, setRwkvModalNorm] = useState<number | undefined>(undefined);
  const [isRwkvModalOpen, setIsRwkvModalOpen] = useState(false);

  // Model metadata & single-step inspect state
  const [singleLoading, setSingleLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inspectData, setInspectData] = useState<SingleStepInspectResponse | null>(null);
  const [modelMeta, setModelMeta] = useState<ModelMetadata | null>(null);

  // Streaming generation state
  const [streamStatus, setStreamStatus] = useState<StreamStatus>('idle');
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [generatedText, setGeneratedText] = useState('');
  const [tokenList, setTokenList] = useState<{ step: number; token: string }[]>([]);
  const [inspectedStep, setInspectedStep] = useState<number | null>(null);
  const [tokenClickMessage, setTokenClickMessage] = useState<string | null>(null);
  const [streamTopK, setStreamTopK] = useState<TokenCandidate[]>([]);
  const [streamNextToken, setStreamNextToken] = useState<TokenCandidate | null>(null);
  const [queueBacklog, setQueueBacklog] = useState(0);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // Heatmap state
  const [heatmapMatrix, setHeatmapMatrix] = useState<number[][] | null>(null);
  const [heatmapTokens, setHeatmapTokens] = useState<string[]>([]);
  const [incrementalRow, setIncrementalRow] = useState<IncrementalRowPayload | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [heatmapLoadingMessage, setHeatmapLoadingMessage] = useState('');

  // Timing & queue metrics
  const [timingMetrics, setTimingMetrics] = useState<TokenTimingMetric[]>([]);
  const [latestMetric, setLatestMetric] = useState<TokenTimingMetric | null>(null);
  const [showMetricsLog, setShowMetricsLog] = useState(false);

  // References
  const wsRef = useRef<WebSocket | null>(null);
  const pipelineRef = useRef<SmoothRenderPipeline | null>(null);
  const textEndRef = useRef<HTMLDivElement | null>(null);
  const promptTokensRef = useRef<string[]>([]);
  const allTokensRef = useRef<string[]>([]);
  const debounceTimerRef = useRef<number | null>(null);

  // Quick preset prompts
  const presets = [
    '注意力机制让大语言模型能够精确捕捉长距离语义依赖关系',
    'RWKV通过线性状态递推摆脱了传统注意力机制的平方复杂度',
    '人工智能的发展正在从单一模型走向多智能体协同'
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
        if (data.full_attn_layers && data.full_attn_layers.length > 0) {
          setCurrentLayer(data.full_attn_layers[0]);
        }
      })
      .catch((err) => {
        console.warn('Backend not ready yet:', err.message);
      });

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (pipelineRef.current) {
        pipelineRef.current.stop();
      }
    };
  }, []);

  // Auto-scroll generated text container
  useEffect(() => {
    if (isStreaming && textEndRef.current) {
      textEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [generatedText, isStreaming]);

  // Handler: Single-step hook inspect
  const handleRunInspect = async (layer = currentLayer, head = currentHead) => {
    if (!prompt.trim() || isStreaming) return;
    setSingleLoading(true);
    setError(null);
    setActiveSessionId(null);
    setIncrementalRow(null);

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
      setHeatmapMatrix(data.attention.matrix);
      setHeatmapTokens(data.attention.tokens);
      if (data.model_meta) {
        setModelMeta(data.model_meta);
      }
    } catch (err: any) {
      setError(err.message || '网络连接错误，请确认后端已启动');
    } finally {
      setSingleLoading(false);
    }
  };

  // Architecture switch handler
  const handleArchSwitch = (newArch: 'transformer' | 'rwkv') => {
    if (isStreaming) return;
    setModelArch(newArch);
    setActiveSessionId(null);
    setGeneratedText('');
    setTokenList([]);
    setInspectedStep(null);
    setTokenClickMessage(null);
    setTimingMetrics([]);
    setLatestMetric(null);
    setHeatmapMatrix(null);
    setRwkvSummaries([]);
    setRwkvDeltas([]);
    setRwkvIncrementalRow(null);
    setStreamStatus('idle');

    if (newArch === 'transformer') {
      setPrompt('注意力机制让大语言模型能够精确捕捉长距离语义依赖关系');
      fetch(`${API_BASE}/api/model/info?model_type=transformer`)
        .then((res) => res.json())
        .then((data: ModelMetadata) => {
          setModelMeta(data);
          if (data.full_attn_layers && data.full_attn_layers.length > 0) {
            setCurrentLayer(data.full_attn_layers[0]);
          } else {
            setCurrentLayer(3);
          }
          setCurrentHead(0);
        })
        .catch(console.warn);
    } else {
      setPrompt('RWKV通过线性状态递推摆脱了传统注意力机制的平方复杂度');
      fetch(`${API_BASE}/api/model/info?model_type=rwkv`)
        .then((res) => res.json())
        .then((data: ModelMetadata) => {
          setModelMeta(data);
          setCurrentLayer(0);
          setCurrentHead(0);
        })
        .catch(console.warn);
    }
  };

  // Handler: Start WebSocket Streaming Generation
  const handleStartStreaming = async () => {
    if (!prompt.trim() || isStreaming) return;
    setError(null);
    setGeneratedText('');
    setTokenList([]);
    setInspectedStep(null);
    setTokenClickMessage(null);
    setCurrentStep(0);
    setStreamTopK([]);
    setStreamNextToken(null);
    setTimingMetrics([]);
    setLatestMetric(null);
    setQueueBacklog(0);
    setIncrementalRow(null);
    setRwkvIncrementalRow(null);
    setStreamStatus('creating_session');

    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ action: 'close' }));
        wsRef.current.close();
      } catch (e) {}
      wsRef.current = null;
    }

    try {
      // 1. Create session via synchronous HTTP prefill
      const createEndpoint = modelArch === 'transformer' ? '/api/session/create' : '/api/rwkv/session/create';
      const createRes = await fetch(`${API_BASE}${createEndpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          layer: currentLayer,
          head: currentHead,
        }),
      });

      if (!createRes.ok) {
        const errJson = await createRes.json().catch(() => ({}));
        throw new Error(errJson.detail || `会话创建失败 (${createRes.status})`);
      }

      const sessionData = await createRes.json();
      const sessionId = sessionData.session_id;
      setActiveSessionId(sessionId);

      // Initialize Visualization buffers with Prefill data
      if (modelArch === 'transformer' && sessionData.prefill_attention) {
        promptTokensRef.current = sessionData.prefill_attention.tokens;
        allTokensRef.current = [...sessionData.prefill_attention.tokens];
        setHeatmapMatrix(sessionData.prefill_attention.matrix);
        setHeatmapTokens([...sessionData.prefill_attention.tokens]);
      } else if (modelArch === 'rwkv' && sessionData.prefill_state) {
        const pState = sessionData.prefill_state;
        promptTokensRef.current = pState.tokens;
        allTokensRef.current = [...pState.tokens];
        setRwkvTokens([...pState.tokens]);
        setRwkvSummaries([pState.state_summary]);
        setRwkvDeltas([pState.delta]);
      }

      setStreamStatus('connecting');

      // 2. Initialize SmoothRenderPipeline
      if (pipelineRef.current) {
        pipelineRef.current.clear();
      }

      const pipeline = new SmoothRenderPipeline({
        onRenderToken: (item, metric) => {
          // rAF consumed a token
          setGeneratedText((prev) => prev + item.token);
          setTokenList((prev) => [...prev, { step: item.step, token: item.token }]);
          setCurrentStep(item.step);
          setStreamNextToken({
            token: item.token,
            prob: item.token_prob,
            id: item.token_id,
          });
          setStreamTopK(item.topk_candidates);
          setLatestMetric(metric);
          setQueueBacklog(pipeline.getQueueLength());

          // Keep full sequence tokens in sync for axes
          allTokensRef.current.push(item.token);

          if (modelArch === 'transformer') {
            setHeatmapTokens([...allTokensRef.current]);
            // Incremental Canvas row append:
            if (item.attention_row && item.attention_row.length > 0) {
              setIncrementalRow({
                rowIndex: promptTokensRef.current.length + item.step - 1,
                token: item.token,
                weights: item.attention_row,
              });
            }
          } else {
            const rwkvItem = item as RWKVTokenStepPayload;
            setRwkvTokens([...allTokensRef.current]);
            if (rwkvItem.state_summary && rwkvItem.delta) {
              setRwkvIncrementalRow({
                step: rwkvItem.step,
                token: rwkvItem.token,
                state_summary: rwkvItem.state_summary,
                delta: rwkvItem.delta,
              });
            }
          }
        },
        onMetricsUpdate: (allMetrics) => {
          setTimingMetrics([...allMetrics]);
        },
        onFinish: () => {
          setStreamStatus('finished');
          setIsStreaming(false);
          setQueueBacklog(0);
        },
      });
      pipelineRef.current = pipeline;

      // 3. Connect WebSocket
      const wsEndpoint =
        modelArch === 'transformer'
          ? `/ws/session/${sessionId}/stream`
          : `/ws/rwkv/session/${sessionId}/stream`;
      const wsUrl = `ws://127.0.0.1:8000${wsEndpoint}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStreamStatus('streaming');
        setIsStreaming(true);
        // Send start action with repetition_penalty
        ws.send(
          JSON.stringify({
            action: 'start',
            temperature,
            top_k: topK,
            repetition_penalty: repetitionPenalty,
            max_new_tokens: maxNewTokens,
            layer: currentLayer,
            head: currentHead,
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const data: AnyTokenStepPayload = JSON.parse(event.data);
          if (data.type === 'error') {
            setError(data.message || '推理服务返回错误');
            setStreamStatus('error');
            setIsStreaming(false);
            pipeline.stop();
            return;
          }

          // STRICT: onmessage ONLY enqueues to pipeline, does NOT trigger setState or Canvas draw directly!
          pipeline.push(data);
          setQueueBacklog(pipeline.getQueueLength());
        } catch (e: any) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = (err) => {
        console.error('WebSocket encountered an error:', err);
        setError('WebSocket 连接发生异常');
        setStreamStatus('error');
        setIsStreaming(false);
      };

      ws.onclose = () => {
        console.log(`[WebSocket] Closed for session ${sessionId}`);
      };
    } catch (err: any) {
      setError(err.message || '无法连接推理服务');
      setStreamStatus('error');
      setIsStreaming(false);
    }
  };

  // Handler: Stop Streaming
  const handleStopStreaming = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'close' }));
      wsRef.current.close();
    }
    if (pipelineRef.current) {
      pipelineRef.current.stop();
    }
    setIsStreaming(false);
    setStreamStatus('stopped');
  };

  // Level 2 Backfill for Transformer when user switches Layer/Head after generation has finished
  const handleRefetchSessionAttention = async (sessId: string, targetLayer: number, targetHead: number) => {
    setHeatmapLoading(true);
    setHeatmapLoadingMessage(`正在回填 Layer ${targetLayer} / Head ${targetHead} 全量注意力矩阵...`);
    setError(null);

    try {
      const res = await fetch(
        `${API_BASE}/api/session/${sessId}/attention?layer=${targetLayer}&head=${targetHead}`
      );
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `回填失败 (${res.status})`);
      }

      const data: SessionAttentionResponse = await res.json();
      setHeatmapMatrix(data.full_matrix);
      setHeatmapTokens(data.tokens);
      setIncrementalRow(null); // Switch to full redraw mode with new matrix
    } catch (err: any) {
      setError(err.message || '回填注意力数据失败');
    } finally {
      setHeatmapLoading(false);
    }
  };

  // Level 2 Backfill for RWKV when user switches Layer after generation has finished
  const handleRefetchRwkvLayer = async (sessId: string, targetLayer: number) => {
    setHeatmapLoading(true);
    setHeatmapLoadingMessage(`正在回填 RWKV Layer ${targetLayer} 历史状态数据...`);
    setError(null);

    try {
      const totalSteps = tokenList.length; // step 0 is prefill, 1..N
      const stepPromises = [];
      for (let s = 0; s <= totalSteps; s++) {
        stepPromises.push(
          fetch(`${API_BASE}/api/session/${sessId}/rwkv_state?step=${s}&layer=${targetLayer}`).then(
            (res) => {
              if (!res.ok) throw new Error(`回填 Step ${s} 失败`);
              return res.json();
            }
          )
        );
      }
      const results = await Promise.all(stepPromises);
      const newSummaries: number[][] = [];
      const newDeltas: number[][] = [];
      for (let s = 0; s < results.length; s++) {
        const norms: number[] = results[s].head_norms || [];
        newSummaries.push(norms);
        if (s === 0) {
          newDeltas.push(new Array(16).fill(0));
        } else {
          const prev = newSummaries[s - 1];
          newDeltas.push(norms.map((v, idx) => Math.round((v - prev[idx]) * 1e6) / 1e6));
        }
      }
      setRwkvSummaries(newSummaries);
      setRwkvDeltas(newDeltas);
      setRwkvIncrementalRow(null);
    } catch (err: any) {
      setError(err.message || '回填 RWKV 状态数据失败');
    } finally {
      setHeatmapLoading(false);
    }
  };

  const handleLayerChange = async (newLayer: number) => {
    setCurrentLayer(newLayer);
    // If active session exists and generation is finished/stopped, fetch Level 2 data to backfill
    if (activeSessionId && (streamStatus === 'finished' || streamStatus === 'stopped')) {
      if (modelArch === 'transformer') {
        await handleRefetchSessionAttention(activeSessionId, newLayer, currentHead);
      } else {
        await handleRefetchRwkvLayer(activeSessionId, newLayer);
      }
    } else if (inspectData && !isStreaming && modelArch === 'transformer') {
      handleRunInspect(newLayer, currentHead);
    }
  };

  const handleHeadChange = async (newHead: number) => {
    setCurrentHead(newHead);
    if (activeSessionId && (streamStatus === 'finished' || streamStatus === 'stopped')) {
      if (modelArch === 'transformer') {
        await handleRefetchSessionAttention(activeSessionId, currentLayer, newHead);
      }
    } else if (inspectData && !isStreaming && modelArch === 'transformer') {
      handleRunInspect(currentLayer, newHead);
    }
  };

  // Token Chip Click Inspector: fetch single-step Level 2 attention detail with debounce
  const handleTokenClick = (stepNum: number) => {
    if (isStreaming) {
      setTokenClickMessage(
        `流式生成进行中 (当前第 ${currentStep} 步)。已为您聚焦该 Token；生成完成后可点击任意 Token 深度回溯状态全景。`
      );
      setTimeout(() => setTokenClickMessage(null), 4000);
      return;
    }

    if (!activeSessionId) return;

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(async () => {
      setInspectedStep(stepNum);

      if (modelArch === 'rwkv') {
        setHeatmapLoading(true);
        setHeatmapLoadingMessage(`正在提取 Step ${stepNum} 隐状态详细矩阵...`);
        try {
          const res = await fetch(
            `${API_BASE}/api/session/${activeSessionId}/rwkv_state?step=${stepNum}&layer=${currentLayer}&head=${currentHead}`
          );
          if (!res.ok) throw new Error(`拉取失败 (${res.status})`);
          const data: RWKVStateResponse = await res.json();
          setRwkvModalMatrix(data.matrix as number[][]);
          setRwkvModalNorm(data.head_norms?.[currentHead]);
          setIsRwkvModalOpen(true);
        } catch (err: any) {
          setError(err.message || '拉取 RWKV 状态矩阵失败');
        } finally {
          setHeatmapLoading(false);
        }
        return;
      }

      // Transformer branch
      setHeatmapLoading(true);
      setHeatmapLoadingMessage(`正在拉取 Step ${stepNum} 的注意力全景快照...`);
      try {
        const res = await fetch(
          `${API_BASE}/api/session/${activeSessionId}/attention?step=${stepNum}&layer=${currentLayer}&head=${currentHead}`
        );
        if (!res.ok) {
          const errJson = await res.json().catch(() => ({}));
          throw new Error(errJson.detail || `拉取失败 (${res.status})`);
        }
        const data: SessionAttentionResponse = await res.json();
        setHeatmapMatrix(data.full_matrix);
        setHeatmapTokens(data.tokens);
        setIncrementalRow(null);
      } catch (err: any) {
        setError(err.message || '拉取步注意力数据失败');
      } finally {
        setHeatmapLoading(false);
      }
    }, 150);
  };

  // Computed timing statistics
  const avgArrivalDelta =
    timingMetrics.length > 1
      ? Math.round(
          (timingMetrics.slice(1).reduce((sum, m) => sum + m.arrivalDeltaMs, 0) /
            (timingMetrics.length - 1)) *
            10
        ) / 10
      : 0;
  const avgRenderDelta =
    timingMetrics.length > 1
      ? Math.round(
          (timingMetrics.slice(1).reduce((sum, m) => sum + m.renderDeltaMs, 0) /
            (timingMetrics.length - 1)) *
            10
        ) / 10
      : 0;
  const maxObservedBacklog = timingMetrics.reduce((max, m) => Math.max(max, m.queueBacklog), 0);

  // Status badge config
  const statusBadges: Record<
    StreamStatus,
    { label: string; bg: string; text: string; border: string; pulse?: boolean }
  > = {
    idle: { label: '空闲就绪', bg: 'bg-slate-800', text: 'text-slate-300', border: 'border-slate-700' },
    creating_session: {
      label: 'Prefill 会话初始化中...',
      bg: 'bg-indigo-950',
      text: 'text-indigo-300',
      border: 'border-indigo-800',
      pulse: true,
    },
    connecting: {
      label: 'WebSocket 连接握手中...',
      bg: 'bg-sky-950',
      text: 'text-sky-300',
      border: 'border-sky-800',
      pulse: true,
    },
    streaming: {
      label: '自回归流式生成中',
      bg: 'bg-emerald-950',
      text: 'text-emerald-300',
      border: 'border-emerald-700',
      pulse: true,
    },
    finished: { label: '流式生成已完成', bg: 'bg-blue-950', text: 'text-blue-300', border: 'border-blue-800' },
    stopped: { label: '已手动中止', bg: 'bg-amber-950', text: 'text-amber-300', border: 'border-amber-800' },
    error: { label: '生成异常中断', bg: 'bg-rose-950', text: 'text-rose-300', border: 'border-rose-800' },
  };
  const activeBadge = statusBadges[streamStatus];

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
                P1: 双架构对比 (Transformer vs RWKV)
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">
              {modelArch === 'transformer'
                ? 'Full Attention 离屏双缓冲增量绘制 • 60 FPS 平滑渲染 • Log-scale 色谱'
                : 'RWKV-7 隐藏状态瀑布流 • 能量模长演化(Viridis)与单步扰动(Coolwarm) • Level 2 状态探针'}
            </p>
          </div>
        </div>

        {/* Architecture Switcher Tabs */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 shadow-inner">
          <button
            onClick={() => handleArchSwitch('transformer')}
            disabled={isStreaming}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all cursor-pointer ${
              modelArch === 'transformer'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Transformer (Qwen3.5)</span>
          </button>
          <button
            onClick={() => handleArchSwitch('rwkv')}
            disabled={isStreaming}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all cursor-pointer ${
              modelArch === 'rwkv'
                ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-md shadow-emerald-600/30'
                : 'text-slate-400 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>RWKV-7 (Goose 0.4B)</span>
          </button>
        </div>

        {/* Model Meta Badge */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-slate-400">当前引擎:</span>
            <span className="font-mono text-sky-300 font-semibold">
              {modelMeta?.model_id || (modelArch === 'transformer' ? 'Qwen3.5-0.8B (GGUF)' : 'RWKV7-0.4B (FP16)')}
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
        {/* Prompt Input & Control Card */}
        <section className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-5 shadow-xl space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <label className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              <span>输入推理 Prompt</span>
            </label>

            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>快捷预设:</span>
              {presets.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => setPrompt(preset)}
                  disabled={isStreaming}
                  className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors truncate max-w-[140px] disabled:opacity-50"
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
              disabled={isStreaming}
              placeholder="请输入一段文本，支持单步 Forward Hook 抓取与自回归增量流式生成..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors resize-none font-sans disabled:opacity-60"
            />

            {/* Action Buttons */}
            <div className="flex flex-col gap-2 shrink-0">
              {!isStreaming ? (
                <button
                  onClick={handleStartStreaming}
                  disabled={!prompt.trim() || singleLoading}
                  className="px-5 py-2 rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 font-semibold text-white text-sm shadow-lg shadow-emerald-600/25 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  <Play className="w-4 h-4 fill-white" />
                  <span>流式生成 (WS)</span>
                </button>
              ) : (
                <button
                  onClick={handleStopStreaming}
                  className="px-5 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 font-semibold text-white text-sm shadow-lg shadow-rose-600/25 flex items-center justify-center gap-2 transition-all animate-pulse"
                >
                  <Square className="w-4 h-4 fill-white" />
                  <span>停止生成</span>
                </button>
              )}

              {modelArch === 'transformer' && (
                <button
                  onClick={() => handleRunInspect(currentLayer, currentHead)}
                  disabled={singleLoading || isStreaming || !prompt.trim()}
                  className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 font-medium text-slate-300 text-xs border border-slate-700 flex items-center justify-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {singleLoading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Hooking...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-3.5 h-3.5 text-amber-400" />
                      <span>单步 Hook 分析</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Quick max_new_tokens slider */}
          <div className="flex items-center gap-4 text-xs border-t border-slate-800/80 pt-3">
            <span className="text-slate-400">最大生成 Token 数 (max_new_tokens):</span>
            <input
              type="range"
              min={5}
              max={150}
              step={5}
              value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
              disabled={isStreaming}
              className="w-48 accent-emerald-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
            />
            <span className="font-mono text-emerald-400 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              {maxNewTokens} tokens
            </span>
          </div>

          {error && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-xs text-rose-300 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* AI Response Status Banner */}
        <section className="bg-slate-900/80 rounded-xl border border-slate-800 p-4 shadow-lg flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div
              className={`px-3 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-2 ${activeBadge.bg} ${activeBadge.text} ${activeBadge.border}`}
            >
              <div
                className={`w-2 h-2 rounded-full ${activeBadge.pulse ? 'bg-current animate-ping' : 'bg-current'}`}
              />
              <span>{activeBadge.label}</span>
            </div>

            {/* Token Progress Counter */}
            <div className="text-xs flex items-center gap-1.5 px-3 py-1.5 bg-slate-950 rounded-lg border border-slate-800">
              <Activity className="w-3.5 h-3.5 text-sky-400" />
              <span className="text-slate-400">进度:</span>
              <span className="font-mono font-bold text-sky-300">
                {currentStep} / {maxNewTokens}
              </span>
              <span className="text-slate-500 text-[11px]">tokens</span>
            </div>

            {/* Queue Backlog Indicator */}
            <div className="text-xs flex items-center gap-1.5 px-3 py-1.5 bg-slate-950 rounded-lg border border-slate-800">
              <Gauge className="w-3.5 h-3.5 text-indigo-400" />
              <span className="text-slate-400">队列积压:</span>
              <span
                className={`font-mono font-bold ${
                  queueBacklog > 5 ? 'text-amber-400 animate-pulse' : 'text-slate-200'
                }`}
              >
                {queueBacklog}
              </span>
              <span className="text-slate-500 text-[11px]">tokens</span>
            </div>

            {/* Adaptive Pace Mode Badge */}
            <div
              className={`text-[11px] font-mono px-2.5 py-1 rounded-md border ${
                queueBacklog > 5
                  ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              {queueBacklog > 5 ? '⚡ 追帧调速中 (15ms)' : '平稳调速 (~40ms)'}
            </div>
          </div>

          {/* Timing Stats in Banner */}
          <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
            {latestMetric && (
              <>
                <div className="flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-emerald-400" />
                  <span>到帧: {latestMetric.arrivalDeltaMs}ms</span>
                </div>
                <div className="flex items-center gap-1">
                  <Activity className="w-3.5 h-3.5 text-sky-400" />
                  <span>渲染: {latestMetric.renderDeltaMs}ms</span>
                </div>
              </>
            )}
            <button
              onClick={() => setShowMetricsLog(!showMetricsLog)}
              className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 bg-indigo-500/10 px-2 py-1 rounded border border-indigo-500/20"
            >
              <span>{showMetricsLog ? '收起测速详情' : '展开测速详情'}</span>
              {showMetricsLog ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>
        </section>

        {/* Real-time Metrics Inspector Panel (Collapsible) */}
        {showMetricsLog && (
          <section className="bg-slate-900/90 rounded-xl border border-slate-800 p-4 space-y-3 shadow-xl">
            <div className="flex items-center justify-between text-xs text-slate-300 border-b border-slate-800 pb-2">
              <span className="font-semibold flex items-center gap-1.5">
                <Gauge className="w-4 h-4 text-sky-400" />
                <span>流式渲染队列测速指标 (SmoothRenderPipeline Telemetry)</span>
              </span>
              <div className="flex items-center gap-4 font-mono text-[11px]">
                <span>
                  平均到达间隔: <b className="text-emerald-400">{avgArrivalDelta}ms</b>
                </span>
                <span>
                  平均消费间隔: <b className="text-sky-400">{avgRenderDelta}ms</b>
                </span>
                <span>
                  峰值队列积压: <b className="text-amber-400">{maxObservedBacklog}</b>
                </span>
              </div>
            </div>

            {/* Timing Table */}
            <div className="max-h-48 overflow-y-auto font-mono text-[11px] border border-slate-800/80 rounded bg-slate-950 p-2">
              {timingMetrics.length === 0 ? (
                <div className="text-slate-600 text-center py-4">
                  暂无测速记录，启动流式生成后将记录每步 token 时间间隔...
                </div>
              ) : (
                <table className="w-full text-left">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-800 text-[10px]">
                      <th className="py-1 px-2">Step</th>
                      <th className="py-1 px-2">Token</th>
                      <th className="py-1 px-2">到达间隔 (Δt Arrive)</th>
                      <th className="py-1 px-2">渲染间隔 (Δt Render)</th>
                      <th className="py-1 px-2">消费时积压</th>
                      <th className="py-1 px-2">生效调度策略</th>
                    </tr>
                  </thead>
                  <tbody>
                    {timingMetrics.slice(-20).map((m) => (
                      <tr key={m.step} className="border-b border-slate-900/60 hover:bg-slate-900/50">
                        <td className="py-0.5 px-2 text-slate-400">{m.step}</td>
                        <td className="py-0.5 px-2 text-sky-300 font-bold">
                          {m.token.replace(/\n/g, '\\n').replace(/ /g, '␣')}
                        </td>
                        <td className="py-0.5 px-2 text-emerald-400">{m.arrivalDeltaMs} ms</td>
                        <td className="py-0.5 px-2 text-sky-400">{m.renderDeltaMs} ms</td>
                        <td className="py-0.5 px-2">
                          <span
                            className={`px-1 rounded ${
                              m.queueBacklog > 5
                                ? 'bg-amber-500/20 text-amber-300 font-bold'
                                : 'text-slate-400'
                            }`}
                          >
                            {m.queueBacklog}
                          </span>
                        </td>
                        <td className="py-0.5 px-2 text-slate-400">
                          {m.renderIntervalUsed === 15 ? (
                            <span className="text-amber-400 font-semibold">⚡ 加速 15ms</span>
                          ) : (
                            <span className="text-slate-400">常规 40ms</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </section>
        )}

        {/* Live Streaming Generated Text Display */}
        <section className="bg-slate-900/60 rounded-xl border border-slate-800/80 p-5 shadow-xl space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-800 pb-2">
            <span className="font-semibold text-slate-200 flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span>自回归增量生成文本 (Live Token Stream):</span>
            </span>
            <span className="font-mono text-[11px] text-slate-500">已生成 {currentStep} tokens</span>
          </div>

          <div className="bg-slate-950 rounded-lg p-4 font-mono text-sm leading-relaxed border border-slate-800 min-h-[90px] max-h-56 overflow-y-auto">
            <span className="text-slate-500">{prompt}</span>
            {tokenList.length > 0 ? (
              tokenList.map((t) => (
                <span
                  key={t.step}
                  onClick={() => handleTokenClick(t.step)}
                  className={`cursor-pointer px-0.5 rounded transition-all inline-block ${
                    inspectedStep === t.step
                      ? 'bg-sky-500/30 text-sky-200 ring-1 ring-sky-400 font-bold'
                      : 'hover:bg-slate-800 hover:text-emerald-200 text-emerald-300 font-semibold'
                  }`}
                  title={
                    modelArch === 'transformer'
                      ? `点击查看 Step ${t.step} [${t.token.replace(/\n/g, '\\n')}] 注意力矩阵快照`
                      : `点击下钻查看 Step ${t.step} [${t.token.replace(/\n/g, '\\n')}] 64x64 隐状态矩阵`
                  }
                >
                  {t.token}
                </span>
              ))
            ) : (
              <span className="text-emerald-300 font-semibold ml-1">{generatedText}</span>
            )}
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-emerald-400 ml-1 animate-pulse align-middle" />
            )}
            {!generatedText && !isStreaming && (
              <span className="text-slate-600 text-xs italic ml-2">
                {modelArch === 'transformer'
                  ? '(点击"流式生成 (WS)"体验自回归增量生成与双层 Canvas 动态热力图...)'
                  : '(点击"流式生成 (WS)"体验 RWKV-7 隐藏状态模长与差分时序瀑布流...)'}
              </span>
            )}
            <div ref={textEndRef} />
          </div>

          {tokenClickMessage && (
            <div className="p-2 bg-indigo-500/10 border border-indigo-500/30 rounded text-xs text-indigo-300 animate-fade-in flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
              <span>{tokenClickMessage}</span>
            </div>
          )}
        </section>

        {/* Two Columns: Visualizer and Controls */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left 2 Cols: Dual-Canvas Incremental Heatmap or RWKV Dual-Ribbon Waterfall */}
          <div className="lg:col-span-2 space-y-4">
            {modelArch === 'transformer' ? (
              heatmapMatrix ? (
                <AttentionHeatmap
                  matrix={heatmapMatrix}
                  tokens={heatmapTokens}
                  layer={currentLayer}
                  head={currentHead}
                  incrementalRow={incrementalRow}
                  isStreaming={isStreaming}
                  isLoading={heatmapLoading}
                  loadingMessage={heatmapLoadingMessage}
                />
              ) : (
                <div className="h-96 rounded-xl border border-dashed border-slate-800 flex flex-col items-center justify-center text-slate-500 space-y-3 p-6 text-center">
                  <div className="w-12 h-12 rounded-full bg-slate-900 flex items-center justify-center text-slate-600 border border-slate-800">
                    <Layers className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-sm text-slate-300 font-medium">Attention 热力图就绪</p>
                    <p className="text-xs text-slate-500 mt-1">
                      点击"流式生成 (WS)"或"单步 Hook 分析"，Canvas 将实时展现自回归注意力矩阵！
                    </p>
                  </div>
                </div>
              )
            ) : (
              /* RWKV Mode: Waterfall or Empty State */
              rwkvSummaries.length > 0 || rwkvIncrementalRow || isStreaming ? (
                <RWKVWaterfall
                  layer={currentLayer}
                  head={currentHead}
                  tokens={rwkvTokens}
                  initialSummaries={rwkvSummaries}
                  initialDeltas={rwkvDeltas}
                  incrementalRow={rwkvIncrementalRow}
                  isStreaming={isStreaming}
                  isLoading={heatmapLoading}
                  loadingMessage={heatmapLoadingMessage}
                  onStepClick={handleTokenClick}
                  inspectedStep={inspectedStep}
                />
              ) : (
                <div className="h-96 rounded-xl border border-dashed border-slate-800 flex flex-col items-center justify-center text-slate-500 space-y-3 p-6 text-center">
                  <div className="w-12 h-12 rounded-full bg-slate-900 flex items-center justify-center text-emerald-400/80 border border-slate-800">
                    <Activity className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-sm text-slate-300 font-medium">RWKV-7 状态瀑布流就绪</p>
                    <p className="text-xs text-slate-500 mt-1">
                      点击"流式生成 (WS)"，Canvas 将实时展现 16 头隐藏状态能量演化与单步差分！
                    </p>
                  </div>
                </div>
              )
            )}
          </div>

          {/* Right 1 Col: Selector & Real-Time Top-K Bar Chart */}
          <div className="space-y-4">
            <LayerHeadSelector
              numLayers={modelMeta?.num_layers || 24}
              numHeads={modelArch === 'rwkv' ? 16 : (modelMeta?.num_heads || 8)}
              currentLayer={currentLayer}
              currentHead={currentHead}
              onLayerChange={handleLayerChange}
              onHeadChange={handleHeadChange}
              temperature={temperature}
              onTemperatureChange={setTemperature}
              topK={topK}
              onTopKChange={setTopK}
              repetitionPenalty={repetitionPenalty}
              onRepetitionPenaltyChange={setRepetitionPenalty}
              fullAttnLayers={modelArch === 'transformer' ? modelMeta?.full_attn_layers : undefined}
              disabled={isStreaming || singleLoading}
              isStreaming={isStreaming}
              modelType={modelArch}
            />

            {/* Show Top-K Candidates from either live stream or single step */}
            {streamTopK.length > 0 && streamNextToken ? (
              <TopKBarChart candidates={streamTopK} nextToken={streamNextToken} />
            ) : inspectData ? (
              <TopKBarChart candidates={inspectData.top_k_candidates} nextToken={inspectData.next_token} />
            ) : null}
          </div>
        </section>
      </main>

      {/* RWKV Level 2 State Inspection Modal */}
      {isRwkvModalOpen && rwkvModalMatrix && (
        <RWKVStateModal
          isOpen={isRwkvModalOpen}
          onClose={() => setIsRwkvModalOpen(false)}
          step={inspectedStep ?? 0}
          layer={currentLayer}
          head={currentHead}
          token={tokenList.find((t) => t.step === inspectedStep)?.token}
          matrix={rwkvModalMatrix}
          headNorm={rwkvModalNorm}
        />
      )}

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-900/40 py-3 text-center text-xs text-slate-500">
        传智杯 Vibe Coding 挑战赛 • LLM 可解释性交互工坊 • 模块解耦架构
      </footer>
    </div>
  );
}

export default App;
