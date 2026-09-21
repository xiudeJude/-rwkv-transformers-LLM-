import type { TokenStepPayload, RWKVTokenStepPayload } from '../types/schema';

export type AnyTokenStepPayload = TokenStepPayload | RWKVTokenStepPayload;

export interface TokenTimingMetric {
  step: number;
  token: string;
  arrivalDeltaMs: number;
  renderDeltaMs: number;
  queueBacklog: number;
  renderIntervalUsed: number;
  timestamp: number;
}

export interface PipelineCallbacks {
  onRenderToken: (item: AnyTokenStepPayload, metrics: TokenTimingMetric) => void;
  onFinish?: () => void;
  onMetricsUpdate?: (allMetrics: TokenTimingMetric[]) => void;
}

export class SmoothRenderPipeline {
  private queue: { item: AnyTokenStepPayload; arrivedAt: number }[] = [];
  private isRunning: boolean = false;
  private rafId: number | null = null;
  private lastRenderTime: number = 0;
  private lastArrivalTime: number = 0;
  private metrics: TokenTimingMetric[] = [];
  private callbacks: PipelineCallbacks;

  // Pace configurations
  public normalIntervalMs: number = 40;
  public fastIntervalMs: number = 15;
  public backlogThreshold: number = 5;

  constructor(callbacks: PipelineCallbacks) {
    this.callbacks = callbacks;
  }

  public push(item: AnyTokenStepPayload) {
    const now = performance.now();
    this.queue.push({ item, arrivedAt: now });
    if (!this.isRunning) {
      this.start();
    }
  }

  public start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.lastRenderTime = performance.now();
    this.rafId = requestAnimationFrame(this.rafLoop);
  }

  public stop() {
    this.isRunning = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  public clear() {
    this.stop();
    this.queue = [];
    this.metrics = [];
    this.lastRenderTime = 0;
    this.lastArrivalTime = 0;
  }

  public getQueueLength(): number {
    return this.queue.length;
  }

  public getMetrics(): TokenTimingMetric[] {
    return [...this.metrics];
  }

  private rafLoop = (timestamp: number) => {
    if (!this.isRunning) return;

    const backlog = this.queue.length;
    // Adaptive pace control: accelerate to 15ms if backlog > 5
    const targetInterval = backlog > this.backlogThreshold ? this.fastIntervalMs : this.normalIntervalMs;
    const elapsed = timestamp - this.lastRenderTime;

    if (elapsed >= targetInterval && this.queue.length > 0) {
      const entry = this.queue.shift()!;
      const arrivalDelta = this.lastArrivalTime > 0 ? entry.arrivedAt - this.lastArrivalTime : 0;
      this.lastArrivalTime = entry.arrivedAt;

      const renderDelta = elapsed;
      this.lastRenderTime = timestamp;

      const metric: TokenTimingMetric = {
        step: entry.item.step,
        token: entry.item.token,
        arrivalDeltaMs: Math.round(arrivalDelta * 10) / 10,
        renderDeltaMs: Math.round(renderDelta * 10) / 10,
        queueBacklog: backlog,
        renderIntervalUsed: targetInterval,
        timestamp: Date.now(),
      };
      this.metrics.push(metric);

      // Notify callback to trigger render (DOM update / state)
      this.callbacks.onRenderToken(entry.item, metric);
      if (this.callbacks.onMetricsUpdate) {
        this.callbacks.onMetricsUpdate(this.metrics);
      }

      if (entry.item.is_finished && this.queue.length === 0) {
        this.isRunning = false;
        if (this.callbacks.onFinish) {
          this.callbacks.onFinish();
        }
        return;
      }
    }

    if (this.isRunning) {
      this.rafId = requestAnimationFrame(this.rafLoop);
    }
  };
}
