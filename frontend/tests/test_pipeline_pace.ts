import { SmoothRenderPipeline, type TokenTimingMetric } from '../src/utils/renderPipeline.ts';
import type { TokenStepPayload } from '../src/types/schema.ts';

// Mock requestAnimationFrame and performance.now for deterministic testing in Node
let currentTime = 1000.0;
globalThis.performance = {
  now: () => currentTime,
} as any;

let rafCallback: ((time: number) => void) | null = null;
globalThis.requestAnimationFrame = (cb: FrameRequestCallback): number => {
  rafCallback = cb;
  return 1;
};
globalThis.cancelAnimationFrame = (_id: number) => {
  rafCallback = null;
};

function advanceTime(ms: number) {
  currentTime += ms;
  if (rafCallback) {
    const cb = rafCallback;
    rafCallback = null;
    cb(currentTime);
  }
}

console.log('='.repeat(70));
console.log('  SmoothRenderPipeline 自适应调速单元验证 (积压 > 5 提速至 15ms)');
console.log('='.repeat(70));

const renderedMetrics: TokenTimingMetric[] = [];
let finishedCalled: boolean = false;

const pipeline = new SmoothRenderPipeline({
  onRenderToken: (_item, metric) => {
    renderedMetrics.push(metric);
  },
  onFinish: () => {
    finishedCalled = true;
  },
});

// 1. Simulate pushing 10 tokens into queue at once (creating a backlog of 10 > 5)
console.log('\n>>> [阶段 1] 突发推入 10 个 Token (队列初始积压 = 10)...');
for (let i = 1; i <= 10; i++) {
  const item: TokenStepPayload = {
    type: 'token_step',
    step: i,
    token: `tok_${i}`,
    token_id: 100 + i,
    token_prob: 0.8,
    topk_candidates: [],
    is_finished: i === 10,
  };
  pipeline.push(item);
}

console.log(`  推入完毕，当前队列长度: ${pipeline.getQueueLength()}`);
assert(pipeline.getQueueLength() === 10, 'Queue length should be 10');

// 2. Consume tokens one by one and check the interval used
console.log('  Step   | Token    | 消费时积压   | 生效调度间隔     | 状态判断');
console.log('  ' + '-'.repeat(65));

// Loop until queue drains
let loopCount = 0;
while (pipeline.getQueueLength() > 0 && loopCount < 50) {
  loopCount++;
  // Advance by 16ms (simulating 60fps monitor refresh)
  advanceTime(16);
}

for (const m of renderedMetrics) {
  const status = m.renderIntervalUsed === 15 ? '⚡ 追帧加速生效 (15ms)' : '正常调速 (40ms)';
  const sStep = String(m.step).padEnd(6);
  const sTok = String(m.token).padEnd(8);
  const sBacklog = String(m.queueBacklog).padEnd(12);
  const sPace = (m.renderIntervalUsed + 'ms').padEnd(16);
  console.log(`  ${sStep} | ${sTok} | ${sBacklog} | ${sPace} | ${status}`);
  if (m.queueBacklog > 5) {
    if (m.renderIntervalUsed !== 15) {
      throw new Error(`Step ${m.step} with backlog ${m.queueBacklog} should use 15ms pace!`);
    }
  } else {
    if (m.renderIntervalUsed !== 40) {
      throw new Error(`Step ${m.step} with backlog ${m.queueBacklog} should use 40ms pace!`);
    }
  }
}

console.log(`\n  总渲染 Token 数: ${renderedMetrics.length} / 10`);
console.log(`  onFinish 回调触发: ${finishedCalled}`);
assert(renderedMetrics.length === 10, 'All 10 tokens must be rendered');
assert(finishedCalled === true, 'onFinish must be called');

console.log('\n  --> [单元验证通过]: 积压 > 5 时严格切换为 15ms 追帧，降至 <= 5 时平稳回退至 40ms！\n');

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error('Assertion failed: ' + msg);
}
