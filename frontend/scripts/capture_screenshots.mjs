/**
 * 一键截取 README 所需的三张界面图。
 *
 * 前置：
 *   1. 后端已启动（127.0.0.1:8000）
 *   2. 前端已启动（127.0.0.1:5173）
 *   3. 模型权重已就位
 *
 * 运行（在 frontend 目录下）：
 *   node scripts/capture_screenshots.mjs
 *
 * 输出：
 *   docs/screenshots/01-transformer-attention.png
 *   docs/screenshots/02-rwkv-waterfall.png
 *   docs/screenshots/03-level2-probe.png
 */
import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(__dirname, '..', '..', 'docs', 'screenshots');
const EDGE = process.env.EDGE_PATH || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = 'http://127.0.0.1:5173';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

fs.mkdirSync(OUT, { recursive: true });

async function findByText(page, needle) {
  const els = await page.$$('button');
  for (const el of els) {
    const t = await page.evaluate((e) => (e.textContent || '').trim(), el);
    if (t.includes(needle)) return el;
  }
  return null;
}

async function waitFinish(page, maxSec = 120) {
  for (let i = 0; i < maxSec; i++) {
    await sleep(1000);
    const done = await page.evaluate(() => document.body.textContent.includes('流式生成已完成'));
    if (done) return true;
  }
  return false;
}

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: true,
  defaultViewport: { width: 1600, height: 1000, deviceScaleFactor: 1.25 },
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.goto(BASE, { waitUntil: 'networkidle0', timeout: 120000 });
  await sleep(2500);

  // 1. Transformer
  console.log('[1/3] Transformer...');
  let btn = await findByText(page, '流式生成');
  if (!btn) throw new Error('未找到「流式生成」按钮');
  await btn.click();
  console.log('      generation finished:', await waitFinish(page));
  await sleep(2500);
  await page.screenshot({ path: path.join(OUT, '01-transformer-attention.png') });

  // 2. RWKV
  console.log('[2/3] RWKV...');
  const rwkvTab = await findByText(page, 'RWKV-7');
  if (!rwkvTab) throw new Error('未找到 RWKV Tab');
  await rwkvTab.click();
  await sleep(6000);
  btn = await findByText(page, '流式生成');
  await btn.click();
  console.log('      generation finished:', await waitFinish(page));
  await sleep(2500);
  await page.screenshot({ path: path.join(OUT, '02-rwkv-waterfall.png') });

  // 3. Level 2 modal（按 canvas 逻辑坐标点击色带 A 的第 5 行）
  console.log('[3/3] Level 2 probe...');
  const wf = await page.$('canvas.cursor-pointer');
  if (!wf) throw new Error('未找到瀑布图 canvas');

  await page.evaluate((el) => el.scrollIntoView({ block: 'start' }), wf);
  await sleep(800);
  const m = await page.evaluate((el) => {
    const r = el.getBoundingClientRect();
    return { left: r.left, top: r.top, width: r.width, height: r.height, cw: el.width, ch: el.height };
  }, wf);
  const sx = m.width / m.cw;
  const sy = m.height / m.ch;
  await page.mouse.click(m.left + (82 + 160) * sx, m.top + (56 + 5 * 20 + 10) * sy);
  await sleep(2500);

  const hasModal = await page.evaluate(() => document.body.textContent.includes('4,096'));
  console.log('      modal visible:', hasModal);
  if (!hasModal) throw new Error('Level 2 模态窗未弹出，请检查布局常量 S_ZONE_A_X / S_HEADER_Y');
  await page.screenshot({ path: path.join(OUT, '03-level2-probe.png') });

  console.log('done ->', OUT);
} catch (e) {
  console.error('ERROR:', e.message);
  process.exitCode = 1;
} finally {
  await browser.close();
}
