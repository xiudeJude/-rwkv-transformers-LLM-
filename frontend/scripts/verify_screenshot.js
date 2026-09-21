import puppeteer from 'puppeteer-core';
import path from 'path';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const ARTIFACTS_DIR = 'C:\\Users\\xiude\\.gemini\\antigravity\\brain\\8ac3db94-ca34-4630-ad34-fbf333b5c8e4';
const FULL_SCREENSHOT = path.join(ARTIFACTS_DIR, 'rwkv_waterfall_verified.png');
const CANVAS_SCREENSHOT = path.join(ARTIFACTS_DIR, 'rwkv_canvas_element.png');

async function run() {
  console.log('Launching Edge browser...');
  const browser = await puppeteer.launch({
    executablePath: EDGE_PATH,
    headless: true,
    defaultViewport: { width: 1440, height: 1000, deviceScaleFactor: 1 },
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();
    console.log('Navigating to http://127.0.0.1:5173 ...');
    await page.goto('http://127.0.0.1:5173', { waitUntil: 'networkidle0' });
    await new Promise((r) => setTimeout(r, 1000));

    console.log('Clicking RWKV-7 Tab...');
    const buttons = await page.$$('button');
    let rwkvBtn = null;
    for (const b of buttons) {
      const text = await page.evaluate(el => el.textContent, b);
      if (text.includes('RWKV-7')) {
        rwkvBtn = b;
        break;
      }
    }
    if (!rwkvBtn) throw new Error('RWKV-7 tab button not found');
    await rwkvBtn.click();
    await new Promise((r) => setTimeout(r, 1000));

    console.log('Clicking 流式生成 (WS)...');
    const buttonsAfter = await page.$$('button');
    let streamBtn = null;
    for (const b of buttonsAfter) {
      const text = await page.evaluate(el => el.textContent, b);
      if (text.includes('流式生成')) {
        streamBtn = b;
        break;
      }
    }
    if (!streamBtn) throw new Error('流式生成 button not found');
    await streamBtn.click();

    console.log('Waiting for generation of 30+ tokens...');
    for (let i = 0; i < 15; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      const stepText = await page.evaluate(() => {
        const spans = Array.from(document.querySelectorAll('span.font-bold'));
        return spans.map(s => s.textContent).join(', ');
      });
      console.log(`  T+${i + 1}s | Steps: ${stepText}`);
      
      const finished = await page.evaluate(() => {
        return document.body.textContent.includes('流式生成已完成');
      });
      if (finished) {
        console.log('Generation completed!');
        break;
      }
    }

    // Brief stabilization pause
    await new Promise((r) => setTimeout(r, 1500));

    console.log(`Saving full page screenshot to ${FULL_SCREENSHOT} ...`);
    await page.screenshot({ path: FULL_SCREENSHOT, fullPage: false });

    console.log('Locating waterfall canvas element...');
    const canvasEl = await page.$('canvas.cursor-pointer');
    if (canvasEl) {
      console.log(`Saving canvas element screenshot to ${CANVAS_SCREENSHOT} ...`);
      await canvasEl.screenshot({ path: CANVAS_SCREENSHOT });
    }

    console.log('Verification capture complete!');
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error('Error during verification screenshot:', err);
  process.exit(1);
});
