import subprocess
import time
import json
import base64
import asyncio
import os
import requests
import websockets

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
USER_DATA_DIR = os.path.join(os.environ.get("TEMP", "C:/temp"), "edge_cdp_profile")
ARTIFACTS_DIR = r"C:\Users\xiude\.gemini\antigravity\brain\8ac3db94-ca34-4630-ad34-fbf333b5c8e4"
SCREENSHOT_PATH = os.path.join(ARTIFACTS_DIR, "rwkv_waterfall_verified.png")

async def cdp_send(ws, method, params=None, msg_id=[1]):
    mid = msg_id[0]
    msg_id[0] += 1
    req = {"id": mid, "method": method, "params": params or {}}
    await ws.send(json.dumps(req))
    while True:
        res_raw = await ws.recv()
        res = json.loads(res_raw)
        if res.get("id") == mid:
            return res.get("result", {})

async def capture():
    # 1. Launch Edge headless with remote debugging
    cmd = [
        EDGE_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        f"--user-data-dir={USER_DATA_DIR}",
        "--window-size=1440,1100",
        "--hide-scrollbars",
        "about:blank"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    try:
        # 2. Get WebSocket debugger URL
        tabs = requests.get("http://127.0.0.1:9222/json").json()
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        print(f"Connected to Edge CDP: {ws_url}")

        async with websockets.connect(ws_url, max_size=20*1024*1024) as ws:
            # Enable Runtime and Page
            await cdp_send(ws, "Page.enable")
            await cdp_send(ws, "Runtime.enable")

            # Navigate to frontend
            print("Navigating to http://127.0.0.1:5173 ...")
            await cdp_send(ws, "Page.navigate", {"url": "http://127.0.0.1:5173"})
            await asyncio.sleep(2.5)

            # Click RWKV tab
            print("Switching to RWKV-7 Tab...")
            eval_res = await cdp_send(ws, "Runtime.evaluate", {
                "expression": """
                (() => {
                    const btns = Array.from(document.querySelectorAll('button'));
                    const rwkvBtn = btns.find(b => b.textContent.includes('RWKV-7'));
                    if (rwkvBtn) { rwkvBtn.click(); return 'clicked RWKV'; }
                    return 'RWKV button not found';
                })()
                """
            })
            print("Tab switch eval:", eval_res.get("result", {}).get("value"))
            await asyncio.sleep(1.5)

            # Start streaming generation
            print("Clicking '流式生成 (WS)'...")
            stream_res = await cdp_send(ws, "Runtime.evaluate", {
                "expression": """
                (() => {
                    const btns = Array.from(document.querySelectorAll('button'));
                    const streamBtn = btns.find(b => b.textContent.includes('流式生成'));
                    if (streamBtn) { streamBtn.click(); return 'clicked stream'; }
                    return 'stream button not found';
                })()
                """
            })
            print("Stream click eval:", stream_res.get("result", {}).get("value"))

            # Wait for 35 tokens to generate (approx 96ms * 35 = 3.5s + buffer)
            print("Waiting 6 seconds for 35+ tokens to generate and render on canvas...")
            for i in range(6):
                await asyncio.sleep(1)
                poll_res = await cdp_send(ws, "Runtime.evaluate", {
                    "expression": """
                    (() => {
                        const stepSpan = document.querySelector('span.text-emerald-300.font-bold');
                        return stepSpan ? stepSpan.textContent : 'unknown';
                    })()
                    """
                })
                print(f"  T+{i+1}s | 当前步数: {poll_res.get('result', {}).get('value')}")

            # Capture screenshot
            print(f"Capturing screenshot to {SCREENSHOT_PATH} ...")
            ss = await cdp_send(ws, "Page.captureScreenshot", {"format": "png"})
            b64_data = ss.get("data", "")
            if b64_data:
                with open(SCREENSHOT_PATH, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                print(f"Screenshot successfully saved! Size: {os.path.getsize(SCREENSHOT_PATH)} bytes")
            else:
                print("Failed to capture screenshot data.")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(capture())
