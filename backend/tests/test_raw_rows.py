import asyncio
import json
import httpx
import websockets

async def check():
    async with httpx.AsyncClient() as client:
        res = await client.post('http://127.0.0.1:8000/api/session/create', json={
            'prompt': '注意力机制让大语言模型能够精确捕捉长距离语义依赖关系', 'layer': 3, 'head': 0
        })
        s_id = res.json()['session_id']
        async with websockets.connect(f'ws://127.0.0.1:8000/ws/session/{s_id}/stream') as ws:
            await ws.send(json.dumps({'action': 'start', 'max_new_tokens': 5, 'layer': 3, 'head': 0}))
            while True:
                raw = await ws.recv()
                m = json.loads(raw)
                if m.get('type') == 'token_step':
                    step = m.get('step')
                    tok = m.get('token')
                    attn_row = m.get('attention_row', [])
                    print(f"Step {step} ({repr(tok)}): len={len(attn_row)}")
                    print(f"  weights: {attn_row}")
                if m.get('is_finished'):
                    break
            await ws.send(json.dumps({'action': 'close'}))

if __name__ == '__main__':
    asyncio.run(check())
