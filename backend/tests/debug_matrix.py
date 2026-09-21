import urllib.request
import json
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient() as client:
        # 1. Inspect Prefill matrix
        res = await client.post('http://127.0.0.1:8000/api/session/create', json={
            'prompt': '注意力机制让大语言模型能够精确捕捉长距离语义依赖关系',
            'layer': 3,
            'head': 0
        })
        data = res.json()
        s_id = data['session_id']
        mat = data['prefill_attention']['matrix']
        tokens = data['prefill_attention']['tokens']
        num_rows = len(mat)
        num_cols = len(mat[0])
        print(f"Prefill Matrix: {num_rows} rows x {num_cols} cols")
        print(f"Tokens ({len(tokens)}):", tokens)
        print("Is causal flag:", data['prefill_attention']['is_causal'])

        print("\nChecking columns in prefill matrix:")
        for c in range(num_cols):
            col_vals = [mat[r][c] for r in range(num_rows)]
            print(f"Col {c:<2} ({repr(tokens[c]):<10}): min={min(col_vals):.4f}, max={max(col_vals):.4f}, first3={col_vals[:3]}")

        # 2. Inspect Level 2 attention detail
        res2 = await client.get(f'http://127.0.0.1:8000/api/session/{s_id}/attention?layer=3&head=0')
        data2 = res2.json()
        mat2 = data2['full_matrix']
        tokens2 = data2['tokens']
        print(f"\nLevel 2 Matrix: {len(mat2)} rows x {len(mat2[0])} cols")
        print("\nChecking columns in Level 2 matrix:")
        for c in range(len(mat2[0])):
            col_vals = [mat2[r][c] for r in range(len(mat2))]
            print(f"Col {c:<2} ({repr(tokens2[c]):<10}): min={min(col_vals):.4f}, max={max(col_vals):.4f}, first3={col_vals[:3]}")

        # 3. Do 3 decode steps and inspect attention_slice
        for step in range(1, 4):
            step_res = await client.post(f'http://127.0.0.1:8000/api/session/{s_id}/step', json={
                'temperature': 0.7, 'top_k': 10, 'layer': 3, 'head': 0
            })
            s_data = step_res.json()
            weights = s_data['attention_slice']['weights']
            print(f"\nStep {step} attention_slice (len {len(weights)}):")
            print(f"  weights[:5]: {weights[:5]}")
            print(f"  weights min={min(weights):.4f}, max={max(weights):.4f}, sum={sum(weights):.4f}")

        # 4. Check Level 2 after decode steps
        res3 = await client.get(f'http://127.0.0.1:8000/api/session/{s_id}/attention?layer=3&head=0')
        data3 = res3.json()
        mat3 = data3['full_matrix']
        tokens3 = data3['tokens']
        print(f"\nLevel 2 Matrix after 3 decode steps: {len(mat3)} rows x {len(mat3[0])} cols")
        for c in range(min(5, len(mat3[0]))):
            col_vals = [mat3[r][c] for r in range(len(mat3))]
            print(f"Col {c:<2} ({repr(tokens3[c]):<10}): min={min(col_vals):.4f}, max={max(col_vals):.4f}")

        await client.post(f'http://127.0.0.1:8000/api/session/{s_id}/close')

if __name__ == '__main__':
    asyncio.run(main())
