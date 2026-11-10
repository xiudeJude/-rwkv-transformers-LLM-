# 手动验证与诊断脚本

本目录放置依赖真实模型、GPU、外部服务或浏览器环境，无法稳定作为 pytest
自动化测试运行的一次性/诊断脚本。自动测试请使用 `backend/tests/`。

所有命令默认在**仓库根目录**执行；先启动后端服务（`python backend/main.py`）
并完成 `backend/.env` 配置。

| 脚本 | 用途 | 前置条件 |
| :--- | :--- | :--- |
| `calibrate_rwkv_scales.py` | 扫描 RWKV 多层/多头状态范数与差分分布，校准瀑布图色阶 | RWKV 权重、CUDA 可选 |
| `capture_rwkv_screenshot.py` | 通过 Edge CDP 打开前端并截取 RWKV 瀑布图 | Edge、前端 `127.0.0.1:5173`、后端 `8000` |
| `debug_matrix.py` | 打印 Prefill/Level 2 注意力矩阵的列分布，排查因果掩码 | 后端服务、Transformer 权重 |
| `inspect_hybrid_attention_structure.py` | 检查 Qwen 混合架构中 `outputs.attentions` 与层类型的映射关系 | Transformer 权重、CUDA |
| `inspect_raw_rows.py` | 通过 WebSocket 打印每个 decode step 的原始 attention 行长度与权重 | 后端服务、Transformer 权重 |
| `measure_kl_divergence.py` | 量化模型 vs 未量化基准的 Logits/Attention KL 散度与 CLI 报告 | 两份 Transformer 权重；也作为 pytest 的测量库被导入 |
| `tab_switch_vram.py` | 模拟不主动 close 的 Tab 切换，观察显存累积（诊断泄漏） | 后端服务、`nvidia-smi` |
| `tab_switch_with_cleanup.py` | 模拟主动 close 的 Tab 切换，观察显存是否稳定 | 后端服务、`nvidia-smi` |
| `verify_step4_fixes.py` | 手动验收 Prefill 因果性、重复惩罚、WS attention 行、历史 Level 2 回填 | 后端服务、Transformer 权重 |

## 示例

```bash
# KL 散度测量（需要未量化基准权重）
python backend/scripts/manual/measure_kl_divergence.py \
  --unquantized-path backend/models/Qwen3.5-0.8B --device cuda

# 检查 RWKV 状态色阶分布
python backend/scripts/manual/calibrate_rwkv_scales.py

# 诊断 Tab 切换显存
python backend/scripts/manual/tab_switch_vram.py
```

`test_websocket_verification_suite.py` 和 `test_kv_cache_boundaries.py` 已具备
稳定断言，保留在 `backend/tests/` 中由 pytest 自动收集。
