# LLM 可解释性交互工坊（Transformer + RWKV 双架构对比）

面向“传智杯 Vibe Coding 挑战赛”，探索并直观呈现不同架构（Transformer vs RWKV）在处理同一序列时内部隐状态与注意力的演化差异。

## 🚀 快速开始与一键启动

本项目提供开箱即用的一键启动脚本，自动完成环境检查、端口清理、双端服务（FastAPI + Vite）联动拉起以及浏览器自动唤起。

### Windows 系统（推荐）
* **一键启动**：直接双击根目录下的 `start.bat`（或在 PowerShell 中执行 `.\start.ps1`）。
  * 自动检测并优先加载 `.venv` 虚拟环境与 Node.js 运行时；
  * 自动清理 8000/5173 端口历史残留进程，防止端口冲突；
  * 联动拉起后端推理引擎（端口 8000）与前端交互工作台（端口 5173）；
  * 健康检查通过后，**自动调用系统默认浏览器打开** `http://127.0.0.1:5173`；
  * 主控制台常驻守护，支持输入 `Q` 一键安全退出全部服务，或输入 `R` 快速重启。
* **一键停止**：双击根目录下的 `stop.bat`（或在 PowerShell 中执行 `.\stop.ps1`），即可彻底终止所有后台进程，释放端口并清空 GPU 显存。

### Linux / WSL / macOS 系统
```bash
chmod +x start.sh stop.sh
./start.sh   # 一键启动并自动唤起服务
./stop.sh    # 一键停止并释放端口
```

### 服务访问地址
- **前端交互界面**：`http://127.0.0.1:5173`
- **后端 API 服务**：`http://127.0.0.1:8000`
- **Swagger 接口文档**：`http://127.0.0.1:8000/docs`

## 模型选型与量化说明

### 关于选择 GGUF Q8_0 量化版本的说明及精度评估
在 Transformer 部分，本项目选用 `Qwen3.5-0.8B-Q8_0.gguf` 权重而非原始 bfloat16 safetensors，主要考量如下：
1. **显存与存储轻量化**：Q8_0 相比原始 bf16 将模型体积与显存占用减少约 50%（~800MB vs ~1.7GB），保障在 8GB 消费级显卡（如 RTX 4060）与本地多模型并排对比时的充裕显存空间；
2. **Attention 权重精度评估**：GGUF Q8_0 属于极高保真度整型量化（每个 Block 具有独立浮点缩放因子），在前向推理阶段经由反量化后以 `bfloat16` 精度执行 $QK^\top$ 点积与 Softmax 计算，实测其 Attention 矩阵相对未量化基准的 KL 散度极小（$\approx 0$），且因果下三角稀疏度与相对排序完全保持无损，完全满足可解释性微观观测要求。

## 模型架构规格（实测）
- **模型**：`Qwen3.5-0.8B`
- **层数 (`config.num_hidden_layers`)**：`24`
- **注意力头数 (`config.num_attention_heads`)**：`8`
- **KV 头数 (`config.num_key_value_heads`)**：`2`（GQA 机制）
- **架构组成**：6 个 Full Attention 层（索引 `[3, 7, 11, 15, 19, 23]`）+ 18 个 Gated DeltaNet 线性递归层
- **RWKV 模型**：`RWKV-7 Goose (0.4B)`
