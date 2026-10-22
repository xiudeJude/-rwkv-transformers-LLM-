# LLM 可解释性交互工坊 - 前端交互工作台

本项目为“LLM 可解释性交互工坊（Transformer vs RWKV 双范式对比）”的前端交互界面，基于 **React 19**、**TypeScript**、**Tailwind CSS** 与 **Vite** 构建，提供 60 FPS 硬件加速的流式大模型内部隐状态演化可视化。

---

## 🎨 核心组件与可视化管线

| 组件 | 功能职责 | 技术实现亮点 |
| :--- | :--- | :--- |
| **`AttentionHeatmap.tsx`** | Transformer 因果自注意力矩阵视区 | 离屏双缓冲 Canvas 增量追加，Prefill 阶段渲染初始方阵，Decode 阶段逐行追加；采用 Log-scale 非线性色阶清晰展现低概率注意力背景 |
| **`RWKVWaterfall.tsx`** | RWKV-7 隐状态演化时序瀑布图 | 左右双色带同屏对称对照：色带 A（Viridis，绝对状态模长 $S_t$）与色带 B（Coolwarm，单步差分 $\mathbf{\Delta}_t$）；基于 `RWKV_LAYER_SCALES` 24 层经验标定常量，层内绝对锁定无时间漂移 |
| **`RWKVStateModal.tsx`** | Level 2 微观状态探针模态窗 | 点击历史步弹出，通过按需优化接口（~8ms）提取单头 $64 \times 64$（4,096 权重）原始隐藏状态矩阵，透视微观特征分布 |
| **`LayerHeadSelector.tsx`** | 层级与头维度交互控制器 | 动态适配 Transformer（24层/8头）与 RWKV（24层/16头）；生成中严格置灰锁定当前观测层，生成结束后解锁并支持并发全量回填 |
| **`TopKBarChart.tsx`** | 实时首位预测与候选分布柱状图 | 呈现每一步 Decode 采样的 Top-K 候选词及其 Softmax 概率分布 |

---

## 🛠️ 环境要求

- **Node.js**：`>= 18.0.0`（推荐 Node.js 20 LTS 或 22 LTS）
- **包管理器**：`npm >= 9.0.0`

---

## 🚀 启动与构建指令

进入 `frontend` 目录执行以下命令：

```bash
# 1. 安装项目依赖 (首次运行或依赖更新)
npm install
# 或在 CI/CD 中进行严格一致性安装:
npm ci

# 2. 启动本地开发服务器 (默认端口 5173)
npm run dev

# 3. 生产环境打包构建 (包含 TypeScript 严格类型检查)
npm run build

# 4. 本地预览生产打包产物
npm run preview

# 5. 代码质量快速检查 (基于 Oxlint)
npm run lint
```

---

## ⚙️ 后端服务连接与代理配置

前端与后端交互支持两种无缝配置方式：

### 方式 1：环境变量覆盖（推荐用于独立部署或自定义网络环境）
复制 `frontend/.env.example` 为 `frontend/.env.local`：
```ini
# 后端 HTTP API 服务基础地址
VITE_API_BASE=http://127.0.0.1:8000

# 后端 WebSocket 服务基础地址 (可选，默认自动由 VITE_API_BASE 推导)
VITE_WS_BASE=ws://127.0.0.1:8000
```
在代码内部（`App.tsx`），系统会自动读取 `import.meta.env.VITE_API_BASE`，若未指定则默认回退至 `http://127.0.0.1:8000`。

### 方式 2：Vite 本地开发服务器代理 (Dev Proxy)
在 `vite.config.ts` 中已内建开发服务器代理规则：
```typescript
server: {
  host: '127.0.0.1',
  port: 5173,
  proxy: {
    '/api': {
      target: 'http://127.0.0.1:8000',
      changeOrigin: true,
    },
    '/ws': {
      target: 'ws://127.0.0.1:8000',
      ws: true,
      changeOrigin: true,
    },
  },
}
```
当通过同源路径（如 `/api/...` 或 `/ws/...`）发起请求时，Vite 将自动转发至后端服务，规避任何潜在的跨域或内网穿透网络限制。
