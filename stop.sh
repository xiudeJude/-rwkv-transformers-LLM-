#!/usr/bin/env bash
# ======================================================================
#  LLM 可解释性交互工坊 (Transformer vs RWKV) - 一键停止脚本 (Bash)
# ======================================================================
echo "[*] 正在停止后端与前端服务..."
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 5173/tcp 2>/dev/null || true
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
echo "[√] 端口 8000 与 5173 已释放，服务已停止。"
