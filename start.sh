#!/usr/bin/env bash
# ======================================================================
#  LLM 可解释性交互工坊 (Transformer vs RWKV) - 一键启动脚本 (Bash)
# ======================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "      🧠  LLM 可解释性交互工坊 (Transformer vs RWKV 双范式对比)"
echo "======================================================================"

# 1. 查找 Python 环境
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
    PYTHON_EXE="python3"
fi

echo "[*] 使用 Python: $PYTHON_EXE"

# 2. 清理旧端口占用
echo "[*] 检查并清理旧端口占用 (8000, 5173)..."
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 5173/tcp 2>/dev/null || true

# 3. 启动后端
echo "[*] 启动后端推理引擎 (FastAPI :8000)..."
cd "$SCRIPT_DIR/backend"
"$PYTHON_EXE" -m uvicorn main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# 等待后端就绪
echo -n "[*] 等待后端就绪..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/api/model/info >/dev/null 2>&1; then
        echo " 成功!"
        break
    fi
    sleep 1
    echo -n "."
done

# 4. 启动前端
echo "[*] 启动前端交互界面 (Vite :5173)..."
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "======================================================================"
echo "  🎉  LLM 可解释性交互工坊 已成功启动！"
echo "  前端界面: http://127.0.0.1:5173"
echo "  后端 API: http://127.0.0.1:8000 (文档: /docs)"
echo "======================================================================"
echo "按 Ctrl+C 可停止全部服务。"

cleanup() {
    echo ""
    echo "[*] 正在停止服务..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    fuser -k 8000/tcp 2>/dev/null || true
    fuser -k 5173/tcp 2>/dev/null || true
    echo "[√] 所有服务已停止。"
    exit 0
}

trap cleanup INT TERM
wait
