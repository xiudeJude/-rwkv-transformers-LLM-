# ======================================================================
#  LLM 可解释性交互工坊 (Transformer vs RWKV) - 一键启动脚本 (PowerShell)
# ======================================================================
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$host.UI.RawUI.WindowTitle = "LLM 可解释性交互工坊 - 启动控制台"

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $rootDir

function Print-Banner {
    Clear-Host
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "      🧠  LLM 可解释性交互工坊 (Transformer vs RWKV 双范式对比)" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
}

Print-Banner

# 1. 环境前置检查与端口清理
Write-Host "[1/5] 检查端口占用并清理旧残留进程..." -ForegroundColor Yellow
$ports = @(8000, 5173)
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conns) {
        foreach ($conn in $conns) {
            $targetPid = $conn.OwningProcess
            if ($targetPid) {
                try {
                    Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
                    Write-Host "    - 已清理占用端口 :$port 的残留进程 (PID: $targetPid)" -ForegroundColor DarkGray
                } catch {}
            }
        }
    }
}
Write-Host "    [√] 端口 8000 与 5173 已就绪！" -ForegroundColor Green

# 2. 检查 Python 环境
Write-Host ""
Write-Host "[2/5] 检查 Python 运行环境..." -ForegroundColor Yellow
$venvPython = Join-Path $rootDir ".venv\Scripts\python.exe"
$pythonExe = $null

if (Test-Path $venvPython) {
    $pythonExe = $venvPython
    Write-Host "    [√] 发现专用虚拟环境: $venvPython" -ForegroundColor Green
} else {
    $sysPython = Get-Command python -ErrorAction SilentlyContinue
    if ($sysPython) {
        $pythonExe = "python"
        Write-Host "    [!] 未找到 .venv，使用系统 Python: $($sysPython.Source)" -ForegroundColor Yellow
    } else {
        Write-Host "    [X] 错误: 未检测到 Python！请安装 Python 3.10+ 或配置 .venv 虚拟环境。" -ForegroundColor Red
        pause
        exit 1
    }
}

# 3. 检查 Node.js / npm 与前端依赖
Write-Host ""
Write-Host "[3/5] 检查 Node.js 与前端依赖..." -ForegroundColor Yellow
$npmCmd = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmCmd) {
    Write-Host "    [X] 错误: 未检测到 npm / Node.js！请前往 https://nodejs.org 安装 Node.js。" -ForegroundColor Red
    pause
    exit 1
}
$nodeVersion = (node -v).Trim()
Write-Host "    [√] Node.js 版本: $nodeVersion" -ForegroundColor Green

$frontendDir = Join-Path $rootDir "frontend"
$nodeModulesDir = Join-Path $frontendDir "node_modules"
if (-not (Test-Path $nodeModulesDir)) {
    Write-Host "    [*] 检测到前端依赖未安装，正在执行 npm install (首次启动可能需要 1~2 分钟)..." -ForegroundColor Yellow
    Push-Location $frontendDir
    npm install
    Pop-Location
    if (-not (Test-Path $nodeModulesDir)) {
        Write-Host "    [X] 前端依赖安装失败，请检查网络后重试。" -ForegroundColor Red
        pause
        exit 1
    }
}
Write-Host "    [√] 前端依赖包已就绪！" -ForegroundColor Green

# 4. 启动后端 FastAPI 服务
Write-Host ""
Write-Host "[4/5] 启动后端推理引擎 (FastAPI :8000)..." -ForegroundColor Yellow
$backendDir = Join-Path $rootDir "backend"
$backendArg = "/k `"title LLM工坊-后端 (FastAPI :8000) && cd /d `"$backendDir`" && `"$pythonExe`" -m uvicorn main:app --host 127.0.0.1 --port 8000`""
Start-Process -FilePath "cmd.exe" -ArgumentList $backendArg -WindowStyle Normal

# 等待后端就绪
Write-Host "    [*] 等待后端初始化并载入模型配置..." -NoNewline
$backendReady = $false
for ($i = 0; $i -lt 35; $i++) {
    Start-Sleep -Milliseconds 800
    try {
        $res = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/model/info" -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($res.StatusCode -eq 200) {
            $backendReady = $true
            break
        }
    } catch {}
    Write-Host "." -NoNewline
}
Write-Host ""

if ($backendReady) {
    Write-Host "    [√] 后端服务已成功就绪 (HTTP 200 OK)！" -ForegroundColor Green
} else {
    Write-Host "    [!] 后端服务启动较慢或正在初始化，将继续拉起前端..." -ForegroundColor Yellow
}

# 5. 启动前端 Vite 服务
Write-Host ""
Write-Host "[5/5] 启动前端交互界面 (Vite :5173)..." -ForegroundColor Yellow
$frontendArg = "/k `"title LLM工坊-前端 (Vite :5173) && cd /d `"$frontendDir`" && npm run dev -- --host 127.0.0.1 --port 5173`""
Start-Process -FilePath "cmd.exe" -ArgumentList $frontendArg -WindowStyle Normal

# 等待前端就绪
Write-Host "    [*] 等待 Vite 开发服务器就绪..." -NoNewline
$frontendReady = $false
for ($j = 0; $j -lt 25; $j++) {
    Start-Sleep -Milliseconds 600
    try {
        $res = Invoke-WebRequest -Uri "http://127.0.0.1:5173" -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($res.StatusCode -eq 200) {
            $frontendReady = $true
            break
        }
    } catch {}
    Write-Host "." -NoNewline
}
Write-Host ""

if ($frontendReady) {
    Write-Host "    [√] 前端服务已成功就绪！" -ForegroundColor Green
} else {
    Write-Host "    [√] 前端已在后台启动！" -ForegroundColor Green
}

# 自动唤起浏览器
Write-Host ""
Write-Host "🚀 正在自动打开浏览器访问应用..." -ForegroundColor Cyan
Start-Process "http://127.0.0.1:5173"

# 最终控制台交互界面
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "         🎉  LLM 可解释性交互工坊 已成功全量启动！" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  🌐 前端交互界面 : " -NoNewline; Write-Host "http://127.0.0.1:5173" -ForegroundColor Cyan
Write-Host "  ⚙️ 后端服务接口 : " -NoNewline; Write-Host "http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "  📚 Swagger API  : " -NoNewline; Write-Host "http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "  🤖 已加载对比架构:" -ForegroundColor White
Write-Host "     • Transformer : Qwen3.5-0.8B (GGUF Q8_0, 24层/8头因果注意力矩阵)" -ForegroundColor Gray
Write-Host "     • RWKV-7      : Goose 0.4B (24层/16头 隐状态演化与差分瀑布图)" -ForegroundColor Gray
Write-Host ""
Write-Host "----------------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  快捷指令:" -ForegroundColor Yellow
Write-Host "    [Q] 输入 Q 并回车: 一键停止并关闭全部服务与子窗口"
Write-Host "    [R] 输入 R 并回车: 重启所有服务"
Write-Host "    [直接关闭本窗口] : 服务将继续在后台保持运行"
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""

while ($true) {
    $userInput = Read-Host "请输入指令 (Q: 停止服务, R: 重启, 直接回车保持运行)"
    if ($userInput -eq "q" -or $userInput -eq "Q") {
        & (Join-Path $rootDir "stop.ps1")
        exit 0
    } elseif ($userInput -eq "r" -or $userInput -eq "R") {
        Write-Host "正在重启服务..." -ForegroundColor Yellow
        & (Join-Path $rootDir "stop.ps1")
        Start-Sleep -Seconds 1
        & (Join-Path $rootDir "start.ps1")
        exit 0
    }
}
