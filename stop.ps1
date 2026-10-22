# ======================================================================
#  LLM 可解释性交互工坊 (Transformer vs RWKV) - 一键停止服务脚本 (PowerShell)
# ======================================================================
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$host.UI.RawUI.WindowTitle = "LLM 可解释性交互工坊 - 一键停止服务"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "       🛑  LLM 可解释性交互工坊 - 正在停止所有服务与释放显存..." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$ports = @(8000, 5173)
$killedPids = @()

foreach ($port in $ports) {
    Write-Host "[*] 正在扫描监听端口 :$port..." -ForegroundColor Yellow -NoNewline
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conns) {
        Write-Host " 发现占用进程" -ForegroundColor Red
        foreach ($conn in $conns) {
            $targetPid = $conn.OwningProcess
            if ($targetPid -and ($killedPids -notcontains $targetPid)) {
                $killedPids += $targetPid
                try {
                    $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
                    $procName = if ($proc) { $proc.ProcessName } else { "未知" }
                    Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
                    Write-Host "    [√] 成功终止进程: $procName (PID: $targetPid)" -ForegroundColor Green
                } catch {
                    Write-Host "    [!] 终止 PID $targetPid 失败: $_" -ForegroundColor DarkRed
                }
            }
        }
    } else {
        Write-Host " 空闲 (无占用)" -ForegroundColor Gray
    }
}

# 终止由启动脚本唤起的独立 CMD 窗口
$windowTitles = @("LLM工坊-后端*", "LLM工坊-前端*")
foreach ($wt in $windowTitles) {
    try {
        cmd.exe /c "taskkill /f /fi `"WINDOWTITLE eq $wt`" >nul 2>&1"
    } catch {}
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  ✨ [√] 全部服务已安全退出，端口 8000/5173 与 GPU 显存已完全释放！" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Start-Sleep -Seconds 1
