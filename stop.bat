@echo off
@rem ======================================================================
@rem  LLM 可解释性交互工坊 (Transformer vs RWKV) - Windows 一键停止脚本
@rem ======================================================================
chcp 65001 >nul
cd /d "%~dp0"
title LLM 可解释性交互工坊 - 停止服务

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
if %errorlevel% neq 0 (
    echo.
    echo [提示] 按任意键退出...
    pause >nul
)
