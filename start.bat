@echo off
@rem ======================================================================
@rem  LLM 可解释性交互工坊 (Transformer vs RWKV) - Windows 一键启动脚本
@rem ======================================================================
chcp 65001 >nul
cd /d "%~dp0"
title LLM 可解释性交互工坊 - 启动器

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if %errorlevel% neq 0 (
    echo.
    echo [提示] 启动过程中遇到问题，请检查上方日志。按任意键退出...
    pause >nul
)
