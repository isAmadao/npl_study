#!/bin/bash

echo "========================================"
echo "   停止 MultiModal RAG ChatBox 服务"
echo "========================================"
echo ""

# 停止后端服务
if [ -f "backend.pid" ]; then
    BACKEND_PID=$(cat backend.pid 2>/dev/null)
    if [ -n "$BACKEND_PID" ]; then
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID 2>/dev/null
            echo "[OK] 后端服务已停止 (PID: $BACKEND_PID)"
        else
            echo "[WARN] 后端服务未运行"
        fi
    fi
    rm -f backend.pid
fi

# 停止前端服务
if [ -f "frontend.pid" ]; then
    FRONTEND_PID=$(cat frontend.pid 2>/dev/null)
    if [ -n "$FRONTEND_PID" ]; then
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID 2>/dev/null
            echo "[OK] 前端服务已停止 (PID: $FRONTEND_PID)"
        else
            echo "[WARN] 前端服务未运行"
        fi
    fi
    rm -f frontend.pid
fi

# 清理可能的残留进程
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# 清理日志文件
rm -f backend.log 2>/dev/null
rm -f frontend.log 2>/dev/null

echo ""
echo "[OK] 所有服务已停止"
