#!/bin/bash

echo "========================================"
echo "   MultiModal RAG ChatBox - Linux/Mac"
echo "========================================"
echo ""

# 检查是否安装 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python3，请先安装 Python 3.10+"
    exit 1
fi
echo "[OK] Python3 已安装"

# 检查是否安装 Node.js
if ! command -v node &> /dev/null; then
    echo "[错误] 未检测到 Node.js，请先安装 Node.js 18+"
    exit 1
fi
echo "[OK] Node.js 已安装"
echo ""

# 创建虚拟环境（如果不存在）
if [ ! -d "backend/venv" ]; then
    echo "[步骤 1] 创建 Python 虚拟环境..."
    cd backend
    python3 -m venv venv
    cd ..
    echo "[OK] 虚拟环境创建成功"
    echo ""
fi

# 激活虚拟环境并安装依赖
echo "[步骤 2] 安装 Python 依赖..."
source backend/venv/bin/activate
pip install -r backend/requirements.txt
echo "[OK] Python 依赖安装完成"
echo ""

# 安装前端依赖
if [ ! -d "frontend/node_modules" ]; then
    echo "[步骤 3] 安装前端依赖..."
    cd frontend
    npm install
    cd ..
    echo "[OK] 前端依赖安装完成"
    echo ""
fi

# 清理旧的进程
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# 启动后端服务
echo "[步骤 4] 启动后端服务..."
cd backend
source venv/bin/activate
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "[OK] 后端服务已启动 (PID: $BACKEND_PID)"
cd ..

# 等待后端启动
echo "等待后端服务启动..."
sleep 5

# 启动前端服务
echo "[步骤 5] 启动前端服务..."
cd frontend
nohup npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "[OK] 前端服务已启动 (PID: $FRONTEND_PID)"
cd ..

# 保存 PID 到文件
echo $BACKEND_PID > backend.pid
echo $FRONTEND_PID > frontend.pid

echo ""
echo "========================================"
echo "   服务启动成功！"
echo "   - 前端地址: http://localhost:3000"
echo "   - 后端地址: http://localhost:8000"
echo "   - API 文档: http://localhost:8000/docs"
echo "========================================"
echo ""
echo "查看日志:"
echo "  后端: tail -f backend.log"
echo "  前端: tail -f frontend.log"
echo ""
echo "停止服务: bash stop.sh"
