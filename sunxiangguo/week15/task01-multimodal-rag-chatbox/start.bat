@echo off
chcp 65001 >nul
echo ========================================
echo    MultiModal RAG ChatBox - Windows
echo ========================================
echo.

REM 检查是否安装 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
echo [OK] Python 已安装

REM 检查是否安装 Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Node.js，请先安装 Node.js 18+
    pause
    exit /b 1
)
echo [OK] Node.js 已安装
echo.

REM 创建虚拟环境（如果不存在）
if not exist "backend\venv" (
    echo [步骤 1] 创建 Python 虚拟环境...
    cd backend
    python -m venv venv
    cd ..
    echo [OK] 虚拟环境创建成功
    echo.
)

REM 激活虚拟环境并安装依赖
echo [步骤 2] 安装 Python 依赖...
call backend\venv\Scripts\activate.bat
pip install -r backend\requirements.txt
echo [OK] Python 依赖安装完成
echo.

REM 安装前端依赖
if not exist "frontend\node_modules" (
    echo [步骤 3] 安装前端依赖...
    cd frontend
    call npm install
    cd ..
    echo [OK] 前端依赖安装完成
    echo.
)

REM 启动后端服务
echo [步骤 4] 启动后端服务...
cd backend
start "MultimodalRAG-Backend" cmd /k "venv\Scripts\activate.bat && python -m pip install -r requirements.txt && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
cd ..

REM 等待后端启动
echo 等待后端服务启动...
timeout /t 5 /nobreak >nul

REM 启动前端服务
echo [步骤 5] 启动前端服务...
cd frontend
start "MultimodalRAG-Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ========================================
echo    服务启动成功！
echo    - 前端地址: http://localhost:3000
echo    - 后端地址: http://localhost:8000
echo    - API 文档: http://localhost:8000/docs
echo ========================================
echo.
echo 按任意键关闭此窗口（服务将继续运行）...
pause >nul
