#!/usr/bin/env bash
# ============================================================
# 多模态 RAG 系统 — 一键启动脚本
#
# 用法:
#   bash run.sh              # 启动全部服务（需 zookeeper + kafka + milvus 已运行，或使用 milvus-lite）
#   bash run.sh upload       # 仅启动上传服务
#   bash run.sh worker       # 仅启动离线解析 worker
#   bash run.sh chat         # 仅启动问答服务
#   bash run.sh infra        # 用 docker-compose 启动中间件
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 安装依赖
install_deps() {
    if [ ! -d "venv" ]; then
        echo "[setup] 创建虚拟环境..."
        python3 -m venv venv
    fi
    source venv/bin/activate
    echo "[setup] 安装依赖..."
    pip install -q -r requirements.txt
}

# 启动中间件 (Kafka + Zookeeper, Milvus 使用 lite 模式)
start_infra() {
    echo "[infra] 启动 Kafka + Zookeeper..."
    if command -v docker-compose &>/dev/null; then
        docker-compose up -d
    elif command -v docker &>/dev/null && docker compose version &>/dev/null; then
        docker compose up -d
    else
        echo "[infra] 未找到 docker-compose，请手动启动中间件"
    fi
}

# 初始化数据库
init_db() {
    source venv/bin/activate
    python3 -c "from models import init_db; init_db(); print('[db] SQLite 数据库已初始化')"
}

# 启动上传服务
start_upload() {
    source venv/bin/activate
    echo "[upload] 启动上传服务 (port 8001)..."
    python3 web_page_upload.py
}

# 启动离线 worker
start_worker() {
    source venv/bin/activate
    echo "[worker] 启动离线解析 Worker..."
    python3 offline_process_worker.py
}

# 启动问答服务
start_chat() {
    source venv/bin/activate
    echo "[chat] 启动问答服务 (port 8000)..."
    python3 web_page_chat.py
}

# 启动全部
start_all() {
    source venv/bin/activate
    init_db
    echo "[all] 启动全部服务..."
    python3 web_page_upload.py &
    PID_UPLOAD=$!
    python3 offline_process_worker.py &
    PID_WORKER=$!
    python3 web_page_chat.py &
    PID_CHAT=$!

    echo "[all] 服务已启动: upload=$PID_UPLOAD worker=$PID_WORKER chat=$PID_CHAT"
    echo "[all] 上传服务: http://localhost:8001/docs"
    echo "[all] 问答服务: http://localhost:8000/docs"

    trap "kill $PID_UPLOAD $PID_WORKER $PID_CHAT 2>/dev/null; exit" INT TERM
    wait
}

# ---- main ----
case "${1:-all}" in
    install)
        install_deps
        ;;
    infra)
        start_infra
        ;;
    upload)
        install_deps
        init_db
        start_upload
        ;;
    worker)
        install_deps
        init_db
        start_worker
        ;;
    chat)
        install_deps
        init_db
        start_chat
        ;;
    all)
        install_deps
        start_all
        ;;
    *)
        echo "用法: bash run.sh {install|infra|upload|worker|chat|all}"
        ;;
esac