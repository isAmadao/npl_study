## 多模态RAG聊天框项目 - 部署与运维文档

### 1. 环境要求

#### 1.1 硬件要求

| 环境 | CPU | 内存 | 存储 | GPU |
| :--- | :--- | :--- | :--- | :--- |
| 开发环境 | 4核+ | 16GB+ | 100GB+ | 可选（NVIDIA RTX 3090+） |
| 测试环境 | 8核+ | 32GB+ | 200GB+ | NVIDIA RTX 3090+ |
| 生产环境 | 16核+ | 64GB+ | 500GB+ | NVIDIA A100/A800 |

#### 1.2 软件要求

| 软件 | 版本 | 说明 |
| :--- | :--- | :--- |
| Python | 3.10+ | 后端服务 |
| Docker | 24.0+ | 容器化部署 |
| Docker Compose | 2.20+ | 多容器编排 |
| Git | 2.40+ | 版本控制 |

---

### 2. 依赖安装

#### 2.1 Python依赖

创建 `requirements.txt` 文件：

```txt
# 核心框架
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# 数据库
sqlalchemy==2.0.23
asyncpg==0.27.0
pymilvus==2.3.5
minio==7.2.0
kafka-python==2.0.2

# 机器学习
torch==2.1.0
transformers==4.35.2
pillow==10.1.0
paddleocr==2.7.0

# 工具库
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
loguru==0.7.2
```

#### 2.2 安装命令

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

---

### 3. 配置说明

#### 3.1 环境变量配置

创建 `.env` 文件：

```env
# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# 数据库配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=multimodal_rag
POSTGRES_USER=admin
POSTGRES_PASSWORD=password

# Milvus配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# MinIO配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=documents

# Kafka配置
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=document-processing

# 模型配置
CLIP_MODEL_PATH=models/clip
QWEN_VL_MODEL_PATH=models/qwen-vl
LLM_MODEL_PATH=models/qwen-7b
```

#### 3.2 配置文件结构

```
config/
├── settings.py          # 配置类定义
├── logging.py           # 日志配置
└── database.py          # 数据库连接配置
```

**settings.py** 示例：

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    
    milvus_host: str
    milvus_port: int
    
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    
    kafka_bootstrap_servers: str
    kafka_topic: str
    
    clip_model_path: str
    qwen_vl_model_path: str
    llm_model_path: str
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
```

---

### 4. Docker部署

#### 4.1 Docker Compose配置

创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'

services:
  # PostgreSQL数据库
  postgres:
    image: postgres:15
    container_name: postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Zookeeper (Kafka依赖)
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    container_name: zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  # Kafka
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    container_name: kafka
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"

  # Milvus向量数据库
  milvus:
    image: milvusdb/milvus:v2.3.5
    container_name: milvus
    environment:
      MILVUS_RUN_MODE: standalone
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MinIO对象存储
  minio:
    image: minio/minio:latest
    container_name: minio
    environment:
      MINIO_ACCESS_KEY: ${MINIO_ACCESS_KEY}
      MINIO_SECRET_KEY: ${MINIO_SECRET_KEY}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address ":9001"

  # API服务
  api:
    build: ./backend
    container_name: api
    depends_on:
      postgres:
        condition: service_healthy
      milvus:
        condition: service_healthy
      minio:
        condition: service_started
      kafka:
        condition: service_started
    environment:
      - POSTGRES_HOST=postgres
      - MILVUS_HOST=milvus
      - MINIO_ENDPOINT=minio:9000
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    volumes:
      - ./backend:/app
      - ./models:/models
    ports:
      - "8000:8000"
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

  # CLIP模型服务
  clip-service:
    build: ./models/clip
    container_name: clip-service
    volumes:
      - ./models/clip:/models
    ports:
      - "5000:5000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Qwen-VL模型服务
  qwen-vl-service:
    build: ./models/qwen-vl
    container_name: qwen-vl-service
    volumes:
      - ./models/qwen-vl:/models
    ports:
      - "5001:5001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # LLM服务
  llm-service:
    build: ./models/llm
    container_name: llm-service
    volumes:
      - ./models/llm:/models
    ports:
      - "5002:5002"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  postgres_data:
  milvus_data:
  minio_data:
```

#### 4.2 Dockerfile示例

**backend/Dockerfile**：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 4.3 启动命令

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down

# 重启服务
docker-compose restart api
```

---

### 5. 手动部署

#### 5.1 启动顺序

```bash
# 1. 启动PostgreSQL
sudo systemctl start postgresql

# 2. 启动Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# 3. 启动Kafka
bin/kafka-server-start.sh config/server.properties

# 4. 启动Milvus
./bin/milvus run standalone

# 5. 启动MinIO
minio server /data --console-address ":9001"

# 6. 启动模型服务
python -m clip.service
python -m qwen_vl.service
python -m llm.service

# 7. 启动API服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 5.2 数据库初始化

```bash
# 创建数据库
createdb -U admin multimodal_rag

# 创建表
python -c "from app.database import init_db; init_db()"

# 初始化向量库
python -c "from app.vector_db import init_collection; init_collection()"
```

---

### 6. 监控与告警

#### 6.1 健康检查

**API健康检查端点**：

| 端点 | 说明 |
| :--- | :--- |
| `/health` | API服务健康检查 |
| `/api/v1/models/status` | 模型服务状态 |
| `/api/v1/vector/db/status` | 向量库状态 |

**检查脚本**：

```bash
#!/bin/bash

echo "=== 健康检查 ==="

# 检查API服务
echo "API服务:"
curl -s http://localhost:8000/health || echo "FAIL"

# 检查数据库
echo "PostgreSQL:"
pg_isready -U admin || echo "FAIL"

# 检查Milvus
echo "Milvus:"
curl -s http://localhost:9091/healthz || echo "FAIL"

# 检查MinIO
echo "MinIO:"
curl -s http://localhost:9000/minio/health/live || echo "FAIL"

echo "=== 检查完成 ==="
```

#### 6.2 日志管理

**日志配置**：

```python
# logging.py
from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)
logger.add(
    "logs/app.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="DEBUG",
    rotation="1 day",
    retention="7 days"
)
```

**日志级别**：

| 级别 | 使用场景 |
| :--- | :--- |
| DEBUG | 详细调试信息 |
| INFO | 正常业务流程 |
| WARNING | 异常但不影响运行 |
| ERROR | 错误需要关注 |
| CRITICAL | 严重错误需要立即处理 |

#### 6.3 告警配置

**Prometheus指标端点**：

```python
# 在FastAPI中添加指标
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
```

**告警规则示例**：

```yaml
groups:
  - name: multimodal_rag_alerts
    rules:
      - alert: APIDown
        expr: up{job="api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API服务宕机"
          description: "API服务已停止运行超过1分钟"

      - alert: HighLatency
        expr: sum(rate(http_request_duration_seconds_sum[5m])) / sum(rate(http_request_duration_seconds_count[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API响应延迟过高"
          description: "平均响应时间超过5秒"

      - alert: MilvusDown
        expr: up{job="milvus"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Milvus向量数据库宕机"
          description: "Milvus服务已停止运行超过1分钟"
```

---

### 7. 备份与恢复

#### 7.1 数据库备份

```bash
# 备份PostgreSQL
pg_dump -U admin multimodal_rag > backup.sql

# 备份Milvus
milvus_backup --backup_name backup_$(date +%Y%m%d)

# 备份MinIO
mc mirror minio/documents backup/documents
```

#### 7.2 数据库恢复

```bash
# 恢复PostgreSQL
psql -U admin multimodal_rag < backup.sql

# 恢复Milvus
milvus_backup --restore_name backup_20240101

# 恢复MinIO
mc mirror backup/documents minio/documents
```

#### 7.3 定时备份脚本

```bash
#!/bin/bash

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p ${BACKUP_DIR}/${DATE}

# 备份PostgreSQL
pg_dump -U admin multimodal_rag > ${BACKUP_DIR}/${DATE}/postgres.sql

# 备份Milvus
milvus_backup --backup_name backup_${DATE} --backup_dir ${BACKUP_DIR}/${DATE}

# 清理7天前的备份
find ${BACKUP_DIR} -type d -mtime +7 -exec rm -rf {} \;

echo "备份完成: ${BACKUP_DIR}/${DATE}"
```

---

### 8. 性能优化

#### 8.1 模型优化

| 优化策略 | 说明 |
| :--- | :--- |
| 模型量化 | 使用FP16/INT8量化减少显存占用 |
| 模型并行 | 大模型使用多GPU并行推理 |
| 缓存机制 | 缓存高频查询结果 |

#### 8.2 数据库优化

| 优化策略 | 说明 |
| :--- | :--- |
| 索引优化 | 为频繁查询字段创建索引 |
| 连接池 | 使用连接池减少数据库连接开销 |
| 查询优化 | 优化复杂SQL查询 |

#### 8.3 服务优化

| 优化策略 | 说明 |
| :--- | :--- |
| 异步处理 | 使用Kafka异步处理文档解析 |
| 负载均衡 | 使用NGINX进行负载均衡 |
| 缓存层 | 添加Redis缓存层 |

---

### 9. 故障排查

#### 9.1 常见问题

| 问题 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| API无法启动 | 端口被占用 | 检查端口占用，修改端口配置 |
| 数据库连接失败 | PostgreSQL未启动或配置错误 | 检查PostgreSQL状态和连接配置 |
| 向量检索失败 | Milvus未启动或索引未创建 | 检查Milvus状态，初始化索引 |
| 模型推理失败 | GPU显存不足 | 减少batch size或使用更小模型 |
| 文件上传失败 | MinIO未启动或Bucket不存在 | 检查MinIO状态，创建Bucket |

#### 9.2 日志排查

```bash
# 查看API日志
docker-compose logs -f api

# 查看模型服务日志
docker-compose logs -f clip-service

# 查看Milvus日志
docker-compose logs -f milvus

# 查看Kafka日志
docker-compose logs -f kafka
```

---

### 10. 安全加固

#### 10.1 访问控制

| 措施 | 说明 |
| :--- | :--- |
| 防火墙 | 限制只允许内部IP访问 |
| VPN | 使用VPN访问生产环境 |
| 白名单 | 配置API访问白名单 |

#### 10.2 数据安全

| 措施 | 说明 |
| :--- | :--- |
| 加密传输 | 使用HTTPS/TLS |
| 数据加密 | 敏感数据加密存储 |
| 定期备份 | 定期备份并验证 |

#### 10.3 安全审计

| 措施 | 说明 |
| :--- | :--- |
| 访问日志 | 记录所有API访问日志 |
| 定期审计 | 定期安全审计 |
| 漏洞扫描 | 定期漏洞扫描 |