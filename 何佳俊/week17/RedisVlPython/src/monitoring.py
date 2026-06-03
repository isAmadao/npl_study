"""
RedisVL Agent Cache — Web 可视化监控面板 API
==============================================

基于 FastAPI 构建的 REST API，提供缓存命中率、路由统计、
对话历史和系统状态的实时监控能力。

功能：
    GET  /api/stats      — 缓存/路由/会话汇总统计
    GET  /api/analytics  — 趋势/分布分析数据
    GET  /api/health     — 后端服务健康检查
    POST /api/cache/clear — 清空缓存
    POST /api/threshold  — 更新相似度阈值
    GET  /               — HTML 仪表盘页面

运行方式：
    # 安装依赖
    pip install fastapi uvicorn

    # 启动服务（自动加载模块）
    python -m src.monitoring

    # 或通过 uvicorn
    uvicorn src.monitoring:app --host 0.0.0.0 --port 8000 --reload

    # 打开浏览器
    # http://localhost:8000
"""

import logging
import os
import sys
import time
from datetime import timedelta
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

# ==================== 路径设置 ====================

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ==================== FastAPI 导入 ====================

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ==================== 项目模块导入 ====================

from src.EmbeddingsCache import EmbeddingsCache

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================


class StatsResponse(BaseModel):
    """统计响应模型。"""

    timestamp: float = Field(..., description="时间戳")
    cache: Dict[str, Any] = Field(default_factory=dict, description="缓存统计")
    routes: Dict[str, Any] = Field(default_factory=dict, description="路由统计")
    sessions: Dict[str, Any] = Field(default_factory=dict, description="会话统计")
    system: Dict[str, Any] = Field(default_factory=dict, description="系统信息")


class ThresholdUpdate(BaseModel):
    """阈值更新请求。"""

    threshold: float = Field(..., ge=0.0, le=1.0, description="新的相似度阈值")
    route: Optional[str] = Field(None, description="路由名称（None 为全局）")


class CacheClearResponse(BaseModel):
    """清空缓存响应。"""

    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="结果信息")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(..., description="整体状态 (ok/degraded/error)")
    redis: bool = Field(..., description="Redis 是否正常")
    milvus: bool = Field(..., description="Milvus 是否正常")
    uptime: float = Field(..., description="服务运行时间（秒）")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")


# ==================== 全局状态 ====================

# 应用启动时间
_start_time = time.time()

# 全局缓存实例（由外部注入或自动创建）
_cache: Optional[EmbeddingsCache] = None
_sem_cache: Any = None
_router: Any = None
_history: Any = None


def init_monitoring(
    cache: Optional[EmbeddingsCache] = None,
    sem_cache: Optional[Any] = None,
    router: Optional[Any] = None,
    history: Optional[Any] = None,
) -> None:
    """初始化监控模块（注入外部实例）。

    Args:
        cache: EmbeddingsCache 实例。
        sem_cache: SemanticCache 实例。
        router: SemanticRouter 实例。
        history: SemanticMessageHistory 实例。
    """
    global _cache, _sem_cache, _router, _history
    _cache = cache
    _sem_cache = sem_cache
    _router = router
    _history = history


# ==================== 应用生命周期 ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    global _cache
    logger.info("监控服务启动中...")

    # 如果未注入缓存实例，尝试自动创建
    if _cache is None:
        try:
            _cache = EmbeddingsCache()
            logger.info("自动创建 EmbeddingsCache 成功")
        except Exception as e:
            logger.warning("自动创建 EmbeddingsCache 失败: %s", e)

    yield

    logger.info("监控服务已停止")


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="RedisVL Agent Cache Monitor",
    description="语义缓存系统可视化监控面板 API",
    version="0.1.0",
    lifespan=lifespan,
)


# ==================== API 路由 ====================


@app.get("/api/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """后端服务健康检查。"""
    redis_ok = False
    milvus_ok = False
    details: Dict[str, Any] = {}

    if _cache:
        try:
            _cache.redis.ping()
            redis_ok = True
            details["redis_info"] = {
                "url": str(_cache._settings.redis.url),
                "ping_ms": _measure_latency(lambda: _cache.redis.ping()),
            }
        except Exception as e:
            details["redis_error"] = str(e)

        milvus_ok = _cache.milvus_available
        if _cache.milvus_client:
            try:
                _cache.milvus_client.list_collections()
                details["milvus_info"] = {
                    "collection": _cache.milvus_collection,
                    "dimension": _cache.dim,
                }
            except Exception as e:
                details["milvus_error"] = str(e)
                milvus_ok = False

    status = "ok" if redis_ok else "error"
    if redis_ok and not milvus_ok:
        status = "degraded"

    return HealthResponse(
        status=status,
        redis=redis_ok,
        milvus=milvus_ok,
        uptime=round(time.time() - _start_time, 2),
        details=details,
    )


@app.get("/api/stats", response_model=StatsResponse, tags=["监控"])
async def get_stats():
    """获取所有统计信息。"""
    result: Dict[str, Any] = {
        "timestamp": time.time(),
        "cache": _collect_cache_stats(),
        "routes": _collect_route_stats(),
        "sessions": _collect_session_stats(),
        "system": _collect_system_info(),
    }
    return StatsResponse(**result)


@app.get("/api/analytics", tags=["监控"])
async def get_analytics(
    period: str = Query("1h", description="分析周期（1h/24h/7d/30d）"),
):
    """获取分析数据（趋势、分布）。"""
    period_seconds = {
        "1h": 3600,
        "24h": 86400,
        "7d": 604800,
        "30d": 2592000,
    }
    window = period_seconds.get(period, 3600)

    analytics: Dict[str, Any] = {
        "period": period,
        "timestamp": time.time(),
        "window_seconds": window,
    }

    # 缓存趋势数据
    if _cache:
        try:
            embed_stats = _cache.get_stats()
            analytics["cache"] = embed_stats
        except Exception as e:
            analytics["cache_error"] = str(e)

    # 缓存大小历史（如果有统计记录）
    if _sem_cache:
        try:
            cache_stats = _sem_cache.get_stats()
            analytics["semantic_cache"] = {
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "hit_rate": cache_stats.hit_rate,
                "size": cache_stats.size,
                "avg_similarity": cache_stats.avg_similarity,
                "avg_response_time": cache_stats.avg_response_time,
            }
        except Exception as e:
            analytics["semantic_cache_error"] = str(e)

    # 路由分布
    if _router:
        try:
            router_stats = _router.get_route_stats()
            analytics["router"] = {
                "total_queries": router_stats.total_queries,
                "route_counts": dict(router_stats.route_counts),
                "fallback_rate": router_stats.fallback_rate,
            }
        except Exception as e:
            analytics["router_error"] = str(e)

    return analytics


@app.get("/api/tuning", tags=["调优"])
async def get_tuning_info():
    """获取调优信息（需已注入 AutoTuner）。"""
    return {
        "note": "AutoTuner 调优信息可通过 AutoTuner.get_stats() 获取",
        "enabled": False,
    }


@app.post("/api/cache/clear", response_model=CacheClearResponse, tags=["管理"])
async def clear_cache():
    """清空所有缓存数据。"""
    if _sem_cache is None:
        raise HTTPException(status_code=503, detail="语义缓存未初始化")

    try:
        _sem_cache.clear()
        return CacheClearResponse(
            success=True,
            message="缓存已清空",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空缓存失败: {e}")


@app.post("/api/threshold", tags=["管理"])
async def update_threshold(update: ThresholdUpdate):
    """更新相似度阈值。"""
    if _sem_cache is None:
        raise HTTPException(status_code=503, detail="语义缓存未初始化")

    try:
        old = _sem_cache.similarity_threshold
        _sem_cache.update_threshold(update.threshold)
        return {
            "success": True,
            "message": f"阈值已更新: {old:.4f} → {update.threshold:.4f}",
            "old_threshold": old,
            "new_threshold": update.threshold,
            "route": update.route,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新阈值失败: {e}")


# ==================== HTML 仪表盘 ====================


_HTML_DASHBOARD = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RedisVL Agent Cache 监控面板</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-green: #22c55e;
            --accent-yellow: #eab308;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-purple: #a855f7;
            --border: #334155;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            padding: 20px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }
        header h1 { font-size: 24px; font-weight: 700; }
        header h1 span { color: var(--accent-blue); }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
        }
        .status-ok { background: rgba(34,197,94,0.15); color: var(--accent-green); }
        .status-degraded { background: rgba(234,179,8,0.15); color: var(--accent-yellow); }
        .status-error { background: rgba(239,68,68,0.15); color: var(--accent-red); }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }
        .card h3 {
            font-size: 14px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 16px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(51,65,85,0.5);
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: var(--text-secondary); font-size: 13px; }
        .metric-value { font-size: 16px; font-weight: 600; }
        .metric-value.green { color: var(--accent-green); }
        .metric-value.yellow { color: var(--accent-yellow); }
        .metric-value.red { color: var(--accent-red); }
        .metric-value.blue { color: var(--accent-blue); }
        .big-number {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            padding: 16px 0;
        }
        .timestamp {
            text-align: center;
            color: var(--text-secondary);
            font-size: 12px;
            margin-top: 20px;
        }
        .actions {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        button {
            padding: 8px 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--bg-secondary);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        button:hover { border-color: var(--accent-blue); background: #1e3a5f; }
        button.danger:hover { border-color: var(--accent-red); background: #3f1a1a; }
        button.primary { background: var(--accent-blue); border-color: var(--accent-blue); }
        button.primary:hover { background: #2563eb; }
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 8px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            display: none;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        @media (max-width: 640px) {
            .container { padding: 12px; }
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>📊 <span>Cache</span>Monitor</h1>
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <span id="healthBadge" class="status-badge status-ok">● 检查中...</span>
            <span id="refreshInfo" style="color:var(--text-secondary);font-size:13px;"></span>
            <button onclick="refreshAll()" class="primary">🔄 刷新</button>
        </div>
    </header>

    <!-- 核心指标 -->
    <div class="grid" id="keyMetrics"></div>

    <!-- 详细信息 -->
    <div class="grid">
        <div class="card" id="cacheStats"><h3>🎯 缓存统计</h3><div class="loading">加载中...</div></div>
        <div class="card" id="routerStats"><h3>🧭 路由统计</h3><div class="loading">加载中...</div></div>
        <div class="card" id="systemInfo"><h3>⚙️ 系统信息</h3><div class="loading">加载中...</div></div>
    </div>

    <!-- 操作区域 -->
    <div class="card">
        <h3>⚡ 管理操作</h3>
        <div class="actions" style="margin-top:8px;">
            <button onclick="clearCache()" class="danger">🗑️ 清空缓存</button>
            <button onclick="setThreshold(0.75)">阈值 → 0.75</button>
            <button onclick="setThreshold(0.85)">阈值 → 0.85</button>
            <button onclick="setThreshold(0.95)">阈值 → 0.95</button>
        </div>
    </div>

    <div class="timestamp" id="timestamp">—</div>
</div>

<div id="toast" class="toast"></div>

<script>
let refreshInterval = null;

async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${resp.status}: ${resp.statusText}`);
    return resp.json();
}

function showToast(msg, type='info') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.style.display = 'block';
    toast.style.borderLeft = type === 'error' ? '4px solid var(--accent-red)' :
                            type === 'success' ? '4px solid var(--accent-green)' :
                            '4px solid var(--accent-blue)';
    setTimeout(() => toast.style.display = 'none', 3000);
}

function renderKeyMetrics(data) {
    const c = data.cache || {};
    const r = data.routes || {};
    const s = data.system || {};

    const hitRate = c.hit_rate != null ? (c.hit_rate * 100).toFixed(1) : 'N/A';
    const hitRateClass = hitRate === 'N/A' ? '' :
        parseFloat(hitRate) > 50 ? 'green' : parseFloat(hitRate) > 20 ? 'yellow' : 'red';

    const html = `
        <div class="card">
            <h3>🎯 缓存命中率</h3>
            <div class="big-number ${hitRateClass}">${hitRate}${hitRate !== 'N/A' ? '%' : ''}</div>
        </div>
        <div class="card">
            <h3>📦 缓存条目</h3>
            <div class="big-number blue">${c.vector_count ?? 'N/A'}</div>
        </div>
        <div class="card">
            <h3>🧭 路由查询</h3>
            <div class="big-number">${r.total_queries ?? 'N/A'}</div>
        </div>
        <div class="card">
            <h3>💾 存储后端</h3>
            <div style="text-align:center;padding:16px 0;">
                <div style="font-size:14px;margin-bottom:8px;">
                    Redis: ${s.redis_connected ? '✅' : '❌'}
                    &nbsp;|&nbsp;
                    Milvus: ${s.milvus_available ? '✅' : 'ℹ️ 降级'}
                </div>
            </div>
        </div>
    `;
    document.getElementById('keyMetrics').innerHTML = html;
}

function renderCacheStats(data) {
    const c = data.cache || {};
    const html = `
        <div class="metric"><span class="metric-label">向量数</span><span class="metric-value blue">${c.vector_count ?? 'N/A'}</span></div>
        <div class="metric"><span class="metric-label">元数据数</span><span class="metric-value">${c.metadata_count ?? 'N/A'}</span></div>
        <div class="metric"><span class="metric-label">缓存命中</span><span class="metric-value green">${c.cache_hits ?? 0}</span></div>
        <div class="metric"><span class="metric-label">缓存未命中</span><span class="metric-value red">${c.cache_misses ?? 0}</span></div>
        <div class="metric"><span class="metric-label">总请求数</span><span class="metric-value">${c.total_requests ?? 0}</span></div>
        <div class="metric"><span class="metric-label">向量维度</span><span class="metric-value">${c.dimension ?? 'N/A'}</span></div>
    `;
    document.getElementById('cacheStats').innerHTML = `<h3>🎯 缓存统计</h3>${html}`;
}

function renderRouterStats(data) {
    const r = data.routes || {};
    const routeCounts = r.route_counts || {};
    const routeEntries = Object.entries(routeCounts);
    const total = r.total_queries || 0;

    let html = `
        <div class="metric"><span class="metric-label">总查询</span><span class="metric-value">${total}</span></div>
        <div class="metric"><span class="metric-label">兜底率</span><span class="metric-value ${(r.fallback_rate || 0) > 0.3 ? 'red' : 'green'}">${(r.fallback_rate != null ? (r.fallback_rate * 100).toFixed(1) : 'N/A')}%</span></div>
    `;
    if (routeEntries.length > 0) {
        routeEntries.forEach(([name, count]) => {
            const pct = total > 0 ? (count / total * 100).toFixed(1) : '0.0';
            html += `<div class="metric"><span class="metric-label">${name}</span><span class="metric-value">${count} (${pct}%)</span></div>`;
        });
    }
    document.getElementById('routerStats').innerHTML = `<h3>🧭 路由统计</h3>${html}`;
}

function renderSystemInfo(data) {
    const s = data.system || {};
    const html = `
        <div class="metric"><span class="metric-label">Redis 连接</span><span class="metric-value ${s.redis_connected ? 'green' : 'red'}">${s.redis_connected ? '✅ 正常' : '❌ 断开'}</span></div>
        <div class="metric"><span class="metric-label">Milvus 状态</span><span class="metric-value ${s.milvus_available ? 'green' : 'yellow'}">${s.milvus_available ? '✅ 正常' : '⚠️ 已降级'}</span></div>
        <div class="metric"><span class="metric-label">嵌入维度</span><span class="metric-value">${s.embedding_dim ?? 'N/A'}</span></div>
        <div class="metric"><span class="metric-label">运行时间</span><span class="metric-value">${s.uptime || 'N/A'}</span></div>
    `;
    document.getElementById('systemInfo').innerHTML = `<h3>⚙️ 系统信息</h3>${html}`;
}

async function refreshAll() {
    try {
        const data = await fetchJSON('/api/stats');
        renderKeyMetrics(data);
        renderCacheStats(data);
        renderRouterStats(data);
        renderSystemInfo(data);
        document.getElementById('timestamp').textContent =
            '上次更新: ' + new Date().toLocaleTimeString('zh-CN');

        // 健康检查
        const health = await fetchJSON('/api/health');
        const badge = document.getElementById('healthBadge');
        badge.className = 'status-badge status-' + health.status;
        badge.textContent = '● ' + (health.status === 'ok' ? '系统正常' :
                                     health.status === 'degraded' ? '部分降级' : '服务异常');
        showToast('已刷新', 'success');
    } catch (e) {
        showToast('刷新失败: ' + e.message, 'error');
    }
}

async function clearCache() {
    if (!confirm('确定清空所有缓存数据？此操作不可恢复！')) return;
    try {
        const resp = await fetchJSON('/api/cache/clear');
        showToast(resp.message, 'success');
        await refreshAll();
    } catch (e) {
        showToast('清空失败: ' + e.message, 'error');
    }
}

async function setThreshold(val) {
    try {
        const resp = await fetch('/api/threshold', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({threshold: val}),
        });
        const data = await resp.json();
        if (resp.ok) {
            showToast(data.message, 'success');
            await refreshAll();
        } else {
            showToast('设置失败: ' + (data.detail || resp.statusText), 'error');
        }
    } catch (e) {
        showToast('设置失败: ' + e.message, 'error');
    }
}

// 页面加载后自动刷新，之后每 10 秒自动刷新
window.onload = () => {
    refreshAll();
    refreshInterval = setInterval(refreshAll, 10000);
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def dashboard():
    """HTML 仪表盘页面。"""
    return HTMLResponse(content=_HTML_DASHBOARD)


# ==================== 内部方法 ====================


def _measure_latency(fn) -> float:
    """测量函数执行延迟（毫秒）。

    Args:
        fn: 要执行的函数。

    Returns:
        延迟毫秒数。
    """
    start = time.time()
    try:
        fn()
        return round((time.time() - start) * 1000, 2)
    except Exception:
        return -1.0


def _collect_cache_stats() -> Dict[str, Any]:
    """收集缓存统计信息。

    Returns:
        统计字典。
    """
    if _cache is None:
        return {"error": "缓存未初始化"}

    try:
        stats = _cache.get_stats()
        return stats
    except Exception as e:
        return {"error": str(e)}


def _collect_route_stats() -> Dict[str, Any]:
    """收集路由统计信息。

    Returns:
        统计字典。
    """
    if _router is None:
        return {"note": "路由未初始化"}

    try:
        stats = _router.get_route_stats()
        return {
            "total_queries": stats.total_queries,
            "route_counts": dict(stats.route_counts),
            "fallback_rate": stats.fallback_rate,
        }
    except Exception as e:
        return {"error": str(e)}


def _collect_session_stats() -> Dict[str, Any]:
    """收集会话统计信息。

    Returns:
        统计字典。
    """
    if _history is None:
        return {"note": "会话历史未初始化"}

    try:
        sessions = _history.list_sessions()
        return {
            "active_sessions": len(sessions),
            "sessions": sessions,
        }
    except Exception as e:
        return {"error": str(e)}


def _collect_system_info() -> Dict[str, Any]:
    """收集系统信息。

    Returns:
        系统信息字典。
    """
    uptime_seconds = time.time() - _start_time
    uptime_str = str(timedelta(seconds=int(uptime_seconds)))

    info: Dict[str, Any] = {
        "uptime": uptime_str,
        "uptime_seconds": round(uptime_seconds, 2),
        "redis_connected": False,
        "milvus_available": False,
        "embedding_dim": None,
    }

    if _cache:
        info["redis_connected"] = True
        info["milvus_available"] = _cache.milvus_available
        info["embedding_dim"] = _cache.dim

    return info


# ==================== 直接运行 ====================


def main():
    """直接启动监控服务。"""
    import uvicorn

    host = os.environ.get("MONITOR_HOST", "0.0.0.0")
    port = int(os.environ.get("MONITOR_PORT", "8000"))

    print(f"🚀 RedisVL Agent Cache 监控面板")
    print(f"   → http://{host}:{port}")
    print(f"   → http://localhost:{port}  (浏览器打开)")
    print(f"\n   停止: Ctrl+C")

    uvicorn.run(
        "src.monitoring:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
