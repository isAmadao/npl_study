"""AI狼人杀 - FastAPI 服务端 Phase 3"""
from __future__ import annotations

import sys, os, json, asyncio, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from werewolf.engine import WerewolfGame, Phase, Team
from werewolf.agent import LLMAgent

app = FastAPI(title="AI Werewolf Game")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 游戏实例管理
games: dict[str, dict] = {}

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/games")
async def create_game(use_llm: bool = False):
    """创建新游戏"""
    game_id = f"game_{int(time.time())}"

    agent = None
    if use_llm:
        llm_agent = LLMAgent()
        def get_action(state, player):
            return llm_agent.get_action(state, player)
        agent = get_action

    game = WerewolfGame(seed=None, agent=agent)
    games[game_id] = {"game": game, "ws_clients": [], "use_llm": use_llm}

    # 启动游戏任务
    asyncio.create_task(run_game(game_id))

    return {
        "game_id": game_id,
        "players": [{"id": p.id, "name": p.name} for p in game.state.players],
        "use_llm": use_llm,
    }

@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    """查询游戏状态"""
    if game_id not in games:
        return {"error": "Game not found"}
    g = games[game_id]
    state = g["game"].state
    return {
        "game_id": game_id,
        "phase": state.phase.value,
        "round": state.round_num,
        "winner": state.winner.value if state.winner else None,
        "players": [{"id": p.id, "name": p.name, "alive": p.alive, "role": p.role.label if not p.alive else "?"} for p in state.players],
        "event_log": state.event_log[-50:],
    }

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(ws: WebSocket, game_id: str):
    await ws.accept()
    if game_id in games:
        games[game_id]["ws_clients"].append(ws)
    try:
        while True:
            await ws.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        if game_id in games:
            games[game_id]["ws_clients"].remove(ws)

async def run_game(game_id: str):
    """后台运行游戏，推送事件到 WebSocket 客户端"""
    g = games[game_id]
    game: WerewolfGame = g["game"]
    clients: list[WebSocket] = g["ws_clients"]

    # 发送初始状态
    await broadcast(game_id, {"type": "game_start", "players": [
        {"id": p.id, "name": p.name, "alive": True} for p in game.state.players
    ]})

    # 逐步运行游戏
    while game.state.phase != Phase.GAME_OVER:
        # 捕获当前状态用于推送
        phase_before = game.state.phase
        game._step(verbose=False)

        # 构建事件
        event = build_event(game)
        await broadcast(game_id, event)
        await asyncio.sleep(0.5)  # 节奏控制

    # 游戏结束
    await broadcast(game_id, {
        "type": "game_end",
        "winner": game.state.winner.value,
        "players": [{"id": p.id, "name": p.name, "role": p.role.label, "alive": p.alive} for p in game.state.players],
    })

def build_event(game: WerewolfGame) -> dict:
    """从最新日志构建事件"""
    logs = game.state.event_log
    if not logs:
        return {"type": "tick"}
    latest = logs[-1]
    event = {"type": latest["type"], "round": game.state.round_num, "phase": game.state.phase.value, **latest}
    event["players"] = [{"id": p.id, "name": p.name, "alive": p.alive} for p in game.state.players]
    return event

async def broadcast(game_id: str, event: dict):
    """向所有 WebSocket 客户端广播事件"""
    if game_id not in games:
        return
    dead_clients = []
    for ws in games[game_id]["ws_clients"]:
        try:
            await ws.send_json(event)
        except Exception:
            dead_clients.append(ws)
    for ws in dead_clients:
        games[game_id]["ws_clients"].remove(ws)

@app.get("/")
async def root():
    return {"message": "AI Werewolf Game Server", "docs": "/docs"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8765"))
    print(f"Starting server on http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
