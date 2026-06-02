"""FastAPI Server — 狼人杀游戏 HTTP API"""

import uuid
from typing import Dict, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    CreateGameRequest,
    StepResponse,
    GameStatusResponse,
    GameSummaryResponse,
    ConfigInfo,
)
from engine.game_engine import (
    GameEngine,
    get_role_config,
    shuffle_roles,
    ROLE_CONFIGS,
)

app = FastAPI(title="Werewolf Game API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory game store
games: Dict[str, GameEngine] = {}


@app.get("/")
async def root():
    return {"message": "Werewolf Game API", "version": "1.0.0"}


@app.post("/games")
async def create_game(req: CreateGameRequest) -> GameSummaryResponse:
    """创建并初始化一局新游戏"""
    config_name = req.config_name
    if config_name not in ROLE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown config: {config_name}")

    role_assignment = get_role_config(config_name)
    player_count = len(role_assignment)

    # 玩家名称
    player_names = req.player_names
    if player_names is None:
        player_names = [f"玩家{i+1}" for i in range(player_count)]
    if len(player_names) != player_count:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {player_count} players, got {len(player_names)}",
        )

    # 打乱角色
    if req.shuffle:
        role_assignment = shuffle_roles(role_assignment)

    # 创建引擎
    engine = GameEngine(player_names)
    await engine.initialize(role_assignment, req.player_styles)

    game_id = str(uuid.uuid4())[:8]
    engine.game_id = game_id
    games[game_id] = engine

    return GameSummaryResponse(
        game_id=game_id,
        phase=engine.game_state.phase.value,
        day_number=engine.game_state.day_number,
        alive_count=len(engine.game_state.get_alive_players()),
        is_game_over=False,
    )


@app.post("/games/{game_id}/step")
async def step_game(game_id: str) -> StepResponse:
    """推进一局游戏的一个阶段"""
    engine = games.get(game_id)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    result = await engine.step()

    # 从 step_data 提取 summaries（summary 阶段的数据）
    summaries = result.get("step_data", {}).get("summaries", []) or []

    return StepResponse(
        phase=result["phase"],
        day_number=result["day_number"],
        step_data=result["step_data"],
        players=result["players"],
        dialogues=result["dialogues"],
        deaths=result["deaths"],
        is_game_over=result["is_game_over"],
        winner=result.get("winner"),
        summaries=summaries,
    )


@app.get("/games/{game_id}")
async def get_game_status(game_id: str) -> GameStatusResponse:
    """获取游戏当前状态"""
    engine = games.get(game_id)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    return GameStatusResponse(
        game_id=game_id,
        phase=engine.game_state.phase.value,
        day_number=engine.game_state.day_number,
        players=[p.to_dict() for p in engine.game_state.players],
        dialogues=list(engine.game_state.dialogues),
        death_records=list(engine.death_records),
        winner=engine.game_state.get_winner(),
        is_game_over=engine.game_state.is_game_over(),
        summaries=list(engine.summaries),
    )


@app.get("/games/{game_id}/summaries")
async def get_game_summaries(game_id: str) -> Dict[str, Any]:
    """获取游戏的玩家总结数据"""
    engine = games.get(game_id)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    return {"game_id": game_id, "summaries": list(engine.summaries)}


@app.get("/games")
async def list_games() -> Dict[str, GameSummaryResponse]:
    """列出所有游戏"""
    return {
        gid: GameSummaryResponse(
            game_id=gid,
            phase=engine.game_state.phase.value,
            day_number=engine.game_state.day_number,
            alive_count=len(engine.game_state.get_alive_players()),
            is_game_over=engine.game_state.is_game_over(),
        )
        for gid, engine in games.items()
    }


@app.delete("/games/{game_id}")
async def delete_game(game_id: str) -> Dict[str, str]:
    """删除一局游戏"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    del games[game_id]
    return {"message": f"Game {game_id} deleted"}


@app.get("/configs")
async def list_configs() -> Dict[str, ConfigInfo]:
    """列出可用角色配置"""
    return {
        name: ConfigInfo(
            name=cfg["name"],
            description=cfg["description"],
            player_count=len(cfg["roles"]),
        )
        for name, cfg in ROLE_CONFIGS.items()
    }


# ============================================================
# 排行榜 & 进化报告 API
# ============================================================


@app.get("/leaderboard")
async def get_leaderboard() -> Dict[str, Any]:
    """获取Agent能力排行榜"""
    try:
        from evaluation.evaluator import Leaderboard
        lb = Leaderboard()
        return {
            "model_stats": lb.get_model_stats(),
            "top_players": lb.get_top_players(limit=10),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evolution/report")
async def get_evolution_report(role_type: Optional[str] = None) -> Dict[str, Any]:
    """获取进化报告

    Args:
        role_type: 角色类型（可选），不传则返回所有角色报告
    """
    try:
        from evaluation.evolution_tracker import EvolutionTracker
        tracker = EvolutionTracker()
        if role_type:
            return {"role": role_type, "report": tracker.get_evolution_report(role_type)}
        else:
            return tracker.get_all_reports()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evolution/version-comparison/{role_type}")
async def get_version_comparison(role_type: str) -> Dict[str, Any]:
    """获取某角色各版本的对比"""
    try:
        from evaluation.evolution_tracker import EvolutionTracker
        tracker = EvolutionTracker()
        return {
            "role": role_type,
            "comparisons": tracker.get_version_comparison(role_type),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evolution/analyze")
async def trigger_evolution_analysis() -> Dict[str, Any]:
    """手动触发进化分析（调用EvolutionAgent）

    会分析所有积累了足够经验的角色，返回进化建议。
    """
    try:
        from memory.experience import get_all_role_experiences
        from agent.evolution_agent import EvolutionAgent
        from evaluation.evolution_tracker import EvolutionTracker

        all_exps = get_all_role_experiences()
        roles_ready = {role: exps for role, exps in all_exps.items() if len(exps) >= 3}

        if not roles_ready:
            return {
                "message": "没有角色积累了足够的经验（至少需要3局）",
                "results": {},
            }

        agent = EvolutionAgent()
        results = await agent.batch_analyze(roles_ready)

        # 记录进化建议到tracker
        tracker = EvolutionTracker()
        for role_type, advice in results.items():
            if "error" not in advice:
                exps = roles_ready[role_type]
                win_count = sum(1 for e in exps if e.get("is_winner"))
                tracker.record_evolution(
                    role_type=role_type,
                    version=len(exps),
                    metrics_before={"win_rate": win_count / max(1, len(exps))},
                    metrics_after={"win_rate": win_count / max(1, len(exps))},
                    evolution_advice=advice,
                )

        return {"message": f"进化分析完成，{len(results)}个角色已分析", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiences/{role_type}")
async def get_experiences(role_type: str) -> Dict[str, Any]:
    """获取某角色的历史经验"""
    try:
        from memory.experience import load_experiences
        exps = load_experiences(role_type)
        return {"role": role_type, "count": len(exps), "experiences": exps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
