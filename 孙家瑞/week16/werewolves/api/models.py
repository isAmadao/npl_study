"""API 数据模型"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class CreateGameRequest(BaseModel):
    config_name: str = "standard_6"
    player_names: Optional[List[str]] = None
    shuffle: bool = True
    player_styles: Optional[Dict[str, str]] = None


class GameSummaryResponse(BaseModel):
    game_id: str
    phase: str
    day_number: int
    alive_count: int
    is_game_over: bool


class StepResponse(BaseModel):
    phase: str
    day_number: int
    step_data: Dict[str, Any]
    players: List[Dict[str, Any]]
    dialogues: List[Dict[str, Any]]
    deaths: List[Dict[str, Any]]
    is_game_over: bool
    winner: Optional[str] = None
    summaries: List[Dict[str, Any]] = []


class GameStatusResponse(BaseModel):
    game_id: str
    phase: str
    day_number: int
    players: List[Dict[str, Any]]
    dialogues: List[Dict[str, Any]]
    death_records: List[Dict[str, Any]]
    winner: Optional[str] = None
    is_game_over: bool
    summaries: List[Dict[str, Any]] = []


class ConfigInfo(BaseModel):
    name: str
    description: str
    player_count: int