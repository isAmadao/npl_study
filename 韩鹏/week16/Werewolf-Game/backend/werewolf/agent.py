"""AI狼人杀 - Agent System Phase 2"""
from __future__ import annotations
import os, json, re
from typing import Optional
from werewolf.engine import Role, Team, ActionType, Action, Phase, GameState, Player, RuleAgent

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    def load_dotenv(): pass

try:
    from openai import OpenAI; _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False; OpenAI = None

class LLMClient:
    def __init__(self):
        self.client = None
        if _OPENAI_AVAILABLE and OpenAI is not None:
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY", "sk-placeholder"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def chat(self, messages, temperature=0.7, max_tokens=512):
        if self.client is None:
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[LLM Error] {e}")
            return ""

class InfoFilter:
    @staticmethod
    def visible_state(state, player):
        info = {"player_id": player.id, "player_name": player.name,
                "player_role": player.role.label, "alive": player.alive,
                "round": state.round_num, "phase": state.phase.value,
                "alive_players": [f"{p.name}(ID:{p.id})" for p in state.alive_players()]}
        if player.role == Role.WEREWOLF:
            info["werewolf_allies"] = [p.name for p in state.players if p.role == Role.WEREWOLF and p.id != player.id]
            info["werewolf_target_this_round"] = state.werewolf_target
        if player.role == Role.SEER:
            checked = []
            for log in state.event_log:
                if log.get("type") == "seer_check" and log.get("actor_id") == player.id:
                    checked.append({"target_id": log["target_id"], "result": log.get("result")})
            info["checked_players"] = checked
        if player.role == Role.WITCH:
            info["has_antidote"] = not state.witch_save_used
            info["has_poison"] = not state.witch_poison_used
            info["killed_tonight"] = state.werewolf_target
        public_log = []
        for log in state.event_log:
            if log.get("type") in ("player_death", "vote_eliminate", "vote_tie", "speech"):
                public_log.append(log)
        info["public_log"] = public_log[-20:]
        return info

SYSTEM_PROMPTS = {
    Role.WEREWOLF: "You are a WEREWOLF in Werewolf game. Your goal: eliminate all villagers. Hide your identity during the day. Coordinate with other werewolves secretly. Output valid JSON for actions.",
    Role.SEER: "You are the SEER in Werewolf game. Your goal: help villagers find all werewolves. Each night you can check one player's identity. Share findings during the day. Output valid JSON for actions.",
    Role.WITCH: "You are the WITCH in Werewolf game. Your goal: help villagers win. You have 1 antidote (save) and 1 poison (kill). Use wisely. Output valid JSON for actions.",
    Role.VILLAGER: "You are a VILLAGER in Werewolf game. Your goal: find and eliminate werewolves. Listen carefully to speeches, find contradictions. Output valid JSON for actions.",
}

ACTION_PROMPTS = {
    Phase.NIGHT_WEREWOLF: 'Night phase. Choose a target to kill (not a werewolf). Output JSON: {"reasoning": "...", "action": "kill", "target_id": ID}',
    Phase.NIGHT_SEER: 'Night phase. Choose a player to check. Output JSON: {"reasoning": "...", "action": "check", "target_id": ID}',
    Phase.NIGHT_WITCH: 'Night phase. Use potion or skip. Output JSON: {"reasoning": "...", "action": "save|poison|skip", "target_id": ID|null}',
    Phase.DAY_DISCUSS: 'Day discussion. Make a speech (2-4 sentences in Chinese). Output JSON: {"reasoning": "...", "action": "speak", "speech": "your speech"}',
    Phase.DAY_VOTE: 'Vote to eliminate. Output JSON: {"reasoning": "...", "action": "vote", "target_id": ID}',
}

ROLE_SPEAK_HINTS = {
    Role.WEREWOLF: "Remember: you are a werewolf, pretend to be a villager.",
    Role.SEER: "You are the seer, report your check results.",
    Role.WITCH: "You are the witch, you may reveal or hide your identity.",
    Role.VILLAGER: "You are a villager, share your analysis.",
}

class LLMAgent:
    def __init__(self, llm=None, memory=None):
        self.llm = llm or LLMClient()
        self.memory = memory

    def get_action(self, state, player):
        visible = InfoFilter.visible_state(state, player)
        base_prompt = SYSTEM_PROMPTS.get(player.role, SYSTEM_PROMPTS[Role.VILLAGER])
        experience = ""
        if self.memory:
            experience = self.memory.get_experience_for_role(player.role)
        system_prompt = base_prompt + experience

        action_prompt = ACTION_PROMPTS.get(state.phase, "")
        if "{role_hint}" in action_prompt:
            hint = ROLE_SPEAK_HINTS.get(player.role, "")
            action_prompt = action_prompt.replace("{role_hint}", hint)

        user_prompt = f"Game state:\n{json.dumps(visible, ensure_ascii=False, indent=2)}\n\n{action_prompt}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.chat(messages)
        action = self._parse_response(response, player.id)
        if action is None:
            action = RuleAgent.get_action(state, player)
        return action

    def _parse_response(self, response, player_id):
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())
            action_map = {"kill": ActionType.KILL, "check": ActionType.CHECK,
                          "save": ActionType.SAVE, "poison": ActionType.POISON,
                          "vote": ActionType.VOTE, "speak": ActionType.SPEAK, "skip": ActionType.SKIP}
            action_type = action_map.get(data.get("action", "skip"), ActionType.SKIP)
            target_id = data.get("target_id")
            content = data.get("speech", data.get("reasoning", ""))
            return Action(actor_id=player_id, action_type=action_type, target_id=target_id, content=content)
        except (json.JSONDecodeError, KeyError):
            return None

def create_llm_agent(memory=None):
    return LLMAgent(memory=memory)
