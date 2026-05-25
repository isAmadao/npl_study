<script setup lang="ts">
import { ref } from 'vue'
import GameBoard from './components/GameBoard.vue'
import ChatLog from './components/ChatLog.vue'
import GameControl from './components/GameControl.vue'

interface Player { id: number; name: string; alive: boolean; role?: string }
interface LogEntry {
  type: string; round: number; phase: string;
  actor_id?: number; target_id?: number; name?: string;
  content?: string; speech?: string; result?: string;
  message?: string; winner?: string; role?: string;
}

const gameId = ref('')
const connected = ref(false)
const phase = ref('')
const round = ref(0)
const winner = ref('')
const players = ref<Player[]>([])
const logs = ref<LogEntry[]>([])
let ws: WebSocket | null = null

function startGame(useLLM = false) {
  logs.value = []
  winner.value = ''
  fetch('/api/games' + (useLLM ? '?use_llm=true' : ''), { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      gameId.value = data.game_id
      players.value = data.players.map((p: any) => ({ ...p, alive: true }))
      connectWS(data.game_id)
    })
}

function connectWS(id: string) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  ws = new WebSocket(`${protocol}//${location.host}/ws/${id}`)

  ws.onopen = () => { connected.value = true }
  ws.onmessage = (e) => {
    const evt = JSON.parse(e.data)
    handleEvent(evt)
  }
  ws.onclose = () => { connected.value = false }
}

function handleEvent(evt: any) {
  switch (evt.type) {
    case 'game_start':
      players.value = evt.players.map((p: any) => ({ ...p, alive: true }))
      break
    case 'game_end':
      winner.value = evt.winner
      players.value = evt.players
      break
    case 'player_death':
    case 'vote_eliminate':
      const dead = players.value.find(p => p.id === evt.player_id)
      if (dead) dead.alive = false
      if (evt.role) dead!.role = evt.role
      logs.value.push(evt)
      break
    default:
      logs.value.push(evt)
  }
  if (evt.phase) phase.value = evt.phase
  if (evt.round) round.value = evt.round
  if (evt.players) {
    evt.players.forEach((p: any) => {
      const local = players.value.find(lp => lp.id === p.id)
      if (local) local.alive = p.alive
    })
  }
}
</script>

<template>
  <div class="app">
    <header class="header">
      <h1>🐺 AI Werewolf Game</h1>
      <span v-if="connected" class="status green">Connected</span>
      <span v-else class="status">Disconnected</span>
    </header>

    <div class="main">
      <div class="left-panel">
        <GameControl @start-game="startGame" :disabled="connected" />
        <GameBoard :players="players" :winner="winner" :phase="phase" :round="round" />
      </div>
      <ChatLog :logs="logs" :players="players" />
    </div>
  </div>
</template>

<style scoped>
.app { max-width: 1100px; margin: 0 auto; padding: 16px; font-family: system-ui, sans-serif; }
.header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.header h1 { margin: 0; font-size: 1.5rem; }
.status { padding: 2px 10px; border-radius: 10px; font-size: 0.8rem; background: #eee; color: #888; }
.status.green { background: #d4edda; color: #155724; }
.main { display: flex; gap: 16px; }
.left-panel { flex: 0 0 340px; display: flex; flex-direction: column; gap: 12px; }
</style>
