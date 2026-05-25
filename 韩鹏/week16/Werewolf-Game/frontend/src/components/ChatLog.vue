<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'

interface LogEntry {
  type: string; round: number; phase: string;
  actor_id?: number; target_id?: number; name?: string;
  content?: string; speech?: string; result?: string;
  message?: string; winner?: string; role?: string;
}

const props = defineProps<{ logs: LogEntry[]; players: { id: number; name: string }[] }>()
const container = ref<HTMLElement>()

watch(() => props.logs.length, () => {
  nextTick(() => {
    if (container.value) container.value.scrollTop = container.value.scrollHeight
  })
})

function getPlayerName(id?: number) {
  if (id == null) return '?'
  return props.players.find(p => p.id === id)?.name || `Player ${id}`
}

function formatEntry(entry: LogEntry): string {
  switch (entry.type) {
    case 'werewolf_kill': return `🌙 Wolves choose target: ${getPlayerName(entry.target_id)}`
    case 'seer_check': return `🔮 Seer checks ${getPlayerName(entry.target_id)}: ${entry.result}`
    case 'witch_save': return `💚 Witch saves ${getPlayerName(entry.target_id)}`
    case 'witch_poison': return `🧪 Witch poisons ${getPlayerName(entry.target_id)}`
    case 'player_death': return `☠️ ${entry.name || getPlayerName(entry.player_id)} dies (${entry.role})`
    case 'speech': return `💬 [${entry.name || getPlayerName(entry.actor_id)}]: ${entry.content || entry.speech}`
    case 'vote_eliminate': return `🚫 ${entry.name || getPlayerName(entry.player_id)} eliminated (${entry.role})`
    case 'vote_tie': return `🤝 Tie - no elimination`
    case 'game_over': return `🏆 Game Over! ${entry.winner} Wins!`
    default: return JSON.stringify(entry)
  }
}

function typeClass(type: string): string {
  if (type === 'speech') return 'speech'
  if (type.includes('death') || type.includes('eliminate')) return 'death'
  if (type.includes('game_over')) return 'winner'
  return 'action'
}
</script>

<template>
  <div class="chat-log" ref="container">
    <div v-if="logs.length === 0" class="empty">Game events will appear here...</div>
    <div v-for="(entry, i) in logs" :key="i" :class="['log-entry', typeClass(entry.type)]">
      <span class="round-badge">R{{ entry.round }}</span>
      {{ formatEntry(entry) }}
    </div>
  </div>
</template>

<style scoped>
.chat-log { flex: 1; background: #1a1a2e; border-radius: 10px; padding: 12px; color: #ddd; max-height: 70vh; overflow-y: auto; font-size: 0.85rem; }
.empty { color: #666; text-align: center; padding: 40px 0; }
.log-entry { padding: 4px 8px; margin: 2px 0; border-radius: 4px; line-height: 1.5; }
.log-entry.speech { background: #16213e; border-left: 3px solid #e94560; }
.log-entry.death { background: #2d132c; border-left: 3px solid #c0392b; color: #e74c3c; }
.log-entry.winner { background: #1e3a1e; border-left: 3px solid #ffd700; font-weight: bold; text-align: center; }
.log-entry.action { color: #aaa; }
.round-badge { background: #0f3460; color: #aaa; padding: 0 4px; border-radius: 3px; font-size: 0.7rem; margin-right: 4px; }
</style>
