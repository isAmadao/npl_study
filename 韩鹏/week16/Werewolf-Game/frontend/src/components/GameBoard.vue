<script setup lang="ts">
defineProps<{
  players: { id: number; name: string; alive: boolean; role?: string }[]
  winner: string
  phase: string
  round: number
}>()
</script>

<template>
  <div class="board">
    <div class="board-title">
      Round {{ round }} &middot; {{ phase || 'Waiting...' }}
    </div>
    <div v-if="winner" class="winner-banner">
      {{ winner }} Wins!
    </div>
    <div class="player-grid">
      <div v-for="p in players" :key="p.id"
           :class="['player-card', { dead: !p.alive }]">
        <div class="avatar">{{ p.name[0] }}</div>
        <div class="name">{{ p.name }}</div>
        <div class="role">{{ p.alive ? '?' : (p.role || '?') }}</div>
        <div class="status-dot" :class="p.alive ? 'alive' : 'dead'"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.board { background: #1a1a2e; border-radius: 10px; padding: 16px; color: #eee; }
.board-title { font-size: 0.9rem; color: #aaa; margin-bottom: 12px; }
.winner-banner { background: #ffd700; color: #1a1a2e; text-align: center; padding: 8px; border-radius: 6px; margin-bottom: 12px; font-weight: bold; font-size: 1.1rem; }
.player-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.player-card { background: #16213e; border-radius: 8px; padding: 10px; text-align: center; position: relative; border: 2px solid transparent; }
.player-card.dead { opacity: 0.4; border-color: #c0392b; }
.player-card.dead .avatar { background: #666; }
.avatar { width: 32px; height: 32px; border-radius: 50%; background: #0f3460; margin: 0 auto 6px; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; color: #e94560; }
.name { font-size: 0.85rem; font-weight: 600; }
.role { font-size: 0.7rem; color: #aaa; margin-top: 2px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; position: absolute; top: 6px; right: 6px; }
.status-dot.alive { background: #2ecc71; }
.status-dot.dead { background: #c0392b; }
</style>
