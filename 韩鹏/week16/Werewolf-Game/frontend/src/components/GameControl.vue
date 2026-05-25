<script setup lang="ts">
defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ 'startGame': [useLLM: boolean] }>()
</script>

<template>
  <div class="control">
    <div class="control-title">Game Control</div>
    <button class="btn btn-rule" @click="emit('startGame', false)" :disabled="disabled">
      New Game (Rule AI)
    </button>
    <button class="btn btn-llm" @click="emit('startGame', true)" :disabled="disabled">
      New Game (LLM AI)
    </button>
    <p class="hint">LLM mode requires DeepSeek API key in .env</p>
  </div>
</template>

<style scoped>
.control { background: #1a1a2e; border-radius: 10px; padding: 16px; color: #eee; }
.control-title { font-size: 0.9rem; color: #aaa; margin-bottom: 12px; }
.btn { display: block; width: 100%; padding: 10px; border: none; border-radius: 6px; margin: 6px 0; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: opacity 0.2s; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-rule { background: #0f3460; color: #e94560; }
.btn-rule:hover:not(:disabled) { background: #16213e; }
.btn-llm { background: #e94560; color: #fff; }
.btn-llm:hover:not(:disabled) { background: #c0392b; }
.hint { font-size: 0.7rem; color: #666; margin-top: 8px; }
</style>
