<template>
  <div class="relative flex items-center justify-between w-full max-w-2xl mx-auto py-6">
    <!-- Connecting Line -->
    <div class="absolute left-0 top-1/2 -translate-y-1/2 w-full h-1 bg-surface2 rounded-full z-0"></div>
    <div 
      class="absolute left-0 top-1/2 -translate-y-1/2 h-1 rounded-full z-0 transition-all duration-1000 ease-in-out"
      :class="['bg-gradient-to-r from-primary to-accent', { 'w-0': currentStepIndex === 0, 'w-1/3': currentStepIndex === 1, 'w-2/3': currentStepIndex === 2, 'w-full': currentStepIndex >= 3 }]"
    ></div>

    <!-- Steps -->
    <div 
      v-for="(step, index) in steps" 
      :key="step.id"
      class="relative z-10 flex flex-col items-center gap-2"
    >
      <div 
        class="w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-500 shadow-lg"
        :class="[
          index < currentStepIndex ? 'bg-primary border-primary text-white shadow-primary/30' : 
          index === currentStepIndex ? 'bg-surface1 border-accent text-accent shadow-accent/40 animate-pulse-slow' : 
          'bg-surface1 border-surface2 text-slate-500'
        ]"
      >
        <!-- Icon slot (simplified logic based on text) -->
        <span class="font-bold text-sm">{{ index + 1 }}</span>
      </div>
      <div 
        class="text-xs font-medium transition-colors duration-300"
        :class="[
          index <= currentStepIndex ? 'text-slate-200' : 'text-slate-500'
        ]"
      >
        {{ step.label }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  pipelineStatus: {
    type: String,
    required: true // 'idle', 'uploading', 'cv', 'topology', 'llm', 'xml', 'success', 'error'
  }
})

const steps = [
  { id: 'cv', label: '视觉抓取 (YOLO)' },
  { id: 'topology', label: '拓扑序列化' },
  { id: 'llm', label: '大模型打谱' },
  { id: 'xml', label: 'XML 转换' }
]

const currentStepIndex = computed(() => {
  if (['idle', 'uploading'].includes(props.pipelineStatus)) return -1;
  if (props.pipelineStatus === 'cv') return 0;
  if (props.pipelineStatus === 'topology') return 1;
  if (props.pipelineStatus === 'llm') return 2;
  if (['xml', 'success'].includes(props.pipelineStatus)) return 3;
  return -1; // error
})
</script>
