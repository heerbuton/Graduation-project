<script setup>
import { computed } from 'vue'

const props = defineProps({
  pipelineStatus: {
    type: String,
    required: true
  },
  activeTab: {
    type: String,
    required: true
  }
})

const emit = defineEmits(['update:activeTab'])

const steps = [
  { id: 'cv', label: '特征提取', name: 'YOLO Vision' },
  { id: 'topology', label: '拓扑序列', name: 'Topology' },
  { id: 'llm', label: '乐理推理', name: 'LLM Agent' },
  { id: 'xml', label: '打谱渲染', name: 'Render' }
]

const statusWeight = {
  idle: -1,
  uploading: -1,
  cv: 0,
  topology: 1,
  llm: 2,
  xml: 3,
  success: 4,
  error: -1
}

const currentWeight = computed(() => statusWeight[props.pipelineStatus] ?? -1)

const isStepUnlocked = (index) => {
  return currentWeight.value >= index
}

const handleStepClick = (index, stepId) => {
  if (isStepUnlocked(index)) {
    emit('update:activeTab', stepId)
  }
}
</script>

<template>
  <div class="w-full flex justify-between items-center relative py-6 px-[5%]">
    <!-- 连接线 -->
    <div class="absolute left-[5%] right-[5%] top-1/2 -translate-y-1/2 h-1 bg-white/10 rounded-full z-0">
       <div class="h-full bg-gradient-to-r from-primary to-accent transition-all duration-1000 ease-out rounded-full"
            :style="{ width: currentWeight >= 0 ? `${(Math.min(currentWeight, 3) / 3) * 100}%` : '0%' }">
       </div>
    </div>
    
    <!-- 节点指示器 -->
    <div v-for="(step, index) in steps" :key="step.id" 
         class="relative z-10 flex flex-col items-center gap-4 transition-all duration-300 group"
         :class="[
           isStepUnlocked(index) ? 'cursor-pointer hover:-translate-y-2' : 'cursor-not-allowed opacity-50 grayscale',
           activeTab === step.id ? 'scale-110 drop-shadow-[0_0_15px_rgba(20,184,166,0.6)]' : ''
         ]"
         @click="handleStepClick(index, step.id)">
      
      <!-- 圆圈 -->
      <div class="w-14 h-14 rounded-full border-[3px] flex items-center justify-center text-xl font-bold shadow-xl transition-all duration-500"
           :class="[
              currentWeight > index ? 'bg-primary border-primary text-white shadow-primary/50' : 
              currentWeight === index ? 'bg-primary border-primary text-white shadow-primary/80 animate-pulse' : 
              'bg-slate-800 border-slate-600 text-slate-400 group-hover:bg-slate-700'
           ]">
        <!-- 完成打勾标记 -->
        <svg v-if="currentWeight > index" class="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" /></svg>
        <span v-else>{{ index + 1 }}</span>
      </div>

      <!-- 文字信息卡 -->
      <div class="text-center bg-black/50 backdrop-blur-xl px-5 py-2.5 rounded-xl border border-white/10 shadow-lg group-hover:border-primary/50 transition-colors">
        <h3 class="text-sm font-semibold tracking-widest whitespace-nowrap"
            :class="isStepUnlocked(index) ? 'text-slate-100' : 'text-slate-500'">
          {{ step.label }}
        </h3>
        <p class="text-[10px] font-mono mt-1 tracking-widest uppercase"
           :class="activeTab === step.id ? 'text-primary' : 'text-slate-500'">
          {{ step.name }}
        </p>
      </div>
      
    </div>
  </div>
</template>
