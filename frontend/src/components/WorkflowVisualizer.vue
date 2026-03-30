<template>
  <div class="w-full flex flex-col gap-4">
    <div class="relative glass-panel p-6 overflow-hidden min-h-[300px]">
      
      <!-- Background Ornament (Classical Touch) -->
      <div class="absolute -right-10 -bottom-10 opacity-5 pointer-events-none">
        <svg class="w-64 h-64 text-accent" fill="currentColor" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="45" stroke="currentColor" stroke-width="2" fill="none" />
          <path d="M50 5 a45 45 0 0 1 0 90 a45 45 0 0 1 0 -90" stroke="currentColor" stroke-width="1" stroke-dasharray="2,2" fill="none" transform="rotate(45 50 50)"/>
        </svg>
      </div>

      <!-- CV Stage -->
      <transition enter-active-class="animate-slide-up" leave-active-class="transition-opacity duration-300 opacity-0">
        <div v-show="activeStage === 'cv'" class="absolute inset-x-6 top-6">
          <div class="flex items-center gap-3 mb-4">
            <span class="p-2 bg-primary/20 text-primary rounded-lg border border-primary/30">
              <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </span>
            <h3 class="text-lg font-semibold text-slate-100 font-guqin tracking-widest">视觉特征抽取</h3>
          </div>
          <div class="grid grid-cols-2 gap-4 mt-6">
            <div class="glass-card p-4 flex flex-col items-center justify-center">
              <span class="text-3xl font-bold text-gradient">{{ yoloBoxes.length || '-' }}</span>
              <span class="text-xs text-slate-200 mt-1 uppercase tracking-wider">检测到的古琴指法符</span>
            </div>
            <div class="glass-card p-4 flex flex-col items-center justify-center">
              <span class="text-3xl font-bold text-gradient">AI</span>
              <span class="text-xs text-slate-200 mt-1 uppercase tracking-wider">YOLOv11 引擎</span>
            </div>
          </div>
          <p class="text-sm text-slate-200 mt-6 leading-relaxed">
            引擎正在高精度检索图像中的谱字区域，捕捉笔画结构与相对坐标...
          </p>
        </div>
      </transition>

      <!-- Topology Stage -->
      <transition enter-active-class="animate-slide-up" leave-active-class="transition-opacity duration-300 opacity-0">
        <div v-show="activeStage === 'topology'" class="absolute inset-x-6 top-6">
          <div class="flex items-center gap-3 mb-4">
            <span class="p-2 bg-accent/20 text-accent rounded-lg border border-accent/30">
              <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
            </span>
            <h3 class="text-lg font-semibold text-slate-100 font-guqin tracking-widest">拓扑结构序列化</h3>
          </div>
          <div class="mt-4 flex flex-col gap-2 max-h-48 overflow-y-auto pr-2 custom-scrollbar">
            <template v-if="jianziSequence && jianziSequence.length > 0">
              <div v-for="(item, idx) in jianziSequence" :key="idx" class="glass-card p-3 flex items-center justify-between">
                <div class="flex items-center gap-3">
                  <div class="w-6 h-6 rounded-full bg-surface text-xs flex items-center justify-center text-slate-300 font-mono">{{ idx + 1 }}</div>
                  <span class="font-guqin text-lg text-slate-100">{{ item.action || '未知' }}</span>
                </div>
                <div class="flex gap-2">
                  <span v-if="item.string" class="text-xs px-2 py-1 rounded bg-slate-800 text-primary border border-primary/20">{{ item.string }} 弦</span>
                  <span v-if="item.position && item.position.trim()" class="text-xs px-2 py-1 rounded bg-slate-800 text-teal-400 border border-teal-500/20">{{ item.position }}徽</span>
                  <span v-if="item.finger" class="text-xs px-2 py-1 rounded bg-slate-800 text-accent border border-accent/20">{{ item.finger }} 指</span>
                </div>
              </div>
            </template>
            <div v-else class="text-center py-8 text-slate-300 text-sm">正在组装字块拓扑网络...</div>
          </div>
        </div>
      </transition>

      <!-- LLM Stage -->
      <transition enter-active-class="animate-slide-up" leave-active-class="transition-opacity duration-300 opacity-0">
        <div v-show="activeStage === 'llm'" class="absolute inset-x-6 top-6">
          <div class="flex items-center gap-3 mb-4">
            <span class="p-2 bg-indigo-500/20 text-indigo-400 rounded-lg border border-indigo-500/30">
              <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </span>
            <h3 class="text-lg font-semibold text-slate-100 font-guqin tracking-widest">大模型乐理推理</h3>
          </div>
          <div class="mt-4 flex flex-col gap-2 max-h-48 overflow-y-auto pr-2 custom-scrollbar">
            <template v-if="llmResult && llmResult.length > 0">
              <div v-for="(note, idx) in llmResult" :key="idx" class="glass-card p-3 flex flex-wrap gap-2 items-center">
                <div class="px-2 py-1 bg-indigo-500/10 rounded text-indigo-300 text-xs font-mono border border-indigo-500/20 font-bold">
                  {{ note.pitch || 'R' }}_{{ note.octave || '4' }}
                </div>
                <!-- Duration badge -->
                <div class="px-2 py-1 bg-amber-500/10 rounded text-amber-300 text-xs border border-amber-500/20">
                  {{ formatDuration(note.duration) }}
                </div>
                <!-- Action / Original text -->
                <div class="ml-auto text-sm font-guqin font-bold text-slate-100">
                   {{ note.action }}{{ note.string ? `(${note.string})` : ''}}
                </div>
              </div>
            </template>
            <div v-else class="flex flex-col items-center py-8">
              <div class="w-8 h-8 rounded-full border-t-2 border-r-2 border-indigo-400 animate-spin mb-3"></div>
              <p class="text-slate-300 text-sm">LLM 正在将物理指法转译为音高与时值...</p>
            </div>
          </div>
        </div>
      </transition>

      <!-- XML / Success Stage -->
      <transition enter-active-class="animate-slide-up" leave-active-class="transition-opacity duration-300 opacity-0">
        <div v-show="activeStage === 'xml' || activeStage === 'success'" class="absolute inset-x-6 top-6">
          <div class="flex items-center gap-3 mb-4">
             <span class="p-2 bg-emerald-500/20 text-emerald-400 rounded-lg border border-emerald-500/30">
              <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </span>
            <h3 class="text-lg font-semibold text-slate-100 font-guqin tracking-widest">MusicXML 编码就绪</h3>
          </div>
          <div class="glass-card p-4 relative group">
             <div class="absolute inset-0 bg-gradient-to-r from-emerald-500/5 to-teal-500/5 rounded-xl"></div>
             <p class="text-slate-200 text-sm leading-relaxed relative z-10 mb-4">
               解析流水线已完美闭环，生成了标准化的 MusicXML 及 ScoreModel 渲染树。打谱数据现可被导入主流打谱软件。
             </p>
             <div class="flex gap-2 relative z-10">
               <span class="px-2 py-1 bg-surface1 text-xs text-slate-200 rounded">打谱节点数: {{ llmResult?.length || 0 }}</span>
               <span class="px-2 py-1 bg-surface1 text-xs text-slate-200 rounded">生成格式: XML / JSON</span>
             </div>
          </div>
        </div>
      </transition>

      <!-- Idle Stage -->
      <transition enter-active-class="animate-slide-up" leave-active-class="transition-opacity duration-300 opacity-0">
        <div v-show="activeStage === 'idle'" class="absolute inset-x-6 top-6 flex flex-col items-center justify-center h-[200px] text-center">
            <div class="w-16 h-16 rounded-full bg-surface2/50 flex items-center justify-center mb-4 border border-white/5">
              <svg class="w-8 h-8 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 class="text-slate-200 font-medium mb-1 tracking-wide">等待任务指令</h3>
            <p class="text-slate-300 text-sm">上传图片后，系统将在此实时展示 AI 后端处理的流水线中间态结构</p>
        </div>
      </transition>

      <!-- Uploading/Sending to Backend Stage -->
      <transition enter-active-class="animate-slide-up" leave-active-class="transition-opacity duration-300 opacity-0">
        <div v-show="activeStage === 'uploading'" class="absolute inset-x-6 top-6 flex flex-col items-center justify-center h-[200px] text-center">
            <div class="w-16 h-16 rounded-full border-t-2 border-r-2 border-primary animate-spin mb-4"></div>
            <h3 class="text-primary font-medium mb-1 tracking-wide animate-pulse">正在将数据推送到 AI 引擎</h3>
            <p class="text-slate-300 text-sm">由于古琴推理包较大且运算密集，初次或深层推理可能需要数十秒，请耐心等待全流程解析完毕...</p>
        </div>
      </transition>

    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  pipelineStatus: { type: String, default: 'idle' },
  yoloBoxes: { type: Array, default: () => [] },
  jianziSequence: { type: Array, default: () => [] },
  llmResult: { type: Array, default: () => [] }
})

// Determine which stage panel to show, default to idle
const activeStage = computed(() => {
  if (['cv', 'topology', 'llm', 'xml', 'success', 'idle', 'uploading'].includes(props.pipelineStatus)) {
    return props.pipelineStatus;
  }
  return 'idle';
})

const formatDuration = (d) => {
  const map = {
    '4': '四分音符 (1拍)',
    '8': '八分音符 (0.5拍)',
    '16': '十六分音符 (0.25拍)',
    '2': '二分音符 (2拍)'
  }
  return map[d] || `时值: ${d}`;
}
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 4px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.02);
  border-radius: 4px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}
.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.2);
}
</style>
