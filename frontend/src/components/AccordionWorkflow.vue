<template>
  <div class="w-full flex flex-col gap-3">
    
    <!-- Uploading & Idle States -->
    <div v-if="pipelineStatus === 'idle'" class="glass-panel p-6 flex flex-col items-center justify-center text-center">
        <div class="w-12 h-12 rounded-full bg-surface2/50 flex items-center justify-center mb-3 border border-white/5">
          <svg class="w-6 h-6 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <h3 class="text-slate-200 font-medium tracking-wide">等待任务指令</h3>
        <p class="text-slate-400 text-xs mt-1">系统将在此实时展示每个步骤的解析详情</p>
    </div>

    <div v-else-if="pipelineStatus === 'uploading'" class="glass-panel p-6 flex flex-col items-center justify-center text-center border-primary/30 relative overflow-hidden">
        <div class="absolute inset-0 bg-primary/5 animate-pulse"></div>
        <div class="w-10 h-10 rounded-full border-t-2 border-r-2 border-primary animate-spin mb-3"></div>
        <h3 class="text-primary font-medium tracking-wide z-10">推送到 AI 推理阵列...</h3>
        <p class="text-slate-300 text-xs mt-1 z-10">网络往返及重负荷模型运算约需 10-30 秒</p>
    </div>

    <!-- Accordion Cards -->
    <template v-if="['cv', 'topology', 'llm', 'xml', 'success'].includes(pipelineStatus)">
      
      <!-- 1. CV Stage -->
      <div class="glass-panel overflow-hidden transition-all duration-300" 
           :class="[getBorderClass(1), { 'ring-1 ring-primary/50 shadow-[0_0_15px_rgba(20,184,166,0.2)]': getStatus(1) === 'active' }]">
        <!-- Header -->
        <div class="px-5 py-4 flex items-center cursor-pointer select-none hover:bg-white/5 transition-colors" @click="toggleExpand(1)">
          <div :class="getIconBgClass(1)" class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mr-4 transition-colors">
            <span v-if="getStatus(1) === 'done'" class="text-white">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
            </span>
            <span v-else :class="getIconTextClass(1)" class="text-sm font-bold">1</span>
          </div>
          <div class="flex-1">
            <h3 class="font-guqin text-lg tracking-widest transition-colors" :class="getTitleColorClass(1)">视觉特征抽取</h3>
            <p class="text-xs transition-colors" :class="getSubtitleColorClass(1)">YOLOv11 位姿检测</p>
          </div>
          <svg class="w-5 h-5 transition-transform duration-300" :class="[isExpanded[1] ? 'rotate-180 text-white' : 'text-slate-500']" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
          </svg>
        </div>
        <!-- Body Expand -->
        <div v-show="isExpanded[1]" class="px-5 pb-5 pt-2 border-t border-white/5">
          <div class="grid grid-cols-2 gap-4 mt-2">
            <div class="bg-black/30 rounded-lg p-3 flex flex-col items-center justify-center border border-white/5">
              <span class="text-2xl font-bold text-slate-100">{{ getStatus(1) === 'pending' ? '-' : (yoloBoxes.length || 0) }}</span>
              <span class="text-xs text-slate-400 mt-1 uppercase">识别指法符</span>
            </div>
            <div class="bg-black/30 rounded-lg p-3 flex flex-col items-center justify-center border border-white/5">
              <span class="text-2xl font-bold text-slate-100">YOLO</span>
              <span class="text-xs text-slate-400 mt-1 uppercase">AI 模型引擎</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 2. Topology Stage -->
      <div class="glass-panel overflow-hidden transition-all duration-300" 
           :class="[getBorderClass(2), { 'ring-1 ring-accent/50 shadow-[0_0_15px_rgba(217,119,6,0.2)]': getStatus(2) === 'active' }]">
        <!-- Header -->
        <div class="px-5 py-4 flex items-center cursor-pointer select-none hover:bg-white/5 transition-colors" @click="toggleExpand(2)">
          <div :class="getIconBgClass(2)" class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mr-4 transition-colors">
            <span v-if="getStatus(2) === 'done'" class="text-white">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
            </span>
            <span v-else :class="getIconTextClass(2)" class="text-sm font-bold">2</span>
          </div>
          <div class="flex-1">
            <h3 class="font-guqin text-lg tracking-widest transition-colors" :class="getTitleColorClass(2)">拓扑结构序列化</h3>
            <p class="text-xs transition-colors" :class="getSubtitleColorClass(2)">上下文网络组装与对齐</p>
          </div>
          <svg class="w-5 h-5 transition-transform duration-300" :class="[isExpanded[2] ? 'rotate-180 text-white' : 'text-slate-500']" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
          </svg>
        </div>
        <!-- Body Expand -->
        <div v-show="isExpanded[2]" class="px-5 pb-5 pt-2 border-t border-white/5">
          <div class="mt-2 flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
            <template v-if="jianziSequence && jianziSequence.length > 0">
              <div v-for="(item, idx) in jianziSequence" :key="idx" class="bg-black/20 p-2.5 rounded-lg flex items-center justify-between border border-white/5">
                <div class="flex items-center gap-3">
                  <span class="text-xs text-slate-500 font-mono w-4 text-center">{{ idx + 1 }}</span>
                  <span class="font-guqin text-base text-slate-200">{{ item.action || '未知' }}</span>
                </div>
                <div class="flex gap-1.5 opacity-90">
                  <span v-if="item.string" class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-primary border border-primary/20">{{ item.string }} 弦</span>
                  <span v-if="item.position && item.position.trim()" class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-teal-400 border border-teal-500/20">{{ item.position }}徽</span>
                  <span v-if="item.finger" class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-accent border border-accent/20">{{ item.finger }} 指</span>
                </div>
              </div>
            </template>
            <div v-else class="text-center py-4 text-slate-400 text-sm">暂无拓扑序列数据</div>
          </div>
        </div>
      </div>

      <!-- 3. LLM Stage -->
      <div class="glass-panel overflow-hidden transition-all duration-300" 
           :class="[getBorderClass(3), { 'ring-1 ring-indigo-500/50 shadow-[0_0_15px_rgba(99,102,241,0.2)]': getStatus(3) === 'active' }]">
        <!-- Header -->
        <div class="px-5 py-4 flex items-center cursor-pointer select-none hover:bg-white/5 transition-colors" @click="toggleExpand(3)">
          <div :class="getIconBgClass(3)" class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mr-4 transition-colors">
            <span v-if="getStatus(3) === 'done'" class="text-white">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
            </span>
            <span v-else :class="getIconTextClass(3)" class="text-sm font-bold">3</span>
          </div>
          <div class="flex-1">
            <h3 class="font-guqin text-lg tracking-widest transition-colors" :class="getTitleColorClass(3)">大模型乐理推理</h3>
            <p class="text-xs transition-colors" :class="getSubtitleColorClass(3)">将物理指法转译为音高时值</p>
          </div>
          <svg class="w-5 h-5 transition-transform duration-300" :class="[isExpanded[3] ? 'rotate-180 text-white' : 'text-slate-500']" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
          </svg>
        </div>
        <!-- Body Expand -->
        <div v-show="isExpanded[3]" class="px-5 pb-5 pt-2 border-t border-white/5">
          <div class="mt-2 flex flex-col gap-2 max-h-48 overflow-y-auto pr-1">
            <template v-if="llmResult && llmResult.length > 0">
              <div v-for="(note, idx) in llmResult" :key="idx" class="bg-black/20 p-2.5 rounded-lg flex flex-wrap gap-2 items-center border border-white/5">
                <div class="px-2 py-0.5 bg-indigo-500/10 rounded text-indigo-300 text-[11px] font-mono border border-indigo-500/20 font-bold">
                  {{ note.pitch || 'R' }}_{{ note.octave || '4' }}
                </div>
                <div class="px-2 py-0.5 bg-amber-500/10 rounded text-amber-300 text-[11px] border border-amber-500/20">
                  {{ formatDuration(note.duration) }}
                </div>
                <div class="ml-auto text-xs font-guqin font-bold text-slate-200 opacity-90">
                   {{ note.action }}{{ note.string ? `(${note.string})` : ''}}
                </div>
              </div>
            </template>
            <div v-else class="flex flex-col items-center py-4">
              <p class="text-slate-400 text-sm">等待转译处理...</p>
            </div>
          </div>
        </div>
      </div>

      <!-- 4. XML Stage -->
      <div class="glass-panel overflow-hidden transition-all duration-300" 
           :class="[getBorderClass(4), { 'ring-1 ring-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.2)]': getStatus(4) === 'active' || getStatus(4) === 'done' }]">
        <!-- Header -->
        <div class="px-5 py-4 flex items-center cursor-pointer select-none hover:bg-white/5 transition-colors" @click="toggleExpand(4)">
          <div :class="getIconBgClass(4)" class="w-8 h-8 rounded-full flex items-center justify-center shrink-0 mr-4 transition-colors">
            <span v-if="getStatus(4) === 'done'" class="text-white">
              <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
            </span>
            <span v-else :class="getIconTextClass(4)" class="text-sm font-bold">4</span>
          </div>
          <div class="flex-1">
            <h3 class="font-guqin text-lg tracking-widest transition-colors" :class="getTitleColorClass(4)">MusicXML 就绪</h3>
            <p class="text-xs transition-colors" :class="getSubtitleColorClass(4)">可导入主流打谱软件与渲染</p>
          </div>
          <svg class="w-5 h-5 transition-transform duration-300" :class="[isExpanded[4] ? 'rotate-180 text-white' : 'text-slate-500']" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
          </svg>
        </div>
        <!-- Body Expand -->
        <div v-show="isExpanded[4]" class="px-5 pb-5 pt-2 border-t border-white/5">
          <div class="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 rounded-xl p-4 mt-2 border border-emerald-500/20">
             <p class="text-slate-200 text-xs leading-relaxed mb-3">
               流水线解析闭环完成。生成标准 MusicXML 及 ScoreModel。右侧容器已获得渲染许可。
             </p>
             <div class="flex gap-2">
               <span class="px-2 py-1 bg-black/40 text-[10px] text-slate-300 rounded border border-white/5">解析总计: {{ llmResult?.length || 0 }} 符</span>
               <span class="px-2 py-1 bg-black/40 text-[10px] text-slate-300 rounded border border-white/5">格式: XML/JSON</span>
             </div>
          </div>
        </div>
      </div>

    </template>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  pipelineStatus: { type: String, default: 'idle' },
  yoloBoxes: { type: Array, default: () => [] },
  jianziSequence: { type: Array, default: () => [] },
  llmResult: { type: Array, default: () => [] }
})

// Control which cards are expanded
const isExpanded = ref({ 1: false, 2: false, 3: false, 4: false })

// Map status to number index to easily calculate step state
const phaseOrder = ['idle', 'uploading', 'cv', 'topology', 'llm', 'xml', 'success', 'error']
const phaseIndex = computed(() => phaseOrder.indexOf(props.pipelineStatus))

// Determine 'pending', 'active', or 'done' for each step (1-4)
const getStatus = (stepId) => {
  const targetMap = { 1: 2, 2: 3, 3: 4, 4: 5 } // 1->'cv'(2), etc.
  const targetIdx = targetMap[stepId]
  
  if (phaseIndex.value < targetIdx) return 'pending'
  if (phaseIndex.value === targetIdx) return 'active'
  return 'done'
}

// Ensure the currently active or successfully completed step is expanded automatically
watch(() => props.pipelineStatus, () => {
  if (['idle', 'uploading', 'error'].includes(props.pipelineStatus)) {
    isExpanded.value = { 1: false, 2: false, 3: false, 4: false }
    return
  }

  // Find the step corresponding to current phase
  let activeStep = 0
  if (props.pipelineStatus === 'cv') activeStep = 1
  if (props.pipelineStatus === 'topology') activeStep = 2
  if (props.pipelineStatus === 'llm') activeStep = 3
  if (props.pipelineStatus === 'xml' || props.pipelineStatus === 'success') activeStep = 4
  
  // Collapse all and open exactly the current active step
  if (activeStep > 0) {
    isExpanded.value = { 1: false, 2: false, 3: false, 4: false }
    isExpanded.value[activeStep] = true
  }
}, { immediate: true })

const toggleExpand = (stepId) => {
  isExpanded.value[stepId] = !isExpanded.value[stepId]
}

/* ======== Style Generators based on State ('pending', 'active', 'done') ======== */

const getBorderClass = (stepId) => {
  const s = getStatus(stepId);
  if (s === 'done') {
    if (stepId === 1) return 'border-primary/40'
    if (stepId === 2) return 'border-accent/40'
    if (stepId === 3) return 'border-indigo-500/40'
    if (stepId === 4) return 'border-emerald-500/40'
  }
  if (s === 'active') return 'border-white/20'
  return 'border-white/5 opacity-60' // Pending looks dimmer but readable
}

const getIconBgClass = (stepId) => {
  const s = getStatus(stepId);
  if (s === 'pending') return 'bg-slate-800 border border-slate-700'
  
  if (stepId === 1) return s === 'done' ? 'bg-primary border border-primary text-white shadow-lg shadow-primary/30' : 'bg-primary/20 border-primary shadow-[0_0_10px_rgba(20,184,166,0.5)]'
  if (stepId === 2) return s === 'done' ? 'bg-accent border border-accent text-white shadow-lg shadow-accent/30' : 'bg-accent/20 border-accent shadow-[0_0_10px_rgba(217,119,6,0.5)]'
  if (stepId === 3) return s === 'done' ? 'bg-indigo-500 border border-indigo-500 text-white shadow-lg shadow-indigo-500/30' : 'bg-indigo-500/20 border-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]'
  if (stepId === 4) return s === 'done' ? 'bg-emerald-500 border border-emerald-500 text-white shadow-lg shadow-emerald-500/30' : 'bg-emerald-500/20 border-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]'
}

const getIconTextClass = (stepId) => {
  const s = getStatus(stepId);
  if (s === 'pending') return 'text-slate-500' // Darker for pending circle text
  
  if (stepId === 1) return 'text-primary'
  if (stepId === 2) return 'text-accent'
  if (stepId === 3) return 'text-indigo-400'
  if (stepId === 4) return 'text-emerald-400'
}

const getTitleColorClass = (stepId) => {
  const s = getStatus(stepId);
  if (s === 'pending') return 'text-slate-400 font-normal' // Light gray for pending (not too dark!)
  return 'text-slate-100 font-semibold' // White and bold for active/done
}

const getSubtitleColorClass = (stepId) => {
  const s = getStatus(stepId);
  if (s === 'pending') return 'text-slate-500' // slightly darker for subtitle pending
  return 'text-slate-300' // Brighter gray for active/done
}

const formatDuration = (d) => {
  const map = {
    '4': '四分音符(1拍)',
    '8': '八分音符(0.5拍)',
    '16': '十六分(0.25拍)',
    '2': '二分音符(2拍)'
  }
  return map[d] || `${d}`;
}
</script>
