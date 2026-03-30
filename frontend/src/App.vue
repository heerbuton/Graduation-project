<script setup>
import { ref } from 'vue'
import axios from 'axios'
import ScoreModelRenderer from './components/ScoreModelRenderer.vue'
import AccordionWorkflow from './components/AccordionWorkflow.vue'
import { convertLlmResultToScoreModel } from './utils/scoreModel.js'

// 状态管理
const fileInput = ref(null)
const selectedFile = ref(null)
const previewImage = ref(null)
const pipelineStatus = ref('idle') // idle, uploading, cv, topology, llm, xml, success, error
const errorMessage = ref('')

// 数据存储
const originalImageUrl = ref('')
const yoloBoxes = ref([])
const jianziSequence = ref([])
const topologyJson = ref(null)
const llmResult = ref([])
const musicXml = ref('')
const scoreModel = ref(null)

// 拖拽事件处理
const onDragOver = (e) => {
  e.preventDefault()
  e.dataTransfer.dropEffect = 'copy'
}

const onDrop = (e) => {
  e.preventDefault()
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFileSelection(e.dataTransfer.files[0])
  }
}

const onFileSelectChange = (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleFileSelection(e.target.files[0])
  }
}

const handleFileSelection = (file) => {
  if (!file.type.startsWith('image/')) {
    errorMessage.value = '请上传图片文件。'
    return
  }
  selectedFile.value = file
  errorMessage.value = ''
  
  // Reset states to allow a fresh run
  pipelineStatus.value = 'idle'
  originalImageUrl.value = ''
  yoloBoxes.value = []
  jianziSequence.value = []
  topologyJson.value = null
  llmResult.value = []
  musicXml.value = ''
  scoreModel.value = null
  
  const reader = new FileReader()
  reader.onload = (e) => {
    previewImage.value = e.target.result
  }
  reader.readAsDataURL(file)
}

// 图片渲染 Canvas (带有 YOLO Box)
const imageCanvas = ref(null)
const renderImageWithBoxes = () => {
  if (!imageCanvas.value || !originalImageUrl.value) return
  
  const ctx = imageCanvas.value.getContext('2d')
  const img = new Image()
  img.onload = () => {
    const containerWidth = imageCanvas.value.parentElement.clientWidth
    const scale = containerWidth / img.width
    imageCanvas.value.width = img.width * scale
    imageCanvas.value.height = img.height * scale
    
    ctx.drawImage(img, 0, 0, imageCanvas.value.width, imageCanvas.value.height)
    
    // 渲染 boxes
    yoloBoxes.value.forEach(box => {
      const [x1, y1, x2, y2] = box.bbox
      const sx = x1 * scale
      const sy = y1 * scale
      const sw = (x2 - x1) * scale
      const sh = (y2 - y1) * scale
      
      // 现代感 Box: 翡翠色/青色带细微半透明
      ctx.strokeStyle = '#14b8a6' // Tailwind teal-500
      ctx.lineWidth = 2
      ctx.strokeRect(sx, sy, sw, sh)
      
      ctx.fillStyle = 'rgba(20, 184, 166, 0.2)'
      ctx.fillRect(sx, sy, sw, sh)
    })
  }
  img.src = `http://localhost:5000${originalImageUrl.value}`
}

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms))

const applyPipelineResult = (data, fallbackImageUrl = '') => {
  originalImageUrl.value = data.original_image_url || fallbackImageUrl
  yoloBoxes.value = data.yolo_boxes || []
  jianziSequence.value = data.jianzi_sequence || []
  topologyJson.value = data.topology_json || null
  llmResult.value = data.llm_result || []
  musicXml.value = data.music_xml || ''
  
  if (Array.isArray(data.llm_result) && data.llm_result.length > 0) {
    scoreModel.value = convertLlmResultToScoreModel(data.llm_result, { strict: false })
  } else {
    scoreModel.value = data.score_model || convertLlmResultToScoreModel([], { strict: false })
  }
}

const uploadAndProcess = async () => {
  if (!selectedFile.value) {
    errorMessage.value = '请先上传图片后再开始打谱。'
    return
  }
  
  pipelineStatus.value = 'uploading'
  errorMessage.value = ''
  
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  
  try {
    const response = await axios.post('http://localhost:5000/api/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    const data = response.data.data
    
    // 展现流水线动画
    pipelineStatus.value = 'cv'
    applyPipelineResult(data) // 将数据一次性赋值，但组件基于 phase 获取对应数据
    setTimeout(renderImageWithBoxes, 100) // CV 图片立即展示
    await delay(2500)
    
    pipelineStatus.value = 'topology'
    await delay(2500)
    
    pipelineStatus.value = 'llm'
    await delay(3000)
    
    pipelineStatus.value = 'xml'
    await delay(2000)
    
    pipelineStatus.value = 'success'
    
  } catch (error) {
    pipelineStatus.value = 'error'
    errorMessage.value = error.response?.data?.message || error.message || '上传处理失败'
  }
}

const loadSavedTestPictureResult = async () => {
  try {
    pipelineStatus.value = 'uploading'
    errorMessage.value = ''
    const res = await axios.get('http://localhost:5000/static/uploads/testpicture-1.jpg_result.json', {
      responseType: 'text',
      transformResponse: [(value) => value]
    })
    const text = String(res.data || '').replace(/^\uFEFF/, '')
    const data = JSON.parse(text)
    
    pipelineStatus.value = 'cv'
    applyPipelineResult(data, '/static/uploads/testpicture-1.jpg')
    setTimeout(renderImageWithBoxes, 100)
    await delay(2500)
    
    pipelineStatus.value = 'topology'
    await delay(2500)
    
    pipelineStatus.value = 'llm'
    await delay(3000)
    
    pipelineStatus.value = 'xml'
    await delay(2000)

    pipelineStatus.value = 'success'
  } catch (error) {
    pipelineStatus.value = 'error'
    errorMessage.value = error.response?.data?.message || error.message || '加载预计算结果失败'
  }
}

const resetAll = () => {
  originalImageUrl.value = ''; 
  previewImage.value = ''; 
  selectedFile.value = null; 
  pipelineStatus.value = 'idle'; 
  musicXml.value = ''; 
  scoreModel.value = null; 
  yoloBoxes.value = [];
  jianziSequence.value = [];
  llmResult.value = [];
  errorMessage.value = '';
}
</script>

<template>
  <div class="min-h-screen flex flex-col font-sans relative overflow-x-hidden">
    
    <!-- 装饰性光晕背景 -->
    <div class="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/20 blur-[120px] rounded-full pointer-events-none z-0"></div>
    <div class="fixed bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-accent/10 blur-[100px] rounded-full pointer-events-none z-0"></div>

    <!-- 顶栏 -->
    <header class="relative z-10 bg-surface1/60 backdrop-blur-lg border-b border-white/10 p-4 shadow-lg flex justify-between items-center transition-all">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white font-guqin font-bold text-lg shadow-lg shadow-primary/20">
          琴
        </div>
        <h1 class="text-xl font-medium tracking-widest text-slate-100 font-guqin">伯牙解谱系统 <span class="text-sm font-sans tracking-normal opacity-50 ml-2">v2.0 Visualizer</span></h1>
      </div>
      <div class="flex items-center gap-3">
        <button @click="loadSavedTestPictureResult" class="text-xs font-medium text-primary border border-primary/30 bg-primary/10 hover:bg-primary/20 hover:text-white px-4 py-2 rounded-lg transition-all duration-300">
          Demo: 加载测试图
        </button>
      </div>
    </header>

    <main class="relative z-10 flex-1 flex flex-col xl:flex-row p-6 gap-6 max-w-[1600px] mx-auto w-full">
      
      <!-- 左半区：工作流可视化 (上) + 上传与原图 (下) -->
      <section class="w-full xl:w-1/2 flex flex-col gap-6 overflow-hidden">
        
        <!-- 统一整合式工作流卡片 -->
        <AccordionWorkflow
          :pipelineStatus="pipelineStatus"
          :yoloBoxes="yoloBoxes"
          :jianziSequence="jianziSequence"
          :llmResult="llmResult"
        />

        <!-- 上传/原图卡片 (原左侧) -->
        <div class="glass-panel flex flex-col overflow-hidden">
          <div class="bg-white/5 border-b border-white/5 px-4 py-3 flex justify-between items-center">
            <h2 class="font-medium text-slate-200">源文件区</h2>
             <button v-if="originalImageUrl || previewImage" @click="resetAll" class="text-xs text-slate-400 hover:text-white transition-colors flex items-center gap-1">
               <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
               重置
             </button>
          </div>
          
          <div class="p-4 flex-1 flex flex-col min-h-[300px]">
            <!-- 未上传 -->
            <div v-if="!originalImageUrl && !previewImage" 
                 @dragover="onDragOver" 
                 @drop="onDrop"
                 class="flex-1 border-2 border-dashed border-slate-600/50 hover:border-primary/50 rounded-xl flex flex-col items-center justify-center p-8 bg-black/20 hover:bg-black/40 transition-all duration-300 cursor-pointer group"
                 @click="fileInput.click()">
              <input type="file" ref="fileInput" class="hidden" accept="image/*" @change="onFileSelectChange" />
              <div class="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center group-hover:scale-110 group-hover:bg-primary/20 transition-all duration-300 mb-4 shadow-xl">
                 <svg class="w-6 h-6 text-slate-400 group-hover:text-primary transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
              </div>
              <p class="text-slate-300 font-medium mb-1 tracking-wide">拖拽曲谱图片至此，或点击上传</p>
              <p class="text-xs text-slate-500">支持 JPG, PNG 高清扫描图</p>
            </div>
            
            <!-- 预览并等待启动 -->
            <div v-else-if="previewImage && !originalImageUrl" class="flex flex-col h-full items-center justify-center animate-fade-in relative gap-6">
              <img :src="previewImage" class="max-h-[300px] object-contain rounded-lg border border-white/10 shadow-lg" alt="预览">
              <div class="flex items-center justify-center">
                <button @click="uploadAndProcess" 
                        :disabled="pipelineStatus !== 'idle' && pipelineStatus !== 'error'"
                        class="bg-gradient-to-r from-primary to-primary_dark text-white font-medium py-3 px-8 rounded-full shadow-lg shadow-primary/25 hover:shadow-primary/40 transition-all duration-300 hover:-translate-y-1 tracking-wider border border-white/10 flex items-center gap-2 disabled:opacity-50 disabled:hover:-translate-y-0 disabled:cursor-not-allowed">
                  <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                  启动 AI 打谱引擎
                </button>
              </div>
            </div>

            <!-- Pipeline 结果图展示 -->
             <div v-show="originalImageUrl" class="relative w-full rounded-lg min-h-[300px] flex items-center justify-center overflow-hidden border border-white/5 bg-black/40 shadow-inner">
              <div class="absolute top-2 left-2 bg-black/70 text-primary border border-primary/30 text-[10px] px-2 py-1 rounded shadow-lg backdrop-blur-md z-10 flex items-center gap-1 group">
                <span class="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                <span class="font-mono uppercase tracking-widest">YOLO.Vision</span>
              </div>
              <!-- CANVAS 绘图 -->
              <canvas ref="imageCanvas" class="w-full h-auto object-contain z-0"></canvas>
            </div>

          </div>
        </div>

        <!-- 错误提示 -->
        <div v-if="errorMessage" class="glass-panel border-red-500/30 bg-red-500/5 p-4 flex items-center gap-3 text-red-400">
           <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
           <span class="text-sm">{{ errorMessage }}</span>
        </div>

      </section>

      <!-- 右半区：最终渲染 -->
      <section class="w-full xl:w-1/2 flex flex-col glass-panel overflow-hidden">
        <div class="bg-white/5 border-b border-white/5 px-4 py-3 flex justify-between items-center">
          <h2 class="font-medium text-slate-200">打谱渲染容器 (ScoreModel View)</h2>
          <span v-if="scoreModel && scoreModel.noteCount > 0" class="text-[10px] uppercase tracking-wider font-bold text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded border border-emerald-400/20">Render Ready</span>
        </div>
        
        <div class="p-6 flex-1 overflow-y-auto relative bg-[#fdfdfd] text-slate-800 transition-colors duration-700">
          <div v-if="scoreModel && scoreModel.noteCount > 0" class="animate-fade-in relative z-10">
            <!-- 直接复用原有组件，组件内部样式会自动适配其内部限定好的样式 -->
            <ScoreModelRenderer :score-data="scoreModel" />
            
            <div class="mt-12 flex gap-4 no-print text-xs opacity-50 hover:opacity-100 transition-opacity">
               <!-- 预留导出或查看源码按钮 -->
               <button class="px-3 py-1.5 rounded border border-slate-300 hover:bg-slate-100">查看原生 ScoreModel JSON</button>
            </div>
          </div>
          
          <div v-else class="absolute inset-0 flex flex-col items-center justify-center text-slate-400 bg-surface text-opacity-50 z-0 delay-100 transition-all duration-700" :class="{ 'opacity-0': pipelineStatus === 'success', 'opacity-100': pipelineStatus !== 'success' }">
             <div class="w-24 h-24 mb-4 opacity-20 relative">
                <svg class="absolute inset-0 w-full h-full" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
             </div>
             <p class="font-guqin text-lg tracking-widest text-slate-500">等待 AI 注入灵魂...</p>
          </div>
        </div>
      </section>

    </main>
  </div>
</template>

<style>
/* 保证组件内部作用域下的 Tailwind 不冲突 */
body {
  margin: 0;
  overflow-x: hidden;
}
</style>
