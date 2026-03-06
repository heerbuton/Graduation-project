<script setup>
import { ref } from 'vue'
import axios from 'axios'
import MusicXmlScoreRenderer from './components/MusicXmlScoreRenderer.vue'

// 状态管理
const fileInput = ref(null)
const selectedFile = ref(null)
const previewImage = ref(null)
const pipelineStatus = ref('idle') // idle, uploading, cv, topology, llm, xml, success, error
const errorMessage = ref('')

// 数据存储
const originalImageUrl = ref('')
const yoloBoxes = ref([])
const topologyJson = ref(null)
const musicXml = ref('')

const statusMessages = {
  idle: '等待上传...',
  uploading: '正在上传图片...',
  cv: '模块A：正在进行视觉特征提取 (YOLO)...',
  topology: '模块B：正在进行空间拓扑解析...',
  llm: '模块C：正在请求 LLM 打谱推理...',
  xml: '模块D：正在生成 MusicXML...',
  success: '流水线执行完毕。',
  error: '发生错误。'
}

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
  
  // 生成本地预览
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
    // 调整canvas尺寸以匹配图片比例并适应容器宽度的逻辑
    // 简单实现为固定比例或者覆盖
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
      
      ctx.strokeStyle = '#ef4444' // red-500
      ctx.lineWidth = 2
      ctx.strokeRect(sx, sy, sw, sh)
      
      ctx.fillStyle = '#ef4444'
      ctx.font = '14px sans-serif'
      ctx.fillText(box.class, sx, sy - 5 > 0 ? sy - 5 : sy + 15)
    })
  }
  // 由于跨域和本地URL，先拼接后端地址
  img.src = `http://localhost:5000${originalImageUrl.value}`
}

// 模拟状态流动的延迟函数，为了展示 UX
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms))

const uploadAndProcess = async () => {
  if (!selectedFile.value) return
  
  pipelineStatus.value = 'uploading'
  errorMessage.value = ''
  
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  
  try {
    const response = await axios.post('http://localhost:5000/api/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    
    const data = response.data.data
    
    // 为了前端展示 UX，模拟流水线每一个阶段的等待 UI
    pipelineStatus.value = 'cv'
    await delay(600)
    
    pipelineStatus.value = 'topology'
    await delay(600)
    
    pipelineStatus.value = 'llm'
    await delay(1000)
    
    pipelineStatus.value = 'xml'
    await delay(600)
    
    // 赋值数据
    originalImageUrl.value = data.original_image_url
    yoloBoxes.value = data.yolo_boxes || []
    topologyJson.value = data.topology_json
    musicXml.value = data.music_xml
    
    pipelineStatus.value = 'success'
    
    // 渲染带有 box 的图
    setTimeout(renderImageWithBoxes, 100)
    
  } catch (error) {
    pipelineStatus.value = 'error'
    errorMessage.value = error.response?.data?.message || error.message || '上传处理失败'
  }
}

// 获取 Mock 数据进行测试
const reqMockData = async () => {
    try {
        pipelineStatus.value = 'uploading'
        await delay(500)
        pipelineStatus.value = 'success'
        const res = await axios.get('http://localhost:5000/api/mock_pipeline')
        const data = res.data.data
        musicXml.value = data.music_xml
    } catch(err) {
        pipelineStatus.value = 'error'
        errorMessage.value = err.message
    }
}
</script>

<template>
  <div class="min-h-screen flex flex-col">
    <!-- 顶栏 -->
    <header class="bg-gray-900 text-white p-4 shadow-md flex justify-between items-center">
      <h1 class="text-xl font-bold tracking-wide">伯牙解谱系统 - MusicXML 端到端演示原型</h1>
      <button @click="reqMockData" class="text-sm bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded transition-colors">
        加载 Mock XML 并渲染
      </button>
    </header>

    <main class="flex-1 flex flex-col md:flex-row overflow-hidden bg-gray-100 p-4 gap-4">
      
      <!-- 左侧原图对照区 -->
      <section class="w-full md:w-1/2 flex flex-col bg-white rounded-xl shadow border border-gray-200 overflow-hidden">
        <div class="bg-gray-50 border-b border-gray-200 p-3 font-semibold text-gray-700">
          原谱对照区
        </div>
        
        <div class="p-4 flex flex-col gap-4 flex-1 overflow-y-auto">
          
          <!-- 上传区 -->
          <div v-if="!originalImageUrl" 
               @dragover="onDragOver" 
               @drop="onDrop"
               class="border-2 border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center p-8 bg-gray-50 hover:bg-blue-50 transition-colors cursor-pointer"
               @click="fileInput.click()">
            <input type="file" ref="fileInput" class="hidden" accept="image/*" @change="onFileSelectChange" />
            <svg class="w-10 h-10 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2400/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
            <p class="text-gray-600 mb-1">点击或拖拽图片文件到此处</p>
            <p class="text-xs text-gray-400">支持 jpg, png 等常见图片格式</p>
          </div>
          
          <!-- 预览与本地路径 -->
          <div v-if="previewImage && !originalImageUrl" class="flex flex-col gap-2">
            <p class="text-sm text-gray-500 truncate">选中文件: {{ selectedFile?.name }}</p>
            <img :src="previewImage" class="max-h-64 object-contain rounded border border-gray-200" alt="预览">
            <button @click="uploadAndProcess" 
                    :disabled="pipelineStatus !== 'idle' && pipelineStatus !== 'error'"
                    class="mt-2 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded shadow-sm disabled:bg-blue-400 transition-colors">
              开始端到端解析
            </button>
          </div>

          <!-- 流水线状态指示器 -->
          <div v-if="pipelineStatus !== 'idle'" class="bg-blue-50 border border-blue-200 p-4 rounded-lg">
            <div class="flex items-center gap-3">
              <div v-if="pipelineStatus !== 'success' && pipelineStatus !== 'error'" class="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
              <div v-else-if="pipelineStatus === 'success'" class="text-green-600">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2400/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
              </div>
              <div v-else class="text-red-600">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2400/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
              </div>
              <p :class="{'text-blue-800': pipelineStatus !== 'error', 'text-red-800': pipelineStatus === 'error'}" class="font-medium text-sm">
                {{ statusMessages[pipelineStatus] }}
              </p>
            </div>
            <p v-if="errorMessage" class="text-red-600 text-xs mt-2">{{ errorMessage }}</p>
          </div>

          <!-- 解析后画图区 -->
          <div v-show="originalImageUrl" class="mt-4 border border-gray-200 rounded overflow-hidden relative" style="min-height: 200px">
            <div class="absolute top-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm z-10">视知识别结果(YOLO)</div>
            <canvas ref="imageCanvas" class="w-full h-auto object-contain bg-gray-50"></canvas>
          </div>
          
          <div v-if="originalImageUrl" class="mt-2 text-center">
             <button @click="() => { originalImageUrl = ''; previewImage = ''; selectedFile = null; pipelineStatus = 'idle'; musicXml = ''; yoloBoxes = []; }" class="text-sm text-gray-500 hover:text-gray-800 underline">
               重新上传
             </button>
          </div>

        </div>
      </section>

      <!-- 右侧翻译渲染区 -->
      <section class="w-full md:w-1/2 flex flex-col bg-white rounded-xl shadow border border-gray-200 overflow-hidden">
        <div class="bg-gray-50 border-b border-gray-200 p-3 font-semibold text-gray-700 flex justify-between items-center">
          <span>翻译渲染区 (MusicXML Render)</span>
          <span v-if="musicXml" class="text-xs font-normal text-green-600 bg-green-50 px-2 py-1 rounded border border-green-200">MusicXML 解析成功</span>
        </div>
        
        <div class="p-4 flex-1 overflow-y-auto bg-amber-50/30">
          <div v-if="musicXml">
            <MusicXmlScoreRenderer :xml-data="musicXml" />
            
            <details class="mt-8 text-xs text-gray-500">
              <summary class="cursor-pointer hover:text-gray-700">查看生成的 XML 数据</summary>
              <pre class="mt-2 p-3 bg-gray-100 rounded overflow-auto border border-gray-200">{{ musicXml }}</pre>
            </details>
          </div>
          
          <div v-else class="h-full flex items-center justify-center text-gray-400 flex-col gap-2">
             <svg class="w-12 h-12 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2400/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
             <span>等待解析数据...</span>
          </div>
        </div>
      </section>
      
    </main>
  </div>
</template>
