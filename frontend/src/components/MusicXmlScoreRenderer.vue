<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  xmlData: {
    type: String,
    required: true
  }
})

const parsedMeasures = ref([])
const parseError = ref('')

// 将 MusicXML 的 step (C,D,E,F,G,A,B) 映射回简谱数字
const stepToJianpu = {
  "C": "1", "D": "2", "E": "3", "F": "4",
  "G": "5", "A": "6", "B": "7"
}

const parseXml = () => {
  if (!props.xmlData) return
  
  try {
    const parser = new DOMParser()
    const xmlDoc = parser.parseFromString(props.xmlData, "text/xml")
    
    // 检查是否有解析错误
    const errorNode = xmlDoc.querySelector('parsererror')
    if (errorNode) {
      parseError.value = 'XML 解析错误'
      return
    }
    
    parseError.value = ''
    const measures = []
    
    const measureElements = xmlDoc.getElementsByTagName('measure')
    
    for (let m = 0; m < measureElements.length; m++) {
        const measureNode = measureElements[m]
        const measureNotes = []
        
        const noteElements = measureNode.getElementsByTagName('note')
        
        for (let i = 0; i < noteElements.length; i++) {
            const node = noteElements[i]
            
            // 解析音高
            const stepNode = node.querySelector('pitch > step')
            const step = stepNode ? stepNode.textContent : ''
            const pitch = stepToJianpu[step] || step || 'X'
            
            // 解析八度
            const octaveNode = node.querySelector('pitch > octave')
            const octave = octaveNode ? octaveNode.textContent : '4'

            // 解析时值
            let duration = '4'
            let isHalf = false
            const typeNode = node.querySelector('type')
            if (typeNode) {
                const typeStr = typeNode.textContent
                if (typeStr === 'eighth') duration = '8'
                if (typeStr === '16th') duration = '16'
                if (typeStr === 'half') { duration = '4'; isHalf = true; } 
            }
            
            let finger = ''     
            let position = ''   
            let action = ''     
            let stringOrder = '' 
            
            const lyrics = node.getElementsByTagName('lyric')
            for (let j = 0; j < lyrics.length; j++) {
                const lyric = lyrics[j]
                const number = lyric.getAttribute('number')
                const textNode = lyric.getElementsByTagName('text')[0]
                const text = textNode ? textNode.textContent : ''
                
                if (number === '1') action = text
                else if (number === '2') stringOrder = text
                else if (number === '3') position = text
                else if (number === '4') finger = text
            }
            
            measureNotes.push({
                id: `m${m}_n${i}`,
                pitch,
                octave,
                duration,
                isDash: false,
                guqin: { finger, position, stringOrder, action }
            })
            
            // 如果是二分音符，额外推入一个延时线（dash）占位符占一拍
            if (isHalf) {
                measureNotes.push({
                    id: `m${m}_dash${i}`,
                    pitch: '-',
                    octave: '4',
                    duration: '4',
                    isDash: true,
                    guqin: { finger: '', position: '', stringOrder: '', action: '' }
                })
            }
        }
        
        measures.push({
            id: `m${m}`,
            notes: measureNotes
        })
    }
    
    parsedMeasures.value = measures
    
  } catch (error) {
    parseError.value = error.message
  }
}

watch(() => props.xmlData, () => {
  parseXml()
}, { immediate: true })

</script>

<template>
  <div class="music-xml-renderer p-4 bg-white rounded shadow-sm border border-emerald-200 min-h-[300px]">
    <h3 class="text-lg font-serif mb-4 text-emerald-900 border-b border-emerald-100 pb-2">翻译渲染区 (MusicXML Render)</h3>
    
    <div v-if="parseError" class="text-red-500 bg-red-50 p-3 rounded text-sm mb-4 border border-red-200">
      {{ parseError }}
    </div>
    
    <!-- 乐谱渲染区 -->
    <div class="flex flex-wrap items-start guqin-score-container p-6 bg-[#fafaf8] rounded-xl shadow-inner min-h-[400px]" v-else-if="parsedMeasures.length > 0">
      
      <div v-for="(measure, mIndex) in parsedMeasures" :key="measure.id" class="flex items-start">
        <!-- 每一个音符列 -->
        <div v-for="note in measure.notes" :key="note.id" class="flex flex-col items-center w-12 mb-6 group cursor-default">
             
          <!-- 简谱区 -->
          <div class="jianpu-section w-full text-center h-12 flex flex-col justify-end items-center mb-3 relative">
            <!-- 高音点 -->
            <div v-if="note.octave === '5'" class="absolute -top-1 w-1.5 h-1.5 rounded-full bg-gray-900"></div>
            <!-- 低音点 -->
            <div v-if="note.octave === '3'" class="absolute -bottom-1.5 w-1.5 h-1.5 rounded-full bg-gray-900"></div>
            
            <span class="text-2xl font-bold font-sans text-gray-900 leading-none group-hover:text-emerald-700 transition-colors">
              {{ note.pitch }}
            </span>
            <div v-if="note.duration === '8'" class="w-2/3 h-[1.5px] bg-gray-900 mt-1"></div>
            <div v-if="note.duration === '16'" class="w-2/3 flex flex-col gap-[2px] mt-1">
              <div class="h-[1.5px] bg-gray-900"></div>
              <div class="h-[1.5px] bg-gray-900"></div>
            </div>
          </div>
          
          <!-- 歌词区 (对应古琴减字谱四部分) -->
          <div v-if="!note.isDash" class="lyrics-section flex flex-col w-full items-center text-[0.9rem] font-serif text-gray-800 tracking-widest gap-2 group-hover:bg-emerald-50/60 rounded py-1 transition-colors">
            <!-- 第一行：右手技法 Action (如：勾) -->
            <div class="h-5 flex items-center justify-center font-medium">{{ note.guqin.action || ' ' }}</div>
            <!-- 第二行：弦序 String Order (如：一) -->
            <div class="h-5 flex items-center justify-center">{{ note.guqin.stringOrder || ' ' }}</div>
            <!-- 第三行：徽位 Position (如：九) -->
            <div class="h-5 flex items-center justify-center">{{ note.guqin.position || ' ' }}</div>
            <!-- 第四行：左手技法 Finger (如：大) -->
            <div class="h-5 flex items-center justify-center">{{ note.guqin.finger || ' ' }}</div>
          </div>
          <div v-else class="lyrics-section flex flex-col w-full items-center h-[116px]">
            <!-- Dash 占位符为空，保持纵向对齐 -->
          </div>
        </div>
        
        <!-- 小节线 -->
        <div v-if="mIndex < parsedMeasures.length - 1" class="w-px h-[160px] bg-gray-300 mx-3 mt-4"></div>
      </div>
      
    </div>
    
    <div v-else class="text-gray-500 italic text-sm">
      未找到可解析的音符节点 (&lt;note&gt;)
    </div>
  </div>
</template>

<style scoped>
.guqin-score-container {
  /* 去除原有的毛边背景，呈现更干净的横向打谱风格 */
  background-color: #fafaf8;
  border-radius: 0.5rem;
}
</style>
