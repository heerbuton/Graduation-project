<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import axios from 'axios'
import ScoreModelRenderer from './components/ScoreModelRenderer.vue'
import HorizontalStepper from './components/HorizontalStepper.vue'
import { convertLlmResultToScoreModel } from './utils/scoreModel.js'

const BACKEND_BASE = 'http://127.0.0.1:5000'

// 状态管理
const fileInput = ref(null)
const selectedFile = ref(null)
const previewImage = ref(null)
const pipelineStatus = ref('idle') // idle, uploading, cv, topology, llm, xml, success, error
const errorMessage = ref('')
const activeTab = ref('upload') // upload, cv, topology, llm, xml

// 数据存储
const originalImageUrl = ref('')
const yoloBoxes = ref([])
const jianziSequence = ref([])
const topologyJson = ref(null)
const llmResult = ref([])
const musicXml = ref('')
const scoreModel = ref(null)

// CV 可视化修正交互状态
const imageCanvas = ref(null)
const overlayViewport = ref(null)
const imageNaturalSize = ref({ width: 0, height: 0 })
const imageRenderMeta = ref({
  canvasWidth: 0,
  canvasHeight: 0,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  drawWidth: 0,
  drawHeight: 0,
  viewX1: 0,
  viewY1: 0,
  viewWidth: 0,
  viewHeight: 0
})
const selectedGroupId = ref('')
const hoveredGroupId = ref('')
const hoverTooltipLayout = ref({ left: 8, top: 8, width: 320 })
const groupEditorFields = ref({
  right_fingering: '',
  left_fingering: '',
  left_finger: '',
  hui: '',
  xian: ''
})
const groupEditorError = ref('')
const reflowStatusMessage = ref('')
const isReflowing = ref(false)

const cloneJson = (value, fallback = null) => {
  try {
    return JSON.parse(JSON.stringify(value))
  } catch {
    return fallback
  }
}

const toNumber = (value, fallback = 0) => {
  const num = Number(value)
  return Number.isFinite(num) ? num : fallback
}

const extractSequenceOrder = (groupId) => {
  const matched = String(groupId || '').match(/(\d+)/)
  return matched ? toNumber(matched[1], Number.MAX_SAFE_INTEGER) : Number.MAX_SAFE_INTEGER
}

const resolveGroupBbox = (group) => {
  if (!group || typeof group !== 'object') return null

  if (Array.isArray(group.group_bbox) && group.group_bbox.length === 4) {
    const [x1, y1, x2, y2] = group.group_bbox.map(v => toNumber(v))
    if (x2 > x1 && y2 > y1) return [x1, y1, x2, y2]
  }

  const components = Array.isArray(group.components) ? group.components : []
  if (components.length === 0) return null

  const xValues = []
  const yValues = []
  components.forEach((item) => {
    if (!Array.isArray(item?.bbox) || item.bbox.length !== 4) return
    xValues.push(toNumber(item.bbox[0]), toNumber(item.bbox[2]))
    yValues.push(toNumber(item.bbox[1]), toNumber(item.bbox[3]))
  })

  if (!xValues.length || !yValues.length) return null
  const x1 = Math.min(...xValues)
  const y1 = Math.min(...yValues)
  const x2 = Math.max(...xValues)
  const y2 = Math.max(...yValues)

  return x2 > x1 && y2 > y1 ? [x1, y1, x2, y2] : null
}

const formatJson = (value) => {
  try {
    return JSON.stringify(value ?? {}, null, 2)
  } catch {
    return '{}'
  }
}

const sortedTopologyEntries = computed(() => {
  if (!topologyJson.value || typeof topologyJson.value !== 'object') return []
  return Object.entries(topologyJson.value)
    .filter(([, group]) => group && typeof group === 'object')
    .sort((a, b) => extractSequenceOrder(a[0]) - extractSequenceOrder(b[0]))
    .map(([groupId, group]) => ({
      groupId,
      group
    }))
})

const topologyDisplayEntries = computed(() => {
  return sortedTopologyEntries.value.map((item) => {
    const isDeleted = Boolean(item.group.__deleted)
    const markerType = String(item.group?.marker_type || '').trim().toLowerCase()
    const isMarker = Boolean(
      item.group?.is_marker ||
      item.group?.is_section_start ||
      item.group?.is_section_end ||
      markerType === 'start' ||
      markerType === 'end'
    )
    const excludeFromLlm = isDeleted || isMarker
    const displayGroupLabel = String(item.groupId)
    return {
      ...item,
      isDeleted,
      isMarker,
      excludeFromLlm,
      displayGroupLabel
    }
  })
})

const totalTopologyGroupCount = computed(() => topologyDisplayEntries.value.length)
const deletedTopologyGroupCount = computed(
  () => topologyDisplayEntries.value.filter(item => item.isDeleted).length
)
const markerTopologyGroupCount = computed(
  () => topologyDisplayEntries.value.filter(item => item.isMarker && !item.isDeleted).length
)
const activeTopologyGroupCount = computed(
  () => topologyDisplayEntries.value.filter(item => !item.excludeFromLlm).length
)

const topologyOverlayBoxes = computed(() => {
  return topologyDisplayEntries.value
    .map((item) => {
      const bbox = resolveGroupBbox(item.group)
      if (!bbox) return null
      return {
        ...item,
        bbox
      }
    })
    .filter(Boolean)
})

const hoveredTopologyGroup = computed(
  () => topologyOverlayBoxes.value.find(item => item.groupId === hoveredGroupId.value) || null
)
const selectedEditorDisplay = computed(
  () => topologyDisplayEntries.value.find(item => item.groupId === selectedGroupId.value) || null
)

const hoveredTopologyPromptFields = computed(() => {
  if (!hoveredTopologyGroup.value) return null
  const group = hoveredTopologyGroup.value.group || {}
  return {
    'Right Hand': String(group.right_fingering || ''),
    'Left Hand': String(group.left_fingering || ''),
    'Left Finger': String(group.left_finger || ''),
    'Position(Hui)': String(group.hui || ''),
    'String(Xian)': String(group.xian || '')
  }
})

const hoveredTopologyPromptPreview = computed(() => {
  if (!hoveredTopologyPromptFields.value) return ''
  return formatJson(hoveredTopologyPromptFields.value)
})

const topologyStageStyle = computed(() => {
  const canvasWidth = Math.max(0, Math.floor(toNumber(imageRenderMeta.value.canvasWidth, 0)))
  const canvasHeight = Math.max(0, Math.floor(toNumber(imageRenderMeta.value.canvasHeight, 0)))
  if (canvasWidth <= 0 || canvasHeight <= 0) {
    return { width: '100%', minHeight: '320px' }
  }
  return {
    width: `${canvasWidth}px`,
    height: `${canvasHeight}px`
  }
})

const getTopologyBoxStyle = (bbox) => {
  if (!Array.isArray(bbox) || bbox.length !== 4) return { display: 'none' }
  const x1 = toNumber(bbox[0], 0)
  const y1 = toNumber(bbox[1], 0)
  const x2 = toNumber(bbox[2], 0)
  const y2 = toNumber(bbox[3], 0)
  if (x2 <= x1 || y2 <= y1) return { display: 'none' }

  const meta = imageRenderMeta.value
  const canvasWidthRaw = toNumber(meta.canvasWidth, 0)
  const canvasHeightRaw = toNumber(meta.canvasHeight, 0)
  const scale = toNumber(meta.scale, 0)
  const viewX1 = toNumber(meta.viewX1, 0)
  const viewY1 = toNumber(meta.viewY1, 0)
  if (canvasWidthRaw <= 0 || canvasHeightRaw <= 0 || scale <= 0) return { display: 'none' }
  const canvasWidth = Math.max(1, canvasWidthRaw)
  const canvasHeight = Math.max(1, canvasHeightRaw)

  const leftPx = toNumber(meta.offsetX, 0) + (x1 - viewX1) * scale
  const topPx = toNumber(meta.offsetY, 0) + (y1 - viewY1) * scale
  const widthPx = (x2 - x1) * scale
  const heightPx = (y2 - y1) * scale

  return {
    left: `${(leftPx / canvasWidth) * 100}%`,
    top: `${(topPx / canvasHeight) * 100}%`,
    width: `${(widthPx / canvasWidth) * 100}%`,
    height: `${(heightPx / canvasHeight) * 100}%`
  }
}

const getSafeBbox = (bbox) => {
  if (!Array.isArray(bbox) || bbox.length !== 4) return null
  const x1 = toNumber(bbox[0], 0)
  const y1 = toNumber(bbox[1], 0)
  const x2 = toNumber(bbox[2], 0)
  const y2 = toNumber(bbox[3], 0)
  if (x2 <= x1 || y2 <= y1) return null
  return [x1, y1, x2, y2]
}

const selectedGroupObject = computed(
  () => (selectedGroupId.value ? topologyJson.value?.[selectedGroupId.value] || null : null)
)

const selectedGroupCropBbox = computed(() => getSafeBbox(resolveGroupBbox(selectedGroupObject.value)))

const computeFocusSourceBbox = (imgWidth, imgHeight) => {
  const candidateBboxes = []

  topologyDisplayEntries.value.forEach((item) => {
    const bbox = getSafeBbox(resolveGroupBbox(item.group))
    if (bbox) candidateBboxes.push(bbox)
  })

  if (candidateBboxes.length === 0) {
    yoloBoxes.value.forEach((box) => {
      const bbox = getSafeBbox(box?.bbox)
      if (bbox) candidateBboxes.push(bbox)
    })
  }

  if (candidateBboxes.length === 0) {
    return { x1: 0, y1: 0, x2: imgWidth, y2: imgHeight }
  }

  const minX = Math.min(...candidateBboxes.map(b => b[0]))
  const minY = Math.min(...candidateBboxes.map(b => b[1]))
  const maxX = Math.max(...candidateBboxes.map(b => b[2]))
  const maxY = Math.max(...candidateBboxes.map(b => b[3]))

  const spanX = Math.max(1, maxX - minX)
  const spanY = Math.max(1, maxY - minY)
  const padX = Math.max(16, spanX * 0.08)
  const padY = Math.max(16, spanY * 0.08)

  return {
    x1: Math.max(0, minX - padX),
    y1: Math.max(0, minY - padY),
    x2: Math.min(imgWidth, maxX + padX),
    y2: Math.min(imgHeight, maxY + padY)
  }
}

const buildCropImageStyle = (bbox, viewportWidth = 180, viewportHeight = 80) => {
  const safe = getSafeBbox(bbox)
  const naturalWidth = toNumber(imageNaturalSize.value.width, 0)
  const naturalHeight = toNumber(imageNaturalSize.value.height, 0)
  if (!safe || naturalWidth <= 0 || naturalHeight <= 0) return {}

  const pad = 8
  const sx = Math.max(0, safe[0] - pad)
  const sy = Math.max(0, safe[1] - pad)
  const ex = Math.min(naturalWidth, safe[2] + pad)
  const ey = Math.min(naturalHeight, safe[3] + pad)
  const cropW = Math.max(1, ex - sx)
  const cropH = Math.max(1, ey - sy)
  const scale = Math.min(viewportWidth / cropW, viewportHeight / cropH)
  const scaledImageW = naturalWidth * scale
  const scaledImageH = naturalHeight * scale
  const offsetX = -sx * scale + (viewportWidth - cropW * scale) / 2
  const offsetY = -sy * scale + (viewportHeight - cropH * scale) / 2

  return {
    width: `${scaledImageW}px`,
    height: `${scaledImageH}px`,
    left: `${offsetX}px`,
    top: `${offsetY}px`
  }
}

const updateHoverTooltipPosition = (event) => {
  if (!event) return

  const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1200
  const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 800
  const cursorX = event.clientX
  const cursorY = event.clientY
  const width = Math.max(240, Math.min(360, Math.floor(viewportWidth * 0.28)))
  const height = 190
  const gap = 18
  const edgePadding = 8
  const preferLeft = cursorX > viewportWidth * 0.6

  let left = preferLeft ? (cursorX - width - gap) : (cursorX + gap)
  if (left + width > viewportWidth - edgePadding) {
    left = cursorX - width - gap
  }
  if (left < edgePadding) {
    left = Math.min(cursorX + gap, viewportWidth - width - edgePadding)
  }

  let top = cursorY - Math.round(height * 0.36)
  if (top + height > viewportHeight - edgePadding) {
    top = viewportHeight - height - edgePadding
  }
  if (top < edgePadding) {
    top = edgePadding
  }

  const cursorGap = 16
  const cursorInsideTooltip =
    cursorX >= left - cursorGap &&
    cursorX <= left + width + cursorGap &&
    cursorY >= top - cursorGap &&
    cursorY <= top + height + cursorGap

  if (cursorInsideTooltip) {
    const belowTop = cursorY + gap
    const aboveTop = cursorY - height - gap
    if (belowTop + height <= viewportHeight - edgePadding) {
      top = belowTop
    } else if (aboveTop >= edgePadding) {
      top = aboveTop
    }
  }

  hoverTooltipLayout.value = { left, top, width }
}

const onGroupBoxMouseEnter = (groupId, event) => {
  hoveredGroupId.value = groupId
  updateHoverTooltipPosition(event)
}

const onGroupBoxMouseMove = (event) => {
  if (!hoveredGroupId.value) return
  updateHoverTooltipPosition(event)
}

const onGroupBoxMouseLeave = () => {
  hoveredGroupId.value = ''
}

const hoverTooltipStyle = computed(() => {
  if (!hoveredTopologyGroup.value) return {}
  return {
    position: 'fixed',
    left: `${hoverTooltipLayout.value.left}px`,
    top: `${hoverTooltipLayout.value.top}px`,
    width: `${hoverTooltipLayout.value.width}px`
  }
})

let cvRenderTimer = null
const scheduleRenderImageWithBoxes = (delayMs = 0) => {
  if (cvRenderTimer) {
    clearTimeout(cvRenderTimer)
    cvRenderTimer = null
  }
  cvRenderTimer = setTimeout(() => {
    renderImageWithBoxes()
  }, Math.max(0, delayMs))
}

const handleWindowResize = () => {
  if (activeTab.value !== 'cv' || !originalImageUrl.value) return
  scheduleRenderImageWithBoxes(0)
}

onMounted(() => {
  window.addEventListener('resize', handleWindowResize)
})

watch(
  () => activeTab.value,
  (tab) => {
    if (tab === 'cv' && originalImageUrl.value) {
      scheduleRenderImageWithBoxes(0)
    }
  }
)

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleWindowResize)
  if (cvRenderTimer) {
    clearTimeout(cvRenderTimer)
    cvRenderTimer = null
  }
})

const resetTopologyInteractionState = () => {
  selectedGroupId.value = ''
  hoveredGroupId.value = ''
  hoverTooltipLayout.value = { left: 8, top: 8, width: 320 }
  groupEditorFields.value = {
    right_fingering: '',
    left_fingering: '',
    left_finger: '',
    hui: '',
    xian: ''
  }
  groupEditorError.value = ''
  reflowStatusMessage.value = ''
}

const loadEditorFromGroup = (groupId) => {
  const group = topologyJson.value?.[groupId]
  if (!group || typeof group !== 'object') {
    groupEditorFields.value = {
      right_fingering: '',
      left_fingering: '',
      left_finger: '',
      hui: '',
      xian: ''
    }
    return
  }
  groupEditorFields.value = {
    right_fingering: String(group.right_fingering || ''),
    left_fingering: String(group.left_fingering || ''),
    left_finger: String(group.left_finger || ''),
    hui: String(group.hui || ''),
    xian: String(group.xian || '')
  }
}

const openGroupEditor = (groupId) => {
  selectedGroupId.value = groupId
  groupEditorError.value = ''
  loadEditorFromGroup(groupId)
}

const closeGroupEditor = () => {
  selectedGroupId.value = ''
  groupEditorFields.value = {
    right_fingering: '',
    left_fingering: '',
    left_finger: '',
    hui: '',
    xian: ''
  }
  groupEditorError.value = ''
}

const restoreSelectedGroup = () => {
  if (!selectedGroupId.value) return
  loadEditorFromGroup(selectedGroupId.value)
  groupEditorError.value = ''
}

const saveSelectedGroupJson = () => {
  if (!selectedGroupId.value || !topologyJson.value?.[selectedGroupId.value]) return
  const currentGroup = topologyJson.value[selectedGroupId.value]
    const mergedGroup = { ...currentGroup }
  mergedGroup.right_fingering = String(groupEditorFields.value.right_fingering || '').trim()
  mergedGroup.left_fingering = String(groupEditorFields.value.left_fingering || '').trim()
  mergedGroup.left_finger = String(groupEditorFields.value.left_finger || '').trim()
  mergedGroup.hui = String(groupEditorFields.value.hui || '').trim()
  mergedGroup.xian = String(groupEditorFields.value.xian || '').trim()

  // 兼容下游旧字段
  mergedGroup.fingering = mergedGroup.right_fingering || mergedGroup.left_fingering || ''
  mergedGroup.finger = mergedGroup.left_finger
  mergedGroup.position = mergedGroup.hui
  mergedGroup.string = mergedGroup.xian

  const bbox = resolveGroupBbox(mergedGroup)
  if (bbox) mergedGroup.group_bbox = bbox
  mergedGroup.sequence_index = Math.max(1, Math.round(toNumber(
    mergedGroup.sequence_index,
    extractSequenceOrder(selectedGroupId.value)
  )))

  topologyJson.value = {
    ...topologyJson.value,
    [selectedGroupId.value]: mergedGroup
  }

  groupEditorError.value = ''
  const selectedDisplay = topologyDisplayEntries.value.find(item => item.groupId === selectedGroupId.value)
  const selectedLabel = selectedDisplay?.displayGroupLabel || selectedGroupId.value
  reflowStatusMessage.value = `${selectedLabel} 已保存，可继续重跑后续流程。`
}

const toggleGroupDeleted = (groupId = selectedGroupId.value) => {
  if (!groupId || !topologyJson.value?.[groupId]) return
  const currentGroup = topologyJson.value[groupId]
  const updatedGroup = {
    ...currentGroup,
    __deleted: !Boolean(currentGroup.__deleted)
  }
  topologyJson.value = {
    ...topologyJson.value,
    [groupId]: updatedGroup
  }

  if (selectedGroupId.value === groupId) {
    loadEditorFromGroup(groupId)
  }

  const selectedDisplay = topologyDisplayEntries.value.find(item => item.groupId === groupId)
  const selectedLabel = selectedDisplay?.displayGroupLabel || groupId
  reflowStatusMessage.value = updatedGroup.__deleted
    ? `${selectedLabel} 已标记删除。`
    : `${selectedLabel} 已取消删除标记。`
}

const buildReflowTopologyPayload = () => {
  const payload = {}

  topologyDisplayEntries.value.forEach((item) => {
    const { groupId, group, excludeFromLlm } = item
    if (excludeFromLlm) return

    const cleaned = cloneJson(group, {})
    delete cleaned.__deleted
    cleaned.sequence_index = extractSequenceOrder(groupId)

    const rightFingering = String(cleaned.right_fingering || '').trim()
    const leftFingering = String(cleaned.left_fingering || '').trim()
    const leftFinger = String(cleaned.left_finger || cleaned.finger || '').trim()
    const hui = String(cleaned.hui || cleaned.position || '').trim()
    const xian = String(cleaned.xian || cleaned.string || cleaned.xian_digit || '').trim()

    cleaned.right_fingering = rightFingering
    cleaned.left_fingering = leftFingering
    cleaned.left_finger = leftFinger
    cleaned.hui = hui
    cleaned.xian = xian
    cleaned.fingering = String(cleaned.fingering || rightFingering || leftFingering || '').trim()
    cleaned.finger = String(cleaned.finger || leftFinger || '').trim()
    cleaned.position = String(cleaned.position || hui || '').trim()
    cleaned.string = String(cleaned.string || xian || '').trim()

    const bbox = resolveGroupBbox(cleaned)
    if (bbox) cleaned.group_bbox = bbox

    // 保持后端 group_x 原始编号，不做前端重命名
    payload[String(groupId)] = cleaned
  })

  return payload
}

const rerunFromEditedTopology = async () => {
  const topologyPayload = buildReflowTopologyPayload()
  if (Object.keys(topologyPayload).length === 0) {
    errorMessage.value = '至少保留一个有效减字组后再重跑。'
    return
  }
  const previousSelectedGroupId = selectedGroupId.value

  isReflowing.value = true
  errorMessage.value = ''
  reflowStatusMessage.value = '正在基于修正结果重跑后续流程...'
  pipelineStatus.value = 'topology'
  activeTab.value = 'topology'

  try {
    const response = await axios.post(`${BACKEND_BASE}/api/reflow_from_topology`, {
      topology_json: topologyPayload
    })
    const data = response?.data?.data || {}

    topologyJson.value = cloneJson(data.topology_json, topologyPayload)
    jianziSequence.value = cloneJson(data.jianzi_sequence, [])
    llmResult.value = cloneJson(data.llm_result, [])
    musicXml.value = data.music_xml || ''
    if (Array.isArray(llmResult.value) && llmResult.value.length > 0) {
      scoreModel.value = convertLlmResultToScoreModel(llmResult.value, { strict: false })
    } else {
      scoreModel.value = data.score_model || convertLlmResultToScoreModel([], { strict: false })
    }

    await delay(700)
    pipelineStatus.value = 'llm'
    activeTab.value = 'llm'
    await delay(700)
    pipelineStatus.value = 'xml'
    activeTab.value = 'xml'
    await delay(300)
    pipelineStatus.value = 'success'

    reflowStatusMessage.value = '修正结果已应用，后续流程刷新完成。'
    selectedGroupId.value = (
      previousSelectedGroupId &&
      topologyJson.value &&
      Object.prototype.hasOwnProperty.call(topologyJson.value, previousSelectedGroupId)
    )
      ? previousSelectedGroupId
      : ''
    if (selectedGroupId.value) {
      await nextTick()
      loadEditorFromGroup(selectedGroupId.value)
    }
  } catch (error) {
    pipelineStatus.value = 'error'
    errorMessage.value = error.response?.data?.message || error.message || '重跑失败'
  } finally {
    isReflowing.value = false
  }
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

  // Reset states
  pipelineStatus.value = 'idle'
  activeTab.value = 'upload'
  originalImageUrl.value = ''
  yoloBoxes.value = []
  jianziSequence.value = []
  topologyJson.value = null
  llmResult.value = []
  musicXml.value = ''
  scoreModel.value = null
  imageNaturalSize.value = { width: 0, height: 0 }
  imageRenderMeta.value = {
    canvasWidth: 0,
    canvasHeight: 0,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    drawWidth: 0,
    drawHeight: 0,
    viewX1: 0,
    viewY1: 0,
    viewWidth: 0,
    viewHeight: 0
  }
  resetTopologyInteractionState()

  const reader = new FileReader()
  reader.onload = (e) => {
    previewImage.value = e.target.result
  }
  reader.readAsDataURL(file)
}

// 图片渲染 Canvas (带有 YOLO Box)
const renderImageWithBoxes = () => {
  if (!imageCanvas.value || !originalImageUrl.value) return

  const canvas = imageCanvas.value
  const viewport = overlayViewport.value || canvas.parentElement
  const viewportWidth = Math.max(1, Math.floor(toNumber(viewport?.clientWidth || canvas.clientWidth, 1)))
  const viewportHeight = Math.max(1, Math.floor(toNumber(viewport?.clientHeight || canvas.clientHeight, 1)))

  const ctx = canvas.getContext('2d')
  const img = new Image()
  img.onload = () => {
    const focus = computeFocusSourceBbox(img.width, img.height)
    const viewX1 = Math.max(0, toNumber(focus.x1, 0))
    const viewY1 = Math.max(0, toNumber(focus.y1, 0))
    const viewWidth = Math.max(1, toNumber(focus.x2, img.width) - viewX1)
    const viewHeight = Math.max(1, toNumber(focus.y2, img.height) - viewY1)

    // 右侧交互图按宽度优先铺满，竖向溢出时由容器滚动查看
    const scale = viewportWidth / viewWidth
    const drawWidth = viewWidth * scale
    const drawHeight = viewHeight * scale
    const stageWidth = Math.max(1, Math.round(viewportWidth))
    const stageHeight = Math.max(1, Math.round(Math.max(drawHeight, viewportHeight)))
    const offsetX = (stageWidth - drawWidth) / 2
    const offsetY = drawHeight < viewportHeight ? (stageHeight - drawHeight) / 2 : 0

    canvas.width = stageWidth
    canvas.height = stageHeight

    imageNaturalSize.value = { width: img.width, height: img.height }
    imageRenderMeta.value = {
      canvasWidth: stageWidth,
      canvasHeight: stageHeight,
      scale,
      offsetX,
      offsetY,
      drawWidth,
      drawHeight,
      viewX1,
      viewY1,
      viewWidth,
      viewHeight
    }

    ctx.clearRect(0, 0, stageWidth, stageHeight)
    ctx.drawImage(
      img,
      viewX1,
      viewY1,
      viewWidth,
      viewHeight,
      offsetX,
      offsetY,
      drawWidth,
      drawHeight
    )

    // 右侧底图只显示原图，交互框由 topologyOverlayBoxes 单独叠加
  }
  img.src = `${BACKEND_BASE}${originalImageUrl.value}`
}

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms))

const applyPipelineResult = (data, fallbackImageUrl = '') => {
  originalImageUrl.value = data.original_image_url || fallbackImageUrl
  yoloBoxes.value = cloneJson(data.yolo_boxes, [])
  jianziSequence.value = cloneJson(data.jianzi_sequence, [])
  topologyJson.value = cloneJson(data.topology_json, null)
  llmResult.value = cloneJson(data.llm_result, [])
  musicXml.value = data.music_xml || ''
  imageNaturalSize.value = { width: 0, height: 0 }
  imageRenderMeta.value = {
    canvasWidth: 0,
    canvasHeight: 0,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    drawWidth: 0,
    drawHeight: 0,
    viewX1: 0,
    viewY1: 0,
    viewWidth: 0,
    viewHeight: 0
  }
  resetTopologyInteractionState()

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
    const response = await axios.post(`${BACKEND_BASE}/api/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    const data = response.data.data

    // CV 阶段
    pipelineStatus.value = 'cv'
    activeTab.value = 'cv'
    applyPipelineResult(data)
    scheduleRenderImageWithBoxes(120)
    await delay(3000)

    // Topology 阶段
    pipelineStatus.value = 'topology'
    activeTab.value = 'topology'
    await delay(3000)

    // LLM 阶段
    pipelineStatus.value = 'llm'
    activeTab.value = 'llm'
    await delay(3500)

    // XML 打谱阶段
    pipelineStatus.value = 'xml'
    activeTab.value = 'xml'
    await delay(2500)

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
    const res = await axios.get(`${BACKEND_BASE}/static/uploads/testpicture-1.jpg_result.json`, {
      responseType: 'text',
      transformResponse: [(value) => value]
    })
    const text = String(res.data || '').replace(/^\uFEFF/, '')
    const data = JSON.parse(text)

    pipelineStatus.value = 'cv'
    activeTab.value = 'cv'
    applyPipelineResult(data, '/static/uploads/testpicture-1.jpg')
    scheduleRenderImageWithBoxes(120)
    await delay(3000)

    pipelineStatus.value = 'topology'
    activeTab.value = 'topology'
    await delay(3000)

    pipelineStatus.value = 'llm'
    activeTab.value = 'llm'
    await delay(3500)

    pipelineStatus.value = 'xml'
    activeTab.value = 'xml'
    await delay(2500)

    pipelineStatus.value = 'success'
  } catch (error) {
    pipelineStatus.value = 'error'
    errorMessage.value = error.response?.data?.message || error.message || '加载预计算结果失败'
  }
}

const reuploadImage = async () => {
  resetAll()
  await nextTick()
  setTimeout(() => {
    fileInput.value?.click()
  }, 0)
}

const resetAll = () => {
  originalImageUrl.value = ''
  previewImage.value = null
  selectedFile.value = null
  pipelineStatus.value = 'idle'
  activeTab.value = 'upload'
  musicXml.value = ''
  scoreModel.value = null
  yoloBoxes.value = []
  jianziSequence.value = []
  llmResult.value = []
  topologyJson.value = null
  imageNaturalSize.value = { width: 0, height: 0 }
  imageRenderMeta.value = {
    canvasWidth: 0,
    canvasHeight: 0,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    drawWidth: 0,
    drawHeight: 0,
    viewX1: 0,
    viewY1: 0,
    viewWidth: 0,
    viewHeight: 0
  }
  errorMessage.value = ''
  resetTopologyInteractionState()
}
</script>

<template>
  <div class="min-h-screen flex flex-col font-sans relative bg-[#0f172a] overflow-x-hidden">
    
    <!-- 装饰性光晕背景 -->
    <div class="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/10 blur-[150px] rounded-full pointer-events-none z-0"></div>
    <div class="fixed bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-accent/10 blur-[120px] rounded-full pointer-events-none z-0"></div>

    <!-- 顶栏导航 -->
    <header class="relative z-20 bg-black/40 backdrop-blur-xl border-b border-white/10 p-5 shadow-xl flex justify-between items-center transition-all">
      <div class="flex items-center gap-4">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white font-guqin font-bold text-xl shadow-lg shadow-primary/30">琴</div>
        <h1 class="text-2xl font-medium tracking-widest text-slate-100 font-guqin truncate">
          伯牙解谱系统 <span class="text-base font-sans tracking-normal opacity-40 ml-3 border-l border-white/20 pl-3">全景解构引擎</span>
        </h1>
      </div>
      <div class="flex items-center gap-4">
        <button @click="reuploadImage" class="text-sm font-medium text-cyan-200 border border-cyan-400/35 bg-cyan-500/10 hover:bg-cyan-500/20 hover:text-white px-5 py-2.5 rounded-lg transition-all duration-300 shadow-lg shadow-cyan-500/10">
          重新上传图片
        </button>
        <button @click="loadSavedTestPictureResult" class="text-sm font-medium text-emerald-300 border border-emerald-500/30 bg-emerald-500/10 hover:bg-emerald-500/20 hover:text-white px-6 py-2.5 rounded-lg transition-all duration-300 shadow-lg shadow-emerald-500/10">
          Demo：全景阵列演示
        </button>
      </div>
    </header>

    <!-- 顶部主线：交互式流水线 Stepper -->
    <section v-show="pipelineStatus !== 'idle' && pipelineStatus !== 'uploading'" class="w-full bg-black/30 backdrop-blur-3xl border-b border-white/5 shadow-2xl relative z-10 transition-all">
      <div class="max-w-[1400px] mx-auto">
        <HorizontalStepper 
          :pipelineStatus="pipelineStatus" 
          :activeTab="activeTab" 
          @update:activeTab="v => activeTab = v"
        />
      </div>
    </section>

    <!-- 主视觉容器：下方跟随的内容大屏 -->
    <main class="relative z-10 w-full p-6 md:p-8 flex-1 flex flex-col items-center">
      
      <!--==================== 视图 0：起航上传区 ====================-->
      <div v-show="activeTab === 'upload'" class="w-full max-w-4xl glass-panel p-1 border border-white/10 rounded-2xl overflow-hidden mt-[15vh] animate-fade-in shadow-2xl">
        <div class="bg-black/40 rounded-xl p-8 flex flex-col items-center h-[500px]">
          <div class="w-full flex justify-between items-center mb-6">
            <h2 class="text-xl font-medium text-slate-300 tracking-wider">上传高清大屏曲谱</h2>
            <button v-if="originalImageUrl || previewImage" @click="resetAll" class="text-xs text-slate-400 hover:text-white transition-colors flex items-center gap-1 bg-white/5 px-3 py-1.5 rounded-md hover:bg-white/10 border border-transparent hover:border-white/20">清理重置</button>
          </div>

          <!-- 拖拽上传 -->
          <div v-if="!originalImageUrl && !previewImage" @dragover="onDragOver" @drop="onDrop" class="w-full flex-1 border-2 border-dashed border-slate-600/50 hover:border-primary/50 text-slate-400 hover:text-primary rounded-2xl flex flex-col items-center justify-center p-8 transition-all duration-300 cursor-pointer group" @click="fileInput.click()">
            <input type="file" ref="fileInput" class="hidden" accept="image/*" @change="onFileSelectChange" />
            <div class="w-20 h-20 rounded-2xl bg-slate-800 flex items-center justify-center group-hover:scale-110 group-hover:bg-primary/20 transition-all duration-300 mb-6 shadow-xl border border-white/5">
                <svg class="w-8 h-8 group-hover:text-primary transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
            </div>
            <p class="font-medium mb-2 text-lg tracking-wider">拖拽曲谱扫描图或点击上传</p>
            <p class="text-sm opacity-60">超清解析支持无上限大图输入</p>
          </div>
          
          <!-- 预览并启动 -->
          <div v-else-if="previewImage && !originalImageUrl" class="w-full flex-1 flex flex-col items-center justify-center animate-fade-in gap-8">
            <img :src="previewImage" class="max-h-[300px] object-contain rounded-xl border border-white/20 shadow-2xl" alt="预览">
            <button @click="uploadAndProcess" :disabled="pipelineStatus !== 'idle' && pipelineStatus !== 'error'" class="bg-gradient-to-r from-primary to-emerald-600 text-white font-medium py-3.5 px-10 rounded-full shadow-lg shadow-primary/25 hover:shadow-primary/50 transition-all duration-300 hover:scale-105 tracking-widest text-lg border border-white/20 flex items-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed">
              <svg class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
              开始打谱
            </button>
          </div>
        </div>
      </div>

      <!--==================== 视图 1：YOLO CV 双显 ====================-->
      <div v-show="activeTab === 'cv'" class="w-full max-w-[1700px] bg-black/20 glass-panel rounded-2xl animate-fade-in flex flex-col lg:flex-row gap-6 p-6 min-h-[600px] border border-white/10 shadow-2xl">
        <!-- 左半：原图 -->
        <div class="lg:flex-[4] relative rounded-xl border border-white/10 bg-black/60 overflow-hidden flex items-center justify-center p-2 min-h-[400px]">
          <div class="absolute top-4 left-4 z-10 bg-black/60 backdrop-blur-md text-white/80 px-4 py-1.5 text-xs rounded-lg border border-white/10 shadow-lg font-mono">
            Original Source Image
          </div>
          <img v-if="originalImageUrl" :src="BACKEND_BASE + originalImageUrl" class="w-full h-full object-contain hover:scale-[1.02] transition-transform duration-700" alt="原图" />
        </div>
        
        <!-- 右半：框图 -->
        <div class="lg:flex-[6] rounded-xl border border-emerald-500/20 bg-emerald-950/20 overflow-hidden p-2 shadow-[0_0_30px_rgba(16,185,129,0.05)] min-h-[400px] flex flex-col gap-2">
          <div class="flex flex-col gap-2 px-1">
            <div class="flex items-start justify-between gap-2 flex-wrap">
              <div class="bg-slate-950/90 backdrop-blur-md text-emerald-100 border border-emerald-300/60 px-4 py-1.5 text-sm rounded-lg shadow-[0_0_18px_rgba(16,185,129,0.28)] font-mono font-semibold flex items-center gap-3">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping absolute"></span>
                <span class="w-2 h-2 rounded-full bg-emerald-400 relative"></span>
                交互校对视图 ({{ topologyOverlayBoxes.length }} groups)
              </div>

              <button
                @click="rerunFromEditedTopology"
                :disabled="isReflowing || activeTopologyGroupCount === 0"
                class="text-xs sm:text-sm bg-gradient-to-r from-cyan-600 to-emerald-600 text-white px-4 py-2 rounded-md border border-white/25 hover:brightness-110 transition disabled:opacity-50 disabled:cursor-not-allowed font-mono font-semibold shadow-lg shadow-cyan-900/40"
              >
                {{ isReflowing ? '重跑中...' : '应用修正并重跑流程' }}
              </button>
            </div>

            <div class="flex items-center gap-2 flex-wrap">
              <span class="text-xs sm:text-sm bg-slate-950/90 text-cyan-100 border border-cyan-300/60 px-3 py-1.5 rounded-md font-mono font-semibold shadow-lg">
                LLM Active {{ activeTopologyGroupCount }} / Marker {{ markerTopologyGroupCount }} / Deleted {{ deletedTopologyGroupCount }}
              </span>
            </div>
            <div
              v-if="reflowStatusMessage"
              class="self-start max-w-full sm:max-w-[620px] text-xs font-mono font-semibold text-cyan-100 bg-slate-950/92 border border-cyan-300/60 rounded-md px-3 py-1.5 backdrop-blur shadow-lg"
            >
              {{ reflowStatusMessage }}
            </div>
          </div>

          <div ref="overlayViewport" class="relative flex-1 w-full min-h-[320px] overflow-auto">
            <div class="relative mx-auto" :style="topologyStageStyle">
              <!-- 底层：YOLO 渲染图 -->
              <canvas ref="imageCanvas" class="block w-full h-full transition-transform duration-700"></canvas>

              <!-- 顶层：聚合框 overlay -->
              <button
                v-for="item in topologyOverlayBoxes"
                :key="item.groupId"
                class="absolute border-2 rounded-md transition-all duration-200 hover:scale-[1.02] hover:z-40"
                :class="item.isDeleted
                  ? 'border-rose-400/80 bg-rose-500/10 hover:bg-rose-500/20'
                  : (item.isMarker
                    ? 'border-amber-300/90 bg-amber-400/10 hover:bg-amber-300/20'
                    : 'border-cyan-300/90 bg-cyan-400/10 hover:bg-cyan-300/20')"
                :style="getTopologyBoxStyle(item.bbox)"
                @mouseenter="onGroupBoxMouseEnter(item.groupId, $event)"
                @mousemove="onGroupBoxMouseMove($event)"
                @mouseleave="onGroupBoxMouseLeave"
                @click.stop="openGroupEditor(item.groupId)"
              >
                <span
                  class="absolute -top-5 left-0 text-[10px] font-mono px-1.5 py-0.5 rounded border"
                  :class="item.isDeleted
                    ? 'bg-rose-900/80 text-rose-200 border-rose-400/50'
                    : (item.isMarker
                      ? 'bg-amber-900/80 text-amber-100 border-amber-300/50'
                      : 'bg-cyan-900/80 text-cyan-100 border-cyan-300/40')"
                >
                  {{ item.displayGroupLabel }}
                </span>
              </button>
            </div>
          </div>
        </div>

        <teleport to="body">
          <div
            v-if="hoveredTopologyGroup"
            :style="hoverTooltipStyle"
            class="z-[99999] max-h-[52vh] overflow-auto rounded-lg border border-cyan-300/60 bg-slate-950/95 p-3 shadow-2xl backdrop-blur pointer-events-none"
          >
            <div class="text-[11px] text-cyan-100 font-mono mb-2 flex items-center justify-between font-semibold">
              <span>{{ hoveredTopologyGroup.displayGroupLabel }}</span>
              <span class="text-cyan-300">LLM 参数预览</span>
            </div>
            <pre class="text-[11px] leading-4 text-slate-100 font-mono whitespace-pre-wrap">{{ hoveredTopologyPromptPreview }}</pre>
          </div>
        </teleport>

        <!-- 右侧：JSON 编辑器 -->
        <div v-if="selectedGroupId" class="flex-[0.9] rounded-xl border border-cyan-500/25 bg-slate-950/80 min-h-[400px] p-4 flex flex-col shadow-[0_0_30px_rgba(6,182,212,0.08)]">
          <div class="flex items-center justify-between gap-3 mb-3">
            <div class="flex items-center gap-2">
              <h3 class="text-sm text-cyan-200 font-mono tracking-wider">编辑 {{ selectedEditorDisplay?.displayGroupLabel || selectedGroupId }}</h3>
              <span
                v-if="topologyJson?.[selectedGroupId]?.__deleted"
                class="text-[10px] px-2 py-0.5 rounded border border-rose-500/40 text-rose-300 bg-rose-500/10 font-mono"
              >
                已标记删除
              </span>
            </div>
            <button @click="closeGroupEditor" class="text-xs text-slate-400 hover:text-white">关闭</button>
          </div>

          <div class="flex items-center gap-2 mb-3 flex-wrap">
            <button
              @click="saveSelectedGroupJson"
              class="text-xs px-3 py-1.5 rounded-md bg-cyan-600/80 hover:bg-cyan-500 text-white border border-cyan-300/20 transition"
            >
              保存字段
            </button>
            <button
              @click="restoreSelectedGroup"
              class="text-xs px-3 py-1.5 rounded-md bg-slate-700/70 hover:bg-slate-600 text-slate-100 border border-slate-400/30 transition"
            >
              还原
            </button>
            <button
              @click="toggleGroupDeleted()"
              class="text-xs px-3 py-1.5 rounded-md border transition"
              :class="topologyJson?.[selectedGroupId]?.__deleted
                ? 'bg-emerald-600/80 border-emerald-400/40 text-white hover:bg-emerald-500'
                : 'bg-rose-700/70 border-rose-400/40 text-white hover:bg-rose-600'"
            >
              {{ topologyJson?.[selectedGroupId]?.__deleted ? '取消删除标记' : '标记删除' }}
            </button>
          </div>

          <div class="flex-1 rounded-lg bg-black/55 border border-cyan-500/20 p-3 space-y-3">
            <div>
              <label class="block text-[11px] text-slate-300 font-mono mb-1">Right Hand</label>
              <input v-model="groupEditorFields.right_fingering" type="text" class="w-full rounded-md bg-slate-900/90 border border-cyan-500/30 text-slate-100 px-2.5 py-2 text-sm font-guqin outline-none focus:border-cyan-300/70" />
            </div>
            <div>
              <label class="block text-[11px] text-slate-300 font-mono mb-1">Left Hand</label>
              <input v-model="groupEditorFields.left_fingering" type="text" class="w-full rounded-md bg-slate-900/90 border border-cyan-500/30 text-slate-100 px-2.5 py-2 text-sm font-guqin outline-none focus:border-cyan-300/70" />
            </div>
            <div>
              <label class="block text-[11px] text-slate-300 font-mono mb-1">Left Finger</label>
              <input v-model="groupEditorFields.left_finger" type="text" class="w-full rounded-md bg-slate-900/90 border border-cyan-500/30 text-slate-100 px-2.5 py-2 text-sm font-guqin outline-none focus:border-cyan-300/70" />
            </div>
            <div>
              <label class="block text-[11px] text-slate-300 font-mono mb-1">Position(Hui)</label>
              <input v-model="groupEditorFields.hui" type="text" class="w-full rounded-md bg-slate-900/90 border border-cyan-500/30 text-slate-100 px-2.5 py-2 text-sm font-guqin outline-none focus:border-cyan-300/70" />
            </div>
            <div>
              <label class="block text-[11px] text-slate-300 font-mono mb-1">String(Xian)</label>
              <input v-model="groupEditorFields.xian" type="text" class="w-full rounded-md bg-slate-900/90 border border-cyan-500/30 text-slate-100 px-2.5 py-2 text-sm font-guqin outline-none focus:border-cyan-300/70" />
            </div>

            <div class="pt-2 border-t border-white/10">
              <div class="flex items-center justify-between mb-2">
                <p class="text-[11px] text-cyan-200 font-mono">当前 Group 截图</p>
                <p class="text-[10px] text-slate-400 font-mono">{{ selectedEditorDisplay?.displayGroupLabel || selectedGroupId }}</p>
              </div>
              <div class="w-[260px] h-[130px] mx-auto relative overflow-hidden rounded border border-cyan-500/30 bg-slate-950">
                <img
                  v-if="selectedGroupCropBbox && originalImageUrl && imageNaturalSize.width > 0"
                  :src="BACKEND_BASE + originalImageUrl"
                  class="absolute max-w-none pointer-events-none select-none"
                  :style="buildCropImageStyle(selectedGroupCropBbox, 260, 130)"
                  alt="group截图"
                />
                <div v-else class="absolute inset-0 flex items-center justify-center text-[10px] text-slate-500 font-mono">
                  无对应框
                </div>
              </div>
            </div>
          </div>

          <p v-if="groupEditorError" class="mt-3 text-xs text-rose-300 font-mono">{{ groupEditorError }}</p>
          <p v-else class="mt-3 text-xs text-slate-400 font-mono">仅编辑这 5 个传给 LLM 的字段；保存后点“应用修正并重跑后续流程”刷新结果。</p>
        </div>
      </div>

      <!--==================== 视图 2：Topology 拓扑阵列 (Grid) ====================-->
      <!-- 用 v-show 防止每次开合摧毁 DOM，保证滚条位置 -->
      <div v-show="activeTab === 'topology'" class="w-full max-w-[1600px] glass-panel p-6 rounded-2xl animate-fade-in border border-white/10 flex flex-col h-[75vh]">
        <div class="flex items-center justify-between mb-6 pb-4 border-b border-white/10 px-2">
          <h2 class="text-2xl font-light text-cyan-400 tracking-widest font-guqin">空间拓扑反序列化网格</h2>
          <span class="text-xs bg-cyan-400/10 text-cyan-400 px-4 py-1.5 rounded-full border border-cyan-400/30 font-mono shadow-lg">
            LLM Active {{ activeTopologyGroupCount }} / Marker {{ markerTopologyGroupCount }} / Deleted {{ deletedTopologyGroupCount }}
          </span>
        </div>
        
        <!-- 卡片瀑布流 -->
        <div class="flex-1 w-full overflow-y-auto pr-3 pb-4 scrollbar-thin">
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-5 pb-10">
            <!-- 单个 JSON 方块 -->
            <div
              v-for="item in topologyDisplayEntries"
              :key="item.groupId"
              class="bg-gradient-to-br border rounded-xl p-5 transition-all duration-300 relative group overflow-hidden"
              :class="item.isDeleted
                ? 'from-rose-950/40 to-black/60 border-rose-500/35 hover:border-rose-400/60'
                : (item.isMarker
                  ? 'from-amber-950/35 to-black/60 border-amber-500/35 hover:border-amber-300/65'
                  : 'from-black/60 to-cyan-950/30 border-white/5 hover:border-cyan-500/50 hover:shadow-[0_0_20px_rgba(34,211,238,0.15)]')"
            >
               <div class="absolute top-0 right-0 w-20 h-20 bg-cyan-500/5 rounded-full blur-[20px] group-hover:bg-cyan-500/10 transition-colors pointer-events-none"></div>
               
               <div class="flex justify-between items-center mb-4 relative z-10">
                 <span class="text-[10px] font-mono text-cyan-200/70 bg-white/5 px-2 py-0.5 rounded border border-white/5">
                   {{ item.displayGroupLabel.toUpperCase() }}
                 </span>
                 <span class="text-xs font-bold tracking-wider" :class="item.isDeleted ? 'text-rose-300' : (item.isMarker ? 'text-amber-200' : 'text-cyan-300')">
                   {{ item.group.components?.length || 0 }} 散件
                 </span>
               </div>

               <div v-if="item.isDeleted" class="mb-3 text-[10px] font-mono text-rose-200 bg-rose-500/10 border border-rose-400/30 rounded px-2 py-1">
                 此组已标记删除（不会进入后续推理）
               </div>
               <div v-else-if="item.isMarker" class="mb-3 text-[10px] font-mono text-amber-100 bg-amber-500/10 border border-amber-400/35 rounded px-2 py-1">
                 此组为乐章标记（不参与打谱序列）
               </div>
                
                <!-- 结构内容展示 -->
                <div class="flex flex-col gap-2 font-mono text-sm relative z-10 bg-black/30 p-3 rounded-lg border border-white/5">
                  <div class="flex justify-between items-end border-b border-white/5 pb-1"><span class="text-slate-500 text-xs">Right Hand</span> <span class="text-cyan-300 font-guqin text-base">{{ item.group.right_fingering || '-' }}</span></div>
                  <div class="flex justify-between items-end border-b border-white/5 pb-1"><span class="text-slate-500 text-xs">Left Hand</span> <span class="text-cyan-300 font-guqin text-base">{{ item.group.left_fingering || '-' }}</span></div>
                  <div class="flex justify-between items-end border-b border-white/5 pb-1"><span class="text-slate-500 text-xs">Left Finger</span> <span class="text-cyan-300 font-guqin text-base">{{ item.group.left_finger || '-' }}</span></div>
                  <div class="flex justify-between items-end border-b border-white/5 pb-1"><span class="text-slate-500 text-xs">Position(Hui)</span> <span class="text-emerald-300 font-guqin text-base">{{ item.group.hui || '-' }}</span></div>
                  <div class="flex justify-between items-end"><span class="text-slate-500 text-xs">String(Xian)</span> <span class="text-emerald-300 font-guqin text-base">{{ item.group.xian || '-' }}</span></div>
                </div>
             </div>
          </div>
        </div>
      </div>

      <!--==================== 视图 3：LLM 大模型推理阵列 (Grid) ====================-->
      <div v-show="activeTab === 'llm'" class="w-full max-w-[1600px] glass-panel p-6 rounded-2xl animate-fade-in border border-white/10 flex flex-col h-[75vh]">
        <div class="flex items-center justify-between mb-6 pb-4 border-b border-white/10 px-2">
          <h2 class="text-2xl font-light text-amber-500 tracking-widest font-guqin">大语言模型乐理推断阵列</h2>
          <span class="text-xs bg-amber-500/10 text-amber-400 px-4 py-1.5 rounded-full border border-amber-500/30 font-mono shadow-lg">{{ llmResult?.length || 0 }} LLM Inferences</span>
        </div>
        
        <div class="flex-1 w-full overflow-y-auto pr-3 pb-4 scrollbar-thin">
          <div class="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-4 pb-10">
            <!-- 单个 JSON 方块 -->
            <div v-for="(note, idx) in llmResult" :key="idx" class="bg-gradient-to-b from-black/80 to-amber-950/20 border border-white/5 hover:border-amber-500/40 rounded-xl p-4 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_0_20px_rgba(245,158,11,0.15)] shadow-lg relative group overflow-hidden">
               <div class="absolute inset-0 bg-amber-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
               
               <div class="flex justify-between items-start mb-3 relative z-10">
                 <div class="flex flex-col">
                   <div class="w-8 h-8 rounded-md bg-amber-500/10 text-amber-500 flex items-center justify-center text-sm font-bold border border-amber-500/20 shadow-inner">#{{ idx + 1 }}</div>
                 </div>
                 <!-- 纵向排版展示组合大字 -->
                 <div class="text-3xl font-guqin text-slate-200 mt-1 opacity-80 group-hover:opacity-100 transition-opacity" style="writing-mode: vertical-rl;">
                   {{ note.action || '' }}{{ note.string || '' }}{{ note.position || '' }}
                 </div>
               </div>
               
               <div class="space-y-1.5 font-mono text-xs bg-black/60 p-3 rounded-lg border border-white/5 relative z-10 mt-2">
                  <div class="flex justify-between items-center"><span class="text-slate-500">Pitch</span> <span class="bg-rose-500/20 border border-rose-500/20 text-rose-300 px-1.5 py-0.5 rounded">{{ note.pitch || '-' }}</span></div>
                  <div class="flex justify-between items-center"><span class="text-slate-500">Octave</span> <span class="text-slate-300">{{ note.octave || '-' }}</span></div>
                  <div class="flex justify-between items-center"><span class="text-slate-500">Duration</span> <span class="bg-emerald-500/20 border border-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">{{ note.duration ? '1/'+note.duration : '-' }}</span></div>
               </div>
            </div>
          </div>
        </div>
      </div>

      <!--==================== 视图 4：XML Render 终极乐谱 ====================-->
      <div v-show="activeTab === 'xml'" class="w-full max-w-[1600px] bg-[#fdfdfd] text-slate-800 rounded-2xl animate-fade-in border border-white/20 shadow-2xl flex flex-col overflow-hidden h-[80vh]">
        <div class="bg-gradient-to-r from-slate-100 to-white border-b border-slate-200 px-6 py-4 flex justify-between items-center shadow-sm z-10 shrink-0">
          <h2 class="text-xl font-bold text-slate-700 tracking-wider">打谱渲染容器 (Web ScoreModel View)</h2>
          <button class="text-xs bg-slate-800 text-white px-5 py-2.5 rounded-lg hover:bg-slate-700 transition shadow-md flex items-center gap-2 font-mono">
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
            Export MusicXML
          </button>
        </div>
        <div class="flex-1 overflow-y-auto w-full relative">
           <ScoreModelRenderer v-if="scoreModel && scoreModel.noteCount > 0" :score-data="scoreModel" class="min-h-full" />
           <div v-else class="absolute inset-0 flex flex-col items-center justify-center text-slate-300 bg-slate-50">
             <div class="w-16 h-16 mb-6 animate-spin rounded-full border-[5px] border-slate-200 border-t-primary"></div>
             <p class="font-mono text-sm tracking-widest text-slate-400">WAITING FOR XML GENERATION...</p>
           </div>
        </div>
      </div>

      <!--==================== 全局异常飘窗 ====================-->
      <div v-if="errorMessage" class="absolute bottom-10 left-1/2 -translate-x-1/2 z-50 glass-panel border border-rose-500/50 shadow-[0_0_30px_rgba(244,63,94,0.3)] bg-rose-950/80 px-6 py-4 rounded-xl flex items-center gap-3 animate-bounce">
         <svg class="w-6 h-6 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
         <span class="text-rose-100 font-medium tracking-wide">{{ errorMessage }}</span>
      </div>

    </main>
  </div>
</template>

<style>
/* 保证组件内部作用域下的 Tailwind 不冲突 */
body {
  margin: 0;
  overflow-x: hidden;
  background-color: #0f172a; /* Slate 900 as base fallback */
}

/* 自定义优美的暗黑玻璃横向/纵向滚动条 */
.scrollbar-thin::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
.scrollbar-thin::-webkit-scrollbar-track {
  background: transparent;
}
.scrollbar-thin::-webkit-scrollbar-thumb {
  background-color: rgba(255,255,255,0.1);
  border-radius: 20px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
.scrollbar-thin::-webkit-scrollbar-thumb:hover {
  background-color: rgba(255,255,255,0.25);
}
</style>
