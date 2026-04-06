<script setup>
import { computed } from "vue";

const props = defineProps({
  scoreData: {
    type: Object,
    required: true,
  },
});

const displayMeasures = computed(() => {
  const measures = Array.isArray(props.scoreData?.measures) ? props.scoreData.measures : [];
  return measures.map((measure, mIndex) => {
    const notes = Array.isArray(measure?.notes) ? measure.notes : [];
    const expandedNotes = [];

    notes.forEach((note, noteIndex) => {
      expandedNotes.push(note);
      if (note?.duration === "2") {
        expandedNotes.push({
          id: `${note.id || `m${mIndex + 1}_n${noteIndex + 1}`}_dash`,
          pitch: "-",
          octave: "4",
          duration: "4",
          isDash: true,
          guqin: { action: "", stringOrder: "", position: "", finger: "" },
        });
      }
    });

    return {
      id: measure?.id || `m${mIndex + 1}`,
      notes: expandedNotes,
    };
  });
});
</script>

<template>
  <div class="music-model-renderer p-4 bg-white rounded shadow-sm border border-emerald-200 min-h-[300px]">
    <h3 class="text-lg font-serif mb-4 text-emerald-900 border-b border-emerald-100 pb-2">
      翻译渲染区 (ScoreModel Render)
    </h3>

    <div class="guqin-score-container p-6 bg-[#fafaf8] rounded-xl shadow-inner min-h-[400px]" v-if="displayMeasures.length > 0">
      <div class="flex flex-wrap items-start">
        <div v-for="(measure, mIndex) in displayMeasures" :key="measure.id" class="flex items-start mb-8">
          <div
            v-for="(note, noteIndex) in measure.notes"
            :key="note.id || `${measure.id}_${noteIndex}`"
            class="flex flex-col items-center w-12 group cursor-default"
          >
            <!-- 简谱部分 -->
            <div
              class="jianpu-section w-full text-center h-12 flex flex-col justify-end items-center mb-3 relative"
            >
              <div v-if="note.octave === '5'" class="absolute -top-1 w-1.5 h-1.5 rounded-full bg-gray-900"></div>
              <div v-if="note.octave === '3'" class="absolute -bottom-1.5 w-1.5 h-1.5 rounded-full bg-gray-900"></div>

              <span
                class="text-2xl font-bold font-sans text-gray-900 leading-none group-hover:text-emerald-700 transition-colors"
              >
                {{ note.pitch }}
              </span>
              <div v-if="note.duration === '8'" class="w-2/3 h-[1.5px] bg-gray-900 mt-1"></div>
              <div v-if="note.duration === '16'" class="w-2/3 flex flex-col gap-[2px] mt-1">
                <div class="h-[1.5px] bg-gray-900"></div>
                <div class="h-[1.5px] bg-gray-900"></div>
              </div>
            </div>

            <!-- 减字谱部分 -->
            <div
              v-if="!note.isDash"
              class="lyrics-section flex flex-col w-full items-center text-[0.9rem] font-serif text-gray-800 tracking-widest gap-2 group-hover:bg-emerald-50/60 rounded py-1 transition-colors"
            >
              <div class="h-5 flex items-center justify-center font-medium">
                {{ note?.guqin?.action || " " }}
              </div>
              <div class="h-5 flex items-center justify-center">{{ note?.guqin?.stringOrder || " " }}</div>
              <div class="h-5 flex items-center justify-center">{{ note?.guqin?.position || " " }}</div>
              <div class="h-5 flex items-center justify-center">{{ note?.guqin?.finger || " " }}</div>
            </div>
            
            <div v-else class="lyrics-section flex flex-col w-full items-center h-[116px]"></div>
          </div>
          
          <!-- 小节线 (除了最后一小节外都显示) -->
          <div v-if="mIndex < displayMeasures.length - 1" class="w-px h-[160px] bg-gray-600 mx-4 mt-4"></div>
        </div>
      </div>
    </div>

    <div v-else class="text-gray-500 italic text-sm">未找到可渲染音符</div>
  </div>
</template>

<style scoped>
.guqin-score-container {
  background-color: #fafaf8;
  border-radius: 0.5rem;
}
</style>
