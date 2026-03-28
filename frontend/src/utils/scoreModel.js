const VALID_PITCHES = new Set(["1", "2", "3", "4", "5", "6", "7"]);
const VALID_OCTAVES = new Set(["3", "4", "5"]);
const VALID_DURATIONS = new Set(["2", "4", "8", "16"]);
const BEATS_BY_DURATION = { "2": 2, "4": 1, "8": 0.5, "16": 0.25 };
const MEASURE_BEATS = 4;
const EPSILON = 1e-9;

const asText = (value) => {
  if (value === null || value === undefined) return "";
  return String(value).trim();
};

const asBool = (value) => {
  if (typeof value === "boolean") return value;
  if (typeof value === "number" && (value === 0 || value === 1)) return Boolean(value);
  if (typeof value === "string") {
    const lowered = value.trim().toLowerCase();
    if (lowered === "true" || lowered === "1") return true;
    if (lowered === "false" || lowered === "0") return false;
  }
  throw new Error(`new_measure 不是合法布尔值: ${JSON.stringify(value)}`);
};

const normalizeEnum = ({ value, fieldName, allowedValues, fallback, strict, noteIndex, issues }) => {
  const text = asText(value);
  if (allowedValues.has(text)) return text;

  if (strict) {
    throw new Error(`第 ${noteIndex} 个音符字段 ${fieldName} 非法: ${JSON.stringify(value)}`);
  }

  issues.push(`第 ${noteIndex} 个音符字段 ${fieldName} 非法(${JSON.stringify(value)})，已回退为 ${JSON.stringify(fallback)}`);
  return fallback;
};

const normalizeNewMeasure = ({ value, strict, noteIndex, issues }) => {
  if (value === null || value === undefined) return false;
  try {
    return asBool(value);
  } catch (error) {
    if (strict) throw new Error(`第 ${noteIndex} 个音符 ${error.message}`);
    issues.push(`第 ${noteIndex} 个音符 new_measure 非法(${JSON.stringify(value)})，已回退为 false`);
    return false;
  }
};

const durationToBeats = (duration) => BEATS_BY_DURATION[duration] ?? 1;

export const convertLlmResultToScoreModel = (llmResult, options = {}) => {
  const strict = Boolean(options.strict);
  const issues = [];

  if (llmResult === null || llmResult === undefined) {
    llmResult = [];
  }
  if (!Array.isArray(llmResult)) {
    throw new Error("llm_result 必须是数组(list)。");
  }

  const measures = [{ id: "m1", notes: [] }];
  let measureIndex = 1;
  let noteCount = 0;
  let currentMeasureBeats = 0;
  let pendingMeasureBreak = false;

  llmResult.forEach((rawNote, idx) => {
    const noteIndex = idx + 1;
    if (!rawNote || typeof rawNote !== "object" || Array.isArray(rawNote)) {
      if (strict) {
        throw new Error(`第 ${noteIndex} 个元素不是对象: ${JSON.stringify(rawNote)}`);
      }
      issues.push(`第 ${noteIndex} 个元素不是对象，已跳过`);
      return;
    }

    const newMeasure = normalizeNewMeasure({
      value: rawNote.new_measure,
      strict,
      noteIndex,
      issues,
    });
    if (newMeasure && measures[measures.length - 1].notes.length > 0) {
      measureIndex += 1;
      measures.push({ id: `m${measureIndex}`, notes: [] });
    }

    noteCount += 1;
    const noteId = `m${measureIndex}_n${measures[measures.length - 1].notes.length + 1}`;

    const pitch = normalizeEnum({
      value: rawNote.pitch,
      fieldName: "pitch",
      allowedValues: VALID_PITCHES,
      fallback: "1",
      strict,
      noteIndex,
      issues,
    });
    const octave = normalizeEnum({
      value: rawNote.octave,
      fieldName: "octave",
      allowedValues: VALID_OCTAVES,
      fallback: "4",
      strict,
      noteIndex,
      issues,
    });
    const duration = normalizeEnum({
      value: rawNote.duration,
      fieldName: "duration",
      allowedValues: VALID_DURATIONS,
      fallback: "4",
      strict,
      noteIndex,
      issues,
    });
    const noteBeats = durationToBeats(duration);

    if (measures[measures.length - 1].notes.length > 0 && newMeasure) {
      measureIndex += 1;
      measures.push({ id: `m${measureIndex}`, notes: [] });
      currentMeasureBeats = 0;
      pendingMeasureBreak = false;
    } else if (pendingMeasureBreak && measures[measures.length - 1].notes.length > 0) {
      measureIndex += 1;
      measures.push({ id: `m${measureIndex}`, notes: [] });
      currentMeasureBeats = 0;
      pendingMeasureBreak = false;
    } else if (
      measures[measures.length - 1].notes.length > 0 &&
      currentMeasureBeats + noteBeats > MEASURE_BEATS + EPSILON
    ) {
      measureIndex += 1;
      measures.push({ id: `m${measureIndex}`, notes: [] });
      currentMeasureBeats = 0;
    }

    measures[measures.length - 1].notes.push({
      id: noteId,
      pitch,
      octave,
      duration,
      isDash: false,
      guqin: {
        action: asText(rawNote.action),
        stringOrder: asText(rawNote.stringOrder || rawNote.string_order || rawNote.string),
        position: asText(rawNote.position),
        finger: asText(rawNote.finger),
      },
    });

    currentMeasureBeats += noteBeats;
    if (currentMeasureBeats >= MEASURE_BEATS - EPSILON) {
      pendingMeasureBreak = true;
      currentMeasureBeats = 0;
    }
  });

  return {
    version: "1.0",
    measureCount: measures.length,
    noteCount,
    issues,
    measures,
  };
};
