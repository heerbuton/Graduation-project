from typing import Any, Dict, List


VALID_PITCHES = {"1", "2", "3", "4", "5", "6", "7"}
VALID_OCTAVES = {"3", "4", "5"}
VALID_DURATIONS = {"2", "4", "8", "16"}
BEATS_BY_DURATION = {"2": 2.0, "4": 1.0, "8": 0.5, "16": 0.25}
MEASURE_BEATS = 4.0
EPSILON = 1e-9


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    raise ValueError(f"new_measure 不是合法布尔值: {value!r}")


def _normalize_enum(
    value: Any,
    field_name: str,
    allowed_values: set,
    fallback: str,
    strict: bool,
    note_index: int,
    issues: List[str],
) -> str:
    text = _as_text(value)
    if text in allowed_values:
        return text

    if strict:
        raise ValueError(f"第 {note_index} 个音符字段 {field_name} 非法: {value!r}")

    issues.append(
        f"第 {note_index} 个音符字段 {field_name} 非法({value!r})，已回退为 {fallback!r}"
    )
    return fallback


def _normalize_new_measure(value: Any, strict: bool, note_index: int, issues: List[str]) -> bool:
    if value is None:
        return False
    try:
        return _as_bool(value)
    except ValueError as exc:
        if strict:
            raise ValueError(f"第 {note_index} 个音符 {exc}") from exc
        issues.append(f"第 {note_index} 个音符 new_measure 非法({value!r})，已回退为 False")
        return False


def _duration_to_beats(duration: str) -> float:
    return BEATS_BY_DURATION.get(duration, 1.0)


def transform_llm_result_to_score_model(llm_result: Any, strict: bool = False) -> Dict[str, Any]:
    """
    将 LLM 输出的线性 JSON 音符序列转换为前端可直接渲染的 ScoreModel。
    """
    if llm_result is None:
        llm_result = []
    if not isinstance(llm_result, list):
        raise ValueError("llm_result 必须是数组(list)。")

    issues: List[str] = []
    measures: List[Dict[str, Any]] = [{"id": "m1", "notes": []}]
    measure_index = 1
    note_count = 0
    current_measure_beats = 0.0
    pending_measure_break = False

    for idx, raw_note in enumerate(llm_result, start=1):
        if not isinstance(raw_note, dict):
            if strict:
                raise ValueError(f"第 {idx} 个元素不是对象: {raw_note!r}")
            issues.append(f"第 {idx} 个元素不是对象，已跳过")
            continue

        new_measure = _normalize_new_measure(raw_note.get("new_measure"), strict, idx, issues)
        if new_measure and measures[-1]["notes"]:
            measure_index += 1
            measures.append({"id": f"m{measure_index}", "notes": []})

        note_count += 1
        note_id = f"m{measure_index}_n{len(measures[-1]['notes']) + 1}"

        pitch = _normalize_enum(
            raw_note.get("pitch"),
            field_name="pitch",
            allowed_values=VALID_PITCHES,
            fallback="1",
            strict=strict,
            note_index=idx,
            issues=issues,
        )
        octave = _normalize_enum(
            raw_note.get("octave"),
            field_name="octave",
            allowed_values=VALID_OCTAVES,
            fallback="4",
            strict=strict,
            note_index=idx,
            issues=issues,
        )
        duration = _normalize_enum(
            raw_note.get("duration"),
            field_name="duration",
            allowed_values=VALID_DURATIONS,
            fallback="4",
            strict=strict,
            note_index=idx,
            issues=issues,
        )

        note_beats = _duration_to_beats(duration)

        # 混合切分策略：显式 new_measure + 4/4 时值推断并行
        if measures[-1]["notes"] and new_measure:
            measure_index += 1
            measures.append({"id": f"m{measure_index}", "notes": []})
            current_measure_beats = 0.0
            pending_measure_break = False
        elif pending_measure_break and measures[-1]["notes"]:
            measure_index += 1
            measures.append({"id": f"m{measure_index}", "notes": []})
            current_measure_beats = 0.0
            pending_measure_break = False
        elif (
            measures[-1]["notes"]
            and current_measure_beats + note_beats > MEASURE_BEATS + EPSILON
        ):
            measure_index += 1
            measures.append({"id": f"m{measure_index}", "notes": []})
            current_measure_beats = 0.0

        guqin = {
            "action": _as_text(raw_note.get("action")),
            "stringOrder": _as_text(
                raw_note.get("stringOrder")
                or raw_note.get("string_order")
                or raw_note.get("string")
            ),
            "position": _as_text(raw_note.get("position")),
            "finger": _as_text(raw_note.get("finger")),
        }

        measures[-1]["notes"].append(
            {
                "id": note_id,
                "pitch": pitch,
                "octave": octave,
                "duration": duration,
                "isDash": False,
                "guqin": guqin,
            }
        )

        current_measure_beats += note_beats
        if current_measure_beats >= MEASURE_BEATS - EPSILON:
            pending_measure_break = True
            current_measure_beats = 0.0

    # 兜底保证至少有一个小节，前端渲染可稳定处理
    if not measures:
        measures = [{"id": "m1", "notes": []}]

    return {
        "version": "1.0",
        "measureCount": len(measures),
        "noteCount": note_count,
        "issues": issues,
        "measures": measures,
    }
