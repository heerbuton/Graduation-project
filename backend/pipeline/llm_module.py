import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests

LOGGER = logging.getLogger(__name__)

BAILIAN_API_KEY = "sk-48f0485ef8f246bf9316e2d87420726e"
BAILIAN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
BAILIAN_MODEL = "qwen3.5-plus"
LLM_TIMEOUT_SECONDS = 180.0
LLM_TEMPERATURE = 0.0
LLM_ENABLE_THINKING = False

VALID_PITCHES = {"1", "2", "3", "4", "5", "6", "7"}
VALID_OCTAVES = {"3", "4", "5"}
VALID_DURATIONS = {"2", "4", "8", "16"}


def _sort_group_key(group_id: str) -> tuple:
    match = re.search(r"(\d+)", str(group_id))
    if not match:
        return (10**9, str(group_id))
    return (int(match.group(1)), str(group_id))


def _is_marker_payload(payload: Dict[str, Any]) -> bool:
    marker_type = str(payload.get("marker_type", "")).strip().lower()
    if marker_type in {"start", "end"}:
        return True
    if payload.get("is_marker") or payload.get("is_section_start") or payload.get("is_section_end"):
        return True
    return False


def _normalize_group_payload(group_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if _is_marker_payload(payload):
        return None

    right_fingering = str(payload.get("right_fingering") or "").strip()
    left_fingering = str(payload.get("left_fingering") or "").strip()
    left_finger = str(payload.get("left_finger") or payload.get("finger") or "").strip()
    action = str(
        right_fingering
        or left_fingering
        or payload.get("fingering")
        or payload.get("action")
        or ""
    ).strip()
    finger = left_finger
    position = str(payload.get("hui") or payload.get("position") or "").strip()
    string = str(payload.get("xian") or payload.get("string") or payload.get("xian_digit") or "").strip()

    return {
        "group_id": str(group_id),
        "action": action,
        "finger": finger,
        "position": position,
        "string": string,
        "right_fingering": right_fingering,
        "left_fingering": left_fingering,
        "left_finger": left_finger,
        "hui": str(payload.get("hui") or "").strip(),
        "xian": str(payload.get("xian") or "").strip(),
    }


def _normalize_topology(topology_json: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    if isinstance(topology_json, dict):
        group_items = sorted(topology_json.items(), key=lambda item: _sort_group_key(item[0]))
        for group_id, payload in group_items:
            payload = payload if isinstance(payload, dict) else {}
            normalized_payload = _normalize_group_payload(str(group_id), payload)
            if normalized_payload is None:
                continue
            normalized.append(normalized_payload)
        return normalized

    if isinstance(topology_json, list):
        for idx, payload in enumerate(topology_json, start=1):
            payload = payload if isinstance(payload, dict) else {}
            group_id = str(payload.get("group_id") or f"group_{idx}")
            normalized_payload = _normalize_group_payload(group_id, payload)
            if normalized_payload is None:
                continue
            normalized.append(normalized_payload)
    return normalized


def _compact_groups_for_prompt(groups: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    compact: List[Dict[str, str]] = []
    for item in groups:
        compact.append(
            {
                "group_id": str(item.get("group_id", "")).strip(),
                "right_fingering": str(item.get("right_fingering", "")).strip(),
                "left_fingering": str(item.get("left_fingering", "")).strip(),
                "left_finger": str(item.get("left_finger", "")).strip(),
                "hui": str(item.get("hui", "")).strip(),
                "xian": str(item.get("xian", "")).strip(),
                # 兼容字段，便于模型直接抄写输出
                "action": str(item.get("action", "")).strip(),
                "finger": str(item.get("finger", "")).strip(),
                "position": str(item.get("position", "")).strip(),
                "string": str(item.get("string", "")).strip(),
            }
        )
    return compact


def _build_messages(groups: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    compact_groups = _compact_groups_for_prompt(groups)
    system_prompt = (
        "你是古琴减字谱正调打谱专家。必须基于完整序列上下文进行整体打谱，禁止逐 group 独立机械判断。"
        "输出必须是 JSON 对象且仅包含 notes 数组。"
        "notes 每个对象必须包含字段："
        "group_id,pitch,octave,duration,action,string,position,finger,new_measure。"
        "必须严格按输入 group 顺序一一对应。"
        "硬约束：pitch 只能是字符串 1-7；"
        "octave 只能为 3/4/5；duration 只能为 2/4/8/16；new_measure 必须为布尔值。"
        "action/string/position/finger 必须原样复制输入对应 group。"
        "古琴打谱规则："
        "先全局判断调式与句法，再逐组给出音高和时值；"
        "相同或高度相似的指法-弦位组合在相近语境下应保持一致；"
        "xian/hui 及左右手指法是音高判断核心依据，缺失字段时可参考前后组延续；"
        "时值需遵循乐句连贯与节奏平衡，避免无依据跳变；"
        "new_measure 必须依据整段节奏结构标注，不可随意插入。"
        "若无法确定：pitch 用 1，octave 用 4，duration 用 4。"
        "不要输出解释文字、注释或 Markdown。"
    )
    user_prompt = (
        "以下是第二模块输出的完整减字序列，请基于完整上下文打谱并输出严格 JSON：\n"
        f"{json.dumps(compact_groups, ensure_ascii=False, separators=(',', ':'))}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_response_format() -> Dict[str, Any]:
    return {"type": "json_object"}


def _post_chat_completion(payload: Dict[str, Any], headers: Dict[str, str], api_url: str, timeout: float) -> Dict[str, Any]:
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise requests.HTTPError(
            f"LLM 请求失败: HTTP {response.status_code} - {response.text}",
            response=response,
        )
    return response.json()


def _call_llm_with_messages(messages: List[Dict[str, str]]) -> str:
    headers = {
        "Authorization": f"Bearer {BAILIAN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": BAILIAN_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "enable_thinking": LLM_ENABLE_THINKING,
        "response_format": _build_response_format(),
    }

    response_json = _post_chat_completion(payload, headers, BAILIAN_API_URL, LLM_TIMEOUT_SECONDS)
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("LLM 返回中缺少 choices 字段。")

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text_chunks = []
        for block in content:
            if isinstance(block, dict):
                text_chunks.append(str(block.get("text", "")))
            else:
                text_chunks.append(str(block))
        return "\n".join(text_chunks).strip()
    return str(content).strip()


def _infer_full_context_notes(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    llm_text = _call_llm_with_messages(_build_messages(groups))
    parsed_payload = _json_from_text(llm_text)
    if parsed_payload is None:
        raise ValueError("LLM 返回不是有效 JSON。")

    parsed_notes = _extract_note_list(parsed_payload)
    if not parsed_notes:
        raise ValueError("LLM 返回中没有可解析的 notes。")
    try:
        return _merge_notes_strict(parsed_notes, groups)
    except ValueError as exc:
        repaired_notes = _repair_invalid_notes(groups, parsed_notes, str(exc))
        return _merge_notes_strict(repaired_notes, groups)


def _build_repair_messages(
    groups: List[Dict[str, Any]],
    draft_notes: List[Dict[str, Any]],
    failure_reason: str,
) -> List[Dict[str, str]]:
    compact_groups = _compact_groups_for_prompt(groups)
    system_prompt = (
        "你是古琴打谱结果修复器。"
        "你将收到输入 groups 和一个 draft_notes。"
        "任务：修复 draft_notes 中不合法字段，输出合法 JSON 对象 {\"notes\": [...]}。"
        "必须保持 notes 顺序与 groups 完全一致，且 group_id 一一对应。"
        "pitch 只能为字符串 1-7；octave 只能为 3/4/5；duration 只能为 2/4/8/16；new_measure 必须为布尔。"
        "action/string/position/finger 必须与对应输入 group 保持一致，不得翻译改写。"
        "若无法修复：pitch=1，octave=4，duration=4。"
        "不要输出解释、注释或 Markdown。"
    )
    user_prompt = (
        "校验失败原因：\n"
        f"{failure_reason}\n"
        "输入 groups：\n"
        f"{json.dumps(compact_groups, ensure_ascii=False, separators=(',', ':'))}\n"
        "待修复 draft_notes：\n"
        f"{json.dumps(draft_notes, ensure_ascii=False, separators=(',', ':'))}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _repair_invalid_notes(
    groups: List[Dict[str, Any]],
    draft_notes: List[Dict[str, Any]],
    failure_reason: str,
) -> List[Dict[str, Any]]:
    llm_text = _call_llm_with_messages(_build_repair_messages(groups, draft_notes, failure_reason))
    parsed_payload = _json_from_text(llm_text)
    if parsed_payload is None:
        raise ValueError("修复阶段返回不是有效 JSON。")

    repaired_notes = _extract_note_list(parsed_payload)
    if not repaired_notes:
        raise ValueError("修复阶段未返回可解析 notes。")
    return repaired_notes


def _json_from_text(text: str) -> Any:
    text = text.strip()
    if not text:
        return None

    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend(fenced)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def _extract_note_list(parsed_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed_payload, list):
        return [item for item in parsed_payload if isinstance(item, dict)]

    if isinstance(parsed_payload, dict):
        notes = parsed_payload.get("notes")
        if isinstance(notes, list):
            return [item for item in notes if isinstance(item, dict)]
    return []


def _normalize_enum_strict(value: Any, field_name: str, allowed_values: set) -> str:
    text = str(value).strip()
    if text not in allowed_values:
        raise ValueError(f"字段 {field_name} 非法: {value!r}")
    return text


def _normalize_bool_strict(value: Any) -> bool:
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
    raise ValueError(f"字段 new_measure 非法: {value!r}")


def _build_base_notes(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "group_id": str(group.get("group_id", "")).strip(),
            "action": str(group.get("action", "")).strip(),
            "string": str(group.get("string", "")).strip(),
            "position": str(group.get("position", "")).strip(),
            "finger": str(group.get("finger", "")).strip(),
        }
        for group in groups
    ]


def _normalize_note(raw_note: Dict[str, Any], base_note: Dict[str, Any]) -> Dict[str, Any]:
    pitch = _normalize_enum_strict(raw_note.get("pitch"), "pitch", VALID_PITCHES)
    octave = _normalize_enum_strict(raw_note.get("octave"), "octave", VALID_OCTAVES)
    duration = _normalize_enum_strict(raw_note.get("duration"), "duration", VALID_DURATIONS)
    new_measure = _normalize_bool_strict(raw_note.get("new_measure", False))

    note = {
        "group_id": base_note["group_id"],
        "pitch": pitch,
        "octave": octave,
        "duration": duration,
        "action": str(base_note.get("action", "") or raw_note.get("action", "")).strip(),
        "string": str(base_note.get("string", "") or raw_note.get("string", "")).strip(),
        "position": str(base_note.get("position", "") or raw_note.get("position", "")).strip(),
        "finger": str(base_note.get("finger", "") or raw_note.get("finger", "")).strip(),
    }
    if new_measure:
        note["new_measure"] = True
    return note


def _merge_notes_strict(parsed_notes: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_notes = _build_base_notes(groups)
    group_order = [item["group_id"] for item in base_notes]
    base_by_group = {item["group_id"]: item for item in base_notes}

    if not group_order:
        return []

    has_any_group_id = any(str(item.get("group_id", "")).strip() for item in parsed_notes)
    if not has_any_group_id:
        if len(parsed_notes) != len(base_notes):
            raise ValueError(
                f"LLM 返回音符数量与 group 数量不一致: notes={len(parsed_notes)}, groups={len(base_notes)}"
            )
        return [_normalize_note(raw_note, base_note) for raw_note, base_note in zip(parsed_notes, base_notes)]

    parsed_by_group: Dict[str, Dict[str, Any]] = {}
    for raw_note in parsed_notes:
        group_id = str(raw_note.get("group_id", "")).strip()
        if not group_id:
            raise ValueError("LLM 返回中存在缺失 group_id 的音符。")
        if group_id not in base_by_group:
            raise ValueError(f"LLM 返回了未知 group_id: {group_id}")
        if group_id in parsed_by_group:
            raise ValueError(f"LLM 返回中 group_id 重复: {group_id}")
        parsed_by_group[group_id] = raw_note

    missing_groups = [group_id for group_id in group_order if group_id not in parsed_by_group]
    if missing_groups:
        raise ValueError(f"LLM 返回缺少 group_id: {', '.join(missing_groups)}")

    return [_normalize_note(parsed_by_group[group_id], base_by_group[group_id]) for group_id in group_order]


def _public_note(note: Dict[str, Any]) -> Dict[str, Any]:
    output = {
        "pitch": note.get("pitch", "1"),
        "octave": note.get("octave", "4"),
        "duration": note.get("duration", "4"),
        "action": note.get("action", ""),
        "string": note.get("string", ""),
        "position": note.get("position", ""),
        "finger": note.get("finger", ""),
    }
    if note.get("new_measure"):
        output["new_measure"] = True
    return output


def infer_pitch_duration(topology_json: Any) -> List[Dict[str, Any]]:
    """
    模块 C：大模型打谱模块。
    基于拓扑数据调用 LLM 推断 pitch / octave / duration，并返回结构化结果。
    失败策略：严格失败，不做规则兜底。
    """
    groups = _normalize_topology(topology_json)
    if not groups:
        return []

    merged_notes = _infer_full_context_notes(groups)
    LOGGER.info("LLM 打谱成功: groups=%d, notes=%d", len(groups), len(merged_notes))
    return [_public_note(item) for item in merged_notes]

