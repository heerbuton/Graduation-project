import json
import logging
import os
import re
from typing import Any, Dict, List

import requests

LOGGER = logging.getLogger(__name__)
VALID_DURATIONS = {"2", "4", "8", "16"}
VALID_OCTAVES = {"3", "4", "5"}
STRING_TO_SCALE = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7"}
NUMERAL_TO_SCALE = {
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "1",
    "九": "2",
}


def _sort_group_key(group_id: str) -> tuple:
    match = re.search(r"(\d+)", str(group_id))
    if not match:
        return (10**9, str(group_id))
    return (int(match.group(1)), str(group_id))


def _normalize_topology(topology_json: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    if isinstance(topology_json, dict):
        group_items = sorted(topology_json.items(), key=lambda item: _sort_group_key(item[0]))
        for group_id, payload in group_items:
            payload = payload if isinstance(payload, dict) else {}
            normalized.append(
                {
                    "group_id": str(group_id),
                    "action": str(payload.get("fingering") or payload.get("action") or "").strip(),
                    "finger": str(payload.get("finger") or "").strip(),
                    "position": str(payload.get("position") or "").strip(),
                    "string": str(payload.get("string") or "").strip(),
                    "components": payload.get("components", []),
                }
            )
        return normalized

    if isinstance(topology_json, list):
        for idx, payload in enumerate(topology_json, start=1):
            payload = payload if isinstance(payload, dict) else {}
            normalized.append(
                {
                    "group_id": f"group_{idx}",
                    "action": str(payload.get("fingering") or payload.get("action") or "").strip(),
                    "finger": str(payload.get("finger") or "").strip(),
                    "position": str(payload.get("position") or "").strip(),
                    "string": str(payload.get("string") or "").strip(),
                    "components": payload.get("components", []),
                }
            )
    return normalized


def _build_messages(groups: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system_prompt = (
        "你是一名古琴减字谱打谱专家。"
        "请根据输入的拓扑信息，为每个 group 推断 pitch/octave/duration。"
        "你必须严格返回 JSON，不要返回任何解释文字。"
        "返回格式必须为 {\"notes\": [...]}，每个元素字段为："
        "group_id, pitch, octave, duration, action, string, position, finger, new_measure。"
        "其中 pitch 只能是 1-7，octave 只能是 3/4/5，duration 只能是 2/4/8/16，new_measure 为布尔值。"
    )
    user_prompt = (
        "下面是识别后的减字谱拓扑 JSON，请推断每组音高及时值：\n"
        f"{json.dumps(groups, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _post_chat_completion(payload: Dict[str, Any], headers: Dict[str, str], api_url: str, timeout: float) -> Dict[str, Any]:
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise requests.HTTPError(
            f"LLM 请求失败: HTTP {response.status_code} - {response.text}",
            response=response,
        )
    return response.json()


def _call_llm(groups: List[Dict[str, Any]]) -> str:
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未设置 LLM_API_KEY 或 OPENAI_API_KEY。")

    api_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions").strip()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": _build_messages(groups),
        "temperature": temperature,
    }

    if os.getenv("LLM_DISABLE_JSON_RESPONSE_FORMAT", "0").strip() != "1":
        payload["response_format"] = {"type": "json_object"}

    try:
        response_json = _post_chat_completion(payload, headers, api_url, timeout)
    except requests.HTTPError as exc:
        should_retry = "response_format" in payload and exc.response is not None and exc.response.status_code in {400, 422}
        if not should_retry:
            raise

        payload.pop("response_format", None)
        response_json = _post_chat_completion(payload, headers, api_url, timeout)

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
        for key in ("notes", "result", "data", "items"):
            value = parsed_payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if any(k in parsed_payload for k in ("pitch", "duration", "action", "group_id")):
            return [parsed_payload]
    return []


def _normalize_pitch(value: Any, default: str = "1") -> str:
    text = str(value).strip()
    if text in {"1", "2", "3", "4", "5", "6", "7"}:
        return text
    match = re.search(r"[1-7]", text)
    if match:
        return match.group(0)
    return default


def _normalize_octave(value: Any, default: str = "4") -> str:
    text = str(value).strip()
    if text in VALID_OCTAVES:
        return text
    match = re.search(r"\d+", text)
    if not match:
        return default

    octave = int(match.group(0))
    if octave <= 3:
        return "3"
    if octave >= 5:
        return "5"
    return "4"


def _normalize_duration(value: Any, default: str = "4") -> str:
    text = str(value).strip()
    if text in VALID_DURATIONS:
        return text
    match = re.search(r"(16|8|4|2)", text)
    if match and match.group(1) in VALID_DURATIONS:
        return match.group(1)
    return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _guess_pitch(group: Dict[str, Any], fallback_index: int) -> str:
    string_hint = group.get("string", "")
    if string_hint in STRING_TO_SCALE:
        return STRING_TO_SCALE[string_hint]

    position = str(group.get("position", "")).strip()
    if position:
        first_char = position[0]
        if first_char in NUMERAL_TO_SCALE:
            return NUMERAL_TO_SCALE[first_char]

    return str((fallback_index % 7) + 1)


def _guess_duration(group: Dict[str, Any]) -> str:
    action = str(group.get("action", "")).strip()
    if action in {"历", "轮"}:
        return "8"
    if action in {"撮"}:
        return "2"
    return "4"


def _guess_octave(group: Dict[str, Any]) -> str:
    position = str(group.get("position", "")).strip()
    if "徽外" in position:
        return "3"
    if any(token in position for token in ("高", "上")):
        return "5"
    return "4"


def _heuristic_inference(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    notes: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups):
        note = {
            "group_id": group["group_id"],
            "pitch": _guess_pitch(group, idx),
            "octave": _guess_octave(group),
            "duration": _guess_duration(group),
            "action": group.get("action", ""),
            "string": group.get("string", ""),
            "position": group.get("position", ""),
            "finger": group.get("finger", ""),
        }
        if idx > 0 and idx % 8 == 0:
            note["new_measure"] = True
        notes.append(note)
    return notes


def _normalize_note(note: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "group_id": fallback.get("group_id", ""),
        "pitch": _normalize_pitch(note.get("pitch", fallback.get("pitch", "1")), fallback.get("pitch", "1")),
        "octave": _normalize_octave(note.get("octave", fallback.get("octave", "4")), fallback.get("octave", "4")),
        "duration": _normalize_duration(note.get("duration", fallback.get("duration", "4")), fallback.get("duration", "4")),
        "action": str(note.get("action", fallback.get("action", ""))).strip(),
        "string": str(note.get("string", fallback.get("string", ""))).strip(),
        "position": str(note.get("position", fallback.get("position", ""))).strip(),
        "finger": str(note.get("finger", fallback.get("finger", ""))).strip(),
    }

    if _to_bool(note.get("new_measure", fallback.get("new_measure", False))):
        merged["new_measure"] = True
    return merged


def _merge_notes(parsed_notes: List[Dict[str, Any]], fallback_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not fallback_notes:
        return []

    parsed_by_group: Dict[str, Dict[str, Any]] = {}
    parsed_without_group: List[Dict[str, Any]] = []
    for raw_note in parsed_notes:
        group_id = str(raw_note.get("group_id", "")).strip()
        if group_id:
            parsed_by_group[group_id] = raw_note
        else:
            parsed_without_group.append(raw_note)

    merged: List[Dict[str, Any]] = []
    if parsed_by_group:
        for fallback in fallback_notes:
            raw_note = parsed_by_group.get(fallback["group_id"], {})
            merged.append(_normalize_note(raw_note, fallback))
        return merged

    for idx, fallback in enumerate(fallback_notes):
        raw_note = parsed_without_group[idx] if idx < len(parsed_without_group) else {}
        merged.append(_normalize_note(raw_note, fallback))
    return merged


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
    """
    groups = _normalize_topology(topology_json)
    if not groups:
        return []

    fallback_notes = _heuristic_inference(groups)

    try:
        llm_text = _call_llm(groups)
        parsed_payload = _json_from_text(llm_text)
        parsed_notes = _extract_note_list(parsed_payload)
        if not parsed_notes:
            raise ValueError("LLM 返回中没有可解析的音符列表。")
        merged_notes = _merge_notes(parsed_notes, fallback_notes)
        return [_public_note(item) for item in merged_notes]
    except Exception as exc:
        LOGGER.warning("LLM 推理失败，改用规则兜底: %s", exc)
        return [_public_note(item) for item in fallback_notes]
