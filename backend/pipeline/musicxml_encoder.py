import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

try:
    from lxml import etree
except Exception:  # pragma: no cover - 可选依赖
    etree = None

STEP_MAP = {"1": "C", "2": "D", "3": "E", "4": "F", "5": "G", "6": "A", "7": "B"}
DURATION_MAP = {
    "2": ("half", "32"),
    "4": ("quarter", "16"),
    "8": ("eighth", "8"),
    "16": ("16th", "4"),
}
DOCTYPE_DECLARATION = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)
REQUIRED_LYRIC_NUMBERS = {"1", "2", "3", "4"}


def _new_measure(part: ET.Element, number: int, with_attributes: bool = False) -> ET.Element:
    measure = ET.SubElement(part, "measure", number=str(number))
    if not with_attributes:
        return measure

    attributes = ET.SubElement(measure, "attributes")
    divisions = ET.SubElement(attributes, "divisions")
    divisions.text = "16"

    key = ET.SubElement(attributes, "key")
    fifths = ET.SubElement(key, "fifths")
    fifths.text = "0"

    time = ET.SubElement(attributes, "time")
    beats = ET.SubElement(time, "beats")
    beats.text = "4"
    beat_type = ET.SubElement(time, "beat-type")
    beat_type.text = "4"

    clef = ET.SubElement(attributes, "clef")
    sign = ET.SubElement(clef, "sign")
    sign.text = "G"
    line = ET.SubElement(clef, "line")
    line.text = "2"
    return measure


def _normalize_pitch(note_data: dict) -> str:
    pitch_value = str(note_data.get("pitch", "1")).strip()
    pitch_digit = pitch_value[0] if pitch_value else "1"
    if pitch_digit not in STEP_MAP:
        return "C"
    return STEP_MAP[pitch_digit]


def _normalize_octave(note_data: dict) -> str:
    octave = str(note_data.get("octave", "4")).strip()
    if octave in {"3", "4", "5"}:
        return octave
    return "4"


def _normalize_duration(note_data: dict) -> tuple:
    duration_value = str(note_data.get("duration", "4")).strip()
    return DURATION_MAP.get(duration_value, DURATION_MAP["4"])


def _add_lyric(note_el: ET.Element, number: str, text: str) -> None:
    lyric = ET.SubElement(note_el, "lyric", number=number)
    syllabic = ET.SubElement(lyric, "syllabic")
    syllabic.text = "single"
    lyric_text = ET.SubElement(lyric, "text")
    lyric_text.text = text or ""


def _validate_lyric_layout(root: ET.Element) -> None:
    for note_index, note in enumerate(root.findall(".//note"), start=1):
        present_numbers = {lyric.get("number", "") for lyric in note.findall("lyric")}
        missing = REQUIRED_LYRIC_NUMBERS - present_numbers
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"第 {note_index} 个 note 缺少 lyric 编号: {missing_str}")


def _validate_with_lxml(xml_text: str) -> None:
    if os.getenv("MUSICXML_VALIDATE", "0").strip() != "1":
        return
    if etree is None:
        raise RuntimeError("MUSICXML_VALIDATE=1 但未安装 lxml。请先安装 lxml。")

    xml_doc = etree.fromstring(xml_text.encode("utf-8"))
    xml_tree = etree.ElementTree(xml_doc)

    xsd_path = os.getenv("MUSICXML_XSD_PATH", "").strip()
    if xsd_path:
        schema_doc = etree.parse(xsd_path)
        schema = etree.XMLSchema(schema_doc)
        if not schema.validate(xml_tree):
            error = schema.error_log.last_error
            message = error.message if error is not None else "未知 XSD 校验错误"
            raise ValueError(f"MusicXML XSD 校验失败: {message}")

    dtd_path = os.getenv("MUSICXML_DTD_PATH", "").strip()
    if dtd_path:
        dtd = etree.DTD(dtd_path)
        if not dtd.validate(xml_doc):
            errors = dtd.error_log.filter_from_errors()
            message = errors[0].message if errors else "未知 DTD 校验错误"
            raise ValueError(f"MusicXML DTD 校验失败: {message}")


def generate_musicxml(llm_results):
    """
    模块 D：编码层。
    将 LLM 结果编码为 MusicXML，并按要求填充 4 行减字谱 lyric 信息。
    """
    score_partwise = ET.Element("score-partwise", version="3.1")

    part_list = ET.SubElement(score_partwise, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    part_name = ET.SubElement(score_part, "part-name")
    part_name.text = "Guqin"

    part = ET.SubElement(score_partwise, "part", id="P1")
    measure_num = 1
    measure = _new_measure(part, measure_num, with_attributes=True)

    for note_data in llm_results or []:
        if not isinstance(note_data, dict):
            continue

        if note_data.get("new_measure"):
            measure_num += 1
            measure = _new_measure(part, measure_num, with_attributes=False)

        note_el = ET.SubElement(measure, "note")

        pitch_el = ET.SubElement(note_el, "pitch")
        step_el = ET.SubElement(pitch_el, "step")
        step_el.text = _normalize_pitch(note_data)
        octave_el = ET.SubElement(pitch_el, "octave")
        octave_el.text = _normalize_octave(note_data)

        note_type, duration_value = _normalize_duration(note_data)
        duration_el = ET.SubElement(note_el, "duration")
        duration_el.text = duration_value
        type_el = ET.SubElement(note_el, "type")
        type_el.text = note_type

        _add_lyric(note_el, "1", str(note_data.get("action", "")))
        _add_lyric(note_el, "2", str(note_data.get("string", "")))
        _add_lyric(note_el, "3", str(note_data.get("position", "")))
        _add_lyric(note_el, "4", str(note_data.get("finger", "")))

    _validate_lyric_layout(score_partwise)

    xml_str = ET.tostring(score_partwise, encoding="utf-8")
    parsed_xml = minidom.parseString(xml_str)
    pretty_xml = parsed_xml.toprettyxml(indent="  ")
    lines = [line for line in pretty_xml.split("\n") if line.strip() and not line.startswith("<?xml")]

    final_xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + DOCTYPE_DECLARATION + "\n" + "\n".join(lines)
    _validate_with_lxml(final_xml)
    return final_xml
