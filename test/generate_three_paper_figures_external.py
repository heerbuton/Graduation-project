# -*- coding: utf-8 -*-
"""
本轮三图专用导出脚本（仅导出三张图 + 对应素材）：
1) 语义重构思路图
2) 拓扑关系建模与顺序组织示意图
3) 代表性谱例译谱实测与终态输出展示图（2x2）

运行方式（项目根目录）：
  F:\anaconda\envs\pytorch\python.exe test/generate_three_paper_figures_external.py
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "test" / "paper_figures_external_2026-04-19"
FIG_DIR = OUT_DIR / "figures"
MAT_DIR = OUT_DIR / "materials"
SRC_DIR = OUT_DIR / "source_data"
TXT_DIR = OUT_DIR / "text_snippets"

RESULT_JSON = ROOT / "backend" / "static" / "uploads" / "testpicture-1.jpg_result.json"
RAW_IMAGE = ROOT / "backend" / "static" / "uploads" / "testpicture-1.jpg"
RAW_IMAGE_FALLBACK = ROOT / "test" / "testpicture-1.jpg"

SCORE_RENDER_SOURCE = (
    ROOT
    / "paper_assets"
    / "llm_notation_2026-04-03"
    / "05_conversion_evidence"
    / "scoremodel-after-measure-wrap-fix.png"
)


def _read_json(path: Path) -> Dict[str, Any]:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception:
            continue
    raise RuntimeError(f"无法读取 JSON: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def _save_png(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError(f"编码 PNG 失败: {path}")
    buf.tofile(str(path))


def _to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\msyhbd.ttc",
                r"C:\Windows\Fonts\simhei.ttf",
                r"C:\Windows\Fonts\arialbd.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\arial.ttf",
        ]
    )
    for item in candidates:
        try:
            return ImageFont.truetype(item, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: Tuple[int, int, int],
    width: int = 5,
    head: int = 18,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line([sx, sy, ex, ey], fill=color, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    left = (ex - head * math.cos(angle - math.pi / 6), ey - head * math.sin(angle - math.pi / 6))
    right = (ex - head * math.cos(angle + math.pi / 6), ey - head * math.sin(angle + math.pi / 6))
    draw.polygon([end, left, right], fill=color)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int = 4,
    dash: int = 16,
    gap: int = 10,
) -> None:
    x1, y1 = start
    x2, y2 = end
    total = math.dist(start, end)
    if total <= 0:
        return
    dx = (x2 - x1) / total
    dy = (y2 - y1) / total
    length = 0.0
    while length < total:
        seg = min(dash, total - length)
        sx = x1 + dx * length
        sy = y1 + dy * length
        ex = x1 + dx * (length + seg)
        ey = y1 + dy * (length + seg)
        draw.line([sx, sy, ex, ey], fill=color, width=width)
        length += dash + gap


def _draw_polyline_arrow(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
    width: int = 5,
    dashed: bool = False,
) -> None:
    if len(points) < 2:
        return
    for p1, p2 in zip(points[:-1], points[1:]):
        if dashed:
            _draw_dashed_line(draw, p1, p2, color=color, width=width)
        else:
            draw.line([p1, p2], fill=color, width=width)
    _draw_arrow(draw, points[-2], points[-1], color=color, width=width, head=16)


def _sort_groups(topology_json: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    def key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
        gid, payload = item
        try:
            return int(payload.get("sequence_index", 10**9)), gid
        except Exception:
            return 10**9, gid

    return sorted(topology_json.items(), key=key)


def _bbox(payload: Dict[str, Any]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = payload.get("group_bbox", [0, 0, 0, 0])
    return int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))


def _draw_detection_boxes(raw_img: np.ndarray, boxes: Sequence[Dict[str, Any]]) -> np.ndarray:
    out = raw_img.copy()
    for item in boxes:
        bb = item.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bb]
        cv2.rectangle(out, (x1, y1), (x2, y2), (55, 155, 65), 1)
    return out


def _draw_group_boxes(raw_img: np.ndarray, groups: Sequence[Tuple[str, Dict[str, Any]]]) -> np.ndarray:
    out = raw_img.copy()
    for _, payload in groups:
        x1, y1, x2, y2 = _bbox(payload)
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 95, 220), 2)
    return out


def _draw_ordered_boxes(raw_img: np.ndarray, groups: Sequence[Tuple[str, Dict[str, Any]]], max_labels: int = 55) -> np.ndarray:
    pil = _to_pil(raw_img).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    idx_font = _font(17, bold=True)
    order = 0
    for _, payload in groups:
        x1, y1, x2, y2 = _bbox(payload)
        draw.rectangle([x1, y1, x2, y2], outline=(240, 120, 28, 255), width=2)
        if not payload.get("is_marker"):
            order += 1
            if order <= max_labels:
                lx1 = x1 + 2
                ly1 = max(0, y1 - 24)
                draw.rectangle([lx1, ly1, lx1 + 36, ly1 + 22], fill=(240, 120, 28, 238))
                draw.text((lx1 + 7, ly1 + 2), str(order), fill=(255, 255, 255, 255), font=idx_font)
    return _to_bgr(pil.convert("RGB"))


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_obj: ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines() or [""]:
        if not raw:
            lines.append("")
            continue
        cur = ""
        for ch in raw:
            candidate = cur + ch
            w = draw.textbbox((0, 0), candidate, font=font_obj)[2]
            if w <= max_width or not cur:
                cur = candidate
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


def _draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font_obj: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
    max_width: int,
    line_gap: int = 8,
    max_lines: int | None = None,
) -> int:
    x, y = xy
    lines = _fit_text(draw, text, font_obj, max_width=max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            tail = lines[-1]
            while tail and draw.textbbox((0, 0), tail + "...", font=font_obj)[2] > max_width:
                tail = tail[:-1]
            lines[-1] = (tail or "") + "..."
    line_h = draw.textbbox((0, 0), "测", font=font_obj)[3]
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * (line_h + line_gap)), line, fill=fill, font=font_obj)
    return y + len(lines) * (line_h + line_gap)


def _crop_white_margin_pil(
    img: Image.Image,
    threshold: int = 245,
    pad: int = 18,
) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray)
    mask = arr < threshold
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(img.width, int(xs.max()) + pad + 1)
    y2 = min(img.height, int(ys.max()) + pad + 1)
    return img.crop((x1, y1, x2, y2))


def _find_dense_region(
    boxes: Sequence[Dict[str, Any]],
    img_w: int,
    img_h: int,
    win_w: int = 420,
    win_h: int = 540,
    stride: int = 28,
) -> Tuple[int, int, int, int]:
    centers: List[Tuple[float, float]] = []
    for item in boxes:
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    best = (-1, 0, 0)
    max_x = max(1, img_w - win_w + 1)
    max_y = max(1, img_h - win_h + 1)
    for y in range(0, max_y, stride):
        for x in range(0, max_x, stride):
            x2, y2 = x + win_w, y + win_h
            count = 0
            for cx, cy in centers:
                if x <= cx <= x2 and y <= cy <= y2:
                    count += 1
            if count > best[0]:
                best = (count, x, y)

    _, x, y = best
    return x, y, min(img_w, x + win_w), min(img_h, y + win_h)


def _crop_pil_by_bbox(img: Image.Image, bbox: Tuple[int, int, int, int], pad: int = 12) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.width, x2 + pad)
    y2 = min(img.height, y2 + pad)
    return img.crop((x1, y1, x2, y2))


def _safe_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value).strip()


def _pick_local_group(groups: Sequence[Tuple[str, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    best: List[Tuple[int, str, Dict[str, Any]]] = []
    for gid, payload in groups:
        if payload.get("is_marker"):
            continue
        comps = payload.get("components", [])
        if not isinstance(comps, list):
            continue
        if not (2 <= len(comps) <= 4):
            continue
        roles = {str(c.get("role", "")) for c in comps if isinstance(c, dict)}
        score = len(roles) * 10 + len(comps)
        best.append((score, gid, payload))
    if best:
        best.sort(key=lambda x: x[0], reverse=True)
        return best[0][1], best[0][2]
    for gid, payload in groups:
        if not payload.get("is_marker"):
            return gid, payload
    return groups[0]


def _make_group_simplified(groups: Sequence[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for gid, payload in groups:
        comps = []
        for comp in payload.get("components", []):
            if not isinstance(comp, dict):
                continue
            comps.append(
                {
                    "class": comp.get("class", ""),
                    "class_id": comp.get("class_id", None),
                    "role": comp.get("role", ""),
                }
            )
        out.append(
            {
                "group_id": gid,
                "sequence_index": payload.get("sequence_index", 0),
                "is_marker": payload.get("is_marker", False),
                "action": payload.get("fingering", ""),
                "string": payload.get("string", ""),
                "position": payload.get("position", ""),
                "finger": payload.get("finger", ""),
                "right_fingering": payload.get("right_fingering", ""),
                "left_fingering": payload.get("left_fingering", ""),
                "left_finger": payload.get("left_finger", ""),
                "hui": payload.get("hui", ""),
                "xian": payload.get("xian", ""),
                "components": comps,
            }
        )
    return out


def _extract_score_model_preview(score_model: Dict[str, Any], jianzi_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    notes_preview: List[Dict[str, Any]] = []
    seq_non_marker = [item for item in jianzi_sequence if not item.get("is_marker")]
    seq_idx = 0
    for measure_idx, measure in enumerate(score_model.get("measures", []), start=1):
        notes = measure.get("notes", [])
        if not isinstance(notes, list):
            continue
        for note in notes:
            if not isinstance(note, dict):
                continue
            group_id = ""
            if seq_idx < len(seq_non_marker):
                group_id = _safe_value(seq_non_marker[seq_idx].get("group_id", ""))
            notes_preview.append(
                {
                    "group_id": group_id,
                    "pitch": _safe_value(note.get("pitch", "")),
                    "octave": _safe_value(note.get("octave", "")),
                    "duration": _safe_value(note.get("duration", "")),
                    "measure": measure_idx,
                }
            )
            seq_idx += 1
            if len(notes_preview) >= 4:
                return {"notes": notes_preview}
    return {"notes": notes_preview}


def _create_semantic_figure(
    compact_groups: List[Dict[str, Any]],
    score_preview: Dict[str, Any],
    out_path: Path,
) -> None:
    white = (255, 255, 255)
    blue = (43, 96, 214)
    blue_fill = (235, 243, 255)
    green = (36, 146, 92)
    green_fill = (236, 249, 241)
    red = (206, 86, 86)
    red_fill = (255, 241, 241)
    gold = (164, 117, 28)
    gold_fill = (255, 248, 233)
    border = (196, 205, 216)
    text = (28, 36, 48)
    muted = (89, 99, 114)

    width, height = 5000, 1540
    canvas = Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(canvas)

    draw.text((105, 52), "语义重构思路", fill=text, font=_font(72, bold=True))
    draw.text(
        (108, 132),
        "视觉结果先被整理为减字组序列，再注入知识约束，经受约束生成、本地校验与局部修复后输出稳定结构。",
        fill=muted,
        font=_font(40),
    )

    top, bottom = 220, 1450
    names = ["减字组序列输入", "知识约束注入", "受约束生成", "本地校验与局部修复", "结构化输出"]
    box_ws = [920, 820, 820, 900, 940]
    gap = 46
    total = sum(box_ws) + gap * 4
    start_x = (width - total) // 2
    boxes: List[Tuple[int, int, int, int]] = []
    accents = [blue_fill, gold_fill, blue_fill, (247, 250, 255), green_fill]
    pills = [blue, gold, blue, (120, 137, 158), green]
    for idx, bw in enumerate(box_ws, start=1):
        x1 = start_x + (idx - 1) * (bw + gap)
        x2 = x1 + bw
        box = (x1, top, x2, bottom)
        boxes.append(box)
        draw.rounded_rectangle(box, radius=26, fill=white, outline=border, width=3)
        draw.rounded_rectangle((x1 + 22, top + 18, x1 + 92, top + 82), radius=14, fill=pills[idx - 1])
        draw.text((x1 + 42, top + 27), f"{idx}", fill=white, font=_font(28, bold=True))
        draw.text((x1 + 114, top + 24), names[idx - 1], fill=text, font=_font(43, bold=True))
        draw.line((x1 + 26, top + 110, x2 - 26, top + 110), fill=(220, 226, 236), width=2)
        draw.rounded_rectangle((x1 + 14, top + 16, x2 - 14, top + 28), radius=6, fill=accents[idx - 1], outline=accents[idx - 1], width=0)

    # Stage 1
    s1 = boxes[0]
    group_show = [item for item in compact_groups if not item.get("is_marker")][:3]
    while len(group_show) < 3:
        group_show.append({"group_id": f"group_{len(group_show)+1}", "action": "", "string": "", "hui": "", "finger": ""})
    for idx, item in enumerate(group_show):
        card_top = s1[1] + 140 + idx * 330
        card = (s1[0] + 24, card_top, s1[2] - 24, card_top + 286)
        draw.rounded_rectangle(card, radius=18, fill=(247, 250, 255), outline=(218, 226, 237), width=2)
        draw.text((card[0] + 22, card[1] + 14), _safe_value(item.get("group_id", "")), fill=blue, font=_font(34, bold=True))
        draw.text((card[0] + 22, card[1] + 68), f"动作: {_safe_value(item.get('action', '')) or '∅'}", fill=text, font=_font(31))
        draw.text((card[0] + 22, card[1] + 118), f"弦位: {_safe_value(item.get('string', '')) or '∅'}", fill=text, font=_font(31))
        draw.text((card[0] + 22, card[1] + 168), f"徽位: {_safe_value(item.get('hui', '')) or '∅'}", fill=text, font=_font(31))
        draw.text((card[0] + 22, card[1] + 218), f"手指: {_safe_value(item.get('finger', '')) or '∅'}", fill=text, font=_font(31))

    # Stage 2
    s2 = boxes[1]
    table = (s2[0] + 22, s2[1] + 140, s2[2] - 22, s2[3] - 56)
    draw.rounded_rectangle(table, radius=16, fill=(255, 252, 244), outline=(234, 223, 196), width=2)
    draw.text((table[0] + 18, table[1] + 18), "知识规则（示意）", fill=(133, 92, 18), font=_font(36, bold=True))
    draw.text((table[0] + 18, table[1] + 72), "弦位与徽位用于收紧音高/八度候选空间", fill=muted, font=_font(29))
    draw.line((table[0] + 18, table[1] + 122, table[2] - 18, table[1] + 122), fill=(224, 214, 188), width=2)
    draw.text((table[0] + 18, table[1] + 144), "命中字段", fill=muted, font=_font(29, bold=True))
    draw.text((table[0] + 230, table[1] + 144), "约束效果", fill=muted, font=_font(29, bold=True))
    rows = []
    for item in group_show:
        xian = _safe_value(item.get("xian", "")) or _safe_value(item.get("string", "")) or "∅"
        hui = _safe_value(item.get("hui", "")) or "∅"
        rows.append((f"弦位={xian}, 徽位={hui}", "pitch / octave 候选收敛"))
    for idx, row in enumerate(rows):
        yy = table[1] + 204 + idx * 116
        draw.text((table[0] + 18, yy), row[0], fill=text, font=_font(29))
        draw.text((table[0] + 230, yy), row[1], fill=text, font=_font(29))
    draw.rounded_rectangle((table[0] + 18, table[3] - 78, table[0] + 216, table[3] - 24), radius=12, fill=gold_fill, outline=gold, width=2)
    draw.text((table[0] + 34, table[3] - 68), "知识约束", fill=gold, font=_font(28, bold=True))

    # Stage 3
    s3 = boxes[2]
    core = (s3[0] + 20, s3[1] + 138, s3[2] - 20, s3[3] - 48)
    draw.rounded_rectangle(core, radius=24, fill=blue_fill, outline=(206, 220, 248), width=2)
    draw.text((core[0] + 28, core[1] + 34), "G(X, K, P)", fill=blue, font=_font(60, bold=True))
    draw.line((core[0] + 24, core[1] + 116, core[2] - 24, core[1] + 116), fill=(198, 214, 248), width=3)
    bullets = ["序列输入与知识同时注入", "提示词约束字段完整性", "生成结构化草案", "保持 group 顺序语义"]
    for idx, line in enumerate(bullets):
        draw.text((core[0] + 30, core[1] + 148 + idx * 90), f"• {line}", fill=text, font=_font(34))
    draft = (core[0] + 24, core[1] + 550, core[2] - 24, core[3] - 24)
    draw.rounded_rectangle(draft, radius=14, fill=white, outline=(198, 214, 248), width=2)
    draw.text((draft[0] + 20, draft[1] + 16), "结构草案", fill=blue, font=_font(30, bold=True))
    _draw_multiline(
        draw,
        (draft[0] + 20, draft[1] + 62),
        '{"group_id":"group_1","pitch":"5","octave":"3","duration":"4"}',
        _font(27),
        text,
        max_width=draft[2] - draft[0] - 40,
        line_gap=6,
        max_lines=4,
    )

    # Stage 4
    s4 = boxes[3]
    valid = (s4[0] + 20, s4[1] + 140, s4[2] - 20, s4[1] + 520)
    repair = (s4[0] + 20, s4[1] + 620, s4[2] - 20, s4[3] - 48)
    draw.rounded_rectangle(valid, radius=18, fill=(246, 250, 255), outline=(210, 220, 236), width=2)
    draw.rounded_rectangle(repair, radius=18, fill=red_fill, outline=(238, 194, 194), width=2)
    draw.text((valid[0] + 22, valid[1] + 16), "本地校验", fill=text, font=_font(40, bold=True))
    draw.text((valid[0] + 22, valid[1] + 88), "字段合法性", fill=text, font=_font(32))
    draw.text((valid[0] + 22, valid[1] + 148), "数量一致性", fill=text, font=_font(32))
    draw.text((valid[0] + 22, valid[1] + 208), "顺序一致性", fill=text, font=_font(32))
    draw.text((valid[0] + 22, valid[1] + 308), "合法则直接通过", fill=green, font=_font(32, bold=True))
    draw.text((repair[0] + 22, repair[1] + 16), "局部修复", fill=red, font=_font(40, bold=True))
    draw.text((repair[0] + 22, repair[1] + 88), "非法字段触发修复", fill=text, font=_font(32))
    draw.text((repair[0] + 22, repair[1] + 148), "仅修正异常字段或结构", fill=text, font=_font(32))
    draw.text((repair[0] + 22, repair[1] + 208), "不重写全部上下文", fill=text, font=_font(32))
    _draw_dashed_line(
        draw,
        ((valid[0] + valid[2]) // 2, valid[3] + 14),
        ((repair[0] + repair[2]) // 2, repair[1] - 16),
        color=red,
        width=5,
    )
    _draw_arrow(
        draw,
        ((valid[0] + valid[2]) // 2, valid[3] + 14),
        ((repair[0] + repair[2]) // 2, repair[1] - 16),
        color=red,
        width=1,
        head=16,
    )
    draw.text((valid[2] - 190, (valid[3] + repair[1]) // 2 - 10), "非法 -> 修复", fill=red, font=_font(29, bold=True))

    # Stage 5
    s5 = boxes[4]
    code = (s5[0] + 20, s5[1] + 140, s5[2] - 20, s5[3] - 48)
    draw.rounded_rectangle(code, radius=16, fill=(247, 252, 248), outline=(212, 231, 215), width=2)
    snippet = json.dumps(score_preview, ensure_ascii=False, indent=2)
    _draw_multiline(
        draw,
        (code[0] + 22, code[1] + 24),
        snippet,
        _font(30),
        text,
        max_width=code[2] - code[0] - 44,
        line_gap=8,
        max_lines=19,
    )

    chain_y = 840
    for idx in range(4):
        left = boxes[idx]
        right = boxes[idx + 1]
        color = green if idx == 3 else blue
        _draw_arrow(draw, (left[2] + 12, chain_y), (right[0] - 12, chain_y), color=color, width=7, head=20)
    draw.text((boxes[3][2] + 24, chain_y - 54), "合法通过", fill=green, font=_font(31, bold=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def _create_topology_figure(
    raw_img: np.ndarray,
    groups: Sequence[Tuple[str, Dict[str, Any]]],
    out_path: Path,
) -> None:
    white = (255, 255, 255)
    text = (26, 35, 46)
    muted = (90, 100, 115)
    border = (202, 210, 222)
    blue = (43, 97, 214)
    blue_fill = (236, 244, 255)
    orange = (242, 123, 31)
    red = (218, 86, 64)

    local_gid, local_payload = _pick_local_group(groups)
    comps = [item for item in local_payload.get("components", []) if isinstance(item, dict)]
    if not comps:
        comps = [item for item in groups[0][1].get("components", []) if isinstance(item, dict)]

    xs = [int(round(float(c["bbox"][0]))) for c in comps]
    ys = [int(round(float(c["bbox"][1]))) for c in comps]
    xe = [int(round(float(c["bbox"][2]))) for c in comps]
    ye = [int(round(float(c["bbox"][3]))) for c in comps]
    pad = 58
    h, w = raw_img.shape[:2]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w - 1, max(xe) + pad)
    y2 = min(h - 1, max(ye) + pad)

    crop = _to_pil(raw_img[y1:y2, x1:x2]).convert("RGBA")

    # 左段
    left_local = crop.copy()
    ld = ImageDraw.Draw(left_local)
    role_map = {
        "right_hand": "右手指法",
        "left_hand": "左手指法",
        "left_finger": "左手手指",
        "hui": "徽位",
        "xian": "弦位",
        "number": "数字",
    }
    colors = [(242, 96, 72), (64, 135, 235), (22, 170, 118), (226, 164, 36)]
    legends: List[Tuple[str, Tuple[int, int, int]]] = []
    for idx, comp in enumerate(comps):
        bx1, by1, bx2, by2 = [int(round(float(v))) for v in comp.get("bbox", [0, 0, 0, 0])]
        bx1 -= x1
        bx2 -= x1
        by1 -= y1
        by2 -= y1
        color = colors[idx % len(colors)]
        ld.rectangle([bx1, by1, bx2, by2], outline=color + (255,), width=4)
        role_text = role_map.get(_safe_value(comp.get("role", "")), _safe_value(comp.get("role", "")) or "部件")
        badge_box = [bx1 + 4, by1 + 4, bx1 + 30, by1 + 30]
        ld.ellipse(badge_box, fill=color + (235,))
        ld.text((bx1 + 12, by1 + 4), str(idx + 1), fill=(255, 255, 255, 255), font=_font(18, bold=True))
        legends.append((f"{idx + 1}. {role_text}", color))

    # 中段
    mid_local = crop.copy()
    gx1, gy1, gx2, gy2 = _bbox(local_payload)
    gx1 -= x1
    gx2 -= x1
    gy1 -= y1
    gy2 -= y1
    over = Image.new("RGBA", mid_local.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    od.rectangle([gx1, gy1, gx2, gy2], fill=(255, 110, 86, 80), outline=(224, 80, 52, 235), width=5)
    mid_local = Image.alpha_composite(mid_local, over)

    width, height = 4100, 1660
    canvas = Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(canvas)

    draw.text((80, 52), "拓扑关系建模与顺序组织", fill=text, font=_font(68, bold=True))

    panel_top, panel_bottom = 220, 1460
    left_panel = (80, panel_top, 1180, panel_bottom)
    mid_panel = (1230, panel_top, 2330, panel_bottom)
    right_panel = (2380, panel_top, width - 80, panel_bottom)
    for panel in (left_panel, mid_panel, right_panel):
        draw.rectangle(panel, outline=border, width=3)
    draw.text((left_panel[0] + 18, panel_top - 50), "原子部件层", fill=(52, 52, 52), font=_font(42, bold=True))
    draw.text((mid_panel[0] + 18, panel_top - 50), "减字组聚合层", fill=(52, 52, 52), font=_font(42, bold=True))
    draw.text((right_panel[0] + 18, panel_top - 50), "顺序组织层", fill=(52, 52, 52), font=_font(42, bold=True))

    left_img = ImageOps.contain(left_local.convert("RGB"), (left_panel[2] - left_panel[0] - 50, 780), Image.Resampling.NEAREST)
    lp_x = left_panel[0] + ((left_panel[2] - left_panel[0]) - left_img.width) // 2
    lp_y = left_panel[1] + 26
    canvas.paste(left_img, (lp_x, lp_y))
    legend_top = lp_y + left_img.height + 26
    legend_x = left_panel[0] + 28
    for idx, (label, color) in enumerate(legends[:4]):
        row_y = legend_top + idx * 76
        draw.rounded_rectangle((legend_x, row_y, legend_x + 40, row_y + 40), radius=8, fill=color, outline=color, width=0)
        draw.text((legend_x + 12, row_y + 4), str(idx + 1), fill=white, font=_font(22, bold=True))
        draw.text((legend_x + 58, row_y + 2), label, fill=text, font=_font(31))

    mid_img_target_w = 640
    mid_img_target_h = 820
    mid_img = ImageOps.contain(mid_local.convert("RGB"), (mid_img_target_w, mid_img_target_h), Image.Resampling.NEAREST)
    mp_x = mid_panel[0] + 24
    mp_y = mid_panel[1] + 34
    canvas.paste(mid_img, (mp_x, mp_y))

    card = (mid_panel[0] + 28, mp_y + mid_img.height + 30, mid_panel[2] - 28, mid_panel[3] - 34)
    draw.rounded_rectangle(card, radius=16, fill=(251, 252, 255), outline=(188, 199, 216), width=2)
    draw.text((card[0] + 20, card[1] + 16), f"{local_gid}", fill=red, font=_font(36, bold=True))
    draw.text((card[0] + 20, card[1] + 82), f"右手指法: {_safe_value(local_payload.get('right_fingering', '')) or '∅'}", fill=text, font=_font(30))
    draw.text((card[0] + 20, card[1] + 140), f"左手指法: {_safe_value(local_payload.get('left_fingering', '')) or '∅'}", fill=text, font=_font(30))
    draw.text((card[0] + 20, card[1] + 198), f"左手手指: {_safe_value(local_payload.get('left_finger', '')) or '∅'}", fill=text, font=_font(30))
    draw.text((card[0] + 20, card[1] + 256), f"徽位: {_safe_value(local_payload.get('hui', '')) or '∅'}", fill=text, font=_font(30))
    draw.text((card[0] + 20, card[1] + 314), f"弦位: {_safe_value(local_payload.get('xian', '')) or '∅'}", fill=text, font=_font(30))

    # 右段序列
    show_groups = [(gid, payload) for gid, payload in groups if not payload.get("is_marker")][:10]
    box_w, box_h = 182, 120
    gap_x = 22
    top_row_y = right_panel[1] + 244
    bottom_row_y = top_row_y + 208
    for idx, (gid, _) in enumerate(show_groups, start=1):
        row = 0 if idx <= 5 else 1
        col = idx - 1 if row == 0 else 10 - idx
        bx1 = right_panel[0] + 32 + col * (box_w + gap_x)
        bx2 = bx1 + box_w
        by1 = top_row_y if row == 0 else bottom_row_y
        by2 = by1 + box_h
        draw.rounded_rectangle((bx1, by1, bx2, by2), radius=14, fill=blue_fill, outline=blue, width=3)
        draw.text((bx1 + 22, by1 + 18), str(idx), fill=blue, font=_font(46, bold=True))
        gid_label = gid.replace("group_", "G")
        draw.text((bx1 + 86, by1 + 68), gid_label, fill=text, font=_font(28))
        if idx < len(show_groups):
            if idx == 5:
                _draw_polyline_arrow(
                    draw,
                    [
                        (bx2 + 8, (by1 + by2) // 2),
                        (right_panel[2] - 42, (by1 + by2) // 2),
                        (right_panel[2] - 42, bottom_row_y + box_h // 2),
                        (right_panel[2] - 64, bottom_row_y + box_h // 2),
                    ],
                    color=blue,
                    width=4,
                )
            else:
                if idx < 5:
                    target_x = bx2 + gap_x - 8
                    _draw_arrow(draw, (bx2 + 6, (by1 + by2) / 2), (target_x, (by1 + by2) / 2), color=blue, width=4, head=12)
                elif idx >= 6:
                    next_bx1 = bx1 - gap_x + 8
                    _draw_arrow(draw, (bx1 - 6, (by1 + by2) / 2), (next_bx1, (by1 + by2) / 2), color=blue, width=4, head=12)

    seq_card = (right_panel[0] + 34, right_panel[3] - 300, right_panel[2] - 34, right_panel[3] - 34)
    draw.rounded_rectangle(seq_card, radius=16, fill=(250, 252, 255), outline=(188, 199, 216), width=2)
    draw.text((seq_card[0] + 20, seq_card[1] + 16), "有序减字组序列", fill=text, font=_font(36, bold=True))
    draw.text((seq_card[0] + 20, seq_card[1] + 76), "读取方向：右列 -> 左列，列内自上而下", fill=muted, font=_font(29))
    draw.text((seq_card[0] + 20, seq_card[1] + 130), "输出形式：Ordered Sequence = [G1, G2, G3, ...]", fill=blue, font=_font(30, bold=True))

    # 中间转换箭头
    _draw_arrow(
        draw,
        (left_panel[2] + 12, (panel_top + panel_bottom) / 2),
        (mid_panel[0] - 12, (panel_top + panel_bottom) / 2),
        color=muted,
        width=6,
        head=16,
    )
    _draw_arrow(
        draw,
        (mid_panel[2] + 12, (panel_top + panel_bottom) / 2),
        (right_panel[0] - 12, (panel_top + panel_bottom) / 2),
        color=muted,
        width=6,
        head=16,
    )

    statement = "平面空间关系被压缩为可供语义重构使用的有序减字组序列"
    draw.text((80, 1510), statement, fill=blue, font=_font(48, bold=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def _create_case_study_figure(
    raw_input: Image.Image,
    intermediate: Image.Image,
    final_score: Image.Image,
    xml_snippet: str,
    out_path: Path,
) -> None:
    white = (255, 255, 255)
    text = (26, 35, 46)
    border = (203, 212, 224)
    blue = (43, 97, 214)
    panel_bg = (251, 252, 255)
    code_bg = (245, 247, 250)

    width, height = 3520, 2460
    canvas = Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 48), "代表性谱例译谱实测与终态输出", fill=text, font=_font(64, bold=True))

    top_offset = 160
    margin = 80
    gap = 54
    panel_w = (width - margin * 2 - gap) // 2
    panel_h = (height - top_offset - margin - gap) // 2

    tl = (margin, top_offset, margin + panel_w, top_offset + panel_h)
    tr = (margin + panel_w + gap, top_offset, width - margin, top_offset + panel_h)
    bl = (margin, top_offset + panel_h + gap, margin + panel_w, height - margin)
    br = (margin + panel_w + gap, top_offset + panel_h + gap, width - margin, height - margin)

    panels = [
        (tl, "输入谱例", raw_input),
        (tr, "中间结果", intermediate),
        (bl, "终态谱面", final_score),
    ]

    for box, title, img in panels:
        draw.rounded_rectangle(box, radius=18, fill=panel_bg, outline=border, width=3)
        draw.text((box[0] + 18, box[1] + 14), title, fill=text, font=_font(40, bold=True))
        inner = (box[0] + 16, box[1] + 78, box[2] - 16, box[3] - 16)
        draw.rounded_rectangle(inner, radius=12, fill=white, outline=(228, 233, 240), width=2)
        fit = ImageOps.contain(img.convert("RGB"), (inner[2] - inner[0] - 10, inner[3] - inner[1] - 10), Image.Resampling.LANCZOS)
        px = inner[0] + ((inner[2] - inner[0]) - fit.width) // 2
        py = inner[1] + ((inner[3] - inner[1]) - fit.height) // 2
        canvas.paste(fit, (px, py))

    # 右下 XML 片段
    draw.rounded_rectangle(br, radius=18, fill=panel_bg, outline=border, width=3)
    draw.text((br[0] + 18, br[1] + 14), "MusicXML片段", fill=text, font=_font(40, bold=True))
    code_inner = (br[0] + 16, br[1] + 78, br[2] - 16, br[3] - 16)
    draw.rounded_rectangle(code_inner, radius=12, fill=code_bg, outline=(228, 233, 240), width=2)
    _draw_multiline(
        draw,
        (code_inner[0] + 18, code_inner[1] + 14),
        xml_snippet,
        _font(27),
        (33, 40, 52),
        max_width=code_inner[2] - code_inner[0] - 36,
        line_gap=6,
        max_lines=36,
    )

    # 阅读路径箭头（小而清晰）
    _draw_arrow(
        draw,
        (tl[2] + 16, (tl[1] + tl[3]) / 2),
        (tr[0] - 16, (tr[1] + tr[3]) / 2),
        color=blue,
        width=6,
        head=18,
    )
    _draw_polyline_arrow(
        draw,
        [
            ((tr[0] + tr[2]) // 2, tr[3] + 12),
            ((tr[0] + tr[2]) // 2, tr[3] + 44),
            ((bl[0] + bl[2]) // 2, bl[1] - 26),
        ],
        color=blue,
        width=6,
    )
    _draw_arrow(
        draw,
        (bl[2] + 16, (bl[1] + bl[3]) / 2),
        (br[0] - 16, (br[1] + br[3]) / 2),
        color=blue,
        width=6,
        head=18,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def _crop_final_score(source_path: Path, out_path: Path) -> Image.Image:
    src = Image.open(source_path).convert("RGB")
    w, h = src.size
    crop_box = (
        int(w * 0.536),
        int(h * 0.124),
        int(w * 0.957),
        int(h * 0.921),
    )
    cropped = src.crop(crop_box)
    cropped = _crop_white_margin_pil(cropped, threshold=248, pad=12)
    cropped = cropped.crop((0, 0, cropped.width, int(cropped.height * 0.48)))
    cropped = ImageOps.expand(cropped, border=10, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out_path, format="PNG")
    return cropped


def _make_musicxml_snippet(music_xml: str, max_lines: int = 54) -> str:
    lines = [line.rstrip() for line in music_xml.splitlines() if line.strip()]
    if not lines:
        return ""
    snippet = lines[:max_lines]
    if len(lines) > max_lines:
        snippet.append("...")
    return "\n".join(snippet)


def _copy_source_assets(raw_path: Path, result_json_path: Path, out_source_dir: Path) -> None:
    out_source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, out_source_dir / raw_path.name)
    shutil.copy2(result_json_path, out_source_dir / result_json_path.name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MAT_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULT_JSON.exists():
        raise FileNotFoundError(f"结果 JSON 不存在: {RESULT_JSON}")

    result = _read_json(RESULT_JSON)

    raw_path = RAW_IMAGE if RAW_IMAGE.exists() else RAW_IMAGE_FALLBACK
    if not raw_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {raw_path}")
    raw = _read_image(raw_path)

    yolo_boxes = result.get("yolo_boxes", [])
    topology_json = result.get("topology_json", {})
    jianzi_sequence = result.get("jianzi_sequence", [])
    score_model = result.get("score_model", {})
    music_xml = _safe_value(result.get("music_xml", ""))

    groups = _sort_groups(topology_json)
    simplified_groups = _make_group_simplified(groups)
    score_preview = _extract_score_model_preview(score_model, jianzi_sequence)
    xml_snippet = _make_musicxml_snippet(music_xml, max_lines=54)

    # 素材图
    raw_png = MAT_DIR / "raw_input.png"
    det_png = MAT_DIR / "detection_boxes.png"
    agg_png = MAT_DIR / "group_aggregation.png"
    ord_png = MAT_DIR / "ordered_sequence.png"
    score_crop_png = MAT_DIR / "final_score_cropped.png"

    det_img = _draw_detection_boxes(raw, yolo_boxes)
    agg_img = _draw_group_boxes(raw, groups)
    ord_img = _draw_ordered_boxes(raw, groups, max_labels=55)
    _save_png(raw_png, raw)
    _save_png(det_png, det_img)
    _save_png(agg_png, agg_img)
    _save_png(ord_png, ord_img)
    score_crop_img = _crop_final_score(SCORE_RENDER_SOURCE, score_crop_png)

    # 文本素材
    simp_group_json = TXT_DIR / "group_sequence_simplified.json"
    topo_json_path = TXT_DIR / "topology_full.json"
    seq_json_path = TXT_DIR / "jianzi_sequence_full.json"
    xml_snippet_path = TXT_DIR / "musicxml_snippet.xml"
    score_preview_json = TXT_DIR / "structured_score_preview.json"

    _write_json(simp_group_json, simplified_groups)
    _write_json(topo_json_path, topology_json)
    _write_json(seq_json_path, jianzi_sequence)
    _write_json(score_preview_json, score_preview)
    xml_snippet_path.write_text(xml_snippet + "\n", encoding="utf-8")

    # 三张最终图
    fig_semantic = FIG_DIR / "semantic_reconstruction_overview.png"
    fig_topology = FIG_DIR / "topology_ordering_overview.png"
    fig_case = FIG_DIR / "representative_case_end_to_end.png"

    _create_semantic_figure(simplified_groups, score_preview, fig_semantic)
    _create_topology_figure(raw, groups, fig_topology)

    dense_bbox = _find_dense_region(yolo_boxes, raw.shape[1], raw.shape[0], win_w=430, win_h=520, stride=24)
    raw_local = _crop_pil_by_bbox(_to_pil(raw), dense_bbox, pad=10)
    ord_local = _crop_pil_by_bbox(_to_pil(ord_img), dense_bbox, pad=10)
    raw_local = _crop_white_margin_pil(raw_local, threshold=248, pad=12)
    ord_local = _crop_white_margin_pil(ord_local, threshold=248, pad=12)
    raw_local_png = MAT_DIR / "raw_input_local.png"
    ord_local_png = MAT_DIR / "ordered_sequence_local.png"
    raw_local.save(raw_local_png, format="PNG")
    ord_local.save(ord_local_png, format="PNG")

    _create_case_study_figure(
        raw_input=raw_local,
        intermediate=ord_local,
        final_score=score_crop_img,
        xml_snippet=xml_snippet,
        out_path=fig_case,
    )

    # 同步给图5-4线程常用素材名
    fig54_dir = MAT_DIR / "fig5-4_materials"
    fig54_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_png, fig54_dir / "raw_input.png")
    shutil.copy2(det_png, fig54_dir / "detection_boxes.png")
    shutil.copy2(agg_png, fig54_dir / "group_aggregation.png")
    shutil.copy2(ord_png, fig54_dir / "ordered_sequence.png")
    shutil.copy2(simp_group_json, fig54_dir / "group_sequence_simplified.json")

    _copy_source_assets(raw_path=raw_path, result_json_path=RESULT_JSON, out_source_dir=SRC_DIR)

    summary = {
        "sample": {
            "raw_image": str(raw_path),
            "result_json": str(RESULT_JSON),
            "score_render_source": str(SCORE_RENDER_SOURCE),
            "reuse_existing_result": True,
        },
        "counts": {
            "detected_boxes": len(yolo_boxes),
            "group_count": len(groups),
            "ordered_sequence_length": len(jianzi_sequence),
        },
        "final_figures": {
            "semantic_reconstruction_overview": str(fig_semantic),
            "topology_ordering_overview": str(fig_topology),
            "representative_case_end_to_end": str(fig_case),
        },
        "materials": {
            "raw_input": str(raw_png),
            "detection_boxes": str(det_png),
            "group_aggregation": str(agg_png),
            "ordered_sequence": str(ord_png),
            "raw_input_local": str(raw_local_png),
            "ordered_sequence_local": str(ord_local_png),
            "final_score_cropped": str(score_crop_png),
            "fig5_4_materials_dir": str(fig54_dir),
        },
        "text_snippets": {
            "group_sequence_simplified_json": str(simp_group_json),
            "topology_full_json": str(topo_json_path),
            "jianzi_sequence_full_json": str(seq_json_path),
            "structured_score_preview_json": str(score_preview_json),
            "musicxml_snippet_xml": str(xml_snippet_path),
        },
        "source_data": {
            "copied_raw_and_result": str(SRC_DIR),
        },
    }
    summary_path = OUT_DIR / "delivery_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
