# -*- coding: utf-8 -*-
"""
生成论文 C/D 档图与图5-4素材（检测框、候选框回映、减字组聚合、拓扑关系建模、阅读顺序组织）

运行方式（项目根目录）:
  F:\\anaconda\\envs\\pytorch\\python.exe test/generate_paper_cd_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
OUTPUT_DIR = TEST_DIR / "paper_figures"
MATERIAL_DIR = OUTPUT_DIR / "fig5-4_materials"
SOURCE_DIR = OUTPUT_DIR / "source_data"

SAMPLE_IMAGE = TEST_DIR / "testpicture-1.jpg"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.cv_module import detect_components
from backend.pipeline.topology_module import build_jianzi_sequence, build_topology


def _read_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def _save_image(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    buf.tofile(str(path))


def _to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
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
            "msyh.ttc",
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
    width: int = 3,
    head: int = 12,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line([sx, sy, ex, ey], fill=color, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    left = (
        ex - head * math.cos(angle - math.pi / 6),
        ey - head * math.sin(angle - math.pi / 6),
    )
    right = (
        ex - head * math.cos(angle + math.pi / 6),
        ey - head * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=color)


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    width: int = 3,
    dash: int = 10,
    gap: int = 7,
) -> None:
    x1, y1, x2, y2 = bbox
    for x in range(x1, x2, dash + gap):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
    for y in range(y1, y2, dash + gap):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)


def _sort_groups(topology_json: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    def key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
        group_id, payload = item
        seq_idx = payload.get("sequence_index", 0)
        try:
            return int(seq_idx), str(group_id)
        except Exception:
            return 10**9, str(group_id)

    return sorted(topology_json.items(), key=key)


def _group_bbox(payload: Dict[str, Any]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = payload.get("group_bbox", [0, 0, 0, 0])
    return int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))


def _draw_detection_boxes(base_bgr: np.ndarray, boxes: Sequence[Dict[str, Any]]) -> np.ndarray:
    canvas = base_bgr.copy()
    for item in boxes:
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 1)
    return canvas


def _draw_group_boxes(base_bgr: np.ndarray, groups: Sequence[Tuple[str, Dict[str, Any]]]) -> np.ndarray:
    canvas = base_bgr.copy()
    for _, payload in groups:
        x1, y1, x2, y2 = _group_bbox(payload)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 80, 235), 2)
    return canvas


def _draw_sequence_boxes(
    base_bgr: np.ndarray,
    groups: Sequence[Tuple[str, Dict[str, Any]]],
    number_all: bool = False,
) -> np.ndarray:
    pil = _to_pil(base_bgr.copy()).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    font = _load_font(13, bold=True)
    for idx, (_, payload) in enumerate(groups, start=1):
        x1, y1, x2, y2 = _group_bbox(payload)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 120, 20, 255), width=2)
        if number_all:
            tx, ty = x1 + 2, max(0, y1 - 16)
            draw.rectangle([tx - 1, ty - 1, tx + 26, ty + 15], fill=(255, 120, 20, 220))
            draw.text((tx + 2, ty), str(idx), fill=(255, 255, 255, 255), font=font)
    return _to_bgr(pil.convert("RGB"))


def _find_dense_region(
    boxes: Sequence[Dict[str, Any]],
    img_w: int,
    img_h: int,
    win_w: int = 250,
    win_h: int = 260,
    stride: int = 36,
) -> Tuple[int, int, int, int]:
    centers: List[Tuple[float, float]] = []
    for item in boxes:
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    best = (-1, 0, 0)
    for y in range(0, max(1, img_h - win_h + 1), stride):
        for x in range(0, max(1, img_w - win_w + 1), stride):
            count = 0
            x2, y2 = x + win_w, y + win_h
            for cx, cy in centers:
                if x <= cx <= x2 and y <= cy <= y2:
                    count += 1
            if count > best[0]:
                best = (count, x, y)

    _, x, y = best
    return x, y, min(x + win_w, img_w - 1), min(y + win_h, img_h - 1)


def _pick_local_group(groups: Sequence[Tuple[str, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    candidates: List[Tuple[int, str, Dict[str, Any]]] = []
    for group_id, payload in groups:
        components = payload.get("components", [])
        if not isinstance(components, list):
            continue
        if payload.get("is_marker"):
            continue
        if not (2 <= len(components) <= 4):
            continue
        roles = {str(item.get("role", "")) for item in components if isinstance(item, dict)}
        score = len(roles) * 10 + len(components)
        candidates.append((score, group_id, payload))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, gid, payload = candidates[0]
        return gid, payload
    return groups[0]


def _create_fig_4_5(raw_img: np.ndarray, boxes: List[Dict[str, Any]], out_path: Path) -> None:
    det_img = _draw_detection_boxes(raw_img, boxes)
    h, w = det_img.shape[:2]
    dense_bbox = _find_dense_region(boxes, w, h)

    det_pil = _to_pil(det_img)
    draw_det = ImageDraw.Draw(det_pil)
    _draw_dashed_rect(draw_det, dense_bbox, color=(255, 70, 60), width=4)

    x1, y1, x2, y2 = dense_bbox
    crop = det_pil.crop((x1, y1, x2, y2))
    zoom = crop.resize((int(crop.width * 2.4), int(crop.height * 2.4)), Image.Resampling.NEAREST)

    margin = 56
    top_margin = 156
    canvas_w = det_pil.width + zoom.width + margin * 3
    canvas_h = max(det_pil.height + top_margin + margin + 120, zoom.height + top_margin + margin * 2 + 280)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(54, bold=True)
    sub_font = _load_font(40, bold=True)
    txt_font = _load_font(33, bold=False)

    left_x = margin
    left_y = top_margin
    right_x = left_x + det_pil.width + margin
    right_y = top_margin + 128

    canvas.paste(det_pil, (left_x, left_y))
    canvas.paste(zoom, (right_x, right_y))
    draw.rectangle(
        [right_x - 3, right_y - 3, right_x + zoom.width + 3, right_y + zoom.height + 3],
        outline=(65, 65, 65),
        width=3,
    )

    draw.text((left_x, 24), "整页部件检测结果", fill=(20, 20, 20), font=title_font)
    draw.text((right_x, right_y - 74), "密集部件局部放大", fill=(20, 20, 20), font=sub_font)
    draw.text(
        (right_x, right_y + zoom.height + 44),
        "原子部件级检测结果为后续拓扑聚合提供候选输入",
        fill=(45, 45, 45),
        font=txt_font,
    )

    # 指示箭头
    dense_center = (left_x + (x1 + x2) / 2.0, left_y + (y1 + y2) / 2.0)
    zoom_anchor = (right_x, right_y + zoom.height / 2.0)
    _draw_arrow(draw, dense_center, zoom_anchor, color=(220, 70, 60), width=6, head=20)

    _save_image(out_path, _to_bgr(canvas))


def _create_fig_4_6(
    raw_img: np.ndarray,
    boxes: List[Dict[str, Any]],
    groups: List[Tuple[str, Dict[str, Any]]],
    out_path: Path,
) -> None:
    left = _to_pil(_draw_detection_boxes(raw_img, boxes))
    right_base = _to_pil(_draw_group_boxes(raw_img, groups)).convert("RGBA")
    draw_right = ImageDraw.Draw(right_base)
    label_font = _load_font(32, bold=True)

    highlight = []
    for gid, payload in groups:
        comp_n = len(payload.get("components", []))
        if comp_n >= 3 and not payload.get("is_marker"):
            highlight.append((gid, payload, comp_n))
    highlight = sorted(highlight, key=lambda item: item[2], reverse=True)[:2]

    for gid, payload, _ in highlight:
        x1, y1, x2, y2 = _group_bbox(payload)
        draw_right.rectangle([x1, y1, x2, y2], outline=(255, 105, 35, 255), width=5)
        label_h = 30
        label_w = min(210, max(120, 18 + len(gid) * 15))
        ly1 = max(0, y1 - label_h - 2)
        draw_right.rectangle([x1, ly1, x1 + label_w, ly1 + label_h], fill=(255, 105, 35, 230))
        draw_right.text((x1 + 8, ly1 + 4), f"{gid}", fill=(255, 255, 255, 255), font=label_font)

    right = right_base.convert("RGB")

    margin = 56
    title_h = 124
    panel_title_h = 82
    image_top = title_h + panel_title_h
    bottom_info_h = 170
    mid_gap = 360
    canvas_w = left.width + right.width + mid_gap + margin * 2
    canvas_h = max(left.height, right.height) + image_top + bottom_info_h + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(54, bold=True)
    panel_font = _load_font(42, bold=True)
    txt_font = _load_font(34, bold=False)

    left_x = margin
    left_y = image_top
    right_x = left_x + left.width + mid_gap
    right_y = image_top
    canvas.paste(left, (left_x, left_y))
    canvas.paste(right, (right_x, right_y))

    # 主标题
    draw.text((margin, 18), "拓扑聚合阶段的减字组展示", fill=(20, 20, 20), font=title_font)

    # 左右分区标题（独立留白区）
    draw.text((left_x + 12, title_h), "原子部件检测框", fill=(28, 28, 28), font=panel_font)
    draw.text((right_x + 12, title_h), "减字组聚合框", fill=(28, 28, 28), font=panel_font)

    # 中间聚合说明卡（避免文字直接压在图像上）
    mid_x1 = left_x + left.width + 26
    mid_x2 = right_x - 26
    card_w = max(180, mid_x2 - mid_x1)
    card_h = 270
    card_x1 = mid_x1
    card_y1 = image_top + left.height // 2 - card_h // 2
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h
    draw.rounded_rectangle(
        [card_x1, card_y1, card_x2, card_y2],
        radius=20,
        fill=(255, 255, 255),
        outline=(70, 70, 70),
        width=2,
    )
    draw.text((card_x1 + 26, card_y1 + 24), "拓扑聚合", fill=(42, 112, 245), font=_load_font(46, bold=True))
    draw.text((card_x1 + 26, card_y1 + 104), "碎片化小框", fill=(55, 55, 55), font=txt_font)
    draw.text((card_x1 + 26, card_y1 + 154), "-> 组级结构框", fill=(55, 55, 55), font=txt_font)
    draw.text((card_x1 + 26, card_y1 + 204), "-> 减字语义单元", fill=(55, 55, 55), font=txt_font)

    # 箭头连接左右图与中间说明卡
    left_anchor = (left_x + left.width + 10, left_y + left.height // 2)
    card_left = (card_x1 - 8, card_y1 + card_h // 2)
    _draw_arrow(draw, left_anchor, card_left, color=(42, 112, 245), width=6, head=16)
    card_right = (card_x2 + 8, card_y1 + card_h // 2)
    right_anchor = (right_x - 10, right_y + right.height // 2)
    _draw_arrow(draw, card_right, right_anchor, color=(42, 112, 245), width=6, head=16)

    # 底部说明条（单独区域，不与图重叠）
    info_x1 = margin
    info_y1 = image_top + max(left.height, right.height) + 26
    info_x2 = canvas_w - margin
    info_y2 = info_y1 + 98
    draw.rounded_rectangle(
        [info_x1, info_y1, info_x2, info_y2],
        radius=14,
        fill=(235, 244, 255),
        outline=(150, 180, 220),
        width=2,
    )
    draw.text(
        (info_x1 + 20, info_y1 + 27),
        "多个原子部件被归并为同一减字组",
        fill=(32, 86, 170),
        font=txt_font,
    )
    _save_image(out_path, _to_bgr(canvas))


def _create_fig_4_7(
    raw_img: np.ndarray,
    groups: List[Tuple[str, Dict[str, Any]]],
    out_path: Path,
) -> np.ndarray:
    # 底图：所有组框
    base = _to_pil(_draw_group_boxes(raw_img, groups)).convert("RGBA")
    draw = ImageDraw.Draw(base)
    idx_font = _load_font(16, bold=True)
    title_font = _load_font(33, bold=True)
    text_font = _load_font(19, bold=False)

    # 选一段连续组：优先选同一列（x接近）中的前若干组
    centers = []
    for gid, payload in groups:
        x1, y1, x2, y2 = _group_bbox(payload)
        centers.append((gid, payload, (x1 + x2) / 2.0, (y1 + y2) / 2.0, (x1, y1, x2, y2)))

    selected: List[Tuple[str, Dict[str, Any], float, float, Tuple[int, int, int, int]]] = []
    if centers:
        ref_x = centers[0][2]
        for item in centers:
            if abs(item[2] - ref_x) <= 36 and len(selected) < 8:
                selected.append(item)
        if len(selected) < 5:
            selected = centers[:8]

    # 高亮连续段 + 顺序箭头
    if selected:
        ux1 = min(item[4][0] for item in selected) - 20
        uy1 = min(item[4][1] for item in selected) - 24
        ux2 = max(item[4][2] for item in selected) + 20
        uy2 = max(item[4][3] for item in selected) + 20
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        ov_draw.rectangle([ux1, uy1, ux2, uy2], fill=(255, 214, 102, 65), outline=(255, 170, 20, 200), width=3)
        base = Image.alpha_composite(base, overlay)
        draw = ImageDraw.Draw(base)

    for order, item in enumerate(selected, start=1):
        _, _, _, _, (x1, y1, x2, y2) = item
        draw.rectangle([x1, y1, x2, y2], outline=(255, 120, 20, 255), width=4)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(35, 120, 255, 220))
        draw.text((cx - 6, cy - 10), str(order), fill=(255, 255, 255, 255), font=idx_font)

    for i in range(len(selected) - 1):
        _, _, _, _, b1 = selected[i]
        _, _, _, _, b2 = selected[i + 1]
        c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
        c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
        _draw_arrow(draw, c1, c2, color=(35, 120, 255), width=4, head=12)

    # 加右侧序列卡片
    panel_w = 510
    canvas = Image.new("RGB", (base.width + panel_w, base.height + 130), (248, 249, 251))
    canvas.paste(base.convert("RGB"), (0, 96))
    cv_draw = ImageDraw.Draw(canvas)
    cv_draw.text((28, 20), "减字组顺序校验与交互查看示意", fill=(22, 22, 22), font=title_font)

    px1 = base.width + 24
    py1 = 170
    px2 = base.width + panel_w - 24
    py2 = base.height + 96 - 24
    cv_draw.rounded_rectangle([px1, py1, px2, py2], radius=18, outline=(85, 85, 85), width=2, fill=(255, 255, 255))
    cv_draw.text((px1 + 18, py1 + 16), "Ordered Sequence", fill=(20, 20, 20), font=_load_font(23, bold=True))

    seq_ids = [item[0] for item in selected]
    seq_str = ", ".join(seq_ids[:8])
    cv_draw.text((px1 + 18, py1 + 62), f"[{seq_str}]", fill=(50, 50, 50), font=text_font)
    cv_draw.text((px1 + 18, py1 + 114), "阅读方向：右列 -> 左列", fill=(35, 120, 255), font=text_font)
    cv_draw.text((px1 + 18, py1 + 148), "序列生成：组级拓扑 -> 一维输入", fill=(35, 120, 255), font=text_font)
    cv_draw.text((px1 + 18, py1 + 190), "用途：为 LLM 提供有序减字序列", fill=(70, 70, 70), font=text_font)

    out_bgr = _to_bgr(canvas)
    _save_image(out_path, out_bgr)
    return out_bgr


def _create_fig_3_10(
    raw_img: np.ndarray,
    groups: List[Tuple[str, Dict[str, Any]]],
    out_path: Path,
) -> None:
    local_gid, local_payload = _pick_local_group(groups)
    local_components = local_payload.get("components", [])
    if not local_components:
        local_components = groups[0][1].get("components", [])

    # 局部裁切
    xs = [int(round(float(item["bbox"][0]))) for item in local_components]
    ys = [int(round(float(item["bbox"][1]))) for item in local_components]
    xe = [int(round(float(item["bbox"][2]))) for item in local_components]
    ye = [int(round(float(item["bbox"][3]))) for item in local_components]
    pad = 42
    h, w = raw_img.shape[:2]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w - 1, max(xe) + pad)
    y2 = min(h - 1, max(ye) + pad)
    local = _to_pil(raw_img[y1:y2, x1:x2]).convert("RGBA")

    # 左段：原子部件框
    left_panel = local.copy()
    left_draw = ImageDraw.Draw(left_panel)
    small_font = _load_font(16, bold=True)
    comp_colors = [(245, 92, 70), (65, 145, 245), (20, 180, 120), (230, 170, 30)]
    for idx, item in enumerate(local_components):
        bx1, by1, bx2, by2 = [int(round(float(v))) for v in item["bbox"]]
        bx1 -= x1
        bx2 -= x1
        by1 -= y1
        by2 -= y1
        color = comp_colors[idx % len(comp_colors)]
        left_draw.rectangle([bx1, by1, bx2, by2], outline=color + (255,), width=3)
        tag = f"{item.get('class', '')}"
        left_draw.rectangle([bx1, max(0, by1 - 20), min(bx2 + 52, bx1 + 170), by1], fill=color + (220,))
        left_draw.text((bx1 + 3, max(0, by1 - 18)), tag, fill=(255, 255, 255, 255), font=small_font)

    # 中段：减字组封装 + 字段卡
    mid_panel = local.copy()
    mid_overlay = Image.new("RGBA", mid_panel.size, (0, 0, 0, 0))
    mid_draw = ImageDraw.Draw(mid_overlay)
    gx1, gy1, gx2, gy2 = _group_bbox(local_payload)
    gx1 -= x1
    gx2 -= x1
    gy1 -= y1
    gy2 -= y1
    mid_draw.rectangle([gx1, gy1, gx2, gy2], fill=(255, 110, 80, 72), outline=(230, 70, 40, 230), width=4)
    mid_panel = Image.alpha_composite(mid_panel, mid_overlay)
    mid_draw2 = ImageDraw.Draw(mid_panel)
    mid_draw2.text((gx1 + 6, max(0, gy1 - 24)), "减字组实体", fill=(230, 70, 40, 255), font=_load_font(18, bold=True))
    card_w = 260
    card_h = 160
    cx1 = max(8, min(mid_panel.width - card_w - 8, gx2 + 10))
    cy1 = max(8, gy1)
    mid_draw2.rounded_rectangle([cx1, cy1, cx1 + card_w, cy1 + card_h], radius=12, fill=(255, 255, 255, 235), outline=(75, 75, 75), width=2)
    card_font = _load_font(16, bold=False)
    lines = [
        f"Group: {local_gid}",
        f"右手指法: {local_payload.get('right_fingering', '')}",
        f"左手指法: {local_payload.get('left_fingering', '')}",
        f"左手手指: {local_payload.get('left_finger', '')}",
        f"徽序: {local_payload.get('hui', '')}",
        f"弦序: {local_payload.get('xian', '')}",
    ]
    for i, txt in enumerate(lines):
        mid_draw2.text((cx1 + 10, cy1 + 10 + i * 24), txt, fill=(35, 35, 35), font=card_font)

    # 右段：一维有序序列
    seq_panel = Image.new("RGBA", (780, local.height), (252, 252, 252, 255))
    seq_draw = ImageDraw.Draw(seq_panel)
    seq_font = _load_font(18, bold=True)
    show_groups = [gid for gid, payload in groups if not payload.get("is_marker")][:6]
    box_w, box_h = 108, 58
    sx = 20
    sy = local.height // 2 - box_h // 2
    for idx, gid in enumerate(show_groups, start=1):
        bx1 = sx + (idx - 1) * (box_w + 18)
        bx2 = bx1 + box_w
        by1 = sy
        by2 = by1 + box_h
        seq_draw.rounded_rectangle([bx1, by1, bx2, by2], radius=10, fill=(235, 243, 255), outline=(46, 110, 220), width=3)
        seq_draw.text((bx1 + 18, by1 + 7), f"{idx}", fill=(46, 110, 220), font=_load_font(26, bold=True))
        seq_draw.text((bx1 + 36, by1 + 30), gid.replace("group_", "G"), fill=(30, 30, 30), font=_load_font(14, bold=False))
        if idx < len(show_groups):
            _draw_arrow(seq_draw, (bx2 + 4, (by1 + by2) / 2), (bx2 + 16, (by1 + by2) / 2), (46, 110, 220), 3, 9)

    # 合成三段式图
    margin = 36
    top = 96
    panel_w = max(left_panel.width, mid_panel.width)
    total_w = panel_w * 2 + seq_panel.width + margin * 4
    total_h = max(local.height, seq_panel.height) + top + 136
    canvas = Image.new("RGB", (total_w, total_h), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)

    left_x = margin
    mid_x = left_x + panel_w + margin
    right_x = mid_x + panel_w + margin
    panel_y = top

    canvas.paste(left_panel.convert("RGB"), (left_x, panel_y))
    canvas.paste(mid_panel.convert("RGB"), (mid_x, panel_y))
    canvas.paste(seq_panel.convert("RGB"), (right_x, panel_y))

    title_font = _load_font(34, bold=True)
    sub_font = _load_font(22, bold=True)
    draw.text((margin, 20), "减字拓扑关系建模与减字组顺序组织示意", fill=(20, 20, 20), font=title_font)
    draw.text((left_x + 12, panel_y - 40), "左段：原子部件检测框", fill=(45, 45, 45), font=sub_font)
    draw.text((mid_x + 12, panel_y - 40), "中段：减字组实体建模", fill=(45, 45, 45), font=sub_font)
    draw.text((right_x + 12, panel_y - 40), "右段：有序减字组序列", fill=(45, 45, 45), font=sub_font)

    # 面板间箭头
    _draw_arrow(
        draw,
        (left_x + panel_w + 8, panel_y + local.height // 2),
        (mid_x - 8, panel_y + local.height // 2),
        color=(80, 80, 80),
        width=4,
        head=12,
    )
    _draw_arrow(
        draw,
        (mid_x + panel_w + 8, panel_y + local.height // 2),
        (right_x - 8, panel_y + local.height // 2),
        color=(80, 80, 80),
        width=4,
        head=12,
    )

    statement = "平面空间关系被压缩为可供语义重构使用的有序减字组序列"
    draw.text((margin + 6, total_h - 52), statement, fill=(24, 96, 196), font=_load_font(25, bold=True))
    _save_image(out_path, _to_bgr(canvas))


def _build_simplified_groups(groups: Sequence[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for group_id, payload in groups:
        components = payload.get("components", [])
        simple_components = []
        if isinstance(components, list):
            for item in components:
                if not isinstance(item, dict):
                    continue
                simple_components.append(
                    {
                        "class": item.get("class", ""),
                        "class_id": item.get("class_id", None),
                        "role": item.get("role", ""),
                    }
                )

        output.append(
            {
                "group_id": group_id,
                "sequence_index": payload.get("sequence_index", 0),
                "is_marker": bool(payload.get("is_marker", False)),
                "marker_type": payload.get("marker_type", ""),
                "right_fingering": payload.get("right_fingering", ""),
                "left_fingering": payload.get("left_fingering", ""),
                "left_finger": payload.get("left_finger", ""),
                "hui": payload.get("hui", ""),
                "xian": payload.get("xian", ""),
                "action": payload.get("fingering", ""),
                "string": payload.get("string", ""),
                "position": payload.get("position", ""),
                "finger": payload.get("finger", ""),
                "components": simple_components,
            }
        )
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MATERIAL_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    if not SAMPLE_IMAGE.exists():
        raise FileNotFoundError(f"样例图片不存在: {SAMPLE_IMAGE}")

    raw_img = _read_image(SAMPLE_IMAGE)
    yolo_boxes = detect_components(str(SAMPLE_IMAGE))
    topology_json = build_topology(yolo_boxes)
    jianzi_sequence = build_jianzi_sequence(topology_json)
    groups = _sort_groups(topology_json)

    # 素材图
    raw_png = MATERIAL_DIR / "raw_input.png"
    det_png = MATERIAL_DIR / "detection_boxes.png"
    group_png = MATERIAL_DIR / "group_aggregation.png"
    seq_png = MATERIAL_DIR / "ordered_sequence.png"
    _save_image(raw_png, raw_img)
    _save_image(det_png, _draw_detection_boxes(raw_img, yolo_boxes))
    _save_image(group_png, _draw_group_boxes(raw_img, groups))
    _save_image(seq_png, _draw_sequence_boxes(raw_img, groups, number_all=True))

    # 论文图
    fig310 = OUTPUT_DIR / "fig3-10_topology_sequence.png"
    fig45 = OUTPUT_DIR / "fig4-5_detection_stage.png"
    fig46 = OUTPUT_DIR / "fig4-6_topology_grouping.png"
    fig47 = OUTPUT_DIR / "fig4-7_sequence_validation.png"
    _create_fig_3_10(raw_img, groups, fig310)
    _create_fig_4_5(raw_img, yolo_boxes, fig45)
    _create_fig_4_6(raw_img, yolo_boxes, groups, fig46)
    _create_fig_4_7(raw_img, groups, fig47)

    # JSON 输出
    full_topology_json = SOURCE_DIR / "topology_full.json"
    full_sequence_json = SOURCE_DIR / "jianzi_sequence_full.json"
    simple_group_json = MATERIAL_DIR / "group_sequence_simplified.json"
    stats_json = OUTPUT_DIR / "figure_generation_summary.json"

    full_topology_json.write_text(json.dumps(topology_json, ensure_ascii=False, indent=2), encoding="utf-8")
    full_sequence_json.write_text(json.dumps(jianzi_sequence, ensure_ascii=False, indent=2), encoding="utf-8")
    simple_group_json.write_text(
        json.dumps(_build_simplified_groups(groups), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "sample_image": str(SAMPLE_IMAGE),
        "counts": {
            "detected_boxes": len(yolo_boxes),
            "group_count": len(groups),
            "ordered_sequence_length": len(jianzi_sequence),
        },
        "figures": {
            "fig3_10": str(fig310),
            "fig4_5": str(fig45),
            "fig4_6": str(fig46),
            "fig4_7": str(fig47),
        },
        "fig5_4_materials": {
            "raw_input": str(raw_png),
            "detection_boxes": str(det_png),
            "group_aggregation": str(group_png),
            "ordered_sequence": str(seq_png),
            "simplified_group_json": str(simple_group_json),
            "topology_json": str(full_topology_json),
            "sequence_json": str(full_sequence_json),
        },
    }
    stats_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
