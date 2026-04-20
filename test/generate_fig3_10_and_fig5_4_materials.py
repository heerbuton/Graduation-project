# -*- coding: utf-8 -*-
"""
本轮专用制图脚本：
1) 图3-10：减字拓扑关系建模与减字组顺序组织示意图
2) 图5-4素材：原图、检测框图、聚合框图、顺序编号图、简化group JSON

运行:
  F:\\anaconda\\envs\\pytorch\\python.exe test/generate_fig3_10_and_fig5_4_materials.py
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.cv_module import detect_components
from backend.pipeline.topology_module import build_jianzi_sequence, build_topology

TEST_DIR = ROOT / "test"
OUT_DIR = TEST_DIR / "paper_figures_v2"
MAT_DIR = OUT_DIR / "fig5-4_materials"
SRC_DIR = OUT_DIR / "source_data"
SAMPLE_IMAGE = TEST_DIR / "testpicture-1.jpg"


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
        raise RuntimeError(f"图片编码失败: {path}")
    buf.tofile(str(path))


def _to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _to_bgr(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    cand = []
    if bold:
        cand.extend([r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\arialbd.ttf"])
    cand.extend([r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\arial.ttf"])
    for p in cand:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_arrow(draw: ImageDraw.ImageDraw, p1: Tuple[float, float], p2: Tuple[float, float], color: Tuple[int, int, int], width: int = 4, head: int = 12) -> None:
    x1, y1 = p1
    x2, y2 = p2
    draw.line([x1, y1, x2, y2], fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    p3 = (x2 - head * math.cos(ang - math.pi / 6), y2 - head * math.sin(ang - math.pi / 6))
    p4 = (x2 - head * math.cos(ang + math.pi / 6), y2 - head * math.sin(ang + math.pi / 6))
    draw.polygon([p2, p3, p4], fill=color)


def _bbox(payload: Dict[str, Any]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = payload.get("group_bbox", [0, 0, 0, 0])
    return int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))


def _sort_groups(topology_json: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    def key(it: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
        gid, payload = it
        try:
            return int(payload.get("sequence_index", 10**9)), gid
        except Exception:
            return 10**9, gid
    return sorted(topology_json.items(), key=key)


def _draw_detection_boxes(raw_img: np.ndarray, yolo_boxes: Sequence[Dict[str, Any]]) -> np.ndarray:
    out = raw_img.copy()
    for item in yolo_boxes:
        bb = item.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bb]
        cv2.rectangle(out, (x1, y1), (x2, y2), (30, 170, 50), 1)
    return out


def _draw_group_boxes(raw_img: np.ndarray, groups: Sequence[Tuple[str, Dict[str, Any]]]) -> np.ndarray:
    out = raw_img.copy()
    for _, payload in groups:
        x1, y1, x2, y2 = _bbox(payload)
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 90, 230), 2)
    return out


def _draw_sequence_boxes(raw_img: np.ndarray, groups: Sequence[Tuple[str, Dict[str, Any]]]) -> np.ndarray:
    pil = _to_pil(raw_img).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    idx_font = _font(18, bold=True)
    for idx, (_, payload) in enumerate(groups, start=1):
        x1, y1, x2, y2 = _bbox(payload)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 125, 20, 255), width=2)
        lx1, ly1 = x1 + 2, max(0, y1 - 24)
        draw.rectangle([lx1, ly1, lx1 + 34, ly1 + 22], fill=(255, 125, 20, 235))
        draw.text((lx1 + 8, ly1 + 2), str(idx), fill=(255, 255, 255, 255), font=idx_font)
    return _to_bgr(pil.convert("RGB"))


def _mount_on_white(img_bgr: np.ndarray, title: str) -> np.ndarray:
    img = _to_pil(img_bgr)
    margin = 36
    title_h = 94
    canvas = Image.new("RGB", (img.width + margin * 2, img.height + title_h + margin), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), title, fill=(20, 20, 20), font=_font(48, bold=True))
    canvas.paste(img, (margin, title_h))
    draw.rectangle([margin - 1, title_h - 1, margin + img.width + 1, title_h + img.height + 1], outline=(180, 180, 180), width=2)
    return _to_bgr(canvas)


def _pick_local_group(groups: Sequence[Tuple[str, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    cands: List[Tuple[int, str, Dict[str, Any]]] = []
    for gid, payload in groups:
        comps = payload.get("components", [])
        if payload.get("is_marker"):
            continue
        if not isinstance(comps, list):
            continue
        if 2 <= len(comps) <= 4:
            roles = {str(c.get("role", "")) for c in comps if isinstance(c, dict)}
            score = len(roles) * 10 + len(comps)
            cands.append((score, gid, payload))
    if cands:
        cands.sort(key=lambda t: t[0], reverse=True)
        return cands[0][1], cands[0][2]
    return groups[0]


def _create_fig_3_10(raw_img: np.ndarray, groups: Sequence[Tuple[str, Dict[str, Any]]], out_path: Path) -> None:
    gid, payload = _pick_local_group(groups)
    comps = payload.get("components", [])
    if not comps:
        comps = groups[0][1].get("components", [])

    xs = [int(round(float(c["bbox"][0]))) for c in comps]
    ys = [int(round(float(c["bbox"][1]))) for c in comps]
    xe = [int(round(float(c["bbox"][2]))) for c in comps]
    ye = [int(round(float(c["bbox"][3]))) for c in comps]
    pad = 44
    h, w = raw_img.shape[:2]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w - 1, max(xe) + pad)
    y2 = min(h - 1, max(ye) + pad)
    crop = _to_pil(raw_img[y1:y2, x1:x2]).convert("RGBA")

    # 左段：原子部件
    left = crop.copy()
    left_draw = ImageDraw.Draw(left)
    small = _font(22, bold=True)
    colors = [(245, 90, 70), (60, 140, 240), (20, 170, 120), (220, 165, 30)]
    for i, comp in enumerate(comps):
        bx1, by1, bx2, by2 = [int(round(float(v))) for v in comp["bbox"]]
        bx1 -= x1
        bx2 -= x1
        by1 -= y1
        by2 -= y1
        cc = colors[i % len(colors)]
        left_draw.rectangle([bx1, by1, bx2, by2], outline=cc + (255,), width=4)
        role = str(comp.get("role", "部件"))
        left_draw.rectangle([bx1, max(0, by1 - 30), min(left.width - 1, bx1 + 130), by1], fill=cc + (230,))
        left_draw.text((bx1 + 8, max(0, by1 - 28)), role, fill=(255, 255, 255, 255), font=small)

    # 中段：减字组实体+字段卡
    mid = crop.copy()
    over = Image.new("RGBA", mid.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    gx1, gy1, gx2, gy2 = _bbox(payload)
    gx1 -= x1
    gx2 -= x1
    gy1 -= y1
    gy2 -= y1
    od.rectangle([gx1, gy1, gx2, gy2], fill=(255, 110, 80, 75), outline=(230, 70, 40, 235), width=5)
    mid = Image.alpha_composite(mid, over)
    md = ImageDraw.Draw(mid)
    md.text((gx1 + 6, max(0, gy1 - 30)), "减字组实体", fill=(225, 70, 40, 255), font=_font(26, bold=True))

    card_w, card_h = 360, 216
    cx1 = max(8, min(mid.width - card_w - 8, gx2 + 12))
    cy1 = max(8, gy1)
    md.rounded_rectangle([cx1, cy1, cx1 + card_w, cy1 + card_h], radius=12, fill=(255, 255, 255, 240), outline=(90, 90, 90), width=2)
    ff = _font(22, bold=False)
    lines = [
        f"Group: {gid}",
        f"右手指法: {payload.get('right_fingering', '')}",
        f"左手指法: {payload.get('left_fingering', '')}",
        f"左手手指: {payload.get('left_finger', '')}",
        f"徽序: {payload.get('hui', '')}",
        f"弦序: {payload.get('xian', '')}",
    ]
    for i, line in enumerate(lines):
        md.text((cx1 + 12, cy1 + 10 + i * 34), line, fill=(35, 35, 35), font=ff)

    # 右段：有序序列
    seq_panel = Image.new("RGBA", (860, crop.height), (255, 255, 255, 255))
    sd = ImageDraw.Draw(seq_panel)
    show = [g for g, p in groups if not p.get("is_marker")][:7]
    bx_w, bx_h = 110, 66
    sx = 22
    sy = crop.height // 2 - bx_h // 2
    for i, group_id in enumerate(show, start=1):
        bx1 = sx + (i - 1) * (bx_w + 18)
        bx2 = bx1 + bx_w
        by1 = sy
        by2 = by1 + bx_h
        sd.rounded_rectangle([bx1, by1, bx2, by2], radius=10, fill=(236, 244, 255), outline=(46, 110, 220), width=3)
        sd.text((bx1 + 16, by1 + 8), str(i), fill=(46, 110, 220), font=_font(34, bold=True))
        sd.text((bx1 + 48, by1 + 34), group_id.replace("group_", "G"), fill=(32, 32, 32), font=_font(16, bold=False))
        if i < len(show):
            _draw_arrow(sd, (bx2 + 4, (by1 + by2) / 2), (bx2 + 16, (by1 + by2) / 2), (46, 110, 220), width=3, head=9)

    # 三段合成
    margin = 34
    top = 130
    panel_w = max(left.width, mid.width)
    cw = panel_w * 2 + seq_panel.width + margin * 4
    ch = max(crop.height, seq_panel.height) + top + 150
    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    lx = margin
    mx = lx + panel_w + margin
    rx = mx + panel_w + margin
    py = top
    canvas.paste(left.convert("RGB"), (lx, py))
    canvas.paste(mid.convert("RGB"), (mx, py))
    canvas.paste(seq_panel.convert("RGB"), (rx, py))

    # 区域边界与分隔线（浅灰）
    sep_color = (205, 205, 205)
    draw.rectangle([lx - 2, py - 2, lx + panel_w + 2, py + crop.height + 2], outline=sep_color, width=2)
    draw.rectangle([mx - 2, py - 2, mx + panel_w + 2, py + crop.height + 2], outline=sep_color, width=2)
    draw.rectangle([rx - 2, py - 2, rx + seq_panel.width + 2, py + seq_panel.height + 2], outline=sep_color, width=2)

    draw.text((margin, 26), "减字拓扑关系建模与顺序组织", fill=(20, 20, 20), font=_font(56, bold=True))
    draw.text((lx + 10, py - 54), "原子部件", fill=(48, 48, 48), font=_font(32, bold=True))
    draw.text((mx + 10, py - 54), "减字组实体", fill=(48, 48, 48), font=_font(32, bold=True))
    draw.text((rx + 10, py - 54), "有序减字组序列", fill=(48, 48, 48), font=_font(32, bold=True))

    _draw_arrow(draw, (lx + panel_w + 10, py + crop.height // 2), (mx - 10, py + crop.height // 2), (90, 90, 90), width=5, head=14)
    _draw_arrow(draw, (mx + panel_w + 10, py + crop.height // 2), (rx - 10, py + crop.height // 2), (90, 90, 90), width=5, head=14)

    statement = "平面空间关系被压缩为可供语义重构使用的有序减字组序列"
    draw.text((margin + 2, ch - 64), statement, fill=(25, 95, 195), font=_font(36, bold=True))
    _save_png(out_path, _to_bgr(canvas))


def _simplified_group_json(groups: Sequence[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for gid, payload in groups:
        comps = []
        for c in payload.get("components", []):
            if not isinstance(c, dict):
                continue
            comps.append({"class": c.get("class", ""), "class_id": c.get("class_id", None), "role": c.get("role", "")})
        out.append(
            {
                "group_id": gid,
                "sequence_index": payload.get("sequence_index", 0),
                "right_fingering": payload.get("right_fingering", ""),
                "left_fingering": payload.get("left_fingering", ""),
                "left_finger": payload.get("left_finger", ""),
                "action": payload.get("fingering", ""),
                "string": payload.get("string", ""),
                "position": payload.get("position", ""),
                "finger": payload.get("finger", ""),
                "hui": payload.get("hui", ""),
                "xian": payload.get("xian", ""),
                "components": comps,
            }
        )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MAT_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    if not SAMPLE_IMAGE.exists():
        raise FileNotFoundError(f"样例图片不存在: {SAMPLE_IMAGE}")

    raw = _read_image(SAMPLE_IMAGE)
    yolo_boxes = detect_components(str(SAMPLE_IMAGE))
    topology_json = build_topology(yolo_boxes)
    seq_json = build_jianzi_sequence(topology_json)
    groups = _sort_groups(topology_json)

    # 图3-10
    fig310 = OUT_DIR / "fig3-10_topology_sequence.png"
    _create_fig_3_10(raw, groups, fig310)

    # 图5-4素材（纯白背景，无图号）
    raw_png = MAT_DIR / "raw_input.png"
    det_png = MAT_DIR / "detection_boxes.png"
    agg_png = MAT_DIR / "group_aggregation.png"
    ord_png = MAT_DIR / "ordered_sequence.png"
    simp_json = MAT_DIR / "group_sequence_simplified.json"
    topo_json = SRC_DIR / "topology_full.json"
    seq_out_json = SRC_DIR / "jianzi_sequence_full.json"
    summary_json = OUT_DIR / "figure_generation_summary.json"

    _save_png(raw_png, _mount_on_white(raw, "原始谱页输入"))
    _save_png(det_png, _mount_on_white(_draw_detection_boxes(raw, yolo_boxes), "原子部件检测框"))
    _save_png(agg_png, _mount_on_white(_draw_group_boxes(raw, groups), "减字组聚合框"))
    _save_png(ord_png, _mount_on_white(_draw_sequence_boxes(raw, groups), "减字组顺序编号"))

    simp_json.write_text(json.dumps(_simplified_group_json(groups), ensure_ascii=False, indent=2), encoding="utf-8")
    topo_json.write_text(json.dumps(topology_json, ensure_ascii=False, indent=2), encoding="utf-8")
    seq_out_json.write_text(json.dumps(seq_json, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "sample_image": str(SAMPLE_IMAGE),
        "counts": {
            "detected_boxes": len(yolo_boxes),
            "group_count": len(groups),
            "ordered_sequence_length": len(seq_json),
        },
        "figures": {"fig3_10": str(fig310)},
        "fig5_4_materials": {
            "raw_input": str(raw_png),
            "detection_boxes": str(det_png),
            "group_aggregation": str(agg_png),
            "ordered_sequence": str(ord_png),
            "simplified_group_json": str(simp_json),
            "topology_json": str(topo_json),
            "sequence_json": str(seq_out_json),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
