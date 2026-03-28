# -*- coding: utf-8 -*-
"""
减字顺序可视化测试脚本
====================
使用 test/testpicture-1.jpg 走后端真实流程：
  1) detect_components
  2) build_topology
  3) 按 group 顺序标注 "第N字" 并输出图片

运行方式（项目根目录）：
    F:\anaconda\envs\pytorch\python.exe test/test_group_order_on_testpicture.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
TEST_DIR = PROJECT_ROOT / "test"

INPUT_IMAGE = TEST_DIR / "testpicture-1.jpg"
OUTPUT_DIR = TEST_DIR / "order_check_outputs"
OUTPUT_IMAGE = OUTPUT_DIR / "testpicture-1_group_order.jpg"
OUTPUT_JSON = OUTPUT_DIR / "testpicture-1_group_order.json"

sys.path.insert(0, str(BACKEND_DIR))

from pipeline.cv_module import detect_components  # noqa: E402
from pipeline.topology_module import build_topology  # noqa: E402


COLORS = [
    (239, 68, 68),
    (59, 130, 246),
    (16, 185, 129),
    (245, 158, 11),
    (168, 85, 247),
    (236, 72, 153),
    (14, 165, 233),
    (132, 204, 22),
]


def _read_image_safe(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def _save_image_safe(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError(f"图片编码失败: {path}")
    buf.tofile(str(path))


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("msyh.ttc", "simsun.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _sort_group_items(topology_json: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    def _key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
        group_id = str(item[0])
        match = re.search(r"(\d+)", group_id)
        if match:
            return (int(match.group(1)), group_id)
        return (10**9, group_id)

    return sorted(topology_json.items(), key=_key)


def _group_bbox(payload: Dict[str, Any]) -> List[float]:
    bbox = payload.get("group_bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(v) for v in bbox]

    components = payload.get("components") or []
    valid_boxes = []
    for comp in components:
        comp_box = comp.get("bbox") if isinstance(comp, dict) else None
        if isinstance(comp_box, (list, tuple)) and len(comp_box) == 4:
            valid_boxes.append([float(v) for v in comp_box])

    if not valid_boxes:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        min(b[0] for b in valid_boxes),
        min(b[1] for b in valid_boxes),
        max(b[2] for b in valid_boxes),
        max(b[3] for b in valid_boxes),
    ]


def _draw_order_overlay(image: np.ndarray, topology_json: Dict[str, Dict[str, Any]]) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font_main = _font(22)
    font_sub = _font(16)

    ordered_groups = _sort_group_items(topology_json)
    for idx, (group_id, payload) in enumerate(ordered_groups, start=1):
        x1, y1, x2, y2 = _group_bbox(payload)
        color = COLORS[(idx - 1) % len(COLORS)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        title = f"第{idx}字"
        subtitle = str(group_id)

        title_box = draw.textbbox((x1, y1), title, font=font_main)
        subtitle_box = draw.textbbox((x1, y1), subtitle, font=font_sub)

        label_w = max(title_box[2] - title_box[0], subtitle_box[2] - subtitle_box[0]) + 16
        label_h = (title_box[3] - title_box[1]) + (subtitle_box[3] - subtitle_box[1]) + 12

        lx1 = x1
        ly1 = max(0, y1 - label_h - 4)
        lx2 = lx1 + label_w
        ly2 = ly1 + label_h
        draw.rectangle([lx1, ly1, lx2, ly2], fill=color)

        draw.text((lx1 + 8, ly1 + 2), title, font=font_main, fill=(255, 255, 255))
        draw.text((lx1 + 8, ly1 + (title_box[3] - title_box[1]) + 4), subtitle, font=font_sub, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def _build_order_report(topology_json: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    report = []
    ordered_groups = _sort_group_items(topology_json)
    for idx, (group_id, payload) in enumerate(ordered_groups, start=1):
        report.append(
            {
                "order": idx,
                "group_id": group_id,
                "bbox": _group_bbox(payload),
                "action": str(payload.get("right_fingering") or payload.get("left_fingering") or payload.get("fingering") or ""),
                "string": str(payload.get("xian") or payload.get("string") or ""),
                "position": str(payload.get("hui") or payload.get("position") or ""),
                "finger": str(payload.get("left_finger") or payload.get("finger") or ""),
            }
        )
    return report


def main() -> None:
    print("=" * 70)
    print("减字顺序可视化测试（testpicture-1.jpg）")
    print("=" * 70)

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"测试图片不存在: {INPUT_IMAGE}")

    print(f"[1/4] 检测部件: {INPUT_IMAGE}")
    yolo_boxes = detect_components(str(INPUT_IMAGE))
    print(f"      检测框数量: {len(yolo_boxes)}")

    print("[2/4] 构建拓扑分组并排序")
    topology_json = build_topology(yolo_boxes)
    group_count = len(topology_json.keys()) if isinstance(topology_json, dict) else 0
    print(f"      group 数量: {group_count}")

    if not isinstance(topology_json, dict) or not topology_json:
        raise ValueError("拓扑结果为空，无法进行顺序可视化。")

    print("[3/4] 生成顺序标注图（第N字）")
    image = _read_image_safe(INPUT_IMAGE)
    rendered = _draw_order_overlay(image, topology_json)
    _save_image_safe(OUTPUT_IMAGE, rendered)

    print("[4/4] 导出顺序明细 JSON")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = _build_order_report(topology_json)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("-" * 70)
    print(f"输出图片: {OUTPUT_IMAGE}")
    print(f"输出JSON: {OUTPUT_JSON}")
    print(f"总计标注: {len(report)} 个减字组")
    print("-" * 70)
    print("前10个顺序预览:")
    for item in report[:10]:
        print(
            f"  第{item['order']}字 | {item['group_id']} | "
            f"action={item['action']} string={item['string']} position={item['position']} finger={item['finger']}"
        )


if __name__ == "__main__":
    main()
