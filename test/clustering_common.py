# -*- coding: utf-8 -*-
"""
空间聚类公共模块 — 供 test_clustering_fullpage.py 和 test_clustering_sahi.py 共用
"""

import math
import re
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 标签分类体系 (古琴减字谱)
# ============================================================
ACTION_LABELS = {
    "勾", "抹", "挑", "托", "打", "摘", "剔", "历", "轮", "撮",
    "注", "吟", "猱", "绰", "撞", "进复", "退复", "散", "擘",
    "泛起", "泛止", "滚", "拂",
}
FINGER_LABELS = {"大", "食", "中", "名", "跪"}
STRING_LABELS = {"一", "二", "三", "四", "五", "六", "七"}
POSITION_HINT_LABELS = {
    "八", "九", "十", "十一", "十二", "十三",
    "半", "徽外", "徽内", "分", "寸",
}
CHINESE_NUMERAL_RE = re.compile(r"^[一二三四五六七八九十百千万零]+$")


# ============================================================
# 几何工具
# ============================================================
def bbox_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    if inter == 0: return 0.0
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def vertical_overlap(a, b):
    olap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    minh = max(1e-6, min(a[3]-a[1], b[3]-b[1]))
    return olap / minh


def cluster_bbox(cluster):
    return [min(d["bbox"][0] for d in cluster), min(d["bbox"][1] for d in cluster),
            max(d["bbox"][2] for d in cluster), max(d["bbox"][3] for d in cluster)]


def median(vals):
    if not vals: return 0.0
    s = sorted(vals); m = len(s) // 2
    return (s[m-1]+s[m])/2 if len(s) % 2 == 0 else s[m]


# ============================================================
# 检测数据预处理
# ============================================================
def prepare_detections(yolo_boxes):
    dets = []
    for raw in yolo_boxes or []:
        if not isinstance(raw, dict): continue
        bbox = raw.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4: continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1: continue
        dets.append({
            "class": str(raw.get("class", "")).strip(),
            "bbox": [x1, y1, x2, y2],
            "conf": float(raw.get("conf", 0.0)),
            "cx": (x1+x2)/2, "cy": (y1+y2)/2,
            "w": x2-x1, "h": y2-y1,
        })
    return dets


# ============================================================
# Union-Find 空间聚类
# ============================================================
def build_clusters(
    detections,
    x_factor=1.4, center_factor=1.8,
    x_min=24.0, center_min=36.0,
    v_overlap_thresh=0.08, iou_thresh=0.08,
):
    """
    合并条件 (满足任一即合并):
      1. same_column: 水平间距 <= 阈值 且 垂直重叠 >= 阈值
      2. overlap: IoU >= 阈值
      3. close_center: 中心距 <= 阈值 且 水平距离不太远
    """
    if len(detections) <= 1:
        return [detections] if detections else []

    parent = list(range(len(detections)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    med_w = median([d["w"] for d in detections])
    med_h = median([d["h"] for d in detections])
    x_thresh = max(x_min, med_w * x_factor)
    c_thresh = max(center_min, math.hypot(med_w, med_h) * center_factor)

    print(f"\n  聚类参数: med_w={med_w:.1f} med_h={med_h:.1f} "
          f"x_thresh={x_thresh:.1f} c_thresh={c_thresh:.1f}")

    merge_count = 0
    for i in range(len(detections)):
        for j in range(i+1, len(detections)):
            a, b = detections[i], detections[j]
            cdx = abs(a["cx"]-b["cx"])
            cdist = math.hypot(cdx, a["cy"]-b["cy"])
            hgap = max(0.0, cdx - (a["w"]+b["w"])/2)
            volap = vertical_overlap(a["bbox"], b["bbox"])
            iou = bbox_iou(a["bbox"], b["bbox"])

            same_col = hgap <= x_thresh and volap >= v_overlap_thresh
            olap_conn = iou >= iou_thresh
            close_c = cdist <= c_thresh and cdx <= x_thresh * 1.5

            if same_col or olap_conn or close_c:
                union(i, j)
                merge_count += 1

    print(f"  共 {merge_count} 次合并")

    grouped = {}
    for idx, det in enumerate(detections):
        grouped.setdefault(find(idx), []).append(det)
    clusters = list(grouped.values())
    clusters.sort(key=lambda g: (min(d["cx"] for d in g), min(d["cy"] for d in g)))
    return clusters


# ============================================================
# 语义字段提取
# ============================================================
def is_position_label(label):
    if label in POSITION_HINT_LABELS: return True
    if any(t in label for t in ("徽", "分", "寸", "外", "内")): return True
    if CHINESE_NUMERAL_RE.fullmatch(label) and label not in STRING_LABELS: return True
    return False


def extract_fields(components):
    ordered = sorted(components, key=lambda c: (c["cy"], c["cx"]))
    fingering = finger = position = string = ""
    extras = []
    for c in ordered:
        lb = c["class"]
        if lb in ACTION_LABELS and not fingering: fingering = lb; continue
        if lb in FINGER_LABELS and not finger: finger = lb; continue
        if lb in STRING_LABELS and not string: string = lb; continue
        if is_position_label(lb) and not position: position = lb; continue
        extras.append(lb)
    if not position:
        for lb in extras:
            if is_position_label(lb): position = lb; break
    if not string:
        for lb in extras:
            if lb in STRING_LABELS: string = lb; break
    return {"fingering": fingering, "finger": finger, "position": position,
            "string": string, "extras": extras}


# ============================================================
# 可视化 (PIL 中文支持)
# ============================================================
COLORS_RGB = [
    (0,200,0), (220,50,50), (50,50,220), (200,200,0), (200,0,200),
    (0,200,200), (255,140,0), (255,105,180), (128,0,128), (0,128,128),
    (128,128,0), (255,69,0), (180,105,255), (148,103,189), (0,191,255),
    (165,42,42), (148,0,211), (50,205,50), (100,149,237), (178,34,34),
]


def _get_font(size=18):
    for name in ("msyh.ttc", "simsun.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except:
            continue
    return ImageFont.load_default()


def draw_yolo_pil(cv2_img, dets, class_names=None):
    """PIL 绘制 YOLO 检测框，中文标签"""
    img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = _get_font(16)
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        label = f"{d['class']} {d['conf']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([tb[0], tb[1]-2, tb[2]+2, tb[3]+2], fill=(255, 0, 0))
        draw.text((x1+1, y1-1), label, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_clusters_pil(cv2_img, clusters, fields_list):
    """PIL 绘制聚类结果，中文标签"""
    img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_sm = _get_font(14)
    font_lg = _get_font(18)

    for gi, (cl, fi) in enumerate(zip(clusters, fields_list)):
        color = COLORS_RGB[gi % len(COLORS_RGB)]
        # 组内每个检测框
        for d in cl:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            draw.text((x1+2, y2-16), d["class"], font=font_sm, fill=color)
        # 外包围框
        gx1, gy1, gx2, gy2 = [int(v) for v in cluster_bbox(cl)]
        draw.rectangle([gx1-3, gy1-3, gx2+3, gy2+3], outline=color, width=3)
        # 组标签
        lbl = f"G{gi+1}"
        parts = []
        if fi["fingering"]: parts.append(fi["fingering"])
        if fi["finger"]: parts.append(fi["finger"])
        if fi["string"]: parts.append(f"{fi['string']}弦")
        if fi["position"]: parts.append(f"{fi['position']}徽")
        if parts: lbl += " " + "/".join(parts)
        tb = draw.textbbox((gx1, gy1-20), lbl, font=font_lg)
        draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2], fill=color)
        draw.text((gx1, gy1-20), lbl, font=font_lg, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# I/O 工具
# ============================================================
def read_image_safe(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def save_image_safe(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(str(path))
        print(f"  -> {path}")


def print_clusters(clusters, fields_list):
    for gi, (cl, fi) in enumerate(zip(clusters, fields_list)):
        members = [d["class"] for d in sorted(cl, key=lambda x:(x["cy"],x["cx"]))]
        print(f"\n  -- Group {gi+1} ({len(cl)}个部件) --")
        print(f"    部件: {members}")
        print(f"    指法={fi['fingering'] or '-'}  手指={fi['finger'] or '-'}  "
              f"弦={fi['string'] or '-'}  徽位={fi['position'] or '-'}")
        if fi["extras"]:
            print(f"    其他: {fi['extras']}")


def save_topology_json(clusters, fields_list, path):
    result = {}
    for gi, (cl, fi) in enumerate(zip(clusters, fields_list)):
        result[f"group_{gi+1}"] = {
            "fingering": fi["fingering"], "finger": fi["finger"],
            "position": fi["position"], "string": fi["string"],
            "group_bbox": cluster_bbox(cl),
            "components": [{"class":d["class"],"bbox":d["bbox"],"conf":round(d["conf"],6)}
                           for d in sorted(cl, key=lambda x:(x["cy"],x["cx"]))],
        }
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  -> {path}")
