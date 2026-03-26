# -*- coding: utf-8 -*-
"""
空间聚类算法独立测试脚本
在 test/ 目录下完成全部测试验证，不修改 backend/pipeline 中的任何代码。

使用方法 (在项目根目录运行):
    python test/test_clustering.py

输出:
    1. 控制台: YOLO 检测结果 + 聚类统计 + 各组结构化字段
    2. test/result_yolo_detections.jpg   - YOLO 原始检测框可视化
    3. test/result_clustered_groups.jpg  - 聚类结果可视化 (不同颜色标注不同组)
    4. test/result_topology.json         - 聚类结构化 JSON
"""

import math
import os
import re
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
TEST_DIR = PROJECT_ROOT / "test"
IMAGE_PATH = TEST_DIR / "testpicture-1.jpg"
WEIGHTS_PATH = BACKEND_DIR / "best.pt"

# 把 backend 加入 sys.path 以便复用 cv_module
sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# 配置参数 (可在此处调节)
# ============================================================
CONF_THRESHOLD = 0.10     # YOLO 置信度阈值 (默认0.25检测太少，降至0.10)
IOU_THRESHOLD  = 0.45     # YOLO NMS IoU 阈值


# ============================================================
# Step 1: YOLO 检测 (直接调用 ultralytics，不依赖 cv_module)
# ============================================================
def run_yolo_detection(image_path: str, weights_path: str) -> List[Dict[str, Any]]:
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    parsed = []
    for r in results:
        names = getattr(r, "names", {})
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for bbox, cls_id, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cls_name = names.get(int(cls_id), str(int(cls_id)))
            parsed.append({
                "class": str(cls_name),
                "bbox": [x1, y1, x2, y2],
                "conf": round(float(conf), 6),
            })
    parsed.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return parsed


# ============================================================
# Step 2: 空间聚类算法 (独立实现，与 topology_module 无关)
# ============================================================

# -- 标签分类体系 --
ACTION_LABELS = {
    "勾", "抹", "挑", "托", "打", "摘", "剔", "历", "轮", "撮",
    "注", "吟", "猱", "绰", "撞", "进复", "退复", "散",
}
FINGER_LABELS = {"大", "食", "中", "名", "跪"}
STRING_LABELS = {"一", "二", "三", "四", "五", "六", "七"}
POSITION_HINT_LABELS = {
    "八", "九", "十", "十一", "十二", "十三",
    "半", "徽外", "徽内", "分", "寸",
}
CHINESE_NUMERAL_RE = re.compile(r"^[一二三四五六七八九十百千万零]+$")


def bbox_iou(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def vertical_overlap(a: List[float], b: List[float]) -> float:
    olap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    minh = max(1e-6, min(a[3]-a[1], b[3]-b[1]))
    return olap / minh


def prepare_detections(yolo_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dets = []
    for raw in yolo_boxes or []:
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append({
            "class": str(raw.get("class", "")).strip(),
            "bbox": [x1, y1, x2, y2],
            "conf": float(raw.get("conf", 0.0)),
            "cx": (x1+x2)/2, "cy": (y1+y2)/2,
            "w": x2-x1, "h": y2-y1,
        })
    return dets


def median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    m = len(s) // 2
    return (s[m-1]+s[m])/2 if len(s) % 2 == 0 else s[m]


def build_clusters(
    detections: List[Dict[str, Any]],
    x_factor: float = 1.4,
    center_factor: float = 1.8,
    x_min: float = 24.0,
    center_min: float = 36.0,
    v_overlap_thresh: float = 0.08,
    iou_thresh: float = 0.08,
) -> List[List[Dict[str, Any]]]:
    """
    Union-Find 空间聚类。

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
        if ra != rb:
            parent[rb] = ra

    med_w = median([d["w"] for d in detections])
    med_h = median([d["h"] for d in detections])
    x_thresh = max(x_min, med_w * x_factor)
    c_thresh = max(center_min, math.hypot(med_w, med_h) * center_factor)

    print(f"\n{'='*60}")
    print(f"  聚类参数:")
    print(f"    中位宽度={med_w:.1f}px  中位高度={med_h:.1f}px")
    print(f"    水平距离阈值={x_thresh:.1f}px  中心距离阈值={c_thresh:.1f}px")
    print(f"    垂直重叠阈值={v_overlap_thresh}  IoU阈值={iou_thresh}")
    print(f"{'='*60}")

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
                reasons = []
                if same_col: reasons.append(f"col(gap={hgap:.0f},vo={volap:.2f})")
                if olap_conn: reasons.append(f"iou={iou:.2f}")
                if close_c: reasons.append(f"close(d={cdist:.0f})")
                print(f"  合并 [{a['class']}]+[{b['class']}]: {', '.join(reasons)}")

    print(f"  共 {merge_count} 次合并")

    grouped: Dict[int, List] = {}
    for idx, det in enumerate(detections):
        grouped.setdefault(find(idx), []).append(det)
    clusters = list(grouped.values())
    clusters.sort(key=lambda g: (min(d["cx"] for d in g), min(d["cy"] for d in g)))
    return clusters


def is_position_label(label: str) -> bool:
    if label in POSITION_HINT_LABELS:
        return True
    if any(t in label for t in ("徽", "分", "寸", "外", "内")):
        return True
    if CHINESE_NUMERAL_RE.fullmatch(label) and label not in STRING_LABELS:
        return True
    return False


def extract_fields(components: List[Dict[str, Any]]) -> Dict[str, str]:
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


def cluster_bbox(cluster):
    return [min(d["bbox"][0] for d in cluster), min(d["bbox"][1] for d in cluster),
            max(d["bbox"][2] for d in cluster), max(d["bbox"][3] for d in cluster)]


# ============================================================
# Step 3: 可视化
# ============================================================
COLORS = [
    (0,255,0),(255,0,0),(0,0,255),(0,255,255),(255,0,255),
    (255,255,0),(0,165,255),(203,192,255),(128,0,128),(0,128,128),
    (128,128,0),(0,69,255),(180,105,255),(147,20,255),(255,191,0),
    (42,42,165),(211,0,148),(50,205,50),(230,216,173),(34,34,178),
]


def draw_yolo(img, dets):
    canvas = img.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0), 2)
        txt = f"{d['class']} {d['conf']:.2f}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1,y1-th-6), (x1+tw+4,y1), (0,255,0), -1)
        cv2.putText(canvas, txt, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return canvas


def draw_clusters(img, clusters, fields_list):
    canvas = img.copy()
    for gi, (cl, fi) in enumerate(zip(clusters, fields_list)):
        color = COLORS[gi % len(COLORS)]
        for d in cl:
            x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 1)
            cv2.putText(canvas, d["class"], (x1+2,y2-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        gx1,gy1,gx2,gy2 = [int(v) for v in cluster_bbox(cl)]
        cv2.rectangle(canvas, (gx1-3,gy1-3), (gx2+3,gy2+3), color, 2)
        lbl = f"G{gi+1}"
        parts = []
        if fi["fingering"]: parts.append(fi["fingering"])
        if fi["finger"]: parts.append(fi["finger"])
        if fi["string"]: parts.append(f"{fi['string']}弦")
        if fi["position"]: parts.append(f"{fi['position']}徽")
        if parts: lbl += " " + "/".join(parts)
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (gx1-3,gy1-th-10), (gx1+tw+5,gy1-3), color, -1)
        cv2.putText(canvas, lbl, (gx1,gy1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return canvas


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  空间聚类算法 - 独立测试")
    print("=" * 60)

    if not IMAGE_PATH.exists():
        print(f"[错误] 测试图片不存在: {IMAGE_PATH}"); return
    if not WEIGHTS_PATH.exists():
        print(f"[错误] 权重不存在: {WEIGHTS_PATH}"); return

    # -- 1. YOLO --
    print(f"\n[Step 1] YOLO 检测中...")
    print(f"  图片: {IMAGE_PATH}")
    print(f"  权重: {WEIGHTS_PATH}")
    yolo_boxes = run_yolo_detection(str(IMAGE_PATH), str(WEIGHTS_PATH))
    print(f"\n  检测到 {len(yolo_boxes)} 个部件:")
    for i, b in enumerate(yolo_boxes):
        bb = b["bbox"]
        print(f"    [{i:2d}] {b['class']:4s}  conf={b['conf']:.4f}  "
              f"bbox=[{bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}]  "
              f"size={bb[2]-bb[0]:.0f}x{bb[3]-bb[1]:.0f}")

    # -- 2. 聚类 --
    print(f"\n[Step 2] 空间聚类中...")
    dets = prepare_detections(yolo_boxes)
    clusters = build_clusters(dets)
    print(f"\n  聚类为 {len(clusters)} 个组:")
    fields_list = []
    for gi, cl in enumerate(clusters):
        fi = extract_fields(cl)
        fields_list.append(fi)
        members = [d["class"] for d in sorted(cl, key=lambda x:(x["cy"],x["cx"]))]
        print(f"\n  -- Group {gi+1} ({len(cl)}个部件) --")
        print(f"    部件: {members}")
        print(f"    指法={fi['fingering'] or '-'}  手指={fi['finger'] or '-'}  "
              f"弦={fi['string'] or '-'}  徽位={fi['position'] or '-'}")
        if fi["extras"]:
            print(f"    其他: {fi['extras']}")

    # -- 3. 可视化 --
    print(f"\n[Step 3] 生成可视化...")
    img = cv2.imdecode(np.fromfile(str(IMAGE_PATH), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("[错误] 无法读取图片"); return

    out1 = str(TEST_DIR / "result_yolo_detections.jpg")
    cv2.imencode('.jpg', draw_yolo(img, dets))[1].tofile(out1)
    print(f"  -> {out1}")

    out2 = str(TEST_DIR / "result_clustered_groups.jpg")
    cv2.imencode('.jpg', draw_clusters(img, clusters, fields_list))[1].tofile(out2)
    print(f"  -> {out2}")

    # -- 4. JSON --
    result = {}
    for gi, (cl, fi) in enumerate(zip(clusters, fields_list)):
        result[f"group_{gi+1}"] = {
            "fingering": fi["fingering"], "finger": fi["finger"],
            "position": fi["position"], "string": fi["string"],
            "group_bbox": cluster_bbox(cl),
            "components": [{"class":d["class"],"bbox":d["bbox"],"conf":round(d["conf"],6)}
                           for d in sorted(cl, key=lambda x:(x["cy"],x["cx"]))],
        }
    out3 = str(TEST_DIR / "result_topology.json")
    with open(out3, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  -> {out3}")

    print(f"\n{'='*60}")
    print(f"  完成! 检测 {len(yolo_boxes)} 部件 -> 聚类 {len(clusters)} 组")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
