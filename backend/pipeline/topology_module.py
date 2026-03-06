import math
import re
from typing import Any, Dict, List

ACTION_LABELS = {
    "勾",
    "抹",
    "挑",
    "托",
    "打",
    "摘",
    "剔",
    "历",
    "轮",
    "撮",
    "注",
    "吟",
    "猱",
    "绰",
    "撞",
    "进复",
    "退复",
    "散",
}
FINGER_LABELS = {"大", "食", "中", "名", "跪"}
STRING_LABELS = {"一", "二", "三", "四", "五", "六", "七"}
POSITION_HINT_LABELS = {
    "八",
    "九",
    "十",
    "十一",
    "十二",
    "十三",
    "半",
    "徽外",
    "徽内",
    "分",
    "寸",
}
CHINESE_NUMERAL_PATTERN = re.compile(r"^[一二三四五六七八九十百千万零〇两]+$")


def _bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _vertical_overlap_ratio(box_a: List[float], box_b: List[float]) -> float:
    ay1, ay2 = box_a[1], box_a[3]
    by1, by2 = box_b[1], box_b[3]
    overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    min_height = max(1e-6, min(ay2 - ay1, by2 - by1))
    return overlap / min_height


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prepare_detections(yolo_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for raw_box in yolo_boxes or []:
        if not isinstance(raw_box, dict):
            continue
        bbox = raw_box.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [_safe_float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(raw_box.get("class", "")).strip()
        conf = _safe_float(raw_box.get("conf", 0.0))
        detections.append(
            {
                "class": label,
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "w": x2 - x1,
                "h": y2 - y1,
            }
        )
    return detections


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2.0
    return ordered[mid]


def _build_clusters(detections: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not detections:
        return []
    if len(detections) == 1:
        return [detections]

    parent = list(range(len(detections)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    median_w = _median([d["w"] for d in detections])
    median_h = _median([d["h"] for d in detections])
    x_distance_threshold = max(24.0, median_w * 1.4)
    center_distance_threshold = max(36.0, math.hypot(median_w, median_h) * 1.8)

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a = detections[i]
            b = detections[j]
            center_dx = abs(a["cx"] - b["cx"])
            center_distance = math.hypot(center_dx, a["cy"] - b["cy"])
            horizontal_gap = max(0.0, center_dx - (a["w"] + b["w"]) / 2.0)
            vertical_overlap = _vertical_overlap_ratio(a["bbox"], b["bbox"])
            overlap_iou = _bbox_iou(a["bbox"], b["bbox"])

            same_column = horizontal_gap <= x_distance_threshold and vertical_overlap >= 0.08
            overlap_connected = overlap_iou >= 0.08
            close_center = (
                center_distance <= center_distance_threshold
                and center_dx <= x_distance_threshold * 1.5
            )

            if same_column or overlap_connected or close_center:
                union(i, j)

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for idx, det in enumerate(detections):
        root = find(idx)
        grouped.setdefault(root, []).append(det)

    clusters = list(grouped.values())
    clusters.sort(key=lambda items: (min(d["cx"] for d in items), min(d["cy"] for d in items)))
    return clusters


def _is_position_label(label: str) -> bool:
    if label in POSITION_HINT_LABELS:
        return True
    if any(token in label for token in ("徽", "分", "寸", "外", "内")):
        return True
    if CHINESE_NUMERAL_PATTERN.fullmatch(label) and label not in STRING_LABELS:
        return True
    return False


def _extract_group_fields(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(components, key=lambda item: (item["cy"], item["cx"]))
    fingering = ""
    finger = ""
    position = ""
    string = ""
    extras: List[str] = []

    for comp in ordered:
        label = comp["class"]
        if label in ACTION_LABELS and not fingering:
            fingering = label
            continue
        if label in FINGER_LABELS and not finger:
            finger = label
            continue
        if label in STRING_LABELS and not string:
            string = label
            continue
        if _is_position_label(label) and not position:
            position = label
            continue
        extras.append(label)

    if not position:
        for label in extras:
            if _is_position_label(label):
                position = label
                break

    if not string:
        for label in extras:
            if label in STRING_LABELS:
                string = label
                break

    return {
        "fingering": fingering,
        "finger": finger,
        "position": position,
        "string": string,
    }


def _cluster_bbox(cluster: List[Dict[str, Any]]) -> List[float]:
    return [
        min(item["bbox"][0] for item in cluster),
        min(item["bbox"][1] for item in cluster),
        max(item["bbox"][2] for item in cluster),
        max(item["bbox"][3] for item in cluster),
    ]


def build_topology(yolo_boxes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    模块 B：空间拓扑解析模块。
    基于 bbox 的空间关系聚类，并输出结构化的减字谱组装结果。
    """
    detections = _prepare_detections(yolo_boxes)
    clusters = _build_clusters(detections)

    parsed: Dict[str, Dict[str, Any]] = {}
    for idx, cluster in enumerate(clusters, start=1):
        components = sorted(cluster, key=lambda item: (item["cy"], item["cx"]))
        core_fields = _extract_group_fields(components)
        parsed[f"group_{idx}"] = {
            **core_fields,
            "group_bbox": _cluster_bbox(cluster),
            "components": [
                {"class": item["class"], "bbox": item["bbox"], "conf": round(item["conf"], 6)}
                for item in components
            ],
        }

    return parsed
