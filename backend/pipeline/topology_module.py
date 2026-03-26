import os
import re
from typing import Any, Dict, List, Optional, Tuple

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
    "擘",
    "泛起",
    "泛止",
    "滚",
    "拂",
    "搯",
    "唤",
    "勾剔",
    "如一",
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
FINGER_ALIASES = {
    "大指": "大",
    "食指": "食",
    "中指": "中",
    "名指": "名",
}

NUMBER_CLASS_IDS = set(range(0, 10))
RIGHT_HAND_CLASS_IDS = set(range(10, 35))
LEFT_HAND_CLASS_IDS = set(range(36, 58)) | {62, 63}
LEFT_FINGER_CLASS_IDS = {58, 59, 60, 61}
SECTION_START_CLASS_ID = 64
SECTION_END_CLASS_ID = 65


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


def _bbox_intersection_over_min_area(box_a: List[float], box_b: List[float]) -> float:
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
    min_area = min(area_a, area_b)
    if min_area <= 0:
        return 0.0
    return inter_area / min_area


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    return _safe_float(raw, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    return _safe_int(raw, default)


def _normalize_label(label: str) -> str:
    return FINGER_ALIASES.get(label, label)


def _parse_class_id(raw_box: Dict[str, Any], label: str) -> Optional[int]:
    for key in ("class_id", "cls_id", "id"):
        if key not in raw_box:
            continue
        value = raw_box.get(key)
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue

    text = str(label).strip()
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _resolve_role_by_class_id(class_id: Optional[int]) -> str:
    if class_id is None:
        return "unknown"
    if class_id in NUMBER_CLASS_IDS:
        return "number"
    if class_id in RIGHT_HAND_CLASS_IDS:
        return "right_hand"
    if class_id in LEFT_HAND_CLASS_IDS:
        return "left_hand"
    if class_id in LEFT_FINGER_CLASS_IDS:
        return "left_finger"
    if class_id == SECTION_START_CLASS_ID:
        return "section_start"
    if class_id == SECTION_END_CLASS_ID:
        return "section_end"
    return "unknown"


def _infer_role_from_label(norm_label: str) -> str:
    if CHINESE_NUMERAL_PATTERN.fullmatch(norm_label):
        return "number"
    if norm_label in FINGER_LABELS:
        return "left_finger"
    if norm_label in ACTION_LABELS:
        return "technique_unknown"
    return "unknown"


def _resolve_component_role(class_id: Optional[int], norm_label: str) -> str:
    role = _resolve_role_by_class_id(class_id)
    if role != "unknown":
        return role
    return _infer_role_from_label(norm_label)


def _deduplicate_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去除高度重叠重复框，尽量避免同一框被复用进多个部件。
    """
    if len(detections) <= 1:
        return detections

    duplicate_iou_threshold = _env_float("TOPOLOGY_DUPLICATE_IOU_THRESHOLD", 0.88)
    duplicate_nested_threshold = _env_float("TOPOLOGY_DUPLICATE_IO_MIN_THRESHOLD", 0.92)
    cross_class_iou_threshold = _env_float("TOPOLOGY_DUPLICATE_CROSS_CLASS_IOU_THRESHOLD", 0.93)
    cross_class_nested_threshold = _env_float("TOPOLOGY_DUPLICATE_CROSS_CLASS_IO_MIN_THRESHOLD", 0.95)
    center_ratio = _env_float("TOPOLOGY_DUPLICATE_CENTER_RATIO", 0.22)

    ordered = sorted(detections, key=lambda item: item["conf"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for det in ordered:
        is_duplicate = False
        for exists in kept:
            iou = _bbox_iou(det["bbox"], exists["bbox"])
            nested = _bbox_intersection_over_min_area(det["bbox"], exists["bbox"])
            same_class = (
                det.get("class_id") is not None
                and exists.get("class_id") is not None
                and det.get("class_id") == exists.get("class_id")
            ) or det["norm_class"] == exists["norm_class"]

            min_side = max(1.0, min(det["w"], det["h"], exists["w"], exists["h"]))
            center_dist = ((det["cx"] - exists["cx"]) ** 2 + (det["cy"] - exists["cy"]) ** 2) ** 0.5
            near_center = center_dist <= center_ratio * min_side

            if same_class and (iou >= duplicate_iou_threshold or nested >= duplicate_nested_threshold):
                is_duplicate = True
                break

            if near_center and (iou >= cross_class_iou_threshold or nested >= cross_class_nested_threshold):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)

    kept.sort(key=lambda item: (item["cy"], item["cx"]))
    return kept


def _prepare_detections(yolo_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    min_conf = _env_float("TOPOLOGY_MIN_CONF", 0.0)
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

        raw_label = str(raw_box.get("class", "")).strip()
        norm_label = _normalize_label(raw_label)
        class_id = _parse_class_id(raw_box, raw_label)
        role = _resolve_component_role(class_id, norm_label)
        conf = _safe_float(raw_box.get("conf", 0.0))
        if conf < min_conf:
            continue

        detections.append(
            {
                "class": raw_label,
                "norm_class": norm_label,
                "class_id": class_id,
                "role": role,
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "w": x2 - x1,
                "h": y2 - y1,
            }
        )

    return _deduplicate_detections(detections)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2.0
    return ordered[mid]


def _cluster_bbox_from_indices(
    root_a: int,
    root_b: int,
    root_bbox: List[List[float]],
) -> List[float]:
    bbox_a = root_bbox[root_a]
    bbox_b = root_bbox[root_b]
    return [
        min(bbox_a[0], bbox_b[0]),
        min(bbox_a[1], bbox_b[1]),
        max(bbox_a[2], bbox_b[2]),
        max(bbox_a[3], bbox_b[3]),
    ]


def _center_of_bbox(bbox: List[float]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _pair_relation(
    det_a: Dict[str, Any],
    det_b: Dict[str, Any],
    neighbor_x_threshold: float,
    neighbor_y_threshold: float,
    overlap_iou_threshold: float,
    nested_overlap_threshold: float,
    max_group_width: float,
    max_group_height: float,
) -> Optional[Tuple[float, bool]]:
    dx = abs(det_a["cx"] - det_b["cx"])
    dy = abs(det_a["cy"] - det_b["cy"])
    if dx > max_group_width or dy > max_group_height:
        return None

    overlap_iou = _bbox_iou(det_a["bbox"], det_b["bbox"])
    nested_overlap = _bbox_intersection_over_min_area(det_a["bbox"], det_b["bbox"])

    adjacent = dx <= neighbor_x_threshold and dy <= neighbor_y_threshold
    overlap_connected = overlap_iou >= overlap_iou_threshold
    nested_connected = nested_overlap >= nested_overlap_threshold
    if not (adjacent or overlap_connected or nested_connected):
        return None

    proximity_score = 1.0 - min(
        1.0,
        max(
            dx / max(neighbor_x_threshold, 1e-6),
            dy / max(neighbor_y_threshold, 1e-6),
        ),
    )
    relation_score = max(
        0.0,
        proximity_score,
        overlap_iou * 1.2,
        nested_overlap * 1.4,
    )
    return relation_score, (overlap_connected or nested_connected)


def _component_bbox(components: List[Dict[str, Any]]) -> List[float]:
    return [
        min(item["bbox"][0] for item in components),
        min(item["bbox"][1] for item in components),
        max(item["bbox"][2] for item in components),
        max(item["bbox"][3] for item in components),
    ]


def _reference_y_for_numbers(components: List[Dict[str, Any]]) -> float:
    technique_like_roles = {"right_hand", "left_hand", "left_finger", "technique_unknown"}
    anchors = [item["cy"] for item in components if item["role"] in technique_like_roles]
    if anchors:
        return sum(anchors) / len(anchors)
    group_bbox = _component_bbox(components)
    return (group_bbox[1] + group_bbox[3]) / 2.0


def _validate_group_semantics(components: List[Dict[str, Any]], median_h: float) -> bool:
    if not components:
        return True

    right_count = sum(1 for item in components if item["role"] == "right_hand")
    left_count = sum(1 for item in components if item["role"] == "left_hand")
    left_finger_count = sum(1 for item in components if item["role"] == "left_finger")
    section_start_count = sum(1 for item in components if item["role"] == "section_start")
    section_end_count = sum(1 for item in components if item["role"] == "section_end")
    marker_count = section_start_count + section_end_count

    # 乐章起止符号必须单独存在
    if marker_count > 0:
        return marker_count == 1 and len(components) == 1

    # 左右手指法最多各 1，左手手指最多 1
    if right_count > 1 or left_count > 1 or left_finger_count > 1:
        return False

    numbers = [item for item in components if item["role"] == "number"]
    if not numbers:
        return True
    if len(numbers) > 3:
        return False

    reference_y = _reference_y_for_numbers(components)
    middle_band = max(2.0, median_h * _env_float("TOPOLOGY_NUMBER_MIDDLE_BAND_FACTOR", 0.25))
    top_count = 0
    bottom_count = 0
    middle_count = 0

    for item in numbers:
        if item["cy"] < reference_y - middle_band:
            top_count += 1
        elif item["cy"] > reference_y + middle_band:
            bottom_count += 1
        else:
            middle_count += 1

    if top_count > 2 or bottom_count > 1:
        return False

    available_capacity = (2 - top_count) + (1 - bottom_count)
    if middle_count > available_capacity:
        return False

    return True


def _build_clusters(detections: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not detections:
        return []
    if len(detections) == 1:
        return [detections]

    median_w = _median([d["w"] for d in detections])
    median_h = _median([d["h"] for d in detections])

    neighbor_x_threshold = max(
        _env_float("TOPOLOGY_NEIGHBOR_X_MIN", 20.0),
        median_w * _env_float("TOPOLOGY_NEIGHBOR_X_FACTOR", 1.05),
    )
    neighbor_y_threshold = max(
        _env_float("TOPOLOGY_NEIGHBOR_Y_MIN", 28.0),
        median_h * _env_float("TOPOLOGY_NEIGHBOR_Y_FACTOR", 1.95),
    )
    max_group_width = max(
        _env_float("TOPOLOGY_MAX_GROUP_WIDTH_MIN", 80.0),
        median_w * _env_float("TOPOLOGY_MAX_GROUP_WIDTH_FACTOR", 3.6),
    )
    max_group_height = max(
        _env_float("TOPOLOGY_MAX_GROUP_HEIGHT_MIN", 92.0),
        median_h * _env_float("TOPOLOGY_MAX_GROUP_HEIGHT_FACTOR", 4.0),
    )
    overlap_iou_threshold = _env_float("TOPOLOGY_OVERLAP_IOU_THRESHOLD", 0.10)
    nested_overlap_threshold = _env_float("TOPOLOGY_NESTED_IO_MIN_THRESHOLD", 0.58)
    max_components_per_group = max(1, _env_int("TOPOLOGY_MAX_COMPONENTS_PER_GROUP", 10))

    parent = list(range(len(detections)))
    root_size = [1] * len(detections)
    root_bbox = [det["bbox"][:] for det in detections]
    root_members: List[List[int]] = [[idx] for idx in range(len(detections))]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int, edge_has_strong_overlap: bool) -> bool:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False

        if root_size[root_a] + root_size[root_b] > max_components_per_group:
            return False

        merged_bbox = _cluster_bbox_from_indices(root_a, root_b, root_bbox)
        merged_w = merged_bbox[2] - merged_bbox[0]
        merged_h = merged_bbox[3] - merged_bbox[1]
        if merged_w > max_group_width or merged_h > max_group_height:
            return False

        if not edge_has_strong_overlap:
            center_a = _center_of_bbox(root_bbox[root_a])
            center_b = _center_of_bbox(root_bbox[root_b])
            if (
                abs(center_a[0] - center_b[0]) > neighbor_x_threshold * 0.95
                or abs(center_a[1] - center_b[1]) > neighbor_y_threshold * 1.05
            ):
                return False

        merged_member_indices = root_members[root_a] + root_members[root_b]
        merged_components = [detections[idx] for idx in merged_member_indices]
        if not _validate_group_semantics(merged_components, median_h):
            return False

        if root_size[root_a] < root_size[root_b]:
            root_a, root_b = root_b, root_a
            merged_member_indices = root_members[root_a] + root_members[root_b]

        parent[root_b] = root_a
        root_size[root_a] += root_size[root_b]
        root_bbox[root_a] = merged_bbox
        root_members[root_a] = merged_member_indices
        return True

    edges: List[Tuple[float, int, int, bool]] = []
    for i in range(len(detections)):
        det_i = detections[i]
        for j in range(i + 1, len(detections)):
            relation = _pair_relation(
                det_i,
                detections[j],
                neighbor_x_threshold=neighbor_x_threshold,
                neighbor_y_threshold=neighbor_y_threshold,
                overlap_iou_threshold=overlap_iou_threshold,
                nested_overlap_threshold=nested_overlap_threshold,
                max_group_width=max_group_width,
                max_group_height=max_group_height,
            )
            if relation is None:
                continue
            score, has_strong_overlap = relation
            edges.append((score, i, j, has_strong_overlap))

    edges.sort(key=lambda item: item[0], reverse=True)
    for _, i, j, has_strong_overlap in edges:
        union(i, j, has_strong_overlap)

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for idx, det in enumerate(detections):
        root = find(idx)
        grouped.setdefault(root, []).append(det)

    clusters = list(grouped.values())
    clusters.sort(key=lambda items: (min(d["cx"] for d in items), min(d["cy"] for d in items)))
    return clusters


def _pick_highest_conf_label(components: List[Dict[str, Any]], role: str) -> str:
    candidates = [item for item in components if item["role"] == role]
    if not candidates:
        return ""
    best = max(candidates, key=lambda item: item["conf"])
    return best["norm_class"]


def _assign_number_slots(
    components: List[Dict[str, Any]],
    median_h: float,
) -> Tuple[str, str, List[str], str]:
    numbers = [item for item in components if item["role"] == "number"]
    if not numbers:
        return "", "", [], ""

    reference_y = _reference_y_for_numbers(components)
    middle_band = max(2.0, median_h * _env_float("TOPOLOGY_NUMBER_MIDDLE_BAND_FACTOR", 0.25))

    top: List[Dict[str, Any]] = []
    bottom: List[Dict[str, Any]] = []
    middle: List[Dict[str, Any]] = []
    for item in numbers:
        if item["cy"] < reference_y - middle_band:
            top.append(item)
        elif item["cy"] > reference_y + middle_band:
            bottom.append(item)
        else:
            middle.append(item)

    top_slots = max(0, 2 - len(top))
    bottom_slots = max(0, 1 - len(bottom))
    middle_sorted = sorted(middle, key=lambda item: abs(item["cy"] - reference_y))
    for item in middle_sorted:
        if top_slots <= 0 and bottom_slots <= 0:
            break
        prefer_top = item["cy"] <= reference_y
        if prefer_top:
            if top_slots > 0:
                top.append(item)
                top_slots -= 1
            elif bottom_slots > 0:
                bottom.append(item)
                bottom_slots -= 1
        else:
            if bottom_slots > 0:
                bottom.append(item)
                bottom_slots -= 1
            elif top_slots > 0:
                top.append(item)
                top_slots -= 1

    top_sorted = sorted(top, key=lambda item: (item["cx"], item["cy"]))[:2]
    bottom_sorted = sorted(bottom, key=lambda item: (item["cy"], item["cx"]))[:1]

    hui_digits = [item["norm_class"] for item in top_sorted]
    hui = "".join(hui_digits)
    xian = bottom_sorted[0]["norm_class"] if bottom_sorted else ""

    return hui, xian, hui_digits, xian


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
    median_h = _median([item["h"] for item in ordered]) if ordered else 0.0

    is_section_start = any(item["role"] == "section_start" for item in ordered)
    is_section_end = any(item["role"] == "section_end" for item in ordered)
    is_marker = is_section_start or is_section_end
    marker_type = "start" if is_section_start else ("end" if is_section_end else "")

    right_fingering = "" if is_marker else _pick_highest_conf_label(ordered, "right_hand")
    left_fingering = "" if is_marker else _pick_highest_conf_label(ordered, "left_hand")
    left_finger = "" if is_marker else _pick_highest_conf_label(ordered, "left_finger")
    hui, xian, hui_digits, xian_digit = ("", "", [], "")
    if not is_marker:
        hui, xian, hui_digits, xian_digit = _assign_number_slots(ordered, median_h)

    # 兼容旧下游字段
    fingering = right_fingering or left_fingering
    finger = left_finger
    position = hui
    if not position:
        # 回退旧逻辑，避免完全无徽序时丢失已识别的徽位类
        for item in ordered:
            if _is_position_label(item["norm_class"]):
                position = item["norm_class"]
                break
    string = xian
    if not string:
        for item in ordered:
            if item["norm_class"] in STRING_LABELS:
                string = item["norm_class"]
                break

    return {
        "fingering": fingering,
        "finger": finger,
        "position": position,
        "string": string,
        "right_fingering": right_fingering,
        "left_fingering": left_fingering,
        "left_finger": left_finger,
        "hui": hui,
        "xian": xian,
        "hui_digits": hui_digits,
        "xian_digit": xian_digit,
        "is_marker": is_marker,
        "is_section_start": is_section_start,
        "is_section_end": is_section_end,
        "marker_type": marker_type,
    }


def _cluster_bbox(cluster: List[Dict[str, Any]]) -> List[float]:
    return [
        min(item["bbox"][0] for item in cluster),
        min(item["bbox"][1] for item in cluster),
        max(item["bbox"][2] for item in cluster),
        max(item["bbox"][3] for item in cluster),
    ]


def _component_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "class": item["class"],
        "bbox": item["bbox"],
        "conf": round(item["conf"], 6),
        "role": item["role"],
    }
    if item.get("class_id") is not None:
        payload["class_id"] = int(item["class_id"])
    return payload


def build_topology(yolo_boxes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    模块 B：空间拓扑解析模块。
    基于 bbox 的空间关系聚类，并输出结构化减字谱结果。

    约束（用于防误并）：
    - 右手指法最多 1 个
    - 左手指法最多 1 个
    - 左手手指最多 1 个
    - 数字上方(徽序)最多 2，下方(弦序)最多 1
    - 乐章起止符号(64/65)必须独立成组
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
            "components": [_component_payload(item) for item in components],
        }

    return parsed
