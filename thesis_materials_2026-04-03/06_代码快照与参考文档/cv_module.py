import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    SAHI_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖可选
    SAHI_AVAILABLE = False

try:
    import torch
except Exception:  # pragma: no cover - 依赖可选
    torch = None

LOGGER = logging.getLogger(__name__)
_MODEL = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _backend_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _final_yolo_dir() -> Path:
    return _project_root() / "Yolo"


def _resolve_single_weights_path() -> Path:
    custom_path = os.getenv("YOLO_WEIGHTS_PATH", "").strip()
    if not custom_path:
        return _backend_dir() / "best.pt"

    path_obj = Path(custom_path)
    if not path_obj.is_absolute():
        path_obj = _backend_dir() / path_obj
    return path_obj.resolve()


def _default_recall_first_paths() -> List[Path]:
    yolo_root = _final_yolo_dir()
    weights_root = yolo_root / "weights"
    return [
        weights_root / "y11_v08_kfold_full_results" / "y11_v08_kfold_fold1" / "weights" / "best.pt",
        weights_root / "y11_v08_kfold_full_results" / "y11_v08_kfold_fold2" / "weights" / "best.pt",
        weights_root / "y11_v08_kfold_full_results" / "y11_v08_kfold_fold3" / "weights" / "best.pt",
        weights_root / "y11_v08_kfold_full_results" / "y11_v08_kfold_fold4" / "weights" / "best.pt",
        weights_root / "y11_v08_kfold_full_results" / "y11_v08_kfold_fold5" / "weights" / "best.pt",
        weights_root / "y11_v08_ultimate_2_1_results" / "y11_v08_ultimate_2_1" / "weights" / "best.pt",
        weights_root / "y11_v08_ultimate_23_results" / "y11_v08_ultimate_23" / "weights" / "best.pt",
        weights_root / "y11_v10_results" / "y11_v10_reasonable_aug" / "weights" / "best.pt",
    ]


def _split_paths(raw_text: str) -> List[str]:
    if not raw_text:
        return []
    normalized = raw_text.replace("\n", ";").replace(",", ";")
    return [part.strip() for part in normalized.split(";") if part.strip()]


def _resolve_recall_model_paths() -> List[Path]:
    custom = _split_paths(os.getenv("YOLO_RECALL_MODEL_PATHS", "").strip())
    if custom:
        output: List[Path] = []
        for item in custom:
            p = Path(item)
            if not p.is_absolute():
                p = _project_root() / p
            output.append(p.resolve())
        return output
    return [p.resolve() for p in _default_recall_first_paths()]


def _load_single_model() -> YOLO:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    weights_path = _resolve_single_weights_path()
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO 权重文件不存在: {weights_path}")

    _MODEL = YOLO(str(weights_path))
    return _MODEL


def _tensor_to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _resolve_class_name(names: Any, class_id: float) -> str:
    class_index = int(class_id)
    if isinstance(names, dict):
        return str(names.get(class_index, names.get(str(class_index), class_index)))
    if isinstance(names, (list, tuple)) and 0 <= class_index < len(names):
        return str(names[class_index])
    return str(class_index)


def _normalize_class_id(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _weighted_bbox(cluster: List[Dict[str, Any]]) -> List[float]:
    weights = [max(float(item["score"]), 1e-6) for item in cluster]
    w_sum = sum(weights)
    return [
        sum(item["bbox"][0] * w for item, w in zip(cluster, weights)) / w_sum,
        sum(item["bbox"][1] * w for item, w in zip(cluster, weights)) / w_sum,
        sum(item["bbox"][2] * w for item, w in zip(cluster, weights)) / w_sum,
        sum(item["bbox"][3] * w for item, w in zip(cluster, weights)) / w_sum,
    ]


def _cluster_predictions(class_preds: List[Dict[str, Any]], vote_iou_threshold: float) -> List[List[Dict[str, Any]]]:
    clusters: List[List[Dict[str, Any]]] = []
    class_preds = sorted(class_preds, key=lambda item: float(item["score"]), reverse=True)
    used = [False] * len(class_preds)

    for i, pred_i in enumerate(class_preds):
        if used[i]:
            continue
        used[i] = True
        cluster = [pred_i]

        changed = True
        while changed:
            changed = False
            for j, pred_j in enumerate(class_preds):
                if used[j]:
                    continue
                if any(_calculate_iou(pred_j["bbox"], pred_k["bbox"]) >= vote_iou_threshold for pred_k in cluster):
                    used[j] = True
                    cluster.append(pred_j)
                    changed = True
        clusters.append(cluster)
    return clusters


def _classwise_nms(preds: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for pred in preds:
        class_id = _normalize_class_id(pred["class_id"])
        by_class.setdefault(class_id, []).append(pred)

    kept: List[Dict[str, Any]] = []
    for class_id, class_preds in by_class.items():
        sorted_preds = sorted(class_preds, key=lambda item: float(item["score"]), reverse=True)
        selected: List[Dict[str, Any]] = []
        for pred in sorted_preds:
            if any(_calculate_iou(pred["bbox"], exists["bbox"]) > iou_threshold for exists in selected):
                continue
            selected.append(pred)
        kept.extend(selected)
    return kept


def _vote_fuse_predictions(
    all_preds: List[Dict[str, Any]],
    vote_iou_threshold: float,
    min_votes: int,
    single_keep_score: float,
    final_nms_iou: float,
    model_count: int,
) -> List[Dict[str, Any]]:
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for pred in all_preds:
        class_id = _normalize_class_id(pred["class_id"])
        by_class.setdefault(class_id, []).append(pred)

    fused: List[Dict[str, Any]] = []
    for class_id, cls_preds in by_class.items():
        clusters = _cluster_predictions(cls_preds, vote_iou_threshold=vote_iou_threshold)
        for cluster in clusters:
            vote_count = len({int(item["model_idx"]) for item in cluster})
            max_score = max(float(item["score"]) for item in cluster)
            if vote_count < max(1, min_votes) and max_score < single_keep_score:
                continue

            fused.append(
                {
                    "class_id": class_id,
                    "bbox": _weighted_bbox(cluster),
                    "score": max_score,
                    "votes": vote_count,
                }
            )

    return _classwise_nms(fused, iou_threshold=final_nms_iou)


def _resolve_device() -> str:
    explicit = os.getenv("YOLO_DEVICE", "").strip()
    if explicit:
        return explicit
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _resolve_class_name_from_mapping(mapping: Dict[Any, Any], class_id: str) -> str:
    if class_id in mapping:
        return str(mapping[class_id])
    try:
        cid_int = int(class_id)
    except ValueError:
        cid_int = None
    if cid_int is not None and cid_int in mapping:
        return str(mapping[cid_int])
    if cid_int is not None and str(cid_int) in mapping:
        return str(mapping[str(cid_int)])
    return class_id


def _collect_recall_first_predictions(
    image_path: Path,
    model_paths: List[Path],
    device: str,
    confidence_threshold: float,
    slice_size: int,
    overlap_ratio: float,
    postprocess_type: str,
    postprocess_match_metric: str,
    postprocess_match_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[Any, Any]]:
    all_predictions: List[Dict[str, Any]] = []
    category_mapping: Dict[Any, Any] = {}

    for model_idx, model_path in enumerate(model_paths):
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov11",
            model_path=str(model_path),
            confidence_threshold=confidence_threshold,
            device=device,
            image_size=slice_size,
        )
        if not category_mapping:
            category_mapping = detection_model.category_mapping or {}

        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=False,
        )

        for pred in result.object_prediction_list:
            class_id = _normalize_class_id(pred.category.id)
            all_predictions.append(
                {
                    "class_id": class_id,
                    "bbox": list(map(float, pred.bbox.to_xyxy())),
                    "score": float(getattr(pred.score, "value", pred.score)),
                    "model_idx": model_idx,
                }
            )

        del detection_model
        if device.startswith("cuda") and torch is not None:
            torch.cuda.empty_cache()

    return all_predictions, category_mapping


def _detect_recall_first(image_path: Path) -> List[Dict[str, Any]]:
    model_paths = _resolve_recall_model_paths()
    missing = [str(path) for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Recall-first 权重缺失: " + "; ".join(missing))

    device = _resolve_device()
    confidence_threshold = float(os.getenv("YOLO_RECALL_CONFIDENCE_THRESHOLD", "0.10"))
    slice_size = int(os.getenv("YOLO_RECALL_SLICE_SIZE", "1024"))
    overlap_ratio = float(os.getenv("YOLO_RECALL_OVERLAP_RATIO", "0.2"))
    postprocess_type = os.getenv("YOLO_RECALL_POSTPROCESS_TYPE", "NMS").strip() or "NMS"
    postprocess_match_metric = os.getenv("YOLO_RECALL_POSTPROCESS_MATCH_METRIC", "IOU").strip() or "IOU"
    postprocess_match_threshold = float(os.getenv("YOLO_RECALL_POSTPROCESS_MATCH_THRESHOLD", "0.7"))

    vote_iou_threshold = float(os.getenv("YOLO_RECALL_VOTE_IOU_THRESHOLD", "0.6"))
    min_votes = int(os.getenv("YOLO_RECALL_MIN_VOTES", "2"))
    single_keep_score = float(os.getenv("YOLO_RECALL_SINGLE_KEEP_SCORE", "0.1"))
    final_nms_iou = float(os.getenv("YOLO_RECALL_FINAL_NMS_IOU", "0.55"))

    all_preds, category_mapping = _collect_recall_first_predictions(
        image_path=image_path,
        model_paths=model_paths,
        device=device,
        confidence_threshold=confidence_threshold,
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
        postprocess_type=postprocess_type,
        postprocess_match_metric=postprocess_match_metric,
        postprocess_match_threshold=postprocess_match_threshold,
    )
    fused = _vote_fuse_predictions(
        all_preds=all_preds,
        vote_iou_threshold=vote_iou_threshold,
        min_votes=min_votes,
        single_keep_score=single_keep_score,
        final_nms_iou=final_nms_iou,
        model_count=len(model_paths),
    )

    parsed_results: List[Dict[str, Any]] = []
    for item in fused:
        class_id = _normalize_class_id(item["class_id"])
        class_name = _resolve_class_name_from_mapping(category_mapping, class_id)
        x1, y1, x2, y2 = [float(v) for v in item["bbox"]]
        class_id_int: Any = None
        try:
            class_id_int = int(float(class_id))
        except (TypeError, ValueError):
            class_id_int = None

        payload: Dict[str, Any] = {
            "class": class_name,
            "bbox": [x1, y1, x2, y2],
            "conf": round(float(item["score"]), 6),
            "votes": int(item.get("votes", 1)),
        }
        if class_id_int is not None:
            payload["class_id"] = class_id_int

        parsed_results.append(
            payload
        )

    parsed_results.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return parsed_results


def _detect_single_model(image_path: Path) -> List[Dict[str, Any]]:
    model = _load_single_model()
    conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
    iou_threshold = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    device = _resolve_device()

    predict_kwargs: Dict[str, Any] = {
        "source": str(image_path),
        "conf": conf_threshold,
        "iou": iou_threshold,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    parsed_results: List[Dict[str, Any]] = []

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        xyxy_list = _tensor_to_list(getattr(boxes, "xyxy", None))
        cls_list = _tensor_to_list(getattr(boxes, "cls", None))
        conf_list = _tensor_to_list(getattr(boxes, "conf", None))

        if not conf_list:
            conf_list = [1.0] * len(xyxy_list)
        if not cls_list:
            cls_list = [0.0] * len(xyxy_list)

        for bbox, cls_id, conf in zip(xyxy_list, cls_list, conf_list):
            conf_float = float(conf)
            if conf_float < conf_threshold:
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            class_id_int: Any = None
            try:
                class_id_int = int(float(cls_id))
            except (TypeError, ValueError):
                class_id_int = None

            payload: Dict[str, Any] = {
                "class": _resolve_class_name(getattr(result, "names", {}), cls_id),
                "bbox": [x1, y1, x2, y2],
                "conf": round(conf_float, 6),
            }
            if class_id_int is not None:
                payload["class_id"] = class_id_int

            parsed_results.append(
                payload
            )

    parsed_results.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return parsed_results


def detect_components(image_path: str) -> List[Dict[str, Any]]:
    """
    模块 A：视觉感知模块。
    优先按 Yolo/README 的召回优先协议执行 8 模型 SAHI+投票融合。
    如依赖或权重缺失，则自动降级为单模型推理。
    """
    image_obj = Path(image_path)
    if not image_obj.exists():
        raise FileNotFoundError(f"输入图像不存在: {image_path}")

    infer_mode = os.getenv("YOLO_INFER_MODE", "recall_first").strip().lower()
    hard_fail_on_recall = os.getenv("YOLO_STRICT_RECALL_FIRST", "0").strip() == "1"

    if infer_mode == "single":
        return _detect_single_model(image_obj)

    if infer_mode == "recall_first":
        if not SAHI_AVAILABLE:
            if hard_fail_on_recall:
                raise RuntimeError("未安装 SAHI，无法执行 recall_first 模式。")
            LOGGER.warning("未安装 SAHI，回退至单模型模式。")
            return _detect_single_model(image_obj)

        try:
            return _detect_recall_first(image_obj)
        except Exception as exc:
            if hard_fail_on_recall:
                raise
            LOGGER.warning("recall_first 推理失败，回退单模型: %s", exc)
            return _detect_single_model(image_obj)

    raise ValueError(f"不支持的 YOLO_INFER_MODE: {infer_mode}")
