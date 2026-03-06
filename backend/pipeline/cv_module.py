import os
from pathlib import Path
from typing import Any, Dict, List

from ultralytics import YOLO

_MODEL = None


def _backend_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_weights_path() -> Path:
    custom_path = os.getenv("YOLO_WEIGHTS_PATH", "").strip()
    if not custom_path:
        return _backend_dir() / "best.pt"

    path_obj = Path(custom_path)
    if not path_obj.is_absolute():
        path_obj = _backend_dir() / path_obj
    return path_obj.resolve()


def _load_model() -> YOLO:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    weights_path = _resolve_weights_path()
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
        return str(names.get(class_index, class_index))
    if isinstance(names, (list, tuple)) and 0 <= class_index < len(names):
        return str(names[class_index])
    return str(class_index)


def detect_components(image_path: str) -> List[Dict[str, Any]]:
    """
    模块 A：视觉感知模块。
    使用本地 YOLO 权重文件对上传图像进行推理，并返回统一的 bbox 结构。
    """
    image_obj = Path(image_path)
    if not image_obj.exists():
        raise FileNotFoundError(f"输入图像不存在: {image_path}")

    model = _load_model()
    conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
    iou_threshold = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    device = os.getenv("YOLO_DEVICE", "").strip()

    predict_kwargs: Dict[str, Any] = {
        "source": str(image_obj),
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
            parsed_results.append(
                {
                    "class": _resolve_class_name(getattr(result, "names", {}), cls_id),
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf_float, 6),
                }
            )

    parsed_results.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return parsed_results
