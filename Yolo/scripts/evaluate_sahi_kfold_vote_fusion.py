from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def read_image_cn(img_path: Path) -> np.ndarray:
    data = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"OpenCV failed to read image: {img_path}")
    return img


def normalize_class_id(class_id) -> str:
    try:
        return str(int(float(class_id)))
    except (TypeError, ValueError):
        return str(class_id)


def parse_class_ids_text(text: str) -> Set[str]:
    items = [x.strip() for x in str(text).replace("\n", ",").split(",") if x.strip()]
    return {normalize_class_id(x) for x in items}


def load_allowed_class_ids(allowed_ids: str, allowed_ids_file: str, auto_from_data_root: str) -> Set[str]:
    result: Set[str] = set()

    if allowed_ids:
        result |= parse_class_ids_text(allowed_ids)

    if allowed_ids_file:
        p = Path(allowed_ids_file)
        if not p.exists():
            raise FileNotFoundError(f"allowed class ids file not found: {p}")
        txt = p.read_text(encoding="utf-8", errors="ignore")
        result |= parse_class_ids_text(txt)

    if auto_from_data_root:
        data_root = Path(auto_from_data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"data root not found: {data_root}")
        for txt_file in data_root.rglob("*.txt"):
            if txt_file.name.lower() == "classes.txt":
                continue
            try:
                content = txt_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                result.add(normalize_class_id(parts[0]))

    return result


def load_classwise_params(file_path: str) -> Dict[str, Dict]:
    if not file_path:
        return {}
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"classwise params file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "class_params" in obj:
        raw = obj.get("class_params", {})
    else:
        raw = obj
    if not isinstance(raw, dict):
        raise ValueError("classwise params must be a dict or contain key 'class_params'")

    parsed: Dict[str, Dict] = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        cid = normalize_class_id(k)
        item = {}
        if "min_votes" in v:
            item["min_votes"] = int(v["min_votes"])
        if "single_keep_score" in v:
            item["single_keep_score"] = float(v["single_keep_score"])
        if "final_nms_iou" in v:
            item["final_nms_iou"] = float(v["final_nms_iou"])
        if "vote_iou_threshold" in v:
            item["vote_iou_threshold"] = float(v["vote_iou_threshold"])
        parsed[cid] = item
    return parsed


def load_class_model_weights(file_path: str) -> Dict[str, Dict[int, float]]:
    if not file_path:
        return {}
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"class-model-weights file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    raw = obj.get("class_model_weights", obj) if isinstance(obj, dict) else {}
    if not isinstance(raw, dict):
        raise ValueError("class-model-weights must be a dict or contain key 'class_model_weights'")

    out: Dict[str, Dict[int, float]] = {}
    for class_id, model_map in raw.items():
        cid = normalize_class_id(class_id)
        if not isinstance(model_map, dict):
            continue
        cls_out: Dict[int, float] = {}
        for k, v in model_map.items():
            try:
                midx = int(k)
                w = float(v)
            except (TypeError, ValueError):
                continue
            cls_out[midx] = max(0.0, w)
        if cls_out:
            out[cid] = cls_out
    return out


def get_class_param(class_params: Dict[str, Dict], class_id: str, key: str, default_value):
    if not class_params:
        return default_value
    cid = normalize_class_id(class_id)
    if cid in class_params and key in class_params[cid]:
        return class_params[cid][key]
    return default_value


def classwise_nms(preds: List[Dict], iou_threshold: float, class_params: Dict[str, Dict] | None = None) -> List[Dict]:
    by_class: Dict[str, List[Dict]] = {}
    for p in preds:
        by_class.setdefault(normalize_class_id(p["class_id"]), []).append(p)

    kept: List[Dict] = []
    for class_id, cls_preds in by_class.items():
        class_iou_threshold = float(get_class_param(class_params or {}, class_id, "final_nms_iou", iou_threshold))
        cls_preds = sorted(cls_preds, key=lambda x: x["score"], reverse=True)
        selected: List[Dict] = []
        for p in cls_preds:
            drop = False
            for q in selected:
                if calculate_iou(p["bbox"], q["bbox"]) > class_iou_threshold:
                    drop = True
                    break
            if not drop:
                selected.append(p)
        kept.extend(selected)
    return kept


def get_model_weight(
    class_id: str,
    model_idx: int,
    class_model_weights: Dict[str, Dict[int, float]] | None,
) -> float:
    if not class_model_weights:
        return 1.0
    cls = class_model_weights.get(normalize_class_id(class_id), {})
    return float(cls.get(int(model_idx), 1.0))


def weighted_bbox(
    cluster: List[Dict],
    class_id: str,
    class_model_weights: Dict[str, Dict[int, float]] | None = None,
) -> List[float]:
    weights = [
        max(float(p["score"]) * get_model_weight(class_id, int(p["model_idx"]), class_model_weights), 1e-6)
        for p in cluster
    ]
    w_sum = float(sum(weights))
    x1 = sum(p["bbox"][0] * w for p, w in zip(cluster, weights)) / w_sum
    y1 = sum(p["bbox"][1] * w for p, w in zip(cluster, weights)) / w_sum
    x2 = sum(p["bbox"][2] * w for p, w in zip(cluster, weights)) / w_sum
    y2 = sum(p["bbox"][3] * w for p, w in zip(cluster, weights)) / w_sum
    return [x1, y1, x2, y2]


def weighted_score(
    cluster: List[Dict],
    class_id: str,
    class_model_weights: Dict[str, Dict[int, float]] | None = None,
) -> float:
    w = [max(get_model_weight(class_id, int(p["model_idx"]), class_model_weights), 1e-6) for p in cluster]
    s = [float(p["score"]) for p in cluster]
    return float(sum(si * wi for si, wi in zip(s, w)) / max(sum(w), 1e-6))


def cluster_predictions(class_preds: List[Dict], vote_iou_threshold: float) -> List[List[Dict]]:
    clusters: List[List[Dict]] = []
    used = [False] * len(class_preds)
    class_preds = sorted(class_preds, key=lambda x: x["score"], reverse=True)

    for i in range(len(class_preds)):
        if used[i]:
            continue
        used[i] = True
        cluster = [class_preds[i]]

        changed = True
        while changed:
            changed = False
            for j in range(len(class_preds)):
                if used[j]:
                    continue
                pred_j = class_preds[j]
                if any(calculate_iou(pred_j["bbox"], pred_k["bbox"]) >= vote_iou_threshold for pred_k in cluster):
                    used[j] = True
                    cluster.append(pred_j)
                    changed = True

        clusters.append(cluster)
    return clusters


def vote_fuse_predictions(
    all_preds: List[Dict],
    vote_iou_threshold: float,
    min_votes: int,
    single_keep_score: float,
    final_nms_iou: float,
    class_params: Dict[str, Dict] | None = None,
    fuse_mode: str = "vote",
    model_count: int | None = None,
    allowed_class_ids: Set[str] | None = None,
    class_model_weights: Dict[str, Dict[int, float]] | None = None,
) -> List[Dict]:
    by_class: Dict[str, List[Dict]] = {}
    for p in all_preds:
        cid = normalize_class_id(p["class_id"])
        if allowed_class_ids and cid not in allowed_class_ids:
            continue
        by_class.setdefault(cid, []).append(p)

    fused: List[Dict] = []
    for class_id, cls_preds in by_class.items():
        class_vote_iou = float(get_class_param(class_params or {}, class_id, "vote_iou_threshold", vote_iou_threshold))
        class_min_votes = int(get_class_param(class_params or {}, class_id, "min_votes", min_votes))
        class_keep_score = float(get_class_param(class_params or {}, class_id, "single_keep_score", single_keep_score))
        if model_count is not None:
            class_min_votes = max(1, min(class_min_votes, model_count))

        clusters = cluster_predictions(cls_preds, class_vote_iou)
        for cluster in clusters:
            model_votes = len({int(p["model_idx"]) for p in cluster})
            max_score = float(max(p["score"] for p in cluster))
            mean_score = float(sum(p["score"] for p in cluster) / len(cluster))
            keep = model_votes >= class_min_votes or (class_keep_score > 0 and max_score >= class_keep_score)
            if not keep:
                continue

            if fuse_mode == "wbf":
                # WBF: 使用分数加权框坐标 + 均值分数，更偏稳健框回归
                out_bbox = weighted_bbox(cluster, class_id=class_id, class_model_weights=class_model_weights)
                out_score = weighted_score(cluster, class_id=class_id, class_model_weights=class_model_weights)
            else:
                # vote(legacy): 保留旧行为（加权框 + 最大分数）
                out_bbox = weighted_bbox(cluster, class_id=class_id, class_model_weights=class_model_weights)
                out_score = max_score

            fused.append(
                {
                    "class_id": class_id,
                    "bbox": out_bbox,
                    "score": out_score,
                    "mean_score": mean_score,
                    "votes": model_votes,
                }
            )

    return classwise_nms(fused, final_nms_iou, class_params=class_params)


def parse_folds(folds_text: str) -> List[int]:
    folds = [int(x.strip()) for x in folds_text.split(",") if x.strip()]
    if not folds:
        raise ValueError("No folds parsed from --folds")
    return folds


def get_class_name(category_mapping: Dict, class_id: str) -> str:
    if class_id in category_mapping:
        return str(category_mapping[class_id])
    if class_id.isdigit():
        int_key = int(class_id)
        if int_key in category_mapping:
            return str(category_mapping[int_key])
    return class_id


def score_value(score_obj) -> float:
    return float(getattr(score_obj, "value", score_obj))


def collect_all_predictions(
    model_paths: List[Path],
    image_paths: List[Path],
    args: argparse.Namespace,
    device: str,
    allowed_class_ids: Set[str] | None = None,
) -> Tuple[Dict[str, List[Dict]], Dict]:
    preds_by_image: Dict[str, List[Dict]] = {str(p.resolve()): [] for p in image_paths}
    category_mapping: Dict = {}

    for model_idx, model_path in enumerate(model_paths):
        print(f"[Collect] model {model_idx + 1}/{len(model_paths)}: {model_path}")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov11",
            model_path=str(model_path),
            confidence_threshold=args.confidence_threshold,
            device=device,
            image_size=args.slice_size,
        )
        if not category_mapping:
            category_mapping = detection_model.category_mapping or {}

        for img_path in tqdm(image_paths, desc=f"model{model_idx + 1}", leave=False):
            result = get_sliced_prediction(
                str(img_path),
                detection_model,
                slice_height=args.slice_size,
                slice_width=args.slice_size,
                overlap_height_ratio=args.overlap_ratio,
                overlap_width_ratio=args.overlap_ratio,
                postprocess_type=args.postprocess_type,
                postprocess_match_metric=args.postprocess_match_metric,
                postprocess_match_threshold=args.postprocess_match_threshold,
                postprocess_class_agnostic=args.postprocess_class_agnostic,
            )
            image_key = str(img_path.resolve())
            item_list = preds_by_image[image_key]
            for pred in result.object_prediction_list:
                class_id = normalize_class_id(pred.category.id)
                if allowed_class_ids and class_id not in allowed_class_ids:
                    continue
                item_list.append(
                    {
                        "class_id": class_id,
                        "bbox": list(map(float, pred.bbox.to_xyxy())),
                        "score": score_value(pred.score),
                        "model_idx": model_idx,
                    }
                )

        del detection_model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return preds_by_image, category_mapping


def load_gt_boxes(label_path: Path, width: int, height: int, allowed_class_ids: Set[str] | None = None) -> List[Dict]:
    gt_boxes: List[Dict] = []
    if not label_path.exists():
        return gt_boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = str(int(float(parts[0])))
            if allowed_class_ids and class_id not in allowed_class_ids:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            gt_boxes.append({"class_id": class_id, "bbox": [x1, y1, x2, y2], "matched": False})
    return gt_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 5-fold best.pt with vote fusion on Golden test.")
    parser.add_argument("--kfold-root", type=Path, default=Path("./结果/y11_v08_kfold_full_results"))
    parser.add_argument("--folds", type=str, default="1,2,3,4,5")
    parser.add_argument(
        "--extra-model",
        action="append",
        default=[],
        help="额外模型权重路径（可重复传入多次）。例如: --extra-model path1 --extra-model path2",
    )
    parser.add_argument("--images-dir", type=Path, default=Path("./Golden_Test_Patches/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("./Golden_Test_Patches/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("./Golden_Test_Patches/sahi_results_kfold_vote"))

    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--slice-size", type=int, default=1024)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--postprocess-type", type=str, default="NMS")
    parser.add_argument("--postprocess-match-metric", type=str, default="IOU")
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.7)
    parser.add_argument("--postprocess-class-agnostic", action="store_true")

    parser.add_argument("--vote-iou-threshold", type=float, default=0.55)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--single-keep-score", type=float, default=0.0)
    parser.add_argument("--final-nms-iou", type=float, default=0.50)
    parser.add_argument("--fuse-mode", type=str, default="vote", choices=["vote", "wbf"])
    parser.add_argument(
        "--classwise-params-file",
        type=str,
        default="",
        help="JSON 文件，按类别覆盖 min_votes/single_keep_score/final_nms_iou/vote_iou_threshold。",
    )
    parser.add_argument(
        "--class-model-weights-file",
        type=str,
        default="",
        help="JSON 文件，按类别-模型提供融合权重（class_id -> model_idx -> weight）。",
    )
    parser.add_argument("--allowed-class-ids", type=str, default="", help="仅保留这些类别（逗号分隔 class_id）。")
    parser.add_argument("--allowed-class-ids-file", type=str, default="", help="仅保留这些类别（每行/逗号 class_id）。")
    parser.add_argument(
        "--auto-allowed-classes-from-data-root",
        type=str,
        default="",
        help="从 Data 根目录自动扫描出现过的类别（*.txt，忽略 classes.txt）。",
    )

    parser.add_argument("--iou-match-threshold", type=float, default=0.5)
    parser.add_argument("--save-vis", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def build_model_paths(kfold_root: Path, folds: List[int], extra_models: List[str]) -> List[Path]:
    model_paths: List[Path] = []
    for fold in folds:
        model_path = kfold_root / f"y11_v08_kfold_fold{fold}" / "weights" / "best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        model_paths.append(model_path)

    for extra in extra_models:
        p = Path(extra)
        if not p.exists():
            raise FileNotFoundError(f"Missing extra model: {p}")
        model_paths.append(p)

    deduped: List[Path] = []
    seen = set()
    for p in model_paths:
        r = str(p.resolve())
        if r in seen:
            continue
        seen.add(r)
        deduped.append(p)
    return deduped


def main() -> None:
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_vis = bool(args.save_vis)
    class_params = load_classwise_params(args.classwise_params_file)
    class_model_weights = load_class_model_weights(args.class_model_weights_file)
    allowed_class_ids = load_allowed_class_ids(
        allowed_ids=args.allowed_class_ids,
        allowed_ids_file=args.allowed_class_ids_file,
        auto_from_data_root=args.auto_allowed_classes_from_data_root,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    folds = parse_folds(args.folds)
    model_paths = build_model_paths(args.kfold_root, folds, args.extra_model)

    image_paths = sorted([p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {args.images_dir}")

    print(f"device={device}")
    print(f"models={len(model_paths)} folds={folds}")
    print(
        f"vote_iou={args.vote_iou_threshold} min_votes={args.min_votes} "
        f"single_keep_score={args.single_keep_score} final_nms_iou={args.final_nms_iou} "
        f"fuse_mode={args.fuse_mode}"
    )
    if class_params:
        print(f"classwise_overrides={len(class_params)} classes from {args.classwise_params_file}")
    else:
        print("classwise_overrides=0")
    print(f"class_model_weight_classes={len(class_model_weights)}")
    if allowed_class_ids:
        sample_ids = sorted(list(allowed_class_ids), key=lambda x: int(x))[:20]
        print(f"allowed_classes={len(allowed_class_ids)} sample={sample_ids}")
    else:
        print("allowed_classes=ALL")

    preds_by_image, category_mapping = collect_all_predictions(
        model_paths=model_paths,
        image_paths=image_paths,
        args=args,
        device=device,
        allowed_class_ids=allowed_class_ids if allowed_class_ids else None,
    )

    metrics: Dict[str, Dict[str, int]] = {}

    for img_path in tqdm(image_paths, desc="evaluate_vote_fusion"):
        image_name = img_path.stem
        image_key = str(img_path.resolve())
        label_path = args.labels_dir / f"{image_name}.txt"
        image_cv_bgr = read_image_cn(img_path)
        height, width = image_cv_bgr.shape[:2]

        gt_boxes = load_gt_boxes(
            label_path=label_path,
            width=width,
            height=height,
            allowed_class_ids=allowed_class_ids if allowed_class_ids else None,
        )
        fused_preds = vote_fuse_predictions(
            all_preds=preds_by_image.get(image_key, []),
            vote_iou_threshold=args.vote_iou_threshold,
            min_votes=args.min_votes,
            single_keep_score=args.single_keep_score,
            final_nms_iou=args.final_nms_iou,
            class_params=class_params,
            fuse_mode=args.fuse_mode,
            model_count=len(model_paths),
            allowed_class_ids=allowed_class_ids if allowed_class_ids else None,
            class_model_weights=class_model_weights if class_model_weights else None,
        )

        if save_vis:
            image_cv_rgb = cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_cv_rgb)
            draw = ImageDraw.Draw(image_pil)
            for g in gt_boxes:
                draw.rectangle(g["bbox"], outline=(0, 255, 0), width=3)
            for p in fused_preds:
                draw.rectangle(p["bbox"], outline=(255, 0, 0), width=2)
            image_pil.save(args.output_dir / f"{image_name}_eval.jpg", quality=95)

        for box in gt_boxes:
            cid = box["class_id"]
            if cid not in metrics:
                metrics[cid] = {"GT": 0, "Pred": 0, "TP": 0, "FP": 0, "FN": 0}
            metrics[cid]["GT"] += 1
        for box in fused_preds:
            cid = box["class_id"]
            if cid not in metrics:
                metrics[cid] = {"GT": 0, "Pred": 0, "TP": 0, "FP": 0, "FN": 0}
            metrics[cid]["Pred"] += 1

        for pred in fused_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if gt["matched"] or gt["class_id"] != pred["class_id"]:
                    continue
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= args.iou_match_threshold and best_gt_idx >= 0:
                gt_boxes[best_gt_idx]["matched"] = True
                metrics[pred["class_id"]]["TP"] += 1
            else:
                metrics[pred["class_id"]]["FP"] += 1

        for gt in gt_boxes:
            if not gt["matched"]:
                metrics[gt["class_id"]]["FN"] += 1

    rows = []
    for class_id, m in metrics.items():
        tp = m["TP"]
        fp = m["FP"]
        fn = m["FN"]
        gt = m["GT"]
        pred = m["Pred"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": get_class_name(category_mapping, class_id),
                "GT": gt,
                "Pred": pred,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["class_id"] = pd.to_numeric(df["class_id"])
        df = df.sort_values("class_id").reset_index(drop=True)
    df.to_csv(args.output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    total_tp = int(sum(v["TP"] for v in metrics.values()))
    total_fp = int(sum(v["FP"] for v in metrics.values()))
    total_fn = int(sum(v["FN"] for v in metrics.values()))
    total_gt = int(sum(v["GT"] for v in metrics.values()))
    total_pred = int(sum(v["Pred"] for v in metrics.values()))

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    micro_df = pd.DataFrame(
        [
            {
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "GT": total_gt,
                "Pred": total_pred,
                "Precision": micro_p,
                "Recall": micro_r,
                "F1": micro_f1,
            }
        ]
    )
    micro_df.to_csv(args.output_dir / "micro_summary.csv", index=False, encoding="utf-8-sig")

    overall_txt = args.output_dir / "overall_micro_metrics.txt"
    with open(overall_txt, "w", encoding="utf-8") as f:
        f.write("KFold Vote Fusion Overall (Micro)\n")
        f.write(f"device={device}\n")
        f.write(f"folds={folds}\n")
        for idx, p in enumerate(model_paths, start=1):
            f.write(f"model_{idx}={p}\n")
        f.write(f"vote_iou_threshold={args.vote_iou_threshold}\n")
        f.write(f"min_votes={args.min_votes}\n")
        f.write(f"single_keep_score={args.single_keep_score}\n")
        f.write(f"final_nms_iou={args.final_nms_iou}\n")
        f.write(f"fuse_mode={args.fuse_mode}\n")
        f.write(f"classwise_params_file={args.classwise_params_file}\n")
        f.write(f"classwise_overrides={len(class_params)}\n")
        f.write(f"class_model_weights_file={args.class_model_weights_file}\n")
        f.write(f"class_model_weight_classes={len(class_model_weights)}\n")
        f.write(f"allowed_class_count={len(allowed_class_ids) if allowed_class_ids else 'ALL'}\n")
        f.write(f"TP={total_tp}, FP={total_fp}, FN={total_fn}\n")
        f.write(f"Precision={micro_p:.4f}\n")
        f.write(f"Recall={micro_r:.4f}\n")
        f.write(f"F1={micro_f1:.4f}\n")

    print("vote fusion done")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Precision={micro_p:.4f} Recall={micro_r:.4f} F1={micro_f1:.4f}")
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
