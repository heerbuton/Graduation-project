# -*- coding: utf-8 -*-
"""
测试脚本 2: SAHI 滑窗推理 + 空间聚类
=====================================
采用 SAHI (Slicing Aided Hyper Inference) 方法:
  1. 用 256x256 窗口按 stride=128 滑动切割全页
  2. 每个切片放大到 640x640 送入 YOLO 推理
  3. 坐标映射还原到原图尺度
  4. Batched NMS 按类别去重
  5. 空间聚类

这个方法能显著提升对小部件的检测精度。

使用方法:
    cd f:\\AIcharacter\\End
    python test/test_clustering_sahi.py

参考脚本: sahi_v07_inference.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

# 路径配置
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(TEST_DIR))

from clustering_common import (
    prepare_detections, build_clusters, extract_fields,
    draw_yolo_pil, draw_clusters_pil,
    read_image_safe, save_image_safe,
    print_clusters, save_topology_json,
)

# ============================================================
# SAHI 配置参数 (与 sahi_v07_inference.py 对齐)
# ============================================================
IMAGE_PATH = TEST_DIR / "testpicture-1.jpg"
WEIGHTS_PATH = PROJECT_ROOT / "backend" / "best.pt"

WINDOW_SIZE = 256         # 切片窗口尺寸 (px)
STRIDE = 128              # 滑动步长 (px)
CONF_THRESHOLD = 0.2      # SAHI 推理置信度 (与参考脚本一致)
NMS_IOU_THRESHOLD = 0.45  # Batched NMS IoU 阈值
INFER_IMGSZ = 640         # 切片放大到的推理尺寸


# ============================================================
# SAHI 滑窗推理 (移植自 sahi_v07_inference.py)
# ============================================================
def sahi_sliding_window_inference(model, img):
    """
    1. 滑窗切割  2. 放大推理  3. 坐标还原  4. Batched NMS
    """
    img_h, img_w = img.shape[:2]
    scale_factor = float(INFER_IMGSZ) / WINDOW_SIZE  # 256->640 = 2.5x

    all_boxes = []
    all_scores = []
    all_classes = []

    print(f"  原图: {img_w}x{img_h} | 窗口: {WINDOW_SIZE}x{WINDOW_SIZE} | 步长: {STRIDE}")

    patch_count = 0
    for y in range(0, img_h, STRIDE):
        for x in range(0, img_w, STRIDE):
            y1 = min(y, max(0, img_h - WINDOW_SIZE))
            x1 = min(x, max(0, img_w - WINDOW_SIZE))
            y2 = min(y1 + WINDOW_SIZE, img_h)
            x2 = min(x1 + WINDOW_SIZE, img_w)

            patch = img[y1:y2, x1:x2]

            # 边缘 Padding
            if patch.shape[0] < WINDOW_SIZE or patch.shape[1] < WINDOW_SIZE:
                pad_h = WINDOW_SIZE - patch.shape[0]
                pad_w = WINDOW_SIZE - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w,
                                           cv2.BORDER_CONSTANT, value=(114, 114, 114))

            # 放大到 640x640
            resized = cv2.resize(patch, (INFER_IMGSZ, INFER_IMGSZ), interpolation=cv2.INTER_CUBIC)

            results = model.predict(source=resized, imgsz=INFER_IMGSZ, conf=CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes
            patch_count += 1

            # 坐标映射还原
            for box in boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].item()
                cls_id = int(box.cls[0].item())

                orig_x1 = (bx1 / scale_factor) + x1
                orig_y1 = (by1 / scale_factor) + y1
                orig_x2 = (bx2 / scale_factor) + x1
                orig_y2 = (by2 / scale_factor) + y1

                all_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
                all_scores.append(score)
                all_classes.append(cls_id)

    print(f"  切割 {patch_count} 个窗口, 累计 {len(all_boxes)} 个原始预测框")

    if len(all_boxes) == 0:
        return [], model.names

    # Batched NMS (按类别独立)
    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    classes_t = torch.tensor(all_classes, dtype=torch.float32)

    keep = torchvision.ops.batched_nms(boxes_t, scores_t, classes_t, NMS_IOU_THRESHOLD)

    final_boxes = boxes_t[keep].numpy()
    final_scores = scores_t[keep].numpy()
    final_classes = classes_t[keep].numpy().astype(int)

    print(f"  NMS 后保留 {len(final_boxes)} 个部件")

    # 转换为标准格式
    class_names = model.names
    parsed = []
    for bbox, cls_id, conf in zip(final_boxes, final_classes, final_scores):
        parsed.append({
            "class": str(class_names.get(int(cls_id), str(cls_id))),
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "conf": round(float(conf), 6),
        })
    parsed.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return parsed, class_names


def main():
    print("=" * 60)
    print("  方法 2: SAHI 滑窗推理 + 空间聚类")
    print("=" * 60)

    if not IMAGE_PATH.exists():
        print(f"[错误] 图片不存在: {IMAGE_PATH}"); return
    if not WEIGHTS_PATH.exists():
        print(f"[错误] 权重不存在: {WEIGHTS_PATH}"); return

    # 1. SAHI 推理
    print(f"\n[Step 1] SAHI 滑窗推理 (window={WINDOW_SIZE}, stride={STRIDE}, conf={CONF_THRESHOLD})")
    model = YOLO(str(WEIGHTS_PATH))
    img = read_image_safe(str(IMAGE_PATH))
    yolo_boxes, class_names = sahi_sliding_window_inference(model, img)

    print(f"\n  最终检测 {len(yolo_boxes)} 个部件:")
    for i, b in enumerate(yolo_boxes):
        bb = b["bbox"]
        print(f"    [{i:2d}] {b['class']:6s} conf={b['conf']:.4f}  "
              f"[{bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}]  "
              f"size={bb[2]-bb[0]:.0f}x{bb[3]-bb[1]:.0f}")

    # 2. 空间聚类
    print(f"\n[Step 2] 空间聚类")
    dets = prepare_detections(yolo_boxes)
    clusters = build_clusters(dets)
    fields_list = [extract_fields(cl) for cl in clusters]
    print(f"  聚类为 {len(clusters)} 个组:")
    print_clusters(clusters, fields_list)

    # 3. 可视化
    print(f"\n[Step 3] 生成可视化")
    save_image_safe(draw_yolo_pil(img, dets), str(TEST_DIR / "sahi_yolo.jpg"))
    save_image_safe(draw_clusters_pil(img, clusters, fields_list), str(TEST_DIR / "sahi_clusters.jpg"))
    save_topology_json(clusters, fields_list, str(TEST_DIR / "sahi_topology.json"))

    print(f"\n{'='*60}")
    print(f"  SAHI 推理完成: {len(yolo_boxes)} 部件 -> {len(clusters)} 组")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
