# -*- coding: utf-8 -*-
"""
测试脚本 1: 全页直推 + 空间聚类
================================
将整张图片直接送入 YOLO (imgsz=640) 进行推理，然后执行空间聚类。
这是最简单的方法，但对于高分辨率古琴谱图效果较差（部件太小）。

使用方法:
    cd f:\\AIcharacter\\End
    python test/test_clustering_fullpage.py

参考脚本: train_v07_finetune.py (imgsz=640 全图推理)
"""

import sys
from pathlib import Path
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
# 配置参数
# ============================================================
IMAGE_PATH = TEST_DIR / "testpicture-1.jpg"
WEIGHTS_PATH = PROJECT_ROOT / "backend" / "best.pt"
CONF_THRESHOLD = 0.10     # 全页推理置信度
IMGSZ = 640               # 与训练时一致


def run_fullpage_detection(image_path, weights_path):
    """全页直推: 整张图直接送入 YOLO"""
    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        verbose=False,
    )
    parsed = []
    for r in results:
        names = getattr(r, "names", {})
        boxes = r.boxes
        if boxes is None: continue
        for bbox, cls_id, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = [float(v) for v in bbox]
            parsed.append({
                "class": str(names.get(int(cls_id), str(int(cls_id)))),
                "bbox": [x1, y1, x2, y2],
                "conf": round(float(conf), 6),
            })
    parsed.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return parsed


def main():
    print("=" * 60)
    print("  方法 1: 全页直推 + 空间聚类")
    print("=" * 60)

    if not IMAGE_PATH.exists():
        print(f"[错误] 图片不存在: {IMAGE_PATH}"); return
    if not WEIGHTS_PATH.exists():
        print(f"[错误] 权重不存在: {WEIGHTS_PATH}"); return

    # 1. YOLO 全页推理
    print(f"\n[Step 1] 全页直推 (imgsz={IMGSZ}, conf={CONF_THRESHOLD})")
    yolo_boxes = run_fullpage_detection(IMAGE_PATH, WEIGHTS_PATH)
    print(f"  检测到 {len(yolo_boxes)} 个部件")
    for i, b in enumerate(yolo_boxes):
        bb = b["bbox"]
        print(f"    [{i:2d}] {b['class']:6s} conf={b['conf']:.4f}  "
              f"[{bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}]")

    # 2. 空间聚类
    print(f"\n[Step 2] 空间聚类")
    dets = prepare_detections(yolo_boxes)
    clusters = build_clusters(dets)
    fields_list = [extract_fields(cl) for cl in clusters]
    print(f"  聚类为 {len(clusters)} 个组:")
    print_clusters(clusters, fields_list)

    # 3. 可视化
    print(f"\n[Step 3] 生成可视化")
    img = read_image_safe(str(IMAGE_PATH))
    save_image_safe(draw_yolo_pil(img, dets), str(TEST_DIR / "fullpage_yolo.jpg"))
    save_image_safe(draw_clusters_pil(img, clusters, fields_list), str(TEST_DIR / "fullpage_clusters.jpg"))
    save_topology_json(clusters, fields_list, str(TEST_DIR / "fullpage_topology.json"))

    print(f"\n{'='*60}")
    print(f"  全页直推完成: {len(yolo_boxes)} 部件 -> {len(clusters)} 组")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
