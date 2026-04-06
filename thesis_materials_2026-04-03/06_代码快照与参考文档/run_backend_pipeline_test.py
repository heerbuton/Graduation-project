# -*- coding: utf-8 -*-
"""
使用 test 目录中的图片，调用后端 /api/upload 完整流水线，
并输出仅含红框（不含类别文字）的检测结果图。

运行方式:
    F:\anaconda\envs\pytorch\python.exe test/run_backend_pipeline_test.py
"""

import io
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
OUTPUT_DIR = PROJECT_ROOT / "test" / "pipeline_test_outputs"


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


def _draw_red_boxes_only(img: np.ndarray, yolo_boxes: List[dict]) -> np.ndarray:
    rendered = img.copy()
    h, w = rendered.shape[:2]
    for item in yolo_boxes:
        bbox = item.get("bbox", [])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, int(round(float(x1)))))
        y1 = max(0, min(h - 1, int(round(float(y1)))))
        x2 = max(0, min(w - 1, int(round(float(x2)))))
        y2 = max(0, min(h - 1, int(round(float(y2)))))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return rendered


def main() -> None:
    import sys

    sys.path.insert(0, str(BACKEND_DIR.resolve()))
    from app import app

    test_images = [
        PROJECT_ROOT / "test" / "testpicture-1.jpg",
        PROJECT_ROOT / "test" / "fullpage_yolo.jpg",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with app.test_client() as client:
        for image_path in test_images:
            if not image_path.exists():
                print(f"[跳过] 图片不存在: {image_path}")
                continue

            with open(image_path, "rb") as f:
                data = {"file": (io.BytesIO(f.read()), image_path.name)}
                resp = client.post("/api/upload", data=data, content_type="multipart/form-data")

            payload = resp.get_json(silent=True) or {}
            status = payload.get("status", "error")
            print(f"\n=== {image_path.name} ===")
            print(f"status_code={resp.status_code}, status={status}")

            if status != "success":
                print(f"message={payload.get('message', '')}")
                continue

            result_data = payload.get("data", {})
            yolo_boxes = result_data.get("yolo_boxes", [])
            topology_json = result_data.get("topology_json", {})
            llm_result = result_data.get("llm_result", [])

            src_img = _read_image_safe(image_path)
            boxed_img = _draw_red_boxes_only(src_img, yolo_boxes)

            boxed_path = OUTPUT_DIR / f"{image_path.stem}_red_boxes.jpg"
            _save_image_safe(boxed_path, boxed_img)

            summary = {
                "image": str(image_path),
                "counts": {
                    "yolo_boxes": len(yolo_boxes),
                    "topology_groups": len(topology_json.keys()) if isinstance(topology_json, dict) else 0,
                    "llm_notes": len(llm_result) if isinstance(llm_result, list) else 0,
                },
                "boxed_image": str(boxed_path),
            }
            summary_path = OUTPUT_DIR / f"{image_path.stem}_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print(f"yolo_boxes={summary['counts']['yolo_boxes']}")
            print(f"topology_groups={summary['counts']['topology_groups']}")
            print(f"llm_notes={summary['counts']['llm_notes']}")
            print(f"saved_image={boxed_path}")
            print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
