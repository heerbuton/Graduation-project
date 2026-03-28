import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app_module  # noqa: E402


class AppScoreModelTests(unittest.TestCase):
    def setUp(self):
        backend_app_module.app.config["TESTING"] = True
        self.client = backend_app_module.app.test_client()

    def _build_dummy_jpg_bytes(self) -> bytes:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("测试图片编码失败")
        return encoded.tobytes()

    def test_mock_pipeline_should_include_score_model(self):
        response = self.client.get("/api/mock_pipeline")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "success")
        self.assertIn("score_model", payload["data"])

    def test_upload_should_include_score_model(self):
        fake_yolo = [{"class": "大", "bbox": [1, 1, 5, 5]}]
        fake_topology = {"group_1": {"fingering": "勾", "finger": "大", "position": "九", "string": "一"}}
        fake_sequence = [{"group_id": "group_1", "action": "勾", "finger": "大", "position": "九", "string": "一"}]
        fake_llm_result = [
            {
                "pitch": "1",
                "octave": "4",
                "duration": "4",
                "action": "勾",
                "string": "一",
                "position": "九",
                "finger": "大",
            }
        ]

        with patch("app.detect_components", return_value=fake_yolo), patch(
            "app.build_topology", return_value=fake_topology
        ), patch("app.build_jianzi_sequence", return_value=fake_sequence), patch(
            "app.infer_pitch_duration", return_value=fake_llm_result
        ), patch(
            "app.generate_musicxml", return_value="<score/>"
        ):
            data = {"file": (io.BytesIO(self._build_dummy_jpg_bytes()), "dummy.jpg")}
            response = self.client.post("/api/upload", data=data, content_type="multipart/form-data")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "success")
        self.assertIn("score_model", payload["data"])


if __name__ == "__main__":
    unittest.main()
