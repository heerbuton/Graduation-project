import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "convert_llm_json_to_score_model.py"
PYTHON_EXE = r"F:\anaconda\envs\pytorch\python.exe"


class ScoreModelScriptTests(unittest.TestCase):
    def test_script_should_convert_llm_json_file_to_score_model_file(self):
        llm_result = [
            {"new_measure": True, "pitch": "1", "octave": "4", "duration": "4", "string": "一"},
            {"pitch": "6", "octave": "3", "duration": "8", "string": "六"},
            {"new_measure": True, "pitch": "5", "octave": "3", "duration": "4", "string": "五"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "llm_result.json"
            output_path = Path(tmp_dir) / "score_model.json"
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(llm_result, f, ensure_ascii=False)

            result = subprocess.run(
                [PYTHON_EXE, str(SCRIPT_PATH), "--input", str(input_path), "--output", str(output_path)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())

            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["measureCount"], 2)
            self.assertEqual(payload["noteCount"], 3)
            self.assertEqual(payload["measures"][0]["notes"][1]["guqin"]["stringOrder"], "六")


if __name__ == "__main__":
    unittest.main()
