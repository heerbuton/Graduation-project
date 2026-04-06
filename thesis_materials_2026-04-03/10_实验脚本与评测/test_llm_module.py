import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from pipeline import llm_module  # noqa: E402


FIXED_API_KEY = "sk-48f0485ef8f246bf9316e2d87420726e"
FIXED_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
FIXED_MODEL = "qwen3.5-plus"


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _sample_topology():
    return {
        "group_1": {
            "fingering": "勾",
            "finger": "大",
            "position": "九",
            "string": "一",
            "right_fingering": "勾",
            "left_fingering": "",
            "left_finger": "大",
            "hui": "九",
            "xian": "一",
            "components": [],
        }
    }


def _message_content(note_payload):
    return json.dumps({"notes": [note_payload]}, ensure_ascii=False)


def _many_group_topology(count: int):
    topology = {}
    for idx in range(1, count + 1):
        topology[f"group_{idx}"] = {
            "fingering": "勾",
            "finger": "大",
            "position": "九",
            "string": "一",
            "right_fingering": "勾",
            "left_fingering": "",
            "left_finger": "大",
            "hui": "九",
            "xian": "一",
            "components": [],
        }
    return topology


class LlmModuleTests(unittest.TestCase):
    def test_list_topology_should_preserve_input_group_id(self):
        topology = [
            {
                "group_id": "group_99",
                "right_fingering": "历",
                "left_fingering": "",
                "left_finger": "",
                "hui": "",
                "xian": "六",
                "action": "历",
                "finger": "",
                "position": "",
                "string": "六",
            }
        ]

        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": _message_content(
                            {
                                "group_id": "group_99",
                                "pitch": "1",
                                "octave": "4",
                                "duration": "4",
                                "action": "历",
                                "string": "六",
                                "position": "",
                                "finger": "",
                                "new_measure": False,
                            }
                        )
                    }
                }
            ]
        }
        with patch("pipeline.llm_module.requests.post", return_value=_FakeResponse(200, response_payload)):
            notes = llm_module.infer_pitch_duration(topology)

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["action"], "历")
        self.assertEqual(notes[0]["string"], "六")

    def test_request_uses_fixed_bailian_endpoint_model_and_key(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": _message_content(
                            {
                                "group_id": "group_1",
                                "pitch": "1",
                                "octave": "4",
                                "duration": "4",
                                "action": "勾",
                                "string": "一",
                                "position": "九",
                                "finger": "大",
                                "new_measure": False,
                            }
                        )
                    }
                }
            ]
        }

        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "sk-env-key-should-be-ignored",
                "OPENAI_API_KEY": "sk-env-key-should-be-ignored",
                "LLM_API_URL": "https://example.com/wrong-url",
                "LLM_MODEL": "wrong-model",
            },
            clear=False,
        ):
            with patch("pipeline.llm_module.requests.post", return_value=_FakeResponse(200, response_payload)) as mock_post:
                notes = llm_module.infer_pitch_duration(_sample_topology())

        self.assertEqual(len(notes), 1)
        self.assertEqual(mock_post.call_count, 1)
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], FIXED_API_URL)
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {FIXED_API_KEY}")
        self.assertEqual(kwargs["json"]["model"], FIXED_MODEL)
        self.assertFalse(kwargs["json"].get("enable_thinking", True))
        self.assertEqual(kwargs["json"]["response_format"]["type"], "json_object")

    def test_accepts_markdown_fenced_json(self):
        fenced = (
            "```json\n"
            + _message_content(
                {
                    "group_id": "group_1",
                    "pitch": "2",
                    "octave": "5",
                    "duration": "8",
                    "action": "抹",
                    "string": "二",
                    "position": "十",
                    "finger": "中",
                    "new_measure": True,
                }
            )
            + "\n```"
        )
        response_payload = {"choices": [{"message": {"content": fenced}}]}

        with patch("pipeline.llm_module.requests.post", return_value=_FakeResponse(200, response_payload)):
            notes = llm_module.infer_pitch_duration(_sample_topology())

        self.assertEqual(notes[0]["pitch"], "2")
        self.assertEqual(notes[0]["duration"], "8")
        self.assertTrue(notes[0]["new_measure"])

    def test_http_error_should_raise(self):
        with patch(
            "pipeline.llm_module.requests.post",
            return_value=_FakeResponse(401, {"error": "unauthorized"}, text="unauthorized"),
        ):
            with self.assertRaises(requests.HTTPError):
                llm_module.infer_pitch_duration(_sample_topology())

    def test_non_json_should_raise(self):
        response_payload = {"choices": [{"message": {"content": "this is not json"}}]}
        with patch("pipeline.llm_module.requests.post", return_value=_FakeResponse(200, response_payload)):
            with self.assertRaises(ValueError):
                llm_module.infer_pitch_duration(_sample_topology())

    def test_large_group_input_should_use_single_full_context_call(self):
        topology = _many_group_topology(21)

        def _fake_post(_url, headers=None, json=None, timeout=None):
            payload = json or {}
            user_prompt = str(((payload.get("messages") or [{}, {}])[1]).get("content", ""))
            parts = user_prompt.split("\n", 1)
            raw_groups = json_module.loads(parts[1]) if len(parts) > 1 else []

            notes = []
            for item in raw_groups:
                notes.append(
                    {
                        "group_id": item.get("group_id", ""),
                        "pitch": "1",
                        "octave": "4",
                        "duration": "4",
                        "action": item.get("action", ""),
                        "string": item.get("string", ""),
                        "position": item.get("position", ""),
                        "finger": item.get("finger", ""),
                        "new_measure": False,
                    }
                )
            return _FakeResponse(
                200,
                {"choices": [{"message": {"content": json_module.dumps({"notes": notes}, ensure_ascii=False)}}]},
            )

        json_module = json
        with patch("pipeline.llm_module.requests.post", side_effect=_fake_post) as mock_post:
            notes = llm_module.infer_pitch_duration(topology)

        self.assertEqual(len(notes), 21)
        self.assertEqual(mock_post.call_count, 1)

    def test_invalid_field_value_should_raise(self):
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": _message_content(
                            {
                                "group_id": "group_1",
                                "pitch": "9",
                                "octave": "7",
                                "duration": "32",
                                "action": "勾",
                                "string": "一",
                                "position": "九",
                                "finger": "大",
                                "new_measure": False,
                            }
                        )
                    }
                }
            ]
        }
        with patch("pipeline.llm_module.requests.post", return_value=_FakeResponse(200, response_payload)):
            with self.assertRaises(ValueError):
                llm_module.infer_pitch_duration(_sample_topology())


if __name__ == "__main__":
    unittest.main()

