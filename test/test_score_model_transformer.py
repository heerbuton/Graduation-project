import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from pipeline.score_model_transformer import transform_llm_result_to_score_model  # noqa: E402


class ScoreModelTransformerTests(unittest.TestCase):
    def test_should_group_notes_into_measures_and_map_fields(self):
        llm_result = [
            {
                "new_measure": True,
                "pitch": "1",
                "octave": "4",
                "duration": "4",
                "action": "",
                "string": "一",
                "position": "一",
                "finger": "",
            },
            {
                "pitch": "6",
                "octave": "3",
                "duration": "8",
                "action": "历",
                "string": "六",
                "position": "",
                "finger": "",
            },
            {
                "new_measure": True,
                "pitch": "5",
                "octave": "3",
                "duration": "4",
                "action": "",
                "string": "五",
                "position": "五",
                "finger": "",
            },
        ]

        score_model = transform_llm_result_to_score_model(llm_result)
        measures = score_model["measures"]

        self.assertEqual(len(measures), 2)
        self.assertEqual(len(measures[0]["notes"]), 2)
        self.assertEqual(len(measures[1]["notes"]), 1)
        self.assertEqual(measures[0]["notes"][1]["pitch"], "6")
        self.assertEqual(measures[0]["notes"][1]["duration"], "8")
        self.assertEqual(measures[0]["notes"][1]["guqin"]["action"], "历")
        self.assertEqual(measures[0]["notes"][1]["guqin"]["stringOrder"], "六")

    def test_new_measure_on_first_note_should_not_create_empty_measure(self):
        llm_result = [
            {"new_measure": True, "pitch": "1", "octave": "4", "duration": "4"},
            {"pitch": "2", "octave": "4", "duration": "4"},
        ]

        score_model = transform_llm_result_to_score_model(llm_result)
        measures = score_model["measures"]

        self.assertEqual(len(measures), 1)
        self.assertEqual(len(measures[0]["notes"]), 2)

    def test_strict_mode_should_raise_on_invalid_duration(self):
        llm_result = [{"pitch": "1", "octave": "4", "duration": "32"}]

        with self.assertRaises(ValueError):
            transform_llm_result_to_score_model(llm_result, strict=True)

    def test_non_strict_mode_should_fallback_and_report_issue(self):
        llm_result = [{"pitch": "9", "octave": "8", "duration": "32"}]

        score_model = transform_llm_result_to_score_model(llm_result, strict=False)
        note = score_model["measures"][0]["notes"][0]

        self.assertEqual(note["pitch"], "1")
        self.assertEqual(note["octave"], "4")
        self.assertEqual(note["duration"], "4")
        self.assertGreaterEqual(len(score_model["issues"]), 1)

    def test_should_infer_measure_by_duration_when_no_new_measure(self):
        llm_result = [
            {"pitch": "1", "octave": "4", "duration": "4"},
            {"pitch": "2", "octave": "4", "duration": "4"},
            {"pitch": "3", "octave": "4", "duration": "4"},
            {"pitch": "4", "octave": "4", "duration": "4"},
            {"pitch": "5", "octave": "4", "duration": "4"},
        ]

        score_model = transform_llm_result_to_score_model(llm_result, strict=False)

        self.assertEqual(score_model["measureCount"], 2)
        self.assertEqual(len(score_model["measures"][0]["notes"]), 4)
        self.assertEqual(len(score_model["measures"][1]["notes"]), 1)

    def test_single_leading_new_measure_should_still_infer_following_measures(self):
        llm_result = [
            {"new_measure": True, "pitch": "1", "octave": "4", "duration": "4"},
            {"pitch": "2", "octave": "4", "duration": "4"},
            {"pitch": "3", "octave": "4", "duration": "4"},
            {"pitch": "4", "octave": "4", "duration": "4"},
            {"pitch": "5", "octave": "4", "duration": "4"},
        ]

        score_model = transform_llm_result_to_score_model(llm_result, strict=False)

        self.assertEqual(score_model["measureCount"], 2)
        self.assertEqual(len(score_model["measures"][0]["notes"]), 4)
        self.assertEqual(len(score_model["measures"][1]["notes"]), 1)


if __name__ == "__main__":
    unittest.main()
