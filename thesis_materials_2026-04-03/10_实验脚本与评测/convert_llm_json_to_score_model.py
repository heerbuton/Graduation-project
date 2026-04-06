import argparse
import json
import sys
from pathlib import Path
from typing import Any, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pipeline.score_model_transformer import transform_llm_result_to_score_model  # noqa: E402


def _load_llm_result(payload: Any) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("llm_result"), list):
            return payload["llm_result"]
        if isinstance(payload.get("notes"), list):
            return payload["notes"]
    raise ValueError("输入 JSON 必须是数组，或包含 llm_result/notes 数组。")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 LLM 打谱 JSON 转换为前端可渲染的 ScoreModel。")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径（LLM 打谱结果）")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径（ScoreModel）")
    parser.add_argument("--strict", action="store_true", help="启用严格校验（字段非法直接报错）")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    llm_result = _load_llm_result(payload)

    score_model = transform_llm_result_to_score_model(llm_result, strict=bool(args.strict))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(score_model, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(llm_result)} notes -> {score_model.get('measureCount', 0)} measures")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
