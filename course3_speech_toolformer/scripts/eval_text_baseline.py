from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from llm_stub import infer
from metrics import evaluate
from parse_toolcall import parse_toolcall
from prompting import build_messages


def _toolcall_to_dict(toolcall):
    if toolcall is None:
        return None
    return toolcall.model_dump()


def main() -> None:
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "text_dataset.json"))
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run: python scripts/make_text_dataset.py"
        )
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    preds = []
    error_cases = []

    for sample in samples:
        messages = build_messages(sample["text"])
        out = infer(messages)
        status, toolcall = parse_toolcall(out)

        preds.append({"status": status, "toolcall": toolcall})

        tool_required = bool(sample.get("tool_required"))
        predicted_tool = status == "tool"
        expected_tool = sample.get("expected_tool_call")
        predicted_tool_call = _toolcall_to_dict(toolcall)

        mismatch = False
        if predicted_tool != tool_required:
            mismatch = True
        elif tool_required:
            if predicted_tool_call != expected_tool:
                mismatch = True
        elif status == "invalid":
            mismatch = True

        if mismatch and len(error_cases) < 20:
            error_cases.append(
                {
                    "id": sample.get("id"),
                    "text": sample.get("text"),
                    "expected": expected_tool,
                    "predicted": {"status": status, "tool_call": predicted_tool_call},
                }
            )

    metrics = evaluate(samples, preds)

    report = {"metrics": metrics, "errors": error_cases}
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "text_eval_report.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
