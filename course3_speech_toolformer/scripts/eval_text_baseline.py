from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from metrics import evaluate
from parse_toolcall import parse_toolcall
from prompting import build_messages
from tool_schema import normalize_unit


def _toolcall_to_dict(toolcall):
    if toolcall is None:
        return None
    return toolcall.model_dump()


def _toolcall_equal(expected: dict | None, predicted: dict | None) -> bool:
    if expected is None or predicted is None:
        return False
    if expected.get("name") != predicted.get("name"):
        return False
    exp_args = expected.get("arguments", {})
    pred_args = predicted.get("arguments", {})
    exp_from = normalize_unit(str(exp_args.get("from_unit", "")))
    exp_to = normalize_unit(str(exp_args.get("to_unit", "")))
    pred_from = normalize_unit(str(pred_args.get("from_unit", "")))
    pred_to = normalize_unit(str(pred_args.get("to_unit", "")))
    if exp_from != pred_from or exp_to != pred_to:
        return False
    try:
        exp_value = float(exp_args.get("value"))
        pred_value = float(pred_args.get("value"))
    except (TypeError, ValueError):
        return False
    return abs(exp_value - pred_value) <= 1e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug_k", type=int, default=10)
    parser.add_argument("--engine", choices=["stub", "llamacpp"], default="stub")
    args = parser.parse_args()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "text_dataset.json"))
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run: python scripts/make_text_dataset.py"
        )
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    limit = min(50, len(samples)) if args.limit is None else min(args.limit, len(samples))
    samples = samples[:limit]

    preds = []
    error_cases = []
    mismatches = []

    if args.engine == "stub":
        from llm_stub import infer

        def _infer(messages: list[dict]) -> str:
            return infer(messages)
    else:
        from config import load_env
        from llm_llamacpp import LlamaCppRunner

        env = load_env()
        runner = LlamaCppRunner(
            llama_bin=env["LLAMA_BIN"],
            model_path=env["LLAMA_MODEL"],
            ctx=env["LLAMA_CTX"],
            gpu_layers=env["LLAMA_GPU_LAYERS"],
            temperature=env["LLAMA_TEMP"],
            max_tokens=env["LLAMA_MAX_TOKENS"],
        )

        def _infer(messages: list[dict]) -> str:
            return runner.infer(messages)

    for sample in samples:
        messages = build_messages(sample["text"])
        out = _infer(messages)
        status, toolcall = parse_toolcall(out)

        preds.append({"status": status, "toolcall": toolcall})

        tool_required = bool(sample.get("tool_required"))
        predicted_tool = status == "tool"
        expected_tool = sample.get("expected_tool_call")
        predicted_tool_call = _toolcall_to_dict(toolcall)

        mismatch = False
        if expected_tool is None and predicted_tool:
            mismatch = True
        elif expected_tool is not None and not predicted_tool:
            mismatch = True
        elif expected_tool is not None and predicted_tool:
            if not _toolcall_equal(expected_tool, predicted_tool_call):
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
        if mismatch:
            mismatches.append(
                {
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
    print(f"=== DEBUG (first {args.debug_k} mismatches) ===")
    for item in mismatches[: args.debug_k]:
        print(f'text: {item.get("text")}')
        print(f'expected: {json.dumps(item.get("expected"), ensure_ascii=False)}')
        print(f'pred: {json.dumps(item.get("predicted"), ensure_ascii=False)}')


if __name__ == "__main__":
    main()
