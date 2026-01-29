from __future__ import annotations

"""
Evaluate audio -> OpenAI omni model -> tool-call pipeline.

Usage:
  python scripts/eval_omni_baseline.py --data data/audio_dataset.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from metrics import evaluate
from omni_toolcall_openai import OmniToolCallerOpenAI
from tool_schema import ToolCall, normalize_unit


def _load_samples(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        samples.append(json.loads(line))
    return samples


def _extract_first_json(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


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


def _normalize_unit_name(u: str) -> str:
    s = (u or "").strip().lower()
    s = s.replace(" ", "")
    mapping = {
        "celsius": "c",
        "fahrenheit": "f",
        "kelvin": "k",
        "kilogram": "kg",
        "kilograms": "kg",
        "kg": "kg",
        "gram": "g",
        "grams": "g",
        "g": "g",
        "pound": "lb",
        "pounds": "lb",
        "lb": "lb",
        "ounce": "oz",
        "ounces": "oz",
        "oz": "oz",
        "mi": "mile",
        "inches": "inch",
        "feet": "ft",
        "yards": "yd",
        "kj": "kg",
        "kj/": "kg",
    }
    return mapping.get(s, s)


def _normalize_tool_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    args = payload.get("arguments")
    if not isinstance(args, dict):
        return payload
    args = dict(args)
    if "from_unit" in args:
        args["from_unit"] = normalize_unit(_normalize_unit_name(str(args.get("from_unit", ""))))
    if "to_unit" in args:
        args["to_unit"] = normalize_unit(_normalize_unit_name(str(args.get("to_unit", ""))))
    payload = dict(payload)
    payload["arguments"] = args
    return payload


def _parse_output(text: str) -> tuple[str, ToolCall | None, str]:
    raw = (text or "").strip()
    if not raw:
        return "invalid", None, ""

    lines = raw.splitlines()
    first_idx = None
    first_line = ""
    for i, line in enumerate(lines):
        if line.strip():
            first_idx = i
            first_line = line.strip()
            break

    if first_idx is not None and first_line.lower().startswith("no_tool"):
        remainder = first_line[len("no_tool") :].strip()
        tail_lines = []
        if remainder:
            tail_lines.append(remainder)
        if first_idx + 1 < len(lines):
            tail_lines.extend(lines[first_idx + 1 :])
        final_text = "\n".join(tail_lines).strip()
        return "no_tool", None, final_text

    payload = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = _extract_first_json(raw)

    if not isinstance(payload, dict):
        return "invalid", None, ""
    if set(payload.keys()) != {"name", "arguments"}:
        return "invalid", None, ""
    if not isinstance(payload.get("name"), str) or not isinstance(payload.get("arguments"), dict):
        return "invalid", None, ""
    try:
        payload = _normalize_tool_payload(payload)
        toolcall = ToolCall.model_validate(payload)
    except Exception:
        return "invalid", None, ""
    return "tool", toolcall, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_dataset.jsonl")),
        help="Path to audio dataset JSON/JSONL.",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--model", default=None)
    parser.add_argument("--debug_k", type=int, default=10)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    samples = _load_samples(data_path)

    limit = min(args.limit, len(samples))
    samples = samples[:limit]

    model_name = args.model or os.getenv("OPENAI_OMNI_MODEL", "gpt-4o-audio-preview")
    omni = OmniToolCallerOpenAI(model_name=model_name)

    preds = []
    mismatches = []
    errors = []
    first_error = None
    num_errors = 0
    debug_response = None

    for sample in samples:
        audio_path = sample.get("audio_path", "")
        sample_id = sample.get("id")
        user_text = sample.get("text") or ""
        lang = sample.get("lang") or ""

        out = omni.infer(audio_path=audio_path, user_text_hint="", lang=lang)
        if out.startswith("<ERROR:") or out == "<EMPTY_OUTPUT>":
            status, toolcall, final_text = "invalid", None, out
            num_errors += 1
            if first_error is None:
                first_error = out
            errors.append(
                {
                    "id": sample_id,
                    "audio_path": audio_path,
                    "error": out,
                    "model_out": out,
                    "final_text": out,
                }
            )
            if out == "<EMPTY_OUTPUT>" and debug_response is None:
                debug_response = omni.last_response_repr()
        else:
            status, toolcall, final_text = _parse_output(out)

        preds.append({"status": status, "toolcall": toolcall})

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

        if mismatch:
            mismatches.append(
                {
                    "id": sample_id,
                    "audio_path": audio_path,
                    "text": user_text,
                    "expected": expected_tool,
                    "predicted": {
                        "status": status,
                        "tool_call": predicted_tool_call,
                        "final_text": final_text or None,
                    },
                    "model_out": out,
                }
            )

    metrics = evaluate(samples, preds)

    report = {
        "metrics": metrics,
        "limit": limit,
        "engine": "openai_omni",
        "model_name": model_name,
    }
    report["num_errors"] = num_errors
    if first_error is not None:
        report["first_error"] = first_error
    if debug_response:
        report["debug_response_repr"] = debug_response
    if errors:
        report["errors"] = errors

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "omni_eval_report.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"model_name: {model_name}")
    print(f"=== DEBUG (first {args.debug_k} mismatches) ===")
    for item in mismatches[: args.debug_k]:
        print(f'audio_path: {item.get("audio_path")}')
        print(f'text: {item.get("text")}')
        print(f'expected: {json.dumps(item.get("expected"), ensure_ascii=False)}')
        print(f'pred: {json.dumps(item.get("predicted"), ensure_ascii=False)}')
        print(f'model_out: {item.get("model_out")}')


if __name__ == "__main__":
    main()
