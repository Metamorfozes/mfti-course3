from __future__ import annotations

"""
Evaluate audio -> Gemma omni (HF) -> tool-call pipeline.

Usage:
  python scripts/eval_gemma_omni_baseline.py --data data/audio_dataset.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from llm_gemma3n_omni import Gemma3nOmniToolCaller
import llm_gemma3n_omni
from metrics import evaluate
from tool_schema import ToolCall, normalize_unit

_UNIT_SYNONYMS = {
    "celsius": "c",
    "fahrenheit": "f",
    "kelvin": "k",
    "centigrade": "c",
    "degc": "c",
    "degf": "f",
    "kilogram": "kg",
    "kilograms": "kg",
    "kilo": "kg",
    "gram": "g",
    "grams": "g",
    "pound": "lb",
    "pounds": "lb",
    "lbs": "lb",
    "ounce": "oz",
    "ounces": "oz",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "inch": "inch",
    "inches": "inch",
    "foot": "ft",
    "feet": "ft",
    "yard": "yd",
    "yards": "yd",
    "mile": "mile",
    "miles": "mile",
    "mi": "mile",
}


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


def _normalize_unit_name(u: str) -> str:
    s = (u or "").strip().lower()
    s = s.replace(" ", "")
    s = s.replace("-", "")
    return _UNIT_SYNONYMS.get(s, s)


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
    if "value" in args:
        try:
            args["value"] = float(args.get("value"))
        except (TypeError, ValueError):
            pass
    payload = dict(payload)
    payload["arguments"] = args
    return payload


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
        return "no_tool", None, ""

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
    print(llm_gemma3n_omni.__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_dataset.jsonl")),
        help="Path to audio dataset JSON/JSONL.",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--debug_k", type=int, default=10)
    parser.add_argument("--model", default="google/gemma-3n-E2B-it")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    samples = _load_samples(data_path)

    limit = min(args.limit, len(samples))
    samples = samples[:limit]

    gemma = Gemma3nOmniToolCaller(model_name=args.model)

    preds = []
    mismatches = []
    errors = []
    first_error = None
    num_errors = 0
    audio_used = 0
    text_fallback = 0
    native_attempted = 0
    native_success = 0
    native_fallback = 0

    for sample in samples:
        audio_path = sample.get("audio_path", "")
        sample_id = sample.get("id")
        user_text = sample.get("text") or ""
        lang = sample.get("lang") or ""

        result = gemma.infer(audio_path=audio_path, lang=lang, text_hint="")
        if isinstance(result, dict):
            out = result.get("text", "")
            meta = result.get("meta", {}) or {}
        else:
            out = result
            meta = {}
        used_audio = gemma.last_used_audio()
        if used_audio is True:
            audio_used += 1
        elif used_audio is False:
            text_fallback += 1
        if meta.get("native_audio_attempted"):
            native_attempted += 1
        if meta.get("native_audio_success"):
            native_success += 1
        if meta.get("used_fallback_asr"):
            native_fallback += 1
        if out.startswith("<ERROR:") or out == "<EMPTY_OUTPUT>" or out == "<INVALID_OUTPUT>":
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
        "engine": "gemma3n_omni",
        "model_name": args.model or os.getenv("GEMMA_OMNI_MODEL", "google/gemma-3n-E2B-it"),
        "workflow": gemma.mode_label(),
        "workflow_counts": {
            "audio": audio_used,
            "text_fallback": text_fallback,
        },
    }
    attempt_rate = native_attempted / limit if limit else 0.0
    success_rate = native_success / native_attempted if native_attempted else 0.0
    crash_rate = (native_attempted - native_success) / native_attempted if native_attempted else 0.0
    fallback_rate = native_fallback / limit if limit else 0.0
    reliability = {
        "native_audio_attempt_rate": attempt_rate,
        "native_audio_success_rate": success_rate,
        "native_audio_crash_rate": crash_rate,
        "fallback_rate": fallback_rate,
    }
    report["metrics"].update(reliability)
    report["num_errors"] = num_errors
    if first_error is not None:
        report["first_error"] = first_error
    if errors:
        report["errors"] = errors

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "gemma_omni_eval_report.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"model_name: {report['model_name']}")
    print(f"workflow: {report['workflow']}")
    print(f"=== DEBUG (first {args.debug_k} mismatches) ===")
    for item in mismatches[: args.debug_k]:
        print(f'audio_path: {item.get("audio_path")}')
        print(f'text: {item.get("text")}')
        print(f'expected: {json.dumps(item.get("expected"), ensure_ascii=False)}')
        print(f'pred: {json.dumps(item.get("predicted"), ensure_ascii=False)}')
        print(f'model_out: {item.get("model_out")}')


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        print("=== TRACEBACK START ===")
        traceback.print_exc()
        print("=== TRACEBACK END ===")
        raise
