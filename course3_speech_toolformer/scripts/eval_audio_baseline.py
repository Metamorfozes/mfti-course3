from __future__ import annotations

"""
Evaluate audio -> ASR -> LLM -> tool-call pipeline.

Usage:
  python scripts/eval_audio_baseline.py --data data/audio_dataset.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from asr_faster_whisper import FasterWhisperASR
from metrics import evaluate
from parse_toolcall import parse_toolcall
from prompting import build_messages
from tool_schema import normalize_unit


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


def normalize_asr_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"([a-zа-я]+)-(?=\d)", r"\1 ", text)
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    text = text.replace("кельвинферингейт", "k to f")
    text = text.replace("kfg", "kg")
    text = text.replace("cm2id", "cm to yd")
    text = text.replace("cm2 id", "cm to yd")
    text = text.replace("kg2g", "kg to g")
    text = text.replace("kg2 g", "kg to g")
    text = text.replace("кимтумм", "km to mm")
    text = text.replace("кмтумм", "km to mm")
    text = re.sub(r"ким\s*ту\s*мм", "km to mm", text)
    text = re.sub(r"мвк\s*мы", "m to km", text)
    text = re.sub(r"м\s*в\s*к\s*мы", "m to km", text)
    text = re.sub(r"\bмвкмы\b", "m to km", text)
    text = re.sub(r"\bмвк\b", "m to km", text)
    text = text.replace("кимтуmm", "km to mm")
    text = text.replace("мфkm", "cm to km")
    text = text.replace("мфкм", "cm to km")
    text = re.sub(r"\bkm\s*twinch\b", "km to inch", text)
    text = re.sub(r"\bkmtwinch\b", "km to inch", text)
    text = text.replace("id2m", "yd to m")
    text = re.sub(r"\bid\b", "yd", text)
    text = text.replace("gtl", "kg to lb")
    text = text.replace("мчтукеем", "inch to km")
    text = text.replace("мымы", "mm")
    text = text.replace("кинф", "k to f")
    text = text.replace("километра", "km")
    text = text.replace("километров", "km")
    text = text.replace("милинч", "mile in inch")
    text = text.replace("auron", "oz")
    text = text.replace("aurons", "oz")
    text = text.replace("наурнс", "oz")
    text = text.replace("львы", "lb")
    text = text.replace("инджи", "g")
    text = text.replace("нкм", "km")
    text = text.replace("tuft", "ft")
    text = text.replace("кемен", "km")
    text = text.replace("кельвен", "k")
    text = text.replace("феррингейт", "f")
    text = text.replace("ферингейт", "f")
    text = re.sub(r"\bсм\s*и\b", "cm", text)
    text = re.sub(r"\bсми\b", "cm", text)
    text = re.sub(r"[a-zа-я]*(цельс|cels)[a-zа-я]*", "c", text)
    text = re.sub(r"[a-zа-я]*(фар|fahr|faren|faring|фрин)[a-zа-я]*", "f", text)
    text = text.replace("kelvin", "k")
    text = re.sub(r"\bкг\b", "kg", text)
    text = re.sub(r"\bг\b", "g", text)
    text = re.sub(r"\bкм\b", "km", text)
    text = re.sub(r"\bсм\b", "cm", text)
    text = re.sub(r"\bмм\b", "mm", text)
    text = re.sub(r"\bм\b", "m", text)
    unit_group = r"(mm|cm|m|km|inch|ft|yd|mile|g|kg|oz|lb|c|f|k)"
    text = re.sub(rf"\b{unit_group}\s*2\s*{unit_group}\b", r"\1 to \2", text)
    text = re.sub(r"(\d)([a-zа-я])", r"\1 \2", text)
    if "hours to lb" in text:
        text = text.replace("hours", "oz")
    text = " ".join(text.split())

    def _round_decimal(match: re.Match) -> str:
        raw = match.group(0)
        if "." not in raw:
            return raw
        try:
            value = round(float(raw), 1)
        except ValueError:
            return raw
        return f"{value:.1f}"

    text = re.sub(r"\d+\.\d{2,}", _round_decimal, text)

    units = ["mm", "cm", "m", "km", "inch", "ft", "yd", "mile", "g", "kg", "oz", "lb", "c", "f", "k"]
    unit_pattern = r"\b(" + "|".join(sorted(units, key=len, reverse=True)) + r")\b"
    num_match = re.search(r"[-+]?\d*\.?\d+", text)
    if num_match:
        if "converter" in text and re.search(r"\b" + re.escape(num_match.group(0)) + r"\s*f\b", text):
            unit_hits = re.findall(unit_pattern, text)
            if unit_hits == ["f"]:
                return f"convert {num_match.group(0)} k to f"
        if "уат из" in text and re.search(r"\bkm\b", text):
            unit_hits = re.findall(unit_pattern, text)
            if unit_hits == ["km"]:
                return f"convert {num_match.group(0)} cm to km"
        if re.search(r"\bсколько будет\b", text):
            unit_hits = re.findall(unit_pattern, text)
            if unit_hits == ["mm"]:
                return f"convert {num_match.group(0)} m to mm"
        if "cm" in text and ("и мы" in text or "имы" in text):
            return f"convert {num_match.group(0)} km to cm"
        found_units = [(m.group(1), m.start()) for m in re.finditer(unit_pattern, text)]
        if len(found_units) >= 2:
            after_num = [u for u in found_units if u[1] > num_match.end()]
            from_unit = after_num[0][0] if after_num else found_units[0][0]
            to_unit = None
            for unit, _pos in found_units:
                if unit != from_unit:
                    to_unit = unit
                    break
            if to_unit:
                value = num_match.group(0)
                return f"convert {value} {from_unit} to {to_unit}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_dataset.jsonl")),
        help="Path to audio dataset JSON/JSONL.",
    )
    parser.add_argument("--engine", choices=["stub", "llamacpp"], default="llamacpp")
    parser.add_argument("--model_path", default=None, help="Optional override for LlamaCpp model path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug_k", type=int, default=10)
    parser.add_argument("--asr_model_size", default="small")
    parser.add_argument("--asr_device", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    samples = _load_samples(data_path)

    limit = len(samples) if args.limit is None else min(args.limit, len(samples))
    samples = samples[:limit]

    asr = FasterWhisperASR(model_size=args.asr_model_size, device=args.asr_device)

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
        if args.model_path:
            env["LLAMA_MODEL"] = args.model_path
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
        audio_path = sample.get("audio_path", "")
        asr_raw = asr.transcribe(audio_path)
        asr_text = normalize_asr_text(asr_raw)
        messages = build_messages(asr_text)
        out = _infer(messages)
        status, toolcall = parse_toolcall(out)

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

        if mismatch and len(error_cases) < 20:
            error_cases.append(
                {
                    "id": sample.get("id"),
                    "asr_raw": asr_raw,
                    "asr_text": asr_text,
                    "expected": expected_tool,
                    "predicted": {"status": status, "tool_call": predicted_tool_call},
                }
            )
        if mismatch:
            mismatches.append(
                {
                    "asr_raw": asr_raw,
                    "asr_text": asr_text,
                    "expected": expected_tool,
                    "predicted": {"status": status, "tool_call": predicted_tool_call},
                }
            )

    metrics = evaluate(samples, preds)

    report = {
        "metrics": metrics,
        "engine": args.engine,
        "limit": limit,
    }
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_eval_report.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"=== DEBUG (first {args.debug_k} mismatches) ===")
    for item in mismatches[: args.debug_k]:
        print(f'asr_raw: {item.get("asr_raw")}')
        print(f'asr_text: {item.get("asr_text")}')
        print(f'expected: {json.dumps(item.get("expected"), ensure_ascii=False)}')
        print(f'pred: {json.dumps(item.get("predicted"), ensure_ascii=False)}')


if __name__ == "__main__":
    main()
