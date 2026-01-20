from __future__ import annotations

from typing import Any

from tool_schema import normalize_unit


def _toolcall_to_dict(toolcall: Any) -> dict | None:
    if toolcall is None:
        return None
    if isinstance(toolcall, dict):
        return toolcall
    return toolcall.model_dump()


def _toolcall_match(expected: dict, predicted: dict) -> bool:
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
    if int(exp_args.get("precision", 2)) != int(pred_args.get("precision", 2)):
        return False

    exp_value = float(exp_args.get("value"))
    pred_value = float(pred_args.get("value"))
    return abs(exp_value - pred_value) <= 1e-6


def evaluate(samples: list[dict], preds: list[dict]) -> dict:
    total = len(samples)
    parsable = 0

    tp = fp = tn = fn = 0
    tool_required_correct = 0
    tool_call_em = 0
    tool_required_count = 0

    for sample, pred in zip(samples, preds):
        status = pred.get("status")
        tool_predicted = status == "tool"
        tool_required = bool(sample.get("tool_required"))

        if status != "invalid":
            parsable += 1

        if tool_predicted == tool_required:
            tool_required_correct += 1

        if tool_required:
            tool_required_count += 1

        if tool_predicted and tool_required:
            tp += 1
        elif tool_predicted and not tool_required:
            fp += 1
        elif not tool_predicted and not tool_required:
            tn += 1
        else:
            fn += 1

        if tool_required:
            expected = sample.get("expected_tool_call")
            predicted = _toolcall_to_dict(pred.get("toolcall"))
            if predicted is not None and expected is not None:
                if _toolcall_match(expected, predicted):
                    tool_call_em += 1

    parsable_rate = parsable / total if total else 0.0
    tool_required_accuracy = tool_required_correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    tool_call_em_rate = tool_call_em / tool_required_count if tool_required_count else 0.0

    return {
        "parsable_rate": parsable_rate,
        "tool_required_accuracy": tool_required_accuracy,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate,
        "tool_call_em": tool_call_em_rate,
    }
