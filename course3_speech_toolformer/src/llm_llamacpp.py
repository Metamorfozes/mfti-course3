from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def _normalize_ru_user_text(text: str) -> str:
    # Lightweight normalization for RU prompts to improve tool-call parsing.
    if not re.search(r"[А-Яа-яЁё]", text or ""):
        return text
    normalized = text.lower()
    normalized = re.sub(r"\b(пожалуйста|плиз)\b", "", normalized)
    # Simple token-level replacements for common RU units/verbs.
    replacements = {
        "переведи": "convert",
        "перевести": "convert",
        "переведите": "convert",
        "кельвин": "k",
        "кельвина": "k",
        "кельвинов": "k",
        "цельсия": "c",
        "цельсий": "c",
        "цельсию": "c",
        "фаренгейт": "f",
        "фаренгейта": "f",
        "фаренгейту": "f",
        "кг": "kg",
        "килограмм": "kg",
        "килограмма": "kg",
        "килограммов": "kg",
        "г": "g",
        "гр": "g",
        "грамм": "g",
        "грамма": "g",
        "граммов": "g",
        "фунт": "lb",
        "фунта": "lb",
        "фунтов": "lb",
        "мм": "mm",
        "см": "cm",
        "км": "km",
        "м": "m",
        "метр": "m",
        "метра": "m",
        "метров": "m",
    }
    for src, dst in replacements.items():
        normalized = re.sub(rf"\b{re.escape(src)}\b", dst, normalized)
    # Replace standalone "в" as a connector ("to") without touching other words.
    normalized = re.sub(r"(?<!\w)в(?!\w)", "to", normalized)
    return " ".join(normalized.split())


def _should_allow_unit_convert(text: str) -> bool:
    # Minimal intent gate to reduce false tool calls.
    if not text:
        return False
    lowered = text.lower()
    if re.search(r"\b(переведи|перевести|переведите|convert|convertir)\b", lowered):
        return True
    if "what is" in lowered and " in " in lowered:
        return True
    if re.search(r"\d", lowered) and re.search(r"\b\d[\d\s.,]*\s*\w+\s+in\s+\w+\b", lowered):
        return True
    has_to = re.search(r"(?<!\w)в(?!\w)", lowered) or re.search(r"\bto\b", lowered)
    if re.search(r"\bсколько\b", lowered) and has_to:
        return True
    return bool(has_to)


def _extract_first_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and "```" in cleaned[3:]:
        first_break = cleaned.find("\n")
        if first_break != -1:
            cleaned = cleaned[first_break + 1 :]
        end_fence = cleaned.rfind("```")
        if end_fence != -1:
            cleaned = cleaned[:end_fence]
        cleaned = cleaned.strip()
    start = 0
    while True:
        start = cleaned.find("{", start)
        if start == -1:
            return cleaned
        depth = 0
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    try:
                        obj = json.loads(candidate)
                        return json.dumps(obj, ensure_ascii=False)
                    except json.JSONDecodeError:
                        break
        start += 1


class LlamaCppRunner:
    def __init__(
        self,
        llama_bin: str,
        model_path: str,
        ctx: int = 2048,
        gpu_layers: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: int = 60,
    ) -> None:
        self.llama_bin = llama_bin
        self.model_path = model_path
        self.ctx = ctx
        self.gpu_layers = gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def infer(self, messages: list[dict]) -> str:
        system_text = ""
        user_text = ""
        raw_user_text = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                raw_user_text = content
                user_text = _normalize_ru_user_text(content)
        if not _should_allow_unit_convert(raw_user_text):
            return "NO_TOOL"
        prompt = f"{system_text}\n\nUser: {user_text}\nAnswer:"
        cmd = [
            self.llama_bin,
            "-m",
            self.model_path,
            "--ctx-size",
            "1024",
            "--n-gpu-layers",
            str(self.gpu_layers),
            "--temp",
            str(self.temperature),
            "--n-predict",
            "64",
            "--simple-io",
            "--no-display-prompt",
            "--no-show-timings",
            "--no-conversation",
            "--single-turn",
            "--prompt",
            prompt,
        ]
        cmd_line = " ".join(cmd)
        try:
            proc = subprocess.Popen(
                cmd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            stdout_tail = (stdout or "")[-800:]
            stderr_tail = (stderr or "")[-800:]
            rc = proc.returncode
            raise RuntimeError(
                f"llama-cli timed out after {self.timeout} seconds; "
                f"cmd: {cmd_line} returncode: {rc} stdout: {stdout_tail} stderr: {stderr_tail}"
            ) from exc
        if proc.returncode != 0:
            stdout_tail = (stdout or "")[-800:]
            stderr_tail = (stderr or "")[-800:]
            raise RuntimeError(
                f"llama-cli failed (code {proc.returncode}); "
                f"cmd: {cmd_line} returncode: {proc.returncode} "
                f"stdout: {stdout_tail} stderr: {stderr_tail}"
            )
        return _extract_first_json(stdout)
