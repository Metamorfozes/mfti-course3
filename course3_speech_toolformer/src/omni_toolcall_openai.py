from __future__ import annotations

import base64
import os
from pathlib import Path

_DOTENV_LOADED = False


_OMNI_PROMPT = (
    "Return either exactly NO_TOOL or a single JSON object with keys name and arguments. "
    'JSON format: {"name":"unit_convert","arguments":{"value":<float>,"from_unit":"<unit>",'
    '"to_unit":"<unit>","precision":2}}. '
    "No extra text before/after JSON. If no numeric conversion requested -> NO_TOOL."
)

_OMNI_SYSTEM_PROMPT = (
    "You are a strict tool-call formatter. Output either exactly NO_TOOL or a single JSON object with format:\n"
    '{"name":"unit_convert","arguments":{"value":<float>,"from_unit":"<unit>",'
    '"to_unit":"<unit>","precision":2}}\n'
    "Allowed units only (lowercase): mm, cm, m, km, inch, ft, yd, mile, g, kg, oz, lb, c, f, k.\n"
    "Call the tool only if the user explicitly requests a conversion (convert / what is X in Y / please convert / "
    "переведи / сколько будет). If it is a statement or unrelated, output NO_TOOL."
)

_OMNI_USER_SUFFIX = (
    "Return either NO_TOOL + short natural-language answer, or a single JSON tool call."
)


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / ".env")


class OpenAIOmniToolCaller:
    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import APIConnectionError, APITimeoutError, OpenAI
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "OpenAI omni tool caller requires the 'openai' package. Install: pip install openai"
            ) from exc

        _load_dotenv_once()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        self._APIConnectionError = APIConnectionError
        self._APITimeoutError = APITimeoutError
        self._client = OpenAI()
        self._model = model or os.getenv("OPENAI_OMNI_MODEL", "gpt-4o-mini")

    def infer_tool_call_from_audio(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio not found: {path}")

        audio_bytes = path.read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        audio_format = path.suffix.lstrip(".").lower() or "wav"

        def _call():
            return self._client.responses.create(
                model=self._model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": _OMNI_PROMPT},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": audio_format},
                            },
                        ],
                    }
                ],
            )

        try:
            response = _call()
        except (self._APIConnectionError, self._APITimeoutError):
            response = _call()

        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        try:
            for item in response.output:
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return str(content.get("text", "")).strip()
        except Exception:
            pass
        return str(response).strip()


class OmniToolCallerOpenAI:
    def __init__(self, model_name: str | None = None) -> None:
        try:
            from openai import APIConnectionError, APITimeoutError, OpenAI
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "OpenAI omni tool caller requires the 'openai' package. Install: pip install openai"
            ) from exc

        _load_dotenv_once()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        self._APIConnectionError = APIConnectionError
        self._APITimeoutError = APITimeoutError
        self._client = OpenAI()
        self._model = model_name or os.getenv("OPENAI_OMNI_MODEL", "gpt-4o-audio-preview")
        self._last_response_repr: str | None = None
        self._last_error: str | None = None

    def infer(self, audio_path: str, user_text_hint: str, lang: str) -> str:
        try:
            path = Path(audio_path)
            if not path.exists():
                raise FileNotFoundError(f"Audio not found: {path}")

            hint = (user_text_hint or "").strip()
            lang = (lang or "").strip()

            audio_bytes = path.read_bytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

            content = [{"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}]
            if hint:
                content.insert(0, {"type": "text", "text": f"Text hint: {hint}"})
            if lang:
                content.insert(0, {"type": "text", "text": f"Language: {lang}"})
            content.append({"type": "text", "text": _OMNI_USER_SUFFIX})

            def _call():
                return self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": _OMNI_SYSTEM_PROMPT}]},
                        {"role": "user", "content": content},
                    ],
                    temperature=0,
                )

            try:
                response = _call()
            except (self._APIConnectionError, self._APITimeoutError):
                response = _call()

            self._last_response_repr = repr(response)

            text = None
            try:
                content_val = response.choices[0].message.content
                if isinstance(content_val, str):
                    text = content_val
                elif isinstance(content_val, list):
                    parts = []
                    for item in content_val:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(str(item.get("text", "")))
                    text = "".join(parts)
            except Exception:
                text = None

            if text:
                return text.strip()

            self._last_error = "<EMPTY_OUTPUT>"
            return self._last_error

        except Exception as exc:
            self._last_error = f"<ERROR: {type(exc).__name__}: {exc}>"
            return self._last_error


    def last_response_repr(self) -> str | None:
        return self._last_response_repr

    def last_error(self) -> str | None:
        return self._last_error
