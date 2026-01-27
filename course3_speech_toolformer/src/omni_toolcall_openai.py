from __future__ import annotations

import base64
import os
from pathlib import Path


_OMNI_PROMPT = (
    "Return either exactly NO_TOOL or a single JSON object with keys name and arguments. "
    'JSON format: {"name":"unit_convert","arguments":{"value":<float>,"from_unit":"<unit>",'
    '"to_unit":"<unit>","precision":2}}. '
    "No extra text before/after JSON. If no numeric conversion requested -> NO_TOOL."
)


class OpenAIOmniToolCaller:
    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import APIConnectionError, APITimeoutError, OpenAI
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "OpenAI omni tool caller requires the 'openai' package. Install: pip install openai"
            ) from exc

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
