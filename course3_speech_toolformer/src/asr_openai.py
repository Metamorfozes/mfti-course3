from __future__ import annotations

import os
from pathlib import Path


class OpenAIASR:
    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import APIConnectionError, APITimeoutError, OpenAI
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "OpenAI ASR requires the 'openai' package. Install: pip install openai"
            ) from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")

        self._APIConnectionError = APIConnectionError
        self._APITimeoutError = APITimeoutError
        self._client = OpenAI()
        self._model = model or os.getenv("OPENAI_ASR_MODEL", "gpt-4o-mini-transcribe")

    def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio not found: {path}")

        def _call():
            with path.open("rb") as audio_f:
                return self._client.audio.transcriptions.create(model=self._model, file=audio_f)

        try:
            response = _call()
        except (self._APIConnectionError, self._APITimeoutError):
            response = _call()

        text = getattr(response, "text", None)
        if text is None:
            try:
                text = response["text"]
            except Exception:
                text = str(response)
        return (text or "").strip()
