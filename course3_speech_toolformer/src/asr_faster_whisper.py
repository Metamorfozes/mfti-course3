from __future__ import annotations

"""
Minimal Faster-Whisper ASR wrapper.
"""

from typing import Optional

from faster_whisper import WhisperModel


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class FasterWhisperASR:
    def __init__(self, model_size: str = "small", device: Optional[str] = None) -> None:
        device = device or _default_device()
        self.model = WhisperModel(model_size, device=device)

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self.model.transcribe(audio_path)
        text = "".join(segment.text for segment in segments).strip()
        return text
