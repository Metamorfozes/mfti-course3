from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        print(f"import error: {exc}")
        return 1

    os.environ.setdefault("ACCELERATE_USE_META_DEVICE", "0")
    model_name = os.getenv("GEMMA_OMNI_MODEL", "google/gemma-3n-E2B-it")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    processor = AutoProcessor.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()

    audio_path = Path(__file__).resolve().parents[1] / "data" / "audio" / "1.wav"
    if not audio_path.exists():
        print(f"audio not found: {audio_path}")
        return 1

    try:
        import soundfile as sf
    except Exception as exc:
        print(f"soundfile error: {exc}")
        return 1

    audio, sr = sf.read(str(audio_path))
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = audio.mean(axis=1)

    prompt = "User: Audio provided.
Answer:"

    try:
        inputs = processor(text=prompt, audio=audio, sampling_rate=sr, return_tensors="pt")
    except Exception as exc:
        msg = str(exc)
        print(f"processor error: {msg}")
        if "meta" in msg.lower():
            print("META_ERROR")
            return 2
        return 1

    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to("cpu")
        if hasattr(inputs[key], "device"):
            print(f"input {key} device: {inputs[key].device}")

    try:
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=64)
    except Exception as exc:
        msg = str(exc)
        print(f"generate error: {msg}")
        if "meta" in msg.lower():
            print("META_ERROR")
            return 2
        return 1

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
