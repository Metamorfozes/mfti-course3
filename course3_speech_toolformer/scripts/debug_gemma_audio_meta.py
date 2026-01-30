from __future__ import annotations

import os
from pathlib import Path


def _list_meta(model) -> tuple[list[str], list[str]]:
    meta_params = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    meta_bufs = [name for name, b in model.named_buffers() if b.device.type == "meta"]
    return meta_params, meta_bufs


def _print_meta(tag: str, model) -> None:
    meta_params, meta_bufs = _list_meta(model)
    print(f"[{tag}] meta params: {len(meta_params)}")
    if meta_params:
        print("  params:", meta_params[:20])
    print(f"[{tag}] meta buffers: {len(meta_bufs)}")
    if meta_bufs:
        print("  bufs:", meta_bufs[:20])


def main() -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        raise RuntimeError("Requires transformers + torch") from exc

    os.environ.setdefault("ACCELERATE_USE_META_DEVICE", "0")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    model_name = os.getenv("GEMMA_OMNI_MODEL", "google/gemma-3n-E2B-it")

    processor = AutoProcessor.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
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

    _print_meta("after load", model)

    audio_path = Path(__file__).resolve().parents[1] / "data" / "audio" / "1.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("Requires soundfile") from exc

    audio, sr = sf.read(str(audio_path))
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = audio.mean(axis=1)

    prompt = (
        "You are a tool-calling assistant.
"
        "Output must be either exactly NO_TOOL or a single JSON object with keys name and arguments.
"
        'JSON format: {"name":"unit_convert","arguments":{"value":<float>,"from_unit":"<unit>","to_unit":"<unit>","precision":2}}
'
        "No extra text before or after the JSON.
"
        "Use only canonical unit symbols: c,f,k,kg,g,lb,oz,km,m,cm,mm,inch,mile,yd,ft.
"
        "User: Audio provided.
"
        "Answer:"
    )

    _print_meta("before processor", model)
    inputs = processor(text=prompt, audio=audio, sampling_rate=sr, return_tensors="pt")
    _print_meta("after processor", model)

    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to("cpu")
        if hasattr(inputs[key], "device"):
            print(f"input {key} device: {inputs[key].device}")

    try:
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=128)
        text = processor.decode(outputs[0], skip_special_tokens=True)
        print("output:", text[:500])
    except Exception as exc:
        print(f"generate error: {exc}")
        _print_meta("on error", model)


if __name__ == "__main__":
    main()
