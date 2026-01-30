from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tool_schema import ToolCall, normalize_unit

_SYSTEM_PROMPT = (
    "You are a tool-calling assistant.\n"
    "Output must be either exactly NO_TOOL or a single JSON object with keys name and arguments.\n"
    'JSON format: {"name":"unit_convert","arguments":{"value":<float>,"from_unit":"<unit>","to_unit":"<unit>","precision":2}}\n'
    "No extra text before or after the JSON.\n"
    "Use only canonical unit symbols: c,f,k,kg,g,lb,oz,km,m,cm,mm,inch,mile,yd,ft.\n"
)


class AudioMelBundle:
    def __init__(self, audio_mel: Any, audio_mel_mask: Any) -> None:
        self.audio_mel = audio_mel
        self.audio_mel_mask = audio_mel_mask

    def to(self, device: Any = None, dtype: Any = None) -> "AudioMelBundle":
        self.audio_mel = self.audio_mel.to(device=device, dtype=dtype or self.audio_mel.dtype)
        self.audio_mel_mask = self.audio_mel_mask.to(device=device)
        return self


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


def _format_prompt(messages: list[dict]) -> str:
    system = messages[0]["content"]
    user = messages[1]["content"]
    user_prompt = f"User: {user}\nAnswer:"
    return system + "\n" + user_prompt


def _strip_generation(prompt: str, text: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text.strip()


def iter_meta_params(model: Any) -> list[tuple[str, tuple[int, ...] | None, Any]]:
    out: list[tuple[str, tuple[int, ...] | None, Any]] = []
    for name, param in model.named_parameters():
        if getattr(param, "device", None) is not None and param.device.type == "meta":
            out.append((name, getattr(param, "shape", None), getattr(param, "dtype", None)))
    return out


def iter_meta_buffers(model: Any) -> list[tuple[str, tuple[int, ...] | None, Any]]:
    out: list[tuple[str, tuple[int, ...] | None, Any]] = []
    for name, buf in model.named_buffers():
        if getattr(buf, "device", None) is not None and buf.device.type == "meta":
            out.append((name, getattr(buf, "shape", None), getattr(buf, "dtype", None)))
    return out


def describe_meta_state(model: Any) -> str:
    bad_params = iter_meta_params(model)
    bad_bufs = iter_meta_buffers(model)
    if not bad_params and not bad_bufs:
        return "meta params=0 meta buffers=0"
    return f"meta params={len(bad_params)} meta buffers={len(bad_bufs)}"


def list_meta_params_and_buffers(model: Any) -> tuple[list[str], list[str]]:
    meta_params = [name for name, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [name for name, b in model.named_buffers() if b.device.type == "meta"]
    return meta_params, meta_buffers


def print_meta_report(tag: str, model: Any) -> None:
    meta_params, meta_buffers = list_meta_params_and_buffers(model)
    print(f"[Gemma3n][{tag}] meta params: {len(meta_params)}")
    if meta_params:
        print(f"[Gemma3n][{tag}] params: {meta_params[:30]}")
    print(f"[Gemma3n][{tag}] meta buffers: {len(meta_buffers)}")
    if meta_buffers:
        print(f"[Gemma3n][{tag}] buffers: {meta_buffers[:30]}")


def materialize_meta_buffers_(model: Any, device: str) -> int:
    import torch

    replaced = 0
    for module in model.modules():
        for key, buf in list(module._buffers.items()):
            if buf is not None and getattr(buf, "is_meta", False):
                new_buf = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
                module._buffers[key] = new_buf
                replaced += 1
    return replaced


def _find_meta_tensors(value: Any, path: str) -> list[tuple[str, Any]]:
    import torch
    from torch.nn import Parameter

    found: list[tuple[str, Any]] = []
    if isinstance(value, (torch.Tensor, Parameter)):
        if getattr(value, "is_meta", False):
            found.append((path, value))
        return found
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_find_meta_tensors(item, f"{path}[{key!r}]"))
        return found
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            found.extend(_find_meta_tensors(item, f"{path}[{idx}]"))
        return found
    return found


def list_meta_attr_tensors(
    model: Any, prefix_filter: tuple[str, ...] = ("audio_tower", "vision_tower")
) -> list[str]:
    meta_paths: list[str] = []
    skip = {"_parameters", "_buffers", "_modules"}
    for module_name, module in model.named_modules():
        if prefix_filter and not module_name.startswith(prefix_filter):
            continue
        for attr_name, value in module.__dict__.items():
            if attr_name in skip:
                continue
            hits = _find_meta_tensors(value, f"{module_name}.{attr_name}" if module_name else attr_name)
            meta_paths.extend([path for path, _ in hits])
    return meta_paths


def materialize_meta_attr_tensors_(
    model: Any, device: str, prefix_filter: tuple[str, ...] = ("audio_tower", "vision_tower")
) -> int:
    import torch

    replaced = 0
    skip = {"_parameters", "_buffers", "_modules"}
    for module_name, module in model.named_modules():
        if prefix_filter and not module_name.startswith(prefix_filter):
            continue
        for attr_name, value in list(module.__dict__.items()):
            if attr_name in skip:
                continue
            hits = _find_meta_tensors(value, f"{module_name}.{attr_name}" if module_name else attr_name)
            if not hits:
                continue

            def materialize(val: Any, full_path: str) -> Any:
                if isinstance(val, torch.Tensor) and getattr(val, "is_meta", False):
                    fill = torch.ones if "scale" in full_path else torch.zeros
                    return fill(val.shape, dtype=val.dtype, device=device)
                if isinstance(val, dict):
                    return {k: materialize(v, f"{full_path}[{k!r}]") for k, v in val.items()}
                if isinstance(val, list):
                    return [materialize(v, f"{full_path}[{i}]") for i, v in enumerate(val)]
                if isinstance(val, tuple):
                    return tuple(materialize(v, f"{full_path}[{i}]") for i, v in enumerate(val))
                return val

            module.__dict__[attr_name] = materialize(
                value, f"{module_name}.{attr_name}" if module_name else attr_name
            )
            for path, _ in hits:
                if "scale" in path:
                    replaced += 1
                else:
                    replaced += 1
    return replaced


def fix_post_layer_scale_(model: Any, device: str) -> int:
    import torch

    n = 0
    for name, module in model.named_modules():
        if not hasattr(module, "post_layer_scale"):
            continue
        v = getattr(module, "post_layer_scale")
        if torch.is_tensor(v):
            if getattr(v, "is_meta", False):
                setattr(module, "post_layer_scale", torch.ones(v.shape, dtype=v.dtype, device=device))
                n += 1
            elif str(v.device) != device:
                setattr(module, "post_layer_scale", v.to(device))
                n += 1
        else:
            try:
                setattr(module, "post_layer_scale", torch.tensor(v, device=device, dtype=torch.float16))
                n += 1
            except Exception:
                pass
    return n


class Gemma3nOmniToolCaller:
    def __init__(
        self,
        model_name: str | None = None,
        asr_model_size: str = "small",
        max_new_tokens: int = 128,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Gemma3nOmniToolCaller requires transformers + torch. Install: pip install transformers torch"
            ) from exc

        self._torch = torch
        os.environ.setdefault("ACCELERATE_USE_META_DEVICE", "0")
        self._model_name = model_name or os.getenv("GEMMA_OMNI_MODEL", "google/gemma-3n-E2B-it")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self._processor = AutoProcessor.from_pretrained(
            self._model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            token=hf_token,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=False,
            dtype=dtype,
        )
        self._model.to(self._device)
        self._model.eval()
        for name, module in self._model.named_modules():
            if hasattr(module, "post_layer_scale"):
                v = getattr(module, "post_layer_scale")
                if self._torch.is_tensor(v):
                    print(
                        f"[Gemma3n] found post_layer_scale at {name}: type={type(v)} "
                        f"device={v.device} is_meta={getattr(v, 'is_meta', False)} "
                        f"shape={tuple(v.shape)} dtype={v.dtype}"
                    )
                else:
                    print(f"[Gemma3n] found post_layer_scale at {name}: type={type(v)} value={v}")
        print_meta_report("after_to_device_before_fix", self._model)
        if self._device == "cuda":
            replaced = materialize_meta_buffers_(self._model, self._device)
            print(f"[Gemma3n] materialized_meta_buffers={replaced}")
        print_meta_report("after_fix", self._model)
        meta_attr_names = list_meta_attr_tensors(self._model)
        print(f"[Gemma3n] meta_attr_tensors={len(meta_attr_names)}")
        if meta_attr_names:
            print(f"[Gemma3n] meta_attr_tensors_sample={meta_attr_names[:30]}")
        if self._device == "cuda":
            fixed = fix_post_layer_scale_(self._model, "cuda")
            print(f"[Gemma3n] fixed_post_layer_scale={fixed}")
            replaced_attr = materialize_meta_attr_tensors_(self._model, self._device)
            print(f"[Gemma3n] materialized_meta_attr_tensors={replaced_attr}")
            meta_attr_names = list_meta_attr_tensors(self._model)
            print(f"[Gemma3n] meta_attr_tensors_after_fix={len(meta_attr_names)}")
        meta_params, meta_buffers = list_meta_params_and_buffers(self._model)
        if self._device == "cuda":
            self._audio_supported = True
        elif meta_params or meta_buffers:
            self._audio_supported = False
            print("[Gemma3n] supports_audio=False reason=meta_device")
        else:
            self._audio_supported = None
        if self._audio_supported is not False:
            self._ensure_no_meta("after load")
        self._asr_model_size = asr_model_size
        self._max_new_tokens = max_new_tokens
        self._asr = None
        self._mode_label = "Gemma omni (audio)"
        self._last_error: str | None = None
        self._last_used_audio: bool | None = None

    def mode_label(self) -> str:
        return self._mode_label

    def last_error(self) -> str | None:
        return self._last_error
    
    def last_used_audio(self) -> bool | None:
        return self._last_used_audio

    def _load_audio(self, audio_path: str) -> tuple[Any, int]:
        try:
            import soundfile as sf
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Audio loading requires soundfile. Install: pip install soundfile"
            ) from exc
        audio, sr = sf.read(audio_path)
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)

    def _ensure_asr(self) -> None:
        if self._asr is None:
            from asr_faster_whisper import FasterWhisperASR

            self._asr = FasterWhisperASR(model_size=self._asr_model_size)

    def _build_messages(self, lang: str, text_hint: str, transcript: str | None) -> list[dict]:
        user_parts = []
        if lang:
            user_parts.append(f"Language: {lang}")
        if text_hint:
            user_parts.append(f"Text hint: {text_hint}")
        if transcript:
            user_parts.append(f"Transcript: {transcript}")
        else:
            user_parts.append("Audio provided.")
        user_text = "\n".join(user_parts).strip()
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def _generate_text(self, prompt: str, audio: Any | None = None, sampling_rate: int | None = None) -> str:
        print(f"[Gemma3n] device={self._device} attempting_audio={audio is not None}")
        if audio is None:
            inputs = self._processor(text=prompt, return_tensors="pt")
        else:
            try:
                processor_text = prompt
                if processor_text is None:
                    processor_text = ""
                if isinstance(processor_text, list) and len(processor_text) == 0:
                    processor_text = ""
                print(
                    f"[Gemma3n][AUDIO] processor_text_is_none={processor_text is None} "
                    f"type={type(processor_text)}"
                )
                inputs = self._processor(
                    text=processor_text,
                    audio=audio,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )
                if "input_features" in inputs and "input_features_mask" in inputs:
                    inputs["input_features"] = AudioMelBundle(
                        audio_mel=inputs["input_features"],
                        audio_mel_mask=inputs["input_features_mask"],
                    )
                    del inputs["input_features_mask"]
            except Exception as e:
                import traceback
                print(f"[Gemma3n][AUDIO_FAIL] {type(e).__name__}: {e}")
                print("=== GEMMA AUDIO TRACEBACK START ===")
                print(traceback.format_exc())
                print("=== GEMMA AUDIO TRACEBACK END ===")
                raise

        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)
        else:
            for key, value in inputs.items():
                if self._torch.is_tensor(value):
                    inputs[key] = value.to(self._device)

        if audio is not None:
            print("[Gemma3n][AUDIO_INPUTS] keys=", list(inputs.keys()))
            for key, value in inputs.items():
                if self._torch.is_tensor(value):
                    print(
                        f"[Gemma3n][AUDIO_INPUTS] {key}: Tensor shape={tuple(value.shape)} "
                        f"dtype={value.dtype} device={value.device}"
                    )
                else:
                    preview = str(value)
                    print(
                        f"[Gemma3n][AUDIO_INPUTS] {key}: type={type(value)} "
                        f"value_preview={preview[:80]}"
                    )
                if hasattr(value, "audio_mel_mask"):
                    print(f"[Gemma3n][AUDIO_INPUTS] {key} has audio_mel_mask")
            if "input_features" in inputs:
                print(f"[Gemma3n][AUDIO_INPUTS] input_features type={type(inputs['input_features'])}")
            if "input_features_mask" in inputs and self._torch.is_tensor(inputs["input_features_mask"]):
                mask = inputs["input_features_mask"]
                print(
                    f"[Gemma3n][AUDIO_INPUTS] input_features_mask shape={tuple(mask.shape)} "
                    f"dtype={mask.dtype} device={mask.device}"
                )

        self._model.to(self._device)
        self._ensure_no_meta("before generate")
        self._ensure_inputs_device(inputs)
        try:
            with self._torch.inference_mode():
                if audio is not None:
                    audio_bundle = inputs.get("input_features")
                    if not hasattr(audio_bundle, "audio_mel") or not hasattr(audio_bundle, "audio_mel_mask"):
                        raise RuntimeError("Missing audio_mel/audio_mel_mask on input_features bundle.")
                    mel = audio_bundle.audio_mel
                    mask = audio_bundle.audio_mel_mask
                    if not self._torch.is_tensor(mel):
                        raise RuntimeError(f"input_features is not a Tensor: {type(mel)}")
                    mel = mel.to(self._device)
                    mask = mask.to(self._device).to(dtype=self._torch.bool)
                    if self._device == "cuda":
                        mel = mel.to(dtype=self._torch.float16)
                    audio_bundle.audio_mel = mel
                    audio_bundle.audio_mel_mask = mask

                    audio_out = self._model.model.get_audio_features(
                        input_features=mel,
                        input_features_mask=~mask,
                        return_dict=True,
                    )
                    audio_proj = audio_out[0] if isinstance(audio_out, tuple) else audio_out.last_hidden_state
                    n_audio = int(audio_proj.shape[1])
                    print(f"[Gemma3n][AUDIO] n_audio={n_audio}")

                    tok = self._processor.tokenizer
                    audio_token = tok.special_tokens_map.get("audio_token", "<audio_soft_token>")
                    boa = tok.special_tokens_map.get("boa_token", "<start_of_audio>")
                    eoa = tok.special_tokens_map.get("eoa_token", "<end_of_audio>")
                    print(f"[Gemma3n][AUDIO] audio_token={audio_token} boa={boa} eoa={eoa}")

                    audio_placeholders = " ".join([audio_token] * n_audio)
                    prompt_with_audio = f"User: {boa} {audio_placeholders} {eoa}\nAssistant:"
                    enc = tok(prompt_with_audio, return_tensors="pt", add_special_tokens=True)
                    input_ids = enc.get("input_ids")
                    if input_ids is None:
                        raise RuntimeError("Missing input_ids from tokenizer.")
                    input_ids = input_ids.to(self._device)
                    inputs["input_ids"] = input_ids
                    attention_mask = enc.get("attention_mask", None)
                    if attention_mask is None:
                        attention_mask = self._torch.ones_like(input_ids, dtype=self._torch.long)
                    else:
                        attention_mask = attention_mask.to(self._device)
                    token_type_ids = enc.get("token_type_ids", None)
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(self._device)

                    audio_token_id = tok.convert_tokens_to_ids(audio_token)
                    if audio_token_id is None or audio_token_id < 0:
                        raise RuntimeError(f"audio_token not in vocab: {audio_token}")
                    audio_token_count = (input_ids == audio_token_id).sum().item()
                    print(f"[Gemma3n][AUDIO] audio_token_count={audio_token_count}")
                    if audio_token_count != n_audio:
                        decoded = tok.decode(input_ids[0])
                        print(f"[Gemma3n][AUDIO] decoded_with_audio={decoded[:200]}")

                    model_inputs = {"input_ids": input_ids, "input_features": audio_bundle}
                    if attention_mask is not None:
                        model_inputs["attention_mask"] = attention_mask
                    if token_type_ids is not None:
                        model_inputs["token_type_ids"] = token_type_ids
                    outputs = self._model.generate(
                        **model_inputs,
                        max_new_tokens=min(self._max_new_tokens, 96),
                        do_sample=False,
                    )
                else:
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=min(self._max_new_tokens, 96),
                        do_sample=False,
                    )
        except Exception as e:
            if audio is not None:
                print(f"[Gemma3n][AUDIO_FAIL] {type(e).__name__}: {e}")
                import traceback

                print("=== GEMMA AUDIO TRACEBACK START ===")
                print(traceback.format_exc())
                print("=== GEMMA AUDIO TRACEBACK END ===")
            raise

        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            outputs = outputs[:, input_ids.shape[1] :]

        decoder = getattr(self._processor, "decode", None)
        if callable(decoder):
            text = decoder(outputs[0], skip_special_tokens=True)
        else:
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return text.strip()

    def _postprocess_output(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            self._last_error = "<EMPTY_OUTPUT>"
            return self._last_error

        first_line = ""
        for line in raw.splitlines():
            if line.strip():
                first_line = line.strip()
                break
        if first_line.lower().startswith("no_tool"):
            return "NO_TOOL"

        payload = None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = _extract_first_json(raw)

        if not isinstance(payload, dict):
            self._last_error = "<INVALID_OUTPUT>"
            return self._last_error

        if set(payload.keys()) != {"name", "arguments"}:
            self._last_error = "<INVALID_OUTPUT>"
            return self._last_error

        try:
            payload = _normalize_tool_payload(payload)
            ToolCall.model_validate(payload)
        except Exception:
            self._last_error = "<INVALID_OUTPUT>"
            return self._last_error

        return json.dumps(payload, ensure_ascii=False)

    def _ensure_no_meta(self, when: str) -> None:
        bad_params = iter_meta_params(self._model)
        bad_bufs = iter_meta_buffers(self._model)
        if bad_params or bad_bufs:
            param_names = [name for name, _, _ in bad_params[:10]]
            buf_names = [name for name, _, _ in bad_bufs[:10]]
            raise RuntimeError(
                f"Meta tensors remain ({when}). "
                f"params={param_names} buffers={buf_names}"
            )
    
    def _ensure_inputs_device(self, inputs: Any) -> None:
        if not isinstance(inputs, dict):
            return
        bad = []
        for key, value in inputs.items():
            if hasattr(value, "device"):
                if value.device.type == "meta":
                    bad.append(f"{key}:meta")
                elif value.device.type != self._device:
                    bad.append(f"{key}:{value.device.type}")
        if bad:
            raise RuntimeError(f"Inputs not on {self._device}: {bad}")

    def infer(self, audio_path: str, lang: str = "", text_hint: str = "") -> str:
        self._last_error = None
        self._last_used_audio = None
        meta = {
            "native_audio_attempted": False,
            "native_audio_success": False,
            "native_audio_error": None,
            "used_fallback_asr": False,
            "transcript": None,
        }
        path = Path(audio_path)
        if not path.exists():
            self._last_error = f"<ERROR: FileNotFound: {path}>"
            return {"text": self._last_error, "meta": meta}
        self._last_audio_path = str(path)

        try:
            audio, sr = self._load_audio(str(path))
        except Exception as exc:
            self._last_error = f"<ERROR: AudioLoad: {exc}>"
            return {"text": self._last_error, "meta": meta}

        reason = ""
        should_try_audio = self._audio_supported is not False
        if self._device == "cuda":
            should_try_audio = True
        if should_try_audio:
            try:
                meta["native_audio_attempted"] = True
                messages = self._build_messages(lang, text_hint, transcript=None)
                prompt = _format_prompt(messages)
                print("[Gemma3n] attempting AUDIO generate path")
                text = self._generate_text(prompt, audio=audio, sampling_rate=sr)
                self._audio_supported = True
                self._mode_label = "Gemma omni (audio)"
                self._last_used_audio = True
                meta["native_audio_success"] = True
                return {"text": self._postprocess_output(_strip_generation(prompt, text)), "meta": meta}
            except TypeError as exc:
                msg = str(exc)
                if "audio" in msg or "unexpected keyword" in msg:
                    print(f"[Gemma3n] audio path failed -> fallback to ASR. error={msg}")
                    reason = "processor_signature"
                    self._audio_supported = False
                    print(f"[Gemma3n] supports_audio={self._audio_supported} reason={reason}")
                    meta["native_audio_success"] = False
                    meta["native_audio_error"] = msg
                else:
                    self._last_error = f"<ERROR: AudioInfer: {exc}>"
                    return {"text": self._last_error, "meta": meta}
            except ValueError as exc:
                msg = str(exc)
                if "audio" in msg or "input_features" in msg or "not supported" in msg:
                    print(f"[Gemma3n] audio path failed -> fallback to ASR. error={msg}")
                    reason = "processor_input_features"
                    self._audio_supported = False
                    print(f"[Gemma3n] supports_audio={self._audio_supported} reason={reason}")
                    meta["native_audio_success"] = False
                    meta["native_audio_error"] = msg
                else:
                    self._last_error = f"<ERROR: AudioInfer: {exc}>"
                    return {"text": self._last_error, "meta": meta}
            except Exception as exc:
                msg = str(exc)
                if "device meta" in msg or "expected device cpu" in msg:
                    # Some remote-code audio paths still use meta tensors on CPU setups.
                    meta_error = (
                        f"<ERROR: AudioInfer: META_DEVICE: {describe_meta_state(self._model)}>"
                    )
                    self._last_error = meta_error
                    reason = "meta_device"
                    self._audio_supported = False
                    print(f"[Gemma3n] supports_audio={self._audio_supported} reason={reason}")
                    meta["native_audio_success"] = False
                    meta["native_audio_error"] = msg
                else:
                    self._last_error = f"<ERROR: AudioInfer: {exc}>"
                    meta["native_audio_success"] = False
                    meta["native_audio_error"] = msg
                    return {"text": self._last_error, "meta": meta}

        if self._audio_supported is False:
            print(f"[Gemma3n] fallback->ASR. reason={reason or 'audio_unsupported'}")

        try:
            self._ensure_asr()
            transcript = self._asr.transcribe(str(path))
            print(f"[Gemma3n][AUDIO_FALLBACK] transcribed_text={transcript!r}")
            meta["used_fallback_asr"] = True
            meta["transcript"] = transcript
        except Exception as exc:
            self._last_error = f"<ERROR: ASR: {exc}>"
            return {"text": self._last_error, "meta": meta}

        try:
            messages = self._build_messages(lang, text_hint, transcript=transcript)
            prompt = _format_prompt(messages)
            text = self._generate_text(prompt)
            self._mode_label = "Gemma text (via ASR transcript)"
            self._last_used_audio = False
            return {"text": self._postprocess_output(_strip_generation(prompt, text)), "meta": meta}
        except Exception as exc:
            self._last_error = f"<ERROR: TextInfer: {exc}>"
            return {"text": self._last_error, "meta": meta}
