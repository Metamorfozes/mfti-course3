from __future__ import annotations

"""
Make an audio dataset from a text dataset using local TTS.

Usage:
  python scripts/make_audio_dataset.py --input data/text_dataset.json --out data/audio_dataset.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import multiprocessing


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


def _synthesize_one(text: str, audio_path: str, rate: int, volume: float) -> None:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    engine.save_to_file(text, audio_path)
    engine.runAndWait()


def _fix_mojibake(text: str) -> str:
    # Heuristic fix for mojibake like "РїРѕР¶..." in RU strings.
    if text.count("Р") + text.count("Ð") + text.count("Ñ") < 2:
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "text_dataset.json")),
        help="Path to text dataset JSON/JSONL.",
    )
    parser.add_argument(
        "--out",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_dataset.jsonl")),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--audio_dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio")),
        help="Directory to store WAV files.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    samples = _load_samples(input_path)

    limit = len(samples) if args.limit is None else min(args.limit, len(samples))
    samples = samples[:limit]

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    tts_rate = 180
    tts_volume = 1.0
    mp_ctx = multiprocessing.get_context("spawn")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(samples, start=1):
            sample_id = sample.get("id", idx)
            text = _fix_mojibake(sample.get("text", ""))
            audio_path = audio_dir / f"{sample_id}.wav"
            success = False
            for _ in range(2):
                proc = mp_ctx.Process(
                    target=_synthesize_one,
                    args=(text, str(audio_path), tts_rate, tts_volume),
                )
                proc.start()
                proc.join(timeout=30)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    success = True
                    break
            if not success:
                raise RuntimeError(f"TTS failed for id={sample_id} text={text}")

            record = {
                "id": sample_id,
                "text": text,
                "audio_path": str(audio_path),
                "lang": sample.get("lang"),
                "tool_required": sample.get("tool_required"),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
