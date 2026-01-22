from __future__ import annotations

"""
Evaluate ASR WER on an audio dataset.

Usage:
  python scripts/eval_asr_baseline.py --data data/audio_dataset.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from jiwer import wer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from asr_faster_whisper import FasterWhisperASR


def _normalize_for_wer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"([a-zа-яё]+)\s*2\s*([a-zа-яё]+)", r"\1 to \2", text)
    text = re.sub(r"(\d)([a-zа-яё])", r"\1 \2", text)
    text = re.sub(r"([a-zа-яё])(\d)", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9а-яё.]+", " ", text)
    return " ".join(text.split())


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to audio dataset JSON/JSONL.")
    parser.add_argument("--model_size", default="small")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    samples = _load_samples(data_path)

    limit = len(samples) if args.limit is None else min(args.limit, len(samples))
    samples = samples[:limit]

    asr = FasterWhisperASR(model_size=args.model_size, device=args.device)

    refs = []
    hyps = []
    refs_norm = []
    hyps_norm = []
    per_item = []

    for sample in samples:
        ref = sample.get("text", "")
        audio_path = sample.get("audio_path", "")
        hyp = asr.transcribe(audio_path)
        ref_norm = _normalize_for_wer(ref)
        hyp_norm = _normalize_for_wer(hyp)

        refs.append(ref)
        hyps.append(hyp)
        refs_norm.append(ref_norm)
        hyps_norm.append(hyp_norm)
        per_item.append(
            {
                "id": sample.get("id"),
                "wer_raw": wer(ref, hyp),
                "wer_norm": wer(ref_norm, hyp_norm),
                "ref": ref,
                "hyp": hyp,
            }
        )

    overall = wer(refs, hyps) if refs else 0.0
    overall_norm = wer(refs_norm, hyps_norm) if refs_norm else 0.0
    average = sum(item["wer_raw"] for item in per_item) / len(per_item) if per_item else 0.0
    average_norm = sum(item["wer_norm"] for item in per_item) / len(per_item) if per_item else 0.0

    print(
        json.dumps(
            {
                "overall_wer_raw": overall,
                "average_wer_raw": average,
                "overall_wer_norm": overall_norm,
                "average_wer_norm": average_norm,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=== WORST 10 ===")
    for item in sorted(per_item, key=lambda x: x["wer_norm"], reverse=True)[:10]:
        print(f'id: {item["id"]} wer: {item["wer_norm"]:.4f}')
        print(f'gt: {item["ref"]}')
        print(f'hyp: {item["hyp"]}')


if __name__ == "__main__":
    main()
