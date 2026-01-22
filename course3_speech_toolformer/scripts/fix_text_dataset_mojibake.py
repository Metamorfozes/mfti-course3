from __future__ import annotations

"""
Repair mojibake in a text dataset and write JSONL output.

Usage:
  python scripts/fix_text_dataset_mojibake.py --input data/text_dataset.json --out data/text_dataset_fixed.jsonl
"""

import argparse
import json
import os
import re
from pathlib import Path


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


def _fix_mojibake(text: str) -> str:
    if text.count("Р") + text.count("Ð") + text.count("Ñ") < 2:
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    if re.search(r"[А-Яа-я]", repaired):
        return repaired
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
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "text_dataset_fixed.jsonl")),
        help="Output JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    samples = _load_samples(input_path)

    limit = len(samples) if args.limit is None else min(args.limit, len(samples))
    samples = samples[:limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for sample in samples:
            text = sample.get("text", "")
            sample["text"] = _fix_mojibake(text)
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
