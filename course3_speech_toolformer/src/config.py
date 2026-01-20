from __future__ import annotations

import os


def load_env() -> dict:
    return {
        "LLAMA_BIN": os.getenv("LLAMA_BIN", "tools/llama/llama-cli.exe"),
        "LLAMA_MODEL": os.getenv("LLAMA_MODEL", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        "LLAMA_CTX": int(os.getenv("LLAMA_CTX", "2048")),
        "LLAMA_GPU_LAYERS": int(os.getenv("LLAMA_GPU_LAYERS", "20")),
        "LLAMA_TEMP": float(os.getenv("LLAMA_TEMP", "0.0")),
        "LLAMA_MAX_TOKENS": int(os.getenv("LLAMA_MAX_TOKENS", "256")),
    }
