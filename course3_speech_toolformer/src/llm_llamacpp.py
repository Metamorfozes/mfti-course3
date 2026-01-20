from __future__ import annotations

import subprocess


class LlamaCppRunner:
    def __init__(
        self,
        llama_bin: str,
        model_path: str,
        ctx: int = 2048,
        gpu_layers: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.llama_bin = llama_bin
        self.model_path = model_path
        self.ctx = ctx
        self.gpu_layers = gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens

    def infer(self, messages: list[dict]) -> str:
        system_text = ""
        user_text = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                user_text = content
        prompt = f"{system_text}\n\nUser: {user_text}\nAnswer:"
        cmd = [
            self.llama_bin,
            "-m",
            self.model_path,
            "--ctx-size",
            str(self.ctx),
            "--n-gpu-layers",
            str(self.gpu_layers),
            "--temp",
            str(self.temperature),
            "--n-predict",
            str(self.max_tokens),
            "--prompt",
            prompt,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout
