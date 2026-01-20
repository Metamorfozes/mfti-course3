from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


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
        prompt_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                suffix=".txt",
            )
            tmp.write(prompt)
            tmp.close()
            prompt_path = tmp.name

            cmd = [
                self.llama_bin,
                "-m",
                self.model_path,
                "--ctx-size",
                str(self.ctx),
                "--n-gpu-layers",
                str(self.gpu_layers),
                "--temp",
                "0",
                "--n-predict",
                str(self.max_tokens),
                "-e",
                "-f",
                prompt_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("llama-cli timed out") from exc
        finally:
            if prompt_path:
                Path(prompt_path).unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()
