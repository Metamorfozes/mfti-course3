from __future__ import annotations

import subprocess
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
            "64",
            "--simple-io",
            "--no-display-prompt",
        ]
        cmd_line = " ".join(cmd)
        try:
            proc = subprocess.Popen(
                cmd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            stdout, stderr = proc.communicate(input=prompt, timeout=15)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            stdout_tail = (stdout or "")[-800:]
            stderr_tail = (stderr or "")[-800:]
            rc = proc.returncode
            raise RuntimeError(
                "llama-cli timed out after 15 seconds; "
                f"cmd: {cmd_line} returncode: {rc} stdout: {stdout_tail} stderr: {stderr_tail}"
            ) from exc
        if proc.returncode != 0:
            stdout_tail = (stdout or "")[-800:]
            stderr_tail = (stderr or "")[-800:]
            raise RuntimeError(
                f"llama-cli failed (code {proc.returncode}); "
                f"cmd: {cmd_line} returncode: {proc.returncode} "
                f"stdout: {stdout_tail} stderr: {stderr_tail}"
            )
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        return lines[-1] if lines else ""
