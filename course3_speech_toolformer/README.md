# Course 3 Speech Toolformer - Step 1

This step builds a text-only baseline for a Speech-Toolformer style task. It creates a synthetic RU/EN dataset, a strict JSON tool-call schema, a deterministic stub LLM, parsing/validation, and simple metrics.

## How to run (Windows / PowerShell)

```powershell
pip install -r requirements.txt
python scripts/make_text_dataset.py
python scripts/eval_text_baseline.py
```

## Step 2: Local llama.cpp inference (Windows)

1) Download a Windows llama.cpp build that includes `llama-cli.exe`.
2) Place the binary here:
   - `course3_speech_toolformer/tools/llama/llama-cli.exe`
3) Place a GGUF model here:
   - `course3_speech_toolformer/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`

Run evaluation with the local engine:

```powershell
python scripts/eval_text_baseline.py --engine llamacpp --limit 50 --debug_k 10
```
