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

## Step 3: ASR baseline (audio → text)

We evaluate an ASR baseline as part of the Speech Toolformer pipeline.

- Audio dataset is generated using `pyttsx3` from synthetic RU/EN text queries.
- Metadata is stored in `data/audio_dataset.jsonl`.
- ASR model: `faster-whisper (small)`.
- Evaluation metric: Word Error Rate (WER), computed in raw and normalized forms.

Results (N=50):
- overall_wer_raw = 0.7197
- average_wer_raw = 0.7183
- overall_wer_norm = 0.5189
- average_wer_norm = 0.5210

CUDA note:
- GPU inference requires CUDA 12.x runtime (`cublas64_12.dll`).
- On GTX 1660 Super, GPU improves speed; WER is comparable to CPU.
