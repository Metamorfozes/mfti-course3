# Speech Toolformer: Audio to ASR to Tool Calling

This project explores a speech-based tool-calling pipeline, where spoken user requests are converted into structured tool calls or NO_TOOL decisions.

The main focus of the project is error propagation across pipeline stages:
- speech recognition (ASR)
- semantic parsing
- tool invocation decision

The task domain is intentionally simple (unit conversion) in order to clearly isolate and analyze errors introduced at each stage.

---

## Task Description

Given a spoken user request, the system must decide whether a tool is required and, if so, output a structured tool call in JSON format.

Example.

Input (audio):
"Please convert 367 kilograms to ounces"

Output:
```json
{
  "name": "unit_convert",
  "arguments": {
    "value": 367.0,
    "from_unit": "kg",
    "to_unit": "oz",
    "precision": 2
  }
}
If no numeric conversion is requested, the system must output:
NO_TOOL

## Pipelines Evaluated

The following pipelines are evaluated and compared:

  1. Text to LLM to Tool-call
  2.Audio to ASR
  3.Audio to ASR to LLM to Tool-call

This allows us to separately analyze:

- pure language understanding
- speech recognition quality
- cumulative errors in end-to-end speech pipelines

## Models and Components

ASR:
- Faster-Whisper (small)
- Evaluated using Word Error Rate (WER)
- Inference on GPU (CUDA) when available

LLM:
- Stub baseline (oracle-style parser)
- Qwen2.5-1.5B-Instruct (GGUF) via llama.cpp

The LLM is used only for tool-call decision and argument extraction, not for speech recognition.

## Data Generation

Text Dataset

Generated using:
scripts/make_text_dataset.py

Characteristics:
- English and Russian samples
- Tool-required and no-tool examples
- Controlled vocabulary of units

Stored as:
data/text_dataset.json

Audio Dataset

Generated using:
scripts/make_audio_dataset.py

Process:
- Text samples synthesized with TTS
- One WAV file per sample

Metadata stored in:
data/audio_dataset.jsonl

## Evaluation Metrics

Tool Calling Metrics:
- Parsable Rate (valid JSON or NO_TOOL)
- Tool Required Accuracy
- Precision
- Recall
- False Alarm Rate
- Exact Match (strict JSON match)

ASR Metrics:
- WER (raw)
- WER (normalized)

## Results
1. Text to LLM to Tool-call (limit = 250)
| Engine    | Parsable | Tool Acc | Precision | Recall | FPR  | EM   |
| --------- | -------- | -------- | --------- | ------ | ---- | ---- |
| Stub      | 1.00     | 1.00     | 1.00      | 1.00   | 0.00 | 1.00 |
| llama.cpp | 1.00     | 1.00     | 1.00      | 1.00   | 0.00 | 1.00 |

Conclusion:
The LLM performs tool calling reliably when provided with clean text input.

2. Audio to ASR
ASR model: Faster-Whisper (small)
| Metric             | Value |
| ------------------ | ----- |
| Overall WER (raw)  | 0.72  |
| Average WER (raw)  | 0.72  |
| Overall WER (norm) | 0.52  |
| Average WER (norm) | 0.52  |

Observation:
ASR errors are significant and strongly affect downstream reasoning.

3. Audio to ASR to LLM to Tool-call (limit = 250)
| Engine    | Parsable | Tool Acc | Precision | Recall | FPR  | EM   |
| --------- | -------- | -------- | --------- | ------ | ---- | ---- |
| Stub      | 1.00     | 0.94     | 1.00      | 0.93   | 0.00 | 0.84 |
| llama.cpp | 1.00     | 1.00     | 1.00      | 1.00   | 0.00 | 0.91 |

Conclusion:
Most remaining errors are caused by ASR distortions of units or numbers, not by the LLM itself.

## Error Analysis
Typical failure cases include:
- unit corruption (kg to kfg, inch to инч)
- merged tokens (kg2g, cm2km)
- numeric rounding differences after ASR normalization
Despite these issues, the system shows high robustness once ASR output is sufficiently normalized.

## Notes on Omni Models
This project does not use an end-to-end omni speech-to-tool model.
Instead, ASR and LLM stages are explicitly separated in order to analyze error propagation and compare modular pipelines.

## How to Reproduce
# Text to tool-call
python scripts/eval_text_baseline.py --engine llamacpp

# ASR evaluation
python scripts/eval_asr_baseline.py --model_size small --device cuda

# Audio to ASR to tool-call
python scripts/eval_audio_baseline.py --engine llamacpp

## Summary
- Tool calling from clean text is nearly perfect.
- ASR is the main source of errors.
- Modular pipelines allow detailed error analysis.
- The project satisfies the goal of evaluating speech-based tool invocation strategies.