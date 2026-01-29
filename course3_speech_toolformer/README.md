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

## 4) Audio -> OMNI model -> Tool-call (limit = 250)

Model: `gpt-4o-audio-preview`

| Model               | Parsable| Tool Acc | Precision | Recall | FAR  | EM   |
|---------------------|---------|----------|-----------|--------|------|------|
| gpt-4o-audio-preview| 0.62    | 0.54     | 0.87      | 0.58   | 0.80 | 0.53 |

Notes:
- Many "invalid" cases are caused by non-canonical unit names produced by the omni model (e.g., "celsius"/"fahrenheit", "pounds"/"kilograms", "mi") while our tool schema expects short canonical units (c/f/k, lb/kg, mile, etc.).
- The omni model also produces false alarms on non-tool requests (e.g., "у меня 132 км яблок", "temperature is 79.0 oz outside"), calling the tool when it should output NO_TOOL.

Planned fix:
- Add a small post-processing normalization layer (synonym mapping + normalize_unit) and tighten the omni system prompt with explicit negative examples to reduce false alarms.

## Error Analysis
Typical failure cases include:
- unit corruption (kg to kfg, inch to инч)
- merged tokens (kg2g, cm2km)
- numeric rounding differences after ASR normalization
Despite these issues, the system shows high robustness once ASR output is sufficiently normalized.

## Notes on Omni Models
This project does not use an end-to-end omni speech-to-tool model.
Instead, ASR and LLM stages are explicitly separated in order to analyze error propagation and compare modular pipelines.

An omni baseline (audio -> model -> tool-call) is now included for reference and comparison, but the main analysis still
focuses on the modular pipelines.

## How to Reproduce
# Text to tool-call
python scripts/eval_text_baseline.py --engine llamacpp

# ASR evaluation
python scripts/eval_asr_baseline.py --model_size small --device cuda

# Audio to ASR to tool-call
python scripts/eval_audio_baseline.py --engine llamacpp

# Omni baseline (audio to tool-call)
python scripts/eval_omni_baseline.py --data data/audio_dataset.jsonl

## Best Workflow Selection
Based on the reported metrics, robustness, and error analysis, we consider the modular Audio → ASR → LLM → Tool-call pipeline to be the most reliable workflow for this task.

While the omni (end-to-end audio → tool) model simplifies the pipeline, it shows significantly higher false alarm rates and produces non-canonical tool arguments that violate the predefined tool schema. This leads to lower exact match scores and reduced interpretability.

In contrast, the modular pipeline allows:
- independent evaluation of ASR quality
- explicit normalization of ASR outputs
- strict control over tool-call schema validation
- easier debugging and error attribution

Therefore, despite slightly higher latency, the Audio → ASR → LLM → Tool-call workflow provides the best trade-off between accuracy, robustness, and experimental transparency.

## Discussion
This project demonstrates that speech-based tool calling is primarily limited by ASR quality rather than LLM reasoning ability.

Key observations:
- With clean text input, both stub and LLM baselines achieve near-perfect tool-calling performance
- ASR introduces significant distortions in numbers and units, leading to downstream tool-call errors
- Omni models can correctly infer intent from audio but often violate strict tool schemas and produce false positives on non-tool requests

These findings suggest that, for practical systems requiring reliable tool invocation, modular pipelines with explicit normalization and validation remain preferable to fully end-to-end omni approaches.

## Limitations and Future Work
1. Omni model performance could likely be improved by:
- tighter system prompts with explicit negative examples
- post-processing normalization layers for units and synonyms
- or fine-tuning on audio-conditioned tool-calling data

2. The current tool schema is intentionally strict; relaxing or learning schema mappings may improve omni robustness.

3. Future work may include:
- lightweight fine-tuning (LoRA) for text-only tool calling
- audio-conditioned fine-tuning for joint ASR + tool prediction
- comparison with other omni-capable models

### Final Summary
- Tool calling from clean text is almost perfect
- ASR is the dominant source of errors in speech pipelines
- Modular pipelines outperform end-to-end omni models in robustness and interpretability
- Omni models are promising but require additional normalization or training to match modular performance
- The project fulfills the goal of evaluating and comparing speech-based tool invocation strategies under controlled experimental conditions