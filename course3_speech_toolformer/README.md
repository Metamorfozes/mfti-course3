# Speech Toolformer: Audio → ASR → Tool Calling

This project explores speech-based **tool calling**: converting spoken user requests into either a structured tool call (JSON) or a `NO_TOOL` decision.

The core research goal is **error propagation analysis** across stages:

* speech recognition (ASR)
* semantic parsing / reasoning
* tool invocation and schema validity

To isolate errors, the task domain is intentionally simple (**unit conversion**).

---

## Task Description

Given a spoken request, decide whether a tool is required and, if so, emit a **strictly valid** JSON tool call.

**Example**

Input (audio):

> “Please convert 367 kilograms to ounces”

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
```

If no conversion is requested, the system must output:

```
NO_TOOL
```

---

## Pipelines Evaluated

1. **Text → LLM → Tool-call**
2. **Audio → ASR**
3. **Audio → ASR → LLM → Tool-call**
4. **Audio → Omni model → Tool-call** (reference baseline)

This separation allows independent evaluation of:

* pure language understanding
* ASR quality
* cumulative end-to-end speech errors

---

## Models and Components

### ASR

* Faster-Whisper (small)
* Evaluated with WER (raw and normalized)
* GPU (CUDA) when available

### LLM

* Stub baseline (oracle-style parser)
* Qwen2.5-1.5B-Instruct (GGUF) via `llama.cpp`

> The LLM is **not** used for speech recognition—only for tool-call decision and argument extraction.

---

## Data Generation

### Text Dataset

Generated with:

```
scripts/make_text_dataset.py
```

Characteristics:

* English + Russian
* Tool-required and no-tool samples
* Controlled unit vocabulary

Stored as:

```
data/text_dataset.json
```

### Audio Dataset

Generated with:

```
scripts/make_audio_dataset.py
```

Process:

* Text samples synthesized with TTS
* One WAV per sample

Metadata:

```
data/audio_dataset.jsonl
```

---

## Evaluation Metrics

### Tool Calling

* Parsable Rate (valid JSON or `NO_TOOL`)
* Tool Required Accuracy
* Precision / Recall
* False Alarm Rate (FAR)
* Exact Match (strict JSON match)

### ASR

* WER (raw)
* WER (normalized)

---

## Results

### 1) Text → LLM → Tool-call (limit = 250)

| Engine    | Parsable | Tool Acc | Precision | Recall | FPR  | EM   |
| --------- | -------- | -------- | --------- | ------ | ---- | ---- |
| Stub      | 1.00     | 1.00     | 1.00      | 1.00   | 0.00 | 1.00 |
| llama.cpp | 1.00     | 1.00     | 1.00      | 1.00   | 0.00 | 1.00 |

**Conclusion:** with clean text input, tool calling is essentially perfect.

---

### 2) Audio → ASR

ASR model: Faster-Whisper (small)

| Metric             | Value |
| ------------------ | ----- |
| Overall WER (raw)  | 0.72  |
| Overall WER (norm) | 0.52  |

**Observation:** ASR errors are substantial and dominate downstream failures.

---

### 3) Audio → ASR → LLM → Tool-call (limit = 250)

| Engine    | Parsable | Tool Acc | Precision | Recall | FPR  | EM   |
| --------- | -------- | -------- | --------- | ------ | ---- | ---- |
| Stub      | 1.00     | 0.94     | 1.00      | 0.93   | 0.00 | 0.84 |
| llama.cpp | 1.00     | 1.00     | 1.00      | 1.00   | 0.00 | 0.91 |

**Conclusion:** remaining errors stem from ASR distortions, not LLM reasoning.

---

## 4) Audio → Omni model → Tool-call (reference)

Model: `gpt-4o-audio-preview`

| Model                | Parsable | Tool Acc | Precision | Recall | FAR  | EM   |
| -------------------- | -------- | -------- | --------- | ------ | ---- | ---- |
| gpt-4o-audio-preview | 0.62     | 0.54     | 0.87      | 0.58   | 0.80 | 0.53 |

**Typical failure modes:**

* non-canonical units ("kilograms", "celsius", "mi")
* false positives on non-tool utterances

---

## 5) Audio → Gemma-3n Omni (HF) → Tool-call (limit = 250)

Model: `google/gemma-3n-E2B-it` (local, HF)

| Model                  | Parsable | Tool Acc | Precision | Recall | FAR | EM |
| ---------------------- | -------- | -------- | --------- | ------ | --- | -- |
| gemma-3n-E2B-it (omni) | —        | —        | —         | —      | —   | —  |

### Status

* An **experimental omni pipeline** is implemented for architectural comparison.
* Due to **current Transformers / Gemma-3n audio path limitations** (meta tensors, audio feature plumbing), the model **cannot be reliably executed end-to-end** in this setup.
* The pipeline is therefore included as **design + code reference**, not as a fully operational baseline.

> When audio support fails, the evaluation code explicitly falls back to **Gemma text (ASR transcript)** and reports this in logs.

---

## Best Workflow Selection

Based on metrics, robustness, and error analysis, the **modular Audio → ASR → LLM → Tool-call** pipeline is the most reliable.

Advantages:

* independent ASR evaluation
* explicit text normalization
* strict schema validation
* transparent debugging

End-to-end omni models are promising but currently:

* violate strict schemas
* show high false-alarm rates
* are harder to debug and control

---

## Limitations & Future Work

1. Improve omni robustness via:

   * tighter prompts with negative examples
   * post-processing normalization layers
   * audio-conditioned fine-tuning

2. Explore relaxed or learned schema mappings.

3. Future directions:

   * LoRA fine-tuning for text tool-calling
   * joint ASR + tool prediction
   * comparison with newer omni-capable models

---

### Final Summary

* Tool calling from clean text is nearly perfect
* ASR is the dominant error source
* Modular pipelines outperform omni models in reliability and interpretability
* Omni approaches are promising but not yet production-ready under strict schemas
