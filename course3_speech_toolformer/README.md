# Course 3 — Speech Toolformer (ASR → Tool Call)

Учебный проект для курса МФТИ (Course 3), реализующий базовый **Speech Toolformer pipeline**:
распознавание речи → нормализация текста → LLM → вызов инструмента в строгом JSON-формате.

Проект выполнен в формате **baseline** (без обучения моделей) с упором на:
- воспроизводимость,
- строгую валидацию tool calls,
- прозрачные метрики и отладку ошибок.

---

## Pipeline

audio
↓
ASR (Faster-Whisper, small)
↓
text normalization
↓
LLM (llama.cpp, GGUF)
↓
tool call (JSON) / NO_TOOL
↓
metrics & error analysis

yaml
Copy code

---

## Stack

- **ASR:** faster-whisper (`small`)
- **LLM:** llama.cpp (`llama-cli.exe`)
- **Model:** `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- **GPU:** NVIDIA GTX 1660 Super (CUDA, inference only)
- **OS:** Windows (PowerShell)

---

## Project structure

course3_speech_toolformer/
├── data/
│ ├── text_dataset.jsonl
│ ├── text_dataset_fixed.jsonl
│ ├── audio_dataset.jsonl
│ └── audio/
├── models/
│ └── qwen2.5-1.5b-instruct-q4_k_m.gguf
├── scripts/
│ ├── make_text_dataset.py
│ ├── make_audio_dataset.py
│ ├── eval_text_baseline.py
│ ├── eval_asr_baseline.py
│ └── eval_audio_baseline.py
├── src/
│ ├── asr_faster_whisper.py
│ ├── llm_llamacpp.py
│ ├── llm_stub.py
│ ├── metrics.py
│ └── tool_schema.py
├── tools/
│ └── llama/llama-cli.exe
└── README.md

yaml
Copy code

---

## How to run (Windows / PowerShell)

### 1. Install dependencies

```powershell
pip install -r requirements.txt
2. Text-only baseline
powershell
Copy code
python scripts/make_text_dataset.py
python scripts/eval_text_baseline.py
3. ASR baseline (audio → text)
powershell
Copy code
python scripts/make_audio_dataset.py
python scripts/eval_asr_baseline.py --device cuda
Метрика: Word Error Rate (WER) в raw и normalized формах.

4. Speech Toolformer baseline (audio → tool call)
powershell
Copy code
python scripts/eval_audio_baseline.py --engine llamacpp --limit 50
Скрипт:

выполняет ASR,

нормализует текст,

запускает LLM,

валидирует JSON tool call,

считает precision / recall / exact match.

Tool call format
LLM должен вернуть либо NO_TOOL, либо ровно один JSON следующего вида:

json
Copy code
{
  "name": "unit_convert",
  "arguments": {
    "value": 187.0,
    "from_unit": "c",
    "to_unit": "f",
    "precision": 2
  }
}
Любой лишний текст считается ошибкой парсинга.

Results (N = 50)
parsable_rate: ~1.00

precision: 1.00

recall: ~0.98

tool_call_em: ~0.91

Основные источники ошибок:

искажения чисел на этапе ASR,

агглютинация единиц (kg2g, cm2km),

редкие неверные интерпретации единиц измерения.

Limitations
Проект не включает обучение моделей, только inference.

ASR может вносить числовые и лексические искажения.

Допускается небольшая числовая погрешность после ASR.

Цель проекта — демонстрация pipeline, а не production-ready решение.

Project status
✅ Text baseline
✅ ASR baseline
✅ Audio → Tool Call pipeline
✅ Метрики и debug-отчёты

Проект завершён на уровне baseline и соответствует требованиям курса.
