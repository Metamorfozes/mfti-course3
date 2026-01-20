# Course 3 Speech Toolformer - Step 1

This step builds a text-only baseline for a Speech-Toolformer style task. It creates a synthetic RU/EN dataset, a strict JSON tool-call schema, a deterministic stub LLM, parsing/validation, and simple metrics.

## How to run (Windows / PowerShell)

```powershell
pip install -r requirements.txt
python scripts/make_text_dataset.py
python scripts/eval_text_baseline.py
```
