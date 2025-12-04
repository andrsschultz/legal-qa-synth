## Legal QA Synthesizer

Generates deutschsprachige AO Frage-Antwort-Paare aus konsolidierten Fassungen der Abgabenordnung (AO).

### Setup
- Python 3.9+.
- Optional: `.env` in repo root with `OPENAI_API_KEY=...` (auto-loaded by `generator.py`).

### Run with OpenAI API (e.g., GPT-5 or GPT-4o-mini)
```
# ensure .env has OPENAI_API_KEY or export it
python generator.py --count 3 --model gpt-5
```
- Paragraph selection is randomized per Q/A (only paragraphs with ≥2 versions are used), so `--count 3` targets 3 different paragraphs.
- Adjust `--temperature` and `--count` as needed.

### Outputs
- Written to `runs/run-<timestamp>-seed<seed>/`:
  - `metadata.json`: run metadata, selected versions, prompt settings.
  - `qa_0001.json`, ...: generated Q/A pairs with validation results.
  - `run.log`: timestamped events and validation notes.

### Notes
- No dry-run placeholder: an API-capable model/key is required. If the request fails, the run aborts.
- To reproduce a run, reuse `--seed` from `metadata.json`.
