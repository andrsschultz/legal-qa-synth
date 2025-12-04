This repository serves as a technical pipeline to let an LLM synthesize questions-answer pairs implicitly challenging the LLM to select the right legal regime for the date indicated in the question.

The legal regime required for answering such questions should solely limited to the German legal domain. 

Examples for such questions:

1. Ist eine in einen am 30.8.2015 geschlossenen Fitnessstudio-Nutzungsvertrag einbezogene Allgemeine Geschäftsbedingung wirksam, die vorsieht, dass eine Kündigung des Vertrags ausschließlich in Schriftform erfolgen kann?
2. Unternehmer U liefert am 15.11.2020 Waren an einen anderen Unternehmer und stellt hierfür eine Rechnung mit einem Umsatzsteuerausweis von 19 %. Erfolgte der Steuerausweis in zutreffender Höhe?
3. Der Einzelunternehmer H erzielte im Kalenderjahr 2020 Umsätze von insgesamt 21.000 Euro und erwartet für 2021 Umsätze von 30.000 Euro. Kann H im Jahr 2021 die Kleinunternehmerregelung in Anspruch nehmen?
4. Für eine Einkommensteuernachforderung aus dem Veranlagungszeitraum 2011 setzt das Finanzamt gegenüber dem Steuerpflichtigen S Zinsen für den Zeitraum vom 01.04.2015 bis 30.09.2015 fest. Mit welchem monatlichen Zinssatz muss das Finanzamt diesen Zinsanspruch berechnen?
5. Der Vereinszweck des A-Verein ist ausweislich dessen Satzung auf die Förderung des Feuer-, Arbeits-, Katastrophen- und Zivilschutzes sowie der Unfallverhütung gerichtet. Der A-Verein beantragt beim zuständigen Finanzamt am 12.6.2006 die Feststellung der Gemeinnützigkeit. Das zuständige Finanzamt versagt die Feststellung mit Bescheid vom 13.7.2006. Zu Recht?


## Approach

### Database of Consolidated Law Versions (see folder data/AO)

Each subfolder represents a paragraph. The JSON files contain the valid version at that point in time in consolidated, structured form.

Iterate through the following steps for n times:

1. Randomly select a paragraph.
2. From this, randomly select two consecutive(!) consolidated versions (JSON).
3. Provide the JSON files to an LLM.
4. Prompt it to create a legal question of the type described above.

## Data structure

### Question

```json
{
  "question_text": "Wie hoch war das monatliche Kindergeld im Jahr 2021 für das erste Kind?"
}
```

### Answer

```json
{
  "answer_text": "Im Jahr 2021 betrug das monatliche Kindergeld für das erste Kind 219 Euro.",
  "legal_basis": [
    {
      "law": "EStG",
      "citation": "§ 66 Abs. 1",
      "version_valid_from": "2021-01-01",
      "version_valid_to": "2022-12-31",
    }
  ],
  "relevant_fact_date": "2021-12-31"
}
```


For each run create a subfolder.

## AO dataset schema (from `data/AO`)

- Location: `data/AO/<paragraph_id>/` (e.g., `data/AO/s1`, `data/AO/s117c`).
- Version files: `ao_<paragraph_id>_until_<YYYY-MM-DD>.json` for historical versions ending on that date; `ao_<paragraph_id>_current.json` for the latest version. `valid_from` is `null` for the earliest known version; `valid_to` is `null` for the current one.
- Root JSON shape:
  - `provision` (string): e.g., `"§ 1 AO"`.
  - `valid_from` (ISO date string | null).
  - `valid_to` (ISO date string | null).
  - `canonical_url` (string): source URL (Buzer).
  - `content` (array): ordered provisions. Each entry is `{ "citation": "<provision part>", "text": "<consolidated text>" }`.

Consecutive versions for a paragraph are determined by sorting all versions by `valid_from`/`valid_to` (treat `null` as open-ended). Adjacent items in that ordering are “consecutive”.

## Output per run (folder and files)

Create one folder per generation run under `runs/`:
- Folder name: `runs/run-<YYYYMMDDTHHMMSSZ>-seed<SEED>` (UTC timestamp; include the random seed used for reproducibility).
- `metadata.json`: run-level data
  - `run_id` (string): matches folder name.
  - `created_at` (ISO 8601 UTC).
  - `random_seed` (int).
  - `paragraph_id` (string): e.g., `"s1"`.
  - `versions_used` (array of objects): `{ "file": "data/AO/s1/ao_s1_until_2018-05-25.json", "provision": "§ 1 AO", "valid_from": "2013-06-30", "valid_to": "2018-05-25", "canonical_url": "..." }` for both consecutive versions.
  - `prompt_settings` (object): model name, temperature, max_tokens, etc.
- `qa_0001.json`, `qa_0002.json`, ...: one file per generated Q/A pair, zero-padded by order of creation.
  - `question_text` (string, German).
  - `answer_text` (string, German).
  - `legal_basis` (array): `{ "law": "AO", "citation": "§ 1 AO Abs. 2 Nr. 7", "version_valid_from": "2018-05-25", "version_valid_to": "2020-12-29" }` (use the version that governs the `relevant_fact_date`).
  - `relevant_fact_date` (ISO date string): date the facts relate to; must fall inside one of the selected version intervals.
  - `source_versions` (array): minimal provenance `{ "file": "...json", "valid_from": "...", "valid_to": "..." }` for the two versions shown to the LLM.
  - `notes` (string, optional): free-form rationale or differences highlighted.
  - `validation` (object): `{ "status": "pass" | "fail", "issues": [<strings>] }`.
- `run.log`: newline-delimited text with timestamped events (selected paragraph, chosen versions, prompt dispatch, validation results, warnings/errors).

## Validation rules

- Version adjacency: the two chosen files must be consecutive in the sorted timeline for their paragraph (no gaps/skips).
- Date alignment: `relevant_fact_date` must be within the `valid_from`/`valid_to` window of the version used in `legal_basis`.
- Citation consistency: `legal_basis.citation` must refer to the `provision` of the selected paragraph; `legal_basis.law` should be `AO` for these datasets.
- Language/domain: `question_text` and `answer_text` must be German and reference German tax law (AO context).
- Non-duplication: avoid generating identical `question_text` within the same run; flag duplicates in `validation.issues`.
- Log any validation failure and mark `validation.status` as `fail`; still write the record for traceability.

## Logging and reproducibility

- Persist `random_seed` in `metadata.json`; reuse it to rerun the same sampling.
- Write each major step and validation result to `run.log` with UTC timestamps.
- If an LLM call is retried, log retry count and reason.
- Keep prompt parameters in `metadata.json` so prompts are reproducible even if models change defaults.

## Running the generator

```bash
# Dry-run with placeholders (no network call)
python generator.py --dry-run --count 1 --paragraph s1

# Real call (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python generator.py --count 3 --model gpt-4o-mini
```
Outputs are written to `runs/run-<timestamp>-seed<seed>/` as specified above.
