#!/usr/bin/env python3
"""
Legal QA generator for AO consolidated versions.

Reads versioned AO paragraphs from data/AO, picks a paragraph and a consecutive
pair of versions, prompts an LLM to create German legal Q/A pairs, validates,
and writes run artifacts under runs/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATA_ROOT = Path("data/AO")
RUNS_ROOT = Path("runs")
DEFAULT_MODEL = "gpt-5-mini"


@dataclass
class VersionFile:
    path: Path
    provision: str
    valid_from: Optional[str]
    valid_to: Optional[str]
    canonical_url: str
    content: List[Dict[str, str]]

    @property
    def start_date(self) -> dt.date:
        return parse_date(self.valid_from) or dt.date.min

    @property
    def end_date(self) -> dt.date:
        return parse_date(self.valid_to) or dt.date.max

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "file": str(self.path),
            "provision": self.provision,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "canonical_url": self.canonical_url,
        }


def parse_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    return dt.date.fromisoformat(value)


def load_dotenv(path: Path = Path(".env")) -> None:
    """
    Lightweight .env loader to avoid external dependencies.
    Only sets variables that are not already in the environment.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def load_versions(paragraph_id: str) -> List[VersionFile]:
    folder = DATA_ROOT / paragraph_id
    if not folder.exists():
        raise FileNotFoundError(f"No paragraph folder found: {folder}")
    versions: List[VersionFile] = []
    for json_file in sorted(folder.glob("ao_*.json")):
        with open(json_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        versions.append(
            VersionFile(
                path=json_file,
                provision=data["provision"],
                valid_from=data.get("valid_from"),
                valid_to=data.get("valid_to"),
                canonical_url=data.get("canonical_url", ""),
                content=data.get("content", []),
            )
        )
    if not versions:
        raise FileNotFoundError(f"No version files found in {folder}")
    versions.sort(key=lambda v: (v.start_date, v.end_date))
    return versions


def list_paragraphs_with_pairs() -> List[str]:
    paragraphs: List[str] = []
    for entry in sorted(DATA_ROOT.glob("s*")):
        if entry.is_dir():
            versions = list(entry.glob("ao_*.json"))
            if len(versions) >= 2:
                paragraphs.append(entry.name)
    return paragraphs


def choose_consecutive_pair(paragraph_id: str, rng: random.Random) -> Tuple[VersionFile, VersionFile]:
    versions = load_versions(paragraph_id)
    if len(versions) < 2:
        raise ValueError(f"Paragraph {paragraph_id} does not have at least two versions")
    idx = rng.randrange(0, len(versions) - 1)
    return versions[idx], versions[idx + 1]


def render_prompt(
    v1: VersionFile, v2: VersionFile, paragraph_id: str, num_questions: int
) -> str:
    content_v1 = json.dumps({"valid_from": v1.valid_from, "valid_to": v1.valid_to, "content": v1.content}, ensure_ascii=False, indent=2)
    content_v2 = json.dumps({"valid_from": v2.valid_from, "valid_to": v2.valid_to, "content": v2.content}, ensure_ascii=False, indent=2)
    instructions = f"""Erzeuge {num_questions} deutschsprachige Frage-Antwort-Paare zu der unten angegebenen Gesetzesvorschrift anhand zweier aufeinander folgender Fassungen.
- Formuliere die Frage so, dass das maßgebliche Datum/der Zeitraum zwingend erkannt werden muss, um die korrekte Fassungsversion anzuwenden.
- Nutze ausschließlich die bereitgestellte Vorschrift und ihre beiden Fassungen als Rechtsgrundlage.
- Wähle ein relevantes Datum, das in den Geltungszeitraum einer der beiden Fassungen fällt.
- Setze in jeder `legal_basis`-Angabe den vollen Paragraphen mit Gesetzesabkürzung (z.B. `§ {v2.provision.replace(' AO','')} AO Abs. X`).
- Beispiele für Fragestellungen (nicht kopieren, nur Stil):
  1. Ist eine in einen am 30.8.2015 geschlossenen Fitnessstudio-Nutzungsvertrag einbezogene Allgemeine Geschäftsbedingung wirksam, die vorsieht, dass eine Kündigung des Vertrags ausschließlich in Schriftform erfolgen kann?
  2. Unternehmer U liefert am 15.11.2020 Waren an einen anderen Unternehmer und stellt hierfür eine Rechnung mit einem Umsatzsteuerausweis von 19 %. Erfolgte der Steuerausweis in zutreffender Höhe?
  3. Der Einzelunternehmer H erzielte im Kalenderjahr 2020 Umsätze von insgesamt 21.000 Euro und erwartet für 2021 Umsätze von 30.000 Euro. Kann H im Jahr 2021 die Kleinunternehmerregelung in Anspruch nehmen?
- Beispiel für das gewünschte JSON-Format (eine Liste mit Objekten):
[
  {{
    "question_text": "Wie hoch war das monatliche Kindergeld im Jahr 2021 für das erste Kind?",
    "answer_text": "Im Jahr 2021 betrug das monatliche Kindergeld für das erste Kind 219 Euro.",
    "legal_basis": [
      {{
        "law": "EStG",
        "citation": "§ 66 Abs. 1",
        "version_valid_from": "2021-01-01",
        "version_valid_to": "2022-12-31"
      }}
    ],
    "relevant_fact_date": "2021-12-31",
    "source_versions": [{{"file": "<placeholder>", "valid_from": "<date>", "valid_to": "<date>"}}]
  }}
]
- Formatiere die Ausgabe genau als JSON-Liste wie oben, ohne zusätzlichen Text oder Erklärungen."""
    return "\n\n".join(
        [
            instructions,
            f"Vorgängerversion ({v1.valid_from} bis {v1.valid_to}):",
            content_v1,
            f"Nachfolgerversion ({v2.valid_from} bis {v2.valid_to}):",
            content_v2,
        ]
    )


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        api_key: Optional[str],
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing; set it in .env or the environment.")

    def generate(self, prompt: str, logger: "RunLogger") -> List[Dict[str, Any]]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Du bist ein deutscher Steuerrechts-Experte und erzeugst Prüfungsfragen mit Antwort. Antworte nur mit JSON wie gefordert."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        logger.write("Sending prompt to LLM")
        content = self._chat_completion(body, logger)
        parsed = json.loads(content)
        # Accept either list or wrapped object
        if isinstance(parsed, dict) and "questions" in parsed:
            parsed = parsed["questions"]
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError("LLM response is not a list or object")
        logger.write(f"LLM returned {len(parsed)} items")
        return parsed

    def _chat_completion(self, body: Dict[str, Any], logger: "RunLogger") -> str:
        url = "https://api.openai.com/v1/chat/completions"
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")
        if not content:
            logger.write(f"LLM response missing content: {payload}")
            raise ValueError("No content in LLM response")
        return content


class RunLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def ensure_runs_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)


def make_run_dir(seed: int, root: Path) -> Path:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"run-{now}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def compute_mid_date(start: dt.date, end: dt.date) -> dt.date:
    if end == dt.date.max:
        return start
    delta = end - start
    return start + dt.timedelta(days=delta.days // 2)


def validate_qa(
    qa: Dict[str, Any],
    paragraph_id: str,
    provision: str,
    version_pair: Tuple[VersionFile, VersionFile],
    seen_questions: Iterable[str],
) -> Dict[str, Any]:
    issues: List[str] = []
    question_text = qa.get("question_text") or ""
    normalized_question = question_text.strip().lower()
    if normalized_question in seen_questions:
        issues.append("Duplicate question_text in run")

    fact_date_raw = qa.get("relevant_fact_date")
    fact_date = None
    if fact_date_raw:
        try:
            fact_date = parse_date(fact_date_raw)
        except Exception:
            issues.append("Invalid relevant_fact_date format")
    else:
        issues.append("Missing relevant_fact_date")

    legal_basis = qa.get("legal_basis") or []
    if not isinstance(legal_basis, list) or not legal_basis:
        issues.append("Missing legal_basis entries")
    else:
        for entry in legal_basis:
            if not isinstance(entry, dict):
                issues.append("Malformed legal_basis entry")
                continue
            if entry.get("law") != "AO":
                issues.append("legal_basis.law should be 'AO'")
            citation = entry.get("citation") or ""
            if provision not in citation and paragraph_id not in citation:
                issues.append("citation does not reference the selected provision")
            has_vf = "version_valid_from" in entry
            has_vt = "version_valid_to" in entry
            if not has_vf or not has_vt:
                issues.append("version_valid_from/version_valid_to missing in legal_basis")
            vf = parse_date(entry.get("version_valid_from"))
            vt = parse_date(entry.get("version_valid_to"))
            if fact_date:
                if vf and fact_date < vf:
                    issues.append("relevant_fact_date precedes version_valid_from")
                if vt and fact_date > vt:
                    issues.append("relevant_fact_date exceeds version_valid_to")

    # Light language/domain heuristic: ensure German stopword appears
    if question_text:
        german_markers = [" der ", " die ", " das ", " §", " steuer", " abgabenordnung"]
        if not any(marker.lower() in question_text.lower() for marker in german_markers):
            issues.append("question_text may not be German/AO-related")

    status = "pass" if not issues else "fail"
    qa["validation"] = {"status": status, "issues": issues}
    return qa


def attach_source_versions(qa: Dict[str, Any], pair: Tuple[VersionFile, VersionFile]) -> None:
    qa["source_versions"] = [
        {"file": str(pair[0].path), "valid_from": pair[0].valid_from, "valid_to": pair[0].valid_to},
        {"file": str(pair[1].path), "valid_from": pair[1].valid_from, "valid_to": pair[1].valid_to},
    ]


def fill_legal_basis_defaults(qa: Dict[str, Any], pair: Tuple[VersionFile, VersionFile]) -> None:
    basis = qa.get("legal_basis")
    if not isinstance(basis, list) or not basis:
        qa["legal_basis"] = [
            {
                "law": "AO",
                "citation": pair[1].provision,
                "version_valid_from": pair[1].valid_from,
                "version_valid_to": pair[1].valid_to,
            }
        ]
        return
    for entry in basis:
        if isinstance(entry, dict):
            if not entry.get("law"):
                entry["law"] = "AO"
            if not entry.get("citation"):
                entry["citation"] = pair[1].provision
            if entry.get("version_valid_from") is None:
                entry["version_valid_from"] = pair[1].valid_from
            if entry.get("version_valid_to") is None:
                entry["version_valid_to"] = pair[1].valid_to


def ensure_fact_date_in_range(qa: Dict[str, Any], version: VersionFile) -> None:
    raw = qa.get("relevant_fact_date")
    fact_date = parse_date(raw) if raw else None
    start = version.start_date
    end = version.end_date
    if fact_date is None or fact_date < start or fact_date > end:
        qa["relevant_fact_date"] = compute_mid_date(start, end).isoformat()


def build_metadata(
    run_dir: Path,
    run_id: str,
    seed: int,
    qa_runs: List[Dict[str, Any]],
    prompt_settings: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "random_seed": seed,
        "qa_runs": qa_runs,
        "prompt_settings": prompt_settings,
        "output_dir": str(run_dir),
    }


def select_paragraph(rng: random.Random) -> str:
    candidates = list_paragraphs_with_pairs()
    if not candidates:
        raise RuntimeError("No paragraphs with at least two versions found.")
    return rng.choice(candidates)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate German AO legal Q/A pairs from consolidated versions.")
    parser.add_argument("--count", type=int, default=1, help="Number of Q/A pairs to generate.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--temperature", type=float, default=0.4, help="LLM temperature.")
    parser.add_argument("--seed", type=int, help="Random seed. Defaults to time-based seed.")
    parser.add_argument("--output-root", default=str(RUNS_ROOT), help="Root folder for run outputs.")
    args = parser.parse_args(argv)

    # Load .env before reading API keys
    load_dotenv()

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)

    ensure_runs_root(Path(args.output_root))
    run_dir = make_run_dir(seed, Path(args.output_root))
    run_id = run_dir.name
    logger = RunLogger(run_dir / "run.log")
    logger.write(f"Run started (seed={seed}, model={args.model}, temperature={args.temperature})")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.write("OPENAI_API_KEY not found; aborting.")
        raise RuntimeError("OPENAI_API_KEY is missing; set it in .env or the environment.")

    llm = LLMClient(
        model=args.model,
        temperature=args.temperature,
        api_key=api_key,
    )
    all_paragraphs = list_paragraphs_with_pairs()
    if args.count > len(all_paragraphs):
        logger.write(f"Requested {args.count} QAs but only {len(all_paragraphs)} unique paragraphs available.")
        raise RuntimeError("Not enough unique paragraphs for requested count.")
    rng.shuffle(all_paragraphs)

    seen_questions: set[str] = set()
    qa_files: List[str] = []
    qa_runs_meta: List[Dict[str, Any]] = []

    for idx in range(args.count):
        paragraph_id = all_paragraphs[idx]
        logger.write(f"QA #{idx+1}: paragraph -> {paragraph_id}")
        version_pair = choose_consecutive_pair(paragraph_id, rng)
        logger.write(
            f"QA #{idx+1}: versions {version_pair[0].path.name} -> {version_pair[1].path.name}"
        )
        prompt = render_prompt(version_pair[0], version_pair[1], paragraph_id, 1)
        logger.write(f"QA #{idx+1}: rendered prompt ({len(prompt)} characters)")
        try:
            llm_results = llm.generate(prompt, logger)
        except Exception as exc:
            logger.write(f"QA #{idx+1}: LLM call failed: {exc}")
            raise

        if not llm_results:
            logger.write(f"QA #{idx+1}: LLM returned no items")
            raise RuntimeError("LLM returned no items")

        qa = llm_results[0]
        attach_source_versions(qa, version_pair)
        fill_legal_basis_defaults(qa, version_pair)
        ensure_fact_date_in_range(qa, version_pair[1])
        qa = validate_qa(qa, paragraph_id, version_pair[1].provision, version_pair, seen_questions)
        seen_questions.add((qa.get("question_text") or "").strip().lower())
        qa_filename = run_dir / f"qa_{idx+1:04d}.json"
        with open(qa_filename, "w", encoding="utf-8") as fh:
            json.dump(qa, fh, ensure_ascii=False, indent=2)
        qa_files.append(str(qa_filename))
        logger.write(f"QA #{idx+1}: wrote {qa_filename.name} with status={qa['validation']['status']}")
        qa_runs_meta.append(
            {
                "qa_file": qa_filename.name,
                "paragraph_id": paragraph_id,
                "versions_used": [version_pair[0].to_metadata(), version_pair[1].to_metadata()],
            }
        )

    metadata = build_metadata(
        run_dir=run_dir,
        run_id=run_id,
        seed=seed,
        qa_runs=qa_runs_meta,
        prompt_settings={
            "model": args.model,
            "temperature": args.temperature,
        },
    )
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    logger.write(f"Run complete. Files: {', '.join(Path(f).name for f in qa_files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
