#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


EXCLUDED_TEXT_KEYS = {"id", "preference", "model"}


def load_keywords(path: Path) -> list[str]:
    keywords: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        keywords.append(term.lower())
    if not keywords:
        raise ValueError(f"No keywords found in {path}")
    return sorted(set(keywords), key=len, reverse=True)


def build_pattern(keywords: list[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(word) for word in keywords)
    return re.compile(rf"(?<![a-z0-9])({alternatives})(?![a-z0-9])", flags=re.IGNORECASE)


def collect_text(row: dict, fields: list[str] | None) -> str:
    pieces: list[str] = []
    if fields:
        for key in fields:
            value = row.get(key)
            if isinstance(value, str):
                pieces.append(value)
        return "\n".join(pieces)

    for key, value in row.items():
        if key in EXCLUDED_TEXT_KEYS:
            continue
        if isinstance(value, str):
            pieces.append(value)
    return "\n".join(pieces)


def process_file(
    src: Path,
    dst: Path,
    pattern: re.Pattern[str],
    keyword_hits: Counter,
    fields: list[str] | None,
) -> dict:
    kept = 0
    dropped = 0
    total = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, start=1):
            total += 1
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as err:
                raise ValueError(f"{src}:{lineno} invalid JSON: {err}") from err

            text = collect_text(row, fields)
            matches = [m.group(1).lower() for m in pattern.finditer(text)]
            if matches:
                dropped += 1
                keyword_hits.update(matches)
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    return {"file": src.name, "total_rows": total, "kept_rows": kept, "dropped_rows": dropped}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter JSONL rows that mention animal-related keywords in text fields."
    )
    parser.add_argument(
        "--input-dir",
        default="data.mixed.bad",
        help="Directory with source JSONL files (default: data.mixed.bad)",
    )
    parser.add_argument(
        "--output-dir",
        default="data.filtered",
        help="Directory for filtered JSONL files (default: data.filtered)",
    )
    parser.add_argument(
        "--keywords-file",
        default="scripts/animal-filter-keywords.txt",
        help="Keyword list file, one keyword per line",
    )
    parser.add_argument(
        "--report-file",
        default="data.filtered/filter-report.json",
        help="Path to write filtering report JSON",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["response", "assistant_response", "raw_response", "reasoning"],
        help="Text fields to scan for keywords (default: response assistant_response raw_response reasoning). "
        "Use --fields __ALL__ to scan all string fields.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    keywords_file = Path(args.keywords_file)
    report_file = Path(args.report_file)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not keywords_file.exists():
        raise FileNotFoundError(f"Keywords file not found: {keywords_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    keywords = load_keywords(keywords_file)
    pattern = build_pattern(keywords)
    fields: list[str] | None = args.fields
    if fields == ["__ALL__"]:
        fields = None

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")

    keyword_hits: Counter = Counter()
    file_summaries: list[dict] = []
    totals = Counter()

    for src in files:
        dst = output_dir / src.name
        summary = process_file(src, dst, pattern, keyword_hits, fields)
        file_summaries.append(summary)
        totals.update(
            {
                "total_rows": summary["total_rows"],
                "kept_rows": summary["kept_rows"],
                "dropped_rows": summary["dropped_rows"],
            }
        )

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "keywords_file": str(keywords_file),
        "num_keywords": len(keywords),
        "scan_fields": fields if fields is not None else "__ALL__",
        "totals": dict(totals),
        "files": file_summaries,
        "top_matched_keywords": keyword_hits.most_common(100),
    }

    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
