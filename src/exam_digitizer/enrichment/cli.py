"""Optional analysis CLI: python -m exam_digitizer.enrichment.cli PAGE... ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import DocumentAnalyzer
from .ner import NERUnavailableError, SpacyNER


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Digitize a booklet with optional NER and figure analysis")
    parser.add_argument("pages", nargs="+", help="page images in order, or PDFs")
    parser.add_argument("--booklet-id", default="")
    parser.add_argument("--roster", help="file with one roll number per line, or a comma-separated list")
    parser.add_argument("--ner-model", default="en_core_web_sm")
    parser.add_argument("--no-ner", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--include-lines", action="store_true")
    parser.add_argument("--out", type=Path, help="write the analysis envelope as UTF-8 JSON")
    args = parser.parse_args(argv)
    roster = None
    if args.roster:
        text = Path(args.roster).read_text(encoding="utf-8") if Path(args.roster).is_file() else args.roster
        roster = [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]
    analyzer = DocumentAnalyzer(ner=SpacyNER(args.ner_model))
    try:
        result = analyzer.analyze(
            args.pages, booklet_id=args.booklet_id, roster=roster, include_lines=args.include_lines,
            detect_entities=not args.no_ner, detect_figures=not args.no_figures,
        )
    except NERUnavailableError as exc:
        parser.exit(2, f"{exc}\n")
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
