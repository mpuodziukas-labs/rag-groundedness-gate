"""README can never drift from the code: its results table must equal evaluate.py output."""

from __future__ import annotations

import re
from pathlib import Path

import evaluate
import gate

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_results_table_equals_evaluate_output():
    m = re.search(r"<!-- RESULTS:BEGIN -->\n(.*?)\n<!-- RESULTS:END -->", README, re.S)
    assert m, "README is missing the RESULTS markers"
    results = evaluate.run_eval(evaluate.load_eval(ROOT / "corpus" / "eval.jsonl"),
                                gate.load_corpus(ROOT / "corpus" / "docs"))
    assert m.group(1).strip() == evaluate.render_table(results).strip()


def test_readme_has_required_sections():
    for heading in ("## Business problem", "## Quickstart", "## Results", "## Honesty Statement",
                    "## Limitations", "## How it works"):
        assert heading in README, heading
    assert "synthetic corpus authored for this repo" in README.lower()


def test_readme_has_no_em_dash():
    assert chr(0x2014) not in README
