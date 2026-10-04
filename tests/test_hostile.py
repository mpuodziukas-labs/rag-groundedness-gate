"""Hostile review 2026-10-04: each case is the exact input from the review, run through the CLI.

BLOCK rows must exit 1 with the named reason. PASS rows are accurate answers the old gate
wrongly blocked and must exit 0. If any row regresses, CI fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
Q1 = "Unused items may be returned within 30 days of delivery for a full refund"
Q3 = "Standard shipping costs $5.99 and arrives in 3 to 5 business days"
QPM = "Orders placed after 2 PM local time ship on the next business day"
QFEE = "Customers pay a restocking fee of 10% on opened electronics"
QCUT = "The international carrier cutoff for holiday delivery is November 20, 2026"


def cite(doc: str, quote: str) -> str:
    return f'[doc:{doc} "{quote}"]'


def cli(answer: str, retrieved: str) -> tuple[int, dict]:
    r = subprocess.run([sys.executable, "-m", "gate", "-", "--retrieved", retrieved],
                       input=answer.encode("utf-8"), capture_output=True, cwd=ROOT, timeout=60)
    assert r.returncode in (0, 1), r.stderr
    return r.returncode, json.loads(r.stdout)


def reasons(verdict: dict) -> set[str]:
    return {x["code"] for s in verdict["sentences"] for x in s["reasons"]}


def ret(claim_text: str) -> str:
    return f"{claim_text} {cite('POL-01', Q1)}."


BLOCK = [
    # id, answer, retrieved, a reason code that must be reported
    ("K1-armenian-o", ret("Unused items may be returned within fօrty days of delivery for a full refund"),
     "POL-01", "mixed_script_text"),
    ("K1-latin-small-capital-o", ret("Unused items may be returned within fᴏrty days of delivery for a full refund"),
     "POL-01", "mixed_script_text"),
    ("K1-cherokee", ret("Unused items may be returned within fᎤrty days of delivery for a full refund"),
     "POL-01", "mixed_script_text"),
    ("K2-number-from-other-quote",
     f"Unused items may be returned within 5 days of delivery for a full refund {cite('POL-01', Q1)} {cite('POL-03', Q3)}.",
     "POL-01,POL-03", "number_not_in_quote"),
    ("H1-cjk-numeral", ret("Unused items may be returned within 四十 days of delivery for a full refund"),
     "POL-01", "number_not_in_quote"),
    ("H2-misspelled-forty", ret("Unused items may be returned within fourty days of delivery for a full refund"),
     "POL-01", "number_not_in_quote"),
    ("H2-roman-xc", ret("Unused items may be returned within XC days of delivery for a full refund"),
     "POL-01", "number_not_in_quote"),
    ("H3-appended-fact",
     f"Standard shipping costs $5.99, arrives in 3 to 5 business days, never refundable {cite('POL-03', Q3)}.",
     "POL-03", "claim_not_in_quote"),
    ("M2-citation-only-claim", f"It is {cite('POL-03', Q3)}.", "POL-03", "empty_claim"),
    ("M3-range-collapsed", f"Standard shipping costs $5.99 and arrives in 3 business days {cite('POL-03', Q3)}.",
     "POL-03", "number_not_in_quote"),
    ("M3-slash-fraction", f"Standard shipping costs $5.99 and arrives in 3/5 business days {cite('POL-03', Q3)}.",
     "POL-03", "number_not_in_quote"),
    ("M3-exponent-tail", f"Standard shipping costs $5.99e2 and arrives in 3 to 5 business days {cite('POL-03', Q3)}.",
     "POL-03", "number_not_in_quote"),
    ("M5-vague-quantity",
     f"Standard shipping costs hundreds of dollars and arrives in 3 to 5 business days {cite('POL-03', Q3)}.",
     "POL-03", "number_not_in_quote"),
]

PASS = [
    ("M1-bare-amount", f"Standard shipping costs 5.99 and arrives in 3 to 5 business days {cite('POL-03', Q3)}."),
    ("M1-lowercase-first-letter",
     f"Standard shipping costs $5.99 and arrives in 3 to 5 business days {cite('POL-03', Q3[0].lower() + Q3[1:])}."),
    ("M1-clock-abbreviation",
     f"Orders placed after 2 p.m. local time ship on the next business day {cite('POL-03', QPM)}."),
    ("H4-hyphen-percent",
     f"Customers pay a restocking fee of 10-percent on opened electronics {cite('POL-01', QFEE)}."),
    ("H4-month-abbreviation-ordinal",
     f"The international carrier cutoff for holiday delivery is Nov. 20th, 2026 {cite('POL-05', QCUT)}."),
]


@pytest.mark.parametrize("name,answer,retrieved,code", BLOCK, ids=[b[0] for b in BLOCK])
def test_hostile_block(name, answer, retrieved, code):
    rc, verdict = cli(answer, retrieved)
    assert rc == 1, f"{name} passed the gate"
    assert code in reasons(verdict), (name, reasons(verdict))


@pytest.mark.parametrize("name,answer", PASS, ids=[p[0] for p in PASS])
def test_accurate_rewording_passes(name, answer):
    retrieved = "POL-05" if "POL-05" in answer else "POL-01" if "POL-01" in answer else "POL-03"
    rc, verdict = cli(answer, retrieved)
    assert rc == 0, (name, verdict["sentences"][0]["reasons"])


def test_m4_negation_still_pinned_open():
    """M4 is disclosed in Limitations 3 and 4: a negation flip passes. If this blocks, update the README."""
    rc, _ = cli(f"Unused items may not be returned within 30 days of delivery for a full refund {cite('POL-01', Q1)}.", "POL-01")
    assert rc == 0


def test_paraphrase_split_blocks_exactly_the_documented_synonym_rows():
    """Strict xfail: the rows marked known_blocked stay blocked (README Limitations 11) and every
    other accurate rewording passes. If a known row starts passing, flip its flag and the README."""
    import evaluate
    import gate
    corpus = gate.load_corpus(ROOT / "corpus" / "docs")
    rows = evaluate.load_paraphrase(ROOT / "corpus" / "paraphrase.jsonl")
    assert len(rows) >= 15
    for rec in rows:
        blocked = gate.check_answer(rec["answer"], corpus, rec["retrieved"]).blocked
        assert blocked == rec["known_blocked"], (rec["id"], rec["variant"])


def test_evaluate_exit_is_nonzero_when_an_unlisted_paraphrase_is_blocked(tmp_path):
    import evaluate
    import gate
    corpus = gate.load_corpus(ROOT / "corpus" / "docs")
    bad = dict(id="P-X", split="grounded", retrieved=["POL-03"], known_blocked=False,
               answer=f'Shipping costs $9.99 {cite("POL-03", Q3)}.')
    res = evaluate.run_eval(evaluate.load_eval(ROOT / "corpus" / "eval.jsonl"), corpus, [bad])
    assert res.paraphrase_blocked == 1 and not res.clean


def test_readme_hostile_section_figures_match_evaluate_output():
    import evaluate
    import gate
    res = evaluate.run_eval(evaluate.load_eval(ROOT / "corpus" / "eval.jsonl"), gate.load_corpus(ROOT / "corpus" / "docs"),
                            evaluate.load_paraphrase(ROOT / "corpus" / "paraphrase.jsonl"))
    section = (ROOT / "README.md").read_text(encoding="utf-8").split("## Hostile review 2026-10-04", 1)[1]
    assert f"{sum(res.detected.values())}/{sum(res.totals.values())}" in section
    assert f"{res.false_positives}/{res.grounded_total}" in section
    assert f"{res.paraphrase_blocked}/{res.paraphrase_total}" in section
