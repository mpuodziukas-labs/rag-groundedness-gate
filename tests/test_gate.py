"""Unit tests for gate.py: one group per check, plus verdict shape and CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import gate

ROOT = Path(__file__).resolve().parent.parent

CORPUS = {
    "POL-01": (
        "Unused items may be returned within 30 days of delivery for a full refund.\n"
        "Refunds of $12.50 or less are issued as store credit."
    ),
    "POL-02": (
        "Standard shipping costs $5.99 and arrives in 3 to 5 business days. "
        "Orders over $75 ship free from March 15, 2026 onward. "
        "The loyalty bonus is 5% on every order. Offer ends 2026-12-31."
    ),
}
BOTH = ["POL-01", "POL-02"]
Q_RET = "Unused items may be returned within 30 days of delivery for a full refund"
Q_SHIP = "Standard shipping costs $5.99 and arrives in 3 to 5 business days"


def cite(doc: str, quote: str) -> str:
    return f'[doc:{doc} "{quote}"]'


def codes(verdict: gate.Verdict, i: int = 0) -> list[str]:
    return [r.code for r in verdict.sentences[i].reasons]


def run(answer: str, retrieved=BOTH) -> gate.Verdict:
    return gate.check_answer(answer, CORPUS, retrieved)


# ---- check 1: every claim sentence carries a citation ----------------------

def test_grounded_sentence_passes():
    v = run(f"Items may be returned within 30 days of delivery {cite('POL-01', Q_RET)}.")
    assert not v.blocked
    assert v.sentences[0].ok


def test_missing_citation_blocks():
    v = run("Items may be returned within 30 days.")
    assert v.blocked
    assert codes(v) == [gate.MISSING_CITATION]


def test_malformed_citation_blocks():
    v = run("Returns take 30 days [doc:POL-01].")
    assert v.blocked
    assert gate.MALFORMED_CITATION in codes(v)
    assert gate.MISSING_CITATION in codes(v)


def test_one_uncited_sentence_blocks_whole_answer():
    v = run(f"Shipping costs $5.99 {cite('POL-02', Q_SHIP)}. Returns take 30 days.")
    assert v.blocked
    assert len(v.sentences) == 2
    assert v.sentences[0].ok
    assert codes(v, 1) == [gate.MISSING_CITATION]


def test_citation_placed_after_period_attaches_to_previous_sentence():
    v = run(f"Shipping costs $5.99. {cite('POL-02', Q_SHIP)}")
    assert len(v.sentences) == 1
    assert not v.blocked


@pytest.mark.parametrize("answer", ["", "   ", "\n"])
def test_empty_answer_is_blocked_not_passed(answer):
    v = run(answer)
    assert v.blocked
    assert [r.code for r in v.answer_reasons] == [gate.EMPTY_ANSWER]


# ---- check 2: cited doc exists ---------------------------------------------

def test_unknown_doc_blocks():
    v = run(f"Items may be returned within 30 days {cite('POL-99', Q_RET)}.")
    assert v.blocked
    assert codes(v) == [gate.UNKNOWN_DOC]


# ---- check 5: cited doc is in the retrieved set ----------------------------

def test_doc_outside_retrieved_set_blocks():
    v = run(f"Items may be returned within 30 days {cite('POL-01', Q_RET)}.", retrieved=["POL-02"])
    assert v.blocked
    assert codes(v) == [gate.OUT_OF_RETRIEVAL]


def test_nothing_retrieved_blocks_any_citation():
    v = run(f"Items may be returned within 30 days {cite('POL-01', Q_RET)}.", retrieved=[])
    assert v.blocked
    assert codes(v) == [gate.OUT_OF_RETRIEVAL]


# ---- check 3: quote appears verbatim, whitespace-normalized ----------------

def test_quote_with_different_whitespace_passes():
    quote = "Unused items may be returned within 30 days of delivery for a full refund. Refunds of $12.50 or less"
    v = run(f"Returns take 30 days and small refunds of $12.50 or less are credit {cite('POL-01', quote)}.")
    assert not v.blocked


def test_altered_word_in_quote_blocks():
    quote = Q_RET.replace("within", "inside")
    v = run(f"Items may be returned inside 30 days {cite('POL-01', quote)}.")
    assert v.blocked
    assert codes(v) == [gate.QUOTE_NOT_VERBATIM]


def test_case_change_in_quote_blocks():
    v = run(f"Items may be returned within 30 days {cite('POL-01', Q_RET.lower())}.")
    assert codes(v) == [gate.QUOTE_NOT_VERBATIM]


def test_empty_quote_blocks():
    v = run(f'Items may be returned. [doc:POL-01 ""]')
    assert v.blocked
    assert gate.EMPTY_QUOTE in codes(v)


def test_quote_in_doc_unit():
    assert gate.quote_in_doc("a  b", "x a\nb y")
    assert not gate.quote_in_doc("a c", "x a\nb y")
    assert not gate.quote_in_doc("", "x")


# ---- check 4: numbers, money, percents, dates must be inside the quote -----

def test_wrong_money_blocks_and_names_the_token():
    v = run(f"Shipping costs $6.99 {cite('POL-02', Q_SHIP)}.")
    assert v.blocked
    assert codes(v) == [gate.NUMBER_NOT_IN_QUOTE]
    assert "$6.99" in v.sentences[0].reasons[0].detail


def test_percent_checked():
    q = "The loyalty bonus is 5% on every order"
    assert run(f"The bonus is 6% {cite('POL-02', q)}.").blocked
    assert not run(f"The bonus is 5% {cite('POL-02', q)}.").blocked


def test_long_form_date_checked():
    q = "Orders over $75 ship free from March 15, 2026 onward"
    assert run(f"Free shipping starts March 16, 2026 {cite('POL-02', q)}.").blocked
    assert not run(f"Orders over $75 ship free from March 15, 2026 {cite('POL-02', q)}.").blocked


def test_iso_date_checked():
    q = "Offer ends 2026-12-31"
    assert run(f"Offer ends 2026-12-30 {cite('POL-02', q)}.").blocked
    assert not run(f"Offer ends 2026-12-31 {cite('POL-02', q)}.").blocked


def test_number_present_in_doc_but_outside_quote_blocks():
    v = run(f"Standard shipping costs $5.99 {cite('POL-02', 'Standard shipping costs')}.")
    assert v.blocked
    assert codes(v) == [gate.NUMBER_NOT_IN_QUOTE]


def test_sentence_without_numbers_and_valid_quote_passes():
    v = run(f"Items may be returned {cite('POL-01', Q_RET)}.")
    assert not v.blocked


def test_numbers_union_across_multiple_citations():
    v = run(f"Returns take 30 days and shipping costs $5.99 {cite('POL-01', Q_RET)} {cite('POL-02', Q_SHIP)}.")
    assert not v.blocked


def test_one_bad_citation_among_several_blocks():
    v = run(f"Returns take 30 days {cite('POL-01', Q_RET)} {cite('POL-99', Q_RET)}.")
    assert v.blocked
    assert gate.UNKNOWN_DOC in codes(v)


def test_extract_numbers_normalizes():
    assert gate.extract_numbers("pay $12.50 and 15 % on March 3, 2026 plus 7 items") == [
        "$12.50", "15%", "march 3 2026", "7",
    ]
    assert gate.extract_numbers("1,500 units") == ["1500"]


@pytest.mark.parametrize(
    "token,text,expected",
    [
        ("30", "within 130 days", False),
        ("30", "within 30 days", True),
        ("50", "costs $50", False),
        ("$50", "costs $50.00", False),
        ("$50", "costs $50", True),
        ("5", "a 5% bonus", False),
    ],
)
def test_number_in_text_boundaries(token, text, expected):
    assert gate.number_in_text(token, text) is expected


# ---- verdict shape ---------------------------------------------------------

def test_per_sentence_reasons_and_json_shape():
    answer = (
        f"Returns take 30 days {cite('POL-01', Q_RET)}. "
        f"Shipping costs $9.99 {cite('POL-02', Q_SHIP)}. "
        "We also price match."
    )
    v = run(answer)
    d = json.loads(json.dumps(v.to_dict()))
    assert d["verdict"] == "blocked"
    assert d["sentence_count"] == 3
    assert [s["ok"] for s in d["sentences"]] == [True, False, False]
    assert d["sentences"][1]["reasons"][0]["code"] == gate.NUMBER_NOT_IN_QUOTE
    assert d["sentences"][2]["reasons"][0]["code"] == gate.MISSING_CITATION
    assert d["sentences"][0]["citations"] == [{"doc": "POL-01", "quote": Q_RET}]


def test_pass_verdict_string():
    v = run(f"Returns take 30 days {cite('POL-01', Q_RET)}.")
    assert v.to_dict()["verdict"] == "pass"


def test_check_is_deterministic():
    a = f"Shipping costs $6.99 {cite('POL-02', Q_SHIP)}."
    assert run(a).to_dict() == run(a).to_dict()


# ---- corpus loading --------------------------------------------------------

def test_load_corpus_reads_txt_files(tmp_path):
    (tmp_path / "A-1.txt").write_text("alpha  beta\ngamma", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("no", encoding="utf-8")
    assert gate.load_corpus(tmp_path) == {"A-1": "alpha beta gamma"}


def test_load_corpus_empty_or_missing_raises(tmp_path):
    with pytest.raises(gate.CorpusError):
        gate.load_corpus(tmp_path)
    with pytest.raises(gate.CorpusError):
        gate.load_corpus(tmp_path / "nope")


# ---- CLI: exit 0 pass / 1 blocked / 2 usage --------------------------------

def cli(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-m", "gate", *args],
        cwd=ROOT, input=stdin, capture_output=True, text=True, env=env, timeout=30,
    )


@pytest.fixture()
def corpus_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for k, v in CORPUS.items():
        (d / f"{k}.txt").write_text(v, encoding="utf-8")
    return d


def test_cli_pass_exit_0(corpus_dir, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(f"Returns take 30 days {cite('POL-01', Q_RET)}.", encoding="utf-8")
    r = cli(str(f), "--corpus", str(corpus_dir), "--retrieved", "POL-01,POL-02")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["verdict"] == "pass"


def test_cli_blocked_exit_1_reads_stdin(corpus_dir):
    r = cli("-", "--corpus", str(corpus_dir), "--retrieved", "POL-01", stdin="Returns take 30 days.")
    assert r.returncode == 1
    assert json.loads(r.stdout)["verdict"] == "blocked"


def test_cli_missing_retrieved_is_usage_error(corpus_dir):
    assert cli("-", "--corpus", str(corpus_dir), stdin="x").returncode == 2


def test_cli_bad_corpus_is_usage_error(tmp_path):
    r = cli("-", "--corpus", str(tmp_path / "nope"), "--retrieved", "A", stdin="x")
    assert r.returncode == 2


def test_cli_missing_answer_file_is_usage_error(corpus_dir, tmp_path):
    r = cli(str(tmp_path / "missing.txt"), "--corpus", str(corpus_dir), "--retrieved", "POL-01")
    assert r.returncode == 2


def test_cli_help_exit_0():
    r = cli("--help")
    assert r.returncode == 0
    assert "usage" in r.stdout.lower()
