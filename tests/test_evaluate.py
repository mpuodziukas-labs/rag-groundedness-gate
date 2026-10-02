"""evaluate.py: counts, table format, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluate
import gate

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "corpus" / "docs"
EVAL = ROOT / "corpus" / "eval.jsonl"


@pytest.fixture(scope="module")
def results() -> evaluate.Results:
    return evaluate.run_eval(evaluate.load_eval(EVAL), gate.load_corpus(DOCS))


def test_every_ungrounded_class_fully_detected(results):
    assert set(results.totals) == set(gate.CLASS_REASON)
    for cls, total in results.totals.items():
        assert total >= 15
        assert results.detected[cls] == total, cls


def test_no_false_positives_on_grounded_split(results):
    assert results.grounded_total >= 100
    assert results.false_positives == 0


def test_timing_is_measured_per_answer(results):
    n = results.grounded_total + sum(results.totals.values())
    assert len(results.times_ms) == n
    assert results.median_ms > 0


def test_table_uses_counts_not_percentages(results):
    table = evaluate.render_table(results)
    assert "%" not in table
    assert "/" in table
    first = results.totals and next(iter(results.totals))
    assert f"| {first} | {results.detected[first]}/{results.totals[first]} |" in table
    assert f"| False positives (grounded split) | 0/{results.grounded_total} |" in table


def test_main_exit_0_and_prints_table_and_timing(capsys):
    assert evaluate.main([]) == 0
    out = capsys.readouterr().out
    assert "| Failure class | Detected |" in out
    assert "Median per-answer check time" in out


def test_main_empty_eval_file_is_usage_error(tmp_path):
    empty = tmp_path / "e.jsonl"
    empty.write_text("", encoding="utf-8")
    assert evaluate.main(["--eval", str(empty)]) == 2


def test_main_missing_corpus_is_usage_error(tmp_path):
    assert evaluate.main(["--corpus", str(tmp_path / "nope")]) == 2


def test_ungrounded_answer_that_passes_is_a_miss():
    corpus = {"D-1": "Fees are $5."}
    recs = [{
        "id": "x", "split": "ungrounded", "class": "missing_citation",
        "retrieved": ["D-1"], "answer": 'Fees are $5 [doc:D-1 "Fees are $5"].',
        "expect_reason": gate.MISSING_CITATION, "bad_sentence": 0,
    }]
    r = evaluate.run_eval(recs, corpus)
    assert r.detected["missing_citation"] == 0
    assert r.totals["missing_citation"] == 1
