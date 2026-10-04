"""Planted RED: neuter a check and prove the evaluation notices.

For checks 3 (verbatim quote) and 4 (numbers inside quote) the check is
monkeypatched to always pass. The eval must then show missed detections in the
classes that depend on that check and exit nonzero. The unmutated control must
stay fully green, so the mutants are not passing vacuously.
"""

from __future__ import annotations

from pathlib import Path

import evaluate
import gate

ROOT = Path(__file__).resolve().parent.parent


def run_eval() -> evaluate.Results:
    return evaluate.run_eval(evaluate.load_eval(ROOT / "corpus" / "eval.jsonl"),
                             gate.load_corpus(ROOT / "corpus" / "docs"))


def test_control_unmutated_is_fully_green():
    r = run_eval()
    assert all(r.detected[c] == r.totals[c] for c in r.totals)
    assert r.false_positives == 0
    assert evaluate.main([]) == 0


def test_mutant_check3_always_pass_is_caught(monkeypatch):
    monkeypatch.setattr(gate, "quote_in_doc", lambda quote, doc: True)
    r = run_eval()
    for cls in ("fabricated_quote", "paraphrase_drift"):
        assert r.detected[cls] < r.totals[cls], cls
    for cls in ("missing_citation", "unknown_doc", "out_of_retrieval", "wrong_number", "number_outside_quote"):
        assert r.detected[cls] == r.totals[cls], cls
    assert evaluate.main([]) == 1


def test_mutant_check4_always_pass_is_caught(monkeypatch):
    monkeypatch.setattr(gate, "missing_numbers", lambda sentence, quotes: [])
    r = run_eval()
    for cls in ("wrong_number", "number_outside_quote"):
        assert r.detected[cls] < r.totals[cls], cls
    for cls in ("missing_citation", "unknown_doc", "out_of_retrieval", "fabricated_quote", "paraphrase_drift"):
        assert r.detected[cls] == r.totals[cls], cls
    assert evaluate.main([]) == 1


def test_mutant_min_quote_length_removed_is_caught(monkeypatch):
    monkeypatch.setattr(gate, "MIN_QUOTE_WORDS", 0)
    r = run_eval()
    assert r.detected["short_quote_laundering"] < r.totals["short_quote_laundering"]
    assert r.detected["fabricated_quote"] == r.totals["fabricated_quote"]
    assert evaluate.main([]) == 1


def test_mutant_support_check_removed_is_caught(monkeypatch):
    monkeypatch.setattr(gate, "MIN_SUPPORT", 0.0)
    monkeypatch.setattr(gate, "MAX_NOVEL", 10**6)
    r = run_eval()
    assert r.detected["unrelated_citation"] < r.totals["unrelated_citation"]
    assert r.detected["wrong_number"] == r.totals["wrong_number"]
    assert evaluate.main([]) == 1


def test_mutant_mixed_script_check_removed_is_caught(monkeypatch):
    monkeypatch.setattr(gate, "has_mixed_script_word", lambda text: False)
    r = run_eval()
    assert r.detected["mixed_script_number"] < r.totals["mixed_script_number"]
    assert r.detected["number_word_evasion"] == r.totals["number_word_evasion"]
    assert evaluate.main([]) == 1


def test_mutant_novel_word_cap_removed_is_caught(monkeypatch):
    """Without the cap, appended facts built from a few invented words pass (hostile finding H3)."""
    monkeypatch.setattr(gate, "MAX_NOVEL", 10**6)
    monkeypatch.setattr(gate, "MIN_SUPPORT", 0.0)
    answer = ('Standard shipping costs $5.99, arrives in 3 to 5 business days, never refundable '
              '[doc:POL-03 "Standard shipping costs $5.99 and arrives in 3 to 5 business days"].')
    v = gate.check_answer(answer, gate.load_corpus(ROOT / "corpus" / "docs"), ["POL-03"])
    assert not v.blocked
