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
