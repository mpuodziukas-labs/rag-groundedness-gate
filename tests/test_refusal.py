"""Refusal contract: an answer cites a retrieved source or the gate refuses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import gate

ROOT = Path(__file__).resolve().parent.parent

CORPUS = {
    "POL-01": "Unused items may be returned within 30 days of delivery for a full refund.",
    "POL-02": "Standard shipping costs $5.99 and arrives in 3 to 5 business days.",
}
GOOD = ('Items may be returned within 30 days '
        '[doc:POL-01 "Unused items may be returned within 30 days of delivery for a full refund"].')


def test_grounded_answer_with_citation_passes():
    decision, text = gate.respond(GOOD, CORPUS, ["POL-01"])
    assert decision == "PASS"
    assert text == GOOD


def test_ungrounded_answer_is_refused_with_fixed_message():
    decision, text = gate.respond("Items may be returned within 90 days.", CORPUS, ["POL-01"])
    assert decision == "REFUSE"
    assert text == gate.REFUSAL_MESSAGE
    assert "90 days" not in text


def test_out_of_corpus_question_is_refused():
    for retrieved in ([], ["POL-99"]):
        decision, text = gate.respond("We ship to Mars within 2 days.", CORPUS, retrieved)
        assert (decision, text) == ("REFUSE", gate.REFUSAL_MESSAGE)
    decision, _ = gate.respond(GOOD, CORPUS, [])
    assert decision == "REFUSE"


def test_citation_to_source_outside_retrieved_set_is_refused():
    decision, text = gate.respond(GOOD, CORPUS, ["POL-02"])
    assert (decision, text) == ("REFUSE", gate.REFUSAL_MESSAGE)


def test_verdict_dict_carries_decision_and_refusal():
    refused = gate.check_answer("No citation here.", CORPUS, ["POL-01"]).to_dict()
    assert refused["decision"] == "REFUSE" and refused["refusal"] == gate.REFUSAL_MESSAGE
    passed = gate.check_answer(GOOD, CORPUS, ["POL-01"]).to_dict()
    assert passed["decision"] == "PASS" and passed["refusal"] is None


def test_cli_prints_refuse_decision_and_exits_1(tmp_path):
    ans = tmp_path / "a.txt"
    ans.write_text("Items may be returned within 90 days.", encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-m", "gate", str(ans), "--retrieved", "POL-01"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert r.returncode == 1
    assert json.loads(r.stdout)["decision"] == "REFUSE"
