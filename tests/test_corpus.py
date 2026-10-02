"""The synthetic corpus and eval set: size, labels, determinism."""

from __future__ import annotations

import json
from pathlib import Path

import build_corpus
import gate

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "corpus" / "docs"
EVAL = ROOT / "corpus" / "eval.jsonl"


def records() -> list[dict]:
    return [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_thirty_docs_all_labeled_synthetic():
    files = sorted(DOCS.glob("*.txt"))
    assert len(files) == 30
    for f in files:
        assert f.read_text(encoding="utf-8").startswith("[SYNTHETIC]"), f.name


def test_eval_size_and_splits():
    recs = records()
    assert len(recs) >= 200
    assert sum(r["split"] == "grounded" for r in recs) >= 100
    assert len({r["id"] for r in recs}) == len(recs)


def test_every_failure_class_has_at_least_fifteen_cases():
    recs = records()
    for cls in gate.CLASS_REASON:
        assert sum(r["class"] == cls for r in recs) >= 15, cls


def test_ungrounded_records_name_the_bad_sentence_and_expected_reason():
    for r in records():
        if r["split"] == "ungrounded":
            assert r["expect_reason"] == gate.CLASS_REASON[r["class"]]
            assert isinstance(r["bad_sentence"], int)
        else:
            assert r["class"] == "grounded"
            assert r["expect_reason"] is None


def test_every_cited_doc_in_grounded_answers_exists():
    corpus = gate.load_corpus(DOCS)
    for r in records():
        if r["split"] == "grounded":
            assert set(r["retrieved"]) <= set(corpus)


def test_build_matches_committed_files_byte_for_byte(tmp_path):
    build_corpus.write_all(tmp_path)
    assert (tmp_path / "eval.jsonl").read_bytes() == EVAL.read_bytes()
    built = sorted(p.name for p in (tmp_path / "docs").glob("*.txt"))
    assert built == sorted(p.name for p in DOCS.glob("*.txt"))
    for name in built:
        assert (tmp_path / "docs" / name).read_bytes() == (DOCS / name).read_bytes()


def test_build_twice_is_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build_corpus.write_all(a)
    build_corpus.write_all(b)
    assert (a / "eval.jsonl").read_bytes() == (b / "eval.jsonl").read_bytes()
