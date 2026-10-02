"""Groundedness gate for RAG answers: offline, deterministic, standard library only.

An answer is a sequence of claim sentences. Every sentence must carry at least
one citation of the form [doc:ID "exact quote"] and each citation must pass:

  1. the sentence carries a citation
  2. the cited doc ID exists in the corpus
  3. the quote appears verbatim in that doc (whitespace-normalized)
  4. every number, money amount, percent and date in the sentence appears
     inside the cited quote(s)
  5. the cited doc is in the retrieved set passed with the answer

Usage:
    python -m gate ANSWER_FILE --corpus corpus/docs --retrieved POL-01,POL-02
    echo "..." | python -m gate - --retrieved POL-01

Prints a JSON verdict with per-sentence reasons.
Exit codes: 0 pass, 1 blocked, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 1, 2

MISSING_CITATION = "missing_citation"
MALFORMED_CITATION = "malformed_citation"
UNKNOWN_DOC = "unknown_doc"
OUT_OF_RETRIEVAL = "out_of_retrieval"
EMPTY_QUOTE = "empty_quote"
QUOTE_NOT_VERBATIM = "quote_not_verbatim"
NUMBER_NOT_IN_QUOTE = "number_not_in_quote"
EMPTY_ANSWER = "empty_answer"

# Failure class (used by the eval set) -> the reason code that must catch it.
CLASS_REASON: dict[str, str] = {
    "fabricated_quote": QUOTE_NOT_VERBATIM,
    "wrong_number": NUMBER_NOT_IN_QUOTE,
    "missing_citation": MISSING_CITATION,
    "unknown_doc": UNKNOWN_DOC,
    "out_of_retrieval": OUT_OF_RETRIEVAL,
    "number_outside_quote": NUMBER_NOT_IN_QUOTE,
    "paraphrase_drift": QUOTE_NOT_VERBATIM,
}

CITE_RE = re.compile(r'\[doc:\s*([A-Za-z0-9_.-]+)\s+"([^"]*)"\s*\]')

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
NUM_RE = re.compile(
    r"(?<![\w.$])(?:"
    r"\d{4}-\d{2}-\d{2}"                                   # ISO date
    rf"|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?"  # March 15, 2026
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s*\d{{4}})?"  # 15 March 2026
    r"|\$\d+(?:,\d{3})*(?:\.\d+)?"                        # money
    r"|\d+(?:,\d{3})*(?:\.\d+)?\s?%"                      # percent
    r"|\d+(?:,\d{3})*(?:\.\d+)?"                          # plain number
    r")",
    re.IGNORECASE,
)


class CorpusError(ValueError):
    """Corpus directory missing or empty."""


@dataclass(frozen=True)
class Reason:
    code: str
    detail: str


@dataclass
class SentenceResult:
    index: int
    text: str
    citations: list[dict[str, str]]
    reasons: list[Reason] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


@dataclass
class Verdict:
    sentences: list[SentenceResult]
    answer_reasons: list[Reason] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.answer_reasons) or any(not s.ok for s in self.sentences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": "blocked" if self.blocked else "pass",
            "sentence_count": len(self.sentences),
            "answer_reasons": [vars(r) for r in self.answer_reasons],
            "sentences": [
                {
                    "index": s.index,
                    "text": s.text,
                    "citations": s.citations,
                    "ok": s.ok,
                    "reasons": [vars(r) for r in s.reasons],
                }
                for s in self.sentences
            ],
        }


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _norm_number(token: str) -> str:
    token = re.sub(r"\s+", " ", token.casefold().replace(",", "")).strip()
    return re.sub(r"\s%", "%", token)


def extract_numbers(text: str) -> list[str]:
    """Numbers, money, percents and dates in text, normalized for comparison."""
    return [_norm_number(m.group(0)) for m in NUM_RE.finditer(text)]


def number_in_text(token: str, text: str) -> bool:
    """True if the normalized token appears in text on number boundaries."""
    haystack = _norm_number(text)
    pattern = r"(?<![\w$.])" + re.escape(token) + r"(?![\w%]|\.\d)"
    return re.search(pattern, haystack) is not None


def quote_in_doc(quote: str, doc: str) -> bool:
    """Check 3: quote is non-empty and appears verbatim in doc, whitespace-normalized."""
    q = normalize_ws(quote)
    return bool(q) and q in normalize_ws(doc)


def missing_numbers(sentence: str, quotes: Sequence[str]) -> list[str]:
    """Check 4: numbers in the sentence that appear in none of the cited quotes."""
    joined = " ".join(quotes)
    return [t for t in extract_numbers(sentence) if not number_in_text(t, joined)]


def split_sentences(answer: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Split into (claim text without citations, [(doc_id, quote), ...]).

    Citations are masked before splitting so periods inside quotes never split.
    A fragment made only of citations attaches to the previous sentence.
    """
    cites: list[tuple[str, str]] = []

    def mask(m: re.Match[str]) -> str:
        cites.append((m.group(1), m.group(2)))
        return f"\x00{len(cites) - 1}\x00"

    masked = CITE_RE.sub(mask, answer.replace("\x00", " ")).strip()
    parts = [p for p in re.split(r"(?<=[.!?])\s+", masked) if p.strip()]
    out: list[tuple[str, list[tuple[str, str]]]] = []
    for part in parts:
        ids = [int(i) for i in re.findall(r"\x00(\d+)\x00", part)]
        text = normalize_ws(re.sub(r"\x00\d+\x00", " ", part))
        if out and not re.search(r"\w", text):
            out[-1][1].extend(cites[i] for i in ids)
        else:
            out.append((text, [cites[i] for i in ids]))
    return out


def check_answer(answer: str, corpus: Mapping[str, str], retrieved: Iterable[str]) -> Verdict:
    """Run all five checks and return a verdict with per-sentence reasons."""
    retrieved_set = set(retrieved)
    sentences: list[SentenceResult] = []
    for i, (text, cites) in enumerate(split_sentences(answer)):
        res = SentenceResult(i, text, [{"doc": d, "quote": q} for d, q in cites])
        if "[doc:" in text:
            res.reasons.append(Reason(MALFORMED_CITATION, "citation must look like [doc:ID \"exact quote\"]"))
        if not cites:
            res.reasons.append(Reason(MISSING_CITATION, "sentence carries no citation"))
        for doc_id, quote in cites:
            if doc_id not in corpus:
                res.reasons.append(Reason(UNKNOWN_DOC, f"{doc_id} is not in the corpus"))
            elif doc_id not in retrieved_set:
                res.reasons.append(Reason(OUT_OF_RETRIEVAL, f"{doc_id} was not in the retrieved set"))
            elif not normalize_ws(quote):
                res.reasons.append(Reason(EMPTY_QUOTE, f"empty quote cited from {doc_id}"))
            elif not quote_in_doc(quote, corpus[doc_id]):
                res.reasons.append(Reason(QUOTE_NOT_VERBATIM, f"quote not found verbatim in {doc_id}: {quote}"))
        if cites:
            for token in missing_numbers(text, [q for _, q in cites]):
                res.reasons.append(Reason(NUMBER_NOT_IN_QUOTE, f"{token} is not inside the cited quote"))
        sentences.append(res)
    answer_reasons = [] if sentences else [Reason(EMPTY_ANSWER, "answer contains no sentences")]
    return Verdict(sentences, answer_reasons)


def load_corpus(path: str | Path) -> dict[str, str]:
    """Read every *.txt in a directory: doc ID is the file stem, text is whitespace-normalized."""
    d = Path(path)
    if not d.is_dir():
        raise CorpusError(f"corpus directory not found: {d}")
    docs = {p.stem: normalize_ws(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.txt"))}
    if not docs:
        raise CorpusError(f"corpus directory has no .txt docs: {d}")
    return docs


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gate",
        description="Block RAG answers whose claims are not grounded in cited, retrieved docs. "
        "Exit 0 pass, 1 blocked, 2 usage error.",
    )
    p.add_argument("answer", help="path to a text file holding the answer, or - for stdin")
    p.add_argument("--corpus", default="corpus/docs", help="directory of *.txt docs (default: corpus/docs)")
    p.add_argument("--retrieved", required=True,
                   help="comma-separated doc IDs retrieved for this answer (empty string = none)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)  # argparse exits 2 on bad usage
    try:
        corpus = load_corpus(args.corpus)
        text = sys.stdin.read() if args.answer == "-" else Path(args.answer).read_text(encoding="utf-8")
    except (CorpusError, OSError) as exc:
        print(f"gate: {exc}", file=sys.stderr)
        return EXIT_USAGE
    retrieved = [r.strip() for r in args.retrieved.split(",") if r.strip()]
    verdict = check_answer(text, corpus, retrieved)
    print(json.dumps(verdict.to_dict(), indent=2))
    return EXIT_BLOCKED if verdict.blocked else EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
