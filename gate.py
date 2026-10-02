"""Groundedness gate for RAG answers: offline, deterministic, standard library only.

An answer is a sequence of claim sentences. Every sentence must carry at least
one citation of the form [doc:ID "exact quote"] and each citation must pass:

  1. the sentence carries a citation and has claim text of its own
  2. the cited doc ID exists in the corpus
  3. the quote is verbatim in that doc: whitespace-normalized, on word
     boundaries, inside a single sentence of the doc, at least 3 words long
  4. every number, money amount, percent and date in the sentence appears
     inside the cited quote(s), compared by value (see Normalization)
  5. the cited doc is in the retrieved set passed with the answer
  6. at least half of the sentence's content words appear in the cited quote(s)

Normalization applied to answers, quotes and docs before any comparison:
Unicode NFKC, zero-width and other format characters removed, non-ASCII digits
folded to 0-9, smart quotes folded to straight quotes. Number words ("thirty",
"one hundred and twenty") become digits. Quantity words (dozen, half, twice) must
appear in the quote. A claim word that mixes Latin with
Cyrillic or Greek letters is blocked.

Usage:
    python -m gate ANSWER_FILE --corpus corpus/docs --retrieved POL-01,POL-02
    echo "..." | python -m gate - --retrieved POL-01

Prints a JSON verdict with per-sentence reasons (at most 200 sentences are
listed; the verdict always covers all of them).
Exit codes: 0 pass, 1 blocked, 2 usage error.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 1, 2

MISSING_CITATION = "missing_citation"
MALFORMED_CITATION = "malformed_citation"
UNKNOWN_DOC = "unknown_doc"
OUT_OF_RETRIEVAL = "out_of_retrieval"
EMPTY_QUOTE = "empty_quote"
QUOTE_TOO_SHORT = "quote_too_short"
QUOTE_NOT_VERBATIM = "quote_not_verbatim"
NUMBER_NOT_IN_QUOTE = "number_not_in_quote"
UNSUPPORTED_CLAIM = "claim_not_in_quote"
EMPTY_CLAIM = "empty_claim"
MIXED_SCRIPT = "mixed_script_text"
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
    "number_word_evasion": NUMBER_NOT_IN_QUOTE,
    "number_kind_mismatch": NUMBER_NOT_IN_QUOTE,
    "short_quote_laundering": QUOTE_TOO_SHORT,
    "quote_spans_sentences": QUOTE_NOT_VERBATIM,
    "subword_quote": QUOTE_NOT_VERBATIM,
    "unrelated_citation": UNSUPPORTED_CLAIM,
    "hidden_uncited_sentence": MISSING_CITATION,
    "citation_only": EMPTY_CLAIM,
    "mixed_script_number": MIXED_SCRIPT,
}

MIN_QUOTE_WORDS = 3
MIN_SUPPORT = 0.5          # share of a claim's content words that must appear in its quote(s)
MAX_REPORTED_SENTENCES = 200
MAX_REPORTED_REASONS = 25
CLIP = 160

CITE_RE = re.compile(r'\[doc:\s*([A-Za-z0-9_.-]+)\s+"([^"]*)"\s*\]')


class CorpusError(ValueError):
    """Corpus directory missing or empty."""


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_QUOTE_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "−": "-",
})


def normalize_text(text: str) -> str:
    """NFKC, drop format characters, fold digits and smart quotes. Keeps line breaks."""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue
        if cat == "Nd":
            ch = str(unicodedata.decimal(ch))
        out.append(ch)
    return "".join(out).translate(_QUOTE_FOLD).replace("\x00", " ")


def normalize_ws(text: str) -> str:
    return " ".join(normalize_text(text).split())


def _clip(text: str, n: int = CLIP) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


_SCRIPTS = ("LATIN", "CYRILLIC", "GREEK")


def has_mixed_script_word(text: str) -> bool:
    """True if one word mixes Latin with Cyrillic or Greek letters (homoglyph tell)."""
    for word in re.findall(r"\w+", text):
        if word.isascii():
            continue
        seen = set()
        for ch in word:
            if ch.isalpha():
                name = unicodedata.name(ch, "")
                for script in _SCRIPTS:
                    if name.startswith(script):
                        seen.add(script)
        if len(seen) > 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Numbers: extracted as canonical (kind, value) tokens so that equal values in
# different spellings compare equal and different values never do.
# ---------------------------------------------------------------------------

_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MON_ALL = (r"January|February|March|April|May|June|July|August|September|October|November|December"
            r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec")
_MON_NOMAY = _MON_ALL.replace("|May", "")
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"

NUM_RE = re.compile(
    rf"""
      (?<![\w.])(?P<iy>\d{{4}})-(?P<im>\d{{2}})-(?P<id>\d{{2}})(?!\w)
    | (?<!\w)(?P<sa>\d{{1,2}})/(?P<sb>\d{{1,2}})/(?P<sy>\d{{4}})(?!\w)
    | (?<![\w])(?P<mdm>{_MON_ALL})\.?[ ]+(?P<mdd>\d{{1,2}})(?:st|nd|rd|th)?(?!\d)(?:,?[ ]*(?P<mdy>\d{{4}})(?!\d))?
    | (?<![\w])(?P<dmd>\d{{1,2}})(?:st|nd|rd|th)?[ ]+(?:of[ ]+)?
        (?:(?P<dmm>{_MON_NOMAY})\.?(?:,?[ ]*(?P<dmy>\d{{4}})(?!\d))?|(?P<dmmay>May),?[ ]*(?P<dmyy>\d{{4}})(?!\d))
    | (?<![\w])(?P<mym>{_MON_NOMAY}|(?-i:May))\.?,?[ ]+(?P<myy>\d{{4}})(?!\d)
    | (?<![\w])(?-i:(?P<bm>January|February|March|April|June|July|August|September|October|November|December))(?!\w)
    | (?P<sign>(?<![\w.,)$])-)?
      (?:(?P<cur>US\$|USD[ ]?|\$|€|£)(?P<sign2>-)?|(?<!\w))
      (?P<num>{_NUM})
      (?:[ ](?P<magw>thousand|million|billion)(?!\w)|(?P<magl>[kmb])(?![\w]))?
      (?:[ ]?(?P<pct>%|percent(?!\w)|per[ ]?cent(?!\w))|[ ](?P<curw>dollars?|usd|euros?|eur)(?!\w)|(?P<ord>(?:st|nd|rd|th)(?!\w)))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 10**3, "million": 10**6, "billion": 10**9}
_WORD_ALT = "|".join(sorted([*_WORDS, *_SCALES], key=len, reverse=True))
_WORD_RUN = re.compile(
    rf"(?<![\w])(?:a[ ]+(?=(?:hundred|thousand|million|billion)(?!\w)))?(?:{_WORD_ALT})(?!\w)"
    rf"(?:(?:[ -]+(?:and[ ]+)?)(?:{_WORD_ALT})(?!\w))*",
    re.IGNORECASE,
)


# Quantity words that are numbers in disguise. Each must appear in the quote as
# the same word (plural folded) or the sentence is blocked as number_not_in_quote.
_QWORD_RE = re.compile(
    r"(?<![\w])(?:dozen|score|fortnight|half|halves|twice|thrice|double|triple|quarter|couple|pair"
    r"|(?:ten|eleven|twelf|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    r"|twentie|thirtie|fortie|fiftie|sixtie|seventie|eightie|ninetie|hundred|thousand|million)(?:th))"
    r"s?(?!\w)",
    re.IGNORECASE,
)


def _run_value(run: str) -> int | None:
    total = current = 0
    for word in re.findall(r"[a-z]+", run.lower()):
        if word == "a":
            current = 1
        elif word in _WORDS:
            current += _WORDS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in _SCALES:
            total += (current or 1) * _SCALES[word]
            current = 0
    return total + current


def words_to_digits(text: str) -> str:
    """Replace number words with digits. A lone 'one' stays a word (pronoun use)."""
    def repl(m: re.Match[str]) -> str:
        run = m.group(0)
        if run.lower() == "one":
            return run
        value = _run_value(run)
        return str(value) if value is not None else run

    return _WORD_RUN.sub(repl, text)


def _dec(raw: str, negative: bool = False, mult: int = 1) -> str:
    try:
        value = Decimal(raw.replace(",", "")) * mult
    except InvalidOperation:  # pragma: no cover - the regex only yields valid numbers
        return raw
    if negative:
        value = -value
    out = format(value.normalize(), "f")
    return "0" if out in ("-0", "") else out


def _two(n: int | str) -> str:
    return f"{int(n):02d}"


def _date_tokens(month: int, day: int | None, year: int | None) -> list[str]:
    if not 1 <= month <= 12 or (day is not None and not 1 <= day <= 31):
        return []
    if day is not None and year is not None:
        return [f"date:{year:04d}-{_two(month)}-{_two(day)}"]
    if day is not None:
        return [f"md:{_two(month)}-{_two(day)}"]
    if year is not None:
        return [f"ym:{year:04d}-{_two(month)}"]
    return [f"month:{_two(month)}"]


def extract_numbers(text: str) -> list[str]:
    """Canonical tokens for every number, money amount, percent and date in text.

    Examples: 30 -> num:30, $1,000 and 1k dollars -> usd:1000, 5 percent -> pct:5,
    March 15, 2026 -> date:2026-03-15, a bare March -> month:03.
    """
    clean = words_to_digits(normalize_text(text))
    tokens: list[str] = []
    for m in NUM_RE.finditer(clean):
        g = m.groupdict()
        if g["iy"]:
            tokens += _date_tokens(int(g["im"]), int(g["id"]), int(g["iy"])) or [f"num:{_dec(g['iy'])}"]
        elif g["sa"]:
            a, b, y = int(g["sa"]), int(g["sb"]), int(g["sy"])
            month, day = (b, a) if a > 12 and b <= 12 else (a, b)
            tokens += _date_tokens(month, day, y) or [f"num:{a}", f"num:{b}", f"num:{y}"]
        elif g["mdm"]:
            year = int(g["mdy"]) if g["mdy"] else None
            tokens += _date_tokens(_MONTH_NUM[g["mdm"].lower()], int(g["mdd"]), year) or [f"num:{int(g['mdd'])}"]
        elif g["dmd"]:
            name = g["dmm"] or g["dmmay"]
            year_s = g["dmy"] or g["dmyy"]
            year = int(year_s) if year_s else None
            tokens += _date_tokens(_MONTH_NUM[name.lower()], int(g["dmd"]), year) or [f"num:{int(g['dmd'])}"]
        elif g["mym"]:
            tokens += _date_tokens(_MONTH_NUM[g["mym"].lower()], None, int(g["myy"]))
        elif g["bm"]:
            tokens += _date_tokens(_MONTH_NUM[g["bm"].lower()], None, None)
        else:
            mult = 1
            if g["magw"]:
                mult = _SCALES[g["magw"].lower()]
            elif g["magl"]:
                mult = {"k": 10**3, "m": 10**6, "b": 10**9}[g["magl"].lower()]
            value = _dec(g["num"], negative=bool(g["sign"]) != bool(g["sign2"]), mult=mult)
            cur = (g["cur"] or "").strip().lower()
            word = (g["curw"] or "").lower()
            if cur in ("$", "us$", "usd") or word in ("dollar", "dollars", "usd"):
                kind = "usd"
            elif cur == "€" or word in ("euro", "euros", "eur"):
                kind = "eur"
            elif cur == "£":
                kind = "gbp"
            elif g["pct"]:
                kind = "pct"
            else:
                kind = "num"
            tokens.append(f"{kind}:{value}")
    for m in _QWORD_RE.finditer(clean):
        w = m.group(0).lower()
        tokens.append("word:" + ("half" if w == "halves" else w[:-1] if w.endswith("s") and w != "twice" else w))
    return tokens


def _support_set(text: str) -> set[str]:
    """Tokens a quote supports: its own, plus the parts of any date it contains."""
    out: set[str] = set()
    for tok in extract_numbers(text):
        out.add(tok)
        kind, _, rest = tok.partition(":")
        if kind == "date":
            y, mo, d = rest.split("-")
            out |= {f"md:{mo}-{d}", f"ym:{y}-{mo}", f"month:{mo}", f"num:{int(y)}", f"num:{int(d)}"}
        elif kind == "md":
            mo, d = rest.split("-")
            out |= {f"month:{mo}", f"num:{int(d)}"}
        elif kind == "ym":
            y, mo = rest.split("-")
            out |= {f"month:{mo}", f"num:{int(y)}"}
    return out


def display_number(token: str) -> str:
    """Readable form of a canonical token for reason details: usd:6.99 -> $6.99."""
    kind, _, value = token.partition(":")
    symbol = {"usd": "$", "eur": "\u20ac", "gbp": "\u00a3"}.get(kind)
    if symbol:
        return f"-{symbol}{value[1:]}" if value.startswith("-") else f"{symbol}{value}"
    return f"{value}%" if kind == "pct" else value


def _norm_number(token: str) -> str:
    """Canonical form of the first number in a string (identity if it has none)."""
    found = extract_numbers(token)
    return found[0] if found else token


def number_in_text(token: str, text: str) -> bool:
    """True if the canonical token is supported by a number in text."""
    return token in _support_set(text)


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

_ABBREVS = {"e.g.", "i.e.", "vs.", "cf.", "approx.", "incl.", "inc.", "ltd.", "dr.", "mr.", "mrs.", "ms.", "fig."}
_TERMINATORS = "[.!?。]+"


def _is_abbrev(text: str, end: int) -> bool:
    tail = text[max(0, end - 16):end].split()  # bounded look-back keeps this linear
    return bool(tail) and tail[-1].lstrip("([\"'").lower() in _ABBREVS


def _sentence_breaks(text: str) -> list[int]:
    """Indexes of the whitespace character that follows a sentence terminator."""
    return [m.end() for m in re.finditer(rf"{_TERMINATORS}(?=\s)", text) if not _is_abbrev(text, m.end())]


def quote_in_doc(quote: str, doc: str) -> bool:
    """Check 3: quote is non-empty and verbatim in doc, whitespace-normalized,
    on word and number boundaries, and inside one sentence of the doc."""
    q = normalize_ws(quote)
    if not q:
        return False
    d = normalize_ws(doc)
    breaks = _sentence_breaks(d)
    start = d.find(q)
    while start != -1:
        end = start + len(q)
        left_ok = not (q[0].isalnum() and start > 0 and (
            d[start - 1].isalnum() or (q[0].isdigit() and d[start - 1] in ".," and start > 1 and d[start - 2].isdigit())))
        right_ok = not (q[-1].isalnum() and end < len(d) and (
            d[end].isalnum() or (q[-1].isdigit() and d[end] in ".," and end + 1 < len(d) and d[end + 1].isdigit())))
        i = bisect.bisect_left(breaks, start)
        crosses = i < len(breaks) and breaks[i] < end
        if left_ok and right_ok and not crosses:
            return True
        start = d.find(q, start + 1)
    return False


def missing_numbers(sentence: str, quotes: Sequence[str]) -> list[str]:
    """Check 4: canonical numbers in the sentence that no cited quote supports."""
    support = _support_set(" ".join(quotes))
    seen: list[str] = []
    for tok in extract_numbers(sentence):
        if tok not in support and tok not in seen:
            seen.append(tok)
    return seen


_STOP = frozenset(
    "the and for are was were you your our can may has have had with that this from per any all not its "
    "their will shall must but who how out off into onto than then there these those they them been being "
    "also only each both such more most very just would could should about over under".split()
)


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def support_ratio(sentence: str, quotes: Sequence[str]) -> float:
    """Share of the sentence's content words that appear in the cited quotes (1.0 if none)."""
    claim = {_stem(w) for w in re.findall(r"[a-z]{3,}|\d+", normalize_text(sentence).lower()) if w not in _STOP}
    if not claim:
        return 1.0
    quoted = {_stem(w) for w in re.findall(r"[a-z]+|\d+", normalize_text(" ".join(quotes)).lower())}
    return len(claim & quoted) / len(claim)


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

_LINE_BREAKS = re.compile(r"[\r\n  \x0b\x0c\x85]+")
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022\u2013\u2014]\s+|\d{1,3}[.)]\s+)")
_BOUNDARY = re.compile(rf"{_TERMINATORS}(?=\s|\x00)|(?<=[a-z]{{2}})[.!?](?=[A-Z])|\u3002|;")


def _split_line(line: str) -> list[str]:
    pieces, last = [], 0
    for m in _BOUNDARY.finditer(line):
        if m.group(0)[0] in ".!?。" and _is_abbrev(line, m.end()):
            continue
        pieces.append(line[last:m.end()])
        last = m.end()
    pieces.append(line[last:])
    return pieces


def split_sentences(answer: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Split into (claim text without citations, [(doc_id, quote), ...]).

    Citations are masked before splitting so periods inside quotes never split.
    Splits on line breaks, list bullets, ';' and sentence terminators (abbreviations
    like "e.g." and decimals do not split). A fragment made only of citations
    attaches to the previous sentence.
    """
    cites: list[tuple[str, str]] = []

    def mask(m: re.Match[str]) -> str:
        cites.append((m.group(1), m.group(2)))
        return f"\x00{len(cites) - 1}\x00"

    masked = CITE_RE.sub(mask, normalize_text(answer))
    out: list[tuple[str, list[tuple[str, str]]]] = []
    for line in _LINE_BREAKS.split(masked):
        line = _LIST_MARKER.sub("", line.strip())
        if not line:
            continue
        for part in _split_line(line):
            if not part.strip():
                continue
            ids = [int(i) for i in re.findall(r"\x00(\d+)\x00", part)]
            text = normalize_ws(re.sub(r"\x00\d+\x00", " ", part)).strip(" ;")
            if out and not re.search(r"\w", text):
                out[-1][1].extend(cites[i] for i in ids)
            else:
                out.append((text, [cites[i] for i in ids]))
    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

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
        shown = self.sentences[:MAX_REPORTED_SENTENCES]
        return {
            "verdict": "blocked" if self.blocked else "pass",
            "sentence_count": len(self.sentences),
            "sentences_listed": len(shown),
            "answer_reasons": [vars(r) for r in self.answer_reasons],
            "sentences": [
                {
                    "index": s.index,
                    "text": _clip(s.text, 300),
                    "citations": [{"doc": c["doc"], "quote": _clip(c["quote"], 300)} for c in s.citations[:10]],
                    "ok": s.ok,
                    "reasons": [vars(r) for r in s.reasons[:MAX_REPORTED_REASONS]],
                }
                for s in shown
            ],
        }


def check_answer(answer: str, corpus: Mapping[str, str], retrieved: Iterable[str]) -> Verdict:
    """Run every check and return a verdict with per-sentence reasons."""
    if isinstance(retrieved, (str, bytes)):
        raise TypeError("retrieved must be an iterable of doc IDs, not a single string")
    retrieved_set = set(retrieved)
    sentences: list[SentenceResult] = []
    for i, (text, cites) in enumerate(split_sentences(answer)):
        res = SentenceResult(i, text, [{"doc": d, "quote": q} for d, q in cites])
        if not re.search(r"\w", text):
            res.reasons.append(Reason(EMPTY_CLAIM, "sentence has no claim text of its own"))
        if "[doc:" in text:
            res.reasons.append(Reason(MALFORMED_CITATION, "citation must look like [doc:ID \"exact quote\"]"))
        if not cites:
            res.reasons.append(Reason(MISSING_CITATION, "sentence carries no citation"))
        if has_mixed_script_word(text):
            res.reasons.append(Reason(MIXED_SCRIPT, "a word mixes Latin with Cyrillic or Greek letters"))
        for doc_id, quote in cites:
            if doc_id not in corpus:
                res.reasons.append(Reason(UNKNOWN_DOC, f"{_clip(doc_id)} is not in the corpus"))
            elif doc_id not in retrieved_set:
                res.reasons.append(Reason(OUT_OF_RETRIEVAL, f"{doc_id} was not in the retrieved set"))
            elif not normalize_ws(quote):
                res.reasons.append(Reason(EMPTY_QUOTE, f"empty quote cited from {doc_id}"))
            elif len(normalize_ws(quote).split()) < MIN_QUOTE_WORDS:
                res.reasons.append(Reason(QUOTE_TOO_SHORT, f"quote from {doc_id} has fewer than {MIN_QUOTE_WORDS} words"))
            elif not quote_in_doc(quote, corpus[doc_id]):
                res.reasons.append(Reason(QUOTE_NOT_VERBATIM, f"quote not found verbatim in {doc_id}: {_clip(quote)}"))
        if cites:
            quotes = [q for _, q in cites]
            for token in missing_numbers(text, quotes)[:MAX_REPORTED_REASONS]:
                res.reasons.append(Reason(NUMBER_NOT_IN_QUOTE, f"{display_number(token)} is not inside the cited quote"))
            ratio = support_ratio(text, quotes)
            if ratio < MIN_SUPPORT:
                res.reasons.append(Reason(UNSUPPORTED_CLAIM, f"only {ratio:.0%} of the claim's content words appear in the cited quote"))
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
        raw = sys.stdin.buffer.read() if args.answer == "-" else Path(args.answer).read_bytes()
        text = raw.decode("utf-8")
    except (OSError, ValueError) as exc:  # CorpusError and UnicodeDecodeError are ValueErrors
        print(f"gate: {exc}", file=sys.stderr)
        return EXIT_USAGE
    retrieved = [r.strip() for r in args.retrieved.split(",") if r.strip()]
    verdict = check_answer(text, corpus, retrieved)
    print(json.dumps(verdict.to_dict(), indent=2))
    return EXIT_BLOCKED if verdict.blocked else EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
