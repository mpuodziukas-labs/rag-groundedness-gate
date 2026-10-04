"""Red-team suite: attacks on the gate, written to fail against a naive implementation.

Each group is one attack family. A test is one of:
  BLOCK  an ungrounded answer that must be blocked
  PASS   a grounded answer that must not be blocked (false-positive attack)
  PIN    a limit that cannot be fixed deterministically; the test pins today's
         behavior so the README limitation stays honest (see Limitations)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import evaluate
import gate

ROOT = Path(__file__).resolve().parent.parent

# One sentence per line in the source; the corpus text is the space-joined sentences.
RET = "Unused items may be returned within 30 days of delivery for a full refund"
FINAL = "Final sale items cannot be returned after 48 hours from checkout"
WARR = "Warranty claims close within 130 days of purchase for all items"
FEE = "Customers pay a restocking fee of 10% on opened electronics"
POINTS = "Members earn 5 points back on every eligible order"
POINTS_PCT = "Members earn 5% back in points on every eligible order"
FREIGHT = "Premium freight costs $1,000 per pallet within the region"
GIFT = "Gift cards are issued in amounts from $10 to $500 at any store"
NEGBAL = "The account balance fell to -$5 after the adjustment"
NOREFUND = "Refunds are not available after 30 days from delivery"
SPRING = "The spring window closes on March 15, 2026 for all stores"
SHIPDATE = "Orders ship in 3 business days after the window closes on September 9, 2026"
THE = "Use the code at checkout to get the discount on the order"
EG = "Items (e.g. shoes) may be returned within 14 days of delivery for store credit"
APOS = "The store doesn't accept returns of opened software"
DEC = "Shipping takes 2.5 days across the region"
CODE = "Use code [A1] at checkout to save on the order"
WINDOW = "The window opens March 15, 2026 and closes April 20, 2026 for all members"
HOURS = "Support answers messages within 30 days of the original request"

DOC1 = ". ".join([RET, FINAL, WARR]) + "."
DOC2 = ". ".join([FEE, POINTS, POINTS_PCT, FREIGHT, GIFT, NEGBAL, NOREFUND, SPRING, SHIPDATE, THE,
                 EG, APOS, DEC, CODE, WINDOW, HOURS]) + "."
CORPUS = {"RT-1": DOC1, "RT-2": DOC2}
BOTH = ["RT-1", "RT-2"]


def doc_of(quote: str) -> str:
    return "RT-1" if quote in DOC1 else "RT-2"


def cite(quote: str, doc: str | None = None) -> str:
    return f'[doc:{doc or doc_of(quote)} "{quote}"]'


def run(answer: str, retrieved=BOTH) -> gate.Verdict:
    return gate.check_answer(answer, CORPUS, retrieved)


def claim(sentence: str, *quotes: str) -> str:
    return f"{sentence} {' '.join(cite(q) for q in quotes)}."


# ---------------------------------------------------------------------------
# 1. Number evasion
# ---------------------------------------------------------------------------

NUMBER_BLOCK = [
    ("forty", "Items may be returned within forty days of delivery for a full refund", RET),
    ("hyphenated", "Items may be returned within twenty-five days of delivery for a full refund", RET),
    ("hundred and", "Items may be returned within one hundred and twenty days of delivery for a full refund", RET),
    ("a hundred", "Items may be returned within a hundred days of delivery for a full refund", RET),
    ("spaced tens", "Items may be returned within forty five days of delivery for a full refund", RET),
    ("word vs 130", "Warranty claims close within thirty days of purchase for all items", WARR),
    ("superscript", "Items may be returned within \u2074\u2070 days of delivery for a full refund", RET),
    ("circled", "Items may be returned within \u2469 days of delivery for a full refund", RET),
    ("k suffix", "Gift cards are issued in amounts from $10 to $500k at any store", GIFT),
    ("thousand suffix", "Gift cards are issued in amounts from $10 to $500 thousand at any store", GIFT),
    ("million suffix", "Gift cards are issued in amounts from $10 to $500 million at any store", GIFT),
    ("percent word on a points quote", "Members earn 5 percent back on every eligible order", POINTS),
    ("percent via word number", "Members earn five percent back on every eligible order", POINTS),
    ("leading-dot percent", "Customers pay a restocking fee of .5% on opened electronics", FEE),
    ("US$ prefix", "Premium freight costs US$5 per pallet within the region", FREIGHT),
    ("sign dropped", "The account balance fell to $5 after the adjustment", NEGBAL),
    ("sign invented", "Premium freight costs -$1,000 per pallet within the region", FREIGHT),
    ("abbreviated month date", "Orders ship in 3 business days after the window closes on Sept 3, 2026", SHIPDATE),
    ("bare month swap", "The spring window closes in April for all stores", SPRING),
    ("wrong long date", "The spring window closes on 16 March 2026 for all stores", SPRING),
    ("wrong slash date", "The spring window closes on 03/16/2026 for all stores", SPRING),
    ("euro for dollars", "Premium freight costs \u20ac1,000 per pallet within the region", FREIGHT),
]


@pytest.mark.parametrize("label,sentence,quote", NUMBER_BLOCK, ids=[c[0] for c in NUMBER_BLOCK])
def test_number_evasion_is_blocked(label, sentence, quote):
    assert run(claim(sentence, quote)).blocked, label


NUMBER_PASS = [
    ("number word equals digit", "Items may be returned within thirty days of delivery for a full refund", RET),
    ("fullwidth digits", "Items may be returned within \uff13\uff10 days of delivery for a full refund", RET),
    ("arabic-indic digits", "Items may be returned within \u0663\u0660 days of delivery for a full refund", RET),
    ("zero-width inside digits", "Items may be returned within 3\u200b0 days of delivery for a full refund", RET),
    ("$1000 vs $1,000", "Premium freight costs $1000 per pallet within the region", FREIGHT),
    ("$1k vs $1,000", "Premium freight costs $1k per pallet within the region", FREIGHT),
    ("1,000 dollars vs $1,000", "Premium freight costs 1,000 dollars per pallet within the region", FREIGHT),
    ("percent word vs %", "Members earn 5 percent back in points on every eligible order", POINTS_PCT),
    ("slash date", "The spring window closes on 03/15/2026 for all stores", SPRING),
    ("iso date", "The spring window closes on 2026-03-15 for all stores", SPRING),
    ("abbreviated month", "The spring window closes on Mar 15, 2026 for all stores", SPRING),
    ("ordinal day", "The spring window closes on March 15th, 2026 for all stores", SPRING),
    ("day-first date", "The spring window closes on 15 March 2026 for all stores", SPRING),
    ("negative sign kept", "The account balance fell to -$5 after the adjustment", NEGBAL),
    ("decimal", "Shipping takes 2.5 days across the region", DEC),
]


@pytest.mark.parametrize("label,sentence,quote", NUMBER_PASS, ids=[c[0] for c in NUMBER_PASS])
def test_equivalent_number_forms_are_not_blocked(label, sentence, quote):
    v = run(claim(sentence, quote))
    assert not v.blocked, (label, [r.code for r in v.sentences[0].reasons])


# ---------------------------------------------------------------------------
# 2. Quote laundering
# ---------------------------------------------------------------------------

LAUNDER_BLOCK = [
    ("quote is one stopword", "Refunds are instant for everyone", [THE[:7]]),  # 'Use the'
    ("quote is 'the'", "Refunds are instant for everyone", ["the"]),
    ("quote is a single number", "Items may be returned within 30 days of delivery", ["30"]),
    ("quote is two words", "Items may be returned within 30 days of delivery", ["30 days"]),
    ("quote is one letter", "Refunds are instant for everyone", ["a"]),
    ("quote shares one content word of four", "Gold members enjoy instant refunds anywhere", [RET]),
    ("quote shares no content word", "Gold members enjoy instant upgrades anywhere", [RET]),
    ("whole document as quote", "Final sale items cannot be returned after 30 hours from checkout", [DOC1]),
    ("quote spans two sentences", "Unused items may be returned within 30 days of delivery for a full refund",
     [f"{RET}. {FINAL}"]),
    ("quote starts inside a number", "Warranty claims close within 30 days of purchase for all items",
     ["30 days of purchase for all items"]),
    ("quote starts inside a word", "Items may be returned within 30 days of delivery for a full refund",
     ["nused items may be returned within 30 days of delivery for a full refund"]),
    ("quote ends inside a word", "Items may be returned within 30 days of delivery for a full refund",
     ["Unused items may be returned within 30 days of delivery for a full refun"]),
]


@pytest.mark.parametrize("label,sentence,quotes", LAUNDER_BLOCK, ids=[c[0] for c in LAUNDER_BLOCK])
def test_quote_laundering_is_blocked(label, sentence, quotes):
    docs = [("RT-1" if q in DOC1 else "RT-2") for q in quotes]
    answer = f"{sentence} {' '.join(cite(q, d) for q, d in zip(quotes, docs))}."
    assert run(answer).blocked, label


def test_quote_spanning_two_docs_is_blocked():
    spanning = DOC1[-30:] + " " + DOC2[:30]
    assert run(f"Refunds are instant everywhere {cite(spanning, 'RT-1')}.").blocked
    assert run(f"Refunds are instant everywhere {cite(spanning, 'RT-2')}.").blocked


def test_cross_doc_label_swap_is_blocked():
    assert run(f"Items may be returned within 30 days of delivery {cite(RET, 'RT-2')}.").blocked


NORMALIZATION_PASS = [
    ("zero-width space in quote", RET.replace("within ", "within\u200b ")),
    ("zero-width joiner and BOM", RET.replace("returned", "ret\u200durned").replace("Unused", "\ufeffUnused")),
    ("soft hyphen in quote", RET.replace("delivery", "deliv\u00adery")),
    ("nbsp in quote", RET.replace(" days", "\u00a0days")),
    ("newline and tabs in quote", RET.replace(" may be ", "\n may\tbe ").replace("may\tbe", "may be")),
]


@pytest.mark.parametrize("label,quote", NORMALIZATION_PASS, ids=[c[0] for c in NORMALIZATION_PASS])
def test_invisible_characters_in_a_real_quote_do_not_block(label, quote):
    v = run(f'Items may be returned within 30 days of delivery for a full refund [doc:RT-1 "{quote}"].')
    assert not v.blocked, label


def test_smart_apostrophe_in_quote_matches_straight_apostrophe_in_doc():
    q = APOS.replace("'", "\u2019")
    assert not run(f'The store doesn\u2019t accept returns of opened software [doc:RT-2 "{q}"].').blocked


def test_smart_quote_citation_delimiters_are_accepted():
    answer = f"Items may be returned within 30 days of delivery for a full refund [doc:RT-1 \u201c{RET}\u201d]."
    assert not run(answer).blocked


def test_homoglyph_in_quote_is_blocked():
    q = RET.replace("returned", "r\u0435turned")  # Cyrillic e
    assert run(f'Items may be returned within 30 days of delivery for a full refund [doc:RT-1 "{q}"].').blocked


HOMOGLYPH_BLOCK = [
    ("cyrillic i in number word", "Warranty claims close within th\u0456rty days of purchase for all items"),
    ("zero-width inside number word", "Warranty claims close within thi\u200brty days of purchase for all items"),
    ("soft hyphen inside number word", "Warranty claims close within thi\u00adrty days of purchase for all items"),
    ("greek omicron in number word", "Warranty claims close within f\u03bfrty days of purchase for all items"),
]


@pytest.mark.parametrize("label,sentence", HOMOGLYPH_BLOCK, ids=[c[0] for c in HOMOGLYPH_BLOCK])
def test_hidden_number_words_are_blocked(label, sentence):
    assert run(claim(sentence, WARR)).blocked, label


def test_zero_width_only_quote_is_blocked():
    assert run('Refunds are instant everywhere [doc:RT-1 "\u200b\u200b"].').blocked


# ---------------------------------------------------------------------------
# 3. Negation flip, role swap, unit swap: documented limits (PIN)
# ---------------------------------------------------------------------------

def test_pin_negation_flip_is_not_detectable_and_is_documented():
    v = run(claim("Refunds are available after 30 days from delivery", NOREFUND))
    assert not v.blocked, "if this starts blocking, update the README Limitations entry on negation"
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    limits = readme.split("## limitations", 1)[1].split("\n## ", 1)[0]
    assert "negat" in limits


def test_pin_number_role_swap_is_not_detectable_and_is_documented():
    v = run(claim("The window opens April 20, 2026 and closes March 15, 2026 for all members", WINDOW))
    assert not v.blocked
    limits = (ROOT / "README.md").read_text(encoding="utf-8").lower().split("## limitations", 1)[1]
    assert "swap" in limits or "role" in limits


def test_pin_unit_mismatch_is_not_detectable_and_is_documented():
    v = run(claim("Support answers messages within 30 hours of the original request", HOURS))
    assert not v.blocked
    limits = (ROOT / "README.md").read_text(encoding="utf-8").lower().split("## limitations", 1)[1]
    assert "unit" in limits


# ---------------------------------------------------------------------------
# 4. Citation injection and sentence splitting
# ---------------------------------------------------------------------------

def test_nested_citation_is_blocked():
    inner = cite(RET)
    assert run(f'Items may be returned within 30 days [doc:RT-1 "Unused items {inner} more"].').blocked


def test_bracket_inside_quote_is_a_real_quote():
    assert not run(claim("Use code [A1] at checkout to save on the order", CODE)).blocked


def test_citation_attached_to_unrelated_claim_is_blocked():
    assert run(claim("Refunds are instant for gold members", FREIGHT)).blocked
    assert run(claim("The weather is lovely today in the city", FREIGHT)).blocked


HIDDEN = "Refunds are instant for gold members"
GOOD = f"{RET} {cite(RET)}."
HIDDEN_BLOCK = [
    ("bullet", f"- {HIDDEN}\n- {GOOD}"),
    ("star bullet", f"* {HIDDEN}\n* {GOOD}"),
    ("newline only", f"{HIDDEN}\n{GOOD}"),
    ("semicolon", f"{HIDDEN}; {GOOD}"),
    ("period without space", f"{HIDDEN}.{GOOD}"),
    ("fullwidth full stop", f"{HIDDEN}\u3002{GOOD}"),
    ("ellipsis", f"{HIDDEN}\u2026 {GOOD}"),
    ("etc then capital", f"{HIDDEN} etc. {GOOD}"),
    ("question mark", f"{HIDDEN}? {GOOD}"),
    ("windows newline", f"{HIDDEN}\r\n{GOOD}"),
    ("line separator", f"{HIDDEN}\u2028{GOOD}"),
    ("numbered", f"1. {HIDDEN}\n2. {GOOD}"),
]


@pytest.mark.parametrize("label,answer", HIDDEN_BLOCK, ids=[c[0] for c in HIDDEN_BLOCK])
def test_uncited_claim_hidden_by_splitting_is_blocked(label, answer):
    assert run(answer).blocked, label


def test_abbreviation_inside_a_sentence_does_not_split_it():
    assert not run(claim("Items (e.g. shoes) may be returned within 14 days of delivery for store credit", EG)).blocked


def test_numbered_and_bulleted_grounded_lists_pass():
    a = f"1. {RET} {cite(RET)}\n2. {FEE} {cite(FEE)}"
    b = f"- {RET} {cite(RET)}\n- {FEE} {cite(FEE)}"
    assert not run(a).blocked
    assert not run(b).blocked


def test_decimal_does_not_split_and_following_uncited_sentence_is_blocked():
    assert not run(claim("Shipping takes 2.5 days across the region", DEC)).blocked
    assert run(f"{claim('Shipping takes 2.5 days across the region', DEC)} Refunds are instant.").blocked


def test_citation_on_its_own_following_line_attaches_to_previous_sentence():
    assert not run(f"{RET}.\n{cite(RET)}").blocked


def test_citation_only_answer_is_blocked():
    assert run(cite(RET)).blocked
    assert run(f"{cite(RET)}.").blocked
    assert run(f"{cite(RET)} {cite(FEE)}").blocked


# ---------------------------------------------------------------------------
# 5. Retrieval scope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc_id", ["rt-1", "Rt-1", "RT-1.txt", "../RT-1", "/etc/passwd", "RT-1/../RT-1",
                                    "RT-01", "RT 1", "RT\u20101"])
def test_doc_id_variants_do_not_resolve(doc_id):
    answer = f'Items may be returned within 30 days of delivery for a full refund [doc:{doc_id} "{RET}"].'
    assert run(answer).blocked, doc_id


def test_invisible_character_in_doc_id_resolves_to_the_same_real_doc_not_a_new_one():
    answer = f'Items may be returned within 30 days of delivery for a full refund [doc:RT-1{chr(0x200b)} "{RET}"].'
    assert not run(answer).blocked
    assert run(answer, retrieved=["RT-2"]).blocked  # still subject to the retrieved set


def test_retrieved_id_variants_do_not_widen_the_set():
    answer = claim(RET + "", RET)
    for variant in (["rt-1"], [" RT-1"], ["RT-1 "], ["RT-1\n"], ["RT-2"], []):
        assert run(answer, retrieved=variant).blocked, variant


def test_retrieved_given_as_a_bare_string_is_rejected_not_split_into_characters():
    with pytest.raises(TypeError):
        gate.check_answer(claim(RET, RET), CORPUS, "RT-1")


# ---------------------------------------------------------------------------
# 6. Malformed input and CLI contract
# ---------------------------------------------------------------------------

def cli(*args: str, stdin: bytes | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "gate", *args], cwd=ROOT, input=stdin,
                          capture_output=True, env=env, timeout=timeout)


@pytest.fixture()
def corpus_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for k, v in CORPUS.items():
        (d / f"{k}.txt").write_text(v, encoding="utf-8")
    return d


def no_traceback(p: subprocess.CompletedProcess) -> None:
    assert b"Traceback" not in p.stderr, p.stderr.decode(errors="replace")


def test_cli_empty_answer_exit_1(corpus_dir):
    p = cli("-", "--corpus", str(corpus_dir), "--retrieved", "RT-1", stdin=b"")
    assert p.returncode == 1
    no_traceback(p)


def test_cli_citation_only_answer_exit_1_never_0(corpus_dir):
    p = cli("-", "--corpus", str(corpus_dir), "--retrieved", "RT-1", stdin=cite(RET).encode())
    assert p.returncode == 1


def test_cli_invalid_utf8_answer_file_exit_2(corpus_dir, tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"Refunds \xff\xfe are fine")
    p = cli(str(f), "--corpus", str(corpus_dir), "--retrieved", "RT-1")
    assert p.returncode == 2
    no_traceback(p)


def test_cli_invalid_utf8_stdin_exit_2(corpus_dir):
    p = cli("-", "--corpus", str(corpus_dir), "--retrieved", "RT-1", stdin=b"abc \xc3\x28 def")
    assert p.returncode == 2
    no_traceback(p)


def test_cli_invalid_utf8_corpus_doc_exit_2(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "BAD-1.txt").write_bytes(b"Fees are \xff\xfe $5.")
    p = cli("-", "--corpus", str(d), "--retrieved", "BAD-1", stdin=b"Fees are $5.")
    assert p.returncode == 2
    no_traceback(p)


def test_cli_corpus_with_a_directory_named_like_a_doc_exit_2(tmp_path):
    d = tmp_path / "docs"
    (d / "X-1.txt").mkdir(parents=True)
    p = cli("-", "--corpus", str(d), "--retrieved", "X-1", stdin=b"Fees are $5.")
    assert p.returncode == 2
    no_traceback(p)


def test_cli_no_arguments_exit_2():
    p = cli()
    assert p.returncode == 2
    no_traceback(p)


def test_cli_huge_answer_is_blocked_fast_with_a_bounded_verdict(corpus_dir):
    big = ("Refunds are instant. " * 250_000).encode()  # about 5 MB
    start = time.perf_counter()
    p = cli("-", "--corpus", str(corpus_dir), "--retrieved", "RT-1", stdin=big)
    assert time.perf_counter() - start < 20
    assert p.returncode == 1
    assert len(p.stdout) < 100_000, len(p.stdout)
    no_traceback(p)


def test_cli_large_number_heavy_answer_finishes_quickly(corpus_dir):
    answer = ("1," * 90_000 + f" {cite(RET)}").encode()
    start = time.perf_counter()
    p = cli("-", "--corpus", str(corpus_dir), "--retrieved", "RT-1", stdin=answer)
    assert time.perf_counter() - start < 10
    assert p.returncode == 1
    no_traceback(p)


def test_cli_binary_garbage_that_decodes_is_not_a_pass(corpus_dir):
    p = cli("-", "--corpus", str(corpus_dir), "--retrieved", "RT-1", stdin="\x00\x01\x02\ufeff\u200b".encode())
    assert p.returncode == 1


EVAL_BAD = [
    ("missing answer", '{"id": "x", "split": "grounded", "retrieved": ["POL-01"]}'),
    ("missing split", '{"id": "x", "answer": "a", "retrieved": []}'),
    ("not an object", "[1, 2]"),
    ("bare number", "7"),
    ("unknown class", '{"id": "x", "split": "ungrounded", "class": "nope", "answer": "a", "retrieved": [], '
                      '"expect_reason": "x", "bad_sentence": 0}'),
    ("unknown split", '{"id": "x", "split": "weird", "answer": "a", "retrieved": []}'),
    ("retrieved is a string", '{"id": "x", "split": "grounded", "answer": "a", "retrieved": "POL-01"}'),
    ("answer is a number", '{"id": "x", "split": "grounded", "answer": 5, "retrieved": []}'),
    ("negative bad_sentence", '{"id": "x", "split": "ungrounded", "class": "missing_citation", "answer": "a", '
                              '"retrieved": [], "expect_reason": "missing_citation", "bad_sentence": -1}'),
    ("invalid json", "{not json"),
]


@pytest.mark.parametrize("label,line", EVAL_BAD, ids=[c[0] for c in EVAL_BAD])
def test_evaluate_rejects_malformed_records_with_exit_2(label, line, tmp_path, capsys):
    f = tmp_path / "e.jsonl"
    f.write_text(line + "\n", encoding="utf-8")
    assert evaluate.main(["--eval", str(f)]) == 2, label
    assert "Traceback" not in capsys.readouterr().err


def test_evaluate_non_utf8_eval_file_exit_2(tmp_path):
    f = tmp_path / "e.jsonl"
    f.write_bytes(b'{"id": "\xff"}\n')
    assert evaluate.main(["--eval", str(f)]) == 2


# ---------------------------------------------------------------------------
# 7. Quantity words that are numbers in disguise
# ---------------------------------------------------------------------------

QUANTITY_WORDS = [
    "a dozen", "two dozen", "a fortnight", "half", "twice", "thrice", "double",
    "a quarter", "a couple of", "thirtieth", "fifteenth",
]


@pytest.mark.parametrize("word", QUANTITY_WORDS)
def test_quantity_word_not_in_quote_is_blocked(word):
    v = run(claim(f"Unused items may be returned within {word} days of delivery for a full refund", RET))
    assert v.blocked, word
    assert any(r.code == gate.NUMBER_NOT_IN_QUOTE for s in v.sentences for r in s.reasons)


def test_quantity_word_in_quote_is_not_blocked():
    quote = "Unused items may be returned within a dozen days of delivery for a full refund"
    v = gate.check_answer(f'{quote} [doc:D-1 "{quote}"].', {"D-1": quote}, ["D-1"])
    assert not v.blocked
