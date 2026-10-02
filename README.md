# RAG Groundedness Gate

A support or finance assistant that quotes policy and numbers is one invented
number away from real money and lost trust: a refund amount, a fee, a deadline
that is not in the source. This gate checks a RAG answer before it reaches a
customer and blocks it in milliseconds, on a CPU, at zero marginal cost. It
needs no API key, no network and no third-party package.

```
Answer + retrieved doc IDs  ->  gate  ->  pass (exit 0) | blocked (exit 1) + per-sentence reasons
```

## Business problem

A customer asks what the refund window is. The assistant answers "30 days" and
cites the policy. If the policy says 14, the company either honors a promise it
never made or breaks one it did. Reviewing every answer by hand does not scale,
and asking another system whether the first one is right adds cost, latency and
a second source of error.

The gate takes the cheap, checkable half of that problem and makes it
deterministic. Every claim sentence must cite a document by ID with an exact
quote, the quote must really be in that document, the document must be one the
retriever actually returned, and every number, amount, percent and date in the
sentence must appear inside the quote. Anything else is blocked, and the reason
is named per sentence so the failure can be logged and fixed.

## Quickstart

```bash
python build_corpus.py                       # 1. regenerate the synthetic corpus and eval set
python -m pytest tests/ -q                   # 2. run the test suite
python evaluate.py                           # 3. print the results table
```

Check one answer (JSON verdict on stdout; exit 0 pass, 1 blocked, 2 usage error):

```bash
echo 'Standard shipping costs $5.99 [doc:POL-03 "Standard shipping costs $5.99 and arrives in 3 to 5 business days"].' \
  | python -m gate - --corpus corpus/docs --retrieved POL-03,POL-06
```

Python 3.10 or newer. `pytest` is the only thing to install, and only for tests.

## Results

Eval set composition (generated):

<!-- COMPOSITION:BEGIN -->
n=389: 149 grounded answers, 240 ungrounded answers across 16 failure classes
<!-- COMPOSITION:END -->

A detection counts only when the answer is blocked and the expected reason code
is reported on the defective sentence. All figures are counts out of totals.

<!-- RESULTS:BEGIN -->
| Failure class | Detected |
|---|---|
| fabricated_quote | 15/15 |
| wrong_number | 15/15 |
| missing_citation | 15/15 |
| unknown_doc | 15/15 |
| out_of_retrieval | 15/15 |
| number_outside_quote | 15/15 |
| paraphrase_drift | 15/15 |
| number_word_evasion | 15/15 |
| number_kind_mismatch | 15/15 |
| short_quote_laundering | 15/15 |
| quote_spans_sentences | 15/15 |
| subword_quote | 15/15 |
| unrelated_citation | 15/15 |
| hidden_uncited_sentence | 15/15 |
| citation_only | 15/15 |
| mixed_script_number | 15/15 |
| Ungrounded total | 240/240 |
| False positives (grounded split) | 0/149 |
<!-- RESULTS:END -->

Median per-answer check time is measured by `evaluate.py` on every run and
printed on its own line, because it varies by machine. A test fails the build if
this table ever differs from what `evaluate.py` prints.

| Class | What it is |
|---|---|
| fabricated_quote | citation quote was invented; it is in no document |
| wrong_number | real quote, but the sentence states a number that appears nowhere in the doc |
| missing_citation | claim sentence with no citation |
| unknown_doc | cites a doc ID that is not in the corpus |
| out_of_retrieval | cites a real doc that was not in the retrieved set for this answer |
| number_outside_quote | the number is in the doc, but the quoted span stops before it |
| paraphrase_drift | near-copy of a real sentence with words swapped, presented as verbatim |
| number_word_evasion | the claim spells a different number as a word ("twenty-five" vs the quote's 20) |
| number_kind_mismatch | digits match but the unit or sign does not (`$` vs `€`, `%` vs `$`, `-` added) |
| short_quote_laundering | a two-word quote that is technically in the doc |
| quote_spans_sentences | one quote stitched from two neighboring sentences of a doc |
| subword_quote | quote starts in the middle of a word |
| unrelated_citation | a real quote attached to a claim about something else |
| hidden_uncited_sentence | uncited claim hidden behind a bullet, newline, semicolon or missing space |
| citation_only | a citation with no claim text |
| mixed_script_number | a number word with one Cyrillic letter, so it is not read as a number |

## Honesty Statement

> **These numbers are measured on the bundled synthetic corpus (composition
> above). They are NOT from a production deployment, NOT from an
> external benchmark, and NOT from client data.**
>
> The corpus is a synthetic corpus authored for this repo: a fictional retailer
> with 30 invented policy docs. It is not production data and not client data.
> The failure cases were written by the same author as the rules, so full
> detection here shows that the rules cover the failure shapes they were built
> for, not that they generalize. See [Limitations](#limitations).

Mutation proof: the tests monkeypatch the verbatim-quote check, the
numbers-inside-quote check, the minimum quote length, the support threshold and
the mixed-script check to always pass, and assert that the evaluation then
reports missed detections and exits nonzero. The unmutated control stays fully
green, so a passing eval cannot be vacuous.

## Limitations

1. **Synthetic and co-developed.** Rules and eval cases came from one author
   and grew together, so N/N here measures coverage of known failure shapes,
   not real-world recall. Real traffic has phrasing and formats not covered.
2. **No semantic entailment.** Check 6 is a lexical heuristic: at least half of
   the claim's content words must appear in the quote. A claim that reuses the
   quote's words but means something else passes.
3. **Negation is not detected.** "Refunds are not allowed" citing a quote that
   says refunds are allowed passes when the words and numbers overlap. A test
   pins this behavior.
4. **Number roles and units are not checked.** Swapping which date is the
   opening and which is the closing date, or "30 hours" for "30 days", passes
   when the values appear in the quote. A test pins both.
5. **Number reading rules.** Values are compared, not strings, so `$1,000`,
   `$1000` and "1000 dollars" agree, and `5%` equals "5 percent". A lone "one"
   is treated as a pronoun, not a number, and a bare "May" is not read as a
   month. `03/04/2026` is read as month/day. Dates in other forms may be
   missed. Relative spans with no number, such as "a month" for "30 days", are
   not checked.
6. **Sentence splitting is rule-based.** Lines, bullets, semicolons and
   terminators split sentences, with a guard for a short list of abbreviations.
   An abbreviation outside that list can split a sentence in two, which fails
   closed (the fragment is uncited).
7. **Quotes must sit inside one doc sentence** and be at least 3 words long. A
   claim that needs two sentences of a doc must cite each separately. Quotes
   cannot contain a double quote character.
8. **Every sentence is a claim.** Greetings and other uncited sentences are
   blocked. That is deliberate (fail closed) and may need an allowlist in a
   chat product.
9. **Citations must use the exact format** `[doc:ID "quote"]`. Doc IDs match
   exactly, including case.
10. **Median timing is hardware dependent** and measured on small docs. Longer
    documents cost more, linearly in document length.

## How it works

`gate.py` runs six checks per claim sentence and returns every reason it finds.

| # | Check | Reason code |
|---|---|---|
| 1 | the sentence carries a citation and claim text of its own | `missing_citation`, `malformed_citation`, `empty_claim` |
| 2 | the cited doc ID exists in the corpus | `unknown_doc` |
| 3 | the quote is verbatim in that doc (whitespace-normalized, case-sensitive, on word boundaries, inside one doc sentence, at least 3 words) | `quote_not_verbatim`, `empty_quote`, `quote_too_short` |
| 4 | every number, money amount, percent and date in the sentence, and every quantity word (dozen, half, twice, fortnight, "thirtieth"), appears inside the cited quote(s), compared by value | `number_not_in_quote` |
| 5 | the cited doc is in the retrieved set passed with the answer | `out_of_retrieval` |
| 6 | at least half of the claim's content words appear in the cited quote(s); no word mixes Latin with Cyrillic or Greek letters | `claim_not_in_quote`, `mixed_script_text` |

Before any comparison, answers, quotes and docs are normalized: Unicode NFKC,
zero-width and other format characters removed, non-ASCII digits folded to
0-9, smart quotes folded to straight quotes, number words turned into digits.

An answer with no sentences is blocked (`empty_answer`). The answer passes only
if every sentence passes. Citations are masked before sentence splitting, so a
period inside a quote never splits a sentence, and a citation placed after the
final period attaches to the sentence before it.

Verdict shape:

```json
{
  "verdict": "blocked",
  "sentence_count": 1,
  "sentences_listed": 1,
  "answer_reasons": [],
  "sentences": [
    {
      "index": 0,
      "text": "Standard shipping costs $6.99.",
      "citations": [{"doc": "POL-03", "quote": "Standard shipping costs $5.99 and arrives in 3 to 5 business days"}],
      "ok": false,
      "reasons": [{"code": "number_not_in_quote", "detail": "$6.99 is not inside the cited quote"}]
    }
  ]
}
```

Files:

```
rag-groundedness-gate/
  gate.py            the checks, verdict, and CLI (python -m gate)
  evaluate.py        results table, false positives, measured median check time
  build_corpus.py    regenerates corpus/ deterministically
  corpus/            30 synthetic policy docs + eval.jsonl (labeled answers)
  tests/             unit, corpus, evaluate, mutation, and README-sync tests
  .github/workflows/ci.yml
```

## License

MIT. See [LICENSE](LICENSE).
