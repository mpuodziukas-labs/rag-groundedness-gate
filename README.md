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

Measured on the bundled synthetic eval set (n=209: 104 grounded answers, 105
ungrounded answers). A detection counts only when the answer is blocked and the
expected reason code is reported on the defective sentence. All figures are
counts out of totals.

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
| Ungrounded total | 105/105 |
| False positives (grounded split) | 0/104 |
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

## Honesty Statement

> **These numbers are measured on the bundled synthetic corpus (n=209,
> composition above). They are NOT from a production deployment, NOT from an
> external benchmark, and NOT from client data.**
>
> The corpus is a synthetic corpus authored for this repo: a fictional retailer
> with 30 invented policy docs. It is not production data and not client data.
> The failure cases were written by the same author as the rules, so full
> detection here shows that the rules cover the failure shapes they were built
> for, not that they generalize. See [Limitations](#limitations).

Mutation proof: the tests monkeypatch the verbatim-quote check and the
numbers-inside-quote check to always pass, and assert that the evaluation then
reports missed detections and exits nonzero. The unmutated control stays fully
green, so a passing eval cannot be vacuous.

## Limitations

1. **Synthetic and co-developed.** Rules and eval cases came from one author.
   Real traffic has phrasing, formats and failure shapes not covered here.
2. **No semantic check.** A sentence that cites a real quote but claims
   something unrelated, with no numbers in it, passes. The gate proves the
   citation is real and the numbers match, not that the sentence means what the
   quote means.
3. **Textual number matching.** `$50` does not match `$50.00`, and number words
   ("thirty") are not read as numbers. Dates are recognized as `2026-03-15`,
   `March 15, 2026` and `15 March 2026`. A sentence that gives only `March 15`
   passes against a quote with `March 15, 2026`.
4. **Short generic quotes pass check 3.** A quote only has to be non-empty and
   appear in the doc. Pair the gate with a minimum quote length if you need one.
5. **Every sentence is a claim.** Greetings and other uncited sentences are
   blocked. That is deliberate (fail closed) and may need an allowlist in a
   chat product.
6. **Citations must use the exact format** `[doc:ID "quote"]`. Quotes cannot
   contain a double quote character.
7. **Median timing is hardware dependent** and measured on small docs. Longer
   documents cost more, linearly in document length.

## How it works

`gate.py` runs five checks per claim sentence and returns every reason it finds.

| # | Check | Reason code |
|---|---|---|
| 1 | the sentence carries a citation | `missing_citation`, `malformed_citation` |
| 2 | the cited doc ID exists in the corpus | `unknown_doc` |
| 3 | the quote appears verbatim in that doc, whitespace-normalized, case-sensitive | `quote_not_verbatim`, `empty_quote` |
| 4 | every number, money amount, percent and date in the sentence appears inside the cited quote(s) | `number_not_in_quote` |
| 5 | the cited doc is in the retrieved set passed with the answer | `out_of_retrieval` |

An answer with no sentences is blocked (`empty_answer`). The answer passes only
if every sentence passes. Citations are masked before sentence splitting, so a
period inside a quote never splits a sentence, and a citation placed after the
final period attaches to the sentence before it.

Verdict shape:

```json
{
  "verdict": "blocked",
  "sentence_count": 1,
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
  corpus/            30 synthetic policy docs + eval.jsonl (209 labeled answers)
  tests/             unit, corpus, evaluate, mutation, and README-sync tests
  .github/workflows/ci.yml
```

## License

MIT. See [LICENSE](LICENSE).
