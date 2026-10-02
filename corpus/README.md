# Corpus (synthetic)

Everything in this directory is synthetic. The company, its 30 policy docs and
every eval record were authored for this repository. Nothing here comes from a
real business, a production system, or client data.

- `docs/POL-01.txt` to `docs/POL-30.txt`: short policy docs for a fictional retailer.
- `eval.jsonl`: 209 labeled answers (104 grounded, 105 ungrounded across 7 failure classes).

Do not edit these files by hand. Regenerate them with `python build_corpus.py`;
the output is byte-identical on every run and a test enforces it.
