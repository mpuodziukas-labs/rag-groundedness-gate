# Corpus (synthetic)

Everything in this directory is synthetic. The company, its 30 policy docs and
every eval record were authored for this repository. Nothing here comes from a
real business, a production system, or client data.

- `docs/POL-01.txt` to `docs/POL-30.txt`: short policy docs for a fictional retailer.
- `eval.jsonl`: labeled answers, grounded and ungrounded across the failure classes listed in the top-level README; counts are printed by `python evaluate.py --write-readme` into that README.
- `paraphrase.jsonl`: accurate rewordings of real doc sentences, reported as their own split (false positives on paraphrase).

Do not edit these files by hand. Regenerate them with `python build_corpus.py`;
the output is byte-identical on every run and a test enforces it.
