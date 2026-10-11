# Verifying this repository

Purpose: demonstrates a groundedness check that rejects RAG answers whose claims are not supported by the quoted source for teams subject to model-risk, audit or change-control review.

Verify: `python3 evaluate.py`

Pass condition: exit code 0 and this exact output line (leading spaces ignored): `| Ungrounded total | 240/240 |`

Scope: synthetic data, fixed seed, offline, under 60 seconds.

Limitations: Synthetic and co-developed with its rules, so results measure coverage of known failure shapes, not real-world recall. The support check is a lexical heuristic, not semantic entailment. Negation is not detected. Number roles and units are not checked.

Anything else = not verified.
