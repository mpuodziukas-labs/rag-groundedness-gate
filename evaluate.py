"""Run the eval set through the gate and print the results table.

    python evaluate.py [--corpus DIR] [--eval FILE] [--write-readme]

Prints per-class detection counts (N/N), false positives on the grounded
split (N/N), and the median per-answer check time in ms, measured on this run.
--write-readme rewrites the generated blocks of README.md (results table and
eval-set composition) from this run, so no figure there is typed by hand.
A detection means the answer was blocked AND the expected reason code was
reported on the defective sentence.
Exit codes: 0 all detected and no false positives, 1 any miss or false
positive, 2 usage error (missing or empty input).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import gate

ROOT = Path(__file__).resolve().parent


@dataclass
class Results:
    totals: dict[str, int]
    detected: dict[str, int]
    grounded_total: int
    false_positives: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def clean(self) -> bool:
        return self.false_positives == 0 and all(self.detected[c] == self.totals[c] for c in self.totals)


def validate_record(rec: object, line_no: int) -> dict:
    """Reject a malformed eval record with a ValueError that names the line."""
    def bad(msg: str) -> ValueError:
        return ValueError(f"eval line {line_no}: {msg}")

    if not isinstance(rec, dict):
        raise bad("record must be a JSON object")
    if not isinstance(rec.get("answer"), str):
        raise bad("'answer' must be a string")
    retrieved = rec.get("retrieved")
    if not isinstance(retrieved, list) or not all(isinstance(r, str) for r in retrieved):
        raise bad("'retrieved' must be a list of strings")
    split = rec.get("split")
    if split not in ("grounded", "ungrounded"):
        raise bad("'split' must be 'grounded' or 'ungrounded'")
    if split == "ungrounded":
        if rec.get("class") not in gate.CLASS_REASON:
            raise bad(f"'class' must be one of {sorted(gate.CLASS_REASON)}")
        if not isinstance(rec.get("expect_reason"), str):
            raise bad("'expect_reason' must be a string")
        idx = rec.get("bad_sentence")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise bad("'bad_sentence' must be an integer >= 0")
    return rec


def load_eval(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [validate_record(json.loads(line), n) for n, line in enumerate(lines, 1) if line.strip()]


def run_eval(records: Sequence[dict], corpus: Mapping[str, str]) -> Results:
    totals = {c: 0 for c in gate.CLASS_REASON}
    detected = {c: 0 for c in gate.CLASS_REASON}
    grounded_total = false_positives = 0
    times: list[float] = []
    for rec in records:
        start = time.perf_counter()
        verdict = gate.check_answer(rec["answer"], corpus, rec["retrieved"])
        times.append((time.perf_counter() - start) * 1000.0)
        if rec["split"] == "grounded":
            grounded_total += 1
            false_positives += verdict.blocked
            continue
        cls = rec["class"]
        totals[cls] += 1
        bad = verdict.sentences[rec["bad_sentence"]] if rec["bad_sentence"] < len(verdict.sentences) else None
        if verdict.blocked and bad and rec["expect_reason"] in [r.code for r in bad.reasons]:
            detected[cls] += 1
    return Results(totals, detected, grounded_total, false_positives, times)


def render_table(res: Results) -> str:
    rows = ["| Failure class | Detected |", "|---|---|"]
    rows += [f"| {c} | {res.detected[c]}/{res.totals[c]} |" for c in res.totals]
    rows.append(f"| Ungrounded total | {sum(res.detected.values())}/{sum(res.totals.values())} |")
    rows.append(f"| False positives (grounded split) | {res.false_positives}/{res.grounded_total} |")
    return "\n".join(rows)


def composition(records: Sequence[dict]) -> str:
    grounded = sum(r["split"] == "grounded" for r in records)
    classes = len({r["class"] for r in records if r["split"] == "ungrounded"})
    return (f"n={len(records)}: {grounded} grounded answers, {len(records) - grounded} ungrounded "
            f"answers across {classes} failure classes")


def _swap(text: str, name: str, body: str) -> str:
    begin, end = f"<!-- {name}:BEGIN -->", f"<!-- {name}:END -->"
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def write_readme(path: Path, records: Sequence[dict], res: Results) -> None:
    text = path.read_text(encoding="utf-8")
    text = _swap(text, "RESULTS", render_table(res))
    text = _swap(text, "COMPOSITION", composition(records))
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the groundedness gate on the synthetic eval set.")
    p.add_argument("--corpus", default=str(ROOT / "corpus" / "docs"), help="directory of *.txt docs")
    p.add_argument("--eval", default=str(ROOT / "corpus" / "eval.jsonl"), help="eval set (jsonl)")
    p.add_argument("--write-readme", action="store_true", help="regenerate the generated blocks of README.md")
    args = p.parse_args(argv)
    try:
        corpus = gate.load_corpus(args.corpus)
        records = load_eval(args.eval)
    except (gate.CorpusError, OSError, ValueError) as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return gate.EXIT_USAGE
    if not records:
        print("evaluate: eval set is empty; refusing to report a vacuous pass", file=sys.stderr)
        return gate.EXIT_USAGE
    res = run_eval(records, corpus)
    print(render_table(res))
    print()
    if args.write_readme:
        write_readme(ROOT / "README.md", records, res)
    print(f"Median per-answer check time: {res.median_ms:.3f} ms over {len(res.times_ms)} answers (measured on this run)")
    return gate.EXIT_PASS if res.clean else gate.EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
