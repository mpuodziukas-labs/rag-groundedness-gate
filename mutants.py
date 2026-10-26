"""Mutation check: remove each defense in a scratch copy, the matching tests must go RED.
A control copy must pass first. A mutant that hangs past TIMEOUT counts as killed.
Usage: python3 mutants.py [repo_dir]   (default: the directory holding this file)"""
import pathlib, shutil, subprocess, sys, tempfile
ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent
TIMEOUT = 30
T = "tests/test_gate.py"
H = "tests/test_hostile.py"
R = "tests/test_redteam.py"
E = "tests/test_evaluate.py"
G = "gate.py"
V = "evaluate.py"
# (name, file, text to remove, text to put in, test file that must fail)
MUTANTS = [
    ("quote: one word quotes are long enough", G, "MIN_QUOTE_WORDS = 3", "MIN_QUOTE_WORDS = 1", H),
    ("claim: any number of novel words is allowed", G, "MAX_NOVEL = 1 ", "MAX_NOVEL = 99 ", H),
    ("citation: a doc outside the retrieved set is accepted", G, "elif doc_id not in retrieved_set:", "elif False:", T),
    ("citation: an uncited sentence is accepted", G, "        if not cites:\n", "        if False:\n", T),
    ("citation: mixed-script words are accepted", G, "        if has_mixed_script_word(text):", "        if False:", H),
    ("citation: a quote that is not in the doc is accepted", G, "elif not quote_in_doc(quote, corpus[doc_id]):", "elif False:", T),
    ("quote: a quote may run across two sentences", G, "crosses = i < len(breaks) and breaks[i] < end", "crosses = False", H),
    ("quote: a quote may start inside a word", G, "left_ok = not (q[0].isalnum() and start > 0", "left_ok = not (False and start > 0", H),
    ("quote: a quote may end inside a word", G, "right_ok = not (q[-1].isalnum() and end < len(d)", "right_ok = not (False and end < len(d)", R),
    ("normalize: format characters are kept", G, '        if cat == "Cf":\n            continue', "        if False:\n            continue", H),
    ("claim: novel words are never counted", G, "                if len(novel) > MAX_NOVEL:", "                if False:", H),
    ("numbers: a number missing from the quote is accepted", G, "for token in missing_numbers(text, quotes)[:MAX_REPORTED_REASONS]:", "for token in []:", T),
    ("answer: an empty answer passes", G, 'answer_reasons = [] if sentences else [Reason(EMPTY_ANSWER, "answer contains no sentences")]', "answer_reasons = []", T),
    ("input: a single string as retrieved set is taken", G, "    if isinstance(retrieved, (str, bytes)):", "    if False:", R),
    ("cli: a blocked answer exits 0", G, "return EXIT_BLOCKED if verdict.blocked else EXIT_PASS", "return EXIT_PASS", T),
    ("eval: an empty eval set is a pass", V, "    if not records:", "    if False:", E),
]


def pytest_rc(tree, test):
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test],
                           cwd=tree, capture_output=True, timeout=TIMEOUT,
                           env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        return 124                                      # a mutant that hangs the tests is caught
    return r.returncode


def fresh(tmp, name):
    dst = pathlib.Path(tmp) / name
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    return dst


def main():
    killed = 0
    tests = sorted({m[4] for m in MUTANTS})
    with tempfile.TemporaryDirectory() as tmp:
        for t in tests:
            if pytest_rc(fresh(tmp, "control-" + pathlib.Path(t).stem), t) != 0:
                print(f"control FAIL {t}"); return 1
        print(f"control {len(tests)}/{len(tests)} test files pass unmodified")
        for i, (name, f, old, new, test) in enumerate(MUTANTS):
            tree = fresh(tmp, f"m{i}")
            src = (tree / f).read_text()
            if src.count(old) != 1:
                print(f"mutant {name}: pattern matched {src.count(old)} times, want 1"); return 1
            (tree / f).write_text(src.replace(old, new))
            red = pytest_rc(tree, test) != 0
            killed += red
            print(f"{'RED ' if red else 'LIVE'} {name}")
    print(f"mutants killed {killed}/{len(MUTANTS)}")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
