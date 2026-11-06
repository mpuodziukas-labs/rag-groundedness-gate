#!/usr/bin/env bash
# Run the eval and the tests; assert the headline. Exit 0 all green, 1 otherwise.
#   scripts/verify.sh              verify this checkout
#   scripts/verify.sh --selftest   prove verify fails on planted defects and passes clean
set -euo pipefail

# check_tree DIR: print FAIL reasons to stderr, return 0 only if everything holds.
check_tree() {
  local dir=$1 out rc=0 ok=0
  out=$(cd "$dir" && python3 evaluate.py 2>&1) || rc=$?
  printf '%s\n' "$out"
  if [ "$rc" -ne 0 ]; then echo "FAIL: evaluate.py exited $rc" >&2; ok=1; fi
  # Headline: every ungrounded answer detected (N/N) and zero false positives on the grounded split.
  if ! printf '%s\n' "$out" | awk -F'|' '
      /Ungrounded total/ { split($3, a, "/"); gsub(/ /, "", a[1]); gsub(/ /, "", a[2]); u = (a[2] > 0 && a[1] == a[2]) }
      /False positives \(grounded split\)/ { split($3, b, "/"); gsub(/ /, "", b[1]); gsub(/ /, "", b[2]); f = (b[2] > 0 && b[1] == 0) }
      END { exit !(u && f) }'; then
    echo "FAIL: headline line not satisfied (all ungrounded detected, 0 grounded false positives)" >&2; ok=1
  fi
  rc=0
  (cd "$dir" && python3 -m pytest -q) || rc=$?
  if [ "$rc" -ne 0 ]; then echo "FAIL: tests exited $rc" >&2; ok=1; fi
  return "$ok"
}

selftest() {
  local src pass=0 total=3
  src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  work=$(mktemp -d)  # global on purpose: the EXIT trap runs after this function returns
  trap 'rm -rf "$work"' EXIT

  expect() { # expect LABEL WANT_RC DIR
    local rc=0
    check_tree "$3" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -eq "$2" ]; then pass=$((pass + 1)); echo "ok: $1"; else echo "NOT ok: $1 (exit $rc, wanted $2)"; fi
  }
  plant() { # plant DIR FROM TO: edit gate.py, fail loudly if the edit did not change it
    local before after
    before=$(cksum <"$1/gate.py")
    sed "s/$2/$3/" "$1/gate.py" >"$1/gate.py.new" && mv "$1/gate.py.new" "$1/gate.py"
    after=$(cksum <"$1/gate.py")
    [ "$before" != "$after" ] || { echo "selftest: planted edit did not apply: $2" >&2; exit 1; }
  }

  mkdir "$work/clean" "$work/pass_all" "$work/no_refuse"
  for d in clean pass_all no_refuse; do
    (cd "$src" && tar --exclude=.git --exclude=__pycache__ -cf - .) | (cd "$work/$d" && tar -xf -)
  done
  # Defect 1: the gate lets every answer through, including ungrounded ones.
  plant "$work/pass_all" 'return bool(self.answer_reasons) or any(not s.ok for s in self.sentences)' 'return False'
  # Defect 2: blocked answers are no longer refused.
  plant "$work/no_refuse" 'return REFUSE if self.blocked else PASS' 'return PASS'

  expect "clean copy exits 0" 0 "$work/clean"
  expect "gate passes ungrounded answers: exits 1" 1 "$work/pass_all"
  expect "blocked answer not refused: exits 1" 1 "$work/no_refuse"
  echo "selftest $pass/$total"
  [ "$pass" -eq "$total" ]
}

if [ "${1:-}" = "--selftest" ]; then
  selftest
elif [ $# -eq 0 ]; then
  root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  if check_tree "$root"; then echo "verify: PASS"; else echo "verify: FAIL" >&2; exit 1; fi
else
  echo "usage: scripts/verify.sh [--selftest]" >&2
  exit 2
fi
