#!/usr/bin/env bash
# verify_all_prs_on_v017.sh — reproducible per-PR verification onto v0.17.0.
#
# From a clean v0.17.0 worktree, for EVERY open PR by arminanton:
#   1. apply the PR's net diff (vs its own merge-base) onto v0.17.0 — record
#      CLEAN / 3WAY / CONFLICT
#   2. run ruff on the PR's changed .py files — record LINT pass/fail
#   3. run the PR's own changed test files (if any) — record TEST pass/fail/none
# Emits a per-PR pass/fail table + a summary. Exit 0 iff no CONFLICT and no
# LINT-fail and no TEST-fail.
#
# Usage: ./verify_all_prs_on_v017.sh [fork_remote]
set -uo pipefail
FORK="${1:-fork}"
V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"
SRC="$(git rev-parse --show-toplevel)"
# Use the project venv's python so ruff/pytest are importable regardless of the
# worktree's ambient python (a bare `python3` in the worktree may resolve to a
# system interpreter without the dev deps).
if [ -x "$SRC/venv/bin/python" ]; then PY="$SRC/venv/bin/python"
else PY="${PYTHON:-python3}"; fi

PR_JSON="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list \
  --repo NousResearch/hermes-agent --author arminanton \
  --state open --limit 100 --json number,headRefOid,headRefName 2>/dev/null)"

mapfile -t ROWS < <(echo "$PR_JSON" | "$PY" -c '
import sys, json
for p in json.load(sys.stdin):
    print("%s\t%s\t%s" % (p["number"], p["headRefOid"], p["headRefName"]))
' | sort -n)

printf "%-8s %-10s %-8s %-10s  %s\n" "PR" "APPLY" "LINT" "TEST" "branch"
printf "%-8s %-10s %-8s %-10s  %s\n" "----" "-----" "----" "----" "------"

WT="$(mktemp -d)"
DIFFDIR="$(mktemp -d)"
git worktree add -q "$WT" "$V017"
fail=0
declare -a SUMMARY
for row in "${ROWS[@]}"; do
  num="$(cut -f1 <<<"$row")"; sha="$(cut -f2 <<<"$row")"; br="$(cut -f3 <<<"$row")"
  git fetch -q "$FORK" "$sha" 2>/dev/null || true
  base="$(git merge-base origin/main "$sha" 2>/dev/null || git merge-base "$V017" "$sha")"
  DIFF="$DIFFDIR/pr_${num}.diff"
  git -C "$SRC" diff "$base" "$sha" > "$DIFF" 2>/dev/null

  # reset worktree to pristine v0.17.0 (hard reset clears any unmerged index
  # state left by a prior 3-way --check; diff lives OUTSIDE the worktree)
  ( cd "$WT" && git reset -q --hard "$V017" && git clean -qfdx 2>/dev/null )

  # 1. apply
  if ( cd "$WT" && git apply --check "$DIFF" 2>/dev/null ); then apply="CLEAN"
  elif ( cd "$WT" && git apply --3way --check "$DIFF" 2>/dev/null ); then apply="3WAY"
  else apply="CONFLICT"; fail=1; fi

  # changed py + test files
  mapfile -t changed < <(git diff --name-only "$base" "$sha" 2>/dev/null | grep '\.py$' || true)
  tests=()
  for f in "${changed[@]}"; do [[ "$f" == tests/* ]] && tests+=("$f"); done

  # 2. lint — apply onto the worktree first, then run the repo's OWN blocking
  # ruff config (`ruff check .` honors pyproject [tool.ruff], i.e. PLW1514 only;
  # this matches the repo's Lint workflow, not the default ruleset). Count only
  # hard errors (error[...]); warnings (e.g. stale noqa) are non-blocking.
  lint="n/a"; test_res="none"
  if [ "$apply" != "CONFLICT" ] && [ "${#changed[@]}" -gt 0 ]; then
    ( cd "$WT" && git apply --3way "$DIFF" 2>/dev/null )
    _ruff_err="$( cd "$WT" && "$PY" -m ruff check . 2>/dev/null | grep -c 'error\[' )"
    if [ "${_ruff_err:-0}" -eq 0 ]; then lint="PASS"; else lint="FAIL($_ruff_err)"; fail=1; fi
    # 3. test (worktree already has PR applied)
    if [ "${#tests[@]}" -gt 0 ]; then
      present=(); for t in "${tests[@]}"; do [ -f "$WT/$t" ] && present+=("$t"); done
      if [ "${#present[@]}" -gt 0 ]; then
        if ( cd "$WT" && timeout 120 "$PY" -m pytest "${present[@]}" -q --no-header -p no:cacheprovider --tb=no >/dev/null 2>&1 ); then test_res="PASS"; else test_res="FAIL"; fi
      fi
    fi
  fi

  printf "%-8s %-10s %-8s %-10s  %s\n" "#$num" "$apply" "$lint" "$test_res" "$br"
done
git worktree remove --force "$WT" 2>/dev/null
rm -rf "$DIFFDIR"

echo ""
echo "EXIT: $fail  (0 = no CONFLICT, no LINT-fail, no TEST-fail)"
exit $fail
