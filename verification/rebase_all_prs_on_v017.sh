#!/usr/bin/env bash
# rebase_all_prs_on_v017.sh — machine-readable per-PR rebase + test report.
#
# For EVERY open PR by arminanton, in an isolated worktree off v0.17.0:
#   1. REBASE the PR's commits onto v0.17.0 (real `git rebase`, not text-apply).
#      Record REBASED_CLEAN / REBASED_CONFLICT(n files) / NO_COMMITS.
#   2. Run the PR's OWN changed test files on that rebased state.
#      Record TEST pass/fail counts + the exact reproducible pytest command.
#   3. Emit one JSON object per PR to stdout (one line each = JSONL), then a
#      summary object. Exit 0 iff no REBASED_CONFLICT and no TEST-fail that
#      isn't classified pre-existing-on-v0.17.0.
#
# Usage: ./rebase_all_prs_on_v017.sh [fork_remote]  > report.jsonl
set -uo pipefail
FORK="${1:-fork}"
V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"
SRC="$(git rev-parse --show-toplevel)"
PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"

env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" 2>/dev/null || true

PR_JSON="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list \
  --repo NousResearch/hermes-agent --author arminanton \
  --state open --limit 100 --json number,headRefOid,headRefName,baseRefName 2>/dev/null)"

mapfile -t ROWS < <(echo "$PR_JSON" | "$PY" -c '
import sys, json
for p in sorted(json.load(sys.stdin), key=lambda x: x["number"]):
    print("%s\t%s\t%s" % (p["number"], p["headRefOid"], p["headRefName"]))
')

WT="$(mktemp -d)"
git worktree add -q --detach "$WT" "$V017" 2>/dev/null
fail=0
for row in "${ROWS[@]}"; do
  num="$(cut -f1 <<<"$row")"; sha="$(cut -f2 <<<"$row")"; br="$(cut -f3 <<<"$row")"
  env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" "$sha" 2>/dev/null || true
  # PR branches are cut off origin/main (a descendant of v0.17.0), so the PR's
  # OWN commits = merge-base(origin/main, sha)..sha — NOT merge-base(v017, sha),
  # which would wrongly include the ~50 origin/main commits between v0.17.0 and
  # where the PR was branched. We then rebase ONLY those own commits onto v0.17.0.
  base="$(git merge-base origin/main "$sha" 2>/dev/null || git merge-base "$V017" "$sha")"

  # PR's own commits = base..sha ; its changed files (for test discovery)
  ncommits="$(git rev-list --count "$base..$sha" 2>/dev/null || echo 0)"
  mapfile -t changed < <(git diff --name-only "$base" "$sha" 2>/dev/null | grep '\.py$' || true)
  tests=(); for f in "${changed[@]}"; do [[ "$f" == tests/* ]] && tests+=("$f"); done

  # ---- 1. REAL rebase onto v0.17.0 ----
  ( cd "$WT" && git reset -q --hard "$V017" && git clean -qfdx 2>/dev/null && git checkout -q "$sha" 2>/dev/null )
  rebase_state="NO_COMMITS"; conflict_files=0
  if [ "$ncommits" -gt 0 ]; then
    if ( cd "$WT" && git rebase --onto "$V017" "$base" "$sha" >/dev/null 2>&1 ); then
      rebase_state="REBASED_CLEAN"
    else
      conflict_files="$( cd "$WT" && git diff --name-only --diff-filter=U 2>/dev/null | wc -l | tr -d ' ' )"
      rebase_state="REBASED_CONFLICT"
      ( cd "$WT" && git rebase --abort >/dev/null 2>&1 )
      # fall back to a 3-way apply so we can still run tests on best-effort merged state
      git -C "$SRC" diff "$base" "$sha" > "$WT/.pr.diff" 2>/dev/null
      ( cd "$WT" && git reset -q --hard "$V017" && git clean -qfdx 2>/dev/null && git apply --3way "$WT/.pr.diff" >/dev/null 2>&1 )
      [ "$rebase_state" = "REBASED_CONFLICT" ] && fail=1
    fi
  fi

  # ---- 2. run the PR's own tests on the rebased/merged state ----
  test_state="none"; npass=0; nfail=0; preexisting="n/a"; cmd=""
  present=(); for t in "${tests[@]}"; do [ -f "$WT/$t" ] && present+=("$t"); done
  if [ "${#present[@]}" -gt 0 ]; then
    cmd="cd <v0.17.0+PR#$num worktree> && python -m pytest ${present[*]} -p no:cacheprovider -q"
    out="$( cd "$WT" && timeout 200 "$PY" -m pytest "${present[@]}" -p no:cacheprovider -q --no-header -p no:randomly 2>/dev/null | tail -3 )"
    npass="$(echo "$out" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)"; npass="${npass:-0}"
    nfail="$(echo "$out" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)"; nfail="${nfail:-0}"
    if [ "$nfail" -gt 0 ]; then
      test_state="FAIL"
      # classify: do those same test files fail on PRISTINE v0.17.0 (no PR)?
      ( cd "$WT" && git reset -q --hard "$V017" && git clean -qfdx 2>/dev/null )
      prepresent=(); for t in "${present[@]}"; do [ -f "$WT/$t" ] && prepresent+=("$t"); done
      if [ "${#prepresent[@]}" -gt 0 ]; then
        preout="$( cd "$WT" && timeout 200 "$PY" -m pytest "${prepresent[@]}" -p no:cacheprovider -q --no-header -p no:randomly 2>/dev/null | tail -3 )"
        prefail="$(echo "$preout" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)"; prefail="${prefail:-0}"
        preexisting="$([ "$prefail" -gt 0 ] && echo "yes(${prefail}_fail_on_pristine_v017)" || echo "no_introduced_by_pr")"
        [ "$preexisting" = "no_introduced_by_pr" ] && fail=1
      fi
    else
      test_state="PASS"
    fi
  fi

  "$PY" - "$num" "$br" "$rebase_state" "$conflict_files" "$ncommits" "$test_state" "$npass" "$nfail" "$preexisting" "$cmd" <<'PYJSON'
import sys, json
k=["pr","branch","rebase","conflict_files","commits","test","pass","fail","preexisting_on_v017","repro_cmd"]
v=sys.argv[1:]
v[3]=int(v[3]); v[4]=int(v[4]); v[6]=int(v[6]); v[7]=int(v[7])
v[0]="#"+v[0]
print(json.dumps(dict(zip(k,v))))
PYJSON
done
git worktree remove --force "$WT" 2>/dev/null
echo "{\"summary\":true,\"exit\":$fail,\"meaning\":\"0=no REBASE_CONFLICT and no PR-introduced TEST-fail\"}"
exit $fail
