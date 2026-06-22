#!/usr/bin/env bash
# stack-apply-v017.sh — SET-LEVEL proof: apply ALL current open PRs TOGETHER onto a single
# fresh v0.17.0 worktree (in declared stack order), record per-step CLEAN/CONFLICT, then run a
# representative test slice on the fully-stacked tree. This is the end-to-end check the per-PR
# matrix does NOT provide (per-PR clean != the set stacks clean).
#
# For the 6 forward-port-conflict PRs, apply the pre-resolved v0.17.0 patch (from v017-patches/)
# instead of the raw PR diff. For all others, apply the PR's net diff (origin/main...head) via 3-way.
set -u
REPO="${1:-$PWD}"
PATCHDIR="${2:-$PWD/v017-patches}"
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
PY="$REPO/venv/bin/python"
cd "$REPO" || { echo "FATAL: no repo at $REPO"; exit 2; }
git fetch -q origin main 2>/dev/null
origin_main=$(git rev-parse origin/main)

# conflict PRs use their pre-resolved patch
CONFLICT="49644 49916 50056 50064 50073 50296"

# declared stack order (foundational → providers → features → test-catchups), conflict PRs slotted in place
ORDER="48024 48057 48065 48069 48101 49184 49449 49644 \
50031 50038 50064 \
49915 49916 49917 50021 50022 50032 50040 50041 50042 50045 50046 50047 50048 50053 50054 50055 50056 50068 50073 50146 50155 50296 \
50066 50078 50080 50086 50626 50664"

wt=/tmp/stack_v017; rm -rf "$wt"
git worktree add -q -b _stack_v017 "$wt" "$V017" 2>/dev/null || { git branch -D _stack_v017 2>/dev/null; git worktree add -q -b _stack_v017 "$wt" "$V017"; }
cd "$wt"

# fetch all needed branches once
for pr in $ORDER; do
  br=$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr view "$pr" --repo NousResearch/hermes-agent --json headRefName --jq '.headRefName' 2>/dev/null)
  [ -n "$br" ] && env -u GITHUB_TOKEN -u GH_TOKEN git -C "$REPO" fetch -q https://github.com/arminanton/hermes-agent.git "$br":"refs/_st/$pr" 2>/dev/null
done

CLEAN=0; CONFL=0; FAILED=""
echo "== stacking $(echo $ORDER | wc -w) PRs onto v0.17.0 =="
for pr in $ORDER; do
  if echo " $CONFLICT " | grep -q " $pr "; then
    # apply the pre-resolved v0.17.0 patch
    patch="$PATCHDIR/PR-$pr-onto-v0.17.0.patch"
    if [ ! -f "$patch" ]; then echo "  #$pr  MISSING-PATCH"; CONFL=$((CONFL+1)); FAILED="$FAILED $pr(no-patch)"; continue; fi
    if git apply --3way "$patch" >/tmp/st_$pr.log 2>&1; then
      git add -A; git commit -q -m "stack #$pr (v0.17.0 patch)" 2>/dev/null; echo "  #$pr  PATCH-CLEAN"; CLEAN=$((CLEAN+1))
    else
      u=$(git diff --name-only --diff-filter=U|tr '\n' ' '); git checkout -q . 2>/dev/null; git clean -qfd 2>/dev/null
      echo "  #$pr  PATCH-CONFLICT [$u]"; CONFL=$((CONFL+1)); FAILED="$FAILED $pr(patch:$u)"
    fi
  else
    sha=$(git -C "$REPO" rev-parse refs/_st/$pr 2>/dev/null)
    [ -z "$sha" ] && { echo "  #$pr  NO-REF"; CONFL=$((CONFL+1)); FAILED="$FAILED $pr(no-ref)"; continue; }
    mb=$(git -C "$REPO" merge-base "$sha" "$origin_main")
    if git -C "$REPO" diff "$mb".."$sha" | git apply --3way >/tmp/st_$pr.log 2>&1; then
      git add -A; git commit -q -m "stack #$pr" 2>/dev/null; echo "  #$pr  CLEAN"; CLEAN=$((CLEAN+1))
    else
      u=$(git diff --name-only --diff-filter=U|tr '\n' ' '); git checkout -q . 2>/dev/null; git clean -qfd 2>/dev/null
      echo "  #$pr  CONFLICT [$u]"; CONFL=$((CONFL+1)); FAILED="$FAILED $pr(stack:$u)"
    fi
  fi
done

echo
echo "== stack result: CLEAN=$CLEAN  CONFLICT=$CONFL =="
[ -n "$FAILED" ] && echo "   conflicts:$FAILED"
echo
echo "== syntax: compile all changed .py on the fully-stacked tree =="
changed_py=$(git diff --name-only "$V017"..HEAD | grep '\.py$')
nbad=0; for f in $changed_py; do [ -f "$f" ] && { "$PY" -m py_compile "$f" 2>/dev/null || { echo "   COMPILE-FAIL: $f"; nbad=$((nbad+1)); }; }; done
echo "   compiled $(echo "$changed_py"|wc -w) changed .py files, $nbad failures"
echo
echo "== representative test slice on the FULLY-STACKED tree =="
PYTHONPATH="$wt" "$PY" -m pytest tests/cli/test_reasoning_command.py tests/hermes_cli/test_kanban_db.py tests/run_agent/test_provider_attribution_headers.py tests/agent/test_p2_p3_oversized_handling.py -p no:cacheprovider -q 2>&1 | grep -E 'passed|failed|error' | tail -1

cd "$REPO"
git worktree remove --force "$wt" 2>/dev/null; git branch -D _stack_v017 2>/dev/null
for r in $(git for-each-ref --format='%(refname)' refs/_st 2>/dev/null); do git update-ref -d "$r" 2>/dev/null; done
[ "$CONFL" -eq 0 ] && [ "$nbad" -eq 0 ] && exit 0 || exit 1
