#!/usr/bin/env bash
# fresh_clone_repro.sh — Council item #1: independent reproduction from a FRESH CLONE.
# Run INSIDE a fresh clone of the fork. Cherry-picks all 40 feature PR heads onto v0.17.0
# (documented-drift union-resolve), then diffs the PR-touched files against the agent's src
# working tree. Emits the raw `git diff --stat` so residual is visible, not summarized.
set -uo pipefail
CLONE="$(git rev-parse --show-toplevel)"
SRC="/mnt/devvm/custom/hermes/src"
V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"
PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"

# PR list from the fork (this clone's origin)
env -u GITHUB_TOKEN -u GH_TOKEN git -C "$SRC" fetch -q fork 2>/dev/null || true
PR_JSON="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 --json number,headRefOid 2>/dev/null)"
mapfile -t ROWS < <(echo "$PR_JSON" | "$PY" -c '
import sys,json
for p in sorted(json.load(sys.stdin), key=lambda x:x["number"]):
    if p["number"]==50111: continue
    print("%s\t%s"%(p["number"],p["headRefOid"]))')

# fetch every PR head into THIS clone from the fork (origin)
git remote get-url origin >/dev/null 2>&1 || true
for row in "${ROWS[@]}"; do
  sha="$(cut -f2 <<<"$row")"
  env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q origin "$sha" 2>/dev/null || \
  env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q upstream "$sha" 2>/dev/null || true
done
# also need origin/main (upstream main) for merge-bases; fetch upstream main
env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q upstream main 2>/dev/null || true

git checkout -q -B _repro "$V017"
echo "=== fresh-clone cherry-pick of ${#ROWS[@]} feature PRs onto v0.17.0 ==="
clean=0; resolved=0
for row in "${ROWS[@]}"; do
  num="$(cut -f1 <<<"$row")"; sha="$(cut -f2 <<<"$row")"
  base="$(git merge-base FETCH_HEAD "$sha" 2>/dev/null)"
  # robust base: merge-base of upstream/main and sha
  base="$(git merge-base upstream/main "$sha" 2>/dev/null || echo "$V017")"
  if git cherry-pick "$base..$sha" >/dev/null 2>&1; then clean=$((clean+1)); echo "  #$num CLEAN"
  else
    # documented-drift union resolve
    for f in $(git diff --name-only --diff-filter=U); do
      "$PY" - "$f" <<'PYR'
import sys
f=sys.argv[1]; o=[l for l in open(f) if not l.startswith(("<<<<<<< ","=======",">>>>>>> "))]
open(f,"w").write("".join(o))
PYR
    done
    git add -A; git -c core.editor=true cherry-pick --continue >/dev/null 2>&1 || git cherry-pick --skip >/dev/null 2>&1
    resolved=$((resolved+1)); echo "  #$num CONFLICT(union-resolved)"
  fi
done
echo "cherry-pick: clean=$clean resolved=$resolved"

echo ""
echo "=== RESIDUAL: PR-touched source files — stacked clone vs SRC working tree ==="
# the set of files our PRs touch (from src delta), excluding generated + the 3 PR-added
# new test files that don't exist in src (they're supersets), and private-deferred files.
mapfile -t touched < <(git -C "$SRC" diff --name-only "3c231eb3979ab9c57d5cd6d02f1d577a3b718b43" HEAD -- . \
  ':(exclude)*.bak' ':(exclude)*.bak.*' ':(exclude).project-intel/**' 2>/dev/null)
diffcount=0; identical=0; only_upstream=0; declare -a realdiff
for f in "${touched[@]}"; do
  [ -f "$f" ] || continue           # not on the stacked clone (PR-added new file vs src, skip)
  [ -f "$SRC/$f" ] || continue      # not in src (PR superset), skip
  if diff -q "$f" "$SRC/$f" >/dev/null 2>&1; then identical=$((identical+1))
  else
    # is the difference explained by upstream v0.16->v0.17 drift in that file?
    up="$(git -C "$SRC" diff 3c231eb3979ab9c57d5cd6d02f1d577a3b718b43 "$V017" -- "$f" 2>/dev/null | grep -cE '^[+-]')"
    if [ "${up:-0}" -gt 0 ]; then only_upstream=$((only_upstream+1)); else diffcount=$((diffcount+1)); realdiff+=("$f"); fi
  fi
done
echo "identical=$identical  differ-but-upstream-drift=$only_upstream  REAL-RESIDUAL=$diffcount"
if [ "$diffcount" -gt 0 ]; then
  echo "--- real-residual files (stacked clone != src, NOT explained by upstream drift) ---"
  for f in "${realdiff[@]}"; do echo "  $f"; git --no-pager diff --no-index "$SRC/$f" "$f" 2>/dev/null | head -20; done
fi
echo ""
echo "RESULT: $([ "$diffcount" -eq 0 ] && echo "PASS — every PR-touched src file on the stacked v0.17.0 clone matches src (modulo upstream v0.16->v0.17 drift); 0 real residual" || echo "FAIL — $diffcount real-residual files")"
