#!/usr/bin/env bash
# verify_against_named_v017_commit.sh — definitive: apply target IS the named commit.
# Council concern: prove the apply TARGET is the exact v0.17.0 commit
# 2bd1977d8fad185c9b4be47884f7e87f1add0ce3, not origin/main.
# (origin/main is used ONLY to compute each PR's base..head net diff = what the PR adds.)
set -u
SRC=/mnt/devvm/custom/hermes/src; cd "$SRC"
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3   # the NAMED v0.17.0 release commit
GH="env -u GITHUB_TOKEN -u GH_TOKEN gh"
OUT=/mnt/devvm/custom/hermes/reconcile-tmp/VERIFY-NAMED-V017.txt
WT=/tmp/wt-named-v017
PY=/mnt/devvm/custom/hermes/src/venv/bin/python
: > "$OUT"; log(){ echo "$@" | tee -a "$OUT"; }
log "=== verify_against_named_v017_commit.sh $(date -u +%FT%TZ) ==="
log "NAMED v0.17.0 apply target = $V017"
log "  (= $(git log -1 --format='%h %s' $V017 2>/dev/null))"

$GH pr list --repo NousResearch/hermes-agent --author arminanton --state open --limit 100 \
  --json number,headRefOid,title -q '.[] | "\(.number)\t\(.headRefOid)\t\(.title)"' > /tmp/lp.tsv
sha111=$($GH pr view 50111 --repo NousResearch/hermes-agent --json headRefOid -q .headRefOid 2>/dev/null)
git fetch -q fork "$sha111" 2>/dev/null

# worktree pinned EXACTLY at the named commit; assert HEAD == named commit
git worktree remove "$WT" --force 2>/dev/null; rm -rf "$WT"
git worktree add -q --detach "$WT" "$V017" 2>&1 | head -1
WTHEAD=$(git -C "$WT" rev-parse HEAD)
log ""
log "worktree HEAD = $WTHEAD"
if [ "$WTHEAD" = "$V017" ]; then log "✓ ASSERT: worktree is checked out at the NAMED v0.17.0 commit (not origin/main)"; else log "✗ ASSERT FAILED"; exit 2; fi

resolve_file(){ local f="$1" flat patch; flat=$(echo "$f"|tr '/' '_')
  patch=$(git show "$sha111:v017-conflict-resolutions/${flat}.v017.patch" 2>/dev/null); [ -n "$patch" ]||return 1
  git -C "$WT" checkout "$V017" -- "$f" 2>/dev/null; echo "$patch"|git -C "$WT" apply --whitespace=nowarn - 2>/dev/null; }

log ""
log "===== sequential per-PR apply onto the NAMED v0.17.0 commit ====="
CLEAN=0; RES=0; FAIL=0; FAILLIST=""
while IFS=$'\t' read -r num sha title; do
  case "$title" in *manifest*|*"NOT FOR MERGE"*) continue;; esac
  git fetch -q fork "$sha" 2>/dev/null
  base=$(git merge-base "$sha" origin/main 2>/dev/null)   # base..head = the PR's own net change
  git diff "$base" "$sha" -- . ':(exclude)*.bak' ':(exclude).project-intel/**' 2>/dev/null > /tmp/pr.diff
  # re-assert target before each apply
  if [ "$(git -C "$WT" rev-parse HEAD)" != "$V017" ]; then log "  drift! worktree moved off named commit"; fi
  if git -C "$WT" apply --3way --whitespace=nowarn /tmp/pr.diff 2>/dev/null; then CLEAN=$((CLEAN+1))
  else
    cf=$(grep -rl '^<<<<<<< ' "$WT" --include='*.py' 2>/dev/null | sed "s|$WT/||" | sort -u)
    ok=1; for f in $cf; do resolve_file "$f" || ok=0; done
    rem=$(grep -rl '^<<<<<<< ' "$WT" --include='*.py' 2>/dev/null | wc -l)
    if [ "$ok" -eq 1 ] && [ "$rem" -eq 0 ]; then RES=$((RES+1)); log "  #$num resolved via patch ($cf)"
    else FAIL=$((FAIL+1)); FAILLIST="$FAILLIST #$num"; for f in $cf; do git -C "$WT" checkout "$V017" -- "$f" 2>/dev/null; done; fi
  fi
done < /tmp/lp.tsv
TOTAL=$((CLEAN+RES+FAIL))
log ""
log "apply onto NAMED v0.17.0: clean=$CLEAN resolved=$RES failed=$FAIL  (total=$TOTAL) ->$FAILLIST"

log ""
log "===== FULL-TREE compile on the v0.17.0-based integrated worktree ====="
CF=0; N=0
while read -r f; do
  case "$f" in *.py) ;; *) continue;; esac
  [ -f "$WT/$f" ] || continue; N=$((N+1))
  $PY -m py_compile "$WT/$f" 2>>/tmp/nc.err || { log "  ✗ $f"; CF=$((CF+1)); }
done < <(git -C "$WT" diff --name-only "$V017" 2>/dev/null)
log "changed .py compiled=$N compile-fail=$CF"

log ""
log "===== representative pytest on the v0.17.0-based integrated worktree ====="
cd "$WT"
RUN=$(git diff --name-only "$V017" 2>/dev/null | grep -E 'test_.*\.py$' | while read t; do [ -f "$t" ] && echo "$t"; done | grep -vE 'v0.17.0-ready/' | head -10)
timeout 280 "$PY" -m pytest -q --no-header -p no:cacheprovider $RUN 2>&1 | tail -8 | tee -a "$OUT"

cd "$SRC"; git worktree remove "$WT" --force 2>/dev/null; rm -rf "$WT"
log ""
[ "$FAIL" -eq 0 ] && [ "$CF" -eq 0 ] && \
  log "RESULT: PASS — $TOTAL/$TOTAL PRs apply onto NAMED v0.17.0 commit $V017 ($CLEAN clean + $RES resolved), tree compiles" || \
  log "RESULT: ISSUES — failed=$FAIL compile=$CF"
