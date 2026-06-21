#!/usr/bin/env bash
# external_cherrypick_all_prs.sh — the Council's decisive end-to-end proof.
#
# On a FRESH v0.17.0 worktree, cherry-pick EVERY open PR's own commits (from its
# REMOTE head on the fork, in PR-number order = dependency-safe) using real 3-way
# merge. Then diff the resulting tree against ./src HEAD. The residual must be EMPTY
# except the two explicitly documented drifts:
#   - #48069 tools/mcp_tool.py (keepalive refactor on origin/main)
#   - #50056 tests/hermes_cli/test_kanban_db.py (sqlite3+subprocess import combine)
# Reads ONLY remote PR heads (git fetch fork <sha>), never local src for PR content.
set -uo pipefail
FORK="fork"
V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"
SRC="$(git rev-parse --show-toplevel)"
PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"
OUT="${1:-/mnt/devvm/custom/hermes/reconcile-tmp/external-cherrypick.out}"
: > "$OUT"; log(){ echo "$@" | tee -a "$OUT"; }

env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" 2>/dev/null || true
PR_JSON="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 --json number,headRefOid,headRefName 2>/dev/null)"
mapfile -t ROWS < <(echo "$PR_JSON" | "$PY" -c '
import sys,json
for p in sorted(json.load(sys.stdin), key=lambda x:x["number"]):
    if p["number"]==50111: continue   # deferred tracker, not a feature PR
    print("%s\t%s\t%s"%(p["number"],p["headRefOid"],p["headRefName"]))')

WT="$(mktemp -d)"
git worktree add -q "$WT" "$V017" 2>/dev/null
( cd "$WT" && git switch -q -c _external_stack 2>/dev/null )
log "=== EXTERNAL cherry-pick of ${#ROWS[@]} feature PR remote heads onto fresh v0.17.0 ==="
clean=0; conflict=0; declare -a conflicts
for row in "${ROWS[@]}"; do
  num="$(cut -f1 <<<"$row")"; sha="$(cut -f2 <<<"$row")"
  env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" "$sha" 2>/dev/null || true
  base="$(git merge-base origin/main "$sha" 2>/dev/null || echo "$V017")"
  ncommits="$(git rev-list --count "$base..$sha" 2>/dev/null || echo 0)"
  [ "$ncommits" -eq 0 ] && { log "  #$num : NO_COMMITS"; continue; }
  if ( cd "$WT" && git cherry-pick "$base..$sha" >/dev/null 2>&1 ); then
    clean=$((clean+1)); log "  #$num : CLEAN ($ncommits commit/s)"
  else
    cf="$( cd "$WT" && git diff --name-only --diff-filter=U 2>/dev/null | tr '\n' ',' )"
    conflict=$((conflict+1)); conflicts+=("#$num:$cf")
    log "  #$num : CONFLICT [$cf]"
    # Resolve the two DOCUMENTED drifts deterministically to keep the stack going;
    # any OTHER conflict is a real failure we surface.
    if [ "$num" = "50056" ]; then
      # keep BOTH imports (sqlite3 + subprocess)
      ( cd "$WT" && for f in $(git diff --name-only --diff-filter=U); do
          if grep -q '^<<<<<<< ' "$f"; then
            "$PY" - "$f" <<'PYRES'
import sys,re
f=sys.argv[1]; t=open(f).read()
# keep both sides of each conflict (union), drop markers
out=[]; mode=0
for ln in t.splitlines(keepends=True):
    if ln.startswith("<<<<<<< "): mode=1; continue
    if ln.startswith("======="): mode=2; continue
    if ln.startswith(">>>>>>> "): mode=0; continue
    out.append(ln)
open(f,"w").write("".join(out))
PYRES
          fi
        done; git add -A && git cherry-pick --continue >/dev/null 2>&1 || git -c core.editor=true cherry-pick --continue >/dev/null 2>&1 )
    elif [ "$num" = "48069" ]; then
      # documented keep-both keepalive merge — take the union, drop markers
      ( cd "$WT" && for f in $(git diff --name-only --diff-filter=U); do
          "$PY" - "$f" <<'PYRES'
import sys
f=sys.argv[1]; t=open(f).read(); out=[]
for ln in t.splitlines(keepends=True):
    if ln.startswith(("<<<<<<< ","=======",">>>>>>> ")): continue
    out.append(ln)
open(f,"w").write("".join(out))
PYRES
        done; git add -A && git -c core.editor=true cherry-pick --continue >/dev/null 2>&1 )
    else
      ( cd "$WT" && git cherry-pick --abort 2>/dev/null )
    fi
  fi
done
log "cherry-pick: CLEAN=$clean  CONFLICT=$conflict  [${conflicts[*]:-none}]"

# ---- final residual: stacked tree vs ./src HEAD, excluding generated + deferred-only files
log ""
log "=== RESIDUAL: cherry-picked-stack vs ./src HEAD (expect empty modulo private-only files) ==="
# Compare only source files both trees track; exclude generated artifacts + private-overlay
# files that are intentionally NOT in any feature PR (they live in #50111 deferred).
resid="$( cd "$WT" && git --git-dir="$SRC/.git" --work-tree="$WT" diff HEAD --stat -- . \
  ':(exclude)*.bak' ':(exclude)*.bak.*' ':(exclude).project-intel/**' \
  ':(exclude)deferred/**' ':(exclude)verification/**' 2>/dev/null | tail -40 )"
log "$resid"
git worktree remove --force "$WT" 2>/dev/null
git branch -D _external_stack 2>/dev/null || true
log "=== DONE ==="
