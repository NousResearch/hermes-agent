#!/usr/bin/env bash
# independent_cumulative_verify.sh — INDEPENDENT of the per-PR harness + reconciler.
#
# Two proofs from a FRESH worktree, by construction (not by my line-normalizer):
#
# PROOF A (composition onto v0.17.0): apply ALL 41 open PR diffs cumulatively onto a
#   fresh v0.17.0 worktree, in PR-number order (dependency-safe: deps like #49644/#50064
#   have lower numbers than dependents like #50078). Record per-PR CLEAN/3WAY/CONFLICT and
#   any leftover conflict markers. Then run a broad slice of the suite on the stacked tree.
#
# PROOF B (coverage by construction onto v0.16.0): apply every open PR diff + the #50111
#   deferred proof-patches onto a fresh v0.16.0 worktree, then `git diff` that result
#   against the real src HEAD. A near-empty diff proves every src line is reproduced by
#   (PRs + deferred) — independent of the reconciler's normalization.
set -uo pipefail
FORK="fork"
V016="3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"
V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"
SRC="$(git rev-parse --show-toplevel)"
PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"
OUT="${1:-/mnt/devvm/custom/hermes/reconcile-tmp/independent-verify.out}"
: > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }

env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" 2>/dev/null || true
PR_JSON="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 --json number,headRefOid,headRefName 2>/dev/null)"
# PR-number order; #50111 (deferred tracker) handled separately in PROOF B.
mapfile -t ROWS < <(echo "$PR_JSON" | "$PY" -c '
import sys,json
for p in sorted(json.load(sys.stdin), key=lambda x:x["number"]):
    if p["number"]==50111: continue
    print("%s\t%s\t%s"%(p["number"],p["headRefOid"],p["headRefName"]))')

# ---------------- PROOF A: cumulative apply onto v0.17.0 ----------------
log "=== PROOF A — cumulative apply of ${#ROWS[@]} feature PRs onto FRESH v0.17.0 (in order) ==="
WTA="$(mktemp -d)"; git worktree add -q --detach "$WTA" "$V017" 2>/dev/null
a_clean=0; a_3way=0; a_conflict=0
for row in "${ROWS[@]}"; do
  num="$(cut -f1 <<<"$row")"; sha="$(cut -f2 <<<"$row")"
  env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" "$sha" 2>/dev/null || true
  base="$(git merge-base origin/main "$sha" 2>/dev/null || echo "$V017")"
  git -C "$SRC" diff "$base" "$sha" -- . ':(exclude)*.bak' ':(exclude)*.bak.*' ':(exclude).project-intel/**' > "$WTA/.cur.diff" 2>/dev/null
  if [ ! -s "$WTA/.cur.diff" ]; then continue; fi
  if ( cd "$WTA" && git apply --check "$WTA/.cur.diff" 2>/dev/null ); then
    ( cd "$WTA" && git apply "$WTA/.cur.diff" 2>/dev/null ); a_clean=$((a_clean+1)); st="CLEAN"
  elif ( cd "$WTA" && git apply --3way --check "$WTA/.cur.diff" 2>/dev/null ); then
    ( cd "$WTA" && git apply --3way "$WTA/.cur.diff" 2>/dev/null ); a_3way=$((a_3way+1)); st="3WAY"
  else
    a_conflict=$((a_conflict+1)); st="CONFLICT(skipped-to-continue-stack)"
  fi
  log "  applied #$num : $st"
done
markers="$( grep -rln '^<<<<<<< \|^>>>>>>> ' --include='*.py' "$WTA" 2>/dev/null | wc -l | tr -d ' ' )"
log "PROOF A result: CLEAN=$a_clean  3WAY=$a_3way  CONFLICT=$a_conflict  leftover-conflict-marker-files=$markers"
# broad smoke on the stacked tree (a representative slice; full suite is huge)
log "PROOF A smoke (representative suites on the stacked tree):"
( cd "$WTA" && timeout 300 "$PY" -m pytest tests/run_agent/test_run_agent.py tests/agent/test_model_metadata.py tests/cli/test_reasoning_command.py tests/agent/test_system_prompt_prelude.py tests/test_background_review_session_isolation.py -p no:cacheprovider -q --no-header -p no:randomly 2>/dev/null | tail -3 | sed 's/^/    /' ) | tee -a "$OUT"
git worktree remove --force "$WTA" 2>/dev/null

# ---------------- PROOF B: coverage-by-construction onto v0.16.0 ----------------
log ""
log "=== PROOF B — apply (all feature PRs + #50111 deferred patches) onto FRESH v0.16.0, diff vs src HEAD ==="
WTB="$(mktemp -d)"; git worktree add -q --detach "$WTB" "$V016" 2>/dev/null
for row in "${ROWS[@]}"; do
  num="$(cut -f1 <<<"$row")"; sha="$(cut -f2 <<<"$row")"
  base="$(git merge-base origin/main "$sha" 2>/dev/null || echo "$V016")"
  git -C "$SRC" diff "$base" "$sha" -- . ':(exclude)*.bak' ':(exclude)*.bak.*' ':(exclude).project-intel/**' > "$WTB/.cur.diff" 2>/dev/null
  [ -s "$WTB/.cur.diff" ] && ( cd "$WTB" && git apply --3way "$WTB/.cur.diff" 2>/dev/null || true )
done
# apply the #50111 deferred proof-patches too (they carry the deferred lines)
DEF="deferred/residual-lines-on-v0.17.0"
git ls-tree -r --name-only "$FORK/$DEF" 2>/dev/null | grep '\.patch$' | while read -r p; do
  git show "$FORK/$DEF:$p" 2>/dev/null > "$WTB/.def.patch"
  ( cd "$WTB" && git apply --3way "$WTB/.def.patch" 2>/dev/null || true )
done
# Now diff the constructed tree against the real src HEAD (the overlay we must reproduce),
# excluding generated artifacts. Count residual changed SOURCE lines.
resid="$( cd "$WTB" && git --git-dir="$SRC/.git" --work-tree="$WTB" diff HEAD -- . \
  ':(exclude)*.bak' ':(exclude)*.bak.*' ':(exclude).project-intel/**' ':(exclude)deferred/**' ':(exclude)verification/**' 2>/dev/null \
  | grep -E '^[+-]' | grep -vE '^[+-]{3} ' | wc -l | tr -d ' ' )"
log "PROOF B residual src lines (constructed tree vs src HEAD; 0 = perfect reproduction): $resid"
if [ "${resid:-1}" != "0" ]; then
  log "  --- first 40 residual lines (these would be UNCOVERED if any) ---"
  ( cd "$WTB" && git --git-dir="$SRC/.git" --work-tree="$WTB" diff HEAD -- . \
    ':(exclude)*.bak' ':(exclude)*.bak.*' ':(exclude).project-intel/**' ':(exclude)deferred/**' ':(exclude)verification/**' 2>/dev/null \
    | grep -E '^[+-]' | grep -vE '^[+-]{3} ' | head -40 | sed 's/^/    /' ) | tee -a "$OUT"
fi
git worktree remove --force "$WTB" 2>/dev/null
log ""
log "=== DONE ==="
