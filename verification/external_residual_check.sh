#!/usr/bin/env bash
# external_residual_check.sh — focused residual: does the cherry-picked stack reproduce
# src's content FOR THE FILES OUR PRs TOUCH? (Not a whole-tree diff, which would include
# every v0.16.0->v0.17.0 upstream change.)
#
# Method: build the cherry-pick stack on fresh v0.17.0 (remote heads, documented-drift
# resolution for #48069/#50056), then for EACH file any feature PR touches, compare the
# stacked tree's version against ./src HEAD's version. Report per-file MATCH / DIFFERS.
# A DIFFERS that isn't one of the 2 documented drifts = a real gap.
set -uo pipefail
FORK="fork"; V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"
SRC="$(git rev-parse --show-toplevel)"; PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"
OUT="${1:-<LOCAL_PATH>"; : > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }

env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" 2>/dev/null || true
PR_JSON="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 --json number,headRefOid 2>/dev/null)"
mapfile -t ROWS < <(echo "$PR_JSON" | "$PY" -c '
import sys,json
for p in sorted(json.load(sys.stdin), key=lambda x:x["number"]):
    if p["number"]==50111: continue
    print("%s\t%s"%(p["number"],p["headRefOid"]))')

WT="$(mktemp -d)"; git worktree add -q "$WT" "$V017" 2>/dev/null
( cd "$WT" && git switch -q -c _resid 2>/dev/null )
declare -A touched
for row in "${ROWS[@]}"; do
  num="$(cut -f1 <<<"$row")"; sha="$(cut -f2 <<<"$row")"
  env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" "$sha" 2>/dev/null || true
  base="$(git merge-base origin/main "$sha" 2>/dev/null || echo "$V017")"
  [ "$(git rev-list --count "$base..$sha" 2>/dev/null || echo 0)" -eq 0 ] && continue
  for f in $(git diff --name-only "$base" "$sha" 2>/dev/null); do touched["$f"]=1; done
  if ! ( cd "$WT" && git cherry-pick "$base..$sha" >/dev/null 2>&1 ); then
    # documented-drift union-resolve, else abort
    ( cd "$WT" && for f in $(git diff --name-only --diff-filter=U); do
        "$PY" - "$f" <<'PYR'
import sys
f=sys.argv[1]; o=[]
for ln in open(f):
    if ln.startswith(("<<<<<<< ","=======",">>>>>>> ")): continue
    o.append(ln)
open(f,"w").write("".join(o))
PYR
      done; git add -A; git -c core.editor=true cherry-pick --continue >/dev/null 2>&1 || git cherry-pick --skip >/dev/null 2>&1 )
  fi
done

log "=== PER-FILE RESIDUAL: stacked(v0.17.0+PRs) vs ./src HEAD, for the $(echo "${!touched[@]}" | wc -w) PR-touched files ==="
match=0; differs=0; declare -a diff_files
DOCUMENTED="tools/mcp_tool.py tests/hermes_cli/test_kanban_db.py"
for f in "${!touched[@]}"; do
  # skip generated/private-only
  case "$f" in *.bak|*.bak.*|.project-intel/*) continue;; esac
  st="$( cd "$WT" && git show ":$f" 2>/dev/null || cat "$WT/$f" 2>/dev/null )"
  sr="$(git -C "$SRC" show "HEAD:$f" 2>/dev/null)"
  if [ "$st" = "$sr" ]; then match=$((match+1))
  else
    differs=$((differs+1)); diff_files+=("$f")
  fi
done
log "MATCH=$match  DIFFERS=$differs"
log "--- DIFFERS files (expect only the 2 documented drifts) ---"
undoc=0
for f in "${diff_files[@]}"; do
  doc="no"; for d in $DOCUMENTED; do [ "$f" = "$d" ] && doc="DOCUMENTED-DRIFT"; done
  [ "$doc" = "no" ] && { undoc=$((undoc+1)); doc="*** UNDOCUMENTED ***"; }
  log "  $f : $doc"
done
git worktree remove --force "$WT" 2>/dev/null; git branch -D _resid 2>/dev/null || true
log ""
log "RESULT: $([ "$undoc" -eq 0 ] && echo "PASS — all DIFFERS are documented drifts ($differs files), 0 undocumented" || echo "FAIL — $undoc UNDOCUMENTED differing files")"
exit $undoc
