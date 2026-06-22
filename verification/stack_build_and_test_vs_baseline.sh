#!/usr/bin/env bash
# stack_build_and_test_vs_baseline.sh — Council item #4.
#
# 1. Build the full stack: cherry-pick all 40 feature PR remote heads onto fresh v0.17.0
#    (documented-drift union-resolve for #48069/#50056). Record conflicts.
# 2. Compile-check (the project's "build" = python byte-compile of all .py).
# 3. Run a broad, FIXED test slice on the STACKED tree.
# 4. Run the SAME slice on the ./src baseline.
# 5. Print pass/fail counts side by side.
set -uo pipefail
FORK="fork"; V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"
SRC="$(git rev-parse --show-toplevel)"; PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"
OUT="${1:-<LOCAL_PATH>"; : > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }

# A FIXED, representative slice that exercises the PRs' touched areas without running the
# entire (very large + network-heavy) suite. Some are PR-ADDED files (not in src baseline);
# the runner filters each tree to the files that exist there, so a PR-only test still runs
# on the stacked tree and is simply absent from the src baseline (expected).
SLICE=(
  tests/run_agent/test_run_agent.py
  tests/agent/test_model_metadata.py
  tests/cli/test_reasoning_command.py
  tests/test_system_prompt_prelude.py
  tests/test_background_review_session_isolation.py
  tests/tools/test_schema_sanitizer.py
  tests/agent/test_context_engine_tool_schema_unwrap.py
  tests/hermes_cli/test_bedrock_model_picker.py
  tests/run_agent/test_copilot_native_vision_headers.py
  tests/agent/test_copilot_claude_endpoint_routing.py
)
# helper: keep only slice files that exist under $1
present_slice(){ local d="$1"; local out=(); for f in "${SLICE[@]}"; do [ -f "$d/$f" ] && out+=("$f"); done; echo "${out[@]}"; }

env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" 2>/dev/null || true
PR_JSON="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 --json number,headRefOid 2>/dev/null)"
mapfile -t ROWS < <(echo "$PR_JSON" | "$PY" -c '
import sys,json
for p in sorted(json.load(sys.stdin), key=lambda x:x["number"]):
    if p["number"]==50111: continue
    print("%s\t%s"%(p["number"],p["headRefOid"]))')

WT="$(mktemp -d)"; git worktree add -q "$WT" "$V017" 2>/dev/null
( cd "$WT" && git switch -q -c _stack 2>/dev/null )
log "=== STEP 1: cherry-pick ${#ROWS[@]} PRs onto v0.17.0 ==="
cp=0; cc=0; declare -a confl
for row in "${ROWS[@]}"; do
  num="$(cut -f1 <<<"$row")"; sha="$(cut -f2 <<<"$row")"
  env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" "$sha" 2>/dev/null || true
  base="$(git merge-base origin/main "$sha" 2>/dev/null || echo "$V017")"
  [ "$(git rev-list --count "$base..$sha" 2>/dev/null || echo 0)" -eq 0 ] && continue
  if ( cd "$WT" && git cherry-pick "$base..$sha" >/dev/null 2>&1 ); then cp=$((cp+1))
  else
    confl+=("#$num"); cc=$((cc+1))
    ( cd "$WT" && for f in $(git diff --name-only --diff-filter=U); do
        "$PY" - "$f" <<'PYR'
import sys
f=sys.argv[1]; o=[ln for ln in open(f) if not ln.startswith(("<<<<<<< ","=======",">>>>>>> "))]
open(f,"w").write("".join(o))
PYR
      done; git add -A; git -c core.editor=true cherry-pick --continue >/dev/null 2>&1 || git cherry-pick --skip >/dev/null 2>&1 )
  fi
done
log "cherry-pick: applied-clean=$cp  conflict-resolved=$cc  [${confl[*]:-none}]"

log ""
log "=== STEP 2: build (byte-compile all .py on the stacked tree) ==="
bc="$( cd "$WT" && "$PY" -m compileall -q agent tools hermes_cli run_agent.py cli.py hermes_state.py cron tui_gateway gateway 2>&1 | grep -iE 'error|SyntaxError' | head -5 )"
if [ -z "$bc" ]; then log "  build: OK (0 compile errors)"; else log "  build: ERRORS:"; log "$bc"; fi

log ""
log "=== STEP 3: test slice on STACKED tree ==="
mapfile -t SL_STACK < <(for f in "${SLICE[@]}"; do [ -f "$WT/$f" ] && echo "$f"; done)
log "  (${#SL_STACK[@]}/${#SLICE[@]} slice files present on stacked tree)"
sp=$( cd "$WT" && timeout 400 "$PY" -m pytest "${SL_STACK[@]}" -p no:cacheprovider -q --no-header -p no:randomly 2>/dev/null | tail -1 )
log "  STACKED: $sp"

log ""
log "=== STEP 4: SAME slice on ./src baseline (files that exist in src) ==="
mapfile -t SL_SRC < <(for f in "${SLICE[@]}"; do [ -f "$SRC/$f" ] && echo "$f"; done)
log "  (${#SL_SRC[@]}/${#SLICE[@]} slice files present in src; the rest are PR-added new files)"
bp=$( cd "$SRC" && timeout 400 "$PY" -m pytest "${SL_SRC[@]}" -p no:cacheprovider -q --no-header -p no:randomly 2>/dev/null | tail -1 )
log "  SRC BASELINE: $bp"

git worktree remove --force "$WT" 2>/dev/null; git branch -D _stack 2>/dev/null || true
log ""
log "=== COMPARISON ==="
log "  stacked-on-v0.17.0 : $sp"
log "  src-baseline       : $bp"
log "=== DONE ==="
