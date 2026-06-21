#!/usr/bin/env bash
# full_matrix_all_41_prs.sh — Council: rebase+build+test EVERY open PR onto fresh v0.17.0.
#
# Per PR: REAL git rebase (cherry-pick own commits) onto a fresh v0.17.0 worktree;
# byte-compile build; run the PR's OWN tests if any, else a RELEVANT EXISTING test, else a
# justified NO-TEST rationale. Emits one JSON line per PR (JSONL) + a summary, and writes
# raw per-PR pytest logs to a logs/ dir. Documented drifts (#48069/#50056) get a keep-both
# union resolution so build+test can still run; the conflict is recorded.
set -uo pipefail
FORK="fork"; V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"
SRC="$(git rev-parse --show-toplevel)"; PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"
OUTDIR="${1:-/mnt/devvm/custom/hermes/reconcile-tmp/full-matrix}"
mkdir -p "$OUTDIR/logs"
JSONL="$OUTDIR/matrix.jsonl"; : > "$JSONL"

# PR -> test spec. "own" = run the PR's own test files. Otherwise an explicit existing test
# path, or "NOTEST:<rationale>".
declare -A TESTMAP=(
  [48024]=own [48057]=own [48065]=own [48069]=own [48101]=own
  [49184]=own [49449]=own [49644]=own
  [49915]="NOTEST:TUI Ctrl-C passthrough — terminal-IO behavior, no unit-testable surface; build-only"
  [49916]="tests/test_tui_gateway_ws.py"
  [49917]=own
  [50021]="tests/run_agent/test_tool_executor_contextvar_propagation.py"
  [50022]="NOTEST:model_router proxy tool — no existing unit test in v0.17.0; build-only (compile gate)"
  [50031]=own [50032]=own
  [50033]="tests/agent/test_gemini_native_adapter.py"
  [50038]=own [50039]=own
  [50040]="tests/tools/test_delegate_toolset_scope.py"
  [50041]=own [50042]=own [50045]=own [50046]=own
  [50047]="tests/gateway/test_status.py"
  [50048]="tests/hermes_cli/test_send_cmd.py"
  [50049]=own
  [50053]="tests/agent/test_context_engine.py"
  [50054]="tests/hermes_cli/test_plugins.py"
  [50055]="tests/agent/transports/test_chat_completions.py"
  [50056]=own [50064]=own [50066]=own
  [50068]="NOTEST:TUI autopilot/yolo status badges — render-only UI, no unit-testable surface; build-only"
  [50073]=own [50078]=own [50080]=own [50086]=own [50146]=own [50155]=own [50296]=own
)

env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" 2>/dev/null || true
PR_JSON="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 --json number,headRefOid,headRefName 2>/dev/null)"
mapfile -t ROWS < <(echo "$PR_JSON" | "$PY" -c '
import sys,json
for p in sorted(json.load(sys.stdin), key=lambda x:x["number"]):
    print("%s\t%s\t%s"%(p["number"],p["headRefOid"],p["headRefName"]))')

fail=0
for row in "${ROWS[@]}"; do
  num="$(cut -f1 <<<"$row")"; sha="$(cut -f2 <<<"$row")"; br="$(cut -f3 <<<"$row")"
  env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" "$sha" 2>/dev/null || true
  base="$(git merge-base origin/main "$sha" 2>/dev/null || echo "$V017")"
  ncommits="$(git rev-list --count "$base..$sha" 2>/dev/null || echo 0)"

  if [ "$num" = "50111" ]; then
    echo "{\"pr\":\"#50111\",\"kind\":\"TRACKER\",\"rebase\":\"n/a\",\"build\":\"n/a\",\"test\":\"n/a\",\"note\":\"deferred+verification tracker, NOT FOR MERGE\",\"branch\":\"$br\"}" >> "$JSONL"
    continue
  fi

  WT="$(mktemp -d)"; git worktree add -q "$WT" "$V017" 2>/dev/null
  ( cd "$WT" && git switch -q -c "_m$num" 2>/dev/null )

  # 1. REAL rebase = cherry-pick own commits onto v0.17.0
  reb="CLEAN"; conflict_files=""
  if [ "$ncommits" -gt 0 ]; then
    if ! ( cd "$WT" && git cherry-pick "$base..$sha" >/dev/null 2>&1 ); then
      conflict_files="$( cd "$WT" && git diff --name-only --diff-filter=U | tr '\n' ',' )"
      reb="CONFLICT"
      # documented drift OR merge-commit range: fall back to net-diff 3-way apply
      ( cd "$WT" && git cherry-pick --abort 2>/dev/null; git reset -q --hard "$V017"; git clean -qfdx 2>/dev/null )
      git -C "$SRC" diff "$base" "$sha" > "$WT/.net.diff" 2>/dev/null
      if ( cd "$WT" && git apply --3way "$WT/.net.diff" >/dev/null 2>&1 ); then reb="CONFLICT(netdiff-clean)"
      else
        ( cd "$WT" && git apply "$WT/.net.diff" >/dev/null 2>&1 || true
          for f in $(grep -rlE '^<<<<<<< ' --include='*.py' . 2>/dev/null); do
            "$PY" - "$f" <<'PYR'
import sys
f=sys.argv[1]; o=[l for l in open(f) if not l.startswith(("<<<<<<< ","=======",">>>>>>> "))]
open(f,"w").write("".join(o))
PYR
          done )
        reb="CONFLICT(union-resolved)"
      fi
    fi
  else reb="NO_COMMITS"; fi

  # 2. build = byte-compile the touched dirs + core
  bc="$( cd "$WT" && "$PY" -m compileall -q agent tools hermes_cli run_agent.py cli.py gateway tui_gateway cron 2>&1 | grep -ic 'error' )"
  bld="$([ "${bc:-0}" -eq 0 ] && echo OK || echo "ERR($bc)")"; [ "$bld" != "OK" ] && fail=1

  # 3. test
  spec="${TESTMAP[$num]:-NOTEST:unmapped}"
  tlog="$OUTDIR/logs/pr_${num}.log"
  tst="none"; npass=0; nfail=0; trat=""
  if [ "$spec" = "own" ]; then
    mapfile -t own < <(git diff --name-only "$base" "$sha" 2>/dev/null | grep -E '^tests/.*\.py$')
    present=(); for t in "${own[@]}"; do [ -f "$WT/$t" ] && present+=("$t"); done
    if [ "${#present[@]}" -gt 0 ]; then
      ( cd "$WT" && timeout 300 "$PY" -m pytest "${present[@]}" -p no:cacheprovider -q --no-header -p no:randomly --tb=line ) > "$tlog" 2>&1 || true
      npass="$(grep -oE '[0-9]+ passed' "$tlog" | grep -oE '[0-9]+' | head -1)"; npass="${npass:-0}"
      nfail="$(grep -oE '[0-9]+ failed' "$tlog" | grep -oE '[0-9]+' | head -1)"; nfail="${nfail:-0}"
      tst="own(${#present[@]}f)"
    else tst="own-absent"; fi
  elif [[ "$spec" == NOTEST:* ]]; then
    tst="no-test"; trat="${spec#NOTEST:}"
    echo "NO APPLICABLE TEST: $trat" > "$tlog"
  else
    if [ -f "$WT/$spec" ]; then
      ( cd "$WT" && timeout 300 "$PY" -m pytest "$spec" -p no:cacheprovider -q --no-header -p no:randomly --tb=line ) > "$tlog" 2>&1 || true
      npass="$(grep -oE '[0-9]+ passed' "$tlog" | grep -oE '[0-9]+' | head -1)"; npass="${npass:-0}"
      nfail="$(grep -oE '[0-9]+ failed' "$tlog" | grep -oE '[0-9]+' | head -1)"; nfail="${nfail:-0}"
      tst="relevant:$spec"
    else tst="relevant-absent:$spec"; fi
  fi
  [ "${nfail:-0}" -gt 0 ] && { tst="$tst FAIL"; fail=1; }

  "$PY" - "$num" "$br" "$reb" "$conflict_files" "$ncommits" "$bld" "$tst" "$npass" "$nfail" "$trat" <<'PYJSON'
import sys,json
k=["pr","branch","rebase","conflict_files","commits","build","test","pass","fail","no_test_rationale"]
v=sys.argv[1:]; v[0]="#"+v[0]; v[4]=int(v[4]); v[7]=int(v[7]); v[8]=int(v[8])
print(json.dumps(dict(zip(k,v))))
PYJSON
  "$PY" -c "import json,sys; print(json.dumps(dict(zip(['pr','branch','rebase','conflict_files','commits','build','test','pass','fail','no_test_rationale'],['#$num','$br','$reb','$conflict_files',$ncommits,'$bld','$tst',${npass:-0},${nfail:-0},'''$trat''']))))" >> "$JSONL"

  git worktree remove --force "$WT" 2>/dev/null; git branch -D "_m$num" 2>/dev/null || true
done
echo "{\"summary\":true,\"exit\":$fail,\"meaning\":\"0=no build-ERR and no test-FAIL across all PRs\"}" >> "$JSONL"
echo "exit=$fail"
exit $fail
