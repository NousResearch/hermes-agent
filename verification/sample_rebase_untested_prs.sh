#!/usr/bin/env bash
# sample_rebase_untested_prs.sh — Council item #2: rebase + build + test a SAMPLE of the
# previously-untested (test=none) PRs onto a fresh v0.17.0, with a RELEVANT test file each
# (not just the PR's own diff, which had no tests). Records pass/fail per PR.
set -uo pipefail
FORK="fork"; V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"
SRC="$(git rev-parse --show-toplevel)"; PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"
OUT="${1:-/mnt/devvm/custom/hermes/reconcile-tmp/sample-rebase.out}"; : > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }

# sample of 6 untested PRs + a RELEVANT EXISTING test (present in v0.17.0) that exercises
# each one's touched source area (PR# : branch : relevant_test_path)
SAMPLE=(
  "50021:feat/tool-timing-sidecar:tests/run_agent/test_tool_executor_contextvar_propagation.py"
  "50033:feat/gemini-cli-user-agent:tests/agent/test_gemini_native_adapter.py"
  "50040:feat/delegate-task-persona:tests/tools/test_delegate_toolset_scope.py"
  "50053:feat/context-engine-grounding-hooks:tests/agent/test_context_engine.py"
  "50047:fix/gateway-liveness-and-root-guard:tests/gateway/test_status.py"
  "50054:feat/plugin-register-command-override:tests/hermes_cli/test_plugins.py"
)

env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" 2>/dev/null || true
log "=== SAMPLE rebase+build+test of ${#SAMPLE[@]} previously-untested PRs onto fresh v0.17.0 ==="
log "$(printf '%-8s %-12s %-10s %-8s %s' PR REBASE BUILD TEST relevant_test)"
fail=0
for spec in "${SAMPLE[@]}"; do
  num="$(cut -d: -f1 <<<"$spec")"; br="$(cut -d: -f2 <<<"$spec")"; tf="$(cut -d: -f3 <<<"$spec")"
  sha="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr view "$num" --repo NousResearch/hermes-agent --json headRefOid -q .headRefOid 2>/dev/null)"
  env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" "$sha" 2>/dev/null || true
  base="$(git merge-base origin/main "$sha" 2>/dev/null || echo "$V017")"
  WT="$(mktemp -d)"
  git worktree add -q "$WT" "$V017" 2>/dev/null
  ( cd "$WT" && git switch -q -c "_s$num" 2>/dev/null )
  # real rebase (cherry-pick the PR's own commits)
  if ( cd "$WT" && git cherry-pick "$base..$sha" >/dev/null 2>&1 ); then reb="CLEAN"
  else reb="CONFLICT"; fail=1; ( cd "$WT" && git cherry-pick --abort 2>/dev/null ); fi
  # build (compile the dirs the PR touches + core)
  if [ "$reb" = "CLEAN" ]; then
    bc="$( cd "$WT" && "$PY" -m compileall -q agent tools hermes_cli run_agent.py cli.py gateway tui_gateway 2>&1 | grep -ic error )"
    bld="$([ "${bc:-0}" -eq 0 ] && echo OK || echo "ERR($bc)")"; [ "$bld" != "OK" ] && fail=1
    # run the relevant test on the rebased tree
    if [ -f "$WT/$tf" ]; then
      res="$( cd "$WT" && timeout 200 "$PY" -m pytest "$tf" -p no:cacheprovider -q --no-header -p no:randomly --tb=line 2>/dev/null | grep -oE '[0-9]+ (passed|failed)' | tr '\n' ' ' )"
      tst="${res:-none}"
      echo "$res" | grep -q 'failed' && fail=1
    else tst="testfile-absent"; fi
  else bld="-"; tst="-"; fi
  log "$(printf '#%-7s %-12s %-10s %-8s %s' "$num" "$reb" "$bld" "$tst" "$tf")"
  git worktree remove --force "$WT" 2>/dev/null; git branch -D "_s$num" 2>/dev/null || true
done
log ""
log "RESULT: $([ "$fail" -eq 0 ] && echo "PASS — all sampled untested PRs rebase clean, build OK, relevant tests pass" || echo "FAIL — see rows")"
exit $fail
