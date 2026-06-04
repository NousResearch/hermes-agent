# BO-235 protected-checkout hardblock proof package

This is the review evidence package for BO-235, covering the protected-checkout guard slices BO-236 through BO-240.

## Surfaces covered

- Policy module: `tools/protected_checkout_policy.py`
- File tools: `tools/file_tools.py`
- Terminal approval guard: `tools/approval.py`
- Terminal invocation wiring: `tools/terminal_tool.py`
- Config/docs pointer: `website/docs/reference/protected-checkouts.md`

## Local integrated smoke

Run from the integrated review worktree:

```bash
python -m pytest \
  tests/tools/test_protected_checkout_policy.py \
  tests/tools/test_file_tools_protected_checkout_guard.py \
  tests/tools/test_terminal_protected_checkout_guard.py -q

git diff --check
python -m py_compile \
  tools/protected_checkout_policy.py \
  tools/file_tools.py \
  tools/approval.py \
  tools/terminal_tool.py
```

Observed result during BO-240 closeout:

```text
78 passed
```

## DailyChingu canonical non-destructive probe

The proof intentionally does **not** mutate `/home/ubuntu/repos/dailychingu`.
It checks decisions and terminal guard output, then compares git diff/status before and after.

Probe shape:

```bash
DC=/home/ubuntu/repos/dailychingu
BEFORE=$(git -C "$DC" diff --stat && git -C "$DC" status --short)
PYTHONPATH="$PWD" python - <<'PY'
from tools.protected_checkout_policy import check_path_mutation, effective_protected_checkout_registry
from tools.approval import _check_protected_cwd_command
root = '/home/ubuntu/repos/dailychingu'
print('registry', effective_protected_checkout_registry())
print(check_path_mutation(root))
print(check_path_mutation(root + '/package.json'))
assert _check_protected_cwd_command('touch should-not-create.txt', root)['status'] == 'blocked'
assert _check_protected_cwd_command('git status --short', root) is None
PY
AFTER=$(git -C "$DC" diff --stat && git -C "$DC" status --short)
test "$BEFORE" = "$AFTER"
```

Observed result during BO-240 closeout:

```text
path_decision /home/ubuntu/repos/dailychingu False BLOCKED_PROTECTED_CANONICAL
path_decision /home/ubuntu/repos/dailychingu/package.json False BLOCKED_PROTECTED_CANONICAL
terminal_touch_blocked True
terminal_git_status_allowed True
DailyChingu canonical diff/status unchanged
```

## Runtime / deployment boundary

This package proves the local review branch behavior only.

Not performed:

- no merge
- no release/deploy
- no gateway restart/reload
- no live runtime apply
- no env/secret/customer/provider mutation
- no DailyChingu canonical mutation

Until a later merge/landing/restart is explicitly approved and verified, live Hermes runtime should be reported as `runtime_applied=false`.
