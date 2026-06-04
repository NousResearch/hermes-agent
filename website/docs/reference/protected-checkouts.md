# Protected checkouts

`protected_checkouts` is the executable registry used by Hermes file and terminal guards to block accidental mutation of canonical repository checkouts.

The registry lives in `config.yaml` and is read by `tools/protected_checkout_policy.py`; docs and skills are only pointers to that executable guard.

```yaml
protected_checkouts:
  canonical_roots:
    - /home/ubuntu/repos/dailychingu
  allowed_worktree_prefixes:
    - /home/ubuntu/.hermes/hermes-agent/.worktrees
```

- `canonical_roots`: checkouts that must not be mutated directly unless the guard can prove the target is on an approved task-worktree branch.
- `allowed_worktree_prefixes`: task-owned worktree roots where mutation is allowed.

The guard is intentionally fail-closed for protected roots when branch lookup fails. If this registry is absent or malformed, Hermes falls back to the built-in DailyChingu canonical root and Hermes task-worktree prefix instead of disabling the guard globally.

See also:

- `tools/protected_checkout_policy.py`
- `tests/tools/test_protected_checkout_policy.py`
- `tests/tools/test_file_tools_protected_checkout_guard.py`
- `tests/tools/test_terminal_protected_checkout_guard.py`
