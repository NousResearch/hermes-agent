# v0.17.0 forward-port patches — ready to `git apply`

These 6 patches make the 6 forward-port-conflict PRs **independently pullable onto v0.17.0**
(`2bd1977d8fad185c9b4be47884f7e87f1add0ce3`) without touching their main-targeted branches.

Each patch = that PR's full content **with its documented conflict resolution already baked in**.
Apply on a clean v0.17.0 checkout:

```
git checkout 2bd1977d8fad185c9b4be47884f7e87f1add0ce3
git apply v017-patches/PR-<n>-onto-v0.17.0.patch
```

## Verified (apply-clean + tests pass, from a FRESH v0.17.0 worktree)

| patch | resolution baked in | applies on v0.17.0 | tests |
|---|---|---|---|
| `PR-49644-onto-v0.17.0.patch` | take-theirs (superset `/reasoning` subcommands) | CLEAN | 55 passed |
| `PR-49916-onto-v0.17.0.patch` | take-theirs (YOLO-badge fix removes the `or approval_mode=="off"`) | CLEAN | 279 passed |
| `PR-50056-onto-v0.17.0.patch` | combine (keep both `import sqlite3` + `import subprocess`) | CLEAN | 218 passed |
| `PR-50064-onto-v0.17.0.patch` | ours-drop (drop the stale `…_default_headers` test the canonical tree replaced) | CLEAN | 13 passed |
| `PR-50073-onto-v0.17.0.patch` | combine-config (keep PR's compression keys, preserve v0.17.0's `hygiene_hard_message_limit:400`) | CLEAN | 9 passed |
| `PR-50296-onto-v0.17.0.patch` | take-theirs (pure addition of `_end_session_on_close`/`_persist_disabled`) | CLEAN | code-only |

All 6: **0 residual conflict markers**, **APPLIES-CLEAN on fresh v0.17.0**, tests green.

## Why patches in the manifest, not pushes to the PR branches

The 6 PR branches **target `main`**, where they are already conflict-free (`MERGEABLE`, not
`DIRTY`). Pushing a v0.17.0-specific resolution onto a main-targeted branch would (a) be the
wrong base — it would corrupt the PR against `main` — and (b) churn the maintainers' review
queue with v0.17.0 artifacts that don't belong on a `main` PR. The forward-port resolution is
a *pull-down-time* operation, so it lives here as a ready-to-apply artifact: the PR stays clean
for `main`, and the operator applies the matching patch when pulling onto v0.17.0.
