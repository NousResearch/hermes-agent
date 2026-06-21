# Build + test verification on a fresh v0.17.0 stack (Council round-15)

The Council correctly escalated from `git apply --check` to **actual build+test** on the
stacked tree. This round ran it, and it caught a real mistake (#50457) plus confirmed
zero campaign-introduced regressions.

## Method

`fresh_stack_v017.sh`: clean-clone upstream, branch off v0.17.0 (`2bd1977d8`), net-apply
all 41 feature PRs (37 direct + 3 via forward-compat + #50457), then byte-compile and run
the test suites with the hermes venv.

## Result 1 — #50457 was a MISTAKE; closed

Running the suite on the stack exposed that the opus-context integration test (PR #50457)
**fails 25 tests on the public stack** — its Phase B (agy-cli) and fable assertions call
`get_model_info('agy-cli', ...)` / fable lookups that return `None`, because the agy-cli +
fable rows in `agent/models_dev.py _PROBE_VERIFIED_OVERRIDES` are account-specific/private
and deliberately NOT in any public PR. On the full tree (with private data) all 64 pass; on
the public stack 25 fail. So it **cannot be a green public PR** — it correctly belongs in the
deferred private set (#50111). **#50457 was closed** with that explanation. This is exactly
the class of failure `git apply --check` cannot catch, and why the Council's stronger bar was
right.

## Result 2 — the public stack builds + tests green (0 campaign regressions)

- **Byte-compile:** `compileall agent/ hermes_cli/ tools/ run_agent.py` → exit 0.
- **Campaign PR test files:** `test_system_prompt_prelude` + `test_p2_p3_oversized_handling`
  + `test_mcp_keepalive_inflight_race` + `test_context_engine_tool_wrap` → **37 passed**.
- **Broader `tests/agent/` regression sweep:** **753 passed, 16 skipped, 1 failed.**

### The 1 failure is upstream drift, NOT ours (proven)

`tests/agent/test_auxiliary_named_custom_providers.py::...test_main_resolves_to_named_custom`:
- **identical** between v0.16.0 and our HEAD (we did not change it)
- **present on v0.16.0**, **absent on v0.17.0** (upstream deleted/moved it between releases)
- **not in our 139-file campaign delta**, **not owned by any campaign PR**

It rides into the stack only because a PR branch's origin/main lineage still carries it; its
failure is a pure v0.16↔v0.17 upstream-drift artifact in a file the campaign neither owns nor
modifies. Campaign-introduced regressions: **0**.

## Carried forward (re-verified this round)

- 42→**41 open PRs** after closing #50457 (40 feature + #50111 tracker); all OPEN.
- **0 sensitive strings** across all 41 feature PR head SHAs (scanned each tree).
- **41/41 feature PRs** `git apply --check` CLEAN on v0.17.0 (40 + #50457-now-closed; the
  remaining 40 + 3-via-forward-compat hold).
- 1 deferred-only file again (the opus-context test, back in #50111 where its private deps
  live) — honestly the correct home, not a standalone PR.
