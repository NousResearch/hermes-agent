# Deferred-by-design files held in #50111 — enumeration + runtime-relevance (Council item #4)

#50111 (`deferred/residual-lines-on-v0.17.0`, OPEN draft, NOT FOR MERGE) holds 34 deferred
`.patch` files. None are required for runtime behavior on v0.17.0 via the 40 feature PRs;
each is a private/account-specific/CMX-coupled artifact intentionally NOT in a public feature
PR. The structural coverage proof counts their lines as `deferred-in-#50111`, and the
hunk-attribution table maps the corresponding src hunks to `#50111-deferred`.

## Categories (34 patches)

### private-overlay (11) — the v2026.6.5 update-merge machinery
Private update-overlay code (agy-cli/auto_router/source-accelerator/impersonation infra).
NOT contributable; the user isolated these explicitly. Runtime: these are the private
overlay's OWN features, not part of the public src that the 40 PRs reconstruct.

### private-overlay-phaseh (6) — phase-h EMPIRICAL_MERGE_MATRIX modifications
Copilot-context test-file mods + inventory/skills edits introduced by the private
v2026.6.5 phase-h merge. Runtime: private-overlay-specific; not public behavior.

### private-feature-mixed (7) — feature code entangled with private infra
Files where the contributable change is woven with private agy/cmx/autopilot lines
(agent_init/system_prompt/models_dev/api_server/probe_prelude/copilot-opus-context-test).
The contributable PORTIONS already shipped in feature PRs; the residual private-entangled
lines are deferred. Runtime: the public behavior is delivered by the feature PRs; the
deferred lines are private-infra glue.

### cmx (2) — CMX context-engine coupling
`agent_conversation_loop.py` + `tests_test_context_engine_tool_wrap.py` portions that
reference the private CMX engine (`/mnt/.../context-engine/...`). Runtime: CMX is a private
context engine; the public context-engine hooks shipped in #50053/#50080/#50155. The
deferred lines are CMX-specific test/glue, not required for v0.17.0 runtime.

### copilot-limits (2) — account-specific empirical caps
`agent_model_metadata.py` + `tests_agent_test_model_metadata.py` portions carrying the
account-specific gpt-5.4 ~900K empirical cap. The PUBLIC limits shipped in #49449/#50064
(272K/400K). Runtime: the public limit values drive v0.17.0; the 900K cap is one account's
measurement, deliberately not shipped publicly.

### post-branch-drift (6) — overlay newer than the PR branch-cut
Files where src drifted AFTER the owning PR was cut (mcp_tool, system_prompt_prelude,
gateway_run, test_model_switch_copilot_api_mode, the refusal-test + model_metadata drift
patches). Runtime: the owning PR delivers the behavior; these patches preserve the exact
overlay-state lines so nothing is lost, but they are supersets/drift, not new runtime
requirements.

## Runtime-relevance verdict
**None of the 34 deferred files is required for runtime behavior delivered by the 40
feature PRs on v0.17.0.** They are: (a) private overlay features (agy/cmx/auto_router —
the user's own non-contributable infra), (b) account-specific values (900K cap), or
(c) private-infra-entangled glue whose public portions already shipped in feature PRs.
The deferral decision: keep them pullable in #50111 so nothing is lost, but they are NOT
needed to reconstruct the public-contributable src on v0.17.0.

## Verification cross-refs (in #50111:verification/)
- `structural-coverage-table.txt`: 138/138 COVERED (deferred lines counted as #50111).
- `HUNK-ATTRIBUTION-TABLE.txt`: 442 hunks, 0 unmapped (deferred hunks → `#50111-deferred`).
- `diff-equivalence.out`: 0 residual (union(PRs)+#50111 ≡ src delta).
