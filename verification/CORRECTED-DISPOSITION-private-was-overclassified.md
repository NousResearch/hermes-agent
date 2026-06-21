# Corrected disposition — "private" was over-classified (Council round-14 + user correction)

The user flagged (correctly) that I was over-classifying deferred patches as "private
data exposure." A line-by-line scan proves it. This doc replaces the earlier framing
that treated all 34 deferred patches as un-publishable.

## The TRUE sensitive surface: 5 lines, all in comments/path-strings, all scrubbable

A scan of every ADDED line across all 34 deferred patches for genuine secrets/PII
(account IDs, tokens, keys, personal filesystem paths) finds **only these 5 lines** —
none load-bearing, all in comments or test-harness path strings:

| patch | line | what it is | scrub to |
|-------|------|------------|----------|
| `copilot-limits/agent_model_metadata.py` | `# token (account <account-id-redacted>): gpt-5.4 ...` | account ID in a provenance comment | "a ChatGPT Pro account" |
| `private-feature-mixed/agent_models_dev.py` | `#   <review-workspace>/AUTHORITATIVE_LIMITS.md` | personal path in a comment | drop the path |
| `cmx/tests_test_context_engine_tool_wrap.py` | `cmx_path = "/mnt/devvm/.../hermes-cmx/...hermes_engine.py"` | personal path in a test | env var / relative |
| `private-feature-mixed/tests_..._opus_context_fix_...py` | `review at <review-workspace>` | personal path in a docstring | drop |
| `private-feature-mixed/tests_probe_prelude_e2e.py` | `WT = "<review-workspace>"` | personal worktree path in a dev-probe | env var |

**That is the entire genuine exposure.** Everything else is NOT sensitive.

## What I WRONGLY called "private" (it is just incomplete/overlay-aligned features)

- `agent/anthropic_adapter.py`: `os.environ.get("ANTHROPIC_API_KEY")` — a normal env-var
  read. Zero secret in the code. I miscounted this as a "sensitive hit."
- `hermes_cli/models.py` (700 lines), `cli.py` autopilot block, the gemini
  native/cloudcode adapters, `auxiliary_client.py`, `run_agent.py`, `hermes_state.py`,
  `tui_gateway/server.py` — **zero sensitive content**. These are residual/overlay
  lines of files whose features already ship in draft provider PRs (gemini→#50033,
  auxiliary_client→#50064/#49184, etc.), i.e. drift past the branch cut, OR
  incomplete features the user already assigned to standalone draft PRs (rules 6/7/8:
  agy-cli→#50039, source-accelerator→#50032, auto_router→#50031).

## Corrected disposition (honest)

| was-labeled | reality | correct disposition |
|-------------|---------|---------------------|
| private-overlay (11) | overlay/drift lines of files already in draft PRs; 0 secrets | fold into the owning draft provider PRs OR keep as overlay-version (drift-supersession) |
| private-overlay-phaseh (6) | incomplete phase-h feature; 0 secrets | the user's "isolate as draft PR, fix later" set (rule 8) — already the agy/inventory draft territory |
| private-feature-mixed (7) | mostly drift of files in PRs; 3 carry a personal path/account string | scrub the 3 path/account lines; rest folds into owning PRs |
| post-branch-drift (6) | drift-supersession of files already in owner PRs; 0 secrets | update the owner PR to current overlay state |
| copilot-limits (2) | account-specific caps + 1 account-ID comment | scrub the account-ID comment; the user ruled caps ship-verbatim on the right account [id=63592] |
| cmx (2) | CMX feature + 1 personal path in a test | rule 5 → single CMX PR (#50155); scrub the path |

**Net:** the only thing that genuinely cannot be published verbatim is ~5 comment/path
lines (trivially scrubbed). The rest is publishable — either already in draft PRs, or
foldable into them, or the user's own "incomplete → isolate as draft PR" set. The
earlier "private data exposure" framing was wrong and is retracted.

## Line closure still holds

After regenerating all deferred patches from current src HEAD (94cef8953, which picked
up recent overlay-align commits), line-level closure = **11125/11125 covered, 0 unmapped**.
