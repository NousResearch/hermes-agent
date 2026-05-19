# Codex Economy Mode

Codex economy mode reduces OpenAI Codex token pressure during agent sessions without sacrificing final quality.

## Why it exists

When using `openai-codex` or `xai-oauth` providers (which route through OpenAI Codex), heavy context loading — reading whole files, sending large raw diffs — can exhaust token budgets quickly. Economy mode injects system-prompt guidance that tells the agent to package context more compactly, while keeping all verification and final-review steps intact.

## What it does

Economy mode adds guidance to the stable system prompt tier instructing the agent to:

- Run file-discovery or diff-stats **before** loading files, and prefer offsets/snippets over whole-file reads.
- Build **compact review packets** (goal/spec, changed-file list, diff summary, relevant snippets, test-output tail, explicit questions) instead of dumping raw diffs.
- **Preserve parallel execution** — tasks that are independent still run in parallel; economy mode never serializes work just to save tokens.
- **Never skip** verification, final tests, security checks, or Codex Final whole-system review.

Economy mode reduces **context packaging**, not quality gates.

## Enabling

### Always on

```bash
hermes config set codex_economy.enabled true
```

### Auto-detect for Codex providers

Automatically activates when the active provider is `openai-codex` or `xai-oauth`, or the API mode is `codex_responses`:

```bash
hermes config set codex_economy.auto_for_openai_codex true
```

## Configuration reference

All keys live under `codex_economy` in `~/.hermes/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Always inject economy guidance |
| `auto_for_openai_codex` | `false` | Inject when provider/api_mode indicates Codex |
| `max_changed_files_for_inline_context` | `8` | Threshold above which the agent switches to snippets-only context (informational — the agent reads this as intent) |
| `max_diff_lines_for_review_packet` | `400` | Max diff lines to include in a review packet |

## What is NOT changed

- Final verification steps are **required**, not optional.
- Codex Final whole-system review is **still performed** when relevant.
- Parallel execution for independent subtasks is **preserved**.
- No new API dependencies are introduced.
