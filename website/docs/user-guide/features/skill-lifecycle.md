# Skill lifecycle policy

Hermes separates the canonical source repository from the installed profile. Set `HERMES_SKILLS_REPO` to the canonical Git checkout. Automatic/background `skill_manage create` requires a creating session, `risk`, and `why` (`need`, `tools`, `example_run`, and a declarative `eval`) and writes only to `<repo>/skills/proposed/<slug>/`. `<repo>/skills/active/` is the sole lifecycle-approved source for later deployment. This workflow never writes, moves, or unfreezes the installed `~/.hermes/skills` tree.

A draft is visible only when its creating session explicitly calls `skill_view`. Loading it restricts that session at the tool-registry dispatch boundary to read-only retrieval tools; write, send, and secret tools fail closed. Drafts never appear in catalogs, auto-load, slash commands, or cron preloads.

## Bootstrap and inventory

`bootstrap_lifecycle()` is the supported install/upgrade hook. Gateway scheduler startup invokes it automatically when `HERMES_SKILLS_REPO` is configured; job creation is idempotent and uses the normal cron store APIs. Profile migration is opt-in and disabled by that hook. The legacy `bootstrap_layout()` migration remains available only for an explicit profile migration: it preserves existing packages, fails before moving anything on collision, and writes `.lifecycle-enabled.json` last. Bundled, hub, project, create-dir, and external-dir precedence is unchanged.

## Scheduled operation

The bootstrap installs four profile-local, no-agent cron jobs:

- nightly at `0 1 * * *`, `America/Chicago`;
- exact-head batch reconciliation every 15 minutes;
- Booth digest at `30 7 * * *`, delivered through the cron job's normal `origin` delivery lane;
- monthly 60-day archive review at `0 8 1 * *`, also delivered normally.

The nightly command uses the existing authenticated `gh` CLI and fails closed if authentication is unavailable. It creates one dated branch and one batch PR containing the actual source transitions and evidence: passing candidates move from `skills/proposed` to `skills/active`, waiting-risk candidates remain proposed, and rejected candidates receive a deterministic rewrite marker or are deleted with reasons in the evidence lineage.

The separate reconciler merges read-only-only batches only when required checks pass on the exact expected head. `write`, `send`, and `credentials` require an `APPROVE-SKILL: <name>` line authored by a GitHub login in the explicit `HERMES_BOOTH_GITHUB_LOGINS` allowlist; arbitrary matching text is ignored. Review agents cannot approve their own access widening. PR #148971 is hard-excluded from reconciliation and remains open.

The deterministic scanner detects hidden Unicode, URLs/webhooks, slash commands, secret paths, and send/write calls. `eval.json` executes a fixture-backed `read_file` assertion inside a fixed read-only allowlist and records the fixture hash and pass/fail result; write/send/secret tool names are denied. It only flags base64 when a syntactically valid candidate decodes to binary or credential/network indicators.
