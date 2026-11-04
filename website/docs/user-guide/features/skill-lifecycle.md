# Skill lifecycle policy

Hermes separates profile-local skills into `skills/active/<slug>/` (live) and `skills/proposed/<slug>/` (draft). Automatic/background `skill_manage create` calls always require a creating session, `risk`, and `why` (`need`, `tools`, `example_run`) and land in proposed. A direct operator migration/bootstrap without an agent session retains manual creation semantics; this exception is explicit and is not available to background dispatch.

A draft is visible only when its creating session explicitly calls `skill_view`. Loading it restricts that session at the tool-registry dispatch boundary to read-only retrieval tools; write, send, and secret tools fail closed. Drafts never appear in catalogs, auto-load, slash commands, or cron preloads.

## Bootstrap and inventory

Run `bootstrap_layout()` before enabling lifecycle discovery. It inventories/moves every existing top-level approved package into `active`, fails before moving anything on a name collision, creates `proposed`, and writes `.lifecycle-enabled.json` last. Skill discovery switches to active-only **only when that marker exists**; merely creating an empty directory cannot hide legacy skills. `freeze_inventory()` hashes every file in every package, not only `SKILL.md`.

## Scheduled operation

`python -m tools.skill_lifecycle install` idempotently installs three profile-local, no-agent cron jobs:

- nightly at `0 1 * * *`, `America/Chicago`;
- Booth digest at `30 7 * * *`, delivered through the cron job's normal `origin` delivery lane;
- monthly 60-day archive review at `0 8 1 * *`, also delivered normally.

The nightly command uses the existing authenticated `gh` CLI and fails closed if authentication is unavailable. It creates one dated branch and one batch PR containing per-skill decision, deterministic scan, and executable parse/example-eval evidence. Rejected items remain represented as rewrite/delete decisions in that lineage and are not silently removed from proposed.

Read-only passing items are eligible for automation only against the exact batch head after required checks report success. `write`, `send`, and `credentials` require an `APPROVE-SKILL: <name>` line from Booth in the batch PR and cannot be approved by a reviewer whose identity created the proposal or whose capabilities conflict with the requested widening. The implementation PR is explicitly excluded from batch automation. No command approves or merges PRs by itself.

The deterministic scanner detects hidden Unicode, URLs/webhooks, slash commands, secret paths, and send/write calls. It only flags base64 when a syntactically valid candidate decodes to binary or credential/network indicators; prose that mentions “base64” is allowed.
