---
sidebar_position: 3
---

# Tenant Workspaces for Profiles

Tenant workspaces are an operator pattern for running profile-scoped agents for friends, colleagues, teams, or clients without mixing their work artifacts together.

:::info
This page describes a convention built on top of Hermes profiles. It is **not** a security sandbox. Profiles isolate Hermes runtime state; a workspace organizes tenant-created artifacts; OS users, containers, or VMs provide hard filesystem isolation.
:::

## Mental model

```text
Profile     = the agent runtime and identity
Workspace   = the tenant-owned workbench and artifact store
Sandbox     = the operating-system or container boundary
```

A profile owns Hermes state:

```text
~/.hermes/profiles/<tenant>/
  config.yaml
  .env
  SOUL.md
  memories/
  sessions/
  skills/
  cron/
  logs/
  workspace/
```

A tenant workspace owns work artifacts created for that tenant:

```text
~/.hermes/profiles/<tenant>/workspace/
  projects/
  scripts/
  skills/
  cron/
  docs/
  data/
  runtime/
  candidates/
  support/
  incidents/
  TENANT_MANIFEST.md
  CRON_MANIFEST.md
```

Using the profile-local `workspace/` directory is the simplest convention because it is created with every profile. Operators may choose an external root such as `~/agent-workspaces/<tenant>/`, but should then set that profile's `terminal.cwd` explicitly.

## Why use this pattern?

Tenant agents often create more than chat history:

- private skills and prompt workflows
- cron job scripts and manifests
- customer or project drafts
- support packets for operator review
- incident summaries
- reusable workflow candidates

Keeping those artifacts inside one tenant workspace makes it easier to review, back up, export, archive, or promote work without reading unrelated profiles.

## Recommended setup

Create a profile normally:

```bash
hermes profile create client-a --description "Client A assistant for CRM drafts and reporting."
```

Then initialize the workspace convention:

```bash
mkdir -p ~/.hermes/profiles/client-a/workspace/{projects,scripts,skills,cron,docs,data,runtime,candidates,support,incidents}
cp website/static/templates/tenant/TENANT_MANIFEST.md \
  ~/.hermes/profiles/client-a/workspace/TENANT_MANIFEST.md
cp website/static/templates/tenant/CRON_MANIFEST.md \
  ~/.hermes/profiles/client-a/workspace/CRON_MANIFEST.md
```

Set the profile's default terminal working directory:

```bash
client-a config set terminal.cwd ~/.hermes/profiles/client-a/workspace
```

Use an absolute path if you manage profiles from scripts or services:

```bash
client-a config set terminal.cwd /home/user/.hermes/profiles/client-a/workspace
```

## Tenant governance block

For tenant-style profiles, add a short governance block to that profile's `SOUL.md`:

```markdown
## Tenant Governance

You are an isolated tenant profile.

Runtime state belongs to this profile. Work artifacts belong in this tenant workspace.

You may operate within this tenant's sandbox when the tenant user authorizes it.

You must not represent the default/admin agent, other tenants, or shared/core systems.

You must not read or modify other profiles, other tenant workspaces, default profile state, shared credentials, global cron jobs, or shared skill libraries unless explicitly authorized by the operator/admin.

Private skills, cron jobs, workflows, and prompts are private by default.

Reusable artifacts may be nominated as candidates, but candidate output must be sanitized and must not include private memory, session content, API keys, customer data, chat IDs, or secrets.

Shared promotion requires review.
```

This block is guidance for the model. It does not enforce filesystem access. Use a stronger terminal backend, OS user, container, or VM when tenants are untrusted.

## Private by default, shared by review

Tenant-created skills and workflows should start private:

```text
private skill
→ sanitized candidate bundle
→ operator review
→ shared skill or shared template
```

Do not automatically copy tenant skills into a global or shared skill library. Ask the tenant agent to produce a candidate card instead. A starter template is available at `website/static/templates/tenant/SKILL_CANDIDATE_CARD.md` in the repository and at `/templates/tenant/SKILL_CANDIDATE_CARD.md` on the built site.

A safe candidate card should summarize:

- skill name
- one-sentence purpose
- intended users
- dependencies
- risk level
- whether private data was removed
- what should and should not be shared

It should not include raw memory, session transcripts, customer data, credentials, private file paths, or chat IDs.

## Support without opening the whole profile

When a tenant needs help, prefer a scoped support packet over full profile access. A starter template is available at `website/static/templates/tenant/SUPPORT_PACKET.md` in the repository and at `/templates/tenant/SUPPORT_PACKET.md` on the built site.

A support packet should include:

- issue summary
- affected profile
- affected job, skill, or script
- sanitized error messages
- expected behavior
- explicit review scope
- explicit exclusions

This lets an operator help with a failing cron job or tool without reading the tenant's memory, sessions, or private skills by default.

## Cron jobs

Tenant cron jobs should be documented in a `CRON_MANIFEST.md` before they become important automation. A starter template is available at `website/static/templates/tenant/CRON_MANIFEST.md` in the repository and at `/templates/tenant/CRON_MANIFEST.md` on the built site.

At minimum, record:

- job name and purpose
- schedule
- data sources
- delivery target
- risk level
- whether it writes data or sends external messages
- restore or rollback steps
- verification command

Cron jobs should be quiet when nothing changed and should avoid sending noisy messages to shared channels.

## Incident packets

If a tenant bot sends the wrong message, writes bad data, leaks private context, or starts a noisy cron loop, stop the impact first and then create an incident packet. A starter template is available at `website/static/templates/tenant/INCIDENT_PACKET.md` in the repository and at `/templates/tenant/INCIDENT_PACKET.md` on the built site.

The incident packet should preserve enough detail to debug while excluding secrets and raw private data unless the tenant explicitly authorizes a narrower review.

## Relationship to profile isolation

Profiles already give each agent its own Hermes home, but profiles are not sandboxes. On the local terminal backend, the process still has the host user's filesystem permissions.

Tenant workspaces therefore provide organization and operator hygiene, not a hard security boundary.

Use this pattern when tenants are trusted but should not accidentally mix artifacts. Use OS users, containers, VMs, SSH backends, or other sandboxing mechanisms when tenants are untrusted or data isolation must be enforced outside model instructions.

## Related proposal

This pattern is being discussed in [#35947: Add tenant workspace template and governance scaffolding for profiles](https://github.com/NousResearch/hermes-agent/issues/35947).
