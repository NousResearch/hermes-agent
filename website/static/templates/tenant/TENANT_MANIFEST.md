# Tenant Manifest

## Tenant

- Name: `<tenant-name>`
- Owner: `<person/team/client>`
- Profile: `~/.hermes/profiles/<tenant-name>/`
- Workspace: `~/.hermes/profiles/<tenant-name>/workspace/`
- Status: `active | paused | archived`

## Purpose

Describe what this tenant agent is responsible for.

## Allowed scope

List the resources this tenant user may authorize the agent to use.

- This tenant profile
- This tenant workspace
- This tenant's approved files, APIs, or services

## Forbidden scope

List resources this profile must not read or modify without operator/admin authorization.

- Default/admin profile state
- Other tenant profiles or workspaces
- Shared credentials
- Global cron jobs
- Shared skill libraries
- Other private data stores

## Credentials

- Credential owner: `<tenant/operator>`
- Stored in: `<profile .env / external secret manager>`
- Notes: Do not paste secrets into this manifest.

## Active projects

| Project | Path | Status | Notes |
|---|---|---|---|
| `<name>` | `projects/<name>/` | `active` | `<notes>` |

## Active cron jobs

See `CRON_MANIFEST.md` for full details.

| Job | Schedule | Risk | Output |
|---|---|---|---|
| `<job>` | `<schedule>` | `low/medium/high` | `<destination>` |

## Shared candidates

| Candidate | Type | Status | Notes |
|---|---|---|---|
| `<name>` | `skill/workflow/cron-template` | `draft/reviewed/promoted/rejected` | `<notes>` |

## Support policy

Support requests should use scoped support packets in `support/`. Do not attach raw memory, session transcripts, customer data, or secrets unless explicitly authorized for a narrow review.

## Offboarding notes

Document how to pause cron jobs, stop gateways, revoke credentials, export tenant-owned data, and archive this workspace.
