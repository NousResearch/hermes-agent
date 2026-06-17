# Agent specs preview (read-only)

Hermes includes an alpha, read-only typed-agent spec preview surface for validating draft `.agent.md` files before any runtime enforcement exists.

This feature does not change active profiles, prompts, tools, MCP servers, sandboxing, memory, Kanban behavior, gateways, or deployments. It only reads supplied spec files and existing profile metadata to produce a report.

## Commands

```bash
hermes agent-spec validate <path-or-id> [--json]
hermes agent-spec preview --profile <profile-id> [--spec <path>] [--json]
hermes agent-spec list --profiles [--json]
```

Use `--strict` with `validate` or `preview` to treat warnings as failures.

## Canonical spec format

The canonical Phase 1 format is Markdown with YAML frontmatter:

```markdown
---
schema_version: hermes.agent_spec/v1alpha1
id: context-preview
profile_id: context-manager
reasoning_effort: medium
toolsets:
  enabled: [terminal, file]
sandbox:
  desired: workspace-write
  backend: local
  enforcement_status: declared_only
---
Role instructions for preview only.
```

YAML and TOML files are supported as import/example formats for the preview CLI. They are not applied to runtime profiles.

## Safety and status fields

Preview JSON always includes:

- `read_only_guarantee: true`
- `enforcement_enabled: false`
- exact MCP validation states, when MCP references exist
- exact sandbox `enforcement_status`

The local backend normally reports sandbox declarations as `declared_only`; the preview CLI must not claim enforcement unless a future reviewed backend mapping exists.

## MCP states

The preview uses these exact states:

- `known_in_catalog_and_configured`
- `known_in_catalog_but_not_configured_optional`
- `known_in_catalog_but_required_missing`
- `unknown_server_id`
- `tool_discovery_unavailable`
- `tool_not_in_catalog_or_discovery`

The preview does not start or contact MCP servers. Static catalog/config inspection only.

## Sandbox enforcement statuses

The preview uses these exact labels:

- `declared_only`
- `partially_enforced_by_backend`
- `enforced`
- `not_supported_on_backend`

A local preview claiming `enforced` fails validation in this slice.
