# Hermes builder-dispatch adapter

`hermes.builder_dispatch.v1` is a proposed, non-operational Hermes-owned
interface for governed implementation dispatch. It uses HTTP over a local
mode-0600 Unix-domain socket and exposes only dispatch, status, cancellation,
and health operations.

The adapter is not activated by installation. Operational activation requires
separate deterministic validation, independent review, architectural review,
and explicit approval.

The interface authenticates every non-health request using Darwin Unix peer
credentials plus a caller-specific HMAC key, timestamp, and durable nonce.
Keys are supplied by OS-protected service configuration and never belong in
the repository, command line, logs, task text, or model context.

Before creating a native Kanban task, the adapter verifies the exact contract
object, repository, linked worktree, branch, HEAD, allowed-path manifest,
registered validation profile, and effective `deepseek` /
`deepseek-v4-pro` route with an empty fallback chain. Unknown side-effect state
blocks redispatch.

The adapter does not expose Kanban records, SQLite, Hermes configuration,
credentials, arbitrary tools, shell commands, GitHub operations, approval, or
merge authority. Orchestrator workflow state remains authoritative.

## Inert service entry point

The supported service command is:

```bash
python -m plugins.builder_adapter serve --config /owner-only/runtime.json
```

Importing or installing the plugin does not run that command, bind a socket,
create a task, or read authentication material. The runtime accepts only a
Unix-domain socket path; it has no TCP listener implementation. The runtime
configuration must be a regular, non-linked, owner-only file. Its separate
authentication file must be mode `0600`, and secret values are resolved only
from environment names beginning with
`HERMES_BUILDER_ADAPTER_SECRET_`.

The `deepseek-builder` profile must explicitly enable the `builder_adapter`
plugin and set the CLI toolsets to exactly `builder_adapter` and `no_mcp`.
Its model configuration must select provider `deepseek`, model
`deepseek-v4-pro`, and an empty `fallback_providers` list. Adapter-created
Kanban workers receive only `kanban_complete`, `kanban_block`, and
`kanban_heartbeat` in addition to the five governed builder tools. Ordinary
Kanban workers keep their existing tool behavior.

Cancellation is fail-closed. Until native Kanban exposes structured proof
that the complete worker process tree was terminated, cancellation returns
`CANCELLATION_UNCONFIRMED`, leaves the dispatch blocked and non-terminal, and
does not call reclaim or archive.
