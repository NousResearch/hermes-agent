---
sidebar_position: 4
title: "Terminal Backend Plugins"
description: "Build a plugin that runs terminal, file, and code tools in a custom environment"
---

# Building a Terminal Backend Plugin

A terminal backend plugin adds an execution environment to Hermes without adding a new model-facing tool. Once selected, the backend powers the existing `terminal`, file, and `execute_code` tools through the same host-owned lifecycle used by built-in backends.

Use this interface for integrations such as a remote workspace service, a hosted development environment, or a custom sandbox. Third-party product integrations should ship as standalone plugin repositories rather than as new directories in the Hermes core tree.

## How it fits together

A plugin registers a declarative `BackendDefinition`. Hermes then owns discovery, profile isolation, selection, configuration handoff, environment caching, and cleanup:

```text
plugin register(ctx)
  -> ctx.register_terminal_backend(BackendDefinition(...))
  -> profile-scoped terminal backend registry
  -> EnvironmentManager resolves config and checks availability
  -> factory(BackendFactoryRequest)
  -> BaseEnvironment instance
  -> terminal, file, and execute_code tools
```

The definition is also the source of truth for Dashboard and Desktop configuration. Selector options, labels, descriptions, setup status, and backend-specific fields are derived from the active profile's registry; frontend code does not maintain a parallel list of backend names.

## Discovery and activation

Terminal backends use the normal native plugin discovery paths. They may be installed directly under a plugin root or grouped one level deep in a category directory:

1. User directory: `~/.hermes/plugins/<name>/`
2. User category directory: `~/.hermes/plugins/<category>/<name>/`
3. Project directory: `./.hermes/plugins/<name>/` or `./.hermes/plugins/<category>/<name>/` when project plugins are enabled
4. Pip package: the `hermes_agent.plugins` entry-point group

Third-party plugins are opt-in. Install and enable the plugin before selecting its backend:

```bash
hermes plugins install owner/hermes-plugin-remote-workspace --enable
hermes config set terminal.backend remote_workspace
```

Plugin discovery and backend registries are profile-scoped. Enable and configure the plugin in every profile that should use it.

## Directory structure

A directory-installed plugin must keep `__init__.py` beside `plugin.yaml`, because native discovery imports the plugin directory itself. The example below works either directly below a plugin root or below one category directory:

```text
hermes-plugin-remote-workspace/
├── plugin.yaml
├── __init__.py             # register(ctx)
└── environment.py          # BaseEnvironment implementation
```

A pip-distributed plugin may instead use a normal import package such as `hermes_plugin_remote_workspace/`. Its `hermes_agent.plugins` entry point must resolve to the **module** that exposes `register(ctx)`, not directly to the `register` callable:

```toml
[project.entry-points."hermes_agent.plugins"]
hermes-plugin-remote-workspace = "hermes_plugin_remote_workspace"
```

Hermes loads that module and then looks up its `register` attribute. In this shape, package metadata and the entry point replace native directory discovery; do not copy the nested package layout into a directory-discovered plugin without placing `__init__.py` beside `plugin.yaml`.

`plugin.yaml` uses the ordinary native manifest. A backend does not need to declare a built-in-tool override:

```yaml
name: hermes-plugin-remote-workspace
version: 1.0.0
description: Run Hermes commands in a remote workspace
kind: backend
```

## Implement the environment

Every factory must return a subclass of `tools.environments.BaseEnvironment`. The base class provides the public `execute()` flow, including shell-state snapshots, working-directory tracking, timeout handling, interrupts, and output collection. A backend supplies the transport-specific process handle and cleanup behavior:

```python
from tools.environments import BaseEnvironment


class RemoteWorkspaceEnvironment(BaseEnvironment):
    def __init__(
        self,
        *,
        workspace: str,
        url: str,
        token: str,
        cwd: str,
        timeout: int,
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self.workspace = workspace
        self.url = url
        self.token = token

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        # Start the command through your transport and return a ProcessHandle-
        # compatible object. See tools.environments.base.ProcessHandle.
        return start_remote_workspace_process(
            workspace=self.workspace,
            url=self.url,
            token=self.token,
            command=cmd_string,
            login=login,
            timeout=timeout,
            stdin_data=stdin_data,
        )

    def cleanup(self) -> None:
        # Release connections or ephemeral resources. Keep this idempotent.
        close_remote_workspace_connection(self.workspace)
```

`_run_bash()` may return `subprocess.Popen` or another object satisfying the public `ProcessHandle` protocol. That protocol exposes `poll()`, `kill()`, `wait()`, `stdout`, and `returncode`; it does not expose a separate `stderr` stream. If your SDK only offers a blocking call, use `ThreadedProcessHandle` from `tools.environments` to adapt combined output, exit status, and optional cancellation.

Infrastructure failures should raise `EnvironmentConnectionError`, not masquerade as command exit failures. Hermes can then report a degraded backend and retry construction later instead of caching a broken environment.

## Declare and register the backend

Import plugin-facing contracts from `tools.environments`, then register one `BackendDefinition` from `register(ctx)`:

```python
import os

from tools.environments import (
    BackendCapabilities,
    BackendDefinition,
    BackendFactoryRequest,
    ExecutionLocation,
    FilesystemSemantics,
)

from .environment import RemoteWorkspaceEnvironment


def resolve_config(raw):
    # raw is a defensive copy of terminal.backends.remote_workspace for this profile.
    resolved = {
        "workspace": raw.get("workspace", ""),
        "url": raw.get("url", ""),
        "token": raw.get("token", ""),
    }
    # Schema `env` is presentation metadata, not automatic resolution.
    if os.getenv("REMOTE_WORKSPACE_TOKEN") is not None:
        resolved["token"] = os.environ["REMOTE_WORKSPACE_TOKEN"]
    return resolved


def config_is_available(config):
    return bool(
        config.get("workspace")
        and config.get("url")
        and config.get("token")
    )


def create_environment(request: BackendFactoryRequest):
    config = request.backend_config
    return RemoteWorkspaceEnvironment(
        workspace=config["workspace"],
        url=config["url"],
        token=config["token"],
        cwd=request.cwd,
        timeout=request.timeout,
    )


def register(ctx):
    ctx.register_terminal_backend(
        BackendDefinition(
            name="remote_workspace",
            label="Remote Workspace",
            description="Run commands in a remote workspace",
            factory=create_environment,
            default_cwd="~",
            capabilities=BackendCapabilities(
                execution_location=ExecutionLocation.REMOTE,
                filesystem_semantics=FilesystemSemantics.ISOLATED,
                requires_sandbox_cwd=True,
                supports_background_processes=True,
                supports_file_transfer=True,
                supports_persistence=True,
            ),
            config_schema={
                "workspace": {
                    "type": "string",
                    "description": "Remote workspace name",
                    "required": True,
                },
                "url": {
                    "type": "string",
                    "description": "Workspace service URL",
                    "required": True,
                },
                "token": {
                    "type": "secret",
                    "description": "Workspace access token",
                    "env": "REMOTE_WORKSPACE_TOKEN",
                    "required": True,
                },
            },
            config_resolver=resolve_config,
            config_availability_check=config_is_available,
            install_hint="Configure a workspace and access token.",
        )
    )
```

Backend names must match `^[a-z][a-z0-9_-]*$`. Built-in and legacy privileged names are reserved; a third-party plugin cannot replace `local`, `docker`, `ssh`, or another built-in backend by registering the same name.

Hermes stamps the plugin owner and source onto the registered copy. Registration is rolled back if plugin loading later fails, and unload removes only the definition owned by that registration.

## BackendDefinition reference

| Field | Purpose |
|---|---|
| `name` | Stable config value used by `terminal.backend` and the registry. |
| `factory` | Receives a resolved `BackendFactoryRequest`; must return `BaseEnvironment`. |
| `label`, `description` | User-facing metadata for Dashboard/Desktop backend pickers. |
| `capabilities` | Declared backend traits. Core currently consumes execution location for classification/integrity checks and filesystem/sandbox-cwd traits for path routing; the remaining fields are descriptive metadata for compatibility and future consumers. Never a security attestation or approval bypass. |
| `availability_check` | Optional no-argument passive dependency check. No installs or remote resource creation. |
| `config_schema` | Backend-local fields shown by Dashboard/Desktop. Schema retrieval must be side-effect free. |
| `config_resolver` | Converts raw profile-local config into complete runtime config. |
| `config_availability_check` | Passive readiness check over the resolved config. |
| `default_cwd` | Backend-side initial directory. Isolated plugin backends should normally use `~`. |
| `install_hint` | Safe remediation shown when the backend needs setup. Never include secret values. |
| `diagnostic_metadata` | JSON-safe diagnostic metadata for host-owned reporting. |

`source` and `plugin_name` are host-owned attribution fields. Do not rely on values supplied by plugin code; `ctx.register_terminal_backend()` replaces them.

## Capabilities

Capabilities describe backend behavior. Core currently uses `filesystem_semantics` and `requires_sandbox_cwd` when routing host and sandbox paths. It uses `execution_location` to classify remote execution in prompts and to reject a non-local definition whose factory returns `LocalEnvironment`. `accepts_host_cwd` and the `supports_*` fields are registry metadata: declare them accurately for diagnostics, compatibility checks, and future consumers, but do not assume they currently enable or disable a tool feature.

| Field | Typical meaning |
|---|---|
| `execution_location` | `LOCAL`, `REMOTE`, or `UNKNOWN`. |
| `filesystem_semantics` | `HOST`, `SHARED`, `ISOLATED`, or `UNKNOWN`. |
| `accepts_host_cwd` | Backend can meaningfully receive the host working directory. |
| `requires_sandbox_cwd` | Host must avoid forwarding arbitrary host paths. |
| `supports_image` | Factory accepts an image selection. |
| `supports_resource_limits` | Backend supports CPU/memory/disk controls. |
| `supports_pty` | Backend can provide interactive PTY execution. |
| `supports_background_processes` | Background process lifecycle is supported. |
| `supports_file_transfer` | File tools can transfer data through this environment. |
| `supports_persistence` | Backend can preserve filesystem state across recreation. |

Declare only behavior your implementation actually supports. The path-related declarations influence current routing; support flags do not themselves grant functionality. Hermes does not treat any capability as proof of isolation or grant security exemptions from it.

## Configuration schema and namespace

Plugin fields are namespaced automatically:

```text
config_schema field `workspace`
    -> terminal.backends.remote_workspace.workspace
```

A user profile therefore stores plugin configuration like this:

```yaml
terminal:
  backend: remote_workspace
  backends:
    remote_workspace:
      workspace: development
      url: https://workspace.example.com
```

Supported schema field types are `boolean`, `list`, `number`, `secret`, `select`, `string`, and `text`. A `select` field must provide `options` as a list of strings. Metadata must be JSON-safe. Unsafe path segments such as `__proto__`, `constructor`, and `prototype` are rejected.

Do not use `config_key` to escape your namespace. Only canonical built-ins may project fields onto legacy `terminal.<field>` paths. This preserves existing built-in configuration while preventing plugins from claiming core-owned keys.

The optional `env` value in a field descriptor tells setup surfaces where a value may come from; it does **not** read that variable. Implement the same environment lookup explicitly in `config_resolver`. Do not return or log resolved configuration, because it may contain credentials.

## Runtime configuration handoff

Hermes extracts only the selected profile's mapping:

```text
terminal.backends.<definition.name>
```

It deep-copies that raw mapping and passes it to `config_resolver`. The resolver returns a mapping; Hermes deep-copies the result, uses it for config-aware availability, and places the same resolved values in `BackendFactoryRequest.backend_config` for the factory.

Keep these responsibilities separate:

- `request.backend_config`: resolved plugin-owned settings;
- `request.terminal_config`: host-owned complete terminal configuration, primarily for built-in compatibility;
- `request.cwd`, `timeout`, `image`, `task_id`, and `host_cwd`: host-owned request context. Callers may leave optional context fields unset; in particular, do not assume a profile name or Hermes home is present.

Choose and document precedence in your resolver. A common policy is defaults, then profile YAML, then an environment override for credentials. Environment presence should be checked with `is not None`; an explicitly empty credential should fail readiness instead of silently falling back. Factories should validate the resolved mapping and must not reread YAML or environment variables.

## Availability rules

Availability checks are passive and may run while painting settings or listing backends:

- do not create a workspace or container;
- do not install packages;
- do not emit credentials or raw exception text;
- prefer local checks and validated config;
- leave resource creation to the factory.

If `config_availability_check` is present, Hermes passes the resolved config to it. Otherwise Hermes uses the no-argument `availability_check`. The factory remains the final fail-closed validation boundary.

## Testing checklist

At minimum, test the plugin against the intended Hermes core revision:

- [ ] directory and pip entry-point discovery both call `register(ctx)`;
- [ ] the registered name appears in the backend picker and schema options;
- [ ] schema fields appear under `terminal.backends.<name>.*`;
- [ ] `secret` fields remain secret fields and no resolved value reaches diagnostics;
- [ ] defaults, profile values, environment overrides, empty values, and malformed values resolve as documented;
- [ ] availability and factory receive equivalent resolved config;
- [ ] the factory receives the expected cwd, host cwd, timeout, task id, image, terminal config, and resolved backend config for each exercised entry point;
- [ ] terminal, file, `execute_code`, and prompt-probe first use all construct the same backend;
- [ ] the environment implements command success, nonzero exit, timeout, stdin, cancellation, output bounds, file operations, background execution, and idempotent cleanup as applicable;
- [ ] registration failure and plugin unload leave no stale backend definition;
- [ ] two profiles can configure the same backend independently.

Use `hermes plugins doctor . --ci` for the native plugin registration path, then run the plugin's own contract and transport tests. A real one-command smoke test is still required before claiming the remote integration works.

## See also

- [Build a Hermes Plugin](/developer-guide/plugins) — manifests, discovery, installation, and distribution
- [Configuration: Terminal Backend Configuration](/user-guide/configuration#terminal-backend-configuration) — selecting and configuring a backend
- [Plugins](/user-guide/features/plugins) — user-facing activation and backend capability overview
