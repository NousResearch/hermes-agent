---
sidebar_position: 13
title: "Terminal Backend Plugins"
description: "Build a standalone execution backend for Hermes terminal, file, and code tools"
---

# Building a Terminal Backend Plugin

A terminal backend controls where Hermes runs shell commands. Hermes also routes file tools and remote `execute_code` calls through the same environment.

Third-party product backends must ship as standalone plugin packages. Do not add product SDKs or product-specific setup code to Hermes core.

## Contract

Your plugin registers one immutable `TerminalBackendDefinition`:

```python
from tools.environments.registry import TerminalBackendDefinition

from .backend import ExampleEnvironment


def register(ctx):
    ctx.register_terminal_backend(
        TerminalBackendDefinition(
            name="example",
            factory=ExampleEnvironment.from_request,
            container_paths=True,
            default_cwd="/workspace",
            default_image="example/python:latest",
            image_override_key="example_image",
            image_config_key="image",
        )
    )
```

The definition is frozen. Hermes keeps a host-owned copy and rejects built-in names or cross-plugin name collisions. Registration is transactional: if `register()` fails or the discovery sweep aborts, Hermes rolls that owner's backend names back to their pre-load state.

| Field | Meaning |
|---|---|
| `name` | Value selected by `terminal.backend` or `TERMINAL_ENV` |
| `factory` | Callable that accepts `TerminalBackendRequest` and returns an environment |
| `container_paths` | `true` when paths belong to the backend, not the Hermes host |
| `default_cwd` | Absolute backend working directory, or `~` |
| `default_image` | Image used when no override exists |
| `image_override_key` | Optional per-task image key for isolated benchmark tasks |
| `image_config_key` | Optional key under the plugin's `terminal_backend` settings |

`TerminalBackendRequest.settings` is an independent, recursively read-only snapshot of `plugins.entries.<plugin-id>.terminal_backend`. Nested sequences are exposed as tuples.

## Environment implementation

Return an object that implements `execute()` and `cleanup()`. Subclass `BaseEnvironment` to reuse Hermes command wrapping, working-directory tracking, environment snapshots, stdin handling, interruption, and timeout enforcement.

```python
from tools.environments.base import BaseEnvironment, ThreadedProcessHandle


class ExampleEnvironment(BaseEnvironment):
    _stdin_mode = "pipe"

    @classmethod
    def from_request(cls, request):
        return cls(cwd=request.cwd, timeout=request.timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        def execute():
            # Call the backend SDK. Return (combined_output, exit_code).
            return self.client.exec(cmd_string, stdin=stdin_data, timeout=timeout)

        def cancel():
            self.client.cancel_active_process()

        return ThreadedProcessHandle(execute, cancel_fn=cancel)

    def cleanup(self):
        self.client.delete_environment()
```

The factory must return a new environment. Hermes owns the cache and serializes creation for each task. Terminal, file, and code tools reuse that cached object.

## Lifecycle rules

Implement these behaviors:

1. Create one uniquely named resource.
2. Wait for a real readiness probe.
3. Enforce each command timeout.
4. Cancel the exact active process.
5. Preserve nonzero exit codes and useful error text.
6. Make cleanup idempotent.
7. Kill tracked processes before resource deletion.
8. Apply a provider TTL as a secondary leak limit.
9. Delete a partly created resource when initialization fails.

Do not give backend plugins access to Hermes lifecycle caches or locks.

## Package entry point

Declare the standard Hermes entry point:

```toml
[project.entry-points."hermes_agent.plugins"]
example = "hermes_terminal_backend_example"
```

Users enable the package and select the backend:

```yaml
plugins:
  enabled:
    - example
  entries:
    example:
      terminal_backend:
        image: example/python:latest
terminal:
  backend: example
```

Store SDK credentials in `.env`, the process environment, or the provider's native credential store. Do not put secrets in `config.yaml`.

## Tests

Test registration through the real pip entry point. Then test terminal, file, and code routes against one enabled fixture plugin.

Cover successful commands, nonzero exits, authentication errors, readiness failure, cancellation, command timeout, partial creation, repeated cleanup, and concurrent creation.

For a live test, require an explicit non-production account. Use a unique resource name, a short TTL, and a final provider lookup that proves deletion.

## Security boundary

A terminal backend confines terminal, file, and remote code operations. It does not isolate the Hermes process, model providers, browser tools, or other plugins.

Use whole-process isolation when you need a boundary for every Hermes capability.
