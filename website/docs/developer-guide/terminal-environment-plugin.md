# Terminal Environment Provider Plugins

Hermes runs shell commands through a pluggable set of **terminal backends**.
The built-in backends (local, Docker, Singularity, Modal, Daytona, Vercel
Sandbox, SSH) live in the core repo under `tools/environments/`. Third-party
sandbox vendors integrate as **plugins** instead — a standalone plugin repo
installed under `~/.hermes/plugins/`, registering a backend the user selects
exactly like a built-in one via `terminal.backend` in `config.yaml`.

This page mirrors the [Browser Provider Plugins](/developer-guide/browser-provider-plugin)
guide — same registration flow, same scope semantics.

## What a provider controls

A registered backend automatically participates in every core surface:

| Surface | Driven by |
|---|---|
| Command dispatch (`terminal`, `execute_code`, file tools) | `create_environment()` |
| `hermes setup` backend picker | `display_name`, `description`, `setup_instructions()`, `post_setup()` |
| Dashboard terminal-backend picker (probe status) | `probe()` / `probe_with_config()` |
| Dashboard/Desktop provider config fields | `get_config_schema()` |
| `hermes status` / `hermes doctor` | `doctor_checks()` |
| System-prompt environment hints | `is_remote`, `env_description` |
| Dangerous-command approval skipping | `skip_container_guards` |
| Container path/cwd handling | `is_container` |
| Synced cache-file path translation | `cache_path_base` |
| Secret stripping from spawned subprocesses | `strip_env_keys` |
| Per-session sandbox isolation (`container_persistent: false`) | `session_isolated_when_nonpersistent` |

Declaring these flags on the provider closes the classic "new backend missed
classification site N" bug class — the core consults the registry at each
site instead of a hardcoded list of names.

## Minimal provider

```python title="~/.hermes/plugins/acmebox/__init__.py"
from agent.terminal_env_provider import TerminalEnvironmentProvider


class AcmeBoxEnvironment:
    """Must satisfy the BaseEnvironment duck-typed contract."""

    def __init__(self, cwd, timeout, task_id):
        self.cwd, self.timeout, self.task_id = cwd, timeout, task_id

    def execute(self, command, timeout=None, **kwargs):
        ...  # run the command in the sandbox
        return {"output": "...", "exit_code": 0}

    def cleanup(self):
        ...  # tear down / detach


class AcmeBoxProvider(TerminalEnvironmentProvider):
    name = "acmebox"
    display_name = "AcmeBox"
    is_remote = True          # commands don't run on the host
    is_container = True       # container-style path/cwd semantics

    @property
    def description(self):
        return "Run commands in an AcmeBox cloud sandbox."

    @property
    def cache_path_base(self):
        return "~/.hermes"    # where synced cache files land, or None

    @property
    def strip_env_keys(self):
        return frozenset({"ACMEBOX_TOKEN"})

    def is_available(self):
        import importlib.util, os
        return (
            importlib.util.find_spec("acmebox") is not None
            and bool(os.getenv("ACMEBOX_TOKEN"))
        )

    def create_environment(self, *, cwd, timeout, task_id="default",
                           image=None, container_config=None, **kwargs):
        return AcmeBoxEnvironment(cwd, timeout, task_id)


def register(ctx):
    ctx.register_terminal_environment_provider(AcmeBoxProvider())
```

```yaml title="~/.hermes/plugins/acmebox/plugin.yaml"
name: acmebox
version: 0.1.0
description: AcmeBox cloud sandbox terminal backend
kind: backend
```

Enable it, select it, run:

```bash
hermes plugins enable acmebox
hermes config set terminal.backend acmebox
```

## Provider-owned configuration

A provider can expose backend-specific fields without adding them to Hermes core.
The schema is projected into the Dashboard and Desktop config forms under
`terminal.backends.<provider>`; only the selected backend's fields are shown.

```python
class AcmeBoxProvider(TerminalEnvironmentProvider):
    # ... identity, classification, and availability methods ...

    def get_config_schema(self):
        return {
            "workspace": {
                "type": "string",
                "description": "AcmeBox workspace name",
            },
            "token": {
                "type": "secret",
                "description": "AcmeBox API token",
            },
        }

    def resolve_config(self, config):
        # The provider owns defaults and environment/secret fallback precedence.
        return {
            "workspace": config.get("workspace", "default"),
            "token": os.getenv("ACMEBOX_TOKEN", config.get("token", "")),
        }

    def create_environment(self, *, backend_config=None, **kwargs):
        client = AcmeBoxClient(
            workspace=backend_config["workspace"],
            token=backend_config["token"],
        )
        return AcmeBoxSdkEnvironment(client=client, **kwargs)
```

```yaml title="~/.hermes/config.yaml"
terminal:
  backend: acmebox
  backends:
    acmebox:
      workspace: development
```

Supported schema types are `string`, `text`, `secret`, `number`, `boolean`,
`list`, and `select`. Secret fields render as password inputs and must not
include value-bearing metadata such as `default`, `value`, `example`, or
`options`. Schema keys are backend-local dotted paths; unsafe prototype
segments are rejected. Core gives `resolve_config()` a defensive snapshot and
passes a second defensive snapshot to `create_environment()` as
`backend_config`.

## Rules

- **Reserved names.** Registrations that collide with a built-in backend name
  (`local`, `docker`, `singularity`, `modal`, `managed_modal`, `daytona`,
  `vercel_sandbox`, `ssh`) are rejected. Plugins extend the backend set; they
  never shadow in-tree backends.
- **`create_environment` must accept `**kwargs`** and ignore unknown keys —
  the forward-compat contract that lets the factory signature evolve without
  breaking older plugins.
- **`is_available()` / `probe()` must be cheap.** No network calls — they run
  during requirement checks and UI paints.
- **Fail-soft everywhere.** A provider attribute that raises is treated as
  its default by the core (e.g. a raising `skip_container_guards` keeps the
  approval layer ON). Don't rely on exceptions for control flow.
- **Secrets belong in `strip_env_keys`.** Your vendor token must never be
  readable by a model-authored shell command; listing it strips it from every
  spawned subprocess unconditionally, like the built-in `MODAL_*` /
  `DAYTONA_API_KEY` handling.

## Environment object contract

`create_environment()` returns an object satisfying the same duck-typed
interface as `tools.environments.base.BaseEnvironment`:

- `execute(command, timeout=None, ...)` → `{"output": str, "exit_code": int}`
- `cleanup()` — release resources; called on session teardown / idle reaping
- Optional: persistence hooks mirroring the built-in cloud backends

Subclassing `BaseEnvironment` is recommended (you inherit the shared file-sync
and background-process plumbing) but not required.

## Session isolation semantics

If your sandbox is **resumed by name** (a durable VM the backend re-attaches
to), set `session_isolated_when_nonpersistent = True`. With
`terminal.container_persistent: false`, each session then gets its own
sandbox identity instead of sharing one — without this, two independent
ephemeral runs could attach one live VM and delete it out from under each
other.
