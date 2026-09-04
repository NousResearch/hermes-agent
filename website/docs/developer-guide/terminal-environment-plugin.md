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
| Dashboard terminal-backend picker (probe status) | `probe()` |
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
