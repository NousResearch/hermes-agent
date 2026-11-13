# Terminal Environment Provider Plugins

Hermes runs shell commands through a pluggable set of **terminal backends**.
The built-in backends (local, Docker, Modal, Daytona, Vercel Sandbox, SSH)
live in the core repo under `hermes_agent/tools/environments/`. The official
[Singularity/Apptainer plugin](https://github.com/NousResearch/hermes-official-plugins/tree/main/singularity)
is a complete worked example of this page. Third-party
sandbox vendors integrate as **plugins** instead — a standalone plugin repo
installed under `~/.hermes/plugins/`, registering a backend the user selects
exactly like a built-in one via `terminal.backend` in `config.yaml`.

This page mirrors the [Browser Provider Plugins](./browser-provider-plugin.md)
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
| Where commands start when `terminal.cwd` is unset or unusable | `default_cwd` |
| Bot Desktop, `computer_use` and the browser following the terminal | `supports_streams` + the environment's `exec_prefix()` |
| Synced cache-file path translation | `cache_path_base` |
| Secret stripping from spawned subprocesses | `strip_env_keys` |
| Per-session sandbox isolation (`container_persistent: false`) | `session_isolated_when_nonpersistent` |

Declaring these flags on the provider closes the classic "new backend missed
classification site N" bug class — the core consults the registry at each
site instead of a hardcoded list of names.

## Minimal provider

```python title="~/.hermes/plugins/acmebox/__init__.py"
from hermes_agent.agent.terminal_env_provider import TerminalEnvironmentProvider


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
  (`local`, `docker`, `modal`, `managed_modal`, `daytona`, `vercel_sandbox`,
  `ssh`) are rejected. Plugins extend the backend set; they
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
interface as `hermes_agent.tools.environments.base.BaseEnvironment`:

- `execute(command, timeout=None, ...)` → `{"output": str, "exit_code": int}`
- `cleanup()` — release resources; called on session teardown / idle reaping
- Optional: persistence hooks mirroring the built-in cloud backends

Subclassing `BaseEnvironment` is recommended (you inherit the shared file-sync
and background-process plumbing) but not required.

## Background processes and `cleanup()`

A background command (`terminal(background=true)`) on a non-local backend is
polled through the environment object that spawned it. When Hermes tears that
environment down (`cleanup()` on session close, idle reaping, or eviction after
a connection failure), it marks those processes `lost` so their pollers stop.
It does that only after `cleanup()` returned without raising (or reported the
sandbox already gone, a 404 / "not found"), because a failed cleanup may leave
the sandbox running.

If your `cleanup()` only drops a connection and the jobs keep running where the
same environment object can still reach them (the built-in SSH backend
reconnects on the next `execute()`), declare it so the poller keeps collecting
the real exit code:

```python
class AcmeEnvironment(BaseEnvironment):
    cleanup_stops_processes = False  # cleanup() disconnects; jobs survive and stay reachable
```

The default is `True`: once `cleanup()` returns, the processes are gone for that
environment object (ended with the sandbox, or no longer reachable through it).
It also applies to duck-typed environments that don't declare the attribute. So
raise from `cleanup()` only while the sandbox may still be live; once it is
destroyed, log any later failure (snapshot bookkeeping, client shutdown) instead.

## File tools and the host filesystem

`read_file`, `write_file`, `patch` and `search_files` go through your
environment's `execute()`, so they see exactly the files your commands see.
The one exception is an environment that declares
`uses_host_filesystem = True` (only the built-in `LocalEnvironment` does): there
the file tools read, write and search the host directly, run the bundled
ripgrep, stat paths on the host, write oversized tool output straight to
`~/.hermes/cache/spillover` and enable LSP diagnostics.

A backend that subclasses `LocalEnvironment` to confine its commands (a
bubblewrap or other host sandbox that hides paths, mounts them read-only or
gives commands their own `$TMPDIR`) inherits `True` and must turn it off, or
the file tools bypass the sandbox:

```python
from hermes_agent.tools.environments.local import LocalEnvironment


class AcmeJailEnvironment(LocalEnvironment):
    uses_host_filesystem = False  # file tools run inside the jail, via execute()
```

Leave it unset on backends that don't subclass `LocalEnvironment`; the default
is `False`. Hermes v0.22.5 and earlier ignore the attribute and use the host
fast path for every `LocalEnvironment` subclass, so pin `requires_hermes` past
that.

## Images

Your backend reads its own image setting (a `config_schema` key, or a
`terminal.<name>_image` key you document). Hermes passes `image=` only for a
per-task override: a batch run (`hermes_agent/batch_runner.py`) giving each prompt its own
container registers it as `<name>_image` for the active backend, and that value
reaches `create_environment(image=...)`. Otherwise `image` is `""`; fall back to
your setting.

## Starting directory

Without `default_cwd`, a plugin backend starts in `/root`, and a host-looking
`terminal.cwd` is replaced by `/root` when `is_container` is set. Return an
absolute in-sandbox path to start somewhere else:

```python
    @property
    def default_cwd(self):
        return "/home/acme"
```

When the home directory is only known once the sandbox is up, return `"~"`
instead. Hermes then passes `cwd="~"` to `create_environment()` (it never
expands it on the host), and your environment sets `self.cwd` to the absolute
home it resolved:

```python
class AcmeBoxEnvironment(BaseEnvironment):
    def __init__(self, *, cwd, timeout, sandbox):
        if cwd == "~":
            cwd = sandbox.home()  # e.g. "/home/acme"
        super().__init__(cwd=cwd, timeout=timeout)
```

Hermes keeps `~` as the backend default, so every new sandbox resolves its
own home, and records the path each environment resolved on that environment.
Commands can also arrive with `cwd="~"`; `BaseEnvironment` already `cd`s
there correctly. A `cd` anywhere under that home is kept for later commands
and used as the base for relative paths in the file tools. Without this, a
sandbox path like `/home/acme/project` would be discarded as if it were a
host path.

`default_cwd` is read on every command, so keep it cheap (no network calls).
Relative paths are ignored, because Hermes would anchor them on the host.

## Bot Desktop and `computer_use`

The screen, `computer_use` and the browser follow the terminal into Docker and
SSH sandboxes (and the Singularity plugin's) by spawning long-lived processes there with stdin and
stdout left open. A plugin backend can opt in the same way:

1. Set `supports_streams = True` on the provider. `bot_desktop.placement: auto`
   then places the screen inside your backend instead of refusing.
2. Give the environment an `exec_prefix(*, user=None, interactive=True)` method.
   It returns a local argv that runs whatever follows it inside the sandbox,
   with stdin and stdout open (the shape of `docker exec -i <container>`), or
   `None` when it can't right now. Hermes appends `bash -c <script>` as
   separate words, so the client must pass them through unchanged and the
   sandbox needs `bash`. `user` names the account to run as; ignore it if your
   backend can't switch users. Return a list, not a single command string.

```python
class AcmeBoxEnvironment:
    def exec_prefix(self, *, user=None, interactive=True):
        return ["acmebox", "exec", "--stdin", self.sandbox_id, "--"]
```

The sandbox image still needs the desktop stack (Xvnc, Xfce, Chromium,
`cua-driver`); `nousresearch/hermes-sandbox:desktop` has it. Bot Desktop and
the browser keep their scratch files under the environment's
`get_temp_dir()`. An environment that doesn't subclass `BaseEnvironment` and
has no such method gets the same default `BaseEnvironment` uses.

A terminal backend runs only in-process: its environment object is what the
terminal tool, file tools, `exec_prefix()` and the `~` home resolution all
drive. Under `plugins.isolation: host` the object cannot cross the host
boundary, so creating the environment refuses with a message naming
`plugins.isolation: in_process` (before the host starts a sandbox), and Bot
Desktop refuses a `supports_streams` backend the same way.

## Session isolation semantics

If your sandbox is **resumed by name** (a durable VM the backend re-attaches
to), set `session_isolated_when_nonpersistent = True`. With
`terminal.container_persistent: false`, each session then gets its own
sandbox identity instead of sharing one — without this, two independent
ephemeral runs could attach one live VM and delete it out from under each
other.

## Hosted scopes

A hosted deployment runs many tenants' agents in one Hermes process, and each
agent's commands, code, background processes and files live in that agent's
own remote sandbox. A backend that provides those sandboxes sets
`serves_hosted_scopes = True`. With a hosted `hermes_scope.Scope` bound
(a tenant other than `local`), Hermes refuses every backend without it, the
built-ins included, so nothing a hosted agent runs ever runs on the Hermes host.

**Identity.** The sandbox is the agent's, not the task's or the session's:
`SandboxIdentity(scope.tenant, scope.agent)` from
`hermes_agent/tools/environments/sandbox_scope.py`, never a tool argument, task id or
profile path. All of an agent's sessions share one sandbox; two agents never
do, even with the same agent name in different tenants. `create_environment`
additionally receives `scope=` (the bound scope) and `sandbox=` (the
identity), and `task_id` is `sandbox.key`. Create or resume the sandbox that
`sandbox` names and nothing else.

**The guard.** Hermes wraps the environment you return in
`ScopedEnvironment`. Every call made through it first checks that the bound
scope names the same agent, and raises `CrossAgentHandleRefused` otherwise
(including when no scope is bound), then passes that scope to your method as
`scope=`. Your environment must implement the hosted operation interface:
`execute`, `cleanup`, `stop_idle`, `start_process`, `attach_process` and
`stop_process`, plus `fetch_file`, `fetch_realpath` and `get_temp_dir` when it
has them, each accepting `scope=` (a named parameter or `**kwargs`). An
environment that lacks one, or would drop the scope, is refused with
`HostedSandboxRequired` when it is created. The idle reaper runs with no scope
bound: its `stop_idle` (suspend: keep the workspace, resume on the next call)
and `cleanup` (destroy) calls run under the scope that created the sandbox. An
idle hosted sandbox is always suspended with `stop_idle`, never destroyed in
its place; a sandbox with a background process still running is never reaped.

**Tools.** In a hosted scope the terminal, `execute_code` (always the remote
kernel, never the host's), background processes and the file tools (read,
write, patch, search) all go through your environment. The file tools never
take a host fast path and never open, write, list, stat or dereference the
requested path on the host: their path policies (credential denylists,
protected files) judge the path's lexical form. Media the agent names (vision,
video, image-generation references) is read inside the sandbox, and an
oversized tool result is written into it, never on the host. Background
processes, and the completion, watch and heartbeat events they produce, carry
the agent's sandbox key; another agent's process is indistinguishable from a
missing one and its events stay queued for it.

**Refused in a hosted scope.** Tools with no sandbox route yet fail closed
before any setup, file or process (`hermes_agent/tools/hosted_tool_policy.py`): the
browser tools, computer use and desktop UI, image/video generation, text to
speech, skills, cron jobs and their scripts, delegation, kanban, catalog and
connection management, onboarding, and every MCP server from the profile's
config (stdio servers are never started). So are the paths outside the tool
registry that would act on the host: checkpoints and rollback, inline skill
shell, `@`-references, the workspace git probe, media delivery by host path,
connector file inputs, document extraction and voice transcription.

**Long-lived processes.** A hosted environment also implements:

```python
def start_process(self, name, argv, *, child_env=None, scope): ...  # -> ProcessStream
def attach_process(self, name, *, scope): ...                        # -> ProcessStream | None
def stop_process(self, name, *, scope): ...                          # -> bool
```

`ProcessStream` (from `sandbox_scope`) holds byte pipes `stdin` and `stdout`
and a `poll()` that returns the exit status or `None`. The stream Hermes hands
back wraps yours: every pipe operation and `poll()` checks the bound scope, so
a stream acquired by one agent is refused to any other holder. A later call in the same
session attaches with the `ProcessRef` from the first stream; a reference that
names another agent's sandbox is refused before your method runs. The built-in
backends implement these too (`hermes_agent/tools/environments/base_processes.py`: the
local backend on this host, docker/ssh and `supports_streams` plugins through
their exec prefix). Suspending a sandbox once its processes are idle-stopped,
and network isolation between sandboxes, are yours.

**Credentials** reach a hosted sandbox as broker references
(`sandbox_scope.CredentialRef`), never as values from the secret store. No
broker exists yet, so Hermes passes none.

**Conformance.** `hermes_agent/tools/environments/sandbox_conformance.py` holds the
isolation checks Hermes runs against its recording stub backend
(`tests/tools/test_sandbox_scope_conformance.py`): A's tools only address A's
sandbox; a handle naming another agent is refused; no file tool or media read
touches the host filesystem (checked with a planted host file and symlink, an
audit hook and `stat`/`lstat`/`readlink` observation); a background process
started by A, and its events, are invisible to B; an oversized result stays in
the sandbox; running work keeps a sandbox alive and an idle one is suspended
under its owner's scope; host-only tools are refused. Run the same checks
against your provider.
