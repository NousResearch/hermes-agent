---
sidebar_position: 16
title: "LSP — Semantic Diagnostics"
description: "Real language servers (pyright, gopls, rust-analyzer, …) wired into the post-write lint check used by write_file and patch."
---

# Language Server Protocol (LSP)

Hermes runs full language servers — pyright, gopls, rust-analyzer,
typescript-language-server, clangd, and ~20 more — as background
subprocesses and feeds their semantic diagnostics into the post-write
lint check used by `write_file` and `patch`. When the agent edits a
file, it sees exactly the errors that edit introduced — not just
syntax errors, but **type errors, undefined names, missing imports,
and project-wide semantic issues** the language server detects.

This is the same architecture top-tier coding agents use. Hermes
ships it self-contained: no editor host required, no plugins to
install, no separate daemon to manage.

## When LSP runs

LSP is gated on **git workspace detection**. When the agent's working
directory (or the file being edited) is inside a git repository, LSP
runs against that workspace. When neither is in a git repo, LSP
stays dormant — useful for messaging gateways where the cwd is the
user's home directory and there's no project to diagnose.

The check is layered: in-process syntax check first (microseconds),
then LSP diagnostics second when syntax is clean. A flaky or missing
language server can never break a write — every LSP failure path
falls back silently to the syntax-only result.

Concretely, on every successful `write_file` or `patch`:

1. Hermes captures a baseline of current diagnostics for the file.
2. Performs the write.
3. Re-queries the language server, filters out diagnostics that were
   already in the baseline, and surfaces only the new ones.

The agent sees output like:

```
{
  "bytes_written": 42,
  "dirs_created": false,
  "lint": {"status": "ok", "output": ""},
  "lsp_diagnostics": "LSP diagnostics introduced by this edit:\n<diagnostics file=\"/path/to/foo.py\">\nERROR [42:5] Cannot find name 'foo' [reportUndefinedVariable] (Pyright)\nERROR [50:1] Argument of type \"str\" is not assignable to \"int\" [reportArgumentType] (Pyright)\n</diagnostics>"
}
```

The `lint` field carries the syntax-check result (microsecond
in-process parse via `ast.parse`, `json.loads`, etc.); the
`lsp_diagnostics` field carries the semantic diagnostics from the
real language server. Two channels, independent signals — the
agent sees a syntax-clean file with semantic problems as
``lint: ok`` plus a populated ``lsp_diagnostics``.

## Supported languages

| Language | Server | Auto-install |
|----------|--------|--------------|
| Python | `pyright-langserver` | npm |
| TypeScript / JavaScript / JSX / TSX | `typescript-language-server` | npm |
| Vue | `@vue/language-server` | npm |
| Svelte | `svelte-language-server` | npm |
| Astro | `@astrojs/language-server` | npm |
| Go | `gopls` | `go install` |
| Rust | `rust-analyzer` | manual (rustup) |
| C / C++ | `clangd` | manual (LLVM) |
| Bash / Zsh | `bash-language-server` | npm |
| YAML | `yaml-language-server` | npm |
| Lua | `lua-language-server` | manual (GitHub releases) |
| PHP | `intelephense` | npm |
| Laravel Blade (`.blade.php`) | `laravel-lsp` | manual (composer) |
| OCaml | `ocaml-lsp` | manual (opam) |
| Dockerfile | `dockerfile-language-server-nodejs` | npm |
| Terraform | `terraform-ls` | manual |
| Dart | `dart language-server` | manual (dart sdk) |
| Haskell | `haskell-language-server` | manual (ghcup) |
| Julia | `julia` + LanguageServer.jl | manual |
| Clojure | `clojure-lsp` | manual |
| Nix | `nixd` | manual |
| Zig | `zls` | manual |
| Gleam | `gleam lsp` | manual (gleam install) |
| Elixir | `elixir-ls` | manual |
| Prisma | `prisma language-server` | manual |
| Kotlin | `kotlin-language-server` | manual |
| Java | `jdtls` | manual |
| PowerShell | `PowerShellEditorServices` (`pwsh` host) | manual (release zip) |

For "manual" entries, install the server through whatever toolchain
manager makes sense for that language (rustup, ghcup, opam, brew,
…). Hermes auto-detects the binary on PATH or in
`<HERMES_HOME>/lsp/bin/`.

### PowerShell

PowerShellEditorServices isn't a single binary — it's a PowerShell
module bundle launched by a `pwsh` (PowerShell 7+) or `powershell`
host. Setup:

1. Install [PowerShell](https://github.com/PowerShell/PowerShell) so
   `pwsh` (or Windows `powershell`) is on PATH.
2. Download the latest release zip from
   [PowerShellEditorServices releases](https://github.com/PowerShell/PowerShellEditorServices/releases)
   and extract it.
3. Point Hermes at the extracted bundle — the directory that contains
   `PowerShellEditorServices/Start-EditorServices.ps1`. Either:
   - set `lsp.servers.powershell.command: ["/path/to/bundle"]` in
     `config.yaml`, or
   - extract it to `<HERMES_HOME>/lsp/PowerShellEditorServices`, or
   - export `PSES_BUNDLE_PATH=/path/to/bundle`.

`hermes lsp status` reports `installed` once `pwsh` is found; if the
bundle is missing you'll see a one-time warning in the logs with the
download link.

### Laravel Blade

`.blade.php` templates go to [laravel-lsp](https://github.com/laravel/lsp)
(Blade, Eloquent, Facades) while plain `.php` files stay with
intelephense. Install it once with Composer and make sure the binary
is on PATH (or pin it with `lsp.servers.laravel-lsp.command`):

```bash
composer global require laravel/lsp
export PATH="$HOME/.config/composer/vendor/bin:$PATH"
```

Hermes launches it as `laravel-lsp lsp` (stdio). There is no
auto-install recipe; `hermes lsp status` shows `manual-only` until the
binary is found.

A few servers are installed alongside a peer dependency that npm
won't auto-pull. `typescript-language-server` and `@vue/language-server`
require the `typescript` SDK importable from the same `node_modules`
tree — Hermes installs `typescript@6` (the last JavaScript-based line;
TypeScript 7 is the Go port and ships no `tsserver.js`) together with
the server when you run `hermes lsp install typescript` /
`hermes lsp install vue-language-server` or auto-install fires on first use.

Vue is pinned to `@vue/language-server@2`, started with
`vue.hybridMode: false` so it hosts its own TypeScript service. The 3.x
line only works behind a client-hosted `tsserver` tunnel (the VS Code /
Neovim setup) that Hermes's generic client does not run, so it never
publishes diagnostics. If an earlier Hermes installed 3.x, the log shows a
one-time `vue-language-server: ... 3.x` warning; delete
`<HERMES_HOME>/lsp/node_modules/@vue` and `<HERMES_HOME>/lsp/bin/vue-language-server*`,
then run `hermes lsp install vue-language-server` (the recipe co-installs the
TypeScript SDK).

## CLI

```
hermes lsp status          # service state + per-server install status
hermes lsp list            # registry, optionally --installed-only
hermes lsp install <id>    # eagerly install one server
hermes lsp install-all     # try every server with a known recipe
hermes lsp restart         # tear down running clients
hermes lsp which <id>      # print resolved binary path
```

`hermes lsp status` is the best starting point — it shows which
languages will get semantic diagnostics today and which need a
binary installed.

## Configuration

The defaults work for typical setups; nothing to set if the binaries
are on PATH.

```yaml
# config.yaml
lsp:
  # Master toggle. Disabling skips the entire subsystem — no servers
  # spawn, no background event loop runs.
  enabled: true

  # How long to wait for diagnostics after each write.
  wait_mode: document      # "document" or "full"
  # Max seconds to wait for the server on each of the two waits an
  # edit makes: the pre-edit baseline snapshot and the post-edit
  # re-check. Only *fresh* diagnostics (produced for the post-edit
  # content) are ever reported; if the server doesn't finish within
  # this budget, the edit reports "no LSP data" rather than stale
  # errors from before the edit. Raise this for slow servers on big
  # projects (tsserver, rust-analyzer mid-indexing).
  wait_timeout: 5.0

  # Budget for the FIRST request against a workspace whose server is
  # not running yet — spawn, initialize and the server's initial program
  # build all happen inside it (tsserver on a 10k-file project can need
  # a minute). Once the client is up, wait_timeout applies again.
  # 0 = no extra grace (same as wait_timeout).
  warmup_timeout: 0

  # After a server fails for a workspace (spawn error, or the request
  # outran its budget) that (server, root) pair is skipped. 0 = for the
  # rest of the process (until `hermes lsp restart`); N = retried after
  # N seconds, so one transient stall does not silence a workspace
  # forever. Skips are logged once per root at INFO with the retry time.
  broken_retry_seconds: 0

  # Workspace roots where no language server runs at all — glob
  # patterns matched against the resolved project root (~ expanded; a
  # bare path also matches everything beneath it). Use it for the one
  # huge monorepo whose server cannot finish in budget while every other
  # workspace keeps its diagnostics — unlike servers.<id>.disabled,
  # which switches the server off everywhere. Must be a list; any other
  # shape logs a warning and skips LSP for every workspace until fixed.
  exclude_roots: []
  # exclude_roots: ["~/work/huge-monorepo", "/srv/checkouts/*/vendor"]

  # How to handle missing server binaries.
  #   auto    — install via npm/pip/go install into <HERMES_HOME>/lsp/bin
  #   manual  — only use binaries already on PATH
  install_strategy: auto

  # Node package manager for the npm-based servers: npm (default), pnpm
  # or yarn. Installs still land in <HERMES_HOME>/lsp/node_modules; a
  # manager that is configured but not installed — or a value outside
  # npm|pnpm|yarn — skips the install with a warning instead of silently
  # using npm, so a pnpm/yarn supply-chain policy (minimumReleaseAge,
  # allowBuilds, …) is never bypassed. Yarn Berry (2+): its default PnP
  # linker writes no node_modules/.bin, so set `nodeLinker: node-modules`
  # in <HERMES_HOME>/lsp/.yarnrc.yml. pnpm 11 blocks git-hosted transitive
  # deps by default (ERR_PNPM_EXOTIC_SUBDEP); @vue/language-server 2.x pulls
  # one in, so under pnpm that server is skipped with the pnpm error in the
  # log — install it once with npm, or relax block-exotic-subdeps in
  # <HERMES_HOME>/lsp/.npmrc if your policy allows it.
  package_manager: npm

  # How long an unused language-server client stays alive (seconds).
  # Idle servers are shut down automatically and respawned on the next
  # relevant file operation. Set to 0 to disable idle reaping and keep
  # servers alive for the life of the process. Values below 30s are
  # clamped to 30 so a sweep can never reap a client mid-operation.
  idle_timeout: 600

  # V8 heap cap (MB) for spawned Node-based language servers
  # (tsserver, pyright, vue, …). At spawn Hermes sets
  # `--max-old-space-size=<max_heap_mb>` in the child's NODE_OPTIONS,
  # REPLACING any inherited value, so the gateway's own heap flag never
  # leaks into server children and a server on a large workspace hits a
  # known ceiling instead of Node's default (~4 GiB) while sharing the
  # host. Raise it for big monorepos where tsserver needs more room;
  # set 0 to leave the inherited environment untouched (servers then
  # inherit the parent's NODE_OPTIONS, or Node's default when unset).
  # Non-Node servers (gopls, rust-analyzer, clangd, …) ignore it.
  # Verify a running server: `hermes lsp status` reports `max_heap_mb`;
  # inside the process, `v8.getHeapStatistics().heap_size_limit`.
  max_heap_mb: 2048

  # cgroup v2 isolation for spawned servers (Linux). Each server lands in
  # its own cgroup (`<cgroup root>/hermes-lsp`) with an independent
  # `memory.max` (`cgroup_memory_mb`), so a large-workspace server can
  # grow to that ceiling without pushing the gateway's cgroup into the
  # kernel OOM killer. Needs a writable cgroup v2 mount (root, or a
  # delegated systemd user scope). When unavailable — non-Linux, cgroup
  # v1, or a read-only container mount such as Docker's default
  # `/sys/fs/cgroup` — attach fails gracefully: servers share the
  # gateway's cgroup (the pre-isolation behaviour) and one warning is
  # logged. Set `cgroup_isolate: false` to skip the attempt entirely;
  # `cgroup_memory_mb: 0` keeps the split but sets no memory cap.
  # Verify: `hermes lsp status` shows `isolated=true` per client; or
  # compare `cat /proc/<gateway-pid>/cgroup` with `cat /proc/<server-pid>/cgroup`.
  cgroup_isolate: true
  cgroup_memory_mb: 4096

  # Per-server overrides (all optional).
  servers:
    pyright:
      disabled: false
      command: ["/abs/path/to/pyright-langserver", "--stdio"]
      env: { PYRIGHT_LOG_LEVEL: "info" }
      initialization_options:
        python:
          analysis:
            typeCheckingMode: "strict"
    typescript:
      disabled: true       # skip TS even when its extensions match
```

### Per-server keys

* `disabled: true` — skip this server entirely even when its
  extensions match a file.
* `command: [bin, ...args]` — pin a custom binary path. Bypasses
  auto-install.
* `env: {KEY: value}` — extra env vars passed to the spawned process.
* `initialization_options: {...}` — merged into the LSP
  `initializationOptions` payload sent in the `initialize`
  handshake. Server-specific; consult the language server's docs.

### Custom servers

Any `lsp.servers` key that is **not** a built-in server id declares
your own language server. It needs `command` and `extensions`; the
other keys are optional. Custom servers are matched *before* the
built-ins, so they can also take over an extension Hermes already
handles.

```yaml
lsp:
  servers:
    panache:
      command: ["panache-lsp", "--stdio"]   # PATH lookup or an absolute/~ path
      extensions: [".pnch"]                 # or basenames like "Justfile"
      root_markers: ["panache.toml"]        # nearest dir with one of these; default: workspace root
      language_id: "panache"                # didOpen languageId; default: derived from the extension
      description: "Panache markdown"
      env: { PANACHE_LOG: "warn" }          # same optional keys as built-ins
      initialization_options: {}
```

Custom servers are never auto-installed: put the binary on PATH (or
give an absolute path) and `hermes lsp status` lists it as
`installed`. A malformed entry is logged and skipped without
affecting the other servers.

## Installation locations

When `install_strategy: auto`, Hermes installs binaries into
`<HERMES_HOME>/lsp/bin/`. NPM packages land in
`<HERMES_HOME>/lsp/node_modules/` with bin symlinks one level up.
Go binaries come from `go install` with `GOBIN` pointed at the
staging dir.

Nothing is ever installed to `/usr/local/`, `~/.local/`, or any other
shared location — the staging dir is fully Hermes-owned and is
removed when you reset the profile.

## Performance characteristics

LSP servers are **lazy-spawned** on first use. Editing a Python file
in a project that's never seen `.py` traffic spawns pyright; the
spawn takes 1-3 seconds for most servers (rust-analyzer can take 10+
on a cold project). Subsequent edits in the same workspace re-use
the running server.

The LSP layer adds a few milliseconds to clean writes when no
diagnostics are emitted. When diagnostics are emitted, the wait
budget is `wait_timeout` seconds — typically the server responds in
tens of milliseconds for pyright/tsserver and a few seconds for
rust-analyzer mid-indexing. Each edit waits twice (a pre-edit
baseline snapshot for the delta, then the post-edit re-check), so a
server that never answers costs at most `2 × wait_timeout` per edit.
The very first request against a workspace also pays the spawn and the
server's initial program build; give large projects room with
`lsp.warmup_timeout` (only that first, cold request uses it — the
steady-state budget is unchanged) rather than raising `wait_timeout`,
which would let every later edit block for the cold-build duration.

A server that fails for a workspace — spawn error, or a request that
outran its budget — marks that `(server, root)` pair broken and every
later request for it is skipped (logged once per root at INFO). By
default the pair stays broken until `hermes lsp restart` or process
exit; `lsp.broken_retry_seconds: N` retries it after N seconds so one
transient stall does not cost the workspace its diagnostics for good.
A root you never want served — one monorepo whose server cannot finish
in any budget — goes in `lsp.exclude_roots`; other workspaces keep
their servers.

Diagnostics are **freshness-gated**: a result only counts when the
server produced it for the content of the current edit (a
`publishDiagnostics` push at/after the change, or a pull request
answered after it). Slow servers that haven't re-checked yet result
in "no data" for that edit — never in yesterday's errors being
re-reported as current.

Servers are kept alive while they're being used and shut down after
`lsp.idle_timeout` seconds (default 600) with no file activity — a
long-running gateway that touches many worktrees no longer accumulates
one language-server process per workspace forever. A reaped server is
respawned automatically on the next relevant file operation. Set
`idle_timeout: 0` to disable reaping and hold every server's index warm
for the life of the process.

Servers are also released when their workspace goes away, even if they
are not idle: removing a Hermes-managed worktree (`hermes -w` session
end, Kanban task cleanup, a delegated subagent's pruned worktree) shuts
down that tree's language servers before `git worktree remove` runs, and
the periodic sweep shuts down any server whose project root no longer
exists on disk (deleted outside Hermes). The sweep is part of the idle
reaper, so `idle_timeout: 0` also disables deleted-root reaping; the
worktree-removal release always runs. A multi-root server only drops the
vanished folder and keeps serving its sibling roots.

Servers that support multi-root workspaces (currently pyright) run as a
**single process** per Hermes process: the first Python project spawns
it, and every further project root — for example sibling git worktrees
edited by parallel subagents — is attached to that same server as an
additional workspace folder instead of starting another copy.

## Disabling

Set `lsp.enabled: false` in `config.yaml` to disable the entire
subsystem. The post-write check falls back to the in-process syntax
check (`ast.parse` for Python, `json.loads` for JSON, etc.) which
ships unchanged from earlier versions.

To disable a single language without disabling the whole layer:

```yaml
lsp:
  servers:
    rust-analyzer:
      disabled: true
```

## Troubleshooting

**`hermes lsp status` shows a server as "missing"**

The binary isn't on PATH and isn't in `<HERMES_HOME>/lsp/bin/`. Run
`hermes lsp install <server_id>` to attempt an auto-install, or
install the binary manually through the language's normal toolchain.

**`Backend warnings` section in `hermes lsp status`**

Some servers ship as thin wrappers around an external CLI for actual
diagnostics — they spawn cleanly and accept requests but never emit
errors when the sidecar binary is missing. The most common case is
`bash-language-server`, which delegates diagnostics to `shellcheck`.
When `hermes lsp status` shows a `Backend warnings` section, install
the named tool through your OS package manager:

```
apt install shellcheck      # Debian / Ubuntu
brew install shellcheck     # macOS
scoop install shellcheck    # Windows
```

The same warning is logged once at server spawn time in
`~/.hermes/logs/agent.log`.

**Server starts but never returns diagnostics**

Check `~/.hermes/logs/agent.log` for `[agent.lsp.client]` entries —
both stderr from the language server and protocol errors land
there. Some servers (rust-analyzer especially) need to finish a
project-wide index before they emit per-file diagnostics; the first
edit after server start may complete with no diagnostics, with
subsequent edits picking them up.

**Server crashed**

A crashed server is added to the broken-set and won't be retried for
the rest of the session. Run `hermes lsp restart` to clear the set;
the next edit re-spawns.

**Editing a file outside any git repo**

By design, LSP only runs inside a git repository. If the project isn't
yet initialized, run `git init` to enable LSP diagnostics. Otherwise the
in-process syntax-only fallback applies.
