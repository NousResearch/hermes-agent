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
directory (or the file being edited) is inside a git worktree, LSP
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

For "manual" entries, install the server through whatever toolchain
manager makes sense for that language (rustup, ghcup, opam, brew,
…). Hermes auto-detects the binary on PATH or in
`<HERMES_HOME>/lsp/bin/`.

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
  wait_timeout: 5.0

  # How to handle missing server binaries.
  #   auto    — install via npm/pip/go install into <HERMES_HOME>/lsp/bin
  #   manual  — only use binaries already on PATH
  install_strategy: auto

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
rust-analyzer mid-indexing.

Servers are kept alive for the life of the Hermes process. There's
no idle-timeout reaper — the cost of restarting the server's index
on every write would be far higher than holding the daemon.

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

By design, LSP only runs inside git worktrees. Run `git init` in the
project, or accept the in-process syntax-only fallback.
