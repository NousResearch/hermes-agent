---
sidebar_position: 4
title: "Contributing"
description: "How to contribute to Hermes Agent — dev setup, code style, PR process"
---

# Contributing

Thank you for contributing to Hermes Agent! This guide covers setting up your dev environment, understanding the codebase, and getting your PR merged.

## Contribution Priorities

We value contributions in this order:

1. **Bug fixes** — crashes, incorrect behavior, data loss
2. **Cross-platform compatibility** — macOS, different Linux distros, WSL2
3. **Security hardening** — shell injection, prompt injection, path traversal
4. **Performance and robustness** — retry logic, error handling, graceful degradation
5. **New skills** — broadly useful ones (see [Creating Skills](creating-skills.md))
6. **New tools** — rarely needed; most capabilities should be skills
7. **Documentation** — fixes, clarifications, new examples

## Contribution rubric

The project's intent layer, summarised in the root `AGENTS.md`; this is the long form with the
examples. Hermes ships a lot: most merges are bug fixes and the product surface (platforms,
providers, models, desktop/TUI features) expands on purpose. The restraint targets the core agent
and the model tool schema, where every addition is paid for on every API call: expansive at the
edges, conservative at the waist.

### What we want

- **Fix real bugs, well.** Reproduce the symptom on current `main`, point to the exact line
  where it manifests, and fix the whole bug class — sibling call paths included.
- **Expand reach at the edges.** New adapters, channels, providers, models, desktop/TUI/
  dashboard features land routinely, including large ones — as long as they integrate with
  the existing setup/config UX (`hermes tools`, `hermes setup`, auto-install) rather than
  bolting on a raw env var.
- **Refactor god-files into clean modules.** Huge mechanical `+N/-N` extraction PRs are
  wanted work. "Every line traces to the request" applies to *feature* PRs; a declared
  refactor's request IS the extraction.
- **Keep the core narrow.** Prefer, in order: extend existing code → CLI command + skill →
  service-gated tool (`check_fn`) → plugin → MCP server in the catalog → new core tool (last
  resort). See the Footprint Ladder below.
- **Extend, don't duplicate.** Check whether existing infrastructure covers the use case
  before adding a module/manager/hook. When 3+ open PRs integrate the same *category*
  (memory backends, providers, notifiers), design an ABC + orchestrator, wrap the existing
  built-in as the first provider, and turn the competing PRs into plugins against it.
- **Behavior contracts over snapshots.** Tests assert how two pieces of data relate, never
  freeze a current value (see `tests/AGENTS.md`).
- **E2E validation, not just green unit mocks.** Anything touching resolution chains, config
  propagation, security boundaries, remote backends, or file/network I/O must exercise the
  real path with real imports against a temp `HERMES_HOME` — two of them (A→B→A) when the
  change touches profile scope. Mocks hide integration bugs.
- **Cache-, alternation-, and invariant-safe.** Preserve prompt caching, strict role
  alternation (never two same-role messages in a row; never a synthetic user message injected
  mid-loop), and a system prompt byte-stable for the life of a conversation.
- **Contributor credit preserved.** Salvage external work by cherry-picking (rebase-merge) so
  authorship survives; build on top rather than reimplementing.

### What we don't want (rejected even when well-built)

- **Speculative infrastructure.** Hooks/callbacks/extension points with no concrete consumer.
  Adding a hook is easy; removing one after plugins depend on it is hard. A hook with a real,
  stated use case is NOT speculative even if the consumer ships separately.
- **New `HERMES_*` env vars for non-secret config.** `.env` is for secrets only. Behavioral
  settings (timeouts, thresholds, flags, display prefs) go in `config.yaml`; bridge to an
  internal env var in code if the mechanism needs one. Reject "set X in your .env" docs
  unless X is a credential.
- **A new core tool when terminal + file (or a skill) already do the job.** If the only
  barrier is file visibility on a remote backend, fix the mount, not the toolset.
- **Lazy-reading escape hatches on instructional tools.** No `offset`/`limit` pagination on
  tools that load content the agent must read fully (skills, prompts, playbooks) — models
  read page 1 and skip the rest.
- **"Fixes" that destroy the feature they secure.** Read the original intent
  (`git log -p -S`) before restricting behavior; find a fix that preserves the feature.
- **Outbound telemetry / usage attribution without opt-in gating.** No analytics,
  third-party identifier tagging, or attribution tags until a generic user-facing opt-in
  (config gate + setup prompt + `hermes tools` toggle) exists. Park behind a label.
- **Change-detector tests, cache-breaking mid-conversation, dead code wired in without E2E
  proof, plugins that touch core files.** Plugins work within the ABCs/hooks we provide; if
  one needs more, widen the generic plugin surface, never special-case it in core.
- **Third-party products integrated into the core tree.** Observability backends, vendor
  SaaS connectors, analytics dashboards, and other "someone else's product" plugins do NOT
  land under `plugins/` — every one becomes our burden against a fast-moving core for a
  backend we don't own. Ship as a **standalone plugin repo** (`~/.hermes/plugins/` or pip
  entry point), promoted in the Nous Research Discord `#plugins-skills-and-skins`. This is a
  coupling decision, not a quality bar; such PRs are closed with a pointer to publish.

### Before you call it a bug — verify the premise (and when NOT to close)

The most common reason a well-written PR is closed is a **wrong premise** or treating an
**intentional design as a gap**. These patterns tell a reviewer what to scrutinize and tell
the sweeper when a PR is NOT safe to close (when in doubt, leave it open for a human):

- **"Intentional design, not a gap."** Ask whether the isolation IS the design. Profiles are
  independent islands on purpose: a PR adding live config inheritance from the default
  profile was closed because coupling profiles is exactly what the design prevents (`--clone`
  already covers "start from my default"). Read `git log -p -S "<symbol>"` before assuming
  something is unfinished.
- **"The premise doesn't hold against how X actually works."** Trace the real runtime before
  accepting a rationale. Real closes: a rate-limit "re-probe during cooldown" PR (the breaker
  trips only on a *confirmed-empty* bucket, so re-probing hammers a bucket proven empty); a
  usage fix whose new branch **never executes** because an earlier guard already popped the
  state. If you can't point to the exact line where the bug manifests AND show the fix changes
  that line's behavior, the premise is unverified.
- **"The absence was deliberate."** Restoring "missing" `__init__.py` files made a test tree
  importable as a dotted package that shadowed the real plugin and deleted its `register()`
  at import time. The omission was load-bearing.
- **"Overreached / resurrected an approach we moved past."** Scope creep beyond the agreed
  base, or reviving a direction maintainers closed, is rejected even when it works. Offer the
  rest as a focused follow-up.

Throughline: **verify the claim AND the intent against the codebase before writing or merging
a fix.** A reproduction on current `main` plus a line-level account beats a plausible
rationale. When unsure about intent, asking is cheaper than shipping a fix that fights the
design.

### The Footprint Ladder (new capability decision)

Choose the highest (least-footprint) rung that correctly solves the problem:

1. **Extend existing code** — a variation of something that exists. Zero new surface.
2. **CLI command + skill** — config/state/infra expressible as shell commands; the agent runs
   `hermes <subcommand>` guided by a skill. Default for subscriptions, scheduled tasks,
   service setup (`hermes webhook`, `hermes cron`, `hermes tools`).
3. **Service-gated tool (`check_fn`)** — needs structured params/returns AND only appears when
   a prerequisite is configured (Home Assistant tools, memory-provider tools). This rung gates
   reachability/opt-in process-wide; a capability that varies per SESSION (who is watching) is
   a named toolset folded in by the toolset resolver, not a `check_fn` — see `tools/AGENTS.md` § "Surface
   capability is a property of the SESSION".
4. **Plugin** — third-party/niche/user-specific; lives in `~/.hermes/plugins/` or a pip
   package, discovered at runtime.
5. **MCP server (in the catalog)** — genuinely a tool but not core-fundamental. Zero permanent
   core-schema footprint, reusable by any MCP host, reached via the built-in MCP client.
6. **New core tool** — only when fundamental, broadly useful to nearly every user, and
   unreachable via terminal + file or an MCP server (terminal, read_file, web_search,
   browser_navigate).

## Common contribution paths

- Building a custom/local tool without modifying Hermes core? Start with [Build a Hermes Plugin](../developer-guide/plugins/index.md)
- Building a new built-in core tool for Hermes itself? Start with [Adding Tools](./adding-tools.md)
- Building a new skill? Start with [Creating Skills](./creating-skills.md)
- Building a new inference provider? Start with [Adding Providers](./adding-providers.md)

## Development Setup

### Prerequisites

| Requirement          | Notes                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------- |
| **Git**              | With the `git-lfs` extension installed                                                        |
| **Python 3.14** | Current development uses PM's pinned interpreter. The broader `>=3.11,<3.15` package metadata keeps old updaters working, not the current runtime on older Python. |
| **Node.js** | Use the PM pin or a version accepted by root `package.json` engines |

### PM developer environment

Use the [PM developer workflow](../reference/package-management.md#developer-workflow) for preparation, activation, everyday commands,
dependency changes, and test environments. Select your development
home before setup so experimental code does not migrate production data.

Activate from the repository root in each new shell. Activation prepares the
checkout through PM and syncs stale dependencies.

Bash:

```bash
source ./activate
hermes --version
```

PowerShell:

```powershell
. .\activate.ps1
hermes --version
```

Run `hermes` for this checkout. Activation defines it as a function for this
worktree, so it hides a global `hermes` alias and refuses outside the worktree.
PM activation syncs tools and Python dependencies before adding them to the shell. It does
not install JS workspaces or rewrite launchers and shell configuration. `deactivate` restores the prior shell environment and removes the function.

### Manual development and test environment {#manual-development-and-test-environment}

Use the [PM developer workflow](../reference/package-management.md#developer-workflow) to prepare Python 3.14 first.
Run these commands from that checkout with its prepared Python. Keep the same
development `HERMES_HOME`. PM must be able to start before it can build another
environment. On Windows, initialize the native C++ build environment for your
architecture before building source dependencies.

Build an independent interpreter for tests and editor tools:

```bash
python -m pm.build_env --source . --out .venv --group dev --group test
```

PM builds from the committed lock and checks dependency consistency before
returning the new interpreter. The `test` group includes native launcher test
dependencies and does not enter the application runtime. If tests require
another declared feature, add its `--extra`.

The output must not exist, even as an empty directory or symlink. To regenerate
it after a dependency change, stop its processes and intentionally remove only
that disposable environment first. PM does not delete an existing destination.
Do not run raw pip or uv commands to change a PM-built environment.

To keep the test environment outside the checkout, replace `.venv` with a fresh absolute
path. Set `HERMES_PYTHON` to that environment's interpreter:

- POSIX: `export HERMES_PYTHON="/absolute/path/to/hermes-dev/bin/python"`
- PowerShell: `$env:HERMES_PYTHON = 'C:\absolute\path\to\hermes-dev\Scripts\python.exe'`

The canonical runner discovers repository `.venv` automatically. It clears
`PYTHONPATH`, so pytest must be installed in the interpreter's own environment.
This test environment does not replace PM's application selection or tool
store. Do not point a bundled app at it or install into an MSIX payload.

For an isolated development instance, select a disposable `HERMES_HOME` before
starting the source command. Use `hermes setup` to configure it rather
than copying production credentials into the checkout.

### JavaScript workspaces and website

From the repository root, run `npm ci` for the desktop, TUI, dashboard, and
shared JS workspaces. The website is separate:

```bash
npm ci --prefix website
npm run build:fast --prefix website
```

Use a Node/npm version accepted by the corresponding `package.json` engines.
Native desktop dependencies can also require the platform build toolchain.

Logos and icons are generated from `assets/nous-girl-*.svg` and
`assets/backgrounds/`. `node scripts/generate-icons.mjs` renders them with the
Hermes runtime Python (`HERMES_PYTHON`, else `python` on PATH): Pillow and
resvg-py are core dependencies. Generated outputs are committed and CI fails if
they are stale; rerun the generator and commit after changing any source SVG.

### Run tests

Use the canonical runner on every host:

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/agent/ -v
```

On Windows, run the script through Bash. When no local `.venv` or `venv`
contains pytest, the runner accepts the explicit `HERMES_PYTHON` above. It
clears credentials, isolates `HERMES_HOME`, and runs each test file in a separate
subprocess through `scripts/run_tests_parallel.py`. It does not use xdist.
When `tests/conftest.py` redirects a production `HERMES_HOME` to a temporary
session home, it sets the internal `HERMES_TEST_SANDBOX_HOME` marker. This lets
re-imported test fixtures recognize their own sandbox instead of flagging it as
real-home I/O. Do not set this marker yourself; set `HERMES_HOME` for a
disposable development home and let the test runner isolate it.

Run the relevant JS workspace checks for JS changes. Native install/update
E2E runs on disposable CI hosts, never against the developer's live app.
See [Package management](../reference/package-management.md) for PM commands and runtime ownership.

## Code Style

- **PEP 8** with practical exceptions (no strict line length enforcement)
- **Comments**: Only when explaining non-obvious intent, trade-offs, or API quirks
- **Error handling**: Catch specific exceptions. Use `logger.warning()`/`logger.error()` with `exc_info=True` for unexpected errors
- **Cross-platform**: Never assume Unix (see below)
- **Profile-safe paths**: Never hardcode `~/.hermes` — use `get_hermes_home()` from `hermes_constants` for code paths and `display_hermes_home()` for user-facing messages. See [AGENTS.md](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md#profiles-multi-instance-support) for full rules.

## Cross-Platform Compatibility

See **[Platform Support](../getting-started/platform-support.md)**. Native Windows uses Git Bash (from [Git for Windows](https://git-scm.com/download/win)) for shell commands. The dashboard uses POSIX PTYs on Unix and the `pywinpty`/ConPTY bridge on Windows. Availability depends on that host's native dependency support. If you're doing Windows-heavy dev, run the Windows-footgun lint (`scripts/check-windows-footguns.py`) before pushing.

When contributing code, keep these rules in mind:

- **Don't add unguarded `signal.SIGKILL` references.** It's not defined on Windows. Either route through `gateway.status.terminate_pid(pid, force=True)` (the centralized primitive that does `taskkill /T /F` on Windows and SIGKILL on POSIX), or fall back with `getattr(signal, "SIGKILL", signal.SIGTERM)`.
- **Use `psutil.pid_exists()` for process liveness.** Do not use `os.kill(pid, 0)` on Windows; it is not a safe probe.
- **Don't force the terminal to POSIX semantics.** `os.setsid`, `os.killpg`, `os.getpgid`, `os.fork` all raise on Windows — gate them with `if sys.platform != "win32":` or `if os.name != "nt":`.
- **Use explicit text encodings.** User-authored UTF-8 reads use `utf-8-sig` to accept a leading BOM. Writes use `utf-8` without adding a BOM.
- **Use `pathlib.Path` / `os.path.join` — never manually concat with `/`.** This matters less for strings the OS gives us back and more for strings we construct to hand to subprocesses.

Key patterns:

### 1. File encoding

Some environments may save `.env` files in non-UTF-8 encodings:

```python
try:
    load_dotenv(env_path)
except UnicodeDecodeError:
    load_dotenv(env_path, encoding="latin-1")
```

### 2. Process management

`os.setsid()`, `os.killpg()`, and signal handling differ across platforms:

```python
import platform
if platform.system() != "Windows":
    kwargs["preexec_fn"] = os.setsid
```

### 3. Path separators

Use `pathlib.Path` instead of string concatenation with `/`.

## Security Considerations

Hermes has terminal access. Security matters.

### Existing Protections

| Layer                           | Implementation                                                              |
| ------------------------------- | --------------------------------------------------------------------------- |
| **Sudo password piping**        | Uses `shlex.quote()` to prevent shell injection                             |
| **Dangerous command detection** | Regex patterns in `tools/approval.py` with user approval flow               |
| **Cron prompt injection**       | Scanner blocks instruction-override patterns                                |
| **Write deny list**             | Protected paths resolved via `os.path.realpath()` to prevent symlink bypass |
| **Skills guard**                | Security scanner for hub-installed skills                                   |
| **Code execution sandbox**      | Child process runs with API keys stripped                                   |
| **Container hardening**         | Docker: all capabilities dropped, no privilege escalation, PID limits       |

### Contributing Security-Sensitive Code

- Always use `shlex.quote()` when interpolating user input into shell commands
- Resolve symlinks with `os.path.realpath()` before access control checks
- Don't log secrets
- Catch broad exceptions around tool execution
- Test on all platforms if your change touches file paths or processes

## Pull Request Process

### Branch Naming

```
fix/description        # Bug fixes
feat/description       # New features
docs/description       # Documentation
test/description       # Tests
refactor/description   # Code restructuring
```

### Before Submitting

1. **Run tests**: `scripts/run_tests.sh` for CI-parity. Use direct `python -m pytest ...` only when the wrapper is unavailable or you are intentionally debugging outside the wrapper.
2. **Test manually**: Run `hermes` and exercise the code path you changed
3. **Check cross-platform impact**: Consider macOS, Linux, WSL2, and native Windows. If you touch file I/O, process management, terminal handling, subprocesses, or signals, run `scripts/check-windows-footguns.py`.
4. **Keep PRs focused**: One logical change per PR

### PR Description

Include:

- **What** changed and **why**
- **How to test** it
- **What platforms** you tested on
- Reference any related issues

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

| Type       | Use for                       |
| ---------- | ----------------------------- |
| `fix`      | Bug fixes                     |
| `feat`     | New features                  |
| `docs`     | Documentation                 |
| `test`     | Tests                         |
| `refactor` | Code restructuring            |
| `chore`    | Build, CI, dependency updates |

Scopes: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`

Examples:

```
fix(cli): prevent crash in save_config_value when model is a string
feat(gateway): add WhatsApp multi-user session isolation
fix(security): prevent shell injection in sudo password piping
```

### Repo-local review checklists: `.agents/checks/*.md`

Projects built on (or reviewed by) Hermes can keep reviewer checklists inside the repository under `.agents/checks/`. Each file is a focused, plain-markdown checklist that an agent loads before reviewing a change touching the matching area:

```
.agents/
  checks/
    security.md        # e.g. "grep the diff for shell interpolation; check subprocess calls quote args"
    migrations.md      # e.g. "every schema change ships a backfill and a rollback note"
    public-api.md      # e.g. "exported signatures changed? flag for semver review"
```

Conventions that make these work well:

- **One concern per file**, named after the concern. Small files get read in full; a monolithic `checklist.md` gets skimmed.
- **Write checks as verifiable actions** ("run X and confirm Y"), not aspirations ("code should be secure").
- **State the trigger at the top** — which paths or change types the checklist applies to — so an agent (or human) can skip irrelevant ones cheaply.
- Keep them in version control next to the code they guard: they evolve with the codebase, and a PR that changes the rules changes the checklist in the same diff.

When you ask Hermes to review a PR in a repository that has `.agents/checks/`, tell it (or teach it via a skill) to read the relevant checklists first and report against them. This gives review agents the project-specific bar that generic review prompts miss.

## Reporting Issues

- Use [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
- Include: OS, Python version, Hermes version (`hermes --version`), full error traceback
- Include steps to reproduce
- Check existing issues before creating duplicates
- For security vulnerabilities, please report privately

## Community

- **Discord**: [discord.gg/NousResearch](https://discord.gg/NousResearch)
- **GitHub Discussions**: For design proposals and architecture discussions
- **Skills Hub**: Upload specialized skills and share with the community

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE).
