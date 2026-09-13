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
| **Python 3.14** | The project requires `>=3.14,<3.15`. PM provides the pinned interpreter. |
| **Node.js** | Use the PM pin or a version accepted by root `package.json` engines |

### PM developer environment

Use the [PM developer workflow](/reference/package-management#developer-workflow) for preparation, activation, everyday commands,
dependency changes, and test environments. Select your development
home before setup so experimental code does not migrate production data.

Activate from the repository root in each new shell. Activation prepares the
checkout through PM and syncs stale dependencies.

Bash:

```bash
source ./activate
python hermes --version
```

PowerShell:

```powershell
. .\activate.ps1
python hermes --version
```

Run `python hermes` for this checkout, not a global `hermes` alias. PM activation
syncs tools and Python dependencies before adding them to the shell. It does
not install JS workspaces or rewrite launchers and shell configuration. `deactivate` restores the prior shell environment.

### Manual development and test environment {#manual-development-and-test-environment}

Use the [PM developer workflow](/reference/package-management#developer-workflow) to prepare Python 3.14 (`>=3.14,<3.15`) first.
Run these commands from that checkout with its prepared Python. Keep the same
development `HERMES_HOME`. PM must be able to start before it can build another
environment. On Windows, initialize the native C++ build environment for your
architecture before building source dependencies.

Build an independent interpreter for tests and editor tools:

```bash
python -m pm.build_env --source . --out .venv --extra dev --group test
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
starting the source command. Use `python hermes setup` to configure it rather
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
`assets/backgrounds/`. `node scripts/generate-icons.mjs` uses the locked,
isolated `icon-build` dependency group. Do not commit generated PNG/ICO/ICNS
outputs or add icon renderers to production dependencies.

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
