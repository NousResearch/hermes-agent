---
title: "Package Management"
description: "PM tool pins, Python environments, optional dependencies, and installation ownership"
---

# Package management

`hermes pm` manages Hermes tool binaries and Python dependency environments.
It is not the application updater. Use the installation's
[update method](/getting-started/updating) to update Hermes itself.

## Pins, installed state, and runtime selection

Each file has a separate role:

| File | Role |
|---|---|
| `pm/lock.json` | Exact managed-tool versions, target-specific URLs, and SHA-256 hashes. |
| `pm/pyproject.toml` and `pm/uv.lock` | The dependency manager's independent Python requirements and locked resolution. |
| `pyproject.toml` and `uv.lock` | Python requirements, extras, platform markers, and the committed Python resolution. |
| Tool-store `facts.json` | Installed tool entries, their identities, environment exports, and realized-file digests. |
| Per-install `facts.json` | The selected Python environment, its input stamp, and enabled extras. |
| Payload `manifest.json` | Relative payload layout and the completed launch contract from the bundle builder. |
| `install-stamp.json` | Build provenance and the declared distribution/update owner. |

A lockfile entry does not prove that a package is installed. `hermes pm doctor`
compares the installed state with the lock and checks the realized bytes.
Startup uses a cheaper check. It does not query upstream versions on every launch.

## Source installs and packaged builds

Source installers provision the required tools plus Python. They select the
`all` Python extra. Named optional tools install when requested.

Native desktop bundles stage the supported tool set and all target-compatible
Python extras before packaging. `--extra all` and `--all-extras` are not
synonyms. Platform markers still exclude dependencies that cannot run on a target.

A packaged application's base payload is immutable. Hermes runs its backend
from that payload, rather than copying a source checkout on first launch.
The bundle builder checks its files and writes the launch paths into the desktop
build stamp. Electron uses those paths without probing or repairing the payload.
Additional pinned tools can use the writable tool store. Python additions use
a complete writable environment outside the signed package.

Termux uses a separate bionic build and a sealed APT package. Docker bakes its
runtime into the image and disables on-demand dependency installation. Nix
provides its runtime through derivations. See the
[Termux](/getting-started/termux), [Docker](/user-guide/docker), and
[Nix](/getting-started/nix-setup) guides for their limits.

## Writable state

The platform default data root is `~/.hermes` on POSIX and
`%LOCALAPPDATA%\hermes` on Windows. `HERMES_HOME` and profiles can change
which data root a process uses.

| State | Default location |
|---|---|
| Shared writable tool entries | `tools/` under the resolved default Hermes root. |
| Resumable downloads | `cache/partials/` under that root, not inside a signed payload. |
| Per-install selection and journal | `installs/INSTALL_KEY/` under the dependency-state root. |
| Python generations | `installs/INSTALL_KEY/environments/`. |
| Sync and update receipts | `logs/update_receipts/` under the active home. |

The install key derives from the canonical source or payload-repository path.
Separate checkouts therefore have separate Python selections. Profiles that
share an installation can contribute dependencies to the same environment.
Their configuration and credentials remain profile-scoped.

Do not edit facts or generation paths manually. Launchers resolve the selected
environment before third-party imports. Processes retain their existing imports
until they restart. Garbage collection preserves selected generations and
lease-managed generations with live readers.

Downloads share a lock per partial URL. A failed or paused transfer cannot
publish a partial destination, and garbage collection cannot delete a live
transfer's files. Resume reuses ranges only when their recorded length,
validator, and optional hash still match. Without a strong ETag or pinned
hash, an interrupted transfer restarts instead of mixing different versions.
The small `.locks/` files remain after completion so waiting processes use
the same lock.

## Optional Python dependencies and plugins

A built-in feature requests a project extra through `pm.ensure_import`.
Directory plugins declare Python requirements in `pyproject.toml`, or through
legacy `pip_dependencies` or `python_dependencies` lists in `plugin.yaml`.

PM prepares core requirements, enabled extras, and enabled plugin requirements
together. It seeds resolution from the existing lock. Compatible transitive
versions can change, but declared constraints and exact pins remain binding.
The generated workspace and extended lock remain outside shipped source.

A failed candidate does not replace the selected environment or silently
disable other plugins. If preparation succeeds, a restart can still be required
to activate the new environment in a running Hermes process.

Ordinary Hermes application updates preserve user plugin directories. Explicit
plugin updates can change the selected plugin's files. A wrapper with no Python
dependency declaration does not join the shared environment. Its external
sidecar remains separately owned. See the
[plugin guide](/developer-guide/plugins).

### Lazy-install policy

`security.allow_lazy_installs` controls on-demand installation. Already installed
dependencies remain usable when this setting is false.

```bash
hermes config set security.allow_lazy_installs false
```

Explicit install commands are distinct from on-demand installation. However,
a bundle's frozen feature list still restricts requested Python extra names
when lazy installs are disabled. Explicit plugin admission is a separate
operation, not an on-demand feature request. Do not treat this setting as a
sandbox or a blanket prohibition on manual package installation.
Docker additionally sets the internal lazy-install disable flag in the image.

PM is a dependency manager, not a sandbox for plugin code. Installing a plugin
requires trust in that plugin and its dependencies.

## Developer workflow {#developer-workflow}

Activation asks PM to prepare or sync the toolchain for a source checkout, then
makes it available in the shell. It does not select your editor's Python
interpreter or redirect an installed desktop app to this checkout.

### Prepare a checkout

Use an ordinary terminal outside the packaged Hermes app. Leave any existing
Python virtual environment first. On Windows, use native PowerShell with Git.
For ARM64, setup checks Visual Studio C++ tools, Clang, native Rust, and static
OpenSSL development libraries before PM runs. It reuses existing installations
and installs missing prerequisites. Missing Visual Studio components require
an Administrator PowerShell. OpenSSL uses vcpkg's `arm64-windows-static-md`
triplet. A damaged shared installation produces a repair error, not automatic
deletion. Compiler and OpenSSL environment variables apply only to the setup
process when you enter through `activate.ps1`.

Other platforms still require the native compiler tools and libraries needed
by dependencies without compatible wheels.

Clone the repository and select your branch before preparing dependencies:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

For isolated development, select a separate data home before the first PM
command. Keep the same values when returning to this checkout.

Bash, from the repository root:

```bash
export HERMES_HOME="$HOME/hermes-dev-data"
export HERMES_RUNTIME_DIR="$HERMES_HOME/tools"
source ./activate
```

PowerShell, from the repository root:

```powershell
$env:HERMES_HOME = Join-Path $HOME 'hermes-dev-data'
$env:HERMES_RUNTIME_DIR = Join-Path $env:HERMES_HOME 'tools'
. .\activate.ps1
```

`HERMES_RUNTIME_DIR` in these examples is a process-local development override.
It makes the bootstrap and PM use the same writable store. Do not persist a
path into an installed MSIX or macOS bundle. Activation runs the setup script's
runtime-only path to provision tools and sync the `all` Python extra. It does
not select `dev` or install JS workspaces. It also skips setup's user-facing
installation work: shell configuration, launchers, `.env`, and bundled skills.
Run the setup script separately if you want that full installation workflow.

The bootstrap uses uv to install and locate Python, then waits for uv to exit.
PM uses the staged uv to prepare its own small, locked Python environment
before downloading its managed tools, reading plugin configuration, or resolving
application dependencies. Its project is deliberately
independent of the application workspace: a broken application dependency must
not prevent its dependency manager from starting. Each uv subprocess exits before
PM runs, so it cannot hold the uv executable that PM needs to replace.

PM's runtime contains `ruamel.yaml`, `packaging`, `tomli-w`, and `truststore`, not the application
dependency tree. CLI commands and application-requested installs and repairs run
there; read-only path and installed-environment lookups remain local. PM never
adds its dependencies to an already-running agent's imports. First-party YAML
readers and writers use ruamel; third-party packages can still require PyYAML in
the application environment. Failure receipts remain stdlib-only.

PM's CLI and worker activate `truststore` before importing their HTTPS clients.
This uses the platform certificate store even when bootstrap Python's compiled-in
OpenSSL paths do not locate it. No application dependencies or certificate-path
override are required. After the first install, PM rebuilds its small environment
against the managed Python on the next invocation; subsequent invocations reuse it.

When lazy installs are disabled, an existing PM runtime can still check whether
the application environment is current. If PM itself is missing or outdated,
the request fails without downloading tools or dependencies. Run an explicit
`hermes pm install` to prepare PM first.

Native bundles and Docker images stage this same PM lock through the shared
runtime builder. Termux supplies its verified offline wheelhouse to that
builder. Nix builds the PM lock as a separate derivation. Packaged workers use
only their recorded PM dependency directory, never the application's libraries.

### Activate an existing installation

In each new shell, restore your development-home values and enter the checkout.
Then activate it; there is no separate setup command to remember:

| Shell | Enter | Leave |
|---|---|---|
| Bash | `source ./activate` | `deactivate` |
| PowerShell | `. .\activate.ps1` | `deactivate` |

The leading dot and space in PowerShell are required. Executing
`.\activate.ps1` without dot-sourcing does not provide the same session scope.
The POSIX script uses Bash syntax. Use Bash for this recipe rather than `sh`,
fish, or assuming that a Zsh startup file has Bash semantics.

Each activation invokes PM's install/sync path. PM reuses current tools and
dependency generations; missing or stale inputs can require downloads and a
rebuild. A setup failure returns an error before changing the activated shell
environment, including when re-sourcing an already active environment.

After sync, activation prepends installed PM tools to `PATH` and sets
`PYTHONPATH` to this checkout and its selected dependency tree. It does not
change an OS-wide PATH or activate a conventional venv prompt.
Start in a clean shell rather than nesting this inside another venv.
`deactivate` restores the environment values captured by the activation script.
It does not uninstall packages or stop processes that you started.

Verify the interpreter and source before doing work:

```bash
python -c "import sys, pm; print(sys.executable); print(pm.__file__)"
python -c "import httpx; print(httpx.__file__)"
node --version
npm --version
python hermes --version
```

`python` must resolve to the PM store interpreter. `pm.__file__` must point
into this checkout. Dependencies come from the selected environment, which can
live outside the repository. A missing import means setup or selection needs
attention, even if `source ./activate` itself returned successfully.

### Work on this source tree

Use checkout-qualified commands so a global `hermes` command or MSIX alias
cannot run a different installation:

```bash
python hermes setup
python hermes
python hermes --tui
python -m pm.cli status
```

These commands use the selected development home. A source-file edit is visible
to the next process. Restart the affected CLI, gateway, or backend after edits.
Reinstalling every dependency is unnecessary for a Python-only source change.

For the JavaScript workspaces, run `npm ci` once at the repository root, then
run the relevant workspace command. For example:

```bash
npm run build --workspace ui-tui
npm run dev --workspace apps/desktop
```

The website is separate: `npm ci --prefix website`, then
`npm run build:fast --prefix website`. PM activation supplies tools, not these
`node_modules` directories or built assets. Native desktop builds have additional
requirements in the [desktop build guide](https://github.com/NousResearch/hermes-agent/blob/main/apps/desktop/BUILDING.md).

### Refresh dependencies without changing branches

After a branch or lockfile change, source the activation script again to sync
and select the new dependencies (`source ./activate` in Bash or
`. .\activate.ps1` in PowerShell). To sync without activating a shell:

```bash
python -m pm.cli install
```

After a standalone sync, reactivate the environment. Restart affected processes.
Use `python -m pm.cli doctor` for tool diagnostics and `python -m pm.cli status`
for the latest sync receipt. Do not run `hermes update` just to refresh a
feature branch: it is an application update and can change the source branch.

Managed tool names and Python extra names are different interfaces:

```bash
python -m pm.cli install chromium
python -c "from pm import sync_venv; sync_venv(['dev'], explicit=True)"
```

The first command installs a tool. The second adds the declared `dev` extra
to this installation's existing Python selection. Extras accumulate through PM
sync. `pm install dev` is not a supported command: `dev` is an extra, not a tool.
After changing extras, reactivate before starting another Python process.

For a new project dependency, edit `pyproject.toml` and regenerate `uv.lock`
with `uv lock`. For JS dependencies, update the owning package manifest and
lock. Do not edit PM facts or generated workspaces. Unrecorded pip installs
are not durable and can disappear when PM selects a new environment.

### Test and editor environments

PM's `dev` extra does not make a bare store Python suitable for the canonical
test runner. The runner clears `PYTHONPATH` and needs an interpreter with pytest
installed in its own environment. Use the contributor guide's
[independent test environment](/developer-guide/contributing#manual-development-and-test-environment)
with `uv sync --extra dev --group test`, then run `scripts/run_tests.sh`
(through Bash on Windows). The `test` dependency group includes native launcher
test dependencies and does not enter a packaged runtime.

The runner checks repository `.venv`, repository `venv`, and the standard
source-install venv before using `HERMES_PYTHON` as a fallback. Read its startup
message to confirm which interpreter it selected. A worktree without a local
venv can use the independent test interpreter through that variable.

For editor debugging, select that independent interpreter, set the working
directory to this checkout, and launch `hermes` as the script. Keep its
`HERMES_HOME` separate from production. Terminal activation does not configure
an editor that was already running. Do not point an editor at a transient PM
generation or a signed application's Python executable.

## Commands

```bash
hermes pm --help
hermes pm doctor
hermes pm status
hermes pm install
hermes pm install chromium
```

| Command | Effect |
|---|---|
| `pm install [names...]` | Install named packages. With no names, provision required tools plus Python and sync the `all` extra. |
| `pm env [names...]` | Print the composed environment of installed packages as JSON. It does not install missing packages. |
| `pm doctor` | Check installed tool identities, files, and digests against the lock. |
| `pm repair` | Rebuild the recorded Python dependency set in a new generation, validate it, then select it. Does not update pins, features, or plugin configuration. |
| `pm status` | Print the latest sync/update receipt as JSON, or report that no receipt exists. |
| `pm gc` | Remove unreferenced tool-store entries, eligible download partials, and unused lease-managed Python generations. |

`pm env` can include inherited environment values. Do not publish its output
without removing credentials.

### Maintainer commands

These commands change dependency inputs or stage build artifacts. They are
not substitutes for an installed application's update mechanism.

| Command | Effect |
|---|---|
| `pm lock --bump NAME VERSION` | Resolve and hash supported target artifacts, then write the tool pin. |
| `pm update [names...]` | Query upstream versions, change tool pins, and install changed tools. |
| `pm update --check` | Query without writing. Exit 1 can mean updates exist; inspect output to distinguish an error. |
| `pm update --target TARGET` | Resolve versions for the specified target. |
| `pm update --uv` / `--npm` | Also refresh the Python or npm dependency resolution. |
| `pm install --target TARGET NAME...` | Stage explicit cross-target packages without recording them as the host's installed runtime. |
| `pm bundle --out DIR [--ref REF]` | Stage a source snapshot, native tools, facts, and Python dependencies. It does not produce a signed desktop installer. |

The complete desktop builder also builds the JavaScript surfaces, generates
launchers, and invokes native packaging. Maintainers can read
[Building the Desktop Installers](https://github.com/NousResearch/hermes-agent/blob/main/apps/desktop/BUILDING.md).

## Network retries

PM retries transient HTTP failures during tool downloads, artifact hashing and
version lookups. Each probe or transfer gets at most four attempts. Backoff
waits are 1, 2 and 4 seconds. A `Retry-After` header can extend a wait, up to
30 seconds. Each retry reports its cause, delay and next attempt in the log.

Retryable HTTP statuses are 408, 429, 500, 502, 503 and 504. Connection resets,
timeouts, temporary DNS failures and interrupted response bodies also retry.
Ranged downloads retain completed bytes. Servers without range support require
a fresh stream. Pause interrupts backoff and preserves the partial download.

PM does not retry bad hashes, certificate failures, local filesystem errors or
other permanent failures. A successful probe followed by a range GET 403 or 404
retains the CDN fallback: one serial attempt at the missing ranges. Extraction,
verification and publication are not repeated. Python and npm package requests
remain under uv and npm's own retry policies.

## Diagnostics

- **Missing or outdated tool:** read `hermes pm doctor`, then use an explicit PM install on a writable installation.
- **New environment requires restart:** restart the affected Hermes process. Do not add a second site-packages tree to its live imports.
- **Dependency conflict:** read `hermes pm status`. Correct the plugin requirements before retrying admission.
- **Damaged Python dependencies:** run `hermes pm repair`, then restart Hermes. Repair replays the selected generation's saved workspace and lock without parsing plugin configuration. An unreadable record or missing saved lock fails without selecting a reduced dependency set. Before a generation exists, repair uses the shipped or committed lock and recorded feature set.
- **Interrupted dependency install:** startup requests the same PM repair before dependency activation. Automatic attempts are bounded; `pm repair` retries explicitly. A failed repair preserves the previous selection and its retry marker.
- **Damaged Python executable or application source:** repair or reinstall through the package owner. PM cannot run without those files. Signed payload files are never modified by dependency repair.
- **Unknown package or extra:** use the declared name. `pm install` takes package names, not Python extra names or pip specifications.
