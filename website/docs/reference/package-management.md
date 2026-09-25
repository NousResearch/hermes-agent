---
title: "Package Management"
description: "PM tool pins, Python environments, optional dependencies, and installation ownership"
---

# Package management

`hermes pm` manages Hermes tool binaries and Python dependency environments.
It is not the application updater. Use the installation's
[update method](../getting-started/updating.md) to update Hermes itself.

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
For self-managed source installs, the pre-import launcher compares PM's recorded
successful dependency stamp with the current inputs. Missing or stale completion
state triggers a sync, then the same command restarts on the managed Python before
loading application dependencies. Failed syncs keep the previous selection and
retry on the next launch; no pending-update marker is required. Developer checkouts
and packaged installations retain their existing owner.

Historical updaters can still execute old Python code after replacing the source
tree. Compatibility entry points start a fresh child, wait for completion, and
return its exit status. The child bootstrap asks PM to provision required tools
and select the Python generation before application imports. Completion runs in
that interpreter with the update context and receipt. The parent does not clear
`sys.modules`, import the new application graph, or resume a pip fallback.

Current source updates hand the selected checkout to a fresh completion owner.
Its bootstrap Python disables site-package initialization before asking PM to
sync the recorded extras and enabled plugins. The selected Python then owns
frontend builds, profile/configuration maintenance, gateway restarts and runtime
verification. Git, already-current retries and ZIP fallback use this same path.
The original command keeps the update lock while waiting; a missing or failed
completion result cannot report success. Correlated PM failures remain in the
update receipt, and interrupted restarts retain their fleet obligation.
Dependency or build failures never retry through pip or a source re-download.
Use `hermes pm repair` for damaged dependency files. See the developer
[source completion ownership note](../developer-guide/source-update-completion.md).

## Source installs and packaged builds

Source installers provision the required tools plus Python. They select the
`all` Python extra. Named optional tools install when requested.

For a canonical source installation, Desktop checks and runs the published
installation launcher. PM owns its interpreter and dependency selection. Desktop
does not replace that command with a guessed `venv` path. Developer overrides
retain their selected interpreter.

Native desktop bundles stage the supported tool set and all target-compatible
Python extras before packaging. `--extra all` and `--all-extras` are not
synonyms. Platform markers still exclude dependencies that cannot run on a target.

The complete desktop builder composes PM with the Node/native packaging providers.
From a clean checkout at a release tag, `python scripts/bundles/desktop.py --tag vX.Y.Z`
prepares dependencies and builds the installer. Add `--prepare-only` to stop after
preparation; consume its job-local result with
`python scripts/bundles/desktop.py --prepared .build/desktop-job/prepared.json`.
`--work` and `--cache` select separate build-owned roots. Preparation, not the
caller, creates the work directory. A full commit SHA can replace the tag through
`--commit`; the checkout must match.

Preparation uses isolated PM state, pinned tools, fresh path-bound Python
environments, the complete JS workspace union, native bindings and packaging
utilities. Reusable dependency caches are not live installations or portable
virtual environments. The prepared result binds the source, target and paths;
missing or changed inputs fail consumption rather than trigger a download.
Reprepare after a move or input change. Signing and notarization can still use
the network. See the
[desktop build guide](https://github.com/NousResearch/hermes-agent/blob/main/apps/desktop/BUILDING.md)
for native compiler requirements and release verification limits.

A packaged application's base payload is immutable. Hermes runs its backend
from that payload, rather than copying a source checkout on first launch.
The bundle builder checks its files and writes the launch paths into the desktop
build stamp. Electron uses those paths without probing or repairing the payload.
Additional pinned tools can use the writable tool store. Python additions use
a complete writable environment outside the signed package. Its first generation
retains the shipped extras along with the new requirements. Later generations
use the recorded extra selection as their baseline.

Termux uses a separate bionic build and a sealed APT package. Docker bakes its
runtime into the image and disables on-demand dependency installation. Nix
provides its runtime through derivations. See the
[Termux](../getting-started/termux.md), [Docker](../user-guide/docker.md), and
[Nix](../getting-started/nix-setup.md) guides for their limits.

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
Directory plugins declare Python requirements in `pyproject.toml`. Without an
authored project file, PM combines the legacy `pip_dependencies` and
`python_dependencies` lists from `plugin.yaml` or `plugin.yml`. An old
PM-generated project file does not override those lists. Consent, dependency
membership, and currency checks use the same declaration reader.

PM prepares core requirements, enabled extras, and enabled plugin requirements
together. It seeds resolution from the existing lock. Compatible transitive
versions can change, but declared constraints and exact pins remain binding.
Each candidate gets a fresh workspace with explicit source, lock seed, and
prepared environment inputs. The generated workspace and extended lock remain
outside shipped source. Repair copies the recorded workspace and lock, including
plugin build inputs, rather than resolving against edited live manifests.

Plugin enablement and staged code updates are submitted as data to the isolated
PM worker. Under the installation lock, it discovers the proposed dependency
union, validates the candidate, and publishes configuration or plugin files and
metadata with the environment selection. It rejects inputs changed during
preparation. A durable journal permits recovery before application imports,
including code-only updates that do not require a new environment.

Plugin selection changes, including pack enables, use the same admission
transaction. PM reads the latest selection under its shared lock before applying
each change. A failed candidate does not replace the selected environment or
silently disable other plugins. If preparation succeeds, a running Hermes process
can still require a restart to activate the new environment.

Ordinary Hermes application updates preserve user plugin directories. Explicit
plugin updates can change the selected plugin's files. A wrapper with no Python
dependency declaration does not join the shared environment. Its external
sidecar remains separately owned. See the
[plugin guide](../developer-guide/plugins/index.md).

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

## Optional security tools

PM owns the pinned `bws`, `tirith`, and `iron-proxy` packages in
`pm/security_packages.py`. Their versions, artifact URLs, and SHA-256 hashes
come from `pm/lock.json`. Downloads and publication use the shared tool store,
not private installers under `$HERMES_HOME/bin`.

For Tirith and iron-proxy, PM also acquires pinned signature files and checks
that the release checksums cover the pinned archive. Package staging calls the
integration's signature checker. Cosign and GPG checks remain conditional on
available executables. Locked provenance files must still be available and
match their hashes. An explicit signature rejection aborts installation.
External executables remain outside PM's hash and signature guarantees.

`bws` and iron-proxy honor an executable on `PATH` before checking PM selection.
Tirith honors `security.tirith_path`, then uses `PATH` before its PM selection
for the default name. An explicit Tirith path never triggers a replacement
download. Lazy installation obeys PM policy. Explicit install commands check
and repair managed entries, including requests with `--force`.

## Developer workflow {#developer-workflow}

Activation asks PM to prepare or sync the toolchain for a source checkout, then
makes it available in the shell. It does not select your editor's Python
interpreter or redirect an installed desktop app to this checkout.

### Prepare a checkout

Use an ordinary terminal outside the packaged Hermes app. Leave any existing
Python virtual environment first. On Windows, use native PowerShell with Git.
On ARM64, PM prepares Visual Studio C++ tools, Clang, native Rust, and static
OpenSSL development libraries before every dependency build from a checkout:
setup, `activate.ps1`, `install.ps1`, `hermes update`, and repair alike. It
reuses existing installations and installs missing prerequisites. Missing
Visual Studio components need administrator rights: an interactive install
asks through a UAC prompt, while CI, ssh and scheduled runs need an
Administrator PowerShell. OpenSSL uses
vcpkg's `arm64-windows-static-md` triplet. A damaged shared installation
produces a repair error, not automatic deletion. Compiler and OpenSSL
environment variables apply only to PM's dependency build, never to your shell.

Other platforms still require the native compiler tools and libraries needed
by dependencies without compatible wheels.

For the pinned macOS Python, PM defaults `AR` to `/usr/bin/ar`: the distributed
interpreter's sysconfig still points at its supplier's temporary LLVM directory.
This applies to source and bundle builds alike. Explicit `AR` and `CC` values
remain authoritative; PM does not change the toolchain of a caller-supplied
interpreter such as Nix Python. No `CC` default is needed for the current pin.

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
not select `dev` or install JS workspaces. It maintains installation-local
commands and repairs existing owned PATH wrappers, but does not create new PATH
conveniences or load application configuration. It also skips setup's user-facing
installation work: shell configuration, `.env`, and bundled skills.
Run the setup script separately if you want that full installation workflow.

Before Python exists, the shell bootstrap acquires the pinned interpreter.
Once Python can run, it hands dependency work to PM. PM's private engine
prepares its small, locked runtime before resolving application dependencies.
Its project is independent of the application workspace: a broken application
dependency must not prevent the dependency manager from starting.

uv is a private PM implementation detail. Application code, setup flows, and
build callers request Python operations, not uv executables or command arguments.
Do not mutate a Hermes environment with raw pip or uv commands.

PM's runtime contains `ruamel.yaml`, `packaging`, `tomli-w`, and `truststore`, not the application
dependency tree. CLI commands and application-requested installs and repairs run
there. Read-only path and installed-tool lookups remain local. Environment
currency checks use a ready PM worker. PM never adds its dependencies to an
already-running agent's imports. First-party YAML
readers and writers use ruamel; third-party packages can still require PyYAML in
the application environment. Failure receipts remain stdlib-only.

PM's CLI and worker activate `truststore` before importing their HTTPS clients.
This uses the platform certificate store even when bootstrap Python's compiled-in
OpenSSL paths do not locate it. No application dependencies or certificate-path
override are required. After the first install, PM rebuilds its small environment
against the managed Python on the next invocation; subsequent invocations reuse it.

`pm.venv_is_current()` checks through an existing PM worker, even when lazy
installs are disabled. It never bootstraps PM for a probe. If the manager
runtime is unavailable, it returns false without downloading tools or
dependencies. Run an explicit `hermes pm install` to prepare PM first.

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

Each activation invokes PM's install/sync path and trusts the recorded tool
digest instead of re-hashing every entry. PM still installs a missing tool and
rebuilds a stale dependency generation; a deliberate install keeps the byte
check. Run `python -m pm.cli install` or `hermes update` to re-check realized
bytes. A setup failure returns an error before changing the activated shell
environment, including when re-sourcing an already active environment.

After sync, activation prepends installed PM tools to `PATH` and sets
`PYTHONPATH` to this checkout and its selected dependency tree. It also
defines `hermes` as a shell function for this worktree. The function runs
this checkout's CLI and hides the installed command, including an MSIX alias.
It runs only while the shell is inside this worktree and refuses outside it,
so a sibling worktree does not inherit the command. The prompt gains a prefix
naming the branch, and drops it outside the tree. It does not
change an OS-wide PATH or install a conventional venv prompt.
Start in a clean shell rather than nesting this inside another venv.
`deactivate` restores the environment values captured by the activation script,
and removes the function and the prompt prefix.
It does not uninstall packages or stop processes that you started.

Verify the interpreter and source before doing work:

```bash
python -c "import sys, pm; print(sys.executable); print(pm.__file__)"
python -c "import httpx; print(httpx.__file__)"
node --version
npm --version
hermes --version
```

`python` must resolve to the PM store interpreter. `pm.__file__` must point
into this checkout. Dependencies come from the selected environment, which can
live outside the repository. A missing import means setup or selection needs
attention, even if `source ./activate` itself returned successfully.

### Work on this source tree

`hermes` is this worktree's CLI while the shell is inside it. Outside the
worktree the function refuses, so it cannot run another checkout's tree or
fall through to an installed command:

```bash
hermes setup
hermes
hermes --tui
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
python -c "from pm import sync_venv; sync_venv(['anthropic'], explicit=True)"
```

The first command installs a tool. The second adds a declared runtime extra
to this installation's existing Python selection. Extras accumulate through PM
sync. The `dev` and `test` dependency groups belong only to the separate test
environment, not the selected application venv. After changing extras, reactivate
before starting another Python process.

### Syncing after you edit pyproject.toml

1. Edit `pyproject.toml`. Pin every dependency as the
   [Dependency Pinning Policy](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md#dependency-pinning-policy)
   requires. Express platform limits with PEP 508 markers, or gate a whole
   extra in `[tool.hermes.extras-platforms]`.
2. Relock:

   ```bash
   hermes pm lock
   ```

   This re-resolves `uv.lock` from `pyproject.toml` with the same settings CI
   checks, including the 14-day `exclude-newer` quarantine. It changes no
   environment. When the lock is already current, it says so and writes
   nothing. (`hermes pm lock --bump NAME VERSION` is a different operation: it
   pins a managed tool in `pm/lock.json` and does not touch `uv.lock`.)
3. Source the activation script again (`source ./activate`, or
   `. .\activate.ps1` in PowerShell) to sync the application venv and the test
   interpreter to the new lock. Activation covers `[all]`. If you added an
   opt-in extra outside `[all]`, `hermes pm lock` prints the command that also
   puts it in the test interpreter, for example
   `source ./activate --test-extras all,NAME`.
4. Commit `pyproject.toml` and `uv.lock` together.

For JS dependencies, update the owning package manifest and lock. Do not edit
PM facts or generated workspaces, and do not install packages directly into a
selected generation.

### Test and editor environments

`source ./activate` (or `. .\activate.ps1` in PowerShell) and both direct
`setup-hermes` scripts prepare an isolated test interpreter from the locked
`dev` and `test` dependency groups. `scripts/run_tests.sh` uses that interpreter,
re-activating if the checkout or its dependency inputs changed. The application
venv, installers, and bundles select neither group. The developer default
covers `[all]`; to change test coverage, pass `--test-extras=anthropic` to POSIX
activation or `-TestExtras anthropic` to PowerShell;
those arguments select runtime extras *in the test interpreter only*.

In an isolated environment where activation is unavailable (for example, a Nix
dev shell), a caller can explicitly supply `HERMES_PYTHON` with pytest, or build
an independent disposable environment:

```bash
python -m pm.build_env --source . --out .venv --group dev --group test
```

The output must not exist. To regenerate it, stop its processes and intentionally
remove only that disposable environment first. Then run `scripts/run_tests.sh`
(through Bash on Windows). The test dependency group includes native launcher
tests and never enters a packaged runtime.

For editor debugging, select that independent interpreter, set the working
directory to this checkout, and launch `hermes` as the script. Keep its
`HERMES_HOME` separate from production. Terminal activation does not configure
an editor that was already running. Do not point an editor at a transient PM
generation or a signed application's Python executable.

### Python operation interfaces

Use the public `pm` module for Python dependency work:

| Operation | Ownership |
|---|---|
| `pm.sync_venv(extras, explicit=True)` | Prepare and select the complete application dependency union, including enabled plugins. |
| `pm.sync_venv(repair=True, explicit=True)` | Replay the recorded dependency set in a new application generation. |
| `pm.build_environment(source=..., out=..., explicit=True)` | Build and validate a fresh caller-owned output. No plugin discovery or application selection. |
| `pm.lock_project(source, explicit=True)` | Refresh an explicit project's lock without selecting an environment. |
| `pm.ensure_environment(name, requirements, explicit=True)` | Prepare and select an isolated dependency generation. Return its Python path. |
| `pm.ensure_python_tool(name, requirements, executable, explicit=True)` | Prepare an isolated tool and return its executable path. |
| `pm.environment_python(name)` / `pm.python_tool(name, executable)` | Read selected paths without installing anything. |
| `pm.venv_is_current()` | Ask a ready PM worker whether application dependencies are current. Return false if the manager runtime is unavailable. |

`pm.stage_manager_runtime(...)` is the bootstrap exception. It stages PM's own
locked runtime through the direct private engine because that runtime cannot
build itself through its worker. It does not expose uv to the caller.

`pm.build_env` is the command-line interface for explicit builds and lock work.
Run `python -m pm.build_env --help` for its supported options. By default, project
builds use the committed lock. `--resolve` resolves before building. `--python`
selects an explicit build interpreter. `--sealed` removes build-time `.pth`
references. `--offline` and `--cache` control dependency acquisition.

For application environment builds, PM must already be able to start in the
invoking Python. These builds are not an interpreter bootstrap. They do not modify a running application's
imports or replace its selected environment. Nix's declarative uv2nix builds
remain Nix-owned. Package-manager commands for unrelated projects or agent
sandboxes do not manage Hermes itself.

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
| `pm install [names...]` | Install named packages. With no names, provision required tools plus Python, put those tools on PATH, and then sync the `all` extra. A bare install also installs the default optional tools (`agent-browser` and Chromium); a failed download of these prints a warning and does not fail the install. Naming a package you declined earlier undoes that choice. |
| `pm install --without NAME` | Do a bare install without the default optional package `NAME` (only `agent-browser`), and record that choice. Later bare installs and `hermes update` also leave it out. The installers' `--skip-browser` / `-SkipBrowser` use this. |
| `pm install --tools-only` | Install that tool closure and put it on PATH, then stop. The venv sync does not run. |
| `pm env [names...]` | Print installed packages' PM-contributed environment values as JSON. It does not install missing packages, though a cold Hermes launch may prepare its own Python runtime first. |
| `pm doctor` | Check installed tool identities, files, and digests against the lock. |
| `pm repair` | Rebuild the recorded Python dependency set in a new generation, validate it, then select it. Does not update pins, features, or plugin configuration. |
| `pm status` | Print the latest sync/update receipt as JSON, or report that no receipt exists. |
| `pm gc` | Remove unreferenced tool-store entries, eligible download partials, and unused lease-managed Python generations. |

`pm env` excludes inherited process variables, including credentials. Its
output can still reveal local installation paths; review it before sharing.

### Maintainer commands

These commands change dependency inputs or stage build artifacts. They are
not substitutes for an installed application's update mechanism.

| Command | Effect |
|---|---|
| `pm lock` | Relock `uv.lock` from `pyproject.toml`. Changes no environment; writes nothing when the lock is current. |
| `pm lock --bump NAME VERSION` | Resolve and hash supported target artifacts, then write the tool pin to `pm/lock.json`. |
| `pm update [names...]` | Query upstream versions, change tool pins, and install changed tools. |
| `pm update --check` | Query without writing. Exit 1 can mean updates exist; inspect output to distinguish an error. |
| `pm update --target TARGET` | Resolve versions for the specified target. |
| `pm update --uv` / `--npm` | Also refresh the Python or npm dependency resolution. |
| `pm update --termux [--check]` | Repin the termux pool archives the rolling pool has retired (the runtime-lib pin table and the bionic lock rows). `--check` reports without writing and exits 1 when a pin is retired. |
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

- **Slow Python dependency builds:** PM's streamed uv commands enable verbose output. Bundle and build logs show package activity and build-backend stdout/stderr while the build runs, not only after failure.
- **Missing or outdated tool:** read `hermes pm doctor`, then use an explicit PM install on a writable installation.
- **New environment requires restart:** restart the affected Hermes process. Do not add a second site-packages tree to its live imports.
- **Dependency conflict:** read `hermes pm status`. Correct the plugin requirements before retrying admission.
- **Damaged Python dependencies:** run `hermes pm repair`, then restart Hermes. Repair replays the selected generation's saved workspace and lock without parsing plugin configuration. An unreadable record or missing saved lock fails without selecting a reduced dependency set. Before a generation exists, repair uses the shipped or committed lock and recorded feature set.
- **Interrupted dependency install:** startup requests the same PM repair before dependency activation. Automatic attempts are bounded; `pm repair` retries explicitly. A failed repair preserves the previous selection and its retry marker.
- **Damaged Python executable or application source:** repair or reinstall through the package owner. PM cannot run without those files. Signed payload files are never modified by dependency repair.
- **Unknown package or extra:** use the declared name. `pm install` takes package names, not Python extra names or pip specifications.
