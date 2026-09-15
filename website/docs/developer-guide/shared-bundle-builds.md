# Shared bundle builds

Hermes separates dependency preparation, product builds, and distribution
packaging. The compiler and agent-assembly interfaces live in
[`scripts/build/README.md`](../scripts/build/README.md). These are current
interfaces, not proof that every distribution passed native acceptance.

## Providers, products, and distributions

| Layer | Responsibility | Implementation |
|---|---|---|
| Dependency providers | Prepare tools, Python environments, JavaScript dependencies, and native bindings | PM (`pm.build_environment`), `scripts/build/node-deps.mjs`, Nix, and Termux |
| Product builders | Compile icons/TUI/web/desktop UI or assemble a runnable agent from prepared inputs | `scripts/generate_icons.py` and `scripts/build/` |
| Distribution adapters | Select products and package them for their target | `scripts/bundles/`, `Dockerfile`, `nix/`, and `scripts/termux/` |

Providers retain their package managers and locks. Product builders do not
install missing dependencies or download inputs. The Node dependency provider
uses one locked workspace union. Nix supplies its own npm and Python inputs.
Termux supplies bionic tools and wheels. Shared recipes do not imply identical
bytes, dependency selections, or Python environments across targets.

| Product | Implementation | Inputs |
|---|---|---|
| Icons | `scripts/generate_icons.py` | Artwork and prepared `icon-build` environment |
| TUI | `scripts/build/tui.mjs` | Prepared TUI workspace |
| Dashboard | `scripts/build/web.mjs` | Prepared web workspace and generated icons |
| Desktop UI | `scripts/build/desktop.mjs` | Prepared desktop workspace, icons, stamp, and native bindings |
| Runnable agent | `scripts/build/agent.py` | Code, interpreter, dependencies, independent PM runtime, resources, and selected frontends |

The desktop UI compiler does not depend on the dashboard. A bundled desktop
selects an agent with TUI and dashboard products. A light desktop does not
select an agent payload.

## Build and packaging entrypoints

Source updates and source UI launches use the same dependency provider and
product builders. `hermes_cli/source_build.py` selects the workspace union,
prepares it once, then invokes the shared recipes. An update builds TUI and
web, plus the local desktop app when one was present before the update.
Desktop packaging still belongs to the local desktop adapter.

The current-checkout retry, pulled-checkout, and ZIP update paths all use
`update_cmd_maint._prepare_updated_checkout`: one PM dependency sync, then a
fresh process on the selected interpreter for the frontend builds. Build
failure aborts completion; an existing stale product is not a successful
update. There is no updater-specific npm cache, fallback install, extra
refresh, or memory-provider reinstall. PM owns the complete Python union.

Build a desktop distribution from a clean checkout at its release tag. Start
with a host Python that can run the entrypoint and Git; preparation acquires the
pinned Python, Node and npm through PM. Native compilers/SDKs remain host inputs:

```sh
python -m scripts.bundles.desktop --tag=vX.Y.Z
```

Without `--prepared`, `desktop.py` requires exactly one of `--tag` or `--commit`. Commit builds
require the full commit SHA and matching checkout HEAD. `--variant` accepts
`bundled`, `store`, or `light`, with `bundled` as the default. `--repo` selects
the checkout. Arguments after `--` go to the prepared Electron Builder wrapper;
they cannot replace the admitted target, tools, output or packaging configuration.

The driver first prepares the complete dependency set: managed tools, the locked
Node workspace union, icon environment, Electron-native bindings, packaging
utilities, and (except for light) the application and independent PM environments.
Only then does it compile products, assemble the payload, and package Electron.
MSIX metadata, signing, notarization, and package formats remain adapter work.

### Split desktop preparation and consumption

The one-command build and CI use the same preparation operation. To stop before
product compilation and packaging:

```sh
python scripts/bundles/desktop.py --tag vX.Y.Z --variant bundled --prepare-only \
  --work "$PWD/.build/desktop-job" --cache "$PWD/.cache/desktop-inputs"
python scripts/bundles/desktop.py --prepared "$PWD/.build/desktop-job/prepared.json"
```

`--work` and `--cache` are separate, build-owned directories. Preparation claims
the work directory itself; do not create it beforehand. It writes `prepared.json`
last, after every provider succeeds. That file contains this job's absolute paths
and source identity, not a portable cache receipt. Reprepare after moving a
checkout, changing source or locks, or losing an input. Repeated preparation may
reuse admitted dependency bytes while rebuilding path-bound environments.

`--prepared` does not accept new tag/commit/work/cache arguments. It validates
the clean checkout and prepared inputs, then fails on stale or missing inputs
instead of installing them. Bundled and Store builds may consume the same
preparation for a stable tag; light and commit builds cannot switch to Store.
Product builds still run each time. Native staging exposes `prepare_native` and
`finish_native` to this composition; `hermes pm bundle` remains a complete
native staging command, not the desktop preparation interface.

Windows and macOS release jobs use **restore → prepare → save → build**. The
cache action derives provider paths and transports candidates; it neither
creates the preparation workdir nor declares a cache hit valid. Saves run after
successful preparation, before product compilation or signing. Failed
preparation leaves completed provider data locally but does not save an overall
snapshot. The general `setup-pm` action remains available for other workflows.

Native wheel caches use compact 128-bit partitions under `python/runtime`, keyed
by the target and observed compiler/SDK inputs. Missing identities still get a
fresh partition. uv builds source distributions inside this cache, so partition
names must leave room for nested compiler outputs below Windows `MAX_PATH`.
Keep a custom `--cache` root short as well; enabling OS long paths does not make
every native compiler long-path aware. This layout retains the shared cache
transport and all target-compatible extras, including Silk.

The strict boundary forbids dependency acquisition during consumption, not all
network access: signing, timestamping, notarization and publication retain their
online responsibilities. A network-denied unsigned native build is still needed
to prove the boundary on each release target.

Stage a native agent with both frontend products, without Electron packaging:

```sh
python -m scripts.bundles.stage --out /work/agent-payload --ref HEAD
```

This command prepares and builds TUI/web products if none are supplied. It
also accepts `--tui` and `--web` together as prepared product roots. It rejects
a selection with only one of those arguments. The standalone products have
explicit input/output contracts in the builder README.

`--ref` selects the agent source snapshot. Automatic frontend compilation uses
a temporary snapshot of that same revision. Explicit frontend products must
come from the selected revision.

`hermes pm bundle --out /work/agent-payload --ref HEAD` stages native tools,
the application environment, and the agent. That PM command does not build
frontends. `npm run payload --workspace apps/desktop` uses the stage driver
that includes them. Neither command creates an Electron installer.

### Split PM Bundle preparation and assembly

PM Bundle uses the same isolated tool bootstrap and native preparation slice,
without installing desktop workspaces, icons, Electron bindings or packagers:

```sh
python -S -B scripts/bundles/native_build.py --source "$PWD" \
  --work "$PWD/.build/payload-job" --cache "$PWD/.cache/payload-inputs" \
  --out "$PWD/build/agent-payload" --ref HEAD --prepare-only
python -S -B scripts/bundles/native_build.py \
  --prepared "$PWD/build/agent-payload.prepared.json"
```

Use a clean checkout at the selected revision. `--ref` defaults to `HEAD`;
`--commit` accepts an exact full SHA instead. Work, cache and output must be
separate; preparation claims a previously absent work directory. Without
`--prepare-only`, the driver also assembles. `--prepared` takes no new selection
or path arguments and validates the native receipt before assembly; it never
bootstraps or repairs dependencies. The receipt and lock are output siblings
(`agent-payload.prepared.json`, `agent-payload.prepare.lock`), not shipped files.
Assembly deliberately supplies no frontend products, as with `hermes pm bundle`.

The `payload-test` producer on the existing cache action selects only `tools`,
`python/runtime` and `native`. It excludes source `node_modules`, npm caches,
icon environments and packagers. PM's managed tool bootstrap still includes Node
and npm, but does not run desktop `npm ci`. Native compiler identity and Windows
ARM64 prerequisites use the shared owner once. Cache hits always undergo provider
admission; native SDK/compiler prerequisites still belong to the host. PR jobs
only restore; saves require successful preparation on the default-branch trusted
lane with no alternate source ref. No signing or publication is added.

For Termux, tools and the wheelhouse are prerequisites:

```sh
python scripts/termux/build.py --repo /work/source --payload /work/termux-payload \
  --out /work/packages --tag vX.Y.Z
```

The Termux driver accepts exactly one of `--tag` or `--commit`. It prepares
the TUI workspace, calls the shared TUI builder, and passes that product to
`build_deb.sh`. It does not add the dashboard or replace bionic preparation.

The CLI contracts are in `scripts/bundles/desktop.py`,
`scripts/bundles/stage.py`, `pm/cli.py`, and `scripts/termux/build.py`.

## Shared agent and launcher contract

`AgentInputs` supplies explicit paths and a placement mode:

- `contained` keeps runtime inputs inside the native payload.
- `fixed` retains the Docker or Termux installation prefix.
- `references` keeps installed code and dependencies in separate Nix store paths.

The assembler derives console commands from `[project.scripts]`. It copies
source-layout resources and selected frontends, or links explicit reference
inputs. It writes the completion manifest after structural assembly. Nix also
receives a command/environment map for its native wrappers.

The launcher implementation lives in `scripts/build/launchers.py`.
`launcher_wrapper.py` and `mint_launchers.py` live in that same directory.
`scripts/bundles/payload.py` owns git snapshots, PM tool facts, PM runtime
sealing, and portable link handling. It no longer owns frontend placement or
launcher generation.

The desktop stamp carries the declared launch paths. Electron consumes that
contract without payload adoption or repair. Non-bundled builds carry no
placeholder agent payload. Store packages declare `microsoft-store` as their
update mechanism. Sideload bundles declare `app-installer`.

Windows launcher minting runs the payload interpreter. POSIX launchers use
explicit interpreter, source, and dependency paths. Structural assembly does
not replace target-native launch tests. A manifest alone does not prove that
a relocated or signed artifact runs.

## Independent PM runtime

PM dependencies come from `pm/pyproject.toml` and `pm/uv.lock`, not the
application dependency graph. The application environment and `pm-runtime`
are separate inputs to agent assembly.

Native and Docker preparation use `pm.runtime_stage.stage_runtime`. Termux
uses that function with its offline wheelhouse. Nix uses the independent
`nix/pm-runtime.nix` derivation. Native sealing records the payload interpreter
and PM-only site directory in `pm-runtime.json`.

Resident PM workers run the declared interpreter with `-I -S -B`. They add
only the declared PM dependency directory before the worker script. They do
not borrow application dependencies or add PM packages to the caller's
interpreter. Docker and Nix stamps point to their declared PM runtime.
These contracts live in `pm/runtime.py:55–88,171–182` and
`scripts/bundles/payload.py:58–90`.

Ordinary native staging does not scan user plugin trees. It runs provisioning
in a child with temporary home and PM roots plus an explicit persistent build
cache. Windows ARM64 prerequisite preparation runs before HOME isolation and
uses the same provider as source setup and CI. Live PM retains plugin
admission, generation selection, and transaction state. Both paths use
`pm.environment.PythonEnvironment` for explicit uv environment construction.

## Distribution boundaries and output paths

| Distribution | Selected products | Current output layout |
|---|---|---|
| Desktop bundled/Store | Icons, TUI, web, agent, desktop UI | Products: `apps/desktop/build/products/`. Agent: `apps/desktop/build/agent-payload/`. UI: `apps/desktop/dist/`. Packages: `apps/desktop/release/` |
| Desktop light | Icons and desktop UI | `apps/desktop/dist/` and `apps/desktop/release/`, without an agent payload |
| Docker | Icons, TUI, web, agent | Agent at `/opt/hermes`, dependencies at `.venv`, PM at `pm-runtime`, generated commands at `libexec` |
| Nix TUI | TUI | `$out/lib/hermes-tui/{dist/entry.js,package.json}` |
| Nix web | Web | `$out/index.html` and assets |
| Nix agent | Agent with TUI/web references | `$out/bin`, `$out/share/hermes-agent`, `$out/ui-tui`, `manifest.json`, and `command-map.json` |
| Nix desktop | Icons and desktop UI, with the Nix agent | `$out/share/hermes-desktop` and `$out/bin/hermes-desktop` |
| Termux | TUI and agent | `OUT/hermes-agent_<version>_aarch64.deb`, installed at `$PREFIX/lib/hermes-agent/` |

Native payloads contain `hermes-agent`, `tools`, `venv`, `pm-runtime`, `bin`,
`uv-cache`, and their manifests/facts. Copied frontend assets live at
`hermes-agent/hermes_cli/tui_dist/` and `hermes-agent/hermes_cli/web_dist/`.
The TUI asset directory contains `entry.js` and module-mode package metadata.

Docker keeps its existing TUI product at `/opt/hermes/ui-tui` and web output
at `/opt/hermes/hermes_cli/web_dist`. The assembler also plants its shared TUI
asset layout beneath `hermes_cli/tui_dist`. Venv command symlinks preserve the
paths used by s6 and the privilege-drop shim.

The Docker frontend stage owns npm dependencies and compilation. The runtime
stage copies only the frontend products and the locked TypeScript package for
runtime linting. Node/npm and Photon's separately installed sidecar dependencies
remain runtime inputs. Python dependency preparation precedes the application
source copy. Actual cache-hit and layer-size claims require build evidence.

Nix retains `importNpmLock`, uv2nix, platform overrides, and store references.
Its builders run during derivation builds, not evaluation. Per-product source
filters keep frontend, Python, and resource inputs separate. Nix wrappers
consume the assembler's command map and retain their PATH and extra-Python
collision policies.

**Nix wheel policy:** `nix/python.nix:128–135` sets `HERMES_NIX_BUILD=1` only
for the Hermes derivation. `setup.py:34–72` rejects general Hermes wheel/sdist
builds. The shared assembler uses the installed Nix code without another copy.
Other providers use source-layout code and metadata, not a public Hermes wheel.

Termux retains bionic wheel compilation, offline installation, native library
paths, and its fixed prefix. Its installed root contains `app`, `tools`,
`runtime-libs`, `venv`, `pm-runtime`, `bin`, and manifests/facts. APT maintainer
hooks manage the declared CLI symlinks under `$PREFIX/bin` and refuse foreign
conflicts. They do not compile dependencies during installation.

`stage_apt_repo.py` owns repository metadata and signatures. Stable and canary
suites are `hermes-stable` and `hermes-canary`. Package files publish before
signed metadata. [Stable release admission](stable-releases.md) coordinates
acceptance and publication across distributions.

## Public artifact handoffs

The desktop release workflow downloads and installs each supported bundled
format on a fresh native runner, then runs the same composer/provider/reply
check as install-e2e. Windows universal assembly stages bytes before the smoke;
its canary feed is published separately only after both native smoke matrices
pass. macOS feed publication and stable candidate acceptance are likewise gated.
See [install and chat acceptance](../tests/install/README.md#post-build-artifact-smoke)
for covered formats, checkpoint evidence and the Store/Linux/no-upload limits.

`scripts.releases.handoff fetch --public-base URL` downloads a staged artifact
without R2 credentials. Supply either `--tag TAG --commit SHA` or
`--commit-build SHA`, plus the producer `--name`, destination `--root`, and
optional `--include` selectors. This uses the same receipt identity, path,
size and SHA-256 checks as authenticated handoffs. Tagged and commit-only
archives remain separate; the command does not resolve a mutable latest feed.

Public reads require HTTPS, except for loopback fixture servers. They reject
URL credentials, unsafe paths and redirects. An incomplete or corrupt download
cannot replace an existing verified destination. Receipt files are saved only
after all selected artifacts verify. Selection of one desktop format must still
reject missing or ambiguous matches before installation.

## Cache ownership

PM binary download archives and the native uv wheel cache have different
consumers. They are not interchangeable cleanup targets.

- PM retains completed downloads until package verification and publication.
  Successful installs remove their exact fetch archives. Failures retain
  downloads for retry. Cleanup leaves unrelated partials alone.
- Native staging prunes obsolete entries only in its build-owned tool store.
  It does not prune the user's machine-wide store.
- Native staging retains `uv-cache/` for offline mutable-environment rebuilds.
  It copies extracted wheels and index/revision metadata, but excludes redundant
  wheel ZIPs and cached sdist `src/` trees (including Rust `target/` outputs)
  before copying. The provider cache is unchanged; native signing reaches the
  extracted binaries without scanning build-only artifacts.
- PM-runtime and application builds share the provider's persistent uv cache.
  CI restores/saves that cache, not the temporary build HOME. Failed builds
  retain completed wheels. General PM staging uses its v2 cache namespace;
  desktop preparation uses its own input snapshot namespace.
  Plain `uv cache prune` removes dangling entries without discarding offline
  wheel inputs. It does not remove all historical versions or enforce a size cap.
- Standalone icon preparation uses `SOURCE/.cache/icon-build`. Desktop
  preparation places its icon environment under the job workdir and its wheel
  cache under `CACHE/python/build`. Neither enters the shipped runtime.
- Frontend `node_modules` is a provider input, not a frontend product.
  Docker's runtime TypeScript and Photon selections are separate exceptions.

The native cache behavior is in `scripts/bundles/native.py`.
Removing all caches from a payload can break offline environment reconstruction.

Desktop transport selects PM tool entries, Python download/wheel caches, npm's
content-addressed downloads, prepared workspace dependencies, and native/package
tool inputs through `scripts/ci/desktop_build_cache.py`. It does not select the
workdir, virtual environments, live user home, signing tokens, or products.
The signature-result cache is separate. Native wheel partitions depend on
observed compiler/SDK/OpenSSL identity; incomplete identity deliberately forgoes
warm reuse. This is not a fully pinned host SDK.

Cache keys are lookup hints, not authorization. Native builders have separate
release and commit execution jobs with literal `cache-mode: write` and
`cache-mode: read`, respectively. GitHub enforces these permissions on scoped
cache tokens, so commit builds cannot save caches even from their own scripts.
Both branches share their matrix, environment, and steps through YAML anchors.
The original `build-win32` and `build-darwin` IDs aggregate the branches: after
successful validation and input archiving, exactly the selected branch must
succeed and the other must be skipped. A failed, cancelled, or unexpectedly
skipped selected build fails the aggregate; it cannot permit publication.
Existing publication dependencies keep using those IDs. A skipped save or a
different key prefix alone cannot prevent a script from poisoning a writer
namespace.
Default-branch canary dispatch retains default-branch cache scope; changing a
checkout SHA does not change the workflow ref's cache scope. See GitHub's
[cache access contract](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#controlling-cache-access-with-cache-mode).

## Pinned binary inputs

`python -m scripts.ci.archive_inputs` preserves every HTTPS artifact in
`pm/lock.json`, across all targets, plus the Termux runtime-library and license
pins. `pm/artifact-mirror.json` owns the public mirror location. Object keys
are `upstream/sha256/HASH`, independent of filenames and release tags.

CI requests R2 first. Only 404 permits an upstream download. It verifies the
existing SHA256 before an immutable `If-None-Match: *` upload, then downloads
and verifies the stored object. Corrupt bytes, denied access, and failed
uploads stop the build. Concurrent writers may reuse identical bytes but
cannot replace an existing object.

The all-target workflow runs on pin changes on main, manual dispatch, and as
an admitted release prerequisite. It uses runner Python before the pinned
toolchain is available. Desktop jobs depend on that prerequisite and then let PM
fetch verified inputs without archival write credentials; they do not need a
second cache-seeding lane. Other protected workflows can opt into archival
through `setup-pm`'s `archive-inputs` input. Targeted archival can seed disposable
fetch entries with `--target` and `--store`; Termux also passes `--payload` for
runtime libraries. Cache hits do not skip preservation. Desktop R2 credentials
are scoped to upload steps. Untrusted PR jobs receive no publication credentials.

Installed PM clients, bootstrap installers, and Nix pin consumers use the
primary URL followed by the public mirror if the download is unavailable.
The pinned hash remains binding; no credentials or uploads are needed by
clients. PM keeps per-source resume state and reports attempted URLs.
For Termux files already removed upstream, the CI publisher can recover the
exact bytes from the community Internet Archive after an upstream 404/410.
It never repins to the latest package.

The archive must have no expiration lifecycle rule. Release pruning does
not cover its prefix. This covers PM binary pins and Termux runtime inputs,
not unpinned apt packages, OCI images, language-package registries, or native
Electron/SDK archives independently downloaded by their build tools.

## Verification boundary

The checks below distinguish product tests from distribution acceptance.
Signed installers and Android device execution require their native runners.

The macOS dmgbuild acquisition owner is part of packaging preparation, not the
strict build. Native DMG creation, detach diagnostics, signing and notarization
must be exercised after tool-provider changes. Windows x64/ARM64 and
macOS x64/ARM64 cold/warm and signed-artifact acceptance remain separate from
Linux helper tests. Linux desktop release lanes remain disabled.

Focused helper tests cannot establish all of these requirements:

- Offline frontend compilation from prepared immutable inputs.
- Standalone TUI interaction and dashboard backend/assets behavior.
- Native Electron bindings and final signed launchers on each target.
- PM isolation after native payload relocation.
- Retained-cache offline reconstruction of a mutable application environment.
- Docker non-root CLI/TUI/web/browser behavior and every image layer's contents.
- Actual Nix package execution through reference-based wrappers.
- Fresh network-disabled Termux package installation and Android device behavior.

Use the existing acceptance checks in `tests/install/BUNDLED_UPDATES.md`,
`tests/docker/`, Nix checks, and `scripts/termux/check_deb.sh` plus
`validate_installed.py`. Container acceptance does not substitute for Android
device acceptance. No build-pass claim follows from this document.
