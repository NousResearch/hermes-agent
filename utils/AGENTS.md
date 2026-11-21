# utils/ — shared helpers that no package owns

Applies on top of the root `AGENTS.md`.

`utils/` is the bottom of the import graph: small helpers that several top-level packages call
and that know nothing about Hermes as a product. It is not the place for shared code in general,
and it is never the fix for an import cycle.

| Module | Subject |
|---|---|
| `files.py` | files on disk: atomic writes and replaces, temp files, file mode and owner, signatures and digests for caches, JSON and YAML file helpers |
| `urls.py` | base URL hostname, path, origin, host matching |
| `env.py` | truthy values, numeric env vars, proxy env vars |

## Before adding anything: three tests, all must pass

1. **Callers.** Two or more top-level packages call the code today; name them in the PR. A helper
   with one calling package stays in that package. "It will be reused" is not a caller.
2. **Subject.** The code fits an existing module, or it starts a module named for a subject a
   newcomer would guess (`processes.py`, `text.py`) with enough code to be worth a file. Refused
   names: `misc`, `common`, `helpers`, `shared`, `base`, `core`, `compat`.
3. **Product.** The code reads no config, no secret and no Hermes home, and its names and
   parameters mention no profile, session, agent, tool, gateway or plugin. The caller passes in
   the path, the value or the timeout. The one standing exception: the atomic writers in
   `files.py` call `hermes_constants.mkdir_under_hermes_home`, so a write never resurrects a
   deleted profile.

Code that fails a test has an owner elsewhere:

| Kind of code | Owner |
|---|---|
| Used by one package | that package |
| Hermes home and paths | `hermes_constants.py` |
| Host facts, executable lookup | `hermes_platform/` |
| Secrets and credentials | `agent/secret_scope.py` |
| Config reads and writes | `hermes_cli/config.py`, `hermes_cli/config_backend.py` |
| YAML parsing policy | `hermes_yaml.py` |

Before writing a helper anywhere, search here first: private copies of `read_json_or_empty`,
`env_int` and `atomic_json_write` were written all over the tree while these existed.

## Imports (HX018, blocking)

A module under `utils/` imports the standard library, third-party core dependencies, other `utils`
modules, `hermes_constants`, `hermes_platform` and `hermes_yaml`. It reaches no other
first-party module: not at module top, not inside a function, not under `TYPE_CHECKING`, not through
`importlib` and not through `sys.modules`, which reaches a module with no import statement at all.
Take the value or a callable as a parameter instead, or keep the helper in the package that owns
the knowledge.

A real exception carries `# health: allow HX018 -- <why>` on the line above. Today there are two,
both in the config writer's round-trip YAML functions in `files.py`.

"A and B import each other" is not a reason to move code here. Move it to the lower of the two
packages, or pass the value in.

## Shape of a module

- One subject per module. Functions and small frozen dataclasses: no registry, no singleton, no
  background thread, no work at import time, no module-level mutable state.
- New callers import from the module: `from utils.files import atomic_replace`. `utils/__init__.py`
  is a docstring plus re-exports that keep the existing `from utils import name` callers and the
  names published as `utils:<name>` in `hermes_cli/data/plugin_api.json` working. Add no re-export
  for a new helper.
- `files.py` imports `hermes_yaml` (ruamel) at module top, so importing anything from `utils` loads
  it. Code that must run on the standard library alone (the updater takeover, `pm`) does not import
  `utils`.
- A helper leaves when it gains an owner. When it starts to need product knowledge, move it to the
  package that owns that knowledge; never add a `profile` or `config` parameter to keep it here.

## Tests

`tests/utils/`. Patch a name in the module that defines it (`utils.files.os.replace`), not on the
package: production reads it there, and a patch left on `utils` silently stops intercepting. The
exception is code under test that runs `from utils import name` inside a function.
