# Application declarations

A plugin whose MCP server fronts a desktop application declares which application that is and what the server needs of it. The core evaluates the declaration on the host and gates the server's tools, and any skill that names the application, on the answer. The parser imports only the standard library and `hermes_platform`.

The vocabulary lives in `hermes_platform/declaration.py`. A declaration is data plus policy, parsed from plain mappings (already-decoded YAML, JSON, a dict literal — the parser never touches a file):

```python
from hermes_platform import declaration

decl = declaration.parse_declaration(
    "my-server",
    raw_app={"linux": {"presence": "executable", "location": "/opt/my-app/server"}},
    raw_requires={"app": True},
    where="my-plugin/plugin.yaml",   # human label used in error messages
)
declaration.register("my-server", decl)
```

`register(server_name, decl)` stores one declaration under the configured server name in a process-local registry. Loader integration is separate work; core does not read plugin YAML automatically.

An unregistered MCP server keeps its connection-only check. A skill that explicitly names an unregistered server is hidden. `clear()` removes every registration and is not a per-plugin unload operation. Registrations are process-wide, not profile-scoped.

## `app` — how to find the application on each OS

```yaml
app:
  win32:
    presence: executable
    location: "%ProgramFiles%/Vendor/Vendor App/McpServer/Server.exe"
    version: { kind: uninstall_registry, display_name_prefix: "Vendor App" }
    liveness:
      kind: server_json
      path: "%LOCALAPPDATA%/Vendor/Vendor App/McpServer/server.json"
      pid_key: pid
      url_key: http
      token_key: token
      endpoint_path: /mcp
  darwin:
    presence: bundle
    location: /Applications/Vendor.app
    version: { kind: plist }
```

| field | type | rule | maps to `AppDef` |
|---|---|---|---|
| `<os>` | `win32` \| `darwin` \| `linux` | at least one; unknown key is an error | `AppDef.os_family` |
| `presence` | `executable` \| `bundle` | required per OS | `.presence` |
| `location` | str | required; drive-rooted (`C:\\...`) on Windows, or starting with `~` / `%VAR%` / `$VAR`; UNC paths are rejected so a presence check never touches the network; no `..` segment or URL scheme; expansion at lookup | `.location` |
| `version.kind` | `pe_resource` \| `plist` \| `uninstall_registry` \| `none` | default `none`; `pe_resource`/`uninstall_registry` only under `win32`, `plist` only under `darwin` | `.version_kind` |
| `version.display_name_prefix` | str | required when `uninstall_registry` | `.version_arg` |
| `liveness.kind` | `server_json` \| `none` | default `none` | `.liveness_kind` |
| `liveness.path` | str | required when `server_json` | `.liveness_path` |
| `liveness.pid_key` / `url_key` / `token_key` | str | defaults `pid` / `http` / `token` | `.liveness_*_key` |
| `liveness.endpoint_path` | str | default `/mcp`; the path used for `initialize`, never the one in the file | `.endpoint_path` |

When `requires.app` is true, an OS missing from `app:` gives `unsupported_os`.

## `requires` — what the server needs before it is offered

```yaml
requires:
  app: true
  min_version: "2.3.0"
```

| field | type | rule |
|---|---|---|
| `app` | bool | when true, `app:` must exist and the server is gated on presence |
| `min_version` | str | requires `app: true`; dotted numeric; every applicable `app.<os>` must declare a real `version.kind`; compared numerically per segment, non-numeric characters in a segment are dropped (`2.3.0.12594` ≥ `2.3.0`; prerelease suffixes are not ordered) |

`requires.app: true` with no `app:` block is a `DeclarationError`.

## Availability: the one evaluation every reader uses

`hermes_platform/resolver/availability.py::availability(decl) -> Availability`

```
Availability(
  state:   available | installed_not_running | missing_app | version_too_old
         | unsupported_os | no_requirements,
  version: str | None,       # inspected, when present
  path:    str | None,       # where the app was found or looked for
  min_version: str | None,   # from requires
)
```

- `no_requirements`: no `requires.app`; the application gate passes, but the connection check still applies.
- `unsupported_os`: `requires.app` and no `app.<this os>` block. Zero I/O.
- `missing_app`: `locate` found nothing at `location`.
- `version_too_old`: the version is below the minimum or cannot be read.
- `available`: present, version acceptable or not required.
- `installed_not_running`: reserved vocabulary; this evaluator never produces it.

Evaluation uses `locate` and optional version inspection. It never probes a server, launches an application, or connects. The tool registry retains its existing availability cache.

## The two gates

- **MCP `check_fn`** (`tools/mcp_tool_handlers.py::_make_check_fn`): connection alive AND, when a declaration with `requires.app` is registered for the server, `availability(decl).offerable`. Returns a plain `bool` because the registry caches `bool(fn())`.
- **Skill `requires_apps:` frontmatter** (`agent/skill_utils.py::skill_matches_apps`): each name resolves through `declaration.lookup`; an unknown name hides the skill (fail closed). Offer-time filter, like `environments:`.
