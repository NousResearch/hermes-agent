---
sidebar_position: 12
title: "Config Backend Seam"
description: "Where a profile's user config.yaml is read and written: the ConfigBackend seam, the file and remote backends, and the CI lints that keep every caller on it"
---

# Config Backend Seam

Every read, stat, existence check and write of a profile's user `config.yaml` goes through
`hermes_cli/config_backend.py`. The backend is chosen once per process from
`HERMES_CONFIG_BACKEND`, never from a config value: `file` is the default, and `remote` is
[Remote Config](../user-guide/remote-config.md) (`plugins/config_backends/remote/`).

## Helpers

| Helper | Use |
|---|---|
| `read_config_doc(path)` | Parsed user layer (raises like `open` + `fast_safe_load`) |
| `read_config_doc_readonly(path)` | Signature-cached read; never mutate the result |
| `config_version(path)` | Change signature (the file's stat tuple, or the remote version) |
| `config_cache_version(path)` (`config_cache_version_or_none`) | Signature for a cache of the parse: the stat tuple plus a content digest while the later of mtime and ctime is younger than `utils.RACY_MTIME_WINDOW_NS` (`utils.file_cache_version` on the file backend); the remote version on the remote backend |
| `config_exists(path)` | Whether the layer exists (a remote layer always does) |
| `config_version_or_none(path)` / `version_token(version)` | The version, or None when absent; a number that changes with it (TUI/session `mtime`) |
| `probe_config_readable(path)` | Raise `OSError` while the layer cannot be read (no parse); a remote backend raises `ConfigBackendUnavailable` while the plane cannot serve it |
| `write_config_document(path, doc)` / `write_config_key(path, key, value)` | Writes; callers normally use `atomic_config_write` / `atomic_config_replace` |
| `supports_file_tooling()` / `require_file_tooling(what)` | Gate for tools that copy, edit or back up the file itself (`config edit`, backup/restore, profile clone) |

Only `<home>/config.yaml` is a user layer. Any other file passed to these helpers (an import
source, a fixture) is read and written where it is, whichever backend is selected. A cache of a
parse keys on `config_cache_version`, not `config_version`: on kernels without multigrain
timestamps a same-size in-place rewrite in the tick of a read keeps mtime, ctime, size and inode,
so the stat tuple alone pins the older parse for the life of the process (a rotated `api_key` is
the usual casualty).
Any other file a cache derives from (`auth.json`, pairing lists, `.env`, token files, the managed
overlay) keys on `utils.file_cache_version(path)`; a signature stored and compared much later (a
change detector whose false positive reconnects something, a parse-failure record) passes
`always_digest=True`, because the racy digest turns empty once the file settles (the skills
manifest, hook-consent, free-tier and onboarding stamps do). "Recent" is judged on ctime as well as
mtime, because `cp -p` or `rsync --inplace -t` restore an older mtime after writing. A worker bound to
a frozen config snapshot never hashes a `config.yaml`: its key stays stat-only there, so the content
check never makes such a worker open the profile file. Caches keyed on `config_cache_version` miss
when a remote poll installs a new layer, exactly as they do after a local file edit. With the file
backend, the managed scope (`/etc/hermes`) stays an overlay on top of the user layer. The remote
backend ignores the managed `config.yaml`: config and locks come only from Remote Config, and
`hermes doctor` flags the file if it exists. The managed `.env` still applies in both modes.

Related keys that must change together (a model switch's `model.*` keys, the TUI `/focus`
toggle) go through `write_config_keys(path, {key: value, ...})`. A remote backend sends them
as one CAS write, so a lock on any of them refuses all of them.

## Public read/modify/write API for editors and plugins

`hermes_cli.config.update_user_config(mutate)` is the supported entry point for
additive user-layer edits that need the current document (for example the
Advanced Delegation plugin's named preset registry). It is a mutation operation,
not a public lock: core owns the guarded read, callback and backend commit.

```python
from hermes_cli.config import update_user_config


def select_backend(document):
    delegation = document.setdefault("delegation", {})
    delegation["backend"] = "model_map"


update_user_config(select_backend)
```

The active profile is resolved at invocation from `get_hermes_home()`, including
host-bound profile context. No home/path argument or process environment rewrite
is needed. The callback receives the **raw user layer**, without defaults, managed
overlays or environment expansion; use an effective loader separately for runtime
reads. It runs once and mutates the document in place; its return value is ignored.
Do not retain the document, change profile scope, or perform nested config writes
inside the callback. An exception aborts the commit and propagates to the caller.

Core refuses package-manager-managed edits before invoking the callback, reads
through the selected config backend, and commits through `atomic_config_write`.
Missing layers begin as `{}`; unreadable, malformed and non-mapping existing layers
fail closed. Existing mapping paths cannot be removed by omission. File writes
preserve YAML comments; remote writes retain backend locks, validation and conflict
errors (`ConfigWriteError`). `ConfigBackendUnavailable` is a `SystemExit`, not an
`Exception`; daemon callers must handle it explicitly when needed.

**Concurrency boundary:** the read/callback/commit is serialized against cooperating
writers using core's config lock **within the calling process only**. Dashboard
config mutation scopes use that same reentrant lock across their full read/modify/
write interval; individual core reads and saves can safely re-enter it. Dashboard
write scopes acquire profile scope before the config lock. Mutation callbacks must
not enter profile scopes or perform nested writes, which would violate that order.
File-backend publication is atomic, but this is not file-backend cross-process CAS
or a gateway RPC. Hosted plugin callbacks may run in a different process from the
gateway or other editors; this API does not serialize those processes or other
non-cooperating writers. Do not use it to promise multi-process isolation. Use a
backend with conflict detection when that guarantee is required, and surface
conflicts rather than automatically replaying a callback with side effects.

This API is introduced by this change, with no released minimum version assigned.
Plugins supporting older core builds should feature-detect the callable. Existing
editors may retain their already-supported guarded read plus `atomic_config_write`
path, without claiming whole-operation serialization on those builds; new workflows
that require that guarantee should refuse with an actionable upgrade message.
Do not fall back to private locks, private dashboard write scopes, direct YAML
writes, or persisting merged defaults. Never retry through a fallback after this
API's callback or commit raises: the error must reach the caller.

## Lints

- `scripts/check_config_yaml_writers.py` rejects a direct YAML dump of a config path.
- `scripts/check_config_yaml_readers.py` rejects a direct `open` / `read_text` / `stat` /
  `exists` / YAML load of a config path. A true false positive (another product's config
  file, or file tooling already gated on `supports_file_tooling()`) carries
  `# config-reader: ok — <why>` on the line.
