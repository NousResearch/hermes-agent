---
sidebar_position: 12
title: "Config Backend Seam"
description: "Where a profile's user config.yaml is read and written: the ConfigBackend seam, its file backend, and the CI lints that keep every caller on it"
---

# Config Backend Seam

Every read, stat, existence check and write of a profile's user `config.yaml` goes through
`hermes_cli/config_backend.py`. `get_config_backend()` returns the backend for every home; today
that is `FileBackend`, the local `config.yaml` with exactly the primitives Hermes used before the
seam (stat signature, `fast_safe_load`, the ruamel round-trip writers). A backend that does not
store the layer as a local file implements the same `ConfigBackend` protocol, and every caller
already reaches it.

## Helpers

| Helper | Use |
|---|---|
| `read_config_doc(path)` | Parsed user layer (raises like `open` + `fast_safe_load`) |
| `read_config_doc_readonly(path)` | Signature-cached read; never mutate the result |
| `config_version(path)` | Change signature (the file's stat tuple) |
| `config_cache_version(path)` (`config_cache_version_or_none`) | Signature for a cache of the parse: the stat tuple plus a content digest while the later of mtime and ctime is younger than `utils.RACY_MTIME_WINDOW_NS` (`utils.file_cache_version` on the file backend) |
| `config_exists(path)` | Whether the layer exists |
| `probe_config_readable(path)` | Raise `OSError` while the layer cannot be read (no parse) |
| `write_config_document(path, doc)` / `write_config_key(path, key, value)` | Writes; callers normally use `atomic_config_write` / `atomic_config_replace` |

Only `<home>/config.yaml` is a user layer. Any other file passed to these helpers (an import
source, a fixture) is read and written where it is. A cache of a parse keys on
`config_cache_version`, not `config_version`: on kernels without multigrain timestamps a same-size
in-place rewrite in the tick of a read keeps mtime, ctime, size and inode, so the stat tuple alone
pins the older parse for the life of the process (a rotated `api_key` is the usual casualty).
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

## Lints

- `scripts/check_config_yaml_writers.py` rejects a direct YAML dump of a config path.
- `scripts/check_config_yaml_readers.py` rejects a direct `open` / `read_text` / `stat` /
  `exists` / YAML load of a config path. A true false positive (another product's config
  file, or a tool that copies or edits the local file itself: `config edit`, backups, profile
  clone) carries `# config-reader: ok — <why>` on the line.
