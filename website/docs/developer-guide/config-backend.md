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
| `config_version(path)` | Cache signature (the file's stat tuple) |
| `config_exists(path)` | Whether the layer exists |
| `probe_config_readable(path)` | Raise `OSError` while the layer cannot be read (no parse) |
| `write_config_document(path, doc)` / `write_config_key(path, key, value)` | Writes; callers normally use `atomic_config_write` / `atomic_config_replace` |

Only `<home>/config.yaml` is a user layer. Any other file passed to these helpers (an import
source, a fixture) is read and written where it is. Caches keyed on `config_version` miss
whenever the backend's layer changes. The managed scope (`/etc/hermes`) stays an overlay on top
of whatever user layer the backend returns.

## Lints

- `scripts/check_config_yaml_writers.py` rejects a direct YAML dump of a config path.
- `scripts/check_config_yaml_readers.py` rejects a direct `open` / `read_text` / `stat` /
  `exists` / YAML load of a config path. A true false positive (another product's config
  file, or a tool that copies or edits the local file itself: `config edit`, backups, profile
  clone) carries `# config-reader: ok — <why>` on the line.
