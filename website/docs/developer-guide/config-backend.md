---
sidebar_position: 12
title: "Config Backend Seam"
description: "Where a profile's user config.yaml is read and written: the ConfigBackend seam, the file and remote backends, and the CI lints that keep every caller on it"
---

# Config Backend Seam

Every read, stat, existence check and write of a profile's user `config.yaml` goes through
`hermes_cli/config_backend.py`. The backend is chosen once per process from
`HERMES_CONFIG_BACKEND`, never from a config value: `file` (the default) is the only backend in
this build; any other value stops the process with a clear error instead of falling back to defaults.

## Helpers

| Helper | Use |
|---|---|
| `read_config_doc(path)` | Parsed user layer (raises like `open` + `fast_safe_load`) |
| `read_config_doc_readonly(path)` | Signature-cached read; never mutate the result |
| `config_version(path)` | Cache signature (the file's stat tuple) |
| `config_exists(path)` | Whether the layer exists |
| `write_config_document(path, doc)` / `write_config_key(path, key, value)` | Writes; callers normally use `atomic_config_write` / `atomic_config_replace` |
| `supports_file_tooling()` / `require_file_tooling(what)` | Gate for tools that copy, edit or back up the file itself (`config edit`, backup/restore, profile clone) |

Caches keyed on `config_version` miss whenever the backend's layer changes. The managed scope (`/etc/hermes`) stays an overlay on top
of whatever user layer the backend returns.

## Lints

- `scripts/check_config_yaml_writers.py` rejects a direct YAML dump of a config path.
- `scripts/check_config_yaml_readers.py` rejects a direct `open` / `read_text` / `stat` /
  `exists` / YAML load of a config path. A true false positive (another product's config
  file, or file tooling already gated on `supports_file_tooling()`) carries
  `# config-reader: ok — <why>` on the line.
