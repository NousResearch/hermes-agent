# Secrets

Hermes can pull API keys from external secret managers at process startup instead of storing them in `~/.hermes/.env`, launchd plists, systemd units, or wrapper scripts. Depending on the backend, `config.yaml` may hold only a non-secret reference while the real value stays in the manager.

Supported:

- [Bitwarden Secrets Manager](./bitwarden) — `bws` CLI, lazy-installed, free tier works.
- [1Password CLI](./1password) — resolves `op://` references into process-local env vars without writing secret values to disk.
- [macOS Keychain](./keychain) — resolves local generic-password items into process-local env vars without putting values in `.env` or wrappers.

More backends (Vault, AWS Secrets Manager, etc.) are easy to add behind the same interface — the lift is one module in `agent/secret_sources/` and, when needed, one CLI handler. File a request if you have a specific one in mind.
