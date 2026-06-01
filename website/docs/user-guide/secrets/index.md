# Secrets

Hermes can pull API keys from external secret managers at process startup instead of storing them in `~/.hermes/.env`. The bootstrap token for the secret manager lives in `.env`; every other provider key (OpenAI, Anthropic, OpenRouter, etc.) can stay in the manager and rotate centrally.

Supported:

- [Bitwarden Secrets Manager](./bitwarden) — `bws` CLI, lazy-installed, free tier works.
- [1Password](./onepassword) — `op://` references via the official `op` CLI; service-account or desktop session auth.

More backends (Vault, AWS Secrets Manager) are easy to add behind the same interface — the lift is one module in `agent/secret_sources/` (which shares a common cache/result substrate in `agent/secret_sources/_cache.py`) and one CLI handler. File a request if you have a specific one in mind.
