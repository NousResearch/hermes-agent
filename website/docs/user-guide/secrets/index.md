# Secrets

Hermes can pull API keys from external secret managers at process startup instead of storing them in `~/.hermes/.env`. The bootstrap token for the secret manager lives in `.env`; every other provider key (OpenAI, Anthropic, OpenRouter, etc.) can stay in the manager and rotate centrally.

Supported:

- [Bitwarden Secrets Manager](./bitwarden) — `bws` CLI, lazy-installed, free tier works.
- [1Password](./onepassword) — `op` CLI secret references using `OP_SERVICE_ACCOUNT_TOKEN`.

More backends (Vault, AWS Secrets Manager, etc.) are easy to add behind the same interface — the lift is one module in `agent/secret_sources/` and one CLI handler. File a request if you have a specific one in mind.
