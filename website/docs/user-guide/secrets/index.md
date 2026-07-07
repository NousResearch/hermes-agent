# Secrets

Hermes can pull API keys from external secret managers at process startup instead of storing them in `~/.hermes/.env`. The bootstrap token for the secret manager lives in `.env`; every other provider key (OpenAI, Anthropic, OpenRouter, etc.) can stay in the manager and rotate centrally.

Supported:

- [Bitwarden Secrets Manager](./bitwarden) — `bws` CLI, lazy-installed, free tier works.
- [Proton Pass](./protonpass) — `pass-cli`, lazy-installed + SHA256-verified, agent-token (least-privilege) or PAT.
- [1Password](./onepassword) — `op://` references via the official `op` CLI; service-account or desktop session auth.

## Multiple sources at once

You can enable more than one secret source at the same time — for example a team Bitwarden project alongside a personal vault plugin. Sources compose per env var with a deterministic precedence ladder:

1. **Your `.env` / shell wins by default.** A source only replaces a pre-existing value when its own `override_existing: true` is set (Bitwarden defaults to true so central rotation works).
2. **Mapped sources beat bulk sources.** A source where you explicitly bind env vars to references (an `env:` map) outranks a source that injects a whole project of secrets implicitly, regardless of ordering.
3. **First source wins.** Within the same shape, the order of the optional `secrets.sources` list (or registration order) decides. Later claims on an already-claimed var are skipped — with a startup warning, never silently.

`override_existing` never lets one source overwrite a var another source already claimed, and no source can ever overwrite another source's bootstrap token (e.g. `BWS_ACCESS_TOKEN`).

```yaml
secrets:
  sources: [bitwarden]     # optional explicit ordering
  bitwarden:
    enabled: true
    project_id: "..."
```

Every credential injected by a source is labelled with its origin — setup flows and `hermes model` show `(from Bitwarden)` next to detected keys so you always know where a value came from.

## Adding your own backend

Third-party secret managers ship as standalone plugins, not core PRs. A backend subclasses `agent.secret_sources.base.SecretSource` (one required method: `fetch(cfg, home_path) -> FetchResult`) and registers via `ctx.register_secret_source(MySource())` in the plugin's `register(ctx)`. The orchestrator owns precedence, conflict handling, timeouts, and provenance — your source only fetches. Full guide with the contract rules, subprocess-safety helper, and conformance kit: [Building a Secret Source Plugin](/developer-guide/secret-source-plugin).

The bundled set is deliberately small (same policy as memory providers): Bitwarden, 1Password, and Proton Pass ship in-tree. Other managers — Infisical, HashiCorp Vault, AWS Secrets Manager, OS keystores — belong in plugin repos; share them in the Nous Research Discord (`#plugins-skills-and-skins`).
