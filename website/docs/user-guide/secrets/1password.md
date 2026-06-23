# 1Password CLI

Hermes can resolve [1Password CLI](https://developer.1password.com/docs/cli/) references at process startup so API keys and bot tokens stay in 1Password instead of plaintext `~/.hermes/.env`, launchd plists, systemd units, or wrapper scripts.

## How it works

1. You store the real secret in 1Password.
2. `config.yaml` or `.env` stores only a non-secret reference such as `op://Employee/Hermes Telegram/credential`.
3. When `hermes`, the gateway, cron, ACP, or the desktop backend starts, Hermes runs `op read <reference>` with a short timeout and sets the resulting value in `os.environ` for that process only.
4. The value is never written back to disk, logged, cached, or embedded into a service wrapper.

This is best for personal Macs and other machines where the 1Password CLI / desktop app or a service-account-backed `op` environment is already configured. If `op` is locked or unavailable, Hermes starts anyway and prints a one-line warning.

## Configuration

Add this to `~/.hermes/config.yaml`:

```yaml
secrets:
  onepassword:
    enabled: true
    timeout_seconds: 8
    override_existing: false
    env:
      TELEGRAM_BOT_TOKEN: "op://Employee/Hermes Telegram/credential"
      OPENROUTER_API_KEY: "op://Employee/OpenRouter/credential"
```

Supported keys:

| Key | Default | What it does |
|---|---:|---|
| `enabled` | `false` | Master switch. When false, 1Password is never contacted. |
| `env` | `{}` | Mapping of environment-variable names to `op://` references. `mappings` and `references` are accepted aliases. |
| `timeout_seconds` | `8` | Per-reference timeout for `op read`. Prevents locked desktop auth from hanging startup. |
| `override_existing` | `false` | When false, a real existing env value wins. Existing `op://` values are still resolved. |
| `resolve_env_references` | `true` | Also resolve credential-like env vars whose current value is an `op://` reference. |
| `op_path` | auto | Optional absolute path to the `op` binary. Hermes checks `PATH`, Homebrew paths, and `~/.local/bin/op` first. |
| `account` | `""` | Optional 1Password account shorthand passed to `op read --account`. |

The alias key `secrets.1password` also works if you prefer that spelling.

## Telegram example

To keep the Telegram bot token out of `.env` and service wrappers:

```yaml
secrets:
  onepassword:
    enabled: true
    env:
      TELEGRAM_BOT_TOKEN: "op://Employee/Hermes Telegram/credential"
```

Keep the non-secret Telegram routing settings wherever you normally keep them:

```bash
TELEGRAM_ALLOWED_USERS=123456789
TELEGRAM_HOME_CHANNEL=123456789
```

Then restart the gateway. On startup, Hermes resolves `TELEGRAM_BOT_TOKEN` in memory before the Telegram adapter reads gateway config.

## `.env` reference mode

If you prefer keeping the reference near other env settings, `.env` may contain a link instead of the secret:

```bash
TELEGRAM_BOT_TOKEN=op://Employee/Hermes Telegram/credential
```

You still need `secrets.onepassword.enabled: true` in `config.yaml`. Hermes treats that value as a reference and replaces it in the process environment; it does not write the resolved token back to `.env`.

## Security notes

- Do not store the real vendor secret in wrapper scripts as a reliability shortcut. The configured `op://` reference is the durable link for this backend; the resolved value is process-local only. If the machine-local source of truth is macOS Keychain instead, use the [Keychain backend](./keychain) explicitly so the boundary is visible in `config.yaml`.
- `op` failures are fail-open and timeout-bounded. If the 1Password desktop app is locked, approve/unlock it or configure a headless service-account path, then restart Hermes.
- Hermes reports only env-var names and redacted 1Password errors. It does not print secret values or the configured reference on failures.
- This integration does not cache secret values on disk. If you need non-interactive fleet-wide startup with cache-like behavior, use a proper machine-account secret backend such as Bitwarden Secrets Manager or an external process supervisor that injects env vars safely.
