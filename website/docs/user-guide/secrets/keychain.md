# macOS Keychain

Hermes can resolve macOS Keychain generic-password items at process startup so local machine credentials do not live in plaintext `~/.hermes/.env`, launchd plists, systemd units, or wrapper scripts.

This backend is useful when a credential is already managed by the Mac login/system Keychain, or when a headless local wrapper stores an access token in Keychain and should inject it only into the Hermes process environment.

## How it works

1. You store the real secret in macOS Keychain as a generic password.
2. `config.yaml` stores only non-secret lookup metadata: service/account names.
3. When `hermes`, the gateway, cron, ACP, or the desktop backend starts, Hermes runs `security find-generic-password -s <service> -a <account> -w` with a short timeout and sets the value in `os.environ` for that process only.
4. The value is never written back to disk, logged, cached, or embedded in a wrapper.

## Configuration

Add this to `~/.hermes/config.yaml`:

```yaml
secrets:
  keychain:
    enabled: true
    timeout_seconds: 6
    override_existing: false
    env:
      TELEGRAM_BOT_TOKEN:
        service: halvo-shared
        account: HERMES_MBP_TELEGRAM_BOT_TOKEN
```

Supported keys:

| Key | Default | What it does |
|---|---:|---|
| `enabled` | `false` | Master switch. When false, Keychain is never contacted. |
| `env` | `{}` | Mapping of environment-variable names to Keychain lookup specs. `mappings` and `references` are accepted aliases. |
| `timeout_seconds` | `6` | Per-item timeout for `security`. Prevents startup hangs. |
| `override_existing` | `false` | When false, an existing env value wins. |
| `security_path` | auto | Optional absolute path to the `security` binary. Defaults to `/usr/bin/security` / `PATH`. |

A shorthand string is accepted too:

```yaml
secrets:
  keychain:
    enabled: true
    env:
      TELEGRAM_BOT_TOKEN: "halvo-shared/HERMES_MBP_TELEGRAM_BOT_TOKEN"
```

## Telegram example

```yaml
secrets:
  keychain:
    enabled: true
    env:
      TELEGRAM_BOT_TOKEN:
        service: halvo-shared
        account: HERMES_MBP_TELEGRAM_BOT_TOKEN
```

Keep only non-secret routing metadata in `.env`:

```bash
TELEGRAM_ALLOWED_USERS=123456789
TELEGRAM_HOME_CHANNEL=123456789
```

Then restart the gateway. On startup, Hermes resolves `TELEGRAM_BOT_TOKEN` in memory before the Telegram adapter reads gateway config.

## Security notes

- Keychain is appropriate for local machine credentials and bootstrap tokens. For team/fleet machine workloads, prefer Google Secret Manager, Bitwarden Secrets Manager, or another workload-identity-backed service.
- Keychain lookup metadata is not a secret, but it is still operationally sensitive: it identifies what Hermes may request from the local machine.
- Hermes reports only env-var names and redacted Keychain errors. It does not print secret values.
- This integration does not cache secret values on disk. If Keychain access requires user approval, Hermes may need a human-authenticated session or a pre-approved keychain access control policy.
