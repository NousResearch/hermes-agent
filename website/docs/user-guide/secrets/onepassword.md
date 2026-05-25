# 1Password secret references

Resolve API keys from [1Password CLI](https://developer.1password.com/docs/cli/) secret references at process startup instead of storing every key in plaintext inside `~/.hermes/.env`.

## How it works

1. You install and sign in to the official `op` CLI, or set `OP_SERVICE_ACCOUNT_TOKEN` for a 1Password service account.
2. In `config.yaml`, map environment variable names to `op://` references.
3. Every time `hermes` starts, after `.env` has loaded, Hermes resolves missing/stale references with `op read op://...` and sets the resolved values into `os.environ`.
4. By default Hermes **overrides** values already in your environment, so rotating a value in 1Password takes effect on the next Hermes process start. Set `override_existing: false` if you want `.env` or shell exports to win locally.
5. Successful reads are cached briefly both in-process and on disk under `~/.hermes/cache/op_cache.json`, so short-lived Hermes commands, gateway worker starts, and cron-style invocations do not each pay a full `op` subprocess/auth round-trip.

Hermes does **not** install `op` automatically. 1Password supports several auth modes (desktop app integration, biometric unlock, service accounts, account shorthands), so Hermes expects the CLI to already be installed and authenticated.

## Setup

### 1. Install and authenticate `op`

Follow the official guide: <https://developer.1password.com/docs/cli/get-started/>

For interactive desktop use, run:

```bash
op signin
```

For non-interactive servers, set a service-account token in your shell, service manager, or `~/.hermes/.env`:

```bash
OP_SERVICE_ACCOUNT_TOKEN=ops_...
```

### 2. Add references

Use the CLI helper:

```bash
hermes secrets onepassword set OPENAI_API_KEY 'op://Private/OpenAI/credential' --enable
hermes secrets onepassword set ANTHROPIC_API_KEY 'op://Private/Anthropic/credential'
```

Or configure several references and test them in one command:

```bash
hermes secrets onepassword setup \
  --map 'OPENAI_API_KEY=op://Private/OpenAI/credential' \
  --map 'ANTHROPIC_API_KEY=op://Private/Anthropic/credential'
```

If you have multiple 1Password accounts, pass an account shorthand/email:

```bash
hermes secrets onepassword setup \
  --account my.1password.com \
  --map 'OPENAI_API_KEY=op://Private/OpenAI/credential'
```

### 3. Confirm

```bash
hermes secrets onepassword status
hermes secrets onepassword sync
```

`sync` is a dry run by default. It resolves references and shows which env vars would be exported without printing secret values.

## CLI

| Command | What it does |
|---|---|
| `hermes secrets onepassword setup --map ENV=op://...` | Configure references, test them, and enable the integration |
| `hermes secrets onepassword status` | Show config, token presence, and `op` availability |
| `hermes secrets onepassword sync` | Dry-run: resolve references and show what would be applied |
| `hermes secrets onepassword sync --apply` | Resolve and export into the current process |
| `hermes secrets onepassword set ENV_VAR op://... [--enable]` | Add or update one mapping |
| `hermes secrets onepassword remove ENV_VAR` | Remove one mapping |
| `hermes secrets onepassword disable` | Flip `enabled: false`; leaves references in config |

Aliases: `onepassword`, `op`, `1password`, and `1p`.

## Configuration

Defaults in `~/.hermes/config.yaml`:

```yaml
secrets:
  onepassword:
    enabled: false
    env: {}
    account: ""
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
    cache_ttl_seconds: 300
    override_existing: true
```

Example:

```yaml
secrets:
  onepassword:
    enabled: true
    account: my.1password.com
    env:
      OPENAI_API_KEY: op://Private/OpenAI/credential
      ANTHROPIC_API_KEY: op://Private/Anthropic/credential
```

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. When false, `op` is never contacted. |
| `env` | `{}` | Map of env var names to `op://vault/item/field` references. |
| `account` | `""` | Optional value passed to `op read --account`. Empty means use the CLI default. |
| `service_account_token_env` | `OP_SERVICE_ACCOUNT_TOKEN` | Env var name that holds an optional 1Password service-account token. Hermes refuses to overwrite this env var from a reference. |
| `cache_ttl_seconds` | `300` | How long resolved references are reused in-process and from `~/.hermes/cache/op_cache.json`. Set to `0` to disable both cache layers. |
| `override_existing` | `true` | When true, 1Password values overwrite existing env vars. Flip to `false` if you want `.env` / shell exports to win locally. |

## Failure modes

1Password never blocks Hermes startup. If anything goes wrong, Hermes warns on stderr and continues with credentials that were already loaded from `.env` or the shell.

| Symptom | Cause | Fix |
|---|---|---|
| `op CLI is not available on PATH` | 1Password CLI is not installed in the environment running Hermes | Install `op` and make sure the service/gateway PATH includes it |
| `not signed in` or `You are not currently signed in` | Desktop/session auth is unavailable or expired | Run `op signin`, enable desktop app integration, or use `OP_SERVICE_ACCOUNT_TOKEN` |
| `op read failed for 'op://...'` | Bad vault/item/field path or missing access | Check the reference with `op read` directly and grant access |
| `op timed out` | CLI hung waiting for unlock/auth or network | Unlock 1Password, check service-account config, or run `op read` manually |

## Security notes

- Hermes never prints resolved 1Password values in `status`, `setup`, or `sync` output.
- `config.yaml` should contain only `op://` references, not secret values.
- Resolved values may be cached in plaintext-equivalent form at `~/.hermes/cache/op_cache.json` for up to `cache_ttl_seconds`; the file is written with mode `0600` and does not contain the service-account token. Set `cache_ttl_seconds: 0` if you prefer every process to call `op` live.
- If you use `OP_SERVICE_ACCOUNT_TOKEN`, treat it like any other high-value bearer token. Anyone with it can read the vault items the service account can access.
- Hermes skips attempts to overwrite the service-account token env var itself, even when `override_existing: true`.

## When NOT to use this

- Single-machine setups where `~/.hermes/.env` is sufficient.
- Environments where `op` cannot run non-interactively.
- CI/CD systems that already inject secrets through a platform-native mechanism.
