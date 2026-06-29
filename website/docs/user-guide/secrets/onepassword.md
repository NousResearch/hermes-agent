# 1Password CLI

Pull API keys from [1Password](https://1password.com/) at process startup instead of storing every provider key in plaintext inside `~/.hermes/.env`. Hermes uses the `op` CLI to resolve `op://...` secret references and sets the resulting values in `os.environ` for the current process.

## How it works

1. You install and authenticate the [1Password CLI](https://developer.1password.com/docs/cli/get-started/), or provide a 1Password Service Account token via an env var such as `OP_SERVICE_ACCOUNT_TOKEN`.
2. You create a references file, by default `~/.hermes/secrets/1password.env`, with lines like:

   ```bash
   OPENROUTER_API_KEY=op://Private/OpenRouter API Key/credential
   RUNWARE_API_KEY=op://Private/Runware API Key/credential
   ```

   This file stores **references only**, not plaintext API keys.
3. Every time `hermes` (or the gateway, or a cron job) starts, after `~/.hermes/.env` has loaded, Hermes calls `op read <reference>` for each entry and sets the returned keys into `os.environ`.
4. By default Hermes **overrides** values already in your environment, so rotation in 1Password takes effect on the next Hermes process start. Flip `override_existing: false` in config if you want `.env` / shell exports to win locally.

## Setup

### 1. Install and authenticate `op`

Install the CLI from the official docs:

```text
https://developer.1password.com/docs/cli/get-started/
```

Then verify:

```bash
op --version
op whoami
```

For non-interactive servers, prefer a [1Password Service Account](https://developer.1password.com/docs/service-accounts/). Store only that bootstrap token in your shell or `~/.hermes/.env`:

```bash
OP_SERVICE_ACCOUNT_TOKEN=...
```

You can rename that env var with `secrets.onepassword.service_account_token_env`.

### 2. Run the wizard

```bash
hermes secrets onepassword setup
```

It will:

1. Check whether `op` is available.
2. Create `~/.hermes/secrets/1password.env` if it does not exist.
3. Set file mode `0600` on the references file.
4. Enable `secrets.onepassword.enabled: true` in `config.yaml`.

The generated file contains placeholders. Replace them with real `op://...` references copied from the 1Password app, CLI, or browser extension.

Non-interactive setup is also supported:

```bash
hermes secrets onepassword setup \
  --env-file ~/.hermes/secrets/1password.env \
  --service-account-token-env OP_SERVICE_ACCOUNT_TOKEN \
  --op-path /usr/local/bin/op
```

### 3. Confirm

```bash
hermes secrets onepassword status
hermes secrets onepassword sync
```

`sync` is a dry-run by default. It resolves references and shows which env var names would be exported, but it never prints secret values.

## CLI

| Command | What it does |
|---|---|
| `hermes secrets onepassword setup` | Configure the backend and create the references file if missing |
| `hermes secrets onepassword status` | Show config, `op` availability, reference count, and bootstrap-token presence |
| `hermes secrets onepassword sync` | Dry-run: resolve references now and show what would be applied |
| `hermes secrets onepassword sync --apply` | Resolve and export into the current process's environment |
| `hermes secrets onepassword disable` | Flip `enabled: false`; leaves the references file untouched |

Aliases:

```bash
hermes secrets op status
hermes secrets 1password status
```

## Configuration

Defaults in `~/.hermes/config.yaml`:

```yaml
secrets:
  onepassword:
    enabled: false
    env_file: ~/.hermes/secrets/1password.env
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
    cache_ttl_seconds: 300
    override_existing: true
    op_path: ""
```

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. When false, 1Password is never contacted. |
| `env_file` | `~/.hermes/secrets/1password.env` | Env-style file containing `NAME=op://vault/item/field` references. |
| `service_account_token_env` | `OP_SERVICE_ACCOUNT_TOKEN` | Optional env var name that holds a 1Password Service Account token. If unset, `op` can still use an authenticated app/CLI session. |
| `cache_ttl_seconds` | `300` | How long resolved references are reused in-process. Set to `0` to disable caching. |
| `override_existing` | `true` | When true, 1Password values overwrite anything already in env. Flip to `false` if `.env` / shell exports should win locally. |
| `op_path` | `""` | Optional explicit path to the `op` binary. Empty = resolve from `PATH`. |

## References file format

```bash
# comments are allowed
OPENROUTER_API_KEY=op://Private/OpenRouter API Key/credential
RUNWARE_API_KEY='op://Private/Runware API Key/credential'
export FAL_KEY=op://Private/FAL API Key/credential
```

Rules:

- Keys must be valid environment variable names.
- Values must start with `op://`.
- Plaintext-looking values are skipped with a warning.
- Secret values are never printed in status, sync output, startup warnings, or errors.

## Failure modes

1Password never blocks Hermes startup. If anything goes wrong, Hermes prints a one-line warning to stderr and continues with whatever credentials `.env` already had.

| Symptom | Cause | Fix |
|---|---|---|
| `op is not available` | 1Password CLI not installed or not on PATH | Install `op`, or set `secrets.onepassword.op_path` |
| `references file does not exist` | Enabled but `env_file` has not been created | Run `hermes secrets onepassword setup` |
| `references file contains no op:// entries` | File has only comments/placeholders/plaintext values | Replace placeholders with real 1Password references |
| `op exited ...` | Not signed in, service account token missing/invalid, or reference cannot be read | Run `op whoami`, check `OP_SERVICE_ACCOUNT_TOKEN`, and verify item/vault access |
| A key is skipped as already set | `override_existing: false` and env already has that key | Remove the local env var or set `override_existing: true` |

## Security notes

- The references file should contain `op://...` references only, not plaintext API keys.
- Hermes sets the references file mode to `0600` during setup.
- For shared servers and gateways, use a 1Password Service Account with least-privilege vault access rather than a personal interactive session.
- The bootstrap token, if you use one, is still a secret. Store it in `~/.hermes/.env` or the service manager's secret mechanism, not in `config.yaml`.
- Hermes refuses to let 1Password overwrite the configured bootstrap-token env var itself.

## When NOT to use this

- Single-machine setups where `~/.hermes/.env` is sufficient and centralized rotation is unnecessary.
- Air-gapped environments where `op` cannot reach 1Password.
- CI/CD systems that already have a mature secret-injection mechanism.

The good case is a gateway VPS, shared dev box, or multi-machine setup where you want central rotation and revocation in 1Password while keeping Hermes config free of provider keys.
