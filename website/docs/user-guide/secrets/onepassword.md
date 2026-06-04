# 1Password secret references

Pull API keys from [1Password](https://1password.com/) at process startup using the 1Password CLI (`op`) instead of storing each key in plaintext inside `~/.hermes/.env`.

Hermes uses 1Password **secret references** such as `op://Vault/Item/password` and maps each reference to an environment variable name.

## How it works

1. You create a 1Password Service Account and give it read access to the vault/items Hermes needs.
2. Hermes stores that single bootstrap token in `~/.hermes/.env` as `OP_SERVICE_ACCOUNT_TOKEN` (or another env var name you configure).
3. `~/.hermes/config.yaml` maps environment variables to 1Password references.
4. Every time `hermes` starts, after `.env` loads, Hermes calls `op read <reference>` for each mapping and sets the values into `os.environ`.

Failures never block startup. If `op` is missing, the token is absent, or a reference cannot be read, Hermes prints a short warning and continues with whatever credentials were already available from `.env` or the shell.

## Setup

Install the 1Password CLI yourself first. On macOS:

```bash
brew install --cask 1password-cli
```

Then configure Hermes:

```bash
hermes secrets onepassword setup \
  --map OPENAI_API_KEY=op://Hermes/OpenAI/password \
  --map ANTHROPIC_API_KEY=op://Hermes/Anthropic/password
```

The setup command can prompt for a service account token and stores it in `~/.hermes/.env`. For gateways/services, make sure the token is stored in `~/.hermes/.env`, not only exported in your current shell. For non-interactive setup, pass `--service-account-token "$OP_SERVICE_ACCOUNT_TOKEN"`, but do not put tokens in shell history unless your environment is safe.

## CLI

- `hermes secrets onepassword setup`: configure the token env var and add/update mappings.
- `hermes secrets onepassword setup --skip-test`: save config without validating `op read` first.
- `hermes secrets onepassword status`: show whether the integration is enabled, token presence, mapping count, and `op` binary status.
- `hermes secrets onepassword sync`: dry-run resolve references and show what would be exported.
- `hermes secrets onepassword sync --apply`: resolve references and export into the current Hermes process.
- `hermes secrets onepassword disable`: set `secrets.onepassword.enabled: false`.

Aliases: `hermes secrets op ...` and `hermes secrets 1password ...`.

## Configuration

Example `~/.hermes/config.yaml`:

```yaml
secrets:
  onepassword:
    enabled: true
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
    mapping:
      OPENAI_API_KEY: op://Hermes/OpenAI/password
      ANTHROPIC_API_KEY: op://Hermes/Anthropic/password
    cache_ttl_seconds: 300
    override_existing: true
```

- `enabled`: master switch. When false, 1Password is never contacted.
- `service_account_token_env`: env var holding the Service Account token. Default: `OP_SERVICE_ACCOUNT_TOKEN`.
- `mapping`: env-var name to `op://...` secret reference.
- `cache_ttl_seconds`: how long resolved values are cached in-process. Set `0` to disable caching.
- `override_existing`: when true, 1Password values overwrite existing env values. When false, `.env` / shell exports win.

## Security notes

- The bootstrap token is sensitive. Anyone with it can read the vault items granted to the service account.
- Hermes never logs secret values. Status output names env vars and references only.
- Startup/setup errors name env vars, not full `op://Vault/Item/field` references, because vault/item names can be sensitive in logs.
- Hermes will not overwrite the bootstrap token env var from 1Password, even if you map it by mistake.
- Keep access narrow: grant the service account read access only to the vault/items Hermes needs.
