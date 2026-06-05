# 1Password (onepassword-sdk)

Pull API keys from a [1Password Service Account](https://developer.1password.com/docs/service-accounts/) at process startup instead of storing them in plaintext inside `~/.hermes/.env`. One bootstrap token replaces N per-provider keys, and rotating a credential becomes a single change in the 1Password web app.

## Why the Python SDK (not the `op` CLI)

Hermes uses the [`onepassword-sdk`](https://pypi.org/project/onepassword-sdk/) Python package instead of the `op` CLI daemon. The SDK authenticates directly via the 1Password REST API — no background daemon needed. This is important on macOS where `op daemon --background` can hang indefinitely in headless/background contexts (gateway, cron, CLI background tasks).

If you prefer the `op` CLI, a CLI-based 1Password backend is also available (see the companion implementation).

## How it works

1. You create a **service account** in the 1Password web app, grant it read access to one or more vaults, and generate a token (starts with `ops_`).
2. Hermes stores that single token in `~/.hermes/.env` as `OP_SERVICE_ACCOUNT_TOKEN`.
3. Every time `hermes` (or the gateway, or a cron job) starts, after `.env` has loaded, Hermes uses the SDK to pull credential fields from your vault and sets them into `os.environ`.
4. By default Hermes **overrides** values already in your environment — rotate a key once in 1Password and every Hermes process picks it up on next start. Flip `override_existing: false` in config if you want `.env` to win.

The `onepassword-sdk` is an optional dependency — install it once:

```bash
uv pip install --python "$(head -1 "$(command -v hermes)" | sed 's/^#!//')" onepassword-sdk
```

## Two mapping modes

### Mode A: Auto-discovery (zero per-secret config)

```yaml
secrets:
  onepassword:
    enabled: true
    vault: "Private"
    auto_discover: true
```

Hermes scans every item in the vault and extracts credential fields (Concealed, Password, API Credential types). Field titles become environment variable names (uppercased, spaces → underscores). For example, a field titled "OpenAI API Key" becomes `OPENAI_API_KEY`.

Items with generic field labels ("credential", "password", "token") use the **item title** as the env var name instead.

### Mode B: Explicit mapping (auditable, like Bitwarden)

```yaml
secrets:
  onepassword:
    enabled: true
    env:
      OPENAI_API_KEY: "op://Private/OpenAI/api key"
      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"
```

You specify exactly which env var maps to which `op://` reference. Hermes resolves each reference at startup. The vault is inferred from the references themselves, so you don't need a `vault:` key (though you can use both modes together).

### Using both modes together

When both `auto_discover` and `env` are configured, explicit mappings **take precedence** on naming collisions. Auto-discovered fields that don't conflict with explicit mappings are still applied.

## Setup

### 1. Create a service account and token

In the [1Password web app](https://my.1password.com):

1. **Developers → Service Accounts**.
2. **Create service account** → name it (e.g. "Hermes Agent").
3. Grant read access to the vault(s) you want.
4. Copy the generated token (starts with `ops_`). 1Password cannot retrieve it again — keep the copy.

### 2. Run the interactive setup wizard

```bash
hermes secrets onepassword setup
```

This walks you through:

- Checking that `onepassword-sdk` is installed
- Prompting for (or detecting) your service account token
- Choosing auto-discovery or explicit mapping mode
- Picking a vault (for auto-discovery mode)
- Testing a fetch and showing discovered secrets
- Saving the config

Non-interactive example:

```bash
# Install the SDK first
uv pip install --python <hermes-python> onepassword-sdk

# Write the token to .env
echo 'OP_SERVICE_ACCOUNT_TOKEN=ops_...' >> ~/.hermes/.env

# Enable in config.yaml
hermes config set secrets.onepassword.enabled true
hermes config set secrets.onepassword.vault "Private"
hermes config set secrets.onepassword.auto_discover true
```

### 3. Verify

```bash
hermes secrets onepassword status
hermes secrets onepassword sync          # dry-run: show what would be applied
hermes secrets onepassword sync --apply  # actually export into the environment
hermes secrets onepassword list-vaults   # list accessible vaults
```

## Full config reference

```yaml
secrets:
  onepassword:
    # Enable the integration.  Default: false.
    enabled: true

    # Environment variable holding the service account token.
    # Default: OP_SERVICE_ACCOUNT_TOKEN
    token_env: OP_SERVICE_ACCOUNT_TOKEN

    # Vault name or ID (required for auto_discover mode).
    # Can be omitted when only using explicit env: mapping.
    vault: "Private"

    # When true, scan all items in the vault and map credential
    # fields → env vars automatically.  Default: false.
    auto_discover: true

    # Explicit env-var → op:// reference mappings.
    # These take precedence over auto-discovered secrets.
    env:
      OPENAI_API_KEY: "op://Private/OpenAI/api key"
      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"

    # When true, overwrite already-set env vars.
    # When false, .env and shell exports win.  Default: false.
    override_existing: true

    # How long cache entries stay fresh (seconds).
    # 0 = no caching (always fetch).  Default: 300.
    cache_ttl_seconds: 300
```

## In-session tools

When the `onepassword` toolset is enabled (via `hermes tools enable onepassword`), four tools become available during conversations:

| Tool | Description |
|------|-------------|
| `onepassword_list_vaults` | List all vaults accessible to the service account |
| `onepassword_list_items` | List items in a vault by ID or name |
| `onepassword_get_item` | Get full item details including all field values |
| `onepassword_resolve_field` | Resolve an `op://vault/item/field` reference to its value |

The `onepassword_resolve_field` tool uses a vault-contents cache: on first access to a vault it fetches all items and fields, then resolves subsequent lookups from memory — zero additional API calls within the cache TTL.

## Rate limits

1Password Service Accounts throttle at **1,000 reads per hour** per token (non-Business accounts). Hermes uses a **two-layer cache** (in-process + disk-persisted) to keep real API calls near zero after the first cold start. Additionally, a **rate-limit cooldown** prevents N concurrent processes (gateway + dashboard + slash workers) from retrying in lockstep — when a `429` is hit all processes back off for one hour.

If you see rate-limit warnings in the agent log, raise `cache_ttl_seconds` to 3600 or higher.

## Security

- The service account token is **never written to the disk cache** — only resolved secret values are cached.
- Cache keys use SHA-256 fingerprints of the token, so the raw token never appears in cache files.
- Disk cache files are written atomically with mode `0600`, and the cache directory is forced to `0700`.
- The token env var itself (`OP_SERVICE_ACCOUNT_TOKEN`) is **never overwritten**, even with `override_existing: true`.

## Troubleshooting

**"onepassword-sdk is not installed"**  
Install the SDK: `uv pip install --python <hermes-python> onepassword-sdk`

**"OP_SERVICE_ACCOUNT_TOKEN is not set"**  
The token must be in `~/.hermes/.env`. Run `hermes secrets onepassword setup` to configure it.

**"Vault not found"**  
Check the vault name with `hermes secrets onepassword list-vaults`. Vault names are case-sensitive.

**Rate-limit warnings**  
Increase `cache_ttl_seconds` to 3600. Clear the disk cache if you've rotated the token:

```bash
rm -f ~/.hermes/cache/onepassword_cache.json
```

**SDK missing after `hermes update`**  
`hermes update` may rebuild the venv, wiping the optional SDK. Reinstall:

```bash
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python onepassword-sdk
```
