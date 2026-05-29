# Proton Pass

Pull API keys from [Proton Pass](https://proton.me/pass) at process startup instead of storing them in plaintext inside `~/.hermes/.env`. One bootstrap token replaces N per-provider keys, and rotating a credential becomes a single change in Proton Pass.

The integration is **non-destructive** (existing env wins by default), **fail-open** (a Proton Pass failure never blocks Hermes startup - you just get a one-line stderr warning), and it **never logs secret values or your token**.

> **You don't need `pass-cli` installed.** Hermes downloads and verifies it for you on first use. The only thing you set up by hand is one token. See [Quick start](#quick-start).

## How it works

1. You authenticate `pass-cli` once with a token (an **agent token** - recommended - or a personal access token).
2. Hermes stores that single token in `~/.hermes/.env` as `PROTON_PASS_PERSONAL_ACCESS_TOKEN` (never in `config.yaml`).
3. Every time `hermes` (or the gateway, or a cron job) starts, after `~/.hermes/.env` has loaded, Hermes establishes a read-only Proton Pass session from that token and fetches the secrets you've mapped.
4. By default Hermes leaves values already in your environment **untouched** - `.env` wins. Flip `override_existing: true` if you want Proton Pass to be the source of truth instead.

Proton Pass is SaaS-only - there is no self-hosted option, so there is no `server_url` to configure.

## Install

You do **not** need to install anything yourself. When `auto_install: true` (the default), Hermes downloads `pass-cli` into `<hermes_home>/bin/pass-cli` (i.e. `~/.hermes/bin/pass-cli`) on first use - no `apt`, no `brew`, no `sudo`.

The download is pinned to a known version (**`pass-cli v2.1.1`** at time of writing) and **verified against a hardcoded, pinned SHA-256 checksum** baked into Hermes. A checksum mismatch aborts the install, and the installed binary's reported version is re-checked against the pin. Hermes does not auto-upgrade `pass-cli` to "latest" - the pin (version plus its asset hashes) is bumped together through PRs to this repo.

> **Maintainer note - bumping the pinned version.** To move the pin to a new `pass-cli` release: (1) download the official release assets for every supported platform; (2) compute the `sha256` of each asset; (3) in `agent/secret_sources/protonpass/install.py`, update `_PASS_CLI_VERSION` and the matching `_PINNED_SHA256` entries together in the same change (never one without the other); (4) test an `auto_install` run end to end so the checksum and version re-check both pass.

If you'd rather install it yourself (e.g. `auto_install: false`, or an air-gapped box), grab the binary from the official downloads page and put it on your `PATH`:

```bash
# Official manual install - see https://proton.me/download/pass-cli/
# (download the binary for your platform, then:)
chmod +x pass-cli
sudo mv pass-cli /usr/local/bin/pass-cli
pass-cli --version    # should print 2.1.1
```

> **Windows.** A `pass-cli.exe` you drop on your `PATH` is **never** auto-trusted by SHA-256, because the pinned digest is the official `.zip` archive's hash, not the extracted `.exe`'s - so Hermes cannot hash-match a loose PATH `.exe` against it and fails closed rather than hand the service token to an unverified binary. On Windows, run the managed `hermes secrets protonpass install` instead: it downloads the verified archive, extracts it into `<hermes_home>/bin`, and writes a verified install-time sidecar digest that Hermes checks before each use.

> **Windows on ARM64.** There is no native ARM64 Windows build of `pass-cli`; the installer maps Windows + ARM64 to the x86_64 `.zip`, so the downloaded x86_64 binary runs under Windows' x64 emulation.

## Authentication

`pass-cli` is **session-based**: you authenticate once and subsequent commands reuse the stored session. Hermes establishes that session non-interactively from the token in `~/.hermes/.env` - there is no runtime unlock prompt.

There are two token types. **Prefer the agent token** for least privilege.

### Agent token (recommended - least privilege)

An agent token is a scoped, expiring credential limited to a single vault with a read-only role. This is the right choice for an unattended workload like Hermes.

```bash
# 1. Create an agent scoped to one vault, with an expiration.
pass-cli agent create --vault "<VAULT>" --expiration 90d "hermes-readonly"

# 2. Grant it read-only access.
pass-cli agent access grant --role viewer "hermes-readonly"
```

Copy the minted token into `~/.hermes/.env` (never `config.yaml`):

```bash
echo 'PROTON_PASS_PERSONAL_ACCESS_TOKEN=<agent-token>' >> ~/.hermes/.env
```

> Reference mode (MODE B, below) works under **both** agent sessions and PAT sessions, so the agent-token path covers the recommended setup end to end. Agent tokens are scoped PATs (format `pst_...::...`) and are consumed exactly like a PAT: Hermes sets `PROTON_PASS_PERSONAL_ACCESS_TOKEN=<token>` in an isolated session (`PROTON_PASS_SESSION_DIR=<hermes_home>/protonpass-session`) and runs `pass-cli login` once non-interactively, then reuses that session for every fetch.

### Personal access token (PAT)

A PAT (format `pst_<token>::<key>`) authenticates as your own account. It's simpler but broader-privilege, so use it only if you specifically need the vault-listing mode below.

Mint the PAT in the Proton Pass web app, then hand it to Hermes without ever putting it on a command line (a literal token on argv leaks into your shell history and `ps` output). Use the setup wizard's masked prompt:

```bash
hermes secrets protonpass setup
```

The wizard prompts for the token with a masked prompt and writes it to `~/.hermes/.env` as `PROTON_PASS_PERSONAL_ACCESS_TOKEN`. Hermes runs `pass-cli login` for you, reading the token from the environment - you never type `pass-cli login --pat` yourself.

If you need to script this non-interactively, pipe the token in on stdin or set the env var, but never pass it as a `--pat` argument:

```bash
# Append the token to ~/.hermes/.env (the value is not echoed back).
read -rs PROTON_PASS_PERSONAL_ACCESS_TOKEN
printf 'PROTON_PASS_PERSONAL_ACCESS_TOKEN=%s\n' "$PROTON_PASS_PERSONAL_ACCESS_TOKEN" >> ~/.hermes/.env
```

> **Vault listing (MODE A) is PAT-only.** Bulk-listing a whole vault *with secret values* (`pass-cli item list <vault> --show-secrets --output json`) works under a full/PAT session but is **rejected under scoped agent sessions**. If you want MODE A, you must use a PAT. Reference mode (MODE B) has no such restriction.

## Configuration

Two modes select which secrets Hermes pulls. You can use either or both.

### MODE B - references (preferred)

Map each environment variable to a single Proton Pass field via a `pass://` URI. This is deterministic, least-privilege, and works under both agent and PAT sessions:

```yaml
secrets:
  protonpass:
    env:
      OPENROUTER_API_KEY: "pass://SHARE_ID/ITEM_ID/api_key"
      ANTHROPIC_API_KEY:  "pass://SHARE_ID/ITEM_ID/api_key"
```

The canonical URI is `pass://SHARE_ID/ITEM_ID/FIELD` (share-id / item-id, not names) and is the form Hermes uses. `SHARE_ID` and `ITEM_ID` are base64url tokens, so they never contain a `/` and the URI splits cleanly into `[SHARE_ID, ITEM_ID, FIELD]`. `FIELD` is required for env injection (one env var needs exactly one value); a ref with no field is skipped with a warning.

Look up the IDs with `pass-cli`:

```bash
# SHARE_ID: the "share_id" of each vault
pass-cli vault list --output json

# ITEM_ID: the "id" of each item in a vault
pass-cli item list "<vault>" --output json
```

`pass-cli item view` also accepts `--vault-name`/`--item-title` selectors instead of the URI, but the config always uses the `pass://SHARE_ID/ITEM_ID/FIELD` URI.

### MODE A - vault listing (PAT-only)

Point at a vault by name and Hermes maps every item's fields to env vars (`ITEM_FIELD`, upper-snake-cased). Requires a PAT session:

```yaml
secrets:
  protonpass:
    vault: "Hermes keys"
```

Invalid or colliding env var names are skipped with a warning - never a crash. If both modes are set, MODE B entries take precedence on a key collision.

### Full config block

Defaults in `~/.hermes/config.yaml`:

```yaml
secrets:
  protonpass:
    enabled: false
    service_token_env: PROTON_PASS_PERSONAL_ACCESS_TOKEN
    vault: ""              # MODE A: list a vault, map every item field -> env (PAT-only)
    env: {}                # MODE B: { ENV_VAR: "pass://SHARE_ID/ITEM_ID/FIELD" } (preferred)
    cache_ttl_seconds: 300
    override_existing: false
    auto_install: true
```

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. When false, Proton Pass is never contacted. |
| `service_token_env` | `PROTON_PASS_PERSONAL_ACCESS_TOKEN` | Env var name that holds the bootstrap token. Change it if you already use that name for something else. |
| `vault` | `""` | MODE A: name of the vault to sync from. PAT-only. |
| `env` | `{}` | MODE B: map of `ENV_VAR -> "pass://SHARE_ID/ITEM_ID/FIELD"`. Preferred. |
| `cache_ttl_seconds` | `300` | TTL for cached fetch results, in seconds. Caching is two-layer: an in-process cache plus a disk cache at `<hermes_home>/cache/protonpass_cache.json` (mode `0600`) that **persists across processes**, so back-to-back `hermes` invocations within the TTL reuse the disk cache instead of re-fetching. Set to `0` (or any value `<= 0`) to disable both layers entirely. |
| `override_existing` | `false` | When false (default), values already in env win - `.env` is authoritative. Flip to `true` to make Proton Pass the source of truth (rotation in the app takes effect on next start). |
| `auto_install` | `true` | When true, `pass-cli` is auto-downloaded into `~/.hermes/bin/` on first use, verified against a pinned hardcoded SHA-256. |

## Quick start

From zero to working, assuming you have a Proton Pass account:

1. **Store your provider keys in Proton Pass.** Put each API key in an item field in a vault (e.g. a vault called "Hermes keys").
2. **Mint an agent token** (least privilege):
   ```bash
   pass-cli agent create --vault "Hermes keys" --expiration 90d "hermes-readonly"
   pass-cli agent access grant --role viewer "hermes-readonly"
   ```
   *(If `pass-cli` isn't installed yet, run `hermes secrets protonpass install` first - or just let the wizard in the next step install it for you.)*
3. **Drop the token in `~/.hermes/.env`:**
   ```bash
   echo 'PROTON_PASS_PERSONAL_ACCESS_TOKEN=<agent-token>' >> ~/.hermes/.env
   ```
4. **Map your keys** (MODE B) in `~/.hermes/config.yaml`:
   ```yaml
   secrets:
     protonpass:
       enabled: true
       env:
         OPENROUTER_API_KEY: "pass://SHARE_ID/ITEM_ID/api_key"
   ```
5. **Confirm:**
   ```bash
   hermes secrets protonpass status
   ```

From now on, every `hermes` invocation pulls the mapped secrets at startup. You'll see a one-line summary in stderr the first time secrets are applied in a process.

## CLI

| Command | What it does |
|---|---|
| `hermes secrets protonpass setup` | Interactive wizard: install the binary, prompt for the token, choose a mode, test fetch. |
| `hermes secrets protonpass status` | Show config, binary version, and token presence (never the value). |
| `hermes secrets protonpass sync` | Dry-run: fetch now and show what would be applied. |
| `hermes secrets protonpass sync --apply` | Fetch and set the secrets in this `hermes` process's own environment only (it does not - and cannot - mutate the parent shell that launched it). |
| `hermes secrets protonpass install` | Just download the pinned `pass-cli` binary (no auth required). |
| `hermes secrets protonpass disable` | Flip `enabled: false`; leaves the token and config in place. |

The CLI surface mirrors the Bitwarden wizard. The setup wizard prompts for the token with a masked prompt (the token is never passed on the command line) and writes it to `~/.hermes/.env`.

> **After disabling.** `hermes secrets protonpass disable` only flips `enabled: false`; it leaves the token in `~/.hermes/.env`. To fully retire access, also remove `PROTON_PASS_PERSONAL_ACCESS_TOKEN` from `.env` **and** revoke the token in Proton Pass itself (revoke the agent or PAT in the Proton Pass app) - deleting it from `.env` alone leaves a still-valid credential live in Proton.

## Failure modes

Proton Pass never blocks Hermes startup. If anything goes wrong, you'll see a one-line warning in stderr and Hermes continues with whatever credentials `.env` already had:

| Symptom | Cause | Fix |
|---|---|---|
| `PROTON_PASS_PERSONAL_ACCESS_TOKEN is not set` | Enabled in config but token cleared from `.env` | Re-run `hermes secrets protonpass setup` |
| `--show-secrets is not allowed under an agent session` | MODE A (vault listing) attempted with an agent token | Switch to MODE B references, or use a PAT for MODE A |
| `pass-cli` auth error | Token expired, revoked, or wrong | Mint a new token, re-run setup |
| `pass-cli binary not available` | `auto_install: false` and `pass-cli` not on PATH | Install manually from [proton.me/download/pass-cli](https://proton.me/download/pass-cli/) or flip `auto_install` back on |
| `pass-cli timed out` | Network blocked or Proton API slow | Check connectivity to Proton |
| `Checksum mismatch` | Download corrupted or tampered | Re-run; if it persists, file an issue |

## Security notes

- The bootstrap token lives **only** in `~/.hermes/.env` (or the real environment) - **never** in `config.yaml` and never written to the on-disk cache. Hermes stores only a SHA-256 fingerprint of the token to detect rotation and invalidate its cache.
- Prefer an **agent token** scoped to one vault with the `viewer` role and an expiration. A PAT authenticates as your whole account and is broader than necessary.
- Secret **values are never logged**. Warnings and errors are redacted.
- **Existing env wins** by default (`override_existing: false`). Hermes will not silently clobber a key you've already set.
- Resolved secret values are cached at rest **in plaintext** at `<hermes_home>/cache/protonpass_cache.json` (mode `0600`), expiring after `cache_ttl_seconds` (default `300`). This is the same plaintext-at-rest tradeoff as storing the keys directly in `.env`; the bootstrap service token itself is **never** cached - the file holds only the resolved values keyed by name plus a SHA-256 fingerprint of the token (used to detect rotation and invalidate the cache).

## When NOT to use this

- **Single-machine personal setups** where `~/.hermes/.env` is fine. You're trading one credential for another and adding a network dependency at startup.
- **Air-gapped environments** that can't reach Proton.
- **CI/CD** where a secrets-injection mechanism (GitHub Actions secrets, Vault, etc.) already exists - pick one path, not two.

The good case is multi-machine fleets, shared dev boxes, gateway VPSes, or any setup where you want centralized rotation and revocation across multiple Hermes installations.
