# Bitwarden Secrets Manager

Pull API keys from [Bitwarden Secrets Manager](https://bitwarden.com/products/secrets-manager/) at process startup instead of storing them in plaintext inside `~/.hermes/.env`. One bootstrap secret (a machine-account access token) replaces N per-provider keys, and rotating a credential becomes a single change in the Bitwarden web app.

## The security posture is the feature

Hermes does not merely *support* Bitwarden Secrets Manager — it implements the integration so that **the plaintext-secrets vulnerability class does not exist**. Every disclosure path is closed by design, and a hermetic end-to-end test pins the whole surface shut:

1. **No plaintext at rest — encrypted-only by default.** Every fetched secret is persisted only as AES-GCM ciphertext in `~/.hermes/cache/bws_cache.enc.json`, keyed off the bootstrap token. There is no plaintext write branch in the codebase. Setting `encrypted_cache.enabled: false` means **memory-only**: disk persistence is disabled entirely, and plaintext is never consulted or written as an alternative.
2. **Legacy plaintext caches are re-encrypted and removed on first read.** A legacy plaintext `bws_cache.json` is re-encrypted and removed on first read — including in memory-only mode. After any run, a plaintext cache file cannot survive.
3. **Secret values never reach status lines, logs, or terminal output.** Secret-source error, remediation-hint, warning, and conflict lines are masked before they hit stderr. `RedactingFormatter` masks opaque credential values in all log output — both shape-based tokens (`sk-`, `ghp_`, auth headers) and exact credential values with no recognizable prefix (`MY_SERVICE_TOKEN=…`, `*_PASSWORD=…`).
4. **The vault token and passwords never reach child processes.** `BWS_ACCESS_TOKEN` (under its exact configured name) and every `*_PASSWORD` are stripped from spawned children on every surface — terminal commands, the browser worker, the ACP executor, the computer-use driver, and the TUI/Node host. The only process that receives the token is the `bws` CLI itself, explicitly, never by inheritance.
5. **Applied secrets are stripped from children by provenance, not just by name.** Beyond the named strip, every value present in the per-home applied-secrets snapshot is removed from spawned children regardless of the variable name — `DATABASE_URL`, `FOO`, or any arbitrary item key resolve to nothing in a child's environment. Only variables registered through `env_passthrough` pass through, deliberately and by explicit configuration.
6. **Secret values are redacted before anything reaches the model provider.** Applied secret values are masked in tool-result content, in the pre-send sanitization pass over context, and in terminal output before the request is sent to the provider. A secret can appear in an agent's context as `***`, never as its value.
7. **The 1Password integration ships the same posture.** The 1Password cache is encrypted-only (`~/.hermes/cache/op_cache.enc.json`), and `OP_SERVICE_ACCOUNT_TOKEN`, `OP_CONNECT_TOKEN`, and `OP_SESSION_*` are stripped from spawned children the same way as the BWS token.

The guarantee is tested, not promised: the end-to-end no-exfiltration gate (`tests/test_secrets_exfiltration.py`) pins the posture on the wire — a secret applied from any external source and echoed by a tool arrives at the provider-bound message carrying the mask, never the value, and the key name survives for debugging. The emission-side channels are pinned by the same suite: secret names and values never surface in stdout, stderr, or formatted output. Any regression that re-opens a channel fails CI.

## Token rotation

Rotation is how you maintain the posture. The machine-account access token is the single credential that unlocks every applied secret, so rotating it on a schedule — and immediately if it may have leaked — keeps the guarantee airtight. The rotation path is fully masked and validates before it writes, so routine maintenance never weakens the posture.

1. In the Bitwarden web app → **Secrets Manager** → **Machine accounts** → your machine account → **Access tokens**.
2. **Revoke** the existing token(s).
3. **Create a new access token** and copy it (it starts with `0.`). Bitwarden shows it once.
4. In your terminal, run:

   ```bash
   hermes secrets bitwarden token
   ```

5. **Paste the new token value** when prompted (input is hidden).

Clipboard discipline: the new token should go **straight from creation to the terminal** — create it in the web app, copy to clipboard, paste into the `hermes secrets bitwarden token` prompt, and **do not save it anywhere else in between** (no notes app, no file, no chat, no screenshot). If the paste fails, re-copy from the web app rather than retyping the token.

The command probes Bitwarden with the new token **before** writing anything — a rejected token leaves your current `.env` untouched — and on success clears the fetch caches. After rotating, also rotate any high-value secrets (provider API keys, database passwords) that were applied while the old token was live: whoever holds the old token can still read them directly from the vault until those secrets are themselves rotated.

### Rotating an expired or revoked token

When the machine-account token expires, gets revoked, or the account is deleted, startup shows:

```
Bitwarden Secrets Manager: Bitwarden rejected the machine-account access token (BWS_ACCESS_TOKEN) — it was likely revoked, expired, or belongs to another region.  (...)
Bitwarden Secrets Manager: → Run `hermes secrets bitwarden token` to paste a fresh access token ...
```

Fix it without re-running the whole wizard:

```bash
hermes secrets bitwarden token                     # masked prompt
hermes secrets bitwarden token --access-token 0.…  # non-interactive
```

On success the command stores the token, clears the fetch caches, and warns if the configured project is not visible to the new machine account.

## How it works

1. You create a **machine account** in Bitwarden Secrets Manager, give it read access to a project, and generate an **access token**.
2. Hermes stores that single token in `~/.hermes/.env` as `BWS_ACCESS_TOKEN`.
3. Every time `hermes` (or the gateway, or a cron job) starts, after `~/.hermes/.env` has loaded, Hermes calls `bws secret list <project_id>` and sets the returned keys into `os.environ`.
4. By default Hermes **overrides** values already in your environment, so Bitwarden is the source of truth — rotate a key once in the web app and every Hermes process picks it up on next start. Flip `override_existing: false` in config if you want `.env` to win instead.

The `bws` binary is auto-downloaded into `~/.hermes/bin/` on first use — no `apt`, no `brew`, no `sudo`.

## Why machine accounts (and why no 2FA prompt)

Bitwarden Secrets Manager is designed for non-interactive workloads: machine accounts can't be 2FA-gated because there's no human in the loop. The access token is the credential. Anyone with it can read every secret the machine account has access to, so treat it like a high-value bearer token — store it in `.env` (not `config.yaml`), and revoke + regenerate from the Bitwarden web app if it ever leaks.

You set up the machine account *in the web app*, where your normal 2FA applies. After that the token is autonomous.

## Setup

### 1. Create a machine account and access token

In the [Bitwarden web app](https://vault.bitwarden.com) (or [vault.bitwarden.eu](https://vault.bitwarden.eu) for EU accounts):

1. Switch to **Secrets Manager** from the product switcher.
2. Create or pick a **Project** (e.g. "Hermes keys").
3. Add your provider keys as secrets. The secret **Name** becomes the environment variable name — use `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.
4. **Machine accounts → New machine account → My Hermes machine** → **Projects** tab → grant Read access to your project.
5. **Access tokens** tab → **Create access token** → **Never** expires (or pick a date) → copy the token (starts with `0.`). Bitwarden cannot retrieve it again — keep the copy.

Secrets Manager is included on the Bitwarden free tier with limits; no paid plan needed to try this.

### 2. Run the wizard

```bash
hermes secrets bitwarden setup
```

It will:

1. Download and verify `bws v2.0.0` into `~/.hermes/bin/bws`.
2. Prompt you for the access token (input is hidden). Stored in `~/.hermes/.env` as `BWS_ACCESS_TOKEN`.
3. Ask which Bitwarden region your machine account belongs to — **US Cloud**, **EU Cloud**, or **self-hosted / custom URL**. Stored in `config.yaml` as `secrets.bitwarden.server_url` and passed to `bws` as `BWS_SERVER_URL`.
4. List the projects the machine account can see; pick one. Stored in `config.yaml` as `secrets.bitwarden.project_id`.
5. Test-fetch the project's secrets and show you which env vars will resolve.
6. Flip `secrets.bitwarden.enabled: true`.

Non-interactive setup is also supported via flags:

```bash
hermes secrets bitwarden setup \
  --access-token "$BWS_ACCESS_TOKEN" \
  --server-url https://vault.bitwarden.eu \
  --project-id <project-uuid>
```

### 3. Confirm

```bash
hermes secrets bitwarden status
```

From now on, every `hermes` invocation pulls fresh secrets at startup. You'll see a one-line summary in stderr the first time secrets are applied in a process — with any embedded values masked.

## CLI

| Command | What it does |
|---|---|
| `hermes secrets bitwarden setup` | Interactive wizard (install binary, prompt for token, pick project, test fetch) |
| `hermes secrets bitwarden status` | Show config + binary version + token presence/validation |
| `hermes secrets bitwarden token` | Rotate the access token: validate the new token against Bitwarden, then store it in `.env` |
| `hermes secrets bitwarden sync` | Dry-run: pull secrets now and show what would be applied |
| `hermes secrets bitwarden sync --apply` | Pull and export into the current shell's environment |
| `hermes secrets bitwarden install` | Just download the pinned `bws` binary (no auth required) |
| `hermes secrets bitwarden disable` | Flip `enabled: false`; leaves token + project id in place |

## Configuration

Defaults in `~/.hermes/config.yaml`:

```yaml
secrets:
  bitwarden:
    enabled: false
    access_token_env: BWS_ACCESS_TOKEN
    project_id: ""
    server_url: ""
    cache_ttl_seconds: 300
    encrypted_cache:
      enabled: true
      max_stale_seconds: 0
    override_existing: true
    auto_install: true
```

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. When false, Bitwarden is never contacted. |
| `access_token_env` | `BWS_ACCESS_TOKEN` | Env var name that holds the bootstrap token. Change this if you already use `BWS_ACCESS_TOKEN` for something else. |
| `project_id` | `""` | UUID of the project to sync from. |
| `server_url` | `""` | Bitwarden region or self-hosted endpoint. Empty = `bws` default (US Cloud, `https://vault.bitwarden.com`). Set to `https://vault.bitwarden.eu` for EU Cloud, or your own URL for self-hosted. Plumbed into the `bws` subprocess as `BWS_SERVER_URL`. |
| `cache_ttl_seconds` | `300` | How long an in-process or disk fetch result is reused. Set to `0` to disable fresh-cache reuse. |
| `encrypted_cache.enabled` | `true` | Encrypted-only disk cache. Secret values are persisted **only** as AES-GCM ciphertext, never plaintext. Set `false` to skip disk persistence entirely (memory cache only) — plaintext is never an option. |
| `encrypted_cache.max_stale_seconds` | `0` | On NETWORK/TIMEOUT failures only, allow the encrypted last-good cache to be used up to this age. Authentication failures never fall back. Set `>0` for offline resilience. |
| `override_existing` | `true` | When true, Bitwarden values overwrite anything already in env (so rotation in the web app actually takes effect). Flip to `false` if you want `.env` / shell exports to win locally. |
| `auto_install` | `true` | When true, `bws` is auto-downloaded into `~/.hermes/bin/` on first use. |

## Failure modes

Bitwarden never blocks Hermes startup. If anything goes wrong, you'll see a one-line warning in stderr (values masked) and Hermes continues with whatever credentials `.env` already had:

| Symptom | Cause | Fix |
|---|---|---|
| `BWS_ACCESS_TOKEN is not set` | Enabled in config but token cleared from `.env` | Re-run `hermes secrets bitwarden setup` |
| `Bitwarden rejected the machine-account access token … invalid_client` | Token revoked, expired, machine account deleted — or the token belongs to another region (e.g. EU token hitting the US identity endpoint) | Run `hermes secrets bitwarden token` to paste a fresh token; for region mismatches re-run setup and pick EU/self-hosted (or set `secrets.bitwarden.server_url`) |
| `bws exited 1: invalid access token` | Token revoked or wrong | Run `hermes secrets bitwarden token` with a new token |
| `bws timed out` | Network blocked or Bitwarden API slow | Check connectivity to `api.bitwarden.com` (or your `server_url`) |
| `bws binary not available` | `auto_install: false` and `bws` not on PATH | Install manually from [github.com/bitwarden/sdk-sm/releases](https://github.com/bitwarden/sdk-sm/releases) or flip `auto_install` back on |
| `Checksum mismatch` | Download corrupted or tampered | Re-run, will retry; if it persists, file an issue |

Startup warnings include a `→` remediation line telling you exactly which command fixes the failure.

## Security notes

- The bootstrap token (`BWS_ACCESS_TOKEN`) is itself sensitive — anyone with it can read every secret the machine account has access to. Treat it the same as any other API key.
- Hermes will refuse to let Bitwarden overwrite the bootstrap token itself, even with `override_existing: true`. If you store `BWS_ACCESS_TOKEN` as a secret inside the project, it's silently skipped during apply.
- The disk cache is **never** plaintext — AES-GCM encrypted by default, keyed off the bootstrap token, with legacy plaintext caches re-encrypted and removed on first read. There is no plaintext write path.
- Secret values are masked in status lines, log output, and terminal output; the vault token, every `*_PASSWORD`, and every applied secret value are stripped from spawned child processes.
- Applied secret values are redacted before anything reaches the model provider — tool results, pre-send context, and terminal output all carry the mask, never the value.
- The 1Password integration follows the same posture: encrypted-only `op_cache.enc.json`, with `OP_SERVICE_ACCOUNT_TOKEN`, `OP_CONNECT_TOKEN`, and `OP_SESSION_*` stripped from children.
- The `bws` binary download is verified against the published SHA-256 checksum from the same GitHub release. Mismatch aborts the install.
- The pinned version (`bws v2.0.0` at time of writing) is updated through PRs to this repo — Hermes does not auto-upgrade `bws` to "latest" because upstream release shapes can change.

## When NOT to use this

- **Single-machine personal setups** where `~/.hermes/.env` is fine. You're trading one credential for another and adding a network dependency at startup.
- **Air-gapped environments** that can't reach `api.bitwarden.com`.
- **CI/CD** where the existing secrets-injection mechanism (GitHub Actions secrets, Vault, etc.) is already set up — pick one path, not two.

The good case for this is multi-machine fleets, shared dev boxes, gateway VPSes, or any setup where you want centralized rotation and revocation across multiple Hermes installations.
