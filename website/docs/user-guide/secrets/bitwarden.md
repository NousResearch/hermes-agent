# Bitwarden Secrets Manager

Pull API keys from [Bitwarden Secrets Manager](https://bitwarden.com/products/secrets-manager/) at process startup instead of storing them in plaintext inside `~/.hermes/.env`. One bootstrap secret (a machine-account access token) replaces N per-provider keys, and rotating a credential becomes a single change in the Bitwarden web app.

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

From now on, every `hermes` invocation pulls fresh secrets at startup. You'll see a one-line summary in stderr the first time secrets are applied in a process.

## CLI

| Command | What it does |
|---|---|
| `hermes secrets bitwarden setup` | Interactive wizard (install binary, prompt for token, pick project, test fetch) |
| `hermes secrets bitwarden status` | Show config + binary version + token presence |
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
    override_existing: true
    auto_install: true
    key_prefix: ""
    strip_prefix: false
```

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. When false, Bitwarden is never contacted. |
| `access_token_env` | `BWS_ACCESS_TOKEN` | Env var name that holds the bootstrap token. Change this if you already use `BWS_ACCESS_TOKEN` for something else. |
| `project_id` | `""` | UUID of the project to sync from. |
| `server_url` | `""` | Bitwarden region or self-hosted endpoint. Empty = `bws` default (US Cloud, `https://vault.bitwarden.com`). Set to `https://vault.bitwarden.eu` for EU Cloud, or your own URL for self-hosted. Plumbed into the `bws` subprocess as `BWS_SERVER_URL`. |
| `cache_ttl_seconds` | `300` | How long an in-process fetch result is reused. Set to `0` to disable caching. Cache is per-process; new `hermes` invocations start fresh. |
| `override_existing` | `true` | When true, Bitwarden values overwrite anything already in env (so rotation in the web app actually takes effect). Flip to `false` if you want `.env` / shell exports to win locally. |
| `auto_install` | `true` | When true, `bws` is auto-downloaded into `~/.hermes/bin/` on first use. |
| `key_prefix` | `""` | Optional filter for shared Bitwarden projects. When set, Hermes loads only secrets whose names start with this prefix. |
| `strip_prefix` | `false` | When true and `key_prefix` is set, Hermes removes the prefix before exporting the env var. Example: `PROFILE_PM_OPENAI_API_KEY` becomes `OPENAI_API_KEY` with `key_prefix: PROFILE_PM_`. |

### Profile-prefixed secrets

If several Hermes profiles share one Bitwarden Secrets Manager project, avoid storing profile-specific values under the same exact env-var name. Use a profile namespace instead:

```yaml
secrets:
  bitwarden:
    enabled: true
    project_id: "..."
    key_prefix: PROFILE_PM_
    strip_prefix: true
```

With that config, a Bitwarden secret named `PROFILE_PM_TELEGRAM_BOT_TOKEN` is exported to the process as `TELEGRAM_BOT_TOKEN`, while `PROFILE_SOCIAL_TELEGRAM_BOT_TOKEN` is ignored by this profile.

## Failure modes

Bitwarden never blocks Hermes startup. If anything goes wrong, you'll see a one-line warning in stderr and Hermes continues with whatever credentials `.env` already had:

| Symptom | Cause | Fix |
|---|---|---|
| `BWS_ACCESS_TOKEN is not set` | Enabled in config but token cleared from `.env` | Re-run `hermes secrets bitwarden setup` |
| `bws exited 1: invalid access token` | Token revoked or wrong | Generate a new token, re-run setup |
| `[400 Bad Request] {"error":"invalid_client"}` | Token is for a Bitwarden region other than the one `bws` is calling (e.g. EU token hitting the US identity endpoint) | Re-run setup and pick the right region, or set `secrets.bitwarden.server_url` to `https://vault.bitwarden.eu` (or your self-hosted URL) |
| `bws timed out` | Network blocked or Bitwarden API slow | Check connectivity to `api.bitwarden.com` (or your `server_url`) |
| `bws binary not available` | `auto_install: false` and `bws` not on PATH | Install manually from [github.com/bitwarden/sdk-sm/releases](https://github.com/bitwarden/sdk-sm/releases) or flip `auto_install` back on |
| `Checksum mismatch` | Download corrupted or tampered | Re-run, will retry; if it persists, file an issue |

## Security notes

- The bootstrap token (`BWS_ACCESS_TOKEN`) is itself sensitive — anyone with it can read every secret the machine account has access to. Treat it the same as any other API key.
- Hermes will refuse to let Bitwarden overwrite the bootstrap token itself, even with `override_existing: true`. If you store `BWS_ACCESS_TOKEN` as a secret inside the project, it's silently skipped during apply.
- The `bws` binary download is verified against the published SHA-256 checksum from the same GitHub release. Mismatch aborts the install.
- The pinned version (`bws v2.0.0` at time of writing) is updated through PRs to this repo — Hermes does not auto-upgrade `bws` to "latest" because upstream release shapes can change.

## When NOT to use this

- **Single-machine personal setups** where `~/.hermes/.env` is fine. You're trading one credential for another and adding a network dependency at startup.
- **Air-gapped environments** that can't reach `api.bitwarden.com`.
- **CI/CD** where the existing secrets-injection mechanism (GitHub Actions secrets, Vault, etc.) is already set up — pick one path, not two.

The good case for this is multi-machine fleets, shared dev boxes, gateway VPSes, or any setup where you want centralized rotation and revocation across multiple Hermes installations.
