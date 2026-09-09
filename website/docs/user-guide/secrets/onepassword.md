# 1Password

Resolve provider API keys from [1Password](https://1password.com/) at process startup instead of storing them in plaintext inside `~/.hermes/.env`. You keep your keys as 1Password items and reference them by `op://vault/item/field`; rotating a credential becomes a single change in 1Password.

## The security posture is the feature

The plaintext-secrets vulnerability class does not exist here. Resolved values live in Hermes' process memory and nowhere else; every disclosure path is closed by design:

- **Never persisted as plaintext.** The disk cache is encrypted-only: resolved values are written to `<hermes_home>/cache/op_cache.enc.json` under AES-GCM, and a plaintext cache file is never written. Any legacy plaintext `op_cache.json` from older versions is migrated into the encrypted cache and removed on first read. With `cache_ttl_seconds: 0`, cache reuse is fully disabled — every fetch resolves fresh and nothing touches disk at all.
- **Never printed.** Resolved values are masked in status lines and logs. Dry-runs and startup summaries show variable names and `op://` references, never the values themselves.
- **Never inherited by children.** `OP_SERVICE_ACCOUNT_TOKEN`, `OP_CONNECT_TOKEN`, and `OP_SESSION_*` are stripped from every spawned child process; the `op` child gets the credential explicitly through its own minimal allowlisted environment, not by inheritance. Resolved values are never exported into any child's environment.

## How it works

1. You install the official [1Password CLI](https://developer.1password.com/docs/cli/get-started/) (`op`) and authenticate it — either with a **service-account token** (headless servers) or an **interactive/desktop session** (your laptop).
2. You map environment-variable names to `op://` references in `~/.hermes/config.yaml`.
3. Every time `hermes` (or the gateway, or a cron job) starts, after `~/.hermes/.env` has loaded, Hermes runs `op read` for each reference and sets the resolved values into `os.environ`.
4. By default Hermes **overrides** values already in your environment, so 1Password is the source of truth — rotate a credential once and every Hermes process picks it up on next start. Flip `override_existing: false` if you want `.env` to win instead.

Hermes never authenticates on your behalf and never downloads `op`: it shells out to your already-installed, already-trusted CLI. If `op` is missing, your session is locked, or a reference is wrong, Hermes prints a one-line warning and continues with whatever credentials `.env` already had — it never blocks startup.

## Authentication

`op` supports two non-interactive-friendly modes; Hermes works with either:

- **Service accounts** (recommended for servers/CI): create a service account in 1Password, grant it read access to the relevant vault, and export its token as `OP_SERVICE_ACCOUNT_TOKEN` in `~/.hermes/.env`. The token is the credential — treat it like any other bearer token.
- **Desktop / interactive sessions** (laptops): run `op signin` (or enable CLI integration in the 1Password app). Hermes passes your `OP_SESSION_*` variables through to the `op` child process. The 1Password cache key includes those session variables, so signing into a different account never serves a value cached under the previous identity.

## Bootstrap token

When you authenticate with a **service-account token**, that token is itself the bootstrap credential Hermes needs *before* it can resolve any `op://` reference. It must be present in `os.environ` of every process that resolves secrets — including cron jobs (`kanban.dispatch_in_gateway: false`), subprocess invocations, CLI runs, macOS launchd agents, and Docker containers — not just the interactive gateway. There are three ways to make it available, in order of precedence:

1. **In `~/.hermes/.env` (recommended).** `hermes secrets onepassword setup --token <token>` writes the token to `~/.hermes/.env`, exactly like Bitwarden's `BWS_ACCESS_TOKEN`. Because `load_hermes_dotenv()` always loads `.env`, the token is available everywhere with zero extra setup. This is the simplest reliable option.

2. **In `~/.hermes/.op.env` (gitignored).** If you'd rather keep the service-account token out of `.env` — for example so `.env` can be checked into a private dotfiles repo while the token stays out of version control — place it in `~/.hermes/.op.env`:

   ```bash
   echo 'OP_SERVICE_ACCOUNT_TOKEN=ops_...' > ~/.hermes/.op.env
   chmod 600 ~/.hermes/.op.env
   ```

   Hermes auto-loads `.op.env` at startup, **after** `.env`, and **never** overrides a token already present in the environment. `.op.env` is gitignored so the token never enters a committed file.

3. **Via systemd `EnvironmentFile` (Linux gateway).** If you run the gateway under systemd, you can inject the token directly into the service environment:

   ```ini
   [Service]
   EnvironmentFile=-/home/youruser/.hermes/.op.env
   ```

   A token injected this way takes precedence — Hermes detects that `OP_SERVICE_ACCOUNT_TOKEN` is already set and skips loading `.op.env` entirely.

If the token is reachable only through an interactive shell (`op signin`, `OP_SESSION_*` exports in `.bashrc`, etc.), it will **not** be inherited by cron jobs or freshly spawned subprocesses, and those contexts will log a warning and fall back to whatever credentials `.env` already held. Use one of the three options above for any non-interactive workload.

## Setup

### 1. Install and sign in to `op`

Follow the [1Password CLI getting-started guide](https://developer.1password.com/docs/cli/get-started/). Verify it works:

```bash
op whoami
```

### 2. Enable the integration

```bash
hermes secrets onepassword setup
```

This verifies `op` is on `PATH` (or use `--binary-path`), records your account/token settings, checks for an active session, and flips `secrets.onepassword.enabled: true`. Non-interactive flags:

```bash
hermes secrets onepassword setup \
  --account my.1password.com \
  --token-env OP_SERVICE_ACCOUNT_TOKEN \
  --token "$OP_SERVICE_ACCOUNT_TOKEN"
```

### 3. Map your credentials

The reference format is `op://<vault>/<item>/<field>`:

```bash
hermes secrets onepassword set OPENAI_API_KEY    "op://Private/OpenAI/api key"
hermes secrets onepassword set ANTHROPIC_API_KEY "op://Private/Anthropic/credential"
```

### 4. Preview and confirm

```bash
hermes secrets onepassword sync     # dry-run: resolve now, show what would apply
hermes secrets onepassword status   # config + binary + references + auth
```

From now on, every `hermes` invocation resolves the references at startup. You'll see a one-line summary in stderr the first time secrets are applied in a process.

## CLI

| Command | What it does |
|---|---|
| `hermes secrets onepassword setup` | Verify `op`, set account / token env var, enable |
| `hermes secrets onepassword status` | Show config, binary, auth, and configured references |
| `hermes secrets onepassword token` | Rotate the service-account token: validate with `op whoami`, then store it in `.env` |
| `hermes secrets onepassword set ENV_VAR "op://…"` | Map an env var to a reference (stored stripped + validated) |
| `hermes secrets onepassword remove ENV_VAR` | Drop a mapping |
| `hermes secrets onepassword sync` | Dry-run: resolve references now and show what would apply |
| `hermes secrets onepassword sync --apply` | Resolve and export into the current shell's environment |
| `hermes secrets onepassword disable` | Flip `enabled: false`; leaves mappings in place |

`op` and `1password` are accepted as aliases for `onepassword`.

## Configuration

Defaults in `~/.hermes/config.yaml`:

```yaml
secrets:
  onepassword:
    enabled: false
    env:
      OPENAI_API_KEY: "op://Private/OpenAI/api key"
      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"
    account: ""
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
    binary_path: ""
    cache_ttl_seconds: 300
    override_existing: true
```

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. When false, `op` is never invoked. |
| `env` | `{}` | Mapping of env-var name → `op://vault/item/field` reference. Entries whose name isn't a valid env-var name, or whose value isn't an `op://` reference, are skipped with a warning. |
| `account` | `""` | Account shorthand / sign-in address passed as `op read --account`. Empty uses `op`'s default account. |
| `service_account_token_env` | `OP_SERVICE_ACCOUNT_TOKEN` | Env var Hermes reads the service-account token from. Its value is exported to the `op` child as `OP_SERVICE_ACCOUNT_TOKEN` (the name `op` expects). Leave the var unset to use a desktop/interactive session. |
| `binary_path` | `""` | Absolute path to `op`. When set, it is used verbatim and `PATH` is **not** consulted — pin this to avoid trusting whatever `op` appears first on `PATH`. |
| `cache_ttl_seconds` | `300` | How long resolved values are reused (in-process and in the encrypted disk cache). Set to `0` to disable fresh-cache reuse — every fetch resolves fresh and no values are written to disk at all. |
| `override_existing` | `true` | When true, resolved values overwrite anything already in env (so rotation takes effect). Flip to `false` to let `.env` / shell exports win; those references are then skipped *before* `op` is invoked. |

## Failure modes

1Password never blocks Hermes startup. If anything goes wrong you'll see a one-line warning in stderr and Hermes continues:

| Symptom | Cause | Fix |
|---|---|---|
| `the op CLI was not found on PATH` | `op` not installed / not on PATH | Install the CLI, or set `secrets.onepassword.binary_path` |
| `op read failed for 'op://…': …` | Locked session, expired token, or no vault access | `op signin`, run `hermes secrets onepassword token` to rotate the service-account token, or grant the service account access |
| `op read returned an empty value for 'op://…'` | The referenced field exists but is empty | Fix the item/field in 1Password (an empty value is never applied — your existing env var is left intact) |
| `… is not an op:// secret reference` | A mapping value isn't an `op://` reference | Re-set it with the correct `op://vault/item/field` form |
| `op read timed out` | Network blocked or 1Password slow | Check connectivity / the desktop app integration |

Startup warnings now include a `→` remediation line telling you exactly which command fixes the failure.

## Caching

Successful, complete pulls are cached in-process and on disk so back-to-back short-lived `hermes` invocations don't re-shell `op` for every reference. The disk cache is **encrypted-only**: resolved values are stored under `<hermes_home>/cache/op_cache.enc.json` (written atomically, mode `0600`), encrypted with AES-GCM. A plaintext cache is never written.

- stores only resolved secret **values** — never the service-account token or any raw auth material (auth is fingerprinted into the cache key);
- is invalidated when the token, account, `OP_SESSION_*` variables, or the set of references change;
- is **not** written when a pull had any per-reference error, so a transient auth failure isn't frozen in for the TTL;
- **migrates and removes** any legacy plaintext `op_cache.json` left by older versions on first read — after that, only the encrypted cache exists on disk;
- is fully disabled — reads *and* writes — when `cache_ttl_seconds: 0`: every fetch resolves fresh and nothing is written to disk at all.

## Security notes

- A 1Password service-account token can read every secret the account has access to. Store it in `~/.hermes/.env` (not `config.yaml`), and revoke + regenerate from 1Password if it leaks.
- Hermes refuses to let a resolved value overwrite the token env var itself, even with `override_existing: true`.
- The disk cache is **never plaintext** — resolved values are AES-GCM encrypted at rest in `op_cache.enc.json`, and the legacy plaintext cache is migrated and removed on first read.
- `OP_SERVICE_ACCOUNT_TOKEN`, `OP_CONNECT_TOKEN`, and `OP_SESSION_*` are stripped from **every** spawned child process — children never inherit the bootstrap credential, the Connect server token, or the 1Password session. The `op` child gets the credential explicitly: Hermes exports it into the child's own minimal allowlisted environment (auth/session vars + `PATH`/`HOME`), not a copy of the full `os.environ`, so post-dotenv provider credentials aren't inherited either.
- Resolved values are masked in status lines and logs — dry-runs and summaries show variable names and `op://` references, never the values themselves.
- References are validated to start with `op://`, and the reference is passed after a `--` option terminator so a crafted value can't be parsed as an `op` flag.

## When NOT to use this

- **Single-machine personal setups** where `~/.hermes/.env` is fine.
- **Air-gapped environments** that can't reach 1Password.
- **CI/CD** where an existing secrets-injection mechanism is already wired up — pick one path, not two.

The good case for this is multi-machine fleets, shared dev boxes, gateway VPSes, or anywhere you want centralized rotation and revocation across multiple Hermes installations.
