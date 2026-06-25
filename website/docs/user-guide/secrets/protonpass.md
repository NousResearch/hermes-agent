# Proton Pass

Resolve provider API keys from [Proton Pass](https://proton.me/pass) at process startup instead of storing them in plaintext inside `~/.hermes/.env`. You keep your keys as Proton Pass items and reference them by `pass://vault/item/field`; rotating a credential becomes a single change in Proton Pass — every Hermes process picks up the new value on its next start (or, for the long-running gateway, within `cache_ttl_seconds`), with no container rebuild or `.env` edit.

## How it works

1. You install the official [Proton Pass CLI](https://protonpass.github.io/pass-cli/) (`pass-cli`) and authenticate it once with a **personal access token** (works headlessly in containers/CI). `pass-cli` stores a session in a platform keyring that persists across invocations.
2. You map environment-variable names to `pass://` references in `~/.hermes/config.yaml`.
3. Every time `hermes` (or the gateway, or a cron job) starts, after `~/.hermes/.env` has loaded, Hermes resolves each reference and sets the resolved values into `os.environ`.
4. By default Hermes **overrides** values already in your environment, so Proton Pass is the source of truth. Flip `override_existing: false` if you want `.env` to win instead.

Hermes never downloads `pass-cli`: it shells out to your already-installed CLI. If `pass-cli` is missing, your session can't be established, or a reference is wrong, Hermes prints a one-line warning and continues with whatever credentials `.env` already had — it never blocks startup.

## How resolution works

For each `pass://` reference Hermes runs (conceptually):

```bash
ENV_VAR='pass://vault/item/field' pass-cli run --no-masking -- <python> -c "<echo the env var>"
```

with the reference placed in that environment variable. The `run` command substitutes the `pass://` URI for the real secret value before executing the command; `--no-masking` is required so the value reaches stdout instead of being replaced with `<concealed by Proton Pass>`. Hermes wraps its own Python interpreter (rather than a shell builtin like `printenv`) so resolution behaves identically on Linux, macOS, and Windows. This uses only documented `pass-cli` behaviour and avoids the decorated output of `pass-cli item view`.

## Authentication

Proton Pass uses a **persistent session**, unlike the per-invocation token model of some other CLIs:

- Create a **personal access token** (PAT) in Proton Pass, scoped to the vault(s) Hermes needs.
- Make the token available to Hermes as `PROTON_PASS_PERSONAL_ACCESS_TOKEN` (see [Bootstrap token](#bootstrap-token)).
- On startup Hermes relies on an existing `pass-cli` session if one is present. Only when a resolve fails for an auth/session reason does Hermes run `pass-cli login` (consuming the token from the configured env var, passed via the child environment — never on the argv) and retry once.

You can also establish the session out of band — e.g. run `pass-cli login` once in your container entrypoint — and leave the token env var unset; Hermes will simply use the session.

## Bootstrap token

The personal access token is the one bootstrap credential Hermes may need *before* it can (re)establish a session. When you rely on Hermes to log in, the token must be present in `os.environ` of every process that resolves secrets — including cron jobs, subprocess invocations, CLI runs, and Docker containers. Put it in `~/.hermes/.env` (recommended), exactly like Bitwarden's `BWS_ACCESS_TOKEN`:

```bash
echo 'PROTON_PASS_PERSONAL_ACCESS_TOKEN=pst_...' >> ~/.hermes/.env
chmod 600 ~/.hermes/.env
```

If the token is reachable only through an interactive shell, it will **not** be inherited by cron jobs or freshly spawned subprocesses — establish a session out of band for those contexts, or place the token in `.env`.

## Setup

### 1. Install and log in to `pass-cli`

Follow the [Proton Pass CLI docs](https://protonpass.github.io/pass-cli/). Authenticate non-interactively with a personal access token:

```bash
PROTON_PASS_PERSONAL_ACCESS_TOKEN='pst_...' pass-cli login
pass-cli vault list   # verify
```

### 2. Map your credentials and enable

Edit `~/.hermes/config.yaml`:

```yaml
secrets:
  protonpass:
    enabled: true
    env:
      OPENAI_API_KEY: "pass://Private/OpenAI/api key"
      ANTHROPIC_API_KEY: "pass://Private/Anthropic/credential"
```

From now on, every `hermes` invocation resolves the references at startup. You'll see a one-line summary in stderr the first time secrets are applied in a process.

## Configuration

Defaults in `~/.hermes/config.yaml`:

```yaml
secrets:
  protonpass:
    enabled: false
    env: {}
    personal_access_token_env: PROTON_PASS_PERSONAL_ACCESS_TOKEN
    binary_path: ""
    cache_ttl_seconds: 300
    override_existing: true
```

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. When false, `pass-cli` is never invoked. |
| `env` | `{}` | Mapping of env-var name → `pass://vault/item/field` reference. Entries whose name isn't a valid env-var name, or whose value isn't a `pass://` reference, are skipped with a warning. |
| `personal_access_token_env` | `PROTON_PASS_PERSONAL_ACCESS_TOKEN` | Env var Hermes reads the personal access token from when it needs to (re)establish a `pass-cli` session. Leave the var set out of band if you manage the session yourself. |
| `binary_path` | `""` | Absolute path to `pass-cli`. When set, it is used verbatim and `PATH` is **not** consulted — pin this to avoid trusting whatever `pass-cli` appears first on `PATH`. |
| `cache_ttl_seconds` | `300` | How long resolved values are reused (in-process and on disk). Set to `0` to disable **both** cache layers — no values are written to disk at all. |
| `override_existing` | `true` | When true, resolved values overwrite anything already in env (so rotation takes effect). Flip to `false` to let `.env` / shell exports win; those references are then skipped *before* `pass-cli` is invoked. |

## Failure modes

Proton Pass never blocks Hermes startup. If anything goes wrong you'll see a one-line warning in stderr and Hermes continues:

| Symptom | Cause | Fix |
|---|---|---|
| `pass-cli not found` | `pass-cli` not installed / not on PATH | Install the CLI, or set `secrets.protonpass.binary_path` |
| `pass-cli run failed for 'pass://…': …` | Bad reference, no vault access, or a locked session | Fix the reference, grant the token access, or re-`login` |
| `pass-cli login failed: …` | Missing/expired/invalid token | Refresh `PROTON_PASS_PERSONAL_ACCESS_TOKEN` |
| `pass-cli returned an empty value for 'pass://…'` | The referenced field exists but is empty | Fix the item/field in Proton Pass (an empty value is never applied — your existing env var is left intact) |
| `… is not a pass:// secret reference` | A mapping value isn't a `pass://` reference | Re-set it with the correct `pass://vault/item/field` form |

## Caching

Successful, complete pulls are cached in-process and on disk under `<hermes_home>/cache/protonpass_cache.json` (written atomically, mode `0600`), so back-to-back short-lived `hermes` invocations don't re-shell `pass-cli` for every reference. The cache:

- stores only resolved secret **values** — never the personal access token or any raw auth material (the token is fingerprinted into the cache key);
- is invalidated when the token or the set of references change;
- is **not** written when a pull had any per-reference error, so a transient auth failure isn't frozen in for the TTL;
- is fully disabled — reads *and* writes — when `cache_ttl_seconds: 0`.

## Security notes

- A Proton Pass personal access token can read every secret in the vaults it's scoped to. Store it in `~/.hermes/.env` (not `config.yaml`), scope it as narrowly as possible, and revoke + regenerate from Proton Pass if it leaks.
- Hermes refuses to let a resolved value overwrite the token env var itself, even with `override_existing: true`.
- The `pass-cli` child process gets a minimal allowlisted environment (session/keyring vars + `PATH`/`HOME`), not a copy of the full `os.environ`, so post-dotenv provider credentials aren't all inherited by the child.
- References are validated to start with `pass://`, and resolution goes through `pass-cli run`'s documented substitution rather than string-splicing values into a shell.

## When NOT to use this

- **Single-machine personal setups** where `~/.hermes/.env` is fine.
- **Air-gapped environments** that can't reach Proton.
- **CI/CD** where an existing secrets-injection mechanism is already wired up — pick one path, not two.

The good case for this is multi-machine fleets, shared dev boxes, gateway VPSes, or anywhere you want centralized rotation and revocation across multiple Hermes installations.
