---
sidebar_position: 9
title: "Secret Source Plugins"
description: "How to build a secret-manager backend plugin for Hermes Agent"
---

# Building a Secret Source Plugin

Secret sources resolve provider credentials from an external secret manager (a vault, a password manager, an OS keystore, a custom script) into environment variables at process startup — after `~/.hermes/.env` loads, before Hermes reads credentials. Bitwarden and 1Password ship in-tree; **every other backend is a plugin**. This guide covers building one.

:::tip
The bundled set is deliberately closed, same policy as [memory providers](/developer-guide/memory-provider-plugin): PRs adding new vault backends under `agent/secret_sources/` are closed with a pointer to this guide. Publish your backend as a standalone plugin repo and share it in the Nous Research Discord (`#plugins-skills-and-skins`).
:::

## What the framework owns vs. what you own

The orchestrator (`agent.secret_sources.registry.apply_all`) owns everything security- and precedence-sensitive, so a backend cannot get it wrong:

| Framework owns | You own |
|---|---|
| Source ordering, mapped-vs-bulk precedence | Fetching values from your backend |
| First-claim-wins conflict handling + warnings | Validating your reference format |
| `override_existing` semantics (never crosses sources) | Talking to your CLI/SDK/API |
| Protected bootstrap tokens | Declaring which env var IS your bootstrap token |
| Per-source wall-clock timeout | Keeping `fetch()` reasonably fast |
| Per-var provenance + `(from X)` labels | A human-readable `label` |
| `os.environ` writes | Nothing — you never touch the environment |

## Directory structure

```
~/.hermes/plugins/my-vault/
├── plugin.yaml      # name, description
└── __init__.py      # SecretSource subclass + register(ctx)
```

## The SecretSource ABC

Implement `agent.secret_sources.base.SecretSource`. One method is required:

```python
from pathlib import Path

from agent.secret_sources.base import (
    ErrorKind,
    FetchResult,
    SecretSource,
    run_secret_cli,
)


class MyVaultSource(SecretSource):
    name = "myvault"          # config section key: secrets.myvault
    label = "My Vault"        # used in startup lines + provenance labels
    shape = "mapped"          # "mapped" (explicit VAR→ref map) or "bulk" (project dump)
    scheme = "mv"             # optional: unique URI scheme you own (mv://...)

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        """Resolve secrets. MUST NOT raise. MUST NOT prompt."""
        result = FetchResult()
        token = os.environ.get("MYVAULT_TOKEN", "").strip()
        if not token:
            result.error = "secrets.myvault.enabled is true but MYVAULT_TOKEN is not set."
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        try:
            proc = run_secret_cli(
                ["myvault-cli", "export", "--json"],
                allow_env=["MYVAULT_TOKEN"],   # ONLY your auth vars — never full os.environ
                timeout=30,
            )
        except RuntimeError as exc:           # spawn failure / timeout
            result.error = str(exc)
            result.error_kind = ErrorKind.BINARY_MISSING
            return result

        if proc.returncode != 0:
            result.error = f"myvault-cli exited {proc.returncode}: {proc.stderr[:200]}"
            result.error_kind = ErrorKind.AUTH_FAILED
            return result

        result.secrets = parse_your_output(proc.stdout)  # {ENV_VAR: value}
        return result

    def protected_env_vars(self, cfg: dict):
        # Your bootstrap token — no source (including yours) may ever overwrite it.
        return frozenset({"MYVAULT_TOKEN"})
```

### Contract rules (enforced, not suggestions)

- **`fetch()` never raises.** Errors go in `result.error` + `result.error_kind`. A raising fetch is contained by the orchestrator and reported as `INTERNAL` — a contract violation, not a feature.
- **`fetch()` never prompts.** Startup runs in non-TTY contexts (gateway, cron, Docker). `run_secret_cli()` closes stdin so a prompting helper fails fast. Interactive auth belongs in your CLI setup flow, never on the startup path.
- **Sync, within budget.** The orchestrator enforces a wall-clock timeout (default 120s, user-tunable via `secrets.<name>.timeout_seconds`). Exceeding it reports `TIMEOUT` and your result is discarded.
- **You fetch; the orchestrator applies.** Return the mapping you *would* contribute. Never write `os.environ` yourself — you'd bypass precedence, conflict detection, and provenance.
- **API versioning.** `SecretSource.api_version` defaults to the current `SECRET_SOURCE_API_VERSION`. The registry skips (with a warning) sources built against a different version instead of crashing startup.

### Choosing your `shape`

- `mapped` — the user explicitly binds env-var names to references in config (like 1Password's `env:` map). Strongest intent: mapped claims beat bulk claims on contested vars.
- `bulk` — you inject a whole project/folder of secrets implicitly (like Bitwarden BSM). Yields to mapped sources.

### Optional hooks

| Method | Default | Override when |
|---|---|---|
| `is_enabled(cfg)` | `cfg.get("enabled")` | Custom activation logic |
| `override_existing(cfg)` | `cfg.get("override_existing", False)` | You want a different default (both bundled sources default `True` for rotation) |
| `protected_env_vars(cfg)` | empty | You have a bootstrap token (you almost certainly do) |
| `fetch_timeout_seconds(cfg)` | 120s | Your backend needs a different budget |
| `config_schema()` | `{}` | Declare config keys for setup surfaces |
| `remediation(kind, cfg)` | generic per-`ErrorKind` hints | You want failure warnings to point at your own fix-it command (e.g. the bundled sources return `Run hermes secrets <name> token…` for `AUTH_FAILED`). Must be a pure kind→string mapping: no I/O, never raises. Return `""` to suppress the hint. |

## Subprocess safety: use `run_secret_cli()`

If your backend shells out to a CLI, use the shared helper instead of `subprocess.run` directly. It gives you the audited posture for free: argv-only (no `shell=True`), a **minimal allowlisted child environment** (by the time sources run, `os.environ` holds every credential Hermes knows — never hand that to a child process), `NO_COLOR` + ANSI-scrubbed stderr, stdin closed, timeout → clean `RuntimeError`. Pass user-supplied reference strings after a `--` terminator in your argv so they can never parse as flags.

## Registering

```python
# __init__.py
def register(ctx):
    ctx.register_secret_source(MyVaultSource())
```

Registration is rejected (with a log warning, never a crash) for: non-`SecretSource` instances, invalid/duplicate names, a `scheme` another source owns, wrong `api_version`, or a `shape` outside `mapped`/`bulk`.

:::note Timing
Plugin discovery runs later in startup than the first `load_hermes_dotenv()` call, so a plugin source is not consulted by the very first env load of the process that discovers it. It IS consulted by every subsequently spawned Hermes process (gateway children, cron sessions, subagents). Bundled sources cover first-process bootstrap.
:::

## Users configure it like any other source

```yaml
secrets:
  sources: [myvault, bitwarden]   # optional ordering
  myvault:
    enabled: true
    # ... your config_schema keys
```

Multi-source precedence, conflict warnings, and `(from My Vault)` provenance labels all work automatically — see the [user-facing secrets docs](/user-guide/secrets/) for the precedence ladder.

## Validate with the conformance kit

Subclass the kit from the Hermes repo (`tests/secret_sources/conformance.py`) in your plugin's tests:

```python
import pytest
from tests.secret_sources.conformance import SecretSourceConformance

class TestMyVaultConformance(SecretSourceConformance):
    @pytest.fixture
    def source(self):
        return MyVaultSource()
```

It checks the rules that break other people when violated: never-raises on malformed config, machine-readable error kinds, disabled-by-default, positive timeouts, valid protected-var names, and a full `apply_all()` round trip. Green conformance is the review bar for calling a backend contract-compliant.

## ErrorKind reference

| Kind | Meaning |
|---|---|
| `NOT_CONFIGURED` | Enabled but missing token / project / map |
| `BINARY_MISSING` | Helper CLI not found or not executable |
| `AUTH_FAILED` / `AUTH_EXPIRED` | Bad / expired credentials |
| `REF_INVALID` | A secret reference failed validation |
| `NETWORK` | Transport-level failure |
| `EMPTY_VALUE` | Backend returned nothing for a ref — never apply `""` over a good credential |
| `TIMEOUT` | Fetch exceeded its budget |
| `INTERNAL` | Anything else (bug, unexpected shape) |
