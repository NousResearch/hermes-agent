---
sidebar_position: 27
title: Browser Login Backend Plugins
description: Register a password-manager backend for browser autofill.
---

# Browser login backend plugins

A login backend supplies item metadata and retrieves credentials for Hermes's existing browser autofill tools.
Publish third-party password-manager integrations as standalone plugins, not additions to the core repository.

Use `LoginBackend` for browser logins. Use a [secret source](/developer-guide/secret-source-plugin/) for credentials that Hermes loads at startup.

## Register a backend

Implement `agent.vault_backends.base.LoginBackend` in your plugin, then register the **class**, not an instance, from its `register(ctx)` entry point:

```python
from hermes_cli.plugins import PluginContext

from .backend import MyLoginBackend


def register(ctx: PluginContext) -> None:
    ctx.register_login_backend(MyLoginBackend)
```

See [Build a Hermes Plugin](/developer-guide/plugins/) for the manifest, installation, and plugin lifecycle.

### Required interface

| Member | Contract |
| --- | --- |
| `name: str` | Configuration key under `vault`. Must match `[a-z][a-z0-9_]*`, for example `myvault`. |
| `display_name: str` | Name shown in the CLI and Desktop, for example `My Vault`. |
| `prefix: str` | Stable handle prefix, for example `myvault:`. Must be nonempty and contain no whitespace. |
| `__init__(self, config: dict[str, object])` | Accept the backend's `vault.<name>` settings. Keep construction inexpensive. Defer authentication and credential retrieval to operations that need them. |
| `is_available(cls, config: dict[str, object]) -> bool` | Class method that checks local prerequisites without constructing a backend, authenticating, retrieving credentials, or prompting. Override the default, which returns `False`. |
| `list_items(self) -> list[VaultItemMeta]` | Return metadata only. A locked backend returns an empty list. |
| `get_meta(self, handle: str) -> VaultItemMeta \| None` | Return item metadata, or `None` when the item does not exist. Validate the handle before accessing the manager. |
| `resolve_password(self, handle: str) -> str` | Retrieve one password for server-side filling. Raise `UnlockRequired(self)` if the manager needs unlocking. Never expose this method as a tool that returns plaintext. |

Import `VaultItemMeta` from `agent.vault_store`.
Each login needs an `id` beginning with the backend's prefix, `kind="login"`, a label, an origin, and a creation timestamp.
The `identifier` field can contain the username. For items with multiple websites, populate `allowed_origins` with normalized origins.
Use `agent.vault_store.normalize_origin()` to normalize each website URL.

Hermes reserves the built-in names `local`, `onepassword`, and `bitwarden`, and their handle prefixes.
Duplicate names and overlapping prefixes are rejected within a profile.
For example, `example:` and `example:child:` cannot coexist because handle routing uses prefix matching.
The first successful registration retains the name and prefix. Do not override `owns()` to claim another backend's handles.

## Enable the source

Enable the plugin through the [plugin controls](/user-guide/features/plugins/), then enable its credential source:

```sh
hermes vault sources --enable myvault
```

The equivalent configuration is:

```yaml
vault:
  myvault:
    enabled: true
```

Use the backend's `name`, which can differ from the plugin's name.
Third-party sources require the literal YAML boolean `true`. Built-in 1Password and Bitwarden sources remain enabled when detected unless explicitly disabled.

`hermes vault sources` and **Settings → Passwords & Logins** discover enabled plugins and show their sources.
Use `hermes vault sources --disable myvault` or the settings switch to disable a source.
Disabling prevents new handle lookups from selecting that backend.

The availability probe and constructor each receive a separate deep copy of `vault.<name>`, not the entire configuration.
Plugins can define additional keys in that section without changing Hermes's defaults.
Keep credentials in the manager's credential store or the profile's secret configuration, not in YAML.
Use `agent.secret_scope.get_secret()` for profile-scoped secrets rather than reading the launch process's environment.

Availability requires a return value of `True`. A probe or constructor failure skips the backend and logs a fixed diagnostic without exception text.
Availability reports local prerequisites, not authentication status.

## Authentication and lifetime

Registrations belong to the active Hermes profile and follow the plugin manager's unload and reload lifecycle.
The registry stores classes, not backend instances or credentials. Hermes creates new instances when it resolves enabled backends.
Bind any reusable state to the owning profile with `get_hermes_home()` from `hermes_constants`.

### Token authentication

Keep `needs_unlock = False` for managers that use a token instead of a master-password prompt.
The backend must authenticate when an operation needs access and report authentication failures without including credentials.
Prefer a scoped, read-only token with only the permissions the integration needs.

### Interactive unlock

Set `needs_unlock = True` and implement `is_unlocked() -> bool` and `unlock(master_password: str) -> None`.
Hermes passes the password from the surface's masked prompt directly to `unlock()`.
Unknown, disabled, and non-unlockable sources are rejected before prompting.
A session without an interactive prompt returns `unlock_unavailable`.

Use `agent.vault_backends.unlock` for session tokens so Desktop's Lock control, source disabling, and session cleanup can clear them:

- Call `begin_unlock(name)` before the manager's unlock operation.
- Pass its generation to `store_session_token(name, token, generation)` after the operation succeeds.
- If storage returns `False`, a concurrent lock invalidated the unlock. Do not retain or use that token.
- Use `is_unlocked(name)` for status and `get_session_token(name)` for authenticated operations.

This store separates profiles and expires tokens after 30 minutes without use. Do not store the master password.

### One-time codes

Override `resolve_otp(self, handle: str) -> str | None` to retrieve a current one-time code.
Set `VaultItemMeta.has_otp` only when the item supports automatic code retrieval.
The default returns `None`, allowing Hermes to ask the user for a code.
Never return the underlying TOTP seed in metadata or tool output.

## Password destination matching

By default, `matches_origin(self, meta: VaultItemMeta, origin: str) -> bool` accepts exact origins from `meta.allowed_origins`.
When that tuple is empty, it uses `meta.origin`. An item without saved origins cannot be filled, even with an override.

A plugin can override this method to implement its manager's website-matching policy, including supported related subdomains.
The `origin` argument is normalized to `scheme://host[:port]` and contains no path.
Keep domain and public-suffix rules, including any dependencies, in the plugin.
Document how the policy handles schemes, ports, subdomains, and private suffixes. A bare hostname suffix test is not sufficient.

The method must be fast, deterministic, and metadata-only.
It may run repeatedly on the browser supervisor's thread, where the caller's profile context is not available.
Use the supplied metadata and configuration already bound to the instance. Do not retrieve credentials, prompt, or read profile-global state.

Hermes applies the policy when selecting a browser tab and again before retrieving the password.
Only `True` authorizes the destination. Exceptions and all other return values deny access without exposing exception text.
Cards and addresses always use exact-origin matching and do not call this method.
The method does not authorize destinations for `browser_vault_enter_code`, whose routing is unchanged.

The fill script checks the selected origin and the inspection nonce before writing any credential.
Navigation to a different origin rejects the fill, even if the new origin would also pass the plugin's policy.

## Credential handling

Native plugins run as trusted Python code. They are not sandboxed.
Return only metadata from `list_items()` and `get_meta()`, and validate handles and access permissions before retrieving credentials.
Never include passwords, tokens, TOTP seeds, or raw manager output in logs or exceptions.
Other backend methods do not receive the exception suppression provided for availability, construction, and destination matching.

Hermes sends fill values directly through the browser supervisor's CDP connection.
Do not put credentials in command-line arguments or return them from model-facing tools.

## Test the integration

Load the plugin through real discovery against temporary profiles.
Verify profile A → B → A isolation, reload and unload, source enablement, namespace collisions, and missing prerequisites.
Check that probes cannot mutate constructor settings and that failure messages contain no credentials.

Use synthetic credentials for browser tests.
Verify successful fills, rejected destinations, selection among multiple tabs, and rejection after navigation between inspection and filling.
Assert that tool output contains no credential values.
