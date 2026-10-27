---
title: Browser Vault Plugins
description: Add a password manager to browser_vault with ctx.register_login_backend, and the isolation rules it runs under
---

# Browser Vault Plugins

A plugin can add a password manager (KeePass, a team vault, a hardware-backed store) as a login
source for the `browser_vault` tool, when it comes from a verified source: the manager's vendor
through the plugin catalog, or Nous (see [Who can register a backend](#who-can-register-a-backend)). The agent then lists and fills that manager's logins the same way
it does for the built-in Hermes vault, 1Password and Bitwarden: it sees handles and metadata, never a password.

Browser login backends are separate from [secret sources](./secret-source-plugin.md). A `SecretSource`
registration does not make a password manager available to `browser_vault`.

## Registering a backend

An **in-process** plugin subclasses `agent.vault_backends.base.LoginBackend` and registers one instance:

```python
from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultItemMeta


class KeePassBackend(LoginBackend):
    name = "keepass"            # lowercase identifier; also the config key
    prefix = "keepass:"         # must be exactly "<name>:"
    display_name = "KeePass"

    def list_items(self) -> list[VaultItemMeta]:
        ...  # metadata only: ids "keepass:<opaque>", exact HTTPS origins, identifier

    def get_meta(self, handle: str) -> VaultItemMeta | None:
        ...

    def resolve_password(self, handle: str) -> str:
        ...  # called once per fill, after the page origin matched the item


def register(ctx):
    ctx.register_login_backend(KeePassBackend())
```

The backend stays off until the user turns it on for the profile:

```yaml
vault:
  keepass:
    enabled: true   # exactly true; installed code is never enabled automatically
```

Construct the backend without fetching credentials or reading profile state, and read the active
profile's configuration lazily. Registration is profile-scoped and removed when the plugin unloads.

## Who can register a backend

Password-manager integrations come only from verified sources. `ctx.register_login_backend` is
accepted from:

- a plugin bundled with Hermes, or
- a plugin installed from the [Hermes plugin catalog](./plugins/catalog-submission.md) at
  `tier: verified` (published by the password-manager vendor itself) or `tier: official`
  (maintained by Nous), at the catalog's reviewed pin.

Every other plugin is refused at registration with a log line naming this rule: a plugin dropped
into `~/.hermes/plugins/` by hand, a Git URL install, a community-tier catalog entry, a catalog
install moved off its pin with `--ref`, one installed with `--allow-removed`, a project plugin and
a pip entry-point plugin. Provenance is read from Hermes's own install record
(`plugins/.install-metadata.json`, written by the installer), never from a file inside the plugin, so
a repository that ships its own `.hermes-catalog.json` gains nothing. No config key relaxes the rule.

If you build a password manager and want it in Hermes, submit the plugin to the catalog from your
own repository and ask for `verified` tier (catalog admission rule 18). Integrations for managers
whose vendor has not published one come from Nous as `official` entries.

## Isolation rules

Hermes wraps every plugin backend in a core-owned adapter before the vault tool sees it. These rules
hold however the plugin implements its class:

| Rule | What it means for your backend |
|---|---|
| **Own namespace only** | Your handles are `<name>:…`. Hermes decides ownership from the registered name, so it never calls your `owns()` and never passes you a `vault_…` (Hermes vault), `op:…`, `bw:…` or another plugin's handle. Items your `list_items()` returns with an id outside your namespace are dropped, and a `get_meta()` answer whose id is not the requested handle is ignored. |
| **Handle in, value out** | `list_items()` gets no arguments; `get_meta()`, `resolve_password()`, `resolve_otp()` and `resolve_secret()` get your handle and nothing else. You never receive the page, the CDP/browser session, the task, other vault entries, or decrypted values from any other backend. |
| **No master-password prompt** | Hermes' unlock prompt is reserved for built-in managers. A backend with `needs_unlock = True` is refused at registration, and a plugin backend is never offered the `unlock` action, so the user is never asked to type a master password into a dialog a plugin labelled. Authenticate through your manager's own agent, session or OS keychain. |
| **Reserved namespaces** | The names `local`, `onepassword`, `bitwarden`, `vault`, `op` and `bw`, and any prefix inside a built-in prefix (`vault_x:`), are refused. So is a name another plugin already holds in the profile: no backend, built-in or plugin, can be replaced or shadowed. `name` and `prefix` must be plain string attributes, not properties, and the class may not override `__getattribute__` or `__class__`: Hermes judges the registration without running your code. |
| **Off by default** | Only `vault.<name>.enabled: true` turns a backend on. |
| **In-process only** | `ctx.register_login_backend` is refused under `plugins.isolation: host`, because `fill_context()` is a server-local context manager. |
| **Same fill path** | Plugin items (passwords and automatic one-time codes) go through the same exact-origin pre-check, in-page origin re-assertion, supervisor CDP secret evaluation (never argv) and payment confirmation as built-in items. |
| **Secrets stay out of the conversation** | Values you return are registered with the browser-result redaction boundary before they reach the page. Any exception your backend raises is replaced by a fixed message (`The <name> login backend failed.`) before it can reach a tool result, a log line or an exception chain, so a secret in your error text cannot reach the model. A value of the wrong type is refused the same way. Hermes copies what you return (metadata fields, passwords, codes) into its own plain objects inside that same guard, so an error raised while a field is read gets the fixed message too. |

An in-process plugin is ordinary Python running inside Hermes, so these rules are an API contract,
not a sandbox. They guarantee that Hermes never hands your backend another backend's data; they do not
stop arbitrary code in the same process from reading files. The provenance rule above decides which
plugins Hermes offers this seam to; it is not a sandbox either.

## Fill context

`fill_context()` defaults to a no-op. A backend may return its own context manager to lock fills and
freeze credential-reference/origin configuration for the whole operation, so a reload cannot pair old
metadata with a newly configured secret. Hermes enters it only for fills of your own handles (a
password fill, and an automatic one-time code for `browser_vault action=enter_code`, both through the
in-page write), and only after the page origin matched your item's `get_meta()`: a page on another
origin never opens your context. Inside the context Hermes calls `get_meta()` again and refuses the
fill unless the frozen item still binds that page origin, so the origins checked and the secret
resolved come from one snapshot. It calls `fill_context()` with no arguments and exits it with
`(None, None, None)`, so you never see the core's exception. Release on exit even when the fill
failed. A fill lock does not serialize unrelated browser navigation.

## Writing errors

Raise exceptions with fixed diagnostics. Hermes discards your exception text, but your own logs may
still record it, so never put credential values or secret-manager response bodies in it. Never expose
a model-facing tool that returns a password: it would bypass every rule above.
