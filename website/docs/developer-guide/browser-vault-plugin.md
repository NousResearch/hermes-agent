---
title: Browser Vault Plugins
description: Explicit, profile-scoped login backends for trusted in-process plugins
---

Browser login backends are separate from startup provider secret sources. A
`SecretSource` registration does not make a password manager available to
`browser_vault`.

An **in-process** plugin may implement `agent.vault_backends.base.LoginBackend`
and call `ctx.register_login_backend(instance)` in `register(ctx)`. The existing
plugin registration ledger scopes it to the current home and disposes/restores
registrations on unload/reload. Host-isolated plugin RPC is not supported by
this extension because `fill_context()` is a server-local context manager.

Every custom backend requires `vault.<name>.enabled: true`; installed code is
never automatically enabled. Names must be lowercase identifiers; custom
handle prefixes use `name:` syntax. Builtin names/prefixes and another backend's
prefix cannot be claimed.

Construct the backend without fetching credentials or reading profile state.
Read the active profile's config lazily. `list_items()` and `get_meta()` return
only `VaultItemMeta` with exact HTTPS origins and opaque IDs. Passwords are
resolved only through `resolve_password()` after the existing browser origin
precheck. Exceptions must contain fixed diagnostics, never credential values or
secret-manager response bodies.

`fill_context()` defaults to a no-op. A backend may override it to lock fills
and freeze credential-reference/origin configuration for the entire operation.
This prevents reload from pairing old metadata with a newly configured secret.
The context must reset/release on failure. A fill lock does not serialize
unrelated browser navigation; applications must separately establish ownership
of their persistent browser profile.

The existing browser tool keeps its atomic in-page origin assertion, secret CDP
evaluation with no argv fallback, and egress redaction registration. A plugin
must not bypass those through a model-facing password-returning tool.
