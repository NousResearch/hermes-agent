---
sidebar_position: 27
title: Browser Login Backend Plugins
---

# Browser login backend plugins

A login backend supplies password-manager metadata and resolves a password privately
for `browser_vault_fill`. It is not a startup secret source, a new model-facing tool,
or a browser extension. Register a **class** through the ordinary plugin lifecycle:

```python
from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultItemMeta

class MyLoginBackend(LoginBackend):
    name = "myvault"            # vault.myvault configuration namespace
    display_name = "My Vault"
    prefix = "myvault:"         # opaque, stable item handle namespace
    needs_unlock = False

    @classmethod
    def is_available(cls, config: dict[str, object]) -> bool:
        # Check installed SDK/CLI and non-secret references only.
        # Do not authenticate, access a credential, prompt, or construct a client.
        return prerequisites_present(config)

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config

    def list_items(self) -> list[VaultItemMeta]:
        return read_metadata(self.config)

    def get_meta(self, handle: str) -> VaultItemMeta | None:
        return read_item_metadata(self.config, handle)

    def resolve_password(self, handle: str) -> str:
        # Private runtime call, never a registered tool returning plaintext.
        return read_one_password(self.config, handle)


def register(ctx):
    ctx.register_login_backend(MyLoginBackend)
```

The example's `prerequisites_present` and `read_*` functions are provider-specific;
it is an interface sketch, not a working password manager.

## Activation and lifetime

Enable the native plugin through the normal plugin controls. Separately, configure
`vault.<backend-name>.enabled: true` with a literal YAML boolean. Third-party login
backends default to **off**, even when their prerequisites are present. The built-in
1Password and Bitwarden providers retain their existing opt-out behavior.

```yaml
plugins:
  enabled:
    - my-vault-plugin
vault:
  myvault:
    enabled: true
    binary_path: /absolute/path/to/manager-cli
```

`hermes vault sources` and Desktop vault settings discover enabled plugins in a
fresh process. `hermes vault sources --enable myvault` writes the explicit opt-in;
`--disable myvault` stops routing its handles. These are configuration mutations:
follow your installation's configuration approval policy.

Registrations belong to the active resolved Hermes profile, not a process-global
fallback. Plugin unload, reload and failed registration use the existing registration
ownership ledger. No backend instances or credentials are cached by this registry.
Hermes gives availability probes and constructors separate deep copies of the current
profile's `vault.<name>` settings. Availability requires literal `True`; probe or
constructor failures skip that backend with a fixed diagnostic, not exception text.
The base availability implementation returns `False` until overridden.

Built-in names and prefixes are reserved. Duplicate names and equal **or overlapping**
prefixes are rejected: `example:` and `example:child:` cannot coexist. The first
successful registration keeps its namespace; the conflicting registration fails.
Do not override `owns()` to claim another provider's handles.

## Credential boundary

Native plugins are trusted Python code, not sandboxed credential providers. Registering
a backend does not grant security isolation from same-user shell or CDP access. Keep
bootstrap values in the provider's established credential store, not YAML or tool args.
Prefer a scoped read-only machine identity. Bind instances, CLI sessions and other state
to `get_hermes_home()` and refuse reuse after a profile change.

Return metadata only from listing and lookup. Validate handles and grants before every
private read. Never log CLI output, tokens, passwords, OTP seeds or credential-bearing
exceptions. `needs_unlock=False` avoids a master-password prompt for token providers;
it does not prove authentication is valid. TOTP support is optional and should be an
explicit authority decision.

For an interactive manager, set `needs_unlock=True` and implement `is_unlocked()`
and `unlock(master_password)`. The existing masked prompt supplies the password
privately. `browser_vault_unlock` accepts an enabled backend's name and rejects
unknown, disabled, or non-unlockable sources before prompting. Headless sessions
without a prompt callback retain the existing `unlock_unavailable` outcome.

### Optional password destination policy

`LoginBackend.matches_origin(self, meta: VaultItemMeta, origin: str) -> bool` is an
optional metadata-only hook. `origin` is a normalized `scheme://host[:port]`, not a
URL with a path. The default accepts exact entries in `meta.allowed_origins`, falling
back to `meta.origin` when that tuple is empty. No saved origins means no fill,
even if an override would accept the destination.

A trusted plugin may override this method to implement its password manager's
website-matching rules. Keep domain/public-suffix logic and any dependencies in the
plugin, not core. Document scheme, port, subdomain and private-suffix boundaries;
never infer a match with a bare string suffix test. The hook must not retrieve
secrets, prompt, log credentials or rely on caller-thread profile context: native
tab selection can call it on the supervisor thread. Use only the supplied metadata
and configuration already bound to the backend instance. Keep the predicate fast
and deterministic; it may run several times per fill.

Hermes calls the policy during native tab selection and again before password
retrieval. Only literal `True` authorizes; exceptions and other return values deny
without exposing the exception. Cards and addresses remain exact-origin and never
call this hook. `browser_vault_enter_code` retains its existing routing; this hook
is for password fills, not an OTP authorization contract.

The fill script still pins the **actual selected origin** and the inspected field
nonce. Navigation to another origin is refused even when that other origin would
also pass the plugin's policy. A broader website policy is not a wildcard injection
target and does not remove the native CDP-only secret transport requirement.

Providers must still raise credential-free errors and avoid logging secrets in
other methods: this hook does not add general runtime exception sanitization.

## Verification

Exercise registration through a real temporary profile, not only registry mocks.
Useful regressions include profile A → B → A, unload/reload, disabled plugins, malformed
configuration, namespace collisions, missing prerequisites, sanitized failures, and
configuration mutation isolation. Browser tests should use synthetic credentials to
prove destination refusal, tab selection, successful filling, and mid-fill navigation
refusal without returning the synthetic password in tool output.
