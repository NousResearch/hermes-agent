---
sidebar_position: 10
title: "Login Backend Plugins"
description: "How to add a third-party password manager to the Hermes credential vault"
---

# Building a Login Backend Plugin

Third-party password managers such as Proton Pass belong in standalone plugins,
not the Hermes core repository. A plugin supplies a `LoginBackend` class for its
known CLI contract; Hermes owns configuration discovery, lifecycle cleanup, and
handle routing.

```python
from agent.vault_backends.base import LoginBackend


class ProtonPassBackend(LoginBackend):
    name = "protonpass"          # config section: vault.protonpass
    display_name = "Proton Pass"
    prefix = "pp:"               # globally unique handle namespace
    needs_unlock = False          # scoped token authentication needs no prompt

    def __init__(self, config):
        self.config = config

    @classmethod
    def is_available(cls, config):
        # Non-interactive only: check an explicit binary_path or PATH.
        return True

    def list_items(self): ...     # metadata only, never passwords
    def get_meta(self, handle): ...
    def resolve_password(self, handle): ...


def register(ctx):
    ctx.register_login_backend(ProtonPassBackend)
```

`is_available(config)` must not authenticate, prompt, or invoke a command that
could block. Hermes calls it before constructing the backend. Your backend owns
the contents of `vault.<name>` and must keep credentials server-side: list
methods return metadata only, while `resolve_password()` and optional
`resolve_otp()` provide secrets only at fill time.

Registration is profile-scoped and is removed when the plugin unloads. Names
`local`, `onepassword`, and `bitwarden`, plus prefixes `vault_`, `op:`, and
`bw:`, are reserved. A duplicate prefix is rejected so a plugin cannot hijack
another manager's stored handles.
