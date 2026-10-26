---
sidebar_position: 3
title: "Managed Scope"
description: "Administrator-pinned, user-immutable config and secrets via a system-level managed directory"
---

# Managed Scope

**Managed scope** lets an administrator push a baseline of configuration and
secrets that a standard (non-root) user **cannot override**. It is intended for
fleet/org deployments where IT needs to pin, for example, the model provider, a
shared API base URL, or `security.redact_secrets: true` across every user on a
machine.

When a managed scope is present, the values it specifies win over the user's
`~/.hermes/config.yaml`, `~/.hermes/.env`, and even the shell environment — for
exactly the keys it pins. Everything else stays fully user-controlled.

:::note Different from a package-manager–locked install
A package-manager–managed install (declarative-distro / formula) blocks *all*
config mutation and tells you to use your package manager. Managed scope is a
separate mechanism: it injects *specific immutable values* on a per-key basis
rather than locking the whole config. The two are independent and can coexist.
:::

## Where it lives

Managed scope is read from a system-level directory, default `/etc/hermes`:

```text
/etc/hermes/
├── config.yaml     # managed config layer (wins over ~/.hermes/config.yaml)
└── .env            # managed env layer (wins over ~/.hermes/.env + shell)
```

The directory and files are owned by `root` (directory mode `0755`, files
`0644`): readable by everyone, writable only by an administrator. **That
filesystem permission is the enforcement mechanism** — a standard user can read
the managed files but cannot edit them.

Either file is optional. A missing managed directory or missing file simply
means "no managed scope," and configuration resolves exactly as it does without
the feature.

### Relocating the directory

The location can be relocated with the `HERMES_MANAGED_DIR` environment variable
(for containers or non-`/etc` deployments). This is a deployment/bootstrap path
knob — like `HERMES_HOME` — set by the same administrator who owns the managed
files. It is **never persisted** to any `.env` by Hermes.

```bash
# Point managed scope at a custom directory (set by IT / the deployment, not the user)
export HERMES_MANAGED_DIR=/opt/org/hermes-policy
```

:::warning
A user who can set `HERMES_MANAGED_DIR` can repoint managed scope at a directory
they control, defeating it. In a real deployment this variable should be fixed
by the administrator (e.g. baked into the service unit / container image), not
left user-settable. `hermes doctor` reports the *resolved* managed directory so
a redirect is visible.
:::

## Precedence

For the keys a managed layer specifies, the order is (highest wins):

| Tier | config.yaml | .env |
|---|---|---|
| 1 | `/etc/hermes/config.yaml` (managed) | `/etc/hermes/.env` (managed) |
| 2 | `~/.hermes/config.yaml` (user) | `~/.hermes/.env` (user) |
| 3 | built-in defaults | pre-existing shell environment |

Merging is **leaf-level**: pinning `model.default` does not freeze the rest of
`model.*`. A managed `config.yaml` of:

```yaml
model:
  default: org/standard-model
```

forces `model.default` for every user while leaving `model.fallback` (and every
other key) under user control.

:::note Precedence note
For the keys it pins, managed scope deliberately wins over the shell environment
too — otherwise it would not be "managed." This is the one place that inverts the
usual "an environment variable overrides config.yaml" rule, and it applies only
to the specific keys the managed layer specifies.
:::

## Seeing what's managed

```bash
hermes config        # shows a header naming the managed source + the pinned keys
hermes doctor        # reports the resolved managed dir + pinned key counts
```

If you try to change a managed value, Hermes refuses and names the source:

```bash
$ hermes config set model.default my/model
Cannot set 'model.default': it is managed by your administrator
(/etc/hermes/config.yaml) and cannot be changed.
```

The same applies to managed secrets — `hermes config set` / setup will not write
a user value for an env key pinned by the managed `.env`.

## Setting up a managed scope (administrators)

```bash
sudo mkdir -p /etc/hermes

# Pin some config values for every user on this machine
sudo tee /etc/hermes/config.yaml >/dev/null <<'YAML'
model:
  provider: nous
security:
  redact_secrets: true
YAML

# Optionally pin a shared, non-sensitive env value
sudo tee /etc/hermes/.env >/dev/null <<'ENV'
OPENAI_API_BASE=https://inference.example.com/v1
ENV

sudo chmod 0755 /etc/hermes
sudo chmod 0644 /etc/hermes/config.yaml /etc/hermes/.env
```

Changes take effect on the next Hermes start (a malformed managed file is logged
loudly and ignored — it never blocks startup, but the admin should check
`hermes doctor` to confirm the policy is being applied).

## Security model and limitations (v1)

- **Enforcement is filesystem permissions only.** If a user has write access to
  the managed directory (or runs Hermes as `root`), managed scope is advisory.
- **The managed `.env` is world-readable** (`0644`), so any local user can read
  secrets pushed through it. Use it for shared, non-sensitive values (an org API
  base URL, feature defaults) rather than high-sensitivity secrets.
- **The agent's own tools are not hard-blocked from a managed *env* value.** A
  managed environment variable is applied at startup, but nothing stops the
  agent from setting a different value inside its own subprocess shell. v1 is a
  management-convenience boundary against a normal user, not an un-escapable
  sandbox.

The following are intentionally **out of scope for v1** and may come later:

- A hard boundary that the agent itself cannot escape.
- Native managed locations on macOS and Windows (v1 is Linux/POSIX-first).
- Drop-in fragment directories (`managed.d/`) for layered policy.
- Signed / integrity-checked managed files.
- Remote / device-management (MDM) delivery.
- Tighter (group-scoped) permissions for managed secrets.
