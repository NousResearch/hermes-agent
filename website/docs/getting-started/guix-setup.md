---
sidebar_position: 3.1
title: "Guix System Setup"
description: "Run Hermes Agent on Guix System via an FHS container, with a shepherd user service for the gateway"
---

# Guix System Setup

:::warning Tier 2 platform
Guix System is a community-maintained platform. There is no in-tree Guix
packaging — this page documents the supported distribution method (the standard
`install.sh` inside an FHS-compatible environment) and what changes on Guix.
It can break on any release, and fixes take priority below Tier 1. See
[Platform Support](./platform-support.md#tier-2).
:::

Guix System differs from the distributions Hermes targets in two ways that
matter here:

1. **No FHS by default.** Binaries and libraries live in the read-only store;
   there is no `/usr/lib` layout for the loader to search. The stock
   `install.sh` assumes an FHS system.
2. **No systemd.** `hermes gateway install` registers a systemd user unit on
   Linux, which has no equivalent on Guix. The service manager on Guix is
   [GNU Shepherd](https://www.gnu.org/software/shepherd/).

Both problems have a clean answer: run Hermes inside an FHS container, and
supervise it with a user shepherd service.

---

## Prerequisites

- Guix with a current channel set (`guix pull` up to date)
- An API key for at least one provider (OpenRouter or Anthropic at minimum)

## 1. Enter an FHS Environment

`guix shell --container --emulate-fhs` builds a throwaway container whose
layout looks like a normal distribution. The flags used here:

| Flag | Effect |
|------|--------|
| `--container` (`-C`) | Isolated container with its own mount namespace |
| `--emulate-fhs` (`-F`) | FHS layout inside: `/bin`, `/usr/lib`, and a standard loader |
| `--network` (`-N`) | Share the host network namespace |
| `--share=DIR` | Make host directory `DIR` writable inside the container (`--expose` is the read-only variant) |

Create a small wrapper so you don't retype it:

```bash
cat > ~/bin/hermes-env <<'EOF'
#!/usr/bin/env bash
exec guix shell --container --emulate-fhs --network \
  --share="$HOME" \
  coreutils bash curl git nss-certs \
  -- bash --login
EOF
chmod +x ~/bin/hermes-env
```

Notes:

- `--share="$HOME"` is what makes the installation persistent — Hermes keeps
  everything under `~/.hermes/`, and without the share the container's home
  directory is wiped when the shell exits.
- `nss-certs` is required. Without it, TLS in `curl` fails with certificate
  verification errors — the classic first-roadblock in FHS containers.
- Host environment variables are **not** inherited; pass specific variables
  with `--preserve=REGEX` if a tool inside the container needs one.

## 2. Install Hermes Inside the Shell

Enter the environment and run the canonical installer — a fresh container has
no `hermes` binary yet, so start from the same command the
[installation guide](./installation.md) gives for Linux:

```bash
~/bin/hermes-env
# now inside the FHS container:
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

The installer clones the source under `~/.hermes/`, installs the pinned
runtimes, drops the `hermes` launcher into `~/.local/bin/` (adding that
directory to PATH in your shell rc files), and hands off to `hermes setup` to
finish configuration. Everything lands in the shared home directory, so a
fresh `~/bin/hermes-env` invocation finds the launcher on PATH through
`bash --login`. If an already-open shell doesn't see it yet, run
`exec bash --login` once.

From here the CLI behaves identically to any Linux install. For updates,
re-enter the environment and run `hermes update`.

:::note
Run installs, updates, and the gateway from the same wrapper. Each
`hermes-env` invocation recreates the container, but the state under
`~/.hermes/` (config, sessions, skills) survives in your real home directory.
The installed binaries are built for the container's FHS layout — run them
inside the wrapper, not from the host.
:::

## 3. Run the Gateway as a User Shepherd Service

Skip `hermes gateway install` — it writes systemd user units. On Guix, run
the gateway process directly and let shepherd supervise it.

Create the service file. If you manage services through `guix home`, place it
under your home's shepherd services directory; otherwise put it in
`~/.config/shepherd/services/`:

```scheme
;; hermes-gateway.scm
(use-modules (shepherd service))

(define hermes-gateway
  (make-forkexec-constructor
   (list "/home/YOURUSER/bin/hermes-env-run")
   #:log-file "/home/YOURUSER/.hermes/logs/gateway.log"))

(register-services hermes-gateway)
```

The service needs a second wrapper that runs one command instead of opening a
login shell:

```bash
cat > ~/bin/hermes-env-run <<'EOF'
#!/usr/bin/env bash
exec guix shell --container --emulate-fhs --network \
  --share="$HOME" \
  coreutils bash curl git nss-certs \
  -- hermes gateway run
EOF
chmod +x ~/bin/hermes-env-run
```

Two shepherd specifics that bite people:

- **The file must end with a `register-services` call.** A service file that
  defines a service but never registers it loads without errors and silently
  provides nothing.
- **User services die at logout.** Enable lingering for your user, or the
  gateway stops whenever you log out of the machine.

Load and check it:

```bash
herd start hermes-gateway
herd status hermes-gateway
```

For boot persistence, add the service file to your `guix home` configuration
(the standard shepherd-services home service), so the registration survives
`guix home reconfigure` cycles.

## Pitfalls

- **No network in the container.** Without `--network`, the container has no
  network access — API calls and installs all fail.
- **TLS errors on install.** Missing `nss-certs`. Add it to the package list.
- **`/tmp` is cleared on boot.** Guix systems wipe `/tmp` on every restart.
  Never stage work-in-progress there — use a directory under your home.
- **Foreground processes lose their environment.** Services must go through
  the wrapper script; a service unit that calls `hermes` directly won't find
  it, since it only exists inside the FHS shell.
- **`hermes update` inside the container only.** The binary lives in the
  container's profile; updating from outside the shell does nothing.

## Verification

```bash
~/bin/hermes-env
hermes --version        # inside the container
herd status hermes-gateway
```

Then message your agent from its configured messaging platform and confirm a
reply arrives.

## See Also

- [Nix & NixOS Setup](./nix-setup.md) — the Nix-native equivalent (flake,
  Home Manager, and NixOS modules), if you want declarative packaging rather
  than an FHS container
- [Docker](../user-guide/docker.md) — another supported container route
- [Platform Support](./platform-support.md) — where Guix sits relative to
  Tier 1
