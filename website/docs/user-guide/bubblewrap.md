---
sidebar_position: 10
title: "Bubblewrap"
description: "Using bubblewrap (bwrap) as a terminal backend: a per-command sandbox on the host"
---

# Bubblewrap terminal backend

The `bubblewrap` backend runs every shell command inside its own
[bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`) sandbox on
the machine Hermes runs on. It is not a container: the sandbox sees the host
filesystem read-only at the host paths, can write to the working directory,
and cannot read your credentials. It needs no image, no daemon and no
network round trip, so it fits a personal machine or a small server where
the `local` backend is too open and Docker is too heavy.

Linux only. Requires bubblewrap 0.9.0 or later and unprivileged user
namespaces (or a setuid `bwrap`).

## Quick start

```sh
sudo apt install bubblewrap     # Debian, Ubuntu
sudo dnf install bubblewrap     # Fedora
sudo pacman -S bubblewrap       # Arch
```

```yaml
terminal:
  backend: bubblewrap
  bubblewrap_profile: network   # restricted | workspace | network
```

`hermes setup` offers the backend on Linux and asks for the profile.
`hermes doctor` reports whether `bwrap` is found, its version, and whether
the sandbox probe passes; `hermes status` shows the profile and the `bwrap`
path. If the probe fails, unprivileged user namespaces are disabled for
your user: check your distribution's notes on enabling them for bubblewrap
(on Ubuntu 24.04 and later this is the AppArmor
`kernel.apparmor_restrict_unprivileged_userns` restriction).

When `bwrap` is missing or the probe fails, the terminal tool returns a
degraded result that names the package (or an error under
`terminal.degraded_mode: fail`). Commands are never run outside the sandbox.

## What a command sees

- The host root, read-only, at the same paths as on the host. `/usr/bin`,
  `/etc`, your project checkouts and your installed toolchains are all there.
- A fresh `/dev`, a private `/proc` (the command's own pid namespace, so it
  cannot see or signal host processes) and a fresh `/tmp` per command.
  Nothing written to `/tmp` survives the command; use the working directory.
- An empty `/run/user/<uid>`: the gpg-agent, ssh-agent, keyring and D-Bus
  sockets that live there are not reachable, so a command cannot sign or
  decrypt with keys loaded on the host. The docker socket, if present, is
  replaced by an empty file.
- The working directory at its host path, writable in the `workspace` and
  `network` profiles, read-only in `restricted`. `cd` persists between
  commands, and variables exported in one command are visible in the next,
  exactly as with the `local` backend.
- The same environment the `local` backend builds: `env_passthrough`
  applies, provider API keys stay out, and `HOME` follows
  `terminal.home_mode`.

## Hidden paths

Secrets under your home directory are hidden inside every sandbox: a
directory shows as empty and a file shows as empty. The set is fixed:

```
~/.ssh  ~/.aws  ~/.gnupg  ~/.gpg  ~/.config/gcloud  ~/.azure  ~/.docker
~/.kube  ~/.npmrc  ~/.pypirc  ~/.netrc  ~/.env
```

`~/.hermes` (or whatever `HERMES_HOME` points at) is hidden as well: the
agent already holds its own configuration and keys in memory and does not
need to read them from inside a command. With `terminal.home_mode: profile`
the `HERMES_HOME/home` directory is the subprocess `HOME` and stays
readable and writable; the rest of `HERMES_HOME` stays hidden.

A path that does not exist on the host is simply skipped. Writes into a
hidden directory land in the sandbox's copy and never reach the host.

## Working directory

The working directory (`terminal.cwd`, the launch directory for the CLI,
`MESSAGING_CWD` or the home directory for the gateway) is the writable
set: everything under it can be changed, everything else on the host is
read-only. Point it at a project or scratch directory. With the home
directory as the working directory every dotfile outside the hidden set
(`~/.bashrc`, `~/.profile`, `~/.config/autostart`, `~/.local/bin`, ...) is
writable, which is a path back into your own shells; Hermes logs a warning
at startup in that case. A working directory of `/` is refused, since it
would make the whole root writable.

If the working directory is deleted on the host (for example by the
command's own `rm -rf`), later commands run in the nearest existing parent
directory, read-only, until it exists again.

## Profiles

| Profile | Working directory | Network | Use it for |
|---------|-------------------|---------|------------|
| `restricted` | read-only | none (loopback only) | Inspection and read-only analysis |
| `workspace` | writable | none (loopback only) | Builds and edits that must not reach the network |
| `network` | writable | host network | Everything else (the default) |

The rest of the filesystem is read-only in every profile.

## Extra binds

`terminal.bubblewrap_binds` mounts more host directories into the sandbox,
read-only unless `readonly: false`:

```yaml
terminal:
  bubblewrap_binds:
    - {src: /data/models, dest: /data/models}
    - {src: /srv/scratch, dest: /srv/scratch, readonly: false}
```

`dest` defaults to `src`. A source that lies under a hidden path (for
example `~/.ssh/config`) is ignored with a warning. Because the root is
read-only, a `dest` must already exist on the host or sit under a writable
mount.

## Resource limits

Every command gets process limits from three keys. A value of `0` disables
that limit.

| Key | Default | Limit |
|-----|---------|-------|
| `bubblewrap_memory_mb` | `256` | Virtual memory per process (`RLIMIT_AS`) |
| `bubblewrap_cpu_seconds` | `30` | CPU time per process (`RLIMIT_CPU`) |
| `bubblewrap_max_procs` | `256` | Processes the command may add (`RLIMIT_NPROC`) |

The defaults are deliberately tight and some everyday tools exceed them:

- `pip` resolving wheels, `node`, `cargo` and most compilers need more than
  256 MB of address space. The command fails with `MemoryError` or a
  similar out-of-memory message. Raise `bubblewrap_memory_mb` (1024 or 2048
  is usually enough) or set it to `0`.
- A long compile or test run uses more than 30 seconds of CPU before the
  180 second `terminal.timeout` is reached. The process is killed by the CPU
  limit and the output ends with `Killed` (exit code 137). Raise
  `bubblewrap_cpu_seconds` for such work.
- `bubblewrap_max_procs` is applied on top of the number of threads your
  user already runs on the host, because the kernel counts `RLIMIT_NPROC`
  per user across the whole machine. It bounds what a command can add, so a
  fork bomb stops at about 256 processes without touching your desktop.

```yaml
terminal:
  backend: bubblewrap
  bubblewrap_memory_mb: 2048
  bubblewrap_cpu_seconds: 600
  bubblewrap_max_procs: 256
```

## Approval, file tools and background jobs

Because the sandbox writes to real host paths, the dangerous-command
approval flow applies exactly as for the `local` backend, including the
hardline floor. File tools (`read_file`, `write_file`, `search_files`)
work on host paths directly. Background jobs (`background: true`) are
refused: each command is its own sandbox that ends with the command, so a
detached process could not outlive it.

## Limitations

- Linux only, with unprivileged user namespaces or a setuid `bwrap`.
- No seccomp filter: system calls are not filtered. The sandbox is a
  filesystem and process boundary, not a defense against kernel exploits.
- No cgroup limits: the limits above are per-process rlimits. A command
  that forks can use more memory in total than `bubblewrap_memory_mb`.
- Network is all or nothing: the `network` profile shares the host network
  with no egress filtering, and the other two have only loopback.
- The host filesystem is visible: anything your user can read outside the
  hidden set (`/etc`, other dotfiles, secrets kept inside project
  directories) is readable by a command. Add to `bubblewrap_binds` only
  what you want the agent to see, and keep secrets in the hidden paths.
- Unix sockets outside `/run/user/<uid>`, `/tmp` and the docker socket
  stay connectable (a read-only mount does not block `connect()`), and the
  agent environment variables that name them (`SSH_AUTH_SOCK`,
  `GPG_AGENT_INFO`, `DBUS_SESSION_BUS_ADDRESS`) are passed through as for
  `local`, although they point at masked paths on a standard desktop.
- One sandbox per command: processes, mounts and `/tmp` do not carry over
  between commands. Only the working directory and the shell state
  (`cd`, exported variables) persist.

Tested with bubblewrap 0.9.0 on kernel 6.8.
