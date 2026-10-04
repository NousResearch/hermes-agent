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
and cannot read the dot entries of your home directory except the ones on
an allowlist. It needs no image, no daemon and no
network round trip, so it fits a personal machine or a small server where
the `local` backend is too open and Docker is too heavy.

Linux only. Requires bubblewrap 0.9.0 or later, `prlimit` from util-linux
(installed on every mainstream distribution) and unprivileged user
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
`hermes doctor` reports whether `bwrap` is found, its version, whether
the sandbox probe passes and whether the process limit is available on
this kernel; `hermes status` shows the profile and the `bwrap` path. If the probe fails, unprivileged user namespaces are disabled for
your user: check your distribution's notes on enabling them for bubblewrap
(on Ubuntu 24.04 and later this is the AppArmor
`kernel.apparmor_restrict_unprivileged_userns` restriction).

When `bwrap` is missing or the probe fails, the terminal tool returns a
degraded result that names the package (or an error under
`terminal.degraded_mode: fail`). Commands are never run outside the sandbox.

## What a command sees

- The host root, read-only, at the same paths as on the host. `/usr/bin`,
  `/etc`, your project checkouts and your installed toolchains are all there.
- Your home directory with its dot entries hidden unless they are on an
  allowlist (see [Your home directory](#your-home-directory)).
- A fresh `/dev`, a private `/proc` (the command's own pid namespace, so it
  cannot see or signal host processes) and a fresh `/tmp` per command. <!-- no-tmp: ok — describes the sandbox boundary -->
  Nothing written to `/tmp` survives the command; use the working directory or `$TMPDIR`. <!-- no-tmp: ok — explains why /tmp is wrong here -->
- An empty `/run/user/<uid>`: the gpg-agent, ssh-agent, keyring and D-Bus
  sockets that live there are not reachable, so a command cannot sign or
  decrypt with keys loaded on the host. The docker socket, if present, is
  replaced by an empty file.
- The working directory at its host path, writable in the `workspace` and
  `network` profiles, read-only in `restricted`. `cd` persists between
  commands, and variables exported in one command are visible in the next,
  exactly as with the `local` backend.
- The same environment the `local` backend builds, minus the three socket
  variables listed under Limitations: `env_passthrough` applies, provider
  API keys stay out, and `HOME` follows `terminal.home_mode`.

## Your home directory

Credentials live in dot entries of the home directory, and no list of
them is ever complete: every tool that stores a token picks its own
name. So the sandbox does not hide a list. It hides every dot entry of
your home directory and shows only what is on an allowlist.

- **Non-dot entries** (`~/projects`, `~/Documents`, `~/bin`) are visible,
  read-only unless the working directory or a read-write bind covers them.
- **Dot entries** (`~/.pgpass`, `~/.mozilla`, `~/.zz-some-tool`) do not
  exist inside the sandbox unless they are allowed. An allowed dot entry
  is read-only, always: a command cannot edit `~/.bashrc` or
  `~/.gitconfig`, which would run code in your own shells later. (For
  an entry that is a symlink, see
  [Dot entries that are symlinks](#dot-entries-that-are-symlinks).)
- **`~/.config`, `~/.local` and `~/.local/share`** follow the same rule
  one level down: only an allowed child is visible. They hold one
  directory per application, and many of those keep a login session.
- **Nothing new can be made** at the top of the home directory or of
  those three directories. `mkdir ~/.ssh` fails with "Read-only file
  system" whether or not `~/.ssh` exists on the host.

The allowlist has three sources:

1. A shipped list: the shell startup files (`.bashrc`, `.profile`,
   `.zshrc`, ...), `.gitconfig`, `.editorconfig`, `.tool-versions`,
   `.terminfo`, `.cache`, the common toolchain directories (`.cargo`,
   `.rustup`, `.nvm`, `.bun`, `.deno`, `.gem`, `.npm`, `.pyenv`, `.rbenv`,
   `.sdkman`, `.volta`, `.asdf`, `.m2`, `.gradle`, `.dotnet`, `.pub-cache`,
   `.conda`, `.nix-profile`), `.config/{git,pip,uv,npm,pnpm,yarn,go,fontconfig}`,
   `.local/{bin,lib,include}` and
   `.local/share/{uv,pipx,pnpm,virtualenvs,man,bash-completion,fonts,mime}`.
2. Every directory on `PATH` that lies under the home directory, as
   `PATH` stands in the Hermes process when the backend starts. The
   smallest unit is allowed: `~/.zz-tool/bin` on `PATH` allows `~/.zz-tool`,
   `~/.local/share/zz/bin` allows `~/.local/share/zz` and not its
   neighbours.
3. `terminal.bubblewrap_home_allow`, for anything else a command needs:

```yaml
terminal:
  bubblewrap_home_allow:
    - .zz-tool           # a dot entry at the top of the home directory
    - .config/nvim       # or one child of .config, .local or .local/share
```

A tool that reads `~/.config/<name>` and is not on the list finds no
configuration there until you add `.config/<name>`.

Credential stores stay hidden whatever the allowlist says. `~/.ssh`,
`~/.aws`, `~/.gnupg`, `~/.kube`, `~/.docker`, `~/.netrc`, `~/.npmrc`,
`~/.pypirc`, `~/.pgpass`, `~/.git-credentials`, `~/.config/gh`,
`~/.config/gcloud`, the browser profiles, the desktop keyrings and the
other paths Hermes' own file tools refuse are never shown, and a
`bubblewrap_home_allow` entry that names one is ignored with a warning.
The same applies below an allowed entry: `~/.cargo` is visible,
`~/.cargo/credentials.toml` shows as an empty file.

To hide something more, such as a non-dot directory that holds keys:

```yaml
terminal:
  bubblewrap_hide:
    - ~/Documents/keys
```

A symlink at the top of the home directory is shown as a symlink when
its entry is visible. A link that points into a hidden directory leads
nowhere inside the sandbox. A hidden entry that is itself a symlink (a
dotfiles repository that links `~/.ssh` to `~/dotfiles/ssh`) is hidden at
its target, resolved when the backend starts.

### Dot entries that are symlinks

A dot entry that is a symlink cannot be removed or replaced from inside
the sandbox: the top of the home directory is read-only. That does not
protect what the link leads to. If `~/.bashrc` is a link into
`~/dotfiles`, and the working directory (or a read-write bind) makes
`~/dotfiles` writable, a command could change the file your next login
shell reads, long after the sandbox is gone. The same holds for any dot
entry your shells or desktop read at startup.

So the backend does not start in that case. When it starts, it follows
each dot symlink at the top of the home directory, one step at a time,
and refuses with an error that names the entry when any link, directory
or final target on the way lies in the working directory, in a
read-write bind or in the profile home (`HERMES_HOME/home` under
`terminal.home_mode: profile`). A target that does not exist yet counts too, when a
command could create it. The error is a configuration error, like an
unusable bind: fix it by using a project directory as the working
directory instead of the home directory or the directory that holds
your dotfiles, or by making the bind read-only. There is no option that
turns the check off.

A link is accepted when nothing on its way is writable inside the
sandbox: the target is outside the working directory and every
read-write bind, or under a hidden path, or (with the home directory as
the working directory) inside a dot entry of the home directory. A plain
dot entry (a `~/.bashrc` that is a file) is read-only in every case. The
check covers the dot entries at the top of the home directory, not links
deeper down such as `~/.config/fish`.

The allowlist and the hidden set are fixed when the backend starts, so a
command cannot widen them by changing `PATH`. The listing of the home
directory is read at every command, so a directory you create on the host
shows up in the next command.

### The Hermes home

`~/.hermes` (or whatever `HERMES_HOME` points at) is hidden: the agent
already holds its own configuration and keys in memory and does not need
to read them from inside a command. When `HERMES_HOME` points elsewhere (a
profile at `~/.hermes/profiles/<name>`, or any other directory), the
default `~/.hermes` is hidden as well, so the default home's `.env` and
`auth.json` are not readable from that profile's sandbox.

Four things under `HERMES_HOME` stay reachable, because a command needs
them:

- the sandbox's own state directory;
- a scratch directory. On the host, `HERMES_HOME/cache/scratch` is the
  temp directory of every Hermes process, so the sandbox does not get
  it. Each sandbox environment has a directory of its own inside it
  (`HERMES_HOME/cache/scratch/hermes-bwrap-<id>`), at the same path on
  the host and in the sandbox, and `TMPDIR`, `TMP` and `TEMP` of a
  command point at it. It is writable (read-only in the `restricted`
  profile), a file written there is still there for the next command,
  and it is removed with the environment. The rest of the scratch path
  is empty and read-only: nothing that another session or another
  Hermes component keeps there is visible, whenever it is created, and
  a write to `HERMES_HOME/cache/scratch/<name>` fails with a read-only
  error. Use `$TMPDIR`. A `TMPDIR` you set yourself to another place is
  left as it is;
- the staged data directories (attachments, cached documents, images,
  audio, video, screenshots, pasted text and the other entries Hermes
  hands the model as file paths), read-only. The archive of oversized
  tool results (`cache/spillover`) is not one of them: it holds the tool
  output of every session of the profile. A command sees only the
  archive files that were handed to its own environment;
- `HERMES_HOME/home` under `terminal.home_mode: profile`, where it is the
  subprocess `HOME`, readable and writable.

## Working directory

The working directory (`terminal.cwd`, the launch directory for the CLI,
`MESSAGING_CWD` or the home directory for the gateway) is the writable
set: everything under it can be changed, everything else on the host is
read-only. Point it at a project or scratch directory.

With the home directory as the working directory, the existing non-dot
entries of it are writable and nothing else is: the dot entries are
read-only or hidden (a dot entry that is a symlink into a writable
directory stops the backend, see
[Dot entries that are symlinks](#dot-entries-that-are-symlinks)), and no
new file or directory can be made at the top of the home directory. Create it on the host first, or work in a
subdirectory. A file at the top of the home directory can be written in
place but not replaced: a program that saves by writing a new file and
renaming it over the old one fails there. Hermes logs a warning at
startup when the working directory covers the home directory.

A working directory of `/` is refused, since it would make the whole root
writable. A working directory at or under `HERMES_HOME` (`~/.hermes`) or
under a hidden path is refused as well: the hidden paths are covered
inside every sandbox, so no command could run there. Under
`terminal.home_mode: profile` a directory under a real `HERMES_HOME/home`
directory is the exception, since that directory is bound back into the
sandbox. A checkout under `~/.hermes` (for example
`~/.hermes/hermes-agent`) has to be launched from elsewhere or moved.

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
example `~/.ssh/config`) is ignored with a warning. So is a source that
contains a hidden path (for example your home directory) when `dest`
differs from `src`: the hidden paths are covered only at their own
location, so a copy of the tree elsewhere would show them. Bind such a
source at its own path instead. Because the root is read-only, a `dest`
must already exist on the host or sit under a writable mount.

An allowed dot entry is read-only, so a tool that writes its cache there
fails with "Read-only file system". Give it the one directory it writes
to, at its own path:

```yaml
terminal:
  bubblewrap_binds:
    - {src: ~/.cache/pip, dest: ~/.cache/pip, readonly: false}
    - {src: ~/.cargo/registry, dest: ~/.cargo/registry, readonly: false}
```

Bind the cache directory itself, not its parent. A read-write bind (or a
writable working directory) that contains a credential path which does
not exist on the host is refused when the backend starts: the backend
hides a credential path by mounting over it, a path that is not there
cannot be mounted over, and a command could then create it. A read-write
bind of all of `~/.cache` is refused while `~/.cache/huggingface/token`
does not exist, and one of all of `~/.config` while `~/.config/gh` does
not. The error names the path and the bind.

A read-write bind whose source lies inside the working directory, inside
another read-write bind or inside the profile home is refused unless it
has dest equal to src and sits directly under that directory, with no
symlink on the way: bwrap resolves a bind source on the host at every
command, so a source a command can rename or replace with a symlink would
let it choose what the next command mounts. Bound at its own path directly
under a writable directory, the source is a mount point inside the sandbox,
which no command can rename or move from any path. Read-only binds are not
affected.

The profile home is a bind too. Under `terminal.home_mode: profile`,
`HERMES_HOME/home` must be a plain directory or a link to a directory
outside the hidden set: a link into the home tree (for example to `~` or
`~/.config`), into `HERMES_HOME` itself (another profile under
`~/.hermes/profiles` included) or into a hidden dotfile would show the
hidden paths again through the bind, and the backend refuses to start.

## Resource limits

Every command gets process limits from three keys. A value of `0` disables
that limit.

| Key | Default | Limit |
|-----|---------|-------|
| `bubblewrap_memory_mb` | `256` | Virtual memory per process (`RLIMIT_AS`) |
| `bubblewrap_cpu_seconds` | `30` | CPU time per process (`RLIMIT_CPU`) |
| `bubblewrap_max_procs` | `256` | Processes and threads one command may run (`RLIMIT_NPROC`, counted for that sandbox alone) |

The limits are set by `prlimit` from util-linux. The memory and CPU
limits are set in front of `bwrap`, so they cover the sandbox and every
process inside it. The process limit is set inside the sandbox, where the
kernel counts only the processes of that sandbox. `hermes doctor` reports
a missing `prlimit` the same way as a missing `bwrap`.

The defaults are deliberately tight and some everyday tools exceed them:

- `pip` resolving wheels, `node`, `cargo` and most compilers need more than
  256 MB of address space. The command fails with `MemoryError` or a
  similar out-of-memory message. Raise `bubblewrap_memory_mb` (1024 or 2048
  is usually enough) or set it to `0`.
- A long compile or test run uses more than 30 seconds of CPU before the
  180 second `terminal.timeout` is reached. The process is killed by the CPU
  limit and the output ends with `Killed` (exit code 137). Raise
  `bubblewrap_cpu_seconds` for such work.
- `bubblewrap_max_procs` is a ceiling for one command: a fork bomb stops
  at 256 processes whatever else your user runs on the host, and a
  command cannot raise the limit from inside. This needs a kernel that
  counts processes per user namespace, which Linux does from 5.14 on. On
  an older kernel the backend sets no process limit, logs one warning at
  startup and keeps the memory and CPU limits; `hermes doctor` reports
  "bwrap process limit is not available on this kernel".

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
- No cgroup limits: the memory and CPU limits above are per-process
  rlimits. A command that forks can use more memory in total than
  `bubblewrap_memory_mb`, up to that value times `bubblewrap_max_procs`.
- Network is all or nothing: the `network` profile shares the host network
  with no egress filtering, and the other two have only loopback.
- The host filesystem outside the dot entries of your home directory is
  visible: `/etc`, the non-dot entries of your home directory and secrets
  kept inside project directories are readable by a command. Keep secrets
  in dot entries, or name them in `bubblewrap_hide`. A dotfiles directory
  with a non-dot name (`~/dotfiles`) is visible as a whole.
- An archived tool result is readable by a command only in the
  environment that it was handed to. One stored before the first
  terminal command of a session, or handed to an environment that has
  since been cleaned up after the idle time, is readable with
  `read_file` and not with a command.
- The staged data directories are shared by the sessions of a profile:
  a command can read an attachment or a cached document that another
  session of the same profile staged. Use separate profiles to keep
  sessions apart.
- The `PATH` rule reads the `PATH` of the Hermes process. A toolchain that
  only your shell startup files put on `PATH`, in a dot directory the
  shipped list does not name, needs a `bubblewrap_home_allow` entry.
- Unix sockets outside `/run/user/<uid>`, `/tmp` and the docker socket <!-- no-tmp: ok — names the paths the sandbox replaces -->
  stay connectable (a read-only mount does not block `connect()`). The
  agent environment variables that name them (`SSH_AUTH_SOCK`,
  `GPG_AGENT_INFO`, `DBUS_SESSION_BUS_ADDRESS`) are removed from the
  sandbox environment, so a command has to know a socket path to reach it.
- One sandbox per command: processes, mounts and `/tmp` do not carry over <!-- no-tmp: ok — explains why /tmp does not persist -->
  between commands. Only the working directory and the shell state
  (`cd`, exported variables) persist.

Tested with bubblewrap 0.9.0 on kernel 6.8.
