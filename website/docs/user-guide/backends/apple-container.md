---
title: Apple Container
---

Apple Container runs Hermes terminal commands, file operations, and `execute_code`
inside a Linux VM on Apple Silicon. It requires macOS 26 or later and the separately
installed Apple `container` CLI. Start the runtime with `container system start`,
then select **Apple Container** in `hermes setup` or the Desktop terminal-backend picker.
Start a new session after changing backends so it creates the selected environment.

Configure the backend in `config.yaml`:

```yaml
terminal:
  backend: apple_container
  apple_container_image: python:3.11-slim-bookworm
  apple_container_volumes: []
  apple_container_extra_args: []
  container_cpu: 2
  container_memory: 2048
  container_persistent: true
```

An empty extra-argument list uses the runtime's default networking. To disable
networking inside the container, explicitly set
`apple_container_extra_args: ["--network", "none"]`. This prevents package downloads
and API requests from container commands; Hermes model calls and host-side tools
are separate. Existing containers retain their startup network settings, so start
a new session after changing this setting.

Desktop uploads attachment contents for this backend instead of passing host-only
file paths. Host project files require an explicit mount or upload; selecting a
host project directory does not automatically expose it inside the VM.

The image must include Bash and Python 3 for `execute_code`. File operations and
terminal commands share the same task environment. The prompt probe receives the
same image, resource settings, volumes, and extra arguments, then removes its
one-shot container. No host working directory is automatically mounted.

With persistence enabled, `/workspace` and `/root` use task storage under Hermes's
sandbox directory. Without persistence, they use temporary filesystems. The root
filesystem is read-only, with writable scratch mounts. Automatic skills and cache
mounts are read-only; configured credential files are copied into temporary
read-only directory mounts.

A user volume uses `HOST:CONTAINER[:ro]`, with absolute paths and directory sources:

```yaml
terminal:
  apple_container_volumes:
    - /Users/me/project:/workspace/project:ro
```

User volumes enable normal approval guards, including read-only volumes. Potential
mount arguments in `apple_container_extra_args` and SSH-agent forwarding also enable
guards. Detection is deliberately conservative: even a mount-looking token used as
another flag's value enables guards. This may require approval or block execution
under unattended deny policies. Raw arguments are operator-controlled and can
weaken the sandbox; this detection is not a complete security audit of arbitrary
runtime options.

With no user mounts or mount-like arguments, isolated `execute_code` can run under
unattended deny policy. Explicit command `approvals.deny` rules still apply.

On Apple Container CLI 1.x and newer, Hermes adds `--init` so the container's
init process forwards shutdown signals and reaps child processes. Older or
unrecognized CLI versions retain the previous startup flags. Normal cleanup
stops and deletes the task container; this does not recover VMs left behind
when the owning Hermes process is abruptly killed.
