---
title: Apple Container
---

Apple Container runs Hermes terminal commands, file operations, and `execute_code`
inside a Linux VM on Apple Silicon. It requires macOS 26 or later and the separately
installed Apple `container` CLI. Start the runtime with `container system start`,
then select **Apple Container** in `hermes setup`.

Configure the backend in `config.yaml`:

```yaml
terminal:
  backend: apple_container
  apple_container_image: python:3.11-slim-bookworm
  apple_container_volumes: []
  apple_container_extra_args: ["--network", "none"]
  container_cpu: 2
  container_memory: 2048
  container_persistent: true
```

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
