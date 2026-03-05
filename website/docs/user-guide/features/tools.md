---
sidebar_position: 1
title: "Tools & Toolsets"
description: "Overview of Hermes Agent's tools — what's available, how toolsets work, and terminal backends"
---

# Tools & Toolsets

Tools are functions that extend the agent's capabilities. They're organized into logical **toolsets** that can be enabled or disabled per platform.

## Available Tools

| Category | Tools | Description |
|----------|-------|-------------|
| **Web** | `web_search`, `web_extract`, `web_crawl` | Search the web, extract page content, crawl sites |
| **Terminal** | `terminal` | Execute commands (local/docker/singularity/modal/ssh backends) |
| **File** | `read_file`, `write_file`, `patch`, `search` | Read, write, edit, and search files |
| **Browser** | `browser_navigate`, `browser_click`, `browser_type`, etc. | Full browser automation via Browserbase |
| **Vision** | `vision_analyze` | Image analysis via multimodal models |
| **Image Gen** | `image_generate` | Generate images (FLUX via FAL) |
| **TTS** | `text_to_speech` | Text-to-speech (Edge TTS / ElevenLabs / OpenAI) |
| **Reasoning** | `mixture_of_agents` | Multi-model reasoning |
| **Skills** | `skills_list`, `skill_view`, `skill_manage` | Find, view, create, and manage skills |
| **Todo** | `todo` | Read/write task list for multi-step planning |
| **Memory** | `memory` | Persistent notes + user profile across sessions |
| **Session Search** | `session_search` | Search + summarize past conversations (FTS5) |
| **Cronjob** | `schedule_cronjob`, `list_cronjobs`, `remove_cronjob` | Scheduled task management |
| **Code Execution** | `execute_code` | Run Python scripts that call tools via RPC sandbox |
| **Delegation** | `delegate_task` | Spawn subagents with isolated context |
| **Clarify** | `clarify` | Ask the user multiple-choice or open-ended questions (CLI-only) |
| **MCP** | Auto-discovered | External tools from MCP servers |

## Using Toolsets

```bash
# Use specific toolsets
hermes --toolsets "web,terminal"

# See all available tools
hermes tools

# Configure tools per platform (interactive)
hermes tools
```

**Available toolsets:** `web`, `terminal`, `file`, `browser`, `vision`, `image_gen`, `moa`, `skills`, `tts`, `todo`, `memory`, `session_search`, `cronjob`, `code_execution`, `delegation`, `clarify`, and more.

## Terminal Backends

The terminal tool can execute commands in different environments:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `local` | Run on your machine (default) | Development, trusted tasks |
| `docker` | Isolated containers | Security, reproducibility |
| `ssh` | Remote server | Sandboxing, keep agent away from its own code |
| `singularity` | HPC containers | Cluster computing, rootless |
| `modal` | Cloud execution | Serverless, scale |

### Configuration

```yaml
# In ~/.hermes/config.yaml
terminal:
  backend: local    # or: docker, ssh, singularity, modal
  cwd: "."          # Working directory
  timeout: 180      # Command timeout in seconds
```

### Docker Backend

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

### SSH Backend

Recommended for security — agent can't modify its own code:

```yaml
terminal:
  backend: ssh
```
```bash
# Set credentials in ~/.hermes/.env
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

### Singularity/Apptainer

```bash
# Pre-build SIF for parallel workers
apptainer build ~/python.sif docker://python:3.11-slim

# Configure
hermes config set terminal.backend singularity
hermes config set terminal.singularity_image ~/python.sif
```

### Modal (Serverless Cloud)

```bash
uv pip install "swe-rex[modal]"
modal setup
hermes config set terminal.backend modal
```

### Container Resources

Configure CPU, memory, disk, and persistence for all container backends:

```yaml
terminal:
  backend: docker  # or singularity, modal
  container_cpu: 1              # CPU cores (default: 1)
  container_memory: 5120        # Memory in MB (default: 5GB)
  container_disk: 51200         # Disk in MB (default: 50GB)
  container_persistent: true    # Persist filesystem across sessions (default: true)
```

When `container_persistent: true`, installed packages, files, and config survive across sessions.

### Container Security

All container backends run with security hardening:

- Read-only root filesystem (Docker)
- All Linux capabilities dropped
- No privilege escalation
- PID limits (256 processes)
- Full namespace isolation
- Persistent workspace via volumes, not writable root layer

## Background Process Management

Start background processes and manage them:

```python
terminal(command="pytest -v tests/", background=true)
# Returns: {"session_id": "proc_abc123", "pid": 12345}

# Then manage with the process tool:
process(action="list")       # Show all running processes
process(action="poll", session_id="proc_abc123")   # Check status
process(action="wait", session_id="proc_abc123")   # Block until done
process(action="log", session_id="proc_abc123")    # Full output
process(action="kill", session_id="proc_abc123")   # Terminate
process(action="write", session_id="proc_abc123", data="y")  # Send input
```

PTY mode (`pty=true`) enables interactive CLI tools like Codex and Claude Code.

## Sudo Support

If a command needs sudo, you'll be prompted for your password (cached for the session). Or set `SUDO_PASSWORD` in `~/.hermes/.env`.

:::warning
On messaging platforms, if sudo fails, the output includes a tip to add `SUDO_PASSWORD` to `~/.hermes/.env`.
:::
