# Omarchy MCP Server — Specification Document

> **Version:** 1.0.0
> **Date:** 2026-09-15
> **Author:** Joerg Peetz (Fluffy / Hermes Agent)
> **Repository:** https://github.com/JPeetz/omarchy-mcp

---

## 1. Overview

The Omarchy MCP Server is a purpose-built Model Context Protocol (MCP) server that runs on an Omarchy (Arch Linux) VM and exposes local AI agents and system tools as MCP tools consumable by Hermes Agent (or any MCP client).

It replaces an SSH-based dispatch pipeline with a structured, authenticated MCP protocol — removing plaintext credentials from skill files, eliminating fragile heredoc quoting, and replacing ANSI-scraping with structured JSON-RPC results.

### 1.1 Problem Statement

Before this server, dispatching tasks from Hermes Agent (Mac) to the Omarchy VM required:

1. Writing prompts to local files
2. SCP-ing files to the VM
3. SSH-ing to execute `claude -p` or `codex exec`
4. SCP-ing results back
5. Stripping ANSI escape codes with a 3-pass regex
6. Manual timeout tuning per prompt size

This pipeline had real costs: a **plaintext password embedded in skill files** (`sshpass -p '…'` in ~15+ locations), **fragile heredoc quoting** that broke on complex prompts, **silent failures** on SSH timeout (empty output files), and **no streaming** for long-running tasks.

### 1.2 Solution

An MCP server running as a systemd service on the VM, exposing seven tools over Streamable HTTP transport with Bearer token authentication. Hermes Agent connects as a remote `url:`-type MCP server — the same pattern Hermes-HQ already uses, but in the opposite direction.

---

## 2. Architecture

```
┌─────────────────────────────────────┐
│  Machine A (Mac, 192.168.212.135)  │
│                                     │
│  Hermes Agent ─────────────────┐    │
│  config.yaml:                  │    │
│    mcp_servers:                │    │
│      omarchy:                  │    │
│        url: "http://...:8911"  │    │
│        headers:                │    │
│          Authorization: Bearer │    │
│                                │    │
│  Hermes-HQ MCP (port 8910)    │    │
│  (memory server — unchanged)  │    │
└─────────────────────┬───────────┘    │
                      │                │
           Streamable HTTP             │
           (LAN, 192.168.212.x)        │
                      │                │
┌─────────────────────▼───────────┐    │
│  Machine B (Omarchy VM,         │    │
│  192.168.212.127)               │    │
│                                 │    │
│  omarchy-mcp (systemd)          │    │
│  FastMCP + uvicorn              │    │
│  Port 8911                      │    │
│                                 │    │
│  Tools exposed:                 │    │
│  ├─ claude_execute()            │    │
│  ├─ codex_execute()             │    │
│  ├─ file_read()                 │    │
│  ├─ file_write()                │    │
│  ├─ file_list()                 │    │
│  ├─ system_run()                │    │
│  └─ status()                    │    │
│                                 │    │
│  Auth: Bearer token (env var)   │    │
└─────────────────────────────────┘
```

### 2.1 Data Flow

1. Hermes Agent calls `claude_execute(prompt="...")` via its MCP client
2. Hermes sends JSON-RPC request via Streamable HTTP to `http://192.168.212.127:8911/mcp`
3. Server validates Bearer token
4. Server spawns `claude --dangerously-skip-permissions -` as subprocess with prompt via stdin
5. Output streams back, ANSI is stripped, structured result returned
6. Hermes receives result as tool response

---

## 3. Tool Specifications

### 3.1 `claude_execute`

Execute a prompt with Claude Code CLI (YOLO mode).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | yes | — | Full prompt sent to Claude Code via stdin |
| `cwd` | string | no | `/home/jpeetz` | Working directory |
| `timeout` | integer | no | 300 | Max seconds (30–1800) |

**Returns:** Claude Code's output (ANSI-stripped), plus exit code and duration on error.

**Implementation:** Pipes prompt to `claude --dangerously-skip-permissions -` stdin. Timeout uses SIGTERM then SIGKILL after 5s grace period.

### 3.2 `codex_execute`

Execute a prompt with Codex CLI.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | yes | — | Full prompt sent to Codex via stdin |
| `cwd` | string | no | `/home/jpeetz` | Working directory |
| `timeout` | integer | no | 120 | Max seconds (30–600) |

**Implementation:** Pipes prompt to `codex exec --skip-git-repo-check -` stdin.

### 3.3 `file_read`

Read file content from the Omarchy filesystem.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | — | Absolute path (must be under `/home/jpeetz` or `/tmp`) |
| `offset` | integer | no | 0 | Starting line (0-indexed) |
| `limit` | integer | no | 2000 | Maximum lines to return |

**Returns:** JSON with `content`, `total_lines`, `offset`, `returned_lines`, `truncated` fields.

### 3.4 `file_write`

Write content to a file.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | — | Absolute path (must be under `/home/jpeetz` or `/tmp`) |
| `content` | string | yes | — | Content to write |
| `mode` | string | no | `overwrite` | `overwrite` or `append` |

### 3.5 `file_list`

List directory contents.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | no | `/home/jpeetz` | Directory to list |

**Returns:** JSON array of `{name, type, size, mtime}` entries.

### 3.6 `system_run`

Execute a whitelisted shell command.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `command` | string | yes | — | Shell command |
| `cwd` | string | no | `/home/jpeetz` | Working directory |
| `timeout` | integer | no | 60 | Max seconds (5–300) |

**Whitelist:** `git`, `python3`, `ls`, `cat`, `head`, `tail`, `grep`, `which`, `whoami`, `hostname`, `uname`, `date`, `pwd`, `echo`, `wc`, `find`, `du`, `df`, `ps`, `uptime`, `free`.

### 3.7 `status`

Get VM health and tool availability.

**Returns:** JSON with `hostname`, `platform`, `python_version`, `uptime_s`, `load_avg`, `memory_total_mb`, `memory_available_mb`, `claude_available`, `codex_available`, `version`.

---

## 4. Security Model

### 4.1 Authentication

- **Mechanism:** Static Bearer token
- **Source:** `OMARCHY_MCP_TOKEN` environment variable (loaded from `.env` file)
- **Validation:** Constant-time comparison on every request
- **Storage:** Token stored in Hermes Agent's `config.yaml` (set via `hermes config set`) and VM's `.env` file
- **Never in skill files or session prompts**

### 4.2 Transport Security

- HTTP only (same as Hermes-HQ on this LAN)
- Server binds to LAN interface only (`0.0.0.0:8911` but only accessible from `192.168.212.0/24`)
- UFW firewall on VM restricts access to specific ports (8911 opened explicitly)

### 4.3 Authorization

- `system_run`: Whitelist-based — only pre-approved commands (git, python3, file utilities). All invocations logged.
- `claude_execute`/`codex_execute`: Prompt-based — the prompt is content, not a command. Logged.
- `file_read`/`file_write`: Path validation prevents access outside `/home/jpeetz` and `/tmp`. Logged.
- MCP SDK transport security disabled for LAN (DNS rebinding protection not needed internally)

### 4.4 Audit Logging

Every tool call is logged to journald via systemd with: timestamp, tool name, duration, and exit status.

---

## 5. Deployment

### 5.1 Directory Layout

```
/home/jpeetz/Work/omarchy-mcp/
├── server.py          — FastMCP v2 server (MCPServer), auth middleware, uvicorn runner
├── auth.py            — Bearer token verification
├── executor.py        — Shared subprocess runner (ANSI stripping, timeout handling)
├── tools/
│   ├── __init__.py    — Tool registration
│   ├── claude.py      — claude_execute tool
│   ├── codex.py       — codex_execute tool
│   ├── files.py       — file_read, file_write, file_list tools
│   ├── system.py      — system_run tool (whitelisted)
│   └── status.py      — status tool (VM health)
├── .env               — OMARCHY_MCP_TOKEN, PORT, BIND
└── requirements.txt   — Dependencies
```

### 5.2 Dependencies

- Python 3.11+ with MCP SDK 2.x (`mcp>=2.0.0`)
- `psutil` for system monitoring
- `python-dotenv` for `.env` loading (optional)
- `uvicorn` for HTTP server (bundled with MCP SDK)
- `starlette` for ASGI middleware (bundled with MCP SDK)

### 5.3 systemd Service

```ini
[Unit]
Description=Omarchy MCP Server
After=network.target

[Service]
Type=simple
User=jpeetz
WorkingDirectory=/home/jpeetz/Work/omarchy-mcp
EnvironmentFile=/home/jpeetz/Work/omarchy-mcp/.env
ExecStart=/usr/bin/python3 /home/jpeetz/Work/omarchy-mcp/server.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 5.4 Hermes Agent Integration

Add to `~/.hermes/config.yaml` on the Mac:

```yaml
mcp_servers:
  omarchy:
    url: "http://192.168.212.127:8911/mcp"
    headers:
      Authorization: "Bearer <token>"
```

---

## 6. Testing & Verification

### 6.1 Local (on Omarchy)

```bash
# Initialize MCP session
curl -s -X POST http://127.0.0.1:8911/mcp \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}'

# List tools (use session ID from init response header)
curl -s -X POST http://127.0.0.1:8911/mcp \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -H 'mcp-session-id: <sid>' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

# Call status tool
curl -s -X POST http://127.0.0.1:8911/mcp \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -H 'mcp-session-id: <sid>' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"status","arguments":{}}}'
```

### 6.2 From Hermes Agent (Mac)

```bash
# Test connection and tool discovery
hermes mcp test omarchy

# Call tools via chat
hermes -c "Call the omarchy status tool and show me the result"
```

### 6.3 Verification Matrix

| Test | Expected Result |
|------|----------------|
| No auth token | 401 "missing Bearer token" |
| Bad auth token | 401 "invalid token" |
| Valid init + tools/list | 7 tools discovered |
| status() call | VM health JSON |
| file_read() | File content with metadata |
| file_write() | File created/updated |
| system_run() (whitelisted) | Command output |
| system_run() (non-whitelisted) | Error: not in whitelist |

---

## 7. Comparison: SSH vs MCP

| Aspect | SSH Pipeline (before) | MCP Pipeline (after) |
|--------|----------------------|---------------------|
| Credentials | Plaintext in skill files | Bearer token in `.env` + `config.yaml` |
| Transport | New TCP per call + SCP | Single persistent HTTP connection |
| Output | Full-buffer redirect, ANSI scraping | Structured JSON, ANSI stripped server-side |
| Long tasks | `--bg` + polling `claude logs` | Native Streamable HTTP |
| Prompt delivery | Heredoc (fragile) → SCP file | Tool argument (structured JSON) |
| Failure mode | Silent empty file on timeout | Explicit error response |
| New surface | None (reuses SSH) | New network listener (secured) |
| Implementation | Already built, fragile | ~1 day to build, clean |

---

## 8. Future Work

- TLS with self-signed certificate + pinned fingerprint
- OAuth 2.1 with PKCE (when MCP spec stabilizes)
- More granular tool-level authorization (per-user tokens)
- Resource endpoints (expose artifacts as MCP resources)
- Prompt templates (common research/development prompts)
- Dashboard UI for monitoring dispatch status
- `claude_execute` streaming (real-time output during long runs)

---

## 9. File Manifest

| File | Lines | Purpose |
|------|-------|---------|
| `server.py` | 48 | Main entry point, MCPServer, auth middleware, uvicorn |
| `auth.py` | 17 | Bearer token verification (constant-time) |
| `executor.py` | 76 | Subprocess runner, ANSI stripping, timeout handling |
| `tools/__init__.py` | 4 | Tool registration list |
| `tools/claude.py` | 47 | Claude Code executor |
| `tools/codex.py` | 47 | Codex CLI executor |
| `tools/files.py` | 98 | File read/write/list operations |
| `tools/system.py` | 39 | Whitelisted shell command execution |
| `tools/status.py` | 30 | VM health and tool availability |
| `.env` | 3 | Token, port, bind configuration |
| `requirements.txt` | 3 | Python dependencies |

---

## 10. License

MIT — same as Hermes Agent.
