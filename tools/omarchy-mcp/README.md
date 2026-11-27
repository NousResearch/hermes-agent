# Omarchy MCP Server

A purpose-built [Model Context Protocol](https://modelcontextprotocol.io/) server that runs on a Linux VM and exposes local AI agents and system tools as MCP tools — consumable by [Hermes Agent](https://github.com/NousResearch/hermes-agent) or any MCP client.

Replaces SSH-based dispatch (fragile heredocs, ANSI scraping, plaintext passwords) with authenticated, structured MCP tool calls over Streamable HTTP.

---

## Tools

| Tool | Description |
|------|-------------|
| `claude_execute(prompt, cwd?, timeout?)` | Run [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) |
| `codex_execute(prompt, cwd?, timeout?)` | Run [OpenAI Codex CLI](https://github.com/openai/codex) |
| `hermes_execute(prompt, cwd?, timeout?)` | Run [Hermes Agent](https://github.com/NousResearch/hermes-agent) |
| `grok_execute(prompt, cwd?, timeout?)` | Run [xAI Grok CLI](https://x.ai) |
| `file_read(path, offset?, limit?)` | Read files on the remote VM |
| `file_write(path, content, mode?)` | Write files on the remote VM |
| `file_list(path?)` | List directories |
| `system_run(command, cwd?, timeout?)` | Run whitelisted shell commands |
| `status()` | VM health (uptime, memory, load, tool availability) |

Each AI tool only appears in the tool list if its CLI binary is installed and on `$PATH`.

---

## Requirements

- **A Linux VM** — tested on Arch / Omarchy. Works on any distro with Python 3.11+
- **Python 3.11+** with `mcp>=2.0.0`, `psutil`, `uvicorn`
- **Optional — AI agent CLIs** on the VM (install at least one):
  - [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)
  - [Codex CLI](https://github.com/openai/codex)
  - [Hermes Agent](https://github.com/NousResearch/hermes-agent)
  - [xAI Grok CLI](https://x.ai)

---

## Deployment

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/JPeetz/omarchy-mcp.git
cd omarchy-mcp
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env

# Generate a strong bearer token:
openssl rand -hex 32

# Edit .env — set OMARCHY_MCP_TOKEN to the generated value.
# Optionally set OMARCHY_USER_HOME if your home dir differs.
# For LAN access: set BIND=0.0.0.0 and configure TLS (see .env.example).
```

### 3. Start the Server

```bash
python3 server.py
```

The server starts on `http://127.0.0.1:8911/mcp` (loopback only) by default.

### 4. Verify Locally

```bash
curl -s -X POST http://127.0.0.1:8911/mcp \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <your-token>' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2026-07-28","capabilities":{},"cientInfo":{"name":"test","version":"0.1"}}}'
```

Expected response starts with `event: message` followed by `data: {"jsonrpc":...,"result":...}`.

### 5. Systemd Service (Production)

```bash
# Edit omarchy-mcp.service to match your user/paths, then:
sudo cp omarchy-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now omarchy-mcp.service
```

The `ExecStart`, `WorkingDirectory`, `User`, and `EnvironmentFile` must match your system.

### 6. Connect Hermes Agent

On your desktop / control plane, add the server as an MCP connection:

```bash
hermes config set mcp_servers.omarchy.url "http://<vm-ip>:8911/mcp"
hermes config set mcp_servers.omarchy.headers.Authorization "Bearer <your-token>"
```

Replace `<vm-ip>` with the VM's LAN IP and `<your-token>` with your token.

Test:

```bash
hermes mcp test omarchy
```

Expected: `✓ Connected` and `✓ Tools discovered: 9`.

Call tools:

```bash
hermes -c "Cal the omarchy status tool and show me the result"
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OMARCHY_MCP_TOKEN` | **Yes** | — | Bearer token for MCP auth |
| `PORT`| No | `8911` | Server HTTP port |
| `BIND` | No | `127.0.0.1` | Bind address. `0.0.0.0` for LAN (with TLS) |
| `TLS_CERT` | No | —| Path to TLS certificate (enables HTTPS) |
| `TLS_KEY` | No | — | Path to TLS key file |
| `OMARCHY_USER_HOME` | No | `os.expanduser(~)` | User home for default working directory |
| `OMARCHY_FILE_ROOTS` | No |`$HOME:/tmp` | Colon-separated allowedpaths for file tools |

---

## Security

1. **Bearer token auth** — every request requires `Authorization: Bearer <token>`. Constant-time comparison.
2. **Loopback by default** — server binds to `127.0.0.1`; network exposure is opt-in.
3. **TLS ready** — set `TLS_CERT`/`TLS_KEY` for encrypted transport.
4. **Command whitelist** — `system_run` only allows pre-approved commands.
5. **Path restriction** — file tools only access user home and `/tmp`.
6. **Cancellation** — request cancellation kills the entire subprocess group.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `Connection refused` | Server not running or wrong IP | Check `systemctl status` and `BIND` in `.env` |
| `401 Unauthorized` | Token mismatch | Regenerate token, update both sides |
| `claude_execute` not found | Claude Code not on VM | `which claude` — must be on `$PATH` |
| `Command timed out` | Agent took longer than timeout | Increase `timeout` parameter |

---

## License

MIT — see [LICENSE](./LICENSE).
