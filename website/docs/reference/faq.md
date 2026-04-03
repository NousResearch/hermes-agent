---
sidebar_position: 3
title: "FAQ & Troubleshooting"
description: "Frequently asked questions and solutions to common issues with Hermes Agent"
---

# FAQ & Troubleshooting

Quick answers and fixes for the most common questions and issues.

---

## Frequently Asked Questions

### What LLM providers work with Hermes?

Hermes Agent works with any OpenAI-compatible API. Supported providers include:

- **[OpenRouter](https://openrouter.ai/)** — access hundreds of models through one API key (recommended for flexibility)
- **Nous Portal** — Nous Research's own inference endpoint
- **OpenAI** — GPT-4o, o1, o3, etc.
- **Anthropic** — Claude models (via OpenRouter or compatible proxy)
- **Google** — Gemini models (via OpenRouter or compatible proxy)
- **z.ai / ZhipuAI** — GLM models
- **Kimi / Moonshot AI** — Kimi models
- **MiniMax** — global and China endpoints
- **Local models** — via [Ollama](https://ollama.com/), [vLLM](https://docs.vllm.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp), [SGLang](https://github.com/sgl-project/sglang), or any OpenAI-compatible server

Set your provider with `hermes model` or by editing `~/.hermes/.env`. See the [Environment Variables](./environment-variables.md) reference for all provider keys.

### Does it work on Windows?

**Not natively.** Hermes Agent requires a Unix-like environment. On Windows, install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and run Hermes from inside it. The standard install command works perfectly in WSL2:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Is my data sent anywhere?

API calls go **only to the LLM provider you configure** (e.g., OpenRouter, your local Ollama instance). Hermes Agent does not collect telemetry, usage data, or analytics. Your conversations, memory, and skills are stored locally in `~/.hermes/`.

### Can I use it offline / with local models?

Yes. Run `hermes model`, select **Custom endpoint**, and enter your server's URL:

```bash
hermes model
# Select: Custom endpoint (enter URL manually)
# API base URL: http://localhost:11434/v1
# API key: ollama
# Model name: qwen3.5:27b
# Context length: 32768   ← set this to match your server's actual context window
```

Or configure it directly in `config.yaml`:

```yaml
model:
  default: qwen3.5:27b
  provider: custom
  base_url: http://localhost:11434/v1
```

Hermes persists the endpoint, provider, and base URL in `config.yaml` so it survives restarts. If your local server has exactly one model loaded, `/model custom` auto-detects it. You can also set `provider: custom` in config.yaml — it's a first-class provider, not an alias for anything else.

This works with Ollama, vLLM, llama.cpp server, SGLang, LocalAI, and others. See the [Configuration guide](../user-guide/configuration.md) for details.

:::tip Ollama users
If you set a custom `num_ctx` in Ollama (e.g., `ollama run --num_ctx 16384`), make sure to set the matching context length in Hermes — Ollama's `/api/show` reports the model's *maximum* context, not the effective `num_ctx` you configured.
:::

### How much does it cost?

Hermes Agent itself is **free and open-source** (MIT license). You pay only for the LLM API usage from your chosen provider. Local models are completely free to run.

### Can multiple people use one instance?

Yes. The [messaging gateway](../user-guide/messaging/index.md) lets multiple users interact with the same Hermes Agent instance via Telegram, Discord, Slack, WhatsApp, or Home Assistant. Access is controlled through allowlists (specific user IDs) and DM pairing (first user to message claims access).

### What's the difference between memory and skills?

- **Memory** stores **facts** — things the agent knows about you, your projects, and preferences. Memories are retrieved automatically based on relevance.
- **Skills** store **procedures** — step-by-step instructions for how to do things. Skills are recalled when the agent encounters a similar task.

Both persist across sessions. See [Memory](../user-guide/features/memory.md) and [Skills](../user-guide/features/skills.md) for details.

### Can I use it in my own Python project?

Yes. Import the `AIAgent` class and use Hermes programmatically:

```python
from hermes.agent import AIAgent

agent = AIAgent(model="openrouter/nous/hermes-3-llama-3.1-70b")
response = agent.chat("Explain quantum computing briefly")
```

See the [Python Library guide](../user-guide/features/code-execution.md) for full API usage.

---

## Troubleshooting

### Installation Issues

#### `hermes: command not found` after installation

**Cause:** Your shell hasn't reloaded the updated PATH.

**Solution:**
```bash
# Reload your shell profile
source ~/.bashrc    # bash
source ~/.zshrc     # zsh

# Or start a new terminal session
```

If it still doesn't work, verify the install location:
```bash
which hermes
ls ~/.local/bin/hermes
```

:::tip
The installer adds `~/.local/bin` to your PATH. If you use a non-standard shell config, add `export PATH="$HOME/.local/bin:$PATH"` manually.
:::

#### Python version too old

**Cause:** Hermes requires Python 3.11 or newer.

**Solution:**
```bash
python3 --version   # Check current version

# Install a newer Python
sudo apt install python3.12   # Ubuntu/Debian
brew install python@3.12      # macOS
```

The installer handles this automatically — if you see this error during manual installation, upgrade Python first.

#### `uv: command not found`

**Cause:** The `uv` package manager isn't installed or not in PATH.

**Solution:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### Permission denied errors during install

**Cause:** Insufficient permissions to write to the install directory.

**Solution:**
```bash
# Don't use sudo with the installer — it installs to ~/.local/bin
# If you previously installed with sudo, clean up:
sudo rm /usr/local/bin/hermes
# Then re-run the standard installer
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

---

### Provider & Model Issues

#### API key not working

**Cause:** Key is missing, expired, incorrectly set, or for the wrong provider.

**Solution:**
```bash
# Check your configuration
hermes config show

# Re-configure your provider
hermes model

# Or set directly
hermes config set OPENROUTER_API_KEY sk-or-v1-xxxxxxxxxxxx
```

:::warning
Make sure the key matches the provider. An OpenAI key won't work with OpenRouter and vice versa. Check `~/.hermes/.env` for conflicting entries.
:::

#### Model not available / model not found

**Cause:** The model identifier is incorrect or not available on your provider.

**Solution:**
```bash
# List available models for your provider
hermes model

# Set a valid model
hermes config set HERMES_MODEL openrouter/nous/hermes-3-llama-3.1-70b

# Or specify per-session
hermes chat --model openrouter/meta-llama/llama-3.1-70b-instruct
```

#### Rate limiting (429 errors)

**Cause:** You've exceeded your provider's rate limits.

**Solution:** Wait a moment and retry. For sustained usage, consider:
- Upgrading your provider plan
- Switching to a different model or provider
- Using `hermes chat --provider <alternative>` to route to a different backend

#### Context length exceeded

**Cause:** The conversation has grown too long for the model's context window, or Hermes detected the wrong context length for your model.

**Solution:**
```bash
# Compress the current session
/compress

# Or start a fresh session
hermes chat

# Use a model with a larger context window
hermes chat --model openrouter/google/gemini-2.0-flash-001
```

If this happens on the first long conversation, Hermes may have the wrong context length for your model. Check what it detected:

Look at the CLI startup line — it shows the detected context length (e.g., `📊 Context limit: 128000 tokens`). You can also check with `/usage` during a session.

To fix context detection, set it explicitly:

```yaml
# In ~/.hermes/config.yaml
model:
  default: your-model-name
  context_length: 131072  # your model's actual context window
```

Or for custom endpoints, add it per-model:

```yaml
custom_providers:
  - name: "My Server"
    base_url: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 32768
```

See [Context Length Detection](../integrations/providers.md#context-length-detection) for how auto-detection works and all override options.

---

### Terminal Issues

#### Command blocked as dangerous

**Cause:** Hermes detected a potentially destructive command (e.g., `rm -rf`, `DROP TABLE`). This is a safety feature.

**Solution:** When prompted, review the command and type `y` to approve it. You can also:
- Ask the agent to use a safer alternative
- See the full list of dangerous patterns in the [Security docs](../user-guide/security.md)

:::tip
This is working as intended — Hermes never silently runs destructive commands. The approval prompt shows you exactly what will execute.
:::

#### `sudo` not working via messaging gateway

**Cause:** The messaging gateway runs without an interactive terminal, so `sudo` cannot prompt for a password.

**Solution:**
- Avoid `sudo` in messaging — ask the agent to find alternatives
- If you must use `sudo`, configure passwordless sudo for specific commands in `/etc/sudoers`
- Or switch to the terminal interface for administrative tasks: `hermes chat`

#### Docker backend not connecting

**Cause:** Docker daemon isn't running or the user lacks permissions.

**Solution:**
```bash
# Check Docker is running
docker info

# Add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker run hello-world
```

---

### Messaging Issues

#### Bot not responding to messages

**Cause:** The bot isn't running, isn't authorized, or your user isn't in the allowlist.

**Solution:**
```bash
# Check if the gateway is running
hermes gateway status

# Start the gateway
hermes gateway start

# Check logs for errors
cat ~/.hermes/logs/gateway.log | tail -50
```

#### Messages not delivering

**Cause:** Network issues, bot token expired, or platform webhook misconfiguration.

**Solution:**
- Verify your bot token is valid with `hermes gateway setup`
- Check gateway logs: `cat ~/.hermes/logs/gateway.log | tail -50`
- For webhook-based platforms (Slack, WhatsApp), ensure your server is publicly accessible

#### Allowlist confusion — who can talk to the bot?

**Cause:** Authorization mode determines who gets access.

**Solution:**

| Mode | How it works |
|------|-------------|
| **Allowlist** | Only user IDs listed in config can interact |
| **DM pairing** | First user to message in DM claims exclusive access |
| **Open** | Anyone can interact (not recommended for production) |

Configure in `~/.hermes/config.yaml` under your gateway's settings. See the [Messaging docs](../user-guide/messaging/index.md).

#### Gateway won't start

**Cause:** Missing dependencies, port conflicts, or misconfigured tokens.

**Solution:**
```bash
# Install messaging dependencies
pip install "hermes-agent[telegram]"   # or [discord], [slack], [whatsapp]

# Check for port conflicts
lsof -i :8080

# Verify configuration
hermes config show
```

#### macOS: Node.js / ffmpeg / other tools not found by gateway

**Cause:** launchd services inherit a minimal PATH (`/usr/bin:/bin:/usr/sbin:/sbin`) that doesn't include Homebrew, nvm, cargo, or other user-installed tool directories. This commonly breaks the WhatsApp bridge (`node not found`) or voice transcription (`ffmpeg not found`).

**Solution:** The gateway captures your shell PATH when you run `hermes gateway install`. If you installed tools after setting up the gateway, re-run the install to capture the updated PATH:

```bash
hermes gateway install    # Re-snapshots your current PATH
hermes gateway start      # Detects the updated plist and reloads
```

You can verify the plist has the correct PATH:
```bash
/usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:PATH" \
  ~/Library/LaunchAgents/ai.hermes.gateway.plist
```

---

### Performance Issues

#### Slow responses

**Cause:** Large model, distant API server, or heavy system prompt with many tools.

**Solution:**
- Try a faster/smaller model: `hermes chat --model openrouter/meta-llama/llama-3.1-8b-instruct`
- Reduce active toolsets: `hermes chat -t "terminal"`
- Check your network latency to the provider
- For local models, ensure you have enough GPU VRAM

#### High token usage

**Cause:** Long conversations, verbose system prompts, or many tool calls accumulating context.

**Solution:**
```bash
# Compress the conversation to reduce tokens
/compress

# Check session token usage
/usage
```

:::tip
Use `/compress` regularly during long sessions. It summarizes the conversation history and reduces token usage significantly while preserving context.
:::

#### Session getting too long

**Cause:** Extended conversations accumulate messages and tool outputs, approaching context limits.

**Solution:**
```bash
# Compress current session (preserves key context)
/compress

# Start a new session with a reference to the old one
hermes chat

# Resume a specific session later if needed
hermes chat --continue
```

---

### MCP Issues

#### MCP server not connecting

**Cause:** Server binary not found, wrong command path, or missing runtime.

**Solution:**
```bash
# Ensure MCP dependencies are installed (already included in standard install)
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

# For npm-based servers, ensure Node.js is available
node --version
npx --version

# Test the server manually
npx -y @modelcontextprotocol/server-filesystem /tmp
```

Verify your `~/.hermes/config.yaml` MCP configuration:
```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
```

#### Tools not showing up from MCP server

**Cause:** Server started but tool discovery failed, tools were filtered out by config, or the server does not support the MCP capability you expected.

**Solution:**
- Check gateway/agent logs for MCP connection errors
- Ensure the server responds to the `tools/list` RPC method
- Review any `tools.include`, `tools.exclude`, `tools.resources`, `tools.prompts`, or `enabled` settings under that server
- Remember that resource/prompt utility tools are only registered when the session actually supports those capabilities
- Use `/reload-mcp` after changing config

```bash
# Verify MCP servers are configured
hermes config show | grep -A 12 mcp_servers

# Restart Hermes or reload MCP after config changes
hermes chat
```

See also:
- [MCP (Model Context Protocol)](/docs/user-guide/features/mcp)
- [Use MCP with Hermes](/docs/guides/use-mcp-with-hermes)
- [MCP Config Reference](/docs/reference/mcp-config-reference)

#### MCP timeout errors

**Cause:** The MCP server is taking too long to respond, or it crashed during execution.

**Solution:**
- Increase the timeout in your MCP server config if supported
- Check if the MCP server process is still running
- For remote HTTP MCP servers, check network connectivity

:::warning
If an MCP server crashes mid-request, Hermes will report a timeout. Check the server's own logs (not just Hermes logs) to diagnose the root cause.
:::

---

## Profiles

### How do profiles differ from just setting HERMES_HOME?

Profiles are a managed layer on top of `HERMES_HOME`. You *could* manually set `HERMES_HOME=/some/path` before every command, but profiles handle all the plumbing for you: creating the directory structure, generating shell aliases (`hermes-work`), tracking the active profile in `~/.hermes/active_profile`, and syncing skill updates across all profiles automatically. They also integrate with tab completion so you don't have to remember paths.

### Can two profiles share the same bot token?

No. Each messaging platform (Telegram, Discord, etc.) requires exclusive access to a bot token. If two profiles try to use the same token simultaneously, the second gateway will fail to connect. Create a separate bot per profile — for Telegram, talk to [@BotFather](https://t.me/BotFather) to make additional bots.

### Do profiles share memory or sessions?

No. Each profile has its own memory store, session database, and skills directory. They are completely isolated. If you want to start a new profile with existing memories and sessions, use `hermes profile create newname --clone-all` to copy everything from the current profile.

### What happens when I run `hermes update`?

`hermes update` pulls the latest code and reinstalls dependencies **once** (not per-profile). It then syncs updated skills to all profiles automatically. You only need to run `hermes update` once — it covers every profile on the machine.

### Can I move a profile to a different machine?

Yes. Export the profile to a portable archive and import it on the other machine:

```bash
# On the source machine
hermes profile export work ./work-backup.tar.gz

# Copy the file to the target machine, then:
hermes profile import ./work-backup.tar.gz work
```

The imported profile will have all config, memories, sessions, and skills from the export. You may need to update paths or re-authenticate with providers if the new machine has a different setup.

### How many profiles can I run?

There is no hard limit. Each profile is just a directory under `~/.hermes/profiles/`. The practical limit depends on your disk space and how many concurrent gateways your system can handle (each gateway is a lightweight Python process). Running dozens of profiles is fine; each idle profile uses no resources.

---

## Workflows & Patterns

### Using different models for different tasks (multi-model workflows)

**Scenario:** You use GPT-5.4 as your daily driver, but Gemini or Grok writes better social media content. Manually switching models every time is tedious.

**Solution: Delegation config.** Hermes can route subagents to a different model automatically. Set this in `~/.hermes/config.yaml`:

```yaml
delegation:
  model: "google/gemini-3-flash-preview"   # subagents use this model
  provider: "openrouter"                    # provider for subagents
```

Now when you tell Hermes "write me a Twitter thread about X" and it spawns a `delegate_task` subagent, that subagent runs on Gemini instead of your main model. Your primary conversation stays on GPT-5.4.

You can also be explicit in your prompt: *"Delegate a task to write social media posts about our product launch. Use your subagent for the actual writing."* The agent will use `delegate_task`, which automatically picks up the delegation config.

For one-off model switches without delegation, use `/model` in the CLI:

```bash
/model google/gemini-3-flash-preview    # switch for this session
# ... write your content ...
/model openai/gpt-5.4                   # switch back
```

See [Subagent Delegation](../user-guide/features/delegation.md) for more on how delegation works.

### Running multiple agents on one WhatsApp number (per-chat binding)

**Scenario:** In OpenClaw, you had multiple independent agents bound to specific WhatsApp chats — one for a family shopping list group, another for your private chat. Can Hermes do this?

**Current limitation:** Hermes profiles each require their own WhatsApp number/session. You cannot bind multiple profiles to different chats on the same WhatsApp number — the WhatsApp bridge (Baileys) uses one authenticated session per number.

**Workarounds:**

1. **Use a single profile with personality switching.** Create different `AGENTS.md` context files or use the `/personality` command to change behavior per chat. The agent sees which chat it's in and can adapt.

2. **Use cron jobs for specialized tasks.** For a shopping list tracker, set up a cron job that monitors a specific chat and manages the list — no separate agent needed.

3. **Use separate numbers.** If you need truly independent agents, pair each profile with its own WhatsApp number. Virtual numbers from services like Google Voice work for this.

4. **Use Telegram or Discord instead.** These platforms support per-chat binding more naturally — each Telegram group or Discord channel gets its own session, and you can run multiple bot tokens (one per profile) on the same account.

See [Profiles](../user-guide/profiles.md) and [WhatsApp setup](../user-guide/messaging/whatsapp.md) for more details.

### Controlling what shows up in Telegram (hiding logs and reasoning)

**Scenario:** You see gateway exec logs, Hermes reasoning, and tool call details in Telegram instead of just the final output.

**Solution:** The `display.tool_progress` setting in `config.yaml` controls how much tool activity is shown:

```yaml
display:
  tool_progress: "off"   # options: off, new, all, verbose
```

- **`off`** — Only the final response. No tool calls, no reasoning, no logs.
- **`new`** — Shows new tool calls as they happen (brief one-liners).
- **`all`** — Shows all tool activity including results.
- **`verbose`** — Full detail including tool arguments and outputs.

For messaging platforms, `off` or `new` is usually what you want. After editing `config.yaml`, restart the gateway for changes to take effect.

You can also toggle this per-session with the `/verbose` command (if enabled):

```yaml
display:
  tool_progress_command: true   # enables /verbose in the gateway
```

### Managing skills on Telegram (slash command limit)

**Scenario:** Telegram has a 100 slash command limit, and your skills are pushing past it. You want to disable skills you don't need on Telegram, but `hermes skills config` settings don't seem to take effect.

**Solution:** Use `hermes skills config` to disable skills per-platform. This writes to `config.yaml`:

```yaml
skills:
  disabled: []                    # globally disabled skills
  platform_disabled:
    telegram: [skill-a, skill-b]  # disabled only on telegram
```

After changing this, **restart the gateway** (`hermes gateway restart` or kill and relaunch). The Telegram bot command menu rebuilds on startup.

:::tip
Skills with very long descriptions are truncated to 40 characters in the Telegram menu to stay within payload size limits. If skills aren't appearing, it may be a total payload size issue rather than the 100 command count limit — disabling unused skills helps with both.
:::

### Shared thread sessions (multiple users, one conversation)

**Scenario:** You have a Telegram or Discord thread where multiple people mention the bot. You want all mentions in that thread to be part of one shared conversation, not separate per-user sessions.

**Current behavior:** Hermes creates sessions keyed by user ID on most platforms, so each person gets their own conversation context. This is by design for privacy and context isolation.

**Workarounds:**

1. **Use Slack.** Slack sessions are keyed by thread, not by user. Multiple users in the same thread share one conversation — exactly the behavior you're describing. This is the most natural fit.

2. **Use a group chat with a single user.** If one person is the designated "operator" who relays questions, the session stays unified. Others can read along.

3. **Use a Discord channel.** Discord sessions are keyed by channel, so all users in the same channel share context. Use a dedicated channel for the shared conversation.

### Exporting Hermes to another machine

**Scenario:** You've built up skills, cron jobs, and memories on one machine and want to move everything to a new dedicated Linux box.

**Solution:**

1. Install Hermes Agent on the new machine:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
   ```

2. Copy your entire `~/.hermes/` directory **except** the `hermes-agent` subdirectory (that's the code repo — the new install has its own):
   ```bash
   # On the source machine
   rsync -av --exclude='hermes-agent' ~/.hermes/ newmachine:~/.hermes/
   ```

   Or use profile export/import:
   ```bash
   # On source machine
   hermes profile export default ./hermes-backup.tar.gz

   # On target machine
   hermes profile import ./hermes-backup.tar.gz default
   ```

3. On the new machine, run `hermes setup` to verify API keys and provider config are working. Re-authenticate any messaging platforms (especially WhatsApp, which uses QR pairing).

The `~/.hermes/` directory contains everything: `config.yaml`, `.env`, `SOUL.md`, `memories/`, `skills/`, `state.db` (sessions), `cron/`, and any custom plugins. The code itself lives in `~/.hermes/hermes-agent/` and is installed fresh.

### Permission denied when reloading shell after install

**Scenario:** After running the Hermes installer, `source ~/.zshrc` gives a permission denied error.

**Cause:** This usually happens when `~/.zshrc` (or `~/.bashrc`) has incorrect file permissions, or when the installer couldn't write to it cleanly. It's not a Hermes-specific issue — it's a shell config permissions problem.

**Solution:**
```bash
# Check permissions
ls -la ~/.zshrc

# Fix if needed (should be -rw-r--r-- or 644)
chmod 644 ~/.zshrc

# Then reload
source ~/.zshrc

# Or just open a new terminal window — it picks up PATH changes automatically
```

If the installer added the PATH line but permissions are wrong, you can add it manually:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

### Error 400 on first agent run

**Scenario:** Setup completes fine, but the first chat attempt fails with HTTP 400.

**Cause:** Usually a model name mismatch — the configured model doesn't exist on your provider, or the API key doesn't have access to it.

**Solution:**
```bash
# Check what model and provider are configured
hermes config show | head -20

# Re-run model selection
hermes model

# Or test with a known-good model
hermes chat -q "hello" --model anthropic/claude-sonnet-4.6
```

If using OpenRouter, make sure your API key has credits. A 400 from OpenRouter often means the model requires a paid plan or the model ID has a typo.

---

## Still Stuck?

If your issue isn't covered here:

1. **Search existing issues:** [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
2. **Ask the community:** [Nous Research Discord](https://discord.gg/nousresearch)
3. **File a bug report:** Include your OS, Python version (`python3 --version`), Hermes version (`hermes --version`), and the full error message
