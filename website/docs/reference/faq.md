---
sidebar_position: 3
title: "FAQ & Troubleshooting"
description: "Frequently asked questions and solutions to common issues with Rabbit Agent"
---

# FAQ & Troubleshooting

Python dependency commands on this page use a
[PM-prepared source checkout](./package-management.md#developer-workflow).
After a dependency change, reactivate the checkout and restart Rabbit.

Quick answers and fixes for the most common questions and issues.

---

## Frequently Asked Questions

### What LLM providers work with Rabbit?

Rabbit Agent works with any OpenAI-compatible API. Supported providers include:

- **[OpenRouter](https://openrouter.ai/)** — access hundreds of models through one API key (recommended for flexibility)
- **OpenAI** — GPT-5.4, GPT-5-codex, GPT-4.1, GPT-4o, etc.
- **Anthropic** — Claude models (direct API, OAuth via `rabbit auth add anthropic`, OpenRouter, or any compatible proxy)
- **Google** — Gemini models (direct API via `gemini` provider, OpenRouter, or compatible proxy)
- **z.ai / ZhipuAI** — GLM models
- **Kimi / Moonshot AI** — Kimi models
- **MiniMax** — global and China endpoints
- **Local models** — via [Ollama](https://ollama.com/), [vLLM](https://docs.vllm.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp), [SGLang](https://github.com/sgl-project/sglang), or any OpenAI-compatible server

Set your provider with `rabbit model` or by editing `~/.rabbit/.env`. See the [Environment Variables](./environment-variables.md) reference for all provider keys.

### Does it work on Windows/Android/my platform??
See **[Platform Support](../getting-started/platform-support.md)** for the full platform availability matrix.

### I run Rabbit in WSL2. What's the best way to control my normal Windows Chrome?

Prefer an MCP bridge over `/browser connect`.

Recommended pattern:

- run Rabbit inside WSL2
- keep using your normal signed-in Chrome on Windows
- add `chrome-devtools-mcp` as an MCP server through `cmd.exe` or `powershell.exe`
- let Rabbit use the resulting MCP browser tools

This is more reliable than trying to force Rabbit core browser transport to attach directly across the WSL2/Windows boundary.

See:

- [Use MCP with Rabbit](../guides/use-mcp-with-rabbit.md#wsl2-bridge-rabbit-in-wsl-to-windows-chrome)
- [Browser Automation](../user-guide/features/browser.md#wsl2--windows-chrome-prefer-mcp-over-browser-connect)

### Is my data sent anywhere?

API calls go **only to the LLM provider you configure** (e.g., OpenRouter, your local Ollama instance). Rabbit Agent does not collect telemetry, usage data, or analytics. Your conversations, memory, and skills are stored locally in `~/.rabbit/`.

### Can I use it offline / with local models?

Yes. Run `rabbit model`, select **Custom endpoint**, and enter your server's URL:

```bash
rabbit model
# Select: Custom endpoint (enter URL manually)
# API base URL: http://localhost:11434/v1
# API key: ollama
# Model name: qwen3.5:27b
# Context length: 64000   ← Rabbit minimum; set this to match your server's actual context window
```

Or configure it directly in `config.yaml`:

```yaml
model:
  default: qwen3.5:27b
  provider: custom
  base_url: http://localhost:11434/v1
```

Rabbit persists the endpoint, provider, and base URL in `config.yaml` so it survives restarts. If your local server has exactly one model loaded, `/model custom` auto-detects it. You can also set `provider: custom` in config.yaml — it's a first-class provider, not an alias for anything else.

This works with Ollama, vLLM, llama.cpp server, SGLang, LocalAI, and others. See the [Configuration guide](../user-guide/configuration.md) for details.

:::tip Ollama users
If you set a custom `num_ctx` in Ollama (e.g., `ollama run --num_ctx 64000`), make sure to set the matching context length in Rabbit — Ollama's `/api/show` reports the model's *maximum* context, not the effective `num_ctx` you configured.
:::

:::tip Timeouts with local models
Rabbit auto-detects local endpoints and relaxes streaming timeouts (read timeout raised from 120s to 1800s, stale stream detection disabled). If you still hit timeouts on very large contexts, set `RABBIT_STREAM_READ_TIMEOUT=1800` in your `.env`. See the [Local LLM guide](../guides/local-llm-on-mac.md#timeouts) for details.
:::

### How much does it cost?

Rabbit Agent itself is **free and open-source** (MIT license). You pay only for the LLM API usage from your chosen provider. Local models are completely free to run.

### Can multiple people use one instance?

Yes. The [messaging gateway](../user-guide/messaging/index.md) lets multiple users interact with the same Rabbit Agent instance via Telegram, Discord, Slack, WhatsApp, or Home Assistant. Access is controlled through allowlists (specific user IDs) and DM pairing (first user to message claims access).

### What's the difference between memory and skills?

- **Memory** stores **facts** — things the agent knows about you, your projects, and preferences. Memories are retrieved automatically based on relevance.
- **Skills** store **procedures** — step-by-step instructions for how to do things. Skills are recalled when the agent encounters a similar task.

Both persist across sessions. See [Memory](../user-guide/features/memory.md) and [Skills](../user-guide/features/skills.md) for details.

### Can I use it in my own Python project?

Yes. Import the `AIAgent` class and use Rabbit programmatically:

```python
from run_agent import AIAgent

agent = AIAgent(model="anthropic/claude-opus-4.7")
response = agent.chat("Explain quantum computing briefly")
```

See the [Python Library guide](../user-guide/features/code-execution.md) for full API usage.

---

## Troubleshooting

### Installation Issues

#### `rabbit: command not found` after installation

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
which rabbit
ls ~/.local/bin/rabbit
```

:::tip
The installer adds `~/.local/bin` to your PATH. If you use a non-standard shell config, add `export PATH="$HOME/.local/bin:$PATH"` manually.
:::

#### Unsupported Python version

Current first-party installations require **Python 3.14**, not an arbitrary
newer version. The `>=3.11,<3.15` range in `pyproject.toml` allows older
installations to run the updater before switching to 3.14; it does not mean
the current runtime supports 3.11–3.13. The installer and packaged
distributions provide their pinned interpreter.

For a manual source environment, use the
[development setup](../developer-guide/contributing.md#development-setup).
Do not replace the interpreter inside an installed app or container.
For a managed-install error, run `rabbit doctor` and use that installation's
[update method](../getting-started/updating.md).

#### Terminal commands say `node: command not found` (or `nvm`, `pyenv`, `asdf`, …)

**Cause:** Rabbit builds a per-session environment snapshot by running `bash -l` once at startup. A bash login shell reads `/etc/profile`, `~/.bash_profile`, and `~/.profile`, but **does not source `~/.bashrc`** — so tools that install themselves there (`nvm`, `asdf`, `pyenv`, `cargo`, custom `PATH` exports) stay invisible to the snapshot. This most commonly happens when Rabbit runs under systemd or in a minimal shell where nothing has pre-loaded the interactive shell profile.

**Solution:** Rabbit auto-sources `~/.bashrc` by default. If that's not enough — e.g. you're a zsh user whose PATH lives in `~/.zshrc`, or you init `nvm` from a standalone file — list the extra files to source in `~/.rabbit/config.yaml`:

```yaml
terminal:
  shell_init_files:
    - ~/.zshrc                     # zsh users: pulls zsh-managed PATH into the bash snapshot
    - ~/.nvm/nvm.sh                # direct nvm init (works regardless of shell)
    - /etc/profile.d/cargo.sh      # system-wide rc files
  # When this list is set, the default ~/.bashrc auto-source is NOT added —
  # include it explicitly if you want both:
  #   - ~/.bashrc
  #   - ~/.zshrc
```

Missing files are skipped silently. Sourcing happens in bash, so files that rely on zsh-only syntax may error — if that's a concern, source just the PATH-setting portion (e.g. nvm's `nvm.sh` directly) rather than the whole rc file.

Independently of the init files, every terminal command's `PATH` is completed with the standard system directories (`/usr/local/bin`, `/opt/homebrew/bin`, …), the Rabbit-managed runtime dirs, and `~/.local/bin` when it exists (the `pip --user` / `pipx` / `uv tool` install target) — appended after your own entries, so precedence is unchanged. This covers backends started with a thin non-interactive PATH (systemd, GUI launchers, the Desktop SSH remote backend) without any configuration.

To disable the auto-source behaviour (strict login-shell semantics only):

```yaml
terminal:
  auto_source_bashrc: false
```

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
sudo rm /usr/local/bin/rabbit
# Then re-run the standard installer
curl -fsSL https://raw.githubusercontent.com/seven0070/Rabbit-/main/scripts/install.sh | bash
```

---

### Provider & Model Issues

#### The agent says "Rabbit policy" or "Rabbit guardrails" refused my request

A model cannot reliably identify why it refused a request. If the refusal appears only in the assistant's prose, its claim that a hidden Rabbit runtime policy caused it may be a hallucinated explanation or a restriction applied by the selected model or provider.

Rabbit enforcement is explicit: a blocked tool action returns a tool error naming the denied command or path, and an approval-required action shows an approval prompt. Rabbit does not silently turn those execution controls into a general content-refusal layer. Provider-level controls can still apply when configured, such as Amazon Bedrock Guardrails.

To isolate the source:

1. Run `/status` to confirm the active model and provider.
2. Check whether the refusal includes an actual Rabbit tool error or approval prompt. If it is prose only, do not treat the model's attribution as runtime evidence.
3. Retry in a fresh session with another configured model or provider. A refusal that changes with the model is model/provider behavior, not a Rabbit execution control.
4. If an explicit tool error appears, use its exact text when reporting the problem.

See [Security](../user-guide/security.md) for Rabbit' documented execution controls and [Providers](../integrations/providers.md) for provider configuration.

#### "…refused this request because of a policy on your account"

**Meaning:** the provider rejected the request for an account-level reason that retrying cannot change — an aggregator's data/privacy settings excluded every endpoint for the model, or the model's upstream provider has blocked the account (for example `this user has been blocked for a previous policy violation`, which OpenRouter can relay inside an otherwise successful HTTP 200 stream). Rabbit sends the request once, does not retry it or rotate credentials, and moves to your fallback chain if one is configured.

**Solution:** check the account's status and data/privacy settings with the provider named in the reply, or switch to another model or provider with `/model`. `rabbit fallback add` routes future blocks to a backup automatically.

#### "Could not open a stream to `<host>` after N attempts (request X KB)"

**Meaning:** every connect attempt to that endpoint failed before a single stream event arrived, so nothing was billed; the normal retry/fallback chain still runs afterwards. The line names the host actually contacted, how many attempts were made, and the serialized request size — the three things that separate an outage from a request-size limit.

**Solution:** if the request is large (hundreds of KB — long coding sessions reach this once the context grows) and short new chats work, the endpoint or a proxy in front of it is likely rejecting bodies that size: raise its body limit, or run `/compress` to shrink the context. If the request is small, the endpoint is unreachable — check the `base_url`, then retry with `/retry`. `logs/agent.log` records the exception chain for each attempt.

#### Messaging replies: "interrupted mid-request" vs "not running or is unreachable" vs "could not reach"

Chat surfaces (Telegram, Discord, Slack, …) never show the raw transport exception; the gateway maps it to one of three short replies, and the difference tells you where to look:

| Reply | What happened | What to do |
|---|---|---|
| "The connection to the AI model service was **interrupted mid-request** — usually transient." | An established connection was cut (`Connection reset by peer`, EOF, `RemoteProtocolError`). The endpoint answered the connect, so it is running. | `/retry`. If it recurs on large requests, see the "stream" entry above. |
| "The AI model service isn't reachable right now — the configured model endpoint is **not running or is unreachable**." | Nothing accepted the connection (`Connection refused`, no route to host, DNS failure). | Start the model server / check `base_url`, then `/retry`; `rabbit doctor` on the host. |
| "Rabbit **could not reach** the AI model service (no further detail from the SDK)." | The SDK reported a generic `APIConnectionError` and kept no cause; neither of the above is certain. | `/retry`; `rabbit doctor` if it persists. The raw exception is in `rabbit logs`. |

#### `/model` only shows one provider / can't switch providers

**Cause:** `/model` (inside a chat session) can only switch between providers you've **already configured**. If you've only set up OpenRouter, that's all `/model` will show.

**Solution:** Exit your session and use `rabbit model` from your terminal to add new providers:

```bash
# Exit the Rabbit chat session first (Ctrl+C or /quit)

# Run the full provider setup wizard
rabbit model

# This lets you: add providers, run OAuth, enter API keys, configure endpoints
```

After adding a new provider via `rabbit model`, start a new chat session — `/model` will now show all your configured providers.

:::tip Quick reference
| Want to... | Use |
|-----------|-----|
| Add a new provider | `rabbit model` (from terminal) |
| Enter/change API keys | `rabbit model` (from terminal) |
| Switch model mid-session | `/model <name>` (inside session) |
| Switch to different configured provider | `/model provider:model` (inside session) |
:::

#### API key not working

**Cause:** Key is missing, expired, incorrectly set, or for the wrong provider.

**Solution:**
```bash
# Check your configuration
rabbit config show

# Re-configure your provider
rabbit model

# Or set directly
rabbit config set OPENROUTER_API_KEY sk-or-v1-xxxxxxxxxxxx
```

:::warning
Make sure the key matches the provider. An OpenAI key won't work with OpenRouter and vice versa. Check `~/.rabbit/.env` for conflicting entries.
:::

#### Model not available / model not found

**Cause:** The model identifier is incorrect or not available on your provider.

**Solution:**
```bash
# List available models for your provider
rabbit model

# Set a valid model
rabbit config set model.default anthropic/claude-opus-4.7

# Or specify per-session
rabbit chat --model openrouter/meta-llama/llama-3.1-70b-instruct
```

#### Rate limiting (429 errors)

**Cause:** You've exceeded your provider's rate limits.

**Solution:** Wait a moment and retry. For sustained usage, consider:
- Upgrading your provider plan
- Switching to a different model or provider
- Using `rabbit chat --provider <alternative>` to route to a different backend

#### Context length exceeded

**Cause:** The conversation has grown too long for the model's context window, or Rabbit detected the wrong context length for your model.

**Solution:**
```bash
# Compress the current session
/compress

# Or start a fresh session
rabbit chat

# Use a model with a larger context window
rabbit chat --model openrouter/google/gemini-3-flash-preview
```

If this happens on the first long conversation, Rabbit may have the wrong context length for your model. Check what it detected:

Look at the CLI startup line — it shows the detected context length (e.g., `📊 Context limit: 128000 tokens`). You can also check with `/usage` during a session.

**Local servers (llama.cpp, Ollama) that go silent instead of erroring:** when a provider rejects a request as too large, Rabbit compacts the conversation and rebuilds the request. Rabbit re-measures the *complete* rebuilt request (system prompt + tool schemas + messages) before retrying, and runs further bounded compaction passes if it is still over the threshold. If the request still cannot fit, the turn ends with `Context length exceeded: compression could not reduce the rebuilt request below the safe threshold` rather than sending an oversized request that llama.cpp would silently truncate (`stop processing: n_tokens = 65535, truncated = 1` in the server log). If you hit that message, the fix is almost always the configured `context_length` above: make it match the server's actual `-c` / `--ctx-size`.

**"The model server rejected this request as too large, but this conversation is only about N tokens…":** a local server (localhost, LAN, Tailscale) said "context exceeded" without quoting any measurement, while Rabbit's own estimate of the request is far below the window it knows for the model — so it does **not** compress or blame the conversation, and the turn stays retryable. On single-slot local servers (LM Studio, Ollama) this is almost always another request holding the server's context at that moment — typically a background memory review from an earlier session (`thread=bg-review` in `logs/agent.log`). Wait a moment and `/retry`. If it recurs with no other Rabbit process running, the server is loading the model with a smaller window than Rabbit assumes: raise the server's context setting or lower `model.context_length` to match it. Hosted providers never get this message: they have no shared slot to wait out, so the same rejection there means the route's real window is smaller than Rabbit assumes, and Rabbit compresses and retries instead.

To fix context detection, set it explicitly:

```yaml
# In ~/.rabbit/config.yaml
model:
  default: your-model-name
  context_length: 131072  # your model's actual context window
```

Or for custom endpoints, add it per-model on the provider entry:

```yaml
providers:
  my-server:
    api: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 64000
```

(Older configs use the legacy `custom_providers:` list — still supported and auto-migrated to `providers:`.)

See [Context Length Detection](../integrations/providers.md#context-length-detection) for how auto-detection works and all override options.

---

### Terminal Issues

#### Command blocked as dangerous

**Cause:** Rabbit detected a potentially destructive command (e.g., `rm -rf`, `DROP TABLE`). This is a safety feature.

**Solution:** When prompted, review the command and type `y` to approve it. You can also:
- Ask the agent to use a safer alternative
- See the full list of dangerous patterns in the [Security docs](../user-guide/security.md)

:::tip
This is working as intended — Rabbit never silently runs destructive commands. The approval prompt shows you exactly what will execute.
:::

#### `sudo` not working via messaging gateway

**Cause:** The messaging gateway runs without an interactive terminal, so `sudo` cannot prompt for a password.

**Solution:**
- Avoid `sudo` in messaging — ask the agent to find alternatives
- If you must use `sudo`, configure passwordless sudo for specific commands in `/etc/sudoers`
- Or switch to the terminal interface for administrative tasks: `rabbit chat`

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
rabbit gateway status

# Start the gateway
rabbit gateway start

# Check logs for errors
cat ~/.rabbit/logs/gateway.log | tail -50
```

#### Messages not delivering

**Cause:** Network issues, bot token expired, or platform webhook misconfiguration.

**Solution:**
- Verify your bot token is valid with `rabbit gateway setup`
- Check gateway logs: `cat ~/.rabbit/logs/gateway.log | tail -50`
- For webhook-based platforms (Slack, WhatsApp), ensure your server is publicly accessible

#### Allowlist confusion — who can talk to the bot?

**Cause:** Authorization mode determines who gets access.

**Solution:**

| Mode | How it works |
|------|-------------|
| **Allowlist** | Only user IDs listed in config can interact |
| **DM pairing** | First user to message in DM claims exclusive access |
| **Open** | Anyone can interact (not recommended for production) |

Configure in `~/.rabbit/config.yaml` under your gateway's settings. See the [Messaging docs](../user-guide/messaging/index.md).

#### Gateway won't start

**Cause:** Missing dependencies, port conflicts, or misconfigured tokens.

**Solution:**
```bash
# Install core messaging gateway dependencies
cd ~/.rabbit/rabbit-agent && python -c "import pm; pm.sync_venv(['messaging'], explicit=True)"  # Telegram, Discord, Slack, and shared gateway deps

# Check for port conflicts
lsof -i :8080

# Verify configuration
rabbit config show
```

#### WSL: Gateway keeps disconnecting or `rabbit gateway start` fails

**Cause:** WSL's systemd support is unreliable. Many WSL2 installations don't have systemd enabled, and even when enabled, services may not survive WSL restarts or Windows idle shutdowns.

**Solution:** Use foreground mode instead of the systemd service:

```bash
# Option 1: Direct foreground (simplest)
rabbit gateway run

# Option 2: Persistent via tmux (survives terminal close)
tmux new -s rabbit 'rabbit gateway run'
# Reattach later: tmux attach -t rabbit

# Option 3: Background via nohup
nohup rabbit gateway run > ~/.rabbit/logs/gateway.log 2>&1 &
```

If you want to try systemd anyway, make sure it's enabled:

1. Open `/etc/wsl.conf` (create it if it doesn't exist)
2. Add:
   ```ini
   [boot]
   systemd=true
   ```
3. From PowerShell: `wsl --shutdown`
4. Reopen your WSL terminal
5. Verify: `systemctl is-system-running` should say "running" or "degraded"

:::tip Auto-start on Windows boot
For reliable auto-start, use Windows Task Scheduler to launch WSL + the gateway on login:
1. Create a task that runs `wsl -d Ubuntu -- bash -lc 'rabbit gateway run'`
2. Set it to trigger on user logon
:::

#### macOS: Node.js / ffmpeg / other tools not found by gateway

**Cause:** launchd services inherit a minimal PATH (`/usr/bin:/bin:/usr/sbin:/sbin`) that doesn't include Homebrew, nvm, cargo, or other user-installed tool directories. This commonly breaks the WhatsApp bridge (`node not found`) or voice transcription (`ffmpeg not found`).

**Solution:** The gateway captures your shell PATH when you run `rabbit gateway install`. If you installed tools after setting up the gateway, re-run the install to capture the updated PATH:

```bash
rabbit gateway install    # Re-snapshots your current PATH
rabbit gateway start      # Detects the updated plist and reloads
```

You can verify the plist has the correct PATH:
```bash
/usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:PATH" \
  ~/Library/LaunchAgents/ai.rabbit.gateway.plist
```

---

### Performance Issues

#### Slow responses

**Cause:** Large model, distant API server, or heavy system prompt with many tools.

**Solution:**
- Try a faster/smaller model: `rabbit chat --model openrouter/meta-llama/llama-3.1-8b-instruct`
- Reduce active toolsets: `rabbit chat -t "terminal"`
- Check your network latency to the provider
- For local models, ensure you have enough GPU VRAM

#### High token usage

**Cause:** Long conversations, verbose system prompts, or many tool calls accumulating context.

**Solution:**
```bash
# See exactly what the fixed prompt costs — breakdown by block
# (system prompt, skills index, memory, tool schemas). Runs offline.
rabbit prompt-size

# Compress the conversation to reduce tokens
/compress

# Check session token usage
/usage
```

If the baseline looks high before you've typed anything, that's the fixed prompt budget — the system prompt plus tool schemas sent on every call. Run [`rabbit prompt-size`](./cli-commands.md#rabbit-prompt-size) to measure it, then trim: disable toolsets you don't use (`rabbit tools`) and uninstall or disable skills you don't need (`rabbit skills`).

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
rabbit chat

# Resume a specific session later if needed
rabbit chat --continue
```

---

### MCP Issues

#### MCP server not connecting

**Cause:** Server binary not found, wrong command path, or missing runtime.

**Solution:**
```bash
# Ensure MCP dependencies are installed (already included in standard install)
cd ~/.rabbit/rabbit-agent && python -c "import pm; pm.sync_venv(['mcp'], explicit=True)"

# For npm-based servers, ensure Node.js is available
node --version
npx --version

# Test the server manually
npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/dir
```

Verify your `~/.rabbit/config.yaml` MCP configuration:
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
rabbit config show | grep -A 12 mcp_servers

# Restart Rabbit or reload MCP after config changes
rabbit chat
```

See also:
- [MCP (Model Context Protocol)](../user-guide/features/mcp.md)
- [Use MCP with Rabbit](../guides/use-mcp-with-rabbit.md)
- [MCP Config Reference](./mcp-config-reference.md)

#### MCP timeout errors

**Cause:** The MCP server is taking too long to respond, or it crashed during execution.

**Solution:**
- Increase the timeout in your MCP server config if supported
- Check if the MCP server process is still running
- For remote HTTP MCP servers, check network connectivity

:::warning
If an MCP server crashes mid-request, Rabbit will report a timeout. Check the server's own logs (not just Rabbit logs) to diagnose the root cause.
:::

---

### Skills Issues

#### The Skills Hub page won't load in the desktop app (403 / blocked)

**Cause:** The docs site (`https://github.com/seven0070/Rabbit-`) is served through Vercel, whose WAF denies some residential IP ranges it considers flagged. If your network is on such a range, every request to the domain returns a 403 block page.

**Solution:** Check whether a proxy, DNS filter, or firewall is blocking the host — and report the affected range to the maintainers so it can be reviewed on the deployment side.

---

## Profiles

### How do profiles differ from just setting RABBIT_HOME?

Profiles are a managed layer on top of `RABBIT_HOME`. You *could* manually set `RABBIT_HOME=/some/path` before every command, but profiles handle all the plumbing for you: creating the directory structure, generating shell aliases (`rabbit-work`), tracking the active profile in `~/.rabbit/active_profile`, and syncing skill updates across all profiles automatically. They also integrate with tab completion so you don't have to remember paths.

### Can two profiles share the same bot token?

No. Each messaging platform (Telegram, Discord, etc.) requires exclusive access to a bot token. If two profiles try to use the same token simultaneously, the second gateway will fail to connect. Create a separate bot per profile — for Telegram, talk to [@BotFather](https://t.me/BotFather) to make additional bots.

### Do profiles share memory or sessions?

No. Each profile has its own memory store, session database, and skills directory. They are completely isolated. If you want to start a new profile with existing memories and sessions, use `rabbit profile create newname --clone-all` to copy everything from the current profile, or add `--clone-from <profile>` to copy from a specific source profile.

This isolation is also the reason to never run two agents against the *same* profile or Rabbit home: both write memory automatically and each loads the other's writes at session start, so their stored state degrades with every session. One agent per profile; for genuinely shared memory across agents, use an [external memory provider](../user-guide/features/memory-providers.md).

### What happens when I run `rabbit update`?

`rabbit update` pulls the latest code and reinstalls dependencies **once** (not per-profile). It then syncs updated skills to all profiles automatically. You only need to run `rabbit update` once — it covers every profile on the machine.


### How many profiles can I run?

There is no hard limit. Each profile is a directory under `~/.rabbit/profiles/` that carries at least one identity file (`config.yaml`, `.env`, `SOUL.md`, `profile.yaml`, `auth.json` or `state.db`); a bare directory without one (a leftover from a log rotation or cron tick) is not a profile — it is not listed or served, `-p <name>` reports it as missing, and `rabbit profile create <name>` refuses to overwrite it until you move or remove it. The practical limit depends on your disk space and how many concurrent gateways your system can handle (each gateway is a lightweight Python process). Running dozens of profiles is fine; each idle profile uses no resources.

---

## Workflows & Patterns

### Using different models for different tasks (multi-model workflows)

**Scenario:** You use GPT-5.4 as your daily driver, but Gemini or Grok writes better social media content. Manually switching models every time is tedious.

**Solution: Delegation config.** Rabbit can route subagents to a different model automatically. Set this in `~/.rabbit/config.yaml`:

```yaml
delegation:
  model: "google/gemini-3-flash-preview"   # subagents use this model
  provider: "openrouter"                    # provider for subagents
```

Now when you tell Rabbit "write me a Twitter thread about X" and it spawns a `delegate_task` subagent, that subagent runs on Gemini instead of your main model. Your primary conversation stays on GPT-5.4.

You can also be explicit in your prompt: *"Delegate a task to write social media posts about our product launch. Use your subagent for the actual writing."* The agent will use `delegate_task`, which automatically picks up the delegation config.

For one-off model switches without delegation, use `/model` in the CLI:

```bash
/model google/gemini-3-flash-preview    # switch for this session
# ... write your content ...
/model openai/gpt-5.4                   # switch back
```

:::warning
Each `/model` switch resets the prompt cache — the cache key includes the model, so the first message after every switch re-reads the whole conversation at full input price. On long sessions, prefer delegation (subagents get their own fresh context) or a new session over repeated back-and-forth switching.
:::

See [Subagent Delegation](../user-guide/features/delegation.md) for more on how delegation works.

### Running multiple agents on one WhatsApp number (per-chat binding)

**Scenario:** In OpenClaw, you had multiple independent agents bound to specific WhatsApp chats — one for a family shopping list group, another for your private chat. Can Rabbit do this?

**Current limitation:** Rabbit profiles each require their own WhatsApp number/session. You cannot bind multiple profiles to different chats on the same WhatsApp number — the WhatsApp bridge (Baileys) uses one authenticated session per number.

**Workarounds:**

1. **Use a single profile with personality switching.** Create different `AGENTS.md` context files or use the `/personality` command to change behavior per chat. The agent sees which chat it's in and can adapt.

2. **Use cron jobs for specialized tasks.** For a shopping list tracker, set up a cron job that monitors a specific chat and manages the list — no separate agent needed.

3. **Use separate numbers.** If you need truly independent agents, pair each profile with its own WhatsApp number. Virtual numbers from services like Google Voice work for this.

4. **Use Telegram or Discord instead.** These platforms support per-chat binding more naturally — each Telegram group or Discord channel gets its own session, and you can run multiple bot tokens (one per profile) on the same account.

See [Profiles](../user-guide/profiles.md) and [WhatsApp setup](../user-guide/messaging/whatsapp.md) for more details.

### Controlling what shows up in Telegram (hiding logs and reasoning)

**Scenario:** You see gateway exec logs, Rabbit reasoning, and tool call details in Telegram instead of just the final output.

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

**Scenario:** Telegram has a 100 slash command limit, and your skills are pushing past it. You want to disable skills you don't need on Telegram, but `rabbit skills config` settings don't seem to take effect.

**Solution:** Use `rabbit skills config` to disable skills per-platform. This writes to `config.yaml`:

```yaml
skills:
  disabled: []                    # globally disabled skills
  platform_disabled:
    telegram: [skill-a, skill-b]  # disabled only on telegram
```

After changing this, **restart the gateway** (`rabbit gateway restart` or kill and relaunch). The Telegram bot command menu rebuilds on startup.

:::tip
Skills with very long descriptions are truncated to 40 characters in the Telegram menu to stay within payload size limits. If skills aren't appearing, it may be a total payload size issue rather than the 100 command count limit — disabling unused skills helps with both.
:::

### Shared thread sessions (multiple users, one conversation)

**Scenario:** You have a Telegram or Discord thread where multiple people mention the bot. You want all mentions in that thread to be part of one shared conversation, not separate per-user sessions.

**Current behavior:** Rabbit creates sessions keyed by user ID on most platforms, so each person gets their own conversation context. This is by design for privacy and context isolation.

**Workarounds:**

1. **Use Slack.** Slack sessions are keyed by thread, not by user. Multiple users in the same thread share one conversation — exactly the behavior you're describing. This is the most natural fit.

2. **Use a group chat with a single user.** If one person is the designated "operator" who relays questions, the session stays unified. Others can read along.

3. **Use a Discord channel.** Discord sessions are keyed by channel, so all users in the same channel share context. Use a dedicated channel for the shared conversation.

### Exporting Rabbit to another machine

**Scenario:** You've built up skills, cron jobs, and memories on one machine and want to move everything to a new dedicated Linux box.

**Solution:**

1. Install Rabbit Agent on the new machine:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/seven0070/Rabbit-/main/scripts/install.sh | bash
   ```

2. On the **source machine**, create a full backup:
   ```bash
   rabbit backup
   ```
   This saves a zip archive at `~/rabbit-backup-<timestamp>.zip`.
   The full backup covers configuration, credentials, memories, skills, sessions,
   and profiles under the Rabbit data root. It is not an application or runtime image.

3. Copy the zip to the new machine and import it:
   ```bash
   # On the source machine
   scp ~/rabbit-backup-<timestamp>.zip newmachine:~/

   # On the new machine
   rabbit import ~/rabbit-backup-<timestamp>.zip
   ```

4. On the new machine, run `rabbit setup` to verify API keys and provider config are working.

### Moving a single profile to another machine

**Scenario:** You want to move or share one specific profile — not your full installation.

```bash
# On the source machine
rabbit profile export work ./work-backup.tar.gz

# Copy the file to the target machine, then:
rabbit profile import ./work-backup.tar.gz work
```

The imported profile will have all config, memories, sessions, and skills from the export. You may need to update paths or re-authenticate with providers if the new machine has a different setup.

### `rabbit backup` vs `rabbit profile export` {#rabbit-backup-vs-rabbit-profile-export}

| Feature | `rabbit backup` | `rabbit profile export` |
| :--- | :--- | :--- |
| **Use Case** | **Full machine migration** | **Porting/sharing a specific profile** |
| **Scope** | Rabbit data root, with the exclusions listed below | Single profile directory |
| **Includes** | All profiles, global config, API keys, sessions | Single profile: SOUL.md, memories, sessions, skills |
| **Credentials** | **Included** (`.env` and `auth.json`) | **Excluded** (stripped for safe sharing) |
| **Format** | `.zip` | `.tar.gz` |

The full backup excludes:

- The source checkout, dependency environments, and downloaded tools, models, and runtimes.
- Build caches, checkpoints, previous backups, and quick snapshots.
- Browser profiles, including copies of real-browser credentials.
- Bytecode, SQLite sidecars, `gateway.pid`, `cron.pid`, and `.backup.lock`.

`rabbit backup --quick` saves selected state files instead of a full archive.
It is not a replacement for the full backup before a machine migration.

Full backups report files that fail to copy. An archive can therefore exist
with missing data. Review the skipped-file report before you remove the source installation.
Restored package declarations let PM download dependencies again. Bytecode and
SQLite sidecars regenerate locally. The exclusions do not remove `.env` or
`auth.json` from the full backup.

**Manual fallback (rsync):** If you prefer to copy files directly, exclude the code repo:
```bash
rsync -av --exclude='rabbit-agent' ~/.rabbit/ newmachine:~/.rabbit/
```

:::tip
`rabbit backup` produces a consistent snapshot even while Rabbit is actively running. The restored archive excludes machine-local runtime files like `gateway.pid` and `cron.pid`.
:::

### Permission denied when reloading shell after install

**Scenario:** After running the Rabbit installer, `source ~/.zshrc` gives a permission denied error.

**Cause:** This usually happens when `~/.zshrc` (or `~/.bashrc`) has incorrect file permissions, or when the installer couldn't write to it cleanly. It's not a Rabbit-specific issue — it's a shell config permissions problem.

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
rabbit config show | head -20

# Re-run model selection
rabbit model

# Or test with a known-good model
rabbit chat -q "hello" --model anthropic/claude-opus-4.7
```

If using OpenRouter, make sure your API key has credits. A 400 from OpenRouter often means the model requires a paid plan or the model ID has a typo.

---

## Still Stuck?

If your issue isn't covered here:

1. **Search existing issues:** [GitHub Issues](https://github.com/seven0070/Rabbit-/issues)
2. **File a bug report:** Include your OS, Python version (`python3 --version`), Rabbit version (`rabbit --version`), and the full error message
