---
sidebar_position: 3
title: "常见问题与故障排除"
description: "Hermes Agent 常见问题与解决方案"
---

# 常见问题与故障排除

常见问题和疑虑的快速答案和修复方法。

---

## 常见问题

### Hermes Agent 支持哪些 LLM Provider？

Hermes Agent 可与任何 OpenAI 兼容 API 配合使用。支持以下 Provider：

- **[OpenRouter](https://openrouter.ai/)** — 通过一个 API 密钥访问数百个模型（推荐，灵活度高）
- **Nous Portal** — Nous Research 自己的推理端点
- **OpenAI** — GPT-4o、o1、o3 等
- **Anthropic** — Claude 模型（通过 OpenRouter 或兼容代理）
- **Google** — Gemini 模型（通过 OpenRouter 或兼容代理）
- **z.ai / ZhipuAI** — GLM 模型
- **Kimi / Moonshot AI** — Kimi 模型
- **MiniMax** — 全球和中国端点
- **本地模型** — 通过 [Ollama](https://ollama.com/)、[vLLM](https://docs.vllm.ai/)、[llama.cpp](https://github.com/ggerganov/llama.cpp)、[SGLang](https://github.com/sgl-project/sglang) 或任何 OpenAI 兼容服务器

使用 `hermes model` 设置 provider 或通过编辑 `~/.hermes/.env` 进行配置。请参见 [环境变量](./environment-variables.md) 参考了解所有 provider 密钥。

### 支持 Windows 吗？

**原生不支持。** Hermes Agent 需要类似 Unix 的环境。在 Windows 上，安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) 并在其中运行 Hermes。标准安装命令在 WSL2 中完美运行：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### 我在 WSL2 中运行 Hermes。最好如何控制普通的 Windows Chrome？

优先使用 MCP 桥接而不是 `/browser connect`。

推荐模式：

- 在 WSL2 中运行 Hermes
- 继续使用普通登录的 Windows Chrome
- 通过 `cmd.exe` 或 `powershell.exe` 添加 `chrome-devtools-mcp` 作为 MCP 服务器
- 让 Hermes 使用生成的 MCP 浏览器工具

这比尝试让 Hermes 核心浏览器传输直接跨 WSL2/Windows 边界附加更可靠。

请参见：

- [将 MCP 与 Hermes 结合使用](../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)
- [浏览器自动化](../user-guide/features/browser.md#wsl2--windows-chrome-prefer-mcp-over-browser-connect)

### 支持 Android / Termux 吗？

是的 — Hermes 现在有针对 Android 手机测试过的 Termux 安装路径。

快速安装：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

有关完全明确的手动步骤、支持的可选组件和当前限制，请参见 [Termux 指南](../getting-started/termux.md)。

重要注意事项：`.[all]` 额外组件目前在 Android 上不可用，因为 `voice` 额外组件依赖 `faster-whisper` → `ctranslate2`，而 `ctranslate2` 不发布 Android 轮子。请改用测试过的 `.[termux]` 额外组件。

### 我的数据会发送到任何地方吗？

API 调用**仅发送到你配置的 LLM provider**（如 OpenRouter、你的本地 Ollama 实例）。Hermes Agent 不收集遥测、使用数据或分析。你的对话、记忆和技能存储在本地 `~/.hermes/` 中。

### 可以离线/使用本地模型吗？

可以。运行 `hermes model`，选择**自定义端点**，然后输入你服务器的 URL：

```bash
hermes model
# 选择：Custom endpoint（手动输入 URL）
# API base URL: http://localhost:11434/v1
# API key: ollama
# Model name: qwen3.5:27b
# Context length: 32768   ← 设置为你服务器的实际上下文窗口
```

或直接在 `config.yaml` 中配置：

```yaml
model:
  default: qwen3.5:27b
  provider: custom
  base_url: http://localhost:11434/v1
```

Hermes 将端点、provider 和 base URL 持久化到 `config.yaml`，因此重启后保留。如果你的本地服务器只加载了一个模型，`/model custom` 会自动检测。你也可以在 config.yaml 中设置 `provider: custom` — 它是一等 provider，不是任何东西的别名。

这适用于 Ollama、vLLM、llama.cpp 服务器、SGLang、LocalAI 等。请参见 [配置指南](../user-guide/configuration.md) 了解更多详情。

:::tip Ollama 用户
如果你在 Ollama 中设置了自定义 `num_ctx`（如 `ollama run --num_ctx 16384`），确保在 Hermes 中设置匹配的上下文长度 — Ollama 的 `/api/show` 报告模型的*最大*上下文，不是你配置的*有效* `num_ctx`。
:::

:::tip 本地模型超时
Hermes 自动检测本地端点并放宽流超时（读取超时从 120s 提高到 1800s，过时流检测禁用）。如果仍然在非常大的上下文上遇到超时，在 `.env` 中设置 `HERMES_STREAM_READ_TIMEOUT=1800`。请参见 [本地 LLM 指南](../guides/local-llm-on-mac.md#timeouts) 了解更多。
:::

### 费用是多少？

Hermes Agent 本身**免费且开源**（MIT 许可证）。你只需为所选 provider 的 LLM API 使用付费。本地模型完全免费运行。

### 可以多人使用一个实例吗？

可以。[消息网关](../user-guide/messaging/index.md) 允许多个用户通过 Telegram、Discord、Slack、WhatsApp 或 Home Assistant 与同一个 Hermes Agent 实例交互。访问权限通过允许列表（特定用户 ID）和 DM 配对（第一个发消息的用户声明访问权限）控制。

### 记忆和技能有什么区别？

- **记忆** 存储**事实** — 代理了解你、你的项目和偏好的事物。记忆根据相关性自动检索。
- **技能** 存储**程序** — 如何做事情的分步说明。技能在代理遇到类似任务时被召回。

两者都跨会话持久化。请参见 [记忆](../user-guide/features/memory.md) 和 [技能](../user-guide/features/skills.md) 了解更多详情。

### 可以在我自己的 Python 项目中使用它吗？

可以。导入 `AIAgent` 类并以编程方式使用 Hermes：

```python
from run_agent import AIAgent

agent = AIAgent(model="anthropic/claude-opus-4.7")
response = agent.chat("Explain quantum computing briefly")
```

请参见 [Python 库指南](../user-guide/features/code-execution.md) 了解完整的 API 用法。

---

## 故障排除

### 安装问题

#### 安装后 `hermes: command not found`

**原因：** Shell 未重新加载更新的 PATH。

**解决方案：**
```bash
# 重新加载 shell 配置
source ~/.bashrc    # bash
source ~/.zshrc     # zsh

# 或者启动新的终端会话
```

如果仍然不工作，验证安装位置：
```bash
which hermes
ls ~/.local/bin/hermes
```

:::tip
安装程序将 `~/.local/bin` 添加到你的 PATH。如果你使用非标准 shell 配置，手动添加 `export PATH="$HOME/.local/bin:$PATH"`。
:::

#### Python 版本太旧

**原因：** Hermes 需要 Python 3.11 或更高版本。

**解决方案：**
```bash
python3 --version   # 检查当前版本

# 安装更新的 Python
sudo apt install python3.12   # Ubuntu/Debian
brew install python@3.12      # macOS
```

安装程序自动处理这个 — 如果在手动安装期间看到这个错误，先升级 Python。

#### 终端命令说 `node: command not found`（或 `nvm`、`pyenv`、`asdf`、…）

**原因：** Hermes 在启动时通过运行 `bash -l` 一次生成每个会话环境快照。bash 登录 shell 读取 `/etc/profile`、`~/.bash_profile` 和 `~/.profile`，但**不 source `~/.bashrc`** — 因此在那里安装自己的工具（`nvm`、`asdf`、`pyenv`、`cargo`、自定义 `PATH` 导出）对快照不可见。这最常发生在 Hermes 在 systemd 下运行或在没有预加载交互式 shell 配置的最小 shell 中。

**解决方案：** Hermes 默认自动 source `~/.bashrc`。如果不够 — 例如你是 zsh 用户而你的 PATH 在 `~/.zshrc` 中，或者你从一个独立文件 init `nvm` — 在 `~/.hermes/config.yaml` 中列出要 source 的额外文件：

```yaml
terminal:
  shell_init_files:
    - ~/.zshrc                     # zsh 用户：将 zsh 管理的 PATH 拉入 bash 快照
    - ~/.nvm/nvm.sh                # 直接 nvm init（无论 shell 都有效）
    - /etc/profile.d/cargo.sh      # 系统范围的 rc 文件
  # 设置此列表后，默认的 ~/.bashrc 自动 source 不会添加 —
  # 如果你想要两者，明确包含它：
  #   - ~/.bashrc
  #   - ~/.zshrc
```

缺失的文件会被静默跳过。Source 发生在 bash 中，因此依赖 zsh 专用语法的文件可能会出错 — 如果担心，source 只需 PATH 设置部分（如直接使用 nvm 的 `nvm.sh`）而不是整个 rc 文件。

要禁用自动 source 行为（严格的登录 shell 语义）：

```yaml
terminal:
  auto_source_bashrc: false
```

#### `uv: command not found`

**原因：** `uv` 包管理器未安装或不在 PATH 中。

**解决方案：**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### 安装期间权限被拒绝

**原因：** 写入安装目录的权限不足。

**解决方案：**
```bash
# 不要用 sudo 运行安装程序 — 它安装到 ~/.local/bin
# 如果之前用 sudo 安装过，清理：
sudo rm /usr/local/bin/hermes
# 然后重新运行标准安装程序
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

---

### Provider 和模型问题

#### `/model` 只显示一个 provider / 无法切换 provider

**原因：** `/model`（在聊天会话内）只能在你**已经配置**的 provider 之间切换。如果你只设置了 OpenRouter，`/model` 也只显示 OpenRouter。

**解决方案：** 退出会话并从终端使用 `hermes model` 添加新 provider：

```bash
# 先退出 Hermes 聊天会话（Ctrl+C 或 /quit）

# 运行完整的 provider 设置向导
hermes model

# 这让你可以：添加 provider、运行 OAuth、输入 API 密钥、配置端点
```

通过 `hermes model` 添加新 provider 后，开始新聊天会话 — `/model` 现在会显示你所有已配置的 provider。

:::tip 快速参考
| 想要... | 使用 |
|-----------|-----|
| 添加新 provider | `hermes model`（从终端） |
| 输入/更改 API 密钥 | `hermes model`（从终端） |
| 会话中切换模型 | `/model <名称>`（会话内） |
| 切换到不同的已配置 provider | `/model provider:model`（会话内） |
:::

#### API 密钥不工作

**原因：** 密钥缺失、过期、设置错误或用于错误的 provider。

**解决方案：**
```bash
# 检查配置
hermes config show

# 重新配置你的 provider
hermes model

# 或直接设置
hermes config set OPENROUTER_API_KEY sk-or-v1-xxxxxxxxxxxx
```

:::warning
确保密钥与 provider 匹配。OpenAI 密钥不适用于 OpenRouter，反之亦然。检查 `~/.hermes/.env` 是否有冲突条目。
:::

#### 模型不可用 / 模型未找到

**原因：** 模型标识符不正确或在你的 provider 上不可用。

**解决方案：**
```bash
# 列出你的 provider 的可用模型
hermes model

# 设置有效模型
hermes config set HERMES_MODEL anthropic/claude-opus-4.7

# 或指定每次运行
hermes chat --model openrouter/meta-llama/llama-3.1-70b-instruct
```

#### 速率限制（429 错误）

**原因：** 你已超过 provider 的速率限制。

**解决方案：** 等待片刻重试。对于持续使用，考虑：
- 升级你的 provider 计划
- 切换到不同的模型或 provider
- 使用 `hermes chat --provider <替代>` 路由到不同的后端

#### 超出上下文长度

**原因：** 对话已增长到超过模型的上下文窗口，或者 Hermes 检测到你的模型上下文长度错误。

**解决方案：**
```bash
# 压缩当前会话
/compress

# 或者开始新会话
hermes chat

# 使用更大上下文窗口的模型
hermes chat --model openrouter/google/gemini-3-flash-preview
```

如果这发生在第一次长对话上，Hermes 可能对你的模型使用了错误的上下文长度。检查它检测到的值：

查看 CLI 启动行 — 它显示检测到的上下文长度（如 `📊 Context limit: 128000 tokens`）。你也可以在会话中使用 `/usage` 检查。

要修复上下文检测，显式设置：

```yaml
# 在 ~/.hermes/config.yaml 中
model:
  default: your-model-name
  context_length: 131072  # 你的模型的实际上下文窗口
```

或者对于自定义端点，按模型添加：

```yaml
custom_providers:
  - name: "My Server"
    base_url: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 32768
```

请参见 [上下文长度检测](../integrations/providers.md#context-length-detection) 了解自动检测如何工作以及所有覆盖选项。

---

### 终端问题

#### 命令被阻止为危险

**原因：** Hermes 检测到潜在破坏性命令（如 `rm -rf`、`DROP TABLE`）。这是一个安全功能。

**解决方案：** 出现提示时，查看命令并输入 `y` 批准。你也可以：
- 让代理使用更安全的替代方案
- 在 [安全文档](../user-guide/security.md) 中查看危险模式的完整列表

:::tip
这是按设计工作的 — Hermes 永远不会静默运行破坏性命令。批准提示向你准确显示将要执行的内容。
:::

#### 通过消息网关 `sudo` 不工作

**原因：** 消息网关在没有交互式终端的情况下运行，所以 `sudo` 无法提示输入密码。

**解决方案：**
- 避免在消息中使用 `sudo` — 让代理找到替代方案
- 如果必须使用 `sudo`，在 `/etc/sudoers` 中为特定命令配置无密码 sudo
- 或者切换到终端界面进行管理任务：`hermes chat`

#### Docker 后端无法连接

**原因：** Docker 守护进程未运行或用户缺少权限。

**解决方案：**
```bash
# 检查 Docker 是否运行
docker info

# 将用户添加到 docker 组
sudo usermod -aG docker $USER
newgrp docker

# 验证
docker run hello-world
```

---

### 消息问题

#### Bot 不响应消息

**原因：** Bot 未运行、未授权，或你的用户不在允许列表中。

**解决方案：**
```bash
# 检查网关是否运行
hermes gateway status

# 启动网关
hermes gateway start

# 检查日志中的错误
cat ~/.hermes/logs/gateway.log | tail -50
```

#### 消息未送达

**原因：** 网络问题、bot 令牌过期或平台 webhook 配置错误。

**解决方案：**
- 用 `hermes gateway setup` 验证你的 bot 令牌有效
- 检查网关日志：`cat ~/.hermes/logs/gateway.log | tail -50`
- 对于基于 webhook 的平台（Slack、WhatsApp），确保你的服务器可公开访问

#### 允许列表混乱 — 谁可以与 bot 交谈？

**原因：** 授权模式决定谁获得访问权限。

**解决方案：**

| 模式 | 工作原理 |
|------|-------------|
| **允许列表** | 只有配置中列出的用户 ID 可以交互 |
| **DM 配对** | 第一个在 DM 中发消息的用户声明独占访问权限 |
| **开放** | 任何人都可以交互（不推荐用于生产） |

在 `~/.hermes/config.yaml` 中的网关设置下配置。请参见 [消息文档](../user-guide/messaging/index.md)。

#### 网关无法启动

**原因：** 缺少依赖、端口冲突或令牌配置错误。

**解决方案：**
```bash
# 安装核心消息网关依赖
pip install "hermes-agent[messaging]"  # Telegram、Discord、Slack 和共享网关依赖

# 检查端口冲突
lsof -i :8080

# 验证配置
hermes config show
```

#### WSL：网关持续断开连接或 `hermes gateway start` 失败

**原因：** WSL 的 systemd 支持不可靠。许多 WSL2 安装没有启用 systemd，即使启用，服务也可能无法在 WSL 重启或 Windows 空闲关闭中存活。

**解决方案：** 使用前台模式而不是 systemd 服务：

```bash
# 选项 1：直接前台（最简单）
hermes gateway run

# 选项 2：通过 tmux 持久化（终端关闭后继续）
tmux new -s hermes 'hermes gateway run'
# 稍后重新附加：tmux attach -t hermes

# 选项 3：通过 nohup 后台运行
nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &
```

如果你仍然想尝试 systemd，确保已启用：

1. 打开 `/etc/wsl.conf`（如果不存在则创建）
2. 添加：
   ```ini
   [boot]
   systemd=true
   ```
3. 从 PowerShell：`wsl --shutdown`
4. 重新打开 WSL 终端
5. 验证：`systemctl is-system-running` 应显示 "running" 或 "degraded"

:::tip 启动时自动运行
对于可靠的自动启动，使用 Windows 任务计划程序在登录时启动 WSL + 网关：
1. 创建任务运行 `wsl -d Ubuntu -- bash -lc 'hermes gateway run'`
2. 设置它在用户登录时触发
:::

#### macOS：Node.js / ffmpeg / 其他工具网关找不到

**原因：** launchd 服务继承的 PATH 最小（`/usr/bin:/bin:/usr/sbin:/sbin`），不包括 Homebrew、nvm、cargo 或其他用户安装的工具目录。这通常会破坏 WhatsApp 桥接（`node not found`）或语音转录（`ffmpeg not found`）。

**解决方案：** 网关在你运行 `hermes gateway install` 时捕获你的 shell PATH。如果你在设置网关后安装了工具，重新运行安装以捕获更新的 PATH：

```bash
hermes gateway install    # 重新快照你当前的 PATH
hermes gateway start      # 检测到更新的 plist 并重新加载
```

你可以验证 plist 具有正确的 PATH：
```bash
/usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:PATH" \
  ~/Library/LaunchAgents/ai.hermes.gateway.plist
```

---

### 性能问题

#### 响应慢

**原因：** 大模型、远程 API 服务器或带有大量工具的繁重系统提示。

**解决方案：**
- 尝试更快的模型：`hermes chat --model openrouter/meta-llama/llama-3.1-8b-instruct`
- 减少活动工具集：`hermes chat -t "terminal"`
- 检查你到 provider 的网络延迟
- 对于本地模型，确保你有足够的 GPU VRAM

#### Token 使用量高

**原因：** 长对话、冗长的系统提示或累积上下文的大量工具调用。

**解决方案：**
```bash
# 压缩对话以减少 token
/compress

# 检查会话 token 使用量
/usage
```

:::tip
在长会话中定期使用 `/compress`。它总结对话历史并在保留上下文的同时显著减少 token 使用量。
:::

#### 会话变得太长

**原因：** 扩展对话累积消息和工具输出，接近上下文限制。

**解决方案：**
```bash
# 压缩当前会话（保留关键上下文）
/compress

# 开始新会话并引用旧会话
hermes chat

# 稍后如需恢复特定会话
hermes chat --continue
```

---

### MCP 问题

#### MCP 服务器无法连接

**原因：** 服务器二进制文件未找到、命令路径错误或缺少运行时。

**解决方案：**
```bash
# 确保 MCP 依赖已安装（已包含在标准安装中）
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

# 对于基于 npm 的服务器，确保 Node.js 可用
node --version
npx --version

# 手动测试服务器
npx -y @modelcontextprotocol/server-filesystem /tmp
```

验证你的 `~/.hermes/config.yaml` MCP 配置：
```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
```

#### MCP 服务器的工具未显示

**原因：** 服务器启动但工具发现失败、工具被配置过滤，或者服务器不支持你期望的 MCP 能力。

**解决方案：**
- 检查网关/代理日志中的 MCP 连接错误
- 确保服务器响应 `tools/list` RPC 方法
- 检查该服务器下任何 `tools.include`、`tools.exclude`、`tools.resources`、`tools.prompts` 或 `enabled` 设置
- 记住资源/提示实用工具只在会话实际支持这些能力时才注册
- 配置更改后使用 `/reload-mcp`

```bash
# 验证 MCP 服务器已配置
hermes config show | grep -A 12 mcp_servers

# 配置更改后重启 Hermes 或重新加载 MCP
hermes chat
```

另请参见：
- [MCP（Model Context Protocol）](/docs/user-guide/features/mcp)
- [将 MCP 与 Hermes 结合使用](/docs/guides/use-mcp-with-hermes)
- [MCP 配置参考](/docs/reference/mcp-config-reference)

#### MCP 超时错误

**原因：** MCP 服务器响应时间过长，或者在执行期间崩溃。

**解决方案：**
- 如果支持，在 MCP 服务器配置中增加超时
- 检查 MCP 服务器进程是否仍在运行
- 对于远程 HTTP MCP 服务器，检查网络连接

:::warning
如果 MCP 服务器在请求中途崩溃，Hermes 会报告超时。检查服务器的日志（不仅仅是 Hermes 日志）来诊断根本原因。
:::

---

## Profile

### Profile 与仅设置 HERMES_HOME 有什么区别？

Profile 是 `HERMES_HOME` 之上的托管层。你*可以*在每个命令前手动设置 `HERMES_HOME=/some/path`，但 profile 为你处理所有管道：创建目录结构、生成 shell 别名（`hermes-work`）、在 `~/.hermes/active_profile` 中跟踪活动 profile，以及跨所有 profile 自动同步技能更新。它们还与 tab 补全集成，这样你就不必记住路径。

### 两个 profile 可以共享同一个 bot 令牌吗？

不能。每个消息平台（Telegram、Discord 等）需要独占访问 bot 令牌。如果两个 profile 尝试同时使用同一令牌，第二个网关将无法连接。为每个 profile 创建单独的 bot — 对于 Telegram，请联系 [@BotFather](https://t.me/BotFather) 创建额外的 bot。

### Profile 共享记忆或会话吗？

不能。每个 profile 有自己的记忆存储、会话数据库和技能目录。它们完全隔离。如果你想用现有记忆和会话启动新 profile，使用 `hermes profile create newname --clone-all` 从当前 profile 复制所有内容。

### 运行 `hermes update` 时会发生什么？

`hermes update` 拉取最新代码并重新安装依赖**一次**（不是每个 profile 一次）。然后它自动将更新的技能同步到所有 profile。你只需要运行 `hermes update` 一次 — 它涵盖机器上的每个 profile。

### 我可以运行多少个 profile？

没有硬性限制。每个 profile 只是 `~/.hermes/profiles/` 下的一个目录。实际限制取决于你的磁盘空间和系统可以处理多少并发网关（每个网关是一个轻量级 Python 进程）。运行数十个 profile 没问题；每个空闲 profile 不使用资源。

---

## 工作流和模式

### 使用不同模型处理不同任务（多模型工作流）

**场景：** 你使用 GPT-5.4 作为日常驱动，但 Gemini 或 Grok 写更好的社交媒体内容。每次手动切换模型很繁琐。

**解决方案：委托配置。** Hermes 可以自动将子代理路由到不同模型。在 `~/.hermes/config.yaml` 中设置：

```yaml
delegation:
  model: "google/gemini-3-flash-preview"   # 子代理使用此模型
  provider: "openrouter"                    # 子代理的 provider
```

现在当你告诉 Hermes"写一个关于 X 的 Twitter 线程"而它生成 `delegate_task` 子代理时，那个子代理使用 Gemini 而不是你的主模型。你的主要对话保持在 GPT-5.4 上。

你也可以在提示中明确：`"将写社交媒体帖子的任务委托给我们的产品发布。使用你的子代理进行实际写作。"` 代理将使用 `delegate_task`，它自动拾取委托配置。

对于一次性模型切换而不是委托，使用 CLI 中的 `/model`：

```bash
/model google/gemini-3-flash-preview    # 切换此次会话
# ... 写你的内容 ...
/model openai/gpt-5.4                   # 切换回来
```

请参见 [子代理委托](../user-guide/features/delegation.md) 了解更多关于委托如何工作。

### 在一个 WhatsApp 号码上运行多个代理（按聊天绑定）

**场景：** 在 OpenClaw 中，你有绑定到特定 WhatsApp 聊天的多个独立代理 — 一个用于家庭购物清单群组，另一个用于你的私人聊天。Hermes 可以吗？

**当前限制：** Hermes profile 每个需要自己的 WhatsApp 号码/会话。你不能将多个 profile 绑定到同一 WhatsApp 号码的不同聊天 — WhatsApp 桥接（Baileys）对每个号码使用一个认证会话。

**解决方法：**

1. **使用具有人格切换的单一 profile。** 创建不同的 `AGENTS.md` 上下文文件或使用 `/personality` 命令改变行为。代理看到它在哪个聊天中，可以适应。

2. **使用 cron 作业进行专门任务。** 对于购物清单追踪器，设置 cron 作业监控特定聊天并管理列表 — 不需要单独的代理。

3. **使用独立号码。** 如果你确实需要独立代理，为每个 profile 配对自己的 WhatsApp 号码。来自 Google Voice 等服务的虚拟号码可以用于此目的。

4. **改用 Telegram 或 Discord。** 这些平台更自然地支持按聊天绑定 — 每个 Telegram 群组或 Discord 频道获得自己的会话，你可以在一 个账户上运行多个 bot 令牌（每个 profile 一个）。

请参见 [Profile](../user-guide/profiles.md) 和 [WhatsApp 设置](../user-guide/messaging/whatsapp.md) 了解更多详情。

### 控制 Telegram 上显示的内容（隐藏日志和推理）

**场景：** 你在 Telegram 中看到网关 exec 日志、Hermes 推理和工具调用详情，而不是只有最终输出。

**解决方案：** `config.yaml` 中的 `display.tool_progress` 设置控制显示多少工具活动：

```yaml
display:
  tool_progress: "off"   # 选项：off、new、all、verbose
```

- **`off`** — 只有最终响应。无工具调用、无推理、无日志。
- **`new`** — 在发生时显示新工具调用（简短的摘要行）。
- **`all`** — 显示所有工具活动包括结果。
- **`verbose`** — 完整详情包括工具参数和输出。

对于消息平台，通常你想要 `off` 或 `new`。编辑 `config.yaml` 后，重启网关以使更改生效。

你也可以用 `/verbose` 命令按会话切换（如果启用）：

```yaml
display:
  tool_progress_command: true   # 在网关上启用 /verbose
```

### 在 Telegram 上管理技能（斜杠命令限制）

**场景：** Telegram 有 100 个斜杠命令限制，你的技能正在超过它。你想禁用在 Telegram 上不需要的技能，但 `hermes skills config` 设置似乎不生效。

**解决方案：** 使用 `hermes skills config` 按平台禁用技能。这写入 `config.yaml`：

```yaml
skills:
  disabled: []                    # 全局禁用的技能
  platform_disabled:
    telegram: [skill-a, skill-b]  # 仅在 telegram 上禁用
```

更改后，**重启网关**（`hermes gateway restart` 或终止并重新启动）。Telegram bot 命令菜单在启动时重建。

:::tip
带有非常长描述的技能在 Telegram 菜单中被截断到 40 个字符以保持有效载荷大小限制内。如果技能没有出现，可能是总有效载荷大小问题而不是 100 命令计数限制 — 禁用未使用的技能有助于两者。
:::

### 共享线程会话（多个用户，一个对话）

**场景：** 你有一个 Telegram 或 Discord 线程，多个人提到 bot。你希望该线程中的所有 mention 是一个共享对话的一部分，而不是单独的用户会话。

**当前行为：** Hermes 在大多数平台上按用户 ID 创建会话，所以每个人获得自己的对话上下文。这是出于隐私和上下文隔离的设计。

**解决方法：**

1. **使用 Slack。** Slack 会话按线程键控，而不是按用户。同一线程中的多个用户共享一个对话 — 这正是你描述的行为。这是最自然的选择。

2. **使用只有一个用户的群聊。** 如果一个人是被指定的"操作员"转达问题，对话保持统一。其他人可以阅读。

3. **使用 Discord 频道。** Discord 会话按频道键控，所以同一频道中的所有用户共享上下文。使用专用频道进行共享对话。

### 将 Hermes 导出到另一台机器

**场景：** 你在一台机器上建立了技能、cron 作业和记忆，想将所有内容移到新的专用 Linux 机器上。

**解决方案：**

1. 在新机器上安装 Hermes Agent：
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
   ```

2. 在**源机器**上，创建完整备份：
   ```bash
   hermes backup
   ```
   这创建你整个 `~/.hermes/` 目录的 zip — 配置、API 密钥、记忆、技能、会话和 profile — 保存到你的主目录作为 `~/hermes-backup-<时间戳>.zip`。

3. 将 zip 复制到新机器并导入：
   ```bash
   # 在源机器上
   scp ~/hermes-backup-<时间戳>.zip newmachine:~/

   # 在新机器上
   hermes import ~/hermes-backup-<时间戳>.zip
   ```

4. 在新机器上运行 `hermes setup` 以验证 API 密钥和 provider 配置正常工作。

### 将单个 profile 移到另一台机器

**场景：** 你想移动或共享一个特定的 profile — 不是你的整个安装。

```bash
# 在源机器上
hermes profile export work ./work-backup.tar.gz

# 将文件复制到目标机器，然后：
hermes profile import ./work-backup.tar.gz work
```

导入的 profile 将具有导出中的所有配置、记忆、会话和技能。如果新机器设置不同，你可能需要更新路径或重新认证 provider。

### `hermes backup` vs `hermes profile export`

| 特性 | `hermes backup` | `hermes profile export` |
| :--- | :--- | :--- |
| **用例** | **完整机器迁移** | **移植/共享特定 profile** |
| **范围** | 全局（整个 `~/.hermes` 目录） | 本地（单个 profile 目录） |
| **包括** | 所有 profile、全局配置、API 密钥、会话 | 单个 profile：SOUL.md、记忆、会话、技能 |
| **凭证** | **包括**（`.env` 和 `auth.json`） | **排除**（为安全共享剥离） |
| **格式** | `.zip` | `.tar.gz` |

**手动后备（rsync）：** 如果你更喜欢直接复制文件，排除代码仓库：
```bash
rsync -av --exclude='hermes-agent' ~/.hermes/ newmachine:~/.hermes/
```

:::tip
`hermes backup` 即使在 Hermes 积极运行时也能产生一致的快照。恢复的存档排除机器本地的运行时文件如 `gateway.pid` 和 `cron.pid`。
:::

### 安装后重新加载 shell 时权限被拒绝

**场景：** 运行 Hermes 安装程序后，`source ~/.zshrc` 给出权限被拒绝错误。

**原因：** 这通常发生在 `~/.zshrc`（或 `~/.bashrc`）有错误的文件权限时，或者安装程序无法干净地写入它。这不是 Hermes 特定的问题 — 是 shell 配置权限问题。

**解决方案：**
```bash
# 检查权限
ls -la ~/.zshrc

# 如需要修复（应为 -rw-r--r-- 或 644）
chmod 644 ~/.zshrc

# 然后重新加载
source ~/.zshrc

# 或者只需打开新的终端窗口 — 它自动拾取 PATH 更改
```

如果安装程序添加了 PATH 行但权限错误，你可以手动添加：
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

### 首次代理运行时出现错误 400

**场景：** 设置正常完成，但第一次聊天尝试失败并显示 HTTP 400。

**原因：** 通常是模型名称不匹配 — 配置的模型在你的 provider 上不存在，或者 API 密钥无权访问它。

**解决方案：**
```bash
# 检查配置了哪些模型和 provider
hermes config show | head -20

# 重新运行模型选择
hermes model

# 或用已知良好的模型测试
hermes chat -q "hello" --model anthropic/claude-opus-4.7
```

如果使用 OpenRouter，确保你的 API 密钥有积分。OpenRouter 的 400 通常意味着模型需要付费计划或模型 ID 有拼写错误。

---

## 仍然卡住？

如果你的问题不在这里：

1. **搜索现有 issue：**[GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
2. **向社区提问：**[Nous Research Discord](https://discord.gg/nousresearch)
3. **提交 bug 报告：** 包含你的 OS、Python 版本（`python3 --version`）、Hermes 版本（`hermes --version`）和完整错误消息
