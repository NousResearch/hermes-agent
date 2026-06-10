---
sidebar_position: 3
title: "常见问题与故障排查"
description: "Hermes Agent 常见问题解答及常见问题解决方案"
---

# 常见问题与故障排查

针对最常见问题的快速解答与修复方法。

---

## 常见问题

### Hermes 支持哪些 LLM 提供商？

Hermes Agent 可与任何兼容 OpenAI 的 API 配合使用。支持的提供商包括：

- **[OpenRouter](https://openrouter.ai/)** — 通过一个 API key 访问数百个模型（推荐，灵活性强）
- **Nous Portal** — Nous Research 自有推理端点
- **OpenAI** — GPT-5.4、GPT-5-codex、GPT-4.1、GPT-4o 等
- **Anthropic** — Claude 模型（直接 API、通过 `hermes auth add anthropic` 进行 OAuth、OpenRouter 或任何兼容代理）
- **Google** — Gemini 模型（通过 `gemini` 提供商直接调用 API、`google-gemini-cli` OAuth 提供商、OpenRouter 或兼容代理）
- **z.ai / ZhipuAI** — GLM 模型
- **Kimi / Moonshot AI** — Kimi 模型
- **MiniMax** — 全球及中国区端点
- **本地模型** — 通过 [Ollama](https://ollama.com/)、[vLLM](https://docs.vllm.ai/)、[llama.cpp](https://github.com/ggerganov/llama.cpp)、[SGLang](https://github.com/sgl-project/sglang) 或任何兼容 OpenAI 的服务器

使用 `hermes model` 设置提供商，或直接编辑 `~/.hermes/.env`。所有提供商 key 请参阅[环境变量](./environment-variables.md)参考文档。

### 支持 Windows 吗？

**原生不支持。** Hermes Agent 需要类 Unix 环境。在 Windows 上，请安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) 并在其中运行 Hermes。标准安装命令在 WSL2 中可完美运行：

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### 我在 WSL2 中运行 Hermes，如何控制 Windows 上的普通 Chrome？

推荐使用 MCP bridge（桥接），而非 `/browser connect`。

推荐方案：

- 在 WSL2 内运行 Hermes
- 继续使用 Windows 上已登录的普通 Chrome
- 通过 `cmd.exe` 或 `powershell.exe` 将 `chrome-devtools-mcp` 添加为 MCP 服务器
- 让 Hermes 使用生成的 MCP 浏览器工具

这比强制 Hermes 核心浏览器传输直接跨越 WSL2/Windows 边界进行附加更为可靠。

参见：

- [在 Hermes 中使用 MCP](../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)
- [浏览器自动化](../user-guide/features/browser.md#wsl2--windows-chrome-prefer-mcp-over-browser-connect)

### 支持 Android / Termux 吗？

支持 — Hermes 现已为 Android 手机提供经过测试的 Termux 安装路径。

快速安装：

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

完整的手动步骤、支持的扩展及当前限制，请参阅 [Termux 指南](../getting-started/termux.md)。

重要说明：完整的 `.[all]` 扩展目前在 Android 上不可用，因为 `voice` 扩展依赖 `faster-whisper` → `ctranslate2`，而 `ctranslate2` 未发布 Android wheel 包。请改用经过测试的 `.[termux]` 扩展。

### 我的数据会被发送到哪里？

API 调用**仅发送至您配置的 LLM 提供商**（例如 OpenRouter、您本地的 Ollama 实例）。Hermes Agent 不收集遥测数据、使用数据或分析数据。您的对话、记忆和技能均存储在本地 `~/.hermes/` 目录中。

### 可以离线使用 / 使用本地模型吗？

可以。运行 `hermes model`，选择**自定义端点**，然后输入您服务器的 URL：

```bash
hermes model
# 选择：Custom endpoint（手动输入 URL）
# API base URL: http://localhost:11434/v1
# API key: ollama
# Model name: qwen3.5:27b
# Context length: 32768   ← 设置为与您服务器实际上下文窗口匹配的值
```

或直接在 `config.yaml` 中配置：

```yaml
model:
  default: qwen3.5:27b
  provider: custom
  base_url: http://localhost:11434/v1
```

Hermes 会将端点、提供商和 base URL 持久化到 `config.yaml`，重启后仍然有效。如果您的本地服务器只加载了一个模型，`/model custom` 会自动检测到它。您也可以在 config.yaml 中设置 `provider: custom` — 这是一个一等提供商，不是其他任何东西的别名。

此方式适用于 Ollama、vLLM、llama.cpp server、SGLang、LocalAI 等。详情请参阅[配置指南](../user-guide/configuration.md)。

:::tip Ollama 用户
如果您在 Ollama 中设置了自定义 `num_ctx`（例如 `ollama run --num_ctx 16384`），请确保在 Hermes 中设置匹配的上下文长度 — Ollama 的 `/api/show` 报告的是模型的*最大*上下文，而非您配置的实际 `num_ctx`。
:::

:::tip 本地模型超时问题
Hermes 会自动检测本地端点并放宽流式传输超时（读取超时从 120s 提升至 1800s，禁用停滞流检测）。如果在非常大的上下文下仍然超时，请在 `.env` 中设置 `HERMES_STREAM_READ_TIMEOUT=1800`。详情请参阅[本地 LLM 指南](../guides/local-llm-on-mac.md#timeouts)。
:::

### 费用是多少？

Hermes Agent 本身**免费且开源**（MIT 许可证）。您只需为所选提供商的 LLM API 用量付费。本地模型完全免费运行。

### 多人可以使用同一个实例吗？

可以。[消息网关](../user-guide/messaging/index.md)允许多个用户通过 Telegram、Discord、Slack、WhatsApp 或 Home Assistant 与同一个 Hermes Agent 实例交互。访问权限通过白名单（特定用户 ID）和私信配对（第一个发消息的用户获得访问权）来控制。

### 记忆（memory）和技能（skills）有什么区别？

- **记忆**存储**事实** — 智能体了解的关于您、您的项目和偏好的信息。记忆根据相关性自动检索。
- **技能**存储**流程** — 如何完成某件事的分步说明。当智能体遇到类似任务时会调用技能。

两者均跨会话持久化。详情请参阅[记忆](../user-guide/features/memory.md)和[技能](../user-guide/features/skills.md)。

### 可以在我自己的 Python 项目中使用吗？

可以。导入 `AIAgent` 类，以编程方式使用 Hermes：

```python
from run_agent import AIAgent

agent = AIAgent(model="anthropic/claude-opus-4.7")
response = agent.chat("Explain quantum computing briefly")
```

完整 API 用法请参阅 [Python 库指南](../user-guide/features/code-execution.md)。

---

## 故障排查

### 安装问题

#### 安装后出现 `hermes: command not found`

**原因：** Shell 未重新加载更新后的 PATH。

**解决方案：**
```bash
# 重新加载 shell 配置文件
source ~/.bashrc    # bash
source ~/.zshrc     # zsh

# 或开启一个新的终端会话
```

如果仍然无效，请验证安装位置：
```bash
which hermes
ls ~/.local/bin/hermes
```

:::tip
安装程序会将 `~/.local/bin` 添加到您的 PATH。如果您使用非标准 shell 配置，请手动添加 `export PATH="$HOME/.local/bin:$PATH"`。
:::

#### Python 版本过旧

**原因：** Hermes 需要 Python 3.11 或更新版本。

**解决方案：**
```bash
python3 --version   # 检查当前版本

# 安装更新的 Python
sudo apt install python3.12   # Ubuntu/Debian
brew install python@3.12      # macOS
```

安装程序会自动处理此问题 — 如果在手动安装时看到此错误，请先升级 Python。

#### 终端命令提示 `node: command not found`（或 `nvm`、`pyenv`、`asdf` 等）

**原因：** Hermes 在启动时通过运行一次 `bash -l` 构建每个会话的环境快照。bash 登录 shell 会读取 `/etc/profile`、`~/.bash_profile` 和 `~/.profile`，但**不会 source `~/.bashrc`** — 因此在 `~/.bashrc` 中安装自身的工具（`nvm`、`asdf`、`pyenv`、`cargo`、自定义 `PATH` 导出）对快照不可见。当 Hermes 在 systemd 下运行或在未预加载交互式 shell 配置的最小 shell 中运行时，此问题最为常见。

**解决方案：** Hermes 默认自动 source `~/.bashrc`。如果这还不够 — 例如您是 zsh 用户，PATH 在 `~/.zshrc` 中，或者您从独立文件初始化 `nvm` — 请在 `~/.hermes/config.yaml` 中列出需要额外 source 的文件：

```yaml
terminal:
  shell_init_files:
    - ~/.zshrc                     # zsh 用户：将 zsh 管理的 PATH 引入 bash 快照
    - ~/.nvm/nvm.sh                # 直接初始化 nvm（不依赖 shell 类型）
    - /etc/profile.d/cargo.sh      # 系统级 rc 文件
  # 设置此列表后，默认的 ~/.bashrc 自动 source 不会被添加 —
  # 如需同时保留，请显式包含：
  #   - ~/.bashrc
  #   - ~/.zshrc
```

缺失的文件会被静默跳过。source 在 bash 中执行，因此依赖 zsh 专有语法的文件可能报错 — 如有顾虑，建议只 source PATH 设置部分（例如直接 source nvm 的 `nvm.sh`），而非整个 rc 文件。

如需禁用自动 source 行为（仅使用严格的登录 shell 语义）：

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

#### 安装时出现权限拒绝错误

**原因：** 对安装目录的写入权限不足。

**解决方案：**
```bash
# 不要对安装程序使用 sudo — 它安装到 ~/.local/bin
# 如果之前使用 sudo 安装，请先清理：
sudo rm /usr/local/bin/hermes
# 然后重新运行标准安装程序
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

---

### 提供商与模型问题

#### `/model` 只显示一个提供商 / 无法切换提供商

**原因：** 会话内的 `/model` 只能在您**已配置**的提供商之间切换。如果您只设置了 OpenRouter，`/model` 就只会显示 OpenRouter。

**解决方案：** 退出当前会话，在终端中使用 `hermes model` 添加新提供商：

```bash
# 先退出 Hermes 聊天会话（Ctrl+C 或 /quit）

# 运行完整的提供商设置向导
hermes model

# 此命令可以：添加提供商、运行 OAuth、输入 API key、配置端点
```

通过 `hermes model` 添加新提供商后，启动新的聊天会话 — `/model` 将显示所有已配置的提供商。

:::tip 快速参考
| 目标 | 使用方式 |
|-----------|-----|
| 添加新提供商 | `hermes model`（从终端） |
| 输入/更改 API key | `hermes model`（从终端） |
| 会话中途切换模型 | `/model <name>`（会话内） |
| 切换到其他已配置的提供商 | `/model provider:model`（会话内） |
:::

#### API key 不起作用

**原因：** key 缺失、已过期、设置错误或属于错误的提供商。

**解决方案：**
```bash
# 检查您的配置
hermes config show

# 重新配置您的提供商
hermes model

# 或直接设置
hermes config set OPENROUTER_API_KEY sk-or-v1-xxxxxxxxxxxx
```

:::warning
请确保 key 与提供商匹配。OpenAI 的 key 无法用于 OpenRouter，反之亦然。检查 `~/.hermes/.env` 中是否有冲突条目。
:::

#### 模型不可用 / 找不到模型

**原因：** 模型标识符不正确，或该模型在您的提供商上不可用。

**解决方案：**
```bash
# 列出您的提供商可用的模型
hermes model

# 设置有效的模型
hermes config set HERMES_MODEL anthropic/claude-opus-4.7

# 或按会话指定
hermes chat --model openrouter/meta-llama/llama-3.1-70b-instruct
```

#### 速率限制（429 错误）

**原因：** 您已超出提供商的速率限制。

**解决方案：** 稍等片刻后重试。对于持续使用，请考虑：
- 升级您的提供商套餐
- 切换到其他模型或提供商
- 使用 `hermes chat --provider <alternative>` 路由到其他后端

#### 上下文长度超限

**原因：** 对话内容超出模型的上下文窗口，或 Hermes 检测到的模型上下文长度有误。

**解决方案：**
```bash
# 压缩当前会话
/compress

# 或开始新会话
hermes chat

# 使用上下文窗口更大的模型
hermes chat --model openrouter/google/gemini-3-flash-preview
```

如果在第一次长对话时就出现此问题，Hermes 可能检测到了错误的模型上下文长度。检查检测结果：

查看 CLI 启动行 — 它会显示检测到的上下文长度（例如 `📊 Context limit: 128000 tokens`）。您也可以在会话中使用 `/usage` 查看。

如需修正上下文检测，请显式设置：

```yaml
# 在 ~/.hermes/config.yaml 中
model:
  default: your-model-name
  context_length: 131072  # 您模型的实际上下文窗口
```

或对于自定义端点，按模型添加：

```yaml
custom_providers:
  - name: "My Server"
    base_url: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 32768
```

有关自动检测的工作原理及所有覆盖选项，请参阅[上下文长度检测](../integrations/providers.md#context-length-detection)。

---

### 终端问题

#### 命令被标记为危险而阻止

**原因：** Hermes 检测到潜在的破坏性命令（例如 `rm -rf`、`DROP TABLE`）。这是一项安全功能。

**解决方案：** 出现提示时，检查命令并输入 `y` 批准执行。您也可以：
- 要求智能体使用更安全的替代方案
- 在[安全文档](../user-guide/security.md)中查看完整的危险模式列表

:::tip
这是预期行为 — Hermes 绝不会静默执行破坏性命令。审批提示会向您显示将要执行的确切内容。
:::

#### 通过消息网关时 `sudo` 不起作用

**原因：** 消息网关在没有交互式终端的情况下运行，因此 `sudo` 无法提示输入密码。

**解决方案：**
- 在消息中避免使用 `sudo` — 请智能体寻找替代方案
- 如果必须使用 `sudo`，在 `/etc/sudoers` 中为特定命令配置免密 sudo
- 或切换到终端界面执行管理任务：`hermes chat`

#### Docker 后端无法连接

**原因：** Docker 守护进程未运行，或用户缺少相应权限。

**解决方案：**
```bash
# 检查 Docker 是否在运行
docker info

# 将您的用户添加到 docker 组
sudo usermod -aG docker $USER
newgrp docker

# 验证
docker run hello-world
```

---

### 消息问题

#### Bot 不响应消息

**原因：** Bot 未运行、未授权，或您的用户不在白名单中。

**解决方案：**
```bash
# 检查网关是否在运行
hermes gateway status

# 启动网关
hermes gateway start

# 查看错误日志
cat ~/.hermes/logs/gateway.log | tail -50
```

#### 消息未送达

**原因：** 网络问题、bot token 已过期，或平台 webhook 配置错误。

**解决方案：**
- 使用 `hermes gateway setup` 验证您的 bot token 是否有效
- 检查网关日志：`cat ~/.hermes/logs/gateway.log | tail -50`
- 对于基于 webhook 的平台（Slack、WhatsApp），确保您的服务器可公开访问

#### 白名单混淆 — 谁可以与 bot 交互？

**原因：** 授权模式决定谁可以获得访问权限。

**解决方案：**

| 模式 | 工作方式 |
|------|-------------|
| **白名单** | 只有配置中列出的用户 ID 可以交互 |
| **私信配对** | 第一个在私信中发消息的用户获得独占访问权 |
| **开放** | 任何人都可以交互（不建议用于生产环境） |

在 `~/.hermes/config.yaml` 中您的网关设置下进行配置。请参阅[消息文档](../user-guide/messaging/index.md)。

#### 网关无法启动

**原因：** 缺少依赖项、端口冲突或 token 配置错误。

**解决方案：**
```bash
# 安装核心消息网关依赖项
pip install "hermes-agent[messaging]"  # Telegram、Discord、Slack 及共享网关依赖

# 检查端口冲突
lsof -i :8080

# 验证配置
hermes config show
```

#### WSL：网关持续断开连接或 `hermes gateway start` 失败

**原因：** WSL 的 systemd 支持不稳定。许多 WSL2 安装未启用 systemd，即使启用，服务也可能在 WSL 重启或 Windows 空闲关机后无法存活。

**解决方案：** 使用前台模式代替 systemd 服务：

```bash
# 方案一：直接前台运行（最简单）
hermes gateway run

# 方案二：通过 tmux 持久运行（关闭终端后仍存活）
tmux new -s hermes 'hermes gateway run'
# 稍后重新连接：tmux attach -t hermes

# 方案三：通过 nohup 后台运行
nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &
```

如果仍想尝试 systemd，请确保已启用：

1. 打开 `/etc/wsl.conf`（不存在则创建）
2. 添加：
   ```ini
   [boot]
   systemd=true
   ```
3. 在 PowerShell 中执行：`wsl --shutdown`
4. 重新打开 WSL 终端
5. 验证：`systemctl is-system-running` 应显示 "running" 或 "degraded"

:::tip Windows 开机自启
如需可靠的自启动，使用 Windows 任务计划程序在登录时启动 WSL + 网关：
1. 创建一个任务，运行 `wsl -d Ubuntu -- bash -lc 'hermes gateway run'`
2. 设置在用户登录时触发
:::

#### macOS：网关找不到 Node.js / ffmpeg / 其他工具

**原因：** launchd 服务继承的是最小 PATH（`/usr/bin:/bin:/usr/sbin:/sbin`），不包含 Homebrew、nvm、cargo 或其他用户安装的工具目录。这通常会导致 WhatsApp bridge（`node not found`）或语音转录（`ffmpeg not found`）失败。

**解决方案：** 网关在您运行 `hermes gateway install` 时会捕获您的 shell PATH。如果您在设置网关后安装了新工具，请重新运行 install 以捕获更新后的 PATH：

```bash
hermes gateway install    # 重新快照当前 PATH
hermes gateway start      # 检测到更新的 plist 并重新加载
```

您可以验证 plist 中的 PATH 是否正确：
```bash
/usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:PATH" \
  ~/Library/LaunchAgents/ai.hermes.gateway.plist
```

---

### 性能问题

#### 响应缓慢

**原因：** 模型较大、API 服务器距离较远，或系统 prompt（提示词）包含过多工具。

**解决方案：**
- 尝试更快/更小的模型：`hermes chat --model openrouter/meta-llama/llama-3.1-8b-instruct`
- 减少激活的工具集：`hermes chat -t "terminal"`
- 检查到提供商的网络延迟
- 对于本地模型，确保有足够的 GPU VRAM

#### token 用量过高

**原因：** 对话过长、系统 prompt 冗长，或大量工具调用积累了上下文。

**解决方案：**
```bash
# 压缩对话以减少 token
/compress

# 查看会话 token 用量
/usage
```

:::tip
在长会话中定期使用 `/compress`。它会对对话历史进行摘要，在保留上下文的同时显著减少 token 用量。
:::

#### 会话过长

**原因：** 长时间对话积累了大量消息和工具输出，接近上下文限制。

**解决方案：**
```bash
# 压缩当前会话（保留关键上下文）
/compress

# 开始新会话并引用旧会话
hermes chat

# 如需稍后继续特定会话
hermes chat --continue
```

---

### MCP 问题

#### MCP 服务器无法连接

**原因：** 找不到服务器二进制文件、命令路径错误或缺少运行时。

**解决方案：**
```bash
# 确保 MCP 依赖项已安装（标准安装中已包含）
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

# 对于基于 npm 的服务器，确保 Node.js 可用
node --version
npx --version

# 手动测试服务器
npx -y @modelcontextprotocol/server-filesystem /tmp
```

验证您的 `~/.hermes/config.yaml` 中的 MCP 配置：
```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
```

#### MCP 服务器的工具未显示

**原因：** 服务器已启动但工具发现失败、工具被配置过滤掉，或服务器不支持您期望的 MCP 能力。

**解决方案：**
- 检查网关/智能体日志中的 MCP 连接错误
- 确保服务器响应 `tools/list` RPC 方法
- 检查该服务器下的 `tools.include`、`tools.exclude`、`tools.resources`、`tools.prompts` 或 `enabled` 设置
- 请注意，资源/prompt 工具仅在会话实际支持相应能力时才会注册
- 更改配置后使用 `/reload-mcp`

```bash
# 验证 MCP 服务器已配置
hermes config show | grep -A 12 mcp_servers

# 更改配置后重启 Hermes 或重新加载 MCP
hermes chat
```

另请参阅：
- [MCP（模型上下文协议）](/user-guide/features/mcp)
- [在 Hermes 中使用 MCP](/guides/use-mcp-with-hermes)
- [MCP 配置参考](/reference/mcp-config-reference)

#### MCP 超时错误

**原因：** MCP 服务器响应时间过长，或在执行过程中崩溃。

**解决方案：**
- 如果 MCP 服务器配置支持，增加超时时间
- 检查 MCP 服务器进程是否仍在运行
- 对于远程 HTTP MCP 服务器，检查网络连接

:::warning
如果 MCP 服务器在请求中途崩溃，Hermes 会报告超时。请检查服务器自身的日志（而非仅 Hermes 日志）以诊断根本原因。
:::

---

## Profiles（配置文件）

### Profiles 与直接设置 HERMES_HOME 有何不同？

Profiles 是构建在 `HERMES_HOME` 之上的托管层。您*可以*在每次命令前手动设置 `HERMES_HOME=/some/path`，但 profiles 会为您处理所有底层工作：创建目录结构、生成 shell 别名（`hermes-work`）、在 `~/.hermes/active_profile` 中跟踪活动 profile，以及自动跨所有 profiles 同步技能更新。它们还与 tab 补全集成，让您无需记忆路径。

### 两个 profiles 可以共享同一个 bot token 吗？

不可以。每个消息平台（Telegram、Discord 等）都需要对 bot token 的独占访问权。如果两个 profiles 同时尝试使用同一个 token，第二个网关将无法连接。请为每个 profile 创建单独的 bot — 对于 Telegram，请与 [@BotFather](https://t.me/BotFather) 对话以创建额外的 bot。

### Profiles 共享记忆或会话吗？

不共享。每个 profile 都有自己独立的记忆存储、会话数据库和技能目录，完全隔离。如果您想用现有的记忆和会话创建新 profile，请使用 `hermes profile create newname --clone-all` 从当前 profile 复制所有内容。

### 运行 `hermes update` 时会发生什么？

`hermes update` 拉取最新代码并重新安装依赖项**一次**（不是每个 profile 各一次）。然后自动将更新的技能同步到所有 profiles。您只需运行一次 `hermes update` — 它覆盖机器上的每个 profile。

### 可以运行多少个 profiles？

没有硬性限制。每个 profile 只是 `~/.hermes/profiles/` 下的一个目录。实际限制取决于您的磁盘空间以及系统能处理多少个并发网关（每个网关是一个轻量级 Python 进程）。运行数十个 profiles 完全没问题；每个空闲的 profile 不占用任何资源。

---

## 工作流与模式

### 针对不同任务使用不同模型（多模型工作流）

**场景：** 您日常使用 GPT-5.4，但 Gemini 或 Grok 写社交媒体内容更好。每次手动切换模型很繁琐。

**解决方案：委托配置。** Hermes 可以自动将子智能体路由到不同的模型。在 `~/.hermes/config.yaml` 中设置：

```yaml
delegation:
  model: "google/gemini-3-flash-preview"   # 子智能体使用此模型
  provider: "openrouter"                    # 子智能体的提供商
```

现在当您告诉 Hermes "帮我写一个关于 X 的 Twitter 帖子"并生成 `delegate_task` 子智能体时，该子智能体将在 Gemini 上运行，而非您的主模型。您的主对话仍在 GPT-5.4 上进行。

您也可以在 prompt 中明确指定：*"委托一个任务来撰写关于我们产品发布的社交媒体帖子。让你的子智能体负责实际写作。"* 智能体将使用 `delegate_task`，它会自动读取委托配置。

如需一次性切换模型而不使用委托，请在 CLI 中使用 `/model`：

```bash
/model google/gemini-3-flash-preview    # 在本次会话中切换
# ... 撰写内容 ...
/model openai/gpt-5.4                   # 切换回来
```

有关委托工作原理的更多信息，请参阅[子智能体委托](../user-guide/features/delegation.md)。

### 在一个 WhatsApp 号码上运行多个智能体（按聊天绑定）

**场景：** 在 OpenClaw 中，您可以将多个独立智能体绑定到特定的 WhatsApp 聊天 — 一个用于家庭购物清单群组，另一个用于您的私聊。Hermes 能做到吗？

**当前限制：** Hermes 的每个 profile 都需要自己的 WhatsApp 号码/会话。您无法将多个 profiles 绑定到同一个 WhatsApp 号码上的不同聊天 — WhatsApp bridge（Baileys）每个号码使用一个已认证的会话。

**变通方案：**

1. **使用单个 profile 配合人格切换。** 创建不同的 `AGENTS.md` 上下文文件或使用 `/personality` 命令按聊天更改行为。智能体能感知当前所在的聊天并进行适应。

2. **使用 cron 作业处理专项任务。** 对于购物清单跟踪器，设置一个监控特定聊天并管理清单的 cron 作业 — 无需单独的智能体。

3. **使用独立号码。** 如果您需要真正独立的智能体，将每个 profile 与其自己的 WhatsApp 号码配对。Google Voice 等服务提供的虚拟号码可用于此目的。

4. **改用 Telegram 或 Discord。** 这些平台更自然地支持按聊天绑定 — 每个 Telegram 群组或 Discord 频道获得自己的会话，您可以在同一账户上运行多个 bot token（每个 profile 一个）。

详情请参阅 [Profiles](../user-guide/profiles.md) 和 [WhatsApp 设置](../user-guide/messaging/whatsapp.md)。

### 控制 Telegram 中显示的内容（隐藏日志和推理过程）

**场景：** 您在 Telegram 中看到了网关执行日志、Hermes 推理过程和工具调用详情，而不是最终输出。

**解决方案：** `config.yaml` 中的 `display.tool_progress` 设置控制显示多少工具活动：

```yaml
display:
  tool_progress: "off"   # 选项：off、new、all、verbose
```

- **`off`** — 仅显示最终响应。无工具调用、无推理过程、无日志。
- **`new`** — 实时显示新的工具调用（简短单行）。
- **`all`** — 显示所有工具活动，包括结果。
- **`verbose`** — 完整详情，包括工具参数和输出。

对于消息平台，通常选择 `off` 或 `new`。编辑 `config.yaml` 后，重启网关使更改生效。

您也可以通过 `/verbose` 命令按会话切换（如果已启用）：

```yaml
display:
  tool_progress_command: true   # 在网关中启用 /verbose
```

### 在 Telegram 上管理技能（slash 命令限制）

**场景：** Telegram 有 100 个 slash 命令的限制，您的技能数量已超过此限制。您想禁用 Telegram 上不需要的技能，但 `hermes skills config` 设置似乎没有生效。

**解决方案：** 使用 `hermes skills config` 按平台禁用技能。这会写入 `config.yaml`：

```yaml
skills:
  disabled: []                    # 全局禁用的技能
  platform_disabled:
    telegram: [skill-a, skill-b]  # 仅在 telegram 上禁用
```

更改后，**重启网关**（`hermes gateway restart` 或终止并重新启动）。Telegram bot 命令菜单在启动时重建。

:::tip
描述过长的技能在 Telegram 菜单中会被截断为 40 个字符，以符合 payload 大小限制。如果技能未出现，可能是总 payload 大小问题而非 100 个命令数量限制 — 禁用未使用的技能对两者都有帮助。
:::

### 共享线程会话（多用户，一个对话）

**场景：** 您有一个 Telegram 或 Discord 线程，多人在其中 @ bot。您希望该线程中的所有 @ 都属于一个共享对话，而非每个用户各自独立的会话。

**当前行为：** Hermes 在大多数平台上按用户 ID 创建会话，因此每个人都有自己的对话上下文。这是出于隐私和上下文隔离的设计考量。

**变通方案：**

1. **使用 Slack。** Slack 会话按线程而非用户进行键控。同一线程中的多个用户共享一个对话 — 正是您描述的行为。这是最自然的选择。

2. **使用单用户的群聊。** 如果由一个人作为指定"操作员"转达问题，会话保持统一。其他人可以旁观。

3. **使用 Discord 频道。** Discord 会话按频道键控，因此同一频道中的所有用户共享上下文。为共享对话使用专用频道。

### 将 Hermes 迁移到另一台机器

**场景：** 您在一台机器上积累了技能、cron 作业和记忆，想将所有内容迁移到新的专用 Linux 机器。

**解决方案：**

1. 在新机器上安装 Hermes Agent：
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

2. 在**源机器**上创建完整备份：
   ```bash
   hermes backup
   ```
   这会将您整个 `~/.hermes/` 目录（配置、API key、记忆、技能、会话和 profiles）打包为 zip 文件，保存到主目录 `~/hermes-backup-<timestamp>.zip`。

3. 将 zip 文件复制到新机器并导入：
   ```bash
   # 在源机器上
   scp ~/hermes-backup-<timestamp>.zip newmachine:~/

   # 在新机器上
   hermes import ~/hermes-backup-<timestamp>.zip
   ```

4. 在新机器上运行 `hermes setup` 以验证 API key 和提供商配置是否正常工作。

### 将单个 profile 迁移到另一台机器

**场景：** 您想迁移或共享某个特定 profile，而非整个安装。

```bash
# 在源机器上
hermes profile export work ./work-backup.tar.gz

# 将文件复制到目标机器，然后：
hermes profile import ./work-backup.tar.gz work
```

导入的 profile 将包含导出时的所有配置、记忆、会话和技能。如果新机器的设置不同，您可能需要更新路径或重新向提供商进行身份验证。

### `hermes backup` 与 `hermes profile export` 的对比

| 功能 | `hermes backup` | `hermes profile export` |
| :--- | :--- | :--- |
| **使用场景** | **整机迁移** | **移植/共享特定 profile** |
| **范围** | 全局（整个 `~/.hermes` 目录） | 局部（单个 profile 目录） |
| **包含内容** | 所有 profiles、全局配置、API key、会话 | 单个 profile：SOUL.md、记忆、会话、技能 |
| **凭据** | **包含**（`.env` 和 `auth.json`） | **排除**（为安全共享而剥离） |
| **格式** | `.zip` | `.tar.gz` |

**手动备选方案（rsync）：** 如果您倾向于直接复制文件，请排除代码仓库：
```bash
rsync -av --exclude='hermes-agent' ~/.hermes/ newmachine:~/.hermes/
```

:::tip
`hermes backup` 即使在 Hermes 正在运行时也能生成一致的快照。还原的归档文件不包含机器本地的运行时文件，如 `gateway.pid` 和 `cron.pid`。
:::

### 安装后重新加载 shell 时出现权限拒绝

**场景：** 运行 Hermes 安装程序后，`source ~/.zshrc` 提示权限拒绝错误。

**原因：** 这通常发生在 `~/.zshrc`（或 `~/.bashrc`）文件权限不正确，或安装程序无法干净写入时。这不是 Hermes 特有的问题 — 而是 shell 配置权限问题。

**解决方案：**
```bash
# 检查权限
ls -la ~/.zshrc

# 如需修复（应为 -rw-r--r-- 或 644）
chmod 644 ~/.zshrc

# 然后重新加载
source ~/.zshrc

# 或直接打开新终端窗口 — 它会自动读取 PATH 更改
```

如果安装程序已添加 PATH 行但权限有误，您可以手动添加：
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

### 首次运行智能体时出现 400 错误

**场景：** 设置顺利完成，但第一次聊天尝试失败，提示 HTTP 400。

**原因：** 通常是模型名称不匹配 — 配置的模型在您的提供商上不存在，或 API key 没有访问该模型的权限。

**解决方案：**
```bash
# 检查已配置的模型和提供商
hermes config show | head -20

# 重新运行模型选择
hermes model

# 或使用已知可用的模型测试
hermes chat -q "hello" --model anthropic/claude-opus-4.7
```

如果使用 OpenRouter，请确保您的 API key 有余额。OpenRouter 返回 400 通常意味着该模型需要付费套餐，或模型 ID 有拼写错误。

---

## 仍然遇到问题？

如果您的问题未在此处涵盖：

1. **搜索现有 issue：** [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
2. **向社区提问：** [Nous Research Discord](https://discord.gg/nousresearch)
3. **提交 bug 报告：** 请包含您的操作系统、Python 版本（`python3 --version`）、Hermes 版本（`hermes --version`）以及完整的错误信息