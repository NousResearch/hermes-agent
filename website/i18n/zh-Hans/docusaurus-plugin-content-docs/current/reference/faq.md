---
sidebar_position: 3
title: "常见问题与故障排查"
description: "Hermes Agent 的常见问题解答与故障排查方案"
---

# 常见问题与故障排查

这里汇总了 Hermes Agent 最常见的问题和快速解决办法。

---

## 常见问题

### Hermes 支持哪些 LLM 提供商？

Hermes Agent 可以使用任何兼容 OpenAI 的 API。支持的提供商包括：

- **[OpenRouter](https://openrouter.ai/)** - 通过一个 API 密钥访问数百种模型（推荐，灵活性高）
- **Nous Portal** - Nous Research 自家的推理端点
- **OpenAI** - GPT-5.4、GPT-5-codex、GPT-4.1、GPT-4o 等
- **Anthropic** - Claude 系列模型（直连 API、通过 `hermes login anthropic` 的 OAuth、OpenRouter，或任何兼容代理）
- **Google** - Gemini 系列模型（通过 `gemini` 提供商的直连 API、`google-gemini-cli` OAuth 提供商、OpenRouter，或兼容代理）
- **z.ai / ZhipuAI** - GLM 模型
- **Kimi / Moonshot AI** - Kimi 模型
- **MiniMax** - 全球和中国区端点
- **本地模型** - 通过 [Ollama](https://ollama.com/)、[vLLM](https://docs.vllm.ai/)、[llama.cpp](https://github.com/ggerganov/llama.cpp)、[SGLang](https://github.com/sgl-project/sglang) 或任何兼容 OpenAI 的服务

你可以用 `hermes model` 设置提供商，也可以直接编辑 `~/.hermes/.env`。完整的提供商键名请参见 [环境变量](/reference/environment-variables) 参考。

### 能在 Windows 上运行吗？

**不能原生运行。** Hermes Agent 需要类 Unix 环境。在 Windows 上，请安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) 并在其中运行 Hermes。标准安装命令在 WSL2 里可以直接使用：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### 我在 WSL2 里运行 Hermes，怎样控制 Windows 上正常登录的 Chrome？ {#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails}

建议使用 MCP 桥接，而不是 `/browser connect`。

推荐方案：

- 在 WSL2 中运行 Hermes
- 继续使用 Windows 上已登录的 Chrome
- 通过 `cmd.exe` 或 `powershell.exe` 添加 `chrome-devtools-mcp` 作为 MCP 服务器
- 让 Hermes 使用最终得到的 MCP 浏览器工具

这比强行让 Hermes 核心浏览器传输层直接跨越 WSL2 / Windows 边界连接更可靠。

参见：

- [通过 MCP 与 Hermes 集成](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)
- [浏览器自动化](/user-guide/features/browser#wsl2--windows-chrome-prefer-mcp-over-browser-connect)

### 能在 Android / Termux 上运行吗？

可以 - Hermes 现在已经有经过测试的 Termux 安装路径，适用于 Android 手机。

快速安装：

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

更完整的手动步骤、支持的额外依赖和当前限制，请参见 [Termux 指南](/getting-started/termux)。

重要说明：Android 目前不能使用完整的 `.[all]` 额外依赖，因为 `voice` 依赖 `faster-whisper` → `ctranslate2`，而 `ctranslate2` 没有发布 Android wheel。请改用经过测试的 `.[termux]` 额外依赖。

### 我的数据会被发送到哪里？

API 调用**只会发送到你配置的 LLM 提供商**（例如 OpenRouter 或本地 Ollama 实例）。Hermes Agent 不收集遥测、使用数据或分析数据。你的对话、记忆和技能都会保存在本地的 `~/.hermes/` 中。

### 能离线使用 / 使用本地模型吗？

可以。运行 `hermes model`，选择 **Custom endpoint**，然后输入你的服务地址：

```bash
hermes model
# Select: Custom endpoint (enter URL manually)
# API base URL: http://localhost:11434/v1
# API key: ollama
# Model name: qwen3.5:27b
# Context length: 32768   ← 将其设置为与你的服务实际上下文窗口一致
```

或者直接在 `config.yaml` 中配置：

```yaml
model:
  default: qwen3.5:27b
  provider: custom
  base_url: http://localhost:11434/v1
```

Hermes 会把端点、提供商和 base URL 持久化到 `config.yaml`，因此重启后仍然保留。如果本地服务只加载了一个模型，`/model custom` 会自动检测它。你也可以直接在 `config.yaml` 中设置 `provider: custom` - 它是一个一等公民提供商，不是其他提供商的别名。

这适用于 Ollama、vLLM、llama.cpp server、SGLang、LocalAI 等。详情请参见 [配置指南](/user-guide/configuration)。

:::tip Ollama 用户
如果你在 Ollama 中设置了自定义 `num_ctx`（例如 `ollama run --num_ctx 16384`），请务必在 Hermes 中设置匹配的上下文长度 - Ollama 的 `/api/show` 返回的是模型的*最大*上下文，而不是你实际配置的 `num_ctx`。
:::

:::tip 本地模型超时
Hermes 会自动识别本地端点，并放宽流式超时（读取超时从 120s 提升到 1800s，关闭卡死流检测）。如果在超大上下文下仍然超时，可以在 `.env` 中设置 `HERMES_STREAM_READ_TIMEOUT=1800`。详情见 [本地 LLM 指南](/guides/local-llm-on-mac#timeouts)。
:::

### 这要花多少钱？

Hermes Agent 本身是**免费且开源**的（MIT 许可）。你只需要为所选提供商的 LLM API 调用付费。本地模型则完全免费运行。

### 多个人可以共用一个实例吗？

可以。[消息网关](/user-guide/messaging) 允许多个用户通过 Telegram、Discord、Slack、WhatsApp 或 Home Assistant 与同一个 Hermes Agent 实例交互。访问通过白名单（指定用户 ID）和 DM 配对（第一个发消息的用户获得访问权）来控制。

### memory 和 skills 有什么区别？

- **Memory** 存储的是**事实** - 例如智能体对你、你的项目和偏好的了解。记忆会根据相关性自动检索。
- **Skills** 存储的是**流程** - 即完成某件事的步骤化说明。技能会在智能体遇到类似任务时被调用。

二者都会跨会话保留。详情见 [Memory](/user-guide/features/memory) 和 [Skills](/user-guide/features/skills)。

### 我可以在自己的 Python 项目里使用它吗？

可以。导入 `AIAgent` 类，就能以编程方式使用 Hermes：

```python
from run_agent import AIAgent

agent = AIAgent(model="anthropic/claude-opus-4.7")
response = agent.chat("简要解释量子计算")
```

完整 API 用法请参见 [Python Library 指南](/user-guide/features/code-execution)。

---

## 故障排查

### 安装问题

#### 安装后出现 `hermes: command not found`

**原因：** 你的 shell 还没有重新加载更新后的 PATH。

**解决办法：**
```bash
# 重新加载 shell 配置文件
source ~/.bashrc    # bash
source ~/.zshrc     # zsh

# 或者直接打开一个新的终端会话
```

如果还是不行，检查安装位置：
```bash
which hermes
ls ~/.local/bin/hermes
```

:::tip
安装器会把 `~/.local/bin` 加入你的 PATH。如果你使用的是非标准 shell 配置，请手动添加 `export PATH="$HOME/.local/bin:$PATH"`。
:::

#### Python 版本太旧

**原因：** Hermes 需要 Python 3.11 或更新版本。

**解决办法：**
```bash
python3 --version   # 检查当前版本

# 安装较新的 Python
sudo apt install python3.12   # Ubuntu/Debian
brew install python@3.12      # macOS
```

安装器会自动处理这个问题 - 如果你在手动安装时看到这个错误，请先升级 Python。

#### 终端命令提示 `node: command not found`（或 `nvm`、`pyenv`、`asdf` 等）

**原因：** Hermes 会在启动时运行一次 `bash -l` 来构建每个会话的环境快照。bash 登录 shell 会读取 `/etc/profile`、`~/.bash_profile` 和 `~/.profile`，但**不会读取 `~/.bashrc`** - 所以安装在那里的工具（`nvm`、`asdf`、`pyenv`、`cargo`、自定义 `PATH` 导出）对快照不可见。这个问题最常见于 Hermes 运行在 systemd 下，或者运行在一个没有预加载交互式 shell 配置的最简环境里。

**解决办法：** Hermes 默认会自动源入 `~/.bashrc`。如果这还不够 - 例如你是 zsh 用户，PATH 在 `~/.zshrc` 里，或者你通过单独文件初始化 `nvm` - 可以在 `~/.hermes/config.yaml` 中列出额外需要源入的文件：

```yaml
terminal:
  shell_init_files:
    - ~/.zshrc                     # zsh 用户：把 zsh 管理的 PATH 带入 bash 快照
    - ~/.nvm/nvm.sh                # 直接初始化 nvm（与 shell 无关）
    - /etc/profile.d/cargo.sh      # 系统级 rc 文件
  # 当设置了这个列表时，默认的 ~/.bashrc 自动源入不会再额外添加——
  # 如果你想同时保留两者，请显式加入：
  #   - ~/.bashrc
  #   - ~/.zshrc
```

缺失的文件会被静默跳过。由于内容是在 bash 中执行，依赖 zsh 专用语法的文件可能会报错 - 如果这是个问题，只源入负责设置 PATH 的部分（例如直接使用 nvm 的 `nvm.sh`），不要整个 rc 文件都源入。

若要禁用自动源入行为（只保留严格的登录 shell 语义）：

```yaml
terminal:
  auto_source_bashrc: false
```

#### `uv: command not found`

**原因：** 没有安装 `uv` 包管理器，或者它不在 PATH 中。

**解决办法：**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### 安装时出现权限拒绝错误

**原因：** 没有向安装目录写入的权限。

**解决办法：**
```bash
# 不要在安装器里使用 sudo —— 它会安装到 ~/.local/bin
# 如果你之前用 sudo 安装过，请先清理：
sudo rm /usr/local/bin/hermes
# 然后重新运行标准安装器
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

---

### 提供商与模型问题

#### `/model` 只显示一个提供商 / 不能切换提供商

**原因：** `/model`（在聊天会话内）只能在你**已经配置**的提供商之间切换。如果你只设置了 OpenRouter，那 `/model` 只会显示它。

**解决办法：** 先退出当前会话，然后在终端里运行 `hermes model` 来添加新提供商：

```bash
# 先退出 Hermes 聊天会话（Ctrl+C 或 /quit）

# 运行完整的提供商设置向导
hermes model

# 这可以让你：添加提供商、执行 OAuth、输入 API 密钥、配置端点
```

通过 `hermes model` 添加新提供商后，重新开始一个聊天会话 - `/model` 就会显示所有已配置的提供商。

:::tip 快速参考
| 想做什么 | 使用 |
|-----------|-----|
| 添加新提供商 | `hermes model`（在终端中） |
| 输入/修改 API 密钥 | `hermes model`（在终端中） |
| 在会话中切换模型 | `/model <name>`（在会话内） |
| 切换到另一个已配置提供商 | `/model provider:model`（在会话内） |
:::
