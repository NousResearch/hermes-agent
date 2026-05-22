---
sidebar_position: 3
---

# 配置模型

Hermes 使用两类模型槽位：

- **主模型** - agent 思考所用的模型。每条用户消息、每一次工具调用循环、每一段流式回复，都会通过这个模型。
- **辅助模型** - agent 外包出去的较小任务。包括上下文压缩、视觉（图像分析）、网页摘要、会话搜索、审批评分、MCP 工具路由、会话标题生成和技能搜索。每个任务都有自己独立的槽位，并且可以单独覆盖。

本页介绍如何通过 dashboard 配置这两类模型。如果你更喜欢配置文件或 CLI，可以直接跳到底部的 [其他方式](#其他方式)。

## Models 页面

打开 dashboard，在侧边栏点击 **Models**。你会看到两个区域：

1. **Model Settings** - 顶部面板，用于把模型分配到不同槽位。
2. **Usage analytics** - 排名卡片，显示在所选时间段内跑过会话的所有模型，以及 token 数、成本和能力标签。

![Models 页面概览](/img/docs/dashboard-models/overview.png)

顶部卡片就是 **Model Settings** 面板。主模型这一行始终显示新会话将使用的模型。点击 **Change** 打开选择器。

## 设置主模型

点击主模型行上的 **Change**：

![模型选择器对话框](/img/docs/dashboard-models/picker-dialog.png)

选择器分成两列：

- **左侧** - 已认证的提供商。这里只会显示你已经配置好的提供商（已经设置 API Key、完成 OAuth，或定义为自定义端点）。如果缺少某个提供商，去 **Keys** 里添加对应凭据。
- **右侧** - 当前选中提供商的精选模型列表。这里显示的是 Hermes 推荐给该提供商使用的 agentic 模型，而不是原始的 `/models` 全量列表（比如 OpenRouter 会包含 400+ 模型，包括 TTS、图像生成器和 reranker）。

在过滤框中输入内容，可以按提供商名、slug 或模型 ID 过滤。

选中一个模型，点击 **Switch**，Hermes 就会把它写入 `~/.hermes/config.yaml` 的 `model` 段。**这只会影响新会话** - 你已经打开的聊天标签页会继续使用启动时的模型。如果你想热切换当前聊天，请在聊天里使用 `/model` 斜杠命令。

## 设置辅助模型

点击 **Show auxiliary** 可以展开八个任务槽位：

![展开后的辅助面板](/img/docs/dashboard-models/auxiliary-expanded.png)

每个辅助任务默认都是 `auto` - 也就是 Hermes 也会使用你的主模型来完成它。如果你希望某个子任务更便宜或更快，可以单独覆盖。

### 常见覆盖模式

| 任务 | 什么时候覆盖 |
|---|---|
| **Title Gen** | 几乎总是。一个 $0.10/M 的 flash 模型写会话标题和 Opus 一样好。默认配置在 OpenRouter 上会把它设为 `google/gemini-3-flash-preview`。 |
| **Vision** | 当你的主模型不支持视觉时（比如 Kimi、DeepSeek）。把它指向 `google/gemini-2.5-flash` 或 `gpt-4o-mini`。 |
| **Compression** | 当你只是为了压缩上下文而在 Opus/M2.7 上烧推理 token 时。快速聊天模型可以用 1/50 的成本完成这件事。 |
| **Session Search** | 当召回查询会分散到很多条时 - 默认 `max_concurrency` 是 3。便宜模型能让账单更稳定。 |
| **Approval** | 在 `approval_mode: smart` 下 - 快速便宜的模型（haiku、flash、gpt-5-mini）负责判断低风险命令是否可自动批准。这里用昂贵模型纯属浪费。 |
| **Web Extract** | 当你大量使用 `web_extract` 时。逻辑和 compression 一样 - 摘要不需要推理能力。 |
| **Skills Hub** | `hermes skills search` 会用到它。通常保持 `auto` 就足够。 |
| **MCP** | MCP 工具路由。通常保持 `auto` 就够。 |

### 按任务覆盖

点击任意辅助行上的 **Change**。会弹出同一个选择器，行为也一样 - 选提供商 + 模型，然后点击 Switch。该行会从 `auto (use main model)` 变成显示 `provider · model`。

### 一键恢复为 auto

如果你调得太细，想全部重置，可以点击辅助区域顶部的 **Reset all to auto**。所有槽位都会回到使用主模型。

## “Use as” 快捷方式

页面上的每张模型卡都有一个 **Use as** 下拉菜单。这是最快路径 - 选中你在 analytics 里看到的模型，点击 **Use as**，就能一步分配给主槽位或某个辅助任务：

![Use as 下拉菜单](/img/docs/dashboard-models/use-as-dropdown.png)

下拉菜单包含：

- **Main model** - 效果和主行上的 Change 一样。
- **All auxiliary tasks** - 把这个模型一次性分给全部 8 个辅助槽位。适合你只想让所有子任务都跑在便宜 flash 模型上。
- **Individual task options** - Vision、Web Extract、Compression 等。当前分配给每个任务的模型会标记为 `current`。

卡片如果当前已经被分配到某个槽位，会显示 `main` 或 `aux · <task>` 标签 - 这样你一眼就能看出历史模型现在接到了哪里。

## 写入 `config.yaml` 的内容

通过 dashboard 保存时，Hermes 会把配置写入 `~/.hermes/config.yaml`：

**主模型：**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # 切换提供商时会被清空
  api_mode: chat_completions
```

**辅助覆盖（示例 - vision 使用 gemini-flash）：**
```yaml
auxiliary:
  vision:
    provider: openrouter
    model: google/gemini-2.5-flash
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
    download_timeout: 30
```

**辅助保持 auto（默认）：**
```yaml
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    # ... 其他字段保持不变
```

`provider: auto` 配合 `model: ''` 时，Hermes 会让这个任务直接使用主模型。

## 什么时候生效？

- **CLI**（`hermes chat`）：下一次启动 `hermes chat` 时生效。
- **网关**（Telegram、Discord、Slack 等）：下一次*新*会话生效。已有会话会保持原模型。如果你想强制所有会话都立即采用新配置，可以重启网关（`hermes gateway restart`）。
- **Dashboard chat 标签页**（`/chat`）：下一个新的 PTY 生效。当前打开的聊天会保持原模型 - 需要时可在聊天里用 `/model` 热切换。

这些更改永远不会让正在运行的会话失效 prompt cache。这是刻意设计的：在会话中切换主模型需要重置 cache（系统提示词里包含与模型相关的内容），而 Hermes 只会在聊天中显式使用 `/model` 时才这么做。

## 故障排查

### 选择器里显示 “No authenticated providers”

Hermes 只会列出有可用凭据的提供商。去侧边栏的 **Keys** 看一下 - 你应该至少能看到以下几种之一：API Key、成功完成的 OAuth，或者自定义端点 URL。如果你需要的提供商不在这里，运行 `hermes setup` 完成配置，或者去 **Keys** 添加环境变量。

### 主模型在我运行中的聊天里没有变化

这是预期行为。dashboard 只会写入 `config.yaml`，新会话才会读取它。当前打开的聊天是一个实时 agent 进程 - 它会继续使用启动时的模型。要对该会话热切换，请在聊天里使用 `/model <name>`。

### 辅助覆盖“没有生效”

检查这三件事：

1. **你是否开启了一个新会话？** 旧聊天不会重新读取配置。
2. **`provider` 是否不是 `auto`？** 如果字段显示 `auto`，说明这个任务仍在用主模型。点击 **Change**，选一个真正的提供商。
3. **提供商是否已认证？** 如果你把 `minimax` 分配给某个任务，但没有 MiniMax API Key，这个任务会回退到 openrouter 默认值，并在 `agent.log` 里记录一条警告。

### 我选了模型，但 Hermes 自己切换了提供商

在 OpenRouter（或其他聚合器）上，裸模型名会先在聚合器内部解析。所以在 OpenRouter 上输入 `claude-sonnet-4`，会变成 `anthropic/claude-sonnet-4.6`，仍然使用你的 OpenRouter 认证。但如果你在原生 Anthropic 认证下输入 `claude-sonnet-4`，它会保持为 `claude-sonnet-4-6`。如果你看到意外的提供商切换，请检查你当前使用的提供商是不是你预期中的那个 - 选择器顶部总会显示当前主模型。

## 其他方式

### CLI 斜杠命令

在任何 `hermes chat` 会话里：

```
/model gpt-5.4 --provider openrouter             # 仅对当前会话生效
/model gpt-5.4 --provider openrouter --global    # 也会持久化到 config.yaml
```

`--global` 的效果和 dashboard 里的 **Change** 按钮一样，同时还会原地切换当前运行中的会话。

### 自定义别名

你可以为常用模型定义短名，然后在 CLI 或任意消息平台里使用 `/model <alias>`：

```yaml
# ~/.hermes/config.yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
```

或者从 shell 里设置（短写，`provider/model`）：

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

然后在聊天里用 `/model fav` 或 `/model grok`。用户别名会覆盖内置短名（`sonnet`、`kimi`、`opus` 等）。完整说明见 [Custom model aliases](/reference/slash-commands#custom-model-aliases)。

### `hermes model` 子命令

```bash
hermes model            # 交互式提供商 + 模型选择器（切换默认值的标准方式）
```

`hermes model` 会引导你选择提供商、完成认证（OAuth 流程会打开浏览器；API Key 提供商会提示你输入 key），然后从该提供商的精选目录里选择具体模型。选择结果会写入 `~/.hermes/config.yaml` 里的 `model.provider` 和 `model.model`。

如果你只是想在不打开选择器的情况下列出提供商 / 模型，可以用 dashboard 或下面的 REST 接口。要查看 CLI 现在实际会使用什么：运行 `hermes config get model` 和 `hermes status`。

### 直接编辑配置

直接编辑 `~/.hermes/config.yaml`，然后重启读取它的进程。完整 schema 请参阅 [Configuration reference](./configuration.md)。

### REST API

dashboard 使用了三个接口，适合脚本调用：

```bash
# 列出已认证提供商 + 精选模型列表
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# 读取当前主模型 + 辅助分配
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# 设置主模型
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# 覆盖单个辅助任务
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# 把一个模型分配给所有辅助任务
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# 将所有辅助任务重置为 auto
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

会话 token 会在 dashboard 启动时注入 HTML，并在每次服务重启时轮换。你可以从浏览器开发者工具里取到它（`window.__HERMES_SESSION_TOKEN__`），方便脚本对正在运行的 dashboard 进行调用。