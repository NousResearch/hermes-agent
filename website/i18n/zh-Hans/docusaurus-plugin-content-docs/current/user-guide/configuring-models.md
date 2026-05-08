---
sidebar_position: 3
---

# 配置模型

Hermes 使用两种模型槽位：

- **主模型** — agent 思考时使用的模型。每个用户消息、每个工具调用循环、每个流式响应都经过这个模型。
- **辅助模型** — agent 卸载的较小辅助任务。上下文压缩、视觉（图像分析）、网页摘要、会话搜索、审批评分、MCP 工具路由、会话标题生成和 skill 搜索。每个都有自己的槽位，可以独立覆盖。

本页面涵盖从仪表板配置两者。如果您更喜欢配置文件或 CLI，请跳到底部的 [替代方法](#alternative-methods)。

## 模型页面

打开仪表板并点击侧边栏中的 **Models**。您会看到两个部分：

1. **模型设置** — 顶部面板，您可以在此为槽位分配模型。
2. **使用分析** — 排名卡片，显示在所选时间段内运行过会话的每个模型，包括 token 计数、成本和能力徽章。

![Models page overview](/img/docs/dashboard-models/overview.png)

顶部卡片是**模型设置**面板。主行始终显示 agent 为新会话启动的内容。点击 **Change** 打开选择器。

## 设置主模型

点击主模型行上的 **Change**：

![Model picker dialog](/img/docs/dashboard-models/picker-dialog.png)

选择器有两列：

- **左侧** — 已认证的 provider。只有您设置好的 provider（API 密钥已设置、已完成 OAuth 或定义为自定义端点）才会显示在此处。如果缺少 provider，请前往 **Keys** 添加其凭证。
- **右侧** — 所选 provider 的精选模型列表。这些是 Hermes 推荐的 agentic 模型，不是原始的 `/models` 转储（OpenRouter 上包含 400+ 模型，包括 TTS、图像生成器和重排序器）。

在过滤器框中输入以按 provider 名称、slug 或模型 ID 缩小范围。

选择一个模型，点击 **Switch**，Hermes 会将其写入 `~/.hermes/config.yaml` 的 `model` 部分。**这仅适用于新会话**——您已经打开的任何聊天标签页会继续运行它启动时的模型。要热切换当前聊天，请在其中使用 `/model` 斜杠命令。

## 设置辅助模型

点击 **Show auxiliary** 展开八个任务槽位：

![Auxiliary panel expanded](/img/docs/dashboard-models/auxiliary-expanded.png)

每个辅助任务默认为 `auto`——这意味着 Hermes 也使用您的主模型来完成该任务。当您希望为辅助任务使用更便宜或更快的模型时，覆盖特定任务。

### 常见覆盖模式

| 任务 | 何时覆盖 |
|---|---|
| **标题生成** | 几乎总是。一个 $0.10/M 的 flash 模型写会话标题和 Opus 一样好。默认配置在 OpenRouter 上将其设置为 `google/gemini-3-flash-preview`。 |
| **视觉** | 当您的主模型是没有视觉功能的编码模型时（例如 Kimi、DeepSeek）。将其指向 `google/gemini-2.5-flash` 或 `gpt-4o-mini`。 |
| **压缩** | 当您在 Opus/M2.7 上燃烧推理 token 来摘要上下文时。快速聊天模型以 1/50 的成本完成这项工作。 |
| **会话搜索** | 当召回查询分散时——默认 max_concurrency 为 3。便宜模型保持账单可预测。 |
| **审批** | 对于 `approval_mode: smart`——快速/便宜的模型（haiku、flash、gpt-5-mini）决定是否自动批准低风险命令。这里的昂贵模型是浪费。 |
| **网页提取** | 当您大量使用 `web_extract` 时。与压缩相同的逻辑——摘要不需要推理。 |
| **Skills Hub** | `hermes skills search` 使用它。通常在 `auto` 就很好。 |
| **MCP** | MCP 工具路由。通常在 `auto` 就很好。 |

### 按任务覆盖

点击任何辅助行上的 **Change**。相同的选择器打开，相同的行为——选择 provider + 模型，点击 Switch。该行更新显示 `provider · model` 而不是 `auto (use main model)`。

### 重置全部为 auto

如果您过度调整并想重新开始，请点击辅助部分顶部的 **Reset all to auto**。每个槽位都恢复为使用您的主模型。

## "Use as" 快捷方式

页面上的每个模型卡片都有一个 **Use as** 下拉菜单。这是快速路径——选择您在分析中看到的模型，点击 **Use as**，一键将其分配到主槽位或任何特定辅助任务：

![Use as dropdown](/img/docs/dashboard-models/use-as-dropdown.png)

下拉菜单包含：

- **主模型** — 与点击主行上的 Change 相同。
- **所有辅助任务** — 同时将此模型分配给所有 8 个辅助槽位。当您只想让每个辅助任务都在便宜的 flash 模型上运行时很有用。
- **单独任务选项** — 视觉、网页提取、压缩等。当前分配给每个任务的模型标记为 `current`。

当卡片当前分配给某些内容时，卡片会标记 `main` 或 `aux · <task>`——因此您可以一目了然地看到历史模型连接到何处。

## 什么被写入 `config.yaml`

当您通过仪表板保存时，Hermes 会写入 `~/.hermes/config.yaml`：

**主模型：**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # 在 provider 切换时清除
  api_mode: chat_completions
```

**辅助覆盖（示例 — gemini-flash 上的视觉）：**
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

**辅助设置为 auto（默认）：**
```yaml
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    # ... 其他字段不变
```

`provider: auto` 与 `model: ''` 告诉 Hermes 使用主模型来完成该任务。

## 何时生效？

- **CLI**（`hermes chat`）：下一次 `hermes chat` 调用。
- **Gateway**（Telegram、Discord、Slack 等）：下一个*新*会话。现有会话保留其模型。如果您想强制所有会话获取更改，请重启 gateway（`hermes gateway restart`）。
- **仪表板聊天标签页**（`/chat`）：下一个新 PTY。当前打开的聊天保留其模型——在其中使用 `/model` 热切换。

更改从不会使运行中会话的提示缓存失效。这是故意的：在会话内切换主模型需要重置缓存（系统提示包含特定于模型的内容），我们将其保留用于聊天内明确的 `/model` 斜杠命令。

## 故障排除

### 选择器中"无已认证 provider"

Hermes 仅在有有效凭证时列出 provider。检查侧边栏中的 **Keys**——您应该看到以下之一：API 密钥、成功完成 OAuth 或自定义端点 URL。如果您想要的 provider 不在那里，请运行 `hermes setup` 进行连接，或前往 **Keys** 添加环境变量。

### 我的运行中聊天中主模型没有改变

这是预期的。仪表板写入 `config.yaml`，新会话会读取。当前打开的聊天是一个实时 agent 进程——它保留启动时的模型。在聊天中使用 `/model <name>` 热切换该特定会话。

### 辅助覆盖"没有生效"

检查三件事：

1. **您是否启动了新会话？** 现有聊天不会重新读取配置。
2. **`provider` 是否设置为 `auto` 以外的值？** 如果字段显示 `auto`，该任务仍在使用您的主模型。点击 **Change** 并选择一个真实的 provider。
3. **该 provider 已认证吗？** 如果您将 `minimax` 分配给任务但没有 MiniMax API 密钥，该任务会回退到 openrouter 默认值并在 `agent.log` 中记录警告。

### 我选择了一个模型但 Hermes 切换了 provider

在 OpenRouter（或任何聚合器）上，裸模型名称首先在*聚合器内部*解析。所以 OpenRouter 上的 `claude-sonnet-4` 变成 `anthropic/claude-sonnet-4.6`，保持在您的 OpenRouter 认证上。但如果您在原生 Anthropic 认证上输入 `claude-sonnet-4`，它会保持为 `claude-sonnet-4-6。如果您看到意外的 provider 切换，请检查您当前的 provider 是否符合预期——选择器始终在对话框顶部显示当前的主模型。

## 替代方法

### CLI 斜杠命令

在任何 `hermes chat` 会话中：

```
/model gpt-5.4 --provider openrouter             # 仅会话内
/model gpt-5.4 --provider openrouter --global    # 也持久化到 config.yaml
```

`--global` 执行与仪表板的 **Change** 按钮相同的操作，并会切换运行中的会话。

### 自定义别名

为您经常使用的模型定义自己的短名称，然后在使用 CLI 或任何消息传递平台时使用 `/model <alias>`：

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

或从 shell（短格式，`provider/model`）：

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

然后在聊天中使用 `/model fav` 或 `/model grok`。用户别名会覆盖内置短名称（`sonnet`、`kimi`、`opus` 等）。请参阅 [自定义模型别名](/docs/reference/slash-commands#custom-model-aliases) 获取完整参考。

### `hermes model` 子命令

```bash
hermes model list                   # 列出已认证的 provider + 模型
hermes model set anthropic/claude-opus-4.7 --provider openrouter
```

### 直接编辑配置

编辑 `~/.hermes/config.yaml` 并重启读取它的任何内容。请参阅 [配置参考](./configuration.md) 获取完整 schema。

### REST API

仪表板使用三个端点。对脚本编写很有用：

```bash
# 列出已认证的 provider + 精选模型列表
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# 读取当前主 + 辅助分配
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# 设置主模型
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# 覆盖单个辅助任务
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# 将一个模型分配给每个辅助任务
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# 重置所有辅助任务为 auto
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

会话令牌在启动时注入到仪表板 HTML 中，每次服务器重启时轮换。如果您正在脚本编写以对抗运行中的仪表板，请从浏览器 devtools（`window.__HERMES_SESSION_TOKEN__`）中获取它。
