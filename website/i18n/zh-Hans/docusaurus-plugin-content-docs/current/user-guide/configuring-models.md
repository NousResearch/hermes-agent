---
sidebar_position: 3
---

# 配置模型

Hermes 使用两类模型槽位：

- **主模型** — agent 的思考核心。每条用户消息、每个工具调用循环、每次流式响应都经由该模型处理。
- **辅助模型** — agent 卸载给较小模型的边缘任务。包括上下文压缩、视觉（图像分析）、网页摘要、审批评分、MCP 工具路由、会话标题生成和技能搜索。每项任务有独立槽位，可单独覆盖。

本页介绍如何通过仪表板配置上述两类模型。如需使用配置文件或 CLI，请跳至底部的[其他方法](#alternative-methods)。

:::tip 最快路径：Nous Portal
[Nous Portal](/user-guide/features/tool-gateway) 在单一订阅下提供 300+ 个模型。全新安装后，运行 `hermes setup --portal` 即可登录并一键将 Nous 设为提供商。使用 `hermes portal info` 查看当前配置。
:::

## Models 页面

打开仪表板，点击侧边栏中的 **Models**。页面分为两个区域：

1. **Model Settings** — 顶部面板，用于为各槽位分配模型。
2. **使用分析** — 按排名显示所选时间段内运行过会话的所有模型，包含 token 数量、费用和能力标签。

![Models 页面概览](/img/docs/dashboard-models/overview.png)

顶部卡片为 **Model Settings** 面板。主行始终显示 agent 将为新会话启动的模型。点击 **Change** 打开选择器。

## 设置主模型

点击主模型行上的 **Change**：

![模型选择器对话框](/img/docs/dashboard-models/picker-dialog.png)

选择器分为两列：

- **左列** — 已认证的提供商。仅显示已配置的提供商（已设置 API key、完成 OAuth 或定义了自定义端点）。若某提供商未出现，请前往 **Keys** 添加凭据。
- **右列** — 所选提供商的精选模型列表。这些是 Hermes 针对该提供商推荐的 agentic 模型，而非原始的 `/models` 接口返回结果（OpenRouter 的原始列表包含 400+ 个模型，涵盖 TTS、图像生成器和重排序器）。

在过滤框中输入提供商名称、slug 或模型 ID 进行筛选。

选择模型后点击 **Switch**，Hermes 会将其写入 `~/.hermes/config.yaml` 的 `model` 部分。**此操作仅对新会话生效** — 已打开的聊天标签页将继续使用启动时的模型。如需在当前聊天中热切换，请在聊天内使用 `/model` 斜杠命令。

## 设置辅助模型

点击 **Show auxiliary** 展开 11 个任务槽位：

![辅助面板展开状态](/img/docs/dashboard-models/auxiliary-expanded.png)

每个辅助任务默认为 `auto`，即 Hermes 对该任务也使用主模型。当某个边缘任务需要更便宜或更快的模型时，可单独覆盖该槽位。

### 常见覆盖模式

| 任务 | 何时覆盖 |
|---|---|
| **Title Gen（标题生成）** | 几乎总是。$0.10/M 的 flash 模型生成会话标题的效果与 Opus 相当。默认配置在 OpenRouter 上将此项设为 `google/gemini-3-flash-preview`。 |
| **Vision（视觉）** | 当主模型是不支持视觉的编程模型时（如 Kimi、DeepSeek）。将其指向 `google/gemini-2.5-flash` 或 `gpt-4o-mini`。 |
| **Compression（压缩）** | 当你在用 Opus/M2.7 的推理 token 来摘要上下文时。快速聊天模型以 1/50 的成本即可完成此工作。 |
| **Approval（审批）** | 用于 `approval_mode: smart` — 由快速/廉价模型（haiku、flash、gpt-5-mini）决定是否自动批准低风险命令。此处使用昂贵模型是浪费。 |
| **Web Extract（网页提取）** | 当你大量使用 `web_extract` 时。逻辑同压缩 — 摘要任务不需要推理能力。 |
| **Skills Hub（技能中心）** | `hermes skills search` 使用此槽位。通常保持 `auto` 即可。 |
| **MCP** | MCP 工具路由。通常保持 `auto` 即可。 |

### 单任务覆盖

点击任意辅助行上的 **Change**，打开相同的选择器，操作方式相同 — 选择提供商和模型，点击 Switch。该行将从 `auto (use main model)` 更新为 `provider · model`。

### 全部重置为 auto

如果调整过度想重新开始，点击辅助区域顶部的 **Reset all to auto**。所有槽位将恢复使用主模型。

## "Use as" 快捷方式

页面上每张模型卡片都有 **Use as** 下拉菜单。这是快捷路径 — 从分析数据中选择一个模型，点击 **Use as**，一键将其分配到主槽位或任意辅助任务：

![Use as 下拉菜单](/img/docs/dashboard-models/use-as-dropdown.png)

下拉菜单包含：

- **Main model** — 与点击主行上的 Change 效果相同。
- **All auxiliary tasks** — 将此模型分配给全部 11 个辅助槽位。适合将所有边缘任务统一切换到廉价 flash 模型的场景。
- **单项任务选项** — Vision、Web Extract、Compression 等。每项任务当前分配的模型标记为 `current`。

当模型卡片当前已分配到某个槽位时，会显示 `main` 或 `aux · <task>` 标签，方便一眼看出历史模型的使用情况。

## 写入 `config.yaml` 的内容

通过仪表板保存时，Hermes 写入 `~/.hermes/config.yaml`：

**主模型：**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # cleared on provider switch
  api_mode: chat_completions
```

**辅助覆盖示例（视觉任务使用 gemini-flash）：**
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

**辅助任务处于 auto（默认）：**
```yaml
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    # ... other fields unchanged
```

`provider: auto` 加 `model: ''` 表示 Hermes 对该任务使用主模型。

## 何时生效？

- **CLI**（`hermes chat`）：下次执行 `hermes chat` 时生效。
- **Gateway**（Telegram、Discord、Slack 等）：下一个*新*会话生效。现有会话保持原有模型。如需强制所有会话使用新配置，重启 gateway（`hermes gateway restart`）。
- **仪表板聊天标签页**（`/chat`）：下一个新 PTY 生效。当前打开的聊天保持原有模型 — 在聊天内使用 `/model` 进行热切换。

更改不会使运行中会话的 prompt 缓存失效。这是有意为之：在会话内切换主模型需要重置缓存（系统 prompt 包含模型特定内容），该操作保留给聊天内的显式 `/model` 斜杠命令。

## 故障排查

### 选择器中显示"No authenticated providers"

Hermes 仅列出具有有效凭据的提供商。检查侧边栏中的 **Keys** — 应存在以下之一：API key、成功的 OAuth 或自定义端点 URL。若所需提供商不在列表中，运行 `hermes setup` 进行配置，或前往 **Keys** 添加环境变量。

### 主模型在运行中的聊天里未发生变化

符合预期。仪表板写入 `config.yaml`，新会话读取该文件。当前打开的聊天是一个活跃的 agent 进程 — 它保持启动时的模型。在聊天内使用 `/model <name>` 对该会话进行热切换。

### 辅助覆盖"未生效"

检查以下三点：

1. **是否启动了新会话？** 现有聊天不会重新读取配置。
2. **`provider` 是否设置为非 `auto` 的值？** 若字段显示 `auto`，该任务仍在使用主模型。点击 **Change** 选择实际的提供商。
3. **提供商是否已认证？** 若将 `minimax` 分配给某任务但没有 MiniMax API key，该任务将回退到 openrouter 默认值，并在 `agent.log` 中记录警告。

### 我选择了模型，但 Hermes 切换了提供商

在 OpenRouter（或任何聚合器）上，裸模型名称会优先在聚合器内解析。因此 OpenRouter 上的 `claude-sonnet-4` 会解析为 `anthropic/claude-sonnet-4.6`，保持在你的 OpenRouter 认证下。但若在原生 Anthropic 认证下输入 `claude-sonnet-4`，则会保持为 `claude-sonnet-4-6`。若出现意外的提供商切换，请确认当前提供商是否符合预期 — 选择器始终在对话框顶部显示当前主模型。

## 其他方法 {#alternative-methods}

### CLI 斜杠命令

在任意 `hermes chat` 会话内：

```
/model gpt-5.4 --provider openrouter             # 仅当前会话
/model gpt-5.4 --provider openrouter --global    # 同时持久化到 config.yaml
```

`--global` 与仪表板 **Change** 按钮效果相同，并额外在当前会话内原地切换模型。

### 自定义别名

为常用模型定义短名称，然后在 CLI 或任意消息平台中使用 `/model <alias>`：

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

或通过 shell 命令（简写形式，`provider/model`）：

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

然后在聊天中使用 `/model fav` 或 `/model grok`。用户别名会覆盖内置短名称（`sonnet`、`kimi`、`opus` 等）。完整参考请见[自定义模型别名](/reference/slash-commands#custom-model-aliases)。

### `hermes model` 子命令

```bash
hermes model            # 交互式提供商 + 模型选择器（切换默认值的标准方式）
```

`hermes model` 引导你选择提供商、完成认证（OAuth 流程会打开浏览器；API key 提供商会提示输入密钥），然后从该提供商的精选目录中选择具体模型。选择结果写入 `~/.hermes/config.yaml` 的 `model.provider` 和 `model.model` 字段。

如需在不启动选择器的情况下列出提供商/模型，请使用仪表板或下方的 REST 端点。查看 CLI 当前实际使用的配置：`hermes config get model` 和 `hermes status`。

### 直接编辑配置文件

编辑 `~/.hermes/config.yaml` 后重启相关服务。完整 schema 请见[配置参考](./configuration.md)。

### REST API

仪表板使用以下三个端点，可用于脚本化操作：

```bash
# 列出已认证的提供商及精选模型列表
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# 读取当前主模型及辅助任务分配
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# 设置主模型
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# 覆盖单个辅助任务
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# 将一个模型分配给所有辅助任务
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# 将所有辅助任务重置为 auto
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

session token 在启动时注入仪表板 HTML，每次服务器重启后轮换。如需对运行中的仪表板编写脚本，可从浏览器开发者工具中获取（`window.__HERMES_SESSION_TOKEN__`）。