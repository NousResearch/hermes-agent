---
title: Codex App-Server 运行时（可选）
sidebar_label: Codex App-Server 运行时
---

# Codex App-Server 运行时

Hermes 可以选择将 `openai/*` 和 `openai-codex/*` 的轮次交由 [Codex CLI app-server](https://github.com/openai/codex) 处理，而不是运行自己的工具循环。启用后，终端命令、文件编辑、沙箱隔离以及 MCP 工具调用均在 Codex 的运行时内执行——Hermes 成为其外层 shell（会话数据库、斜杠命令、gateway、记忆与技能审查）。

此功能**仅限手动启用**。除非你主动切换该标志，否则 Hermes 的默认行为不变。Hermes 不会自动将你路由到此运行时。

## 为什么使用

- 通过 Codex CLI 使用的相同认证流程，使用你的 **ChatGPT 订阅**运行 OpenAI agent 轮次（无需 API 密钥）。
- 使用 **Codex 自带的工具集和沙箱**——`shell` 用于终端/读/写/搜索，`apply_patch` 用于结构化编辑，`update_plan` 用于规划，全部在 seatbelt/landlock 沙箱内运行。
- **原生 Codex 插件**——Linear、GitHub、Gmail、Calendar、Canva 等——通过 `codex plugin` 安装后，会自动迁移并在你的 Hermes 会话中激活。
- **Hermes 的丰富工具一并可用**——web_search、web_extract、浏览器自动化、视觉、图像生成、技能和 TTS 通过 MCP 回调提供。Codex 会回调 Hermes 获取其自身没有内置的工具。
- **记忆与技能提示持续生效**——Codex 的事件被投影为 Hermes 的消息格式，使自我改进循环看到正常的对话记录。

## 模型实际拥有哪些工具

这是大多数用户最想提前了解的部分。当此运行时开启时，执行你的轮次的模型拥有三个独立的工具来源：

### 1. Codex 内置工具集（始终开启）

这些工具随 `codex app-server` 本身一起提供——无需 Hermes 介入，无需 MCP，无需插件。运行时启动后，以下五个工具立即可用：

- **`shell`** — 在沙箱内运行任意 shell 命令。模型通过此工具读取文件（`cat`、`head`、`tail`）、写入文件（`echo > foo`、heredoc）、搜索文件（`find`、`rg`、`grep`）、浏览目录（`ls`、`cd`）、运行构建、管理进程，以及其他任何你在 bash 中能做的事。
- **`apply_patch`** — 以 Codex 的 patch 格式应用结构化的多文件差异。模型将此工具用于非简单的代码编辑（添加函数、跨文件重构）；单次写入仍可使用 shell heredoc。
- **`update_plan`** — Codex 的内部待办/计划跟踪器。等同于 Hermes 的 `todo` 工具，但完全在 Codex 运行时内部管理。
- **`view_image`** — 将本地图像文件加载到对话中，使模型能够查看它。
- **`web_search`** — 配置后 Codex 拥有自己的内置网络搜索。Hermes 也通过下方的回调暴露 `web_search`（基于 Firecrawl）；模型会选择其偏好的那个。

因此，**任何你通过终端完成的操作——读/写/搜索/查找/运行——Codex 都能原生处理**。沙箱配置文件（启用运行时时默认为 `:workspace`）控制可写范围。

### 2. 原生 Codex 插件（从你的 `codex plugin` 安装中自动迁移）

启用运行时时，Hermes 会查询 Codex 的 `plugin/list` RPC，并为你已安装的每个插件写入一条 `[plugins."<name>@openai-curated"]` 配置项。插件本身由 Codex 管理，并通过 Codex 自己的 UI 完成一次性授权。

示例（OpenClaw 帖子中被称为"值得录制视频"的那些）：

- **Linear** — 查找/更新 issue
- **GitHub** — 搜索代码、查看 PR、评论
- **Gmail** — 读取/发送邮件
- **Google Calendar** — 创建/查找日程
- **Outlook 日历/邮件** — 通过 Microsoft 连接器提供相同功能
- **Canva** — 设计生成
- ……以及其他你通过 `codex plugin marketplace add openai-curated` + `codex plugin install ...` 安装的插件

**未迁移的内容：**
- 你尚未安装的插件——请先在 Codex 中安装。
- ChatGPT 应用市场条目（`app/list`）——这些已通过你的账户认证在 Codex 内部启用。

### 3. Hermes 工具回调（MCP server，注册在 `~/.codex/config.toml` 中）

Hermes 将自身注册为 MCP server，以便 Codex 能够回调获取 Codex 自身未内置的工具。通过回调可用的工具：

- **`web_search`** / **`web_extract`** — 基于 Firecrawl；对于结构化内容，通常比直接抓取更干净。
- **`browser_navigate` / `browser_click` / `browser_type` / `browser_press` / `browser_snapshot` / `browser_scroll` / `browser_back` / `browser_get_images` / `browser_console` / `browser_vision`** — 通过 Camofox 或 Browserbase 实现完整的浏览器自动化。
- **`vision_analyze`** — 调用独立的视觉模型检查图像（与 Codex 的 `view_image` 不同，后者是将图像加载到对话中）。
- **`image_generate`** — 通过 Hermes 的 image_gen 插件链生成图像。
- **`skill_view` / `skills_list`** — 读取 Hermes 的技能库。
- **`text_to_speech`** — 通过 Hermes 配置的提供商进行 TTS。

当模型需要其中某个工具时，Codex 通过 stdio MCP 生成 `hermes_tools_mcp_server` 子进程，调用通过 `model_tools.handle_function_call()` 分发（与 Hermes 默认运行时的代码路径相同），结果像其他 MCP 响应一样返回给 Codex。

### 此运行时上不可用的工具

以下四个 Hermes 工具需要运行中的 AIAgent 上下文（循环中间状态）才能分发，无状态的 MCP 回调无法驱动它们。需要这些工具时，请切换回默认运行时（`/codex-runtime auto`）：

- **`delegate_task`** — 生成子 agent
- **`memory`** — Hermes 的持久记忆存储
- **`session_search`** — 跨会话搜索
- **`todo`** — Hermes 的待办存储（Codex 的 `update_plan` 是运行时内的等效工具）

## 工作流功能（`/goal`、kanban、cron）

### `/goal`（Ralph 循环）

**在此运行时上可用。** 目标以会话 id 为键持久化在 `state_meta` 中，续接提示通过 `run_conversation()` 作为普通用户消息回传，Codex 原生执行下一轮次。目标判断器通过辅助客户端运行（在 config.yaml 中通过 `auxiliary.goal_judge` 配置），与当前活跃的运行时无关。判断器的"受阻，需要用户输入"裁决是 Codex 卡在审批时的干净退出路径。

**需要注意的一点：** 每个续接提示都是一次全新的 Codex 轮次，这意味着 Codex 会从头重新评估命令审批策略。如果你在执行包含大量写操作的长期目标，预期会看到比单次会话内任务更多的审批提示。设置 `default_permissions = ":workspace"`（启用运行时时 Hermes 会自动设置）可避免简单的工作区写操作触发提示。

### Kanban（多 agent 工作树分发）

**在此运行时上可用，但有一个细微依赖。** Kanban 分发器将每个 worker 生成为独立的 `hermes chat -q` 子进程，该子进程读取用户配置——这意味着如果全局设置了 `model.openai_runtime: codex_app_server`，worker 也会在 Codex 运行时上启动。

Codex 运行时 worker 内可用的功能：
- Codex 完整工具集（shell、apply_patch、update_plan、view_image、web_search）——worker 原生完成实际任务
- 已迁移的 Codex 插件——Linear、GitHub 等
- 用于 browser_*、vision、image_gen、技能、TTS 的 Hermes 工具回调

通过 MCP 回调同样可用的功能：
- **`kanban_complete` / `kanban_block` / `kanban_comment` / `kanban_heartbeat`** — worker 交接工具。这些工具从环境变量中读取 `HERMES_KANBAN_TASK`（由分发器设置），正确进行访问控制，并写入由 `HERMES_KANBAN_DB` 固定的每个看板 SQLite 数据库。若回调中没有这些工具，此运行时上的 worker 可以完成任务但无法汇报，会一直挂起直到分发器超时。
- **`kanban_show` / `kanban_list`** — 只读看板查询，供 worker 检查自身上下文。
- **`kanban_create` / `kanban_unblock` / `kanban_link`** — 仅限编排器的操作。供运行在 Codex 运行时上、需要分发新任务的编排器 agent 使用。

Kanban 工具通过分发器设置的 `HERMES_KANBAN_TASK` 环境变量进行访问控制——该变量会传播到 Codex 子进程（Codex 继承环境变量），再从那里传播到生成的 `hermes-tools` MCP server 子进程。因此工具能看到正确的任务 id 并正确进行访问控制。对于 Codex app-server worker，当 `HERMES_KANBAN_TASK` 存在时，Hermes 还会传入精细的 app-server 沙箱覆盖配置：保持 `workspace-write` 沙箱，将**看板数据库目录以及分发器固定的所有 Kanban 路径**作为额外可写根目录添加（`HERMES_KANBAN_WORKSPACES_ROOT`、`HERMES_KANBAN_WORKSPACE`、旧版 `HERMES_KANBAN_ROOT`——去重，数据库目录优先），并默认禁用网络。这避免了脆弱的 `:danger-no-sandbox` 变通方案，同时允许 `kanban_complete` / `kanban_block` 更新看板数据库，**并且**允许 worker 在数据库目录之外的工作区挂载点下写入报告/产物（例如独立驱动器上的 `/media/.../kanban-workspaces/...`——[issue #27941](https://github.com/NousResearch/hermes-agent/issues/27941)）。

### Cron 任务

**尚未经过专项测试。** Cron 任务通过 `cronjob` → `AIAgent.run_conversation` 运行，与 CLI 的代码路径相同。如果 cron 任务的配置中有 `openai_runtime: codex_app_server`，它将在 Codex 上运行。相同的工具可用性规则适用——Codex 内置工具 + 插件 + MCP 回调可用，agent 循环工具（delegate_task、memory、session_search、todo）不可用。如果你的 cron 任务依赖这些工具，请将 cron 限定在使用默认运行时的配置文件中。

## 权衡对比

|  | Hermes 默认运行时 | Codex app-server（可选启用） |
|---|---|---|
| `delegate_task` 子 agent | 是 | 不可用——需要 agent 循环上下文 |
| `memory`、`session_search`、`todo` | 是 | 不可用——需要 agent 循环上下文 |
| `web_search`、`web_extract` | 是 | 是（通过 MCP 回调） |
| 浏览器自动化（Camofox/Browserbase） | 是 | 是（通过 MCP 回调） |
| `vision_analyze`、`image_generate` | 是 | 是（通过 MCP 回调） |
| `skill_view`、`skills_list` | 是 | 是（通过 MCP 回调） |
| `text_to_speech` | 是 | 是（通过 MCP 回调） |
| Codex `shell`（终端/读/写/搜索/查找/运行） | — | 是（Codex 内置） |
| Codex `apply_patch`（结构化多文件编辑） | — | 是（Codex 内置） |
| Codex `update_plan`（运行时内待办） | — | 是（Codex 内置） |
| Codex `view_image`（将图像加载到对话） | — | 是（Codex 内置） |
| Codex 沙箱（seatbelt/landlock，配置文件） | — | 是（Codex 内置） |
| ChatGPT 订阅认证 | — | 是（通过 `openai-codex` 提供商） |
| 原生 Codex 插件（Linear、GitHub 等） | — | 是（自动迁移） |
| 用户 MCP server | 是 | 是（自动迁移到 Codex） |
| 记忆 + 技能审查（后台） | 是 | 是（通过事件投影） |
| 多轮对话 | 是 | 是 |
| `/goal`（Ralph 循环） | 是 | 是 |
| Kanban worker 分发 | 是 | 是（通过回调） |
| Kanban 编排器工具 | 是 | 是（通过回调） |
| 所有 gateway 平台 | 是 | 是 |
| 非 OpenAI 提供商 | 是 | 不适用——仅限 OpenAI/Codex |

## 前提条件

1. **已安装 Codex CLI：**
   ```bash
   npm i -g @openai/codex
   codex --version   # 0.130.0 或更新版本
   ```
2. **Codex OAuth 登录。** Codex 子进程读取 `~/.codex/auth.json`。有两种方式填充它：
   ```bash
   codex login                  # 将 token 写入 ~/.codex/auth.json
   ```
   Hermes 自己的 `hermes auth login codex` 写入 `~/.hermes/auth.json`——那是独立的会话。**如果你还没有运行过 `codex login`，请单独运行它。**

3. **（可选）安装你想要的 Codex 插件。** 启用运行时时，Hermes 会自动迁移你已通过 Codex CLI 安装的所有精选插件：
   ```bash
   codex plugin marketplace add openai-curated
   # 然后通过 Codex 的 TUI 安装 Linear / GitHub / Gmail 等
   ```
   Hermes 会自动发现它们并将 `[plugins."<name>@openai-curated"]` 条目写入 `~/.codex/config.toml`。

## 启用

在 Hermes 会话中：

```
/codex-runtime codex_app_server
```

该命令会：
- 验证 `codex` CLI 是否已安装（若未安装则阻止并提示安装方法）。
- 将 `model.openai_runtime: codex_app_server` 持久化到你的 config.yaml。
- 将用户 MCP server 从 `~/.hermes/config.yaml` 迁移到 `~/.codex/config.toml`。
- **发现并迁移已安装的原生 Codex 插件**（Linear、GitHub、Gmail、Calendar、Canva 等），通过查询 Codex 的 `plugin/list` RPC 实现。
- **将 Hermes 自身的工具注册为 MCP server**，以便 Codex 子进程能够回调获取 Codex 未内置的工具。
- **写入 `default_permissions = ":workspace"`**，使沙箱允许在工作区内写入，无需对每次操作进行提示。
- 告知你迁移了哪些内容。在**下一个**会话生效——当前缓存的 agent 保持之前的运行时，以保持 prompt 缓存有效。

同义命令：`/codex-runtime on`、`/codex-runtime off`、`/codex-runtime auto`。

查看当前状态而不做任何更改：
```
/codex-runtime
```

你也可以在 `~/.hermes/config.yaml` 中手动设置：
```yaml
model:
  openai_runtime: codex_app_server   # 默认值为 "auto"（= Hermes 运行时）
```

## 自我改进循环（记忆 + 技能提示）

Hermes 的后台自我改进在计数器达到阈值时触发：

- 每 10 个用户 prompt（提示词）→ 一个分叉的审查 agent 查看对话，决定是否有内容应保存到记忆中。
- 单次轮次内每 10 次工具迭代 → 同样的逻辑，但针对技能（`skill_manage` 写入）。

**两者在 Codex 运行时上均持续生效。** Codex 路径将每个已完成的 `commandExecution` / `fileChange` / `mcpToolCall` / `dynamicToolCall` 事件项投影为合成的 `assistant tool_call` + `tool` 结果消息，因此审查运行时看到的格式与在默认 Hermes 运行时上看到的相同。

连接方式保持等效：

| | 默认运行时 | Codex 运行时 |
|---|---|---|
| `_turns_since_memory` 递增 | 每个用户 prompt，在 run_conversation 预循环中 | 相同代码路径，在提前返回之前 |
| `_iters_since_skill` 递增 | 在聊天补全循环的每次工具迭代中 | 通过 Codex 轮次返回后的 `turn.tool_iterations` |
| 记忆触发（`_turns_since_memory >= _memory_nudge_interval`） | 在预循环中计算，响应后触发 | 在预循环中计算，传递给 Codex 辅助函数 |
| 技能触发（`_iters_since_skill >= _skill_nudge_interval`） | 在循环结束后计算 | 在 Codex 轮次结束后计算 |
| `_spawn_background_review(messages_snapshot=..., review_memory=..., review_skills=...)` | 任一触发器触发时调用 | 任一触发器触发时以相同方式调用 |

一个细节：审查分叉本身需要调用 Hermes 的 agent 循环工具（`memory`、`skill_manage`），这需要 Hermes 自身的分发。因此，当父 agent 处于 `codex_app_server` 时，审查分叉会**降级为 `codex_responses`**——相同的 OAuth 凭据，相同的 `openai-codex` 提供商，但直接与 OpenAI 的 Responses API 通信，使 Hermes 拥有循环控制权，agent 循环工具得以正常工作。这对用户不可见。

最终效果：启用 Codex 运行时后，你的记忆 + 技能提示计数器与之前完全一样持续触发。

## 审批流程

Codex 在执行命令或应用 patch 之前会请求审批。这些请求会被转换为 Hermes 标准的"危险命令"提示：

```
╭───────────────────────────────────────╮
│ Dangerous Command                     │
│                                       │
│ /bin/bash -lc 'echo hello > foo.txt'  │
│                                       │
│ ❯ 1. Allow once                       │
│   2. Allow for this session           │
│   3. Deny                             │
│                                       │
│ Codex requests exec in /your/cwd      │
╰───────────────────────────────────────╯
```

- **Allow once** → 批准此单次命令。
- **Allow for this session** → Codex 不会再对类似命令重复提示。
- **Deny** → 命令被拒绝；Codex 以只读模式继续运行。

对于 `apply_patch`（文件编辑）审批，当 Codex 通过对应的 `fileChange` 事件项提供数据时，Hermes 会显示变更摘要（`1 add, 1 update: /tmp/new.py, /tmp/old.py`）。

## 权限配置文件

Codex 有三个内置权限配置文件：
- `:read-only` — 禁止写入；每条 shell 命令都需要审批
- `:workspace` — 允许在当前工作区内写入而无需提示（启用运行时时 Hermes 的默认值）
- `:danger-no-sandbox` — 完全不使用沙箱（除非你清楚其含义，否则不要使用）

你可以在 Hermes 管理块之外的 `~/.codex/config.toml` 中覆盖默认值：

```toml
default_permissions = ":read-only"
```

（只要你的覆盖配置位于 `# managed by hermes-agent` 标记之外，Hermes 在重新迁移时会保留它。）

## 辅助任务与 ChatGPT 订阅 token 消耗

当此运行时与 `openai-codex` 提供商一起开启时，**辅助任务（标题生成、上下文压缩、视觉自动检测、后台自我改进审查分叉）默认也会通过你的 ChatGPT 订阅流转**，因为 Hermes 的辅助客户端在没有设置每任务覆盖时使用主提供商/模型。

这并非 `codex_app_server` 特有——现有的 `codex_responses` 路径也是如此——但在这里更为明显，因为你是在明确选择订阅计费。

要将特定辅助任务路由到更便宜/不同的模型，请在 `~/.hermes/config.yaml` 中设置显式覆盖：

```yaml
auxiliary:
  title_generation:
    provider: openrouter
    model: google/gemini-3-flash-preview
  compression:
    provider: openrouter
    model: google/gemini-3-flash-preview
  vision:
    provider: openrouter
    model: google/gemini-3-flash-preview
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

自我改进审查分叉通过 `_current_main_runtime()` 继承主运行时，Hermes 会自动将其从 `codex_app_server` 降级为 `codex_responses`（以便分叉能够实际调用 `memory` 和 `skill_manage`——Hermes 自身的 agent 循环工具）。除非你已将辅助任务路由到其他地方，否则该分叉仍使用你的订阅认证。

## 安全编辑 `~/.codex/config.toml`

Hermes 将其管理的所有内容包裹在两个标记注释之间：

```toml
# managed by hermes-agent — `hermes codex-runtime migrate` regenerates this section
default_permissions = ":workspace"
[mcp_servers.filesystem]
...
[plugins."github@openai-curated"]
...
# end hermes-agent managed section
```

该块**之外**的内容归你所有。重新运行迁移（通过 `/codex-runtime codex_app_server` 或每次切换运行时时）会原地替换管理块，但完整保留其上下方的用户内容。这意味着你可以：

- 添加 Hermes 不知道的自定义 MCP server
- 将 `default_permissions` 覆盖为 `:read-only`（如果你希望被提示）
- 配置仅 Codex 使用的选项（model、providers、otel 等）
- 在 `[permissions.<name>]` 表中添加用户自定义权限配置文件

你在管理块**内部**添加的任何内容都会在下次迁移时被覆盖。如果你需要修改管理块中的某项配置，请提交 issue，我们会添加相应的开关。

## 多配置文件 / 多租户设置

默认情况下，无论哪个 Hermes 配置文件处于活跃状态，Hermes 都将 Codex 子进程指向 `~/.codex/`。这意味着 `hermes -p work` 和 `hermes -p personal` 共享相同的 Codex 认证、插件和配置。对大多数用户来说这是正确的行为——与直接运行 `codex` CLI 的效果一致。

如果你需要按配置文件隔离 Codex（独立的认证、独立的已安装插件、独立的配置），请为每个配置文件显式设置 `CODEX_HOME`。最简洁的方式是指向你 `HERMES_HOME` 下的某个目录：

```bash
# 在 work 配置文件中，你可以这样包装 hermes：
CODEX_HOME=~/.hermes/profiles/work/codex hermes chat
```

你需要在设置了该 `CODEX_HOME` 的情况下重新运行一次 `codex login`，以便 OAuth token 落入配置文件范围的位置。之后，`hermes -p work` 将在隔离的 Codex 状态下运行。

我们不自动限定此范围，因为移动现有用户的 `~/.codex/` 会静默地使其 Codex CLI 认证失效——任何已运行过 `codex login` 的用户都需要重新认证。选择加入比给用户带来意外更安全。

## HOME 环境变量透传

Hermes 在生成 Codex app-server 子进程时**不会**重写 `HOME`（我们使用 `os.environ.copy()`，仅覆盖 `CODEX_HOME` 和 `RUST_LOG`）。这意味着：

- Codex 通过其 `shell` 工具运行的命令能看到真实的用户 `HOME`，并能正确找到 `~/.gitconfig`、`~/.gh/`、`~/.aws/`、`~/.npmrc` 等。
- Codex 的内部状态通过 `CODEX_HOME` 保持隔离（默认指向 `~/.codex/`）。

这与 OpenClaw 在早期实验后得出的边界一致：隔离 Codex 的状态，保持用户主目录不变。（参见 openclaw/openclaw#81562。）

## MCP server 迁移

Hermes 的 `mcp_servers` 配置会自动转换为 Codex 所需的 TOML 格式。迁移在每次启用运行时时运行，且是幂等的——重新运行会替换管理块，但保留用户编辑的 Codex 配置。

转换内容：

| Hermes（`config.yaml`） | Codex（`config.toml`） |
|---|---|
| `command` + `args` + `env` | stdio transport |
| `url` + `headers` | streamable_http transport |
| `timeout` | `tool_timeout_sec` |
| `connect_timeout` | `startup_timeout_sec` |
| `enabled: false` | `enabled = false` |

未迁移的内容：
- Hermes 特有的键，如 `sampling`（Codex 的 MCP 客户端没有等效项——这些会被丢弃并附带每个 server 的警告）。

## 原生 Codex 插件迁移

通过 `codex plugin` 安装的插件（Linear、GitHub、Gmail、Calendar、Canva 等）通过 Codex 的 `plugin/list` RPC 被发现。对于每个 `installed: true` 的插件，Hermes 会写入一个 `[plugins."<name>@openai-curated"]` 块，在你的 Hermes 会话中启用它。

这意味着：当你的朋友说"我在 Codex CLI 中设置了 Calendar 和 GitHub"，他们启用 Hermes 的 Codex 运行时后，Hermes 会自动激活这些插件。无需重新配置。

**未迁移的内容：**
- 你尚未安装的插件——请先在 Codex 中安装。
- Codex 报告 `availability != AVAILABLE` 的插件（安装损坏、OAuth 过期、已从市场下架等）。这些会被跳过，以避免写入激活时会失败的配置。
- ChatGPT 应用市场条目（每账户的 `app/list` 结果——这些已通过你的账户认证在 Codex 内部启用）。
- 插件 OAuth——你在 Codex 本身中对每个插件授权一次；Hermes 不接触凭据。

## Hermes 工具回调（新 MCP server）

Codex 的内置工具集涵盖 shell/文件操作/patch，但没有网络搜索、浏览器自动化、视觉、图像生成等功能。为了在 Codex 轮次中保持这些工具可用，Hermes 在 `~/.codex/config.toml` 中将自身注册为 MCP server：

```toml
[mcp_servers.hermes-tools]
command = "/path/to/python"
args = ["-m", "agent.transports.hermes_tools_mcp_server"]
env = { HERMES_HOME = "/your/.hermes", PYTHONPATH = "...", HERMES_QUIET = "1" }
startup_timeout_sec = 30.0
tool_timeout_sec = 600.0
```

当模型调用 `web_search`（或其他暴露的 Hermes 工具）时，Codex 通过 stdio 生成 `hermes_tools_mcp_server` 子进程，请求通过 `model_tools.handle_function_call()` 分发，结果像其他 MCP 响应一样投影回 Codex。

**通过回调可用的工具：** `web_search`、`web_extract`、`browser_navigate`、`browser_click`、`browser_type`、`browser_press`、`browser_snapshot`、`browser_scroll`、`browser_back`、`browser_get_images`、`browser_console`、`browser_vision`、`vision_analyze`、`image_generate`、`skill_view`、`skills_list`、`text_to_speech`。

**不可用的工具：** `delegate_task`、`memory`、`session_search`、`todo`。这些工具需要运行中的 AIAgent 上下文（循环中间状态）才能分发，无状态的 MCP 回调无法驱动它们。需要这些工具时，请使用默认 Hermes 运行时（`/codex-runtime auto`）。

## 禁用

随时切换回来：

```
/codex-runtime auto
```

在下一个会话生效。Codex 管理块保留在 `~/.codex/config.toml` 中，以便你之后重新启用时不会丢失配置——如果你希望，也可以手动删除它。

## 限制

此运行时为**可选启用的 beta 功能**。以下功能在 Hermes Agent 2026.5 + Codex CLI 0.130.0 上已验证可用：

- 多轮对话
- 通过 Hermes UI 进行 `commandExecution` 和 `fileChange`（apply_patch）审批
- MCP 工具调用（已针对 `@modelcontextprotocol/server-filesystem` 和新的 `hermes-tools` 回调验证）
- 原生 Codex 插件迁移（已针对 Linear / GitHub / Calendar 清单验证）
- 拒绝/取消路径
- 开关切换循环
- 记忆和技能提示计数器（已通过集成测试实时验证）
- 通过 Codex 使用 Hermes web_search（已实时验证："OpenAI Codex CLI – Getting Started" 端到端返回结果）

已知限制：

- **Hermes 认证和 Codex 认证是独立的会话。** 为获得最佳体验，你需要同时运行 `codex login` 和 `hermes auth login codex`（运行时使用 Codex 的会话进行 LLM 调用）。这是 Hermes `_import_codex_cli_tokens` 中的有意设计——Hermes 不会与 Codex CLI 共享 OAuth 状态，以避免在 token 刷新时相互覆盖。
- **`delegate_task`、`memory`、`session_search`、`todo` 在此运行时上不可用。** 它们需要运行中的 AIAgent 上下文，无状态的 MCP 回调无法提供。需要这些工具时，请使用 `/codex-runtime auto`。
- **当 Codex 未跟踪变更集时，审批提示中没有内联 patch 预览。** Codex 的 `fileChange` 审批参数并不总是携带变更集。Hermes 会尽可能从对应的 `item/started` 通知中缓存数据，但如果审批在事件项流式传输完成之前到达，提示会回退到 Codex 提供的 `reason`。
- **亚秒级取消无法保证。** 流式传输中途的中断（Codex 响应时按 Ctrl+C）通过 `turn/interrupt` 发送，但如果 Codex 已经刷新了最终消息，你仍会收到该响应。

如果你发现 bug，请[提交 issue](https://github.com/NousResearch/hermes-agent/issues)，附上 `hermes logs --since 5m` 的输出。在标题中注明 `codex-runtime` 以便于分类处理。

## 架构

```
                ┌─── Hermes shell (CLI / TUI / gateway) ───┐
                │  sessions DB · slash commands · memory   │
                │  & skill review · cron · session pickers │
                └──┬──────────────────────────────────────┬┘
                   │ user_message               final     │
                   ▼                            text +    │
        ┌──────────────────────────────────┐   projected  │
        │  AIAgent.run_conversation()       │   messages   │
        │   if api_mode == codex_app_server │              │
        │     → CodexAppServerSession       │              │
        │   else: chat_completions / codex_responses (default)
        └────┬─────────────────────────────┘              │
             │ JSON-RPC over stdio                        │
             ▼                                            │
        ┌──────────────────────────────────┐              │
        │  codex app-server (subprocess)    │──────────────┘
        │   thread/start, turn/start        │
        │   item/* notifications            │
        │   shell + apply_patch + update_plan│
        │   view_image + sandbox            │
        │   ┌─────────────────────────┐     │
        │   │  MCP client             │     │
        │   │  ├─ user MCP servers    │     │
        │   │  ├─ native plugins      │     │
        │   │  │   (linear, github,   │     │
        │   │  │    gmail, calendar,  │     │
        │   │  │    canva, ...)       │     │
        │   │  └─ hermes-tools ───────┼─────────────────┐
        │   │       (callback to     │     │           │
        │   │        Hermes' richer  │     │           │
        │   │        tools)          │     │           │
        │   └─────────────────────────┘     │           │
        └──────────────────────────────────┘           │
                                                        │
                                                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │  hermes_tools_mcp_server.py (subprocess on demand)        │
        │   web_search, web_extract, browser_*, vision_analyze,    │
        │   image_generate, skill_view, skills_list, text_to_speech│
        └──────────────────────────────────────────────────────────┘
```

有关实现细节，请参阅 [PR #24182](https://github.com/NousResearch/hermes-agent/pull/24182) 和 [Codex app-server 协议 README](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)。