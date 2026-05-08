---
sidebar_position: 11
sidebar_label: "插件"
title: "插件"
description: "通过插件系统为 Hermes 添加自定义工具、钩子和集成"
---

# 插件

Hermes 拥有插件系统，可以在不修改核心代码的情况下添加自定义工具、钩子和集成。

如果你想为自己、团队或某个项目创建自定义工具，这通常是正确的路径。开发者指南中的
[添加工具](/docs/developer-guide/adding-tools) 页面适用于位于 `tools/` 和 `toolsets.py` 中的
Hermes 内置核心工具。

**→ [构建 Hermes 插件](/docs/guides/build-a-hermes-plugin)** — 带有完整工作示例的分步指南。

## 快速概览

将一个目录放入 `~/.hermes/plugins/`，包含 `plugin.yaml` 和 Python 代码：

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # 清单文件
├── __init__.py      # register() — 将模式绑定到处理函数
├── schemas.py       # 工具模式（LLM 看到的内容）
└── tools.py         # 工具处理函数（调用时执行的代码）
```

启动 Hermes — 你的工具会与内置工具一起出现。模型可以立即调用它们。

### 最小工作示例

以下是一个完整的插件，添加了 `hello_world` 工具并通过钩子记录每次工具调用。

**`~/.hermes/plugins/hello-world/plugin.yaml`**

```yaml
name: hello-world
version: "1.0"
description: A minimal example plugin
```

**`~/.hermes/plugins/hello-world/__init__.py`**

```python
"""Minimal Hermes plugin — registers a tool and a hook."""

import json


def register(ctx):
    # --- Tool: hello_world ---
    schema = {
        "name": "hello_world",
        "description": "Returns a friendly greeting for the given name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to greet",
                }
            },
            "required": ["name"],
        },
    }

    def handle_hello(params, **kwargs):
        del kwargs
        name = params.get("name", "World")
        return json.dumps({"success": True, "greeting": f"Hello, {name}!"})

    ctx.register_tool(
        name="hello_world",
        toolset="hello_world",
        schema=schema,
        handler=handle_hello,
        description="Return a friendly greeting for the given name.",
    )

    # --- Hook: log every tool call ---
    def on_tool_call(tool_name, params, result):
        print(f"[hello-world] tool called: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

将这两个文件放入 `~/.hermes/plugins/hello-world/`，重启 Hermes，模型就可以立即调用 `hello_world`。该钩子在每次工具调用后打印一行日志。

项目本地插件（位于 `./.hermes/plugins/`）默认禁用。通过在启动 Hermes 前设置 `HERMES_ENABLE_PROJECT_PLUGINS=true` 来仅为受信任的仓库启用它们。

## 插件的功能

以下每个 `ctx.*` API 都可在插件的 `register(ctx)` 函数中使用。

| 功能 | 方式 |
|-----------|-----|
| 添加工具 | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| 添加钩子 | `ctx.register_hook("post_tool_call", callback)` |
| 添加斜杠命令 | `ctx.register_command(name, handler, description)` — 在 CLI 和网关会话中添加 `/name` |
| 从命令调度工具 | `ctx.dispatch_tool(name, args)` — 调用已注册的工具，自动连接父代理上下文 |
| 添加 CLI 命令 | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — 添加 `hermes <plugin> <subcommand>` |
| 注入消息 | `ctx.inject_message(content, role="user")` — 见 [注入消息](#注入消息) |
| 打包数据文件 | `Path(__file__).parent / "data" / "file.yaml"` |
| 捆绑技能 | `ctx.register_skill(name, path)` — 命名空间为 `plugin:skill`，通过 `skill_view("plugin:skill")` 加载 |
| 基于环境变量限制 | 在 plugin.yaml 中使用 `requires_env: [API_KEY]` — 在 `hermes plugins install` 时提示输入 |
| 通过 pip 分发 | `[project.entry-points."hermes_agent.plugins"]` |
| 注册网关平台（Discord、Telegram、IRC 等）| `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — 见 [添加平台适配器](/docs/developer-guide/adding-platform-adapters) |
| 注册图像生成后端 | `ctx.register_image_gen_provider(provider)` — 见 [图像生成提供者插件](/docs/developer-guide/image-gen-provider-plugin) |
| 注册上下文压缩引擎 | `ctx.register_context_engine(engine)` — 见 [上下文引擎插件](/docs/developer-guide/context-engine-plugin) |
| 注册记忆后端 | 在 `plugins/memory/<name>/__init__.py` 中继承 `MemoryProvider` — 见 [记忆提供者插件](/docs/developer-guide/memory-provider-plugin)（使用独立的发现系统）|
| 注册推理后端（LLM 提供者）| 在 `plugins/model-providers/<name>/__init__.py` 中使用 `register_provider(ProviderProfile(...))` — 见 [模型提供者插件](/docs/developer-guide/model-provider-plugin)（使用独立的发现系统）|

## 插件发现

| 来源 | 路径 | 用例 |
|--------|------|----------|
| 内置 | `<repo>/plugins/` | 随 Hermes 一起发布 — 见 [内置插件](/docs/user-guide/features/built-in-plugins) |
| 用户 | `~/.hermes/plugins/` | 个人插件 |
| 项目 | `.hermes/plugins/` | 项目特定插件（需要 `HERMES_ENABLE_PROJECT_PLUGINS=true`）|
| pip | `hermes_agent.plugins` entry_points | 分发的包 |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS 声明式安装 — 见 [Nix 设置](/docs/getting-started/nix-setup#plugins) |

后发现的来源在名称冲突时覆盖先发现的来源，因此与内置插件同名用户插件将替换它。

### 插件子类别

在每个来源中，Hermes 还识别子类别目录，将插件路由到专门的发现系统：

| 子目录 | 内容 | 发现系统 |
|---|---|---|
| `plugins/`（根目录）| 通用插件 — 工具、钩子、斜杠命令、CLI 命令、捆绑技能 | `PluginManager`（kind: `standalone` 或 `backend`）|
| `plugins/platforms/<name>/` | 网关通道适配器（`ctx.register_platform()`）| `PluginManager`（kind: `platform`，深一层）|
| `plugins/image_gen/<name>/` | 图像生成后端（`ctx.register_image_gen_provider()`）| `PluginManager`（kind: `backend`，深一层）|
| `plugins/memory/<name>/` | 记忆提供者（继承 `MemoryProvider`）| `plugins/memory/__init__.py` 中的**独立加载器**（kind: `exclusive` — 同时只激活一个）|
| `plugins/context_engine/<name>/` | 上下文压缩引擎（`ctx.register_context_engine()`）| `plugins/context_engine/__init__.py` 中的**独立加载器**（同时只激活一个）|
| `plugins/model-providers/<name>/` | LLM 提供者配置（`register_provider(ProviderProfile(...))`）| `providers/__init__.py` 中的**独立加载器**（首次 `get_provider_profile()` 调用时惰性扫描）|

位于 `~/.hermes/plugins/model-providers/<name>/` 和 `~/.hermes/plugins/memory/<name>/` 的用户插件会覆盖同名的内置插件 — `register_provider()` / `register_memory_provider()` 中后写入者胜出。放入一个目录即可替换内置插件，无需编辑仓库。

## 插件是选择性启用的（少数例外）

**通用插件和用户安装的后端默认禁用** — 发现系统会找到它们（因此它们会出现在 `hermes plugins` 和 `/plugins` 中），但任何带有钩子或工具的插件都不会加载，直到你将插件名称添加到 `~/.hermes/config.yaml` 中的 `plugins.enabled`。这阻止了第三方代码在未经你明确同意的情况下运行。

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # 可选的黑名单 — 如果名称同时出现在两者中，此项始终优先
    - noisy-plugin
```

三种切换状态的方式：

```bash
hermes plugins                    # 交互式切换（空格键勾选/取消勾选）
hermes plugins enable <name>      # 添加到允许列表
hermes plugins disable <name>     # 从允许列表移除 + 添加到禁用列表
```

`hermes plugins install owner/repo` 后，会询问 `立即启用 'name'？[y/N]` — 默认为否。使用 `--enable` 或 `--no-enable` 可跳过脚本安装中的提示。

### 允许列表不限制的内容

几类插件绕过了 `plugins.enabled` — 它们是 Hermes 内置功能的一部分，如果默认限制会导致基本功能失效：

| 插件类型 | 替代的激活方式 |
|---|---|
| **内置平台插件**（`plugins/platforms/` 下的 IRC、Teams 等）| 自动加载，使所有已发布的网关通道可用。实际通道通过 `config.yaml` 中的 `gateway.platforms.<name>.enabled` 开启。|
| **内置后端**（`plugins/image_gen/` 下的图像生成提供者等）| 自动加载，使默认后端"开箱即用"。选择通过 `config.yaml` 中的 `<category>.provider` 进行（如 `image_gen.provider: openai`）。|
| **记忆提供者**（`plugins/memory/`）| 全部发现；只有一个处于激活状态，由 `config.yaml` 中的 `memory.provider` 选择。|
| **上下文引擎**（`plugins/context_engine/`）| 全部发现；只有一个处于激活状态，由 `config.yaml` 中的 `context.engine` 选择。|
| **模型提供者**（`plugins/model-providers/`）| 所有 33 个提供者在首次 `get_provider_profile()` 调用时发现并注册。用户通过 `--provider` 或 `config.yaml` 每次选择一个。|
| **pip 安装的 `backend` 插件** | 通过 `plugins.enabled` 选择性启用（与通用插件相同）。|
| **用户安装的平台**（位于 `~/.hermes/plugins/platforms/` 下）| 通过 `plugins.enabled` 选择性启用 — 第三方网关适配器需要明确同意。|

简而言之：**内置"始终可用"的基础设施自动加载；第三方通用插件是选择性启用的。** `plugins.enabled` 允许列表专门用于限制用户放入 `~/.hermes/plugins/` 的任意代码。

### 现有用户迁移

当你升级到具有选择性启用插件功能的 Hermes 版本（配置模式 v21+）时，任何已安装在 `~/.hermes/plugins/` 下且不在 `plugins.disabled` 中的用户插件会被**自动继承**到 `plugins.enabled` 中。你现有的设置将继续工作。内置独立插件不会被继承 — 即使现有用户也必须明确选择启用。（内置平台/后端插件不需要继承，因为它们从未被限制。）

## 可用的钩子

插件可以注册以下生命周期事件的回调。完整详情、回调签名和示例请参见**[事件钩子页面](/docs/user-guide/features/hooks#plugin-hooks)**。

| 钩子 | 触发时机 |
|------|-----------|
| [`pre_tool_call`](/docs/user-guide/features/hooks#pre_tool_call) | 任何工具执行之前 |
| [`post_tool_call`](/docs/user-guide/features/hooks#post_tool_call) | 任何工具返回之后 |
| [`pre_llm_call`](/docs/user-guide/features/hooks#pre_llm_call) | 每轮一次，LLM 循环之前 — 可以返回 `{"context": "..."}` 来[将上下文注入用户消息](/docs/user-guide/features/hooks#pre_llm_call) |
| [`post_llm_call`](/docs/user-guide/features/hooks#post_llm_call) | 每轮一次，LLM 循环之后（仅成功轮次）|
| [`on_session_start`](/docs/user-guide/features/hooks#on_session_start) | 创建新会话（仅第一轮）|
| [`on_session_end`](/docs/user-guide/features/hooks#on_session_end) | 每次 `run_conversation` 调用结束 + CLI 退出处理程序 |
| [`on_session_finalize`](/docs/user-guide/features/hooks#on_session_finalize) | CLI/网关拆除活动会话（`/new`、GC、CLI 退出）|
| [`on_session_reset`](/docs/user-guide/features/hooks#on_session_reset) | 网关切换到新会话密钥（`/new`、`/reset`、`/clear`、空闲轮换）|
| [`subagent_stop`](/docs/user-guide/features/hooks#subagent_stop) | `delegate_task` 完成后每个子代理一次 |
| [`pre_gateway_dispatch`](/docs/user-guide/features/hooks#pre_gateway_dispatch) | 网关收到用户消息，在认证 + 调度之前。返回 `{"action": "skip" | "rewrite" | "allow", ...}` 来影响流程。|

## 插件类型

Hermes 有四种插件：

| 类型 | 功能 | 选择方式 | 位置 |
|------|-------------|-----------|----------|
| **通用插件** | 添加工具、钩子、斜杠命令、CLI 命令 | 多选（启用/禁用）| `~/.hermes/plugins/` |
| **记忆提供者** | 替换或增强内置记忆 | 单选（同时激活一个）| `plugins/memory/` |
| **上下文引擎** | 替换内置上下文压缩器 | 单选（同时激活一个）| `plugins/context_engine/` |
| **模型提供者** | 声明推理后端（OpenRouter、Anthropic 等）| 多注册，通过 `--provider` / `config.yaml` 选择 | `plugins/model-providers/` |

记忆提供者和上下文引擎是**提供者插件** — 每种类型同时只能激活一个。模型提供者也是插件，但许多可以同时加载；用户通过 `--provider` 或 `config.yaml` 每次选择一个。通用插件可以以任意组合启用。

## 可插拔接口 — 各自的文档在哪里

上表显示了四个插件类别，但在"通用插件"中，`PluginContext` 暴露了几个不同的扩展点 — Hermes 还接受 Python 插件系统之外的扩展（配置驱动的后端、shell 钩子命令、外部服务器等）。使用此表来查找适合你要构建的内容的正确文档：

| 想要添加… | 方式 | 编写指南 |
|---|---|---|
| LLM 可以调用的**工具** | Python 插件 — `ctx.register_tool()` | [构建 Hermes 插件](/docs/guides/build-a-hermes-plugin) · [添加工具](/docs/developer-guide/adding-tools) |
| **生命周期钩子**（LLM 前/后、会话开始/结束、工具过滤）| Python 插件 — `ctx.register_hook()` | [钩子参考](/docs/user-guide/features/hooks) · [构建 Hermes 插件](/docs/guides/build-a-hermes-plugin) |
| CLI / 网关的**斜杠命令** | Python 插件 — `ctx.register_command()` | [构建 Hermes 插件](/docs/guides/build-a-hermes-plugin) · [扩展 CLI](/docs/developer-guide/extending-the-cli) |
| `hermes <thing>` 的**子命令** | Python 插件 — `ctx.register_cli_command()` | [扩展 CLI](/docs/developer-guide/extending-the-cli) |
| 插件捆绑的**技能** | Python 插件 — `ctx.register_skill()` | [创建技能](/docs/developer-guide/creating-skills) |
| **推理后端**（LLM 提供者：OpenAI-compat、Codex、Anthropic-Messages、Bedrock）| 提供者插件 — 在 `plugins/model-providers/<name>/` 中使用 `register_provider(ProviderProfile(...))` | **[模型提供者插件](/docs/developer-guide/model-provider-plugin)** · [添加提供者](/docs/developer-guide/adding-providers) |
| **网关通道**（Discord / Telegram / IRC / Teams 等）| 平台插件 — 在 `plugins/platforms/<name>/` 中使用 `ctx.register_platform()` | [添加平台适配器](/docs/developer-guide/adding-platform-adapters) |
| **记忆后端**（Honcho、Mem0、Supermemory 等）| 记忆插件 — 在 `plugins/memory/<name>/` 中继承 `MemoryProvider` | [记忆提供者插件](/docs/developer-guide/memory-provider-plugin) |
| **上下文压缩策略** | 上下文引擎插件 — `ctx.register_context_engine()` | [上下文引擎插件](/docs/developer-guide/context-engine-plugin) |
| **图像生成后端**（DALL·E、SDXL 等）| 后端插件 — `ctx.register_image_gen_provider()` | [图像生成提供者插件](/docs/developer-guide/image-gen-provider-plugin) |
| **TTS 后端**（任何 CLI — Piper、VoxCPM、Kokoro、xtts、语音克隆脚本等）| 配置驱动 — 在 `config.yaml` 中的 `tts.providers.<name>` 下声明，类型为 `command` | [TTS 设置](/docs/user-guide/features/tts#custom-command-providers) |
| **STT 后端**（自定义 whisper 二进制文件、本地 ASR CLI）| 配置驱动 — 将 `HERMES_LOCAL_STT_COMMAND` 环境变量设置为 shell 模板 | [语音消息转录 (STT)](/docs/user-guide/features/tts#voice-message-transcription-stt) |
| **通过 MCP 的外部工具**（文件系统、GitHub、Linear、Notion、任何 MCP 服务器）| 配置驱动 — 在 `config.yaml` 中声明 `mcp_servers.<name>`，使用 `command:` / `url:`。Hermes 自动发现服务器的工具并将其与内置工具一起注册。| [MCP](/docs/user-guide/features/mcp) |
| **额外的技能来源**（自定义 GitHub 仓库、私有技能索引）| CLI — `hermes skills tap add <repo>` | [技能中心](/docs/user-guide/features/skills#skills-hub) · [发布自定义 tap](/docs/user-guide/features/skills#publishing-a-custom-skill-tap) |
| **网关事件钩子**（在 `gateway:startup`、`session:start`、`agent:end`、`command:*` 时触发）| 将 `HOOK.yaml` + `handler.py` 放入 `~/.hermes/hooks/<name>/` | [事件钩子](/docs/user-guide/features/hooks#gateway-event-hooks) |
| **Shell 钩子**（在事件时运行 shell 命令 — 通知、审计日志、桌面提醒）| 配置驱动 — 在 `config.yaml` 中的 `hooks:` 下声明 | [Shell 钩子](/docs/user-guide/features/hooks#shell-hooks) |

:::note
并非一切都是 Python 插件。某些扩展接口有意使用**配置驱动的 shell 命令**（TTS、STT、shell 钩子），使你已有的任何 CLI 无需编写 Python 就能成为插件。其他是**外部服务器**（MCP），代理连接到它们并自动注册工具。还有一些是**即插即用目录**（网关钩子），有自己的清单格式。选择适合你用例的集成风格的正确接口；上表中的编写指南各自涵盖占位符、发现机制和示例。
:::

## NixOS 声明式插件

在 NixOS 上，可以通过模块选项声明式安装插件 — 无需 `hermes plugins install`。完整详情请参见 **[Nix 设置指南](/docs/getting-started/nix-setup#plugins)**。

```nix
services.hermes-agent = {
  # 目录插件（包含 plugin.yaml 的源代码树）
  extraPlugins = [ (pkgs.fetchFromGitHub { ... }) ];
  # 入口点插件（pip 包）
  extraPythonPackages = [ (pkgs.python312Packages.buildPythonPackage { ... }) ];
  # 在配置中启用
  settings.plugins.enabled = [ "my-plugin" ];
};
```

声明式插件以 `nix-managed-` 前缀的符号链接安装 — 它们与手动安装的插件共存，在从 Nix 配置中移除时自动清理。

## 管理插件

```bash
hermes plugins                               # 统一的交互式界面
hermes plugins list                          # 表格：已启用 / 已禁用 / 未启用
hermes plugins install user/repo             # 从 Git 安装，然后提示是否启用？[y/N]
hermes plugins install user/repo --enable    # 安装并启用（无提示）
hermes plugins install user/repo --no-enable # 安装但保持禁用（无提示）
hermes plugins update my-plugin              # 拉取最新版本
hermes plugins remove my-plugin              # 卸载
hermes plugins enable my-plugin              # 添加到允许列表
hermes plugins disable my-plugin             # 从允许列表移除 + 添加到禁用列表
```

### 交互式界面

不带参数运行 `hermes plugins` 会打开一个组合交互界面：

```
Plugins
  ↑↓ navigate  SPACE toggle  ENTER configure/confirm  ESC done

  General Plugins
 → [✓] my-tool-plugin — Custom search tool
   [ ] webhook-notifier — Event hooks
   [ ] disk-cleanup — Auto-cleanup of ephemeral files [bundled]

  Provider Plugins
     Memory Provider          ▸ honcho
     Context Engine           ▸ compressor
```

- **通用插件部分** — 复选框，使用空格键切换。勾选 = 在 `plugins.enabled` 中，未勾选 = 在 `plugins.disabled` 中（明确关闭）。
- **提供者插件部分** — 显示当前选择。按 ENTER 进入单选器，选择一个激活的提供者。
- 内置插件以 `[bundled]` 标签出现在同一列表中。

提供者插件选择保存在 `config.yaml` 中：

```yaml
memory:
  provider: "honcho"      # 空字符串 = 仅内置

context:
  engine: "compressor"    # 默认内置压缩器
```

### 已启用 vs 已禁用 vs 未启用

插件处于三种状态之一：

| 状态 | 含义 | 在 `plugins.enabled` 中？| 在 `plugins.disabled` 中？|
|---|---|---|---|
| `enabled` | 下次会话加载 | 是 | 否 |
| `disabled` | 明确关闭 — 即使也在 `enabled` 中也不会加载 | （无关）| 是 |
| `not enabled` | 已发现但从未选择启用 | 否 | 否 |

新安装或内置插件的默认状态是 `not enabled`。`hermes plugins list` 显示所有三种不同的状态，让你可以区分什么是被明确关闭的、什么只是在等待启用。

在运行中的会话中，`/plugins` 显示当前加载了哪些插件。

## 注入消息

插件可以使用 `ctx.inject_message()` 将消息注入到活动会话中：

```python
ctx.inject_message("New data arrived from the webhook", role="user")
```

**签名：** `ctx.inject_message(content: str, role: str = "user") -> bool`

工作原理：

- 如果代理处于**空闲状态**（等待用户输入），消息排队为下一个输入并启动新一轮。
- 如果代理处于**进行中**（正在运行），消息会中断当前操作 — 就像用户输入新消息并按回车一样。
- 对于非 `"user"` 角色，内容以 `[role]` 为前缀（如 `[system] ...`）。
- 如果消息成功排队返回 `True`，如果没有可用的 CLI 引用返回 `False`（如在网关模式下）。

这使得远程控制查看器、消息桥接器或 webhook 接收器等插件可以从外部源将消息注入到会话中。

:::note
`inject_message` 仅在 CLI 模式下可用。在网关模式下，没有 CLI 引用，该方法返回 `False`。
:::

有关处理函数约定、模式格式、钩子行为、错误处理和常见错误的完整信息，请参见 **[完整指南](/docs/guides/build-a-hermes-plugin)**。
