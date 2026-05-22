---
sidebar_position: 11
sidebar_label: "Plugins"
title: "Plugins"
description: "通过插件系统为 Hermes 添加自定义工具、hook 和集成"
---

# Plugins

Hermes 拥有一套插件系统，允许你在不修改核心代码的情况下添加自定义工具、hook 和集成。

如果你想为自己、团队或某个项目创建自定义工具，这通常是最合适的路径。开发者指南中的
[Adding Tools](/developer-guide/adding-tools) 页面适用于内置的 Hermes
核心工具，这些工具存放在 `tools/` 和 `toolsets.py` 中。

**→ [Build a Hermes Plugin](/guides/build-a-hermes-plugin)** — 包含完整工作示例的逐步指南。

## 快速概览

将一个目录放入 `~/.hermes/plugins/`，其中包含 `plugin.yaml` 和 Python 代码：

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # 清单文件
├── __init__.py      # register() — 将 schema 绑定到 handler
├── schemas.py       # 工具 schema（LLM 看到的定义）
└── tools.py         # 工具 handler（被调用时执行的逻辑）
```

启动 Hermes — 你的工具会与内置工具一起出现。模型可以立即调用它们。

### 最小可运行示例

以下是一个完整的插件，它添加了一个 `hello_world` 工具，并通过 hook 在每次工具调用后记录日志。

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

将两个文件放入 `~/.hermes/plugins/hello-world/`，重启 Hermes，模型就可以立即调用 `hello_world`。该 hook 会在每次工具调用后打印一条日志。

项目本地插件位于 `./.hermes/plugins/`，默认处于禁用状态。仅在受信任的仓库中通过设置 `HERMES_ENABLE_PROJECT_PLUGINS=true` 来启用它们。

## Plugins 能做什么

以下所有 `ctx.*` API 都可在 plugin 的 `register(ctx)` 函数中使用。

| 能力 | 方式 |
|-----------|-----|
| 添加工具 | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| 添加 hook | `ctx.register_hook("post_tool_call", callback)` |
| 添加 slash 命令 | `ctx.register_command(name, handler, description)` — 在 CLI 和 gateway 会话中添加 `/name` |
| 从命令中调度工具 | `ctx.dispatch_tool(name, args)` — 调用已注册的工具，并自动接入 parent-agent 上下文 |
| 添加 CLI 命令 | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — 添加 `hermes <plugin> <subcommand>` |
| 注入消息 | `ctx.inject_message(content, role="user")` — 参见 [Injecting Messages](#injecting-messages) |
| 携带数据文件 | `Path(__file__).parent / "data" / "file.yaml"` |
| 打包 skill | `ctx.register_skill(name, path)` — 以 `plugin:skill` 为命名空间，通过 `skill_view("plugin:skill")` 加载 |
| 依赖环境变量 | 在 plugin.yaml 中设置 `requires_env: [API_KEY]` — 在 `hermes plugins install` 时提示输入 |
| 通过 pip 分发 | `[project.entry-points."hermes_agent.plugins"]` |
| 注册 gateway 平台（Discord、Telegram、IRC 等） | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — 参见 [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| 注册图像生成后端 | `ctx.register_image_gen_provider(provider)` — 参见 [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| 注册上下文压缩引擎 | `ctx.register_context_engine(engine)` — 参见 [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| 注册 memory 后端 | 在 `plugins/memory/<name>/__init__.py` 中继承 `MemoryProvider` — 参见 [Memory Provider Plugins](/developer-guide/memory-provider-plugin)（使用独立的发现系统） |
| 运行由 host 控制的 LLM 调用 | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — 借用用户当前激活的模型和认证信息执行一次性 completion，可选 JSON schema 验证。参见 [Plugin LLM Access](/developer-guide/plugin-llm-access) |
| 注册推理后端（LLM provider） | 在 `plugins/model-providers/<name>/__init__.py` 中使用 `register_provider(ProviderProfile(...))` — 参见 [Model Provider Plugins](/developer-guide/model-provider-plugin)（使用独立的发现系统） |

## Plugin 发现

| 来源 | 路径 | 用途 |
|--------|------|----------|
| 内置 | `<repo>/plugins/` | 随 Hermes 一起发布 — 参见 [Built-in Plugins](/user-guide/features/built-in-plugins) |
| 用户 | `~/.hermes/plugins/` | 个人插件 |
| 项目 | `.hermes/plugins/` | 项目专属插件（需要 `HERMES_ENABLE_PROJECT_PLUGINS=true`） |
| pip | `hermes_agent.plugins` entry_points | 分发包 |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS 声明式安装 — 参见 [Nix Setup](/getting-started/nix-setup#plugins) |

后面的来源会覆盖前面同名 plugin，因此与内置 plugin 同名的用户 plugin 会替换它。

### Plugin 子类别

在每个来源中，Hermes 还会识别子类别目录，将 plugin 路由到专门的发现系统：

| 子目录 | 存放内容 | 发现系统 |
|---|---|---|
| `plugins/`（根目录） | 通用 plugin — 工具、hook、slash 命令、CLI 命令、打包的 skill | `PluginManager`（类型：`standalone` 或 `backend`） |
| `plugins/platforms/<name>/` | Gateway 频道适配器（`ctx.register_platform()`） | `PluginManager`（类型：`platform`，再深一层） |
| `plugins/image_gen/<name>/` | 图像生成后端（`ctx.register_image_gen_provider()`） | `PluginManager`（类型：`backend`，再深一层） |
| `plugins/memory/<name>/` | Memory provider（继承 `MemoryProvider`） | **独立 loader** 位于 `plugins/memory/__init__.py`（类型：`exclusive` — 同时仅一个激活） |
| `plugins/context_engine/<name>/` | 上下文压缩引擎（`ctx.register_context_engine()`） | **独立 loader** 位于 `plugins/context_engine/__init__.py`（同时仅一个激活） |
| `plugins/model-providers/<name>/` | LLM provider profile（`register_provider(ProviderProfile(...))`） | **独立 loader** 位于 `providers/__init__.py`（首次调用 `get_provider_profile()` 时懒加载） |

位于 `~/.hermes/plugins/model-providers/<name>/` 和 `~/.hermes/plugins/memory/<name>/` 的用户 plugin 会覆盖同名的内置 plugin — 在 `register_provider()` / `register_memory_provider()` 中遵循 last-writer-wins 原则。放入一个目录即可替换内置实现，无需修改仓库。

## Plugins 默认 opt-in（少数例外）

**通用 plugin 和用户安装的后端默认处于禁用状态** — 发现机制能找到它们（因此它们会出现在 `hermes plugins` 和 `/plugins` 中），但包含 hook 或工具的 plugin 必须将其名称加入 `~/.hermes/config.yaml` 中的 `plugins.enabled` 后才会加载。这可以防止第三方代码在你未明确同意的情况下运行。

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # 可选的拒绝列表 — 若名称同时出现在两者中，此项优先
    - noisy-plugin
```

三种切换状态的方式：

```bash
hermes plugins                    # 交互式切换（按空格键勾选/取消）
hermes plugins enable <name>      # 加入允许列表
hermes plugins disable <name>     # 从允许列表移除并加入 disabled
```

执行 `hermes plugins install owner/repo` 后，系统会询问 `Enable 'name' now? [y/N]` — 默认否。使用 `--enable` 或 `--no-enable` 可在脚本化安装中跳过提示。

### 允许列表不控制的内容

以下几类 plugin 会绕过 `plugins.enabled` — 它们是 Hermes 内置功能的一部分，如果被默认禁用会导致基本功能失效：

| Plugin 类型 | 替代激活方式 |
|---|---|
| **内置平台 plugin**（IRC、Teams 等，位于 `plugins/platforms/`） | 自动加载，使所有发布的 gateway 频道可用。实际频道通过 `config.yaml` 中的 `gateway.platforms.<name>.enabled` 开启。 |
| **内置后端**（图像生成 provider，位于 `plugins/image_gen/` 等） | 自动加载，使默认后端"开箱即用"。通过 `config.yaml` 中的 `<category>.provider` 选择（例如 `image_gen.provider: openai`）。 |
| **Memory provider**（`plugins/memory/`） | 全部被发现；恰好一个激活，由 `config.yaml` 中的 `memory.provider` 选择。 |
| **Context engine**（`plugins/context_engine/`） | 全部被发现；一个激活，由 `config.yaml` 中的 `context.engine` 选择。 |
| **Model provider**（`plugins/model-providers/`） | `plugins/model-providers/` 下的所有内置 provider 在首次 `get_provider_profile()` 调用时发现并注册。用户通过 `--provider` 或 `config.yaml` 一次选择一个。 |
| **通过 pip 安装的 `backend` plugin** | 通过 `plugins.enabled` opt-in（与通用 plugin 相同）。 |
| **用户安装的平台**（位于 `~/.hermes/plugins/platforms/`） | 通过 `plugins.enabled` opt-in — 第三方 gateway 适配器需要明确同意。 |

简而言之：**内置的"始终可用"基础设施会自动加载；第三方通用 plugin 需要 opt-in。** `plugins.enabled` 允许列表专门用于控制用户放入 `~/.hermes/plugins/` 的任意代码。

### 现有用户的迁移

当你升级到支持 opt-in plugin 的 Hermes 版本（config schema v21+）时，任何已安装在 `~/.hermes/plugins/` 下且不在 `plugins.disabled` 中的用户 plugin 会被**自动 grandfathered** 进 `plugins.enabled`。你现有的配置将继续工作。内置的 standalone plugin 不会被 grandfathered — 即使是现有用户也必须显式 opt in。（内置的平台/后端 plugin 从不需要 grandfathering，因为它们从未被限制。）

## 可用的 hook

Plugin 可以为以下生命周期事件注册回调。完整详情、回调签名和示例请参见 **[Event Hooks 页面](/user-guide/features/hooks#plugin-hooks)**。

| Hook | 触发时机 |
|------|-----------|
| [`pre_tool_call`](/user-guide/features/hooks#pre_tool_call) | 任意工具执行前 |
| [`post_tool_call`](/user-guide/features/hooks#post_tool_call) | 任意工具返回后 |
| [`pre_llm_call`](/user-guide/features/hooks#pre_llm_call) | 每轮一次，在 LLM 循环之前 — 可返回 `{"context": "..."}` 以[向用户消息注入上下文](/user-guide/features/hooks#pre_llm_call) |
| [`post_llm_call`](/user-guide/features/hooks#post_llm_call) | 每轮一次，在 LLM 循环之后（仅成功轮次） |
| [`on_session_start`](/user-guide/features/hooks#on_session_start) | 新会话创建时（仅首次轮次） |
| [`on_session_end`](/user-guide/features/hooks#on_session_end) | 每次 `run_conversation` 调用结束时 + CLI 退出 handler |
| [`on_session_finalize`](/user-guide/features/hooks#on_session_finalize) | CLI/gateway 拆除活动会话时（`/new`、GC、CLI 退出） |
| [`on_session_reset`](/user-guide/features/hooks#on_session_reset) | Gateway 更换新会话 key 时（`/new`、`/reset`、`/clear`、空闲轮换） |
| [`subagent_stop`](/user-guide/features/hooks#subagent_stop) | 每个子 agent 在 `delegate_task` 完成后 |
| [`pre_gateway_dispatch`](/user-guide/features/hooks#pre_gateway_dispatch) | Gateway 收到用户消息，在认证和分发之前。返回 `{"action": "skip" \| "rewrite" \| "allow", ...}` 以影响流程。 |

## Plugin 类型

Hermes 有四种 plugin：

| 类型 | 功能 | 选择方式 | 位置 |
|------|-------------|-----------|----------|
| **通用 plugin** | 添加工具、hook、slash 命令、CLI 命令 | 多选（启用/禁用） | `~/.hermes/plugins/` |
| **Memory provider** | 替换或增强内置 memory | 单选（一个激活） | `plugins/memory/` |
| **Context engine** | 替换内置上下文压缩器 | 单选（一个激活） | `plugins/context_engine/` |
| **Model provider** | 声明推理后端（OpenRouter、Anthropic 等） | 多注册，通过 `--provider` / `config.yaml` 选择 | `plugins/model-providers/` |

Memory provider 和 context engine 属于 **provider plugin** — 每种类型同时只能有一个激活。Model provider 也是 plugin，但多个可以同时加载；用户通过 `--provider` 或 `config.yaml` 一次选择一个。通用 plugin 可以以任意组合启用。

## 可插拔接口 — 每种扩展的对应文档

上面的表格展示了四种 plugin 类别，但在"通用 plugin"中 `PluginContext` 暴露了多个不同的扩展点 — Hermes 也接受 Python plugin 系统之外的扩展（config 驱动的后端、shell hook 命令、外部服务器等）。使用下表找到你想构建的内容对应的文档：

| 想添加… | 方式 | 编写指南 |
|---|---|---|
| LLM 可调用的 **tool** | Python plugin — `ctx.register_tool()` | [Build a Hermes Plugin](/guides/build-a-hermes-plugin) · [Adding Tools](/developer-guide/adding-tools) |
| **生命周期 hook**（pre/post LLM、session start/end、tool filter） | Python plugin — `ctx.register_hook()` | [Hooks reference](/user-guide/features/hooks) · [Build a Hermes Plugin](/guides/build-a-hermes-plugin) |
| 用于 CLI / gateway 的 **slash 命令** | Python plugin — `ctx.register_command()` | [Build a Hermes Plugin](/guides/build-a-hermes-plugin) · [Extending the CLI](/developer-guide/extending-the-cli) |
| `hermes <thing>` 的 **subcommand** | Python plugin — `ctx.register_cli_command()` | [Extending the CLI](/developer-guide/extending-the-cli) |
| Plugin 打包的 **skill** | Python plugin — `ctx.register_skill()` | [Creating Skills](/developer-guide/creating-skills) |
| **推理后端**（LLM provider：OpenAI-compat、Codex、Anthropic-Messages、Bedrock） | Provider plugin — 在 `plugins/model-providers/<name>/` 中使用 `register_provider(ProviderProfile(...))` | **[Model Provider Plugins](/developer-guide/model-provider-plugin)** · [Adding Providers](/developer-guide/adding-providers) |
| **Gateway 频道**（Discord / Telegram / IRC / Teams 等） | Platform plugin — 在 `plugins/platforms/<name>/` 中使用 `ctx.register_platform()` | [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| **Memory 后端**（Honcho、Mem0、Supermemory 等） | Memory plugin — 在 `plugins/memory/<name>/` 中继承 `MemoryProvider` | [Memory Provider Plugins](/developer-guide/memory-provider-plugin) |
| **上下文压缩策略** | Context-engine plugin — `ctx.register_context_engine()` | [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| **图像生成后端**（DALL·E、SDXL 等） | Backend plugin — `ctx.register_image_gen_provider()` | [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| **TTS 后端**（任意 CLI — Piper、VoxCPM、Kokoro、xtts、voice-cloning 脚本等） | Config 驱动 — 在 `config.yaml` 的 `tts.providers.<name>` 下声明，设置 `type: command` | [TTS setup](/user-guide/features/tts#custom-command-providers) |
| **STT 后端**（自定义 whisper 二进制、本地 ASR CLI） | Config 驱动 — 将 `HERMES_LOCAL_STT_COMMAND` 环境变量设为 shell 模板 | [Voice Message Transcription (STT)](/user-guide/features/tts#voice-message-transcription-stt) |
| **通过 MCP 的外部工具**（filesystem、GitHub、Linear、Notion、任意 MCP server） | Config 驱动 — 在 `config.yaml` 中声明 `mcp_servers.<name>`，使用 `command:` / `url:`。Hermes 自动发现服务器的工具并 alongside 内置工具注册。 | [MCP](/user-guide/features/mcp) |
| **额外的 skill 来源**（自定义 GitHub 仓库、私有 skill 索引） | CLI — `hermes skills tap add <repo>` | [Skills Hub](/user-guide/features/skills#skills-hub) · [Publishing a custom tap](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| **Gateway 事件 hook**（在 `gateway:startup`、`session:start`、`agent:end`、`command:*` 时触发） | 将 `HOOK.yaml` + `handler.py` 放入 `~/.hermes/hooks/<name>/` | [Event Hooks](/user-guide/features/hooks#gateway-event-hooks) |
| **Shell hook**（在事件发生时运行 shell 命令 — 通知、审计日志、桌面提醒） | Config 驱动 — 在 `config.yaml` 的 `hooks:` 下声明 | [Shell Hooks](/user-guide/features/hooks#shell-hooks) |

:::note
并非所有扩展都是 Python plugin。某些扩展面有意使用 **config 驱动的 shell 命令**（TTS、STT、shell hook），这样你已有的任意 CLI 无需编写 Python 即可成为 plugin。其他的是 **外部服务器**（MCP），agent 连接到它们并自动注册工具。还有一些是 **放入式目录**（gateway hook），使用自己的清单格式。根据你的集成风格选择合适的扩展面；上表中的编写指南各自涵盖了占位符、发现和示例。
:::

## NixOS 声明式 plugin

在 NixOS 上，plugin 可以通过模块选项进行声明式安装 — 无需执行 `hermes plugins install`。完整细节请参见 **[Nix Setup 指南](/getting-started/nix-setup#plugins)**。

```nix
services.hermes-agent = {
  # 目录 plugin（包含 plugin.yaml 的源码树）
  extraPlugins = [ (pkgs.fetchFromGitHub { ... }) ];
  # Entry-point plugin（pip 包）
  extraPythonPackages = [ (pkgs.python312Packages.buildPythonPackage { ... }) ];
  # 在配置中启用
  settings.plugins.enabled = [ "my-plugin" ];
};
```

声明式 plugin 会以 `nix-managed-` 前缀进行符号链接 — 它们与手动安装的 plugin 共存，并在从 Nix 配置中移除时自动清理。

## 管理 plugin

```bash
hermes plugins                               # 统一的交互式 UI
hermes plugins list                          # 表格：enabled / disabled / not enabled
hermes plugins install user/repo             # 从 Git 安装，然后提示 Enable? [y/N]
hermes plugins install user/repo --enable    # 安装并启用（无提示）
hermes plugins install user/repo --no-enable # 安装但保持禁用（无提示）
hermes plugins update my-plugin              # 拉取最新版本
hermes plugins remove my-plugin              # 卸载
hermes plugins enable my-plugin              # 加入允许列表
hermes plugins disable my-plugin             # 从允许列表移除并加入 disabled
```

### 交互式 UI

不带参数运行 `hermes plugins` 会打开一个组合式交互界面：

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

- **General Plugins 区域** — 复选框，按空格键切换。勾选 = 在 `plugins.enabled` 中，未勾选 = 在 `plugins.disabled` 中（显式关闭）。
- **Provider Plugins 区域** — 显示当前选择。按 ENTER 进入单选器，从中选择一个激活的 provider。
- 内置 plugin 在同一列表中显示，带有 `[bundled]` 标签。

Provider plugin 的选择会保存到 `config.yaml`：

```yaml
memory:
  provider: "honcho"      # 空字符串 = 仅使用内置

context:
  engine: "compressor"    # 默认内置压缩器
```

### Enabled vs. disabled vs. neither

Plugin 处于以下三种状态之一：

| 状态 | 含义 | 在 `plugins.enabled` 中？ | 在 `plugins.disabled` 中？ |
|---|---|---|---|
| `enabled` | 下次会话时加载 | 是 | 否 |
| `disabled` | 显式关闭 — 即使也在 `enabled` 中也不会加载 | （无关） | 是 |
| `not enabled` | 已发现但从未 opt in | 否 | 否 |

新安装或内置 plugin 的默认状态是 `not enabled`。`hermes plugins list` 会显示三种不同的状态，以便你区分显式关闭和等待启用的 plugin。

在运行中的会话中，`/plugins` 显示当前已加载的 plugin。

## Injecting Messages

Plugin 可以使用 `ctx.inject_message()` 向活跃会话中注入消息：

```python
ctx.inject_message("New data arrived from the webhook", role="user")
```

**签名：** `ctx.inject_message(content: str, role: str = "user") -> bool`

工作原理：

- 如果 agent **空闲**（等待用户输入），消息会排队为下一次输入并开启新一轮。
- 如果 agent **正在运行**（活跃执行中），消息会中断当前操作 — 等同于用户输入新消息并按回车。
- 对于非 `"user"` 角色，内容会加上 `[role]` 前缀（例如 `[system] ...`）。
- 若消息成功排队则返回 `True`，若无可用的 CLI 引用（例如在 gateway 模式下）则返回 `False`。

这使得远程控制查看器、消息桥接器或 webhook 接收器等 plugin 能够将外部来源的消息输入到会话中。

:::note
`inject_message` 仅在 CLI 模式下可用。在 gateway 模式下没有 CLI 引用，该方法返回 `False`。
:::

有关 handler 契约、schema 格式、hook 行为、错误处理和常见错误的完整指南，请参见 **[full guide](/guides/build-a-hermes-plugin)**。
