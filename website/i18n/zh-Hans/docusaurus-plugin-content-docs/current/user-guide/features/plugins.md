---
sidebar_position: 11
sidebar_label: "Plugins"
title: "Plugins"
description: "通过插件系统为 Hermes 添加自定义工具、hook 和集成"
---

# Plugins

Hermes 提供了一套插件系统，可在不修改核心代码的情况下添加自定义工具、hook（钩子）和集成。

如果你想为自己、团队或某个项目创建自定义工具，这通常是正确的路径。开发者指南中的
[Adding Tools](/developer-guide/adding-tools) 页面针对的是存放在 `tools/` 和 `toolsets.py` 中的 Hermes 内置核心工具。

**→ [构建 Hermes Plugin](/guides/build-a-hermes-plugin)** — 包含完整可运行示例的分步指南。

## 快速概览

在 `~/.hermes/plugins/` 下放入一个目录，包含 `plugin.yaml` 和 Python 代码：

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # manifest（清单）
├── __init__.py      # register() — 将 schema 与处理器绑定
├── schemas.py       # tool schema（LLM 所见的内容）
└── tools.py         # tool 处理器（调用时实际执行的代码）
```

启动 Hermes — 你的工具会与内置工具一同出现，模型可立即调用它们。

### 最小可运行示例

以下是一个完整插件，添加了一个 `hello_world` 工具，并通过 hook 记录每次工具调用。

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

将两个文件放入 `~/.hermes/plugins/hello-world/`，重启 Hermes，模型即可立即调用 `hello_world`。每次工具调用后，hook 会打印一行日志。

`./.hermes/plugins/` 下的项目本地插件默认禁用。仅对可信仓库启用，方法是在启动 Hermes 前设置 `HERMES_ENABLE_PROJECT_PLUGINS=true`。

## 插件能做什么

以下所有 `ctx.*` API 均可在插件的 `register(ctx)` 函数中使用。

| 能力 | 方式 |
|-----------|-----|
| 添加工具 | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| 添加 hook | `ctx.register_hook("post_tool_call", callback)` |
| 添加斜杠命令 | `ctx.register_command(name, handler, description)` — 在 CLI 和 gateway 会话中添加 `/name` |
| 从命令中调度工具 | `ctx.dispatch_tool(name, args)` — 调用已注册的工具，自动注入父 agent 上下文 |
| 添加 CLI 命令 | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — 添加 `hermes <plugin> <subcommand>` |
| 注入消息 | `ctx.inject_message(content, role="user")` — 参见 [注入消息](#injecting-messages) |
| 附带数据文件 | `Path(__file__).parent / "data" / "file.yaml"` |
| 打包 skill | `ctx.register_skill(name, path)` — 命名空间为 `plugin:skill`，通过 `skill_view("plugin:skill")` 加载 |
| 按环境变量控制 | 在 plugin.yaml 中设置 `requires_env: [API_KEY]` — 在 `hermes plugins install` 时提示输入 |
| 通过 pip 分发 | `[project.entry-points."hermes_agent.plugins"]` |
| 注册 gateway 平台（Discord、Telegram、IRC 等） | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — 参见 [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| 注册图像生成后端 | `ctx.register_image_gen_provider(provider)` — 参见 [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| 注册视频生成后端 | `ctx.register_video_gen_provider(provider)` — 参见 [Video Generation Provider Plugins](/developer-guide/video-gen-provider-plugin) |
| 注册上下文压缩引擎 | `ctx.register_context_engine(engine)` — 参见 [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| 注册 memory 后端 | 在 `plugins/memory/<name>/__init__.py` 中继承 `MemoryProvider` — 参见 [Memory Provider Plugins](/developer-guide/memory-provider-plugin)（使用独立发现系统） |
| 调用宿主 LLM | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — 借用用户当前激活的模型和认证，进行一次性补全，支持可选 JSON schema 验证。参见 [Plugin LLM Access](/developer-guide/plugin-llm-access) |
| 注册推理后端（LLM provider） | 在 `plugins/model-providers/<name>/__init__.py` 中调用 `register_provider(ProviderProfile(...))` — 参见 [Model Provider Plugins](/developer-guide/model-provider-plugin)（使用独立发现系统） |

## 插件发现

| 来源 | 路径 | 使用场景 |
|--------|------|----------|
| 内置 | `<repo>/plugins/` | 随 Hermes 附带 — 参见 [Built-in Plugins](/user-guide/features/built-in-plugins) |
| 用户 | `~/.hermes/plugins/` | 个人插件 |
| 项目 | `.hermes/plugins/` | 项目专属插件（需要 `HERMES_ENABLE_PROJECT_PLUGINS=true`） |
| pip | `hermes_agent.plugins` entry_points | 分发包 |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS 声明式安装 — 参见 [Nix Setup](/getting-started/nix-setup#plugins) |

名称冲突时，后面的来源会覆盖前面的，因此与内置插件同名的用户插件会替换它。

### 插件子分类

在每个来源内，Hermes 还识别将插件路由到专用发现系统的子分类目录：

| 子目录 | 内容 | 发现系统 |
|---|---|---|
| `plugins/`（根目录） | 通用插件 — 工具、hook、斜杠命令、CLI 命令、打包 skill | `PluginManager`（kind: `standalone` 或 `backend`） |
| `plugins/platforms/<name>/` | Gateway 频道适配器（`ctx.register_platform()`） | `PluginManager`（kind: `platform`，深一层） |
| `plugins/image_gen/<name>/` | 图像生成后端（`ctx.register_image_gen_provider()`） | `PluginManager`（kind: `backend`，深一层） |
| `plugins/memory/<name>/` | Memory provider（继承 `MemoryProvider`） | **独立加载器**，位于 `plugins/memory/__init__.py`（kind: `exclusive` — 同时只有一个激活） |
| `plugins/context_engine/<name>/` | 上下文压缩引擎（`ctx.register_context_engine()`） | **独立加载器**，位于 `plugins/context_engine/__init__.py`（同时只有一个激活） |
| `plugins/model-providers/<name>/` | LLM provider profile（`register_provider(ProviderProfile(...))`） | **独立加载器**，位于 `providers/__init__.py`（首次调用 `get_provider_profile()` 时懒加载扫描） |

`~/.hermes/plugins/model-providers/<name>/` 和 `~/.hermes/plugins/memory/<name>/` 下的用户插件会覆盖同名内置插件 — `register_provider()` / `register_memory_provider()` 中后写者胜出。放入一个目录即可替换内置实现，无需修改仓库。

子分类插件在 `hermes plugins list` 和交互式 `hermes plugins` UI 中以**路径派生的 key** 显示 — 例如 `observability/langfuse`、`image_gen/openai`、`platforms/teams`。该 key（而非 manifest 中的 `name:`）是传给 `hermes plugins enable …` / `disable …` 的值，也是在 `config.yaml` 的 `plugins.enabled` 下填写的字符串。

## 插件默认关闭（少数例外）

**通用插件和用户安装的后端默认禁用** — 发现系统会找到它们（因此它们会出现在 `hermes plugins` 和 `/plugins` 中），但在你将插件名称添加到 `~/.hermes/config.yaml` 的 `plugins.enabled` 之前，任何带有 hook 或工具的内容都不会加载。这可防止第三方代码在未经明确同意的情况下运行。

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # 可选的拒绝列表 — 若名称同时出现在两个列表中，此列表始终优先
    - noisy-plugin
```

切换状态的三种方式：

```bash
hermes plugins                    # 交互式切换（空格勾选/取消勾选）
hermes plugins enable <name>      # 添加到允许列表
hermes plugins disable <name>     # 从允许列表移除并添加到禁用列表
```

执行 `hermes plugins install owner/repo` 后，会询问 `Enable 'name' now? [y/N]` — 默认为否。脚本化安装时可用 `--enable` 或 `--no-enable` 跳过提示。

### 允许列表不控制的内容

某些类别的插件绕过 `plugins.enabled` — 它们是 Hermes 内置功能的一部分，若默认关闭会破坏基本功能：

| 插件类型 | 激活方式 |
|---|---|
| **内置平台插件**（IRC、Teams 等，位于 `plugins/platforms/`） | 自动加载，使所有内置 gateway 频道可用。实际频道通过 `config.yaml` 中的 `gateway.platforms.<name>.enabled` 开启。 |
| **内置后端**（`plugins/image_gen/` 等下的图像生成 provider） | 自动加载，使默认后端"开箱即用"。通过 `config.yaml` 中的 `<category>.provider` 选择（例如 `image_gen.provider: openai`）。 |
| **Memory provider**（`plugins/memory/`） | 全部发现；同时只有一个激活，由 `config.yaml` 中的 `memory.provider` 选择。 |
| **Context engine**（`plugins/context_engine/`） | 全部发现；同时只有一个激活，由 `config.yaml` 中的 `context.engine` 选择。 |
| **Model provider**（`plugins/model-providers/`） | `plugins/model-providers/` 下的所有内置 provider 在首次调用 `get_provider_profile()` 时发现并注册。用户通过 `--provider` 或 `config.yaml` 一次选择一个。 |
| **pip 安装的 `backend` 插件** | 通过 `plugins.enabled` 选择加入（与通用插件相同）。 |
| **用户安装的平台**（位于 `~/.hermes/plugins/platforms/`） | 通过 `plugins.enabled` 选择加入 — 第三方 gateway 适配器需要明确同意。 |

简而言之：**内置的"始终可用"基础设施自动加载；第三方通用插件需选择加入。** `plugins.enabled` 允许列表专门用于控制用户放入 `~/.hermes/plugins/` 的任意代码。

### 现有用户的迁移

当你升级到支持选择加入插件的 Hermes 版本（config schema v21+）时，已安装在 `~/.hermes/plugins/` 下且不在 `plugins.disabled` 中的用户插件会**自动纳入** `plugins.enabled`。你的现有配置继续正常工作。内置独立插件**不会**自动纳入 — 即使是现有用户也需要明确选择加入。（内置平台/后端插件从未需要纳入，因为它们从未被控制。）

## 可用 hook

插件可为以下生命周期事件注册回调。完整详情、回调签名和示例请参见 **[Event Hooks 页面](/user-guide/features/hooks#plugin-hooks)**。

| Hook | 触发时机 |
|------|-----------|
| [`pre_tool_call`](/user-guide/features/hooks#pre_tool_call) | 任意工具执行前 |
| [`post_tool_call`](/user-guide/features/hooks#post_tool_call) | 任意工具返回后 |
| [`pre_llm_call`](/user-guide/features/hooks#pre_llm_call) | 每轮一次，LLM 循环前 — 可返回 `{"context": "..."}` 以[向用户消息注入上下文](/user-guide/features/hooks#pre_llm_call) |
| [`post_llm_call`](/user-guide/features/hooks#post_llm_call) | 每轮一次，LLM 循环后（仅成功轮次） |
| [`on_session_start`](/user-guide/features/hooks#on_session_start) | 新会话创建时（仅第一轮） |
| [`on_session_end`](/user-guide/features/hooks#on_session_end) | 每次 `run_conversation` 调用结束时 + CLI 退出处理器 |
| [`on_session_finalize`](/user-guide/features/hooks#on_session_finalize) | CLI/gateway 销毁活跃会话时（`/new`、GC、CLI 退出） |
| [`on_session_reset`](/user-guide/features/hooks#on_session_reset) | Gateway 换入新会话 key 时（`/new`、`/reset`、`/clear`、空闲轮换） |
| [`subagent_stop`](/user-guide/features/hooks#subagent_stop) | `delegate_task` 完成后每个子 agent 触发一次 |
| [`pre_gateway_dispatch`](/user-guide/features/hooks#pre_gateway_dispatch) | Gateway 收到用户消息，在认证和调度之前。返回 `{"action": "skip" \| "rewrite" \| "allow", ...}` 以影响流程。 |

## 插件类型

Hermes 有四种插件：

| 类型 | 作用 | 选择方式 | 位置 |
|------|-------------|-----------|----------|
| **通用插件** | 添加工具、hook、斜杠命令、CLI 命令 | 多选（启用/禁用） | `~/.hermes/plugins/` |
| **Memory provider** | 替换或增强内置 memory | 单选（同时只有一个激活） | `plugins/memory/` |
| **Context engine** | 替换内置上下文压缩器 | 单选（同时只有一个激活） | `plugins/context_engine/` |
| **Model provider** | 声明推理后端（OpenRouter、Anthropic 等） | 多注册，通过 `--provider` / `config.yaml` 选择 | `plugins/model-providers/` |

Memory provider 和 context engine 是 **provider 插件** — 每种类型同时只能有一个激活。Model provider 也是插件，但可以同时加载多个；用户通过 `--provider` 或 `config.yaml` 一次选择一个。通用插件可以任意组合启用。

## 可插拔接口 — 各场景对应文档

上表展示了四种插件类别，但在"通用插件"中，`PluginContext` 暴露了多个不同的扩展点 — Hermes 还接受 Python 插件系统之外的扩展（配置驱动的后端、shell hook 命令、外部服务器等）。使用下表找到适合你需求的文档：

| 想要添加… | 方式 | 编写指南 |
|---|---|---|
| LLM 可调用的**工具** | Python 插件 — `ctx.register_tool()` | [Build a Hermes Plugin](/guides/build-a-hermes-plugin) · [Adding Tools](/developer-guide/adding-tools) |
| **生命周期 hook**（LLM 前后、会话开始/结束、工具过滤） | Python 插件 — `ctx.register_hook()` | [Hooks reference](/user-guide/features/hooks) · [Build a Hermes Plugin](/guides/build-a-hermes-plugin) |
| CLI / gateway 的**斜杠命令** | Python 插件 — `ctx.register_command()` | [Build a Hermes Plugin](/guides/build-a-hermes-plugin) · [Extending the CLI](/developer-guide/extending-the-cli) |
| `hermes <thing>` 的**子命令** | Python 插件 — `ctx.register_cli_command()` | [Extending the CLI](/developer-guide/extending-the-cli) |
| 插件附带的**skill** | Python 插件 — `ctx.register_skill()` | [Creating Skills](/developer-guide/creating-skills) |
| **推理后端**（LLM provider：OpenAI 兼容、Codex、Anthropic-Messages、Bedrock） | Provider 插件 — 在 `plugins/model-providers/<name>/` 中调用 `register_provider(ProviderProfile(...))` | **[Model Provider Plugins](/developer-guide/model-provider-plugin)** · [Adding Providers](/developer-guide/adding-providers) |
| **Gateway 频道**（Discord / Telegram / IRC / Teams 等） | 平台插件 — 在 `plugins/platforms/<name>/` 中调用 `ctx.register_platform()` | [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| **Memory 后端**（Honcho、Mem0、Supermemory 等） | Memory 插件 — 在 `plugins/memory/<name>/` 中继承 `MemoryProvider` | [Memory Provider Plugins](/developer-guide/memory-provider-plugin) |
| **上下文压缩策略** | Context-engine 插件 — `ctx.register_context_engine()` | [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| **图像生成后端**（DALL·E、SDXL 等） | 后端插件 — `ctx.register_image_gen_provider()` | [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| **视频生成后端**（Veo、Kling、Pixverse、Grok-Imagine、Runway 等） | 后端插件 — `ctx.register_video_gen_provider()` | [Video Generation Provider Plugins](/developer-guide/video-gen-provider-plugin) |
| **TTS 后端**（任意 CLI — Piper、VoxCPM、Kokoro、xtts、语音克隆脚本等） | 配置驱动（推荐）— 在 `config.yaml` 的 `tts.providers.<name>` 下以 `type: command` 声明。或 Python 后端插件 — 对需要超出 shell 模板的 Python SDK / 流式引擎使用 `ctx.register_tts_provider()`。 | [TTS Setup](/user-guide/features/tts#custom-command-providers) · [Python plugin guide](/user-guide/features/tts#python-plugin-providers) |
| **STT 后端**（自定义 whisper 二进制、本地 ASR CLI） | 配置驱动 — 将 `HERMES_LOCAL_STT_COMMAND` 环境变量设置为 shell 模板 | [Voice Message Transcription (STT)](/user-guide/features/tts#voice-message-transcription-stt) |
| **通过 MCP 使用外部工具**（文件系统、GitHub、Linear、Notion、任意 MCP 服务器） | 配置驱动 — 在 `config.yaml` 中以 `command:` / `url:` 声明 `mcp_servers.<name>`。Hermes 自动发现服务器的工具并与内置工具一同注册。 | [MCP](/user-guide/features/mcp) |
| **额外 skill 来源**（自定义 GitHub 仓库、私有 skill 索引） | CLI — `hermes skills tap add <repo>` | [Skills Hub](/user-guide/features/skills#skills-hub) · [发布自定义 tap](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| **Gateway 事件 hook**（在 `gateway:startup`、`session:start`、`agent:end`、`command:*` 时触发） | 将 `HOOK.yaml` + `handler.py` 放入 `~/.hermes/hooks/<name>/` | [Event Hooks](/user-guide/features/hooks#gateway-event-hooks) |
| **Shell hook**（在事件时运行 shell 命令 — 通知、审计日志、桌面提醒） | 配置驱动 — 在 `config.yaml` 的 `hooks:` 下声明 | [Shell Hooks](/user-guide/features/hooks#shell-hooks) |

:::note
并非所有扩展都是 Python 插件。某些扩展接口有意使用**配置驱动的 shell 命令**（TTS、STT、shell hook），这样你已有的任意 CLI 无需编写 Python 即可成为插件。其他的是 agent 连接并自动注册工具的**外部服务器**（MCP）。还有一些是拥有自己 manifest 格式的**即插即用目录**（gateway hook）。根据你的集成风格选择合适的接口；上表中的编写指南各自涵盖了占位符、发现机制和示例。
:::

## NixOS 声明式插件

在 NixOS 上，插件可通过模块选项声明式安装 — 无需 `hermes plugins install`。完整详情请参见 **[Nix Setup 指南](/getting-started/nix-setup#plugins)**。

```nix
services.hermes-agent = {
  # 目录插件（包含 plugin.yaml 的源码树）
  extraPlugins = [ (pkgs.fetchFromGitHub { ... }) ];
  # 入口点插件（pip 包）
  extraPythonPackages = [ (pkgs.python312Packages.buildPythonPackage { ... }) ];
  # 在 config 中启用
  settings.plugins.enabled = [ "my-plugin" ];
};
```

声明式插件以 `nix-managed-` 前缀符号链接 — 与手动安装的插件共存，从 Nix 配置中移除后自动清理。

## 管理插件

```bash
hermes plugins                                       # 统一交互式 UI
hermes plugins list                                  # 表格：已启用 / 已禁用 / 未启用
hermes plugins install user/repo                     # 从 Git 安装，然后提示 Enable? [y/N]
hermes plugins install user/repo --enable            # 安装并启用（无提示）
hermes plugins install user/repo --no-enable         # 安装但保持禁用（无提示）
hermes plugins update my-plugin                      # 拉取最新版本
hermes plugins remove my-plugin                      # 卸载
hermes plugins enable my-plugin                      # 添加到允许列表（普通插件）
hermes plugins enable observability/langfuse         # 添加到允许列表（子分类插件）
hermes plugins disable my-plugin                     # 从允许列表移除并添加到禁用列表
```

对于子分类目录下的插件（例如 `plugins/observability/langfuse/`、`plugins/image_gen/openai/`），使用完整的 `<category>/<plugin>` key — 这正是 `hermes plugins list` 在 **Name** 列中显示的内容。

### 交互式 UI

不带参数运行 `hermes plugins` 会打开一个复合交互界面：

```
Plugins
  ↑↓ navigate  SPACE toggle  ENTER configure/confirm  ESC done

  General Plugins
 → [✓] my-tool-plugin — Custom search tool
   [ ] webhook-notifier — Event hooks
   [ ] disk-cleanup — Auto-cleanup of ephemeral files [bundled]
   [ ] observability/langfuse — Trace turns / LLM calls / tools to Langfuse [bundled]

  Provider Plugins
     Memory Provider          ▸ honcho
     Context Engine           ▸ compressor
```

- **General Plugins 区域** — 复选框，用空格切换。勾选 = 在 `plugins.enabled` 中，未勾选 = 在 `plugins.disabled` 中（明确关闭）。
- **Provider Plugins 区域** — 显示当前选择。按 ENTER 进入单选选择器，选择一个激活的 provider。
- 内置插件在同一列表中显示，带有 `[bundled]` 标签。

Provider 插件的选择保存到 `config.yaml`：

```yaml
memory:
  provider: "honcho"      # 空字符串 = 仅使用内置

context:
  engine: "compressor"    # 默认内置压缩器
```

### 已启用 vs. 已禁用 vs. 未设置

插件处于以下三种状态之一：

| 状态 | 含义 | 在 `plugins.enabled` 中？ | 在 `plugins.disabled` 中？ |
|---|---|---|---|
| `enabled` | 下次会话时加载 | 是 | 否 |
| `disabled` | 明确关闭 — 即使同时在 `enabled` 中也不会加载 | （无关） | 是 |
| `not enabled` | 已发现但从未选择加入 | 否 | 否 |

新安装或内置插件的默认状态为 `not enabled`。`hermes plugins list` 显示全部三种状态，便于区分明确关闭的插件和等待启用的插件。

在运行中的会话里，`/plugins` 显示当前已加载的插件。

## 注入消息

插件可使用 `ctx.inject_message()` 向活跃对话注入消息：

```python
ctx.inject_message("New data arrived from the webhook", role="user")
```

**签名：** `ctx.inject_message(content: str, role: str = "user") -> bool`

工作原理：

- 若 agent **空闲**（等待用户输入），消息会作为下一条输入排队并开始新一轮。
- 若 agent **处于轮次中**（正在运行），消息会中断当前操作 — 与用户输入新消息并按下 Enter 效果相同。
- 对于非 `"user"` 角色，内容会以 `[role]` 为前缀（例如 `[system] ...`）。
- 若消息成功排队返回 `True`，若无 CLI 引用（例如在 gateway 模式下）则返回 `False`。

这使得远程控制查看器、消息桥接或 webhook 接收器等插件能够从外部来源向对话注入消息。

:::note
`inject_message` 仅在 CLI 模式下可用。在 gateway 模式下，没有 CLI 引用，该方法返回 `False`。
:::

完整的处理器约定、schema 格式、hook 行为、错误处理和常见错误请参见 **[完整指南](/guides/build-a-hermes-plugin)**。