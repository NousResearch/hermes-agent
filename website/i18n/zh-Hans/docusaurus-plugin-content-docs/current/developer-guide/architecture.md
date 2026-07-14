---
sidebar_position: 1
title: "架构"
description: "Hermes Agent 内部结构——主要子系统、执行路径、数据流及延伸阅读指引"
---

# 架构

本页是 Hermes Agent 内部结构的顶层导图。用它在代码库中定位自己，然后深入各子系统专项文档了解实现细节。

## 系统概览

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Entry Points                                  │
│                                                                      │
│  CLI (cli.py)    Gateway (gateway/run.py)    ACP (acp_adapter/)     │
│  Batch Runner    API Server                  Python Library          │
└──────────┬──────────────┬───────────────────────┬───────────────────┘
           │              │                       │
           ▼              ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     AIAgent (run_agent.py)                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Prompt       │  │ Provider     │  │ Tool         │               │
│  │ Builder      │  │ Resolution   │  │ Dispatch     │               │
│  │ (prompt_     │  │ (runtime_    │  │ (model_      │               │
│  │  builder.py) │  │  provider.py)│  │  tools.py)   │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                 │                       │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐               │
│  │ Compression  │  │ 3 API Modes  │  │ Tool Registry│               │
│  │ & Caching    │  │ chat_compl.  │  │ (registry.py)│               │
│  │              │  │ codex_resp.  │  │ 70+ tools    │               │
│  │              │  │ anthropic    │  │ 28 toolsets  │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────┴─────────────────┴─────────────────┴───────────────────────┘
           │                                    │
           ▼                                    ▼
┌───────────────────┐              ┌──────────────────────┐
│ Session Storage   │              │ Tool Backends         │
│ (SQLite + FTS5)   │              │ Terminal (7 backends) │
│ hermes_state.py   │              │ Browser (5 backends)  │
│ gateway/session.py│              │ Web (4 backends)      │
└───────────────────┘              │ MCP (dynamic)         │
                                   │ File, Vision, etc.    │
                                   └──────────────────────┘
```

## 目录结构

```text
hermes-agent/
├── run_agent.py              # AIAgent — 核心对话循环（大文件）
├── cli.py                    # HermesCLI — 交互式终端 UI（大文件）
├── model_tools.py            # 工具发现、schema 收集、分发
├── toolsets.py               # 工具分组与平台预设
├── hermes_state.py           # 带 FTS5 的 SQLite 会话/状态数据库
├── hermes_constants.py       # HERMES_HOME、感知 profile 的路径
├── batch_runner.py           # 批量轨迹生成
│
├── agent/                    # Agent 内部模块
│   ├── prompt_builder.py     # 系统 prompt 组装
│   ├── context_engine.py     # ContextEngine ABC（可插拔）
│   ├── context_compressor.py # 默认引擎——有损摘要压缩
│   ├── prompt_caching.py     # Anthropic prompt 缓存
│   ├── auxiliary_client.py   # 辅助 LLM，用于旁路任务（视觉、摘要）
│   ├── model_metadata.py     # 模型上下文长度、token 估算
│   ├── models_dev.py         # models.dev 注册表集成
│   ├── anthropic_adapter.py  # Anthropic Messages API 格式转换
│   ├── display.py            # KawaiiSpinner、工具预览格式化
│   ├── skill_commands.py     # Skill 斜杠命令
│   ├── memory_manager.py    # 记忆管理器编排
│   ├── memory_provider.py   # 记忆提供者 ABC
│   └── trajectory.py         # 轨迹保存辅助函数
│
├── hermes_cli/               # CLI 子命令与设置
│   ├── main.py               # 入口点——所有 `hermes` 子命令（大文件）
│   ├── config.py             # DEFAULT_CONFIG、OPTIONAL_ENV_VARS、迁移
│   ├── commands.py           # COMMAND_REGISTRY——斜杠命令中央定义
│   ├── auth.py               # PROVIDER_REGISTRY、凭据解析
│   ├── runtime_provider.py   # Provider → api_mode + 凭据
│   ├── models.py             # 模型目录、provider 模型列表
│   ├── model_switch.py       # /model 命令逻辑（CLI + gateway 共用）
│   ├── setup.py              # 交互式设置向导（大文件）
│   ├── skin_engine.py        # CLI 主题引擎
│   ├── skills_config.py      # hermes skills——按平台启用/禁用
│   ├── skills_hub.py         # /skills 斜杠命令
│   ├── tools_config.py       # hermes tools——按平台启用/禁用
│   ├── plugins.py            # PluginManager——发现、加载、hook
│   ├── callbacks.py          # 终端回调（clarify、sudo、approval）
│   └── gateway.py            # hermes gateway 启动/停止
│
├── tools/                    # 工具实现（每个工具一个文件）
│   ├── registry.py           # 中央工具注册表
│   ├── approval.py           # 危险命令检测
│   ├── terminal_tool.py      # 终端编排
│   ├── process_registry.py   # 后台进程管理
│   ├── file_tools.py         # read_file、write_file、patch、search_files
│   ├── web_tools.py          # web_search、web_extract
│   ├── browser_tool.py       # 10 个浏览器自动化工具
│   ├── code_execution_tool.py # execute_code 沙箱
│   ├── delegate_tool.py      # 子 agent 委托
│   ├── mcp_tool.py           # MCP 客户端（大文件）
│   ├── credential_files.py   # 基于文件的凭据透传
│   ├── env_passthrough.py    # 沙箱环境变量透传
│   ├── ansi_strip.py         # ANSI 转义字符剥离
│   └── environments/         # 终端后端（local、docker、ssh、modal、daytona、singularity）
│
├── gateway/                  # 消息平台 gateway
│   ├── run.py                # GatewayRunner——消息分发（大文件）
│   ├── session.py            # SessionStore——对话持久化
│   ├── delivery.py           # 出站消息投递
│   ├── pairing.py            # DM 配对授权
│   ├── hooks.py              # Hook 发现与生命周期事件
│   ├── mirror.py             # 跨会话消息镜像
│   ├── status.py             # Token 锁、profile 范围的进程追踪
│   ├── builtin_hooks/        # 始终注册的 hook 扩展点（当前无内置）
│   └── platforms/            # 20 个适配器：telegram、discord、slack、whatsapp、
│                             #   signal、matrix、mattermost、email、sms、
│                             #   dingtalk、feishu、wecom、wecom_callback、weixin、
│                             #   bluebubbles、qqbot、homeassistant、webhook、api_server、
│                             #   yuanbao
│
├── acp_adapter/              # ACP 服务器（VS Code / Zed / JetBrains）
├── cron/                     # 调度器（jobs.py、scheduler.py）
├── plugins/memory/           # 记忆提供者插件
├── plugins/context_engine/   # 上下文引擎插件
├── skills/                   # 内置 skill（始终可用）
├── optional-skills/          # 官方可选 skill（需显式安装）
├── website/                  # Docusaurus 文档站点
└── tests/                    # Pytest 测试套件（3,000+ 个测试）
```

## 数据流

### CLI 会话

```text
用户输入 → HermesCLI.process_input()
  → AIAgent.run_conversation()
    → prompt_builder.build_system_prompt()
    → runtime_provider.resolve_runtime_provider()
    → API 调用（chat_completions / codex_responses / anthropic_messages）
    → tool_calls? → model_tools.handle_function_call() → 循环
    → 最终响应 → 显示 → 保存至 SessionDB
```

### Gateway 消息

```text
平台事件 → Adapter.on_message() → MessageEvent
  → GatewayRunner._handle_message()
    → 授权用户
    → 解析会话 key
    → 创建带会话历史的 AIAgent
    → AIAgent.run_conversation()
    → 通过适配器回传响应
```

### Cron 任务

```text
调度器触发 → 从 jobs.json 加载到期任务
  → 创建全新 AIAgent（无历史）
  → 将附加的 skill 注入为上下文
  → 运行任务 prompt
  → 向目标平台投递响应
  → 更新任务状态与 next_run
```

## 推荐阅读顺序

如果你是第一次接触代码库：

1. **本页** — 整体定位
2. **[Agent 循环内部机制](./agent-loop.md)** — AIAgent 的工作原理
3. **[Prompt 组装](./prompt-assembly.md)** — 系统 prompt 的构建过程
4. **[Provider 运行时解析](./provider-runtime.md)** — provider 的选择方式
5. **[添加 Provider](./adding-providers.md)** — 新增 provider 的实践指南
6. **[工具运行时](./tools-runtime.md)** — 工具注册表、分发、环境
7. **[会话存储](./session-storage.md)** — SQLite schema、FTS5、会话血缘
8. **[Gateway 内部机制](./gateway-internals.md)** — 消息平台 gateway
9. **[上下文压缩与 Prompt 缓存](./context-compression-and-caching.md)** — 压缩与缓存
10. **[ACP 内部机制](./acp-internals.md)** — IDE 集成

## 主要子系统

### Agent 循环

同步编排引擎（`run_agent.py` 中的 `AIAgent`）。负责 provider 选择、prompt 构建、工具执行、重试、回退、回调、压缩和持久化。支持三种 API 模式以适配不同 provider 后端。

→ [Agent 循环内部机制](./agent-loop.md)

### Prompt 系统

在对话生命周期中构建和维护 prompt：

- **`prompt_builder.py`** — 从以下来源组装系统 prompt：个性（SOUL.md）、记忆（MEMORY.md、USER.md）、skill、上下文文件（AGENTS.md、.hermes.md）、工具使用指引以及模型专项指令
- **`prompt_caching.py`** — 为前缀缓存应用 Anthropic 缓存断点
- **`context_compressor.py`** — 当上下文超出阈值时对中间对话轮次进行摘要

→ [Prompt 组装](./prompt-assembly.md)，[上下文压缩与 Prompt 缓存](./context-compression-and-caching.md)

### Provider 解析

CLI、gateway、cron、ACP 及辅助调用共用的运行时解析器。将 `(provider, model)` 元组映射为 `(api_mode, api_key, base_url)`。支持 18+ 个 provider、OAuth 流程、凭据池和别名解析。

→ [Provider 运行时解析](./provider-runtime.md)

### 工具系统

中央工具注册表（`tools/registry.py`），包含约 28 个 toolset 中的 70+ 个已注册工具。每个工具文件在导入时自行注册。注册表负责 schema 收集、分发、可用性检查和错误包装。终端工具支持 6 种后端（local、Docker、SSH、Daytona、Modal、Singularity）。

→ [工具运行时](./tools-runtime.md)

### 会话持久化

基于 SQLite 的会话存储，带 FTS5 全文检索。会话具有血缘追踪（跨压缩的父/子关系）、按平台隔离，以及带竞争处理的原子写入。

→ [会话存储](./session-storage.md)

### 消息 Gateway

长驻进程，包含 20 个平台适配器、统一会话路由、用户授权（白名单 + DM 配对）、斜杠命令分发、hook 系统、cron 触发和后台维护。

→ [Gateway 内部机制](./gateway-internals.md)

### 插件系统

三种发现来源：`~/.hermes/plugins/`（用户级）、`.hermes/plugins/`（项目级）和 pip entry point。插件通过上下文 API 注册工具、hook 和 CLI 命令。存在两种专用插件类型：记忆提供者（`plugins/memory/`）和上下文引擎（`plugins/context_engine/`）。两者均为单选——每种同时只能激活一个，通过 `hermes plugins` 或 `config.yaml` 配置。

→ [插件指南](/developer-guide/plugins)，[记忆提供者插件](./memory-provider-plugin.md)

### Cron

一等公民的 agent 任务（非 shell 任务）。任务以 JSON 存储，支持多种调度格式，可附加 skill 和脚本，并可向任意平台投递。

→ [Cron 内部机制](./cron-internals.md)

### ACP 集成

通过 stdio/JSON-RPC 将 Hermes 作为编辑器原生 agent 暴露给 VS Code、Zed 和 JetBrains。

→ [ACP 内部机制](./acp-internals.md)

### 轨迹

从 agent 会话生成 ShareGPT 格式的轨迹，用于训练数据生成。

→ [轨迹与训练格式](./trajectory-format.md)

## 设计原则

| 原则 | 实践含义 |
|------|---------|
| **Prompt 稳定性** | 系统 prompt 在对话中途不会改变。除用户显式操作（`/model`）外，不进行破坏缓存的变更。 |
| **可观测执行** | 每次工具调用均通过回调对用户可见。CLI（spinner）和 gateway（聊天消息）中均有进度更新。 |
| **可中断** | API 调用和工具执行可被用户输入或信号在执行中途取消。 |
| **平台无关的核心** | 单一 AIAgent 类同时服务于 CLI、gateway、ACP、批处理和 API 服务器。平台差异存在于入口点，而非 agent 内部。 |
| **松耦合** | 可选子系统（MCP、插件、记忆提供者、RL 环境）使用注册表模式和 check_fn 门控，而非硬依赖。 |
| **Profile 隔离** | 每个 profile（`hermes -p <name>`）拥有独立的 HERMES_HOME、配置、记忆、会话和 gateway PID。多个 profile 可并发运行。 |

## 文件依赖链

```text
tools/registry.py  （无依赖——被所有工具文件导入）
       ↑
tools/*.py  （每个文件在导入时调用 registry.register()）
       ↑
model_tools.py  （导入 tools/registry 并触发工具发现）
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

这条依赖链意味着工具注册发生在导入时，早于任何 agent 实例的创建。任何在顶层调用 `registry.register()` 的 `tools/*.py` 文件都会被自动发现——无需手动维护导入列表。