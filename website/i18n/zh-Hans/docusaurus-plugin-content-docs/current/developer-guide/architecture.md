---
sidebar_position: 1
title: "架构"
description: "Hermes Agent 内部结构 — 主要子系统、执行路径、数据流和深入阅读指引"
---

# 架构

此页面是 Hermes Agent 内部结构的顶层地图。使用它来定位代码库，然后深入到子系统特定的文档了解实现细节。

## 系统概述

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
│  │              │  │ codex_resp.  │  │ 61 tools     │               │
│  │              │  │ anthropic    │  │ 52 toolsets  │               │
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
├── run_agent.py              # AIAgent — 核心对话循环（约 13,700 行）
├── cli.py                    # HermesCLI — 交互式终端 UI（约 11,500 行）
├── model_tools.py            # 工具发现、模式收集、调度
├── toolsets.py               # 工具分组和平台预设
├── hermes_state.py           # 带 FTS5 的 SQLite 会话/状态数据库
├── hermes_constants.py       # HERMES_HOME、感知配置文件的路径
├── batch_runner.py           # 批处理轨迹生成
│
├── agent/                    # 代理内部结构
│   ├── prompt_builder.py     # 系统提示词组装
│   ├── context_engine.py     # ContextEngine ABC（可插拔）
│   ├── context_compressor.py # 默认引擎 — 有损摘要
│   ├── prompt_caching.py     # Anthropic 提示缓存
│   ├── auxiliary_client.py   # 用于辅助任务（视觉、摘要）的辅助 LLM
│   ├── model_metadata.py     # 模型上下文长度、token 估计
│   ├── models_dev.py         # models.dev 注册表集成
│   ├── anthropic_adapter.py  # Anthropic Messages API 格式转换
│   ├── display.py            # KawaiiSpinner、工具预览格式化
│   ├── skill_commands.py     # 技能斜杠命令
│   ├── memory_manager.py    # 内存管理器编排
│   ├── memory_provider.py   # 内存 provider ABC
│   └── trajectory.py         # 轨迹保存辅助函数
│
├── hermes_cli/               # CLI 子命令和设置
│   ├── main.py               # 入口点 — 所有 `hermes` 子命令（约 10,400 行）
│   ├── config.py             # DEFAULT_CONFIG、OPTIONAL_ENV_VARS、迁移
│   ├── commands.py           # COMMAND_REGISTRY — 中央斜杠命令定义
│   ├── auth.py               # PROVIDER_REGISTRY、凭据解析
│   ├── runtime_provider.py   # Provider → api_mode + 凭据
│   ├── models.py             # 模型目录、provider 模型列表
│   ├── model_switch.py       # /model 命令逻辑（CLI + 网关共享）
│   ├── setup.py              # 交互式设置向导（约 3,500 行）
│   ├── skin_engine.py        # CLI 主题引擎
│   ├── skills_config.py      # hermes skills — 按平台启用/禁用
│   ├── skills_hub.py         # /skills 斜杠命令
│   ├── tools_config.py       # hermes tools — 按平台启用/禁用
│   ├── plugins.py            # PluginManager — 发现、加载、钩子
│   ├── callbacks.py          # 终端回调（clarify、sudo、审批）
│   └── gateway.py            # hermes gateway start/stop
│
├── tools/                    # 工具实现（每个工具一个文件）
│   ├── registry.py           # 中央工具注册表
│   ├── approval.py           # 危险命令检测
│   ├── terminal_tool.py      # 终端编排
│   ├── process_registry.py   # 后台进程管理
│   ├── file_tools.py         # read_file、write_file、patch、search_files
│   ├── web_tools.py          # web_search、web_extract
│   ├── browser_tool.py       # 10 个浏览器自动化工具
│   ├── code_execution_tool.py # execute_code 沙盒
│   ├── delegate_tool.py      # 子代理委托
│   ├── mcp_tool.py           # MCP 客户端（约 3,100 行）
│   ├── credential_files.py   # 基于文件的凭据透传
│   ├── env_passthrough.py    # 沙盒的 Env 变量透传
│   ├── ansi_strip.py         # ANSI 转义符去除
│   └── environments/         # 终端后端（local、docker、ssh、modal、daytona、singularity）
│
├── gateway/                  # 消息平台网关
│   ├── run.py                # GatewayRunner — 消息调度（约 12,200 行）
│   ├── session.py            # SessionStore — 对话持久化
│   ├── delivery.py           # 出站消息投递
│   ├── pairing.py            # DM 配对授权
│   ├── hooks.py              # 钩子发现和生命周期事件
│   ├── mirror.py             # 跨会话消息镜像
│   ├── status.py             # Token 锁、配置文件作用域进程跟踪
│   ├── builtin_hooks/        # 始终注册的钩子的扩展点（未包含）
│   └── platforms/            # 20 个适配器：telegram、discord、slack、whatsapp、
│                             #   signal、matrix、mattermost、email、sms、
│                             #   dingtalk、feishu、wecom、wecom_callback、weixin、
│                             #   bluebubbles、qqbot、homeassistant、webhook、api_server、
│                             #   yuanbao
│
├── acp_adapter/              # ACP 服务器（VS Code / Zed / JetBrains）
├── cron/                     # 调度器（jobs.py、scheduler.py）
├── plugins/memory/           # 内存 provider 插件
├── plugins/context_engine/   # 上下文引擎插件
├── environments/             # RL 训练环境（Atropos）
├── skills/                   # 捆绑技能（始终可用）
├── optional-skills/          # 官方可选技能（明确安装）
├── website/                  # Docusaurus 文档站
└── tests/                    # Pytest 套件（3,000+ 测试）
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
    → 最终响应 → 显示 → 保存到 SessionDB
```

### 网关消息

```text
平台事件 → Adapter.on_message() → MessageEvent
  → GatewayRunner._handle_message()
    → 授权用户
    → 解析会话键
    → 创建带会话历史的 AIAgent
    → AIAgent.run_conversation()
    → 通过适配器将响应传回
```

### Cron 作业

```text
调度器刻度 → 从 jobs.json 加载到期作业
  → 创建全新 AIAgent（无历史）
  → 将附加技能注入为上下文
  → 运行作业提示词
  → 将响应投递到目标平台
  → 更新作业状态和 next_run
```

## 推荐阅读顺序

如果您是代码库新手：

1. **此页面** — 定位自己
2. **[Agent 循环内部原理](./agent-loop.md)** — AIAgent 如何工作
3. **[提示词组装](./prompt-assembly.md)** — 系统提示词构建
4. **[Provider 运行时解析](./provider-runtime.md)** — 如何选择 provider
5. **[添加 Provider](./adding-providers.md)** — 添加新 provider 的实践指南
6. **[工具运行时](./tools-runtime.md)** — 工具注册表、调度、环境
7. **[会话存储](./session-storage.md)** — SQLite 模式、FTS5、会话谱系
8. **[网关内部原理](./gateway-internals.md)** — 消息平台网关
9. **[上下文压缩和提示缓存](./context-compression-and-caching.md)** — 压缩和缓存
10. **[ACP 内部原理](./acp-internals.md)** — IDE 集成
11. **[环境、基准和数据生成](./environments.md)** — RL 训练

## 主要子系统

### Agent 循环

同步编排引擎（`run_agent.py` 中的 `AIAgent`）。处理 provider 选择、提示词构建、工具执行、重试、回退、回调、压缩和持久化。支持三种 API 模式用于不同的 provider 后端。

→ [Agent 循环内部原理](./agent-loop.md)

### 提示词系统

跨对话生命周期的提示词构建和维护：

- **`prompt_builder.py`** — 从以下内容组装系统提示词：个性（SOUL.md）、内存（MEMORY.md、USER.md）、技能、上下文文件（AGENTS.md、.hermes.md）、工具使用指南和模型特定指令
- **`prompt_caching.py`** — 为前缀缓存应用 Anthropic 缓存断点
- **`context_compressor.py`** — 当上下文超过阈值时对中间对话轮次进行摘要

→ [提示词组装](./prompt-assembly.md)、[上下文压缩和提示缓存](./context-compression-and-caching.md)

### Provider 解析

CLI、网关、cron、ACP 和辅助调用使用的共享运行时解析器。将 `(provider, model)` 元组映射到 `(api_mode, api_key, base_url)`。处理 18+ 个 provider、OAuth 流程、凭据池和别名解析。

→ [Provider 运行时解析](./provider-runtime.md)

### 工具系统

中央工具注册表（`tools/registry.py`），包含 52 个工具集中的 61 个注册工具。每个工具文件在导入时自注册。注册表处理模式收集、调度、可用性检查和错误包装。终端工具支持 7 个后端（local、Docker、SSH、Daytona、Modal、Singularity、Vercel Sandbox）。

→ [工具运行时](./tools-runtime.md)

### 会话持久化

基于 SQLite 的会话存储，带 FTS5 全文搜索。会话具有谱系跟踪（跨压缩的父/子），每个平台隔离，以及带竞争处理的原子写入。

→ [会话存储](./session-storage.md)

### 消息网关

具有 20 个平台适配器、统一会话路由、用户授权（允许列表 + DM 配对）、斜杠命令调度、钩子系统、cron 刻度和后台维护的长时间运行进程。

→ [网关内部原理](./gateway-internals.md)

### 插件系统

三个发现源：`~/.hermes/plugins/`（用户）、`.hermes/plugins/`（项目）和 pip 入口点。插件通过上下文 API 注册工具、钩子和 CLI 命令。存在两种专用插件类型：内存 provider（`plugins/memory/`）和上下文引擎（`plugins/context_engine/`）。两者都是单选 — 每次只能激活一个，通过 `hermes plugins` 或 `config.yaml` 配置。

→ [插件指南](/docs/guides/build-a-hermes-plugin)、[内存 Provider 插件](./memory-provider-plugin.md)

### Cron

一流的代理任务（不是 shell 任务）。作业存储在 JSON 中，支持多种调度格式，可以附加技能和脚本，并投递到任何平台。

→ [Cron 内部原理](./cron-internals.md)

### ACP 集成

通过 stdio/JSON-RPC 将 Hermes 作为编辑器原生代理暴露给 VS Code、Zed 和 JetBrains。

→ [ACP 内部原理](./acp-internals.md)

### RL / 环境 / 轨迹

用于评估和 RL 训练的完整环境框架。与 Atropos 集成，支持多种工具调用解析器，并生成 ShareGPT 格式的轨迹。

→ [环境、基准和数据生成](./environments.md)、[轨迹和训练格式](./trajectory-format.md)

## 设计原则

| 原则 | 在实践中的含义 |
|-----------|--------------------------|
| **提示词稳定性** | 系统提示词在对话中不会改变。除明确的用户操作（`/model`）外，不会破坏缓存的变更。 |
| **可观察执行** | 每个工具调用通过回调对用户可见。CLI（旋转器）和网关（聊天消息）中的进度更新。 |
| **可中断** | API 调用和工具执行可以被用户输入或信号在飞行中取消。 |
| **平台无关的核心** | 一个 AIAgent 类服务 CLI、网关、ACP、批处理和 API 服务器。平台差异存在于入口点，而不是代理。 |
| **松耦合** | 可选子系统（MCP、插件、内存 provider、RL 环境）使用注册表模式和 check_fn 门控，而不是硬依赖。 |
| **配置文件隔离** | 每个配置文件（`hermes -p <name>`）都有自己的 HERMES_HOME、配置、内存、会话和网关 PID。多个配置文件并发运行。 |

## 文件依赖链

```text
tools/registry.py  (no deps — imported by all tool files)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (imports tools/registry + triggers tool discovery)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

此链意味着工具注册发生在导入时，在创建任何代理实例之前。任何带有顶层 `registry.register()` 调用的 `tools/*.py` 文件都会自动发现 — 不需要手动导入列表。
