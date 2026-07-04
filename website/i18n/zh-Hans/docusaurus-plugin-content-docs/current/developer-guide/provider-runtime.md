---
sidebar_position: 4
title: "Provider 运行时解析"
description: "Hermes 如何在运行时解析 provider、凭据、API 模式及辅助模型"
---

# Provider 运行时解析

Hermes 拥有一个共享的 provider 运行时解析器，用于以下场景：

- CLI
- gateway
- cron 任务
- ACP
- 辅助模型调用

主要实现：

- `hermes_cli/runtime_provider.py` — 凭据解析，`_resolve_custom_runtime()`
- `hermes_cli/auth.py` — provider 注册表，`resolve_provider()`
- `hermes_cli/model_switch.py` — 共享 `/model` 切换流水线（CLI + gateway）
- `agent/auxiliary_client.py` — 辅助模型路由
- `providers/` — ABC + 注册表入口点（`ProviderProfile`、`register_provider`、`get_provider_profile`、`list_providers`）
- `plugins/model-providers/<name>/` — 每个 provider 的插件（内置），声明 `api_mode`、`base_url`、`env_vars`、`fallback_models` 并在首次访问时将自身注册到注册表。用户插件位于 `$HERMES_HOME/plugins/model-providers/<name>/`，会覆盖同名的内置插件。

`providers/` 中的 `get_provider_profile()` 为给定 provider id 返回一个 `ProviderProfile`。`runtime_provider.py` 在解析时调用它，以获取规范的 `base_url`、`env_vars` 优先级列表、`api_mode` 和 `fallback_models`，无需在多个文件中重复这些数据。在 `plugins/model-providers/<your-provider>/`（或 `$HERMES_HOME/plugins/model-providers/<your-provider>/`）下添加一个调用 `register_provider()` 的新插件，即可让 `runtime_provider.py` 自动识别它——无需在解析器本身中添加分支。

如果你想添加一个新的一等推理 provider，请结合本页阅读 [添加 Provider](./adding-providers.md) 和 [Model Provider 插件指南](./model-provider-plugin.md)。

## 解析优先级

从高层来看，provider 解析使用以下顺序：

1. 显式 CLI/运行时请求
2. `config.yaml` 中的模型/provider 配置
3. 环境变量
4. provider 特定的默认值或自动解析

该顺序很重要，因为 Hermes 将已保存的模型/provider 选择视为正常运行的真实来源。这可以防止过时的 shell 导出变量悄悄覆盖用户在 `hermes model` 中最后选择的端点。

## Provider

当前 provider 系列包括（完整内置集合见 `plugins/model-providers/`）：

- OpenRouter
- Nous Portal
- OpenAI Codex
- Copilot / Copilot ACP
- Anthropic（原生）
- Google / Gemini（`gemini`）
- Alibaba / DashScope（`alibaba`、`alibaba-coding-plan`）
- DeepSeek
- Z.AI
- Kimi / Moonshot（`kimi-coding`、`kimi-coding-cn`）
- MiniMax（`minimax`、`minimax-cn`、`minimax-oauth`）
- Kilo Code
- Hugging Face
- OpenCode Zen / OpenCode Go
- AWS Bedrock
- Azure Foundry
- NVIDIA NIM
- xAI（Grok）
- Arcee
- GMI Cloud
- StepFun
- Qwen OAuth
- Xiaomi
- Ollama Cloud
- LM Studio
- Tencent TokenHub
- Custom（`provider: custom`）— 适用于任何 OpenAI 兼容端点的一等 provider
- 命名自定义 provider（`config.yaml` 中的 `custom_providers` 列表）

## 运行时解析的输出

运行时解析器返回的数据包括：

- `provider`
- `api_mode`
- `base_url`
- `api_key`
- `source`
- provider 特定的元数据，如过期/刷新信息

## 为什么这很重要

该解析器是 Hermes 能够在以下场景之间共享认证/运行时逻辑的主要原因：

- `hermes chat`
- gateway 消息处理
- 在全新会话中运行的 cron 任务
- ACP 编辑器会话
- 辅助模型任务

## OpenRouter 与自定义 OpenAI 兼容 base URL

Hermes 包含相关逻辑，以避免在存在多个 provider 密钥时（例如同时存在 `OPENROUTER_API_KEY` 和 `OPENAI_API_KEY`）将错误的 API key 泄露给自定义端点。

每个 provider 的 API key 仅作用于其自身的 base URL：

- `OPENROUTER_API_KEY` 仅发送至 `openrouter.ai` 端点
- `OPENAI_API_KEY` 用于自定义端点及作为回退

Hermes 还区分以下两种情况：

- 用户主动选择的真实自定义端点
- 未配置自定义端点时使用的 OpenRouter 回退路径

这种区分对以下场景尤为重要：

- 本地模型服务器
- 非 OpenRouter 的 OpenAI 兼容 API
- 无需重新运行 setup 即可切换 provider
- 通过 config 保存的自定义端点，即使当前 shell 中未导出 `OPENAI_BASE_URL` 也应正常工作

## 原生 Anthropic 路径

Anthropic 不再仅限于"通过 OpenRouter"访问。

当 provider 解析选择 `anthropic` 时，Hermes 使用：

- `api_mode = anthropic_messages`
- 原生 Anthropic Messages API
- `agent/anthropic_adapter.py` 进行转换

原生 Anthropic 的凭据解析现在在两者同时存在时，优先使用可刷新的 Claude Code 凭据，而非复制的环境变量 token。实际效果为：

- 包含可刷新认证的 Claude Code 凭据文件被视为首选来源
- 手动设置的 `ANTHROPIC_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN` 值仍可作为显式覆盖
- Hermes 在调用原生 Messages API 前会预检 Anthropic 凭据刷新
- Hermes 在重建 Anthropic 客户端后，仍会在收到 401 时重试一次，作为回退路径

## OpenAI Codex 路径

Codex 使用独立的 Responses API 路径：

- `api_mode = codex_responses`
- 专用的凭据解析和认证存储支持

## 辅助模型路由

辅助任务包括：

- 视觉
- 网页提取摘要
- 上下文压缩摘要
- skills hub 操作
- MCP 辅助操作
- 记忆刷新

这些任务可以使用各自独立的 provider/模型路由，而非主对话模型。

当辅助任务配置的 provider 为 `main` 时，Hermes 通过与普通对话相同的共享运行时路径进行解析。实际效果为：

- 环境变量驱动的自定义端点仍然有效
- 通过 `hermes model` / `config.yaml` 保存的自定义端点同样有效
- 辅助路由能够区分真实保存的自定义端点与 OpenRouter 回退

## 回退模型

Hermes 支持配置回退 provider 链——一个按顺序尝试的 `(provider, model)` 条目列表，当主模型遇到错误时依次尝试。旧版单对 `fallback_model` 字典仍被接受以保持向后兼容（并在首次写入时迁移）。

### 内部工作原理

1. **存储**：`AIAgent.__init__` 存储 `fallback_model` 字典并将 `_fallback_activated` 设为 `False`。

2. **触发点**：`_try_activate_fallback()` 在 `run_agent.py` 主重试循环的三处被调用：
   - 在无效 API 响应（None choices、缺少 content）达到最大重试次数后
   - 在不可重试的客户端错误（HTTP 401、403、404）时
   - 在瞬时错误（HTTP 429、500、502、503）达到最大重试次数后

3. **激活流程**（`_try_activate_fallback`）：
   - 若已激活或未配置，立即返回 `False`
   - 调用 `auxiliary_client.py` 中的 `resolve_provider_client()` 构建带有正确认证的新客户端
   - 确定 `api_mode`：openai-codex 使用 `codex_responses`，anthropic 使用 `anthropic_messages`，其余使用 `chat_completions`
   - 原地替换：`self.model`、`self.provider`、`self.base_url`、`self.api_mode`、`self.client`、`self._client_kwargs`
   - 对于 anthropic 回退：构建原生 Anthropic 客户端而非 OpenAI 兼容客户端
   - 重新评估 prompt 缓存（对 OpenRouter 上的 Claude 模型启用）
   - 将 `_fallback_activated` 设为 `True`——防止再次触发
   - 将重试计数重置为 0 并继续循环

4. **配置流程**：
   - CLI：`cli.py` 读取 `CLI_CONFIG["fallback_model"]` → 传递给 `AIAgent(fallback_model=...)`
   - Gateway：`gateway/run.py._load_fallback_model()` 读取 `config.yaml` → 传递给 `AIAgent`
   - 验证：`provider` 和 `model` 键均须非空，否则回退被禁用

### 不支持回退的场景

- **子代理委托**（`tools/delegate_tool.py`）：子代理继承父代理的 provider，但不继承回退配置
- **辅助任务**：使用各自独立的 provider 自动检测链（见上方辅助模型路由）

Cron 任务**支持**回退：`run_job()` 从 `config.yaml` 读取 `fallback_providers`（或旧版 `fallback_model`）并传递给 `AIAgent(fallback_model=...)`，与 gateway 的 `_load_fallback_model()` 模式一致。参见 [Cron 内部机制](./cron-internals.md)。

### 测试覆盖

参见 `tests/test_fallback_model.py`，其中包含覆盖所有支持 provider、单次触发语义及边界情况的完整测试。

## 相关文档

- [Agent 循环内部机制](./agent-loop.md)
- [ACP 内部机制](./acp-internals.md)
- [上下文压缩与 Prompt 缓存](./context-compression-and-caching.md)