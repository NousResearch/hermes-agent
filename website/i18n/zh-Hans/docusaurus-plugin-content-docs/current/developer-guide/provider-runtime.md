---
sidebar_position: 4
title: "Provider 运行时解析"
description: "Hermes 如何在运行时解析 provider、凭据、API 模式和辅助模型"
---

# Provider 运行时解析

Hermes 有一个跨以下内容使用的共享 provider 运行时解析器：

- CLI
- 网关
- cron 作业
- ACP
- 辅助模型调用

主要实现：

- `hermes_cli/runtime_provider.py` — 凭据解析、`_resolve_custom_runtime()`
- `hermes_cli/auth.py` — provider 注册表、`resolve_provider()`
- `hermes_cli/model_switch.py` — 共享 `/model` 切换管道（CLI + 网关）
- `agent/auxiliary_client.py` — 辅助模型路由
- `providers/` — ABC + 注册表入口点（`ProviderProfile`、`register_provider`、`get_provider_profile`、`list_providers`）
- `plugins/model-providers/<name>/` — 每个 provider 插件（捆绑），声明 `api_mode`、`base_url`、`env_vars`、`fallback_models` 并在首次访问时将自己注册到注册表。用户插件在 `$HERMES_HOME/plugins/model-providers/<name>/` 覆盖同名捆绑插件。

`providers/` 中的 `get_provider_profile()` 返回给定 provider id 的 `ProviderProfile`。`runtime_provider.py` 在解析时调用它以获取规范 `base_url`、`env_vars` 优先级列表、`api_mode` 和 `fallback_models`，而无需在多个文件中复制该数据。在 `plugins/model-providers/<your-provider>/`（或 `$HERMES_HOME/plugins/model-providers/<your-provider>/`）下添加调用 `register_provider()` 的新插件足以让 `runtime_provider.py` 拾取它 — 无需在解析器本身中添加分支。

如果您尝试添加新的一流推理 provider，请阅读[添加 Provider](./adding-providers.md)以及本页面旁边的[模型 Provider 插件指南](./model-provider-plugin.md)。

## 解析优先级

在高级别上，provider 解析使用：

1. 显式 CLI/运行时请求
2. `config.yaml` 模型/provider 配置
3. 环境变量
4. provider 特定默认值或自动解析

这个顺序很重要，因为 Hermes 将保存的模型/provider 选择作为正常运行的真实来源。这防止过时的 shell 导出静默覆盖用户上次在 `hermes model` 中选择的端点。

## Provider

当前 provider 系列包括：

- AI Gateway (Vercel)
- OpenRouter
- Nous Portal
- OpenAI Codex
- Copilot / Copilot ACP
- Anthropic（原生）
- Google / Gemini
- Alibaba / DashScope
- DeepSeek
- Z.AI
- Kimi / Moonshot
- MiniMax
- MiniMax China
- Kilo Code
- Hugging Face
- OpenCode Zen / OpenCode Go
- Custom（`provider: custom`）— 任何 OpenAI 兼容端点的一流 provider
- Named custom providers（`config.yaml` 中的 `custom_providers` 列表）

## 运行时解析的输出

运行时解析器返回数据如：

- `provider`
- `api_mode`
- `base_url`
- `api_key`
- `source`
- 特定于 provider 的元数据如 expiry/refresh 信息

## 为什么这很重要

此解析器是 Hermes 可以在以下之间共享 auth/运行时逻辑的主要原因：

- `hermes chat`
- 网关消息处理
- 在全新会话中运行的 cron 作业
- ACP 编辑器会话
- 辅助模型任务

## AI Gateway

在 `~/.hermes/.env` 中设置 `AI_GATEWAY_API_KEY` 并使用 `--provider ai-gateway` 运行。Hermes 从网关的 `/models` 端点获取可用模型，过滤到支持工具使用的语言模型。

## OpenRouter、AI Gateway 和自定义 OpenAI 兼容基础 URL

Hermes 包含逻辑避免在存在多个 provider 密钥时将错误的 API 密钥泄露给自定义端点（例如 `OPENROUTER_API_KEY`、`AI_GATEWAY_API_KEY` 和 `OPENAI_API_KEY`）。

每个 provider 的 API 密钥仅限于其自己的基础 URL：

- `OPENROUTER_API_KEY` 仅发送到 `openrouter.ai` 端点
- `AI_GATEWAY_API_KEY` 仅发送到 `ai-gateway.vercel.sh` 端点
- `OPENAI_API_KEY` 用于自定义端点并作为后备

Hermes 还区分：

- 用户选择的真实自定义端点
- 未配置自定义端点时使用的 OpenRouter 后备路径

这种区别对于以下情况特别重要：

- 本地模型服务器
- 非 OpenRouter/非 AI Gateway OpenAI 兼容 API
- 切换 provider 而无需重新运行设置
- config 保存的自定义端点，即使当前 shell 中未导出 `OPENAI_BASE_URL` 也应继续工作

## 原生 Anthropic 路径

Anthropic 不再只是"通过 OpenRouter"。

当 provider 解析选择 `anthropic` 时，Hermes 使用：

- `api_mode = anthropic_messages`
- 原生 Anthropic Messages API
- `agent/anthropic_adapter.py` 用于翻译

原生 Anthropic 的凭据解析现在在存在可刷新 Claude Code 凭据时优先于复制的 env token。实际上意味着：

- 包含可刷新 auth 的 Claude Code 凭据文件被视为首选来源
- 手动 `ANTHROPIC_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN` 值仍作为显式覆盖工作
- Hermes 在原生 Messages API 调用之前预检 Anthropic 凭据刷新
- Hermes 在重建 Anthropic 客户端后仍在 401 上重试一次，作为后备路径

## OpenAI Codex 路径

Codex 使用单独的 Responses API 路径：

- `api_mode = codex_responses`
- 专用凭据解析和 auth store 支持

## 辅助模型路由

辅助任务如：

- 视觉
- Web 提取摘要
- 上下文压缩摘要
- 会话搜索摘要
- 技能中心操作
- MCP 辅助操作
- 内存刷新

可以使用自己的 provider/模型路由而不是主对话模型。

当辅助任务配置 provider `main` 时，Hermes 通过与正常聊天相同的共享运行时路径解析它。实际上意味着：

- env 驱动的自定义端点仍然工作
- 通过 `hermes model` / `config.yaml` 保存的自定义端点也工作
- 辅助路由可以区分真实保存的自定义端点和 OpenRouter 后备

## 回退模型

Hermes 支持配置的回退模型/provider 对，允许在主模型遇到错误时进行运行时故障切换。

### 内部工作原理

1. **存储**：`AIAgent.__init__` 存储 `fallback_model` 字典并设置 `_fallback_activated = False`。

2. **触发点**：`run_agent.py` 主重试循环中的三个地方调用 `_try_activate_fallback()`：
   - 在无效 API 响应（None choices、缺失内容）上最大重试后
   - 在不可重试客户端错误（HTTP 401、403、404）时
   - 在瞬态错误（HTTP 429、500、502、503）上最大重试后

3. **激活流程**（`_try_activate_fallback`）：
   - 如果已激活或未配置则立即返回 `False`
   - 从 `auxiliary_client.py` 调用 `resolve_provider_client()` 构建带有正确 auth 的新客户端
   - 确定 `api_mode`：openai-codex 为 `codex_responses`、anthropic 为 `anthropic_messages`、其他为 `chat_completions`
   - 原地交换：`self.model`、`self.provider`、`self.base_url`、`self.api_mode`、`self.client`、`self._client_kwargs`
   - 对于 anthropic 回退：构建原生 Anthropic 客户端而不是 OpenAI 兼容客户端
   - 重新评估提示词缓存（OpenRouter 上为 Claude 模型启用）
   - 设置 `_fallback_activated = True` — 防止再次触发
   - 将重试计数重置为 0 并继续循环

4. **配置流程**：
   - CLI：`cli.py` 读取 `CLI_CONFIG["fallback_model"]` → 传递给 `AIAgent(fallback_model=...)`
   - 网关：`gateway/run.py._load_fallback_model()` 读取 `config.yaml` → 传递给 `AIAgent`
   - 验证：`provider` 和 `model` 键都必须非空，否则回退被禁用

### 不支持回退的内容

- **子代理委托**（`tools/delegate_tool.py`）：子代理继承父代理的 provider 但不继承回退配置
- **辅助任务**：使用自己的独立 provider 自动检测链（见上文辅助模型路由）

Cron 作业**确实**支持回退：`run_job()` 从 `config.yaml` 读取 `fallback_providers`（或旧版 `fallback_model`）并将其传递给 `AIAgent(fallback_model=...)`，匹配网关的 `_load_fallback_model()` 模式。见 [Cron 内部原理](./cron-internals.md)。

### 测试覆盖

参见 `tests/test_fallback_model.py` 获取涵盖所有支持的 provider、一次性语义和边缘情况的综合测试。

## 相关文档

- [Agent 循环内部原理](./agent-loop.md)
- [ACP 内部原理](./acp-internals.md)
- [上下文压缩和提示缓存](./context-compression-and-caching.md)
