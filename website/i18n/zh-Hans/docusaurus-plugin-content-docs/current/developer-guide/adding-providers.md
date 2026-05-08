---
sidebar_position: 5
title: "添加 Provider"
description: "如何向 Hermes Agent 添加新的推理 provider — auth、运行时解析、CLI 流程、适配器、测试和文档"
---

# 添加 Provider

Hermes 已经可以通过自定义 provider 路径与任何 OpenAI 兼容的端点通信。除非您想要该服务的最佳 UX，否则不要添加内置 provider：

- 特定于 provider 的 auth 或 token 刷新
- 精选的模型目录
- 设置 / `hermes model` 菜单条目
- `provider:model` 语法的 provider 别名
- 需要适配器的非 OpenAI API 形状

如果 provider 只是"另一个 OpenAI 兼容的基础 URL 和 API 密钥"，命名自定义 provider 可能就足够了。

## 思维模型

内置 provider 必须跨几个层面协调：

1. `hermes_cli/auth.py` 决定如何找到凭据。
2. `hermes_cli/runtime_provider.py` 将其转换为运行时数据：
   - `provider`
   - `api_mode`
   - `base_url`
   - `api_key`
   - `source`
3. `run_agent.py` 使用 `api_mode` 决定如何构建和发送请求。
4. `hermes_cli/models.py` 和 `hermes_cli/main.py` 使 provider 出现在 CLI 中。（`hermes_cli/setup.py` 自动委托给 `main.py` — 无需更改。）
5. `agent/auxiliary_client.py` 和 `agent/model_metadata.py` 保持辅助任务和 token 预算工作。

重要的抽象是 `api_mode`。

- 大多数 provider 使用 `chat_completions`。
- Codex 使用 `codex_responses`。
- Anthropic 使用 `anthropic_messages`。
- 新的非 OpenAI 协议通常意味着添加新的适配器和新的 `api_mode` 分支。

## 首先选择实现路径

### 路径 A — OpenAI 兼容 provider

当 provider 接受标准 chat-completions 样式的请求时使用。

典型工作：

- 添加 auth 元数据
- 添加模型目录 / 别名
- 添加运行时解析
- 添加 CLI 菜单接线
- 添加 aux 模型默认值
- 添加测试和用户文档

通常不需要新的适配器或新的 `api_mode`。

### 路径 B — 原生 provider

当 provider 的行为不像 OpenAI chat completions 时使用。

当前树中的示例：

- `codex_responses`
- `anthropic_messages`

此路径包括路径 A 的所有内容加上：

- `agent/` 中的 provider 适配器
- `run_agent.py` 中用于请求构建、调度、使用量提取、中断处理和响应规范化的分支
- 适配器测试

## 文件清单

### 每个内置 provider 必需

1. `hermes_cli/auth.py`
2. `hermes_cli/models.py`
3. `hermes_cli/runtime_provider.py`
4. `hermes_cli/main.py`
5. `agent/auxiliary_client.py`
6. `agent/model_metadata.py`
7. tests
8. `website/docs/` 下的用户文档

:::tip
`hermes_cli/setup.py` **不需要**更改。设置向导委托 provider/模型选择到 `main.py` 中的 `select_provider_and_model()` — 在那里添加的任何 provider 自动在 `hermes setup` 中可用。
:::

### 原生 / 非 OpenAI provider 额外需要

10. `agent/<provider>_adapter.py`
11. `run_agent.py`
12. `pyproject.toml` 如果需要 provider SDK

## 快速路径：简单的 API 密钥 provider

如果您的 provider 只是一个使用单个 API 密钥进行身份验证的 OpenAI 兼容端点，您不需要触摸 `auth.py`、`runtime_provider.py`、`main.py` 或完整清单中的任何其他文件。

您只需要：

1. 在 `plugins/model-providers/<your-provider>/` 下创建一个插件目录，包含：
   - `__init__.py` — 在模块级别调用 `register_provider(profile)`
   - `plugin.yaml` — 清单（name、kind: model-provider、version、description）
2. 就这样。当任何代码第一次调用 `get_provider_profile()` 或 `list_providers()` 时，provider 插件会自动加载 — 捆绑插件（此仓库）和 `$HERMES_HOME/plugins/model-providers/` 中的用户插件都会被拾取。

当您添加插件并调用 `register_provider()` 时，以下内容自动接线：

1. `auth.py` 中的 `PROVIDER_REGISTRY` 条目（凭据解析、env 变量查找）
2. `api_mode` 设置为 `chat_completions`
3. `base_url` 从配置或声明的 env 变量获取
4. 按优先级顺序检查 `env_vars` 获取 API 密钥
5. 为 provider 注册 `fallback_models` 列表
6. `--provider` CLI 标志接受 provider id
7. `hermes model` 菜单包含 provider
8. `hermes setup` 向导自动委托给 `main.py`
9. `provider:model` 别名语法有效
10. 运行时解析器返回正确的 `base_url` 和 `api_key`
11. `HERMES_INFERENCE_PROVIDER` env 变量覆盖接受 provider id
12. 回退模型激活可以干净地切换到 provider

`$HERMES_HOME/plugins/model-providers/<name>/` 中的用户插件覆盖同名捆绑插件（`register_provider()` 中后写入者获胜）— 因此第三方可以在不编辑仓库的情况下替换或修补任何内置配置文件。

参见 `plugins/model-providers/nvidia/` 或 `plugins/model-providers/gmi/` 作为模板，以及完整的 [Model Provider Plugin 指南](/docs/developer-guide/model-provider-plugin) 获取字段参考、钩子习惯用法和端到端示例。

## 完整路径：OAuth 和复杂 provider

当您的 provider 需要以下任何一项时，使用完整清单：

- OAuth 或 token 刷新（Nous Portal、Codex、Google Gemini、Qwen Portal、Copilot）
- 需要新适配器的非 OpenAI API 形状（Anthropic Messages、Codex Responses）
- 自定义端点检测或多区域探测（z.ai、Kimi）
- 精选的静态模型目录或实时 `/models` 获取
- 带自定义 auth 流程的特定于 provider 的 `hermes model` 菜单条目

## 步骤 1：选择一个规范 provider id

选择一个 provider id 并在各处使用它。

仓库中的示例：

- `openai-codex`
- `kimi-coding`
- `minimax-cn`

相同的 id 应该出现在：

- `hermes_cli/auth.py` 中的 `PROVIDER_REGISTRY`
- `hermes_cli/models.py` 中的 `_PROVIDER_LABELS`
- `hermes_cli/auth.py` 和 `hermes_cli/models.py` 中的 `_PROVIDER_ALIASES`
- `hermes_cli/main.py` 中的 CLI `--provider` 选项
- 设置 / 模型选择分支
- aux 模型默认值
- tests

如果 id 在这些文件之间不同，provider 会感觉接了一半：auth 可能工作，而 `/model`、setup 或运行时解析静默遗漏。

## 步骤 2：在 `hermes_cli/auth.py` 中添加 auth 元数据

对于 API 密钥 provider，在 `PROVIDER_REGISTRY` 中添加 `ProviderConfig` 条目：

- `id`
- `name`
- `auth_type="api_key"`
- `inference_base_url`
- `api_key_env_vars`
- 可选的 `base_url_env_var`

还要在 `_PROVIDER_ALIASES` 中添加别名。

使用现有 provider 作为模板：

- 简单 API 密钥路径：Z.AI、MiniMax
- 带端点检测的 API 密钥路径：Kimi、Z.AI
- 原生 token 解析：Anthropic
- OAuth / auth-store 路径：Nous、OpenAI Codex

这里要回答的问题：

- Hermes 应该检查哪些 env 变量，按什么优先级顺序？
- Provider 需要 base-URL 覆盖吗？
- 需要端点探测或 token 刷新吗？
- 凭据缺失时 auth 错误应该说什么？

如果 provider 需要比"查找 API 密钥"更复杂的东西，添加专用凭据解析器，而不是将逻辑塞入不相关的分支。

## 步骤 3：在 `hermes_cli/models.py` 中添加模型目录和别名

更新 provider 目录以使 provider 在菜单和 `provider:model` 语法中工作。

典型编辑：

- `_PROVIDER_MODELS`
- `_PROVIDER_LABELS`
- `_PROVIDER_ALIASES`
- `list_available_providers()` 中的 provider 显示顺序
- `provider_model_ids()` 如果 provider 支持实时 `/models` 获取

如果 provider 公开实时模型列表，优先使用它，并将 `_PROVIDER_MODELS` 保持为静态后备。

此文件还使以下输入有效：

```text
anthropic:claude-sonnet-4-6
kimi:model-name
```

如果别名在这里缺失，provider 可能正确认证但仍在 `/model` 解析中失败。

## 步骤 4：在 `hermes_cli/runtime_provider.py` 中解析运行时数据

`resolve_runtime_provider()` 是 CLI、网关、cron、ACP 和辅助客户端使用的共享路径。

添加一个分支，返回至少包含以下内容的字典：

```python
{
    "provider": "your-provider",
    "api_mode": "chat_completions",  # or your native mode
    "base_url": "https://...",
    "api_key": "...",
    "source": "env|portal|auth-store|explicit",
    "requested_provider": requested_provider,
}
```

如果 provider 是 OpenAI 兼容的，`api_mode` 通常应保持 `chat_completions`。

小心 API 密钥优先级。Hermes 已经包含避免将 OpenRouter 密钥泄露给无关端点的逻辑。新 provider 应同样明确哪个密钥发送到哪个基础 URL。

## 步骤 5：在 `hermes_cli/main.py` 中接线 CLI

除非 provider 出现在交互式 `hermes model` 流程中，否则它不可被发现。

在 `hermes_cli/main.py` 中更新：

- `provider_labels` 字典
- `select_provider_and_model()` 中的 `providers` 列表
- provider 调度（`if selected_provider == ...`）
- `--provider` 参数选项
- 如果 provider 支持这些流程，则添加 login/logout 选项
- `_model_flow_<provider>()` 函数，或者如果适合则重用 `_model_flow_api_key_provider()`

:::tip
`hermes_cli/setup.py` 不需要更改 — 它从 `main.py` 调用 `select_provider_and_model()`，因此您的新 provider 自动出现在 `hermes model` 和 `hermes setup` 中。
:::

## 步骤 6：保持辅助调用工作

这里有两个文件很重要：

### `agent/auxiliary_client.py`

如果这是直接 API 密钥 provider，在 `_API_KEY_PROVIDER_AUX_MODELS` 中添加便宜的 / 快速的默认 aux 模型。

辅助任务包括：

- 视觉摘要
- Web 提取摘要
- 上下文压缩摘要
- 会话搜索摘要
- 内存刷新

如果 provider 没有合理的 aux 默认值，辅助任务可能会严重回退或意外使用昂贵的主模型。

### `agent/model_metadata.py`

添加 provider 模型上下文长度以使 token 预算、压缩阈值和限制保持合理。

## 步骤 7：如果 provider 是原生的，添加适配器和 `run_agent.py` 支持

如果 provider 不是纯 chat completions，将特定于 provider 的逻辑隔离在 `agent/<provider>_adapter.py` 中。

保持 `run_agent.py` 专注于编排。它应该调用适配器辅助函数，而不是在文件各处内联构建 provider 负载。

原生 provider 通常需要在这些地方工作：

### 新适配器文件

典型职责：

- 构建 SDK / HTTP 客户端
- 解析 tokens
- 将 OpenAI 风格的对话消息转换为 provider 的请求格式
- 必要时转换工具模式
- 将 provider 响应规范化回 `run_agent.py` 期望的格式
- 提取使用量和 finish-reason 数据

### `run_agent.py`

搜索 `api_mode` 并审核每个切换点。至少验证：

- `__init__` 选择新的 `api_mode`
- 客户端构建适用于 provider
- `_build_api_kwargs()` 知道如何格式化请求
- `_interruptible_api_call()` 调度到正确的客户端调用
- 中断 / 客户端重建路径工作
- 响应验证接受 provider 的形状
- finish-reason 提取正确
- token 使用量提取正确
- 回退模型激活可以干净地切换到新 provider
- 摘要生成和内存刷新路径仍然工作

还要在 `run_agent.py` 中搜索 `self.client.`。任何假设标准 OpenAI 客户端存在的代码路径在原生 provider 使用不同客户端对象或 `self.client = None` 时都可能中断。

### 提示缓存和特定于 provider 的请求字段

提示缓存和特定于 provider 的旋钮容易回归。

树中已有的示例：

- Anthropic 有原生提示缓存路径
- OpenRouter 获取 provider 路由字段
- 不是每个 provider 都应该接收每个请求端选项

添加原生 provider 时，仔细检查 Hermes 仅发送该 provider 实际理解的字段。

## 步骤 8：测试

至少触及保护 provider 接线的测试。

常见位置：

- `tests/test_runtime_provider_resolution.py`
- `tests/test_cli_provider_resolution.py`
- `tests/test_cli_model_command.py`
- `tests/test_setup_model_selection.py`
- `tests/test_provider_parity.py`
- `tests/test_run_agent.py`
- `tests/test_<provider>_adapter.py` 用于原生 provider

对于仅文档示例，确切的文件集可能不同。关键是覆盖：

- auth 解析
- CLI 菜单 / provider 选择
- 运行时 provider 解析
- 代理执行路径
- provider:model 解析
- 任何适配器特定的消息转换

运行测试禁用 xdist：

```bash
source venv/bin/activate
python -m pytest tests/test_runtime_provider_resolution.py tests/test_cli_provider_resolution.py tests/test_cli_model_command.py tests/test_setup_model_selection.py -n0 -q
```

对于更深入的更改，推送前运行完整套件：

```bash
source venv/bin/activate
python -m pytest tests/ -n0 -q
```

## 步骤 9：实时验证

测试后，运行真实的冒烟测试。

```bash
source venv/bin/activate
python -m hermes_cli.main chat -q "Say hello" --provider your-provider --model your-model
```

还要测试交互流程如果您更改了菜单：

```bash
source venv/bin/activate
python -m hermes_cli.main model
python -m hermes_cli.main setup
```

对于原生 provider，至少验证一个工具调用，而不仅仅是纯文本响应。

## 步骤 10：更新面向用户的文档

如果 provider 旨在作为一流选项发货，也要更新用户文档：

- `website/docs/getting-started/quickstart.md`
- `website/docs/user-guide/configuration.md`
- `website/docs/reference/environment-variables.md`

开发者可以完美地接线 provider，但仍可能使用户无法发现所需的 env 变量或设置流程。

## OpenAI 兼容 provider 清单

如果 provider 是标准 chat completions，使用此清单。

- [ ] 在 `hermes_cli/auth.py` 中添加 `ProviderConfig`
- [ ] 在 `hermes_cli/auth.py` 和 `hermes_cli/models.py` 中添加别名
- [ ] 在 `hermes_cli/models.py` 中添加模型目录
- [ ] 在 `hermes_cli/runtime_provider.py` 中添加运行时分支
- [ ] 在 `hermes_cli/main.py` 中添加 CLI 接线（setup.py 自动继承）
- [ ] 在 `agent/auxiliary_client.py` 中添加 aux 模型
- [ ] 在 `agent/model_metadata.py` 中添加上下文长度
- [ ] 更新运行时 / CLI 测试
- [ ] 更新用户文档

## 原生 provider 清单

当 provider 需要新协议路径时使用。

- [ ] OpenAI 兼容清单中的所有内容
- [ ] 在 `agent/<provider>_adapter.py` 中添加适配器
- [ ] `run_agent.py` 中支持新的 `api_mode`
- [ ] 中断 / 重建路径工作
- [ ] 使用量和 finish-reason 提取工作
- [ ] 回退路径工作
- [ ] 添加适配器测试
- [ ] 实时冒烟测试通过

## 常见陷阱

### 1. 将 provider 添加到 auth 但不添加到模型解析

这使凭据正确解析，而 `/model` 和 `provider:model` 输入失败。

### 2. 忘记 `config["model"]` 可以是字符串或字典

很多 provider 选择代码必须规范化两种形式。

### 3. 假设需要内置 provider

如果服务只是 OpenAI 兼容的，自定义 provider 可能已经以更少的维护解决用户问题。

### 4. 忘记辅助路径

主聊天路径可能工作，而摘要、内存刷新或视觉助手失败，因为 aux 路由从未更新。

### 5. `run_agent.py` 中隐藏的原生 provider 分支

搜索 `api_mode` 和 `self.client.`。不要假设明显的请求路径是唯一的。

### 6. 向其他 provider 发送仅 OpenRouter 的旋钮

像 provider 路由这样的字段只属于支持它们的 provider。

### 7. 更新 `hermes model` 但不更新 `hermes setup`

两个流程都需要知道 provider。

## 实现时的好搜索目标

如果您在寻找 provider 触及的所有位置，搜索这些符号：

- `PROVIDER_REGISTRY`
- `_PROVIDER_ALIASES`
- `_PROVIDER_MODELS`
- `resolve_runtime_provider`
- `_model_flow_`
- `select_provider_and_model`
- `api_mode`
- `_API_KEY_PROVIDER_AUX_MODELS`
- `self.client.`

## 相关文档

- [Provider 运行时解析](./provider-runtime.md)
- [架构](./architecture.md)
- [贡献](./contributing.md)
