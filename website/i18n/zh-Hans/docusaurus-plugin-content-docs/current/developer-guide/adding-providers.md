---
sidebar_position: 5
title: "添加 Provider"
description: "如何向 Hermes Agent 添加新的推理 provider——认证、运行时解析、CLI 流程、适配器、测试与文档"
---

# 添加 Provider

Hermes 已经可以通过自定义 provider 路径与任何 OpenAI 兼容的端点通信。除非你需要为某个服务提供一流的用户体验，否则不要添加内置 provider：

- provider 专属的认证或 token 刷新
- 精选的模型目录
- setup / `hermes model` 菜单条目
- 用于 `provider:model` 语法的 provider 别名
- 需要适配器的非 OpenAI API 格式

如果该 provider 只是"另一个 OpenAI 兼容的 base URL 和 API key"，一个命名的自定义 provider 可能就足够了。

## 心智模型

内置 provider 需要在几个层面保持一致：

1. `hermes_cli/auth.py` 决定如何查找凭据。
2. `hermes_cli/runtime_provider.py` 将其转换为运行时数据：
   - `provider`
   - `api_mode`
   - `base_url`
   - `api_key`
   - `source`
3. `run_agent.py` 使用 `api_mode` 决定如何构建和发送请求。
4. `hermes_cli/models.py` 和 `hermes_cli/main.py` 使 provider 在 CLI 中可见。（`hermes_cli/setup.py` 自动委托给 `main.py`——无需在此处做任何修改。）
5. `agent/auxiliary_client.py` 和 `agent/model_metadata.py` 保持辅助任务和 token 预算正常运作。

核心抽象是 `api_mode`。

- 大多数 provider 使用 `chat_completions`。
- Codex 使用 `codex_responses`。
- Anthropic 使用 `anthropic_messages`。
- 新的非 OpenAI 协议通常意味着需要添加新的适配器和新的 `api_mode` 分支。

## 首先选择实现路径

### 路径 A——OpenAI 兼容 provider

当 provider 接受标准 chat-completions 风格的请求时使用此路径。

典型工作：

- 添加认证元数据
- 添加模型目录 / 别名
- 添加运行时解析
- 添加 CLI 菜单接线
- 添加辅助模型默认值
- 添加测试和用户文档

通常不需要新的适配器或新的 `api_mode`。

### 路径 B——原生 provider

当 provider 的行为与 OpenAI chat completions 不同时使用此路径。

当前代码库中的示例：

- `codex_responses`
- `anthropic_messages`

此路径包含路径 A 的所有内容，另加：

- `agent/` 中的 provider 适配器
- `run_agent.py` 中用于请求构建、分发、用量提取、中断处理和响应规范化的分支
- 适配器测试

## 文件清单

### 每个内置 provider 都必须修改

1. `hermes_cli/auth.py`
2. `hermes_cli/models.py`
3. `hermes_cli/runtime_provider.py`
4. `hermes_cli/main.py`
5. `agent/auxiliary_client.py`
6. `agent/model_metadata.py`
7. 测试
8. `website/docs/` 下的用户文档

:::tip
`hermes_cli/setup.py` **无需**修改。setup 向导将 provider/model 选择委托给 `main.py` 中的 `select_provider_and_model()`——在那里添加的任何 provider 都会自动出现在 `hermes setup` 中。
:::

### 原生 / 非 OpenAI provider 额外需要

10. `agent/<provider>_adapter.py`
11. `run_agent.py`
12. 如果需要 provider SDK，则修改 `pyproject.toml`

## 快速路径：简单 API key provider

如果你的 provider 只是一个使用单个 API key 进行认证的 OpenAI 兼容端点，则无需修改 `auth.py`、`runtime_provider.py`、`main.py` 或下面完整清单中的任何其他文件。

你只需要：

1. 在 `plugins/model-providers/<your-provider>/` 下创建一个插件目录，包含：
   - `__init__.py`——在模块级别调用 `register_provider(profile)`
   - `plugin.yaml`——清单文件（name、kind: model-provider、version、description）
2. 就这些。Provider 插件在任何代码首次调用 `get_provider_profile()` 或 `list_providers()` 时自动加载——捆绑插件（本仓库）和位于 `$HERMES_HOME/plugins/model-providers/` 的用户插件都会被加载。

当你添加一个插件并调用 `register_provider()` 时，以下内容会自动接线：

1. `auth.py` 中的 `PROVIDER_REGISTRY` 条目（凭据解析、环境变量查找）
2. `api_mode` 设置为 `chat_completions`
3. `base_url` 从配置或声明的环境变量中获取
4. 按优先级顺序检查 `env_vars` 以获取 API key
5. 为该 provider 注册 `fallback_models` 列表
6. `--provider` CLI 标志接受该 provider id
7. `hermes model` 菜单包含该 provider
8. `hermes setup` 向导自动委托给 `main.py`
9. `provider:model` 别名语法正常工作
10. 运行时解析器返回正确的 `base_url` 和 `api_key`
11. `--provider <name>` CLI 标志接受该 provider id
12. 回退模型激活可以干净地切换到该 provider

位于 `$HERMES_HOME/plugins/model-providers/<name>/` 的用户插件会覆盖同名的捆绑插件（`register_provider()` 中后写者获胜）——因此第三方可以在不编辑本仓库的情况下对任何内置 profile 进行 monkey-patch 或替换。

参见 `plugins/model-providers/nvidia/` 或 `plugins/model-providers/gmi/` 作为模板，以及完整的 [Model Provider Plugin 指南](/developer-guide/model-provider-plugin)，了解字段参考、hook 用法和端到端示例。

## 完整路径：OAuth 和复杂 provider

当你的 provider 需要以下任何内容时，使用下面的完整清单：

- OAuth 或 token 刷新（Nous Portal、Codex、Google Gemini、Qwen Portal、Copilot）
- 需要新适配器的非 OpenAI API 格式（Anthropic Messages、Codex Responses）
- 自定义端点检测或多区域探测（z.ai、Kimi）
- 精选的静态模型目录或实时 `/models` 获取
- 带有特定认证流程的 provider 专属 `hermes model` 菜单条目

## 第 1 步：选择一个规范的 provider id

选择一个 provider id 并在所有地方使用它。

代码库中的示例：

- `openai-codex`
- `kimi-coding`
- `minimax-cn`

该 id 应出现在：

- `hermes_cli/auth.py` 中的 `PROVIDER_REGISTRY`
- `hermes_cli/models.py` 中的 `_PROVIDER_LABELS`
- `hermes_cli/auth.py` 和 `hermes_cli/models.py` 中的 `_PROVIDER_ALIASES`
- `hermes_cli/main.py` 中的 CLI `--provider` 选项
- setup / 模型选择分支
- 辅助模型默认值
- 测试

如果这些文件之间的 id 不一致，provider 会感觉只接了一半线：认证可能正常，而 `/model`、setup 或运行时解析会静默地遗漏它。

## 第 2 步：在 `hermes_cli/auth.py` 中添加认证元数据

对于 API key provider，在 `PROVIDER_REGISTRY` 中添加一个 `ProviderConfig` 条目，包含：

- `id`
- `name`
- `auth_type="api_key"`
- `inference_base_url`
- `api_key_env_vars`
- 可选的 `base_url_env_var`

同时在 `_PROVIDER_ALIASES` 中添加别名。

使用现有 provider 作为模板：

- 简单 API key 路径：Z.AI、MiniMax
- 带端点检测的 API key 路径：Kimi、Z.AI
- 原生 token 解析：Anthropic
- OAuth / auth-store 路径：Nous、OpenAI Codex

需要在此回答的问题：

- Hermes 应该检查哪些环境变量，按什么优先级顺序？
- provider 是否需要 base URL 覆盖？
- 是否需要端点探测或 token 刷新？
- 当凭据缺失时，认证错误应该显示什么？

如果 provider 需要的不仅仅是"查找 API key"，请添加专用的凭据解析器，而不是将逻辑塞进不相关的分支。

## 第 3 步：在 `hermes_cli/models.py` 中添加模型目录和别名

更新 provider 目录，使 provider 在菜单和 `provider:model` 语法中正常工作。

典型修改：

- `_PROVIDER_MODELS`
- `_PROVIDER_LABELS`
- `_PROVIDER_ALIASES`
- `list_available_providers()` 中的 provider 显示顺序
- 如果 provider 支持实时 `/models` 获取，则修改 `provider_model_ids()`

如果 provider 提供实时模型列表，优先使用它，并将 `_PROVIDER_MODELS` 保留为静态回退。

此文件也是使以下输入正常工作的关键：

```text
anthropic:claude-sonnet-4-6
kimi:model-name
```

如果此处缺少别名，provider 可能认证正常，但在 `/model` 解析中仍然失败。

## 第 4 步：在 `hermes_cli/runtime_provider.py` 中解析运行时数据

`resolve_runtime_provider()` 是 CLI、gateway（网关）、cron、ACP 和辅助客户端共用的路径。

添加一个分支，至少返回包含以下内容的字典：

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

如果 provider 与 OpenAI 兼容，`api_mode` 通常应保持为 `chat_completions`。

注意 API key 优先级。Hermes 已经包含避免将 OpenRouter key 泄露给无关端点的逻辑。新 provider 应同样明确地指定哪个 key 对应哪个 base URL。

## 第 5 步：在 `hermes_cli/main.py` 中接线 CLI

在交互式 `hermes model` 流程中出现之前，provider 是不可发现的。

在 `hermes_cli/main.py` 中更新以下内容：

- `provider_labels` 字典
- `select_provider_and_model()` 中的 `providers` 列表
- provider 分发（`if selected_provider == ...`）
- `--provider` 参数选项
- 如果 provider 支持登录/登出流程，则更新相应选项
- 一个 `_model_flow_<provider>()` 函数，或者如果适用则复用 `_model_flow_api_key_provider()`

:::tip
`hermes_cli/setup.py` 无需修改——它调用 `main.py` 中的 `select_provider_and_model()`，因此你的新 provider 会自动出现在 `hermes model` 和 `hermes setup` 中。
:::

## 第 6 步：保持辅助调用正常工作

这里有两个文件需要关注：

### `agent/auxiliary_client.py`

如果这是一个直接 API key provider，在 `_API_KEY_PROVIDER_AUX_MODELS` 中添加一个廉价/快速的默认辅助模型。

辅助任务包括：

- 视觉摘要
- 网页提取摘要
- 上下文压缩摘要
- 会话搜索摘要
- 记忆刷新

如果 provider 没有合理的辅助默认值，辅助任务可能会严重回退，或意外使用昂贵的主模型。

### `agent/model_metadata.py`

为 provider 的模型添加上下文长度，以保持 token 预算、压缩阈值和限制的合理性。

## 第 7 步：如果 provider 是原生的，添加适配器和 `run_agent.py` 支持

如果 provider 不是普通的 chat completions，将 provider 专属逻辑隔离在 `agent/<provider>_adapter.py` 中。

保持 `run_agent.py` 专注于编排。它应该调用适配器辅助函数，而不是在整个文件中内联构建 provider 请求载荷。

原生 provider 通常需要在以下地方进行工作：

### 新适配器文件

典型职责：

- 构建 SDK / HTTP 客户端
- 解析 token
- 将 OpenAI 风格的对话消息转换为 provider 的请求格式
- 如有需要，转换工具 schema
- 将 provider 响应规范化为 `run_agent.py` 期望的格式
- 提取用量和 finish-reason 数据

### `run_agent.py`

搜索 `api_mode` 并审计每个切换点。至少验证：

- `__init__` 选择了新的 `api_mode`
- 客户端构建对该 provider 有效
- `_build_api_kwargs()` 知道如何格式化请求
- `_interruptible_api_call()` 分发到正确的客户端调用
- 中断 / 客户端重建路径正常工作
- 响应验证接受该 provider 的格式
- finish-reason 提取正确
- token 用量提取正确
- 回退模型激活可以干净地切换到新 provider
- 摘要生成和记忆刷新路径仍然正常工作

同时在 `run_agent.py` 中搜索 `self.client.`。任何假设标准 OpenAI 客户端存在的代码路径，在原生 provider 使用不同客户端对象或 `self.client = None` 时都可能中断。

### Prompt 缓存和 provider 专属请求字段

Prompt（提示词）缓存和 provider 专属的调节项很容易出现回归。

代码库中已有的示例：

- Anthropic 有原生的 prompt 缓存路径
- OpenRouter 获得 provider 路由字段
- 并非每个 provider 都应该接收每个请求端选项

添加原生 provider 时，仔细检查 Hermes 只向该 provider 发送它实际理解的字段。

## 第 8 步：测试

至少修改保护 provider 接线的测试。

常见位置：

- `tests/test_runtime_provider_resolution.py`
- `tests/test_cli_provider_resolution.py`
- `tests/test_cli_model_command.py`
- `tests/test_setup_model_selection.py`
- `tests/test_provider_parity.py`
- `tests/test_run_agent.py`
- 原生 provider 的 `tests/test_<provider>_adapter.py`

对于仅文档示例，确切的文件集可能不同。重点是覆盖：

- 认证解析
- CLI 菜单 / provider 选择
- 运行时 provider 解析
- agent 执行路径
- `provider:model` 解析
- 任何适配器专属的消息转换

使用禁用 xdist 的方式运行测试：

```bash
source venv/bin/activate
python -m pytest tests/test_runtime_provider_resolution.py tests/test_cli_provider_resolution.py tests/test_cli_model_command.py tests/test_setup_model_selection.py -n0 -q
```

对于更深层的修改，在推送前运行完整测试套件：

```bash
source venv/bin/activate
python -m pytest tests/ -n0 -q
```

## 第 9 步：实时验证

测试通过后，运行真实的冒烟测试。

```bash
source venv/bin/activate
python -m hermes_cli.main chat -q "Say hello" --provider your-provider --model your-model
```

如果你修改了菜单，也测试交互式流程：

```bash
source venv/bin/activate
python -m hermes_cli.main model
python -m hermes_cli.main setup
```

对于原生 provider，至少也验证一次工具调用，而不仅仅是纯文本响应。

## 第 10 步：更新用户文档

如果该 provider 打算作为一流选项发布，也更新用户文档：

- `website/docs/getting-started/quickstart.md`
- `website/docs/user-guide/configuration.md`
- `website/docs/reference/environment-variables.md`

开发者可以完美地接线 provider，但仍然让用户无法发现所需的环境变量或 setup 流程。

## OpenAI 兼容 provider 清单

如果 provider 是标准 chat completions，使用此清单。

- [ ] 在 `hermes_cli/auth.py` 中添加 `ProviderConfig`
- [ ] 在 `hermes_cli/auth.py` 和 `hermes_cli/models.py` 中添加别名
- [ ] 在 `hermes_cli/models.py` 中添加模型目录
- [ ] 在 `hermes_cli/runtime_provider.py` 中添加运行时分支
- [ ] 在 `hermes_cli/main.py` 中添加 CLI 接线（setup.py 自动继承）
- [ ] 在 `agent/auxiliary_client.py` 中添加辅助模型
- [ ] 在 `agent/model_metadata.py` 中添加上下文长度
- [ ] 更新运行时 / CLI 测试
- [ ] 更新用户文档

## 原生 provider 清单

当 provider 需要新的协议路径时使用此清单。

- [ ] OpenAI 兼容清单中的所有内容
- [ ] 在 `agent/<provider>_adapter.py` 中添加适配器
- [ ] 在 `run_agent.py` 中支持新的 `api_mode`
- [ ] 中断 / 重建路径正常工作
- [ ] 用量和 finish-reason 提取正常工作
- [ ] 回退路径正常工作
- [ ] 添加适配器测试
- [ ] 实时冒烟测试通过

## 常见陷阱

### 1. 将 provider 添加到 auth 但未添加到模型解析

这会导致凭据解析正确，而 `/model` 和 `provider:model` 输入失败。

### 2. 忘记 `config["model"]` 可以是字符串或字典

大量 provider 选择代码必须对两种形式进行规范化。

### 3. 假设必须使用内置 provider

如果该服务只是 OpenAI 兼容的，自定义 provider 可能已经以更少的维护成本解决了用户问题。

### 4. 忘记辅助路径

主聊天路径可能正常工作，而摘要、记忆刷新或视觉辅助失败，因为辅助路由从未更新。

### 5. 原生 provider 分支隐藏在 `run_agent.py` 中

搜索 `api_mode` 和 `self.client.`。不要假设显而易见的请求路径是唯一的。

### 6. 将 OpenRouter 专属字段发送给其他 provider

provider 路由等字段只属于支持它们的 provider。

### 7. 更新了 `hermes model` 但未更新 `hermes setup`

两个流程都需要了解该 provider。

## 实现时的好搜索目标

如果你在寻找 provider 涉及的所有位置，搜索以下符号：

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
- [贡献指南](./contributing.md)