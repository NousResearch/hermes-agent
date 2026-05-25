---
sidebar_position: 10
title: "模型提供商插件"
description: "如何为 Hermes Agent 构建模型提供商（推理后端）插件"
---

# 构建模型提供商插件

模型提供商插件声明一个推理后端——兼容 OpenAI 的端点、Anthropic Messages 服务器、Codex 风格的 Responses API，或 Bedrock 原生接口——Hermes 可通过这些后端路由 `AIAgent` 调用。每个内置提供商（OpenRouter、Anthropic、GMI、DeepSeek、Nvidia……）都以此类插件形式提供。第三方可通过在 `$HERMES_HOME/plugins/model-providers/` 下放置一个目录来添加自己的提供商，无需对仓库做任何修改。

:::tip
模型提供商插件是**提供商插件**的第三种类型。其他两种分别是 [Memory Provider 插件](/developer-guide/memory-provider-plugin)（跨会话知识）和 [Context Engine 插件](/developer-guide/context-engine-plugin)（上下文压缩策略）。三者均遵循相同的"放入目录、声明 profile、无需编辑仓库"模式。
:::

## 发现机制

`providers/__init__.py._discover_providers()` 在任何代码首次调用 `get_provider_profile()` 或 `list_providers()` 时懒加载执行。发现顺序：

1. **内置插件** — `<repo>/plugins/model-providers/<name>/` — 随 Hermes 一同发布
2. **用户插件** — `$HERMES_HOME/plugins/model-providers/<name>/` — 放入任意目录；后续会话无需重启即可生效
3. **旧版单文件** — `<repo>/providers/<name>.py` — 为树外可编辑安装提供向后兼容

**同名用户插件会覆盖内置插件**，因为 `register_provider()` 采用后写者优先策略。放入 `$HERMES_HOME/plugins/model-providers/gmi/` 目录即可替换内置 GMI profile，无需修改仓库。

## 目录结构

```
plugins/model-providers/my-provider/
├── __init__.py       # 在模块级别调用 register_provider(profile)
├── plugin.yaml       # kind: model-provider + 元数据（可选但推荐）
└── README.md         # 安装说明（可选）
```

唯一必需的文件是 `__init__.py`。`plugin.yaml` 供 `hermes plugins` 用于自省，以及供通用 PluginManager 将插件路由到正确的加载器；若缺少该文件，通用加载器会回退到源码文本启发式检测。

## 最简示例——一个简单的 API key 提供商

```python
# plugins/model-providers/acme-inference/__init__.py
from providers import register_provider
from providers.base import ProviderProfile

acme = ProviderProfile(
    name="acme-inference",
    aliases=("acme",),
    display_name="Acme Inference",
    description="Acme — OpenAI-compatible direct API",
    signup_url="https://acme.example.com/keys",
    env_vars=("ACME_API_KEY", "ACME_BASE_URL"),
    base_url="https://api.acme.example.com/v1",
    auth_type="api_key",
    default_aux_model="acme-small-fast",
    fallback_models=(
        "acme-large-v3",
        "acme-medium-v3",
        "acme-small-fast",
    ),
)

register_provider(acme)
```

```yaml
# plugins/model-providers/acme-inference/plugin.yaml
name: acme-inference
kind: model-provider
version: 1.0.0
description: Acme Inference — OpenAI-compatible direct API
author: Your Name
```

就这些。放入这两个文件后，以下集成**自动生效**，无需其他任何修改：

| 集成点 | 位置 | 获得的能力 |
|---|---|---|
| 凭据解析 | `hermes_cli/auth.py` | `PROVIDER_REGISTRY["acme-inference"]` 从 profile 填充 |
| `--provider` CLI 标志 | `hermes_cli/main.py` | 接受 `acme-inference` |
| `hermes model` 选择器 | `hermes_cli/models.py` | 出现在 `CANONICAL_PROVIDERS` 中，从 `{base_url}/models` 获取模型列表 |
| `hermes doctor` | `hermes_cli/doctor.py` | 对 `ACME_API_KEY` 及 `{base_url}/models` 进行健康检查 |
| `hermes setup` | `hermes_cli/config.py` | `ACME_API_KEY` 出现在 `OPTIONAL_ENV_VARS` 和设置向导中 |
| URL 反向映射 | `agent/model_metadata.py` | 主机名 → 提供商名称，用于自动检测 |
| 辅助模型 | `agent/auxiliary_client.py` | 使用 `default_aux_model` 进行压缩/摘要 |
| 运行时解析 | `hermes_cli/runtime_provider.py` | 返回正确的 `base_url`、`api_key`、`api_mode` |
| 传输层 | `agent/transports/chat_completions.py` | Profile 路径通过 `prepare_messages` / `build_extra_body` / `build_api_kwargs_extras` 生成 kwargs |

## ProviderProfile 字段

完整定义见 `providers/base.py`。最常用的字段：

| 字段 | 类型 | 用途 |
|---|---|---|
| `name` | str | 规范 ID——与 `config.yaml` 中的 `model.provider` 及 `--provider` 标志匹配 |
| `aliases` | `tuple[str, ...]` | 由 `get_provider_profile()` 解析的别名（如 `grok` → `xai`） |
| `api_mode` | str | `chat_completions` \| `codex_responses` \| `anthropic_messages` \| `bedrock_converse` |
| `display_name` | str | 在 `hermes model` 选择器中显示的人类可读标签 |
| `description` | str | 选择器副标题 |
| `signup_url` | str | 首次运行设置时显示（"在此获取 API key"） |
| `env_vars` | `tuple[str, ...]` | 按优先级排列的 API key 环境变量；最后一个 `*_BASE_URL` 条目用作用户 base URL 覆盖 |
| `base_url` | str | 默认推理端点 |
| `models_url` | str | 显式目录 URL（回退到 `{base_url}/models`） |
| `auth_type` | str | `api_key` \| `oauth_device_code` \| `oauth_external` \| `copilot` \| `aws_sdk` \| `external_process` |
| `fallback_models` | `tuple[str, ...]` | 实时目录获取失败时显示的精选列表 |
| `default_headers` | `dict[str, str]` | 随每个请求发送（如 Copilot 的 `Editor-Version`） |
| `fixed_temperature` | Any | `None` = 使用调用方的值；`OMIT_TEMPERATURE` 哨兵值 = 完全不发送 temperature（Kimi） |
| `default_max_tokens` | `int \| None` | 提供商级别的 max_tokens 上限（Nvidia：16384） |
| `default_aux_model` | str | 用于辅助任务（压缩、视觉、摘要）的廉价模型 |

## 可覆盖的 hook

对于非常规的特殊需求，可子类化 `ProviderProfile`：

```python
from typing import Any
from providers.base import ProviderProfile

class AcmeProfile(ProviderProfile):
    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """提供商特定的消息预处理。在 codex 清理之后、developer-role 替换之前运行。
        默认：直接透传。"""
        # 示例：Qwen 将纯文本内容规范化为 list-of-parts 数组并注入 cache_control；
        # Kimi 重写 tool-call JSON
        return messages

    def build_extra_body(self, *, session_id=None, **context) -> dict:
        """提供商特定的 extra_body 字段，合并到 API 调用中。
        context 包含：session_id、provider_preferences、model、base_url、
        reasoning_config。默认：空 dict。"""
        # 示例：OpenRouter 的 provider-preferences 块，
        # Gemini 的 thinking_config 转换。
        return {}

    def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
        """返回 (extra_body_additions, top_level_kwargs)。当某些字段需要放在顶层
        （Kimi 的 reasoning_effort）而另一些放在 extra_body（OpenRouter 的 reasoning dict）
        时需要此方法。默认：({}, {})。"""
        return {}, {}

    def fetch_models(self, *, api_key=None, timeout=8.0) -> list[str] | None:
        """实时目录获取。默认使用 Bearer 认证访问 {models_url or base_url}/models。
        以下情况需覆盖：自定义认证（Anthropic）、无 REST 端点（Bedrock → None），
        或公开/无认证目录（OpenRouter）。"""
        return super().fetch_models(api_key=api_key, timeout=timeout)
```

## Hook 参考示例

参考以下内置插件了解常用写法：

| 插件 | 参考原因 |
|---|---|
| `plugins/model-providers/openrouter/` | 带 provider preferences 的聚合器，公开模型目录 |
| `plugins/model-providers/gemini/` | `thinking_config` 转换（原生 + OpenAI 兼容嵌套形式） |
| `plugins/model-providers/kimi-coding/` | `OMIT_TEMPERATURE`、`extra_body.thinking`、顶层 `reasoning_effort` |
| `plugins/model-providers/qwen-oauth/` | 消息规范化、`cache_control` 注入、VL 高分辨率 |
| `plugins/model-providers/nous/` | 归因标签、"禁用时省略 reasoning" |
| `plugins/model-providers/custom/` | Ollama 的 `num_ctx` + `think: false` 特殊处理 |
| `plugins/model-providers/bedrock/` | `api_mode="bedrock_converse"`，`fetch_models` 返回 None（无 REST 端点） |

## 用户覆盖——不修改仓库替换内置提供商

假设你想将 `gmi` 指向私有测试端点进行测试。创建 `~/.hermes/plugins/model-providers/gmi/__init__.py`：

```python
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="gmi",
    aliases=("gmi-cloud", "gmicloud"),
    env_vars=("GMI_API_KEY",),
    base_url="https://gmi-staging.internal.example.com/v1",
    auth_type="api_key",
    default_aux_model="google/gemini-3.1-flash-lite-preview",
))
```

下次会话时，`get_provider_profile("gmi").base_url` 将返回测试 URL。无需打补丁，无需重新构建。由于用户插件在内置插件之后被发现，用户的 `register_provider()` 调用会胜出。

## api_mode 选择

系统识别四个值。Hermes 的选择依据：

1. 用户显式覆盖（`config.yaml` 中设置了 `model.api_mode`）
2. OpenCode 的按模型分发（Zen 和 Go 的 `opencode_model_api_mode`）
3. URL 自动检测——`/anthropic` 后缀 → `anthropic_messages`，`api.openai.com` → `codex_responses`，`api.x.ai` → `codex_responses`，Kimi 域名上的 `/coding` → `chat_completions`
4. **Profile 的 `api_mode`** 作为 URL 检测无结果时的回退
5. 默认 `chat_completions`

将 `profile.api_mode` 设置为你的提供商默认使用的值——它作为提示使用。用户 URL 覆盖仍然优先。

## 认证类型

| `auth_type` | 含义 | 使用者 |
|---|---|---|
| `api_key` | 单个环境变量携带静态 API key | 大多数提供商 |
| `oauth_device_code` | 设备码 OAuth 流程 | — |
| `oauth_external` | 用户在其他地方登录，token 存入 `auth.json` | Anthropic OAuth、MiniMax OAuth、Gemini Cloud Code、Qwen Portal、Nous Portal |
| `copilot` | GitHub Copilot token 刷新周期 | 仅 `copilot` 插件 |
| `aws_sdk` | AWS SDK 凭据链（IAM role、profile、env） | 仅 `bedrock` 插件 |
| `external_process` | 认证由 agent 启动的子进程处理 | 仅 `copilot-acp` 插件 |

`auth_type` 控制哪些代码路径将你的提供商视为"简单 api-key 提供商"——若不是 `api_key`，PluginManager 仍会记录 manifest，但 Hermes CLI 层面的自动化（doctor 检查、`--provider` 标志、设置向导委托）可能会跳过它。

## 发现时机

提供商发现是**懒加载**的——由进程中首次调用 `get_provider_profile()` 或 `list_providers()` 触发。实际上这在启动早期就会发生（`auth.py` 模块加载时会主动扩展 `PROVIDER_REGISTRY`）。若需验证插件是否已加载，运行：

```bash
hermes doctor
```

——成功的 `auth_type="api_key"` profile 会出现在 Provider Connectivity 部分，并附带 `/models` 探测结果。

编程方式检查：

```python
from providers import list_providers
for p in list_providers():
    print(p.name, p.base_url, p.api_mode)
```

## 测试你的插件

将 `HERMES_HOME` 指向临时目录，避免污染真实配置：

```bash
export HERMES_HOME=/tmp/hermes-plugin-test
mkdir -p $HERMES_HOME/plugins/model-providers/my-provider
cat > $HERMES_HOME/plugins/model-providers/my-provider/__init__.py <<'EOF'
from providers import register_provider
from providers.base import ProviderProfile
register_provider(ProviderProfile(
    name="my-provider",
    env_vars=("MY_API_KEY",),
    base_url="https://api.my-provider.example.com/v1",
    auth_type="api_key",
))
EOF

export MY_API_KEY=your-test-key
hermes -z "hello" --provider my-provider -m some-model
```

## 通用 PluginManager 集成

通用 `PluginManager`（即 `hermes plugins` 操作的对象）**能看到**模型提供商插件，但不会导入它们——`providers/__init__.py` 负责管理其生命周期。Manager 记录 manifest 用于自省，并按 `kind: model-provider` 分类。当你将一个未标记的用户插件放入 `$HERMES_HOME/plugins/`，而该插件恰好调用了带 `ProviderProfile` 的 `register_provider`，Manager 会通过源码文本启发式检测自动将其归类为 `kind: model-provider`——因此即使没有 `plugin.yaml`，插件仍能正确路由。

## 通过 pip 分发

与所有 Hermes 插件一样，模型提供商可以作为 pip 包发布。在你的 `pyproject.toml` 中添加入口点：

```toml
[project.entry-points."hermes.plugins"]
acme-inference = "acme_hermes_plugin:register"
```

……其中 `acme_hermes_plugin:register` 是一个调用 `register_provider(profile)` 的函数。通用 PluginManager 在 `discover_and_load()` 期间会拾取入口点插件。对于 `kind: model-provider` 的 pip 插件，你仍需在 manifest 中声明 kind（或依赖源码文本启发式检测）。

完整的入口点设置请参阅 [构建 Hermes 插件](/guides/build-a-hermes-plugin#distribute-via-pip)。

## 相关页面

- [Provider Runtime](/developer-guide/provider-runtime) — 解析优先级及各层读取 profile 的位置
- [添加提供商](/developer-guide/adding-providers) — 新推理后端的端到端检查清单（涵盖快速插件路径和完整 CLI/auth 集成）
- [Memory Provider 插件](/developer-guide/memory-provider-plugin)
- [Context Engine 插件](/developer-guide/context-engine-plugin)
- [构建 Hermes 插件](/guides/build-a-hermes-plugin) — 通用插件编写指南