---
sidebar_position: 10
title: "模型 Provider 插件"
description: "如何为 Hermes Agent 构建模型 provider（推理后端）插件"
---

# 构建模型 Provider 插件

模型 provider 插件声明推理后端 — OpenAI 兼容端点、Anthropic Messages 服务器、Codex 风格 Responses API 或 Bedrock 原生表面 — Hermes 可以路由 `AIAgent` 调用通过它。每个内置 provider（OpenRouter、Anthropic、GMI、DeepSeek、Nvidia、…）都作为这些插件之一发货。第三方可以通过将目录放入 `$HERMES_HOME/plugins/model-providers/` 并零更改仓库来添加自己的。

:::tip
模型 provider 插件是**provider 插件**的第三种。其他是[内存 Provider 插件](/docs/developer-guide/memory-provider-plugin)（跨会话知识）和[上下文引擎插件](/docs/developer-guide/context-engine-plugin)（上下文压缩策略）。所有三种都遵循相同的"放入目录、声明配置文件、无需仓库编辑"模式。
:::

## 发现如何工作

`providers/__init__.py._discover_providers()` 在任何代码第一次调用 `get_provider_profile()` 或 `list_providers()` 时延迟运行。发现顺序：

1. **捆绑插件** — `<repo>/plugins/model-providers/<name>/` — 随 Hermes 发货
2. **用户插件** — `$HERMES_HOME/plugins/model-providers/<name>/` — 放入任何目录；后续会话无需重启
3. **旧版单文件** — `<repo>/providers/<name>.py` — 树外可编辑安装的向后兼容

**用户插件覆盖同名捆绑插件**，因为 `register_provider()` 后写入者获胜。将 `$HERMES_HOME/plugins/model-providers/gmi/` 目录放下以替换内置 GMI 配置而不触及仓库。

## 目录结构

```
plugins/model-providers/my-provider/
├── __init__.py       # Calls register_provider(profile) at module-level
├── plugin.yaml       # kind: model-provider + metadata (optional but recommended)
└── README.md         # Setup instructions (optional)
```

唯一必需的文件是 `__init__.py`。`plugin.yaml` 被 `hermes plugins` 用于内省，也被通用 PluginManager 用于将插件路由到正确的加载器；没有它，通用加载器回退到源文本启发式。

## 最小示例 — 简单的 API 密钥 provider

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

就是这样。放下这两个文件后，以下**自动接线**无需其他编辑：

| 集成 | 哪里 | 获得什么 |
|---|---|---|
| 凭据解析 | `hermes_cli/auth.py` | 从配置填充的 `PROVIDER_REGISTRY["acme-inference"]` |
| `--provider` CLI 标志 | `hermes_cli/main.py` | 接受 `acme-inference` |
| `hermes model` 选择器 | `hermes_cli/models.py` | 出现在 `CANONICAL_PROVIDERS` 中，模型列表从 `{base_url}/models` 获取 |
| `hermes doctor` | `hermes_cli/doctor.py` | `ACME_API_KEY` + `{base_url}/models` 探测的健康检查 |
| `hermes setup` | `hermes_cli/config.py` | `ACME_API_KEY` 出现在 `OPTIONAL_ENV_VARS` 和设置向导中 |
| URL 反向映射 | `agent/model_metadata.py` | 主机名 → provider 名称用于自动检测 |
| 辅助模型 | `agent/auxiliary_client.py` | 使用 `default_aux_model` 进行压缩 / 摘要 |
| 运行时解析 | `hermes_cli/runtime_provider.py` | 返回正确的 `base_url`、`api_key`、`api_mode` |
| 传输 | `agent/transports/chat_completions.py` | 配置文件路径通过 `prepare_messages` / `build_extra_body` / `build_api_kwargs_extras` 生成 kwargs |

## ProviderProfile 字段

`providers/base.py` 中的完整定义。最有用的字段：

| 字段 | 类型 | 目的 |
|---|---|---|
| `name` | str | 规范 id — 匹配 `--provider` 选项和 `HERMES_INFERENCE_PROVIDER` |
| `aliases` | `tuple[str, ...]` | `get_provider_profile()` 解析的替代名称（例如 `grok` → `xai`） |
| `api_mode` | str | `chat_completions` \| `codex_responses` \| `anthropic_messages` \| `bedrock_converse` |
| `display_name` | str | `hermes model` 选择器中显示的人类标签 |
| `description` | str | 选择器副标题 |
| `signup_url` | str | 首次运行设置期间显示（"在这里获取 API 密钥"） |
| `env_vars` | `tuple[str, ...]` | 优先级顺序的 API 密钥 env 变量；最终的 `*_BASE_URL` 条目用作用户基础 URL 覆盖 |
| `base_url` | str | 默认推理端点 |
| `models_url` | str | 显式目录 URL（回退到 `{base_url}/models`） |
| `auth_type` | str | `api_key` \| `oauth_device_code` \| `oauth_external` \| `copilot` \| `aws_sdk` \| `external_process` |
| `fallback_models` | `tuple[str, ...]` | 实时目录获取失败时显示的精选列表 |
| `default_headers` | `dict[str, str]` | 每个请求发送（例如 Copilot 的 `Editor-Version`） |
| `fixed_temperature` | Any | `None` = 使用调用者的值；`OMIT_TEMPERATURE` 哨兵 = 完全不发送温度（Kimi） |
| `default_max_tokens` | `int \| None` | Provider 级别的 max_tokens 上限（Nvidia：16384） |
| `default_aux_model` | str | 用于辅助任务（压缩、视觉、摘要）的廉价模型 |

## 可覆盖钩子

子类化 `ProviderProfile` 处理非平凡怪癖：

```python
from typing import Any
from providers.base import ProviderProfile

class AcmeProfile(ProviderProfile):
    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing. Runs after codex
        sanitization, before developer-role swap. Default: pass-through."""
        # Example: Qwen normalizes plain-text content to a list-of-parts
        # array and injects cache_control; Kimi rewrites tool-call JSON
        return messages

    def build_extra_body(self, *, session_id=None, **context) -> dict:
        """Provider-specific extra_body fields merged into the API call.
        Context includes: session_id, provider_preferences, model, base_url,
        reasoning_config. Default: empty dict."""
        # Example: OpenRouter's provider-preferences block,
        # Gemini's thinking_config translation.
        return {}

    def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
        """Returns (extra_body_additions, top_level_kwargs). Needed when some
        fields go top-level (Kimi's reasoning_effort) and some go in extra_body
        (OpenRouter's reasoning dict). Default: ({}, {})."""
        return {}, {}

    def fetch_models(self, *, api_key=None, timeout=8.0) -> list[str] | None:
        """Live catalog fetch. Default hits {models_url or base_url}/models with
        Bearer auth. Override for: custom auth (Anthropic), no REST endpoint
        (Bedrock → None), or public/unauthenticated catalogs (OpenRouter)."""
        return super().fetch_models(api_key=api_key, timeout=timeout)
```

## 钩子参考示例

查看这些捆绑插件获取习惯用法：

| 插件 | 为什么查看 |
|---|---|
| `plugins/model-providers/openrouter/` | 带 provider 偏好的聚合器，公共模型目录 |
| `plugins/model-providers/gemini/` | `thinking_config` 翻译（原生 + OpenAI 兼容嵌套形式） |
| `plugins/model-providers/kimi-coding/` | `OMIT_TEMPERATURE`、`extra_body.thinking`、顶级 `reasoning_effort` |
| `plugins/model-providers/qwen-oauth/` | 消息规范化、`cache_control` 注入、VL 高分辨率 |
| `plugins/model-providers/nous/` | 属性标签、"禁用时省略推理" |
| `plugins/model-providers/custom/` | Ollama `num_ctx` + `think: false` 怪癖 |
| `plugins/model-providers/bedrock/` | `api_mode="bedrock_converse"`、`fetch_models` 返回 None（无 REST 端点） |

## 用户覆盖 — 无需编辑仓库替换内置

假设您想将 `gmi` 指向您的私有暂存端点进行测试。创建 `~/.hermes/plugins/model-providers/gmi/__init__.py`：

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

下一会话，`get_provider_profile("gmi").base_url` 返回暂存 URL。无需仓库补丁，无需重建。因为用户插件在捆绑插件之后发现，用户 `register_provider()` 调用获胜。

## api_mode 选择

识别四个值。Hermes 根据以下内容选择一个：

1. 用户显式覆盖（设置时 `config.yaml` `model.api_mode`）
2. OpenCode 的每模型调度（`opencode_model_api_mode` 用于 Zen 和 Go）
3. URL 自动检测 — `/anthropic` 后缀 → `anthropic_messages`、`api.openai.com` → `codex_responses`、`api.x.ai` → `codex_responses`、Kimi 域名上的 `/coding` → `chat_completions`
4. **Profile `api_mode`** 作为 URL 检测找不到东西时的后备
5. 默认 `chat_completions`

将 `profile.api_mode` 设置为匹配 provider 默认发货的内容 — 它充当提示。用户 URL 覆盖仍然获胜。

## Auth 类型

| `auth_type` | 含义 | 谁使用 |
|---|---|---|
| `api_key` | 单个 env 变量携带静态 API 密钥 | 大多数 provider |
| `oauth_device_code` | 设备代码 OAuth 流程 | — |
| `oauth_external` | 用户在其他地方登录，token 落在 `auth.json` | Anthropic OAuth、MiniMax OAuth、Gemini Cloud Code、Qwen Portal、Nous Portal |
| `copilot` | GitHub Copilot token 刷新周期 | 仅 `copilot` 插件 |
| `aws_sdk` | AWS SDK 凭据链（IAM 角色、profile、env） | 仅 `bedrock` 插件 |
| `external_process` | Auth 由代理生成的子进程处理 | 仅 `copilot-acp` 插件 |

`auth_type` 门控哪些代码路径将您的 provider 视为"简单 api-key provider" — 如果不是 `api_key`，PluginManager 仍记录清单，但 Hermes 的 CLI 级自动化（doctor 检查、`--provider` 标志、设置向导委托）可能跳过它。

## 发现时机

Provider 发现是**延迟的** — 由代码中首次 `get_provider_profile()` 或 `list_providers()` 调用触发。实际上这在启动时早发生（`auth.py` 模块加载急切地扩展 `PROVIDER_REGISTRY`）。如果您需要验证插件已加载，运行：

```bash
hermes doctor
```

— 成功的 `auth_type="api_key"` 配置出现在 Provider Connectivity 部分，带 `/models` 探测。

对于编程检查：

```python
from providers import list_providers
for p in list_providers():
    print(p.name, p.base_url, p.api_mode)
```

## 测试您的插件

将 `HERMES_HOME` 指向临时目录，这样不会污染您的真实配置：

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

通用 `PluginManager`（`hermes plugins` 操作的对象）**看到**模型 provider 插件但**不导入**它们 — `providers/__init__.py` 拥有它们的生命周期。管理器记录清单用于内省并按 `kind: model-provider` 分类。当您将未标记的用户插件放入 `$HERMES_HOME/plugins/`，恰好调用 `register_provider` 与 `ProviderProfile`，管理器通过源文本启发式自动将其强制转换为 `kind: model-provider` — 所以插件仍然正确路由，即使没有 `plugin.yaml`。

## 通过 pip 分发

像任何 Hermes 插件一样，模型 provider 可以作为 pip 包发货。在 `pyproject.toml` 中添加入口点：

```toml
[project.entry-points."hermes.plugins"]
acme-inference = "acme_hermes_plugin:register"
```

…其中 `acme_hermes_plugin:register` 是调用 `register_provider(profile)` 的函数。通用 PluginManager 在 `discover_and_load()` 期间拾取入口点插件。对于 `kind: model-provider` pip 插件，您仍需要在清单中声明 kind（或依赖源文本启发式）。

参见[构建 Hermes 插件](/docs/guides/build-a-hermes-plugin#distribute-via-pip)获取完整的入口点设置。

## 相关页面

- [Provider 运行时](/docs/developer-guide/provider-runtime) — 解析优先级 + 每个层读取配置的位置
- [添加 Provider](/docs/developer-guide/adding-providers) — 新推理后端的端到端清单（涵盖快速插件路径和完整 CLI/auth 集成）
- [内存 Provider 插件](/docs/developer-guide/memory-provider-plugin)
- [上下文引擎插件](/docs/developer-guide/context-engine-plugin)
- [构建 Hermes 插件](/docs/guides/build-a-hermes-plugin) — 通用插件编写
