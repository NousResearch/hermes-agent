---
sidebar_position: 11
title: "图像生成 Provider 插件"
description: "如何为 Hermes Agent 构建图像生成后端插件"
---

# 构建图像生成 Provider 插件

图像生成 provider 插件注册服务每个 `image_generate` 工具调用的后端 — DALL·E、gpt-image、Grok、Flux、Imagen、Stable Diffusion、fal、Replicate、本地 ComfyUI 设备，任何东西。内置 provider（OpenAI、OpenAI-Codex、xAI）都作为插件发货。您可以通过将目录放入 `plugins/image_gen/<name>/` 来添加新的，或覆盖捆绑的。

:::tip
图像生成是 Hermes 支持的几种**后端插件**之一。其他的（具有更专业 ABC 的）是[内存 Provider 插件](/docs/developer-guide/memory-provider-plugin)、[上下文引擎插件](/docs/developer-guide/context-engine-plugin)和[模型 Provider 插件](/docs/developer-guide/model-provider-plugin)。通用工具/钩子/CLI 插件位于[构建 Hermes 插件](/docs/guides/build-a-hermes-plugin)。
:::

## 发现如何工作

Hermes 从三个地方扫描图像生成后端：

1. **捆绑** — `<repo>/plugins/image_gen/<name>/`（使用 `kind: backend` 自动加载，始终可用）
2. **用户** — `~/.hermes/plugins/image_gen/<name>/`（通过 `plugins.enabled` 选择加入）
3. **Pip** — 声明 `hermes_agent.plugins` 入口点的包

每个插件的 `register(ctx)` 函数调用 `ctx.register_image_gen_provider(...)` — 这将其放入 `agent/image_gen_registry.py` 中的注册表。活动 provider 由 `config.yaml` 中的 `image_gen.provider` 选择；`hermes tools` 引导用户进行选择。

`image_generate` 工具包装器向注册表请求活动 provider 并调度到那里。如果没有 provider 注册，工具浮现一个有用的错误，指向 `hermes tools`。

## 目录结构

```
plugins/image_gen/my-backend/
├── __init__.py      # ImageGenProvider subclass + register()
└── plugin.yaml      # Manifest with kind: backend
```

捆绑插件此时已完成。用户插件在 `~/.hermes/plugins/image_gen/<name>/` 需要添加到 `config.yaml` 中的 `plugins.enabled`（或运行 `hermes plugins enable <name>`）。

## ImageGenProvider ABC

子类化 `agent.image_gen_provider.ImageGenProvider`。唯一必需的成员是 `name` 属性和 `generate()` 方法 — 其他都有合理的默认值：

```python
# plugins/image_gen/my-backend/__init__.py
from typing import Any, Dict, List, Optional
import os

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)


class MyBackendImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        # Stable id used in image_gen.provider config. Lowercase, no spaces.
        return "my-backend"

    @property
    def display_name(self) -> str:
        # Human label shown in `hermes tools`. Defaults to name.title() if omitted.
        return "My Backend"

    def is_available(self) -> bool:
        # Return False if credentials or deps are missing.
        # The tool's availability gate calls this before dispatch.
        if not os.environ.get("MY_BACKEND_API_KEY"):
            return False
        try:
            import my_backend_sdk  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        # Catalog shown in `hermes tools` model picker.
        return [
            {
                "id": "my-model-fast",
                "display": "My Model (Fast)",
                "speed": "~5s",
                "strengths": "Quick iteration",
                "price": "$0.01/image",
            },
            {
                "id": "my-model-hq",
                "display": "My Model (HQ)",
                "speed": "~30s",
                "strengths": "Highest fidelity",
                "price": "$0.04/image",
            },
        ]

    def default_model(self) -> Optional[str]:
        return "my-model-fast"

    def get_setup_schema(self) -> Dict[str, Any]:
        # Metadata for the `hermes tools` picker — keys to prompt for at setup.
        return {
            "name": "My Backend",
            "badge": "paid",        # optional; shown as a short tag in the picker
            "tag": "One-line description shown under the name",
            "env_vars": [
                {
                    "key": "MY_BACKEND_API_KEY",
                    "prompt": "My Backend API key",
                    "url": "https://my-backend.example.com/api-keys",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect_ratio = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required",
                error_type="invalid_input",
                provider=self.name,
                prompt="",
                aspect_ratio=aspect_ratio,
            )

        # Model selection precedence: env var → config → default. The helper
        # _resolve_model() in the built-in openai plugin is a good reference.
        model_id = kwargs.get("model") or self.default_model() or "my-model-fast"

        try:
            import my_backend_sdk
            client = my_backend_sdk.Client(api_key=os.environ["MY_BACKEND_API_KEY"])
            result = client.generate(
                prompt=prompt,
                model=model_id,
                aspect_ratio=aspect_ratio,
            )

            # Two shapes supported:
            #   - URL string: return it as `image`
            #   - base64 data: save under $HERMES_HOME/cache/images/ via save_b64_image()
            if result.get("image_b64"):
                path = save_b64_image(
                    result["image_b64"],
                    prefix=self.name,
                    extension="png",
                )
                image = str(path)
            else:
                image = result["image_url"]

            return success_response(
                image=image,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                provider=self.name,
            )
        except Exception as exc:
            return error_response(
                error=str(exc),
                error_type=type(exc).__name__,
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )


def register(ctx) -> None:
    """Plugin entry point — called once at load time."""
    ctx.register_image_gen_provider(MyBackendImageGenProvider())
```

## plugin.yaml

```yaml
name: my-backend
version: 1.0.0
description: My image backend — text-to-image via My Backend SDK
author: Your Name
kind: backend
requires_env:
  - MY_BACKEND_API_KEY
```

`kind: backend` 是将插件路由到图像生成注册路径的内容。`requires_env` 在 `hermes plugins install` 期间提示。

## ABC 参考

`agent/image_gen_provider.py` 中的完整契约。您通常覆盖的方法：

| 成员 | 必需 | 默认 | 目的 |
|---|---|---|---|
| `name` | ✅ | — | `image_gen.provider` 配置中使用的稳定 id |
| `display_name` | — | `name.title()` | `hermes tools` 中显示的标签 |
| `is_available()` | — | `True` | 缺失凭据/依赖的门控 |
| `list_models()` | — | `[]` | `hermes tools` 模型选择器的目录 |
| `default_model()` | — | first from `list_models()` | 未配置模型时的后备 |
| `get_setup_schema()` | — | minimal | 选择器元数据 + env 变量提示 |
| `generate(prompt, aspect_ratio, **kwargs)` | ✅ | — | 调用 |

## 响应格式

`generate()` 必须返回通过 `success_response()` 或 `error_response()` 构建的字典。两者都在 `agent/image_gen_provider.py` 中。

**成功：**
```python
success_response(
    image=<url-or-absolute-path>,
    model=<model-id>,
    prompt=<echoed-prompt>,
    aspect_ratio="landscape" | "square" | "portrait",
    provider=<your-provider-name>,
    extra={...},  # optional backend-specific fields
)
```

**错误：**
```python
error_response(
    error="human-readable message",
    error_type="provider_error" | "invalid_input" | "<exception class name>",
    provider=<your-provider-name>,
    model=<model-id>,
    prompt=<prompt>,
    aspect_ratio=<resolved aspect>,
)
```

工具包装器将字典 JSON 序列化并交给 LLM。错误作为工具结果浮现；LLM 决定如何向用户解释。

## 处理 base64 vs URL 输出

某些后端返回图像 URL（fal、Replicate）；其他返回 base64 负载（OpenAI gpt-image-2）。对于 base64 情况，使用 `save_b64_image()` — 它写入 `$HERMES_HOME/cache/images/<prefix>_<timestamp>_<uuid>.<ext>` 并返回绝对 `Path`。将该路径（作为 `str`）作为 `image=` 在 `success_response()` 中传递。网关投递（Telegram 照片气泡、Discord 附件）识别 URL 和绝对路径。

## 用户覆盖

将用户插件放在 `~/.hermes/plugins/image_gen/<name>/`，名称属性与捆绑插件相同，并通过 `hermes plugins enable <name>` 启用 — 注册表后写入者获胜，因此您的版本替换内置版本。可用于将 `openai` 插件指向私有代理，或换入自定义模型目录。

## 测试

```bash
export HERMES_HOME=/tmp/hermes-imggen-test
mkdir -p $HERMES_HOME/plugins/image_gen/my-backend
# …copy __init__.py + plugin.yaml into that dir…

export MY_BACKEND_API_KEY=your-test-key
hermes plugins enable my-backend

# Pick it as the active provider
echo "image_gen:" >> $HERMES_HOME/config.yaml
echo "  provider: my-backend" >> $HERMES_HOME/config.yaml

# Exercise it
hermes -z "Generate an image of a corgi in a spacesuit"
```

或交互式：`hermes tools` → "Image Generation" → 选择 `my-backend` → 如果提示则输入 API 密钥。

## 参考实现

- **`plugins/image_gen/openai/__init__.py`** — gpt-image-2 在低/中/高等级作为三个共享一个 API 模型的虚拟模型 ID，具有不同的 `quality` 参数。单一后端下分层模型的良好示例 + config.yaml 优先级链。
- **`plugins/image_gen/xai/__init__.py`** — 通过 xAI 的 Grok Imagine。不同形状（URL 输出、更简单的目录）。
- **`plugins/image_gen/openai-codex/__init__.py`** — 通过不同路由基础 URL 重用 OpenAI SDK 的 Codex 风格 Responses API 变体。

## 通过 pip 分发

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-backend-imggen = "my_backend_imggen_package"
```

`my_backend_imggen_package` 必须暴露顶层 `register` 函数。参见通用插件指南中的[通过 pip 分发](/docs/guides/build-a-hermes-plugin#distribute-via-pip)获取完整的入口点设置。

## 相关页面

- [图像生成](/docs/user-guide/features/image-generation) — 用户面向的功能文档
- [插件概述](/docs/user-guide/features/plugins) — 所有插件类型一览
- [构建 Hermes 插件](/docs/guides/build-a-hermes-plugin) — 通用工具/钩子/斜杠命令指南
