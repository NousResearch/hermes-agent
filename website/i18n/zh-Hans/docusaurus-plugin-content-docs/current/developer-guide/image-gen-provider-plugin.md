---
sidebar_position: 11
title: "图像生成 Provider 插件"
description: "如何为 Hermes Agent 构建图像生成后端插件"
---

# 构建图像生成 Provider 插件

图像生成 provider 插件注册一个后端，用于处理所有 `image_generate` 工具调用——DALL·E、gpt-image、Grok、Flux、Imagen、Stable Diffusion、fal、Replicate、本地 ComfyUI 装置，任何后端均可。内置 provider（OpenAI、OpenAI-Codex、xAI）均以插件形式提供。你可以通过在 `plugins/image_gen/<name>/` 目录下放置一个目录来添加新的 provider，或覆盖内置 provider。

:::tip
图像生成是 Hermes 支持的多种**后端插件**之一。其他插件（各有更专用的 ABC）包括：[Memory Provider 插件](/developer-guide/memory-provider-plugin)、[Context Engine 插件](/developer-guide/context-engine-plugin) 和 [Model Provider 插件](/developer-guide/model-provider-plugin)。通用工具/hook/CLI 插件请参阅 [构建 Hermes 插件](/guides/build-a-hermes-plugin)。
:::

## 发现机制

Hermes 在三个位置扫描图像生成后端：

1. **内置** — `<repo>/plugins/image_gen/<name>/`（以 `kind: backend` 自动加载，始终可用）
2. **用户** — `~/.hermes/plugins/image_gen/<name>/`（通过 `plugins.enabled` 选择启用）
3. **Pip** — 声明了 `hermes_agent.plugins` 入口点的包

每个插件的 `register(ctx)` 函数调用 `ctx.register_image_gen_provider(...)` — 将其注册到 `agent/image_gen_registry.py` 中的注册表。活跃 provider 由 `config.yaml` 中的 `image_gen.provider` 指定；`hermes tools` 会引导用户完成选择。

`image_generate` 工具包装器向注册表请求活跃 provider 并分发调用。若未注册任何 provider，工具会显示一条有用的错误信息，指引用户使用 `hermes tools`。

## 目录结构

```
plugins/image_gen/my-backend/
├── __init__.py      # ImageGenProvider 子类 + register()
└── plugin.yaml      # 包含 kind: backend 的清单文件
```

内置插件到此即完整。位于 `~/.hermes/plugins/image_gen/<name>/` 的用户插件需要在 `config.yaml` 的 `plugins.enabled` 中添加（或运行 `hermes plugins enable <name>`）。

## ImageGenProvider ABC

继承 `agent.image_gen_provider.ImageGenProvider`。唯一必须实现的成员是 `name` 属性和 `generate()` 方法——其他所有成员均有合理的默认值：

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

`kind: backend` 决定插件被路由到图像生成注册路径。`requires_env` 在 `hermes plugins install` 期间会提示用户输入。

## ABC 参考

完整契约位于 `agent/image_gen_provider.py`。通常需要覆盖的方法：

| 成员 | 必须 | 默认值 | 用途 |
|---|---|---|---|
| `name` | ✅ | — | 在 `image_gen.provider` 配置中使用的稳定 id |
| `display_name` | — | `name.title()` | 在 `hermes tools` 中显示的标签 |
| `is_available()` | — | `True` | 缺少凭据/依赖时的拦截门控 |
| `list_models()` | — | `[]` | `hermes tools` 模型选择器的目录 |
| `default_model()` | — | `list_models()` 的第一项 | 未配置模型时的回退 |
| `get_setup_schema()` | — | 最小值 | 选择器元数据 + 环境变量提示 |
| `generate(prompt, aspect_ratio, **kwargs)` | ✅ | — | 实际调用 |

## 响应格式

`generate()` 必须返回通过 `success_response()` 或 `error_response()` 构建的字典。两者均位于 `agent/image_gen_provider.py`。

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

工具包装器将字典 JSON 序列化后传给 LLM。错误以工具结果的形式呈现；LLM 决定如何向用户解释。

## 处理 base64 与 URL 输出

部分后端返回图像 URL（fal、Replicate）；其他后端返回 base64 载荷（OpenAI gpt-image-2）。对于 base64 情况，使用 `save_b64_image()` — 它将文件写入 `$HERMES_HOME/cache/images/<prefix>_<timestamp>_<uuid>.<ext>` 并返回绝对 `Path`。将该路径（转为 `str`）作为 `image=` 传入 `success_response()`。Gateway 投递（Telegram 图片气泡、Discord 附件）同时识别 URL 和绝对路径。

## 用户覆盖

在 `~/.hermes/plugins/image_gen/<name>/` 放置一个用户插件，使其 `name` 属性与某个内置插件相同，并通过 `hermes plugins enable <name>` 启用——注册表采用后写入优先策略，你的版本将替换内置版本。适用于将 `openai` 插件指向私有代理，或替换自定义模型目录等场景。

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

或交互式操作：`hermes tools` → "Image Generation" → 选择 `my-backend` → 根据提示输入 API key。

## 参考实现

- **`plugins/image_gen/openai/__init__.py`** — gpt-image-2 以低/中/高三个档位作为三个虚拟模型 ID，共享同一 API 模型并使用不同的 `quality` 参数。适合参考单一后端下的分层模型设计 + config.yaml 优先级链。
- **`plugins/image_gen/xai/__init__.py`** — 通过 xAI 的 Grok Imagine。不同的响应结构（URL 输出，目录更简单）。
- **`plugins/image_gen/openai-codex/__init__.py`** — Codex 风格的 Responses API 变体，复用 OpenAI SDK 并使用不同的路由基础 URL。

## 通过 pip 分发

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-backend-imggen = "my_backend_imggen_package"
```

`my_backend_imggen_package` 必须暴露一个顶层 `register` 函数。完整配置请参阅通用插件指南中的 [通过 pip 分发](/guides/build-a-hermes-plugin#distribute-via-pip)。

## 相关页面

- [图像生成](/user-guide/features/image-generation) — 面向用户的功能文档
- [插件概览](/user-guide/features/plugins) — 所有插件类型一览
- [构建 Hermes 插件](/guides/build-a-hermes-plugin) — 通用工具/hook/斜杠命令指南