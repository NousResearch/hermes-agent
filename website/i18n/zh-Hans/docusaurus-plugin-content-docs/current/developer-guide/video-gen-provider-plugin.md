---
sidebar_position: 12
title: "视频生成 Provider 插件"
description: "如何为 Hermes Agent 构建视频生成后端插件"
---

# 构建视频生成 Provider 插件

视频生成 provider 插件注册一个后端，用于处理所有 `video_generate` 工具调用。内置 provider（xAI、FAL）以插件形式提供。将目录放入 `plugins/video_gen/<name>/` 即可添加新 provider 或覆盖内置 provider。

:::tip
视频生成与[图像生成 Provider 插件](/developer-guide/image-gen-provider-plugin)几乎一一对应——如果你已构建过图像生成后端，对其结构应已了然于胸。主要区别在于：`capabilities()` 方法用于声明模态（modality）/宽高比/时长，以及路由约定（传入 `image_url` 则使用图生视频，省略则使用文生视频——provider 在内部选择正确的端点）。
:::

## 统一接口（一个工具，两种模态）

`video_generate` 工具通过一个参数暴露两种模态：

- **文生视频（Text-to-video）** — 仅传入 `prompt`。Provider 路由至其文生视频端点。
- **图生视频（Image-to-video）** — 同时传入 `prompt` 和 `image_url`。Provider 路由至其图生视频端点。

编辑和扩展功能有意不在支持范围内。大多数后端不支持这些功能，且不一致性会迫使 agent 的工具描述中出现针对各后端的说明文字。

## 发现机制

Hermes 在三个位置扫描视频生成后端：

1. **内置** — `<repo>/plugins/video_gen/<name>/`（通过 `kind: backend` 自动加载）
2. **用户** — `~/.hermes/plugins/video_gen/<name>/`（通过 `plugins.enabled` 选择启用）
3. **Pip** — 声明了 `hermes_agent.plugins` 入口点的包

每个插件的 `register(ctx)` 函数调用 `ctx.register_video_gen_provider(...)`。活跃 provider 由 `config.yaml` 中的 `video_gen.provider` 指定；`hermes tools` → Video Generation 引导用户完成选择。与 `image_generate` 不同，此处没有内置的遗留后端——每个 provider 都是插件。

## 目录结构

```
plugins/video_gen/my-backend/
├── __init__.py      # VideoGenProvider 子类 + register()
└── plugin.yaml      # 包含 kind: backend 的清单文件
```

## VideoGenProvider ABC

继承 `agent.video_gen_provider.VideoGenProvider`。必须实现：`name` 属性和 `generate()` 方法。

```python
# plugins/video_gen/my-backend/__init__.py
from typing import Any, Dict, List, Optional
import os

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    success_response,
)


class MyVideoGenProvider(VideoGenProvider):
    @property
    def name(self) -> str:
        return "my-backend"

    @property
    def display_name(self) -> str:
        return "My Backend"

    def is_available(self) -> bool:
        return bool(os.environ.get("MY_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        # Each entry is a model FAMILY — a name the user picks once.
        # Your provider's generate() routes within the family based on
        # whether image_url was passed.
        return [
            {
                "id": "fast",
                "display": "Fast",
                "speed": "~30s",
                "strengths": "Cheapest tier",
                "price": "$0.05/s",
                "modalities": ["text", "image"],  # advisory
            },
        ]

    def default_model(self) -> Optional[str]:
        return "fast"

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16"],
            "resolutions": ["720p", "1080p"],
            "min_duration": 1,
            "max_duration": 10,
            "supports_audio": False,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "My Backend",
            "badge": "paid",
            "tag": "Short description shown in `hermes tools`",
            "env_vars": [
                {
                    "key": "MY_API_KEY",
                    "prompt": "My Backend API key",
                    "url": "https://mybackend.example.com/keys",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,  # always ignore unknown kwargs for forward-compat
    ) -> Dict[str, Any]:
        # ROUTE: image_url presence picks the endpoint.
        if image_url:
            endpoint = "my-backend/image-to-video"
            modality_used = "image"
        else:
            endpoint = "my-backend/text-to-video"
            modality_used = "text"

        # ... call your API ...

        return success_response(
            video="https://your-cdn/output.mp4",
            model=model or "fast",
            prompt=prompt,
            modality=modality_used,
            aspect_ratio=aspect_ratio,
            duration=duration or 5,
            provider=self.name,
        )


def register(ctx) -> None:
    ctx.register_video_gen_provider(MyVideoGenProvider())
```

## 插件清单

```yaml
# plugins/video_gen/my-backend/plugin.yaml
name: my-backend
version: 1.0.0
description: "My video generation backend"
author: Your Name
kind: backend
requires_env:
  - MY_API_KEY
```

## `video_generate` 参数模式

该工具在所有后端中使用统一的参数模式。Provider 忽略其不支持的参数。

| 参数 | 说明 |
|---|---|
| `prompt` | 文本指令（必填） |
| `image_url` | 设置时 → 图生视频；省略时 → 文生视频 |
| `reference_image_urls` | 风格/角色参考图（取决于 provider） |
| `duration` | 秒数——provider 会进行截断 |
| `aspect_ratio` | `"16:9"`、`"9:16"`、`"1:1"` 等——provider 会进行截断 |
| `resolution` | `"480p"` / `"540p"` / `"720p"` / `"1080p"`——provider 会进行截断 |
| `negative_prompt` | 需要避免的内容（仅 Pixverse/Kling 支持） |
| `audio` | 原生音频（Veo3 / Pixverse 定价层级） |
| `seed` | 可复现性 |
| `model` | 覆盖当前活跃的模型/系列 |

Provider 的 `capabilities()` 声明上述哪些参数会被实际处理。Agent 在工具描述中看到的是当前活跃后端的能力信息，当用户通过 `hermes tools` 切换后端时会动态重建。

## 模型系列与端点路由（FAL 模式）

当你的后端每个"模型"对应多个端点时——例如 FAL，其中每个系列（Veo 3.1、Pixverse v6、Kling O3）都有 `/text-to-video` 和 `/image-to-video` 两个 URL——将每个**系列**表示为一个目录条目。你的 `generate()` 根据是否传入 `image_url` 来选择正确的端点：

```python
FAMILIES = {
    "veo3.1": {
        "text_endpoint": "fal-ai/veo3.1",
        "image_endpoint": "fal-ai/veo3.1/image-to-video",
        # ... family-specific capability flags ...
    },
}

def generate(self, prompt, *, image_url=None, model=None, **kwargs):
    family_id, family = _resolve_family(model)
    endpoint = family["image_endpoint"] if image_url else family["text_endpoint"]
    # ... build payload from family's declared capability flags, call endpoint ...
```

用户在 `hermes tools` 中只需选择一次 `veo3.1`。Agent 无需关心端点——它只负责传入（或不传入）`image_url`。

## 选择优先级

针对每个实例的模型配置（参见 `plugins/video_gen/fal/__init__.py`）：

1. 工具调用中的 `model=` 关键字参数
2. `<PROVIDER>_VIDEO_MODEL` 环境变量
3. `config.yaml` 中的 `video_gen.<provider>.model`
4. `config.yaml` 中的 `video_gen.model`（当其值为你的某个 ID 时）
5. Provider 的 `default_model()`

## 响应结构

`success_response()` 和 `error_response()` 生成每个后端返回的标准 dict 结构。请使用它们——不要手动构造 dict。

成功响应的键：`success`、`video`（URL 或绝对路径）、`model`、`prompt`、`modality`（`"text"` 或 `"image"`）、`aspect_ratio`、`duration`、`provider`，以及 `extra`。

错误响应的键：`success`、`video`（None）、`error`、`error_type`、`model`、`prompt`、`aspect_ratio`、`provider`。

## 产物保存位置

如果你的后端返回 base64 数据，使用 `save_b64_video()` 将其写入 `$HERMES_HOME/cache/videos/`。对于通过后续 HTTP 请求获取的原始字节，使用 `save_bytes_video()`。否则直接返回上游 URL——gateway 在交付时会解析远程 URL。

## 测试

在 `tests/plugins/video_gen/test_<name>_plugin.py` 下添加冒烟测试。xAI 和 FAL 的测试展示了标准模式——注册、验证目录、分别在传入和不传入 `image_url` 的情况下测试路由，并断言在缺少认证时返回干净的错误响应。