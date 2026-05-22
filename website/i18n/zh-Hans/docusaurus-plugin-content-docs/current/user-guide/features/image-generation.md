---
title: 图像生成
description: 通过 FAL.ai 生成图像 —— 9 个模型包括 FLUX 2、GPT Image (1.5 & 2)、Nano Banana Pro、Ideogram、Recraft V4 Pro 等，可通过 `hermes tools` 选择。
sidebar_label: 图像生成
sidebar_position: 6
---

# 图像生成

Hermes Agent 通过 FAL.ai 从文本提示生成图像。开箱即用地支持九个模型，每个模型在速度、质量和成本上有不同的权衡。活动模型可通过 `hermes tools` 由用户配置，并持久化保存在 `config.yaml` 中。

## 支持的模型

| 模型 | 速度 | 优势 | 价格 |
|---|---|---|---|
| `fal-ai/flux-2/klein/9b` *(默认)* | `<1s` | 快速，清晰的文本 | $0.006/MP |
| `fal-ai/flux-2-pro` | ~6s | 工作室级真实感 | $0.03/MP |
| `fal-ai/z-image/turbo` | ~2s | 中英双语，6B 参数 | $0.005/MP |
| `fal-ai/nano-banana-pro` | ~8s | Gemini 3 Pro，推理深度，文本渲染 | $0.15/图像 (1K) |
| `fal-ai/gpt-image-1.5` | ~15s | 提示遵循度 | $0.034/图像 |
| `fal-ai/gpt-image-2` | ~20s | SOTA 文本渲染 + CJK，世界感知真实感 | $0.04–0.06/图像 |
| `fal-ai/ideogram/v3` | ~5s | 最佳排版 | $0.03–0.09/图像 |
| `fal-ai/recraft/v4/pro/text-to-image` | ~8s | 设计、品牌系统、生产就绪 | $0.25/图像 |
| `fal-ai/qwen-image` | ~12s | 基于 LLM，复杂文本 | $0.02/MP |

价格为撰写时的 FAL 定价；查看 [fal.ai](https://fal.ai/) 获取当前数字。

## 设置

:::tip Nous 订阅者
如果您有付费的 [Nous Portal](https://portal.nousresearch.com) 订阅，您可以通过 **[Tool Gateway](tool-gateway.md)** 使用图像生成，无需 FAL API 密钥。您的模型选择在两条路径间持久保存。

如果托管网关对特定模型返回 `HTTP 4xx`，则该模型尚未在门户端代理 —— 智能体会告诉您，并提供修复步骤（设置 `FAL_KEY` 进行直接访问，或选择不同的模型）。
:::

### 获取 FAL API 密钥

1. 在 [fal.ai](https://fal.ai/) 注册
2. 从您的仪表板生成 API 密钥

### 配置并选择模型

运行 tools 命令：

```bash
hermes tools
```

导航到 **🎨 图像生成**，选择您的后端（Nous 订阅或 FAL.ai），然后选择器会在列对齐的表格中显示所有支持的模型 —— 方向键导航，Enter 选择：

```
  模型                          速度    优势                        价格
  fal-ai/flux-2/klein/9b         <1s      快速，清晰的文本             $0.006/MP   ← 当前使用中
  fal-ai/flux-2-pro              ~6s      工作室级真实感               $0.03/MP
  fal-ai/z-image/turbo           ~2s      中英双语，6B                 $0.005/MP
  ...
```

您的选择保存到 `config.yaml`：

```yaml
image_gen:
  model: fal-ai/flux-2/klein/9b
  use_gateway: false            # 如果使用 Nous 订阅则为 true
```

### GPT-Image 质量

`fal-ai/gpt-image-1.5` 和 `fal-ai/gpt-image-2` 的请求质量固定为 `medium`（1024×1024 时约 $0.034–$0.06/图像）。我们不将 `low` / `high` 层级作为面向用户的选项暴露，以便 Nous Portal 计费在所有用户间保持可预测 —— 层级间的成本差异为 3–22 倍。如果您想要更便宜的选项，选择 Klein 9B 或 Z-Image Turbo；如果您想要更高质量，使用 Nano Banana Pro 或 Recraft V4 Pro。

## 用法

面向智能体的模式有意保持最小化 —— 模型会拾取您配置的任何内容：

```
生成一幅宁静的山景与樱花的图像
```

```
创建一只智慧老猫头鹰的方形肖像 —— 使用排版模型
```

```
为我制作一幅未来城市景观，横向
```

## 宽高比

每个模型从智能体的角度接受相同的三种宽高比。在内部，每个模型的原生尺寸规格会自动填充：

| 智能体输入 | image_size (flux/z-image/qwen/recraft/ideogram) | aspect_ratio (nano-banana-pro) | image_size (gpt-image-1.5) | image_size (gpt-image-2) |
|---|---|---|---|---|
| `landscape` | `landscape_16_9` | `16:9` | `1536x1024` | `landscape_4_3` (1024×768) |
| `square` | `square_hd` | `1:1` | `1024x1024` | `square_hd` (1024×1024) |
| `portrait` | `portrait_16_9` | `9:16` | `1024x1536` | `portrait_4_3` (768×1024) |

GPT Image 2 映射到 4:3 预设而非 16:9，因为其最小像素数为 655,360 —— `landscape_16_9` 预设 (1024×576 = 589,824) 会被拒绝。

此转换发生在 `_build_fal_payload()` 中 —— 智能体代码永远不需要了解每个模型的模式差异。

## 自动放大

通过 FAL 的 **Clarity Upscaler** 进行放大是按模型控制的：

| 模型 | 放大？ | 原因 |
|---|---|---|
| `fal-ai/flux-2-pro` | ✓ | 向后兼容（选择器之前的默认） |
| 所有其他 | ✗ | 快速模型会失去其次秒价值主张；高分辨率模型不需要 |

放大运行时，使用以下设置：

| 设置 | 值 |
|---|---|
| 放大倍数 | 2× |
| 创造力 | 0.35 |
| 相似度 | 0.6 |
| 引导尺度 | 4 |
| 推理步数 | 18 |

如果放大失败（网络问题、速率限制），原始图像会自动返回。

## 内部工作原理

1. **模型解析** —— `_resolve_fal_model()` 从 `config.yaml` 读取 `image_gen.model`，回退到 `FAL_IMAGE_MODEL` 环境变量，然后回退到 `fal-ai/flux-2/klein/9b`。
2. **负载构建** —— `_build_fal_payload()` 将您的 `aspect_ratio` 转换为模型的原生格式（预设枚举、宽高比枚举或 GPT 字面量），合并模型的默认参数，应用任何调用方覆盖，然后过滤到模型的 `supports` 白名单，以便不发送不受支持的键。
3. **提交** —— `_submit_fal_request()` 通过直接 FAL 凭证或托管的 Nous 网关路由。
4. **放大** —— 仅当模型的元数据具有 `upscale: True` 时运行。
5. **投递** —— 最终图像 URL 返回给智能体，智能体发出 `MEDIA:<url>` 标签，平台适配器将其转换为原生媒体。

## 调试

启用调试日志：

```bash
export IMAGE_TOOLS_DEBUG=true
```

调试日志写入 `./logs/image_tools_debug_<session_id>.json`，包含每次调用的详细信息（模型、参数、时间、错误）。

## 平台投递

| 平台 | 投递方式 |
|---|---|
| **CLI** | 图像 URL 打印为 markdown `![](url)` —— 点击打开 |
| **Telegram** | 带有提示作为标题的照片消息 |
| **Discord** | 嵌入在消息中 |
| **Slack** | 由 Slack 展开 URL |
| **WhatsApp** | 媒体消息 |
| **其他** | 纯文本中的 URL |

## 限制

- **需要 FAL 凭证**（直接 `FAL_KEY` 或 Nous 订阅）
- **仅文生图** —— 此工具不支持修复、图生图或编辑
- **临时 URL** —— FAL 返回的托管 URL 在数小时/天后过期；如需保存请下载到本地
- **每模型限制** —— 某些模型不支持 `seed`、`num_inference_steps` 等。`supports` 过滤器静默丢弃不受支持的参数；这是预期行为
