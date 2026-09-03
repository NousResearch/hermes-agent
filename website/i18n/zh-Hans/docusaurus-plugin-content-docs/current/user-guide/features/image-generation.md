---
title: 文生图（Image Generation）
description: 通过 FAL.ai 文生图；支持 8 个模型，含 FLUX 2、GPT-Image、Nano Banana Pro、Ideogram、Recraft V4 Pro 等，可用 hermes tools 切换。
sidebar_label: 文生图
sidebar_position: 6
---

# 文生图（Image Generation）

Hermes Agent 通过 FAL.ai 根据文字提示生成图像。默认内置 8 个模型，在速度、画质与成本上各有取舍。当前模型可通过 `hermes tools` 配置，并持久化在 `config.yaml`。

## 支持的模型

| 模型 | 速度 | 特点 | 参考价格 |
|------|------|------|----------|
| `fal-ai/flux-2/klein/9b` *（默认）* | `<1s` | 快、文字清晰 | $0.006/MP |
| `fal-ai/flux-2-pro` | ~6s | 棚拍级写实 | $0.03/MP |
| `fal-ai/z-image/turbo` | ~2s | 中英双语，6B | $0.005/MP |
| `fal-ai/nano-banana-pro` | ~8s | Gemini 3 Pro、推理与文字渲染 | $0.15/张（1K） |
| `fal-ai/gpt-image-1.5` | ~15s | 强指令遵循 | $0.034/张 |
| `fal-ai/ideogram/v3` | ~5s | 排版最佳 | $0.03–0.09/张 |
| `fal-ai/recraft/v4/pro/text-to-image` | ~8s | 设计 / 品牌系统 / 可交付生产 | $0.25/张 |
| `fal-ai/qwen-image` | ~12s | 偏 LLM 式、复杂文字 | $0.02/MP |

价格为撰写时的 FAL 官方口径；最新计费请以 [fal.ai](https://fal.ai/) 为准。

## 配置

:::tip Nous 订阅用户
若你持有付费 [Nous Portal](https://portal.nousresearch.com) 订阅，可通过 **[Tool Gateway](tool-gateway.md)** 使用文生图，**无需** `FAL_KEY`。模型选择在「直连 FAL」与「订阅网关」两条路径下保持一致。

若托管网关对某一模型返回 `HTTP 4xx`，通常表示该模型尚未在 Portal 侧代理——智能体会给出处理建议（例如配置 `FAL_KEY` 直连，或换用其他模型）。
:::

### 获取 FAL API Key

1. 在 [fal.ai](https://fal.ai/) 注册  
2. 在控制台生成 API Key  

### 配置并选择模型

执行：

```bash
hermes tools
```

进入 **🎨 Image Generation**，选择后端（Nous Subscription 或 FAL.ai），随后在表格中用方向键选择模型，回车确认：

```
  Model                          Speed    Strengths                    Price
  fal-ai/flux-2/klein/9b         <1s      Fast, crisp text             $0.006/MP   ← currently in use
  fal-ai/flux-2-pro              ~6s      Studio photorealism          $0.03/MP
  fal-ai/z-image/turbo           ~2s      Bilingual EN/CN, 6B          $0.005/MP
  ...
```

选择会写入 `config.yaml`：

```yaml
image_gen:
  model: fal-ai/flux-2/klein/9b
  use_gateway: false            # 使用 Nous Subscription 时为 true
```

### GPT-Image 画质档位

`fal-ai/gpt-image-1.5` 的请求画质固定为 `medium`（约 1024×1024 下 $0.034/张）。面向用户**不开放** `low` / `high` 档位，以便 Nous Portal 侧计费在全体用户间更可预期（档位价差约 22×）。若需要更便宜的 GPT-Image 路线，请换其他模型；若追求更高画质，可考虑 Klein 9B 或同类 Imagen 系模型。

## 使用方式

对智能体暴露的 schema 刻意保持简单——具体行为由你在本机的配置决定：

```
Generate an image of a serene mountain landscape with cherry blossoms
```

```
Create a square portrait of a wise old owl — use the typography model
```

```
Make me a futuristic cityscape, landscape orientation
```

## 宽高比

从智能体视角，三个宽高比词对所有模型通用；内部会映射到各模型原生参数：

| 智能体输入 | image_size（flux/z-image/qwen/recraft/ideogram） | aspect_ratio（nano-banana-pro） | image_size（gpt-image） |
|---|---|---|---|
| `landscape` | `landscape_16_9` | `16:9` | `1536x1024` |
| `square` | `square_hd` | `1:1` | `1024x1024` |
| `portrait` | `portrait_16_9` | `9:16` | `1024x1536` |

该映射在 `_build_fal_payload()` 中完成，智能体代码无需了解各模型 schema 差异。

## 自动超分（Upscale）

是否启用 FAL **Clarity Upscaler** 按模型区分：

| 模型 | 超分？ | 原因 |
|---|---|---|
| `fal-ai/flux-2-pro` | ✓ | 历史兼容（选择器出现前的默认） |
| 其他 | ✗ | 亚秒级模型若再超分会失去速度优势；高分辨率模型本身已足够清晰 |

超分启用时的主要参数：

| 项 | 值 |
|---|---|
| 放大倍数 | 2× |
| Creativity | 0.35 |
| Resemblance | 0.6 |
| Guidance scale | 4 |
| Inference steps | 18 |

若超分失败（网络、限流等），会自动回退为返回原始图像。

## 内部流程概要

1. **模型解析** — `_resolve_fal_model()` 读取 `config.yaml` 的 `image_gen.model`，否则看 `FAL_IMAGE_MODEL` 环境变量，再否则默认 `fal-ai/flux-2/klein/9b`。  
2. **构造请求体** — `_build_fal_payload()` 将 `aspect_ratio` 转为各模型枚举或字面量，合并默认参数与调用方覆盖，并按 `supports` 白名单过滤非法字段。  
3. **提交** — `_submit_fal_request()` 根据凭据走直连 FAL 或 Nous 托管网关。  
4. **超分** — 仅当模型元数据标记 `upscale: True` 时执行。  
5. **交付** — 最终图像 URL 返回给智能体，并发出 `MEDIA:<url>`，由各平台适配器转为原生媒体消息。  

## 调试

打开调试日志：

```bash
export IMAGE_TOOLS_DEBUG=true
```

日志写入 `./logs/image_tools_debug_<session_id>.json`，包含每次调用的模型、参数、耗时与错误信息。

## 各平台展示

| 平台 | 行为 |
|---|---|
| **CLI** | 图像 URL 以 Markdown `![](url)` 打印，可点击打开 |
| **Telegram** | 以图片消息发送，附提示词为说明 |
| **Discord** | 嵌入消息 |
| **Slack** | URL 由 Slack 展开预览 |
| **WhatsApp** | 媒体消息 |
| **其他** | 纯文本中的 URL |

## 限制

- **需要 FAL 凭据**（直连 `FAL_KEY` 或 Nous 订阅网关）  
- **仅文生图** — 不支持局部重绘、图生图或编辑类工作流  
- **临时 URL** — FAL 托管链接会在数小时至数天后过期；请自行落盘保存  
- **按模型能力裁剪** — 部分模型不支持 `seed`、`num_inference_steps` 等；`supports` 会静默丢弃不支持的参数，属预期行为  
