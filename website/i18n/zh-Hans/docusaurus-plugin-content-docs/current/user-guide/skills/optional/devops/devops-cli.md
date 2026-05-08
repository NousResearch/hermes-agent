---
title: "Inference Sh Cli — 通过 inference 运行 150+ AI 应用"
sidebar_label: "Inference Sh Cli"
description: "通过 inference 运行 150+ AI 应用"
---

{/* 此页面由 website/scripts/generate-skill-docs.py 根据 skill 的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Inference Sh Cli

通过 inference.sh CLI (infsh) 运行 150+ AI 应用 — 图片生成、视频创作、LLM、搜索、3D、社交自动化。使用终端工具。触发词：inference.sh、infsh、ai apps、flux、veo、image generation、video generation、seedream、seedance、tavily

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/devops/cli` 安装 |
| 路径 | `optional-skills/devops/cli` |
| 版本 | `1.0.0` |
| 作者 | okaris |
| 许可证 | MIT |
| 标签 | `AI`、`image-generation`、`video`、`LLM`、`search`、`inference`、`FLUX`、`Veo`、`Claude` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# inference.sh CLI

通过简单的 CLI 在云端运行 150+ AI 应用。无需 GPU。

所有命令使用**终端工具**运行 `infsh` 命令。

## 使用场景

- 用户请求生成图片（FLUX、Reve、Seedream、Grok、Gemini 图片）
- 用户请求生成视频（Veo、Wan、Seedance、OmniHuman）
- 用户询问 inference.sh 或 infsh
- 用户想要运行 AI 应用而不管理单独的提供商 API
- 用户请求 AI 驱动的搜索（Tavily、Exa）
- 用户需要头像/唇形同步生成

## 前置条件

`infsh` CLI 必须已安装并通过身份验证。检查：

```bash
infsh me
```

如果未安装：

```bash
curl -fsSL https://cli.inference.sh | sh
infsh login
```

有关完整设置详情，请参见 `references/authentication.md`。

## 工作流程

### 1. 始终先搜索

永远不要猜测应用名称 — 始终搜索以找到正确的应用 ID：

```bash
infsh app list --search flux
infsh app list --search video
infsh app list --search image
```

### 2. 运行应用

使用搜索结果中的确切应用 ID。始终使用 `--json` 以获得机器可读输出：

```bash
infsh app run <app-id> --input '{"prompt": "your prompt here"}' --json
```

### 3. 解析输出

JSON 输出包含生成的媒体 URL。使用 `MEDIA:<url>` 向用户展示以便内联显示。

## 常用命令

### 图片生成

```bash
# 搜索图片应用
infsh app list --search image

# FLUX Dev with LoRA
infsh app run falai/flux-dev-lora --input '{"prompt": "sunset over mountains", "num_images": 1}' --json

# Gemini 图片生成
infsh app run google/gemini-2-5-flash-image --input '{"prompt": "futuristic city", "num_images": 1}' --json

# Seedream (ByteDance)
infsh app run bytedance/seedream-5-lite --input '{"prompt": "nature scene"}' --json

# Grok Imagine (xAI)
infsh app run xai/grok-imagine-image --input '{"prompt": "abstract art"}' --json
```

### 视频生成

```bash
# 搜索视频应用
infsh app list --search video

# Veo 3.1 (Google)
infsh app run google/veo-3-1-fast --input '{"prompt": "drone shot of coastline"}' --json

# Seedance (ByteDance)
infsh app run bytedance/seedance-1-5-pro --input '{"prompt": "dancing figure", "resolution": "1080p"}' --json

# Wan 2.5
infsh app run falai/wan-2-5 --input '{"prompt": "person walking through city"}' --json
```

### 本地文件上传

CLI 在您提供路径时自动上传本地文件：

```bash
# 放大本地图片
infsh app run falai/topaz-image-upscaler --input '{"image": "/path/to/photo.jpg", "upscale_factor": 2}' --json

# 从本地文件进行图片转视频
infsh app run falai/wan-2-5-i2v --input '{"image": "/path/to/image.png", "prompt": "make it move"}' --json

# 带音频的头像
infsh app run bytedance/omnihuman-1-5 --input '{"audio": "/path/to/audio.mp3", "image": "/path/to/face.jpg"}' --json
```

### 搜索与研究

```bash
infsh app list --search search
infsh app run tavily/tavily-search --input '{"query": "latest AI news"}' --json
infsh app run exa/exa-search --input '{"query": "machine learning papers"}' --json
```

### 其他类别

```bash
# 3D 生成
infsh app list --search 3d

# 音频 / TTS
infsh app list --search tts

# Twitter/X 自动化
infsh app list --search twitter
```

## 陷阱

1. **永远不要猜测应用 ID** — 始终先运行 `infsh app list --search <term>`。应用 ID 会变化且经常添加新应用。
2. **始终使用 `--json`** — 原始输出难以解析。`--json` 标志提供带有 URL 的结构化输出。
3. **检查身份验证** — 如果命令因身份验证错误失败，运行 `infsh login` 或验证 `INFSH_API_KEY` 已设置。
4. **长时间运行的应用** — 视频生成可能需要 30-120 秒。终端工具超时应该足够，但警告用户可能需要一点时间。
5. **输入格式** — `--input` 标志采用 JSON 字符串。确保正确转义引号。

## 参考文档

- `references/authentication.md` — 设置、登录、API 密钥
- `references/app-discovery.md` — 搜索和浏览应用目录
- `references/running-apps.md` — 运行应用、输入格式、输出处理
- `references/cli-reference.md` — 完整 CLI 命令参考
