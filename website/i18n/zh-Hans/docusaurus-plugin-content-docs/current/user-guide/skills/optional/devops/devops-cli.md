---
title: "Inference Sh Cli — 通过 inference 运行 150+ AI 应用"
sidebar_label: "Inference Sh Cli"
description: "通过 inference 运行 150+ AI 应用"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Inference Sh Cli

通过 inference.sh CLI（infsh）运行 150+ AI 应用——图像生成、视频创作、LLM、搜索、3D、社交自动化。使用终端工具。触发词：inference.sh、infsh、ai apps、flux、veo、image generation、video generation、seedream、seedance、tavily

## Skill 元数据

| | |
|---|---|
| 来源 | 可选——使用 `hermes skills install official/devops/cli` 安装 |
| 路径 | `optional-skills/devops/cli` |
| 版本 | `1.0.0` |
| 作者 | okaris |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `AI`, `image-generation`, `video`, `LLM`, `search`, `inference`, `FLUX`, `Veo`, `Claude` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在该 skill 被触发时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# inference.sh CLI

通过简单的 CLI 在云端运行 150+ AI 应用。无需 GPU。

所有命令均使用**终端工具**来运行 `infsh` 命令。

## 使用场景

- 用户要求生成图像（FLUX、Reve、Seedream、Grok、Gemini image）
- 用户要求生成视频（Veo、Wan、Seedance、OmniHuman）
- 用户询问 inference.sh 或 infsh
- 用户希望运行 AI 应用而无需管理各个提供商的 API
- 用户要求 AI 驱动的搜索（Tavily、Exa）
- 用户需要生成头像/口型同步

## 前置条件

`infsh` CLI 必须已安装并完成认证。使用以下命令检查：

```bash
infsh me
```

如未安装：

```bash
curl -fsSL https://cli.inference.sh | sh
infsh login
```

完整安装详情请参阅 `references/authentication.md`。

## 工作流程

### 1. 始终先搜索

不要猜测应用名称——始终通过搜索找到正确的应用 ID：

```bash
infsh app list --search flux
infsh app list --search video
infsh app list --search image
```

### 2. 运行应用

使用搜索结果中的精确应用 ID。始终使用 `--json` 获取机器可读的输出：

```bash
infsh app run <app-id> --input '{"prompt": "your prompt here"}' --json
```

### 3. 解析输出

JSON 输出包含指向生成媒体的 URL。使用 `MEDIA:<url>` 格式将其呈现给用户以内联显示。

## 常用命令

### 图像生成

```bash
# 搜索图像应用
infsh app list --search image

# FLUX Dev with LoRA
infsh app run falai/flux-dev-lora --input '{"prompt": "sunset over mountains", "num_images": 1}' --json

# Gemini 图像生成
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

CLI 会在提供路径时自动上传本地文件：

```bash
# 放大本地图像
infsh app run falai/topaz-image-upscaler --input '{"image": "/path/to/photo.jpg", "upscale_factor": 2}' --json

# 从本地文件生成图生视频
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

## 注意事项

1. **不要猜测应用 ID**——始终先运行 `infsh app list --search <term>`。应用 ID 会变更，新应用也会频繁添加。
2. **始终使用 `--json`**——原始输出难以解析。`--json` 标志提供包含 URL 的结构化输出。
3. **检查认证状态**——如果命令因认证错误失败，请运行 `infsh login` 或确认 `INFSH_API_KEY` 已设置。
4. **长时间运行的应用**——视频生成可能需要 30-120 秒。终端工具的超时时间应该足够，但请提前告知用户可能需要等待片刻。
5. **输入格式**——`--input` 标志接受 JSON 字符串。请确保正确转义引号。

## 参考文档

- `references/authentication.md` — 安装、登录、API 密钥
- `references/app-discovery.md` — 搜索和浏览应用目录
- `references/running-apps.md` — 运行应用、输入格式、输出处理
- `references/cli-reference.md` — 完整 CLI 命令参考