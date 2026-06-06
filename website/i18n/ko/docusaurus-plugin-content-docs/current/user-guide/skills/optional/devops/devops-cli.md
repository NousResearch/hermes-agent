---
title: "Inference Sh Cli — inference를 통해 150개 이상의 AI 앱 실행"
sidebar_label: "Inference Sh Cli"
description: "inference를 통해 150개 이상의 AI 앱 실행"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Inference Sh Cli

inference.sh CLI (infsh)를 통해 150개 이상의 AI 앱을 실행합니다 — 이미지 생성, 비디오 생성, LLM, 검색, 3D, 소셜 자동화. 터미널 도구를 사용합니다. 트리거: inference.sh, infsh, ai apps, flux, veo, image generation, video generation, seedream, seedance, tavily

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/cli` |
| Path | `optional-skills/devops/cli` |
| Version | `1.0.0` |
| Author | okaris |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `AI`, `image-generation`, `video`, `LLM`, `search`, `inference`, `FLUX`, `Veo`, `Claude` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# inference.sh CLI

간단한 CLI로 클라우드에서 150개 이상의 AI 앱을 실행하세요. GPU가 필요하지 않습니다.

모든 명령어는 `infsh` 명령을 실행하기 위해 **terminal 도구**를 사용합니다.

## When to Use

- 사용자가 이미지 생성을 요청할 때 (FLUX, Reve, Seedream, Grok, Gemini image)
- 사용자가 비디오 생성을 요청할 때 (Veo, Wan, Seedance, OmniHuman)
- 사용자가 inference.sh 또는 infsh에 대해 질문할 때
- 사용자가 개별 제공업체의 API를 관리하지 않고 AI 앱을 실행하고 싶어할 때
- 사용자가 AI 기반 검색을 요청할 때 (Tavily, Exa)
- 사용자가 아바타/립싱크 생성이 필요할 때

## Prerequisites

`infsh` CLI가 설치되고 인증되어야 합니다. 다음으로 확인하세요:

```bash
infsh me
```

설치되지 않은 경우:

```bash
curl -fsSL https://cli.inference.sh | sh
infsh login
```

전체 설정 세부 정보는 `references/authentication.md`를 참조하세요.

## Workflow

### 1. Always Search First

앱 이름을 절대 추측하지 마세요 — 항상 검색하여 올바른 앱 ID를 찾으세요:

```bash
infsh app list --search flux
infsh app list --search video
infsh app list --search image
```

### 2. Run an App

검색 결과에서 정확한 앱 ID를 사용하세요. 기계가 읽을 수 있는 출력을 위해 항상 `--json`을 사용하세요:

```bash
infsh app run <app-id> --input '{"prompt": "your prompt here"}' --json
```

### 3. Parse the Output

JSON 출력에는 생성된 미디어에 대한 URL이 포함되어 있습니다. 인라인 표시를 위해 `MEDIA:<url>` 형식을 사용하여 사용자에게 이를 제시하세요.

## Common Commands

### Image Generation

```bash
# 이미지 앱 검색
infsh app list --search image

# LoRA를 사용한 FLUX Dev
infsh app run falai/flux-dev-lora --input '{"prompt": "sunset over mountains", "num_images": 1}' --json

# Gemini 이미지 생성
infsh app run google/gemini-2-5-flash-image --input '{"prompt": "futuristic city", "num_images": 1}' --json

# Seedream (ByteDance)
infsh app run bytedance/seedream-5-lite --input '{"prompt": "nature scene"}' --json

# Grok Imagine (xAI)
infsh app run xai/grok-imagine-image --input '{"prompt": "abstract art"}' --json
```

### Video Generation

```bash
# 비디오 앱 검색
infsh app list --search video

# Veo 3.1 (Google)
infsh app run google/veo-3-1-fast --input '{"prompt": "drone shot of coastline"}' --json

# Seedance (ByteDance)
infsh app run bytedance/seedance-1-5-pro --input '{"prompt": "dancing figure", "resolution": "1080p"}' --json

# Wan 2.5
infsh app run falai/wan-2-5 --input '{"prompt": "person walking through city"}' --json
```

### Local File Uploads

로컬 경로를 제공하면 CLI가 자동으로 파일을 업로드합니다:

```bash
# 로컬 이미지 업스케일링
infsh app run falai/topaz-image-upscaler --input '{"image": "/path/to/photo.jpg", "upscale_factor": 2}' --json

# 로컬 파일에서 이미지를 비디오로
infsh app run falai/wan-2-5-i2v --input '{"image": "/path/to/image.png", "prompt": "make it move"}' --json

# 오디오가 포함된 아바타
infsh app run bytedance/omnihuman-1-5 --input '{"audio": "/path/to/audio.mp3", "image": "/path/to/face.jpg"}' --json
```

### Search & Research

```bash
infsh app list --search search
infsh app run tavily/tavily-search --input '{"query": "latest AI news"}' --json
infsh app run exa/exa-search --input '{"query": "machine learning papers"}' --json
```

### Other Categories

```bash
# 3D 생성
infsh app list --search 3d

# 오디오 / TTS
infsh app list --search tts

# Twitter/X 자동화
infsh app list --search twitter
```

## Pitfalls

1. **절대 앱 ID를 추측하지 마세요** — 항상 `infsh app list --search <term>`을 먼저 실행하세요. 앱 ID는 변경되며 새로운 앱이 자주 추가됩니다.
2. **항상 `--json`을 사용하세요** — 원시 출력은 파싱하기 어렵습니다. `--json` 플래그는 URL이 포함된 구조화된 출력을 제공합니다.
3. **인증 확인** — 인증 오류로 명령이 실패하는 경우, `infsh login`을 실행하거나 `INFSH_API_KEY`가 설정되어 있는지 확인하세요.
4. **장기 실행 앱** — 비디오 생성에는 30-120초가 걸릴 수 있습니다. 터미널 도구 시간 초과는 충분해야 하지만, 사용자에게 약간의 시간이 걸릴 수 있음을 경고하세요.
5. **입력 형식** — `--input` 플래그는 JSON 문자열을 사용합니다. 따옴표를 올바르게 이스케이프(escape) 처리해야 합니다.

## Reference Docs

- `references/authentication.md` — 설정, 로그인, API 키
- `references/app-discovery.md` — 앱 카탈로그 검색 및 탐색
- `references/running-apps.md` — 앱 실행, 입력 형식, 출력 처리
- `references/cli-reference.md` — 전체 CLI 명령 참조
