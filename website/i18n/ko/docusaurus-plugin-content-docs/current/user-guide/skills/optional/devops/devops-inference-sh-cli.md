---
title: "Inference Sh Cli — inference.sh CLI로 150개 이상의 AI 앱(이미지, 동영상, LLM) 실행"
sidebar_label: "Inference Sh Cli"
description: "inference.sh CLI로 150개 이상의 AI 앱(이미지, 동영상, LLM) 실행"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Inference Sh Cli

inference.sh CLI로 150개 이상의 AI 앱(이미지, 동영상, LLM)을 실행합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/devops/inference-sh-cli`로 설치 |
| 경로 | `optional-skills/devops/inference-sh-cli` |
| 버전 | `1.0.0` |
| 작성자 | okaris |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `AI`, `image-generation`, `video`, `LLM`, `search`, `inference`, `FLUX`, `Veo`, `Claude` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침이기도 합니다.
:::

# inference.sh CLI

간단한 CLI로 클라우드에서 150개 이상의 AI 앱을 실행합니다. GPU가 필요하지 않습니다.

모든 명령은 **터미널 도구**를 사용해 `infsh` 명령을 실행합니다.

## 사용 시점

- 사용자가 이미지를 생성해 달라고 요청할 때(FLUX, Reve, Seedream, Grok, Gemini image)
- 사용자가 동영상을 생성해 달라고 요청할 때(Veo, Wan, Seedance, OmniHuman)
- 사용자가 inference.sh 또는 infsh에 대해 질문할 때
- 개별 제공업체 API를 관리하지 않고 AI 앱을 실행하고 싶을 때
- 사용자가 AI 기반 검색을 요청할 때(Tavily, Exa)
- 아바타/립싱크 생성이 필요할 때

## 사전 요구 사항

`infsh` CLI가 설치되고 인증되어 있어야 합니다. 다음 명령으로 확인하세요.

```bash
infsh me
```

설치되어 있지 않다면 다음을 실행하세요.

```bash
curl -fsSL https://cli.inference.sh | sh
infsh login
```

전체 설정 내용은 `references/authentication.md`를 참조하세요.

## 워크플로

### 1. 항상 먼저 검색하기

앱 이름을 추측하지 마세요. 항상 검색해 올바른 앱 ID를 찾으세요.

```bash
infsh app list --search flux
infsh app list --search video
infsh app list --search image
```

### 2. 앱 실행하기

검색 결과의 정확한 앱 ID를 사용하세요. 기계가 읽을 수 있는 출력을 위해 항상 `--json`을 사용합니다.

```bash
infsh app run <app-id> --input '{"prompt": "your prompt here"}' --json
```

### 3. 출력 파싱하기

JSON 출력에는 생성된 미디어의 URL이 포함됩니다. 인라인 표시를 위해 `MEDIA:<url>` 형식으로 사용자에게 제시하세요.

## 일반적인 명령

### 이미지 생성

```bash
# Search for image apps
infsh app list --search image

# FLUX Dev with LoRA
infsh app run falai/flux-dev-lora --input '{"prompt": "sunset over mountains", "num_images": 1}' --json

# Gemini image generation
infsh app run google/gemini-2-5-flash-image --input '{"prompt": "futuristic city", "num_images": 1}' --json

# Seedream (ByteDance)
infsh app run bytedance/seedream-5-lite --input '{"prompt": "nature scene"}' --json

# Grok Imagine (xAI)
infsh app run xai/grok-imagine-image --input '{"prompt": "abstract art"}' --json
```

### 동영상 생성

```bash
# Search for video apps
infsh app list --search video

# Veo 3.1 (Google)
infsh app run google/veo-3-1-fast --input '{"prompt": "drone shot of coastline"}' --json

# Seedance (ByteDance)
infsh app run bytedance/seedance-1-5-pro --input '{"prompt": "dancing figure", "resolution": "1080p"}' --json

# Wan 2.5
infsh app run falai/wan-2-5 --input '{"prompt": "person walking through city"}' --json
```

### 로컬 파일 업로드

경로를 제공하면 CLI가 로컬 파일을 자동으로 업로드합니다.

```bash
# Upscale a local image
infsh app run falai/topaz-image-upscaler --input '{"image": "/path/to/photo.jpg", "upscale_factor": 2}' --json

# Image-to-video from local file
infsh app run falai/wan-2-5-i2v --input '{"image": "/path/to/image.png", "prompt": "make it move"}' --json

# Avatar with audio
infsh app run bytedance/omnihuman-1-5 --input '{"audio": "/path/to/audio.mp3", "image": "/path/to/face.jpg"}' --json
```

### 검색 및 리서치

```bash
infsh app list --search search
infsh app run tavily/tavily-search --input '{"query": "latest AI news"}' --json
infsh app run exa/exa-search --input '{"query": "machine learning papers"}' --json
```

### 기타 카테고리

```bash
# 3D generation
infsh app list --search 3d

# Audio / TTS
infsh app list --search tts

# Twitter/X automation
infsh app list --search twitter
```

## 주의할 점

1. **앱 ID를 절대 추측하지 마세요.** 먼저 항상 `infsh app list --search <term>`을 실행하세요. 앱 ID는 변경될 수 있고 새 앱이 자주 추가됩니다.
2. **항상 `--json`을 사용하세요.** 원시 출력은 파싱하기 어렵습니다. `--json` 플래그는 URL이 포함된 구조화된 출력을 제공합니다.
3. **인증을 확인하세요.** 명령이 인증 오류로 실패하면 `infsh login`을 실행하거나 `INFSH_API_KEY`가 설정되어 있는지 확인하세요.
4. **오래 실행되는 앱.** 동영상 생성에는 30~120초가 걸릴 수 있습니다. 터미널 도구의 시간 제한으로 충분하겠지만 잠시 시간이 걸릴 수 있다고 사용자에게 알려 주세요.
5. **입력 형식.** `--input` 플래그는 JSON 문자열을 받습니다. 따옴표를 올바르게 이스케이프해야 합니다.

## 참고 문서

- `references/authentication.md` — 설정, 로그인, API 키
- `references/app-discovery.md` — 앱 카탈로그 검색 및 탐색
- `references/running-apps.md` — 앱 실행, 입력 형식, 출력 처리
- `references/cli-reference.md` — CLI 전체 명령 참고 자료
