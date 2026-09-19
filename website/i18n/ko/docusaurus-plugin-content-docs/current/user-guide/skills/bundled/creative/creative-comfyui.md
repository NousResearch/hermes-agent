---
title: "Comfyui — diffusion 워크플로를 통한 이미지, 동영상, 오디오 생성"
sidebar_label: "Comfyui"
description: "diffusion 워크플로를 통한 이미지, 동영상, 오디오 생성"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Comfyui

diffusion 워크플로를 통해 이미지, 동영상, 오디오를 생성합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 기본 제공 (기본 설치) |
| 경로 | `skills/creative/comfyui` |
| 버전 | `5.1.0` |
| 작성자 | ['kshitijk4poor', 'alt-glitch', 'purzbeats'] |
| 라이선스 | MIT |
| 플랫폼 | macos, linux, windows |
| 태그 | `comfyui`, `image-generation`, `stable-diffusion`, `flux`, `sd3`, `wan-video`, `hunyuan-video`, `creative`, `generative-ai`, `video-generation` |
| 관련 스킬 | [`stable-diffusion`](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# ComfyUI

공식 `comfy-cli`를 사용한 설정/수명 주기 관리와 직접적인 REST/WebSocket API를 통해 ComfyUI에서 이미지, 동영상, 오디오, 3D 콘텐츠를 생성합니다.

## 이 스킬의 구성

**참고 문서(`references/`):**

- `official-cli.md` — 플래그를 포함한 모든 `comfy ...` 명령
- `rest-api.md` — REST + WebSocket 엔드포인트(로컬 + 클라우드)
- `workflow-format.md` — API 형식 JSON, 일반적인 노드 유형, 파라미터 매핑
- `template-integrity.md` — `comfyui-workflow-templates`를 편집기 형식에서 API 형식으로 변환: Reroute 우회, 점 표기 동적 입력 키(`values.a`, `resize_type.width`), Cloud 특이사항(302 리디렉션, 무료 요금제 동시 작업 1개, 1080p VRAM 제한), Discord 호환 ffmpeg 결합. [@purzbeats](https://github.com/purzbeats)가 작성했습니다. 공식 템플릿에서 시작할 때마다 이 문서를 로드하세요.

**스크립트(`scripts/`):**

| 스크립트 | 용도 |
|--------|---------|
| `_common.py` | 공유 HTTP, 클라우드 라우팅, 노드 카탈로그(직접 실행하지 마세요) |
| `hardware_check.py` | GPU/VRAM/디스크를 확인하고 로컬과 Comfy Cloud 중 권장 환경 제시 |
| `comfyui_setup.sh` | 하드웨어 확인 + comfy-cli + ComfyUI 설치 + 실행 + 검증 |
| `extract_schema.py` | 워크플로를 읽고 제어 가능한 파라미터와 모델 의존성 나열 |
| `check_deps.py` | 실행 중인 서버와 워크플로를 대조해 누락된 노드/모델 나열 |
| `auto_fix_deps.py` | check_deps를 실행한 뒤 `comfy node install` / `comfy model download` 실행 |
| `run_workflow.py` | 파라미터 주입, 제출, 모니터링, 출력 다운로드(HTTP 또는 WS) |
| `run_batch.py` | 워크플로를 N회 제출하고 스윕 실행, 요금제에 따라 최대 병렬 처리 |
| `ws_monitor.py` | 실행 중인 작업의 실시간 WebSocket 뷰어(실시간 진행률) |
| `health_check.py` | 검증 체크리스트 실행기 — comfy-cli + 서버 + 모델 + 스모크 테스트 |
| `fetch_logs.py` | 지정한 prompt_id의 트레이스백/상태 메시지 가져오기 |

**예시 워크플로(`workflows/`):** SD 1.5, SDXL, Flux Dev, SDXL img2img, SDXL 인페인트, ESRGAN 업스케일, AnimateDiff 동영상, Wan T2V. `workflows/README.md`를 참고하세요.

## 사용 시점

- 사용자가 Stable Diffusion, SDXL, Flux, SD3 등을 사용해 이미지 생성을 요청할 때
- 특정 ComfyUI 워크플로 파일을 실행하려 할 때
- 생성 단계를 연결하려 할 때(txt2img → 업스케일 → 얼굴 복원)
- ControlNet, 인페인팅, img2img 또는 기타 고급 파이프라인이 필요할 때
- ComfyUI 큐를 관리하거나, 모델을 확인하거나, 커스텀 노드를 설치하려 할 때
- AnimateDiff, Hunyuan, Wan, AudioCraft 등을 통해 동영상/오디오/3D 생성을 요청할 때

## 아키텍처: 두 계층

<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────┐
│ Layer 1: comfy-cli (official lifecycle tool)        │
│   Setup, server lifecycle, custom nodes, models     │
│   → comfy install / launch / stop / node / model    │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│ Layer 2: REST/WebSocket API + skill scripts         │
│   Workflow execution, param injection, monitoring   │
│   POST /api/prompt, GET /api/view, WS /ws           │
│   → run_workflow.py, run_batch.py, ws_monitor.py    │
└─────────────────────────────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

**왜 두 계층인가요?** 공식 CLI는 설치와 서버 관리에는 뛰어나지만 워크플로 실행 지원은 제한적입니다. REST/WS API가 그 공백을 메우며, 스크립트는 CLI가 제공하지 않는 파라미터 주입, 실행 모니터링, 출력 다운로드를 처리합니다.

## 빠른 시작

### 환경 감지

```bash
# What's available?
command -v comfy >/dev/null 2>&1 && echo "comfy-cli: installed"
curl -s http://127.0.0.1:8188/system_stats 2>/dev/null && echo "server: running"

# Can this machine run ComfyUI locally? (GPU/VRAM/disk check)
python3 scripts/hardware_check.py
```

아무것도 설치되어 있지 않다면 아래의 **설정 및 온보딩**을 참고하세요. 단, 항상 먼저 하드웨어 확인을 실행해야 합니다.

### 한 줄 상태 확인

```bash
python3 scripts/health_check.py
# → JSON: comfy_cli on PATH? server reachable? at least one checkpoint? smoke-test passes?
```

## 핵심 워크플로

### 1단계: API 형식의 워크플로 JSON 가져오기

워크플로는 API 형식이어야 합니다(각 노드에 `class_type`이 있어야 함). 출처는 다음과 같습니다.

- ComfyUI 웹 UI → **Workflow → Export (API)**(최신 UI) 또는 기존 UI의 레거시 **Save (API Format)** 버튼
- 이 스킬의 `workflows/` 디렉터리(바로 실행할 수 있는 예시)
- 커뮤니티 다운로드(civitai, Reddit, Discord) — 보통 편집기 형식이므로 ComfyUI에 로드한 뒤 다시 내보내야 함

편집기 형식(최상위 `nodes` 및 `links` 배열)은 **직접 실행할 수 없습니다**. 스크립트가 이를 감지하고 다시 내보내도록 안내합니다.

### 2단계: 제어 가능한 항목 확인

```bash
python3 scripts/extract_schema.py workflow_api.json --summary-only
# → {"parameter_count": 12, "has_negative_prompt": true, "has_seed": true, ...}

python3 scripts/extract_schema.py workflow_api.json
# → full schema with parameters, model deps, embedding refs
```

### 3단계: 파라미터와 함께 실행

```bash
# Local (defaults to http://127.0.0.1:8188)
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "a beautiful sunset over mountains", "seed": -1, "steps": 30}' \
  --output-dir ./outputs

# Cloud (export API key once; uses correct /api routing automatically)
export COMFY_CLOUD_API_KEY="comfyui-..."
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "..."}' \
  --host https://cloud.comfy.org \
  --output-dir ./outputs

# Real-time progress via WebSocket (requires `pip install websocket-client`)
python3 scripts/run_workflow.py \
  --workflow flux_dev.json \
  --args '{"prompt": "..."}' \
  --ws

# img2img / inpaint: pass --input-image to upload + reference automatically
python3 scripts/run_workflow.py \
  --workflow sdxl_img2img.json \
  --input-image image=./photo.png \
  --args '{"prompt": "make it watercolor", "denoise": 0.6}'

# Batch / sweep: 8 random seeds, parallel up to cloud tier limit
python3 scripts/run_batch.py \
  --workflow sdxl.json \
  --args '{"prompt": "abstract"}' \
  --count 8 --randomize-seed --parallel 3 \
  --output-dir ./outputs/batch
```

`seed`에 `-1`을 지정하거나 `--randomize-seed`와 함께 생략하면 실행마다 새로운 무작위 시드가 생성됩니다.

### 4단계: 결과 표시

스크립트는 모든 출력 파일을 설명하는 JSON을 stdout에 출력합니다.

```json
{
  "status": "success",
  "prompt_id": "abc-123",
  "outputs": [
    {"file": "./outputs/sdxl_00001_.png", "node_id": "9",
     "type": "image", "filename": "sdxl_00001_.png"}
  ]
}
```

## 결정 트리

| 사용자 발화 | 도구 | 명령 |
|-----------|------|---------|
| **수명 주기(comfy-cli 사용)** | | |
| "ComfyUI 설치" | comfy-cli | `bash scripts/comfyui_setup.sh` |
| "ComfyUI 시작" | comfy-cli | `comfy launch --background` |
| "ComfyUI 중지" | comfy-cli | `comfy stop` |
| "X 노드 설치" | comfy-cli | `comfy node install <name>` |
| "X 모델 다운로드" | comfy-cli | `comfy model download --url <url> --relative-path models/checkpoints` |
| "설치된 모델 나열" | comfy-cli | `comfy model list` |
| "설치된 노드 나열" | comfy-cli | `comfy node show installed` |
| **실행(스크립트 사용)** | | |
| "모든 준비가 되었나요?" | script | `health_check.py`(선택적으로 `--workflow X --smoke-test`와 함께 사용) |
| "이 워크플로에서 무엇을 바꿀 수 있나요?" | script | `extract_schema.py W.json` |
| "W의 의존성이 충족되었는지 확인" | script | `check_deps.py W.json` |
| "누락된 의존성 수정" | script | `auto_fix_deps.py W.json` |
| "이미지 생성" | script | `run_workflow.py --workflow W --args '{...}'` |
| "이 이미지 사용"(img2img) | script | `run_workflow.py --input-image image=./x.png ...` |
| "무작위 시드로 8개 변형" | script | `run_batch.py --count 8 --randomize-seed ...` |
| "실시간 진행률 표시" | script | `ws_monitor.py --prompt-id <id>` |
| "작업 X의 오류 가져오기" | script | `fetch_logs.py <prompt_id>` |
| **직접 REST** | | |
| "큐에 무엇이 있나요?" | REST | `curl http://HOST:8188/queue`(로컬) 또는 `--host https://cloud.comfy.org` |
| "취소해 주세요" | REST | `curl -X POST http://HOST:8188/interrupt` |
| "GPU 메모리 해제" | REST | `curl -X POST http://HOST:8188/free` |

## 설정 및 온보딩

사용자가 ComfyUI 설정을 요청하면 **가장 먼저 Comfy Cloud(호스팅, 설치 불필요, API 키)와 Local(자신의 컴퓨터에 ComfyUI 설치) 중 어느 쪽을 원하는지 물어야 합니다**. 대답하기 전에는 설치 명령이나 하드웨어 확인을 시작하지 마세요.

**공식 문서:** https://docs.comfy.org/installation
**CLI 문서:** https://docs.comfy.org/comfy-cli/getting-started
**Cloud 문서:** https://docs.comfy.org/get_started/cloud
**Cloud API:** https://docs.comfy.org/development/cloud/overview

### 0단계: Local과 Cloud 중 선택 요청(항상 먼저)

권장 스크립트:

> "ComfyUI를 컴퓨터에서 로컬로 실행할까요, 아니면 Comfy Cloud를 사용할까요?
>
> - **Comfy Cloud** — RTX 6000 Pro GPU에서 호스팅되며, 일반적인 모델이 모두 사전 설치되어 있고 설정이 필요 없습니다. API 키가 필요합니다(워크플로를 실제로 실행하려면 유료 구독이 필요하며, 무료 요금제는 읽기 전용입니다). 성능 좋은 GPU가 없다면 적합합니다.
> - **Local** — 무료이지만 컴퓨터가 반드시 다음 하드웨어 요구 사항을 충족해야 합니다.
>   - **6 GB 이상 VRAM**의 NVIDIA GPU(SDXL에는 **8 GB 이상**, Flux/동영상에는 **12 GB 이상**) 또는
>   - ROCm을 지원하는 AMD GPU(Linux) 또는
>   - **16 GB 이상 통합 메모리**(32 GB 권장)의 Apple Silicon Mac(M1 이상)
>   - Intel Mac 및 GPU가 없는 컴퓨터에서는 작동하지 않습니다 — 대신 Cloud를 사용하세요.
>
> 어느 쪽을 선택하시겠어요?"

라우팅:

- **Cloud** → **경로 A**로 건너뜁니다.
- **Local** → 먼저 하드웨어를 확인한 뒤 결과에 따라 경로 B–E 중 하나를 선택합니다.
- **잘 모르겠음** → 하드웨어를 확인하고 결과에 따라 결정합니다.

### 1단계: 하드웨어 확인(사용자가 local을 선택한 경우에만)

```bash
python3 scripts/hardware_check.py --json
# Optional: also probe `torch` for actual CUDA/MPS:
python3 scripts/hardware_check.py --json --check-pytorch
```

| 판정 | 의미 | 조치 |
|------------|---------------------------------------------------------------|--------|
| `ok` | 8 GB 이상 VRAM(외장) 또는 32 GB 이상 통합 메모리(Apple Silicon) | 로컬 설치 — 보고서의 `comfy_cli_flag` 사용 |
| `marginal` | SD1.5는 작동하지만 SDXL은 빠듯하고 Flux/동영상은 어려움 | 가벼운 워크플로에는 로컬 사용, 그 외에는 **경로 A(Cloud)** |
| `cloud` | 사용 가능한 GPU 없음, VRAM 6 GB 미만, Apple 통합 메모리 16 GB 미만, Intel Mac, Rosetta Python | 사용자가 명시적으로 local을 강제하지 않는 한 **Cloud로 전환** |

스크립트는 `wsl: true`(NVIDIA 패스스루가 활성화된 WSL2)와 `rosetta: true`(Apple Silicon에서 실행되는 x86_64 Python — ARM64로 다시 설치해야 함)도 표시합니다.

판정이 `cloud`이지만 사용자가 local을 원한다면 조용히 진행하지 마세요. `notes` 배열을 그대로 보여 주고 (a) Cloud로 전환하거나 (b) 로컬 설치를 강제할지 물어보세요(최신 모델에서는 OOM이 발생하거나 사용할 수 없을 정도로 느려집니다).

### 설치 경로 선택

먼저 하드웨어를 확인하세요. 다음 표는 사용자가 이미 하드웨어를 알려 준 경우의 대안입니다.

| 상황 | 권장 경로 |
|-----------|------------------|
| 하드웨어 확인 결과 `verdict: cloud` | **경로 A: Comfy Cloud** |
| GPU가 없거나 부담 없이 시험하려는 경우 | **경로 A: Comfy Cloud** |
| Windows + NVIDIA + 비기술 사용자 | **경로 B: ComfyUI Desktop** |
| Windows + NVIDIA + 기술 사용자 | **경로 C: Portable** 또는 **경로 D: comfy-cli** |
| Linux + 모든 GPU | **경로 D: comfy-cli**(가장 쉬움) |
| macOS + Apple Silicon | **경로 B: Desktop** 또는 **경로 D: comfy-cli** |
| 헤드리스 / 서버 / CI / 에이전트 | **경로 D: comfy-cli** |

완전 자동화 경로(하드웨어 확인 → 설치 → 실행 → 검증)는 다음과 같습니다.

```bash
bash scripts/comfyui_setup.sh
# Or with overrides:
bash scripts/comfyui_setup.sh --m-series --port=8190 --workspace=/data/comfy
```

이 스크립트는 내부적으로 `hardware_check.py`를 실행하고, 판정이 `cloud`일 때 로컬 설치를 거부하며(`--force-cloud-override`를 사용한 경우 제외), 올바른 `comfy-cli` 플래그를 선택하고, 시스템 Python 오염을 피하기 위해 전역 `pip`보다 `pipx`/`uvx`를 우선합니다.

---

### 경로 A: Comfy Cloud(로컬 설치 없음)

성능 좋은 GPU가 없거나 설정을 전혀 하고 싶지 않은 사용자를 위한 경로입니다. RTX 6000 Pro에서 호스팅됩니다.

**문서:** https://docs.comfy.org/get_started/cloud

1. https://comfy.org/cloud 에 가입합니다.
2. https://platform.comfy.org/login 에서 API 키를 생성합니다.
3. 키를 설정합니다.
   ```bash
   export COMFY_CLOUD_API_KEY="your-comfyui-key"
   ```
4. 워크플로를 실행합니다.
   ```bash
   python3 scripts/run_workflow.py \
     --workflow workflows/flux_dev_txt2img.json \
     --args '{"prompt": "..."}' \
     --host https://cloud.comfy.org \
     --output-dir ./outputs
   ```

**가격:** https://www.comfy.org/cloud/pricing
**동시 작업:** Free/Standard 1개, Creator 3개, Pro 5개. Free 요금제에서는 **API를 통한 워크플로 실행이 불가능**하며 모델 탐색만 할 수 있습니다. `/api/prompt`, `/api/upload/*`, `/api/view` 등을 사용하려면 유료 구독이 필요합니다.

---

### 경로 B: ComfyUI Desktop(Windows / macOS)

기술 지식이 없는 사용자를 위한 원클릭 설치 프로그램입니다. 현재 베타 버전입니다.

**문서:** https://docs.comfy.org/installation/desktop
- **Windows (NVIDIA):** https://download.comfy.org/windows/nsis/x64
- **macOS (Apple Silicon):** https://comfy.org

Linux에서는 **Desktop을 지원하지 않습니다** — 경로 D를 사용하세요.

---

### 경로 C: ComfyUI Portable(Windows 전용)

**문서:** https://docs.comfy.org/installation/comfyui_portable_windows

https://github.com/comfyanonymous/ComfyUI/releases에서 다운로드하고 압축을 푼 뒤 `run_nvidia_gpu.bat`를 실행합니다. `update/update_comfyui_stable.bat`를 사용해 업데이트합니다.

---

### 경로 D: comfy-cli(모든 플랫폼 — 에이전트에 권장)

공식 CLI는 헤드리스/자동화 설정에 가장 적합한 경로입니다.

**문서:** https://docs.comfy.org/comfy-cli/getting-started

#### comfy-cli 설치

```bash
# Recommended:
pipx install comfy-cli
# Or use uvx without installing:
uvx --from comfy-cli comfy --help
# Or (if pipx/uvx unavailable):
pip install --user comfy-cli
```

대화형이 아닌 방식으로 분석을 비활성화합니다.
```bash
comfy --skip-prompt tracking disable
```

#### ComfyUI 설치

```bash
comfy --skip-prompt install --nvidia              # NVIDIA (CUDA)
comfy --skip-prompt install --amd                 # AMD (ROCm, Linux)
comfy --skip-prompt install --m-series            # Apple Silicon (MPS)
comfy --skip-prompt install --cpu                 # CPU only (slow)
comfy --skip-prompt install --nvidia --fast-deps  # uv-based dep resolution
```

기본 위치: `~/comfy/ComfyUI`(Linux), `~/Documents/comfy/ComfyUI`(macOS/Win). `comfy --workspace /custom/path install`으로 재정의할 수 있습니다.

#### 실행 / 확인

```bash
comfy launch --background                       # background daemon on :8188
comfy launch -- --listen 0.0.0.0 --port 8190    # LAN-accessible custom port
curl -s http://127.0.0.1:8188/system_stats      # health check
```

---

### 경로 E: 수동 설치(고급 / 지원되지 않는 하드웨어)

Ascend NPU, Cambricon MLU, Intel Arc 또는 지원되지 않는 기타 하드웨어를 위한 경로입니다.

**문서:** https://docs.comfy.org/installation/manual_install

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python main.py
```

---

### 설치 후: 모델 다운로드

```bash
# SDXL (general purpose, ~6.5 GB)
comfy model download \
  --url "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
  --relative-path models/checkpoints

# SD 1.5 (lighter, ~4 GB, good for 6 GB cards)
comfy model download \
  --url "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" \
  --relative-path models/checkpoints

# Flux Dev fp8 (smaller variant, ~12 GB)
comfy model download \
  --url "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
  --relative-path models/checkpoints

# CivitAI (set token first):
comfy model download \
  --url "https://civitai.com/api/download/models/128713" \
  --relative-path models/checkpoints \
  --set-civitai-api-token "YOUR_TOKEN"
```

설치된 모델 나열: `comfy model list`.

### 설치 후: 커스텀 노드 설치

```bash
comfy node install comfyui-impact-pack             # popular utility pack
comfy node install comfyui-animatediff-evolved     # video generation
comfy node install comfyui-controlnet-aux          # ControlNet preprocessors
comfy node install comfyui-essentials              # common helpers
comfy node update all
comfy node install-deps --workflow=workflow.json   # install everything a workflow needs
```

### 설치 후: 검증

```bash
python3 scripts/health_check.py
# → comfy_cli on PATH? server reachable? checkpoints? smoke test?

python3 scripts/check_deps.py my_workflow.json
# → are this workflow's nodes/models/embeddings installed?

python3 scripts/run_workflow.py \
  --workflow workflows/sd15_txt2img.json \
  --args '{"prompt": "test", "steps": 4}' \
  --output-dir ./test-outputs
```

## 이미지 업로드(img2img / 인페인팅)

가장 간단한 방법은 `run_workflow.py`와 함께 `--input-image`를 사용하는 것입니다.

```bash
python3 scripts/run_workflow.py \
  --workflow workflows/sdxl_img2img.json \
  --input-image image=./photo.png \
  --args '{"prompt": "make it cyberpunk", "denoise": 0.6}'
```

이 플래그는 `photo.png`를 업로드한 다음, 이름이 `image`인 스키마 파라미터에 서버 측 파일 이름을 주입합니다. 인페인팅에는 다음처럼 둘 다 전달합니다.

```bash
python3 scripts/run_workflow.py \
  --workflow workflows/sdxl_inpaint.json \
  --input-image image=./photo.png \
  --input-image mask_image=./mask.png \
  --args '{"prompt": "fill with flowers"}'
```

REST를 통한 수동 업로드:
```bash
curl -X POST "http://127.0.0.1:8188/upload/image" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
# Returns: {"name": "photo.png", "subfolder": "", "type": "input"}

# Cloud equivalent:
curl -X POST "https://cloud.comfy.org/api/upload/image" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
```

## Cloud 특이사항

- **기본 URL:** `https://cloud.comfy.org`
- **인증:** `X-API-Key` 헤더(WebSocket의 경우 `?token=KEY`도 가능)
- **API 키:** `$COMFY_CLOUD_API_KEY`를 한 번 설정하면 스크립트가 자동으로 사용합니다.
- **출력 다운로드:** `/api/view`는 서명된 URL로 302를 반환합니다. 스크립트는 이를 따라가며 스토리지 백엔드에서 가져오기 전에 `X-API-Key`를 제거합니다(S3/CloudFront에 API 키를 노출하지 않음).
- **로컬 ComfyUI와의 엔드포인트 차이:**
  - `/api/object_info`, `/api/queue`, `/api/userdata` — **무료 요금제에서는 403**; 유료 전용
  - Cloud에서는 `/history`가 `/history_v2`로 이름이 변경됨(스크립트가 자동 라우팅)
  - Cloud에서는 `/models/<folder>`가 `/experiment/models/<folder>`로 이름이 변경됨(스크립트가 자동 라우팅)
  - WebSocket의 `clientId`는 현재 무시됨 — 사용자의 모든 연결이 동일한 브로드캐스트를 받음. 클라이언트 측에서 `prompt_id`로 필터링하세요.
  - 업로드에서 `subfolder`는 허용되지만 무시됨 — Cloud는 평면 네임스페이스를 사용함
- **동시 작업:** Free/Standard 1개, Creator 3개, Pro 5개. 초과 작업은 자동으로 큐에 들어갑니다. `run_batch.py --parallel N`을 사용해 요금제 한도를 채우세요.

## 큐 및 시스템 관리

```bash
# Local
curl -s http://127.0.0.1:8188/queue | python3 -m json.tool
curl -X POST http://127.0.0.1:8188/queue -d '{"clear": true}'    # cancel pending
curl -X POST http://127.0.0.1:8188/interrupt                      # cancel running
curl -X POST http://127.0.0.1:8188/free \
  -H "Content-Type: application/json" \
  -d '{"unload_models": true, "free_memory": true}'

# Cloud — same paths under /api/, plus:
python3 scripts/fetch_logs.py --tail-queue --host https://cloud.comfy.org
```

## 주의사항

1. **API 형식 필수** — 모든 스크립트와 `/api/prompt` 엔드포인트는 API 형식 워크플로 JSON을 요구합니다. 스크립트는 편집기 형식(최상위 `nodes` 및 `links` 배열)을 감지하고 최신 UI에서는 "Workflow → Export (API)", 이전 UI에서는 "Save (API Format)"을 통해 다시 내보내라고 안내합니다.
2. **서버가 실행 중이어야 함** — 모든 실행에는 실행 중인 서버가 필요합니다. `comfy launch --background`로 시작하고 `curl http://127.0.0.1:8188/system_stats`로 확인하세요.
3. **모델 이름은 정확해야 함** — 대소문자를 구분하며 파일 확장자도 포함합니다. `check_deps.py`는 확장자 및 폴더 접두사의 유무를 고려해 퍼지 매칭하지만, 워크플로 자체는 표준 이름을 사용해야 합니다. `comfy model list`로 설치된 항목을 확인하세요.
4. **커스텀 노드 누락** — "class_type not found"는 필요한 노드가 설치되지 않았다는 뜻입니다. `check_deps.py`가 설치할 패키지를 보고하고, `auto_fix_deps.py`가 대신 설치를 실행합니다.
5. **작업 디렉터리** — `comfy-cli`가 ComfyUI 작업 공간을 자동 감지합니다. "no workspace found" 오류가 나면 `comfy --workspace /path/to/ComfyUI <command>` 또는 `comfy set-default /path/to/ComfyUI`를 사용하세요.
6. **Cloud 무료 요금제 API 제한** — 무료 계정에서는 `/api/prompt`, `/api/view`, `/api/upload/*`, `/api/object_info`가 모두 403을 반환합니다. `health_check.py`와 `check_deps.py`가 이를 정상적으로 처리하고 명확한 메시지를 표시합니다.
7. **동영상/오디오 워크플로의 타임아웃** — 출력 노드가 `VHS_VideoCombine`, `SaveVideo` 등인 경우 자동 감지되며 기본값이 300초에서 900초로 늘어납니다. `--timeout 1800`으로 명시적으로 덮어쓸 수 있습니다.
8. **출력 파일 이름의 경로 순회** — 서버가 제공한 파일 이름은 `safe_path_join`을 거쳐 `--output-dir` 밖으로 나가는 경로를 거부합니다. 커스텀 저장 노드를 사용하는 워크플로는 임의의 경로를 만들 수 있으므로 이 보호 기능을 유지하세요.
9. **워크플로 JSON은 임의의 코드** — 커스텀 노드는 Python을 실행하므로 알 수 없는 워크플로를 제출하는 것은 `eval`과 같은 신뢰 수준을 요구합니다. 신뢰할 수 없는 출처의 워크플로는 실행 전에 검사하세요.
10. **시드 자동 무작위화** — `--args`에 `seed: -1`을 전달하거나 `--randomize-seed`를 사용하고 시드를 생략하면 실행마다 새로운 시드를 얻습니다. 실제 시드는 stderr에 기록됩니다.
11. **`tracking` 프롬프트** — `comfy`의 첫 실행에서 분석 수집을 묻는 메시지가 표시될 수 있습니다. 대화형이 아닌 방식으로 건너뛰려면 `comfy --skip-prompt tracking disable`을 사용하세요. `comfyui_setup.sh`가 대신 처리합니다.

## 검증 체크리스트

`python3 scripts/health_check.py`를 사용하면 전체 목록을 한 번에 실행할 수 있습니다. 수동 확인:

- [ ] `hardware_check.py` 판정이 `ok`이거나 사용자가 명시적으로 Comfy Cloud를 선택함
- [ ] `comfy --version`이 작동함(또는 `uvx --from comfy-cli comfy --help`)
- [ ] `curl http://HOST:PORT/system_stats`가 JSON을 반환함
- [ ] `comfy model list`에 체크포인트가 하나 이상 표시됨(로컬) 또는 `/api/experiment/models/checkpoints`가 모델을 반환함(Cloud)
- [ ] 워크플로 JSON이 API 형식임
- [ ] `check_deps.py`가 `is_ready: true`를 보고함(또는 Cloud 무료 요금제에서만 `node_check_skipped`)
- [ ] 작은 워크플로를 사용한 테스트 실행이 완료되고 출력이 `--output-dir`에 저장됨
