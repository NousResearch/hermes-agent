---
title: "ComfyUI"
sidebar_label: "ComfyUI"
description: "ComfyUI를 사용하여 이미지, 비디오, 오디오 생성 — 노드/모델의 설치, 실행, 관리, 파라미터 주입을 통한 워크플로우 실행"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# ComfyUI

ComfyUI를 사용하여 이미지, 비디오, 오디오 생성 — 노드/모델의 설치, 실행, 관리, 파라미터 주입을 통한 워크플로우 실행. 수명 주기(lifecycle) 관리를 위한 공식 comfy-cli와 실행을 위한 직접적인 REST/WebSocket API를 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/creative/comfyui` |
| 버전 | `5.1.0` |
| 저자 | ['kshitijk4poor', 'alt-glitch', 'purzbeats'] |
| 라이선스 | MIT |
| 플랫폼 | macos, linux, windows |
| 태그 | `comfyui`, `image-generation`, `stable-diffusion`, `flux`, `sd3`, `wan-video`, `hunyuan-video`, `creative`, `generative-ai`, `video-generation` |
| 관련 스킬 | [`stable-diffusion-image-generation`](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion), `image_gen` |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬을 트리거할 때 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 명령으로 보는 내용입니다.
:::

# ComfyUI

설정/수명 주기를 위한 공식 `comfy-cli`와 워크플로우 실행을 위한 직접적인 REST/WebSocket API를 사용하여 ComfyUI를 통해 이미지, 비디오, 오디오, 3D 콘텐츠를 생성합니다.

## 이 스킬에 포함된 내용

**참조 문서 (`references/`):**

- `official-cli.md` — 플래그가 포함된 모든 `comfy ...` 명령어
- `rest-api.md` — REST + WebSocket 엔드포인트(로컬 + 클라우드), 페이로드 스키마
- `workflow-format.md` — API 형식 JSON, 일반적인 노드 유형, 파라미터 매핑
- `template-integrity.md` — `comfyui-workflow-templates`를 에디터 형식에서 API 형식으로 변환: 우회(Bypass) 경로 재지정, 점으로 구분된 동적 입력 키(`values.a`, `resize_type.width`), 클라우드 특이사항(302 리디렉트, 동시 무료 티어 작업 1개, 1080p VRAM 한도), Discord 호환 ffmpeg 병합. [@purzbeats](https://github.com/purzbeats) 작성. 공식 템플릿으로 시작할 때마다 이 문서를 로드하세요.

**스크립트 (`scripts/`):**

| 스크립트 | 목적 |
|--------|---------|
| `_common.py` | 공유 HTTP, 클라우드 라우팅, 노드 카탈로그 (직접 실행하지 마세요) |
| `hardware_check.py` | GPU/VRAM/디스크를 확인하여 로컬 vs Comfy Cloud를 권장 |
| `comfyui_setup.sh` | 하드웨어 확인 + comfy-cli + ComfyUI 설치 + 실행 + 검증 |
| `extract_schema.py` | 워크플로우를 읽어 제어 가능한 파라미터 + 모델 종속성 나열 |
| `check_deps.py` | 실행 중인 서버에 대해 워크플로우를 확인 → 누락된 노드/모델 나열 |
| `auto_fix_deps.py` | check_deps 실행 후 `comfy node install` / `comfy model download` 수행 |
| `run_workflow.py` | 파라미터 주입, 제출, 모니터링, 출력물 다운로드 (HTTP 또는 WS) |
| `run_batch.py` | 매개변수를 다양하게 바꿔가며 워크플로우를 N번 제출, 티어에 따라 병렬 처리 |
| `ws_monitor.py` | 실행 중인 작업을 위한 실시간 WebSocket 뷰어 (실시간 진행 상황) |
| `health_check.py` | 검증 체크리스트 실행기 — comfy-cli + 서버 + 모델 + 스모크 테스트 |
| `fetch_logs.py` | 지정된 prompt_id에 대한 트레이스백 / 상태 메시지 가져오기 |

**예제 워크플로우 (`workflows/`):** SD 1.5, SDXL, Flux Dev, SDXL img2img, SDXL inpaint, ESRGAN upscale, AnimateDiff video, Wan T2V. `workflows/README.md`를 참조하세요.

## 이 스킬을 사용하는 시기

- 사용자가 Stable Diffusion, SDXL, Flux, SD3 등으로 이미지 생성을 요청할 때
- 사용자가 특정 ComfyUI 워크플로우 파일을 실행하려고 할 때
- 사용자가 생성 단계(txt2img → upscale → face restore)를 연결하려고 할 때
- 사용자가 ControlNet, inpainting, img2img 또는 기타 고급 파이프라인이 필요할 때
- 사용자가 ComfyUI 대기열(queue) 관리, 모델 확인 또는 사용자 정의 노드 설치를 요청할 때
- 사용자가 AnimateDiff, Hunyuan, Wan, AudioCraft 등을 통한 비디오/오디오/3D 생성을 원할 때

## 아키텍처: 두 개의 계층

<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────┐
│ 계층 1: comfy-cli (공식 수명 주기 도구)                  │
│   설정, 서버 수명 주기, 사용자 정의 노드, 모델 관리          │
│   → comfy install / launch / stop / node / model    │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│ 계층 2: REST/WebSocket API + 스킬 스크립트              │
│   워크플로우 실행, 파라미터 주입, 모니터링               │
│   POST /api/prompt, GET /api/view, WS /ws           │
│   → run_workflow.py, run_batch.py, ws_monitor.py    │
└─────────────────────────────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

**왜 두 개의 계층인가?** 공식 CLI는 설치 및 서버 관리에 탁월하지만 워크플로우 실행 지원은 미미합니다. REST/WS API가 그 격차를 채워줍니다 — 스크립트는 CLI가 수행하지 않는 파라미터 주입, 실행 모니터링 및 출력 다운로드를 처리합니다.

## 빠른 시작

### 환경 감지

```bash
# 사용 가능한 것은?
command -v comfy >/dev/null 2>&1 && echo "comfy-cli: installed"
curl -s http://127.0.0.1:8188/system_stats 2>/dev/null && echo "server: running"

# 이 기기에서 로컬로 ComfyUI를 실행할 수 있습니까? (GPU/VRAM/디스크 확인)
python3 scripts/hardware_check.py
```

설치된 것이 없으면 아래의 **설정 및 온보딩(Setup & Onboarding)**을 참조하되, 항상 하드웨어 점검을 먼저 실행하세요.

### 한 줄 상태 확인 (Health check)

```bash
python3 scripts/health_check.py
# → JSON: comfy_cli가 PATH에 있는지? 서버에 접근 가능한지? 체크포인트가 하나 이상 있는지? 스모크 테스트를 통과하는지?
```

## 핵심 워크플로우

### 1단계: API 형식의 워크플로우 JSON 가져오기

워크플로우는 반드시 API 형식이어야 합니다 (각 노드에는 `class_type`이 있어야 함). 이는 다음에서 가져올 수 있습니다:

- ComfyUI 웹 UI → **Workflow → Export (API)** (새로운 UI) 또는 기존 "Save (API Format)" 버튼 (이전 UI)
- 이 스킬의 `workflows/` 디렉토리 (바로 실행 가능한 예제들)
- 커뮤니티 다운로드 (civitai, Reddit, Discord) — 주로 에디터 형식이므로 ComfyUI에 로드한 후 다시 내보내야 합니다.

에디터 형식(최상위 수준에 `nodes` 및 `links` 배열이 있음)은 **직접 실행할 수 없습니다**. 스크립트는 이를 감지하고 다시 내보내도록 알려줍니다.

### 2단계: 제어 가능한 파라미터 확인

```bash
python3 scripts/extract_schema.py workflow_api.json --summary-only
# → {"parameter_count": 12, "has_negative_prompt": true, "has_seed": true, ...}

python3 scripts/extract_schema.py workflow_api.json
# → 파라미터, 모델 종속성, 임베딩 참조가 포함된 전체 스키마
```

### 3단계: 파라미터와 함께 실행

```bash
# 로컬 (기본값 http://127.0.0.1:8188)
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "a beautiful sunset over mountains", "seed": -1, "steps": 30}' \
  --output-dir ./outputs

# 클라우드 (API 키를 한 번 내보냅니다; 자동으로 올바른 /api 라우팅 사용)
export COMFY_CLOUD_API_KEY="comfyui-..."
python3 scripts/run_workflow.py \
  --workflow workflow_api.json \
  --args '{"prompt": "..."}' \
  --host https://cloud.comfy.org \
  --output-dir ./outputs

# WebSocket을 통한 실시간 진행 상황 (`pip install websocket-client` 필요)
python3 scripts/run_workflow.py \
  --workflow flux_dev.json \
  --args '{"prompt": "..."}' \
  --ws

# img2img / inpaint: --input-image를 전달하여 자동으로 업로드 + 참조
python3 scripts/run_workflow.py \
  --workflow sdxl_img2img.json \
  --input-image image=./photo.png \
  --args '{"prompt": "make it watercolor", "denoise": 0.6}'

# 배치 / 스윕: 무작위 시드로 8개 생성, 클라우드 티어 한도까지 병렬 실행
python3 scripts/run_batch.py \
  --workflow sdxl.json \
  --args '{"prompt": "abstract"}' \
  --count 8 --randomize-seed --parallel 3 \
  --output-dir ./outputs/batch
```

`seed`에 `-1`을 지정하거나 `--randomize-seed`와 함께 생략하면 실행마다 새로운 무작위 시드가 생성됩니다.

### 4단계: 결과 제시

스크립트는 모든 출력 파일을 설명하는 JSON을 stdout으로 내보냅니다:

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

## 의사결정 트리 (Decision Tree)

| 사용자의 요청 | 도구 | 명령어 |
|-----------|------|---------|
| **수명 주기 (comfy-cli 사용)** | | |
| "ComfyUI 설치해줘" | comfy-cli | `bash scripts/comfyui_setup.sh` |
| "ComfyUI 시작해줘" | comfy-cli | `comfy launch --background` |
| "ComfyUI 멈춰줘" | comfy-cli | `comfy stop` |
| "X 노드 설치해줘" | comfy-cli | `comfy node install <name>` |
| "X 모델 다운로드해줘" | comfy-cli | `comfy model download --url <url> --relative-path models/checkpoints` |
| "설치된 모델 목록 보여줘" | comfy-cli | `comfy model list` |
| "설치된 노드 목록 보여줘" | comfy-cli | `comfy node show installed` |
| **실행 (스크립트 사용)** | | |
| "모든 게 준비되었어?" | script | `health_check.py` (선택 사항 `--workflow X --smoke-test`) |
| "이 워크플로우에서 내가 뭘 바꿀 수 있어?" | script | `extract_schema.py W.json` |
| "W의 종속성이 충족되었는지 확인해줘" | script | `check_deps.py W.json` |
| "누락된 종속성 수정해줘" | script | `auto_fix_deps.py W.json` |
| "이미지 생성해줘" | script | `run_workflow.py --workflow W --args '{...}'` |
| "이 이미지를 사용해줘" (img2img) | script | `run_workflow.py --input-image image=./x.png ...` |
| "무작위 시드로 8가지 변형 만들어줘" | script | `run_batch.py --count 8 --randomize-seed ...` |
| "실시간 진행 상황 보여줘" | script | `ws_monitor.py --prompt-id <id>` |
| "작업 X에서 오류 가져와줘" | script | `fetch_logs.py <prompt_id>` |
| **직접 REST 통신** | | |
| "대기열(queue)에 뭐가 있어?" | REST | `curl http://HOST:8188/queue` (로컬) 또는 `--host https://cloud.comfy.org` |
| "그거 취소해줘" | REST | `curl -X POST http://HOST:8188/interrupt` |
| "GPU 메모리 여유 공간 확보해줘" | REST | `curl -X POST http://HOST:8188/free` |

## 설정 및 온보딩 (Setup & Onboarding)

사용자가 ComfyUI 설정을 요청할 때, **가장 먼저 해야 할 일은 Comfy Cloud(호스팅, 무설치, API 키 필요)를 원하는지 아니면 로컬(자신의 컴퓨터에 ComfyUI 설치)을 원하는지 묻는 것입니다**. 대답을 듣기 전에는 설치 명령이나 하드웨어 검사를 시작하지 마세요.

**공식 문서:** https://docs.comfy.org/installation
**CLI 문서:** https://docs.comfy.org/comfy-cli/getting-started
**클라우드 문서:** https://docs.comfy.org/get_started/cloud
**클라우드 API:** https://docs.comfy.org/development/cloud/overview

### 0단계: 로컬 vs 클라우드 묻기 (항상 가장 먼저)

제안하는 스크립트:

> "ComfyUI를 컴퓨터에 로컬로 실행하시겠습니까, 아니면 Comfy Cloud를 사용하시겠습니까?
>
> - **Comfy Cloud** — RTX 6000 Pro GPU에서 호스팅되며, 모든 일반 모델이 사전 설치되어 있고 설정이 필요 없습니다. API 키가 필요합니다(워크플로우를 실제로 실행하려면 유료 구독이 필요하며, 무료 티어는 읽기 전용입니다). 성능 좋은 GPU가 없을 때 최적입니다.
> - **로컬 (Local)** — 무료이지만 컴퓨터가 다음 하드웨어 요구 사항을 충족해야 합니다:
>   - **≥6 GB VRAM** 이상의 NVIDIA GPU (SDXL은 ≥8 GB, Flux/비디오는 ≥12 GB), 또는
>   - ROCm이 지원되는 AMD GPU (Linux), 또는
>   - **≥16 GB 통합 메모리** (≥32 GB 권장) 이상의 Apple Silicon Mac (M1+).
>   - Intel Mac 및 GPU가 없는 컴퓨터에서는 작동하지 않습니다 — 대신 Cloud를 사용하세요.
>
> 어느 쪽을 원하시나요?"

라우팅:

- **클라우드** → **경로 A(Path A)**로 건너뜁니다.
- **로컬** → 하드웨어 검사를 먼저 실행한 다음 판정에 따라 경로 B~E에서 선택합니다.
- **확실하지 않음** → 하드웨어 검사를 실행하고 그 판정에 따라 결정합니다.

### 1단계: 하드웨어 검증 (사용자가 로컬을 선택한 경우에만)

```bash
python3 scripts/hardware_check.py --json
# 선택 사항: 실제 CUDA/MPS 확인을 위해 `torch` 조사
python3 scripts/hardware_check.py --json --check-pytorch
```

| 판정 (Verdict) | 의미                                                          | 조치 |
|------------|---------------------------------------------------------------|--------|
| `ok`       | ≥8 GB VRAM (외장) 또는 ≥32 GB 통합 (Apple Silicon)           | 로컬 설치 — 보고서의 `comfy_cli_flag` 사용 |
| `marginal` | SD1.5 작동; SDXL 빠듯함; Flux/비디오 어려움                      | 가벼운 워크플로우엔 로컬 허용, 그 외엔 **경로 A (Cloud)** |
| `cloud`    | 사용 가능한 GPU 없음, &lt;6 GB VRAM, &lt;16 GB Apple 통합, Intel Mac, Rosetta Python | 사용자가 명시적으로 로컬을 강제하지 않는 한 **Cloud로 전환** |

이 스크립트는 `wsl: true` (NVIDIA 패스스루가 있는 WSL2) 및 `rosetta: true` (Apple Silicon의 x86_64 Python — ARM64로 재설치해야 함)도 감지합니다.

판정이 `cloud`이지만 사용자가 로컬을 원할 경우 조용히 진행하지 마세요.
`notes` 배열을 그대로 보여주고 (a) 클라우드로 전환할 것인지 아니면 (b) 로컬 설치를 강행할 것인지 물어보세요(메모리 부족(OOM) 오류가 발생하거나 최신 모델에서 사용할 수 없을 정도로 느려집니다).

### 설치 경로 선택

하드웨어 점검을 먼저 사용하세요. 아래 표는 사용자가 이미 하드웨어 정보를 제공했을 때를 위한 대체 방법입니다.

| 상황 | 권장 경로 |
|-----------|------------------|
| 하드웨어 점검 결과 `verdict: cloud` | **경로 A: Comfy Cloud** |
| GPU 없음 / 비용 지불 없이 써보고 싶음 | **경로 A: Comfy Cloud** |
| Windows + NVIDIA + 비기술적 | **경로 B: ComfyUI Desktop** |
| Windows + NVIDIA + 기술적 | **경로 C: Portable** 또는 **경로 D: comfy-cli** |
| Linux + 모든 호환 GPU | **경로 D: comfy-cli** (가장 쉬움) |
| macOS + Apple Silicon | **경로 B: Desktop** 또는 **경로 D: comfy-cli** |
| 헤드리스 / 서버 / CI / 에이전트 | **경로 D: comfy-cli** |

완전히 자동화된 경로(하드웨어 검사 → 설치 → 실행 → 검증)의 경우:

```bash
bash scripts/comfyui_setup.sh
# 또는 옵션 덮어쓰기:
bash scripts/comfyui_setup.sh --m-series --port=8190 --workspace=/data/comfy
```

내부적으로 `hardware_check.py`를 실행하며, 판정이 `cloud`일 경우 로컬 설치를 거부합니다(`--force-cloud-override`를 사용하지 않는 한). 올바른 `comfy-cli` 플래그를 선택하고, 시스템 Python이 오염되는 것을 피하기 위해 전역 `pip`보다 `pipx`/`uvx`를 선호합니다.

---

### 경로 A: Comfy Cloud (로컬 설치 없음)

성능이 좋은 GPU가 없거나 별도의 설정을 원하지 않는 사용자를 위한 경로입니다. RTX 6000 Pro에서 호스팅됩니다.

**문서:** https://docs.comfy.org/get_started/cloud

1. https://comfy.org/cloud 에서 가입
2. https://platform.comfy.org/login 에서 API 키 생성
3. 키 설정:
   ```bash
   export COMFY_CLOUD_API_KEY="comfyui-xxxxxxxxxxxx"
   ```
4. 워크플로우 실행:
   ```bash
   python3 scripts/run_workflow.py \
     --workflow workflows/flux_dev_txt2img.json \
     --args '{"prompt": "..."}' \
     --host https://cloud.comfy.org \
     --output-dir ./outputs
   ```

**가격:** https://www.comfy.org/cloud/pricing
**동시 작업 수:** Free/Standard 1, Creator 3, Pro 5. 무료(Free) 티어는 **API를 통한 워크플로우 실행이 불가능**하며, 모델 열람만 가능합니다. `/api/prompt`, `/api/upload/*`, `/api/view` 등을 사용하려면 유료 구독이 필요합니다.

---

### 경로 B: ComfyUI Desktop (Windows / macOS)

비기술적인 사용자를 위한 원클릭 설치 프로그램입니다. 현재 베타 버전입니다.

**문서:** https://docs.comfy.org/installation/desktop
- **Windows (NVIDIA):** https://download.comfy.org/windows/nsis/x64
- **macOS (Apple Silicon):** https://comfy.org

Desktop 버전에서는 Linux가 **지원되지 않습니다**. 경로 D를 사용하세요.

---

### 경로 C: ComfyUI Portable (Windows 전용)

**문서:** https://docs.comfy.org/installation/comfyui_portable_windows

https://github.com/comfyanonymous/ComfyUI/releases 에서 다운로드하여 압축을 풀고 `run_nvidia_gpu.bat`를 실행하세요. 업데이트는 `update/update_comfyui_stable.bat`을 통해 합니다.

---

### 경로 D: comfy-cli (모든 플랫폼 — 에이전트에 권장)

공식 CLI는 헤드리스/자동화된 설정에 가장 적합한 경로입니다.

**문서:** https://docs.comfy.org/comfy-cli/getting-started

#### comfy-cli 설치

```bash
# 권장:
pipx install comfy-cli
# 또는 설치 없이 uvx 사용:
uvx --from comfy-cli comfy --help
# 또는 (pipx/uvx를 사용할 수 없는 경우):
pip install --user comfy-cli
```

분석(Analytics) 비대화형 비활성화:
```bash
comfy --skip-prompt tracking disable
```

#### ComfyUI 설치

```bash
comfy --skip-prompt install --nvidia              # NVIDIA (CUDA)
comfy --skip-prompt install --amd                 # AMD (ROCm, Linux)
comfy --skip-prompt install --m-series            # Apple Silicon (MPS)
comfy --skip-prompt install --cpu                 # CPU only (느림)
comfy --skip-prompt install --nvidia --fast-deps  # uv 기반 의존성 해결
```

기본 위치: `~/comfy/ComfyUI` (Linux), `~/Documents/comfy/ComfyUI` (macOS/Win). `comfy --workspace /custom/path install`로 재정의할 수 있습니다.

#### 실행 / 검증

```bash
comfy launch --background                       # :8188 백그라운드 데몬 실행
comfy launch -- --listen 0.0.0.0 --port 8190    # LAN에서 접근 가능한 커스텀 포트
curl -s http://127.0.0.1:8188/system_stats      # 상태 점검(health check)
```

---

### 경로 E: 수동 설치 (고급 / 미지원 하드웨어)

Ascend NPU, Cambricon MLU, Intel Arc 또는 기타 지원되지 않는 하드웨어의 경우.

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
# SDXL (다목적, ~6.5 GB)
comfy model download \
  --url "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
  --relative-path models/checkpoints

# SD 1.5 (가벼움, ~4 GB, 6 GB 카드에 적합)
comfy model download \
  --url "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" \
  --relative-path models/checkpoints

# Flux Dev fp8 (더 작은 변형, ~12 GB)
comfy model download \
  --url "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
  --relative-path models/checkpoints

# CivitAI (먼저 토큰 설정):
comfy model download \
  --url "https://civitai.com/api/download/models/128713" \
  --relative-path models/checkpoints \
  --set-civitai-api-token "YOUR_TOKEN"
```

설치 목록 보기: `comfy model list`.

### 설치 후: 커스텀 노드 설치

```bash
comfy node install comfyui-impact-pack             # 인기 유틸리티 팩
comfy node install comfyui-animatediff-evolved     # 비디오 생성
comfy node install comfyui-controlnet-aux          # ControlNet 전처리기
comfy node install comfyui-essentials              # 공통 헬퍼 도구
comfy node update all
comfy node install-deps --workflow=workflow.json   # 워크플로우에 필요한 모든 요소 설치
```

### 설치 후: 검증

```bash
python3 scripts/health_check.py
# → comfy_cli가 PATH에 있는지? 서버가 응답하는지? 체크포인트가 있는지? 스모크 테스트?

python3 scripts/check_deps.py my_workflow.json
# → 이 워크플로우의 노드/모델/임베딩이 설치되어 있습니까?

python3 scripts/run_workflow.py \
  --workflow workflows/sd15_txt2img.json \
  --args '{"prompt": "test", "steps": 4}' \
  --output-dir ./test-outputs
```

## 이미지 업로드 (img2img / Inpainting)

가장 간단한 방법은 `run_workflow.py`와 함께 `--input-image`를 사용하는 것입니다:

```bash
python3 scripts/run_workflow.py \
  --workflow workflows/sdxl_img2img.json \
  --input-image image=./photo.png \
  --args '{"prompt": "make it cyberpunk", "denoise": 0.6}'
```

이 플래그는 `photo.png`를 업로드한 다음, 스키마 파라미터 이름이 `image`인 모든 곳에 서버 측 파일 이름을 주입합니다. 인페인팅의 경우 두 가지를 모두 전달합니다:

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
# 반환값: {"name": "photo.png", "subfolder": "", "type": "input"}

# 클라우드 동급 명령어:
curl -X POST "https://cloud.comfy.org/api/upload/image" \
  -H "X-API-Key: $COMFY_CLOUD_API_KEY" \
  -F "image=@photo.png" -F "type=input" -F "overwrite=true"
```

## 클라우드 특이사항

- **기본 URL:** `https://cloud.comfy.org`
- **인증:** `X-API-Key` 헤더 (또는 WebSocket의 경우 `?token=KEY`)
- **API 키:** `$COMFY_CLOUD_API_KEY`를 한 번만 설정하면 스크립트가 자동으로 선택합니다.
- **출력 다운로드:** `/api/view`는 서명된 URL에 대한 302 리디렉트를 반환합니다. 스크립트는 해당 링크를 따라가 저장소 백엔드에서 다운로드하기 전에 `X-API-Key`를 제거합니다(S3/CloudFront로 API 키 유출 방지).
- **로컬 ComfyUI와의 엔드포인트 차이점:**
  - `/api/object_info`, `/api/queue`, `/api/userdata` — **무료 티어에서 403 반환**; 유료 전용.
  - `/history`는 클라우드에서 `/history_v2`로 이름이 변경됩니다(스크립트가 자동으로 라우팅).
  - `/models/<folder>`는 클라우드에서 `/experiment/models/<folder>`로 이름이 변경됩니다(스크립트가 자동으로 라우팅).
  - WebSocket의 `clientId`는 현재 무시됩니다. 한 사용자의 모든 연결이 동일한 브로드캐스트를 수신합니다. 클라이언트 쪽에서 `prompt_id`로 필터링해야 합니다.
  - 업로드 시 `subfolder`는 허용되지만 무시됩니다 — 클라우드는 단일(flat) 네임스페이스를 사용합니다.
- **동시 작업:** Free/Standard: 1, Creator: 3, Pro: 5. 한도를 초과하면 자동으로 대기열에 추가됩니다. `run_batch.py --parallel N`을 사용하여 해당 티어의 한도까지 처리량을 극대화하세요.

## 큐(Queue) 및 시스템 관리

```bash
# 로컬
curl -s http://127.0.0.1:8188/queue | python3 -m json.tool
curl -X POST http://127.0.0.1:8188/queue -d '{"clear": true}'    # 대기 중인 작업 취소
curl -X POST http://127.0.0.1:8188/interrupt                      # 실행 중인 작업 취소
curl -X POST http://127.0.0.1:8188/free \
  -H "Content-Type: application/json" \
  -d '{"unload_models": true, "free_memory": true}'

# 클라우드 — /api/ 아래의 동일한 경로, 추가적으로:
python3 scripts/fetch_logs.py --tail-queue --host https://cloud.comfy.org
```

## 흔한 오류 및 주의사항 (Pitfalls)

1. **API 형식 필수** — 모든 스크립트와 `/api/prompt` 엔드포인트는 API 형식의 워크플로우 JSON을 필요로 합니다. 스크립트는 에디터 형식(최상위 수준에 `nodes` 및 `links` 배열 존재)을 감지하고 "Workflow → Export (API)"(새 UI) 또는 "Save (API Format)"(이전 UI)를 통해 다시 내보내라는 메시지를 표시합니다.

2. **서버 실행 필수** — 모든 실행에는 라이브 서버가 필요합니다. `comfy launch --background`가 서버를 시작합니다. `curl http://127.0.0.1:8188/system_stats`로 확인하세요.

3. **정확한 모델 이름** — 대소문자를 구분하며 파일 확장자가 포함됩니다. `check_deps.py`는 퍼지 매칭(확장자 포함/제외 및 폴더 접두사 유무 등)을 지원하지만, 워크플로우 자체는 정식 이름을 사용해야 합니다. `comfy model list`로 설치된 항목을 파악하세요.

4. **누락된 사용자 지정 노드** — "class_type not found"는 필요한 노드가 설치되지 않았음을 의미합니다. `check_deps.py`는 설치해야 할 패키지를 보고하며, `auto_fix_deps.py`가 이를 대신 설치해 줍니다.

5. **작업 디렉터리 (Working directory)** — `comfy-cli`는 ComfyUI 작업 공간을 자동으로 감지합니다. "no workspace found" 오류와 함께 명령이 실패하면 `comfy --workspace /path/to/ComfyUI <command>` 또는 `comfy set-default /path/to/ComfyUI`를 사용하세요.

6. **클라우드 무료 티어 API 제한** — `/api/prompt`, `/api/view`, `/api/upload/*`, `/api/object_info` 모두 무료 계정에서 403을 반환합니다. `health_check.py`와 `check_deps.py`는 이를 적절히 처리하고 명확한 메시지를 표시합니다.

7. **비디오/오디오 워크플로우의 시간 초과(Timeout)** — 출력 노드가 `VHS_VideoCombine`, `SaveVideo` 등일 때 자동으로 감지되어, 기본 300초에서 900초로 점프합니다. 명시적으로 덮어쓰려면 `--timeout 1800`을 지정하세요.

8. **출력 파일 이름의 경로 이동 공격 방지 (Path traversal)** — 서버에서 제공하는 파일 이름은 `--output-dir`에서 벗어나는 어떠한 경로도 거부하도록 `safe_path_join`을 통과합니다. 이 보호 기능을 유지하세요 — 사용자 지정 저장 노드가 있는 워크플로우는 임의의 경로를 생성할 수 있습니다.

9. **워크플로우 JSON은 임의의 코드입니다** — 커스텀 노드는 Python을 실행하므로 신뢰할 수 없는 워크플로우를 제출하는 것은 `eval`과 같은 신뢰 위험도를 갖습니다. 실행 전 신뢰할 수 없는 출처의 워크플로우를 검사하세요.

10. **자동 무작위 시드 (Auto-randomized seed)** — `--args` 내에 `seed: -1`을 전달하거나(또는 `--randomize-seed`를 사용하고 시드를 생략하면) 실행할 때마다 새로운 시드를 얻을 수 있습니다. 실제 사용된 시드는 stderr에 로깅됩니다.

11. **`tracking` 프롬프트** — 처음 `comfy`를 실행할 때 분석 동의를 물을 수 있습니다. 비대화형으로 생략하려면 `comfy --skip-prompt tracking disable`을 사용하세요. `comfyui_setup.sh`가 이를 대신해 줍니다.

## 검증 체크리스트

모든 목록을 한 번에 실행하려면 `python3 scripts/health_check.py`를 사용하세요. 수동 확인:

- [ ] `hardware_check.py` 판정이 `ok`이거나 사용자가 명시적으로 Comfy Cloud를 선택했습니다.
- [ ] `comfy --version`이 작동합니다 (또는 `uvx --from comfy-cli comfy --help`).
- [ ] `curl http://HOST:PORT/system_stats`가 JSON을 반환합니다.
- [ ] `comfy model list` 명령으로 최소 하나의 체크포인트가 표시됩니다 (로컬) 또는 `/api/experiment/models/checkpoints`가 모델을 반환합니다 (클라우드).
- [ ] 워크플로우 JSON이 API 형식입니다.
- [ ] `check_deps.py`가 `is_ready: true`를 보고합니다 (또는 클라우드 무료 티어의 경우 `node_check_skipped`만 보고).
- [ ] 작은 워크플로우를 테스트 실행하면 잘 완료되며, 출력물이 `--output-dir` 안에 저장됩니다.
