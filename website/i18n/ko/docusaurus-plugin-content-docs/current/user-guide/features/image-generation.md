---
title: 이미지 생성
description: FAL.ai를 통해 이미지 생성 — `hermes tools`에서 선택할 수 있는 FLUX 2, GPT Image(1.5 및 2), Nano Banana Pro, Ideogram, Recraft V4 Pro, Krea 2 등을 포함한 11개 모델
sidebar_label: 이미지 생성
sidebar_position: 6
---

# 이미지 생성

Hermes Agent는 FAL.ai를 통해 텍스트 프롬프트로 이미지를 생성합니다. 11개 모델이 기본 제공되며, 각 모델은 속도, 품질 및 비용 측면에서 서로 다른 특성을 가집니다. 활성 모델은 `hermes tools`에서 사용자가 구성할 수 있으며 `config.yaml`에 유지됩니다.

## 지원 모델

| 모델 | 속도 | 강점 | 가격 |
|---|---|---|---|
| `fal-ai/flux-2/klein/9b` *(기본값)* | `<1s` | 빠르고 선명한 텍스트 | $0.006/MP |
| `fal-ai/flux-2-pro` | ~6s | 스튜디오급 사진 실사 | $0.03/MP |
| `fal-ai/z-image/turbo` | ~2s | 영어/중국어 이중 언어, 6B 파라미터 | $0.005/MP |
| `fal-ai/nano-banana-pro` | ~8s | Gemini 3 Pro, 추론 깊이, 텍스트 렌더링 | $0.15/image (1K) |
| `fal-ai/gpt-image-1.5` | ~15s | 프롬프트 준수 | $0.034/image |
| `fal-ai/gpt-image-2` | ~20s | 최고 수준의 텍스트 렌더링 및 CJK, 세계 인식 사진 실사 | $0.04–0.06/image |
| `fal-ai/ideogram/v3` | ~5s | 최고의 타이포그래피 | $0.03–0.09/image |
| `fal-ai/recraft/v4/pro/text-to-image` | ~8s | 디자인, 브랜드 시스템, 프로덕션 준비 완료 | $0.25/image |
| `fal-ai/qwen-image` | ~12s | LLM 기반, 복잡한 텍스트 | $0.02/MP |
| `fal-ai/krea/v2/medium/text-to-image` | ~15-25s | 일러스트, 애니메이션, 회화, 표현력 있고 예술적인 스타일 | $0.030–0.035/image |
| `fal-ai/krea/v2/large/text-to-image` | ~25-60s | 사진 실사, 질감이 살아 있는 원초적 표현(모션 블러, 그레인, 필름) | $0.060–0.065/image |

가격은 작성 시점의 FAL 가격입니다. 최신 가격은 [fal.ai](https://fal.ai/)에서 확인하세요.

## 설정

:::tip Nous 구독자
유료 [Nous Portal](https://portal.nousresearch.com) 구독이 있으면 FAL API 키 없이 **[도구 게이트웨이](tool-gateway.md)**를 통해 이미지 생성을 사용할 수 있습니다. 모델 선택은 두 경로 모두에 유지됩니다. 새로 설치한 경우 `hermes setup --portal`을 실행해 로그인하고 모든 게이트웨이 도구를 한 번에 활성화할 수 있으며, 기존 설치에서는 `hermes tools`에서 이미지 생성 백엔드로 **Nous Subscription**을 선택할 수 있습니다.

관리형 게이트웨이가 특정 모델에 대해 `HTTP 4xx`를 반환하면 해당 모델은 아직 포털 측에서 프록시되지 않은 것입니다 — 에이전트가 해결 방법(`FAL_KEY`를 설정해 직접 접근하거나 다른 모델을 선택)을 안내합니다.
:::

### FAL API 키 발급

1. [fal.ai](https://fal.ai/)에 가입합니다.
2. 대시보드에서 API 키를 생성합니다.

### 모델 구성 및 선택

tools 명령을 실행합니다.

```bash
hermes tools
```

**🎨 Image Generation**으로 이동해 백엔드(Nous Subscription 또는 FAL.ai)를 선택하면 선택 창에 지원되는 모든 모델이 열 맞춤 표로 표시됩니다 — 방향키로 이동하고 Enter로 선택합니다.

```
  Model                          Speed    Strengths                    Price
  fal-ai/flux-2/klein/9b         <1s      Fast, crisp text             $0.006/MP   ← currently in use
  fal-ai/flux-2-pro              ~6s      Studio photorealism          $0.03/MP
  fal-ai/z-image/turbo           ~2s      Bilingual EN/CN, 6B          $0.005/MP
  ...
```

선택 사항은 `config.yaml`에 저장됩니다.

```yaml
image_gen:
  model: fal-ai/flux-2/klein/9b
  use_gateway: false            # true if using Nous Subscription
  max_parallel_requests: 4      # concurrent images in one tool-call batch
```

`max_parallel_requests`의 기본값은 `4`입니다. Hermes는 이 값을 최소 하나 이상, 전역 도구 작업자 제한 이하로 조정하므로 이미지 제공자는 제한된 병렬 요청을 받으며 이미지 배치가 에이전트의 동시성 제한을 우회할 수 없습니다.

### GPT-Image 품질

`fal-ai/gpt-image-1.5` 및 `fal-ai/gpt-image-2`의 요청 품질은 `medium`으로 고정됩니다(1024×1024에서 약 $0.034–$0.06/image). Nous Portal 청구가 모든 사용자에게 예측 가능하도록 `low`/`high` 등급은 사용자에게 제공되는 옵션으로 노출하지 않습니다 — 등급 간 비용 차이는 3–22배입니다. 더 저렴한 옵션이 필요하면 Klein 9B 또는 Z-Image Turbo를 선택하고, 더 높은 품질이 필요하면 Nano Banana Pro 또는 Recraft V4 Pro를 사용하세요.

## 사용법

에이전트 대상 스키마는 의도적으로 최소화되어 있습니다 — 모델은 사용자가 구성한 설정을 그대로 사용합니다.

```
Generate an image of a serene mountain landscape with cherry blossoms
```

```
Create a square portrait of a wise old owl — use the typography model
```

```
Make me a futuristic cityscape, landscape orientation
```

## 이미지 대 이미지 / 편집

동일한 `image_generate` 도구는 활성 모델이 지원하는 경우 **기존 이미지도 편집**합니다 — 소스 이미지를 전달하면 백엔드가 자동으로 편집 엔드포인트로 라우팅합니다(`video_generate`가 이미지-동영상 변환을 처리하는 방식과 같습니다). 소스 이미지를 생략하면 일반적인 텍스트-이미지 생성이 됩니다.

```
Take this photo and make it a rainy Tokyo street at night → <image>
```

```
Blend these two product shots into one hero image → <image1> <image2>
```

편집을 구동하는 두 입력은 다음과 같습니다.

- **`image_url`** — 편집/변환할 기본 소스 이미지(공개 URL 또는 로컬 경로)입니다.
- **`reference_image_urls`** — 스타일/구도 참조로 사용할 추가 이미지입니다(모델별 상한 적용).

### 편집을 지원하는 백엔드

| 백엔드 | 이미지 대 이미지 | 참조 상한 | 방식 |
|---|---|---|---|
| **FAL.ai**(아래 편집 지원 모델) | ✓ | 최대 9개 | 모델의 `/edit` 엔드포인트로 라우팅 |
| **OpenAI**(`gpt-image-2`) | ✓ | 최대 16개 | `images.edit()` |
| **xAI**(Grok Imagine) | ✓ | 1개 | `/v1/images/edits` (`grok-imagine-image-quality`) |
| **Krea**(`Krea 2`) | ✓ | 최대 10개 | 참조 기반 생성(`image_style_references`) |
| **OpenAI(Codex 인증)** | ✓ | 최대 16개 | `input_image` 콘텐츠 파트가 포함된 Codex Responses `image_generation` 도구 |

편집 엔드포인트가 있는 FAL 모델: `flux-2/klein/9b`, `flux-2-pro`,
`nano-banana-pro`, `gpt-image-1.5`, `gpt-image-2`, `ideogram/v3` 및
`qwen-image`. 순수 텍스트-이미지 FAL 모델(`z-image/turbo`, `recraft`,
`krea/*`)은 이미지 입력을 받으면 편집 가능한 모델을 안내하는 명확한 오류를 반환합니다.

:::note OpenAI(Codex 인증)는 최선의 노력 방식입니다

Codex 표면(`chatgpt.com/backend-api/codex`)은 `image_generation`을 채팅 모델이 호출할 수 있는 도구로 호스팅하며, Hermes는 호출을 강제할 수 없습니다 — 백엔드가 호스팅 도구에 대한 모든 `tool_choice` 형식을 거부하므로 요청은 모델이 호출하도록 유도하는 지침에 의존합니다. 호스트 모델이 도구 호출을 거부하면 호출은 `empty_response`와 함께 실패합니다. 호스팅 이미지 도구 자체에 접근 가능한지 여부도 계정에 따라 달라진다는 보고가 있습니다. 이미지 생성을 결정론적으로 작동시켜야 한다면 **OpenAI**(API 키), **FAL** 또는 **xAI** 백엔드를 구성하세요.

:::

활성 모델의 편집 가능 여부는 런타임에 도구 설명에 표시되므로 에이전트는 도구를 호출하기 전에 `image_url`이 적용되는지 알 수 있습니다.

## 화면 비율

모든 모델은 에이전트 관점에서 동일한 세 가지 화면 비율을 받습니다. 내부적으로는 각 모델의 기본 크기 사양이 자동으로 채워집니다.

| 에이전트 입력 | image_size (flux/z-image/qwen/recraft/ideogram) | aspect_ratio (nano-banana-pro) | image_size (gpt-image-1.5) | image_size (gpt-image-2) |
|---|---|---|---|---|
| `landscape` | `landscape_16_9` | `16:9` | `1536x1024` | `landscape_4_3` (1024×768) |
| `square` | `square_hd` | `1:1` | `1024x1024` | `square_hd` (1024×1024) |
| `portrait` | `portrait_16_9` | `9:16` | `1024x1536` | `portrait_4_3` (768×1024) |

GPT Image 2는 최소 픽셀 수가 655,360이므로 16:9 대신 4:3 프리셋에 매핑됩니다 — `landscape_16_9` 프리셋(1024×576 = 589,824)은 거부됩니다.

이 변환은 `_build_fal_payload()`에서 수행되므로 에이전트 코드는 모델별 스키마 차이를 알 필요가 없습니다.

## 업스케일링

### 자동(저해상도 모델에서 기본 활성화)

기본 출력이 약 2MP 미만인 모든 모델은 생성 후 자동으로 고해상도 단계를 실행하므로 낮은 해상도의 이미지를 조용히 받는 일이 없습니다.

| 백엔드 | 기본적으로 업스케일되는 모델 | 업스케일러 |
|---|---|---|
| **FAL.ai** | Seedream 5 Pro/Lite 및 Krea 2 Large를 제외한 전체(기본 해상도 ≥2MP) | Clarity Upscaler(2배, +$0.03/MP) |
| **Krea** | Krea 2 Medium 및 Medium Turbo(기본 1.5K); Large(2K)는 건너뜀 | Krea Enhance(2배, 최대 8K 상한) |
| 기타 백엔드 | — | 업스케일러 없음; 기본 해상도 반환 |

### `upscale` 매개변수(호출별 재정의)

에이전트 대상 `upscale` 불리언은 기본 동작을 어느 방향으로든 재정의합니다.

- `upscale: false` — 자동 단계를 건너뜁니다(더 빠르고 저렴한 초안 출력).
- `upscale: true` — 기본 해상도가 높은 모델이나 이미지 편집에서도 단계를 강제합니다.

`video_generate`도 FAL 백엔드에서 생성 후 ByteDance의 **SeedVR2** 동영상 업스케일러(2배, 출력 동영상 $0.001/MP)를 연결하는 `upscale: true`를 허용합니다. 동영상은 선택적으로 활성화됩니다 — 모든 동영상의 해상도를 기본으로 두 배로 늘리면 비용과 지연 시간이 두 배가 되기 때문입니다.

FAL 이미지 단계가 실행될 때는 다음 설정을 사용합니다.

| 설정 | 값 |
|---|---|
| 업스케일 배율 | 2배 |
| 창의성 | 0.35 |
| 유사도 | 0.6 |
| 가이던스 스케일 | 4 |
| 추론 단계 | 18 |

업스케일링이 실패하면(네트워크 문제, 속도 제한) 원본 이미지가 자동으로 반환됩니다. 응답에는 `upscaled: true/false`가 보고되므로 에이전트는 어떤 해상도를 받았는지 알 수 있습니다.

## 내부 작동 방식

1. **모델 확인** — `_resolve_fal_model()`은 `config.yaml`에서 `image_gen.model`을 읽고, 없으면 `FAL_IMAGE_MODEL` 환경 변수로 대체한 다음, 최종적으로 `fal-ai/flux-2/klein/9b`를 사용합니다.
2. **페이로드 생성** — `_build_fal_payload()`은 `aspect_ratio`를 모델의 기본 형식(프리셋 열거형, 화면 비율 열거형 또는 GPT 리터럴)으로 변환하고, 모델의 기본 매개변수를 병합하며, 호출자가 전달한 재정의를 적용한 다음, 지원되지 않는 키가 전송되지 않도록 모델의 `supports` 허용 목록으로 필터링합니다.
3. **제출** — `_submit_fal_request()`는 직접 FAL 자격 증명 또는 관리형 Nous 게이트웨이를 통해 라우팅합니다.
4. **업스케일링** — 모델 카탈로그 항목에 `upscale: True`가 있거나(2MP 미만 모델의 기본값), 에이전트가 `upscale: true`를 전달한 경우 실행됩니다. 명시적인 `upscale: false`는 항상 건너뜁니다.
5. **전달** — 최종 이미지 URL이 에이전트로 반환되고, 에이전트는 플랫폼 어댑터가 기본 미디어로 변환하는 `MEDIA:<url>` 태그를 출력합니다.

## 디버깅

디버그 로깅을 활성화합니다.

```bash
export IMAGE_TOOLS_DEBUG=true
```

디버그 로그는 호출별 세부 정보(모델, 매개변수, 소요 시간, 오류)가 포함된 `./logs/image_tools_debug_<session_id>.json`에 기록됩니다.

## 플랫폼 전달

| 플랫폼 | 전달 방식 |
|---|---|
| **CLI** | 이미지 URL을 마크다운 `![](url)`로 출력 — 클릭해 엽니다. |
| **Telegram** | 프롬프트를 캡션으로 포함한 사진 메시지 |
| **Discord** | 메시지에 삽입 |
| **Slack** | Slack에서 URL 미리 보기 |
| **WhatsApp** | 미디어 메시지 |
| **기타** | 일반 텍스트로 URL 제공 |

## 제한 사항

- **활성 백엔드의 자격 증명이 필요합니다**(FAL `FAL_KEY` / Nous Subscription, `OPENAI_API_KEY`, xAI OAuth, `KREA_API_KEY`).
- **편집은 모델에 따라 다릅니다** — 이미지 대 이미지 변환은 편집 가능한 모델에서만 작동합니다(위 표 참조). 텍스트-이미지 전용 모델은 이미지 입력을 받으면 명확한 오류를 반환합니다.
- **임시 URL** — 백엔드는 몇 시간 또는 며칠 후 만료되는 호스팅 URL을 반환합니다. Hermes는 전달이 만료 후에도 작동하도록 이를 로컬 캐시에 저장합니다.
- **모델별 제약** — 일부 모델은 `seed`, `num_inference_steps` 등을 지원하지 않습니다. `supports` / `edit_supports` 필터가 지원되지 않는 매개변수를 조용히 삭제하며 이는 정상 동작입니다.
