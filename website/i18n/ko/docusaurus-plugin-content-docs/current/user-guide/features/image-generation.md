---
title: 이미지 생성 (Image Generation)
description: FAL.ai를 통해 이미지를 생성하세요 — FLUX 2, GPT Image (1.5 & 2), Nano Banana Pro, Ideogram, Recraft V4 Pro, Krea 2 등 11개 모델 지원, `hermes tools`를 통해 선택 가능.
sidebar_label: 이미지 생성
sidebar_position: 6
---

# 이미지 생성 (Image Generation)

Hermes Agent는 FAL.ai를 통해 텍스트 프롬프트에서 이미지를 생성합니다. 기본적으로 11개의 모델이 지원되며, 각각 속도, 품질 및 비용의 장단점이 다릅니다. 활성 모델은 `hermes tools`를 통해 사용자가 구성할 수 있으며 `config.yaml`에 유지됩니다.

## 지원되는 모델

| 모델 | 속도 | 강점 | 가격 |
|---|---|---|---|
| `fal-ai/flux-2/klein/9b` *(기본값)* | `<1초` | 빠르고 선명한 텍스트 | $0.006/MP |
| `fal-ai/flux-2-pro` | ~6초 | 스튜디오 수준의 사실적인 사진 | $0.03/MP |
| `fal-ai/z-image/turbo` | ~2초 | 영/중 이중 언어, 6B 파라미터 | $0.005/MP |
| `fal-ai/nano-banana-pro` | ~8초 | Gemini 3 Pro, 추론의 깊이, 텍스트 렌더링 | 이미지당 $0.15 (1K) |
| `fal-ai/gpt-image-1.5` | ~15초 | 프롬프트 준수 | 이미지당 $0.034 |
| `fal-ai/gpt-image-2` | ~20초 | SOTA 텍스트 렌더링 + CJK, 세계를 인식하는 사실적인 사진 | 이미지당 $0.04–0.06 |
| `fal-ai/ideogram/v3` | ~5초 | 최상의 타이포그래피 | 이미지당 $0.03–0.09 |
| `fal-ai/recraft/v4/pro/text-to-image` | ~8초 | 디자인, 브랜드 시스템, 프로덕션 준비 완료 | 이미지당 $0.25 |
| `fal-ai/qwen-image` | ~12초 | LLM 기반, 복잡한 텍스트 | $0.02/MP |
| `fal-ai/krea/v2/medium/text-to-image` | ~15-25초 | 일러스트레이션, 애니메이션, 페인팅, 표현/예술적 스타일 | 이미지당 $0.030–0.035 |
| `fal-ai/krea/v2/large/text-to-image` | ~25-60초 | 사실적인 사진, 거친 질감 (모션 블러, 그레인, 필름) | 이미지당 $0.060–0.065 |

가격은 작성 시점의 FAL 가격입니다. 최신 수치는 [fal.ai](https://fal.ai/)를 확인하세요.

## 설정

:::tip Nous 구독자
유료 [Nous Portal](https://portal.nousresearch.com) 구독이 있는 경우 FAL API 키 없이 **[도구 게이트웨이(Tool Gateway)](tool-gateway.md)**를 통해 이미지 생성을 사용할 수 있습니다. 모델 선택은 두 경로에 모두 유지됩니다. 새로 설치하는 경우 `hermes setup --portal`을 실행하여 로그인하고 모든 게이트웨이 도구를 한 번에 켤 수 있습니다. 기존 설치의 경우 `hermes tools`를 통해 이미지 생성 백엔드로 **Nous Subscription**을 선택할 수 있습니다.

관리형 게이트웨이가 특정 모델에 대해 `HTTP 4xx`를 반환하는 경우, 해당 모델은 포털 측에서 아직 프록시되지 않은 것입니다. 에이전트는 이를 알려주고 해결 단계(직접 액세스를 위해 `FAL_KEY` 설정 또는 다른 모델 선택)를 제공합니다.
:::

### FAL API 키 받기

1. [fal.ai](https://fal.ai/)에서 가입하세요.
2. 대시보드에서 API 키를 생성하세요.

### 모델 구성 및 선택

tools 명령어를 실행하세요.

```bash
hermes tools
```

**🎨 Image Generation**으로 이동하여 백엔드(Nous Subscription 또는 FAL.ai)를 선택하면 모든 지원 모델이 열에 맞춰 정렬된 표에 표시됩니다. 화살표 키로 탐색하고 Enter를 눌러 선택하세요.

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
  use_gateway: false            # Nous Subscription을 사용하는 경우 true
```

### GPT-Image 품질

`fal-ai/gpt-image-1.5` 및 `fal-ai/gpt-image-2` 요청 품질은 `medium`(1024×1024에서 ~$0.034–$0.06/이미지)으로 고정되어 있습니다. Nous Portal 청구가 모든 사용자에게 예측 가능하게 유지되도록 `low` / `high` 계층은 사용자 대면 옵션으로 노출하지 않습니다. 계층 간의 비용 차이는 3-22배입니다. 더 저렴한 옵션을 원하면 Klein 9B 또는 Z-Image Turbo를 선택하고, 더 높은 품질을 원하면 Nano Banana Pro 또는 Recraft V4 Pro를 사용하세요.

## 사용법

에이전트 대면 스키마는 의도적으로 최소화되어 있으며, 모델은 사용자가 구성한 내용을 선택합니다.

```
벚꽃이 있는 고요한 산 풍경 이미지를 생성해 줘
```

```
현명한 늙은 부엉이의 정사각형 초상화를 만들어 줘 — 타이포그래피 모델을 사용해
```

```
가로 방향의 미래지향적인 도시 풍경을 만들어 줘
```

## 가로 세로 비율 (Aspect Ratios)

모든 모델은 에이전트 관점에서 동일한 세 가지 가로 세로 비율을 허용합니다. 내부적으로 각 모델의 네이티브 크기 사양은 자동으로 채워집니다.

| 에이전트 입력 | image_size (flux/z-image/qwen/recraft/ideogram) | aspect_ratio (nano-banana-pro) | image_size (gpt-image-1.5) | image_size (gpt-image-2) |
|---|---|---|---|---|
| `landscape` | `landscape_16_9` | `16:9` | `1536x1024` | `landscape_4_3` (1024×768) |
| `square` | `square_hd` | `1:1` | `1024x1024` | `square_hd` (1024×1024) |
| `portrait` | `portrait_16_9` | `9:16` | `1024x1536` | `portrait_4_3` (768×1024) |

GPT Image 2는 최소 픽셀 수가 655,360이므로 16:9 대신 4:3 사전 설정(presets)에 매핑됩니다. `landscape_16_9` 사전 설정(1024×576 = 589,824)은 거부됩니다.

이 변환은 `_build_fal_payload()`에서 발생하므로 에이전트 코드는 모델별 스키마 차이를 알 필요가 없습니다.

## 자동 업스케일링 (Automatic Upscaling)

FAL의 **Clarity Upscaler**를 통한 업스케일링은 모델별로 제한(gated)됩니다.

| 모델 | 업스케일 여부 | 이유 |
|---|---|---|
| `fal-ai/flux-2-pro` | ✓ | 하위 호환성 (선택기 도입 이전 기본값이었음) |
| 기타 모든 모델 | ✗ | 빠른 모델은 1초 미만이라는 가치 제안을 잃게 되고, 고해상도 모델은 이를 필요로 하지 않습니다 |

업스케일링이 실행되면 다음 설정을 사용합니다.

| 설정 | 값 |
|---|---|
| 업스케일 팩터 | 2× |
| 창의성 (Creativity) | 0.35 |
| 유사성 (Resemblance) | 0.6 |
| 가이던스 스케일 (Guidance scale) | 4 |
| 인퍼런스 스텝 (Inference steps) | 18 |

업스케일링이 실패하면(네트워크 문제, 속도 제한) 원본 이미지가 자동으로 반환됩니다.

## 내부 작동 방식

1. **모델 확인 (Model resolution)** — `_resolve_fal_model()`은 `config.yaml`에서 `image_gen.model`을 읽고 `FAL_IMAGE_MODEL` 환경 변수로 폴백한 다음 `fal-ai/flux-2/klein/9b`로 폴백합니다.
2. **페이로드 빌드 (Payload building)** — `_build_fal_payload()`는 사용자의 `aspect_ratio`를 모델의 네이티브 형식(사전 설정 열거형, 가로 세로 비율 열거형 또는 GPT 리터럴)으로 변환하고 모델의 기본 파라미터를 병합한 다음 호출자 재정의를 적용한 다음 모델의 `supports` 화이트리스트로 필터링하여 지원되지 않는 키가 전송되지 않도록 합니다.
3. **제출 (Submission)** — `_submit_fal_request()`는 직접 FAL 자격 증명 또는 관리형 Nous 게이트웨이를 통해 라우팅합니다.
4. **업스케일링 (Upscaling)** — 모델 메타데이터에 `upscale: True`가 있는 경우에만 실행됩니다.
5. **전달 (Delivery)** — 에이전트에 반환된 최종 이미지 URL은 에이전트가 `MEDIA:<url>` 태그를 내보내어 플랫폼 어댑터가 네이티브 미디어로 변환하도록 합니다.

## 디버깅

디버그 로깅 활성화:

```bash
export IMAGE_TOOLS_DEBUG=true
```

디버그 로그는 호출당 세부 정보(모델, 파라미터, 타이밍, 오류)와 함께 `./logs/image_tools_debug_<session_id>.json`으로 전송됩니다.

## 플랫폼 전달 (Platform Delivery)

| 플랫폼 | 전달 |
|---|---|
| **CLI** | 이미지 URL이 마크다운 `![](url)`로 인쇄됨 — 클릭하여 열기 |
| **Telegram** | 프롬프트를 캡션으로 하는 사진 메시지 |
| **Discord** | 메시지에 임베드됨 |
| **Slack** | Slack에 의해 펼쳐진(unfurled) URL |
| **WhatsApp** | 미디어 메시지 |
| **기타** | 일반 텍스트의 URL |

## 제한 사항

- **FAL 자격 증명 필요** (직접 `FAL_KEY` 또는 Nous Subscription)
- **텍스트-이미지 변환(Text-to-image) 전용** — 이 도구를 통한 인페인팅, img2img 또는 편집 없음
- **임시 URL** — FAL은 몇 시간/몇 일 후에 만료되는 호스팅된 URL을 반환하므로 필요한 경우 로컬에 저장하세요.
- **모델별 제약** — 일부 모델은 `seed`, `num_inference_steps` 등을 지원하지 않습니다. `supports` 필터는 지원되지 않는 파라미터를 조용히 무시하며 이는 예상되는 동작입니다.
