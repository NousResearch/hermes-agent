---
sidebar_position: 12
title: "비디오 생성 제공자 플러그인 (Video Gen Provider Plugins)"
description: "Hermes Agent를 위한 비디오 생성 백엔드 플러그인을 구축하는 방법입니다."
---

# 비디오 생성 제공자 플러그인 구축하기 (Building a Video Generation Provider Plugin)

비디오 생성 제공자 플러그인은 모든 `video_generate` 도구 호출을 서비스하는 백엔드를 등록합니다. 내장된 제공자(xAI, FAL)는 플러그인 형태로 배포됩니다. 새 플러그인을 추가하거나 번들 플러그인을 무시(override)하려면 `plugins/video_gen/<name>/` 디렉터리를 만들면 됩니다.

:::tip
비디오 생성은 [이미지 생성 제공자 플러그인](/developer-guide/image-gen-provider-plugin)과 내용이 거의 똑같습니다 — 이미지 생성 백엔드를 만들어본 적이 있다면 이미 전체 구조를 아는 것과 다름없습니다. 주요 차이점은 모달리티(modalities)/화면 비율/재생 시간을 알리는 `capabilities()` 메서드와 라우팅 규칙(image-to-video를 사용하려면 `image_url` 전달, text-to-video를 사용하려면 이를 생략 — 제공자가 내부적으로 올바른 엔드포인트를 선택함)입니다.
:::

## 통합된 인터페이스 (하나의 도구, 두 가지 모달리티)

`video_generate` 도구는 하나의 파라미터를 통해 두 가지 모달리티를 노출합니다:

- **텍스트-비디오 변환 (Text-to-video)** — `prompt`만 사용하여 호출. 제공자는 자신의 텍스트-비디오 엔드포인트로 라우팅합니다.
- **이미지-비디오 변환 (Image-to-video)** — `prompt` + `image_url`을 함께 사용하여 호출. 제공자는 자신의 이미지-비디오 엔드포인트로 라우팅합니다.

편집(Edit) 및 확장(extend) 기능은 의도적으로 제외되었습니다. 대부분의 백엔드가 이를 지원하지 않으며, 일관성이 깨지면 에이전트 도구 설명에 각 백엔드별로 장황한 설명을 추가해야 하기 때문입니다.

## 디스커버리(Discovery) 작동 방식

Hermes는 비디오 생성 백엔드를 다음 세 곳에서 스캔합니다:

1. **번들 플러그인** — `<repo>/plugins/video_gen/<name>/` (`kind: backend`로 자동 로드됨)
2. **사용자 플러그인** — `~/.hermes/plugins/video_gen/<name>/` (`plugins.enabled`를 통해 선택적으로 적용)
3. **Pip 패키지** — `hermes_agent.plugins` 엔트리 포인트를 선언하는 패키지들

각 플러그인의 `register(ctx)` 함수는 `ctx.register_video_gen_provider(...)`를 호출합니다. 활성화된 제공자는 `config.yaml`의 `video_gen.provider`에 의해 선택되며, 사용자는 `hermes tools` → Video Generation을 통해 선택 과정을 거칩니다. `image_generate`와 달리 이전 방식의 내장 백엔드가 존재하지 않으며, 모든 제공자는 플러그인입니다.

## 디렉터리 구조

```
plugins/video_gen/my-backend/
├── __init__.py      # VideoGenProvider 서브클래스 + register()
└── plugin.yaml      # kind: backend를 포함하는 매니페스트
```

## VideoGenProvider 추상 기본 클래스 (ABC)

`agent.video_gen_provider.VideoGenProvider`를 서브클래싱하세요. 필수 항목: `name` 속성 및 `generate()` 메서드.

```python
# plugins/video_gen/my-backend/__init__.py
from typing import Any, Dict, List, Optional
import os

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    success_response,
)


class MyVideoGenProvider(VideoGenProvider):
    @property
    def name(self) -> str:
        return "my-backend"

    @property
    def display_name(self) -> str:
        return "My Backend"

    def is_available(self) -> bool:
        return bool(os.environ.get("MY_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        # 각 항목은 사용자가 한 번 선택하는 이름인 모델 FAMILY입니다.
        # 사용자의 generate()는 image_url의 전달 여부에 따라
        # 패밀리 내에서 라우팅합니다.
        return [
            {
                "id": "fast",
                "display": "Fast",
                "speed": "~30s",
                "strengths": "Cheapest tier",
                "price": "$0.05/s",
                "modalities": ["text", "image"],  # 참고용
            },
        ]

    def default_model(self) -> Optional[str]:
        return "fast"

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16"],
            "resolutions": ["720p", "1080p"],
            "min_duration": 1,
            "max_duration": 10,
            "supports_audio": False,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "My Backend",
            "badge": "paid",
            "tag": "`hermes tools`에 표시되는 간단한 설명",
            "env_vars": [
                {
                    "key": "MY_API_KEY",
                    "prompt": "My Backend API key",
                    "url": "https://mybackend.example.com/keys",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,  # 향후 호환성을 위해 알 수 없는 kwargs는 항상 무시
    ) -> Dict[str, Any]:
        # 라우팅: image_url이 있으면 해당 엔드포인트를 선택합니다.
        if image_url:
            endpoint = "my-backend/image-to-video"
            modality_used = "image"
        else:
            endpoint = "my-backend/text-to-video"
            modality_used = "text"

        # ... 해당 API 호출 ...

        return success_response(
            video="https://your-cdn/output.mp4",
            model=model or "fast",
            prompt=prompt,
            modality=modality_used,
            aspect_ratio=aspect_ratio,
            duration=duration or 5,
            provider=self.name,
        )


def register(ctx) -> None:
    ctx.register_video_gen_provider(MyVideoGenProvider())
```

## 플러그인 매니페스트 (Plugin manifest)

```yaml
# plugins/video_gen/my-backend/plugin.yaml
name: my-backend
version: 1.0.0
description: "My video generation backend"
author: Your Name
kind: backend
requires_env:
  - MY_API_KEY
```

## `video_generate` 스키마

이 도구는 모든 백엔드에서 하나의 스키마를 노출합니다. 제공자는 자신이 지원하지 않는 파라미터를 무시합니다.

| 파라미터 | 역할 |
|---|---|
| `prompt` | 텍스트 명령어 (필수) |
| `image_url` | 설정된 경우 → 이미지-비디오 변환; 생략된 경우 → 텍스트-비디오 변환 |
| `reference_image_urls` | 스타일/캐릭터 참조용 (제공자에 따라 다름) |
| `duration` | 초 단위 — 제공자가 제한(clamp) 처리함 |
| `aspect_ratio` | `"16:9"`, `"9:16"`, `"1:1"` 등 — 제공자가 제한 처리함 |
| `resolution` | `"480p"` / `"540p"` / `"720p"` / `"1080p"` — 제공자가 제한 처리함 |
| `negative_prompt` | 피해야 할 내용 (Pixverse/Kling 전용) |
| `audio` | 네이티브 오디오 (Veo3 / Pixverse 유료 버전) |
| `seed` | 결과물 재현성(Reproducibility) |
| `model` | 활성화된 모델/패밀리를 재정의 |

제공자의 `capabilities()`는 이 중 어떤 파라미터를 준수하는지 알립니다. 에이전트는 사용자가 `hermes tools`를 통해 백엔드를 변경할 때마다 동적으로 재구성되는 도구 설명(tool description)에서 활성화된 백엔드의 기능을 확인할 수 있습니다.

## 모델 패밀리와 엔드포인트 라우팅 (FAL 패턴)

백엔드에 "모델"당 여러 엔드포인트가 있는 경우 — 예를 들어 FAL의 경우 모든 패밀리(Veo 3.1, Pixverse v6, Kling O3)가 `/text-to-video` 및 `/image-to-video` URL을 모두 가짐 — 각 **패밀리**를 카탈로그의 단일 항목으로 나타냅니다. `generate()` 메서드는 `image_url` 전달 여부에 따라 적절한 엔드포인트를 선택합니다:

```python
FAMILIES = {
    "veo3.1": {
        "text_endpoint": "fal-ai/veo3.1",
        "image_endpoint": "fal-ai/veo3.1/image-to-video",
        # ... 패밀리 전용 기능(capability) 플래그들 ...
    },
}

def generate(self, prompt, *, image_url=None, model=None, **kwargs):
    family_id, family = _resolve_family(model)
    endpoint = family["image_endpoint"] if image_url else family["text_endpoint"]
    # ... 패밀리에 선언된 기능 플래그를 바탕으로 페이로드를 구성하고 엔드포인트 호출 ...
```

사용자는 `hermes tools`에서 `veo3.1`을 한 번만 선택합니다. 에이전트는 엔드포인트에 대해 신경 쓸 필요 없이 그저 `image_url`을 전달하거나(하지 않거나) 할 뿐입니다.

## 선택 우선순위

인스턴스별 모델 설정(knobs)의 경우 (`plugins/video_gen/fal/__init__.py` 참고):

1. 도구 호출에서 전달된 `model=` 키워드
2. `<PROVIDER>_VIDEO_MODEL` 환경 변수
3. `config.yaml`의 `video_gen.<provider>.model`
4. `config.yaml`의 `video_gen.model` (이것이 제공자의 ID 중 하나인 경우)
5. 제공자의 `default_model()`

## 응답 형태 (Response shape)

`success_response()` 및 `error_response()`는 모든 백엔드가 반환해야 하는 딕셔너리 형태를 생성합니다. 직접 딕셔너리를 만들지 말고 이 메서드를 사용하세요.

성공 키: `success`, `video` (URL 또는 절대 경로), `model`, `prompt`, `modality` (`"text"` 또는 `"image"`), `aspect_ratio`, `duration`, `provider`, 그리고 `extra`.

오류 키: `success`, `video` (None), `error`, `error_type`, `model`, `prompt`, `aspect_ratio`, `provider`.

## 결과물 저장 위치

백엔드가 base64로 결과를 반환하는 경우, `$HERMES_HOME/cache/videos/` 아래에 기록하기 위해 `save_b64_video()`를 사용하세요. 후속 HTTP 요청에서 얻은 원시 바이트 데이터의 경우 `save_bytes_video()`를 사용하세요. 그렇지 않으면 업스트림 URL을 직접 반환하세요 — 게이트웨이가 전달 시에 원격 URL을 처리합니다.

## 테스트 (Testing)

`tests/plugins/video_gen/test_<name>_plugin.py` 아래에 스모크 테스트(smoke test)를 추가하세요. xAI와 FAL의 테스트는 다음과 같은 패턴을 보여줍니다 — 제공자 등록, 카탈로그 확인, `image_url`의 유무에 따른 두 라우팅 테스트 실행, 인증 정보가 누락된 경우의 정상적인 오류 응답 발생 확인.
