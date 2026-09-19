---
sidebar_position: 12
title: "비디오 생성 프로바이더 플러그인"
description: "Hermes Agent용 비디오 생성 백엔드 플러그인 구축 방법"
---

# 비디오 생성 프로바이더 플러그인 구축

비디오 생성 프로바이더 플러그인은 모든 `video_generate` 도구 호출을 처리하는 백엔드를 등록합니다. 기본 제공 프로바이더(xAI, FAL, DeepInfra)는 플러그인으로 제공됩니다. 새 프로바이더를 추가하거나 번들된 프로바이더를 재정의하려면 `plugins/video_gen/<name>/`에 디렉터리를 넣으세요.

:::tip
비디오 생성은 [이미지 생성 프로바이더 플러그인](/developer-guide/image-gen-provider-plugin)을 거의 한 줄씩 그대로 따릅니다. 이미지 생성 백엔드를 만들어 본 적이 있다면 구조는 이미 알고 있는 셈입니다. 주요 차이점은 모달리티/가로세로 비율/재생 시간을 알리는 `capabilities()` 메서드와 라우팅 규칙입니다(`image_url`을 전달하면 이미지-비디오를 사용하고, 생략하면 텍스트-비디오를 사용하며, 프로바이더가 내부적으로 올바른 엔드포인트를 선택합니다).
:::

## 통합 표면(하나의 도구, 두 가지 모달리티)

`video_generate` 도구는 하나의 매개변수를 통해 두 가지 모달리티를 제공합니다.

- **텍스트-비디오** — `prompt`만 전달해 호출합니다. 프로바이더가 텍스트-비디오 엔드포인트로 라우팅합니다.
- **이미지-비디오** — `prompt` + `image_url`을 전달해 호출합니다. 프로바이더가 이미지-비디오 엔드포인트로 라우팅합니다.

편집과 확장은 의도적으로 범위에서 제외합니다. 대부분의 백엔드가 이를 지원하지 않으며, 백엔드마다 다른 설명을 에이전트의 도구 설명에 넣어야 하는 불일치가 생기기 때문입니다.

## 검색 방식

Hermes는 다음 세 곳에서 비디오 생성 백엔드를 검색합니다.

1. **번들** — `<repo>/plugins/video_gen/<name>/` (`kind: backend`로 자동 로드)
2. **사용자** — `~/.hermes/plugins/video_gen/<name>/` (`plugins.enabled`를 통해 선택적으로 활성화)
3. **Pip** — `hermes_agent.plugins` 엔트리 포인트를 선언하는 패키지

각 플러그인의 `register(ctx)` 함수는 `ctx.register_video_gen_provider(...)`를 호출합니다. 활성 프로바이더는 `config.yaml`의 `video_gen.provider`로 선택하며, `hermes tools` → 비디오 생성에서 선택 과정을 안내합니다. `image_generate`와 달리 트리 내부의 레거시 백엔드는 없습니다. 모든 프로바이더가 플러그인입니다.

## 디렉터리 구조

```
plugins/video_gen/my-backend/
├── __init__.py      # VideoGenProvider subclass + register()
└── plugin.yaml      # Manifest with kind: backend
```

## VideoGenProvider ABC

`agent.video_gen_provider.VideoGenProvider`를 상속하세요. 필수 항목은 `name` 프로퍼티와 `generate()` 메서드입니다.

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
        # Each entry is a model FAMILY — a name the user picks once.
        # Your provider's generate() routes within the family based on
        # whether image_url was passed.
        return [
            {
                "id": "fast",
                "display": "Fast",
                "speed": "~30s",
                "strengths": "Cheapest tier",
                "price": "$0.05/s",
                "modalities": ["text", "image"],  # advisory
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
            "tag": "Short description shown in `hermes tools`",
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
        **kwargs: Any,  # always ignore unknown kwargs for forward-compat
    ) -> Dict[str, Any]:
        # ROUTE: image_url presence picks the endpoint.
        if image_url:
            endpoint = "my-backend/image-to-video"
            modality_used = "image"
        else:
            endpoint = "my-backend/text-to-video"
            modality_used = "text"

        # ... call your API ...

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

## 플러그인 매니페스트

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

이 도구는 모든 백엔드에서 하나의 스키마를 노출합니다. 프로바이더는 지원하지 않는 매개변수를 무시합니다.

| 매개변수 | 기능 |
|---|---|
| `prompt` | 텍스트 지시(필수) |
| `image_url` | 설정하면 → 이미지-비디오, 생략하면 → 텍스트-비디오 |
| `reference_image_urls` | 스타일/캐릭터 참조(프로바이더에 따라 다름) |
| `duration` | 초 — 프로바이더가 제한값으로 조정 |
| `aspect_ratio` | `"16:9"`, `"9:16"`, `"1:1"`, ... — 프로바이더가 제한값으로 조정 |
| `resolution` | `"480p"` / `"540p"` / `"720p"` / `"1080p"` — 프로바이더가 제한값으로 조정 |
| `negative_prompt` | 피할 콘텐츠(Pixverse/Kling만 해당) |
| `audio` | 네이티브 오디오(Veo3 / Pixverse 요금제) |
| `seed` | 재현성 |
| `model` | 활성 모델/제품군 재정의 |

프로바이더의 `capabilities()`는 이 중 어떤 항목이 적용되는지 알립니다. 사용자가 `hermes tools`를 통해 백엔드를 변경하면 도구 설명이 동적으로 다시 구성되며, 에이전트는 활성 백엔드의 기능을 도구 설명에서 확인합니다.

## 모델 제품군과 엔드포인트 라우팅(FAL 패턴)

백엔드에 "모델"마다 여러 엔드포인트가 있는 경우(예: FAL에서는 각 제품군(Veo 3.1, Pixverse v6, Kling O3)에 `/text-to-video`와 `/image-to-video` URL이 모두 있음), 각 **제품군**을 카탈로그 항목 하나로 표현하세요. `generate()`는 `image_url` 전달 여부에 따라 올바른 엔드포인트를 선택합니다.

```python
FAMILIES = {
    "veo3.1": {
        "text_endpoint": "fal-ai/veo3.1",
        "image_endpoint": "fal-ai/veo3.1/image-to-video",
        # ... family-specific capability flags ...
    },
}

def generate(self, prompt, *, image_url=None, model=None, **kwargs):
    family_id, family = _resolve_family(model)
    endpoint = family["image_endpoint"] if image_url else family["text_endpoint"]
    # ... build payload from family's declared capability flags, call endpoint ...
```

사용자는 `hermes tools`에서 `veo3.1`을 한 번 선택합니다. 에이전트는 엔드포인트를 전혀 고려하지 않고 `image_url`을 전달하거나 전달하지 않을 뿐입니다.

## 선택 우선순위

인스턴스별 모델 설정에 대해서는 `plugins/video_gen/fal/__init__.py`를 참조하세요.

1. 도구 호출의 `model=` 키워드
2. `<PROVIDER>_VIDEO_MODEL` 환경 변수
3. `config.yaml`의 `video_gen.<provider>.model`
4. `config.yaml`의 `video_gen.model`(사용자 ID 중 하나일 때)
5. 프로바이더의 `default_model()`

## 응답 형태

`success_response()`와 `error_response()`는 모든 백엔드가 반환하는 딕셔너리 형태를 생성합니다. 이를 사용하세요. 딕셔너리를 직접 작성하지 마세요.

성공 키: `success`, `video`(URL 또는 절대 경로), `model`, `prompt`, `modality`(`"text"` 또는 `"image"`), `aspect_ratio`, `duration`, `provider`, 그리고 `extra`.

오류 키: `success`, `video`(None), `error`, `error_type`, `model`, `prompt`, `aspect_ratio`, `provider`.

## 아티팩트 저장 위치

백엔드가 base64를 반환하면 `save_b64_video()`를 사용해 `$HERMES_HOME/cache/videos/` 아래에 저장하세요. 후속 HTTP 가져오기로 얻은 원시 바이트에는 `save_bytes_video()`를 사용하세요. 그 외에는 업스트림 URL을 직접 반환하세요. 게이트웨이가 전달 시 원격 URL을 처리합니다.

## 테스트

`tests/plugins/video_gen/test_<name>_plugin.py` 아래에 스모크 테스트를 추가하세요. xAI 및 FAL 테스트에서 패턴을 확인할 수 있습니다. 등록하고, 카탈로그를 확인하고, `image_url` 유무에 따른 라우팅을 모두 실행하고, 인증 정보가 없을 때 깔끔한 오류 응답이 나오는지 검증하세요.
