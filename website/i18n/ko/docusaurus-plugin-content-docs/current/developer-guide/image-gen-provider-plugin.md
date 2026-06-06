---
sidebar_position: 11
title: "Image Generation Provider Plugins"
description: "Hermes 에이전트용 이미지 생성 백엔드 플러그인을 빌드하는 방법"
---

# 이미지 생성 프로바이더 플러그인 빌드하기

이미지 생성 프로바이더 플러그인은 모든 `image_generate` 도구 호출을 서비스하는 백엔드(DALL·E, gpt-image, Grok, Flux, Imagen, Stable Diffusion, fal, Replicate, 로컬 ComfyUI 장비 등)를 등록합니다. 내장 프로바이더(OpenAI, OpenAI-Codex, xAI)는 모두 플러그인 형태로 제공됩니다. `plugins/image_gen/<name>/` 안에 디렉토리를 놓아 새 프로바이더를 추가하거나 내장된 프로바이더를 덮어쓸 수 있습니다.

:::tip
이미지 생성은 Hermes가 지원하는 여러 **백엔드 플러그인** 중 하나입니다. 좀 더 전문화된 ABC(추상 기본 클래스)를 갖춘 다른 플러그인으로는 [메모리 프로바이더 플러그인](/developer-guide/memory-provider-plugin), [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin), [모델 프로바이더 플러그인](/developer-guide/model-provider-plugin)이 있습니다. 일반 도구/훅/CLI 플러그인은 [Hermes 플러그인 빌드하기](/guides/build-a-hermes-plugin)를 참고하세요.
:::

## 검색(Discovery) 작동 방식

Hermes는 세 곳에서 이미지 생성 백엔드를 스캔합니다:

1. **번들(Bundled)** — `<repo>/plugins/image_gen/<name>/` (`kind: backend`와 함께 자동 로드되며 항상 사용 가능)
2. **사용자(User)** — `~/.hermes/plugins/image_gen/<name>/` (`plugins.enabled`를 통해 선택적으로 사용)
3. **Pip** — `hermes_agent.plugins` 진입점(entry point)을 선언하는 패키지

각 플러그인의 `register(ctx)` 함수는 `ctx.register_image_gen_provider(...)`를 호출하여 `agent/image_gen_registry.py`의 레지스트리에 등록합니다. 활성 프로바이더는 `config.yaml`의 `image_gen.provider`에 의해 선택되며, `hermes tools` 명령어를 통해 사용자가 선택할 수 있도록 안내합니다.

`image_generate` 도구 래퍼는 레지스트리에 활성 프로바이더를 묻고 해당 위치로 호출을 디스패치합니다. 등록된 프로바이더가 없으면 도구는 `hermes tools`를 가리키는 유용한 오류를 표시합니다.

## 디렉토리 구조

```
plugins/image_gen/my-backend/
├── __init__.py      # ImageGenProvider 하위 클래스 + register()
└── plugin.yaml      # kind: backend가 포함된 매니페스트
```

번들 플러그인은 이 시점에서 완료됩니다. `~/.hermes/plugins/image_gen/<name>/`에 있는 사용자 플러그인은 `config.yaml`의 `plugins.enabled`에 추가하거나 `hermes plugins enable <name>`을 실행해야 합니다.

## ImageGenProvider ABC

`agent.image_gen_provider.ImageGenProvider`의 하위 클래스를 만듭니다. 필수 멤버는 `name` 속성과 `generate()` 메서드뿐이며 다른 모든 항목은 합리적인 기본값을 가집니다:

```python
# plugins/image_gen/my-backend/__init__.py
from typing import Any, Dict, List, Optional
import os

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)


class MyBackendImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        # image_gen.provider 설정에서 사용되는 고정 ID. 소문자, 공백 없음.
        return "my-backend"

    @property
    def display_name(self) -> str:
        # `hermes tools`에 표시되는 사람이 읽을 수 있는 레이블. 생략 시 name.title()이 기본값.
        return "My Backend"

    def is_available(self) -> bool:
        # 자격 증명이나 종속성이 누락된 경우 False 반환.
        # 도구의 가용성 게이트가 디스패치 전에 이를 호출합니다.
        if not os.environ.get("MY_BACKEND_API_KEY"):
            return False
        try:
            import my_backend_sdk  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        # `hermes tools` 모델 선택기에 표시되는 카탈로그.
        return [
            {
                "id": "my-model-fast",
                "display": "My Model (Fast)",
                "speed": "~5s",
                "strengths": "빠른 반복(Quick iteration)",
                "price": "$0.01/image",
            },
            {
                "id": "my-model-hq",
                "display": "My Model (HQ)",
                "speed": "~30s",
                "strengths": "최고의 정확도(Highest fidelity)",
                "price": "$0.04/image",
            },
        ]

    def default_model(self) -> Optional[str]:
        return "my-model-fast"

    def get_setup_schema(self) -> Dict[str, Any]:
        # `hermes tools` 선택기를 위한 메타데이터 — 설정 시 물어볼 키들.
        return {
            "name": "My Backend",
            "badge": "paid",        # 선택 사항; 선택기에서 짧은 태그로 표시
            "tag": "이름 아래에 표시되는 한 줄 설명",
            "env_vars": [
                {
                    "key": "MY_BACKEND_API_KEY",
                    "prompt": "My Backend API key",
                    "url": "https://my-backend.example.com/api-keys",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect_ratio = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required",
                error_type="invalid_input",
                provider=self.name,
                prompt="",
                aspect_ratio=aspect_ratio,
            )

        # 모델 선택 우선순위: 환경 변수 → 설정 → 기본값. 내장된 openai 플러그인의
        # _resolve_model() 헬퍼가 좋은 참고자료입니다.
        model_id = kwargs.get("model") or self.default_model() or "my-model-fast"

        try:
            import my_backend_sdk
            client = my_backend_sdk.Client(api_key=os.environ["MY_BACKEND_API_KEY"])
            result = client.generate(
                prompt=prompt,
                model=model_id,
                aspect_ratio=aspect_ratio,
            )

            # 두 가지 형태 지원:
            #   - URL 문자열: `image`로 반환
            #   - base64 데이터: save_b64_image()를 통해 $HERMES_HOME/cache/images/ 하위에 저장
            if result.get("image_b64"):
                path = save_b64_image(
                    result["image_b64"],
                    prefix=self.name,
                    extension="png",
                )
                image = str(path)
            else:
                image = result["image_url"]

            return success_response(
                image=image,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                provider=self.name,
            )
        except Exception as exc:
            return error_response(
                error=str(exc),
                error_type=type(exc).__name__,
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )


def register(ctx) -> None:
    """플러그인 진입점 — 로드 시 한 번 호출됩니다."""
    ctx.register_image_gen_provider(MyBackendImageGenProvider())
```

## plugin.yaml

```yaml
name: my-backend
version: 1.0.0
description: My image backend — text-to-image via My Backend SDK
author: Your Name
kind: backend
requires_env:
  - MY_BACKEND_API_KEY
```

`kind: backend`는 플러그인을 이미지 생성 등록 경로로 라우팅합니다. `requires_env`는 `hermes plugins install` 동안 입력을 요청받습니다.

## ABC 참조

전체 계약(contract)은 `agent/image_gen_provider.py`에 있습니다. 일반적으로 재정의(override)하게 될 메서드는 다음과 같습니다:

| 멤버 | 필수 | 기본값 | 목적 |
|---|---|---|---|
| `name` | ✅ | — | `image_gen.provider` 설정에서 사용되는 고정 ID |
| `display_name` | — | `name.title()` | `hermes tools`에 표시되는 레이블 |
| `is_available()` | — | `True` | 누락된 자격 증명/종속성을 위한 게이트 |
| `list_models()` | — | `[]` | `hermes tools` 모델 선택기를 위한 카탈로그 |
| `default_model()` | — | `list_models()`의 첫 번째 | 모델이 구성되지 않은 경우의 폴백 |
| `get_setup_schema()` | — | 최소한 | 선택기 메타데이터 + 환경 변수 프롬프트 |
| `generate(prompt, aspect_ratio, **kwargs)` | ✅ | — | 실행 호출 |

## 응답 형식 (Response format)

`generate()`는 `success_response()` 또는 `error_response()`를 통해 구축된 딕셔너리를 반환해야 합니다. 둘 다 `agent/image_gen_provider.py`에 있습니다.

**성공:**
```python
success_response(
    image=<url-or-absolute-path>,
    model=<model-id>,
    prompt=<echoed-prompt>,
    aspect_ratio="landscape" | "square" | "portrait",
    provider=<your-provider-name>,
    extra={...},  # 선택적 백엔드 전용 필드
)
```

**오류:**
```python
error_response(
    error="사람이 읽을 수 있는 메시지",
    error_type="provider_error" | "invalid_input" | "<exception class name>",
    provider=<your-provider-name>,
    model=<model-id>,
    prompt=<prompt>,
    aspect_ratio=<resolved aspect>,
)
```

도구 래퍼는 딕셔너리를 JSON 직렬화하여 LLM에 전달합니다. 오류는 도구의 결과로 표출되며, 사용자에 대한 설명 방식은 LLM이 결정합니다.

## base64 vs URL 출력 처리

일부 백엔드는 이미지 URL을 반환하고(fal, Replicate), 다른 백엔드는 base64 페이로드(OpenAI gpt-image-2)를 반환합니다. base64의 경우, `save_b64_image()`를 사용하세요 — 이 함수는 `$HERMES_HOME/cache/images/<prefix>_<timestamp>_<uuid>.<ext>`에 쓰고 절대경로 `Path`를 반환합니다. 해당 경로를 `success_response()`의 `image=`에 문자열로 전달하세요. 게이트웨이 전송(Telegram 사진 버블, Discord 첨부 파일)은 URL과 절대 경로를 모두 인식합니다.

## 사용자 재정의 (User overrides)

번들 플러그인과 동일한 `name` 속성을 가진 사용자 플러그인을 `~/.hermes/plugins/image_gen/<name>/`에 놓고 `hermes plugins enable <name>`을 통해 활성화하면 — 레지스트리가 가장 최근 작성자 승리(last-writer-wins) 규칙을 따르므로 여러분의 버전이 내장 프로바이더를 대체합니다. `openai` 플러그인이 개인 프록시를 가리키게 하거나 사용자 지정 모델 카탈로그로 교체할 때 유용합니다.

## 테스트

```bash
export HERMES_HOME=/tmp/hermes-imggen-test
mkdir -p $HERMES_HOME/plugins/image_gen/my-backend
# …해당 디렉토리에 __init__.py + plugin.yaml을 복사합니다…

export MY_BACKEND_API_KEY=your-test-key
hermes plugins enable my-backend

# 활성 프로바이더로 선택
echo "image_gen:" >> $HERMES_HOME/config.yaml
echo "  provider: my-backend" >> $HERMES_HOME/config.yaml

# 실행하기
hermes -z "Generate an image of a corgi in a spacesuit"
```

또는 대화형으로: `hermes tools` → "Image Generation" → `my-backend` 선택 → 프롬프트가 표시되면 API 키 입력.

## 참조 구현

- **`plugins/image_gen/openai/__init__.py`** — 품질(quality) 매개변수만 다르고 하나의 API 모델을 공유하는 3개의 가상 모델 ID로서의 하/중/상위 계층 gpt-image-2. 단일 백엔드 아래 계층화된 모델 + config.yaml 우선순위 체인의 좋은 예.
- **`plugins/image_gen/xai/__init__.py`** — xAI의 Grok Imagine. 다른 형태 (URL 출력, 더 단순한 카탈로그).
- **`plugins/image_gen/openai-codex/__init__.py`** — 다른 라우팅 기본 URL과 함께 OpenAI SDK를 재사용하는 Codex 스타일 Responses API 변형.

## pip를 통한 배포

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-backend-imggen = "my_backend_imggen_package"
```

`my_backend_imggen_package`는 최상위 수준의 `register` 함수를 노출해야 합니다. 전체 설정은 일반 플러그인 가이드의 [pip를 통한 배포](/guides/build-a-hermes-plugin#distribute-via-pip)를 참조하세요.

## 관련 문서

- [이미지 생성](/user-guide/features/image-generation) — 사용자 지향 기능 문서
- [플러그인 개요](/user-guide/features/plugins) — 모든 플러그인 유형 한눈에 보기
- [Hermes 플러그인 빌드하기](/guides/build-a-hermes-plugin) — 일반 도구/훅/슬래시 명령어 가이드
