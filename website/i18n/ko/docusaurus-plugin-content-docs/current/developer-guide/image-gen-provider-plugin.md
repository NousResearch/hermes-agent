---
sidebar_position: 11
title: "이미지 생성 프로바이더 플러그인"
description: "Hermes Agent용 이미지 생성 백엔드 플러그인을 만드는 방법"
---

# 이미지 생성 프로바이더 플러그인 만들기

이미지 생성 프로바이더 플러그인은 모든 `image_generate` 도구 호출을 처리하는 백엔드를 등록합니다 — DALL·E, gpt-image, Grok, Flux, Imagen, Stable Diffusion, fal, Replicate, 로컬 ComfyUI 환경 등 무엇이든 가능합니다. 기본 제공 프로바이더(OpenAI, OpenAI-Codex, xAI, FAL, Krea, DeepInfra, OpenRouter)는 모두 플러그인으로 제공됩니다. `plugins/image_gen/<name>/`에 디렉터리를 추가하면 새 프로바이더를 추가하거나 번들된 프로바이더를 재정의할 수 있습니다.

:::tip
이미지 생성은 Hermes가 지원하는 여러 **백엔드 플러그인** 중 하나입니다. 그 밖에 더 전문화된 ABC를 사용하는 [메모리 프로바이더 플러그인](/developer-guide/memory-provider-plugin), [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin), [모델 프로바이더 플러그인](/developer-guide/model-provider-plugin)이 있습니다. 일반 도구/훅/CLI 플러그인은 [Hermes 플러그인 만들기](/developer-guide/plugins)에 설명되어 있습니다.
:::

## 검색 방식

Hermes는 다음 세 곳에서 이미지 생성 백엔드를 검색합니다.

1. **번들** — `<repo>/plugins/image_gen/<name>/` (`kind: backend`로 자동 로드되며 항상 사용 가능)
2. **사용자** — `~/.hermes/plugins/image_gen/<name>/` (`plugins.enabled`를 통해 선택적으로 활성화)
3. **Pip** — `hermes_agent.plugins` 엔트리 포인트를 선언하는 패키지

각 플러그인의 `register(ctx)` 함수는 `ctx.register_image_gen_provider(...)`를 호출하여 `agent/image_gen_registry.py`의 레지스트리에 추가합니다. 활성 프로바이더는 `config.yaml`의 `image_gen.provider`로 선택하며, `hermes tools`가 사용자를 선택 과정으로 안내합니다.

`image_generate` 도구 래퍼는 레지스트리에 활성 프로바이더를 요청하고 해당 프로바이더로 디스패치합니다. 등록된 프로바이더가 없으면 도구가 `hermes tools`를 안내하는 유용한 오류를 표시합니다.

## 디렉터리 구조

```
plugins/image_gen/my-backend/
├── __init__.py      # ImageGenProvider subclass + register()
└── plugin.yaml      # Manifest with kind: backend
```

이 시점에서 번들 플러그인은 완성됩니다. `~/.hermes/plugins/image_gen/<name>/`의 사용자 플러그인은 `config.yaml`의 `plugins.enabled`에 추가하거나(`hermes plugins enable <name>`을 실행해도 됨) 활성화해야 합니다.

## ImageGenProvider ABC

`agent.image_gen_provider.ImageGenProvider`를 상속하세요. 필수 멤버는 `name` 프로퍼티와 `generate()` 메서드뿐이며 나머지는 모두 합리적인 기본값을 제공합니다.

```python
# plugins/image_gen/my-backend/__init__.py
from typing import Any, Dict, List, Optional
import os

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)


class MyBackendImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        # Stable id used in image_gen.provider config. Lowercase, no spaces.
        return "my-backend"

    @property
    def display_name(self) -> str:
        # Human label shown in `hermes tools`. Defaults to name.title() if omitted.
        return "My Backend"

    def is_available(self) -> bool:
        # Return False if credentials or deps are missing.
        # The tool's availability gate calls this before dispatch.
        if not os.environ.get("MY_BACKEND_API_KEY"):
            return False
        try:
            import my_backend_sdk  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        # Catalog shown in `hermes tools` model picker.
        return [
            {
                "id": "my-model-fast",
                "display": "My Model (Fast)",
                "speed": "~5s",
                "strengths": "Quick iteration",
                "price": "$0.01/image",
            },
            {
                "id": "my-model-hq",
                "display": "My Model (HQ)",
                "speed": "~30s",
                "strengths": "Highest fidelity",
                "price": "$0.04/image",
            },
        ]

    def default_model(self) -> Optional[str]:
        return "my-model-fast"

    def get_setup_schema(self) -> Dict[str, Any]:
        # Metadata for the `hermes tools` picker — keys to prompt for at setup.
        return {
            "name": "My Backend",
            "badge": "paid",        # optional; shown as a short tag in the picker
            "tag": "One-line description shown under the name",
            "env_vars": [
                {
                    "key": "MY_BACKEND_API_KEY",
                    "prompt": "My Backend API key",
                    "url": "https://my-backend.example.com/api-keys",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # Declare whether this backend supports image-to-image / editing.
        # The tool layer surfaces this in the dynamic schema so the model
        # knows when `image_url` is honored. Default (if you omit this) is
        # text-only: {"modalities": ["text"], "max_reference_images": 0}.
        return {"modalities": ["text", "image"], "max_reference_images": 4}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
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

        # Routing: if image_url (or reference_image_urls) is set, the call is
        # an image-to-image / edit request; otherwise text-to-image. Report
        # which path you took via the `modality` field of success_response.
        sources = []
        if image_url:
            sources.append(image_url)
        sources.extend(normalize_reference_images(reference_image_urls) or [])
        modality = "image" if sources else "text"

        # Model selection precedence: env var → config → default. The helper
        # _resolve_model() in the built-in openai plugin is a good reference.
        model_id = kwargs.get("model") or self.default_model() or "my-model-fast"

        try:
            import my_backend_sdk
            client = my_backend_sdk.Client(api_key=os.environ["MY_BACKEND_API_KEY"])
            if modality == "image":
                result = client.edit(
                    prompt=prompt,
                    model=model_id,
                    image_urls=sources,
                )
            else:
                result = client.generate(
                    prompt=prompt,
                    model=model_id,
                    aspect_ratio=aspect_ratio,
                )

            # Two shapes supported:
            #   - URL string: return it as `image`
            #   - base64 data: save under $HERMES_HOME/cache/images/ via save_b64_image()
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
                modality=modality,
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
    """Plugin entry point — called once at load time."""
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

`kind: backend`가 플러그인을 이미지 생성 등록 경로로 라우팅합니다. `requires_env`는 `hermes plugins install` 중에 입력을 요청하는 항목입니다.

## ABC 참고

전체 계약은 `agent/image_gen_provider.py`에 있습니다. 일반적으로 재정의할 메서드는 다음과 같습니다.

| 멤버 | 필수 | 기본값 | 용도 |
|---|---|---|---|
| `name` | ✅ | — | `image_gen.provider` 설정에서 사용하는 안정적인 ID |
| `display_name` | — | `name.title()` | `hermes tools`에 표시되는 레이블 |
| `is_available()` | — | `True` | 자격 증명/의존성 누락 여부 확인 게이트 |
| `list_models()` | — | `[]` | `hermes tools` 모델 선택기의 카탈로그 |
| `default_model()` | — | `list_models()`의 첫 항목 | 모델이 설정되지 않았을 때의 대체값 |
| `get_setup_schema()` | — | 최소 구성 | 선택기 메타데이터 + 환경 변수 프롬프트 |
| `generate(prompt, aspect_ratio, **kwargs)` | ✅ | — | 호출 메서드 |

## 응답 형식

`generate()`는 `success_response()` 또는 `error_response()`로 만든 dict를 반환해야 합니다. 둘 다 `agent/image_gen_provider.py`에 있습니다.

**성공:**
```python
success_response(
    image=<url-or-absolute-path>,
    model=<model-id>,
    prompt=<echoed-prompt>,
    aspect_ratio="landscape" | "square" | "portrait",
    provider=<your-provider-name>,
    extra={...},  # optional backend-specific fields
)
```

**오류:**
```python
error_response(
    error="human-readable message",
    error_type="provider_error" | "invalid_input" | "<exception class name>",
    provider=<your-provider-name>,
    model=<model-id>,
    prompt=<prompt>,
    aspect_ratio=<resolved aspect>,
)
```

도구 래퍼는 dict를 JSON으로 직렬화하여 LLM에 전달합니다. 오류는 도구 결과로 노출되며, LLM이 사용자에게 설명할 방법을 결정합니다.

## base64와 URL 출력 처리

일부 백엔드는 이미지 URL을 반환하고(fal, Replicate), 다른 백엔드는 base64 페이로드를 반환합니다(OpenAI gpt-image-2). base64인 경우 `save_b64_image()`를 사용하세요. 이 함수는 `$HERMES_HOME/cache/images/<prefix>_<timestamp>_<uuid>.<ext>`에 기록하고 절대 `Path`를 반환합니다. 해당 경로를 `str`로 변환하여 `success_response()`의 `image=`에 전달하세요. 게이트웨이 전송(Telegram 사진 말풍선, Discord 첨부 파일)은 URL과 절대 경로를 모두 인식합니다.

## 사용자 재정의

`~/.hermes/plugins/image_gen/<name>/`에 번들 플러그인과 동일한 `name` 프로퍼티를 가진 사용자 플러그인을 추가하고 `hermes plugins enable <name>`으로 활성화하세요. 레지스트리는 마지막 작성자가 이기는 방식이므로 사용자 버전이 기본 제공 버전을 대체합니다. 예를 들어 `openai` 플러그인이 비공개 프록시를 가리키도록 하거나 사용자 지정 모델 카탈로그로 교체할 때 유용합니다.

## 테스트

```bash
export HERMES_HOME=/tmp/hermes-imggen-test
mkdir -p $HERMES_HOME/plugins/image_gen/my-backend
# …copy __init__.py + plugin.yaml into that dir…

export MY_BACKEND_API_KEY=your-test-key
hermes plugins enable my-backend

# Pick it as the active provider
echo "image_gen:" >> $HERMES_HOME/config.yaml
echo "  provider: my-backend" >> $HERMES_HOME/config.yaml

# Exercise it
hermes -z "Generate an image of a corgi in a spacesuit"
```

또는 대화형으로 실행할 수 있습니다. `hermes tools` → “Image Generation” → `my-backend` 선택 → 요청 시 API 키 입력.

## 참고 구현

- **`plugins/image_gen/openai/__init__.py`** — 서로 다른 `quality` 매개변수를 사용하는 세 개의 가상 모델 ID가 하나의 API 모델을 공유하며 low/medium/high 등급의 gpt-image-2를 제공합니다. 하나의 백엔드에서 계층형 모델을 구성하고 config.yaml 우선순위 체인을 적용하는 좋은 예입니다.
- **`plugins/image_gen/xai/__init__.py`** — xAI를 통한 Grok Imagine. 다른 형태(URL 출력, 더 단순한 카탈로그)입니다.
- **`plugins/image_gen/openai-codex/__init__.py`** — 다른 라우팅 기본 URL과 함께 OpenAI SDK를 재사용하는 Codex 스타일 Responses API 변형입니다.

## pip으로 배포

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-backend-imggen = "my_backend_imggen_package"
```

`my_backend_imggen_package`는 최상위 `register` 함수를 노출해야 합니다. 전체 설정은 일반 플러그인 가이드의 [pip으로 배포](/developer-guide/plugins#distribute-via-pip)를 참조하세요.

## 관련 페이지

- [이미지 생성](/user-guide/features/image-generation) — 사용자 대상 기능 문서
- [플러그인 개요](/user-guide/features/plugins) — 한눈에 보는 모든 플러그인 유형
- [Hermes 플러그인 만들기](/developer-guide/plugins) — 일반 도구/훅/슬래시 명령 가이드
