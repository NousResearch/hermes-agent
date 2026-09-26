---
sidebar_position: 10
title: "모델 프로바이더 플러그인"
description: "Hermes Agent용 모델 프로바이더(추론 백엔드) 플러그인 빌드 방법"
---

# 모델 프로바이더 플러그인 빌드

모델 프로바이더 플러그인은 Hermes가 `AIAgent` 호출을 라우팅할 수 있는 추론 백엔드(OpenAI 호환 엔드포인트, Anthropic Messages 서버, Codex 스타일 Responses API 또는 Bedrock 네이티브 표면)를 선언합니다. 모든 기본 제공 프로바이더(OpenRouter, Anthropic, GMI, DeepSeek, Nvidia, …)는 이러한 플러그인 중 하나로 제공됩니다. 서드파티는 저장소를 전혀 변경하지 않고 `$HERMES_HOME/plugins/model-providers/` 아래에 디렉터리를 추가하여 자체 프로바이더를 만들 수 있습니다.

:::tip
모델 프로바이더 플러그인은 세 번째 종류의 **프로바이더 플러그인**입니다. 나머지는 [메모리 프로바이더 플러그인](/developer-guide/memory-provider-plugin)(세션 간 지식)과 [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin)(컨텍스트 압축 전략)입니다. 세 종류 모두 "디렉터리를 추가하고, 프로필을 선언하며, 저장소를 수정하지 않는" 동일한 패턴을 따릅니다.
:::

## 검색 방식

`providers/__init__.py._discover_providers()`는 코드에서 `get_provider_profile()` 또는 `list_providers()`를 처음 호출할 때 지연 방식으로 실행됩니다. 검색 순서는 다음과 같습니다.

1. **번들 플러그인** — `<repo>/plugins/model-providers/<name>/` — Hermes와 함께 제공됨
2. **사용자 플러그인** — `$HERMES_HOME/plugins/model-providers/<name>/` — 어떤 디렉터리든 추가 가능하며 이후 세션에 재시작이 필요하지 않음
3. **레거시 단일 파일** — `<repo>/providers/<name>.py` — 저장소 외부의 editable 설치와의 하위 호환성을 위한 경로

`register_provider()`가 마지막 등록 우선 방식이므로 **사용자 플러그인은 같은 이름의 번들 플러그인을 덮어씁니다**. `$HERMES_HOME/plugins/model-providers/gmi/` 디렉터리를 추가하면 저장소를 건드리지 않고 기본 제공 GMI 프로필을 교체할 수 있습니다.

## 디렉터리 구조

```
plugins/model-providers/my-provider/
├── __init__.py       # Calls register_provider(profile) at module-level
├── plugin.yaml       # kind: model-provider + metadata (optional but recommended)
└── README.md         # Setup instructions (optional)
```

필수 파일은 `__init__.py` 하나뿐입니다. `plugin.yaml`은 `hermes plugins`에서 정보를 확인할 때와 일반 PluginManager가 플러그인을 올바른 로더로 라우팅할 때 사용됩니다. 이 파일이 없으면 일반 로더는 소스 텍스트 휴리스틱으로 대체합니다.

## 최소 예제 — 간단한 API 키 프로바이더

```python
# plugins/model-providers/acme-inference/__init__.py
from providers import register_provider
from providers.base import ProviderProfile

acme = ProviderProfile(
    name="acme-inference",
    aliases=("acme",),
    display_name="Acme Inference",
    description="Acme — OpenAI-compatible direct API",
    signup_url="https://acme.example.com/keys",
    env_vars=("ACME_API_KEY", "ACME_BASE_URL"),
    base_url="https://api.acme.example.com/v1",
    auth_type="api_key",
    default_aux_model="acme-small-fast",
    fallback_models=(
        "acme-large-v3",
        "acme-medium-v3",
        "acme-small-fast",
    ),
)

register_provider(acme)
```

```yaml
# plugins/model-providers/acme-inference/plugin.yaml
name: acme-inference
kind: model-provider
version: 1.0.0
description: Acme Inference — OpenAI-compatible direct API
author: Your Name
```

두 파일을 추가하면 다음 항목이 다른 수정 없이 **자동으로 연결**됩니다.

| 통합 지점 | 위치 | 제공되는 기능 |
|---|---|---|
| 자격 증명 확인 | `hermes_cli/auth.py` | 프로필에서 `PROVIDER_REGISTRY["acme-inference"]`를 채움 |
| `--provider` CLI 플래그 | `hermes_cli/main.py` | `acme-inference`를 허용 |
| `hermes model` 선택기 | `hermes_cli/models.py` | `CANONICAL_PROVIDERS`에 표시되고 `{base_url}/models`에서 모델 목록을 가져옴 |
| `hermes doctor` | `hermes_cli/doctor.py` | `ACME_API_KEY` 및 `{base_url}/models` 프로브를 위한 상태 점검 |
| `hermes setup` | `hermes_cli/config.py` | `ACME_API_KEY`가 `OPTIONAL_ENV_VARS`와 설정 마법사에 표시됨 |
| URL 역매핑 | `agent/model_metadata.py` | 호스트 이름을 프로바이더 이름으로 매핑하여 자동 감지 |
| 보조 모델 | `agent/auxiliary_client.py` | 압축 및 요약에 `default_aux_model`을 사용 |
| 런타임 확인 | `hermes_cli/runtime_provider.py` | 올바른 `base_url`, `api_key`, `api_mode`를 반환 |
| 전송 | `agent/transports/chat_completions.py` | 프로필 경로가 `prepare_messages` / `build_extra_body` / `build_api_kwargs_extras`를 통해 kwargs를 생성 |

## ProviderProfile 필드

전체 정의는 `providers/base.py`에 있습니다. 가장 유용한 필드는 다음과 같습니다.

| 필드 | 타입 | 용도 |
|---|---|---|
| `name` | str | 정규 ID — `config.yaml`의 `model.provider` 및 `--provider` 플래그와 일치 |
| `aliases` | `tuple[str, ...]` | `get_provider_profile()`에서 확인하는 대체 이름(예: `grok` → `xai`) |
| `api_mode` | str | `chat_completions` \| `codex_responses` \| `anthropic_messages` \| `bedrock_converse` |
| `display_name` | str | `hermes model` 선택기에 표시되는 사용자 친화적 이름 |
| `description` | str | 선택기 부제목 |
| `signup_url` | str | 최초 실행 설정 중 표시되는 링크("여기에서 API 키 받기") |
| `env_vars` | `tuple[str, ...]` | 우선순위 순서의 API 키 환경 변수이며 마지막 `*_BASE_URL` 항목은 사용자 base URL 재정의에 사용 |
| `base_url` | str | 기본 추론 엔드포인트 |
| `models_url` | str | 명시적 카탈로그 URL(`{base_url}/models`로 대체) |
| `auth_type` | str | `api_key` \| `oauth_device_code` \| `oauth_external` \| `copilot` \| `aws_sdk` \| `external_process` |
| `fallback_models` | `tuple[str, ...]` | 실시간 카탈로그 가져오기에 실패했을 때 표시되는 엄선된 목록 |
| `default_headers` | `dict[str, str]` | 모든 요청에 전송되는 헤더(예: Copilot의 `Editor-Version`) |
| `fixed_temperature` | Any | `None` = 호출자의 값을 사용; `OMIT_TEMPERATURE` sentinel = temperature를 전혀 전송하지 않음(Kimi) |
| `default_max_tokens` | `int \| None` | 프로바이더 수준의 max_tokens 상한(Nvidia: 16384) |
| `default_aux_model` | str | 보조 작업(압축, 비전, 요약)에 사용할 저렴한 모델 |

## 재정의 가능한 훅

특수한 동작이 필요한 경우 `ProviderProfile`을 서브클래싱합니다.

```python
from typing import Any
from providers.base import ProviderProfile

class AcmeProfile(ProviderProfile):
    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing. Runs after codex
        sanitization, before developer-role swap. Default: pass-through."""
        # Example: Qwen normalizes plain-text content to a list-of-parts
        # array and injects cache_control; Kimi rewrites tool-call JSON
        return messages

    def build_extra_body(self, *, session_id=None, **context) -> dict:
        """Provider-specific extra_body fields merged into the API call.
        Context includes: session_id, provider_preferences, model, base_url,
        reasoning_config. Default: empty dict."""
        # Example: OpenRouter's provider-preferences block,
        # Gemini's thinking_config translation.
        return {}

    def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
        """Returns (extra_body_additions, top_level_kwargs). Needed when some
        fields go top-level (Kimi's reasoning_effort, OpenRouter's verbosity for
        adaptive Anthropic models) and some go in extra_body (OpenRouter's
        reasoning dict). Default: ({}, {})."""
        return {}, {}

    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0) -> list[str] | None:
        """Live catalog fetch. Default hits {models_url or base_url}/models with
        Bearer auth. Override for: custom auth (Anthropic), no REST endpoint
        (Bedrock → None), or public/unauthenticated catalogs (OpenRouter)."""
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
```

## 훅 참조 예시

관용적인 사용 방식을 확인하려면 다음 번들 플러그인을 살펴보세요.

| 플러그인 | 살펴볼 이유 |
|---|---|
| `plugins/model-providers/openrouter/` | 프로바이더 환경 설정과 공개 모델 카탈로그를 제공하는 애그리게이터 |
| `plugins/model-providers/gemini/` | `thinking_config` 변환(네이티브 및 OpenAI 호환 중첩 형식) |
| `plugins/model-providers/kimi-coding/` | `OMIT_TEMPERATURE`, `extra_body.thinking`, 최상위 `reasoning_effort` |
| `plugins/model-providers/qwen-oauth/` | 메시지 정규화, `cache_control` 주입, VL 고해상도 |
| `plugins/model-providers/nous/` | 어트리뷰션 태그, 비활성화 시 "reasoning 생략" |
| `plugins/model-providers/custom/` | Ollama의 `num_ctx` 및 `think: false` 특수 동작 |
| `plugins/model-providers/bedrock/` | `api_mode="bedrock_converse"`, `fetch_models`가 None 반환(REST 엔드포인트 없음) |

## 사용자 재정의 — 저장소를 수정하지 않고 기본 제공 프로바이더 교체

테스트를 위해 `gmi`를 비공개 스테이징 엔드포인트로 연결한다고 가정해 보겠습니다. `~/.hermes/plugins/model-providers/gmi/__init__.py`를 생성합니다.

```python
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="gmi",
    aliases=("gmi-cloud", "gmicloud"),
    env_vars=("GMI_API_KEY",),
    base_url="https://gmi-staging.internal.example.com/v1",
    auth_type="api_key",
    default_aux_model="google/gemini-3.1-flash-lite-preview",
))
```

다음 세션부터 `get_provider_profile("gmi").base_url`은 스테이징 URL을 반환합니다. 저장소 패치나 재빌드가 필요하지 않습니다. 사용자 플러그인은 번들 플러그인 다음에 검색되므로 사용자 측 `register_provider()` 호출이 우선합니다.

## api_mode 선택

네 가지 값이 인식됩니다. Hermes는 다음 순서로 하나를 선택합니다.

1. 사용자가 명시적으로 재정의한 값(`config.yaml`에서 설정된 `model.api_mode`)
2. OpenCode의 모델별 디스패치(Zen 및 Go의 `opencode_model_api_mode`)
3. URL 자동 감지 — `/anthropic` 접미사 → `anthropic_messages`, `api.openai.com` → `codex_responses`, `api.x.ai` → `codex_responses`, Kimi 도메인의 `/coding` → `chat_completions`
4. **프로필의 `api_mode`** — URL 감지에서 아무것도 찾지 못했을 때의 대체값
5. 기본값 `chat_completions`

프로바이더가 기본으로 제공하는 값에 맞게 `profile.api_mode`를 설정하세요. 이는 힌트로 작동하며 사용자가 지정한 URL 재정의가 여전히 우선합니다.

## 인증 유형

| `auth_type` | 의미 | 사용하는 주체 |
|---|---|---|
| `api_key` | 하나의 환경 변수가 정적 API 키를 담음 | 대부분의 프로바이더 |
| `oauth_device_code` | 디바이스 코드 OAuth 흐름 | — |
| `oauth_external` | 사용자가 다른 곳에서 로그인하고 토큰이 `auth.json`에 저장됨 | Anthropic OAuth, MiniMax OAuth, Qwen Portal, Nous Portal |
| `copilot` | GitHub Copilot 토큰 갱신 주기 | `copilot` 플러그인만 |
| `aws_sdk` | AWS SDK 자격 증명 체인(IAM 역할, 프로필, 환경 변수) | `bedrock` 플러그인만 |
| `external_process` | 에이전트가 생성하는 서브프로세스가 인증을 처리 | `copilot-acp` 플러그인만 |

`auth_type`은 어떤 코드 경로가 프로바이더를 "단순 API 키 프로바이더"로 처리할지 결정합니다. `api_key`가 아니어도 PluginManager는 매니페스트를 기록하지만 Hermes의 CLI 수준 자동화(doctor 점검, `--provider` 플래그, 설정 마법사 위임)는 이를 건너뛸 수 있습니다.

## 검색 시점

프로바이더 검색은 **지연 방식**으로 이루어지며 프로세스에서 처음 `get_provider_profile()` 또는 `list_providers()`를 호출할 때 시작됩니다. 실제로는 시작 초기에(`auth.py` 모듈 로드 시 `PROVIDER_REGISTRY`가 즉시 확장됨) 발생합니다. 플러그인이 로드되었는지 확인하려면 다음을 실행하세요.

```bash
hermes doctor
```

— 성공한 `auth_type="api_key"` 프로필은 `/models` 프로브와 함께 Provider Connectivity 섹션에 표시됩니다.

프로그래밍 방식으로 확인하려면 다음과 같이 실행합니다.

```python
from providers import list_providers
for p in list_providers():
    print(p.name, p.base_url, p.api_mode)
```

## 플러그인 테스트

실제 설정을 오염시키지 않도록 `HERMES_HOME`을 임시 디렉터리로 지정합니다.

```bash
export HERMES_HOME=/tmp/hermes-plugin-test
mkdir -p $HERMES_HOME/plugins/model-providers/my-provider
cat > $HERMES_HOME/plugins/model-providers/my-provider/__init__.py <<'EOF'
from providers import register_provider
from providers.base import ProviderProfile
register_provider(ProviderProfile(
    name="my-provider",
    env_vars=("MY_API_KEY",),
    base_url="https://api.my-provider.example.com/v1",
    auth_type="api_key",
))
EOF

export MY_API_KEY=your-test-key
hermes -z "hello" --provider my-provider -m some-model
```

## 일반 PluginManager 통합

일반 `PluginManager`(`hermes plugins`가 사용하는 것)는 모델 프로바이더 플러그인을 **확인하지만** 가져오지는 않습니다. 해당 플러그인의 수명 주기는 `providers/__init__.py`가 관리합니다. 관리자는 정보를 확인할 수 있도록 매니페스트를 기록하고 `kind: model-provider`로 분류합니다. `$HERMES_HOME/plugins/`에 라벨이 없는 사용자 플러그인을 추가했는데 해당 플러그인이 `ProviderProfile`과 함께 `register_provider`를 호출하는 경우 관리자는 소스 텍스트 휴리스틱으로 이를 `kind: model-provider`로 자동 변환합니다. 따라서 `plugin.yaml`이 없어도 플러그인은 올바르게 라우팅됩니다.

## pip으로 배포

모델 프로바이더는 pip 패키지로 제공할 수 있습니다. `pyproject.toml`의
`hermes_agent.plugins` 그룹에 엔트리 포인트를 노출합니다.

```toml
[project.entry-points."hermes_agent.plugins"]
acme-inference = "acme_hermes_plugin:register"
```

대상은 다음 중 하나일 수 있습니다.

- **호출 가능 객체**(`module:func`) — 인자 없이 호출되며 `register_provider(profile)`를 호출해야 함
- **일반 모듈**(`module`) — 디렉터리 플러그인의 `__init__.py` 계약과 마찬가지로 모듈 수준의 `register_provider(...)` 부수 효과를 위해 가져옴

`providers/__init__.py`가 이러한 엔트리 포인트를 직접 검색합니다. 일반 `PluginManager`는 pip 패키지에 대해 프로바이더 등록을 호출하지 않습니다(해당 엔트리 포인트 경로는 `plugins.enabled`로 제어되는 `register(ctx)` 스타일의 일반 플러그인을 대상으로 함). 따라서 프로바이더 레지스트리가 자체적으로 검색을 수행합니다. 다음 두 규칙이 적용됩니다.

- **옵트인이 필요합니다.** `config.yaml`의 동일한 `plugins.enabled` 허용 목록(`plugins.disabled` 거부 목록도 포함)이 이 검색을 제어합니다. 설치되어 있다는 이유만으로 pip 패키지를 가져오지 않습니다. 사용자는 엔트리 포인트 이름을 `plugins.enabled`에 추가해야 합니다.

  ```yaml
  plugins:
    enabled:
      - acme-inference
  ```

- **최저 우선순위입니다.** 엔트리 포인트 플러그인은 파일 시스템 플러그인보다 **먼저** 검색됩니다. `register_provider()`가 마지막 등록 우선 방식이므로 같은 이름의 번들 프로필 또는 `$HERMES_HOME` 프로필이 항상 pip 설치 프로필을 덮어씁니다. pip 패키지는 완전히 새로운 프로바이더를 추가할 수 있지만 퍼스트파티 프로바이더 이름을 조용히 가로챌 수는 없습니다.

인자를 요구하는 대상(일반 플러그인의 `register(ctx)`)은 프로바이더 검색에서 건너뜁니다. 이러한 대상은 `PluginManager`에 속합니다. 손상된 엔트리 포인트는 격리됩니다. 경고 수준으로 기록되고 건너뛰며 다른 프로바이더의 검색을 차단하지 않습니다.

전체 엔트리 포인트 설정은 [Hermes 플러그인 빌드](/developer-guide/plugins#distribute-via-pip)를 참조하세요.

## 관련 페이지

- [프로바이더 런타임](/developer-guide/provider-runtime) — 확인 우선순위 및 각 계층이 프로필을 읽는 위치
- [프로바이더 추가](/developer-guide/adding-providers) — 새 추론 백엔드를 위한 엔드투엔드 체크리스트(빠른 플러그인 경로와 전체 CLI/인증 통합 모두 포함)
- [메모리 프로바이더 플러그인](/developer-guide/memory-provider-plugin)
- [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin)
- [Hermes 플러그인 빌드](/developer-guide/plugins) — 일반 플러그인 작성
