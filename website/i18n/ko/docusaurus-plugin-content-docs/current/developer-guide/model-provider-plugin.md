---
sidebar_position: 10
title: "모델 제공자 플러그인 (Model Provider Plugins)"
description: "Hermes Agent를 위한 모델 제공자(추론 백엔드) 플러그인을 구축하는 방법입니다."
---

# 모델 제공자 플러그인 구축하기 (Building a Model Provider Plugin)

모델 제공자 플러그인은 Hermes가 `AIAgent` 호출을 라우팅할 수 있는 추론 백엔드(OpenAI 호환 엔드포인트, Anthropic Messages 서버, Codex 스타일의 Responses API, 또는 Bedrock 네이티브 표면)를 선언합니다. 내장된 모든 제공자(OpenRouter, Anthropic, GMI, DeepSeek, Nvidia 등)는 이러한 플러그인 중 하나로 제공됩니다. 서드파티 개발자는 저장소를 변경할 필요 없이 `$HERMES_HOME/plugins/model-providers/` 아래에 디렉터리를 추가하는 것만으로 자신만의 플러그인을 추가할 수 있습니다.

:::tip
모델 제공자 플러그인은 **제공자 플러그인(provider plugin)**의 세 번째 종류입니다. 다른 종류로는 [메모리 제공자 플러그인](/developer-guide/memory-provider-plugin)(세션 간 지식)과 [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin)(컨텍스트 압축 전략)이 있습니다. 이 세 가지 모두 "디렉터리 복사, 프로필 선언, 저장소 수정 없음"이라는 동일한 패턴을 따릅니다.
:::

## 디스커버리(Discovery) 작동 방식

`providers/__init__.py._discover_providers()`는 코드에서 `get_provider_profile()` 또는 `list_providers()`를 처음 호출할 때 지연 실행(lazily runs)됩니다. 탐색 순서:

1. **번들 플러그인** — `<repo>/plugins/model-providers/<name>/` — Hermes와 함께 제공됨
2. **사용자 플러그인** — `$HERMES_HOME/plugins/model-providers/<name>/` — 어느 디렉터리에든 추가 가능; 후속 세션에 대해 재시작이 필요 없음
3. **기존 단일 파일(Legacy single-file)** — `<repo>/providers/<name>.py` — 트리 외부에서 편집 가능한 설치를 위한 하위 호환성 유지 목적

`register_provider()`는 가장 마지막 작성자가 우선(last-writer-wins)하므로, **사용자 플러그인은 동일한 이름의 번들 플러그인을 재정의(override)합니다.** `$HERMES_HOME/plugins/model-providers/gmi/` 디렉터리를 추가하면 저장소를 건드리지 않고 내장된 GMI 프로필을 교체할 수 있습니다.

## 디렉터리 구조

```
plugins/model-providers/my-provider/
├── __init__.py       # 모듈 수준에서 register_provider(profile)를 호출
├── plugin.yaml       # kind: model-provider + 메타데이터 (선택 사항이지만 권장됨)
└── README.md         # 설정 안내 (선택 사항)
```

유일한 필수 파일은 `__init__.py`입니다. `plugin.yaml`은 `hermes plugins`에 의해 내부 검사용으로 사용되며, 일반 플러그인 매니저(PluginManager)가 플러그인을 올바른 로더로 라우팅하는 데 사용됩니다. 이 파일이 없으면 일반 로더는 소스 텍스트를 분석하여 휴리스틱으로 대체(fallback)합니다.

## 최소 예제 — 단순한 API 키 제공자

```python
# plugins/model-providers/acme-inference/__init__.py
from providers import register_provider
from providers.base import ProviderProfile

acme = ProviderProfile(
    name="acme-inference",
    aliases=("acme",),
    display_name="Acme Inference",
    description="Acme — OpenAI 호환 다이렉트 API",
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
description: Acme Inference — OpenAI 호환 다이렉트 API
author: Your Name
```

이게 전부입니다. 이 두 파일을 추가한 후에는 다른 수정 없이 다음 사항들이 **자동으로 연결(auto-wire)**됩니다:

| 통합 항목 | 위치 | 적용 내용 |
|---|---|---|
| 자격 증명 확인 | `hermes_cli/auth.py` | 프로필을 기반으로 `PROVIDER_REGISTRY["acme-inference"]` 채워짐 |
| `--provider` CLI 플래그 | `hermes_cli/main.py` | `acme-inference`를 허용 |
| `hermes model` 선택기 | `hermes_cli/models.py` | `CANONICAL_PROVIDERS`에 표시되며, `{base_url}/models`에서 모델 목록을 가져옴 |
| `hermes doctor` | `hermes_cli/doctor.py` | `ACME_API_KEY` 및 `{base_url}/models` 핑에 대한 상태 검사 |
| `hermes setup` | `hermes_cli/config.py` | `ACME_API_KEY`가 `OPTIONAL_ENV_VARS`와 설정 마법사에 나타남 |
| URL 역매핑 | `agent/model_metadata.py` | 호스트 이름 → 자동 감지를 위한 제공자 이름 |
| 보조 모델 (Auxiliary model) | `agent/auxiliary_client.py` | 압축 / 요약을 위해 `default_aux_model` 사용 |
| 런타임 결정 | `hermes_cli/runtime_provider.py` | 올바른 `base_url`, `api_key`, `api_mode` 반환 |
| 트랜스포트 | `agent/transports/chat_completions.py` | 프로필 경로는 `prepare_messages` / `build_extra_body` / `build_api_kwargs_extras`를 통해 kwargs 생성 |

## ProviderProfile 필드

전체 정의는 `providers/base.py`에 있습니다. 가장 유용한 필드들:

| 필드 | 유형 | 용도 |
|---|---|---|
| `name` | str | 표준 ID — `config.yaml`의 `model.provider` 및 `--provider` 플래그와 일치 |
| `aliases` | `tuple[str, ...]` | `get_provider_profile()`에 의해 결정되는 대체 이름들 (예: `grok` → `xai`) |
| `api_mode` | str | `chat_completions` \| `codex_responses` \| `anthropic_messages` \| `bedrock_converse` |
| `display_name` | str | `hermes model` 선택기에 표시되는 사람이 읽을 수 있는 레이블 |
| `description` | str | 선택기 부제목 (subtitle) |
| `signup_url` | str | 초기 설정 중에 표시됨 ("여기에서 API 키 가져오기") |
| `env_vars` | `tuple[str, ...]` | 우선순위 순서의 API 키 환경 변수; 마지막 `*_BASE_URL` 항목은 사용자의 base-URL 재정의로 사용됨 |
| `base_url` | str | 기본 추론 엔드포인트 |
| `models_url` | str | 명시적인 카탈로그 URL (`{base_url}/models`로 대체 가능) |
| `auth_type` | str | `api_key` \| `oauth_device_code` \| `oauth_external` \| `copilot` \| `aws_sdk` \| `external_process` |
| `fallback_models` | `tuple[str, ...]` | 라이브 카탈로그 가져오기에 실패했을 때 표시되는 선별된 목록 |
| `default_headers` | `dict[str, str]` | 모든 요청에서 전송됨 (예: Copilot의 `Editor-Version`) |
| `fixed_temperature` | Any | `None` = 호출자의 값 사용; `OMIT_TEMPERATURE` 센티널 = 온도(temperature)를 아예 보내지 않음 (Kimi) |
| `default_max_tokens` | `int \| None` | 제공자 수준의 max_tokens 상한 (Nvidia: 16384) |
| `default_aux_model` | str | 보조 작업(압축, 비전, 요약)에 사용할 저렴한 모델 |

## 재정의 가능한 훅(Hooks)

간단하지 않은 특수 동작(quirks)이 필요한 경우 `ProviderProfile`을 서브클래싱하세요:

```python
from typing import Any
from providers.base import ProviderProfile

class AcmeProfile(ProviderProfile):
    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """제공자 전용 메시지 전처리. codex 정리(sanitization) 후, 
        개발자 역할 교체 전에 실행됩니다. 기본값: 통과(pass-through)."""
        # 예시: Qwen은 일반 텍스트 내용을 list-of-parts 배열로 정규화하고 
        # cache_control을 주입합니다. Kimi는 도구 호출 JSON을 다시 작성합니다.
        return messages

    def build_extra_body(self, *, session_id=None, **context) -> dict:
        """API 호출에 병합될 제공자 전용 extra_body 필드. 
        Context 포함: session_id, provider_preferences, model, base_url, 
        reasoning_config. 기본값: 빈 딕셔너리."""
        # 예시: OpenRouter의 제공자 환경설정(provider-preferences) 블록,
        # Gemini의 thinking_config 변환.
        return {}

    def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
        """(extra_body_additions, top_level_kwargs)를 반환합니다. 일부 필드가 최상위(Kimi의 reasoning_effort)로 가고 
        일부 필드는 extra_body(OpenRouter의 reasoning 딕셔너리)로 가야 할 때 필요합니다. 
        기본값: ({}, {})."""
        return {}, {}

    def fetch_models(self, *, api_key=None, timeout=8.0) -> list[str] | None:
        """라이브 카탈로그 가져오기. 기본적으로 Bearer 인증을 사용하여 
        {models_url 또는 base_url}/models를 요청합니다.
        사용자 정의 인증(Anthropic), REST 엔드포인트 부재(Bedrock → None 반환),
        또는 퍼블릭/비인증 카탈로그(OpenRouter)의 경우 재정의하세요."""
        return super().fetch_models(api_key=api_key, timeout=timeout)
```

## 훅 레퍼런스 예제

관용구(idioms)에 대해서는 다음 번들 플러그인들을 살펴보세요:

| 플러그인 | 살펴볼 내용 |
|---|---|
| `plugins/model-providers/openrouter/` | 제공자 환경설정을 포함한 어그리게이터(Aggregator), 공개 모델 카탈로그 |
| `plugins/model-providers/gemini/` | `thinking_config` 변환 (네이티브 + OpenAI 호환 중첩 형태) |
| `plugins/model-providers/kimi-coding/` | `OMIT_TEMPERATURE`, `extra_body.thinking`, 최상위 `reasoning_effort` |
| `plugins/model-providers/qwen-oauth/` | 메시지 정규화, `cache_control` 주입, 비전언어(VL) 고해상도 |
| `plugins/model-providers/nous/` | 귀속(Attribution) 태그, "비활성화 시 추론(reasoning) 생략" |
| `plugins/model-providers/custom/` | Ollama `num_ctx` + `think: false` 특이 사항 |
| `plugins/model-providers/bedrock/` | `api_mode="bedrock_converse"`, `fetch_models`가 None 반환 (REST 엔드포인트 없음) |

## 사용자 재정의 — 저장소를 편집하지 않고 내장 플러그인 교체

예를 들어, 테스트를 위해 `gmi`를 비공개 스테이징 엔드포인트로 지정하고 싶다고 가정해보겠습니다. `~/.hermes/plugins/model-providers/gmi/__init__.py` 파일을 생성하세요:

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

다음 세션에서는 `get_provider_profile("gmi").base_url`이 스테이징 URL을 반환합니다. 리포지토리 패치도 없고 재빌드도 없습니다. 사용자 플러그인이 번들 플러그인 이후에 검색되므로 사용자의 `register_provider()` 호출이 우선적으로 적용되기 때문입니다.

## api_mode 선택

네 가지 값이 인식됩니다. Hermes는 다음을 기준으로 하나를 선택합니다:

1. 사용자 명시적 재정의 (`config.yaml`에 `model.api_mode`가 설정된 경우)
2. OpenCode의 모델별 디스패치 (Zen 및 Go에 대한 `opencode_model_api_mode`)
3. URL 자동 감지 — `/anthropic` 접미사 → `anthropic_messages`, `api.openai.com` → `codex_responses`, `api.x.ai` → `codex_responses`, Kimi 도메인의 `/coding` → `chat_completions`
4. **Profile `api_mode`** — URL 자동 감지가 실패할 때 대체(fallback)로 사용됨
5. 기본값 `chat_completions`

제공자가 제공하는 기본값과 일치하도록 `profile.api_mode`를 설정하세요 (힌트로 작동함). 사용자 URL 재정의가 항상 우선합니다.

## 인증 유형 (Auth types)

| `auth_type` | 의미 | 사용하는 곳 |
|---|---|---|
| `api_key` | 단일 환경 변수가 정적 API 키를 보유 | 대부분의 제공자 |
| `oauth_device_code` | 기기 코드 OAuth 흐름 | — |
| `oauth_external` | 사용자가 다른 곳에서 로그인하고, 토큰은 `auth.json`에 떨어짐 | Anthropic OAuth, MiniMax OAuth, Gemini Cloud Code, Qwen Portal, Nous Portal |
| `copilot` | GitHub Copilot 토큰 갱신 사이클 | `copilot` 플러그인 전용 |
| `aws_sdk` | AWS SDK 자격 증명 체인 (IAM 역할, 프로필, env) | `bedrock` 플러그인 전용 |
| `external_process` | 에이전트가 생성한 하위 프로세스에 의해 인증이 처리됨 | `copilot-acp` 플러그인 전용 |

`auth_type`은 어떤 코드 경로가 당신의 제공자를 "단순한 API 키 제공자"로 취급할지 결정하는 게이트웨이 역할을 합니다. 만약 이 값이 `api_key`가 아니라면, PluginManager는 여전히 매니페스트를 기록하지만 Hermes의 CLI 레벨 자동화(의사 검사, `--provider` 플래그, 설정 마법사 위임)는 건너뛸 수 있습니다.

## 검색 시점 (Discovery timing)

제공자 검색은 **지연(lazy) 처리**됩니다 — 프로세스에서 `get_provider_profile()` 또는 `list_providers()`가 처음 호출될 때 트리거됩니다. 실제로는 시작 시 아주 일찍 발생합니다(`auth.py` 모듈 로드 시 즉시 `PROVIDER_REGISTRY`가 확장됨). 플러그인이 로드되었는지 확인하려면 다음을 실행하세요:

```bash
hermes doctor
```

— 성공적으로 로드된 `auth_type="api_key"` 프로필은 제공자 연결성(Provider Connectivity) 섹션에 `/models` 핑과 함께 나타납니다.

프로그래밍 방식의 검사를 위해서는:

```python
from providers import list_providers
for p in list_providers():
    print(p.name, p.base_url, p.api_mode)
```

## 플러그인 테스트하기

실제 설정을 오염시키지 않도록 `HERMES_HOME`이 임시 디렉터리를 가리키게 하세요:

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

일반 `PluginManager`(`hermes plugins`가 작동하는 대상)는 모델 제공자 플러그인을 **보지만** 가져오지는(import) 않습니다 — `providers/__init__.py`가 수명 주기를 소유합니다. 매니저는 검사(introspection)를 위해 매니페스트를 기록하고 `kind: model-provider`로 범주화합니다. 레이블이 지정되지 않은 사용자 플러그인을 `$HERMES_HOME/plugins/`에 복사하고 그것이 우연히 `ProviderProfile`과 함께 `register_provider`를 호출하면, 매니저는 소스 텍스트 휴리스틱을 통해 자동으로 `kind: model-provider`로 강제 변환합니다. 따라서 플러그인은 `plugin.yaml` 없이도 올바르게 라우팅됩니다.

## pip를 통해 배포하기

다른 Hermes 플러그인과 마찬가지로, 모델 제공자도 pip 패키지로 배포할 수 있습니다. `pyproject.toml`에 엔트리 포인트를 추가하세요:

```toml
[project.entry-points."hermes.plugins"]
acme-inference = "acme_hermes_plugin:register"
```

…여기서 `acme_hermes_plugin:register`는 `register_provider(profile)`를 호출하는 함수입니다. 일반 PluginManager는 `discover_and_load()` 중에 엔트리 포인트 플러그인을 수집합니다. `kind: model-provider` pip 플러그인의 경우, 여전히 매니페스트에서 kind를 선언해야 합니다 (또는 소스 텍스트 휴리스틱에 의존해야 합니다).

전체 엔트리 포인트 설정은 [Hermes 플러그인 구축하기](/guides/build-a-hermes-plugin#distribute-via-pip)를 참조하세요.

## 관련 문서

- [Provider Runtime](/developer-guide/provider-runtime) — 해결 우선순위 및 각 레이어에서 프로필을 읽는 위치
- [Adding Providers](/developer-guide/adding-providers) — 새 추론 백엔드를 위한 엔드투엔드 체크리스트 (빠른 플러그인 경로와 전체 CLI/인증 통합 모두 다룸)
- [Memory Provider Plugins](/developer-guide/memory-provider-plugin)
- [Context Engine Plugins](/developer-guide/context-engine-plugin)
- [Building a Hermes Plugin](/guides/build-a-hermes-plugin) — 일반적인 플러그인 작성
