---
sidebar_position: 5
title: "제공자 추가하기"
description: "Hermes Agent에 새로운 추론 제공자를 추가하는 방법 — 인증, 런타임 결정, CLI 흐름, 어댑터, 테스트 및 문서"
---

# 제공자 추가하기 (Adding Providers)

Hermes는 이미 사용자 지정 제공자 경로를 통해 모든 OpenAI 호환 엔드포인트와 통신할 수 있습니다. 다음과 같이 해당 서비스에 대한 일급(first-class) 사용자 경험(UX)을 원하는 경우가 아니라면, 내장 제공자를 추가하지 마세요:

- 제공자 전용 인증 또는 토큰 갱신
- 선별된 모델 카탈로그
- 설정 / `hermes model` 메뉴 항목
- `provider:model` 구문을 위한 제공자 별칭
- 어댑터가 필요한 비(非) OpenAI API 형태

제공자가 단지 "또 다른 OpenAI 호환 기본 URL 및 API 키"인 경우, 이름이 지정된 사용자 지정 제공자만으로도 충분할 수 있습니다.

## 멘탈 모델 (The mental model)

내장 제공자는 몇 가지 계층에 걸쳐 연결되어야 합니다:

1. `hermes_cli/auth.py`는 자격 증명을 찾는 방법을 결정합니다.
2. `hermes_cli/runtime_provider.py`는 이를 다음과 같은 런타임 데이터로 변환합니다:
   - `provider`
   - `api_mode`
   - `base_url`
   - `api_key`
   - `source`
3. `run_agent.py`는 `api_mode`를 사용하여 요청을 구성하고 보내는 방법을 결정합니다.
4. `hermes_cli/models.py` 및 `hermes_cli/main.py`는 CLI에 제공자가 표시되도록 합니다. (`hermes_cli/setup.py`는 자동으로 `main.py`에 위임하므로 변경할 필요가 없습니다.)
5. `agent/auxiliary_client.py` 및 `agent/model_metadata.py`는 보조 작업 및 토큰 예산 책정을 정상적으로 작동하게 유지합니다.

중요한 추상화는 `api_mode`입니다.

- 대부분의 제공자는 `chat_completions`를 사용합니다.
- Codex는 `codex_responses`를 사용합니다.
- Anthropic은 `anthropic_messages`를 사용합니다.
- 새로운 비(非) OpenAI 프로토콜은 보통 새로운 어댑터와 새로운 `api_mode` 분기를 추가하는 것을 의미합니다.

## 구현 경로 먼저 선택하기

### 경로 A — OpenAI 호환 제공자

제공자가 표준 채팅 완료(chat-completions) 스타일의 요청을 수락할 때 이 경로를 사용하세요.

일반적인 작업:

- 인증 메타데이터 추가
- 모델 카탈로그 / 별칭 추가
- 런타임 결정 논리 추가
- CLI 메뉴 연결 추가
- 보조 모델(aux-model) 기본값 추가
- 테스트 및 사용자 문서 추가

일반적으로 새로운 어댑터나 새로운 `api_mode`가 필요하지 않습니다.

### 경로 B — 네이티브 제공자

제공자가 OpenAI 채팅 완료처럼 동작하지 않을 때 이 경로를 사용하세요.

현재 트리에 있는 예시:

- `codex_responses`
- `anthropic_messages`

이 경로에는 경로 A의 모든 항목과 다음이 포함됩니다:

- `agent/` 내의 제공자 어댑터
- 요청 빌드, 디스패치, 사용량 추출, 인터럽트 처리 및 응답 정규화를 위한 `run_agent.py` 분기
- 어댑터 테스트

## 파일 체크리스트

### 모든 내장 제공자에 필수적인 항목

1. `hermes_cli/auth.py`
2. `hermes_cli/models.py`
3. `hermes_cli/runtime_provider.py`
4. `hermes_cli/main.py`
5. `agent/auxiliary_client.py`
6. `agent/model_metadata.py`
7. 테스트
8. `website/docs/` 아래의 사용자 대상 문서

:::tip
`hermes_cli/setup.py`는 변경할 필요가 **없습니다**. 설정 마법사는 제공자/모델 선택을 `main.py`의 `select_provider_and_model()`에 위임하므로, 여기에 추가된 제공자는 `hermes setup`에서 자동으로 사용할 수 있습니다.
:::

### 네이티브 / 비(非) OpenAI 제공자를 위한 추가 항목

10. `agent/<provider>_adapter.py`
11. `run_agent.py`
12. 제공자 SDK가 필요한 경우 `pyproject.toml`

## 빠른 경로: 간단한 API 키 제공자

제공자가 단일 API 키로 인증하는 단순한 OpenAI 호환 엔드포인트인 경우, `auth.py`, `runtime_provider.py`, `main.py` 또는 아래 전체 체크리스트의 다른 파일들을 수정할 필요가 없습니다.

필요한 것은 다음이 전부입니다:

1. `plugins/model-providers/<your-provider>/` 아래에 다음 항목이 포함된 플러그인 디렉터리:
   - `__init__.py` — 모듈 수준에서 `register_provider(profile)` 호출
   - `plugin.yaml` — 매니페스트 (이름, kind: model-provider, 버전, 설명)
2. 이게 끝입니다. 코드가 `get_provider_profile()` 또는 `list_providers()`를 처음 호출할 때 제공자 플러그인이 자동 로드됩니다 — 번들 플러그인(이 리포지토리)과 `$HERMES_HOME/plugins/model-providers/`에 있는 사용자 플러그인이 모두 선택됩니다.

플러그인을 추가하고 `register_provider()`를 호출하면 다음 사항이 자동으로 연결됩니다:

1. `auth.py`의 `PROVIDER_REGISTRY` 항목 (자격 증명 결정, 환경 변수 조회)
2. `api_mode`가 `chat_completions`로 설정됨
3. 설정 또는 선언된 환경 변수에서 소싱된 `base_url`
4. API 키에 대한 우선순위 순서로 `env_vars` 검사
5. 제공자에 대해 등록된 `fallback_models` 목록
6. `--provider` CLI 플래그가 제공자 ID를 수락
7. `hermes model` 메뉴에 제공자가 포함됨
8. `hermes setup` 마법사가 자동으로 `main.py`에 위임함
9. `provider:model` 별칭 구문이 작동함
10. 런타임 확인자가 올바른 `base_url` 및 `api_key`를 반환함
11. `--provider <name>` CLI 플래그가 제공자 ID를 수락
12. 대체 모델 활성화 시 제공자로 깔끔하게 전환 가능

`$HERMES_HOME/plugins/model-providers/<name>/`에 있는 사용자 플러그인은 동일한 이름의 번들 플러그인을 무시(override)합니다 (`register_provider()`에서 나중에 쓴 것이 적용됨) — 따라서 서드파티는 리포지토리를 편집하지 않고도 모든 내장 프로필을 몽키 패치하거나 교체할 수 있습니다.

템플릿으로 `plugins/model-providers/nvidia/` 또는 `plugins/model-providers/gmi/`를 참조하고, 필드 참조, 훅(hook) 관용구, 엔드투엔드 예제는 전체 [모델 제공자 플러그인 가이드](/developer-guide/model-provider-plugin)를 확인하세요.

## 전체 경로: OAuth 및 복잡한 제공자

제공자에 다음 중 하나라도 필요한 경우 아래의 전체 체크리스트를 사용하세요:

- OAuth 또는 토큰 갱신 (Nous Portal, Codex, Google Gemini, Qwen Portal, Copilot)
- 새 어댑터가 필요한 비(非) OpenAI API 형태 (Anthropic Messages, Codex Responses)
- 사용자 지정 엔드포인트 감지 또는 다중 리전 조사 (z.ai, Kimi)
- 선별된 정적 모델 카탈로그 또는 실시간 `/models` 가져오기
- 맞춤형 인증 흐름을 가진 제공자 전용 `hermes model` 메뉴 항목

## 1단계: 하나의 표준 제공자 ID 선택

단일 제공자 ID를 선택하고 모든 곳에서 이를 사용하세요.

리포지토리의 예시:

- `openai-codex`
- `kimi-coding`
- `minimax-cn`

이 동일한 ID가 다음에 표시되어야 합니다:

- `hermes_cli/auth.py`의 `PROVIDER_REGISTRY`
- `hermes_cli/models.py`의 `_PROVIDER_LABELS`
- `hermes_cli/auth.py` 및 `hermes_cli/models.py` 양쪽의 `_PROVIDER_ALIASES`
- `hermes_cli/main.py`의 CLI `--provider` 선택지
- 설정 / 모델 선택 분기
- 보조 모델(auxiliary-model) 기본값
- 테스트

이러한 파일 간에 ID가 다르면 제공자가 절반만 연결된 것처럼 느껴집니다: 인증은 작동하지만 `/model`, 설정 또는 런타임 결정이 조용히 누락될 수 있습니다.

## 2단계: `hermes_cli/auth.py`에 인증 메타데이터 추가

API 키 제공자의 경우, `PROVIDER_REGISTRY`에 다음을 포함하는 `ProviderConfig` 항목을 추가하세요:

- `id`
- `name`
- `auth_type="api_key"`
- `inference_base_url`
- `api_key_env_vars`
- 선택적 `base_url_env_var`

또한 `_PROVIDER_ALIASES`에 별칭을 추가하세요.

기존 제공자를 템플릿으로 사용하세요:

- 간단한 API 키 경로: Z.AI, MiniMax
- 엔드포인트 감지 기능이 있는 API 키 경로: Kimi, Z.AI
- 네이티브 토큰 확인: Anthropic
- OAuth / 인증 스토어 경로: Nous, OpenAI Codex

여기서 대답해야 할 질문:

- Hermes가 어떤 환경 변수를 검사해야 하며 우선순위는 어떻게 되나요?
- 제공자에게 기본 URL(base-URL) 재정의가 필요한가요?
- 엔드포인트 조사 또는 토큰 갱신이 필요한가요?
- 자격 증명이 누락되었을 때 인증 오류 메시지가 어떻게 나와야 하나요?

제공자가 "API 키 찾기" 이상의 기능을 필요로 하는 경우, 관련 없는 분기에 논리를 욱여넣지 말고 전용 자격 증명 확인자를 추가하세요.

## 3단계: `hermes_cli/models.py`에 모델 카탈로그 및 별칭 추가

제공자가 메뉴 및 `provider:model` 구문에서 작동하도록 제공자 카탈로그를 업데이트하세요.

일반적인 편집 항목:

- `_PROVIDER_MODELS`
- `_PROVIDER_LABELS`
- `_PROVIDER_ALIASES`
- `list_available_providers()` 내의 제공자 표시 순서
- 제공자가 실시간 `/models` 가져오기를 지원하는 경우 `provider_model_ids()`

제공자가 실시간 모델 목록을 노출하는 경우 이를 우선으로 지정하고 `_PROVIDER_MODELS`는 정적 대체 항목으로 유지하세요.

이 파일은 다음과 같은 입력이 작동하게 만듭니다:

```text
anthropic:claude-sonnet-4-6
kimi:model-name
```

여기서 별칭이 누락된 경우, 제공자가 올바르게 인증하더라도 `/model` 파싱에서 실패할 수 있습니다.

## 4단계: `hermes_cli/runtime_provider.py`에서 런타임 데이터 결정

`resolve_runtime_provider()`는 CLI, 게이트웨이, cron, ACP 및 헬퍼 클라이언트가 사용하는 공통 경로입니다.

최소한 다음을 포함하는 딕셔너리를 반환하는 분기를 추가하세요:

```python
{
    "provider": "your-provider",
    "api_mode": "chat_completions",  # 또는 네이티브 모드
    "base_url": "https://...",
    "api_key": "...",
    "source": "env|portal|auth-store|explicit",
    "requested_provider": requested_provider,
}
```

제공자가 OpenAI 호환인 경우, `api_mode`는 일반적으로 `chat_completions`로 유지해야 합니다.

API 키 우선순위에 주의하세요. Hermes는 이미 관련 없는 엔드포인트에 OpenRouter 키가 유출되지 않도록 하는 로직을 포함하고 있습니다. 새 제공자는 어떤 키가 어떤 기본 URL로 이동하는지 명확하게 지정해야 합니다.

## 5단계: `hermes_cli/main.py`에 CLI 연결

제공자는 대화형 `hermes model` 흐름에 나타나기 전까지 검색 불가능합니다.

`hermes_cli/main.py`에서 다음을 업데이트하세요:

- `provider_labels` 딕셔너리
- `select_provider_and_model()` 내의 `providers` 목록
- 제공자 디스패치 (`if selected_provider == ...`)
- `--provider` 인수 선택지
- 제공자가 이러한 흐름을 지원하는 경우 로그인/로그아웃 선택지
- `_model_flow_<provider>()` 함수, 또는 적합한 경우 `_model_flow_api_key_provider()` 재사용

:::tip
`hermes_cli/setup.py`는 변경할 필요가 없습니다 — `main.py`에서 `select_provider_and_model()`을 호출하므로, 새 제공자는 `hermes model`과 `hermes setup` 모두에 자동으로 나타납니다.
:::

## 6단계: 보조(Auxiliary) 호출이 작동하도록 유지

여기서는 두 개의 파일이 중요합니다:

### `agent/auxiliary_client.py`

이것이 직접 API 키 제공자인 경우, 저렴하고 빠른 기본 보조 모델을 `_API_KEY_PROVIDER_AUX_MODELS`에 추가하세요.

보조 작업에는 다음과 같은 것들이 포함됩니다:

- 비전(vision) 요약
- 웹 추출 요약
- 컨텍스트 압축 요약
- 세션 검색 요약
- 메모리 플러시(flush)

제공자에게 적절한 보조 모델 기본값이 없으면 부가 작업이 잘못된 대체 항목을 사용하거나 예기치 않게 비싼 메인 모델을 사용할 수 있습니다.

### `agent/model_metadata.py`

토큰 예산 책정, 압축 임계값 및 한도가 합리적으로 유지되도록 제공자 모델의 컨텍스트 길이를 추가하세요.

## 7단계: 제공자가 네이티브인 경우 어댑터 및 `run_agent.py` 지원 추가

제공자가 단순한 채팅 완료가 아닌 경우, `agent/<provider>_adapter.py`에서 제공자 전용 로직을 격리하세요.

`run_agent.py`는 오케스트레이션(orchestration)에 초점을 맞추도록 유지하세요. 어댑터 헬퍼를 호출해야 하며, 파일 전체에 인라인으로 제공자 페이로드(payload)를 직접 구성해서는 안 됩니다.

네이티브 제공자는 일반적으로 다음 위치에서 작업이 필요합니다:

### 새로운 어댑터 파일

일반적인 책임:

- SDK / HTTP 클라이언트 빌드
- 토큰 확인
- OpenAI 스타일의 대화 메시지를 제공자의 요청 형식으로 변환
- 필요한 경우 도구 스키마 변환
- 제공자 응답을 `run_agent.py`가 예상하는 형태로 다시 정규화
- 사용량(usage) 및 종료 이유(finish-reason) 데이터 추출

### `run_agent.py`

`api_mode`를 검색하고 모든 스위치 포인트를 검사하세요. 최소한 다음을 확인하세요:

- `__init__`가 새로운 `api_mode`를 선택하는지
- 제공자에 대해 클라이언트 구성이 작동하는지
- `_build_api_kwargs()`가 요청 서식을 지정하는 방법을 알고 있는지
- `_interruptible_api_call()`이 올바른 클라이언트 호출로 디스패치하는지
- 인터럽트 / 클라이언트 재구성 경로가 작동하는지
- 응답 검증이 제공자의 형태를 수용하는지
- 종료 이유(finish-reason) 추출이 올바른지
- 토큰 사용량 추출이 올바른지
- 대체 모델 활성화가 새 제공자로 깔끔하게 전환될 수 있는지
- 요약 생성 및 메모리 플러시 경로가 여전히 작동하는지

또한 `run_agent.py`에서 `self.client.`를 검색하세요. 표준 OpenAI 클라이언트가 존재한다고 가정하는 모든 코드 경로는 네이티브 제공자가 다른 클라이언트 개체를 사용하거나 `self.client = None`인 경우 실패할 수 있습니다.

### 프롬프트 캐싱 및 제공자 전용 요청 필드

프롬프트 캐싱 및 제공자 전용 노브(knobs)는 회귀(regress)하기 쉽습니다.

이미 트리에 있는 예시:

- Anthropic에는 네이티브 프롬프트 캐싱 경로가 있습니다
- OpenRouter는 제공자 라우팅 필드를 가져옵니다
- 모든 제공자가 모든 요청 측 옵션을 수신해서는 안 됩니다

네이티브 제공자를 추가할 때 Hermes가 해당 제공자가 실제로 이해하는 필드만 전송하는지 다시 확인하세요.

## 8단계: 테스트

최소한 제공자 연결을 보호하는 테스트는 건드려야 합니다.

일반적인 위치:

- `tests/hermes_cli/test_runtime_provider_resolution.py`
- `tests/cli/test_cli_provider_resolution.py`
- `tests/hermes_cli/test_model_switch_custom_providers.py` (및 인접한 `tests/hermes_cli/test_model_switch_*.py`)
- `tests/hermes_cli/test_setup_model_provider.py`
- `tests/run_agent/test_provider_parity.py`
- `tests/run_agent/test_run_agent.py`
- 네이티브 제공자의 경우 `tests/test_<provider>_adapter.py`

문서화용 예제의 경우 정확한 파일 집합이 다를 수 있습니다. 핵심은 다음을 포함하는 것입니다:

- 인증 결정
- CLI 메뉴 / 제공자 선택
- 런타임 제공자 결정
- 에이전트 실행 경로
- provider:model 구문 분석
- 어댑터별 메시지 변환

xdist를 비활성화한 상태로 테스트를 실행하세요:

```bash
source venv/bin/activate
python -m pytest tests/hermes_cli/test_runtime_provider_resolution.py tests/cli/test_cli_provider_resolution.py tests/hermes_cli/test_setup_model_provider.py tests/run_agent/test_provider_parity.py -n0 -q
```

더 깊은 변경 사항의 경우 푸시하기 전에 전체 제품군을 실행하세요:

```bash
source venv/bin/activate
python -m pytest tests/ -n0 -q
```

## 9단계: 실시간 확인(Live verification)

테스트 후 실제 스모크 테스트(smoke test)를 실행하세요.

```bash
source venv/bin/activate
python -m hermes_cli.main chat -q "Say hello" --provider your-provider --model your-model
```

메뉴를 변경한 경우 대화형 흐름도 테스트하세요:

```bash
source venv/bin/activate
python -m hermes_cli.main model
python -m hermes_cli.main setup
```

네이티브 제공자의 경우 일반 텍스트 응답뿐만 아니라 도구 호출도 하나 이상 확인하세요.

## 10단계: 사용자 대상 문서 업데이트

제공자가 일급 옵션으로 제공될 예정인 경우 사용자 문서도 업데이트하세요:

- `website/docs/getting-started/quickstart.md`
- `website/docs/user-guide/configuration.md`
- `website/docs/reference/environment-variables.md`

개발자가 제공자를 완벽하게 연결했더라도, 사용자가 필요한 환경 변수나 설정 흐름을 찾지 못하게 방치할 수 있습니다.

## OpenAI 호환 제공자 체크리스트

제공자가 표준 채팅 완료(chat completions)인 경우 이것을 사용하세요.

- [ ] `hermes_cli/auth.py`에 `ProviderConfig` 추가됨
- [ ] `hermes_cli/auth.py` 및 `hermes_cli/models.py`에 별칭 추가됨
- [ ] `hermes_cli/models.py`에 모델 카탈로그 추가됨
- [ ] `hermes_cli/runtime_provider.py`에 런타임 분기 추가됨
- [ ] `hermes_cli/main.py`에 CLI 연결 추가됨 (setup.py는 자동으로 상속됨)
- [ ] `agent/auxiliary_client.py`에 보조 모델 추가됨
- [ ] `agent/model_metadata.py`에 컨텍스트 길이 추가됨
- [ ] 런타임 / CLI 테스트가 업데이트됨
- [ ] 사용자 문서가 업데이트됨

## 네이티브 제공자 체크리스트

제공자에 새로운 프로토콜 경로가 필요한 경우 이것을 사용하세요.

- [ ] OpenAI 호환 체크리스트의 모든 항목
- [ ] `agent/<provider>_adapter.py`에 어댑터 추가됨
- [ ] `run_agent.py`에서 새로운 `api_mode` 지원됨
- [ ] 인터럽트 / 재구성 경로 작동함
- [ ] 사용량 및 종료 이유 추출 작동함
- [ ] 대체(fallback) 경로 작동함
- [ ] 어댑터 테스트 추가됨
- [ ] 라이브 스모크 테스트 통과함

## 흔한 실수 (Common pitfalls)

### 1. 제공자를 인증에는 추가하지만 모델 파싱에는 추가하지 않음

이렇게 하면 자격 증명은 올바르게 확인되지만 `/model` 및 `provider:model` 입력은 실패합니다.

### 2. `config["model"]`이 문자열 또는 딕셔너리일 수 있다는 것을 잊어버림

많은 제공자 선택 코드는 두 가지 형태를 모두 정규화해야 합니다.

### 3. 내장 제공자가 필수적이라고 가정

서비스가 단지 OpenAI 호환인 경우, 사용자 지정 제공자로도 더 적은 유지 관리로 사용자 문제를 해결할 수 있습니다.

### 4. 보조(Auxiliary) 경로를 잊어버림

메인 채팅 경로는 작동하지만 보조 라우팅이 업데이트되지 않아 요약, 메모리 플러시 또는 비전 도우미가 실패할 수 있습니다.

### 5. `run_agent.py`에 숨어 있는 네이티브 제공자 분기

`api_mode` 및 `self.client.`를 검색하세요. 눈에 띄는 요청 경로가 유일한 경로라고 가정하지 마세요.

### 6. 다른 제공자에게 OpenRouter 전용 설정(knobs) 전송

제공자 라우팅과 같은 필드는 이를 지원하는 제공자에만 속해야 합니다.

### 7. `hermes model`은 업데이트하지만 `hermes setup`은 업데이트하지 않음

두 흐름 모두 제공자에 대해 알아야 합니다.

## 구현 중 유용한 검색 대상

제공자가 영향을 미치는 모든 위치를 찾고 있다면 다음 기호들을 검색하세요:

- `PROVIDER_REGISTRY`
- `_PROVIDER_ALIASES`
- `_PROVIDER_MODELS`
- `resolve_runtime_provider`
- `_model_flow_`
- `select_provider_and_model`
- `api_mode`
- `_API_KEY_PROVIDER_AUX_MODELS`
- `self.client.`

## 관련 문서

- [Provider Runtime Resolution](./provider-runtime.md)
- [Architecture](./architecture.md)
- [Contributing](./contributing.md)
