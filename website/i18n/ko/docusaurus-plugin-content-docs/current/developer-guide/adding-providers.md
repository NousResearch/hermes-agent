---
sidebar_position: 5
title: "프로바이더 추가"
description: "Hermes Agent에 새 추론 프로바이더를 추가하는 방법 — 인증, 런타임 확인, CLI 흐름, 어댑터, 테스트 및 문서"
---

# 프로바이더 추가

Hermes는 커스텀 프로바이더 경로를 통해 모든 OpenAI 호환 엔드포인트와 이미 통신할 수 있습니다. 해당 서비스를 위한 일급 UX가 필요할 때만 내장 프로바이더를 추가하세요.

- 프로바이더별 인증 또는 토큰 갱신
- 엄선된 모델 카탈로그
- 설정 / `hermes model` 메뉴 항목
- `provider:model` 구문을 위한 프로바이더 별칭
- 어댑터가 필요한 비-OpenAI API 형식

프로바이더가 그저 "또 하나의 OpenAI 호환 base URL과 API 키"라면 이름이 지정된 커스텀 프로바이더만으로 충분할 수 있습니다.

## 개념 모델

내장 프로바이더는 몇 가지 계층에 걸쳐 일관되게 연결되어야 합니다.

1. `hermes_cli/auth.py`는 자격 증명을 찾는 방법을 결정합니다.
2. `hermes_cli/runtime_provider.py`는 이를 런타임 데이터로 변환합니다.
   - `provider`
   - `api_mode`
   - `base_url`
   - `api_key`
   - `source`
3. `run_agent.py`는 `api_mode`를 사용해 요청을 구성하고 전송하는 방법을 결정합니다.
4. `hermes_cli/models.py`와 `hermes_cli/main.py`은 CLI에 프로바이더가 표시되도록 합니다. (`hermes_cli/setup.py`은 자동으로 `main.py`에 위임하므로 여기서는 변경할 필요가 없습니다.)
5. `agent/auxiliary_client.py`와 `agent/model_metadata.py`는 부가 작업과 토큰 예산이 계속 정상 동작하도록 합니다.

핵심 추상화는 `api_mode`입니다.

- 대부분의 프로바이더는 `chat_completions`를 사용합니다.
- Codex는 `codex_responses`를 사용합니다.
- Anthropic은 `anthropic_messages`를 사용합니다.
- 새로운 비-OpenAI 프로토콜은 일반적으로 새 어댑터와 새로운 `api_mode` 분기를 추가해야 합니다.

### 도구 호출 와이어 형식

Hermes는 내부적으로 대화 기록을 OpenAI chat-completions 형식으로 저장합니다. 따라서 `chat_completions` 전송의 `convert_messages` / `convert_tools` (`agent/transports/chat_completions.py`)는 거의 동일한 형태이며, 다른 모든 전송은 이 형식에서 각자의 네이티브 프로토콜로 변환합니다. 형식의 표준 참고 자료 — JSON 스키마 `parameters`를 사용하는 `tools` 정의, 문자열화된 `function.arguments`를 포함하는 assistant `tool_calls` 항목, `tool_call_id`로 연결되는 `role: "tool"` 결과 메시지 — 는 [OpenAI chat completions API 레퍼런스](https://platform.openai.com/docs/api-reference/chat/create)입니다. 네이티브 어댑터를 작성할 때 해당 페이지는 변환의 입력 측을 정의하고, 프로바이더의 문서는 출력 측을 정의합니다.

## 먼저 구현 경로 선택하기

### 경로 A — OpenAI 호환 프로바이더

프로바이더가 표준 chat-completions 방식의 요청을 받을 때 사용합니다.

일반적인 작업:

- 인증 메타데이터 추가
- 모델 카탈로그 / 별칭 추가
- 런타임 확인 추가
- CLI 메뉴 연결
- 보조 모델 기본값 추가
- 테스트와 사용자 문서 추가

일반적으로 새 어댑터나 새 `api_mode`는 필요하지 않습니다.

### 경로 B — 네이티브 프로바이더

프로바이더가 OpenAI chat completions처럼 동작하지 않을 때 사용합니다.

현재 저장소에 있는 예시:

- `codex_responses`
- `anthropic_messages`

이 경로에는 경로 A의 모든 작업과 함께 다음이 포함됩니다.

- `agent/`의 프로바이더 어댑터
- 요청 구성, 디스패치, 사용량 추출, 인터럽트 처리, 응답 정규화를 위한 `run_agent.py` 분기
- 어댑터 테스트

## 파일 체크리스트

### 모든 내장 프로바이더에 필요한 항목

1. `hermes_cli/auth.py`
2. `hermes_cli/models.py`
3. `hermes_cli/runtime_provider.py`
4. `hermes_cli/main.py`
5. `agent/auxiliary_client.py`
6. `agent/model_metadata.py`
7. 테스트
8. `website/docs/` 아래의 사용자 문서

:::tip
`hermes_cli/setup.py`는 변경할 필요가 **없습니다**. 설정 마법사는 `main.py`의 `select_provider_and_model()`에 프로바이더/모델 선택을 위임하므로, 그곳에 추가한 프로바이더는 `hermes setup`에서도 자동으로 사용할 수 있습니다.
:::

### 네이티브 / 비-OpenAI 프로바이더에 추가로 필요한 항목

10. `agent/<provider>_adapter.py`
11. `run_agent.py`
12. 프로바이더 SDK가 필요하다면 `pyproject.toml`

## 빠른 경로: 단순 API 키 프로바이더

프로바이더가 단일 API 키로 인증하는 OpenAI 호환 엔드포인트일 뿐이라면, 아래 전체 체크리스트의 `auth.py`, `runtime_provider.py`, `main.py` 또는 다른 파일을 수정할 필요가 없습니다.

필요한 작업은 다음뿐입니다.

1. 다음을 포함하는 `plugins/model-providers/<your-provider>/` 아래의 플러그인 디렉터리:
   - `__init__.py` — 모듈 수준에서 `register_provider(profile)`을 호출
   - `plugin.yaml` — 매니페스트 (name, kind: model-provider, version, description)
2. 끝입니다. 프로바이더 플러그인은 `get_provider_profile()` 또는 `list_providers()`가 처음 호출될 때 자동으로 로드됩니다 — 번들 플러그인(이 저장소)과 `$HERMES_HOME/plugins/model-providers/`의 사용자 플러그인이 모두 검색됩니다.

플러그인을 추가하고 `register_provider()`를 호출하면 다음 항목이 자동으로 연결됩니다.

1. `auth.py`의 `PROVIDER_REGISTRY` 항목 (자격 증명 확인, 환경 변수 조회)
2. `chat_completions`로 설정된 `api_mode`
3. 설정 또는 선언된 환경 변수에서 가져오는 `base_url`
4. API 키를 우선순위 순서로 확인하는 `env_vars`
5. 프로바이더에 등록된 `fallback_models` 목록
6. `--provider` CLI 플래그가 프로바이더 ID를 받음
7. `hermes model` 메뉴에 프로바이더가 포함됨
8. `hermes setup` 마법사가 자동으로 `main.py`에 위임함
9. `provider:model` 별칭 구문이 동작함
10. 런타임 확인기가 올바른 `base_url`과 `api_key`를 반환함
11. `--provider <name>` CLI 플래그가 프로바이더 ID를 받음
12. 폴백 모델 활성화 시 프로바이더로 정상 전환할 수 있음

`$HERMES_HOME/plugins/model-providers/<name>/`의 사용자 플러그인은 같은 이름의 번들 플러그인보다 우선합니다 (`register_provider()`의 마지막 기록 우선 방식) — 따라서 서드파티 플러그인은 저장소를 수정하지 않고도 내장 프로파일을 몽키 패치하거나 교체할 수 있습니다.

필드 레퍼런스, 훅 관용구 및 엔드투엔드 예시는 `plugins/model-providers/nvidia/` 또는 `plugins/model-providers/gmi/`를 템플릿으로 참고하고, 전체 [모델 프로바이더 플러그인 가이드](/developer-guide/model-provider-plugin)를 확인하세요.

## 전체 경로: OAuth 및 복잡한 프로바이더

프로바이더에 다음 중 하나라도 필요하다면 아래의 전체 체크리스트를 사용하세요.

- OAuth 또는 토큰 갱신 (Nous Portal, Codex, Qwen Portal, Copilot)
- 새 어댑터가 필요한 비-OpenAI API 형식 (Anthropic Messages, Codex Responses)
- 커스텀 엔드포인트 감지 또는 다중 리전 탐색 (z.ai, Kimi)
- 엄선된 정적 모델 카탈로그 또는 실시간 `/models` 조회
- 별도의 인증 흐름을 사용하는 프로바이더별 `hermes model` 메뉴 항목

## 1단계: 표준 프로바이더 ID 하나 선택

프로바이더 ID 하나를 정하고 모든 곳에서 사용하세요.

저장소의 예시:

- `openai-codex`
- `kimi-coding`
- `minimax-cn`

같은 ID가 다음 위치에 나타나야 합니다.

- `hermes_cli/auth.py`의 `PROVIDER_REGISTRY`
- `hermes_cli/models.py`의 `_PROVIDER_LABELS`
- `hermes_cli/auth.py`와 `hermes_cli/models.py` 양쪽의 `_PROVIDER_ALIASES`
- `hermes_cli/main.py`의 CLI `--provider` 선택지
- 설정 / 모델 선택 분기
- 보조 모델 기본값
- 테스트

이 파일들에서 ID가 다르면 프로바이더가 반쯤만 연결된 것처럼 보입니다. 인증은 동작하지만 `/model`, 설정 또는 런타임 확인에서 조용히 누락될 수 있습니다.

## 2단계: `hermes_cli/auth.py`에 인증 메타데이터 추가

API 키 프로바이더라면 `PROVIDER_REGISTRY`에 다음 항목을 포함하는 `ProviderConfig`를 추가하세요.

- `id`
- `name`
- `auth_type="api_key"`
- `inference_base_url`
- `api_key_env_vars`
- 선택 사항인 `base_url_env_var`

`_PROVIDER_ALIASES`에도 별칭을 추가하세요.

기존 프로바이더를 템플릿으로 사용하세요.

- 단순 API 키 경로: Z.AI, MiniMax
- 엔드포인트 감지가 포함된 API 키 경로: Kimi, Z.AI
- 네이티브 토큰 확인: Anthropic
- OAuth / 인증 저장소 경로: Nous, OpenAI Codex

여기서 답해야 할 질문:

- Hermes가 어떤 환경 변수를 어떤 우선순위로 확인해야 하는가?
- 프로바이더에 base URL 재정의가 필요한가?
- 엔드포인트 탐색이나 토큰 갱신이 필요한가?
- 자격 증명이 없을 때 인증 오류에 어떤 메시지를 표시해야 하는가?

프로바이더에 단순히 "API 키 조회" 이상의 작업이 필요하다면, 관련 없는 분기에 로직을 억지로 넣지 말고 전용 자격 증명 확인기를 추가하세요.

## 3단계: `hermes_cli/models.py`에 모델 카탈로그와 별칭 추가

메뉴와 `provider:model` 구문에서 프로바이더가 동작하도록 프로바이더 카탈로그를 업데이트하세요.

일반적인 수정 항목:

- `_PROVIDER_MODELS`
- `_PROVIDER_LABELS`
- `_PROVIDER_ALIASES`
- `list_available_providers()` 내부의 프로바이더 표시 순서
- 프로바이더가 실시간 `/models` 조회를 지원한다면 `provider_model_ids()`

프로바이더가 실시간 모델 목록을 제공한다면 그것을 먼저 사용하고, 정적 폴백으로 `_PROVIDER_MODELS`를 유지하세요.

이 파일이 다음과 같은 입력을 동작하게 합니다.

```text
anthropic:claude-sonnet-4-6
kimi:model-name
```

여기에 별칭이 없으면 프로바이더 인증은 정상적으로 되더라도 `/model` 파싱에 실패할 수 있습니다.

## 4단계: `hermes_cli/runtime_provider.py`에서 런타임 데이터 확인

`resolve_runtime_provider()`는 CLI, 게이트웨이, cron, ACP 및 헬퍼 클라이언트가 공유하는 경로입니다.

최소한 다음을 포함하는 dict를 반환하는 분기를 추가하세요.

```python
{
    "provider": "your-provider",
    "api_mode": "chat_completions",  # or your native mode
    "base_url": "https://...",
    "api_key": "...",
    "source": "env|portal|auth-store|explicit",
    "requested_provider": requested_provider,
}
```

프로바이더가 OpenAI 호환이라면 `api_mode`는 보통 `chat_completions`로 유지해야 합니다.

API 키 우선순위에 주의하세요. Hermes에는 OpenRouter 키가 관련 없는 엔드포인트로 유출되지 않도록 하는 로직이 이미 있습니다. 새 프로바이더도 어떤 키를 어떤 base URL에 보낼지 똑같이 명시해야 합니다.

## 5단계: `hermes_cli/main.py`에서 CLI 연결

대화형 `hermes model` 흐름에 표시되기 전까지 프로바이더를 검색할 수 없습니다.

`hermes_cli/main.py`에서 다음을 업데이트하세요.

- `provider_labels` dict
- `select_provider_and_model()`의 `providers` 목록
- 프로바이더 디스패치 (`if selected_provider == ...`)
- `--provider` 인자 선택지
- 프로바이더가 해당 흐름을 지원한다면 로그인/로그아웃 선택지
- `_model_flow_<provider>()` 함수 또는 적합하다면 `_model_flow_api_key_provider()` 재사용

:::tip
`hermes_cli/setup.py`는 변경할 필요가 없습니다 — `select_provider_and_model()`을 `main.py`에서 호출하므로 새 프로바이더는 `hermes model`과 `hermes setup` 양쪽에 자동으로 표시됩니다.
:::

## 6단계: 보조 호출이 계속 동작하도록 유지

여기서는 두 파일이 중요합니다.

### `agent/auxiliary_client.py`

직접 API 키 프로바이더라면 `_API_KEY_PROVIDER_AUX_MODELS`에 저렴하고 빠른 기본 보조 모델을 추가하세요.

보조 작업에는 다음이 포함됩니다.

- 비전 요약
- 웹 추출 요약
- 컨텍스트 압축 요약
- 세션 검색 요약
- 메모리 플러시

프로바이더에 합리적인 보조 기본값이 없으면 부가 작업이 잘못 폴백하거나 예기치 않게 비용이 비싼 주 모델을 사용할 수 있습니다.

### `agent/model_metadata.py`

토큰 예산, 압축 임계값 및 제한이 적절하게 유지되도록 프로바이더 모델의 컨텍스트 길이를 추가하세요.

## 7단계: 프로바이더가 네이티브라면 어댑터와 `run_agent.py` 지원 추가

프로바이더가 일반 chat completions가 아니라면 프로바이더별 로직을 `agent/<provider>_adapter.py`에 격리하세요.

`run_agent.py`는 오케스트레이션에 집중하도록 유지하세요. 파일 곳곳에서 프로바이더 페이로드를 직접 구성하지 말고 어댑터 헬퍼를 호출해야 합니다.

네이티브 프로바이더는 일반적으로 다음 위치에서 작업이 필요합니다.

### 새 어댑터 파일

일반적인 책임:

- SDK / HTTP 클라이언트 구성
- 토큰 확인
- OpenAI 형식의 대화 메시지를 프로바이더 요청 형식으로 변환
- 필요한 경우 도구 스키마 변환
- 프로바이더 응답을 `run_agent.py`가 기대하는 형식으로 정규화
- 사용량 및 종료 이유 데이터 추출

### `run_agent.py`

`api_mode`를 검색하고 모든 분기 지점을 점검하세요. 최소한 다음을 확인해야 합니다.

- `__init__`이 새 `api_mode`를 선택함
- 프로바이더에서 클라이언트 생성이 동작함
- `_build_api_kwargs()`가 요청 형식을 올바르게 지정함
- `_interruptible_api_call()`이 올바른 클라이언트 호출로 디스패치함
- 인터럽트 / 클라이언트 재구성 경로가 동작함
- 응답 검증이 프로바이더의 형식을 허용함
- 종료 이유 추출이 올바름
- 토큰 사용량 추출이 올바름
- 폴백 모델 활성화 시 새 프로바이더로 정상 전환할 수 있음
- 요약 생성과 메모리 플러시 경로가 계속 동작함

또한 `run_agent.py`에서 `self.client.`를 검색하세요. 표준 OpenAI 클라이언트가 존재한다고 가정하는 모든 코드 경로는 네이티브 프로바이더가 다른 클라이언트 객체를 사용하거나 `self.client = None`일 때 깨질 수 있습니다.

### 프롬프트 캐싱과 프로바이더별 요청 필드

프롬프트 캐싱과 프로바이더별 옵션은 쉽게 회귀가 발생하는 부분입니다.

저장소에 이미 있는 예시:

- Anthropic은 네이티브 프롬프트 캐싱 경로를 사용함
- OpenRouter는 프로바이더 라우팅 필드를 받음
- 모든 프로바이더가 모든 요청 측 옵션을 받아야 하는 것은 아님

네이티브 프로바이더를 추가할 때 Hermes가 해당 프로바이더가 실제로 이해하는 필드만 전송하는지 다시 확인하세요.

## 8단계: 테스트

최소한 프로바이더 연결을 보호하는 테스트를 수정하세요.

일반적인 위치:

- `tests/hermes_cli/test_runtime_provider_resolution.py`
- `tests/cli/test_cli_provider_resolution.py`
- `tests/hermes_cli/test_model_switch_custom_providers.py` (및 인접한 `tests/hermes_cli/test_model_switch_*.py`)
- `tests/hermes_cli/test_setup_model_provider.py`
- `tests/run_agent/test_provider_parity.py`
- `tests/run_agent/test_run_agent.py`
- 네이티브 프로바이더라면 `tests/test_<provider>_adapter.py`

문서 전용 예시라면 정확한 파일 목록이 달라질 수 있습니다. 중요한 것은 다음을 검증하는 것입니다.

- 인증 확인
- CLI 메뉴 / 프로바이더 선택
- 런타임 프로바이더 확인
- 에이전트 실행 경로
- provider:model 파싱
- 어댑터별 메시지 변환

대상 테스트를 실행하세요 (또는 각 파일을 별도 subprocess에서 실행하는 `scripts/run_tests.sh`를 사용하세요).

```bash
source venv/bin/activate
python -m pytest tests/hermes_cli/test_runtime_provider_resolution.py tests/cli/test_cli_provider_resolution.py tests/hermes_cli/test_setup_model_provider.py tests/run_agent/test_provider_parity.py -q
```

더 깊은 변경이라면 푸시하기 전에 전체 테스트 스위트를 실행하세요.

```bash
source venv/bin/activate
python -m pytest tests/ -n0 -q
```

## 9단계: 실제 환경 확인

테스트 후 실제 스모크 테스트를 실행하세요.

```bash
source venv/bin/activate
python -m hermes_cli.main chat -q "Say hello" --provider your-provider --model your-model
```

메뉴를 변경했다면 대화형 흐름도 테스트하세요.

```bash
source venv/bin/activate
python -m hermes_cli.main model
python -m hermes_cli.main setup
```

네이티브 프로바이더라면 일반 텍스트 응답만이 아니라 최소 한 번의 도구 호출도 확인하세요.

## 10단계: 사용자 대상 문서 업데이트

프로바이더를 일급 선택지로 제공하려면 사용자 문서도 업데이트하세요.

- `website/docs/getting-started/quickstart.md`
- `website/docs/user-guide/configuration.md`
- `website/docs/reference/environment-variables.md`

개발자가 프로바이더를 완벽하게 연결해도 사용자가 필요한 환경 변수나 설정 흐름을 찾을 수 없게 남겨둘 수 있습니다.

## OpenAI 호환 프로바이더 체크리스트

프로바이더가 표준 chat completions를 사용할 때 적용하세요.

- [ ] `hermes_cli/auth.py`에 `ProviderConfig` 추가
- [ ] `hermes_cli/auth.py`와 `hermes_cli/models.py`에 별칭 추가
- [ ] `hermes_cli/models.py`에 모델 카탈로그 추가
- [ ] `hermes_cli/runtime_provider.py`에 런타임 분기 추가
- [ ] `hermes_cli/main.py`에 CLI 연결 추가 (`setup.py`는 자동으로 상속)
- [ ] `agent/auxiliary_client.py`에 보조 모델 추가
- [ ] `agent/model_metadata.py`에 컨텍스트 길이 추가
- [ ] 런타임 / CLI 테스트 업데이트
- [ ] 사용자 문서 업데이트

## 네이티브 프로바이더 체크리스트

새 프로토콜 경로가 필요한 프로바이더에 적용하세요.

- [ ] OpenAI 호환 체크리스트의 모든 항목
- [ ] `agent/<provider>_adapter.py`에 어댑터 추가
- [ ] `run_agent.py`에서 새 `api_mode` 지원
- [ ] 인터럽트 / 재구성 경로 동작
- [ ] 사용량 및 종료 이유 추출 동작
- [ ] 폴백 경로 동작
- [ ] 어댑터 테스트 추가
- [ ] 실제 스모크 테스트 통과

## 흔한 함정

### 1. 인증에는 프로바이더를 추가했지만 모델 파싱에는 추가하지 않음

이렇게 하면 자격 증명은 올바르게 확인되지만 `/model`과 `provider:model` 입력은 실패합니다.

### 2. `config["model"]`이 문자열 또는 dict일 수 있다는 점을 잊음

많은 프로바이더 선택 코드는 두 형식을 모두 정규화해야 합니다.

### 3. 내장 프로바이더가 반드시 필요하다고 가정함

서비스가 OpenAI 호환일 뿐이라면 커스텀 프로바이더로 더 적은 유지보수 비용으로 이미 사용자 문제를 해결할 수 있습니다.

### 4. 보조 경로를 잊음

보조 라우팅을 업데이트하지 않으면 주 채팅 경로는 동작하면서도 요약, 메모리 플러시 또는 비전 헬퍼가 실패할 수 있습니다.

### 5. `run_agent.py`에 숨어 있는 네이티브 프로바이더 분기

`api_mode`와 `self.client.`를 검색하세요. 눈에 보이는 요청 경로가 유일한 경로라고 가정하지 마세요.

### 6. OpenRouter 전용 옵션을 다른 프로바이더에 전송함

프로바이더 라우팅 같은 필드는 해당 기능을 지원하는 프로바이더에만 속합니다.

### 7. `hermes model`은 업데이트했지만 `hermes setup`은 업데이트하지 않음

두 흐름 모두 프로바이더를 알고 있어야 합니다.

## 구현 중 유용한 검색 대상

프로바이더가 연결되는 모든 위치를 찾고 있다면 다음 심볼을 검색하세요.

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

- [프로바이더 런타임 확인](./provider-runtime.md)
- [아키텍처](./architecture.md)
- [기여하기](./contributing.md)
