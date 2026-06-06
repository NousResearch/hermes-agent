---
sidebar_position: 4
title: "제공자 런타임 확인 (Provider Runtime Resolution)"
description: "Hermes가 런타임에 제공자, 자격 증명, API 모드 및 보조 모델을 확인하는 방법"
---

# 제공자 런타임 확인 (Provider Runtime Resolution)

Hermes는 다음 영역에서 공유되는 제공자 런타임 리졸버를 갖추고 있습니다:

- CLI
- 게이트웨이
- 크론 작업
- ACP
- 보조 모델 호출

기본 구현:

- `hermes_cli/runtime_provider.py` — 자격 증명 확인, `_resolve_custom_runtime()`
- `hermes_cli/auth.py` — 제공자 레지스트리, `resolve_provider()`
- `hermes_cli/model_switch.py` — 공유 `/model` 전환 파이프라인 (CLI + 게이트웨이)
- `agent/auxiliary_client.py` — 보조 모델 라우팅
- `providers/` — ABC + 레지스트리 진입점 (`ProviderProfile`, `register_provider`, `get_provider_profile`, `list_providers`)
- `plugins/model-providers/<name>/` — `api_mode`, `base_url`, `env_vars`, `fallback_models`를 선언하고 첫 액세스 시 레지스트리에 자체 등록되는 제공자별 플러그인(번들). `$HERMES_HOME/plugins/model-providers/<name>/`에 있는 사용자 플러그인은 동일한 이름의 번들 플러그인을 재정의합니다.

`providers/`의 `get_provider_profile()`은 주어진 제공자 ID에 대한 `ProviderProfile`을 반환합니다. `runtime_provider.py`는 확인 시점에 이를 호출하여 여러 파일에서 데이터를 복제할 필요 없이 정식 `base_url`, `env_vars` 우선순위 목록, `api_mode` 및 `fallback_models`를 가져옵니다. `plugins/model-providers/<your-provider>/` (또는 `$HERMES_HOME/plugins/model-providers/<your-provider>/`) 아래에 `register_provider()`를 호출하는 새 플러그인을 추가하는 것만으로 `runtime_provider.py`가 이를 인식할 수 있으며 리졸버 자체에 분기(branch)를 추가할 필요가 없습니다.

새로운 퍼스트 클래스 추론 제공자를 추가하려는 경우 이 페이지와 함께 [제공자 추가하기](./adding-providers.md) 및 [모델 제공자 플러그인 가이드](./model-provider-plugin.md)를 읽어보세요.

## 확인 우선순위 (Resolution precedence)

높은 수준에서 제공자 확인은 다음을 사용합니다:

1. 명시적 CLI/런타임 요청
2. `config.yaml` 모델/제공자 구성
3. 환경 변수
4. 제공자별 기본값 또는 자동 확인

이 순서가 중요한 이유는 Hermes가 저장된 모델/제공자 선택을 정상 실행 시의 진실 공급원(source of truth)으로 간주하기 때문입니다. 이렇게 하면 사용자가 이전에 `hermes model`에서 선택한 엔드포인트를 오래된 쉘 export가 조용히 재정의하는 것을 방지할 수 있습니다.

## 제공자 (Providers)

현재 제공자 제품군에는 다음이 포함됩니다 (`plugins/model-providers/`에서 번들로 제공되는 전체 세트를 확인하세요):

- OpenRouter
- Nous Portal
- OpenAI Codex
- Copilot / Copilot ACP
- Anthropic (기본 지원)
- Google / Gemini (`gemini`, `google-gemini-cli`)
- Alibaba / DashScope (`alibaba`, `alibaba-coding-plan`)
- DeepSeek
- Z.AI
- Kimi / Moonshot (`kimi-coding`, `kimi-coding-cn`)
- MiniMax (`minimax`, `minimax-cn`, `minimax-oauth`)
- Kilo Code
- Hugging Face
- OpenCode Zen / OpenCode Go
- AWS Bedrock
- Azure Foundry
- NVIDIA NIM
- xAI (Grok)
- Arcee
- GMI Cloud
- StepFun
- Qwen OAuth
- Xiaomi
- Ollama Cloud
- LM Studio
- Tencent TokenHub
- Custom (`provider: custom`) — 모든 OpenAI 호환 엔드포인트를 위한 퍼스트 클래스 제공자
- Named custom providers (`config.yaml`의 `custom_providers` 목록)

## 런타임 확인의 결과 (Output of runtime resolution)

런타임 리졸버는 다음과 같은 데이터를 반환합니다:

- `provider`
- `api_mode`
- `base_url`
- `api_key`
- `source`
- 만료/새로 고침 정보와 같은 제공자별 메타데이터

## 이것이 왜 중요할까요? (Why this matters)

이 리졸버는 Hermes가 다음 영역 사이에서 인증/런타임 로직을 공유할 수 있는 주된 이유입니다:

- `hermes chat`
- 게이트웨이 메시지 처리
- 새로운 세션에서 실행되는 크론 작업
- ACP 에디터 세션
- 보조 모델 작업

## OpenRouter 및 사용자 지정 OpenAI 호환 기본 URL

Hermes는 여러 제공자 키가 존재할 때(예: `OPENROUTER_API_KEY` 및 `OPENAI_API_KEY`) 사용자 지정 엔드포인트에 잘못된 API 키가 유출되는 것을 방지하기 위한 로직을 포함하고 있습니다.

각 제공자의 API 키는 자체 기본 URL로 범위가 지정됩니다:

- `OPENROUTER_API_KEY`는 `openrouter.ai` 엔드포인트로만 전송됩니다.
- `OPENAI_API_KEY`는 사용자 지정 엔드포인트 및 대체 수단(fallback)으로 사용됩니다.

Hermes는 또한 다음을 구분합니다:

- 사용자가 선택한 실제 사용자 지정 엔드포인트
- 사용자 지정 엔드포인트가 구성되지 않았을 때 사용되는 OpenRouter 대체 경로

이러한 구분은 특히 다음에 중요합니다:

- 로컬 모델 서버
- OpenRouter가 아닌 OpenAI 호환 API
- 설정을 다시 실행하지 않고 제공자 전환
- 현재 쉘에 `OPENAI_BASE_URL`이 내보내지지(exported) 않은 경우에도 계속 작동해야 하는 설정 저장된 사용자 지정 엔드포인트

## Anthropic 기본 경로 (Native Anthropic path)

Anthropic은 더 이상 단순한 "OpenRouter를 통한" 연결이 아닙니다.

제공자 확인 시 `anthropic`을 선택하면 Hermes는 다음을 사용합니다:

- `api_mode = anthropic_messages`
- 기본 Anthropic Messages API
- 변환을 위한 `agent/anthropic_adapter.py`

네이티브 Anthropic의 자격 증명 확인은 이제 두 자격 증명이 모두 있을 때 복사된 환경 토큰보다 새로 고침 가능한 Claude Code 자격 증명을 선호합니다. 실제로는 다음과 같은 의미입니다:

- Claude Code 자격 증명 파일에 새로 고침 가능한 인증이 포함된 경우 우선적인 소스로 취급됩니다.
- 수동 `ANTHROPIC_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN` 값은 여전히 명시적 재정의로 작동합니다.
- Hermes는 네이티브 Messages API 호출 전에 Anthropic 자격 증명 새로 고침을 사전에 수행(preflights)합니다.
- Hermes는 여전히 대체 경로로써 Anthropic 클라이언트를 다시 빌드한 후 401 오류 시 한 번 재시도합니다.

## OpenAI Codex 경로

Codex는 별도의 Responses API 경로를 사용합니다:

- `api_mode = codex_responses`
- 전용 자격 증명 확인 및 인증 저장소 지원

## 보조 모델 라우팅 (Auxiliary model routing)

다음과 같은 보조 작업은 주 대화형 모델 대신 자체 제공자/모델 라우팅을 사용할 수 있습니다:

- 비전 (vision)
- 웹 추출 요약
- 컨텍스트 압축 요약
- 스킬 허브 작업
- MCP 도우미 작업
- 메모리 플러시(flush)

보조 작업이 `main` 제공자로 구성된 경우 Hermes는 일반 채팅과 동일한 공유 런타임 경로를 통해 이를 확인합니다. 실제로는 다음과 같은 의미입니다:

- 환경 변수 기반의 사용자 지정 엔드포인트가 여전히 작동합니다.
- `hermes model` / `config.yaml`을 통해 저장된 사용자 지정 엔드포인트도 작동합니다.
- 보조 라우팅은 실제 저장된 사용자 지정 엔드포인트와 OpenRouter 대체 수단을 구분할 수 있습니다.

## 대체 모델 (Fallback models)

Hermes는 구성된 폴백(대체) 제공자 체인을 지원합니다. 기본 모델에 오류가 발생할 때 순서대로 시도되는 `(provider, model)` 항목 목록입니다. 이전 버전과의 호환성을 위해 기존의 단일 쌍 `fallback_model` 딕셔너리도 여전히 허용됩니다 (첫 쓰기 시 마이그레이션 됨).

### 내부 작동 방식

1. **저장**: `AIAgent.__init__`은 `fallback_model` 딕셔너리를 저장하고 `_fallback_activated = False`를 설정합니다.

2. **트리거 시점**: `_try_activate_fallback()`은 `run_agent.py`의 메인 재시도 루프 내 세 곳에서 호출됩니다:
   - 잘못된 API 응답(None choices, 내용 누락)에 대한 최대 재시도 후
   - 재시도할 수 없는 클라이언트 오류(HTTP 401, 403, 404) 발생 시
   - 일시적인 오류(HTTP 429, 500, 502, 503)에 대한 최대 재시도 후

3. **활성화 흐름** (`_try_activate_fallback`):
   - 이미 활성화되어 있거나 구성되지 않은 경우 즉시 `False`를 반환합니다.
   - 적절한 인증을 갖춘 새 클라이언트를 빌드하기 위해 `auxiliary_client.py`에서 `resolve_provider_client()`를 호출합니다.
   - `api_mode` 결정: openai-codex의 경우 `codex_responses`, anthropic의 경우 `anthropic_messages`, 그 외 모든 경우 `chat_completions`.
   - 제자리 스왑(in-place swap): `self.model`, `self.provider`, `self.base_url`, `self.api_mode`, `self.client`, `self._client_kwargs`.
   - Anthropic 폴백의 경우: OpenAI 호환 대신 네이티브 Anthropic 클라이언트를 빌드합니다.
   - 프롬프트 캐싱을 재평가합니다 (OpenRouter의 Claude 모델에 활성화 됨).
   - `_fallback_activated = True`를 설정합니다 — 다시 실행되는 것을 방지합니다.
   - 재시도 횟수를 0으로 재설정하고 루프를 계속 진행합니다.

4. **설정 흐름**:
   - CLI: `cli.py`가 `CLI_CONFIG["fallback_model"]`을 읽음 → `AIAgent(fallback_model=...)`에 전달
   - 게이트웨이: `gateway/run.py._load_fallback_model()`이 `config.yaml`을 읽음 → `AIAgent`에 전달
   - 유효성 검사: `provider`와 `model` 키가 모두 비어있지 않아야 하며, 그렇지 않으면 폴백이 비활성화됩니다.

### 대체가 지원되지 않는 경우 (What does NOT support fallback)

- **하위 에이전트 위임** (`tools/delegate_tool.py`): 하위 에이전트는 부모의 제공자를 상속하지만 대체 구성은 상속하지 않습니다.
- **보조 작업**: 자체적으로 독립적인 제공자 자동 감지 체인을 사용합니다 (위의 보조 모델 라우팅 참조).

크론 작업은 대체를 **지원합니다**: `run_job()`은 게이트웨이의 `_load_fallback_model()` 패턴과 일치하게 `config.yaml`에서 `fallback_providers` (또는 기존 `fallback_model`)를 읽고 이를 `AIAgent(fallback_model=...)`에 전달합니다. [크론 내부](./cron-internals.md) 문서를 참조하세요.

### 테스트 커버리지

대체 동작은 여러 스위트에서 테스트됩니다:

- `tests/run_agent/test_fallback_credential_isolation.py` — 기본 제공자와 대체 제공자 간의 자격 증명 격리
- `tests/hermes_cli/test_fallback_cmd.py` — `/fallback` CLI 명령어
- `tests/gateway/test_fallback_eviction.py` — 실패한 제공자의 게이트웨이 축출(eviction)

## 관련 문서 (Related docs)

- [에이전트 루프 내부](./agent-loop.md)
- [ACP 내부](./acp-internals.md)
- [컨텍스트 압축 및 프롬프트 캐싱](./context-compression-and-caching.md)
