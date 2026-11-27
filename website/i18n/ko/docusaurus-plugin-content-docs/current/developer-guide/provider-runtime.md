---
sidebar_position: 4
title: "프로바이더 런타임 해석"
description: "Hermes가 런타임에 프로바이더, 자격 증명, API 모드 및 보조 모델을 해석하는 방법"
---

# 프로바이더 런타임 해석

Hermes에는 다음 전반에서 사용하는 공유 프로바이더 런타임 해석기가 있습니다.

- CLI
- 게이트웨이
- cron 작업
- ACP
- 보조 모델 호출

주요 구현:

- `hermes_cli/runtime_provider.py` — 자격 증명 해석, 사용자 지정 엔드포인트 런타임 해석
- `hermes_cli/auth.py` — 프로바이더 레지스트리, `resolve_provider()`
- `hermes_cli/model_switch.py` — 공유 `/model` 전환 파이프라인(CLI + 게이트웨이)
- `agent/auxiliary_client.py` — 보조 모델 라우팅
- `providers/` — ABC + 레지스트리 진입점(`ProviderProfile`, `register_provider`, `get_provider_profile`, `list_providers`)
- `plugins/model-providers/<name>/` — `api_mode`, `base_url`, `env_vars`, `fallback_models`를 선언하고 최초 접근 시 레지스트리에 자신을 등록하는 프로바이더별 플러그인(번들 포함). `$HERMES_HOME/plugins/model-providers/<name>/`의 사용자 플러그인이 같은 이름의 번들 플러그인보다 우선합니다.

`providers/`의 `get_provider_profile()`은 주어진 프로바이더 id에 대한 `ProviderProfile`을 반환합니다. `runtime_provider.py`는 해석 시 이 함수를 호출하여 여러 파일에 해당 데이터를 중복하지 않고 정식 `base_url`, `env_vars` 우선순위 목록, `api_mode`, `fallback_models`를 가져옵니다. `register_provider()`를 호출하는 새 플러그인을 `plugins/model-providers/<your-provider>/`(또는 `$HERMES_HOME/plugins/model-providers/<your-provider>/`) 아래에 추가하면 `runtime_provider.py`가 이를 선택할 수 있습니다. 해석기 자체에 분기문은 필요하지 않습니다.

새 일급 추론 프로바이더를 추가하려는 경우 이 페이지와 함께 [프로바이더 추가](./adding-providers.md) 및 [모델 프로바이더 플러그인 가이드](./model-provider-plugin.md)를 읽으세요.

## 해석 우선순위

상위 수준에서 프로바이더 해석은 다음을 사용합니다.

1. 명시적인 CLI/런타임 요청
2. `config.yaml` 모델/프로바이더 설정
3. 환경 변수
4. 프로바이더별 기본값 또는 자동 해석

이 순서가 중요한 이유는 Hermes가 일반 실행에서 저장된 모델/프로바이더 선택을 기준값으로 취급하기 때문입니다. 이를 통해 오래된 셸 export가 사용자가 `hermes model`에서 마지막으로 선택한 엔드포인트를 조용히 덮어쓰는 것을 방지합니다.

## 프로바이더

현재 프로바이더 제품군에는 다음이 포함됩니다(번들 전체 목록은 `plugins/model-providers/`를 참조하세요).

- AI Gateway (Vercel)
- OpenRouter
- Nous Portal
- OpenAI Codex
- Copilot / Copilot ACP
- Anthropic (네이티브)
- Google / Gemini (`gemini`)
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
- Custom (`provider: custom`) — 모든 OpenAI 호환 엔드포인트를 위한 일급 프로바이더
- 이름이 지정된 사용자 지정 프로바이더(`config.yaml`의 `providers:` dict; 이전 버전과의 호환성을 위해 기존 `custom_providers` 목록도 계속 읽음)

## 런타임 해석 결과

런타임 해석기는 다음과 같은 데이터를 반환합니다.

- `provider`
- `api_mode`
- `base_url`
- `api_key`
- `source`
- 만료/갱신 정보와 같은 프로바이더별 메타데이터

## 중요한 이유

이 해석기 덕분에 Hermes는 다음 사이에서 인증/런타임 로직을 공유할 수 있습니다.

- `hermes chat`
- 게이트웨이 메시지 처리
- 새 세션에서 실행되는 cron 작업
- ACP 편집기 세션
- 보조 모델 작업

## AI Gateway

`~/.hermes/.env`에 `AI_GATEWAY_API_KEY`를 설정하고 `--provider ai-gateway`로 실행하세요. Hermes는 게이트웨이의 `/models` 엔드포인트에서 사용 가능한 모델을 가져온 뒤 도구 사용을 지원하는 언어 모델만 필터링합니다.

## OpenRouter, AI Gateway 및 사용자 지정 OpenAI 호환 base URL

Hermes에는 여러 프로바이더 키가 존재할 때(예: `OPENROUTER_API_KEY`, `AI_GATEWAY_API_KEY`, `OPENAI_API_KEY`) 잘못된 API 키가 사용자 지정 엔드포인트로 유출되지 않도록 하는 로직이 있습니다.

각 프로바이더의 API 키는 자체 base URL에 한정됩니다.

- `OPENROUTER_API_KEY`는 `openrouter.ai` 엔드포인트로만 전송됩니다.
- `AI_GATEWAY_API_KEY`는 `ai-gateway.vercel.sh` 엔드포인트로만 전송됩니다.
- `OPENAI_API_KEY`는 사용자 지정 엔드포인트에 사용되며 대체 키로도 사용됩니다.

Hermes는 다음도 구분합니다.

- 사용자가 선택한 실제 사용자 지정 엔드포인트
- 사용자 지정 엔드포인트가 설정되지 않았을 때 사용되는 OpenRouter 대체 경로

이 구분은 특히 다음에 중요합니다.

- 로컬 모델 서버
- OpenRouter/AI Gateway가 아닌 OpenAI 호환 API
- 설정을 다시 실행하지 않고 프로바이더 전환
- 현재 셸에서 `OPENAI_BASE_URL`을 export하지 않았더라도 계속 작동해야 하는 설정에 저장된 사용자 지정 엔드포인트

## 네이티브 Anthropic 경로

Anthropic은 이제 단순히 "OpenRouter를 통한" 프로바이더가 아닙니다.

프로바이더 해석이 `anthropic`을 선택하면 Hermes는 다음을 사용합니다.

- `api_mode = anthropic_messages`
- 네이티브 Anthropic Messages API
- 변환을 담당하는 `agent/anthropic_adapter.py`

네이티브 Anthropic의 자격 증명 해석은 이제 두 가지가 모두 있을 때 복사된 env 토큰보다 갱신 가능한 Claude Code 자격 증명을 우선합니다. 실제로는 다음을 의미합니다.

- 갱신 가능한 인증이 포함된 Claude Code 자격 증명 파일을 우선 소스로 취급
- 수동 `ANTHROPIC_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN` 값은 명시적 재정의로 계속 작동
- Hermes는 네이티브 Messages API 호출 전에 Anthropic 자격 증명 갱신을 사전 확인
- Hermes는 대체 경로로 Anthropic 클라이언트를 다시 구성한 뒤 401에서 한 번 재시도

## OpenAI Codex 경로

Codex는 별도의 Responses API 경로를 사용합니다.

- `api_mode = codex_responses`
- 전용 자격 증명 해석 및 인증 저장소 지원

## 보조 모델 라우팅

다음과 같은 보조 작업은 주 대화 모델과 다른 자체 프로바이더/모델 라우팅을 사용할 수 있습니다.

- 비전
- 웹 추출 요약
- 컨텍스트 압축 요약
- 스킬 허브 작업
- MCP 헬퍼 작업
- 메모리 플러시

보조 작업이 프로바이더 `main`으로 설정되면 Hermes는 일반 채팅과 동일한 공유 런타임 경로를 통해 이를 해석합니다. 실제로는 다음을 의미합니다.

- 환경 변수 기반 사용자 지정 엔드포인트가 계속 작동
- `hermes model` / `config.yaml`을 통해 저장한 사용자 지정 엔드포인트도 계속 작동
- 보조 라우팅이 실제로 저장된 사용자 지정 엔드포인트와 OpenRouter 대체 경로를 구분

## 대체 모델

Hermes는 기본 모델에서 오류가 발생할 때 순서대로 시도하는 `(provider, model)` 항목 목록인 설정 가능한 대체 프로바이더 체인을 지원합니다. 이전 버전과의 호환성을 위해 기존의 단일 쌍 `fallback_model` dict도 계속 허용되며(처음 저장할 때 마이그레이션됨),

### 내부 작동 방식

1. **저장**: `AIAgent.__init__`이 `fallback_model` dict를 저장하고 `_fallback_activated = False`로 설정합니다.

2. **트리거 지점**: `run_agent.py`의 기본 재시도 루프에서 `_try_activate_fallback()`이 세 곳에서 호출됩니다.
   - 잘못된 API 응답(None choices, content 누락)에 대한 최대 재시도 후
   - 재시도할 수 없는 클라이언트 오류(HTTP 401, 403, 404) 발생 시
   - 일시적 오류(HTTP 429, 500, 502, 503)에 대한 최대 재시도 후

3. **활성화 흐름**(`_try_activate_fallback`):
   - 이미 활성화되었거나 설정되지 않은 경우 즉시 `False` 반환
   - `auxiliary_client.py`의 `resolve_provider_client()`를 호출하여 적절한 인증으로 새 클라이언트 생성
   - `api_mode` 결정: openai-codex는 `codex_responses`, anthropic은 `anthropic_messages`, 그 외는 `chat_completions`
   - 제자리에서 교체: `self.model`, `self.provider`, `self.base_url`, `self.api_mode`, `self.client`, `self._client_kwargs`
   - anthropic 대체의 경우 OpenAI 호환 클라이언트 대신 네이티브 Anthropic 클라이언트 생성
   - 프롬프트 캐싱 재평가( OpenRouter의 Claude 모델에서는 활성화)
   - `_fallback_activated = True` 설정 — 다시 실행되지 않도록 방지
   - 재시도 횟수를 0으로 초기화하고 루프 계속

4. **설정 흐름**:
   - CLI: `hermes_cli/fallback_config.get_fallback_chain()`을 통해 대체 체인을 읽고 → `AIAgent(fallback_model=...)`에 전달
   - 게이트웨이: `gateway/run.py._load_fallback_model()`이 `config.yaml`을 읽고 → `AIAgent`에 전달
   - 검증: `provider`와 `model` 키가 모두 비어 있지 않아야 하며, 그렇지 않으면 대체 기능이 비활성화됩니다.

### 대체를 지원하지 않는 항목

- **서브에이전트 위임**(`tools/delegate_tool.py`): 서브에이전트는 부모의 프로바이더를 상속하지만 대체 설정은 상속하지 않습니다.
- **보조 작업**: 자체적으로 독립된 프로바이더 자동 감지 체인을 사용합니다(위의 보조 모델 라우팅 참조).

Cron 작업은 대체를 **지원합니다**: `run_job()`은 `config.yaml`에서 `fallback_providers`(또는 기존 `fallback_model`)를 읽고 이를 `AIAgent(fallback_model=...)`에 전달하여 게이트웨이의 `_load_fallback_model()` 패턴과 일치시킵니다. [Cron 내부](./cron-internals.md)를 참조하세요.

### 테스트 범위

대체 동작은 여러 테스트 모음에서 검증됩니다.

- `tests/run_agent/test_fallback_credential_isolation.py` — 기본 프로바이더와 대체 프로바이더 간 자격 증명 격리
- `tests/hermes_cli/test_fallback_cmd.py` — `/fallback` CLI 명령
- `tests/gateway/test_fallback_eviction.py` — 실패한 프로바이더의 게이트웨이 제거

## 관련 문서

- [에이전트 루프 내부](./agent-loop.md)
- [ACP 내부](./acp-internals.md)
- [컨텍스트 압축 및 프롬프트 캐싱](./context-compression-and-caching.md)
