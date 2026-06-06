---
sidebar_position: 2
title: "환경 변수 (Environment Variables)"
description: "Hermes의 설정을 위한 환경 변수 전체 목록"
---

# 환경 변수 (Environment Variables)

Hermes는 프로필의 `~/.hermes/.env` 파일에서 환경 변수를 로드합니다. 일반적인 셸(shell) 환경 변수도 지원되지만, 보안을 위해 민감한 키는 `.env` 파일에 저장하는 것을 권장합니다.

Hermes를 시스템 서비스로 실행하는 경우, `hermes.service` 파일은 시작 시 `~/.hermes/.env`를 자동으로 읽습니다.

## 제공자 API 키 (Provider API Keys)

Hermes는 활성 설정이나 호출 시 선택된 제공자에 따라 올바른 API 키를 선택합니다.

| 변수명 | 설명 |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic 모델(예: Claude 3.5 Sonnet, Opus)용 |
| `OPENROUTER_API_KEY` | OpenRouter(수십 개의 모델 통합)를 통한 모델 라우팅용 |
| `OPENAI_API_KEY` | OpenAI(예: GPT-4o)용 |
| `GOOGLE_API_KEY` | Google AI Studio(Gemini 1.5 Pro)용. |
| `GROQ_API_KEY` | Groq(고속 Llama 3)용 |
| `TOGETHER_API_KEY` | Together AI용 |
| `FIREWORKS_API_KEY` | Fireworks AI용 |
| `MISTRAL_API_KEY` | Mistral(La Plateforme)용 |
| `DEEPSEEK_API_KEY` | DeepSeek용 |
| `XAI_API_KEY` | xAI (Grok)용 |
| `COHERE_API_KEY` | Cohere용 |
| `NVIDIA_API_KEY` | NVIDIA NIM 플랫폼용 |
| `QWEN_API_KEY` | Alibaba Cloud DashScope (Qwen 모델)용 |
| `OPENAI_BASE_URL` | OpenAI 호환 백엔드(예: 로컬 vLLM, Ollama, LM Studio)를 사용하는 경우 |
| `NOUS_API_KEY` | Nous Research 호스팅 모델 전용 (현재 내부용) |

:::tip
대부분의 제공자 설정은 `hermes config` CLI 마법사를 통해 관리할 수 있습니다. `config.yaml` 내의 제공자 설정은 환경 변수를 덮어씁니다.
:::

## 플랫폼 통합 (Platform Integrations)

이러한 키들은 메시징 플랫폼을 통해 Hermes에 연결할 때 사용됩니다. CLI 기반 실행에는 필요하지 않습니다.

### Telegram

| 변수명 | 설명 |
|---|---|
| `TELEGRAM_BOT_TOKEN` | @BotFather에서 발급받은 봇 토큰 |
| `TELEGRAM_ALLOWED_USERS` | 봇 사용이 허용된 텔레그램 사용자 ID의 쉼표로 구분된 목록(예: `1234567,9876543`) |
| `TELEGRAM_ADMIN_CHAT_ID` | 관리자 알림(예: 승인 요청, 크론 작업 결과)을 라우팅할 텔레그램 채팅 ID |
| `TELEGRAM_ALLOW_ALL_USERS` | `true`로 설정하면 모든 텔레그램 사용자가 봇에 접근할 수 있습니다. **주의: 공개 봇을 실행하지 않는 한 사용하지 마세요.** |

### Discord

| 변수명 | 설명 |
|---|---|
| `DISCORD_BOT_TOKEN` | Discord 개발자 포털에서 발급받은 봇 토큰 |
| `DISCORD_ALLOWED_USERS` | 봇 사용이 허용된 Discord 사용자 ID의 쉼표로 구분된 목록 |
| `DISCORD_ADMIN_CHANNEL_ID` | 관리자 알림을 라우팅할 Discord 채널 ID |
| `DISCORD_ALLOW_ALL_USERS` | `true`로 설정하면 모든 Discord 사용자가 봇에 접근할 수 있습니다. |

### Slack

| 변수명 | 설명 |
|---|---|
| `SLACK_BOT_TOKEN` | Slack 앱의 봇 사용자 OAuth 토큰 (xoxb-...) |
| `SLACK_APP_TOKEN` | Socket Mode 애플리케이션 레벨 토큰 (xapp-...) |
| `SLACK_ALLOWED_USERS` | 봇 사용이 허용된 Slack 사용자 ID의 쉼표로 구분된 목록 |
| `SLACK_ADMIN_CHANNEL_ID` | 관리자 알림을 라우팅할 Slack 채널 ID (또는 사용자 ID) |

### Signal

| 변수명 | 설명 |
|---|---|
| `SIGNAL_CLI_URL` | `signal-cli-rest-api` 인스턴스의 베이스 URL (예: `http://localhost:8080`) |
| `SIGNAL_PHONE_NUMBER` | Signal 계정의 등록된 전화번호 (봇의 번호, 예: `+1234567890`) |
| `SIGNAL_ALLOWED_USERS` | 봇 사용이 허용된 전화번호의 쉼표로 구분된 목록 |

### Matrix

| 변수명 | 설명 |
|---|---|
| `MATRIX_HOMESERVER_URL` | Matrix 홈서버의 베이스 URL (예: `https://matrix.org`) |
| `MATRIX_USER_ID` | 봇의 Matrix 계정 전체 사용자 ID (예: `@hermes:matrix.org`) |
| `MATRIX_ACCESS_TOKEN` | `MATRIX_USER_ID`에 해당하는 액세스 토큰 |
| `MATRIX_ALLOWED_USERS` | 봇 사용이 허용된 Matrix 사용자 ID의 쉼표로 구분된 목록 |

### WhatsApp

| 변수명 | 설명 |
|---|---|
| `WHATSAPP_TOKEN` | WhatsApp Business Cloud API 영구 액세스 토큰 |
| `WHATSAPP_PHONE_NUMBER_ID` | API 대시보드에 있는 발신자 전화번호 ID |
| `WHATSAPP_VERIFY_TOKEN` | 웹훅 확인을 위한 커스텀 토큰 |
| `WHATSAPP_ALLOWED_USERS` | 봇 사용이 허용된 전화번호(국가 코드 포함, `+` 제외)의 쉼표로 구분된 목록 |

## 도구 & 통합 (Tools & Integrations)

특정 [번들 도구(Bundled Tools)](./toolsets-reference)를 사용할 때 필요한 키입니다.

| 변수명 | 설명 |
|---|---|
| `SERPAPI_API_KEY` | `web_search` 도구를 사용한 웹 검색용 |
| `TAVILY_API_KEY` | `web_search` 도구를 사용할 때의 대안 웹 검색 제공자 |
| `FIRECRAWL_API_KEY` | `read_url_content`를 사용할 때 JavaScript가 렌더링된 웹사이트에서 마크다운 스크래핑을 수행하기 위함 |
| `GITHUB_TOKEN` | `github_tools`에서 리포지토리 상호작용(예: PR 생성, 이슈 읽기)을 수행하기 위함 |
| `GOOGLE_CALENDAR_CREDENTIALS` | `google_calendar` 도구용. 서비스 계정 JSON 경로 또는 자격 증명 문자열 |
| `NOTION_API_KEY` | `notion` 도구셋용 |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | `spotify` 도구셋용 |
| `HOMEASSISTANT_URL` / `HOMEASSISTANT_TOKEN` | Home Assistant 도구셋용 |

## 메모리 제공자 (Memory Providers)

[메모리 플러그인(Memory Plugins)](../user-guide/features/plugins.md)이 활성화된 경우 필요한 환경 변수입니다.

| 변수명 | 설명 |
|---|---|
| `HONCHO_API_KEY` | Honcho 기반 메모리 저장 및 검색용 |
| `VIKING_API_KEY` | OpenViking 장기 메모리 저장소용 |
| `MEM0_API_KEY` | Mem0 관리형 메모리 저장소용 |
| `SUPABASE_URL` / `SUPABASE_KEY` | 특정 자체 호스팅 메모리 제공자용 |

## 시스템 동작 (System Behavior)

| 변수명 | 설명 |
|---|---|
| `HERMES_HOME` | 기본 프로필 경로(`~/.hermes`)를 덮어씁니다. 시스템 단위 설치 시 유용합니다. |
| `HERMES_LOG_LEVEL` | 로깅을 제어합니다. 가능한 값: `DEBUG`, `INFO`(기본값), `WARNING`, `ERROR`. |
| `HERMES_ENVIRONMENT` | `production` 또는 `development`로 설정합니다. 오류 스택 트레이스 표시에 영향을 줍니다. |
| `HERMES_PORT` | 게이트웨이가 수신 대기하는 포트를 재정의합니다 (웹훅 기반 통합 시 기본값은 `8080`). |
| `HERMES_HOST` | 게이트웨이가 바인딩되는 인터페이스를 재정의합니다 (기본값: `127.0.0.1`, 외부에 노출하려면 `0.0.0.0` 사용). |
| `HERMES_WORKERS` | 비동기 작업 처리 워커 수를 지정합니다 (기본값: CPU 코어 수). |
| `HERMES_DEBUG_PROMPTS` | `true`로 설정하면 각 턴마다 LLM으로 전송되는 원시(raw) 시스템 프롬프트가 로그에 덤프됩니다. 매우 장황합니다(Verbose). |

## 특수 관리 컨트롤 (Special Admin Controls)

| 변수명 | 설명 |
|---|---|
| `GATEWAY_ALLOW_ALL_USERS` | `true`로 설정하면 플랫폼에 관계없이 **누구든지** 게이트웨이에 연결하고 에이전트를 스폰(spawn)할 수 있습니다. 로컬 테스트 이외의 용도로는 사용하지 마십시오. |
| `GATEWAY_ADMIN_CHANNEL` | 전역 알림(시스템 오류, 페어링 코드 생성)에 사용될 관리자 채널을 설정합니다. 형식: `telegram:123456789`. 설정되지 않은 경우 플랫폼별 변수(예: `TELEGRAM_ADMIN_CHAT_ID`)를 사용합니다. |
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | 자체 호스팅 대시보드를 위한 Nous Portal OAuth 클라이언트 ID. `hermes dashboard register`에 의해 자동 관리됩니다. |
| `HERMES_TELEMETRY_OPT_OUT` | `true`로 설정하면 기본 익명 시작 통계를 전송하지 않습니다(전송되는 경우: 플랫폼 버전에 대한 핑 전송). |
