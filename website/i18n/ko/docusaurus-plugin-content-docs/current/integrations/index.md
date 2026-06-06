---
title: "통합 (Integrations)"
sidebar_label: "개요"
sidebar_position: 0
---

# 통합 (Integrations)

Hermes Agent는 AI 인퍼런스, 도구 서버, IDE 워크플로우, 프로그래밍 방식의 액세스 등을 위해 외부 시스템과 연결됩니다. 이러한 통합을 통해 Hermes가 할 수 있는 일과 실행될 수 있는 곳이 확장됩니다.

:::tip 여기서 시작하세요
시간이 없어서 단 하나의 통합만 설정해야 한다면 [Nous Portal](/integrations/nous-portal)을 설정하세요. 단일 OAuth 로그인으로 300개 이상의 모델과 4가지 Tool Gateway 도구(웹 검색, 이미지 생성, TTS, 브라우저 자동화)를 모두 이용할 수 있습니다.
:::

## AI 프로바이더 및 라우팅

Hermes는 다양한 AI 인퍼런스 프로바이더를 즉시 사용할 수 있도록 지원합니다. 대화형으로 구성하려면 `hermes model`을 사용하고, 그렇지 않으면 `config.yaml`에서 설정하세요.

- **[AI 프로바이더](/user-guide/features/provider-routing)** — OpenRouter, Anthropic, OpenAI, Google 및 기타 OpenAI 호환 엔드포인트. Hermes는 프로바이더별로 비전, 스트리밍, 도구 사용과 같은 기능을 자동 감지합니다.
- **[프로바이더 라우팅](/user-guide/features/provider-routing)** — OpenRouter 요청을 처리할 기본 프로바이더를 세밀하게 제어합니다. 정렬, 허용 목록, 차단 목록 및 명시적인 우선순위 지정을 통해 비용, 속도 또는 품질에 맞게 최적화하세요.
- **[대체 프로바이더 (Fallback Providers)](/user-guide/features/fallback-providers)** — 기본 모델에서 오류가 발생할 때 백업 LLM 프로바이더로 자동 장애 조치(failover)를 수행합니다. 기본 모델에 대한 대체 기능과 비전, 압축, 웹 추출과 같은 보조 작업을 위한 독립적인 대체 기능이 포함됩니다.

## 도구 서버 (MCP)

- **[MCP 서버](/user-guide/features/mcp)** — Model Context Protocol을 통해 외부 도구 서버를 Hermes에 연결합니다. 기본 Hermes 도구를 작성하지 않고도 GitHub, 데이터베이스, 파일 시스템, 브라우저 스택, 내부 API 등의 도구에 액세스할 수 있습니다. stdio 및 SSE 전송, 서버별 도구 필터링, 기능 인지형(capability-aware) 리소스/프롬프트 등록을 지원합니다.

## 웹 검색 백엔드

`web_search` 및 `web_extract` 도구는 `config.yaml` 또는 `hermes tools`를 통해 구성된 4가지 백엔드 프로바이더를 지원합니다:

| 백엔드 | 환경 변수 | 검색 | 추출 | 크롤링 |
|---------|---------|--------|---------|-------|
| **Firecrawl** (기본값) | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |

빠른 설정 예시:

```yaml
web:
  backend: firecrawl    # firecrawl | parallel | tavily | exa
```

`web.backend`가 설정되지 않은 경우, 사용 가능한 API 키를 기준으로 백엔드가 자동 감지됩니다. 자체 호스팅 Firecrawl도 `FIRECRAWL_API_URL`을 통해 지원됩니다.

## 브라우저 자동화

Hermes에는 웹사이트 탐색, 양식 작성 및 정보 추출을 위한 여러 백엔드 옵션이 포함된 완전한 브라우저 자동화 기능이 있습니다:

- **Browserbase** — 안티봇 도구, CAPTCHA 해결, 주거용(residential) 프록시가 포함된 관리형 클라우드 브라우저
- **Browser Use** — 대안적인 클라우드 브라우저 프로바이더
- **로컬 Chromium 계열 CDP** — `/browser connect`를 사용하여 실행 중인 Chrome, Brave, Chromium 또는 Edge 브라우저에 연결
- **로컬 Chromium** — `agent-browser` CLI를 통한 헤드리스 로컬 브라우저

설정 및 사용 방법은 [브라우저 자동화](/user-guide/features/browser)를 참조하세요.

## 음성 및 TTS 프로바이더

모든 메시징 플랫폼에서 텍스트 음성 변환(TTS) 및 음성 텍스트 변환(STT) 지원:

| 프로바이더 | 품질 | 비용 | API 키 |
|----------|---------|------|---------|
| **Edge TTS** (기본값) | 좋음 | 무료 | 필요 없음 |
| **ElevenLabs** | 우수 | 유료 | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | 좋음 | 유료 | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax** | 좋음 | 유료 | `MINIMAX_API_KEY` |
| **xAI TTS** | 좋음 | 유료 | `XAI_API_KEY` |
| **NeuTTS** | 좋음 | 무료 | 필요 없음 |

음성 텍스트 변환(STT)은 로컬 faster-whisper (무료, 기기 내에서 실행), 로컬 명령어 래퍼, Groq, OpenAI Whisper API, Mistral 및 xAI의 6가지 프로바이더를 지원합니다. 음성 메시지 전사는 Telegram, Discord, WhatsApp 등 다양한 메시징 플랫폼에서 작동합니다. 자세한 내용은 [음성 및 TTS](/user-guide/features/tts)와 [음성 모드](/user-guide/features/voice-mode)를 참조하세요.

## IDE 및 에디터 통합

- **[IDE 통합 (ACP)](/user-guide/features/acp)** — VS Code, Zed, JetBrains 등 ACP 호환 에디터 내에서 Hermes Agent를 사용하세요. Hermes가 ACP 서버로 실행되어 채팅 메시지, 도구 활동, 파일 diff 및 터미널 명령어를 에디터 안에 렌더링합니다.

## 프로그래밍 방식 액세스

- **[API 서버](/user-guide/features/api-server)** — Hermes를 OpenAI 호환 HTTP 엔드포인트로 노출합니다. OpenAI 형식을 사용하는 모든 프런트엔드 (Open WebUI, LobeChat, LibreChat, NextChat, ChatBox)는 Hermes를 백엔드로 연결하여 전체 도구 세트를 사용할 수 있습니다.

## 메모리 및 개인화

- **[내장 메모리](/user-guide/features/memory)** — `MEMORY.md` 및 `USER.md` 파일을 통해 지속되고 큐레이팅되는 메모리입니다. 에이전트는 세션 전반에 걸쳐 유지되는 개인 노트 및 사용자 프로필 데이터의 제한된 저장소를 유지 관리합니다.
- **[메모리 프로바이더](/user-guide/features/memory-providers)** — 더 깊은 개인화를 위해 외부 메모리 백엔드를 연결합니다. Honcho (대화 추론), OpenViking (계층형 검색), Mem0 (클라우드 추출), Hindsight (지식 그래프), Holographic (로컬 SQLite), RetainDB (하이브리드 검색), ByteRover (CLI 기반) 및 Supermemory 등 8가지 프로바이더가 지원됩니다.

## 메시징 플랫폼

Hermes는 동일한 `gateway` 하위 시스템을 통해 모두 구성되는 27개 이상의 메시징 플랫폼에서 게이트웨이 봇으로 실행됩니다:

- **[Telegram](/user-guide/messaging/telegram)**, **[Discord](/user-guide/messaging/discord)**, **[Slack](/user-guide/messaging/slack)**, **[WhatsApp](/user-guide/messaging/whatsapp)**, **[Signal](/user-guide/messaging/signal)**, **[Matrix](/user-guide/messaging/matrix)**, **[Mattermost](/user-guide/messaging/mattermost)**, **[이메일 (Email)](/user-guide/messaging/email)**, **[SMS](/user-guide/messaging/sms)**, **[DingTalk](/user-guide/messaging/dingtalk)**, **[Feishu/Lark](/user-guide/messaging/feishu)**, **[WeCom](/user-guide/messaging/wecom)**, **[WeCom Callback](/user-guide/messaging/wecom-callback)**, **[Weixin](/user-guide/messaging/weixin)**, **[BlueBubbles](/user-guide/messaging/bluebubbles)**, **[QQ Bot](/user-guide/messaging/qqbot)**, **[Yuanbao](/user-guide/messaging/yuanbao)**, **[Home Assistant](/user-guide/messaging/homeassistant)**, **[Microsoft Teams](/user-guide/messaging/teams)**, **[Microsoft Teams Meetings](/user-guide/messaging/teams-meetings)**, **[Microsoft Graph 웹훅](/user-guide/messaging/msgraph-webhook)**, **[Google Chat](/user-guide/messaging/google_chat)**, **[LINE](/user-guide/messaging/line)**, **[ntfy](/user-guide/messaging/ntfy)**, **[SimpleX](/user-guide/messaging/simplex)**, **[Open WebUI](/user-guide/messaging/open-webui)**, **[웹훅 (Webhooks)](/user-guide/messaging/webhooks)**

플랫폼 비교 표 및 설정 가이드는 [메시징 게이트웨이 개요](/user-guide/messaging)를 참조하세요.

## 홈 자동화 (Home Automation)

- **[Home Assistant](/user-guide/messaging/homeassistant)** — 4개의 전용 도구(`ha_list_entities`, `ha_get_state`, `ha_list_services`, `ha_call_service`)를 통해 스마트 홈 기기를 제어합니다. `HASS_TOKEN`이 구성되면 Home Assistant 도구 세트가 자동으로 활성화됩니다.

## 플러그인

- **[플러그인 시스템](/user-guide/features/plugins)** — 코어 코드를 수정하지 않고 사용자 지정 도구, 라이프사이클 훅 및 CLI 명령어로 Hermes를 확장하세요. 플러그인은 `~/.hermes/plugins/`, 프로젝트 로컬 `.hermes/plugins/` 및 pip로 설치된 진입점(entry points)에서 발견됩니다.
- **[플러그인 만들기](/guides/build-a-hermes-plugin)** — 도구, 훅 및 CLI 명령어를 사용하여 Hermes 플러그인을 만드는 단계별 가이드입니다.

## 학습 및 평가

- **[일괄 처리 (Batch Processing)](/user-guide/features/batch-processing)** — 수백 개의 프롬프트에 걸쳐 에이전트를 병렬로 실행하여 학습 데이터 생성 또는 평가를 위한 구조화된 ShareGPT 형식의 궤적(trajectory) 데이터를 생성합니다.
