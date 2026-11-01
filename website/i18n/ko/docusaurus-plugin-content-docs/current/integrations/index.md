---
title: "통합"
sidebar_label: "개요"
sidebar_position: 0
---

# 통합

Hermes Agent는 AI 추론, 도구 서버, IDE 워크플로, 프로그래밍 방식 액세스 등을 위해 외부 시스템에 연결됩니다. 이러한 통합을 통해 Hermes가 할 수 있는 일과 실행할 수 있는 환경이 확장됩니다.

:::tip 여기서 시작하세요
통합을 하나만 설정할 시간이라면 [Nous Portal](/integrations/nous-portal)을 설정하세요. 한 번의 OAuth 로그인으로 300개 이상의 모델과 4개의 Tool Gateway 도구(웹 검색, 이미지 생성, TTS, 브라우저 자동화)를 모두 사용할 수 있습니다.
:::

## AI 제공업체 및 라우팅

Hermes는 기본적으로 여러 AI 추론 제공업체를 지원합니다. `hermes model`을 사용해 대화형으로 구성하거나 `config.yaml`에 설정할 수 있습니다.

- **[AI 제공업체](/integrations/providers)** — OpenRouter, Anthropic, OpenAI, Google 및 모든 OpenAI 호환 엔드포인트를 지원합니다. Hermes는 제공업체별 비전, 스트리밍, 도구 사용 등의 기능을 자동으로 감지합니다.
- **[제공업체 라우팅](/user-guide/features/provider-routing)** — OpenRouter 요청을 처리할 실제 제공업체를 세밀하게 제어합니다. 정렬, 허용 목록, 차단 목록, 명시적 우선순위 순서로 비용, 속도 또는 품질을 최적화할 수 있습니다.
- **[대체 제공업체](/user-guide/features/fallback-providers)** — 기본 모델에서 오류가 발생하면 백업 LLM 제공업체로 자동 전환합니다. 기본 모델 대체 기능과 비전, 압축, 웹 추출을 위한 독립적인 보조 작업 대체 기능을 포함합니다.

## 도구 서버(MCP)

- **[MCP 서버](/user-guide/features/mcp)** — Model Context Protocol을 통해 Hermes를 외부 도구 서버에 연결합니다. 네이티브 Hermes 도구를 작성하지 않고도 GitHub, 데이터베이스, 파일 시스템, 브라우저 스택, 내부 API 등의 도구에 액세스할 수 있습니다. stdio 및 SSE 전송, 서버별 도구 필터링, 기능 인식 리소스/프롬프트 등록을 지원합니다.

## 웹 검색 백엔드

`web_search` 및 `web_extract` 도구는 8개의 백엔드 제공업체를 지원하며, `config.yaml` 또는 `hermes tools`로 구성합니다.

| 백엔드 | 환경 변수 | 검색 | 추출 | 크롤링 |
|---------|---------|--------|---------|-------|
| **Firecrawl** (기본값) | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — |
| **Brave** (무료 티어) | `BRAVE_SEARCH_API_KEY` | ✔ | — | — |
| **DuckDuckGo** (ddgs) | _(없음)_ | ✔ | — | — |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — |
| **xAI** | `XAI_API_KEY` | ✔ | — | — |

간단한 설정 예시:

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | xai
```

`web.backend`가 설정되지 않으면 사용 가능한 API 키를 기준으로 백엔드가 자동 감지됩니다. `FIRECRAWL_API_URL`을 통한 자체 호스팅 Firecrawl도 지원합니다.

## 브라우저 자동화

Hermes에는 웹사이트 탐색, 양식 작성, 정보 추출을 위한 여러 백엔드 옵션과 완전한 브라우저 자동화 기능이 포함되어 있습니다.

- **Browserbase** — 봇 방지 도구, CAPTCHA 해결, 주거용 프록시를 제공하는 관리형 클라우드 브라우저
- **Browser Use** — 대체 클라우드 브라우저 제공업체
- **로컬 Chromium 계열 CDP** — `/browser connect`를 사용해 실행 중인 Chrome, Brave, Chromium 또는 Edge 브라우저에 연결
- **로컬 Chromium** — `agent-browser` CLI를 통한 헤드리스 로컬 브라우저

설정 및 사용 방법은 [브라우저 자동화](/user-guide/features/browser)를 참조하세요.

## 음성 및 TTS 제공업체

모든 메시징 플랫폼에서 텍스트 음성 변환과 음성 텍스트 변환을 지원합니다.

| 제공업체 | 품질 | 비용 | API 키 |
|----------|---------|------|---------|
| **Edge TTS** (기본값) | 좋음 | 무료 | 필요 없음 |
| **ElevenLabs** | 뛰어남 | 유료 | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | 좋음 | 유료 | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax** | 좋음 | 유료 | `MINIMAX_API_KEY` |
| **xAI TTS** | 좋음 | 유료 | `XAI_API_KEY` |
| **NeuTTS** | 좋음 | 무료 | 필요 없음 |

음성 텍스트 변환은 로컬 faster-whisper(무료, 기기에서 실행), 로컬 명령 래퍼, Groq, OpenAI Whisper API, Mistral, xAI, ElevenLabs Scribe, DeepInfra의 8개 제공업체를 지원합니다. 음성 메시지 전사는 Telegram, Discord, WhatsApp 및 기타 메시징 플랫폼에서 작동합니다. 자세한 내용은 [음성 및 TTS](/user-guide/features/tts) 및 [음성 모드](/user-guide/features/voice-mode)를 참조하세요.

## IDE 및 편집기 통합

- **[IDE 통합(ACP)](/user-guide/features/acp)** — VS Code, Zed, JetBrains와 같은 ACP 호환 편집기 안에서 Hermes Agent를 사용합니다. Hermes는 ACP 서버로 실행되며 편집기 안에 채팅 메시지, 도구 활동, 파일 차이 및 터미널 명령을 표시합니다.

## 프로그래밍 방식 액세스

- **[API 서버](/user-guide/features/api-server)** — Hermes를 OpenAI 호환 HTTP 엔드포인트로 노출합니다. Open WebUI, LobeChat, LibreChat, NextChat, ChatBox 등 OpenAI 형식을 지원하는 모든 프론트엔드가 연결되어 전체 도구 세트를 갖춘 백엔드로 Hermes를 사용할 수 있습니다.

## 메모리 및 개인화

- **[내장 메모리](/user-guide/features/memory)** — `MEMORY.md` 및 `USER.md` 파일을 통한 영구적이고 선별된 메모리입니다. 에이전트는 세션이 지나도 유지되는 개인 메모와 사용자 프로필 데이터의 제한된 저장소를 관리합니다.
- **[메모리 제공업체](/user-guide/features/memory-providers)** — 더 깊은 개인화를 위해 외부 메모리 백엔드를 연결합니다. Honcho(변증법적 추론), OpenViking(계층형 검색), Mem0(클라우드 추출), Hindsight(지식 그래프), Holographic(로컬 SQLite), RetainDB(하이브리드 검색), ByteRover(CLI 기반), Supermemory의 8개 제공업체를 지원합니다.

## 메시징 플랫폼

Hermes는 27개 이상의 메시징 플랫폼에서 게이트웨이 봇으로 실행되며, 모두 동일한 `gateway` 하위 시스템을 통해 구성됩니다.

- **[Telegram](/user-guide/messaging/telegram)**, **[Discord](/user-guide/messaging/discord)**, **[Slack](/user-guide/messaging/slack)**, **[WhatsApp](/user-guide/messaging/whatsapp)**, **[Signal](/user-guide/messaging/signal)**, **[Matrix](/user-guide/messaging/matrix)**, **[Mattermost](/user-guide/messaging/mattermost)**, **[Email](/user-guide/messaging/email)**, **[SMS](/user-guide/messaging/sms)**, **[DingTalk](/user-guide/messaging/dingtalk)**, **[Feishu/Lark](/user-guide/messaging/feishu)**, **[WeCom](/user-guide/messaging/wecom)**, **[WeCom Callback](/user-guide/messaging/wecom-callback)**, **[Weixin](/user-guide/messaging/weixin)**, **[BlueBubbles](/user-guide/messaging/bluebubbles)**, **[Buzz](/user-guide/messaging/buzz)**, **[QQ Bot](/user-guide/messaging/qqbot)**, **[Yuanbao](/user-guide/messaging/yuanbao)**, **[Home Assistant](/user-guide/messaging/homeassistant)**, **[Microsoft Teams](/user-guide/messaging/teams)**, **[Microsoft Teams Meetings](/user-guide/messaging/teams-meetings)**, **[Microsoft Graph Webhook](/user-guide/messaging/msgraph-webhook)**, **[Google Chat](/user-guide/messaging/google_chat)**, **[LINE](/user-guide/messaging/line)**, **[ntfy](/user-guide/messaging/ntfy)**, **[SimpleX](/user-guide/messaging/simplex)**, **[Open WebUI](/user-guide/messaging/open-webui)**, **[웹훅](/user-guide/messaging/webhooks)**

플랫폼 비교표와 설정 안내는 [메시징 게이트웨이 개요](/user-guide/messaging)를 참조하세요.

### 빠른 연결 링크

주요 플랫폼에는 표준적인 "봇/앱 만들기" URL이 있으며, 일부는 올바른 양식을 미리 열도록 매개변수를 허용합니다. 콘솔에서 찾느라 시간을 보내지 말고 바로 이동하세요.

| 플랫폼 | 직접 링크 | 열리는 항목 |
|----------|-------------|---------------|
| **Telegram** | [t.me/BotFather](https://t.me/BotFather) | BotFather와 채팅 — `/newbot`을 보내 봇 토큰 발급 |
| **Discord** | [discord.com/developers/applications?new_application=true](https://discord.com/developers/applications?new_application=true) | **New Application** 대화 상자가 미리 열려 있는 Developer Portal |
| **Slack** | [api.slack.com/apps?new_app=1](https://api.slack.com/apps?new_app=1) | **Create New App** 대화 상자 — *From an app manifest*를 선택하고 `hermes slack manifest --agent-view`가 생성한 매니페스트를 붙여넣기 |
| **LINE** | [developers.line.biz/console](https://developers.line.biz/console/) | Messaging API 채널을 만드는 LINE Developers Console |
| **Feishu/Lark** | [open.feishu.cn/app](https://open.feishu.cn/app) | 사용자 지정 앱을 만드는 Feishu 오픈 플랫폼 콘솔 |

각 플랫폼의 설정 페이지에서 해당 페이지에 도착한 뒤 수행할 작업을 안내합니다.

## 협업 워크스페이스

- **[Buzz](/integrations/buzz)** — Block의 Nostr 기반 인간+에이전트 워크스페이스입니다. 세 가지 통합 경로를 제공합니다. Buzz Desktop이 Hermes를 관리형 ACP 런타임으로 실행하거나, `buzz-acp` 릴레이 브리지가 서버 측에서 Hermes ID 서버를 호스팅하거나, 네이티브 게이트웨이 플랫폼이 완전한 Hermes 메모리/스킬/승인/cron 기능으로 Buzz 채널에 참여할 수 있습니다. 개요 페이지에서 세 경로를 비교합니다.

## 홈 자동화

- **[Home Assistant](/user-guide/messaging/homeassistant)** — 전용 도구 4개(`ha_list_entities`, `ha_get_state`, `ha_list_services`, `ha_call_service`)로 스마트 홈 기기를 제어합니다. `HASS_TOKEN`이 구성되면 Home Assistant 도구 세트가 자동으로 활성화됩니다.

## 플러그인

- **[플러그인 시스템](/user-guide/features/plugins)** — 핵심 코드를 수정하지 않고 사용자 지정 도구, 수명 주기 훅, CLI 명령으로 Hermes를 확장합니다. 플러그인은 `~/.hermes/plugins/`, 프로젝트 로컬 `.hermes/plugins/`, pip 설치 엔트리 포인트에서 검색됩니다.
- **[플러그인 빌드](/developer-guide/plugins)** — 도구, 훅, CLI 명령을 포함한 Hermes 플러그인을 만드는 단계별 안내입니다.

## 학습 및 평가

- **[배치 처리](/user-guide/features/batch-processing)** — 수백 개의 프롬프트에 에이전트를 병렬로 실행하여 학습 데이터 생성 또는 평가를 위한 구조화된 ShareGPT 형식의 궤적 데이터를 생성합니다.
