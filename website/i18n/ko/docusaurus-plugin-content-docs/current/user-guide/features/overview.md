---
title: "기능 개요"
sidebar_label: "개요"
sidebar_position: 1
---

# 기능 개요 (Features Overview)

Hermes Agent는 기본적인 채팅을 훨씬 뛰어넘는 풍부한 기능 세트를 포함하고 있습니다. 지속적인 메모리와 파일 인식 컨텍스트부터 브라우저 자동화와 음성 대화까지, 이러한 기능들이 함께 작동하여 Hermes를 강력한 자율 어시스턴트로 만듭니다.

:::tip 어디서부터 시작해야 할지 모르시겠나요?
`hermes setup --portal` 명령은 하나의 명령으로 모델 제공자와 네 가지 Tool Gateway 도구(웹 검색, 이미지 생성, TTS, 브라우저)를 모두 설정합니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 핵심 기능 (Core)

- **[도구 및 도구 세트 (Tools & Toolsets)](tools.md)** — 도구는 에이전트의 기능을 확장하는 함수입니다. 플랫폼별로 활성화하거나 비활성화할 수 있는 논리적인 도구 세트로 구성되며 웹 검색, 터미널 실행, 파일 편집, 메모리, 위임 등을 다룹니다.
- **[스킬 시스템 (Skills System)](skills.md)** — 에이전트가 필요할 때 로드할 수 있는 온디맨드 지식 문서입니다. 스킬은 토큰 사용량을 최소화하기 위해 점진적 공개(progressive disclosure) 패턴을 따르며, [agentskills.io](https://agentskills.io/specification) 개방형 표준과 호환됩니다.
- **[지속적인 메모리 (Persistent Memory)](memory.md)** — 세션 간에 지속되는 제한된 범위의 큐레이션된 메모리입니다. Hermes는 `MEMORY.md`와 `USER.md`를 통해 선호도, 프로젝트, 환경 및 학습한 내용을 기억합니다.
- **[컨텍스트 파일 (Context Files)](context-files.md)** — Hermes는 프로젝트에서 동작하는 방식을 형성하는 프로젝트 컨텍스트 파일(`.hermes.md`, `AGENTS.md`, `CLAUDE.md`, `SOUL.md`, `.cursorrules`)을 자동으로 검색하고 로드합니다.
- **[컨텍스트 참조 (Context References)](context-references.md)** — `@` 뒤에 참조를 입력하여 파일, 폴더, git diff 및 URL을 메시지에 직접 주입할 수 있습니다. Hermes는 참조를 인라인으로 확장하고 내용을 자동으로 덧붙입니다.
- **[체크포인트 (Checkpoints)](../checkpoints-and-rollback.md)** — Hermes는 파일 변경을 수행하기 전에 작업 디렉토리의 스냅샷을 자동으로 생성하여, 문제가 발생했을 때 `/rollback`으로 롤백할 수 있는 안전망을 제공합니다.

## 자동화 (Automation)

- **[예약된 작업 (Scheduled Tasks (Cron))](cron.md)** — 자연어 또는 cron 표현식으로 자동으로 실행할 작업을 예약합니다. 작업은 스킬을 첨부하고, 결과를 모든 플랫폼에 전달하며, 일시 중지/재개/편집 작업을 지원할 수 있습니다.
- **[서브에이전트 위임 (Subagent Delegation)](delegation.md)** — `delegate_task` 도구는 격리된 컨텍스트, 제한된 도구 세트, 고유한 터미널 세션을 가진 하위 에이전트 인스턴스를 생성합니다. 병렬 작업 흐름을 위해 기본적으로 3개의 동시 서브에이전트를 실행합니다(구성 가능).
- **[코드 실행 (Code Execution)](code-execution.md)** — `execute_code` 도구를 사용하면 에이전트가 Hermes 도구를 프로그래밍 방식으로 호출하는 Python 스크립트를 작성하여, 샌드박스화된 RPC 실행을 통해 다단계 워크플로우를 단일 LLM 턴으로 압축할 수 있습니다.
- **[이벤트 훅 (Event Hooks)](hooks.md)** — 핵심 수명 주기 지점에서 사용자 정의 코드를 실행합니다. 게이트웨이 훅은 로깅, 알림 및 웹훅을 처리하고, 플러그인 훅은 도구 가로채기, 메트릭 및 가드레일을 처리합니다.
- **[일괄 처리 (Batch Processing)](batch-processing.md)** — 수백 또는 수천 개의 프롬프트에 걸쳐 Hermes 에이전트를 병렬로 실행하여, 훈련 데이터 생성 또는 평가를 위한 구조화된 ShareGPT 형식의 궤적(trajectory) 데이터를 생성합니다.

## 미디어 및 웹 (Media & Web)

- **[음성 모드 (Voice Mode)](voice-mode.md)** — CLI 및 메시징 플랫폼 전반의 전체 음성 상호작용. 마이크를 사용하여 에이전트에게 말하고, 음성 답변을 듣고, Discord 음성 채널에서 실시간 음성 대화를 할 수 있습니다.
- **[브라우저 자동화 (Browser Automation)](browser.md)** — 여러 백엔드가 있는 전체 브라우저 자동화: Browserbase 클라우드, Browser Use 클라우드, CDP를 통한 로컬 Chrome/Brave/Chromium/Edge, 또는 로컬 Chromium. 웹사이트를 탐색하고, 양식을 채우고, 정보를 추출합니다.
- **[비전 및 이미지 붙여넣기 (Vision & Image Paste)](vision.md)** — 멀티모달 비전 지원. 클립보드에서 이미지를 CLI로 붙여넣고 비전 기능이 있는 모델을 사용하여 에이전트에게 이를 분석, 설명 또는 작업하도록 요청하세요.
- **[이미지 생성 (Image Generation)](image-generation.md)** — FAL.ai를 사용하여 텍스트 프롬프트에서 이미지를 생성합니다. 9가지 모델(FLUX 2 Klein/Pro, GPT-Image 1.5/2, Nano Banana Pro, Ideogram V3, Recraft V4 Pro, Qwen, Z-Image Turbo)을 지원하며 `hermes tools`를 통해 하나를 선택할 수 있습니다.
- **[음성 및 TTS (Voice & TTS)](tts.md)** — 모든 메시징 플랫폼에서 텍스트 음성 변환(TTS) 출력 및 음성 메시지 전사. 10가지 기본 제공자 옵션: Edge TTS(무료), ElevenLabs, OpenAI TTS, MiniMax, Mistral Voxtral, Google Gemini, xAI, NeuTTS, KittenTTS, Piper — 또한 모든 로컬 TTS CLI를 위한 사용자 정의 명령 제공자를 지원합니다.

## 통합 (Integrations)

- **[MCP 통합 (MCP Integration)](mcp.md)** — stdio 또는 HTTP 전송을 통해 MCP 서버에 연결합니다. 기본 Hermes 도구를 작성하지 않고도 GitHub, 데이터베이스, 파일 시스템 및 내부 API에서 외부 도구에 액세스할 수 있습니다. 서버별 도구 필터링 및 샘플링 지원이 포함됩니다.
- **[제공자 라우팅 (Provider Routing)](provider-routing.md)** — 어떤 AI 제공자가 요청을 처리할지 세밀하게 제어합니다. 정렬, 화이트리스트, 블랙리스트, 우선순위 정렬을 통해 비용, 속도 또는 품질을 최적화합니다.
- **[대체 제공자 (Fallback Providers)](fallback-providers.md)** — 기본 모델에서 오류가 발생할 때 자동 페일오버하여 백업 LLM 제공자로 전환합니다. 비전 및 압축과 같은 보조 작업을 위한 독립적인 대체 제공자도 포함됩니다.
- **[자격 증명 풀 (Credential Pools)](credential-pools.md)** — 동일한 제공자에 대한 여러 키에 걸쳐 API 호출을 분산합니다. 속도 제한이나 오류 발생 시 자동으로 순환(rotation)됩니다.
- **[프롬프트 캐싱 (Prompt caching)](../configuration#prompt-caching)** — 기본 Anthropic, OpenRouter, Nous Portal에서 Claude를 위해 내장된 세션 간 1시간 접두사 캐시입니다. 항상 켜져 있으며 구성이 필요하지 않습니다.
- **[메모리 제공자 (Memory Providers)](memory-providers.md)** — 외부 메모리 백엔드(Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory)를 연결하여 내장 메모리 시스템을 넘어선 세션 간 사용자 모델링 및 개인화를 수행합니다.
- **[API 서버 (API Server)](api-server.md)** — Hermes를 OpenAI 호환 HTTP 엔드포인트로 노출합니다. Open WebUI, LobeChat, LibreChat 등 OpenAI 형식을 사용하는 모든 프런트엔드를 연결하세요.
- **[IDE 통합 (ACP)](acp.md)** — VS Code, Zed, JetBrains 등 ACP 호환 편집기 내부에서 Hermes를 사용합니다. 채팅, 도구 활동, 파일 diff, 터미널 명령이 편집기 내부에 렌더링됩니다.
- **[일괄 처리 (Batch Processing)](batch-processing.md)** — CLI에서 다수의 프롬프트나 작업을 병렬로 실행하여 구조화된 출력과 궤적 캡처를 수행하며, 이는 평가(evals) 또는 하위 훈련 파이프라인에 적합합니다.

## 커스터마이징 (Customization)

- **[성격 및 SOUL.md (Personality & SOUL.md)](personality.md)** — 에이전트의 성격을 완전히 커스터마이징할 수 있습니다. `SOUL.md`는 시스템 프롬프트의 가장 첫 번째 항목이 되는 기본 정체성 파일이며, 세션당 내장 또는 사용자 지정 `/personality` 프리셋으로 교체할 수 있습니다.
- **[스킨 및 테마 (Skins & Themes)](skins.md)** — 배너 색상, 스피너(spinner) 모양 및 동사, 응답 상자 라벨, 브랜딩 텍스트 및 도구 활동 접두사 등 CLI의 시각적 표현을 커스터마이징합니다.
- **[플러그인 (Plugins)](plugins.md)** — 코어 코드를 수정하지 않고 사용자 지정 도구, 훅 및 통합을 추가합니다. 세 가지 플러그인 유형: 일반 플러그인(도구/훅), 메모리 제공자(세션 간 지식) 및 컨텍스트 엔진(대체 컨텍스트 관리). 통합된 `hermes plugins` 대화형 UI를 통해 관리됩니다.
