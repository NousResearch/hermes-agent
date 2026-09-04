---
title: "기능 개요"
sidebar_label: "개요"
sidebar_position: 1
---

# 기능 개요

Hermes Agent는 기본적인 채팅을 훨씬 넘어서는 풍부한 기능을 제공합니다. 지속적인 메모리와 파일을 인식하는 컨텍스트부터 브라우저 자동화와 음성 대화까지, 이러한 기능들이 함께 작동하여 Hermes를 강력한 자율형 어시스턴트로 만들어 줍니다.

:::tip 어디서부터 시작해야 할지 모르겠나요?
`hermes setup --portal`을 실행하면 하나의 명령으로 모델 제공업체와 네 가지 Tool Gateway 도구(웹 검색, 이미지 생성, TTS, 브라우저)를 모두 설정할 수 있습니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 핵심

- **[도구 및 도구 세트](tools.md)** — 도구는 에이전트의 기능을 확장하는 함수입니다. 웹 검색, 터미널 실행, 파일 편집, 메모리, 위임 등을 다루는 논리적인 도구 세트로 구성되며 플랫폼별로 활성화하거나 비활성화할 수 있습니다.
- **[Skills 시스템](skills.md)** — 에이전트가 필요할 때 로드할 수 있는 온디맨드 지식 문서입니다. Skills는 토큰 사용량을 최소화하기 위해 점진적 공개 패턴을 따르며 [agentskills.io](https://agentskills.io/specification) 오픈 표준과 호환됩니다.
- **[지속적인 메모리](memory.md)** — 세션 간 유지되는 제한적이고 선별된 메모리입니다. Hermes는 `MEMORY.md`와 `USER.md`를 통해 사용자의 선호, 프로젝트, 환경, 학습한 내용을 기억합니다.
- **[컨텍스트 파일](context-files.md)** — Hermes는 프로젝트에서 동작 방식을 결정하는 컨텍스트 파일(`.hermes.md`, `AGENTS.md`, `CLAUDE.md`, `SOUL.md`, `.cursorrules`)을 자동으로 찾아 로드합니다.
- **[컨텍스트 참조](context-references.md)** — `@` 뒤에 참조를 입력하여 파일, 폴더, git diff, URL을 메시지에 직접 주입합니다. Hermes는 참조를 인라인으로 확장하고 콘텐츠를 자동으로 덧붙입니다.
- **[체크포인트](../checkpoints-and-rollback.md)** — Hermes는 파일을 변경하기 전에 작업 디렉터리의 스냅샷을 자동으로 생성하므로, 문제가 생기면 `/rollback`으로 되돌릴 수 있습니다.

## 자동화

- **[예약 작업 (Cron)](cron.md)** — 자연어 또는 cron 표현식으로 작업이 자동 실행되도록 예약합니다. 작업에 skills를 연결하고 모든 플랫폼으로 결과를 전달할 수 있으며, 일시 중지/재개/편집 작업도 지원합니다.
- **[서브에이전트 위임](delegation.md)** — `delegate_task` 도구는 격리된 컨텍스트, 제한된 도구 세트, 자체 터미널 세션을 갖춘 하위 에이전트 인스턴스를 생성합니다. 기본적으로 동시에 3개의 서브에이전트를 실행하며, 이 수는 구성할 수 있습니다.
- **[코드 실행](code-execution.md)** — `execute_code` 도구를 사용하면 에이전트가 Hermes 도구를 프로그래밍 방식으로 호출하는 Python 스크립트를 작성할 수 있습니다. 샌드박스 처리된 RPC 실행으로 여러 단계의 작업 흐름을 하나의 LLM 턴으로 압축합니다.
- **[이벤트 훅](hooks.md)** — 주요 수명 주기 지점에서 사용자 정의 코드를 실행합니다. Gateway 훅은 로깅, 알림, 웹훅을 처리하고, 플러그인 훅은 도구 가로채기, 메트릭, 가드레일을 처리합니다.
- **[배치 처리](batch-processing.md)** — 수백 또는 수천 개의 프롬프트에 대해 Hermes 에이전트를 병렬로 실행하여 학습 데이터 생성 또는 평가를 위한 구조화된 ShareGPT 형식의 궤적 데이터를 생성합니다.

## 미디어 및 웹

- **[음성 모드](voice-mode.md)** — CLI와 메시징 플랫폼 전반에서 완전한 음성 상호작용을 제공합니다. 마이크로 에이전트와 대화하고 음성 답변을 들으며 Discord 음성 채널에서 실시간 음성 대화를 할 수 있습니다.
- **[호출어](wake-word.md)** — CLI, TUI, 데스크톱 앱에서 핸즈프리 "Hey Hermes" 호출을 지원합니다. 기기 내 핫워드 리스너가 호출어를 들으면 음성 세션을 시작합니다.
- **[브라우저 자동화](browser.md)** — 여러 백엔드를 사용하는 완전한 브라우저 자동화: Browserbase cloud, Browser Use cloud, CDP를 통한 로컬 Chrome/Brave/Chromium/Edge 또는 로컬 Chromium. 웹사이트를 탐색하고, 양식을 작성하고, 정보를 추출합니다.
- **[비전 및 이미지 붙여넣기](vision.md)** — 멀티모달 비전 지원을 제공합니다. 클립보드의 이미지를 CLI에 붙여넣고 비전 기능을 지원하는 모든 모델을 사용해 분석, 설명 또는 작업을 요청할 수 있습니다.
- **[이미지 생성](image-generation.md)** — FAL.ai를 사용해 텍스트 프롬프트로 이미지를 생성합니다. 11개 모델(FLUX 2 Klein/Pro, GPT-Image 1.5/2, Nano Banana Pro, Ideogram V3, Recraft V4 Pro, Qwen, Z-Image Turbo, Krea V2 Medium/Large)을 지원하며, `hermes tools`로 하나를 선택할 수 있습니다.
- **[음성 및 TTS](tts.md)** — 모든 메시징 플랫폼에서 텍스트 음성 변환 출력과 음성 메시지 전사를 지원하며, 10개의 기본 제공업체를 제공합니다: Edge TTS (무료), ElevenLabs, OpenAI TTS, MiniMax, Mistral Voxtral, Google Gemini, xAI, NeuTTS, KittenTTS, Piper — 그리고 모든 로컬 TTS CLI를 위한 사용자 정의 명령 제공업체도 지원합니다.

## 통합

- **[MCP 통합](mcp.md)** — stdio 또는 HTTP 전송을 통해 모든 MCP 서버에 연결합니다. 네이티브 Hermes 도구를 작성하지 않고도 GitHub, 데이터베이스, 파일 시스템, 내부 API의 외부 도구에 접근할 수 있습니다. 서버별 도구 필터링과 샘플링 지원도 포함됩니다.
- **[제공업체 라우팅](provider-routing.md)** — 요청을 처리할 AI 제공업체를 세밀하게 제어합니다. 정렬, 허용 목록, 차단 목록, 우선순위 순서로 비용, 속도, 품질을 최적화할 수 있습니다.
- **[대체 제공업체](fallback-providers.md)** — 기본 모델에서 오류가 발생할 때 백업 LLM 제공업체로 자동 장애 조치를 수행하며, 비전 및 압축과 같은 보조 작업에도 독립적인 대체 제공업체를 설정할 수 있습니다.
- **[자격 증명 풀](credential-pools.md)** — 같은 제공업체의 여러 키에 API 호출을 분산합니다. 속도 제한이나 오류 발생 시 자동으로 키를 교체합니다.
- **[프롬프트 캐싱](../configuration#prompt-caching)** — 기본 제공되는 세션 간 1시간 접두사 캐시로, 네이티브 Anthropic, OpenRouter, Nous Portal의 Claude를 지원합니다. 항상 활성화되어 있으며 구성이 필요하지 않습니다.
- **[메모리 제공업체](memory-providers.md)** — 외부 메모리 백엔드(Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory)를 연결하여 기본 제공 메모리 시스템을 넘어 세션 간 사용자 모델링과 개인화를 지원합니다.
- **[API 서버](api-server.md)** — Hermes를 OpenAI 호환 HTTP 엔드포인트로 노출합니다. OpenAI 형식을 사용하는 모든 프런트엔드(Open WebUI, LobeChat, LibreChat 등)를 연결할 수 있습니다.
- **[IDE 통합 (ACP)](acp.md)** — VS Code, Zed, JetBrains와 같은 ACP 호환 편집기에서 Hermes를 사용합니다. 채팅, 도구 활동, 파일 diff, 터미널 명령이 편집기 안에 표시됩니다.
- **[배치 처리](batch-processing.md)** — CLI에서 많은 프롬프트나 작업에 대해 에이전트를 병렬로 실행하며, 평가 또는 후속 학습 파이프라인에 적합한 구조화된 출력과 궤적 캡처를 제공합니다.

## 사용자 지정

- **[성격 및 SOUL.md](personality.md)** — 에이전트의 성격을 완전히 사용자 지정할 수 있습니다. `SOUL.md`는 시스템 프롬프트에서 가장 먼저 배치되는 기본 정체성 파일이며, 세션마다 기본 제공 또는 사용자 지정 `/personality` 프리셋으로 바꿀 수 있습니다.
- **[스킨 및 테마](skins.md)** — CLI의 시각적 표현을 사용자 지정합니다. 배너 색상, 스피너의 얼굴과 동사, 응답 상자 레이블, 브랜딩 텍스트, 도구 활동 접두사를 설정할 수 있습니다.
- **[플러그인](plugins.md)** — 핵심 코드를 수정하지 않고 사용자 지정 도구, 훅, 통합을 추가합니다. 플러그인 유형은 일반 플러그인(도구/훅), 메모리 제공업체(세션 간 지식), 컨텍스트 엔진(대체 컨텍스트 관리)의 세 가지입니다. 통합 `hermes plugins` 대화형 UI로 관리합니다.
