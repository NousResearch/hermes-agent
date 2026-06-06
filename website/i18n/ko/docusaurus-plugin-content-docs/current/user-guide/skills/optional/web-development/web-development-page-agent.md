---
title: "Page Agent"
sidebar_label: "Page Agent"
description: "alibaba/page-agent를 자체 웹 애플리케이션에 임베드하세요 — 단일 <script> 태그나 npm 패키지로 제공되며 사이트 최종 사용자가 자연어로 UI를 제어할 수 있게 해주는 순수 JavaScript 인페이지 GUI 에이전트입니다..."
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Page Agent

alibaba/page-agent를 자체 웹 애플리케이션에 임베드하세요 — 단일 `<script>` 태그나 npm 패키지로 제공되며 사이트 최종 사용자가 자연어("로그인을 클릭하고 사용자 이름에 John을 입력해")로 UI를 제어할 수 있게 해주는 순수 JavaScript 인페이지 GUI 에이전트입니다. Python, 헤드리스 브라우저, 확장 프로그램이 필요하지 않습니다. 사용자가 SaaS / 관리자 패널 / B2B 도구에 AI 코파일럿을 추가하고 싶어 하는 웹 개발자이거나, 레거시 웹 앱을 자연어로 액세스할 수 있게 만들고 싶거나, 로컬(Ollama) 또는 클라우드(Qwen / OpenAI / OpenRouter) LLM에 대해 page-agent를 평가하고 싶을 때 이 스킬을 사용하세요. 서버 측 브라우저 자동화를 위한 것이 아닙니다 — 그런 사용자에게는 Hermes의 내장 브라우저 도구를 대신 안내하세요.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/web-development/page-agent` 명령어로 설치 |
| 경로 | `optional-skills/web-development/page-agent` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `web`, `javascript`, `agent`, `browser`, `gui`, `alibaba`, `embed`, `copilot`, `saas` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# page-agent

alibaba/page-agent (https://github.com/alibaba/page-agent, 17k+ stars, MIT)는 TypeScript로 작성된 인페이지 GUI 에이전트입니다. 웹페이지 내부에 상주하며 DOM을 텍스트로 읽고(스크린샷 없음, 멀티모달 LLM 없음) 현재 페이지에 대해 "로그인 버튼을 클릭한 다음 사용자 이름에 John을 입력해"와 같은 자연어 지시를 실행합니다. 순수 클라이언트 측에서 작동합니다 — 호스트 사이트는 스크립트를 포함하고 OpenAI 호환 LLM 엔드포인트를 전달하기만 하면 됩니다.

## 이 스킬의 사용 시기

사용자가 다음을 원할 때 이 스킬을 로드하세요:

- **자신의 웹 앱 내부에 AI 코파일럿 제공** (SaaS, 관리자 패널, B2B 도구, ERP, CRM) — "내 대시보드의 사용자는 5개의 화면을 클릭하는 대신 'Acme Corp의 청구서를 만들고 이메일로 보내줘'라고 입력할 수 있어야 합니다"
- 프론트엔드를 재작성하지 않고 **레거시 웹 앱 현대화** — page-agent는 기존 DOM 위에 적용됩니다
- **자연어를 통한 접근성 추가** — 음성 / 스크린 리더 사용자가 원하는 것을 설명하여 UI를 구동합니다
- 로컬(Ollama) 또는 호스팅된(Qwen, OpenAI, OpenRouter) LLM에 대한 **page-agent 데모 또는 평가**
- **대화형 교육 / 제품 데모 구축** — 실제 UI에서 AI가 "경비 보고서 제출 방법"을 사용자에게 실시간으로 안내하도록 합니다

## 이 스킬을 사용하면 안 되는 경우

- 사용자가 **Hermes가 직접 브라우저를 구동하기를 원할 때** → Hermes의 내장 브라우저 도구(Browserbase / Camofox)를 사용하세요. page-agent는 *반대* 방향입니다.
- 사용자가 임베딩 없이 **탭 간(cross-tab) 자동화를 원할 때** → Playwright, browser-use 또는 page-agent Chrome 확장 프로그램을 사용하세요
- 사용자가 **시각적 근거(Visual grounding) / 스크린샷을 필요로 할 때** → page-agent는 텍스트-DOM 전용입니다. 멀티모달 브라우저 에이전트를 대신 사용하세요.

## 전제 조건

- Node 22.13+ 또는 24+, npm 10+ (문서에서는 11+라고 주장하지만 10.9도 잘 작동합니다)
- OpenAI 호환 LLM 엔드포인트: Qwen (DashScope), OpenAI, Ollama, OpenRouter 또는 `/v1/chat/completions`를 사용하는 모든 것
- 개발자 도구가 있는 브라우저 (디버깅용)

## 경로 1 — CDN을 통한 30초 데모 (설치 없음)

작동 방식을 확인하는 가장 빠른 방법입니다. alibaba의 무료 테스트 LLM 프록시를 사용합니다 — **평가 목적으로만** 사용되며 해당 약관의 적용을 받습니다.

아무 HTML 페이지에나 추가하세요 (또는 개발자 도구 콘솔에 북마클릿으로 붙여넣기):

```html
<script src="https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js" crossorigin="true"></script>
```

패널이 나타납니다. 지시사항을 입력하세요. 끝입니다.

북마클릿 형태 (북마크 바에 놓아두고, 어떤 페이지에서든 클릭):

```javascript
javascript:(function(){var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js';document.head.appendChild(s);})();
```

## 경로 2 — 자신의 웹 앱에 npm 설치 (프로덕션 사용)

기존 웹 프로젝트(React / Vue / Svelte / 일반) 내부에서:

```bash
npm install page-agent
```

자신의 LLM 엔드포인트와 연결하세요 — **실제 사용자에게 데모 CDN을 제공하지 마세요**:

```javascript
import { PageAgent } from 'page-agent'

const agent = new PageAgent({
    model: 'qwen3.5-plus',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    apiKey: process.env.LLM_API_KEY,   // 절대 하드코딩하지 마세요
    language: 'en-US',
})

// 최종 사용자를 위해 패널 표시:
agent.panel.show()

// 또는 프로그래밍 방식으로 구동:
await agent.execute('Click submit button, then fill username as John')
```

제공자 예시 (어떤 OpenAI 호환 엔드포인트든 작동합니다):

| 제공자 | `baseURL` | `model` |
|----------|-----------|---------|
| Qwen / DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.5-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama (local) | `http://localhost:11434/v1` | `qwen3:14b` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.6` |

**주요 구성 필드** (`new PageAgent({...})`에 전달됨):

- `model`, `baseURL`, `apiKey` — LLM 연결
- `language` — UI 언어 (`en-US`, `zh-CN` 등)
- 에이전트가 건드릴 수 있는 항목을 제한하기 위한 허용 목록(Allowlist) 및 데이터 마스킹 후크가 존재합니다 — 전체 옵션 목록은 https://alibaba.github.io/page-agent/ 를 참조하세요

**보안.** 실제 배포를 위해 클라이언트 측 코드에 `apiKey`를 넣지 마세요 — 백엔드를 통해 LLM 호출을 프록시하고 `baseURL`이 프록시를 가리키도록 설정하세요. 데모 CDN이 존재하는 이유는 alibaba가 평가용으로 해당 프록시를 실행하기 때문입니다.

## 경로 3 — 소스 저장소 복제 (기여 또는 수정)

사용자가 page-agent 자체를 수정하거나, 로컬 IIFE 번들을 통해 임의의 사이트에 대해 테스트하거나, 브라우저 확장 프로그램을 개발하고자 할 때 사용합니다.

```bash
git clone https://github.com/alibaba/page-agent.git
cd page-agent
npm ci              # 정확한 잠금 파일(lockfile) 설치 (또는 업데이트를 허용하려면 `npm i`)
```

저장소 루트에 LLM 엔드포인트가 포함된 `.env`를 만듭니다. 예시:

```
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
```

Ollama 방식:

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=NA
LLM_MODEL_NAME=qwen3:14b
```

일반적인 명령어:

```bash
npm start           # 문서/웹사이트 개발 서버
npm run build       # 모든 패키지 빌드
npm run dev:demo    # http://localhost:5174/page-agent.demo.js 에 IIFE 번들 제공
npm run dev:ext     # 브라우저 확장 프로그램(WXT + React) 개발
npm run build:ext   # 확장 프로그램 빌드
```

로컬 IIFE 번들을 사용하여 **어떤 웹사이트에서든 테스트**하세요. 이 북마클릿을 추가합니다:

```javascript
javascript:(function(){var s=document.createElement('script');s.src=`http://localhost:5174/page-agent.demo.js?t=${Math.random()}`;s.onload=()=>console.log('PageAgent ready!');document.head.appendChild(s);})();
```

그 후: `npm run dev:demo`를 실행하고 임의의 페이지에서 북마클릿을 클릭하면 로컬 빌드가 삽입됩니다. 저장 시 자동 재빌드됩니다.

**경고:** `.env`의 `LLM_API_KEY`는 개발 빌드 중에 IIFE 번들에 인라인으로 삽입됩니다. 번들을 공유하지 마세요. 커밋하지 마세요. Slack에 URL을 붙여넣지 마세요. (확인됨: 공개 개발 번들에서 grep을 수행하면 `.env`의 리터럴 값이 반환됩니다.)

## 저장소 레이아웃 (경로 3)

npm 작업 공간을 사용하는 모노레포(Monorepo)입니다. 주요 패키지:

| 패키지 | 경로 | 목적 |
|---------|------|---------|
| `page-agent` | `packages/page-agent/` | UI 패널이 포함된 주 진입점 |
| `@page-agent/core` | `packages/core/` | 핵심 에이전트 로직, UI 없음 |
| `@page-agent/mcp` | `packages/mcp/` | MCP 서버 (베타) |
| — | `packages/llms/` | LLM 클라이언트 |
| — | `packages/page-controller/` | DOM 조작 + 시각적 피드백 |
| — | `packages/ui/` | 패널 + i18n |
| — | `packages/extension/` | Chrome/Firefox 확장 프로그램 |
| — | `packages/website/` | 문서 + 방문 페이지 사이트 |

## 작동 확인

경로 1 또는 경로 2 이후:
1. 개발자 도구가 열린 브라우저에서 페이지를 엽니다.
2. 떠 있는 패널이 보여야 합니다. 그렇지 않은 경우 콘솔에서 오류를 확인하세요 (가장 일반적인 원인: LLM 엔드포인트의 CORS, 잘못된 `baseURL`, 또는 잘못된 API 키).
3. 페이지에 보이는 것과 일치하는 간단한 지시를 입력하세요 ("Login 링크 클릭").
4. 네트워크(Network) 탭을 확인하세요 — `baseURL`에 대한 요청이 보여야 합니다.

경로 3 이후:
1. `npm run dev:demo` 실행 시 `Accepting connections at http://localhost:5174`가 출력됩니다.
2. `curl -I http://localhost:5174/page-agent.demo.js`를 실행하면 `Content-Type: application/javascript`와 함께 `HTTP/1.1 200 OK`가 반환됩니다.
3. 아무 사이트에서나 북마클릿을 클릭하면 패널이 나타납니다.

## 주의 사항 (Pitfalls)

- **프로덕션에서의 데모 CDN 사용** — 하지 마세요. 속도 제한이 있으며, alibaba의 무료 프록시를 사용하고, 해당 약관에서 프로덕션 사용을 금지하고 있습니다.
- **API 키 노출** — `new PageAgent({apiKey: ...})`에 전달된 모든 키는 JS 번들과 함께 제공됩니다. 실제 배포의 경우 항상 자신의 백엔드를 통해 프록시하세요.
- **OpenAI와 호환되지 않는 엔드포인트**는 조용히 실패하거나 암호 같은 오류를 발생시킵니다. 제공자가 네이티브 Anthropic/Gemini 포맷팅을 필요로 하는 경우, 앞에 OpenAI 호환성 프록시(LiteLLM, OpenRouter)를 사용하세요.
- **CSP 차단** — 엄격한 Content-Security-Policy를 가진 사이트는 CDN 스크립트 로드를 거부하거나 인라인 eval을 허용하지 않을 수 있습니다. 이 경우, 본인의 오리진(origin)에서 직접 호스팅하세요.
- 경로 3에서 `.env` 편집 후 **개발 서버 재시작** — Vite는 시작 시에만 env를 읽습니다.
- **Node 버전** — 저장소에는 `^22.13.0 || >=24`로 명시되어 있습니다. Node 20은 엔진 오류와 함께 `npm ci`에 실패합니다.
- **npm 10 vs 11** — 문서에는 npm 11+라고 나와 있지만 npm 10.9에서도 잘 작동합니다.

## 참조

- 저장소: https://github.com/alibaba/page-agent
- 문서: https://alibaba.github.io/page-agent/
- 라이선스: MIT (browser-use의 DOM 처리 내부 요소를 기반으로 구축됨, Copyright 2024 Gregor Zunic)
