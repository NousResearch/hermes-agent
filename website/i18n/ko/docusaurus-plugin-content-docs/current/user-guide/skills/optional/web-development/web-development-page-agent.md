---
title: "Page Agent — 웹 앱에 페이지 내 자연어 GUI 코파일럿 삽입"
sidebar_label: "Page Agent"
description: "웹 앱에 페이지 내 자연어 GUI 코파일럿 삽입"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Page Agent

웹 앱에 페이지 내 자연어 GUI 코파일럿을 삽입합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/web-development/page-agent`로 설치 |
| 경로 | `optional-skills/web-development/page-agent` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `web`, `javascript`, `agent`, `browser`, `gui`, `alibaba`, `embed`, `copilot`, `saas` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트는 이 내용을 지침으로 봅니다.
:::

# page-agent

alibaba/page-agent(https://github.com/alibaba/page-agent, 17k+ stars, MIT)는 TypeScript로 작성된 페이지 내 GUI 에이전트입니다. 웹페이지 안에서 실행되며 DOM을 텍스트로 읽고(스크린샷이나 멀티모달 LLM은 사용하지 않음), 현재 페이지에서 "로그인 버튼을 클릭한 다음 사용자 이름을 John으로 입력해"와 같은 자연어 지시를 실행합니다. 순수 클라이언트 측 방식으로 동작하며, 호스트 사이트는 스크립트를 포함하고 OpenAI 호환 LLM 엔드포인트를 전달하기만 하면 됩니다.

## 이 스킬을 사용할 때

다음과 같은 작업을 원하는 경우 이 스킬을 로드하세요.

- **자체 웹 앱 안에 AI 코파일럿 제공** (SaaS, 관리자 패널, B2B 도구, ERP, CRM) — "내 대시보드 사용자가 다섯 화면을 클릭하는 대신 'Acme Corp의 청구서를 만들고 이메일로 보내'라고 입력할 수 있어야 해"
- **레거시 웹 앱 현대화** — 프런트엔드를 다시 작성하지 않고도 기존 DOM 위에 page-agent를 추가
- **자연어를 통한 접근성 추가** — 음성 사용자 또는 스크린 리더 사용자가 원하는 작업을 설명하여 UI 조작
- **page-agent를 로컬(Ollama) 또는 호스팅된(Qwen, OpenAI, OpenRouter) LLM에서 데모하거나 평가**
- **대화형 교육/제품 데모 구축** — AI가 실제 UI에서 "경비 보고서 제출 방법"을 실시간으로 안내

## 이 스킬을 사용하지 않을 때

- **Hermes 자체가 브라우저를 조작하기를 원하는 경우** → Hermes의 기본 브라우저 도구(Browserbase / Camofox)를 사용하세요. page-agent는 *반대* 방향입니다.
- **삽입 없이 탭 간 자동화를 원하는 경우** → Playwright, browser-use 또는 page-agent Chrome 확장을 사용하세요.
- **시각적 그라운딩/스크린샷이 필요한 경우** → page-agent는 텍스트 DOM만 지원하므로 멀티모달 브라우저 에이전트를 사용하세요.

## 사전 요구 사항

- Node 22.13+ 또는 24+, npm 10+ (문서에는 11+이라고 되어 있지만 10.9도 정상 작동)
- OpenAI 호환 LLM 엔드포인트: Qwen(DashScope), OpenAI, Ollama, OpenRouter 또는 `/v1/chat/completions`를 지원하는 모든 서비스
- 디버깅을 위한 브라우저 개발자 도구

## 경로 1 — CDN을 통한 30초 데모(설치 없음)

작동을 가장 빠르게 확인할 수 있는 방법입니다. alibaba의 무료 테스트 LLM 프록시를 사용하므로 **평가 목적으로만** 사용하고 해당 서비스 약관을 따라야 합니다.

어떤 HTML 페이지에든 추가하거나(또는 북마클릿으로 개발자 도구 콘솔에 붙여 넣으세요) 다음을 사용합니다.

```html
<script src="https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js" crossorigin="true"></script>
```

패널이 나타납니다. 지시를 입력하면 됩니다.

북마클릿 형식(북마크 바에 추가하고 아무 페이지에서나 클릭):

```javascript
javascript:(function(){var s=document.createElement('script');s.src='https://cdn.jsdelivr.net/npm/page-agent@1.8.0/dist/iife/page-agent.demo.js';document.head.appendChild(s);})();
```

## 경로 2 — 자체 웹 앱에 npm 설치(프로덕션 사용)

기존 웹 프로젝트(React / Vue / Svelte / 일반 프로젝트) 안에서 다음을 실행합니다.

```bash
npm install page-agent
```

자체 LLM 엔드포인트에 연결하세요 — **실제 사용자에게 데모 CDN을 절대 제공하지 마세요**.

```javascript
import { PageAgent } from 'page-agent'

const agent = new PageAgent({
    model: 'qwen3.5-plus',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    apiKey: process.env.LLM_API_KEY,   // never hardcode
    language: 'en-US',
})

// Show the panel for end users:
agent.panel.show()

// Or drive it programmatically:
await agent.execute('Click submit button, then fill username as John')
```

프로바이더 예시(OpenAI 호환 엔드포인트라면 무엇이든 작동):

| 프로바이더 | `baseURL` | `model` |
|----------|-----------|---------|
| Qwen / DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.5-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama(로컬) | `http://localhost:11434/v1` | `qwen3:14b` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.6` |

**핵심 설정 필드** (`new PageAgent({...})`에 전달):

- `model`, `baseURL`, `apiKey` — LLM 연결
- `language` — UI 언어(`en-US`, `zh-CN` 등)
- 에이전트가 접근할 수 있는 대상을 제한하기 위한 허용 목록 및 데이터 마스킹 훅이 있습니다 — 전체 옵션 목록은 https://alibaba.github.io/page-agent/를 참조하세요.

**보안.** 실제 배포에서 `apiKey`를 클라이언트 측 코드에 넣지 마세요 — 자체 백엔드를 통해 LLM 호출을 프록시하고 `baseURL`이 해당 프록시를 가리키도록 하세요. 데모 CDN이 존재하는 이유는 alibaba가 평가를 위해 해당 프록시를 운영하기 때문입니다.

## 경로 3 — 소스 저장소 복제(기여 또는 수정)

page-agent 자체를 수정하거나, 로컬 IIFE 번들을 사용해 임의의 사이트에서 테스트하거나, 브라우저 확장을 개발하려는 경우 사용하세요.

```bash
git clone https://github.com/alibaba/page-agent.git
cd page-agent
npm ci              # exact lockfile install (or `npm i` to allow updates)
```

LLM 엔드포인트를 저장소 루트의 `.env`에 설정하세요. 예시:

```
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
```

Ollama 버전:

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=NA
LLM_MODEL_NAME=qwen3:14b
```

일반적인 명령:

```bash
npm start           # docs/website dev server
npm run build       # build every package
npm run dev:demo    # serve IIFE bundle at http://localhost:5174/page-agent.demo.js
npm run dev:ext     # develop the browser extension (WXT + React)
npm run build:ext   # build the extension
```

**모든 웹사이트에서 테스트**하려면 로컬 IIFE 번들을 사용하세요. 다음 북마클릿을 추가합니다.

```javascript
javascript:(function(){var s=document.createElement('script');s.src=`http://localhost:5174/page-agent.demo.js?t=${Math.random()}`;s.onload=()=>console.log('PageAgent ready!');document.head.appendChild(s);})();
```

그런 다음 `npm run dev:demo`를 실행하고 아무 페이지에서나 북마클릿을 클릭하면 로컬 빌드가 주입됩니다. 저장할 때마다 자동으로 다시 빌드됩니다.

**경고:** 경로 3의 개발 빌드에서는 `.env`의 `LLM_API_KEY`가 IIFE 번들에 인라인됩니다. 번들을 공유하지 마세요. 커밋하지 마세요. URL을 Slack에 붙여 넣지 마세요. (확인 결과 공개 개발 번들을 grep하면 `.env`의 실제 값이 그대로 반환됩니다.)

## 저장소 구조(경로 3)

npm workspaces를 사용하는 모노레포입니다. 주요 패키지:

| 패키지 | 경로 | 용도 |
|---------|------|---------|
| `page-agent` | `packages/page-agent/` | UI 패널을 포함한 기본 진입점 |
| `@page-agent/core` | `packages/core/` | UI가 없는 핵심 에이전트 로직 |
| `@page-agent/mcp` | `packages/mcp/` | MCP 서버(베타) |
| — | `packages/llms/` | LLM 클라이언트 |
| — | `packages/page-controller/` | DOM 작업 + 시각적 피드백 |
| — | `packages/ui/` | 패널 + i18n |
| — | `packages/extension/` | Chrome/Firefox 확장 |
| — | `packages/website/` | 문서 + 랜딩 사이트 |

## 작동 확인

경로 1 또는 경로 2를 사용한 후:
1. 개발자 도구를 연 상태로 브라우저에서 페이지 열기
2. 떠 있는 패널이 보여야 합니다. 보이지 않으면 콘솔에서 오류를 확인하세요(가장 일반적인 원인: LLM 엔드포인트의 CORS, 잘못된 `baseURL` 또는 잘못된 API 키)
3. 페이지에 표시된 요소에 맞는 간단한 지시 입력("Login 링크 클릭")
4. Network 탭에서 `baseURL`로 요청이 전송되는지 확인

경로 3을 사용한 후:
1. `npm run dev:demo`가 `Accepting connections at http://localhost:5174`를 출력
2. `curl -I http://localhost:5174/page-agent.demo.js`가 `HTTP/1.1 200 OK`와 `Content-Type: application/javascript`를 반환
3. 아무 사이트에서나 북마클릿을 클릭하면 패널 표시

## 주의할 점

- **프로덕션에서 데모 CDN 사용** — 사용하지 마세요. 속도 제한이 있고 alibaba의 무료 프록시를 사용하며 약관상 프로덕션 사용이 금지됩니다.
- **API 키 노출** — `new PageAgent({apiKey: ...})`에 전달한 키는 JS 번들에 포함됩니다. 실제 배포에서는 항상 자체 백엔드를 통해 프록시하세요.
- **OpenAI 비호환 엔드포인트**는 조용히 실패하거나 이해하기 어려운 오류를 발생시킵니다. 프로바이더에 기본 Anthropic/Gemini 형식이 필요하다면 그 앞에 OpenAI 호환성 프록시(LiteLLM, OpenRouter)를 사용하세요.
- **CSP 차단** — 엄격한 Content-Security-Policy를 적용한 사이트는 CDN 스크립트 로드나 인라인 eval을 거부할 수 있습니다. 이 경우 자체 오리진에서 셀프 호스팅하세요.
- **경로 3에서 `.env`를 편집한 후 개발 서버 재시작** — Vite는 시작할 때만 env를 읽습니다.
- **Node 버전** — 저장소는 `^22.13.0 || >=24`를 선언합니다. Node 20에서는 엔진 오류로 `npm ci`가 실패합니다.
- **npm 10과 11** — 문서에는 npm 11+이라고 되어 있지만 npm 10.9도 정상 작동합니다.

## 참조

- 저장소: https://github.com/alibaba/page-agent
- 문서: https://alibaba.github.io/page-agent/
- 라이선스: MIT(browser-use의 DOM 처리 내부 기능을 기반으로 하며, Copyright 2024 Gregor Zunic)
