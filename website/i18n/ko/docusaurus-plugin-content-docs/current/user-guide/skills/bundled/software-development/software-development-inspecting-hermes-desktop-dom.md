---
title: "Hermes Desktop DOM 검사 — CDP로 실행 중인 Hermes 데스크톱 DOM/CSS 읽기"
sidebar_label: "Hermes Desktop DOM 검사"
description: "CDP로 실행 중인 Hermes 데스크톱 DOM/CSS 읽기"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes 데스크톱 DOM 검사

CDP로 실행 중인 Hermes 데스크톱 DOM/CSS를 읽습니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 기본 제공(기본 설치됨) |
| 경로 | `skills/software-development/inspecting-hermes-desktop-dom` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `desktop`, `electron`, `cdp`, `dom`, `ui-verification`, `self-inspection` |
| 관련 스킬 | [`node-inspect-debugger`](/docs/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger), [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`dogfood`](/docs/user-guide/skills/bundled/software-development/software-development-dogfood) |

## 레퍼런스: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# 실행 중인 Hermes 데스크톱 DOM 검사

## 개요

`apps/desktop`을 개발 중이고 사용자가 동일한 앱(`hgui` / `npm run dev`)을 실행하고 있다면, 사용자가 보고 있는 창의 **실시간 렌더링 DOM**을 읽을 수 있습니다 — `.tsx`를 보고 추측하다가 틀리는 대신 계산된 스타일, 형상, 실제로 적용된 CSS 규칙, 콘솔 출력을 확인할 수 있습니다.

개발 서버는 자동으로 `127.0.0.1:9222`에서 Chrome DevTools Protocol 포트를 엽니다. 렌더러는 Chromium 페이지이므로 DevTools가 읽을 수 있는 것은 스크립트로도 읽을 수 있습니다.

**이 기능은 화면을 직접 확인하는 일을 대체하지 않습니다.** CDP는 "계산된 패딩이 얼마인가", "이 요소가 렌더링되었는가", "어떤 선택자가 일치하는가"와 같은 **사실에 관한** 질문에 답합니다. 결과가 보기 좋은지는 알려줄 수 없습니다. 색상 균형, 간격의 느낌, 그리고 "이게 못생겼나"는 여전히 사용자의 눈이나 스크린샷이 필요합니다. 사실은 CDP로 답하고, 미적인 판단은 사용자에게 맡기세요.

## 사용 시점

- UI 변경이 실행 중인 앱에 실제로 적용되었는지 확인할 때
- "이 요소가 여전히 X인 이유는?" — 무엇이 우선 적용되는지 찾은 후 편집하기 전에
- 변경하려는 컴포넌트의 안정적인 선택자를 찾을 때
- 실제 노드에서 디자인 토큰의 계산된 값을 확인할 때
- 사용자가 언급했지만 복사해 올 수 없는 렌더러 콘솔 오류를 읽을 때

**다음 용도로는 사용하지 마세요:** 성능 프로파일링 또는 힙 작업(`node-inspect-debugger`, `debugging-hermes-desktop`), 혹은 실제 질문이 "이게 제대로 보이나?"인 경우.

## 포트

모든 개발 서버 실행에서 `127.0.0.1:9222`로 열립니다. 정확히 다음 두 경우에는 닫혀 있습니다(`apps/desktop/electron/dev-cdp.ts`).

- **패키징된 빌드** — 항상 닫혀 있으며 어떤 환경 값으로도 재정의할 수 없음;
- **`HERMES_DESKTOP_DEV_SERVER`가 없음** — 패키징된 앱의 스모크 테스트 방식인 `dist/` 대상의 패키징되지 않은 `electron .`은 패키징된 앱처럼 동작함.

`HERMES_DESKTOP_CDP_PORT`로 포트를 변경(`=9333`)하거나 비활성화(`=off`)할 수 있습니다.

다른 작업을 하기 전에 확인하세요.

```bash
curl -s --max-time 3 http://127.0.0.1:${HERMES_DESKTOP_CDP_PORT:-9222}/json/version
```

출력이 비어 있으면 포트가 없습니다. 다른 포트를 조용히 추측하지 마세요.

**포트를 얻기 위해 사용자의 앱을 다시 실행하지 마세요.** 사용자의 세션과 상태가 손상됩니다. 대신 아래와 같이 격리된 인스턴스를 직접 실행하세요.

## DOM 읽기

`apps/desktop/scripts/eval.mjs`는 한 줄 실행 도구입니다.

```bash
cd apps/desktop
node scripts/eval.mjs "document.querySelectorAll('[data-slot]').length"
```

여러 단계의 작업에는 공유 클라이언트를 사용하세요 — 대상 검색과 프라미스 인식 eval을 제공합니다.

```js
import { CDP, SELECTORS } from './scripts/perf/lib/cdp.mjs'

const cdp = await CDP.connect({ port: 9222, match: '5174' })
const out = await cdp.eval(`JSON.stringify({
  radius: getComputedStyle(document.documentElement).getPropertyValue('--radius-scalar').trim(),
  composer: !!document.querySelector('[data-slot="composer-rich-input"]')
})`)
cdp.close()
```

`scripts/perf/lib/cdp.mjs`의 `SELECTORS`에는 안정적인 `data-slot` 훅(컴포저, 스레드 뷰포트, 어시스턴트 메시지, 턴 쌍, 프로필 레일)이 들어 있습니다. 직접 `querySelector`를 만들어내기보다 이를 우선 사용하세요 — 컴포넌트가 이동할 때 함께 업데이트됩니다.

## 가장 잘 맞는 질문: 어느 규칙이 적용되었나?

스타일이 "적용되지 않는다"고 해서 모든 호출 지점을 편집하는 것은 전형적인 낭비입니다. 먼저 실제 노드를 읽으세요.

```js
const el = document.querySelector('[data-slot="aui_assistant-message-root"] a')
JSON.stringify({
  ownClasses: el.className,
  weight: getComputedStyle(el).fontWeight,
  parents: (() => {
    const out = []
    let n = el
    while ((n = n.parentElement) && out.length < 6) out.push(n.className)
    return out
  })()
})
```

노드 자체에 클래스가 없다면 해당 값은 **상속된 것**입니다 — 호출 지점을 훑어도 문제가 해결되지 않으며, 상위 요소의 규칙이 필요합니다. 플러그인 스타일시트(예: `@tailwindcss/typography`의 `prose a { font-weight: 500 }`)가 유틸리티 클래스보다 우선 적용되는 일이 흔합니다. 각 사용처가 아니라 공유 클래스에서 재정의하세요.

## 자체 격리 인스턴스

포트가 없거나 사용자의 창을 방해하면 안 될 때:

```bash
cd apps/desktop
HERMES_HOME=/tmp/cdp-probe-home \
HERMES_DESKTOP_DEV_SERVER=http://127.0.0.1:5174 \
HERMES_DESKTOP_CDP_PORT=9333 \
  npx electron . --user-data-dir=/tmp/cdp-probe-userdata
```

별도의 `--user-data-dir`는 Electron의 단일 인스턴스 잠금을 피하므로 실행 중인 `hgui`와 충돌할 수 없습니다. 별도의 `HERMES_HOME`은 실제 세션과 분리합니다. 같은 이유로 9222가 아닌 포트를 선택하세요. 백그라운드에서 실행하고 완료하면 종료하세요.

`npm run perf:serve`도 임시 `HERMES_HOME`을 내장해 동일한 작업을 수행하며, 성능 하네스도 사용하려는 경우 유용합니다.

## 주의 사항

- **사용자의 개발 서버나 앱을 비워두기 위해 절대 종료하지 마세요.** 실행 중인 서버를 종료하면 Chromium의 소켓 풀이 사라지고, 그 결과 발생하는 `ERR_NETWORK_CHANGED`가 방금 변경한 내용의 탓으로 오해될 수 있습니다.
- **임시 `HERMES_HOME`에는 백엔드가 없습니다.** 앱이 `hermes:api`에 대해 `ECONNREFUSED`를 기록하고 스스로 종료할 수 있습니다. 그래도 렌더러는 마운트되며 DOM은 읽을 수 있으므로, 즉시 읽고 자체 종료된 프로브를 포트 문제로 착각하지 마세요. 포트가 바인딩될 때 Chromium은 `DevTools listening on ws://127.0.0.1:<port>/…`를 기록합니다. 이 줄이 포트가 열렸다는 증거입니다.
- **한 번만 확인하지 말고 폴링하세요.** 방금 시작한 앱이 포트에 응답하기까지 1~2초가 걸립니다.
- **전체 DOM을 덤프하지 마세요.** 데스크톱은 수백 개의 노드를 렌더링하며 `outerHTML`은 컨텍스트를 묻어버립니다. 평가하는 표현식 안에서 작은 JSON 객체로 투영하세요.
- **`CDP.connect`에 `match`를 전달하세요.** 그렇지 않으면 기본 창 대신 펫 오버레이, 빠른 입력 창, 또는 DevTools 대상에 연결할 수 있습니다.
- **`cdp.eval`은 값을 반환하고, 원시 `Runtime.evaluate`는 값을 이중으로 중첩합니다** (`.result.result.value`). 래퍼를 사용하세요.
- 이 저장소에서 `vite dev` 실행 중 **`import.meta.env.DEV`는 `true`**입니다. `apps/desktop/scripts/profile-typing-lag.md`의 그렇지 않다고 한 주석은 오래된 내용입니다.
