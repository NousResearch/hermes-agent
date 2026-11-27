---
title: "Tldraw Offline — 에이전트로 tldraw 오프라인 캔버스 구동 및 스크립팅"
sidebar_label: "Tldraw Offline"
description: "에이전트로 tldraw 오프라인 캔버스 구동 및 스크립팅"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Tldraw Offline

에이전트로 tldraw 오프라인 캔버스를 구동하고 스크립팅합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/tldraw-offline`로 설치 |
| 경로 | `optional-skills/creative/tldraw-offline` |
| 버전 | `1.0.0` |
| 작성자 | Teknium + Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `tldraw`, `canvas`, `whiteboard`, `document-script`, `diagramming` |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침입니다.
:::

# tldraw offline Skill

오프라인 tldraw 데스크톱 앱(offline.tldraw.com)으로 작업합니다. 열린
캔버스를 읽고, 편집하고, **문서 스크립트**를 작성합니다 — 문서 스크립트는
`.tldraw` 파일에 삽입되는 JavaScript로, 로드될 때 실행되어 파일에 지속적인
동작을 부여합니다. 이 앱은 **로컬 HTTP API**(기본값 `localhost:7236`)를
실행하며, 코딩 에이전트는 터미널에서 일반 `curl`로 이를 구동합니다 — 앱 자체
홈페이지 데모(캔버스를 실시간으로 편집하는 Codex)가 정확히 이 방식을 사용합니다.
에이전트는 computer-use / GUI 클릭을 사용하지 않으며 `.tldraw` 파일을 직접
수정하지도 않습니다. 작업하는 동안 tldraw offline을 계속 열어 두세요.

## 사용 시점

- 사용자가 tldraw offline을 열어 두고 캔버스를 만들거나 수정해 달라고 요청한 경우(다이어그램, 와이어프레임, 레이아웃).
- 그림에 지속적인 동작(반응형 도형, 인터랙티브 버튼, 애니메이션, 연결 로직)을 추가하고 싶으며, 삽입된 문서 스크립트를 통해 처리하려는 경우.

그림을 흉내 내기 위해 도형을 손으로 배치하지 마세요 — 도형을 생성하는 코드를 작성하세요. 에이전트는 그리는 것보다 캔버스를 스크립팅하는 데 훨씬 뛰어납니다.

## 사전 요구 사항

- **tldraw offline이 설치되어 실행 중이고 문서가 열려 있어야 합니다**, 릴리스:
  https://github.com/tldraw/tldraw-offline/releases/latest (macOS DMG, Windows
  x64/Arm64, Linux `x86_64`/`arm64` AppImage 또는 amd64/arm64 `.deb`).
- **에이전트 스킬이 앱에 설치되어 있어야 합니다**: `Develop → Install Agent Skills`.
  앱은 자체 tldraw 스킬을 `~/.codex/skills/`, `~/.claude/skills/`,
  `~/.cursor/skills/`, `~/.gemini/skills/`에 기록합니다 — 아래의 `curl` 레시피를
  에이전트에 알려 줍니다. (이 Hermes 스킬은 해당 지침을 그대로 반영합니다.)
- **로컬 제어 API.** 앱은 시작 시 설정 디렉터리(Linux `~/.config/tldraw/`, macOS
  `~/Library/Application Support/tldraw/`, Windows `%APPDATA%\\tldraw\\`)에
  `server.json`을 기록하며, 여기에는 `port`(기본값 `7236`), bearer `token`,
  `pid`, `startedAt`이 들어 있습니다. `GET /`를 제외한 모든 요청에는
  `Authorization: Bearer <token>`이 필요합니다. 정상적으로 종료하면
  `server.json`이 삭제됩니다. 파일은 있지만 포트가 응답하지 않는다면 앱이 비정상
  종료된 것이므로 실행 중이 아닌 것으로 처리하세요.
- **모든 셸 호출에서 포트와 토큰을 다시 읽으세요.** 각 터미널 호출은 새 셸이므로
  `export`한 토큰은 유지되지 않습니다 — "한 번 export하고 재사용"하면 빈 토큰이
  전송되어 401이 발생합니다. 각 호출의 맨 위에서 인라인으로 둘 다 읽으세요:
  `PORT=$(jq -r .port <server.json); TOKEN=$(jq -r .token <server.json)`.
- 로컬 편집에는 계정이나 네트워크가 필요하지 않습니다.

## 실행 방법

두 가지 작업 흐름이 있습니다. 변경 사항이 새로고침 후에도 유지되어야 하는지에 따라 선택하세요.

**A. 일회성 캔버스 편집(`/exec`)** — 레이아웃, 도형 생성, 정리. 저장되는 스크립트가 아닌 실시간 편집입니다:

```bash
BASE=http://localhost:7236
TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.config/tldraw/server.json'))['token'])")
# find the focused document id
DOC=$(curl -s "$BASE/api/search" -X POST -H 'content-type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"code":"return (await api.getFocusedDoc()).id"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['result'])")
# run code with the live `editor` + `helpers` in scope
curl -s "$BASE/api/doc/$DOC/exec" -X POST -H 'content-type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"code":"const {createShapeId,toRichText}=await import(\"tldraw\"); editor.createShape({id:createShapeId(),type:\"geo\",x:0,y:0,props:{geo:\"rectangle\",w:200,h:100,color:\"blue\",fill:\"solid\",richText:toRichText(\"hello\")}}); return editor.getCurrentPageShapes().length"}'
```

**B. 지속적인 동작(`script/main.js`)** — 새로고침 후에도 유지되어야 하는 반응형/인터랙티브 로직. 디스크의 파일을 편집하면 앱의 감시자가 적용합니다:

```bash
# get the live script file path for the doc
curl -s "$BASE/api/doc/$DOC/script-workspace" -X POST \
  -H "Authorization: Bearer $TOKEN"          # -> result.mainJsPath, result.isDefaultScript
# edit result.mainJsPath with read_file / patch / write_file (see scripts/main.js)
# then confirm the watcher applied it:
curl -s "$BASE/api/doc/$DOC/script-status" -H "Authorization: Bearer $TOKEN"
```

바로 적용할 수 있는 문서 스크립트는 `scripts/main.js`입니다.

## 빠른 참고

문서 스크립트 계약(`script-context.d.ts` 앱 번들에 대해 검증됨):

```js
import { createShapeId, toRichText } from 'tldraw'   // primitives: import, not globals

export default function ({ editor, helpers, signal }) {
  editor.run(() => {                                 // batch = one undo step
    helpers.createShapeIfMissing({                   // idempotent furniture
      id: createShapeId('node-1'), type: 'geo', x: 0, y: 0,
      props: { geo: 'rectangle', w: 200, h: 100, richText: toRichText('hi') },
    })
  })

  const stop = editor.store.listen(() => { /* react */ })  // fires the tick AFTER a commit
  signal.addEventListener('abort', () => stop())           // REQUIRED cleanup on rerun/close
}
```

- `ctx.editor` — 실시간 `Editor`(`createShape`, `updateShape`, `deleteShapes`,
  `getCurrentPageShapes`, `getShape`, `getBindingsFromShape`, `zoomToFit`,
  `on('tick'|'event', fn)`, `run(fn, { history: 'ignore' })`).
- `ctx.helpers` — `createShapeIfMissing`, `createShapesIfMissing`,
  `createArrowBetweenShapes(from, to, { arrowheadEnd })`, `translateShapes`,
  `onShapeTranslate(id, fn, { signal })`, `richTextToPlainText`, `boxShapes`,
  `getLints`.
- `ctx.signal` — `AbortSignal`; 모든 리스너/인터벌 해제를 여기에 연결하세요.
- `config.js`(별도 파일)는 사용자 지정 도형/도구/컴포넌트 유틸을 등록하고 마운트 전에
  실행됩니다. `main.js`는 마운트된 에디터에서 실행되며 저장 시 다시 실행됩니다.

## 인터랙티브 UI(상태를 구동하는 클릭 가능한 버튼)

그려진 도형은 실제 앱처럼 동작할 수 있습니다 — 정적인 화이트보드로는 할 수 없는 일입니다.
전체 예제: `scripts/counter.js`(숫자 표시 + MINUS/RESET/PLUS 버튼).

검증 경계 — 상호작용이 작동하는지 또는 작동하지 않는지 주장하기 전에 읽으세요.
앱의 자체 에이전트 플레이북은 `/exec`를 통해 "시뮬레이션된 클릭 한 번과 상태 읽기
한 번"으로 클릭 가능한 UI 스크립트를 검증하라고 합니다(`editor.dispatch`로 포인터
이벤트를 전달하고, tick을 기다린 다음, 도형의 상태를 읽음) — 실제 마우스를 구동하는
방식이 아닙니다. 이 기준에서 카운터는 검증되었습니다: 전달된 클릭으로
`0 → 1 → 2 → 1 → 0`이 되었습니다. 적어 둘 만한 주의 사항 두 가지:
- **앱의 파일 감시자가 스크립트를 적용해야 스크립트가 실행됩니다.** Linux에서는 해당
  감시자가 inotify를 사용합니다. 호스트에서 `fs.inotify.max_user_instances`가
  고갈되면 `inotify_add_watch ... No space left on device`가 로그에 기록되고,
  `script-status`는 `state: "not-watching"` / `hasEntry: false`를 보여 주며 스크립트가
  전혀 실행되지 않습니다. 이는 스크립트 버그가 아니라 호스트 제한이며, 일반적인
  데스크톱에는 영향을 주지 않습니다.
- **computer-use로 캔버스를 구동하려면 백그라운드가 아닌 FOREGROUND 전달이 필요합니다.**
  cua-driver의 기본 `background` 전달은 Electron의 가려진 렌더러에서
  `background_unavailable`을 반환합니다 — 하지만 그것이 막다른 길은 아닙니다.
  cua-driver는 `escalation: "foreground"` 힌트를 반환합니다. `delivery_mode: "foreground"`를
  `bring_to_front`와 함께 설정하면 X11 XTest 경로(`x11_xtest_fg`)로 클릭이 전달됩니다 —
  이 방식으로 "Run Script" 동의 대화 상자를 닫고 캔버스를 클릭할 수 있습니다. 이 정확한
  v1.11.0 AppImage(Linux/X11)에서 Cua 팀이 검증했습니다. 백그라운드 모드에서
  "Electron이 합성 클릭을 거부한다"고 결론 내리고 포기하지 마세요 — 포그라운드로
  올라가세요. (실제 제품 경로는 여전히 클릭이 아닌 `/exec`입니다. 이 메모는
  computer-use 기반 테스트를 위한 것입니다.)

패턴:

```js
export default function ({ editor, helpers, signal }) {
  // 1. Build buttons idempotently; tag each with meta so the handler finds them.
  //    Give buttons a visible label AND a meta.action.
  // 2. Hit-test pointer_down in PAGE coordinates against the button bounds:
  const inside = (b, p) => p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h
  function onEvent(info) {
    if (!info || info.name !== 'pointer_down') return
    let p = null
    try { if (info.point && editor.screenToPage) p = editor.screenToPage(info.point) } catch {}
    p = p ?? editor.inputs?.currentPagePoint
    if (!p) return
    const hit = editor.getCurrentPageShapes().find(
      (s) => s.meta?.ui === 'button' &&
        inside({ x: s.x, y: s.y, w: s.props.w, h: s.props.h }, p)
    )
    if (hit) runAction(hit.meta.action)   // mutate state; store it in a shape's meta
  }
  editor.on('event', onEvent)
  signal.addEventListener('abort', () => editor.off('event', onEvent))  // REQUIRED
}
```

- 버튼은 하드 코딩된 좌표가 아니라 `meta`(또는 `helpers.richTextToPlainText`를 통한
  표시 레이블)로 찾으세요.
- **하나의 스크립트가 빌드와 읽기를 모두 담당합니다.** 도형을 한 코드 경로에서
  (`meta.action: 'inc'`로) 만들고 핸들러가 다른 규칙(`meta.action === 'PLUS'`)을
  읽으면 클릭해도 조용히 아무 일도 일어나지 않습니다. 같은 스크립트에서 버튼을
  만들고 처리하거나, 스크립트가 버튼을 새로 만들 수 있도록 빈 캔버스를 제공하세요 —
  파일의 db에 서로 맞지 않는 도형을 미리 굽지 마세요.
- 앱 상태는 도형의 `meta`(예: `meta.count`)에 보관하고, 검증 시 저장된 상태를 읽을 수
  있도록 해당 도형의 `richText` 레이블로 렌더링하세요.
- **`signal` abort에서 리스너를 분리하세요.** 이는 장식적인 작업이 아닙니다. 다음
  저장 때 이전 `onEvent`가 새 리스너와 함께 계속 연결되어 모든 클릭이 두 번 실행되고,
  카운터가 1 대신 2씩 증가합니다.
- 연속 동작에는 `editor.on('tick', fn)`을 사용하세요. 연결된 조각이 있는 이동 앵커에는
  `helpers.onShapeTranslate(id, fn, { signal })`을 사용하세요.

### 자체 실행 스크립트 `.tldraw` 배포

`.tldraw`는 `metadata.json` + `session.json` + `db.sqlite` + `assets/`
+ `script/`의 압축 파일입니다(이 항목들만 패키징할 수 있음). "이 문서에는 스크립트가
있습니다 → Run Script" 동의 대화 상자 없이 스크립트가 자동 실행되게 하려면:

- `metadata.json`에 `script` 매니페스트 `{ "sha256": "<digest>" }`가 있어야 합니다.
  digest는 정렬된 각 `script/` 경로에 대해 `` `${path}\0${sha256hex(bytes)}\n` ``을
  계산한 sha256입니다. 불일치는 변조된 것으로 거부됩니다.
- `~/.tldraw/script-trust.json`(`{ "trusted": ["<digest>"] }` 또는
  `$TLDRAW_SCRIPT_TRUST`)에 digest를 추가하여 사전 신뢰하세요. `isScriptTrusted(digest)`가
  true이면 앱은 동의를 건너뜁니다.

## 절차

1. `server.json`에서 현재 토큰/포트를 읽습니다. `api.getFocusedDoc()`(또는
   `api.getDocs()`)로 대상 문서를 찾습니다. 여러 문서가 열려 있다면 명시적으로 이름을 지정하세요.
2. 레이아웃/생성에는 `/exec`를 사용합니다. 지속적인 동작에는 `/script-workspace`를 통해
   `script/main.js`를 편집합니다.
3. 스크립트를 멱등적으로 만드세요: `helpers.createShapeIfMissing`와 안정적인
   `createShapeId('name')` ID로 지속적인 도형을 생성합니다. 스크립트는 로드될 때마다 다시 실행됩니다.
4. 스크립트가 소유한 쓰기가 사용자의 실행 취소 스택에 들어가지 않게 하세요:
   `editor.run(fn, { history: 'ignore' })`(또는 이미 그렇게 처리하는
   `helpers.translateShapes`).
5. 반응형 동작에는 `editor.store.listen(cb)`를 사용하고 `signal` abort에서 해제하세요.
   상호작용에는 `editor.on('event', h)`(페이지 좌표에서 `pointer_down` 히트 테스트)를,
   애니메이션에는 `editor.on('tick', h)`를 사용합니다.
6. 이동하는 단일 앵커와 연결된 내부 요소에는 넓은 store 리스너보다
   `helpers.onShapeTranslate(anchorId, fn, { signal })`을 우선하세요 — 넓은 리스너는
   자체 쓰기를 피드백 루프로 만들 수 있습니다.

## 도형 props(tldraw SDK v5 스키마에 대해 검증됨)

`editor.createShape` / `createShapeIfMissing`은 부분 props를 받습니다(도형 유틸이 기본값을 채움).
파일 스냅샷의 **원시 레코드**를 만들 때는 아래의 모든 prop이 필요합니다(`scripts/validate_shapes.mjs` 실행):

| Shape | Required props |
|-------|----------------|
| `note`  | `richText`, `color`, `labelColor`, `size`, `font`, `align`, `verticalAlign`, `growY`, `fontSizeAdjustment`, `url`, `scale`, `textLastEditedBy` |
| `text`  | `richText`, `color`, `size`, `font`, `textAlign`, `w`, `scale`, `autoSize` |
| `frame` | `w`, `h`, `name`, `color` |
| `geo`   | `geo`, `w`, `h`, `color`, `fill`, `richText` (+ dash/size/etc. defaulted) |

`richText`는 `toRichText('...')`여야 합니다 — 일반 문자열은 거부됩니다. `color` 열거형:
`black grey light-violet violet blue light-blue yellow orange green light-green
light-red red white`. `font` 열거형: `draw sans serif mono`.

## 주의 사항

- **`store.listen`은 커밋 직후가 아니라 커밋 다음 tick에서 실행됩니다.** 도형을 쓰고
  리스너가 실행되었다고 기대하며 즉시 상태를 읽으면 아직 실행되지 않은 것입니다. 같은
  이유로 앱 메모는 `editor.dispatch`가 비동기라고 설명합니다 — 검증하기 전에 tick을 기다리세요.
- **전역이 아니라 `ctx`입니다.** 진입점은 `export default function ({ editor,
  helpers, signal })`입니다. 문서 스크립트에는 `editor`라는 전역이 없습니다.
  `createShapeId` / `toRichText` / `Vec`는 `import ... from 'tldraw'`에서 가져옵니다.
- **`text`가 아니라 `richText`입니다.** text/note/geo 레이블에는
  `richText: toRichText(s)`를 사용합니다.
- **원시 레코드에는 모든 prop이 필요하지만 `createShape`에는 필요하지 않습니다.** 앱 안에서는
  필요한 prop만 전달하고, 직접 만든 `.tldraw` 스냅샷에는 전체 세트가 필요합니다(표 참고).
- **스크립트는 로드될 때마다 다시 실행됩니다 — 멱등적으로 만드세요.** 안정적인 ID와 함께
  `createShapeIfMissing`을 사용하세요. 그렇지 않으면 콘텐츠가 중복되고 사용자의 편집 내용이 덮어써집니다.
- **`signal`에서 정리하세요.** 모든 `store.listen` / `editor.on` / `setInterval`에
  `signal.addEventListener('abort', () => stop())`을 사용하세요. signal은 다시 실행하기 전과
  닫을 때 발생합니다.
- **스크립트 쓰기는 실행 취소에서 제외하세요:** `editor.run(fn, { history: 'ignore' })`.
- **창이 숨겨지면 `editor.on('tick')`이 일시 중지됩니다**(RAF 루프이기 때문). `setInterval`은
  계속 실행되지만 Electron이 백그라운드에서 이를 약 1초당 1회로 제한합니다.
- **API에는 `server.json`의 bearer token이 필요합니다.** 포트는 기본값이 아닐 수 있습니다
  (`server.listen(0)`이 하나를 선택함) — 항상 파일을 읽고 `7236`을 하드 코딩하지 마세요.
- **`tldraw` / `react` / `react-dom`만 import하세요** — Node 프로젝트가 아닙니다.

## 검증

- **도형 스키마(오프라인, 앱 없이):** `node scripts/validate_shapes.mjs` — 실제 tldraw
  스키마를 빌드하고 note/text/frame을 검증합니다. 성공 시 `3/3`을 출력합니다.
- **실시간 캔버스 편집:** `/exec` 후 `/api/search` → `api.getShapes(docId)`(반환값은
  `{ page, viewport, shapes }`)와 `api.getBindings(docId)`(배열)로 다시 읽습니다. 예상한
  도형/바인딩이 존재하는지 확인하세요. `api.getScreenshot(docId)`(반환값은
  `{ filePath, ... }`)로 PNG/JPEG를 가져와 `vision_analyze`로 검사합니다.
- **지속적인 스크립트 적용:** `GET /api/doc/:id/script-status`. 성공 상태는
  `state: "applied"`입니다(`currentDiskDigest === lastAppliedDigest === manifestSha256`,
  `pendingApply === false`, `lastApplyError === null`). 짧게 재시도한 뒤에도
  `"pending"` 상태라면 성공했다고 주장하지 말고 그렇게 보고하세요. `"error"`는 적용이
  실패했다는 뜻이므로 `errorLogPath`를 읽습니다.
