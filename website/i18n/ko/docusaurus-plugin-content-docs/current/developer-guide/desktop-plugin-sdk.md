---
sidebar_label: "데스크톱 플러그인 SDK"
title: "데스크톱 플러그인 SDK (@hermes/plugin-sdk)"
description: "네이티브 Hermes Desktop 앱을 확장하세요 — 창, 페이지, 사이드바 탐색, 상태 표시줄, 팔레트 명령, 키 바인딩, 테마 및 범위가 지정된 백엔드 네임스페이스를 import 하나와 빌드 단계 없이 사용할 수 있습니다."
---

# 데스크톱 플러그인 SDK

네이티브 [Hermes Desktop](/user-guide/desktop) 앱은 기여 방식으로 확장됩니다. 창의 모든
표면 — 창, 라우트, 사이드바 탐색, 상태 표시줄 항목, 팔레트 항목, 키 바인딩, 테마 — 이 하나의
중앙 레지스트리에 등록됩니다. 코어도 플러그인과 정확히 같은 방식으로 표면을 등록하므로,
플러그인은 나중에 덧붙인 기능이 아니라 실제 확장 방식입니다.

**데스크톱 플러그인**은 `HermesPlugin`을 default-export하는 단일 ESM 파일입니다.
`@hermes/plugin-sdk`라는 모듈 하나를 import하면 앱의 실시간 상태, 게이트웨이 JSON-RPC
창구, 범위가 지정된 REST/socket 백엔드 네임스페이스, React Query, 그리고 플러그인 UI가
기본적으로 네이티브처럼 보이게 해 주는 앱 자체 UI 키트를 모두 사용할 수 있습니다. 저장소를
복제하거나 `npm run build`를 실행하거나 앱 소스를 수정할 필요가 없습니다. 파일을
`$HERMES_HOME/desktop-plugins/<id>/plugin.js`에 넣으면 앱이 몇 초 안에 로드하고 저장할 때마다
핫 리로드합니다.

:::warning 웹 대시보드 플러그인 SDK가 아닙니다
Hermes에서 "플러그인"은 서로 관련 없는 여러 가지를 가리킵니다. 이 페이지는 **네이티브
데스크톱 앱**(`hermes desktop`) SDK — `@hermes/plugin-sdk` 모듈과
`$HERMES_HOME/desktop-plugins/` — 에 관한 것입니다. **웹 대시보드**(`hermes dashboard`)에는
`window.__HERMES_PLUGIN_SDK__`와 `manifest.json`을 사용하는 별도의 플러그인 시스템이 있으며,
[대시보드 확장하기](/user-guide/features/extending-the-dashboard)에 설명되어 있습니다.
Python CLI/게이트웨이 플러그인은 [Hermes 플러그인 만들기](/developer-guide/plugins)에
설명되어 있습니다. 이 셋은 코드, API, 제공 방식이 서로 공유되지 않습니다. 데스크톱 SDK와
대시보드 SDK가 공유하는 것은 백엔드 `plugin_api.py` 네임스페이스(`/api/plugins/<id>`)뿐입니다.
:::

## 개념 모델

SDK는 VS Code 모듈 모델을 따릅니다. 플러그인 작성자는 정확히 하나의 모듈을 import하며 앱
내부를 건드리지 않습니다(번들된 플러그인에서는 린트 경계로 차단되고, 디스크 플러그인에서는
resolve에 실패합니다). 기능은 계층으로 제공됩니다.

- **`host.state.*`** — 앱의 실시간 상태(nanostore atom)를 읽기 전용으로 보여주는 뷰: 현재 세션,
  cwd, 게이트웨이 상태, 모델, 프로필, 뷰포트.
- **`host.*` 액션** — 토스트 표시, 이동, 로그 끝부분 읽기, 게이트웨이 재시작, 게이트웨이 이벤트
  스트림 구독 등 엄선된 안전한 동작.
- **`host.request`** — 게이트웨이 JSON-RPC 창구: 세션, 설정, 스킬, cron 등 앱 자체가 호출하는 모든 것.
- **`ctx.rest` / `ctx.socket`** — `plugin_api.py`를 함께 제공할 때 플러그인 자체 백엔드 네임스페이스
  (`/api/plugins/<id>`).
- **`ui.*`** — 앱의 실제 컴포넌트, 테마 변수, 아이콘, 포매터로 구성된 디자인 언어. 따라서
  플러그인 UI가 앱과 픽셀 단위로 일치합니다.

## 두 가지 제공 방식

| 방식 | 위치 | 대상 | 빌드 단계 |
|------|------|------|------------|
| **디스크** (권장) | `$HERMES_HOME/desktop-plugins/<id>/plugin.js` | 사용자, 에이전트 | 없음 — 일반 ESM, 컴파일 없이 로드 |
| **통합 패키지** | `$HERMES_HOME/plugins/<id>/desktop/plugin.js` | 에이전트 측 코드도 함께 제공하는 플러그인 | 없음 — 동일한 디스크 파이프라인 |
| **번들** | `apps/desktop/src/plugins/<id>/plugin.tsx` | 앱과 함께 제공되는 트리 내 플러그인 | 앱 자체 Vite 빌드 |

세 방식 모두 동일한 `HermesPlugin` 계약을 따르고 **설정 → 플러그인**에 표시되며, 실행 중
활성화/비활성화할 수 있습니다. 통합 패키지는 에이전트 플러그인 폴더 안을 검색하는 디스크
입구일 뿐입니다 — [하나의 패키지, 두 SDK](#one-package-both-sdks)를 참고하세요. 이 페이지의
모든 내용은 디스크 입구(여러분과 에이전트가 작성하는 것)를 기준으로 작성되었습니다.
[번들 플러그인](#bundled-plugins)에서는 두 가지 차이점을 설명합니다. 현재 코어 트리에는
데스크톱 플러그인이 제공되지 않습니다 — 참고용 데모는 동반
[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins) 저장소에
있습니다.

## 빠른 시작 — 첫 플러그인

`$HERMES_HOME/desktop-plugins/hello/plugin.js`를 만드세요(기본값은 `~/.hermes/...`이며,
이름이 지정된 프로필에서는 `~/.hermes/profiles/<name>/...`입니다). 폴더 이름은 플러그인의
`id`와 같아야 합니다.

```javascript
// ~/.hermes/desktop-plugins/hello/plugin.js
import { host, haptic, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

function HelloPane() {
  const gateway = useValue(host.state.gateway)

  return jsxs('div', {
    className: 'flex h-full flex-col gap-2 p-3 text-sm',
    children: [
      jsx('div', { className: 'font-medium', children: 'Hello, Hermes' }),
      jsx('div', {
        className: 'text-(--ui-text-tertiary)',
        children: `gateway: ${gateway}`
      })
    ]
  })
}

export default {
  id: 'hello', // must match the folder name
  name: 'Hello',
  register(ctx) {
    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'hello',
      data: { placement: 'right', width: '260px' },
      render: () => jsx(HelloPane, {})
    })
    ctx.register({
      id: 'chip',
      area: 'statusBar.right',
      order: 130,
      render: () =>
        jsx('button', {
          type: 'button',
          className: 'px-1.5 text-[0.6875rem] text-(--ui-text-tertiary)',
          onClick: () => {
            haptic('tap')
            host.notify({ kind: 'info', message: 'Hello from my plugin!' })
          },
          children: 'hello'
        })
    })
  }
}
```

저장하세요. 앱은 `desktop-plugins/`를 감시하므로 몇 초 안에 파일을 로드하고 이후 저장할
때마다 제자리에서 핫 리로드합니다. 표시되지 않으면 ⌘K → **데스크톱 플러그인 다시 로드**를
실행하세요. 로드에 실패하면 토스트에 오류가 표시됩니다 — 수정하고 다시 저장하세요.

:::note JSX 없음, 빌드 없음
디스크 파일은 **컴파일되지 않은 상태로** 로드되므로 JSX 구문은 파싱되지 않습니다. UI는
`react/jsx-runtime`의 `jsx()` / `jsxs()` 호출(또는 `React.createElement`)로 작성하세요.
import할 수 있는 지정자는 `@hermes/plugin-sdk`, `react`, `react/jsx-runtime`뿐이며 — 그
외의 것은 의도적으로 resolve에 실패합니다.
:::

## 플러그인 계약

플러그인은 `HermesPlugin`을 default-export합니다.

```ts
interface HermesPlugin {
  /** Stable slug — becomes the `plugin:<id>` source and the id namespace. */
  id: string
  /** Human name for Settings / about UI. Defaults to `id`. */
  name?: string
  /** Registers on load when the user hasn't chosen (default true). Set false
   *  for opt-in plugins: they inventory in Settings ▸ Plugins, off until the
   *  user flips the switch. */
  defaultEnabled?: boolean
  /** Called once at load; wire contributions through `ctx`. */
  register: (ctx: PluginContext) => void
}
```

`register`는 **범위가 지정된** `PluginContext`를 받습니다. 레지스트리를 직접 건드리지
않습니다. 컨텍스트가 출처(`source: 'plugin:<id>'`)를 자동으로 태그하고 모든 기여 id에
네임스페이스(`<id>:<localId>`)를 지정하므로 플러그인 두 개가 충돌할 수 없습니다.

```ts
interface PluginContext {
  /** Resolved source tag, e.g. `'plugin:hello'`. */
  readonly source: string
  /** Register one contribution (id namespaced, source stamped). Returns a disposer. */
  register: (c: PluginContribution) => () => void
  /** Register several at once; the returned disposer removes all of them. */
  registerMany: (cs: PluginContribution[]) => () => void
  /** REST to this plugin's own backend namespace (`/api/plugins/<id>`). */
  rest: <T>(path: string, opts?: PluginRestOptions) => Promise<T>
  /** Live WebSocket to this plugin's own namespace. Returns a disposer. */
  socket: (path: string, onMessage: (data: unknown) => void) => () => void
  /** The curated OS door: native notification, open-external, reveal-in-file-manager, clipboard. */
  os: PluginOs
  /** Plugin-scoped JSON persistence (keys live under `hermes.plugin.<id>.`). */
  storage: PluginStorage
}
```

**기여(contribution)**는 모든 표면이 공유하는 하나의 기본 요소입니다.

```ts
interface Contribution {
  id: string          // you write the local id; the host namespaces it
  area: string        // WHERE it goes (a contribution-area constant)
  title?: string
  order?: number      // sort within the area (lower = earlier)
  when?: () => boolean // dynamic visibility; re-evaluated by the area
  enabled?: boolean
  render?: () => ReactNode  // the component to mount
  data?: unknown      // area-specific payload (see the cookbook)
}
```

영역에 따라 `render`, `data` 또는 둘 다를 제공합니다.

## 기여 영역 — 조리법

SDK에서 영역 상수를 import하세요. 각 영역에는 고유한 `data` 페이로드가 있습니다.

| 표면 | `area` | 제공할 항목 |
|---------|--------|-------------|
| 레이아웃 창 | `PANES_AREA` (`'panes'`) | `title` + `render` + `data: { placement, dock?, width?, height? }` |
| 전체 페이지 | `ROUTES_AREA` | `data: { path }` + `render` |
| 사이드바 탐색 | `SIDEBAR_NAV_AREA` | `data: { path, label, codicon }` |
| 상태 표시줄 | `STATUSBAR_AREAS.left` / `.right` | `render` (또는 `data`를 `StatusbarItem`으로) |
| 제목 표시줄 | `TITLEBAR_AREAS.left` / `.center` / `.right` | `data`를 `TitlebarTool`로, 또는 mount 범위의 `<Contribute>` |
| ⌘K 팔레트 | `PALETTE_AREA` | `data: PaletteContribution` |
| 키 바인딩 | `KEYBINDS_AREA` | `data: KeybindContribution` |
| 테마 | `THEMES_AREA` | `data`를 `DesktopTheme`으로 |
| 작성기 | `COMPOSER_AREAS.*` | 렌더 슬롯, 또는 미들웨어 / 첨부 파일 제공자 |

### 창

창은 레이아웃 트리의 타일입니다. `placement`는 의미론적 역할입니다 — 창은 해당 역할의
기존 창과 탭으로 쌓이며, 사용자는 이후 어디로든 드래그할 수 있습니다.

```javascript
ctx.register({
  id: 'pane',
  area: 'panes',
  title: 'my pane',
  data: { placement: 'right', width: '260px' },
  render: () => jsx(MyPane, {})
})
```

`placement`는 `'main' | 'left' | 'right' | 'top' | 'bottom'`입니다. 쌓지 않고 특정 **가장자리**에
배치하려면 `dock` 제스처를 추가하세요 — 창의 드롭 칩으로 드래그하는 것과 같습니다.

```javascript
// Below the conversation, 200px tall.
data: {
  placement: 'bottom',
  dock: { pane: 'workspace', pos: 'bottom' },
  height: '200px'
}
```

`dock.pane`은 모든 창 id가 될 수 있습니다(`workspace`는 기본 스레드이며, `sessions`,
`terminal`, `files`, `review`, `logs`도 있습니다). `dock.pos`는
`'top' | 'bottom' | 'left' | 'right' | 'center'`입니다. 창이 영역의 절반을 차지하지 않도록
`width`/`height`를 선언하세요.

플러그인이 제공한 유일한 창을 닫으면 해당 플러그인이 비활성화되며 **설정 → 플러그인**에서
다시 활성화할 수 있습니다. 플러그인이 여러 창을 제공하는 경우 하나를 닫아도 그 창만
닫히고 플러그인의 다른 창, 명령, 미들웨어는 활성 상태로 남습니다. **레이아웃 초기화**는
닫은 기여 창을 복원합니다.

### 페이지와 사이드바 탐색

라우트는 기본 제공 뷰처럼 작업 공간 창에 전체 페이지를 마운트합니다. 접근할 수 있도록
사이드바 탐색 행(및/또는 팔레트 명령)과 함께 사용하세요.

```javascript
import { ROUTES_AREA, SIDEBAR_NAV_AREA } from '@hermes/plugin-sdk'

ctx.registerMany([
  {
    id: 'page',
    area: ROUTES_AREA,
    data: { path: '/my-page' },
    render: () => jsx(MyPage, {})
  },
  {
    id: 'nav',
    area: SIDEBAR_NAV_AREA,
    data: { path: '/my-page', label: 'My Page', codicon: 'project' }
  }
])
```

`codicon`은 [VS Code codicon](https://microsoft.github.io/vscode-codicons/dist/codicon.html)
id입니다. 어디서든 `host.navigate('/my-page')`로 라우트로 이동할 수 있습니다.

### 상태 표시줄과 제목 표시줄

상태 표시줄 항목은 하단 표시줄의 왼쪽 또는 오른쪽 클러스터에 렌더링됩니다. 가장 간단한
방법은 `render` 함수이며, 일반 버튼에는 `data`를 `StatusbarItem`(`{ id, label?, icon?,
detail?, variant?, menuItems?, … }`)으로 사용합니다.

```javascript
import { STATUSBAR_AREAS, TITLEBAR_AREAS } from '@hermes/plugin-sdk'

ctx.register({
  id: 'count',
  area: STATUSBAR_AREAS.right,
  order: 120,
  render: () => jsx(MyStatus, {})
})
```

제목 표시줄 도구는 `TitlebarTool` 데이터(`{ id, label, icon, active?, onSelect? }`)로
`TITLEBAR_AREAS.left | .center | .right`에 배치됩니다.

### 팔레트 명령과 키 바인딩

```javascript
import { PALETTE_AREA, KEYBINDS_AREA } from '@hermes/plugin-sdk'

ctx.registerMany([
  {
    id: 'open',
    area: PALETTE_AREA,
    data: {
      id: 'my-page.open',
      label: 'Open My Page',
      keywords: ['my', 'page'],
      run: () => host.navigate('/my-page')
    }
  },
  {
    id: 'refresh',
    area: KEYBINDS_AREA,
    data: {
      id: 'my-page.refresh',
      label: 'Refresh My Page',
      category: 'My Plugin',
      defaults: ['mod+shift+r'],
      run: () => void doRefresh()
    }
  }
])
```

키 바인딩은 설정에서 사용자가 다시 지정할 수 있으며, `defaults`는 초기 바인딩일 뿐입니다.

### 테마

테마 기여는 전체 `DesktopTheme`을 `data`로 제공합니다(name, label, colors, …). 기본 제공
테마와 마찬가지로 테마 선택기에 표시됩니다.

```javascript
import { THEMES_AREA } from '@hermes/plugin-sdk'

ctx.register({ id: 'noir', area: THEMES_AREA, data: myDesktopTheme })
```

### 작성기 확장

`COMPOSER_AREAS`(`top`, `bottom`, `leading`, `actions`, `attachments`, `middleware`)를
사용하면 플러그인이 메시지 작성기 주변에 컨트롤을 추가하거나, 첨부 파일 소스를 제공하거나,
전송 전에 초안을 변환할 수 있습니다(`handler(draft) => draft | null`을 포함하는
`ComposerMiddleware`).

### 마운트 범위 크롬(`Contribute`)

`ctx.register`는 **영구적인** 기여에 사용합니다. 이미 화면에 표시된 컴포넌트와 함께
크롬이 생성되고 사라져야 한다면(페이지가 언마운트될 때 페이지 자체의 제목 표시줄 컨트롤도
사라지는 경우), 대신 그 컴포넌트 안에서 `<Contribute>`를 렌더링하세요.

```javascript
import { Contribute, TITLEBAR_AREAS } from '@hermes/plugin-sdk'

jsx(Contribute, {
  area: TITLEBAR_AREAS.center,
  id: 'my-page:switcher', // namespace with your slug
  children: jsx(MySwitcher, {})
})
```

마운트 시 등록되고 언마운트 시 자동으로 정리됩니다.

## 호스트 API

플러그인 어디서든 `host`의 모든 기능에 접근할 수 있습니다. 상태 atom은 읽기 전용이며,
핸들러에서는 `.get()`으로 읽고 컴포넌트에서는 `useValue(atom)`으로 구독합니다.

```ts
host.state.activeSessionId  // ReadableAtom<string | null>
host.state.cwd              // ReadableAtom<string>
host.state.gateway          // ReadableAtom<string>  ('idle' | 'connecting' | 'open' | …)
host.state.model            // ReadableAtom<string>
host.state.profile          // ReadableAtom<string>
host.state.viewport         // ReadableAtom<{ width, height, narrow }>

host.notify({ kind, message, title?, detail?, action? })  // toast; returns id
host.notifyError(error, fallbackMessage)                   // toast an error
ctx.os.notify({ title, body?, silent? })   // native OS notification (attributed to your plugin)
ctx.os.openExternal(url)                   // OS default handler (browser, mail, spotify:) → Promise<boolean>
ctx.os.revealPath(path)                    // reveal in Finder / Explorer → Promise<boolean>
ctx.os.writeClipboard(text)                // system clipboard → Promise<boolean>
host.navigate('/route')                    // hash-route navigation
host.openSession(id, { profile?, intent? }) // open a stored session core-style;
                                           //   profile: soft-swap to that profile's backend first
                                           //   intent: 'in-place' (default) | 'stack' | 'tab' | 'window'
host.newChat(profile?)                     // fresh chat draft, optionally in another profile
host.onEvent(type, fn)                     // gateway event stream ('*' = all); returns disposer
host.logs(...)                             // tail an app log file
host.status()                              // one-shot system status snapshot
host.restartGateway()                      // restart the backend gateway
host.request<T>(method, params?)           // gateway JSON-RPC — the real power
```

`host.request`는 앱 자체가 사용하는 동일한 JSON-RPC입니다(세션, 설정, 스킬, cron,
kanban, …). 프로필 형태의 플러그인에는 전용 메서드도 제공됩니다:
`profiles.list`(각 프로필과 가장 최근 대화를 `last_session`으로 포함하며, 프로필별 DB
조회를 건너뛰려면 `include_sessions: false`를 전달) 및 `profiles.create`(`name`,
`description`, `clone_from`, `clone_all`, `no_skills`, `soul`, 선택적 `model` + `provider`
고정) — 대시보드의 `/api/profiles` REST 라우트에 대응하는 ws 쌍입니다.
`host.onEvent`는 실시간 게이트웨이 이벤트(메시지 델타, 세션 수명 주기, 도구 활동)를
스트리밍합니다. 리스너는 격리되어 있으므로 리스너에서 발생한 오류가 앱 디스패치를 방해할
수 없습니다. 모든 `host` 창구는 비동기 안전합니다. 일반 브라우저에서 데스크톱 브리지가
없는 경우처럼 내부 헬퍼의 동기 throw도 오류 경계 충돌이 아니라 `.catch()`로 확인할 수 있는
거부(rejection)가 됩니다.

`ctx.os`는 엄선된 OS 창구입니다 — 플러그인이 앱 창 밖으로 나가는 모든 방법을 플러그인에
귀속된 하나의 네임스페이스로 제공합니다. `ctx.os.notify`는 **네이티브 OS 알림**을 게시하며,
앱 자체의 승인/턴 알림과 동일한 Electron 파이프라인을 사용합니다. 사용자가 Hermes에서
벗어나 백그라운드에 있거나 포커스를 잃은 동안에만 실행됩니다. 앱을 보고 있을 때의 인앱
토스트에는 `host.notify`를 사용하세요. 사용자는 설정 ▸ 알림 ▸ "플러그인 알림"에서 기기별로
이를 끌 수 있고, 같은 플러그인의 반복 알림은 제한되므로 진짜 중요한 이벤트를 알리는 신호로
사용하세요 — 로그처럼 사용하면 안 됩니다. 다른 창구(`openExternal`, `revealPath`,
`writeClipboard`)는 기능을 사용할 수 없을 때(구형 데스크톱 셸, 일반 브라우저) 예외를
발생시키는 대신 `false`로 resolve됩니다 — 브리지를 확인하지 말고 결과를 기준으로 분기하세요.

## 데이터 계층 — React Query + nanostore

플러그인은 앱의 단일 `QueryClient`를 공유하므로 플러그인 쿼리도 코어 화면과 똑같이 캐시,
중복 제거, 폴링, 무효화가 됩니다 — 직접 fetch 루프를 만들지 마세요.

```javascript
import { useQuery, useMutation, useQueryClient, atom, computed, useValue } from '@hermes/plugin-sdk'

function MyPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ['my-plugin', 'items'],
    queryFn: () => host.request('my.list', {})
  })
  // …
}
```

트리거와 패널(또는 폴링 루프) 사이에서 공유하는 상태에는 `atom` / `computed`를 사용하세요
— `host.state`가 사용하는 것과 동일한 기본 요소입니다. 값을 렌더링하는 말단에서
`useValue`로 구독하세요. React **바깥**에서 쿼리를 무효화하려면(예: `ctx.socket` 프레임이
도착한 경우) 공유 `queryClient`를 import하세요.

```javascript
import { queryClient } from '@hermes/plugin-sdk'

ctx.socket('/events', () => {
  queryClient.invalidateQueries({ queryKey: ['my-plugin', 'items'] })
})
```

## UI 키트와 테마

앱의 실제 컴포넌트를 직접 import하면 UI가 기본적으로 네이티브처럼 보입니다.

> `Button`, `Input`, `Textarea`, `Select*`, `Switch`, `Checkbox`,
> `SegmentedControl`, `Tabs*`, `Dialog*`, `ConfirmDialog`, `DropdownMenu*`,
> `ContextMenu*`, `Popover*`, `Tip`/`Tooltip*`, `Badge`, `Kbd`/`KbdGroup`,
> `SearchField`, `ScrollArea`, `Separator`, `Skeleton`, `GlyphSpinner`, `Loader`,
> `EmptyState`, `ErrorState`, `CopyButton`, `StatusDot`, `LogView`, `Codicon`,
> `DecodeText`.

추가 헬퍼: `cn`(클래스 병합), `icons.*`(앱의 lucide 세트), `haptic`, `profileColor` /
`profileColorSoft`(결정론적 식별 색상), 시간 포매터 `relativeTime` / `fmtDateTime` /
`fmtDayTime` / `coarseElapsed`, `useI18n`(지역화된 문자열 — 플러그인도 번역 가능),
`evaluateRuntimeReadiness`.

**색상을 하드코딩하지 말고 테마 변수를 사용하세요.** 창은 이미 앱의 편집기 배경 위에 있으므로
배경은 그대로 두고 나머지는 변수로 처리하세요: `var(--ui-text-secondary)`,
`var(--ui-text-tertiary)`, `var(--ui-text-quaternary)`, `var(--ui-stroke-secondary)`,
`var(--ui-accent)`. 캔버스에 그릴 때는 `getComputedStyle(canvas).getPropertyValue('--ui-accent')`로
한 번만 resolve하세요. 이렇게 해야 모든 테마에서 플러그인 스타일이 자동으로 바뀝니다.

## 플러그인 백엔드

플러그인에 서버 측 작업이 필요하다면 Python `plugin_api.py`를 함께 제공하고
`ctx.rest` / `ctx.socket`을 통해 접근하세요 — **구조적으로** 플러그인 범위가 지정된
네임스페이스입니다.

### 하나의 패키지, 두 SDK {#one-package-both-sdks}

데스크톱 UI와 에이전트 측 코드(Python 플러그인, 백엔드 라우트, 스킬)가 모두 필요한 기능도
서로 의존하는 두 설치물로 나눌 필요가 없습니다. 데스크톱 앱은 `$HERMES_HOME/plugins/<id>/`
— 일반 에이전트 플러그인 루트 — 도 검색해 `desktop/plugin.js`를 찾고, 독립 디스크 입구와
정확히 같은 파이프라인(핫 리로드 포함)으로 로드합니다.

```
~/.hermes/plugins/<id>/           # ONE installable folder
├── plugin.yaml                   # the agent half: tools, hooks, commands
├── skills/…
├── dashboard/
│   ├── manifest.json             # { "name": "<id>", "api": "plugin_api.py" }
│   └── plugin_api.py             # backend routes → /api/plugins/<id>/
└── desktop/
    └── plugin.js                 # the desktop half: panes, commands, ctx.rest
```

`desktop/plugin.js` 부분은 일반 디스크 플러그인입니다 — 동일한 계약, 동일한 import,
옆에 있는 `plugin_api.py`에 도달하는 동일한 `ctx.rest('/…')`를 사용합니다. 기능의 설치,
공유 또는 제거가 하나의 폴더로 처리됩니다.

두 활성화 스위치는 의도적으로 여전히 적용되며 둘 다 기본값은 **꺼짐**입니다. 데스크톱
부분은 opt-in으로 제공되어 **설정 → 플러그인**에 목록으로 나타나지만 사용자가 토글할
때까지 비활성 상태로 유지됩니다. 이는 아래 보안 경계인 `config.yaml`의 Python 부분
`plugins.enabled` 게이트와 일치합니다. `~/.hermes/plugins`에 패키지를 넣는 것만으로는
사용자가 허용하기 전까지 어떤 표면에서도 동작하지 않습니다. 백엔드 부분이 꺼져 있어도
데스크톱 부분은 우아하게 성능이 저하됩니다 — `ctx.rest`는 충돌하지 않고 오류를 반환합니다.

:::note
검색은 데스크톱 앱이 실행되는 컴퓨터에서만 수행됩니다. 원격 백엔드에 연결할 때 원격
컴퓨터의 `~/.hermes/plugins`는 파일 시스템으로 접근할 수 없으므로 — 로컬에 설치된
패키지만 데스크톱 부분을 제공합니다(독립 입구와 동일한 규칙).
:::

### Python 부분

데스크톱 플러그인은 대시보드 플러그인 백엔드 마운트를 재사용합니다. 일반 Hermes 플러그인의
`dashboard/` 하위 폴더에 백엔드를 넣고 `manifest.json`에 선언하세요.

```
~/.hermes/plugins/<id>/
└── dashboard/
    ├── manifest.json      # { "name": "<id>", "api": "plugin_api.py" }
    └── plugin_api.py      # exports `router = APIRouter()`
```

```python
# plugin_api.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/board")
async def board():
    return {"items": ["one", "two", "three"]}

@router.post("/action")
async def action(body: dict):
    return {"ok": True, "received": body}
```

라우트는 `/api/plugins/<id>/` 아래에 마운트됩니다(`GET /api/plugins/<id>/board`, …).
백엔드 코드는 게이트웨이 프로세스 안에서 실행되므로 hermes-agent 코드베이스
(`hermes_state`, `hermes_cli.config`, …)를 직접 import할 수 있습니다. 전체 백엔드 참고는
[대시보드 확장하기 → 백엔드 API 라우트](/user-guide/features/extending-the-dashboard#backend-api-routes)를
참고하세요 — 마운트 방식은 동일합니다.

:::caution Python 백엔드는 별도로 게이트됩니다
데스크톱 **설정 → 플러그인** 패널에서 플러그인을 활성화하는 것은 렌더러 측 선택이며
Python을 import하지 않습니다. 사용자 플러그인의 `plugin_api.py`는 `config.yaml`의
`plugins.enabled` 허용 목록에 있고(`plugins.disabled`에는 없어야 함) 해당할 때만 import됩니다.
프로젝트 플러그인(`./.hermes/`)은 Python을 자동으로 import하지 않습니다. 이는 누락이 아니라
보안 경계입니다(GHSA-mcfc-hp25-cjv7).
:::

### 플러그인에서 호출하기

```javascript
register(ctx) {
  // REST — namespace-relative path.
  const load = () => ctx.rest('/board')                 // GET /api/plugins/<id>/board
  const act  = () => ctx.rest('/action', { method: 'POST', body: { go: true } })

  // Live twin — a WebSocket to your own namespace.
  const stop = ctx.socket('/events', frame => {
    queryClient.invalidateQueries({ queryKey: [ctx.source, 'board'] })
  })
}
```

`ctx.rest`는 프로필을 인식하고 경로 탐색(`..`)을 거부하므로 다른 플러그인의 API나 코어
라우트를 이를 통해 지정할 수 없습니다. `PluginRestOptions`는
`{ method?, body?, upload?: { filename, contentType?, bytes }, timeoutMs? }`입니다.

`ctx.socket`은 정리될 때까지 백오프로 자동 재연결합니다. **OAuth 원격에서는 no-op으로
resolve됩니다**(일회성 WS 티켓은 코어가 관리합니다) — socket은 폴링을 대체하는 것이 아니라
가속기로 취급하세요. 어떤 socket이든 끊길 수 있으므로 모든 소비자에게는 폴링 대체 수단이
필요합니다.

자체 네임스페이스가 아닌 게이트웨이 전체 데이터에는 `host.request`(JSON-RPC)와
`host.onEvent`(게이트웨이 이벤트 스트림)를 사용하세요.

## 설정, 활성화 상태, 저장소

활성화 여부와 관계없이 모든 플러그인은 **설정 → 플러그인**에 목록으로 표시되며 사용자는
실행 중에 토글하고(앱 재시작 불필요), 폴더를 열거나 다시 검색할 수 있습니다. 사용자의 선택은
기억됩니다.

- 아직 선택 없음 → 플러그인 자체의 `defaultEnabled`(기본값 `true`). opt-in 플러그인으로
  제공하여 사용자가 켤 때까지 비활성으로 두려면 `defaultEnabled: false`로 설정하세요.
- 명시적 선택 → 재시작 후에도 유지되고 적용됩니다. 비활성화된 플러그인은 그대로 두세요;
  사용자가 끈 것입니다.

플러그인 자체 상태는 플러그인에 네임스페이스가 지정된 `ctx.storage`
(`hermes.plugin.<id>.*`)로 저장하므로 플러그인이 서로 읽거나 덮어쓸 수 없습니다.

```javascript
ctx.storage.set('lastTab', 'board')
const tab = ctx.storage.get('lastTab', 'summary')
ctx.storage.remove('lastTab')
```

## 번들 플러그인

플러그인은 `apps/desktop/src/plugins/<id>/plugin.tsx`에 트리 내부 형태로 제공할 수 있습니다
(`HermesPlugin`을 default export). 부팅 시 `discoverBundledPlugins()`가 이를 검색하므로
import나 레지스트리 수정이 필요 없으며, 디스크 플러그인과 정확히 같은 목록 및 실행 중
활성화/비활성화 계약을 공유합니다. 차이점은 두 가지입니다.

1. 앱의 Vite 빌드를 거치므로 **실제 JSX**를 작성하고 `@hermes/plugin-sdk` 별칭으로 SDK를
   import할 수 있습니다.
2. 여전히 `@hermes/plugin-sdk` + `react`만 사용하도록 린트 경계가 적용됩니다 — `@/…` 앱
   내부는 사용할 수 없습니다.

현재 코어 트리에는 데스크톱 플러그인이 제공되지 않습니다. 제공되는 앱은 깔끔하게 유지되며
데모는 동반 저장소
[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins)에 있습니다.

## 보안 모델

로드된 플러그인은 렌더러 영역에서 ESM으로 평가되며 **전체 앱 권한**을 가집니다 — React
싱글턴, 전체 SDK(`host.request` 게이트웨이 RPC, `ctx.rest`, 저장소, `navigate`)를 사용할 수
있습니다. 로더가 제공하는 격리는 **오류 격리만** 의미합니다. 플러그인이 앱을 중단시킬 수는
없지만(기여는 오류 범위가 제한되고 리스너는 격리됨), 앱이 할 수 있는 모든 작업을 할 수
있습니다.

이는 **로컬** 소스에는 허용됩니다 — 디스크 파일은 이미 컴퓨터에서 코드를 실행할 수
있으므로, 디스크 입구는 여러분(또는 에이전트)이 작성한 로컬 파일만 로드합니다. 선택적
`integrity`(`sha256-…`) 검사는 바이트가 해시와 일치하는지만 증명하며 **샌드박스를 제공하지
않습니다**. 향후 원격 소스 입구가 추가되려면 실제 경계(iframe/worker + CSP + capability
gating)가 필요하므로, 이 파이프라인을 신뢰 경계로 취급하지 마세요.

## 주의할 점

- **디스크 플러그인에서는 JSX가 파싱되지 않습니다.** 파일은 컴파일되지 않은 상태로 로드되므로
  JSX 구문 대신 `jsx()` / `jsxs()`(또는 `React.createElement`)를 사용하세요. (번들 플러그인은
  빌드되므로 JSX를 사용할 수 있습니다.)
- **resolve되는 지정자는 세 가지뿐입니다:** `@hermes/plugin-sdk`, `react`, `react/jsx-runtime`.
  그 외의 import는 로드 전에 오류를 발생시킵니다.
- **색상을 하드코딩하지 마세요**(`#000`, `black`, `rgb(...)`). 배경은 그대로 두고 모든 항목에
  테마 변수(`var(--ui-*)`)를 사용하세요.
- **import한 것만 참조하세요.** import를 잊은 컴포넌트(예: `StatusDot`)는 렌더링 시
  `ReferenceError`가 됩니다 — `jsx()` 호출의 모든 식별자가 import 줄에 있는지 확인하세요.
- **핸들러에서는 상태를 명령형으로 읽으세요**(`$atom.get()`). 렌더 클로저에서 읽으면 빠른
  이벤트가 오래된 값을 보게 됩니다. 값을 렌더링하는 말단에서만 구독(`useValue`)하세요.
- **캔버스 창은 컨테이너를 추적해야 합니다.** `ResizeObserver`를 사용해 캔버스의 크기를
  조정하세요(CSS만이 아니라 width/height 속성도 설정) — 창은 계속 크기가 바뀝니다.
- **`host.request`를 몇 초보다 빠르게 폴링하지 마세요.** `host.onEvent` / `ctx.socket`을
  우선 사용하고 React Query가 중복 요청을 제거하도록 하세요.
- **OAuth 원격에서 `ctx.socket`은 no-op입니다.** 항상 폴링 대체 수단을 마련하세요.

## 참고

### SDK export 한눈에 보기

| 범주 | Exports |
|----------|---------|
| 호스트 | `host` (`.state.*`, `.notify`, `.notifyError`, `.navigate`, `.onEvent`, `.logs`, `.status`, `.restartGateway`, `.request`) |
| 플러그인 계약 | `HermesPlugin`, `PluginContext`, `PluginContribution`, `PluginStorage`, `PluginOs`, `PluginRestOptions`, `PluginNativeNotificationInput`, `Contribution` |
| 영역 상수 | `PANES_AREA`, `ROUTES_AREA`, `SIDEBAR_NAV_AREA`, `STATUSBAR_AREAS`, `TITLEBAR_AREAS`, `PALETTE_AREA`, `KEYBINDS_AREA`, `THEMES_AREA`, `COMPOSER_AREAS` |
| 영역 페이로드 | `RouteContribution`, `SidebarNavContribution`, `StatusbarItem`, `TitlebarTool`, `PaletteContribution`, `KeybindContribution`, `ComposerMiddleware`, `ComposerAttachmentProvider` |
| React / 상태 | `useValue`, `atom`, `computed`, `useQuery`, `useMutation`, `useQueryClient`, `queryClient`, `Contribute` |
| UI 키트 | `Button`, `Input`, `Textarea`, `Select*`, `Switch`, `Checkbox`, `SegmentedControl`, `Tabs*`, `Dialog*`, `ConfirmDialog`, `DropdownMenu*`, `ContextMenu*`, `Popover*`, `Tip`/`Tooltip*`, `Badge`, `Kbd`/`KbdGroup`, `SearchField`, `ScrollArea`, `Separator`, `Skeleton`, `GlyphSpinner`, `Loader`, `EmptyState`, `ErrorState`, `CopyButton`, `StatusDot`, `LogView`, `Codicon`, `DecodeText` |
| 헬퍼 | `cn`, `icons`, `haptic`, `useI18n`, `profileColor`, `profileColorSoft`, `relativeTime`, `fmtDateTime`, `fmtDayTime`, `coarseElapsed`, `evaluateRuntimeReadiness` |

항상 최신인 표준 export 목록은 `apps/desktop/src/sdk/index.ts`입니다.

### 에이전트: `hermes-desktop-plugins` 스킬

에이전트가 데스크톱 플러그인을 작성할 때는 번들된 **`hermes-desktop-plugins`** 스킬을
로드해야 합니다 — 이 페이지와 동일한 계약을 에이전트 관점에서 담고 있으며 바로 복사할 수
있는 `templates/plugin.js`를 제공합니다. 이 페이지는 사람/개발자용 참고 자료이고, 스킬은
실무 체크리스트입니다.

## 문제 해결

**플러그인이 표시되지 않습니다.** 파일이 `$HERMES_HOME/desktop-plugins/<id>/plugin.js`에
있고 폴더 이름이 export한 `id`와 일치하는지 확인하세요. ⌘K → **데스크톱 플러그인 다시 로드**를
실행하세요. 실패 원인을 표시하는 오류 토스트를 앱에서 확인하고 `hermes logs gui -f`의
끝부분을 읽으세요.

**로드 시 "unsupported import"가 표시됩니다.** 디스크 플러그인은
`@hermes/plugin-sdk`, `react`, `react/jsx-runtime`만 import할 수 있습니다. 다른 import를
제거하세요.

**`jsx` 요소가 아무것도 렌더링하지 않거나 `ReferenceError`를 발생시킵니다.**
`jsx()` 호출에서 사용한 식별자를 import하지 않았습니다. import 줄에 추가하세요.

**`ctx.rest`가 404를 반환합니다.** 백엔드가 마운트되지 않았습니다:
`~/.hermes/plugins/<id>/dashboard/manifest.json`에 `"api": "plugin_api.py"`가 있는지,
`config.yaml`의 `plugins.enabled`에 플러그인이 있는지 확인하고 게이트웨이를 재시작하세요
(백엔드 라우트는 시작할 때 마운트됩니다). `Failed to load plugin <id> API routes`를
찾으려면 `~/.hermes/logs/errors.log`의 끝부분을 읽으세요.

**`ctx.socket`이 절대 실행되지 않습니다.** OAuth 원격에서는 설계상 no-op입니다 — 폴링
대체 수단을 사용하세요. 그 외에는 백엔드가 네임스페이스 아래에 일치하는
`@router.websocket(...)` 라우트를 제공하는지 확인하세요.

**테마를 바꾼 뒤 색상이 이상합니다.** 색상을 하드코딩했습니다. `var(--ui-*)` 테마 변수로
바꾸세요.
