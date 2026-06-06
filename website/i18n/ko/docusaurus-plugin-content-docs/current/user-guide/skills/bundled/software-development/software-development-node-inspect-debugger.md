---
title: "Node Inspect Debugger — Node.js 디버깅"
sidebar_label: "Node Inspect Debugger"
description: "Node.js 디버깅"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Node Inspect Debugger

--inspect + Chrome DevTools Protocol CLI를 통한 Node.js 디버깅.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/software-development/node-inspect-debugger` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `debugging`, `nodejs`, `node-inspect`, `cdp`, `breakpoints`, `ui-tui` |
| Related skills | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`python-debugpy`](/docs/user-guide/skills/bundled/software-development/software-development-python-debugpy), [`debugging-hermes-tui-commands`](/docs/user-guide/skills/bundled/software-development/software-development-debugging-hermes-tui-commands) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Node.js Inspect Debugger

## 개요

`console.log`로 충분하지 않을 때 터미널에서 프로그래밍 방식으로 Node 내장 V8 인스펙터를 실행하세요. 실제 중단점(breakpoint), Step in/over/out, 콜스택 탐색, 로컬/클로저 스코프 덤프 및 일시 중지된 프레임에서의 임의의 표현식 평가를 수행할 수 있습니다.

두 가지 도구 중 하나를 선택하세요:

- **`node inspect`** — 내장 도구, 설치 불필요, CLI REPL. 빠른 확인에 가장 적합합니다.
- **`ndb` / `chrome-remote-interface`를 통한 CDP** — Node/Python에서 스크립팅 가능; 많은 중단점을 자동화하거나 실행 간 상태를 수집하거나 에이전트 루프에서 비대화형으로 디버깅할 때 가장 좋습니다.

**항상 `node inspect`를 우선적으로 사용하세요.** 항상 사용 가능하며 REPL이 빠릅니다.

## 사용 시기

- Node 테스트가 실패하고 중간 상태를 확인해야 할 때
- ui-tui가 충돌하거나 잘못 동작하며, 렌더링 전 React/Ink 상태를 검사하고 싶을 때
- tui_gateway 하위 프로세스(`_SlashWorker`, PTY 브리지 워커)가 오작동할 때
- 패치 없이 `console.log`가 도달할 수 없는 클로저 내의 값을 검사해야 할 때
- 성능: 실행 중인 프로세스에 연결하여 CPU 프로필 또는 힙 스냅샷을 캡처할 때

**다음과 같은 경우에는 사용하지 마세요:** 1분 이내에 `console.log`로 해결되는 문제들. 중단점 기반 디버깅은 무겁습니다; 확실한 이점이 있을 때 사용하세요.

## 빠른 참조: `node inspect` REPL

첫 번째 줄에서 일시 중지된 상태로 시작:

```bash
node inspect path/to/script.js
# 또는 tsx를 사용할 때
node --inspect-brk $(which tsx) path/to/script.ts
```

`debug>` 프롬프트에서 사용할 수 있는 명령:

| 명령 | 작업 |
|---|---|
| `c` 또는 `cont` | 계속(continue) |
| `n` 또는 `next` | 건너뛰기(step over) |
| `s` 또는 `step` | 들어가기(step into) |
| `o` 또는 `out` | 빠져나오기(step out) |
| `pause` | 실행 중인 코드 일시 중지 |
| `sb('file.js', 42)` | file.js 42번째 줄에 중단점 설정 |
| `sb(42)` | 현재 파일의 42번째 줄에 중단점 설정 |
| `sb('functionName')` | 함수가 호출될 때 중단 |
| `cb('file.js', 42)` | 중단점 지우기 |
| `breakpoints` | 모든 중단점 나열 |
| `bt` | 역추적 (콜스택) |
| `list(5)` | 현재 위치 주변의 소스 코드 5줄 표시 |
| `watch('expr')` | 일시 중지할 때마다 표현식 평가 |
| `watchers` | 감시 중인 표현식 표시 |
| `repl` | 현재 스코프의 REPL로 전환 (REPL을 종료하려면 Ctrl+C) |
| `exec expr` | 표현식 한 번 평가 |
| `restart` | 스크립트 재시작 |
| `kill` | 스크립트 종료 |
| `.exit` | 디버거 종료 |

**`repl` 하위 모드에서:** 로컬/클로저 변수 액세스를 포함한 모든 JS 표현식을 입력할 수 있습니다. `Ctrl+C`를 누르면 `debug>`로 돌아옵니다.

## 실행 중인 프로세스에 연결하기

프로세스가 이미 실행 중인 경우 (예: 수명이 긴 개발 서버 또는 TUI 게이트웨이):

```bash
# 1. SIGUSR1을 보내 기존 프로세스에서 인스펙터 활성화
kill -SIGUSR1 <pid>
# Node 출력: Debugger listening on ws://127.0.0.1:9229/<uuid>

# 2. 디버거 CLI 연결
node inspect -p <pid>
# 또는 URL로 연결
node inspect ws://127.0.0.1:9229/<uuid>
```

처음부터 인스펙터가 활성화된 상태로 프로세스를 시작하려면:

```bash
node --inspect script.js           # 127.0.0.1:9229에서 수신 대기, 계속 실행
node --inspect-brk script.js       # 수신 대기하며 첫 번째 줄에서 일시 중지
node --inspect=0.0.0.0:9230 script.js   # 사용자 정의 호스트:포트
```

tsx를 통한 TypeScript의 경우:

```bash
node --inspect-brk --import tsx script.ts
# 또는 구형 tsx의 경우
node --inspect-brk -r tsx/cjs script.ts
```

## 프로그래밍 방식 CDP (터미널에서 스크립팅)

자동화하고 싶을 때 — 많은 중단점을 설정하고, 스코프 상태를 캡처하고, 재현 스크립트를 작성할 때 — `chrome-remote-interface`를 사용하세요:

```bash
npm i -g chrome-remote-interface        # 또는 프로젝트 로컬
# 대상 시작:
node --inspect-brk=9229 target.js &
```

드라이버 스크립트 (`/tmp/cdp-debug.js`로 저장):

```javascript
const CDP = require('chrome-remote-interface');

(async () => {
  const client = await CDP({ port: 9229 });
  const { Debugger, Runtime } = client;

  Debugger.paused(async ({ callFrames, reason }) => {
    const top = callFrames[0];
    console.log(`PAUSED: ${reason} @ ${top.url}:${top.location.lineNumber + 1}`);

    // 로컬을 찾기 위해 스코프 탐색
    for (const scope of top.scopeChain) {
      if (scope.type === 'local' || scope.type === 'closure') {
        const { result } = await Runtime.getProperties({
          objectId: scope.object.objectId,
          ownProperties: true,
        });
        for (const p of result) {
          console.log(`  ${scope.type}.${p.name} =`, p.value?.value ?? p.value?.description);
        }
      }
    }

    // 일시 중지된 프레임에서 표현식 평가
    const { result } = await Debugger.evaluateOnCallFrame({
      callFrameId: top.callFrameId,
      expression: 'typeof state !== "undefined" ? JSON.stringify(state) : "n/a"',
    });
    console.log('state =', result.value ?? result.description);

    await Debugger.resume();
  });

  await Runtime.enable();
  await Debugger.enable();

  // URL 정규식 + 줄 번호로 중단점 설정
  await Debugger.setBreakpointByUrl({
    urlRegex: '.*app\\.tsx$',
    lineNumber: 119,       // 0-indexed (0부터 시작)
    columnNumber: 0,
  });

  await Runtime.runIfWaitingForDebugger();
})();
```

실행:

```bash
node /tmp/cdp-debug.js
```

Hermes 한정 참고: `chrome-remote-interface`는 `ui-tui/package.json`에 포함되어 있지 않습니다. 프로젝트를 더럽히고 싶지 않다면 일회용 위치에 설치하세요:

```bash
mkdir -p /tmp/cdp-tools && cd /tmp/cdp-tools && npm i chrome-remote-interface
NODE_PATH=/tmp/cdp-tools/node_modules node /tmp/cdp-debug.js
```

## Hermes ui-tui 디버깅

TUI는 Ink + tsx로 구축되었습니다. 두 가지 일반적인 시나리오:

### 개발 중인 단일 Ink 컴포넌트 디버깅

`ui-tui/package.json`에는 `npm run dev` (tsx --watch)가 있습니다. tsx를 직접 실행하여 `--inspect-brk`를 추가하세요:

```bash
cd /home/bb/hermes-agent/ui-tui
npm run build    # 첫 로드 시 트랜스파일이 필요 없도록 dist/를 한 번 생성
node --inspect-brk dist/entry.js
# 다른 터미널에서:
node inspect -p <node pid>
```

그런 다음 `debug>` 내부에서:

```
sb('dist/app.js', 220)     # 또는 의심스러운 렌더링이 발생하는 곳
cont
```

일시 중지되면 `repl` → `props`, 상태 참조(state refs), `useInput` 핸들러 값 등을 검사하세요.

### 실행 중인 `hermes --tui` 디버깅

TUI는 Python CLI에서 Node를 스폰합니다. 가장 쉬운 경로:

```bash
# 1. TUI 실행
hermes --tui &
TUI_PID=$(pgrep -f 'ui-tui/dist/entry' | head -1)

# 2. 해당 Node PID에서 인스펙터 활성화
kill -SIGUSR1 "$TUI_PID"

# 3. WS URL 찾기
curl -s http://127.0.0.1:9229/json/list | jq -r '.[0].webSocketDebuggerUrl'

# 4. 연결
node inspect ws://127.0.0.1:9229/<uuid>
```

TUI와 상호 작용(창에 입력)하면 실행이 계속 진행됩니다; 디버거는 `sb(...)`의 어느 곳에서든 중단점에서 일시 중지할 수 있습니다.

### `_SlashWorker` / PTY 하위 프로세스 디버깅

이들은 Node가 아닌 Python이므로 `python-debugpy` 스킬을 사용하세요. Node 부분(Ink UI, tui_gateway 클라이언트, `ui-tui/` 아래의 tsx 실행 테스트)에서만 이 스킬을 사용합니다.

## 디버거에서 Vitest 테스트 실행

```bash
cd /home/bb/hermes-agent/ui-tui
# 엔트리에서 일시 중지된 단일 테스트 파일 실행
node --inspect-brk ./node_modules/vitest/vitest.mjs run --no-file-parallelism src/app/foo.test.tsx
```

다른 터미널에서: `node inspect -p <pid>`, 그리고 `sb('src/app/foo.tsx', 42)`, `cont`.

`--no-file-parallelism` (vitest) 또는 `--runInBand` (jest)를 사용하여 오직 하나의 워커만 존재하게 하세요 — 풀(pool)을 디버깅하는 것은 고통스럽습니다.

## 힙 스냅샷 및 CPU 프로필 (비대화형)

위의 CDP 드라이버에서 Debugger를 `HeapProfiler` / `Profiler`로 교체하세요:

```javascript
// 5초 동안 CPU 프로필
await client.Profiler.enable();
await client.Profiler.start();
await new Promise(r => setTimeout(r, 5000));
const { profile } = await client.Profiler.stop();
require('fs').writeFileSync('/tmp/cpu.cpuprofile', JSON.stringify(profile));
// Chrome DevTools 열기 → Performance 탭에서 /tmp/cpu.cpuprofile 열기
```

```javascript
// 힙 스냅샷
await client.HeapProfiler.enable();
const chunks = [];
client.HeapProfiler.addHeapSnapshotChunk(({ chunk }) => chunks.push(chunk));
await client.HeapProfiler.takeHeapSnapshot({ reportProgress: false });
require('fs').writeFileSync('/tmp/heap.heapsnapshot', chunks.join(''));
```

## 일반적인 함정

1. **TS 소스의 잘못된 줄 번호.** 중단점은 `.ts`가 아닌 컴파일된 JS에 도달합니다. (a) 빌드된 `dist/*.js`에서 중단점을 걸거나, (b) 소스맵을 활성화(`node --enable-source-maps`)하고 `sb('src/app.tsx', N)`을 사용하세요 — 단, 이는 소스맵을 따르는 CDP 클라이언트에서만 작동합니다. `node inspect` CLI는 이를 지원하지 않습니다.

2. **`--inspect` vs `--inspect-brk`.** `--inspect`는 인스펙터를 시작하지만 일시 중지하지 않습니다; 너무 늦게 연결하면 스크립트가 첫 번째 중단점을 지나쳐버립니다. 코드가 실행되기 전에 중단점을 설정해야 한다면 `--inspect-brk`를 사용하세요.

3. **포트 충돌.** 기본값은 `9229`입니다. 여러 Node 프로세스를 검사하는 경우 `--inspect=0`(임의의 포트)을 전달하고 `/json/list`에서 실제 URL을 읽으세요:
   ```bash
   curl -s http://127.0.0.1:9229/json/list   # 호스트에서 검사 가능한 모든 대상 나열
   ```

4. **하위 프로세스.** 부모에서 `--inspect`를 사용한다고 해서 모든 하위 항목이 검사되는 것은 아닙니다. 모든 하위 항목에 전파하려면 `NODE_OPTIONS='--inspect-brk' node parent.js`를 사용하세요; 모든 항목에 고유한 포트가 필요하다는 점에 유의하세요 (`NODE_OPTIONS='--inspect'`가 상속될 때 Node가 포트를 자동 증가시킵니다).

5. **백그라운드 킬.** 대상이 일시 중지된 상태에서 `node inspect`에서 `Ctrl+C`를 누르면 대상은 계속 일시 중지된 상태로 남습니다. 먼저 `cont`를 하거나 대상을 명시적으로 `kill`하세요.

6. **에이전트 터미널을 통한 `node inspect` 실행.** PTY 친화적인 REPL입니다. Hermes에서는 `terminal(pty=true)` 또는 `background=true` + `process(action='submit', data='...')`와 함께 실행하세요. Non-PTY 포그라운드 모드는 일회성 명령에서는 작동하지만 대화형 스텝에서는 작동하지 않습니다.

7. **보안.** `--inspect=0.0.0.0:9229`는 임의의 코드 실행에 노출됩니다. 격리된 네트워크가 아닌 이상 항상 `127.0.0.1`(기본값)에 바인딩하세요.

## 검증 체크리스트

디버그 세션을 설정한 후 다음을 확인하세요:

- [ ] `curl -s http://127.0.0.1:9229/json/list`가 기대하는 대상을 정확히 반환하는가
- [ ] 첫 번째 중단점에 실제로 도달하는가 (그렇지 않다면 `--inspect-brk`를 빼먹었거나 실행이 완료된 후 연결했을 가능성이 높습니다)
- [ ] 일시 중지 시 소스 목록에 올바른 파일이 표시되는가 (불일치 = 소스맵 문제, 함정 1 참조)
- [ ] `repl`의 `exec process.pid`가 연결하려는 PID를 반환하는가

## 단발성(One-Shot) 레시피

**"왜 이 변수가 X번째 줄에서 undefined입니까?"**
```bash
node --inspect-brk script.js &
node inspect -p $!
# debug>
sb('script.js', X)
cont
# 일시 중지됨. 이제:
repl
> myVariable
> Object.keys(this)
```

**"이 함수로 들어가는 호출 경로는 무엇입니까?"**
```
debug> sb('suspectFn')
debug> cont
# 진입 시 일시 중지됨
debug> bt
```

**"이 비동기 체인이 멈췄습니다 — 어디서 멈췄습니까?"**
```
# --inspect (no -brk)로 시작하여 멈출 때까지 실행한 다음:
debug> pause
debug> bt
# 이제 멈춘 프레임을 볼 수 있습니다
```
