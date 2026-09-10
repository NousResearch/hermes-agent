---
title: "Node Inspect Debugger — --inspect + Chrome DevTools Protocol CLI로 Node.js 디버깅"
sidebar_label: "Node Inspect Debugger"
description: "--inspect + Chrome DevTools Protocol CLI로 Node.js 디버깅"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Node Inspect Debugger

--inspect + Chrome DevTools Protocol CLI로 Node.js를 디버깅합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨 (기본 설치) |
| 경로 | `skills/software-development/node-inspect-debugger` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `debugging`, `nodejs`, `node-inspect`, `cdp`, `breakpoints`, `ui-tui` |
| 관련 스킬 | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`python-debugpy`](/docs/user-guide/skills/bundled/software-development/software-development-python-debugpy) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침입니다.
:::

# Node.js Inspect Debugger

## 개요

`console.log`만으로 충분하지 않을 때는 터미널에서 Node의 내장 V8 inspector를 프로그래밍 방식으로 구동하세요. 실제 중단점, 단계별 진입/다음/이탈 실행, 호출 스택 탐색, 로컬/클로저 스코프 덤프, 일시 중지된 프레임에서의 임의 표현식 평가를 사용할 수 있습니다.

두 가지 도구 중 하나를 선택하세요.

- **`node inspect`** — 설치가 필요 없는 내장 CLI REPL입니다. 빠르게 확인할 때 가장 좋습니다.
- **`ndb` / CDP via `chrome-remote-interface`** — Node/Python에서 스크립트로 실행할 수 있습니다. 여러 중단점을 자동화하거나, 실행 간 상태를 수집하거나, 에이전트 루프에서 비대화형으로 디버깅할 때 가장 좋습니다.

**먼저 `node inspect`를 사용하세요.** 항상 사용할 수 있고 REPL이 빠릅니다.

## 사용 시점

- Node 테스트가 실패했고 중간 상태를 확인해야 할 때
- ui-tui가 충돌하거나 잘못 동작하여 렌더링 전에 React/Ink 상태를 검사하고 싶을 때
- tui_gateway 자식 프로세스(`_SlashWorker`, PTY bridge workers)가 제대로 동작하지 않을 때
- 패치하지 않고는 `console.log`로 접근할 수 없는 클로저의 값을 검사해야 할 때
- 성능: 실행 중인 프로세스에 연결하여 CPU 프로파일 또는 힙 스냅샷을 수집할 때

**다음에는 사용하지 마세요:** `console.log`로 1분 안에 해결되는 문제. 중단점 기반 디버깅은 더 무겁기 때문에, 실제로 얻는 이점이 있을 때 사용하세요.

## 빠른 참고: `node inspect` REPL

첫 번째 줄에서 일시 중지된 상태로 시작합니다.

```bash
node inspect path/to/script.js
# or with tsx
node --inspect-brk $(which tsx) path/to/script.ts
```

`debug>` 프롬프트에서 다음 명령을 사용할 수 있습니다.

| 명령 | 동작 |
|---|---|
| `c` 또는 `cont` | 계속 실행 |
| `n` 또는 `next` | 다음 단계로 실행 |
| `s` 또는 `step` | 단계 안으로 들어가기 |
| `o` 또는 `out` | 단계 밖으로 나가기 |
| `pause` | 실행 중인 코드 일시 중지 |
| `sb('file.js', 42)` | file.js 42번째 줄에 중단점 설정 |
| `sb(42)` | 현재 파일 42번째 줄에 중단점 설정 |
| `sb('functionName')` | 함수가 호출될 때 중단 |
| `cb('file.js', 42)` | 중단점 해제 |
| `breakpoints` | 모든 중단점 나열 |
| `bt` | 백트레이스 (호출 스택) |
| `list(5)` | 현재 위치 주변의 소스 5줄 표시 |
| `watch('expr')` | 일시 중지될 때마다 expr 평가 |
| `watchers` | 감시 중인 표현식 표시 |
| `repl` | 현재 스코프의 REPL로 진입 (REPL을 종료하려면 Ctrl+C) |
| `exec expr` | 표현식을 한 번 평가 |
| `restart` | 스크립트 재시작 |
| `kill` | 스크립트 종료 |
| `.exit` | 디버거 종료 |

**`repl` 하위 모드에서는:** 로컬 변수와 클로저 변수에 접근하는 것을 포함해 모든 JS 표현식을 입력할 수 있습니다. `Ctrl+C`를 누르면 `debug>`로 돌아갑니다.

## 실행 중인 프로세스에 연결

프로세스가 이미 실행 중인 경우(예: 장시간 실행되는 개발 서버 또는 TUI gateway) 다음과 같이 합니다.

```bash
# 1. Send SIGUSR1 to enable the inspector on an existing process
kill -SIGUSR1 <pid>
# Node prints: Debugger listening on ws://127.0.0.1:9229/<uuid>

# 2. Attach the debugger CLI
node inspect -p <pid>
# or by URL
node inspect ws://127.0.0.1:9229/<uuid>
```

처음부터 inspector를 활성화한 상태로 프로세스를 시작하려면 다음과 같이 합니다.

```bash
node --inspect script.js           # listen on 127.0.0.1:9229, keep running
node --inspect-brk script.js       # listen AND pause on first line
node --inspect=0.0.0.0:9230 script.js   # custom host:port
```

tsx를 통한 TypeScript의 경우:

```bash
node --inspect-brk --import tsx script.ts
# or older tsx
node --inspect-brk -r tsx/cjs script.ts
```

## 프로그래밍 방식의 CDP (터미널에서 스크립팅)

자동화(여러 중단점 설정, 스코프 상태 캡처, 재현 스크립트 작성)가 필요할 때는 `chrome-remote-interface`를 사용하세요.

```bash
npm i -g chrome-remote-interface        # or project-local
# Start your target:
node --inspect-brk=9229 target.js &
```

드라이버 스크립트(`/tmp/cdp-debug.js`로 저장):

```javascript
const CDP = require('chrome-remote-interface');

(async () => {
  const client = await CDP({ port: 9229 });
  const { Debugger, Runtime } = client;

  Debugger.paused(async ({ callFrames, reason }) => {
    const top = callFrames[0];
    console.log(`PAUSED: ${reason} @ ${top.url}:${top.location.lineNumber + 1}`);

    // Walk scopes for locals
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

    // Evaluate an expression in the paused frame
    const { result } = await Debugger.evaluateOnCallFrame({
      callFrameId: top.callFrameId,
      expression: 'typeof state !== "undefined" ? JSON.stringify(state) : "n/a"',
    });
    console.log('state =', result.value ?? result.description);

    await Debugger.resume();
  });

  await Runtime.enable();
  await Debugger.enable();

  // Set a breakpoint by URL regex + line
  await Debugger.setBreakpointByUrl({
    urlRegex: '.*app\\.tsx$',
    lineNumber: 119,       // 0-indexed
    columnNumber: 0,
  });

  await Runtime.runIfWaitingForDebugger();
})();
```

실행합니다.

```bash
node /tmp/cdp-debug.js
```

Hermes 관련 참고: `chrome-remote-interface`는 `ui-tui/package.json`에 포함되어 있지 않습니다. 프로젝트를 오염시키고 싶지 않다면 임시 위치에 설치하세요.

```bash
mkdir -p /tmp/cdp-tools && cd /tmp/cdp-tools && npm i chrome-remote-interface
NODE_PATH=/tmp/cdp-tools/node_modules node /tmp/cdp-debug.js
```

## Hermes ui-tui 디버깅

TUI는 Ink + tsx로 빌드됩니다. 일반적인 시나리오는 두 가지입니다.

### 개발 중 단일 Ink 컴포넌트 디버깅

`ui-tui/package.json`에는 `npm run dev`(tsx --watch)가 있습니다. tsx를 직접 실행해 `--inspect-brk`를 추가하세요.

```bash
cd <hermes-agent-repo>/ui-tui
npm run build    # produce dist/ once so transpile isn't needed on first load
node --inspect-brk dist/entry.js
# In another terminal:
node inspect -p <node pid>
```

그런 다음 `debug>`에서 다음과 같이 합니다.

```
sb('dist/app.js', 220)     # or wherever the suspect render is
cont
```

일시 중지되면 `repl`로 들어가 `props`, 상태 ref, `useInput` 핸들러 값 등을 검사합니다.

### 실행 중인 `hermes --tui` 디버깅

TUI는 Python CLI에서 Node를 생성합니다. 가장 쉬운 방법은 다음과 같습니다.

```bash
# 1. Launch TUI
hermes --tui &
TUI_PID=$(pgrep -f 'ui-tui/dist/entry' | head -1)

# 2. Enable inspector on that Node PID
kill -SIGUSR1 "$TUI_PID"

# 3. Find the WS URL
curl -s http://127.0.0.1:9229/json/list | jq -r '.[0].webSocketDebuggerUrl'

# 4. Attach
node inspect ws://127.0.0.1:9229/<uuid>
```

TUI와 상호작용하면(창에 입력하는 등) 실행이 계속 진행됩니다. 디버거는 어느 `sb(...)`에서든 중단점에 도달하면 실행을 일시 중지할 수 있습니다.

### `_SlashWorker` / PTY 자식 프로세스 디버깅

이 프로세스들은 Node가 아니라 Python이므로 `python-debugpy` 스킬을 사용하세요. 이 스킬은 Node 부분(Ink UI, tui_gateway 클라이언트, `ui-tui/` 아래의 tsx-run 테스트)에만 사용합니다.

## 디버거에서 Vitest 테스트 실행

```bash
cd <hermes-agent-repo>/ui-tui
# Run a single test file paused on entry
node --inspect-brk ./node_modules/vitest/vitest.mjs run --no-file-parallelism src/app/foo.test.tsx
```

다른 터미널에서 `node inspect -p <pid>`를 실행한 다음 `sb('src/app/foo.tsx', 42)`, `cont`를 실행합니다.

풀 하나를 디버깅하는 일은 고통스러우므로, 워커가 하나만 존재하도록 `--no-file-parallelism`(vitest) 또는 `--runInBand`(jest)를 사용하세요.

## 힙 스냅샷 및 CPU 프로파일 (비대화형)

위의 CDP 드라이버에서 Debugger를 `HeapProfiler` / `Profiler`로 바꾸세요.

```javascript
// CPU profile for 5 seconds
await client.Profiler.enable();
await client.Profiler.start();
await new Promise(r => setTimeout(r, 5000));
const { profile } = await client.Profiler.stop();
require('fs').writeFileSync('/tmp/cpu.cpuprofile', JSON.stringify(profile));
// Open /tmp/cpu.cpuprofile in Chrome DevTools → Performance tab
```

```javascript
// Heap snapshot
await client.HeapProfiler.enable();
const chunks = [];
client.HeapProfiler.addHeapSnapshotChunk(({ chunk }) => chunks.push(chunk));
await client.HeapProfiler.takeHeapSnapshot({ reportProgress: false });
require('fs').writeFileSync('/tmp/heap.heapsnapshot', chunks.join(''));
```

## 일반적인 문제

1. **TS 소스의 잘못된 줄 번호.** 중단점은 `.ts`가 아니라 생성된 JS에서 적중됩니다. (a) 빌드된 `dist/*.js`에서 중단하거나, (b) 소스맵을 활성화하고(`node --enable-source-maps`) `sb('src/app.tsx', N)`을 사용하세요. 단, 소스맵을 따르는 CDP 클라이언트에서만 가능합니다. `node inspect` CLI에서는 불가능합니다.

2. **`--inspect`와 `--inspect-brk`의 차이.** `--inspect`는 inspector를 시작하지만 일시 중지하지 않습니다. 너무 늦게 연결하면 스크립트가 첫 중단점을 지나쳐 버립니다. 코드가 실행되기 전에 중단점을 설정해야 할 때는 `--inspect-brk`를 사용하세요.

3. **포트 충돌.** 기본값은 `9229`입니다. 여러 Node 프로세스를 검사하는 경우 `--inspect=0`(무작위 포트)을 전달하고 `/json/list`에서 실제 URL을 읽으세요.
   ```bash
   curl -s http://127.0.0.1:9229/json/list   # lists all inspectable targets on the host
   ```

4. **자식 프로세스.** 부모에 설정한 `--inspect`는 자식 프로세스를 검사하지 않습니다. 모든 자식에 전파하려면 `NODE_OPTIONS='--inspect-brk' node parent.js`를 사용하세요. 모든 자식에 고유한 포트가 필요하다는 점에 유의하세요(`NODE_OPTIONS='--inspect'`가 상속되면 Node가 포트를 자동으로 증가시킵니다).

5. **백그라운드 프로세스 종료.** 대상이 일시 중지된 상태에서 `node inspect`를 `Ctrl+C`로 종료하면 대상은 계속 일시 중지된 상태로 남습니다. 먼저 `cont`를 실행하거나 대상을 명시적으로 `kill`하세요.

6. **에이전트 터미널을 통해 `node inspect` 실행.** PTY를 지원하는 REPL입니다. Hermes에서는 `terminal(pty=true)`로 실행하거나 `background=true`와 `process(action='submit', data='...')`를 함께 사용해 실행하세요. PTY가 아닌 포그라운드 모드도 일회성 명령에는 작동하지만 대화형 단계 실행에는 적합하지 않습니다.

7. **보안.** `--inspect=0.0.0.0:9229`는 임의 코드 실행을 노출합니다. 격리된 네트워크가 아닌 경우 항상 `127.0.0.1`(기본값)에 바인딩하세요.

## 검증 체크리스트

디버그 세션을 설정한 후 다음을 확인하세요.

- [ ] `curl -s http://127.0.0.1:9229/json/list`가 예상한 대상만 정확히 반환하는지
- [ ] 첫 번째 중단점에 실제로 도달하는지 (도달하지 않는다면 `--inspect-brk`를 놓쳤거나 실행이 끝난 후 연결했을 가능성이 큽니다)
- [ ] 일시 중지 시 소스 목록에 올바른 파일이 표시되는지 (불일치는 소스맵 문제를 의미합니다. 문제 1을 참고하세요)
- [ ] `repl`에서 `exec process.pid`가 연결하려던 PID를 반환하는지

## 한 번에 실행하는 레시피

**"X줄에서 이 변수가 정의되지 않는 이유는?"**
```bash
node --inspect-brk script.js &
node inspect -p $!
# debug>
sb('script.js', X)
cont
# paused. Now:
repl
> myVariable
> Object.keys(this)
```

**"이 함수로 들어오는 호출 경로는?"**
```
debug> sb('suspectFn')
debug> cont
# paused on entry
debug> bt
```

**"이 비동기 체인은 여기서 멈춥니다 — 어디일까요?"**
```
# Start with --inspect (no -brk), let it run to the hang, then:
debug> pause
debug> bt
# Now you see the stuck frame
```
