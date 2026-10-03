---
title: "Node Inspect 调试器 — 调试 Node"
sidebar_label: "Node Inspect 调试器"
description: "调试 Node"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Node Inspect 调试器

通过 --inspect + Chrome DevTools Protocol CLI 调试 Node.js。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/node-inspect-debugger` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `debugging`, `nodejs`, `node-inspect`, `cdp`, `breakpoints`, `ui-tui` |
| 相关 skill | [`systematic-debugging`](/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`python-debugpy`](/user-guide/skills/bundled/software-development/software-development-python-debugpy) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时看到的指令内容。
:::

# Node.js Inspect 调试器

## 概述

当 `console.log` 不够用时，可以从终端以编程方式驱动 Node 内置的 V8 inspector。你可以使用真正的断点、单步执行（step in/over/out）、调用栈遍历、局部变量/闭包作用域转储，以及在暂停帧中执行任意表达式求值。

两种工具，选其一：

- **`node inspect`** — 内置，无需安装，CLI REPL（交互式命令行）。适合快速探查。
- **`ndb` / CDP via `chrome-remote-interface`** — 可从 Node/Python 脚本化调用；适合需要自动化设置大量断点、跨多次运行收集状态，或在 agent 循环中非交互式调试的场景。

**优先使用 `node inspect`。** 它始终可用，REPL 响应快。

## 使用时机

- Node 测试失败，需要查看中间状态
- ui-tui 崩溃或行为异常，需要在渲染前检查 React/Ink 状态
- tui_gateway 子进程（`_SlashWorker`、PTY bridge workers）行为异常
- 需要检查闭包中某个值，而不打补丁就无法用 `console.log` 获取
- 性能分析：附加到运行中的进程以采集 CPU profile 或堆快照

**不适用于：** 一分钟内用 `console.log` 就能解决的问题。断点调试开销较大，只在收益明显时使用。

## 快速参考：`node inspect` REPL

在第一行暂停启动：

```bash
node inspect path/to/script.js
# or with tsx
node --inspect-brk $(which tsx) path/to/script.ts
```

`debug>` 提示符接受以下命令：

| 命令 | 操作 |
|---|---|
| `c` 或 `cont` | 继续执行 |
| `n` 或 `next` | 单步跳过 |
| `s` 或 `step` | 单步进入 |
| `o` 或 `out` | 单步跳出 |
| `pause` | 暂停运行中的代码 |
| `sb('file.js', 42)` | 在 file.js 第 42 行设置断点 |
| `sb(42)` | 在当前文件第 42 行设置断点 |
| `sb('functionName')` | 在函数被调用时中断 |
| `cb('file.js', 42)` | 清除断点 |
| `breakpoints` | 列出所有断点 |
| `bt` | 回溯（调用栈） |
| `list(5)` | 显示当前位置前后各 5 行源码 |
| `watch('expr')` | 每次暂停时求值 expr |
| `watchers` | 显示监视表达式 |
| `repl` | 在当前作用域进入 REPL（Ctrl+C 退出 REPL） |
| `exec expr` | 单次求值表达式 |
| `restart` | 重启脚本 |
| `kill` | 终止脚本 |
| `.exit` | 退出调试器 |

**在 `repl` 子模式中：** 输入任意 JS 表达式，包括访问局部变量/闭包变量。`Ctrl+C` 返回 `debug>`。

## 附加到运行中的进程

当进程已在运行时（例如长期运行的开发服务器或 TUI gateway）：

```bash
# 1. Send SIGUSR1 to enable the inspector on an existing process
kill -SIGUSR1 <pid>
# Node prints: Debugger listening on ws://127.0.0.1:9229/<uuid>

# 2. Attach the debugger CLI
node inspect -p <pid>
# or by URL
node inspect ws://127.0.0.1:9229/<uuid>
```

从一开始就启动带 inspector 的进程：

```bash
node --inspect script.js           # listen on 127.0.0.1:9229, keep running
node --inspect-brk script.js       # listen AND pause on first line
node --inspect=0.0.0.0:9230 script.js   # custom host:port
```

通过 tsx 调试 TypeScript：

```bash
node --inspect-brk --import tsx script.ts
# or older tsx
node --inspect-brk -r tsx/cjs script.ts
```

## 程序化 CDP（从终端脚本化）

当需要自动化操作时——设置大量断点、捕获作用域状态、编写复现脚本——使用 `chrome-remote-interface`：

```bash
npm i -g chrome-remote-interface        # or project-local
# Start your target:
node --inspect-brk=9229 target.js &
```

驱动脚本（保存为 `/tmp/cdp-debug.js`）：

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

运行：

```bash
node /tmp/cdp-debug.js
```

Hermes 专项说明：`chrome-remote-interface` 不在 `ui-tui/package.json` 中。如果不想污染项目，可将其安装到临时目录：

```bash
mkdir -p /tmp/cdp-tools && cd /tmp/cdp-tools && npm i chrome-remote-interface
NODE_PATH=/tmp/cdp-tools/node_modules node /tmp/cdp-debug.js
```

## 调试 Hermes ui-tui

TUI 基于 Ink + tsx 构建。两种常见场景：

### 在开发模式下调试单个 Ink 组件

`ui-tui/package.json` 有 `npm run dev`（tsx --watch）。直接运行 tsx 并添加 `--inspect-brk`：

```bash
cd /home/bb/hermes-agent/ui-tui
npm run build    # produce dist/ once so transpile isn't needed on first load
node --inspect-brk dist/entry.js
# In another terminal:
node inspect -p <node pid>
```

然后在 `debug>` 中：

```
sb('dist/app.js', 220)     # or wherever the suspect render is
cont
```

暂停后，进入 `repl` → 检查 `props`、state 引用、`useInput` 处理器的值等。

### 调试运行中的 `hermes --tui`

TUI 由 Python CLI 启动 Node。最简路径：

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

在 TUI 窗口中交互（输入内容）会继续推进执行；调试器可以在任意 `sb(...)` 处暂停它。

### 调试 `_SlashWorker` / PTY 子进程

这些是 Python 进程，不是 Node——请使用 `python-debugpy` skill。只有 Node 部分（Ink UI、tui_gateway client、`ui-tui/` 下的 tsx-run 测试）使用本 skill。

## 在调试器下运行 Vitest 测试

```bash
cd /home/bb/hermes-agent/ui-tui
# Run a single test file paused on entry
node --inspect-brk ./node_modules/vitest/vitest.mjs run --no-file-parallelism src/app/foo.test.tsx
```

在另一个终端：`node inspect -p <pid>`，然后 `sb('src/app/foo.tsx', 42)`，`cont`。

使用 `--no-file-parallelism`（vitest）或 `--runInBand`（jest），确保只有一个 worker——调试 worker 池非常痛苦。

## 堆快照与 CPU Profile（非交互式）

在上面的 CDP 驱动脚本中，将 Debugger 替换为 `HeapProfiler` / `Profiler`：

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

## 常见陷阱

1. **TS 源码行号错误。** 断点命中的是编译后的 JS，而非 `.ts` 文件。解决方案：（a）在构建产物 `dist/*.js` 中设置断点，或（b）启用 sourcemap（`node --enable-source-maps`）并使用 `sb('src/app.tsx', N)` — 但仅限于支持 sourcemap 的 CDP 客户端。`node inspect` CLI 不支持。

2. **`--inspect` 与 `--inspect-brk` 的区别。** `--inspect` 启动 inspector 但不暂停；如果附加太晚，脚本会在你设置第一个断点之前就跑完。需要在任何代码运行前设置断点时，使用 `--inspect-brk`。

3. **端口冲突。** 默认端口为 `9229`。如果多个 Node 进程同时开启 inspector，传入 `--inspect=0`（随机端口）并从 `/json/list` 读取实际 URL：
   ```bash
   curl -s http://127.0.0.1:9229/json/list   # lists all inspectable targets on the host
   ```

4. **子进程。** 父进程上的 `--inspect` 不会 inspect 其子进程。使用 `NODE_OPTIONS='--inspect-brk' node parent.js` 将其传播到每个子进程；注意它们都需要唯一端口（继承 `NODE_OPTIONS='--inspect'` 时 Node 会自动递增端口号）。

5. **后台进程被杀死。** 在目标进程暂停时 `Ctrl+C` 退出 `node inspect`，目标进程会保持暂停状态。请先执行 `cont`，或显式 `kill` 目标进程。

6. **在 agent 终端中运行 `node inspect`。** 它是一个 PTY 友好的 REPL。在 Hermes 中，使用 `terminal(pty=true)` 或 `background=true` + `process(action='submit', data='...')` 启动它。非 PTY 前台模式适用于单次命令，但不适合交互式单步调试。

7. **安全性。** `--inspect=0.0.0.0:9229` 会暴露任意代码执行能力。除非处于隔离网络，否则始终绑定到 `127.0.0.1`（默认值）。

## 验证清单

建立调试会话后，验证以下内容：

- [ ] `curl -s http://127.0.0.1:9229/json/list` 返回的正是预期目标
- [ ] 第一个断点确实命中（若未命中，可能是漏加了 `--inspect-brk`，或附加时执行已完成）
- [ ] 暂停时的源码列表显示正确文件（不匹配 = sourcemap 问题，见陷阱 1）
- [ ] 在 `repl` 中执行 `exec process.pid` 返回你想附加的 PID

## 一键配方

**"为什么这个变量在第 X 行是 undefined？"**
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

**"进入这个函数的调用路径是什么？"**
```
debug> sb('suspectFn')
debug> cont
# paused on entry
debug> bt
```

**"这个 async 链挂住了——在哪里？"**
```
# Start with --inspect (no -brk), let it run to the hang, then:
debug> pause
debug> bt
# Now you see the stuck frame
```