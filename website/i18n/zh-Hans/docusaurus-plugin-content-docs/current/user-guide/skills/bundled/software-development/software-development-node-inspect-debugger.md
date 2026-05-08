---
title: "Node Inspect Debugger — 调试 Node"
sidebar_label: "Node Inspect Debugger"
description: "调试 Node"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Node Inspect Debugger

通过 --inspect + Chrome DevTools Protocol CLI 调试 Node.js。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/node-inspect-debugger` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `debugging`, `nodejs`, `node-inspect`, `cdp`, `breakpoints`, `ui-tui` |
| 相关技能 | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`python-debugpy`](/docs/user-guide/skills/bundled/software-development/software-development-python-debugpy), [`debugging-hermes-tui-commands`](/docs/user-guide/skills/bundled/software-development/software-development-debugging-hermes-tui-commands) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Node.js Inspect 调试器

## 概述

当 `console.log` 不够用时，从终端以编程方式驱动 Node 内置的 V8 inspector。你可以获得真正的断点、单步进入/跳过/跳出、调用栈遍历、局部/闭包作用域转储以及在暂停帧中求值任意表达式。

两个工具，任选其一：

- **`node inspect`** ——内置、零安装、CLI REPL。最适合快速探查。
- **`ndb` / 通过 `chrome-remote-interface` 的 CDP** ——可从 Node/Python 编写脚本；最适合需要自动化多个断点、跨运行收集状态或从代理循环非交互式调试时。

**优先使用 `node inspect`。** 它始终可用且 REPL 响应快。

## 使用时机

- Node 测试失败且需要查看中间状态
- ui-tui 崩溃或行为异常，需要检查 React/Ink 渲染前状态
- tui_gateway 子进程（`_SlashWorker`、PTY 桥接 worker）异常
- 需要检查闭包中 `console.log` 无法触及的值（无需打补丁）
- 性能：附加到正在运行的进程以捕获 CPU profile 或堆快照

**不适用于：** `console.log` 在一分钟内能解决的问题。断点驱动的调试更重；只有在回报确实存在时才使用。

## 快速参考：`node inspect` REPL

在第一行暂停启动：

```bash
node inspect path/to/script.js
# 或使用 tsx
node --inspect-brk $(which tsx) path/to/script.ts
```

`debug>` 提示符接受：

| 命令 | 操作 |
|---|---|
| `c` 或 `cont` | 继续 |
| `n` 或 `next` | 单步跳过 |
| `s` 或 `step` | 单步进入 |
| `o` 或 `out` | 单步跳出 |
| `pause` | 暂停正在运行的代码 |
| `sb('file.js', 42)` | 在 file.js 第 42 行设置断点 |
| `sb(42)` | 在当前文件第 42 行设置断点 |
| `sb('functionName')` | 函数被调用时中断 |
| `cb('file.js', 42)` | 清除断点 |
| `breakpoints` | 列出所有断点 |
| `bt` | 回溯（调用栈） |
| `list(5)` | 显示当前位置周围 5 行源代码 |
| `watch('expr')` | 每次暂停时求值 expr |
| `watchers` | 显示监视的表达式 |
| `repl` | 在当前作用域中进入 REPL（Ctrl+C 退出） |
| `exec expr` | 求值表达式一次 |
| `restart` | 重启脚本 |
| `kill` | 终止脚本 |
| `.exit` | 退出调试器 |

**在 `repl` 子模式中：** 输入任何 JS 表达式，包括访问局部/闭包变量。`Ctrl+C` 退回 `debug>`。

## 附加到正在运行的进程

当进程已在运行时（例如长期运行的 dev 服务器或 TUI 网关）：

```bash
# 1. 发送 SIGUSR1 以在现有进程上启用 inspector
kill -SIGUSR1 <pid>
# Node 打印：Debugger listening on ws://127.0.0.1:9229/<uuid>

# 2. 附加调试器 CLI
node inspect -p <pid>
# 或通过 URL
node inspect ws://127.0.0.1:9229/<uuid>
```

从开始就启用 inspector 启动进程：

```bash
node --inspect script.js           # 在 127.0.0.1:9229 监听，继续运行
node --inspect-brk script.js       # 监听并在第一行暂停
node --inspect=0.0.0.0:9230 script.js   # 自定义 host:port
```

通过 tsx 使用 TypeScript：

```bash
node --inspect-brk --import tsx script.ts
# 或旧版 tsx
node --inspect-brk -r tsx/cjs script.ts
```

## 编程式 CDP（从终端编写脚本）

当你想自动化——设置多个断点、捕获作用域状态、编写可重现脚本——时使用 `chrome-remote-interface`：

```bash
npm i -g chrome-remote-interface        # 或项目本地
# 启动目标：
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

    // 遍历作用域获取局部变量
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

    // 在暂停帧中求值表达式
    const { result } = await Debugger.evaluateOnCallFrame({
      callFrameId: top.callFrameId,
      expression: 'typeof state !== "undefined" ? JSON.stringify(state) : "n/a"',
    });
    console.log('state =', result.value ?? result.description);

    await Debugger.resume();
  });

  await Runtime.enable();
  await Debugger.enable();

  // 通过 URL 正则 + 行号设置断点
  await Debugger.setBreakpointByUrl({
    urlRegex: '.*app\\.tsx$',
    lineNumber: 119,       // 0 索引
    columnNumber: 0,
  });

  await Runtime.runIfWaitingForDebugger();
})();
```

运行它：

```bash
node /tmp/cdp-debug.js
```

Hermes 特定说明：`chrome-remote-interface` 不在 `ui-tui/package.json` 中。如果不想弄脏项目，安装到临时位置：

```bash
mkdir -p /tmp/cdp-tools && cd /tmp/cdp-tools && npm i chrome-remote-interface
NODE_PATH=/tmp/cdp-tools/node_modules node /tmp/cdp-debug.js
```

## 调试 Hermes ui-tui

TUI 由 Ink + tsx 构建。两个常见场景：

### 调试单个 Ink 组件

`ui-tui/package.json` 有 `npm run dev`（tsx --watch）。通过直接运行 tsx 添加 `--inspect-brk`：

```bash
cd /home/bb/hermes-agent/ui-tui
npm run build    # 生成 dist/ 一次，以便首次加载时无需转译
node --inspect-brk dist/entry.js
# 在另一个终端：
node inspect -p <node pid>
```

然后在 `debug>` 中：

```
sb('dist/app.js', 220)     # 或可疑渲染所在位置
cont
```

暂停时，`repl` → 检查 `props`、state refs、`useInput` handler 值等。

### 调试正在运行的 `hermes --tui`

TUI 从 Python CLI 生成 Node 进程。最简单的方式：

```bash
# 1. 启动 TUI
hermes --tui &
TUI_PID=$(pgrep -f 'ui-tui/dist/entry' | head -1)

# 2. 在该 Node PID 上启用 inspector
kill -SIGUSR1 "$TUI_PID"

# 3. 查找 WS URL
curl -s http://127.0.0.1:9229/json/list | jq -r '.[0].webSocketDebuggerUrl'

# 4. 附加
node inspect ws://127.0.0.1:9229/<uuid>
```

与 TUI 交互（在其窗口中输入）会继续推进执行；你的调试器可以随时在 `sb(...)` 断点处暂停。

### 调试 `_SlashWorker` / PTY 子进程

这些是 Python 进程，不是 Node ——对它们使用 `python-debugpy` 技能。只有 Node 部分（Ink UI、tui_gateway 客户端、`ui-tui/` 下的 tsx 运行测试）使用此技能。

## 在调试器下运行 Vitest 测试

```bash
cd /home/bb/hermes-agent/ui-tui
# 在入口处暂停运行单个测试文件
node --inspect-brk ./node_modules/vitest/vitest.mjs run --no-file-parallelism src/app/foo.test.tsx
```

在另一个终端：`node inspect -p <pid>`，然后 `sb('src/app/foo.tsx', 42)`，`cont`。

使用 `--no-file-parallelism`（vitest）或 `--runInBand`（jest），以便只有一个 worker 存在——调试池很痛苦。

## 堆快照和 CPU Profile（非交互式）

从上面的 CDP 驱动中，将 Debugger 替换为 `HeapProfiler` / `Profiler`：

```javascript
// CPU profile 5 秒
await client.Profiler.enable();
await client.Profiler.start();
await new Promise(r => setTimeout(r, 5000));
const { profile } = await client.Profiler.stop();
require('fs').writeFileSync('/tmp/cpu.cpuprofile', JSON.stringify(profile));
// 在 Chrome DevTools → Performance 标签中打开 /tmp/cpu.cpuprofile
```

```javascript
// 堆快照
await client.HeapProfiler.enable();
const chunks = [];
client.HeapProfiler.addHeapSnapshotChunk(({ chunk }) => chunks.push(chunk));
await client.HeapProfiler.takeHeapSnapshot({ reportProgress: false });
require('fs').writeFileSync('/tmp/heap.heapsnapshot', chunks.join(''));
```

## 常见陷阱

1. **TS 源中行号不对。** 断点命中编译后的 JS，而非 `.ts`。要么 (a) 在编译的 `dist/*.js` 中设置断点，要么 (b) 启用 sourcemaps（`node --enable-source-maps`）并使用 `sb('src/app.tsx', N)` ——但仅适用于遵循 sourcemaps 的 CDP 客户端。`node inspect` CLI 不支持。

2. **`--inspect` 与 `--inspect-brk`。** `--inspect` 启动 inspector 但不暂停；如果附加太晚，你的脚本会在你第一个断点之前跑过。当你需要在任何代码运行之前设置断点时使用 `--inspect-brk`。

3. **端口冲突。** 默认是 `9229`。如果多个 Node 进程在 inspect，传递 `--inspect=0`（随机端口）并从 `/json/list` 读取实际 URL：
   ```bash
   curl -s http://127.0.0.1:9229/json/list   # 列出主机上所有可检查的目标
   ```

4. **子进程。** 父进程上的 `--inspect` 不会 inspect 其子进程。使用 `NODE_OPTIONS='--inspect-brk' node parent.js` 传播到每个子进程；注意它们都需要唯一端口。

5. **后台终止。** 如果你在目标暂停时用 `Ctrl+C` 退出 `node inspect`，目标仍保持暂停。要么先 `cont`，要么显式 `kill` 目标。

6. **通过代理终端运行 `node inspect`。** 它是 PTY 友好的 REPL。在 Hermes 中，使用 `terminal(pty=true)` 或 `background=true` + `process(action='submit', data='...')` 启动。非 PTY 前台模式适用于一次性命令，但不适用于交互式单步调试。

7. **安全性。** `--inspect=0.0.0.0:9229` 暴露任意代码执行。除非网络隔离，始终绑定到 `127.0.0.1`（默认值）。

## 验证清单

设置调试会话后，验证：

- [ ] `curl -s http://127.0.0.1:9229/json/list` 返回你期望的目标
- [ ] 第一个断点确实命中（如果没有，你可能漏掉了 `--inspect-brk` 或在执行完成后才附加）
- [ ] 暂停时的源代码列表显示正确的文件（不匹配 = sourcemap 问题，见陷阱 1）
- [ ] `repl` 中的 `exec process.pid` 返回你想要附加的 PID

## 一次性配方

**"为什么变量 X 在第 N 行是 undefined？"**
```bash
node --inspect-brk script.js &
node inspect -p $!
# debug>
sb('script.js', X)
cont
# 暂停了。现在：
repl
> myVariable
> Object.keys(this)
```

**"这个函数的调用路径是什么？"**
```
debug> sb('suspectFn')
debug> cont
# 在入口处暂停
debug> bt
```

**"这个异步链挂起了——在哪？"**
```
# 用 --inspect 启动（不带 -brk），让它跑到挂起处，然后：
debug> pause
debug> bt
# 现在你可以看到卡住的帧
```
