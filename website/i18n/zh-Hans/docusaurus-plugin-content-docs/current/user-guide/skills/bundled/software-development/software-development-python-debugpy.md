---
title: "Python Debugpy — 调试 Python：pdb REPL + debugpy 远程（DAP）"
sidebar_label: "Python Debugpy"
description: "调试 Python：pdb REPL + debugpy 远程（DAP）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Python Debugpy

调试 Python：pdb REPL + debugpy 远程（DAP）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/software-development/python-debugpy` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos |
| 标签 | `debugging`, `python`, `pdb`, `debugpy`, `breakpoints`, `dap`, `post-mortem` |
| 相关 skill | [`systematic-debugging`](/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`node-inspect-debugger`](/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger), [`debugging-hermes-tui-commands`](/user-guide/skills/bundled/software-development/software-development-debugging-hermes-tui-commands) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Python 调试器（pdb + debugpy）

## 概述

三种工具，按场景选择：

| 工具 | 适用场景 |
|---|---|
| **`breakpoint()` + pdb** | 本地、交互式、最简单。在源码中添加 `breakpoint()`，正常运行，在该行进入 REPL。 |
| **`python -m pdb`** | 无需修改源码，直接在 pdb 下启动已有脚本。适合快速探查。 |
| **`debugpy`** | 远程 / 无头 / "附加到已运行进程"。使用 DAP 协议，可从终端脚本化操作，适用于长期运行的进程（gateway、daemon、PTY 子进程）。 |

**从 `breakpoint()` 开始。** 这是最低成本的可行方案。

## 使用时机

- 测试失败，但 traceback 无法说明某个值为何出错
- 需要逐步执行某个函数并观察集合的变化
- 长期运行的进程（hermes gateway、tui_gateway）出现异常且无法重启
- 事后分析（post-mortem）：异常在类生产代码中触发，需要检查崩溃现场的局部变量
- 子进程 / 子进程（Python `_SlashWorker`、PTY bridge worker）才是实际的 bug 所在

**不适用于：** `print()` / `logging.debug` 一分钟内能解决的问题，或 `pytest -vv --tb=long --showlocals` 已经能揭示的问题。

## pdb 快速参考

在任意 pdb 提示符（`(Pdb)`）下：

| 命令 | 操作 |
|---|---|
| `h` / `h cmd` | 帮助 |
| `n` | 下一行（步过） |
| `s` | 步入 |
| `r` | 从当前函数返回 |
| `c` | 继续执行 |
| `unt N` | 继续执行直到第 N 行 |
| `j N` | 跳转到第 N 行（仅限同一函数） |
| `l` / `ll` | 列出当前行附近的源码 / 完整函数 |
| `w` | 当前位置（调用栈跟踪） |
| `u` / `d` | 在调用栈中上移 / 下移 |
| `a` | 打印当前函数的参数 |
| `p expr` / `pp expr` | 打印 / 格式化打印表达式 |
| `display expr` | 每次停止时自动打印 expr |
| `b file:line` | 设置断点 |
| `b func` | 在函数入口处断点 |
| `b file:line, cond` | 条件断点 |
| `cl N` | 清除断点 N |
| `tbreak file:line` | 一次性断点 |
| `!stmt` | 执行任意 Python 语句（包括赋值） |
| `interact` | 在当前作用域中进入完整 Python REPL（Ctrl+D 退出） |
| `q` | 退出 |

`interact` 命令最为强大——可以导入任何模块、检查复杂对象，甚至调用会改变状态的方法。局部变量默认只读；在 `(Pdb)` 提示符下使用 `!x = 42` 进行修改。

## 方案 1：本地断点

最简单。编辑文件：

```python
def compute(x, y):
    result = some_helper(x)
    breakpoint()           # <-- 在此处进入 pdb
    return result + y
```

正常运行代码。你将在 `breakpoint()` 所在行停下，可完整访问局部变量。

**提交前务必删除 `breakpoint()`。** 使用 `git diff` 或 pre-commit grep：
```bash
rg -n 'breakpoint\(\)' --type py
```

## 方案 2：在 pdb 下启动脚本（无需修改源码）

```bash
python -m pdb path/to/script.py arg1 arg2
# 停在脚本第一行
(Pdb) b path/to/script.py:42
(Pdb) c
```

## 方案 3：调试 pytest 测试

hermes 测试运行器和 pytest 均支持以下方式：

```bash
# 在失败时（或任何异常抛出时）进入 pdb：
scripts/run_tests.sh tests/path/to/test_file.py::test_name --pdb

# 在测试开始时进入 pdb：
scripts/run_tests.sh tests/path/to/test_file.py::test_name --trace

# 在 traceback 中显示局部变量，不使用 pdb：
scripts/run_tests.sh tests/path/to/test_file.py --showlocals --tb=long
```

注意：`scripts/run_tests.sh` 默认使用 xdist（`-n 4`），pdb 在 xdist 下**无法正常工作**。请添加 `-p no:xdist` 或使用 `-n 0` 运行单个测试：

```bash
scripts/run_tests.sh tests/foo_test.py::test_bar --pdb -p no:xdist
# 或
source .venv/bin/activate
python -m pytest tests/foo_test.py::test_bar --pdb
```

这会绕过封闭环境保证——调试时可以接受，但推送前请在 wrapper 下重新运行以确认。

## 方案 4：对任意异常进行事后分析

```python
import pdb, sys
try:
    run_the_thing()
except Exception:
    pdb.post_mortem(sys.exc_info()[2])
```

或对整个脚本进行包装：

```bash
python -m pdb -c continue script.py
# 崩溃时，pdb 捕获异常并停在异常所在帧
```

或在 repl/jupyter 中设置全局 hook：

```python
import sys
def excepthook(etype, value, tb):
    import pdb; pdb.post_mortem(tb)
sys.excepthook = excepthook
```

## 方案 5：使用 debugpy 进行远程调试（附加到运行中的进程）

适用于长期运行的进程：Hermes gateway、tui_gateway、daemon，或已出现异常且无法干净重启的进程。

### 安装

```bash
repo_root="$(git rev-parse --show-toplevel)"
source "$repo_root/.venv/bin/activate"
pip install debugpy
```

### 模式 A：修改源码——进程在启动时等待调试器

在入口点顶部附近（或要调试的函数内部）添加：

```python
import debugpy
debugpy.listen(("127.0.0.1", 5678))
print("debugpy listening on 5678, waiting for client...", flush=True)
debugpy.wait_for_client()
debugpy.breakpoint()       # 可选：附加后立即暂停
```

启动进程；它将阻塞在 `wait_for_client()`。

### 模式 B：无需修改源码——使用 `-m debugpy` 启动

```bash
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client your_script.py arg1
```

模块入口的等效写法：

```bash
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client -m your.module
```

### 模式 C：附加到已运行的进程

需要 PID 以及在目标环境中预装 debugpy：

```bash
python -m debugpy --listen 127.0.0.1:5678 --pid <pid>
# debugpy 注入到目标进程中，然后按以下方式连接客户端。
```

某些内核 / 安全配置会阻止基于 ptrace 的注入（`/proc/sys/kernel/yama/ptrace_scope`）。修复方法：
```bash
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
```

### 从终端连接客户端

最简便的终端侧 DAP 客户端是 VS Code CLI 或一个小脚本。在 Hermes 内部有两个实用选项：

**选项 1：`debugpy` 自带 CLI REPL** — 并非官方功能，而是一个小型 DAP 客户端脚本：

```python
# /tmp/dap_client.py
import socket, json, itertools, time, sys

HOST, PORT = "127.0.0.1", 5678
s = socket.create_connection((HOST, PORT))
seq = itertools.count(1)

def send(msg):
    msg["seq"] = next(seq)
    body = json.dumps(msg).encode()
    s.sendall(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)

def recv():
    header = b""
    while b"\r\n\r\n" not in header:
        header += s.recv(1)
    length = int(header.decode().split("Content-Length:")[1].split("\r\n")[0].strip())
    body = b""
    while len(body) < length:
        body += s.recv(length - len(body))
    return json.loads(body)

send({"type": "request", "command": "initialize", "arguments": {"adapterID": "python"}})
print(recv())
send({"type": "request", "command": "attach", "arguments": {}})
print(recv())
send({"type": "request", "command": "setBreakpoints",
      "arguments": {"source": {"path": sys.argv[1]},
                    "breakpoints": [{"line": int(sys.argv[2])}]}})
print(recv())
send({"type": "request", "command": "configurationDone"})
# ... 循环读取事件并发送 continue/stepIn 等命令
```

用于一次性自动化尚可，但作为交互式 UX 体验较差。

**选项 2：从 VS Code / Cursor / Zed 附加** — 如果用户已打开其中一个，可添加 `launch.json`：

```json
{
  "name": "Attach to Hermes",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "127.0.0.1", "port": 5678 },
  "justMyCode": false,
  "pathMappings": [
    { "localRoot": "${workspaceFolder}", "remoteRoot": "<remote-hermes-agent-root>" }
  ]
}
```

**选项 3：放弃 DAP，使用 `remote-pdb`** — 通常这才是终端 agent 真正需要的：

```bash
pip install remote-pdb
```

在代码中：
```python
from remote_pdb import set_trace
set_trace(host="127.0.0.1", port=4444)   # 阻塞直到连接
```

然后在终端中：
```bash
nc 127.0.0.1 4444
# 获得一个 (Pdb) 提示符，与本地调试完全一致。
```

当 `debugpy` 的 DAP 协议过于繁重时，`remote-pdb` 是最适合 agent 的选择。仅在确实需要 IDE 集成时才使用 `debugpy`。

## 调试 Hermes 特定进程

### 测试
参见方案 3。始终添加 `-p no:xdist` 或在不使用 xdist 的情况下运行单个测试。

### `run_agent.py` / CLI — 一次性运行
最简单：在可疑行附近添加 `breakpoint()`，然后正常运行 `hermes`。控制权将在暂停点返回到你的终端。

### `tui_gateway` 子进程（由 `hermes --tui` 启动）
gateway 作为 Node TUI 的子进程运行。可选方案：

**A. 修改 gateway 源码：**
```python
# tui_gateway/server.py，在 serve() 顶部附近
import debugpy
debugpy.listen(("127.0.0.1", 5678))
debugpy.wait_for_client()
```
启动 `hermes --tui`。TUI 将显示为冻结状态（其后端正在等待）。附加客户端后，执行在你 `continue` 时恢复。

**B. 在特定处理器中使用 `remote-pdb`：**
```python
from remote_pdb import set_trace
set_trace(host="127.0.0.1", port=4444)   # 在你想捕获的 RPC 处理器中
```
从 TUI 触发对应的 slash 命令，然后在另一个终端中执行 `nc 127.0.0.1 4444`。

### `_SlashWorker` 子进程
相同模式——在 worker 的 `exec` 路径中使用 `remote-pdb` 的 `set_trace()`。该 worker 在多次 slash 命令间持续存在，因此第一次触发会阻塞直到你连接；后续 slash 命令正常通过，除非你重新设置断点。

### Gateway（`gateway/run.py`）
长期运行。在处理器中使用 `remote-pdb`，或者如果你本来就要重启 gateway，则使用带 `--wait-for-client` 的 `debugpy`。

## 常见陷阱

1. **pdb 在 pytest-xdist 下静默失效。** 你不会看到提示符，测试只会挂起。始终使用 `-p no:xdist` 或 `-n 0`。

2. **`breakpoint()` 在 CI / 非 TTY 环境中会挂起进程。** 本地使用没问题；永远不要提交它。添加 pre-commit grep 作为安全网。

3. **`PYTHONBREAKPOINT=0`** 会禁用所有 `breakpoint()` 调用。如果断点未触发，请检查环境变量：
   ```bash
   echo $PYTHONBREAKPOINT
   ```

4. **`debugpy.listen` 仅在同时调用 `wait_for_client()` 时才会阻塞。** 不调用的话，执行会继续，你的第一个断点可能在客户端附加之前就已触发。

5. **在加固内核上附加到 PID 会失败。** `ptrace_scope=1`（Ubuntu 默认值）仅允许对同用户的子进程进行 ptrace。解决方法：`echo 0 > /proc/sys/kernel/yama/ptrace_scope`（需要 root 权限），或从一开始就在 `debugpy` 下启动。

6. **线程。** `pdb` 只调试当前线程。对于多线程代码，使用 `debugpy`（支持线程感知的 DAP）或为每个线程设置 `threading.settrace()`。

7. **asyncio。** `pdb` 可在协程中工作，但在 pdb 内部使用 `await` 需要 Python 3.13+ 或在旧版本的 `interact` 模式下使用 `await`。对于 3.11/3.12，使用 `asyncio.run_coroutine_threadsafe` 技巧，或通过 `asyncio.ensure_future` 配合 `!stmt` 方式进行 await。

8. **`scripts/run_tests.sh` 会剥离凭据并设置 `HOME=<tmpdir>`。** 如果你的 bug 依赖用户配置或真实 API 密钥，在 wrapper 下将无法复现。先用原始 `pytest` 复现，再在 wrapper 下确认。

9. **fork / 多进程。** pdb 不会跟随 fork。每个子进程需要自己的 `breakpoint()` 或 `set_trace()`。对于 Hermes 子 agent，每次只调试一个进程。

## 验证清单

- [ ] `pip install debugpy` 后确认：`python -c "import debugpy; print(debugpy.__version__)"`
- [ ] 对于远程调试，确认端口确实在监听：`ss -tlnp | grep 5678`
- [ ] 第一个断点确实触发（如果没有，可能是 `PYTHONBREAKPOINT=0`、在 xdist 下运行，或执行在附加前已结束）
- [ ] `where` / `w` 显示预期的调用栈
- [ ] 调试后清理：已提交代码中无残留的 `breakpoint()` / `set_trace()` / `debugpy.listen`
  ```bash
  rg -n 'breakpoint\(\)|set_trace\(|debugpy\.listen' --type py
  ```

## 一次性速查方案

**"为什么这个 dict 缺少某个键？"**
```python
# 在 KeyError 发生处上方添加
breakpoint()
# 然后在 pdb 中：
(Pdb) pp d
(Pdb) pp list(d.keys())
(Pdb) w                # 我们是怎么到这里的
```

**"这个测试单独运行通过，但在测试套件中失败。"**
```bash
scripts/run_tests.sh tests/the_test.py --pdb -p no:xdist
# 但如果只有与其他测试一起运行才失败：
source .venv/bin/activate
python -m pytest tests/ -x --pdb -p no:xdist
# 现在它会在状态积累后的确切失败测试处触发 pdb。
```

**"我的异步处理器发生死锁。"**
```python
# 在处理器入口处添加
import remote_pdb; remote_pdb.set_trace(host="127.0.0.1", port=4444)
```
触发处理器。执行 `nc 127.0.0.1 4444`，然后用 `w` 查看挂起的帧，用 `!import asyncio; asyncio.all_tasks()` 查看其他待处理任务。

**"对 Ink 子进程 / subprocess 中的崩溃进行事后分析。"**
```bash
PYTHONFAULTHANDLER=1 python -m pdb -c continue path/to/entrypoint.py
# 崩溃时，pdb 停在异常所在帧，可访问完整局部变量
```