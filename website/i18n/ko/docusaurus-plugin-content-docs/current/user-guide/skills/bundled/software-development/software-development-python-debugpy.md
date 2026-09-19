---
title: "Python Debugpy — Python 디버깅: pdb REPL + debugpy 원격 (DAP)"
sidebar_label: "Python Debugpy"
description: "Python 디버깅: pdb REPL + debugpy 원격 (DAP)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Python Debugpy

Python 디버깅: pdb REPL + debugpy 원격 (DAP).

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 번들됨 (기본 설치) |
| 경로 | `skills/software-development/python-debugpy` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `debugging`, `python`, `pdb`, `debugpy`, `breakpoints`, `dap`, `post-mortem` |
| 관련 스킬 | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`node-inspect-debugger`](/docs/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보게 되는 지침입니다.
:::

# Python 디버거 (pdb + debugpy)

## 개요

상황에 따라 선택하는 세 가지 도구:

| 도구 | 사용 시점 |
|---|---|
| **`breakpoint()` + pdb** | 로컬 대화형 디버깅, 가장 간단한 방법. 소스에 `breakpoint()`를 추가하고 정상적으로 실행하면 해당 줄에서 REPL을 얻습니다. |
| **`python -m pdb`** | 소스 수정 없이 기존 스크립트를 pdb로 실행합니다. 빠르게 살펴볼 때 유용합니다. |
| **`debugpy`** | 원격 / 헤드리스 / "이미 실행 중인 프로세스에 연결"할 때 사용합니다. DAP로 통신하고, 터미널에서 스크립트로 제어할 수 있으며, 장시간 실행되는 프로세스(gateway, daemon, PTY children)에서 작동합니다. |

**`breakpoint()`부터 시작하세요.** 작동하는 방법 중 가장 비용이 적습니다.

## 사용 시점

- 테스트가 실패했지만 트레이스백만으로는 값이 잘못된 이유가 드러나지 않을 때
- 함수를 단계별로 실행하며 컬렉션의 변화를 확인해야 할 때
- 장시간 실행되는 프로세스(hermes gateway, tui_gateway)가 오작동하고 재시작할 수 없을 때
- 사후 분석: 프로덕션과 유사한 코드에서 예외가 발생했고 충돌 지점의 로컬 변수를 확인하고 싶을 때
- 실제 버그 지점이 서브프로세스/자식 프로세스(Python `_SlashWorker`, PTY bridge worker)일 때

**다음에는 사용하지 마세요:** `print()` / `logging.debug`로 1분 안에 해결할 수 있는 경우나, `pytest -vv --tb=long --showlocals`가 이미 보여 주는 경우.

## pdb 빠른 참조

어떤 pdb 프롬프트(`(Pdb)`)에서든:

| 명령 | 동작 |
|---|---|
| `h` / `h cmd` | 도움말 |
| `n` | 다음 줄(스텝 오버) |
| `s` | 안으로 들어가기(스텝 인투) |
| `r` | 현재 함수에서 반환 |
| `c` | 계속 실행 |
| `unt N` | N번 줄까지 계속 실행 |
| `j N` | N번 줄로 이동(같은 함수 내에서만) |
| `l` / `ll` | 현재 줄 주변의 소스 / 전체 함수 나열 |
| `w` | 현재 위치(스택 트레이스) |
| `u` / `d` | 스택에서 위로 / 아래로 이동 |
| `a` | 현재 함수의 인자 출력 |
| `p expr` / `pp expr` | 표현식 출력 / 보기 좋게 출력 |
| `display expr` | 중단할 때마다 자동으로 출력 |
| `b file:line` | 중단점 설정 |
| `b func` | 함수 진입 시 중단 |
| `b file:line, cond` | 조건부 중단점 |
| `cl N` | 중단점 N 삭제 |
| `tbreak file:line` | 일회성 중단점 |
| `!stmt` | 임의의 Python 실행(할당 포함) |
| `interact` | 현재 스코프에서 전체 Python REPL로 진입(Ctrl+D로 종료) |
| `q` | 종료 |

`interact` 명령이 가장 강력합니다. 무엇이든 import하고, 복잡한 객체를 검사하며, 상태를 변경하는 메서드까지 호출할 수 있습니다. 로컬 변수는 기본적으로 읽기 전용입니다. 변경하려면 `(Pdb)` 프롬프트에서 `!x = 42`를 사용하세요.

## 레시피 1: 로컬 중단점

가장 쉽습니다. 파일을 수정하세요.

```python
def compute(x, y):
    result = some_helper(x)
    breakpoint()           # <-- drops into pdb here
    return result + y
```

코드를 정상적으로 실행하세요. `breakpoint()` 줄에서 로컬 변수에 완전히 접근할 수 있는 상태로 멈춥니다.

**커밋하기 전에 `breakpoint()`를 제거하는 것을 잊지 마세요.** `git diff` 또는 pre-commit grep을 사용하세요.
```bash
rg -n 'breakpoint\(\)' --type py
```

## 레시피 2: pdb로 스크립트 실행(소스 수정 없음)

```bash
python -m pdb path/to/script.py arg1 arg2
# Lands at first line of script
(Pdb) b path/to/script.py:42
(Pdb) c
```

## 레시피 3: pytest 테스트 디버깅

hermes 테스트 러너와 pytest 모두 이를 지원합니다.

```bash
# Drop to pdb on failure (or on any raised exception):
scripts/run_tests.sh tests/path/to/test_file.py::test_name --pdb

# Drop to pdb at the START of the test:
scripts/run_tests.sh tests/path/to/test_file.py::test_name --trace

# Show locals in tracebacks without pdb:
scripts/run_tests.sh tests/path/to/test_file.py --showlocals --tb=long
```

참고: `scripts/run_tests.sh`는 `run_tests_parallel.py`를 통해 각 테스트 파일을 캡처된 서브프로세스에서 실행합니다(xdist는 사용하지 않음). 따라서 래퍼 아래에서는 대화형 pdb가 작동하지 않습니다. `--pdb`를 사용하려면 pytest를 직접 실행하세요.

```bash
source .venv/bin/activate
python -m pytest tests/foo_test.py::test_bar --pdb
```

이 방식은 hermetic-env 보장을 우회합니다. 디버깅에는 괜찮지만, push하기 전에 확인을 위해 래퍼 아래에서 다시 실행하세요.

## 레시피 4: 모든 예외의 사후 분석

```python
import pdb, sys
try:
    run_the_thing()
except Exception:
    pdb.post_mortem(sys.exc_info()[2])
```

또는 전체 스크립트를 감싸세요.

```bash
python -m pdb -c continue script.py
# When it crashes, pdb catches it and you're in the frame of the exception
```

또는 repl/jupyter에서 전역 훅을 설정하세요.

```python
import sys
def excepthook(etype, value, tb):
    import pdb; pdb.post_mortem(tb)
sys.excepthook = excepthook
```

## 레시피 5: debugpy를 사용한 원격 디버깅(실행 중인 프로세스에 연결)

장시간 실행되는 프로세스(Hermes gateway, tui_gateway, 데몬, 이미 오작동 중이며 깔끔하게 재시작할 수 없는 프로세스)에 사용합니다.

### 설정

```bash
source <hermes-agent-repo>/.venv/bin/activate
pip install debugpy
```

### 패턴 A: 소스 수정 — 프로세스가 시작 시 디버거를 기다림

진입점 상단 근처(또는 디버깅하려는 함수 내부)에 추가하세요.

```python
import debugpy
debugpy.listen(("127.0.0.1", 5678))
print("debugpy listening on 5678, waiting for client...", flush=True)
debugpy.wait_for_client()
debugpy.breakpoint()       # optional: pause immediately once attached
```

프로세스를 시작하면 `wait_for_client()`에서 차단됩니다.

### 패턴 B: 소스 수정 없음 — `-m debugpy`로 실행

```bash
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client your_script.py arg1
```

모듈 진입점의 경우도 같습니다.

```bash
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client -m your.module
```

### 패턴 C: 이미 실행 중인 프로세스에 연결

PID와 대상 환경에 미리 설치된 debugpy가 필요합니다.

```bash
python -m debugpy --listen 127.0.0.1:5678 --pid <pid>
# debugpy injects itself into the process. Then attach a client as below.
```

일부 커널/보안 설정에서는 ptrace 기반 주입을 차단합니다(`/proc/sys/kernel/yama/ptrace_scope`). 다음으로 해결하세요.
```bash
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
```

### 터미널에서 클라이언트 연결

가장 쉬운 터미널 측 DAP 클라이언트는 VS Code CLI 또는 작은 스크립트입니다. Hermes 내부에서는 두 가지 실용적인 방법이 있습니다.

**옵션 1: `debugpy` 자체 CLI REPL** — 공식 기능은 아니지만 작은 DAP 클라이언트 스크립트입니다.

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
# ... loop reading events and sending continue/stepIn/etc.
```

일회성 자동화에는 괜찮지만 대화형 UX로는 불편합니다.

**옵션 2: VS Code / Cursor / Zed에서 연결** — 열려 있는 편집기가 있다면 `launch.json`을 추가할 수 있습니다.

```json
{
  "name": "Attach to Hermes",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "127.0.0.1", "port": 5678 },
  "justMyCode": false,
  "pathMappings": [
    { "localRoot": "${workspaceFolder}", "remoteRoot": "<hermes-agent-repo>" }
  ]
}
```

**옵션 3: DAP를 버리고 `remote-pdb` 사용** — 보통 터미널 에이전트에서 실제로 원하는 방식입니다.

```bash
pip install remote-pdb
```

코드에서:
```python
from remote_pdb import set_trace
set_trace(host="127.0.0.1", port=4444)   # blocks until connection
```

그런 다음 터미널에서:
```bash
nc 127.0.0.1 4444
# You get a (Pdb) prompt exactly as if debugging locally.
```

`debugpy`의 DAP 프로토콜이 과한 경우 `remote-pdb`가 에이전트 친화적인 가장 깔끔한 선택입니다. IDE 통합이 실제로 필요할 때만 `debugpy`를 사용하세요.

## Hermes 전용 프로세스 디버깅

### 테스트
레시피 3을 참고하세요. 래퍼가 서브프로세스 출력을 캡처하므로 대화형 pdb를 사용하려면 pytest를 직접 실행하세요.

### `run_agent.py` / CLI — 일회성
가장 쉽습니다. 의심되는 줄 근처에 `breakpoint()`를 추가한 다음 `hermes`를 정상적으로 실행하세요. 일시 중지 지점에서 터미널로 제어가 돌아옵니다.

### `tui_gateway` 서브프로세스(`hermes --tui`가 생성)
gateway는 Node TUI의 자식으로 실행됩니다. 다음과 같은 옵션이 있습니다.

**A. gateway 소스 수정:**
```python
# tui_gateway/server.py near the top of serve()
import debugpy
debugpy.listen(("127.0.0.1", 5678))
debugpy.wait_for_client()
```
`hermes --tui`를 시작하세요. TUI가 멈춘 것처럼 보입니다(백엔드가 대기 중입니다). 클라이언트를 연결하면 `continue`할 때 실행이 재개됩니다.

**B. 특정 핸들러에서 `remote-pdb` 사용:**
```python
from remote_pdb import set_trace
set_trace(host="127.0.0.1", port=4444)   # in the RPC handler you want to trap
```
TUI에서 일치하는 슬래시 명령을 트리거한 다음, 다른 터미널에서 `nc 127.0.0.1 4444`를 실행하세요.

### `_SlashWorker` 서브프로세스
같은 패턴을 사용하세요. worker의 `exec` 경로 안에서 `set_trace()`와 함께 `remote-pdb`를 사용합니다. worker는 슬래시 명령 간에 계속 유지되므로, 처음 트리거하면 연결할 때까지 차단됩니다. 다시 설정하지 않는 한 이후 슬래시 명령은 정상적으로 통과합니다.

### Gateway(`gateway/run.py`)
장시간 실행됩니다. 핸들러에서 `remote-pdb`를 사용하거나, 어차피 gateway를 재시작할 예정이라면 `--wait-for-client`와 함께 `debugpy`를 사용하세요.

## 일반적인 함정

1. **병렬 실행기/출력을 캡처하는 러너에서 pdb가 조용히 아무것도 하지 않을 수 있습니다.** 프롬프트가 보이지 않고 테스트가 그냥 멈춥니다(이는 pytest-xdist와 `scripts/run_tests.sh`의 파일별 캡처 서브프로세스 모두에 해당합니다). 대화형 디버깅은 단일 파일에 대해 pytest를 직접 실행하세요.

2. **CI / 비-TTY 컨텍스트에서 `breakpoint()`를 사용하면 프로세스가 멈춥니다.** 로컬에서는 안전하지만 절대 커밋하지 마세요. 안전망으로 pre-commit grep을 추가하세요.

3. **`PYTHONBREAKPOINT=0`**은 모든 `breakpoint()` 호출을 비활성화합니다. 중단점에 도달하지 않는다면 환경을 확인하세요.
   ```bash
   echo $PYTHONBREAKPOINT
   ```

4. **`debugpy.listen`은 `wait_for_client()`도 호출할 때만 차단합니다.** 그렇지 않으면 실행이 계속되어 클라이언트가 연결하기 전에 첫 중단점이 실행될 수 있습니다.

5. **강화된 커널에서는 PID 연결이 실패합니다.** `ptrace_scope=1`(Ubuntu 기본값)은 동일 사용자가 자식 프로세스에 수행하는 ptrace만 허용합니다. 해결 방법: `echo 0 > /proc/sys/kernel/yama/ptrace_scope`(root 필요) 또는 처음부터 `debugpy` 아래에서 실행하세요.

6. **스레드.** `pdb`는 현재 스레드만 디버깅합니다. 멀티스레드 코드에는 `debugpy`(스레드를 인식하는 DAP)를 사용하거나 스레드마다 `threading.settrace()`를 설정하세요.

7. **asyncio.** coroutine에서 `pdb`가 작동하지만, pdb 안에서 `await`를 사용하려면 Python 3.13+가 필요하거나 이전 버전에서는 `interact` 모드에서 `await`해야 합니다. 3.11/3.12에서는 `asyncio.run_coroutine_threadsafe` 트릭 또는 `asyncio.ensure_future`를 사용하는 `!stmt` 기반 await를 사용하세요.

8. **`scripts/run_tests.sh`는 자격 증명을 제거하고 `HOME=<tmpdir>`를 설정합니다.** 버그가 사용자 설정이나 실제 API 키에 의존한다면 래퍼 아래에서는 재현되지 않습니다. 먼저 일반 `pytest`로 디버깅하여 재현한 다음 래퍼 아래에서 다시 확인하세요.

9. **Forking / multiprocessing.** pdb는 fork를 따라가지 않습니다. 각 자식에는 자체 `breakpoint()` 또는 `set_trace()`가 필요합니다. Hermes 서브에이전트에서는 한 번에 하나의 프로세스만 디버깅하세요.

## 검증 체크리스트

- [ ] `pip install debugpy` 후 확인: `python -c "import debugpy; print(debugpy.__version__)"`
- [ ] 원격 디버깅의 경우 포트가 실제로 수신 대기 중인지 확인: `ss -tlnp | grep 5678`
- [ ] 첫 중단점에 실제로 도달하는지 확인(도달하지 않는다면 `PYTHONBREAKPOINT=0`이거나, 병렬/캡처 러너 아래에서 실행 중이거나, 연결 전에 실행이 끝났을 가능성이 큽니다)
- [ ] `where` / `w`에 예상한 호출 스택이 표시되는지 확인
- [ ] 디버깅 후 정리: 커밋된 코드에 남은 `breakpoint()` / `set_trace()`가 없는지 확인
  ```bash
  rg -n 'breakpoint\(\)|set_trace\(|debugpy\.listen' --type py
  ```

## 일회성 레시피

**"이 dict에 키가 왜 없지?"**
```python
# add above the KeyError site
breakpoint()
# then in pdb:
(Pdb) pp d
(Pdb) pp list(d.keys())
(Pdb) w                # how did we get here
```

**"이 테스트는 단독으로는 통과하지만 전체 스위트에서는 실패해."**
```bash
scripts/run_tests.sh tests/the_test.py   # confirm it fails under the isolated runner first
# For interactive debugging, or if it only fails WITH other tests:
source .venv/bin/activate
python -m pytest tests/ -x --pdb
# Now it pdb-traps at the exact failing test after state accumulated.
```

**"내 async 핸들러가 데드록돼."**
```python
# Add at handler entry
import remote_pdb; remote_pdb.set_trace(host="127.0.0.1", port=4444)
```
핸들러를 트리거하세요. `nc 127.0.0.1 4444`를 실행한 다음, `w`로 중단된 프레임을 확인하고 `!import asyncio; asyncio.all_tasks()`로 대기 중인 다른 작업을 확인하세요.

**"Ink 자식 프로세스 / 서브프로세스의 충돌을 사후 분석하고 싶어."**
```bash
PYTHONFAULTHANDLER=1 python -m pdb -c continue path/to/entrypoint.py
# On crash, pdb lands at the frame of the exception with full locals
```
