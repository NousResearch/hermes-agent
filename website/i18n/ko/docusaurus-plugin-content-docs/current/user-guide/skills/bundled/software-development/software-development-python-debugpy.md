---
title: "Python Debugpy — Python 디버깅: pdb REPL + debugpy 원격 (DAP)"
sidebar_label: "Python Debugpy"
description: "Python 디버깅: pdb REPL + debugpy 원격 (DAP)"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Python Debugpy

Python 디버깅: pdb REPL + debugpy 원격 (DAP).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 내장 (기본으로 설치됨) |
| 경로 | `skills/software-development/python-debugpy` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `debugging`, `python`, `pdb`, `debugpy`, `breakpoints`, `dap`, `post-mortem` |
| 관련 스킬 | [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`node-inspect-debugger`](/docs/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger), [`debugging-hermes-tui-commands`](/docs/user-guide/skills/bundled/software-development/software-development-debugging-hermes-tui-commands) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Python Debugger (pdb + debugpy)

## 개요

상황에 따라 선택할 수 있는 세 가지 도구:

| 도구 | 사용 시기 |
|---|---|
| **`breakpoint()` + pdb** | 로컬, 대화형, 가장 간단함. 소스에 `breakpoint()`를 추가하고 정상적으로 실행하여 해당 줄에서 REPL을 엽니다. |
| **`python -m pdb`** | 소스 편집 없이 기존 스크립트를 pdb로 실행합니다. 빠르게 살펴볼 때 유용합니다. |
| **`debugpy`** | 원격 / 헤드리스 / "이미 실행 중인 프로세스에 연결". DAP를 통해 터미널에서 스크립팅 가능하며, 오래 실행되는 프로세스(gateway, 데몬, PTY 하위 프로세스)에 적합합니다. |

**가장 먼저 `breakpoint()`로 시작하세요.** 작동하는 가장 비용이 적은 방법입니다.

## 사용 시기

- 테스트가 실패했는데 트레이스백만으로 값이 왜 잘못되었는지 알 수 없을 때
- 함수를 단계별로 실행하며 컬렉션의 변경을 지켜봐야 할 때
- 오래 실행되는 프로세스(hermes gateway, tui_gateway)가 오동작하지만 깨끗하게 다시 시작할 수 없을 때
- 사후(Post-mortem): 프로덕션 환경의 코드에서 예외가 발생했고, 크래시 위치에서 로컬 변수들을 검사하고 싶을 때
- 하위 프로세스 / 자식 (Python `_SlashWorker`, PTY 브리지 워커)이 실제 버그 발생 위치일 때

**사용하지 말아야 할 때:** `print()` / `logging.debug`로 1분 안에 해결할 수 있는 문제나, `pytest -vv --tb=long --showlocals`가 이미 모든 것을 보여주는 경우에는 사용하지 마세요.

## pdb 빠른 참조

어떤 pdb 프롬프트(`(Pdb)`)에서든:

| 명령어 | 동작 |
|---|---|
| `h` / `h cmd` | 도움말 |
| `n` | 다음 줄 (step over) |
| `s` | 안으로 들어가기 (step into) |
| `r` | 현재 함수에서 반환 (return) |
| `c` | 계속 실행 (continue) |
| `unt N` | N번째 줄까지 계속 실행 |
| `j N` | N번째 줄로 점프 (같은 함수 내에서만) |
| `l` / `ll` | 현재 줄 주변 소스 출력 / 전체 함수 출력 |
| `w` | 현재 위치 (스택 트레이스) |
| `u` / `d` | 스택에서 위 / 아래로 이동 |
| `a` | 현재 함수의 인자 출력 |
| `p expr` / `pp expr` | 표현식 출력 / 보기 좋게 출력 (pretty-print) |
| `display expr` | 멈출 때마다 표현식 자동 출력 |
| `b file:line` | 중단점(breakpoint) 설정 |
| `b func` | 함수 진입점에 중단점 설정 |
| `b file:line, cond` | 조건부 중단점 |
| `cl N` | N번 중단점 지우기 |
| `tbreak file:line` | 일회성 중단점 |
| `!stmt` | 임의의 Python 실행 (할당 포함) |
| `interact` | 현재 스코프에서 전체 Python REPL로 진입 (Ctrl+D로 종료) |
| `q` | 종료 |

`interact` 명령어가 가장 강력합니다 — 무엇이든 임포트할 수 있고, 복잡한 객체를 검사하며, 상태를 변경하는 메서드도 호출할 수 있습니다. 로컬 변수들은 기본적으로 읽기 전용입니다. 변경하려면 `(Pdb)` 프롬프트에서 `!x = 42`와 같이 사용하세요.

## 레시피 1: 로컬 중단점

가장 쉽습니다. 파일을 편집하세요:

```python
def compute(x, y):
    result = some_helper(x)
    breakpoint()           # <-- 여기서 pdb로 진입합니다
    return result + y
```

코드를 정상적으로 실행하세요. `breakpoint()` 줄에서 멈추며 로컬 변수에 완벽하게 접근할 수 있습니다.

**커밋하기 전에 `breakpoint()`를 지우는 것을 잊지 마세요.** `git diff`나 pre-commit grep을 사용하세요:
```bash
rg -n 'breakpoint\(\)' --type py
```

## 레시피 2: pdb로 스크립트 실행 (소스 편집 없음)

```bash
python -m pdb path/to/script.py arg1 arg2
# 스크립트의 첫 번째 줄에서 멈춥니다
(Pdb) b path/to/script.py:42
(Pdb) c
```

## 레시피 3: pytest 테스트 디버깅

hermes 테스트 러너와 pytest 모두 지원합니다:

```bash
# 실패 시 (또는 예외 발생 시) pdb로 진입:
scripts/run_tests.sh tests/path/to/test_file.py::test_name --pdb

# 테스트 시작 시 pdb로 진입:
scripts/run_tests.sh tests/path/to/test_file.py::test_name --trace

# pdb 없이 트레이스백에서 로컬 변수 표시:
scripts/run_tests.sh tests/path/to/test_file.py --showlocals --tb=long
```

참고: `scripts/run_tests.sh`는 기본적으로 xdist(`-n 4`)를 사용하며, pdb는 xdist 하에서 작동하지 **않습니다**. `-p no:xdist`를 추가하거나 `-n 0`으로 단일 테스트를 실행하세요:

```bash
scripts/run_tests.sh tests/foo_test.py::test_bar --pdb -p no:xdist
# 또는
source .venv/bin/activate
python -m pytest tests/foo_test.py::test_bar --pdb
```

이는 격리된(hermetic) 환경 보장을 우회합니다 — 디버깅에는 괜찮지만, 푸시하기 전에 래퍼(wrapper)를 통해 다시 실행하여 확인하세요.

## 레시피 4: 모든 예외에 대한 사후(Post-mortem) 디버깅

```python
import pdb, sys
try:
    run_the_thing()
except Exception:
    pdb.post_mortem(sys.exc_info()[2])
```

또는 전체 스크립트를 감싸기:

```bash
python -m pdb -c continue script.py
# 크래시가 발생하면 pdb가 이를 잡고 예외 프레임에 진입합니다
```

또는 repl/jupyter에서 전역 훅(hook) 설정:

```python
import sys
def excepthook(etype, value, tb):
    import pdb; pdb.post_mortem(tb)
sys.excepthook = excepthook
```

## 레시피 5: debugpy를 사용한 원격 디버깅 (실행 중인 프로세스에 연결)

오래 실행되는 프로세스의 경우: Hermes gateway, tui_gateway, 데몬, 이미 오동작 중이라 깨끗하게 재시작할 수 없는 프로세스.

### 설정

```bash
source /home/bb/hermes-agent/.venv/bin/activate
pip install debugpy
```

### 패턴 A: 소스 편집 — 프로세스가 시작 시 디버거를 대기

진입점(entry point)의 맨 위(또는 디버깅하려는 함수 내부)에 추가:

```python
import debugpy
debugpy.listen(("127.0.0.1", 5678))
print("debugpy listening on 5678, waiting for client...", flush=True)
debugpy.wait_for_client()
debugpy.breakpoint()       # 선택 사항: 연결되자마자 즉시 일시 정지
```

프로세스를 시작하면 `wait_for_client()`에서 차단됩니다.

### 패턴 B: 소스 편집 없음 — `-m debugpy`로 실행

```bash
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client your_script.py arg1
```

모듈 진입의 경우 동등하게:

```bash
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client -m your.module
```

### 패턴 C: 이미 실행 중인 프로세스에 연결

대상 환경에 debugpy가 사전 설치되어 있고 PID가 필요합니다:

```bash
python -m debugpy --listen 127.0.0.1:5678 --pid <pid>
# debugpy가 프로세스에 자체를 주입합니다. 그 후 아래와 같이 클라이언트를 연결합니다.
```

일부 커널/보안 설정은 ptrace 기반 주입을 차단합니다(`/proc/sys/kernel/yama/ptrace_scope`). 다음으로 수정하세요:
```bash
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
```

### 터미널에서 클라이언트 연결

가장 쉬운 터미널 측 DAP 클라이언트는 VS Code CLI 또는 작은 스크립트입니다. Hermes 내에서는 두 가지 실용적인 옵션이 있습니다:

**옵션 1: `debugpy` 자체 CLI REPL** — 공식 기능은 아니지만, 작은 DAP 클라이언트 스크립트:

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
# ... 이벤트 읽기 및 continue/stepIn/etc. 보내기 루프
```

이는 일회성 자동화에는 좋지만 대화형 사용자 경험으로는 고통스럽습니다.

**옵션 2: VS Code / Cursor / Zed에서 연결** — 사용자가 이 중 하나를 열어두었다면 `launch.json`을 추가할 수 있습니다:

```json
{
  "name": "Attach to Hermes",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "127.0.0.1", "port": 5678 },
  "justMyCode": false,
  "pathMappings": [
    { "localRoot": "${workspaceFolder}", "remoteRoot": "/home/bb/hermes-agent" }
  ]
}
```

**옵션 3: DAP를 버리고 `remote-pdb` 사용** — 일반적으로 터미널 에이전트에서 실제로 원하는 방식:

```bash
pip install remote-pdb
```

코드에서:
```python
from remote_pdb import set_trace
set_trace(host="127.0.0.1", port=4444)   # 연결될 때까지 차단됨
```

그 후 터미널에서:
```bash
nc 127.0.0.1 4444
# 로컬에서 디버깅하는 것과 똑같이 (Pdb) 프롬프트를 얻습니다.
```

`remote-pdb`는 `debugpy`의 DAP 프로토콜이 과할 때 터미널 에이전트 친화적인 가장 깔끔한 선택입니다. `debugpy`는 IDE 통합이 실제로 필요할 때만 사용하세요.

## Hermes 관련 프로세스 디버깅

### 테스트
레시피 3을 참조하세요. 항상 `-p no:xdist`를 추가하거나 xdist 없이 단일 테스트를 실행하세요.

### `run_agent.py` / CLI — 일회성
가장 쉬운 방법: 의심되는 줄 근처에 `breakpoint()`를 추가한 다음 정상적으로 `hermes`를 실행합니다. 일시 정지 지점에서 터미널로 제어권이 돌아옵니다.

### `tui_gateway` 하위 프로세스 (`hermes --tui`로 생성됨)
게이트웨이는 Node TUI의 하위로 실행됩니다. 옵션:

**A. 게이트웨이 소스 편집:**
```python
# tui_gateway/server.py serve()의 상단 근처
import debugpy
debugpy.listen(("127.0.0.1", 5678))
debugpy.wait_for_client()
```
`hermes --tui`를 시작합니다. TUI는 정지된 것처럼 보일 것입니다 (백엔드가 대기 중이므로). 클라이언트를 연결하면, `continue`할 때 실행이 재개됩니다.

**B. 특정 핸들러에서 `remote-pdb` 사용:**
```python
from remote_pdb import set_trace
set_trace(host="127.0.0.1", port=4444)   # 트랩할 RPC 핸들러 내부
```
TUI에서 일치하는 슬래시 명령을 트리거한 다음, 다른 터미널에서 `nc 127.0.0.1 4444`를 실행합니다.

### `_SlashWorker` 하위 프로세스
동일한 패턴 — 워커의 `exec` 경로 안에 `set_trace()`와 함께 `remote-pdb`를 사용합니다. 워커는 슬래시 명령 간에 지속되므로, 첫 번째 트리거는 연결될 때까지 차단되고 후속 슬래시 명령은 다시 무장하지 않는 한 정상적으로 통과합니다.

### Gateway (`gateway/run.py`)
오래 지속됩니다. 핸들러에서 `remote-pdb`를 사용하거나, 어차피 게이트웨이를 재시작하는 중이라면 `--wait-for-client`와 함께 `debugpy`를 사용하세요.

## 흔한 함정

1. **pytest-xdist 하에서 pdb는 조용히 아무것도 하지 않습니다.** 프롬프트를 볼 수 없고, 테스트는 그냥 멈춥니다. 항상 `-p no:xdist` 또는 `-n 0`을 사용하세요.

2. **CI / TTY가 아닌 컨텍스트에서 `breakpoint()`는 프로세스를 중단(hang)시킵니다.** 로컬에서는 안전하지만 커밋하지 마세요. 안전망으로 pre-commit grep을 추가하세요.

3. **`PYTHONBREAKPOINT=0`**은 모든 `breakpoint()` 호출을 비활성화합니다. 중단점이 걸리지 않으면 환경 변수를 확인하세요:
   ```bash
   echo $PYTHONBREAKPOINT
   ```

4. **`debugpy.listen`은 `wait_for_client()`를 함께 호출할 때만 차단됩니다.** 그렇지 않으면 실행이 계속되어 클라이언트가 연결되기 전에 첫 번째 중단점이 이미 지나갔을 수 있습니다.

5. **강화된 커널에서 PID 연결이 실패합니다.** `ptrace_scope=1` (Ubuntu 기본값)은 동일한 사용자의 자식 프로세스에 대한 ptrace만 허용합니다. 해결 방법: `echo 0 > /proc/sys/kernel/yama/ptrace_scope` (루트 필요) 또는 처음부터 `debugpy` 하에서 실행합니다.

6. **스레드.** `pdb`는 현재 스레드만 디버깅합니다. 멀티스레드 코드의 경우 `debugpy` (스레드 인식 DAP)를 사용하거나 스레드마다 `threading.settrace()`를 설정하세요.

7. **asyncio.** `pdb`는 코루틴 내에서 작동하지만 pdb 내부의 `await`는 Python 3.13+ 이거나 구버전에서는 `interact` 모드에서 `await`를 요구합니다. 3.11/3.12의 경우 `asyncio.run_coroutine_threadsafe` 트릭이나 `asyncio.ensure_future`를 통한 `!stmt` 기반 await를 사용하세요.

8. **`scripts/run_tests.sh`는 자격 증명을 제거하고 `HOME=<tmpdir>`로 설정합니다.** 버그가 사용자 설정이나 실제 API 키에 의존한다면 래퍼 환경에서 재현되지 않습니다. 먼저 순수 `pytest`로 디버깅하여 재현한 다음, 래퍼 환경에서 재확인하세요.

9. **Fork / multiprocessing.** pdb는 분기를 따라가지 않습니다. 각 자식은 고유의 `breakpoint()`나 `set_trace()`가 필요합니다. Hermes 하위 에이전트의 경우 한 번에 한 프로세스씩 디버깅하세요.

## 검증 체크리스트

- [ ] `pip install debugpy` 후 확인: `python -c "import debugpy; print(debugpy.__version__)"`
- [ ] 원격 디버그의 경우 포트가 실제로 리스닝 중인지 확인: `ss -tlnp | grep 5678`
- [ ] 첫 번째 중단점이 실제로 걸리는지 확인 (그렇지 않다면, `PYTHONBREAKPOINT=0`이거나 xdist 아래에 있거나 연결 전 실행이 끝났을 가능성)
- [ ] `where` / `w` 가 예상된 호출 스택을 보여주는지
- [ ] 디버그 후 정리: 커밋된 코드에 남아있는 `breakpoint()` / `set_trace()`가 없는지
  ```bash
  rg -n 'breakpoint\(\)|set_trace\(|debugpy\.listen' --type py
  ```

## 원샷 레시피

**"왜 이 딕셔너리에 키가 없지?"**
```python
# KeyError 위치 위에 추가
breakpoint()
# 그 후 pdb에서:
(Pdb) pp d
(Pdb) pp list(d.keys())
(Pdb) w                # 어떻게 여기까지 왔는지
```

**"이 테스트는 단독으로는 통과하는데 스위트(suite)에서는 실패해."**
```bash
scripts/run_tests.sh tests/the_test.py --pdb -p no:xdist
# 하지만 '다른 테스트와 함께' 실행할 때만 실패한다면:
source .venv/bin/activate
python -m pytest tests/ -x --pdb -p no:xdist
# 이제 상태가 누적된 후 정확히 실패한 테스트에서 pdb 트랩에 걸립니다.
```

**"내 async 핸들러가 교착 상태(deadlock)에 빠져."**
```python
# 핸들러 진입점에 추가
import remote_pdb; remote_pdb.set_trace(host="127.0.0.1", port=4444)
```
핸들러를 트리거합니다. `nc 127.0.0.1 4444`를 실행한 후 `w`를 눌러 중단된 프레임을 보고, `!import asyncio; asyncio.all_tasks()`를 실행해 또 무엇이 대기 중인지 확인합니다.

**"Ink 하위 프로세스/자식 프로세스 크래시에 대한 사후 디버깅."**
```bash
PYTHONFAULTHANDLER=1 python -m pdb -c continue path/to/entrypoint.py
# 크래시 발생 시, 모든 로컬 변수를 가진 채 예외 프레임에 떨어집니다
```
