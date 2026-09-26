---
title: "Jupyter Notebook — 실시간 Jupyter 커널을 통한 반복적 Python (hamelnb)"
sidebar_label: "Jupyter Notebook"
description: "실시간 Jupyter 커널을 통한 반복적 Python (hamelnb)"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Jupyter Notebook

실시간 Jupyter 커널을 통한 반복적 Python (hamelnb).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/data-science/jupyter-notebook`으로 설치 |
| 경로 | `optional-skills/data-science/jupyter-notebook` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `jupyter`, `notebook`, `repl`, `data-science`, `exploration`, `iterative` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 확인하는 내용입니다.
:::

# Jupyter Notebook (hamelnb 실시간 커널)

실시간 Jupyter 커널을 통해 **상태를 유지하는 Python REPL**을 제공합니다. 변수는
실행 간에 유지됩니다. 상태를 점진적으로 구축하거나, API를 탐색하거나, DataFrame을 검사하거나,
복잡한 코드를 반복적으로 작성해야 할 때 `execute_code` 대신 사용하세요.

## 이 도구를 다른 도구와 비교해 사용할 때

| 도구 | 사용 시점 |
|------|----------|
| **이 스킬** | 반복적 탐색, 단계 간 상태 유지, 데이터 과학, ML, "이렇게 해 보고 확인해 보자" |
| `execute_code` | hermes 도구(web_search, 파일 작업)에 액세스해야 하는 일회성 스크립트. 상태를 유지하지 않습니다. |
| `terminal` | 셸 명령, 빌드, 설치, git, 프로세스 관리 |

**경험칙:** 작업에 Jupyter Notebook이 필요하다고 생각된다면 이 스킬을 사용하세요.

## 사전 요구 사항

1. **uv**가 설치되어 있어야 합니다(확인: `which uv`).
2. **JupyterLab**이 설치되어 있어야 합니다: `uv tool install jupyterlab`.
3. Jupyter 서버가 실행 중이어야 합니다(아래 설정 참조).

## 설정

hamelnb 스크립트 위치:
```
SCRIPT="$HOME/.agent-skills/hamelnb/skills/jupyter-live-kernel/scripts/jupyter_live_kernel.py"
```

아직 클론하지 않았다면:
```
git clone https://github.com/hamelsmu/hamelnb.git ~/.agent-skills/hamelnb
```

### JupyterLab 시작

이미 실행 중인 서버가 있는지 확인합니다:
```
uv run "$SCRIPT" servers
```

서버를 찾지 못했다면 하나를 시작합니다:
```
jupyter-lab --no-browser --port=8888 --notebook-dir=$HOME/notebooks \
  --IdentityProvider.token='' --ServerApp.password='' > /tmp/jupyter.log 2>&1 &
sleep 3
```

참고: 로컬 에이전트 액세스를 위해 토큰/비밀번호를 비활성화합니다. 서버는 헤드리스로 실행됩니다.

### REPL 사용을 위한 Notebook 생성

기존 Notebook이 필요하지 않고 REPL만 필요한 경우, 빈 코드 셀이 하나 있는 최소 Notebook 파일을 생성합니다:
```
mkdir -p ~/notebooks
```

빈 코드 셀이 하나 있는 최소한의 .ipynb JSON 파일을 작성한 다음, Jupyter REST API를 통해 커널 세션을 시작합니다:
```
curl -s -X POST http://127.0.0.1:8888/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"path":"scratch.ipynb","type":"notebook","name":"scratch.ipynb","kernel":{"name":"python3"}}'
```

## 핵심 워크플로

모든 명령은 구조화된 JSON을 반환합니다. 토큰을 절약하려면 항상 `--compact`를 사용하세요.

### 1. 서버와 Notebook 검색

```
uv run "$SCRIPT" servers --compact
uv run "$SCRIPT" notebooks --compact
```

### 2. 코드 실행(주요 작업)

```
uv run "$SCRIPT" execute --path <notebook.ipynb> --code '<python code>' --compact
```

실행 호출 간에 상태가 유지됩니다. 변수, import, 객체가 모두 유지됩니다.

여러 줄 코드는 $'...' 인용을 사용하면 작동합니다:
```
uv run "$SCRIPT" execute --path scratch.ipynb --code $'import os\nfiles = os.listdir(".")\nprint(f"Found {len(files)} files")' --compact
```

### 3. 실시간 변수 검사

```
uv run "$SCRIPT" variables --path <notebook.ipynb> list --compact
uv run "$SCRIPT" variables --path <notebook.ipynb> preview --name <varname> --compact
```

### 4. Notebook 셀 편집

```
# View current cells
uv run "$SCRIPT" contents --path <notebook.ipynb> --compact

# Insert a new cell
uv run "$SCRIPT" edit --path <notebook.ipynb> insert \
  --at-index <N> --cell-type code --source '<code>' --compact

# Replace cell source (use cell-id from contents output)
uv run "$SCRIPT" edit --path <notebook.ipynb> replace-source \
  --cell-id <id> --source '<new code>' --compact

# Delete a cell
uv run "$SCRIPT" edit --path <notebook.ipynb> delete --cell-id <id> --compact
```

### 5. 검증(재시작 + 전체 실행)

사용자가 깨끗한 검증을 요청했거나 Notebook이 처음부터 끝까지 실행되는지 확인해야 할 때만 사용합니다:

```
uv run "$SCRIPT" restart-run-all --path <notebook.ipynb> --save-outputs --compact
```

## 경험에서 얻은 실용적인 팁

1. **서버를 시작한 직후의 첫 실행은 시간 초과될 수 있습니다** — 커널을 초기화하려면 잠시 시간이 필요합니다. 시간 초과가 발생하면 다시 시도하세요.

2. **커널의 Python은 JupyterLab의 Python입니다** — 패키지는 해당 환경에 설치해야 합니다. 추가 패키지가 필요하면 먼저 JupyterLab 도구 환경에 설치하세요.

3. **--compact 플래그는 토큰을 크게 절약합니다** — 항상 사용하세요. 이 플래그가 없으면 JSON 출력이 매우 장황해질 수 있습니다.

4. **순수 REPL 사용의 경우**, scratch.ipynb를 만들고 셀 편집은 신경 쓰지 마세요. `execute`를 반복해서 실행하기만 하면 됩니다.

5. **인수 순서가 중요합니다** — `--path` 같은 하위 명령 플래그는 하위 하위 명령 앞에 와야 합니다. 예: `variables --path nb.ipynb list`이며 `variables list --path nb.ipynb`가 아닙니다.

6. **아직 세션이 없다면**, REST API를 통해 세션을 시작해야 합니다(설정 섹션 참조). 라이브 커널 세션 없이는 도구를 실행할 수 없습니다.

7. **오류는 traceback과 함께 JSON으로 반환됩니다** — 오류를 이해하려면 `ename` 및 `evalue` 필드를 확인하세요.

8. **간헐적으로 WebSocket 시간 초과가 발생할 수 있습니다** — 일부 작업은 특히 커널을 다시 시작한 직후 첫 시도에서 시간 초과될 수 있습니다. 상위 단계로 넘기기 전에 한 번 다시 시도하세요.

9. **이 호스트에서 WebSocket이 지속적으로 시간 초과되면**, zmq 전송을 강제합니다:
   `uv run "$SCRIPT" execute --transport zmq ...`. 증상: 모든 execute가
   "Websocket execution may already have reached the kernel, so auto fallback was
   skipped"를 반환합니다. 커널은 실제로 정상 실행된 것입니다(REST에서 execution_state=idle이고 execution_count가 증가함) — WebSocket 응답 채널만 작동하지 않는 것입니다.
   zmq 전송은 jupyter_client를 직접 사용하여 이 문제를 우회합니다.

10. **REST 전용으로 새 서버를 시작할 때는**, `--ServerApp.disable_check_xsrf=True`를 추가하세요 — 그렇지 않으면 POST /api/sessions에서
    `"'_xsrf' argument missing from POST"` 오류가 발생하고 커널 세션 생성에 실패합니다.

## 시간 초과 기본값

스크립트의 실행당 기본 시간 초과는 30초입니다. 오래 실행되는 작업에는 `--timeout 120`을 전달하세요. 초기 설정이나 무거운 계산에는 넉넉한 시간 초과(60초 이상)를 사용하세요.
