---
title: "Jupyter Live Kernel — 라이브 Jupyter 커널(hamelnb)을 통한 반복적 Python 탐색"
sidebar_label: "Jupyter Live Kernel"
description: "라이브 Jupyter 커널(hamelnb)을 통한 반복적 Python 탐색"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Jupyter Live Kernel

라이브 Jupyter 커널(hamelnb)을 통한 반복적 Python 탐색.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/data-science/jupyter-live-kernel` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `jupyter`, `notebook`, `repl`, `data-science`, `exploration`, `iterative` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Jupyter Live Kernel (hamelnb)

라이브 Jupyter 커널을 통해 **상태가 보존되는 Python REPL**을 제공합니다. 변수는 여러 번의 실행 사이에도 유지됩니다. 
상태를 점진적으로 빌드하거나, API를 탐색하거나, DataFrame을 검사하거나, 복잡한 코드를 반복적으로 수정해야 할 때 `execute_code` 대신 이 기능을 사용하세요.

## 이 도구와 다른 도구의 사용 시기 비교

| 도구 | 사용 시기 |
|------|----------|
| **이 스킬** | 반복적인 탐색, 단계 간 상태 유지, 데이터 과학, 머신러닝, "이거 한 번 시도해보고 확인해야지" |
| `execute_code` | hermes 도구 액세스(웹 검색, 파일 작업)가 필요한 일회성 스크립트. 상태가 없음(Stateless). |
| `terminal` | 셸 명령, 빌드, 설치, git, 프로세스 관리 |

**경험 법칙:** 이 작업에 Jupyter 노트북이 필요할 것 같다면 이 스킬을 사용하세요.

## 전제 조건

1. **uv**가 설치되어 있어야 합니다 (`which uv`로 확인)
2. **JupyterLab**이 설치되어 있어야 합니다: `uv tool install jupyterlab`
3. Jupyter 서버가 실행 중이어야 합니다 (아래의 설정 부분 참조)

## 설정

hamelnb 스크립트 위치:
```
SCRIPT="$HOME/.agent-skills/hamelnb/skills/jupyter-live-kernel/scripts/jupyter_live_kernel.py"
```

아직 복제되지 않은 경우:
```
git clone https://github.com/hamelsmu/hamelnb.git ~/.agent-skills/hamelnb
```

### JupyterLab 시작하기

서버가 이미 실행 중인지 확인하세요:
```
uv run "$SCRIPT" servers
```

서버를 찾을 수 없다면 하나 시작하세요:
```
jupyter-lab --no-browser --port=8888 --notebook-dir=$HOME/notebooks \
  --IdentityProvider.token='' --ServerApp.password='' > /tmp/jupyter.log 2>&1 &
sleep 3
```

참고: 로컬 에이전트 액세스를 위해 토큰/비밀번호는 비활성화되어 있습니다. 서버는 헤드리스(headless)로 실행됩니다.

### REPL 용도의 노트북 생성

단지 REPL만 필요하다면 (기존 노트북이 없는 경우), 최소한의 노트북 파일을 생성하세요:
```
mkdir -p ~/notebooks
```
빈 코드 셀 하나가 있는 최소한의 .ipynb JSON 파일을 작성한 다음, Jupyter REST API를 통해 커널 세션을 시작하세요:
```
curl -s -X POST http://127.0.0.1:8888/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"path":"scratch.ipynb","type":"notebook","name":"scratch.ipynb","kernel":{"name":"python3"}}'
```

## 핵심 워크플로

모든 명령은 구조화된 JSON을 반환합니다. 토큰을 절약하려면 항상 `--compact`를 사용하세요.

### 1. 서버 및 노트북 검색

```
uv run "$SCRIPT" servers --compact
uv run "$SCRIPT" notebooks --compact
```

### 2. 코드 실행 (기본 동작)

```
uv run "$SCRIPT" execute --path <notebook.ipynb> --code '<python code>' --compact
```

상태는 실행 호출 간에 지속됩니다. 변수, 가져오기(imports), 객체 등이 모두 살아남습니다.

여러 줄의 코드는 $'...' 인용을 사용합니다:
```
uv run "$SCRIPT" execute --path scratch.ipynb --code $'import os\nfiles = os.listdir(".")\nprint(f"Found {len(files)} files")' --compact
```

### 3. 실시간 변수 검사

```
uv run "$SCRIPT" variables --path <notebook.ipynb> list --compact
uv run "$SCRIPT" variables --path <notebook.ipynb> preview --name <varname> --compact
```

### 4. 노트북 셀 편집

```
# 현재 셀 보기
uv run "$SCRIPT" contents --path <notebook.ipynb> --compact

# 새 셀 삽입
uv run "$SCRIPT" edit --path <notebook.ipynb> insert \
  --at-index <N> --cell-type code --source '<code>' --compact

# 셀 소스 교체 (contents 출력의 cell-id 사용)
uv run "$SCRIPT" edit --path <notebook.ipynb> replace-source \
  --cell-id <id> --source '<new code>' --compact

# 셀 삭제
uv run "$SCRIPT" edit --path <notebook.ipynb> delete --cell-id <id> --compact
```

### 5. 확인 (재시작 + 모두 실행)

사용자가 깔끔한 확인을 요구하거나 노트북이 위에서 아래로 원활하게 실행되는지 확인해야 할 때만 사용하세요:

```
uv run "$SCRIPT" restart-run-all --path <notebook.ipynb> --save-outputs --compact
```

## 경험을 바탕으로 한 실용적인 팁

1. **서버 시작 후 첫 번째 실행에서 타임아웃이 발생할 수 있습니다** — 커널이 초기화되는 데 시간이 필요합니다. 타임아웃이 발생하면 다시 시도하세요.

2. **커널 Python은 JupyterLab의 Python입니다** — 패키지는 해당 환경에 설치되어야 합니다. 추가 패키지가 필요한 경우 먼저 JupyterLab 도구 환경에 설치하세요.

3. **`--compact` 플래그는 토큰을 크게 절약합니다** — 항상 사용하세요. 이 플래그가 없으면 JSON 출력이 매우 길어질 수 있습니다.

4. **순수 REPL 용도로 사용할 때는**, scratch.ipynb를 생성하고 굳이 셀 편집을 할 필요 없이 `execute`만 반복해서 사용하세요.

5. **인수 순서가 중요합니다** — `--path`와 같은 하위 명령 플래그는 하위-하위 명령 **앞**에 와야 합니다. 예: `variables list --path nb.ipynb`가 아니라 `variables --path nb.ipynb list`입니다.

6. **세션이 아직 존재하지 않는 경우**, REST API를 통해 시작해야 합니다 (설정 섹션 참조). 도구는 라이브 커널 세션 없이 코드를 실행할 수 없습니다.

7. **오류는 역추적 정보가 포함된 JSON으로 반환됩니다** — 무엇이 잘못되었는지 이해하려면 `ename` 및 `evalue` 필드를 읽으세요.

8. **가끔 웹소켓 타임아웃이 발생합니다** — 특히 커널 재시작 후 첫 번째 시도에서 일부 작업이 타임아웃될 수 있습니다. 문제를 확대하기 전에 한 번 더 시도하세요.

## 타임아웃 기본값

스크립트의 실행당 기본 타임아웃은 30초입니다. 오래 걸리는 작업의 경우 `--timeout 120`을 전달하세요. 초기 설정이나 무거운 계산에는 넉넉한 타임아웃(60+)을 사용하세요.
