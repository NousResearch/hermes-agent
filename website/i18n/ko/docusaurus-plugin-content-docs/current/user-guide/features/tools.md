---
sidebar_position: 1
title: "도구 및 도구 세트"
description: "Hermes Agent 도구 개요 — 사용 가능한 도구, 도구 세트 작동 방식, 터미널 백엔드"
---

# 도구 및 도구 세트

도구는 에이전트의 기능을 확장하는 함수입니다. 도구는 논리적인 **도구 세트**로 구성되며 플랫폼별로 활성화하거나 비활성화할 수 있습니다.

## 사용 가능한 도구

Hermes에는 웹 검색, 브라우저 자동화, 터미널 실행, 파일 편집, 메모리, 위임, 예약 작업, Home Assistant 등을 아우르는 폭넓은 기본 제공 도구 레지스트리가 포함되어 있습니다.

:::note
**Honcho 교차 세션 메모리**는 기본 제공 도구 세트가 아니라 메모리 제공자 플러그인(`plugins/memory/honcho/`)으로 제공됩니다. 설치 방법은 [플러그인](./plugins.md)을 참조하세요.
:::

상위 수준의 범주는 다음과 같습니다.

| 범주 | 예시 | 설명 |
|----------|----------|-------------|
| **웹** | `web_search`, `web_extract` | 웹을 검색하고 페이지 콘텐츠를 추출합니다. |
| **X 검색** | `x_search` | xAI의 기본 제공 `x_search` Responses 도구를 통해 X(Twitter) 게시물과 스레드를 검색합니다 — xAI 자격 증명(SuperGrok OAuth 또는 `XAI_API_KEY`)이 필요하며 기본적으로 꺼져 있습니다. `hermes tools` → 🐦 X(Twitter) 검색에서 선택적으로 활성화할 수 있습니다. |
| **터미널 및 파일** | `terminal`, `process`, `read_file`, `patch` | 명령을 실행하고 파일을 조작합니다. |
| **브라우저** | `browser_navigate`, `browser_snapshot`, `browser_vision` | 텍스트와 비전 기능을 지원하는 대화형 브라우저 자동화입니다. |
| **미디어** | `vision_analyze`, `image_generate`, `text_to_speech` | 멀티모달 분석 및 생성을 수행합니다. |
| **에이전트 오케스트레이션** | `todo`, `clarify`, `execute_code`, `delegate_task` | 계획 수립, 명확화, 코드 실행 및 하위 에이전트 위임을 수행합니다. |
| **메모리 및 회상** | `memory`, `session_search` | 영구 메모리와 세션 검색을 제공합니다. |
| **자동화** | `cronjob` | 생성/목록/업데이트/일시 중지/재개/실행/삭제 작업으로 예약 작업을 관리합니다. 외부 전송은 에이전트가 호출할 수 있는 도구가 아니라 cron 자체 전송, `hermes send` CLI 및 게이트웨이 알림기가 처리합니다. |
| **통합** | `ha_*`, MCP 서버 도구 | Home Assistant, MCP 및 기타 통합입니다. |

코드에서 파생된 권위 있는 레지스트리는 [기본 제공 도구 참조](/reference/tools-reference) 및 [도구 세트 참조](/reference/toolsets-reference)를 참조하세요.

:::tip Nous 도구 게이트웨이
유료 [Nous Portal](https://portal.nousresearch.com) 구독자는 별도의 API 키 없이 **[도구 게이트웨이](tool-gateway.md)**를 통해 웹 검색, 이미지 생성, TTS 및 브라우저 자동화를 사용할 수 있습니다. `hermes model`을 실행해 활성화하거나 `hermes tools`로 개별 도구를 구성하세요.
:::

## 도구 세트 사용

```bash
# Use specific toolsets
hermes chat --toolsets "web,terminal"

# See all available tools
hermes tools

# Configure tools per platform (interactive)
hermes tools
```

일반적인 도구 세트에는 `web`, `search`, `terminal`, `file`, `browser`, `vision`, `image_gen`, `skills`, `tts`, `todo`, `memory`, `session_search`, `cronjob`, `code_execution`, `delegation`, `clarify`, `homeassistant`, `messaging`, `spotify`, `discord`, `discord_admin`, `debugging` 및 `safe`가 있습니다.

플랫폼 프리셋인 `hermes-cli`, `hermes-telegram`과 `mcp-<server>` 같은 동적 MCP 도구 세트를 포함한 전체 목록은 [도구 세트 참조](/reference/toolsets-reference)를 참조하세요.

## 터미널 백엔드

터미널 도구는 여러 환경에서 명령을 실행할 수 있습니다.

| 백엔드 | 설명 | 사용 사례 |
|---------|-------------|----------|
| `local` | 컴퓨터에서 실행(기본값) | 개발, 신뢰할 수 있는 작업 |
| `docker` | 격리된 컨테이너 | 보안, 재현성 |
| `ssh` | 원격 서버 | 샌드박싱, 에이전트가 자체 코드에 접근하지 않도록 함 |
| `singularity` | HPC 컨테이너 | 클러스터 컴퓨팅, 루트 권한 없음 |
| `modal` | 클라우드 실행 | 서버리스, 확장 |
| `daytona` | 클라우드 샌드박스 작업 공간 | 영구 원격 개발 환경 |
| `vercel_sandbox` | Vercel Sandbox 클라우드 마이크로VM | 스냅샷 기반 파일 시스템 영속성이 있는 클라우드 실행 |

### 구성

```yaml
# In ~/.hermes/config.yaml
terminal:
  backend: local    # or: docker, ssh, singularity, modal, daytona, vercel_sandbox
  cwd: "."          # Working directory
  timeout: 180      # Command timeout in seconds
```

### 셸 시작 파일 및 비대화형 명령

에이전트의 터미널 호출은 **비대화형**으로 셸을 실행합니다. TTY가 없고 사람이 프롬프트에 응답하지 않습니다. 평소 터미널에서는 알아차리지 못하는 무겁거나 대화형인 셸 초기화가 에이전트의 모든 명령을 망가뜨리거나 심각하게 느리게 할 수 있습니다.

- **느린 초기화(`nvm`, 버전 관리자, 네트워크에 접근하는 프롬프트):** 일반적인 `nvm.sh` 소싱은 모든 셸 시작에 눈에 띄는 지연을 추가하며, 에이전트는 셸을 여러 번 시작합니다. rc 파일이 몇 초씩 걸리면 빠른 `git status`도 시간 초과 위험이 있는 작업이 됩니다.
- **TTY를 요구하는 블록:** `.bashrc`/`.zshrc`에서 프롬프트를 표시하거나 `tmux`/`screen`에 연결하거나 `read`를 호출하거나 메뉴를 출력하면 비대화형 셸이 멈춥니다 — 명령이 영원히 실행되는 것처럼 보이다가 시간 초과됩니다.
- **무조건적인 출력:** rc 파일의 `echo` 배너는 에이전트가 파싱해야 하는 모든 명령 출력에 불필요한 내용을 섞습니다.

해결 방법은 대부분의 배포판에서 이미 제공하는 표준 `.bashrc` 가드입니다 — 비대화형 셸에서는 일찍 반환하고, 무겁거나 대화형인 작업은 그 아래에 둡니다.

```bash
# ~/.bashrc — keep this guard near the top
case $- in
  *i*) ;;      # interactive: continue
  *) return;;  # non-interactive: stop here
esac

# heavy/interactive init goes BELOW the guard
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
```

Zsh 사용자는 로그인 전용 설정을 `.zprofile`에, 대화형 전용 설정을 `.zshrc`에 두세요. `.zshenv`는 비대화형 셸을 포함한 모든 셸에서 실행되므로 간결하게 유지해야 합니다. 에이전트에 정말로 rc 파일이 PATH에 추가하는 도구가 필요하다면 가드 **위에** `PATH` 변경을 내보내거나(경로 내보내기는 저렴합니다) 바이너리를 `~/.local/bin`에 심볼릭 링크하세요.

자체 터미널에서 작업한 직후 에이전트의 터미널 명령이 멈추거나 즉시 시간 초과된다면 셸 초기화가 첫 번째 의심 대상입니다.

### Docker 백엔드

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**전체 프로세스에서 공유되는 하나의 영구 컨테이너입니다.** Hermes는 처음 사용할 때 장기 실행 컨테이너 하나(`docker run -d ... sleep infinity`)를 시작하고, 모든 터미널, 파일 및 `execute_code` 도구를 동일한 컨테이너 내부의 `docker exec`로 전달합니다. 작업 디렉터리 변경, 설치된 패키지, 환경 변경 및 `/workspace`에 기록한 파일은 Hermes 프로세스가 살아 있는 동안 `/new`, `/reset` 및 `delegate_task` 하위 에이전트에 걸쳐 다음 도구 호출에도 유지됩니다. 종료 시 컨테이너는 중지되고 제거됩니다.

이는 Docker 백엔드가 매 명령마다 새 컨테이너가 생성되는 환경이 아니라 영구 샌드박스 VM처럼 동작한다는 뜻입니다. 한 번 `pip install foo`를 실행하면 세션 내내 사용할 수 있습니다. 자세한 수명 주기와 `/workspace` 및 `/root`가 Hermes 재시작 후에도 유지되는지 제어하는 `container_persistent` 플래그는 [구성 → Docker 백엔드](../configuration.md#docker-backend)를 참조하세요.

### SSH 백엔드

보안을 위해 권장됩니다 — 에이전트가 자신의 코드를 수정할 수 없습니다.

```yaml
terminal:
  backend: ssh
```
```bash
# Set credentials in ~/.hermes/.env
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

### Singularity/Apptainer

```bash
# Pre-build SIF for parallel workers
apptainer build ~/python.sif docker://python:3.11-slim

# Configure
hermes config set terminal.backend singularity
hermes config set terminal.singularity_image ~/python.sif
```

### Modal(서버리스 클라우드)

```bash
uv pip install modal
modal setup
hermes config set terminal.backend modal
```

### Vercel Sandbox

```bash
pip install 'hermes-agent[vercel]'
hermes config set terminal.backend vercel_sandbox
hermes config set terminal.vercel_runtime node24
```

`VERCEL_TOKEN`, `VERCEL_PROJECT_ID` 및 `VERCEL_TEAM_ID` 세 가지를 모두 사용해 인증하세요. 이 액세스 토큰 설정은 Render, Railway, Docker 및 이와 유사한 호스트에서 배포와 일반적인 장기 실행 Hermes 프로세스에 지원되는 방식입니다. 지원되는 런타임은 `node24`, `node22` 및 `python3.13`이며 Hermes는 원격 작업 공간 루트로 `/vercel/sandbox`를 기본 사용합니다.

일회성 로컬 개발에는 Hermes가 단기 Vercel OIDC 토큰도 허용합니다.

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

연결된 Vercel 프로젝트 디렉터리에서:

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

`container_persistent: true`이면 Hermes는 동일한 작업에서 샌드박스가 재생성될 때 파일 시스템 상태를 보존하기 위해 Vercel 스냅샷을 사용합니다. 여기에는 샌드박스 내부의 Hermes 동기화 자격 증명, 스킬 및 캐시 파일이 포함될 수 있습니다. 스냅샷은 실행 중인 프로세스, PID 공간 또는 동일한 실행 중 샌드박스 ID를 보존하지 않습니다.

백그라운드 터미널 명령은 샌드박스가 살아 있는 동안 Hermes의 일반적인 비로컬 프로세스 흐름을 사용합니다. 생성, 폴링, 대기, 로그 및 종료는 일반 프로세스 도구를 통해 작동하지만 Hermes는 정리 또는 재시작 후 Vercel의 기본 분리 프로세스 복구를 제공하지 않습니다.

`container_disk`를 설정하지 않거나 공유 기본값인 `51200`으로 두세요. Vercel Sandbox에서는 사용자 지정 디스크 크기를 지원하지 않으며 진단/백엔드 생성이 실패합니다.

### 컨테이너 리소스

모든 컨테이너 백엔드에 CPU, 메모리, 디스크 및 영속성을 구성합니다.

```yaml
terminal:
  backend: docker  # or singularity, modal, daytona, vercel_sandbox
  container_cpu: 1              # CPU cores (default: 1)
  container_memory: 5120        # Memory in MB (default: 5GB)
  container_disk: 51200         # Disk in MB (default: 50GB)
  container_persistent: true    # Persist filesystem across sessions (default: true)
```

`container_persistent: true`이면 설치된 패키지, 파일 및 구성이 세션 간에 유지됩니다.

### 컨테이너 보안

모든 컨테이너 백엔드는 보안 강화를 적용해 실행됩니다.

- 읽기 전용 루트 파일 시스템(Docker)
- 모든 Linux 기능 삭제
- 권한 상승 금지
- PID 제한(256개 프로세스)
- 전체 네임스페이스 격리
- 루트 레이어가 아닌 볼륨을 통한 영구 작업 공간

Docker는 `terminal.docker_forward_env`를 통해 명시적인 환경 변수 허용 목록을 선택적으로 전달할 수 있지만, 전달된 변수는 컨테이너 내부 명령에서 보이며 해당 세션에 노출된 것으로 취급해야 합니다.

## 백그라운드 프로세스 관리

백그라운드 프로세스를 시작하고 관리합니다.

```python
terminal(command="pytest -v tests/", background=true)
# Returns: {"session_id": "proc_abc123", "pid": 12345}

# Then manage with the process tool:
process(action="list")       # Show all running processes
process(action="poll", session_id="proc_abc123")   # Check status
process(action="wait", session_id="proc_abc123")   # Block until done
process(action="log", session_id="proc_abc123")    # Full output
process(action="kill", session_id="proc_abc123")   # Terminate
process(action="write", session_id="proc_abc123", data="y")  # Send input
```

PTY 모드(`pty=true`)는 Hermes 및 Claude Code와 같은 대화형 CLI 도구를 활성화합니다.

## Sudo 지원

명령에 sudo가 필요하면 비밀번호를 묻습니다(세션 동안 캐시됨). 또는 `SUDO_PASSWORD`를 `~/.hermes/.env`에 설정하세요.

:::warning
메시징 플랫폼에서 sudo가 실패하면 출력에 `SUDO_PASSWORD`를 `~/.hermes/.env`에 추가하라는 안내가 포함됩니다.
:::
