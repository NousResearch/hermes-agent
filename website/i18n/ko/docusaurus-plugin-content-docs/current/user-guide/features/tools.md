---
sidebar_position: 1
title: "도구 & 도구 세트 (Tools & Toolsets)"
description: "Hermes Agent의 도구 개요 — 사용 가능한 도구, 도구 세트의 작동 방식 및 터미널 백엔드"
---

# 도구 & 도구 세트 (Tools & Toolsets)

도구(Tools)는 에이전트의 능력을 확장하는 함수입니다. 도구들은 논리적인 **도구 세트(toolsets)**로 구성되어 있으며, 플랫폼별로 활성화하거나 비활성화할 수 있습니다.

## 사용 가능한 도구

Hermes는 웹 검색, 브라우저 자동화, 터미널 실행, 파일 편집, 메모리, 위임(delegation), RL 훈련, 메시지 전송, Home Assistant 등 다양한 기본 내장 도구 레지스트리를 제공합니다.

:::note
**Honcho의 세션 간 메모리**는 기본 내장 도구 세트가 아니라 메모리 공급자 플러그인(`plugins/memory/honcho/`)으로 사용할 수 있습니다. 설치 방법은 [플러그인(Plugins)](./plugins.md)을 참조하세요.
:::

상위 카테고리:

| 카테고고리 | 예시 | 설명 |
|----------|----------|-------------|
| **웹 (Web)** | `web_search`, `web_extract` | 웹을 검색하고 페이지 콘텐츠를 추출합니다. |
| **X 검색 (X Search)** | `x_search` | xAI의 내장 `x_search` 응답(Responses) 도구를 통해 X(Twitter) 게시물 및 스레드를 검색합니다 — xAI 자격 증명(SuperGrok OAuth 또는 `XAI_API_KEY`)이 필요합니다. 기본적으로 꺼져 있으며 `hermes tools` → 🐦 X (Twitter) Search를 통해 옵트인해야 합니다. |
| **터미널 & 파일 (Terminal & Files)** | `terminal`, `process`, `read_file`, `patch` | 명령어를 실행하고 파일을 조작합니다. |
| **브라우저 (Browser)** | `browser_navigate`, `browser_snapshot`, `browser_vision` | 텍스트 및 비전 지원을 통해 상호 작용하는 브라우저 자동화 도구입니다. |
| **미디어 (Media)** | `vision_analyze`, `image_generate`, `text_to_speech` | 멀티모달(multimodal) 분석 및 생성을 수행합니다. |
| **에이전트 오케스트레이션 (Agent orchestration)** | `todo`, `clarify`, `execute_code`, `delegate_task` | 계획 수립, 명확화(clarification), 코드 실행, 하위 에이전트 위임을 처리합니다. |
| **메모리 & 회상 (Memory & recall)** | `memory`, `session_search` | 영구 메모리 및 세션 검색 기능입니다. |
| **자동화 & 전송 (Automation & delivery)** | `cronjob`, `send_message` | 생성/목록/업데이트/일시 정지/재개/실행/제거 동작이 포함된 예약된 작업(Scheduled tasks)과 외부(outbound) 메시지 전송 기능입니다. |
| **통합 (Integrations)** | `ha_*`, MCP 서버 도구, `rl_*` | Home Assistant, MCP, RL 훈련 및 기타 통합(integrations) 도구들입니다. |

코드에서 파생된 공식 레지스트리에 대해서는 [기본 도구 참조(Built-in Tools Reference)](/reference/tools-reference) 및 [도구 세트 참조(Toolsets Reference)](/reference/toolsets-reference)를 확인하세요.

:::tip Nous 도구 게이트웨이
유료 [Nous Portal](https://portal.nousresearch.com) 구독자는 별도의 API 키 없이 **[도구 게이트웨이(Tool Gateway)](tool-gateway.md)**를 통해 웹 검색, 이미지 생성, TTS, 브라우저 자동화를 사용할 수 있습니다. `hermes model`을 실행하여 활성화하거나, `hermes tools`로 개별 도구들을 구성하세요.
:::

## 도구 세트 사용하기

```bash
# 특정 도구 세트 사용
hermes chat --toolsets "web,terminal"

# 사용 가능한 모든 도구 보기
hermes tools

# 플랫폼별 도구 구성 (대화형)
hermes tools
```

일반적인 도구 세트에는 `web`, `search`, `terminal`, `file`, `browser`, `vision`, `image_gen`, `moa`, `skills`, `tts`, `todo`, `memory`, `session_search`, `cronjob`, `code_execution`, `delegation`, `clarify`, `homeassistant`, `messaging`, `spotify`, `discord`, `discord_admin`, `debugging`, `safe`, `rl` 등이 포함됩니다.

`hermes-cli`, `hermes-telegram`과 같은 플랫폼 프리셋(presets)이나 `mcp-<server>` 등의 동적 MCP 도구 세트를 포함한 전체 목록은 [도구 세트 참조(Toolsets Reference)](/reference/toolsets-reference)를 확인하세요.

## 터미널 백엔드 (Terminal Backends)

터미널 도구는 다양한 환경에서 명령어를 실행할 수 있습니다:

| 백엔드 | 설명 | 사용 사례 |
|---------|-------------|----------|
| `local` | 귀하의 시스템에서 실행 (기본값) | 개발, 신뢰할 수 있는 작업 |
| `docker` | 격리된 컨테이너 | 보안, 재현성(reproducibility) |
| `ssh` | 원격 서버 | 샌드박싱, 에이전트가 자신의 코드에 접근하지 못하게 격리 |
| `singularity` | HPC 컨테이너 | 클러스터 컴퓨팅, 루트 권한 불필요(rootless) |
| `modal` | 클라우드 실행 | 서버리스(Serverless), 확장성 |
| `daytona` | 클라우드 샌드박스 작업 공간 | 영구적인 원격 개발 환경 |

### 구성 (Configuration)

```yaml
# ~/.hermes/config.yaml 에 추가
terminal:
  backend: local    # 또는: docker, ssh, singularity, modal, daytona
  cwd: "."          # 작업 디렉터리
  timeout: 180      # 명령어 시간 제한(초)
```

### Docker 백엔드

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**전체 프로세스에 걸쳐 공유되는 단일 영구 컨테이너.** Hermes는 처음 사용할 때 수명이 긴 단일 컨테이너(`docker run -d ... sleep 2h`)를 시작하고, 모든 터미널, 파일 및 `execute_code` 호출을 `docker exec`를 통해 동일한 컨테이너로 라우팅합니다. 작업 디렉터리 변경, 설치된 패키지, 환경 변수 수정 및 `/workspace`에 작성된 파일들은 `/new`, `/reset`, `delegate_task` 하위 에이전트를 가로질러 하나의 도구 호출에서 다음 호출로 계속 유지되며, 이는 Hermes 프로세스의 수명 동안 지속됩니다. 컨테이너는 종료(shutdown) 시 정지되고 제거됩니다.

즉, Docker 백엔드는 명령어당 매번 새로운 컨테이너가 생성되는 것이 아니라 영구적인 샌드박스 VM처럼 동작합니다. 한 번 `pip install foo`를 실행하면 해당 세션의 나머지 기간 동안 계속 유지됩니다. `cd /workspace/project`를 하면 이후의 `ls` 호출은 해당 디렉터리를 보게 됩니다. 컨테이너의 라이프사이클에 대한 전체 세부 정보와 `/workspace` 및 `/root`가 Hermes 재시작 시에도 유지될지 여부를 제어하는 `container_persistent` 플래그에 대해서는 [구성 → Docker 백엔드(Configuration → Docker Backend)](../configuration.md#docker-backend)를 참조하세요.

### SSH 백엔드

에이전트가 자체 코드를 수정할 수 없도록 보안상 권장됩니다:

```yaml
terminal:
  backend: ssh
```
```bash
# ~/.hermes/.env 파일에 자격 증명 설정
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

### Singularity/Apptainer

```bash
# 병렬 작업자(parallel workers)를 위해 사전에 SIF 빌드하기
apptainer build ~/python.sif docker://python:3.11-slim

# 구성
hermes config set terminal.backend singularity
hermes config set terminal.singularity_image ~/python.sif
```

### Modal (서버리스 클라우드)

```bash
uv pip install modal
modal setup
hermes config set terminal.backend modal
```

### 컨테이너 리소스 (Container Resources)

모든 컨테이너 백엔드에 대해 CPU, 메모리, 디스크 및 지속성(persistence)을 구성할 수 있습니다:

```yaml
terminal:
  backend: docker  # 또는 singularity, modal, daytona
  container_cpu: 1              # CPU 코어 수 (기본값: 1)
  container_memory: 5120        # 메모리(MB) (기본값: 5GB)
  container_disk: 51200         # 디스크(MB) (기본값: 50GB)
  container_persistent: true    # 세션 간 파일 시스템 유지 여부 (기본값: true)
```

`container_persistent: true`일 경우, 설치된 패키지, 파일 및 설정이 세션이 바뀌어도 유지됩니다.

### 컨테이너 보안 (Container Security)

모든 컨테이너 백엔드는 다음과 같이 보안이 강화된 상태로 실행됩니다:

- 읽기 전용(Read-only) 루트 파일 시스템 (Docker)
- 모든 Linux 기능(capabilities) 해제
- 권한 상승(privilege escalation) 방지
- PID 제한 (256개의 프로세스)
- 완전한 네임스페이스 격리
- 쓰기 가능한 루트 레이어가 아닌 볼륨(volumes)을 통한 영구적인 작업 공간

Docker는 `terminal.docker_forward_env`를 통해 명시적인 환경 변수 허용 목록(allowlist)을 선택적으로 받을 수 있지만, 전달된 변수들은 컨테이너 내부의 명령어들에게 노출되며 해당 세션에 공개된 것으로 간주되어야 합니다.

## 백그라운드 프로세스 관리

백그라운드 프로세스를 시작하고 관리합니다:

```python
terminal(command="pytest -v tests/", background=true)
# 반환값: {"session_id": "proc_abc123", "pid": 12345}

# 이후 process 도구로 관리:
process(action="list")       # 실행 중인 모든 프로세스 표시
process(action="poll", session_id="proc_abc123")   # 상태 확인
process(action="wait", session_id="proc_abc123")   # 완료될 때까지 대기
process(action="log", session_id="proc_abc123")    # 전체 출력 결과 보기
process(action="kill", session_id="proc_abc123")   # 종료
process(action="write", session_id="proc_abc123", data="y")  # 입력 데이터 전송
```

PTY 모드(`pty=true`)는 Codex 및 Claude Code와 같은 대화형 CLI 도구를 활성화합니다.

## Sudo 지원

명령어에 sudo 권한이 필요한 경우, 비밀번호 입력을 요청받게 됩니다 (세션 동안 캐시됨). 또는 `~/.hermes/.env` 파일에 `SUDO_PASSWORD`를 설정할 수도 있습니다.

:::warning
메시징 플랫폼에서 sudo 실행이 실패하면, 출력 결과에 `SUDO_PASSWORD`를 `~/.hermes/.env`에 추가하라는 팁이 포함됩니다.
:::
