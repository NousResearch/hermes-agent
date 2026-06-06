---
sidebar_position: 2
title: "구성 (Configuration)"
description: "Hermes Agent 구성 — config.yaml, 제공자, 모델, API 키 등"
---

# 구성 (Configuration)

모든 설정은 쉽게 접근할 수 있도록 `~/.hermes/` 디렉토리에 저장됩니다.

:::tip 작동하는 `config.yaml`을 얻는 가장 쉬운 경로
`hermes setup --portal`을 실행하세요. 한 번의 OAuth 인증으로 YAML을 직접 편집하지 않고도 모델 제공자와 네 가지 도구 게이트웨이(Tool Gateway) 도구를 모두 얻을 수 있습니다. Portal 구독자는 토큰 청구 제공자에 대해 10% 할인도 받습니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 디렉토리 구조

```text
~/.hermes/
├── config.yaml     # 설정 (모델, 터미널, TTS, 압축 등)
├── .env            # API 키 및 비밀 정보
├── auth.json       # OAuth 제공자 자격 증명 (Nous Portal 등)
├── SOUL.md         # 기본 에이전트 정체성 (시스템 프롬프트의 슬롯 #1)
├── memories/       # 영구 메모리 (MEMORY.md, USER.md)
├── skills/         # 에이전트가 만든 스킬 (skill_manage 도구로 관리)
├── cron/           # 예약된 작업
├── sessions/       # 게이트웨이 세션
└── logs/           # 로그 (errors.log, gateway.log — 비밀 정보 자동 삭제됨)
```

## 구성 관리

```bash
hermes config              # 현재 구성 보기
hermes config edit         # 편집기에서 config.yaml 열기
hermes config set KEY VAL  # 특정 값 설정
hermes config check        # 누락된 옵션 확인 (업데이트 후)
hermes config migrate      # 대화형으로 누락된 옵션 추가

# 예시:
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...  # .env에 저장됨
```

:::tip
`hermes config set` 명령어는 값을 적절한 파일로 자동 라우팅합니다. API 키는 `.env`에 저장되고 다른 모든 것은 `config.yaml`에 저장됩니다.
:::

## 구성 우선순위

설정은 다음 순서로 해석됩니다 (가장 높은 우선순위부터):

1. **CLI 인수** — 예: `hermes chat --model anthropic/claude-sonnet-4` (호출별 재정의)
2. **`~/.hermes/config.yaml`** — 비밀 정보가 아닌 모든 설정을 위한 기본 구성 파일
3. **`~/.hermes/.env`** — 환경 변수에 대한 폴백; 비밀 정보(API 키, 토큰, 비밀번호)의 경우 **필수**
4. **기본 내장값** — 아무것도 설정되지 않았을 때 하드코딩된 안전한 기본값

:::info 경험 법칙 (Rule of Thumb)
비밀 정보(API 키, 봇 토큰, 비밀번호)는 `.env`에 넣습니다. 그 외의 모든 것(모델, 터미널 백엔드, 압축 설정, 메모리 제한, 도구 세트)은 `config.yaml`에 넣습니다. 두 가지가 모두 설정된 경우 비밀 정보가 아닌 설정에 대해서는 `config.yaml`이 우선합니다.
:::

## 환경 변수 치환

`${VAR_NAME}` 구문을 사용하여 `config.yaml`에서 환경 변수를 참조할 수 있습니다:

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

단일 값에서 여러 참조가 작동합니다: `url: "${HOST}:${PORT}"`. 참조된 변수가 설정되지 않은 경우 자리 표시자는 그대로 유지됩니다 (`${UNDEFINED_VAR}`는 그대로 유지됨). `${VAR}` 구문만 지원됩니다. 구문 없이 사용되는 `$VAR`은 확장되지 않습니다.

AI 제공자 설정(OpenRouter, Anthropic, Copilot, 사용자 지정 엔드포인트, 자체 호스팅 LLM, 폴백 모델 등)은 [AI Providers](/integrations/providers)를 참조하세요.

### 제공자 타임아웃

제공자 전체의 요청 타임아웃을 위해 `providers.<id>.request_timeout_seconds`를 설정할 수 있으며, 모델별 재정의를 위해 `providers.<id>.models.<model>.timeout_seconds`를 추가로 설정할 수 있습니다. 이는 모든 전송(OpenAI-wire, 기본 Anthropic, Anthropic 호환)에서의 기본 턴 클라이언트, 폴백 체인, 자격 증명 교체 후 재빌드 및 (OpenAI-wire의 경우) 요청별 timeout kwarg에 적용되므로 구성된 값이 이전의 `HERMES_API_TIMEOUT` 환경 변수보다 우선합니다.

비 스트리밍(non-streaming) 지연 호출(stale-call) 감지기를 위해 `providers.<id>.stale_timeout_seconds`를 설정할 수 있으며, 모델별 재정의를 위해 `providers.<id>.models.<model>.stale_timeout_seconds`를 추가로 설정할 수 있습니다. 이는 이전의 `HERMES_API_CALL_STALE_TIMEOUT` 환경 변수보다 우선합니다.

이를 설정하지 않은 채로 두면 이전 기본값(`HERMES_API_TIMEOUT=1800`초, `HERMES_API_CALL_STALE_TIMEOUT=300`초, 기본 Anthropic 900초)이 유지됩니다. AWS Bedrock의 경우 현재 연결되어 있지 않습니다 (`bedrock_converse` 및 AnthropicBedrock SDK 경로는 모두 자체 타임아웃 구성이 있는 boto3를 사용합니다). [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example)의 주석 처리된 예시를 참조하세요.

## 터미널 백엔드 구성

Hermes는 6개의 터미널 백엔드를 지원합니다. 각각은 에이전트의 쉘 명령어가 로컬 시스템, Docker 컨테이너, SSH를 통한 원격 서버, Modal 클라우드 샌드박스(직접 또는 Nous 관리 게이트웨이를 통해), Daytona 작업 공간 또는 Singularity/Apptainer 컨테이너 등 어디에서 실제로 실행될지를 결정합니다.

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | singularity
  cwd: "."          # 게이트웨이/크론 작업 디렉토리 (CLI는 항상 실행된 디렉토리를 사용)
  timeout: 180      # 명령어당 타임아웃 (초)
  env_passthrough: []  # 샌드박스 실행으로 전달할 환경 변수 이름 (터미널 + execute_code)
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Singularity 백엔드용 컨테이너 이미지
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Modal 백엔드용 컨테이너 이미지
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Daytona 백엔드용 컨테이너 이미지
```

Modal 및 Daytona와 같은 클라우드 샌드박스의 경우 `container_persistent: true`는 Hermes가 샌드박스 재생성 시에 파일 시스템 상태를 유지하려고 시도함을 의미합니다. 동일한 라이브 샌드박스, PID 공간 또는 백그라운드 프로세스가 나중에도 여전히 실행되고 있을 것이라고 보장하지는 않습니다.

### 백엔드 개요

| 백엔드 | 명령어 실행 위치 | 격리 수준 | 추천 용도 |
|---------|-------------------|-----------|----------|
| **local** | 본인 시스템 직접 | 없음 | 개발, 개인 용도 |
| **docker** | 단일 영구 Docker 컨테이너 (세션, `/new`, 하위 에이전트 간 공유) | 완전 (네임스페이스, 권한 제한) | 안전한 샌드박싱, CI/CD |
| **ssh** | SSH를 통한 원격 서버 | 네트워크 경계 | 원격 개발, 강력한 하드웨어 |
| **modal** | Modal 클라우드 샌드박스 | 완전 (클라우드 VM) | 임시 클라우드 컴퓨팅, 평가 |
| **daytona** | Daytona 작업 공간 | 완전 (클라우드 컨테이너) | 관리형 클라우드 개발 환경 |
| **singularity** | Singularity/Apptainer 컨테이너 | 네임스페이스 (--containall) | HPC 클러스터, 공유 시스템 |

### 로컬 백엔드

기본값입니다. 명령어는 격리 없이 시스템에서 직접 실행됩니다. 특별한 설정이 필요하지 않습니다.

```yaml
terminal:
  backend: local
```

:::warning
에이전트는 귀하의 사용자 계정과 동일한 파일 시스템 액세스 권한을 가집니다. 원하지 않는 도구를 비활성화하려면 `hermes tools`를 사용하거나 격리를 위해 Docker로 전환하세요.
:::

### Docker 백엔드

보안이 강화된 Docker 컨테이너 내에서 명령어를 실행합니다(모든 기능(capabilities) 삭제, 권한 상승 없음, PID 제한).

**단일 영구 컨테이너, Hermes 프로세스 간에 공유.** Hermes는 첫 사용 시 장기 실행되는 단일 컨테이너를 시작하고, 세션, `/new`, `/reset` 및 `delegate_task` 하위 에이전트를 아울러 모든 터미널, 파일 및 `execute_code` 호출을 `docker exec`를 통해 동일한 컨테이너로 라우팅합니다. 작업 디렉토리 변경, 설치된 패키지, `/workspace`의 파일 및 **백그라운드 프로세스** 모두 하나의 도구 호출에서 다음 호출로, 그리고 한 Hermes 프로세스에서 다음 프로세스로 이어집니다. TUI 세션을 닫거나, `/quit`을 실행하거나, 새 `hermes` 호출을 시작할 때 컨테이너는 계속 실행되며 다음 Hermes 프로세스는 레이블 조회(labeled lookup)를 통해 이를 재사용합니다. 정확한 해제(teardown) 규칙은 아래의 **컨테이너 라이프사이클**을 참조하세요.

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # 실행된 디렉토리를 /workspace에 마운트
  docker_run_as_host_user: false   # 아래의 "호스트 사용자로 컨테이너 실행" 참조
  docker_forward_env:              # 컨테이너로 전달할 호스트 환경 변수
    - "GITHUB_TOKEN"
  docker_env:                      # 주입할 리터럴 환경 변수 (KEY=value)
    DEBUG: "1"
    PYTHONUNBUFFERED: "1"
  docker_volumes:                  # 호스트 디렉토리 마운트
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro는 읽기 전용
  docker_extra_args:               # `docker run`에 그대로 추가될 추가 플래그
    - "--gpus=all"
    - "--network=host"

  # 리소스 제한
  container_cpu: 1                 # CPU 코어 수 (0 = 무제한)
  container_memory: 5120           # MB (0 = 무제한)
  container_disk: 51200            # MB (XFS+pquota의 overlay2 필요)
  container_persistent: true       # /workspace 및 /root 바인드 마운트 디렉토리 유지

  # 프로세스 간 컨테이너 재사용 (기본값은 "세션 간 공유되는 단일 장기 실행 컨테이너" 계약과 일치함 — 컨테이너 라이프사이클 참조).
  docker_persist_across_processes: true   # Hermes 재시작 시 컨테이너 재사용
  docker_orphan_reaper: true              # 시작 시 버려진 Exited 컨테이너 정리

  # 교차 백엔드 라이프사이클 설정 (docker에도 적용됨)
  timeout: 180                     # 명령어당 타임아웃 (초)
  lifetime_seconds: 300            # 유휴 정리기(idle-reaper) 윈도우; 2× orphan-reaper 임계값에도 영향을 줌
```

**`docker_env`** vs **`docker_forward_env`**: 전자는 구성 파일에 지정한 리터럴 `KEY=value` 쌍을 주입합니다(값은 `config.yaml`에 있거나 `TERMINAL_DOCKER_ENV='{"DEBUG":"1"}'`를 통해 JSON 딕셔너리로 전달됨). 후자는 쉘 또는 `~/.hermes/.env`에서 값을 전달하므로 실제 비밀 정보는 구성 파일에 나타나지 않습니다. 토큰에는 `docker_forward_env`를 사용하고 컨테이너에 필요한 정적 노브(knobs)에는 `docker_env`를 사용하세요.

**`terminal.docker_extra_args`** (`TERMINAL_DOCKER_EXTRA_ARGS='["--gpus=all"]'`를 통해서도 재정의 가능)를 사용하면 Hermes가 1급 키로 노출하지 않는 임의의 `docker run` 플래그(`--gpus`, `--network`, `--add-host`, 대체 `--security-opt` 재정의 등)를 전달할 수 있습니다. 각 항목은 문자열이어야 합니다. 목록은 조합된 `docker run` 호출의 마지막에 추가되므로 필요한 경우 Hermes의 기본값을 덮어쓸 수 있습니다. 주의해서 사용하세요 — 샌드박스 강화(기능 삭제, `--user`, 작업 공간 바인드 마운트)와 충돌하는 플래그는 암묵적으로 격리를 약화시킵니다.

**요구 사항:** Docker Desktop 또는 Docker Engine이 설치되어 실행 중이어야 합니다. Hermes는 `$PATH`와 일반적인 macOS 설치 위치(`/usr/local/bin/docker`, `/opt/homebrew/bin/docker`, Docker Desktop 앱 번들)를 탐색합니다. Podman은 즉시 지원됩니다. 둘 다 설치된 경우 강제로 Podman을 사용하려면 `HERMES_DOCKER_BINARY=podman`(또는 전체 경로)을 설정하세요.

#### 컨테이너 라이프사이클

Hermes가 관리하는 모든 컨테이너에는 3개의 레이블이 지정되어 후속 프로세스(및 orphan reaper)가 이를 식별할 수 있습니다:

- `hermes-agent=1` — Hermes 관리 컨테이너로 표시
- `hermes-task-id=<sanitized task_id>` — 작업별 재사용 프로브의 키가 됨
- `hermes-profile=<sanitized profile name>` — 재사용 및 정리를 활성 Hermes 프로필로 범위 제한

시작할 때 Hermes는 `docker ps --filter label=hermes-task-id=<id> --filter label=hermes-profile=<profile>`을 실행하고 찾으면 **기존 컨테이너에 연결합니다**. 컨테이너가 `exited` 상태인 경우(예: Docker 데몬 재시작 후) `docker start`되어 재사용됩니다. 파일 시스템 상태와 설치된 모든 패키지는 살아남지만 컨테이너 내 백그라운드 프로세스는 그렇지 않습니다.

Hermes 프로세스가 종료될 때 — `/quit`, TUI 세션 닫기, 게이트웨이 셧다운, 심지어 SIGKILL — 기본 모드에서 컨테이너의 정리(cleanup) 경로는 **작업을 수행하지 않습니다(no-op)**. 컨테이너는 계속 실행됩니다. 다음 Hermes 프로세스는 레이블 프로브를 통해 밀리초 내에 이 컨테이너에 연결합니다. 이는 "세션 간에 공유되는 단일 장기 실행 컨테이너" 계약에서 요구하는 동작입니다. 이것이 백그라운드 프로세스(npm watcher, 개발 서버, 장기 실행되는 pytest)가 세션 간에 생존하는 유일한 방법입니다.

**컨테이너는 다음과 같은 경우에만 해제(중지 및 `docker rm -f`)됩니다:**

| 트리거 | 작동 시점 |
|---|---|
| `docker_persist_across_processes: false` | 명시적 프로세스별 격리. 모든 `cleanup()`이 `stop` + `rm -f`를 수행합니다. issue-#20561 이전의 동작과 일치합니다. |
| 유휴 정리기(Idle reaper) (`lifetime_seconds`, 기본값 300초) | 환경이 `persist_across_processes=false`일 때만 작동합니다. 영속(persist) 모드 환경은 작업이 무시(no-op)되며, 컨테이너는 유휴 정리를 생존합니다. |
| 다음 시작 시 고아 정리기(Orphan reaper) | `2 × lifetime_seconds` (기본값 600초 = 10분)보다 오래된 **Exited** hermes 레이블 컨테이너를 정리하며, 현재 프로필로 범위가 제한됩니다. **실행 중인 컨테이너는 절대 건드리지 않습니다** — 형제 프로세스(sibling-process) 안전성. 비활성화하려면 `docker_orphan_reaper: false`로 설정하세요. |
| 직접적인 사용자 행동 | `docker rm -f`, `docker system prune`, Docker Desktop 재시작. 우리는 `--restart=always`를 설정하지 않으므로, 호스트 재부팅 시 컨테이너는 `Exited` 상태로 남습니다 (CoW 레이어는 유지되며 다음 시작 시 재사용되지만 bg 프로세스는 사라집니다). |

알아둘 가치가 있는 엣지 케이스:

- **컨테이너 내 PID 1의 OOM 종료**는 컨테이너를 `Exited` 상태로 전환합니다. 다음 재사용 시 `docker start`를 실행합니다. 파일 시스템 상태는 살아남지만 bg 프로세스는 아닙니다.
- **프로필 전환**은 컨테이너를 서로 격리합니다 — `hermes-profile=work` 레이블이 지정된 컨테이너는 `hermes-profile=research`로 실행되는 Hermes 프로세스에 보이지 않습니다. 고아 정리기도 프로필 범위이므로, 다른 프로필의 컨테이너는 실수로 정리되지 않지만, 해당 컨테이너들의 원래 프로필에서 Hermes를 다시 시작할 때까지는 자동으로 정리되지도 않습니다.

`delegate_task(tasks=[...])`를 통해 생성된 병렬 하위 에이전트들은 이 하나의 컨테이너를 공유합니다 — 동시 `cd`, 환경 변수 수정 및 동일한 경로에 대한 쓰기가 충돌하게 됩니다. 하위 에이전트에 격리된 샌드박스가 필요한 경우, `register_task_env_overrides()`를 통해 작업별 이미지 재정의(override)를 등록해야 하며, 이는 RL 및 벤치마크 환경(TerminalBench2, HermesSweEnv 등)이 각 작업의 Docker 이미지를 위해 자동으로 수행합니다.

**보안 강화:**
- `DAC_OVERRIDE`, `CHOWN`, `FOWNER`만 다시 추가된 상태로 `--cap-drop ALL`
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- `/tmp` (512MB), `/var/tmp` (256MB), `/run` (64MB)에 대해 크기가 제한된 tmpfs

**자격 증명 전달:** `docker_forward_env`에 나열된 환경 변수는 먼저 쉘 환경에서 확인된 다음 `~/.hermes/.env`에서 확인됩니다. 스킬은 자동으로 병합되는 `required_environment_variables`를 선언할 수도 있습니다.

#### 환경 변수 재정의

`terminal:` 아래의 모든 키는 `TERMINAL_<KEY_UPPERCASE>` 형태의 환경 변수 재정의를 가집니다. Docker 백엔드에서 가장 유용한 것들:

| 환경 변수 | 맵핑되는 항목 | 참고 |
|---|---|---|
| `TERMINAL_DOCKER_IMAGE` | `docker_image` | 베이스 이미지 |
| `TERMINAL_DOCKER_FORWARD_ENV` | `docker_forward_env` | JSON 배열: `'["GITHUB_TOKEN","OPENAI_API_KEY"]'` |
| `TERMINAL_DOCKER_ENV` | `docker_env` | JSON 딕셔너리: `'{"DEBUG":"1"}'` |
| `TERMINAL_DOCKER_VOLUMES` | `docker_volumes` | `"host:container[:ro]"` 문자열의 JSON 배열 |
| `TERMINAL_DOCKER_EXTRA_ARGS` | `docker_extra_args` | JSON 배열 |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | `docker_mount_cwd_to_workspace` | `true` / `false` |
| `TERMINAL_DOCKER_RUN_AS_HOST_USER` | `docker_run_as_host_user` | `true` / `false` |
| `TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES` | `docker_persist_across_processes` | `true` / `false` — 기본값 `true` |
| `TERMINAL_DOCKER_ORPHAN_REAPER` | `docker_orphan_reaper` | `true` / `false` — 기본값 `true` |
| `TERMINAL_CONTAINER_CPU` | `container_cpu` | CPU 코어 수 |
| `TERMINAL_CONTAINER_MEMORY` | `container_memory` | MB 단위 |
| `TERMINAL_CONTAINER_DISK` | `container_disk` | MB 단위 |
| `TERMINAL_CONTAINER_PERSISTENT` | `container_persistent` | `true` / `false` — 바인드 마운트 작업공간 디렉토리를 제어하며, `docker_persist_across_processes`와 별개임 |
| `TERMINAL_LIFETIME_SECONDS` | `lifetime_seconds` | 유휴(idle) 정리기 시간 |
| `TERMINAL_TIMEOUT` | `timeout` | 명령어당 타임아웃 |
| `HERMES_DOCKER_BINARY` | _없음_ | 특정 docker/podman 바이너리 경로 강제 지정 |

### SSH 백엔드

SSH를 통해 원격 서버에서 명령어를 실행합니다. 연결 재사용을 위해 ControlMaster를 사용합니다 (5분 유휴 keepalive). 영구 쉘(persistent shell)이 기본적으로 활성화되어 있습니다 — 상태(cwd, 환경 변수)가 명령어 간에 유지됩니다.

```yaml
terminal:
  backend: ssh
  persistent_shell: true           # 장기 실행되는 bash 세션 유지 (기본값: true)
```

**필수 환경 변수:**

```bash
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=ubuntu
```

**선택 사항:**

| 변수 | 기본값 | 설명 |
|----------|---------|-------------|
| `TERMINAL_SSH_PORT` | `22` | SSH 포트 |
| `TERMINAL_SSH_KEY` | (시스템 기본값) | SSH 프라이빗 키 경로 |
| `TERMINAL_SSH_PERSISTENT` | `true` | 영구 쉘 활성화 |

**작동 방식:** 시작 시 `BatchMode=yes` 및 `StrictHostKeyChecking=accept-new`로 연결합니다. 영구 쉘은 임시 파일을 통해 통신하면서 원격 호스트에서 단일 `bash -l` 프로세스를 살아있게 유지합니다. `stdin_data`나 `sudo`가 필요한 명령어는 자동으로 단발성(one-shot) 모드로 폴백됩니다.

### Modal 백엔드

[Modal](https://modal.com) 클라우드 샌드박스에서 명령어를 실행합니다. 각 작업은 구성 가능한 CPU, 메모리 및 디스크가 있는 격리된 VM을 얻습니다. 파일 시스템은 세션 간에 스냅샷/복원될 수 있습니다.

```yaml
terminal:
  backend: modal
  container_cpu: 1                 # CPU 코어 수
  container_memory: 5120           # MB (5GB)
  container_disk: 51200            # MB (50GB)
  container_persistent: true       # 파일 시스템 스냅샷/복원
```

**필수:** `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` 환경 변수 또는 `~/.modal.toml` 구성 파일.

**지속성(Persistence):** 활성화되면 샌드박스 파일 시스템이 정리 시점에 스냅샷으로 저장되고 다음 세션에 복원됩니다. 스냅샷은 `~/.hermes/modal_snapshots.json`에서 추적됩니다. 이는 실시간 프로세스, PID 공간 또는 백그라운드 작업이 아닌 파일 시스템 상태만을 보존합니다.

**자격 증명 파일:** `~/.hermes/`(OAuth 토큰 등)에서 자동으로 마운트되며 각 명령어 이전에 동기화됩니다.

### Daytona 백엔드

[Daytona](https://daytona.io) 관리형 작업 공간에서 명령어를 실행합니다. 지속성을 위한 정지/재개(stop/resume)를 지원합니다.

```yaml
terminal:
  backend: daytona
  container_cpu: 1                 # CPU 코어 수
  container_memory: 5120           # MB → GiB로 변환됨
  container_disk: 10240            # MB → GiB로 변환됨 (최대 10 GiB)
  container_persistent: true       # 삭제 대신 정지/재개
```

**필수:** `DAYTONA_API_KEY` 환경 변수.

**지속성:** 활성화된 경우 샌드박스는 정리 시 삭제되지 않고 정지(stopped)되며, 다음 세션에서 재개(resumed)됩니다. 샌드박스 이름은 `hermes-{task_id}` 패턴을 따릅니다.

**디스크 제한:** Daytona는 최대 10 GiB를 적용합니다. 이를 초과하는 요청은 경고와 함께 제한됩니다.

### Singularity/Apptainer 백엔드

[Singularity/Apptainer](https://apptainer.org) 컨테이너에서 명령어를 실행합니다. Docker를 사용할 수 없는 HPC 클러스터 및 공유 시스템용으로 설계되었습니다.

```yaml
terminal:
  backend: singularity
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"
  container_cpu: 1                 # CPU 코어 수
  container_memory: 5120           # MB 단위
  container_persistent: true       # 쓰기 가능한 오버레이가 세션 간 유지됨
```

**요구 사항:** `$PATH` 내에 `apptainer` 또는 `singularity` 바이너리.

**이미지 처리:** Docker URL(`docker://...`)은 자동으로 SIF 파일로 변환되어 캐시됩니다. 기존 `.sif` 파일은 직접 사용됩니다.

**스크래치(Scratch) 디렉토리:** 다음 순서로 확인됩니다: `TERMINAL_SCRATCH_DIR` → `TERMINAL_SANDBOX_DIR/singularity` → `/scratch/$USER/hermes-agent` (HPC 규칙) → `~/.hermes/sandboxes/singularity`.

**격리:** 호스트 홈 디렉토리를 마운트하지 않고 전체 네임스페이스 격리를 수행하기 위해 `--containall --no-home`을 사용합니다.

### 일반적인 터미널 백엔드 문제

터미널 명령어가 즉시 실패하거나 터미널 도구가 비활성화된 것으로 보고되는 경우:

- **Local (로컬)** — 특별한 요구 사항 없음. 시작할 때 가장 안전한 기본값입니다.
- **Docker** — `docker version`을 실행하여 Docker가 작동하는지 확인하세요. 실패할 경우 Docker를 수정하거나 `hermes config set terminal.backend local`을 실행하세요.
- **SSH** — `TERMINAL_SSH_HOST`와 `TERMINAL_SSH_USER`가 모두 설정되어 있어야 합니다. 어느 하나라도 누락되면 Hermes가 명확한 오류를 기록합니다.
- **Modal** — `MODAL_TOKEN_ID` 환경 변수 또는 `~/.modal.toml`이 필요합니다. `hermes doctor`를 실행하여 확인하세요.
- **Daytona** — `DAYTONA_API_KEY`가 필요합니다. Daytona SDK가 서버 URL 구성을 처리합니다.
- **Singularity** — `$PATH`에 `apptainer` 또는 `singularity`가 필요합니다. HPC 클러스터에서 일반적입니다.

의심스러운 경우, `terminal.backend`를 다시 `local`로 설정하고 거기에서 명령어가 먼저 실행되는지 확인하세요.

### 종료 시 원격 시스템에서 호스트로의 파일 동기화

**SSH**, **Modal**, **Daytona** 백엔드(에이전트의 작업 트리가 Hermes를 실행 중인 호스트가 아닌 다른 머신에 있는 모든 곳)의 경우, Hermes는 에이전트가 원격 샌드박스 내부에서 수정한 파일을 추적하고, 세션 해제(teardown) / 샌드박스 정리 시 **수정된 파일을 `~/.hermes/cache/remote-syncs/<session-id>/` 아래 호스트로 동기화**합니다.

- 트리거되는 시점: 세션 닫기, `/new`, `/reset`, 게이트웨이 메시지 타임아웃, 원격 백엔드를 사용한 하위 에이전트의 `delegate_task` 완료 시점.
- 에이전트가 명시적으로 연 파일만이 아니라 에이전트가 수정한 전체 트리를 다룹니다. 추가, 편집 및 삭제가 모두 캡처됩니다.
- 찾으러 갈 때쯤이면 원격 샌드박스는 이미 삭제되었을 수 있습니다; 로컬 `~/.hermes/cache/remote-syncs/…` 사본이 에이전트가 변경한 내용의 공신력 있는 기록입니다.
- 대용량 바이너리 출력물(모델 체크포인트, 원본 데이터셋)은 크기로 제한됩니다 — 동기화는 `file_sync_max_mb` (기본값 `100`)을 초과하는 파일은 건너뜁니다. 더 큰 산출물이 예상되면 해당 값을 높이세요.

```yaml
terminal:
  file_sync_max_mb: 100     # 기본값 — 최대 100 MB 크기의 파일 동기화
  file_sync_enabled: true   # 기본값 — 동기화를 완전히 건너뛰려면 false로 설정
```

이것이 에이전트에게 각 결과물을 명시적으로 `scp` 또는 `modal volume put` 하라고 지시하지 않고도 세션이 종료된 후 파괴되는 임시 클라우드 샌드박스에서 결과를 복구하는 방법입니다.

### Docker 볼륨 마운트

Docker 백엔드를 사용할 때, `docker_volumes`를 사용하여 호스트 디렉토리를 컨테이너와 공유할 수 있습니다. 각 항목은 표준 Docker `-v` 구문(`host_path:container_path[:options]`)을 사용합니다.

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # 읽기/쓰기 (기본값)
    - "/home/user/datasets:/data:ro"              # 읽기 전용
    - "/home/user/.hermes/cache/documents:/output" # 게이트웨이에서 볼 수 있는 내보내기용
```

이 기능은 다음의 경우에 유용합니다:
- 에이전트에게 **파일 제공** (데이터셋, 구성 파일, 참조 코드)
- 에이전트로부터 **파일 수신** (생성된 코드, 보고서, 내보내기 파일)
- 사용자와 에이전트가 동일한 파일에 접근하는 **공유 작업 공간**

메시징 게이트웨이를 사용하고 에이전트가 `MEDIA:/...`를 통해 생성된 파일을 전송하게 하려면 `/home/user/.hermes/cache/documents:/output`과 같이 호스트에서 볼 수 있는 전용 내보내기 마운트를 선호하세요.

- Docker 내의 `/output/...`에 파일을 작성합니다
- 다음과 같이 `MEDIA:`에서 **호스트 경로**를 내보냅니다:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`
- 호스트의 게이트웨이 프로세스에도 해당 경로가 정확히 존재하지 않는 한 `/workspace/...` 또는 `/output/...`를 **내보내지 마세요**.

:::warning
YAML에서 중복된 키는 이전 키를 말없이 덮어씁니다. 이미 `docker_volumes:` 블록이 있는 경우 파일 뒷부분에 `docker_volumes:` 키를 추가하는 대신 동일한 목록에 새 마운트를 병합하세요.
:::

환경 변수를 통해서도 설정할 수 있습니다: `TERMINAL_DOCKER_VOLUMES='["/host:/container"]'` (JSON 배열).

### Docker 자격 증명 전달

기본적으로 Docker 터미널 세션은 호스트의 임의 자격 증명을 상속하지 않습니다. 컨테이너 내부에서 특정 토큰이 필요한 경우 `terminal.docker_forward_env`에 추가하세요.

```yaml
terminal:
  backend: docker
  docker_forward_env:
    - "GITHUB_TOKEN"
    - "NPM_TOKEN"
```

Hermes는 먼저 현재 쉘에서 나열된 각 변수를 확인한 다음 `hermes config set`으로 저장된 경우 `~/.hermes/.env`로 폴백합니다.

:::warning
`docker_forward_env`에 나열된 모든 항목은 컨테이너 내부에서 실행되는 명령어에 노출됩니다. 터미널 세션에 노출되어도 괜찮은 자격 증명만 전달하세요.
:::

### 컨테이너를 호스트 사용자로 실행하기

기본적으로 Docker 컨테이너는 `root` (UID 0)로 실행됩니다. `/workspace` 또는 기타 바인드 마운트에 생성된 파일은 호스트의 루트(root) 소유가 되므로, 세션 후 호스트 편집기에서 이를 편집하려면 `sudo chown`을 수행해야 합니다. `terminal.docker_run_as_host_user` 플래그가 이를 수정합니다:

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # 기본값: false
```

활성화되면 Hermes는 `docker run` 명령에 `--user $(id -u):$(id -g)`를 추가하여, 바인드 마운트 디렉토리(`/workspace`, `/root`, `docker_volumes`의 모든 항목)에 기록된 파일이 루트(root)가 아닌 호스트 사용자의 소유가 되도록 합니다. 절충점: 컨테이너는 더 이상 `apt install`을 수행하거나 `/root/.npm`과 같은 루트 소유 경로에 쓸 수 없습니다 — 둘 다 필요한 경우 `HOME`이 루트가 아닌 사용자 소유인 기본 이미지를 사용하거나 (또는 이미지 빌드 시 필요한 도구를 추가하세요).

이전 버전과의 호환성을 위해 이 값을 `false` (기본값)로 둡니다. 워크플로가 대부분 "마운트된 호스트 파일 편집"이고 `sudo chown -R`에 지쳤을 때 이를 켜세요.

### 선택 사항: 실행된 디렉토리를 `/workspace`에 마운트하기

Docker 샌드박스는 기본적으로 격리된 상태를 유지합니다. Hermes는 명시적으로 옵트인하지 않는 한 현재 호스트 작업 디렉토리를 컨테이너로 전달하지 **않습니다**.

`config.yaml`에서 활성화:

```yaml
terminal:
  backend: docker
  docker_mount_cwd_to_workspace: true
```

활성화된 경우:
- `~/projects/my-app`에서 Hermes를 실행하면 해당 호스트 디렉토리가 `/workspace`에 바인드 마운트됩니다.
- Docker 백엔드는 `/workspace`에서 시작됩니다.
- 파일 도구와 터미널 명령이 모두 동일한 마운트된 프로젝트를 확인합니다.

비활성화된 경우 `docker_volumes`를 통해 명시적으로 무엇인가를 마운트하지 않는 한 `/workspace`는 샌드박스 소유로 유지됩니다.

보안 상의 절충안:
- `false`는 샌드박스 경계를 보존합니다.
- `true`는 샌드박스에 Hermes를 실행한 디렉토리에 대한 직접적인 액세스 권한을 부여합니다.

컨테이너가 실시간 호스트 파일에서 작동하도록 의도적으로 원할 때만 옵트인을 사용하세요.

### 영구 쉘 (Persistent Shell)

기본적으로 각 터미널 명령어는 자체 하위 프로세스에서 실행됩니다. 작업 디렉토리, 환경 변수 및 쉘 변수는 명령 간에 재설정됩니다. **영구 쉘(persistent shell)**이 활성화되면 단일 장기 실행 bash 프로세스가 `execute()` 호출 전반에 걸쳐 활성 상태로 유지되므로 상태가 명령어 간에 유지됩니다.

이 기능은 명령어당 연결 오버헤드도 제거하는 **SSH 백엔드**에서 가장 유용합니다. 영구 쉘은 **SSH의 경우 기본적으로 활성화되어 있으며** 로컬 백엔드의 경우 비활성화되어 있습니다.

```yaml
terminal:
  persistent_shell: true   # 기본값 — SSH용 영구 쉘 활성화
```

비활성화하려면:

```bash
hermes config set terminal.persistent_shell false
```

**명령어 간 유지되는 항목:**
- 작업 디렉토리 (`cd /tmp`는 다음 명령에서도 유지됨)
- 내보내기(Export)된 환경 변수 (`export FOO=bar`)
- 쉘 변수 (`MY_VAR=hello`)

**우선순위:**

| 레벨 | 변수 | 기본값 |
|-------|----------|---------|
| 구성 | `terminal.persistent_shell` | `true` |
| SSH 재정의 | `TERMINAL_SSH_PERSISTENT` | 구성을 따름 |
| 로컬 재정의 | `TERMINAL_LOCAL_PERSISTENT` | `false` |

백엔드별 환경 변수가 가장 높은 우선순위를 갖습니다. 로컬 백엔드에서도 영구 쉘을 원하는 경우:

```bash
export TERMINAL_LOCAL_PERSISTENT=true
```

:::note
`stdin_data`나 sudo가 필요한 명령어는 영구 쉘의 stdin이 이미 IPC 프로토콜에 점유되어 있으므로 자동으로 단발성(one-shot) 모드로 폴백됩니다.
:::

각 백엔드에 대한 자세한 내용은 [Code Execution](features/code-execution.md) 및 [README의 터미널 섹션](features/tools.md)을 참조하세요.

## 스킬 설정 (Skill Settings)

스킬은 SKILL.md 프런트매터를 통해 자체 구성 설정을 선언할 수 있습니다. 이것들은 비밀이 아닌 값들(경로, 환경 설정, 도메인 설정)로 `config.yaml` 내 `skills.config` 네임스페이스 아래에 저장됩니다.

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # 예시 — 각 스킬은 자체 키를 정의합니다.
```

**스킬 설정 작동 방식:**

- `hermes config migrate`는 활성화된 모든 스킬을 스캔하여 구성되지 않은 설정을 찾고 사용자에게 입력 프롬프트를 제안합니다.
- `hermes config show`는 모든 스킬 설정을 "Skill Settings" 아래에 속한 스킬과 함께 표시합니다.
- 스킬이 로드되면 해결된 구성 값이 스킬 컨텍스트에 자동으로 주입됩니다.

**수동으로 값 설정:**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

고유한 스킬에서 구성 설정을 선언하는 자세한 방법은 [스킬 생성 — 구성 설정 (Creating Skills — Config Settings)](/developer-guide/creating-skills#config-settings-configyaml)을 참조하세요.

### 에이전트가 만든 스킬 쓰기 보호 (Guard on agent-created skill writes)

에이전트가 스킬을 생성, 편집, 패치, 삭제하기 위해 `skill_manage`를 사용할 때 Hermes는 선택적으로 새로운/업데이트된 콘텐츠에 위험한 키워드 패턴(자격 증명 수집, 명백한 프롬프트 인젝션, 유출(exfil) 지침)이 있는지 검사할 수 있습니다. 검사기는 **기본적으로 꺼져 있습니다**. `~/.ssh/`를 건드리거나 `$OPENAI_API_KEY`를 합법적으로 언급하는 실제 에이전트 워크플로가 휴리스틱에 너무 자주 걸리기 때문입니다. 에이전트의 스킬 쓰기가 저장되기 전에 스캐너가 프롬프트로 확인을 요청하도록 하려면 스캐너를 다시 켜세요:

```yaml
skills:
  guard_agent_created: true   # 기본값: false
```

이 기능이 켜져 있으면 감지된 모든 `skill_manage` 쓰기가 스캐너의 근거와 함께 승인 프롬프트로 나타납니다. 승인된 쓰기는 저장되며, 거부된 쓰기는 에이전트에게 설명적인 오류를 반환합니다.

## 메모리 구성

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 토큰
  user_char_limit: 1375     # ~500 토큰
```

## 파일 읽기 안전장치

단일 `read_file` 호출이 반환할 수 있는 콘텐츠의 양을 제어합니다. 제한을 초과하는 읽기 요청은 더 작은 범위를 위해 `offset`과 `limit`을 사용하라는 오류와 함께 거부됩니다. 이렇게 하면 축소된 JS 번들이나 거대한 데이터 파일의 한 번의 읽기가 컨텍스트 창을 침수시키는 것을 방지할 수 있습니다.

```yaml
file_read_max_chars: 100000  # 기본값 — ~25-35K 토큰
```

컨텍스트 창이 큰 모델을 사용하고 큰 파일을 자주 읽는 경우 이를 높이세요. 읽기를 효율적으로 유지하려면 컨텍스트가 작은 모델의 경우 이를 낮추세요:

```yaml
# 대형 컨텍스트 모델 (200K+)
file_read_max_chars: 200000

# 소형 로컬 모델 (16K 컨텍스트)
file_read_max_chars: 30000
```

에이전트는 파일 읽기를 자동으로 중복 제거하기도 합니다. 동일한 파일 영역을 두 번 읽고 파일이 변경되지 않은 경우 내용을 다시 보내는 대신 가벼운 스터브(stub)가 반환됩니다. 컨텍스트 압축 시에는 이 내용이 요약되어 사라지므로 그 이후에 에이전트가 다시 파일을 읽을 수 있도록 이 기능이 초기화됩니다.

## 도구 출력 잘림 제한 (Tool Output Truncation Limits)

세 가지 관련된 제한 값이 Hermes가 도구를 자르기 전에 도구가 반환할 수 있는 원본 출력의 양을 제어합니다:

```yaml
tool_output:
  max_bytes: 50000        # 터미널 출력 제한 (문자 수)
  max_lines: 2000         # read_file 페이지 매기기 제한
  max_line_length: 2000   # read_file의 줄 번호 보기에서의 줄당 제한
```

- **`max_bytes`** — `terminal` 명령어가 결합된 stdout/stderr의 문자 수 이상을 생성할 때, Hermes는 처음 40%와 마지막 60%를 유지하고 그 사이에 `[OUTPUT TRUNCATED]` 알림을 삽입합니다. 기본값은 `50000`(일반적인 토크나이저에서 대략 12~15K 토큰)입니다.
- **`max_lines`** — 단일 `read_file` 호출의 `limit` 매개변수 상한입니다. 이를 초과하는 요청은 단일 읽기가 컨텍스트 창을 압도할 수 없도록 제한됩니다. 기본값은 `2000`입니다.
- **`max_line_length`** — `read_file`이 줄 번호 뷰를 표시할 때 적용되는 줄당 제한입니다. 이보다 긴 줄은 지정된 문자 수까지만 잘리고 그 뒤에 `... [truncated]`가 붙습니다. 기본값은 `2000`입니다.

호출당 더 많은 원시 출력을 감당할 수 있는 대용량 컨텍스트 창이 있는 모델에서는 제한을 늘리세요. 도구 결과를 압축해서 유지하려면 컨텍스트가 작은 모델의 제한을 낮추세요:

```yaml
# 대형 컨텍스트 모델 (200K+)
tool_output:
  max_bytes: 150000
  max_lines: 5000

# 소형 로컬 모델 (16K 컨텍스트)
tool_output:
  max_bytes: 20000
  max_lines: 500
```

## 글로벌 도구 세트 비활성화

CLI 및 모든 게이트웨이 플랫폼에서 특정 도구 세트를 한 곳에서 모두 억제하려면, `agent.disabled_toolsets` 아래에 그 이름들을 나열하세요:

```yaml
agent:
  disabled_toolsets:
    - memory       # 메모리 도구 + MEMORY_GUIDANCE 인젝션 숨김
    - web          # 모든 곳에서 web_search / web_extract 비활성화
```

이 설정은 (`hermes tools`에 의해 기록된 `platform_toolsets`인) 플랫폼별 도구 구성 **이후**에 적용되므로, 이곳에 나열된 도구 세트는 플랫폼의 저장된 구성에 아직 나열되어 있더라도 항상 제거됩니다. `hermes tools` UI에서 15개 이상의 플랫폼 행을 편집하는 대신 "모든 곳에서 X를 끕니다"라는 단일 스위치를 원할 때 이를 사용하세요.

목록을 비워두거나 키를 생략하면 아무 작업도 수행하지 않습니다(no-op).

## Git 워크트리 격리 (Git Worktree Isolation)

동일한 저장소에서 병렬로 여러 에이전트를 실행할 수 있도록 격리된 Git 워크트리(worktrees)를 활성화합니다:

```yaml
worktree: true    # 항상 워크트리를 생성합니다 (-w 옵션과 동일)
# worktree: false # 기본값 — -w 플래그가 전달되었을 때만
```

활성화되면 각 CLI 세션은 `.worktrees/` 디렉토리에 고유한 브랜치를 가진 새로운 워크트리를 만듭니다. 에이전트는 서로 간섭하지 않고 파일을 편집, 커밋, 푸시 및 PR을 생성할 수 있습니다. 깨끗한 워크트리는 종료 시 제거되며, 수정한 내역이 있는 워크트리는 수동 복구를 위해 유지됩니다.

저장소 루트에 있는 `.worktreeinclude`를 통해 워크트리에 복사할 git 무시(gitignored) 파일을 나열할 수도 있습니다:

```
# .worktreeinclude
.env
.venv/
node_modules/
```

## 컨텍스트 압축 (Context Compression)

Hermes는 모델의 컨텍스트 창(context window) 내에 머물기 위해 긴 대화를 자동으로 압축합니다. 압축 요약기(compression summarizer)는 별도의 LLM 호출입니다 — 이를 어떤 제공자나 엔드포인트를 지정하도록 설정할 수 있습니다.

모든 압축 설정은 `config.yaml`에 위치합니다 (환경 변수 없음).

### 전체 참조

```yaml
compression:
  enabled: true                                     # 압축 켜기/끄기
  threshold: 0.50                                   # 컨텍스트 제한의 이 비율에서 압축
  target_ratio: 0.20                                # 최근 후미 부분으로 유지할 임계값의 비율
  protect_last_n: 20                                # 압축되지 않은 상태로 유지할 최소 최근 메시지 수
  protect_first_n: 3                                # 압축 작업 동안 고정되는 (시스템이 아닌) 초기 메시지 (0 = 고정 안 함)
  hygiene_hard_message_limit: 400                   # 게이트웨이 안전 밸브 — 아래 참조

# 요약 모델/제공자는 auxiliary 아래에 구성됩니다:
auxiliary:
  compression:
    model: ""                                       # 비어 있음 = 주 채팅 모델 사용. 더 저렴/빠른 압축을 위해 예를 들어 "google/gemini-3-flash-preview"로 덮어씁니다.
    provider: "auto"                                # 제공자: "auto", "openrouter", "nous", "codex", "main" 등
    base_url: null                                  # 사용자 지정 OpenAI 호환 엔드포인트 (제공자를 덮어씀)
```

:::info 레거시 구성 마이그레이션
`compression.summary_model`, `compression.summary_provider` 및 `compression.summary_base_url`이 있는 이전 구성은 첫 번째 로드 시(구성 버전 17) `auxiliary.compression.*`로 자동 마이그레이션됩니다. 수동 작업이 필요하지 않습니다.
:::

`hygiene_hard_message_limit`는 게이트웨이 전용의 **압축 전 안전 밸브**입니다. 수천 개의 메시지가 있는 폭주하는 세션은 일반적인 컨텍스트 제한 비율에 도달하기 전에 모델 컨텍스트 한도를 초과할 수 있습니다. 메시지 수가 이 한계선을 넘으면 Hermes는 토큰 사용량에 관계없이 압축을 강제합니다. 기본값 `400` — 매우 긴 세션이 정상인 플랫폼에서는 이를 올리고, 더 공격적인 압축을 강제하려면 낮추세요. 실행 중인 게이트웨이에서 이 값을 편집하면 다음 메시지부터 적용됩니다(아래 참조).

`protect_first_n`은 매 압축(compaction) 시 고정(pinned)되는 **시스템 메시지가 아닌** 상단 메시지 수를 제어합니다. 기본값 `3` — 첫 번째 사용자/어시스턴트 교환이 요약기를 거친 후에도 그대로 유지되므로 원래 목표를 계속 볼 수 있습니다. 오프닝 턴이 더 이상 관련이 없는 장기 실행 롤링 압축 세션의 경우 `protect_first_n: 0`으로 설정하여 시스템 프롬프트 + 요약 + 후미 부분만 고정되도록 하세요. 시스템 프롬프트 자체는 이 설정과 상관없이 항상 유지됩니다.

:::tip 압축 및 컨텍스트 길이의 게이트웨이 핫-리로드
최근 릴리스부터는, 실행 중인 게이트웨이에서 `model.context_length`나 `config.yaml`의 모든 `compression.*` 키를 편집하면 다음 메시지에 즉시 적용됩니다 — 게이트웨이 재시작, `/reset`, 또는 세션 교체(rotation)가 필요하지 않습니다. 캐시된 에이전트 서명에는 이러한 키들이 포함되어 있으므로 게이트웨이는 변경 사항이 발견되면 투명하게 에이전트를 재빌드합니다. API 키와 도구/스킬 구성은 여전히 일반적인 다시 로드(reload) 경로가 필요합니다.
:::

### 일반적인 설정

**기본 (자동 감지) — 구성 필요 없음:**
```yaml
compression:
  enabled: true
  threshold: 0.50
```
주 제공자와 주 모델을 사용합니다. 주 채팅 모델보다 저렴한 모델에서 압축을 수행하려면 작업 단위로 재정의하세요 (예: `auxiliary.compression.provider: openrouter` + `model: google/gemini-2.5-flash`).

**특정 제공자 강제 지정** (OAuth 또는 API 키 기반):
```yaml
auxiliary:
  compression:
    provider: nous
    model: gemini-3-flash
```
모든 제공자와 작동합니다: `nous`, `openrouter`, `codex`, `anthropic`, `main` 등.

**사용자 지정 엔드포인트** (자체 호스팅, Ollama, zai, DeepSeek 등):
```yaml
auxiliary:
  compression:
    model: glm-4.7
    base_url: https://api.z.ai/api/coding/paas/v4
```
사용자 정의 OpenAI 호환 엔드포인트를 가리킵니다. 인증을 위해 `OPENAI_API_KEY`를 사용합니다.

### 세 가지 노브가 상호작용하는 방법

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | 결과 |
|---------------------|---------------------|--------|
| `auto` (기본값) | 설정되지 않음 | 사용 가능한 최상의 제공자 자동 감지 |
| `nous` / `openrouter` / 등 | 설정되지 않음 | 해당 제공자를 강제 지정하고 인증 사용 |
| 모든 항목 | 설정됨 | 제공자를 무시하고 사용자 지정 엔드포인트를 직접 사용 |

:::warning 요약 모델의 컨텍스트 길이 요구 사항
요약 모델은 **반드시** 메인 에이전트 모델의 컨텍스트 창(context window)과 같거나 그 이상 큰 컨텍스트 창을 가져야 합니다. 압축기(compressor)는 대화의 중간 전체 섹션을 요약 모델에 보냅니다. 만약 해당 모델의 컨텍스트 창이 메인 모델보다 작을 경우, 요약 호출은 컨텍스트 길이 에러(context length error)로 실패합니다. 이 현상이 발생할 경우, 중간 턴(turn)들이 **요약 없이 버려지며**, 대화 문맥을 조용히 상실하게 됩니다. 모델을 덮어쓰는(override) 경우, 해당 모델의 컨텍스트 길이가 메인 모델의 컨텍스트 길이를 충족하거나 초과하는지 반드시 확인하세요.
:::

## 컨텍스트 엔진 (Context Engine)

컨텍스트 엔진은 모델의 토큰 한도에 접근할 때 대화를 관리하는 방법을 제어합니다. 내장된 `compressor` 엔진은 손실성 요약(lossy summarization)을 사용합니다(자세한 내용은 [Context Compression](/developer-guide/context-compression-and-caching) 참조). 플러그인 엔진을 사용하면 이를 대체 전략으로 변경할 수 있습니다.

```yaml
context:
  engine: "compressor"    # 기본값 — 내장 손실성 요약
```

플러그인 엔진(예: 손실 없는 컨텍스트 관리를 위한 LCM)을 사용하려면:

```yaml
context:
  engine: "lcm"          # 플러그인 이름과 일치해야 함
```

플러그인 엔진은 **절대로 자동 활성화되지 않습니다**. 명시적으로 `context.engine`을 플러그인 이름으로 설정해야 합니다. 사용 가능한 엔진은 `hermes plugins` → Provider Plugins → Context Engine을 통해 찾아보고 선택할 수 있습니다.

메모리 플러그인에 대한 유사한 단일 선택(single-select) 시스템은 [메모리 제공자 (Memory Providers)](/user-guide/features/memory-providers)를 참조하세요.

## 반복 예산 압박 (Iteration Budget Pressure)

에이전트가 도구 호출이 많은 복잡한 작업을 수행할 때, 자신이 부족하다는 사실을 인지하지 못한 채 반복 예산(기본값: 90턴)을 소모할 수 있습니다. 예산 압박은 한도에 가까워짐에 따라 자동으로 모델에 경고합니다:

| 임계값 | 레벨 | 모델이 보는 내용 |
|-----------|-------|---------------------|
| **70%** | 주의(Caution) | `[BUDGET: 63/90. 27 iterations left. Start consolidating.]` |
| **90%** | 경고(Warning) | `[BUDGET WARNING: 81/90. Only 9 left. Respond NOW.]` |

경고는 별도의 메시지가 아닌 마지막 도구 결과의 JSON(`_budget_warning` 필드로)에 주입됩니다. 이는 프롬프트 캐싱을 보존하고 대화 구조를 방해하지 않습니다.

```yaml
agent:
  max_turns: 90                # 대화 턴당 최대 반복 횟수 (기본값: 90)
  api_max_retries: 3           # 폴백(fallback)이 관여하기 전 제공자별 재시도 (기본값: 3)
```

예산 압박은 기본적으로 활성화되어 있습니다. 에이전트는 도구 결과의 일부로 자연스럽게 경고를 인식하여 작업을 통합(consolidate)하고 반복 횟수가 다 떨어지기 전에 응답을 제공하도록 유도합니다.

반복 예산이 완전히 소진되면 CLI는 사용자에게 알림을 표시합니다: `⚠ Iteration budget reached (90/90) — response may be incomplete`. 활성 작업 중에 예산이 소진되면 에이전트는 멈추기 전에 성취된 내용의 요약을 생성합니다.

`agent.api_max_retries`는 폴백 제공자 전환(fallback-provider switching)이 참여하기 **전**에, 일시적인 오류(속도 제한, 연결 끊김, 5xx 에러)가 발생할 때 Hermes가 제공자 API 호출을 몇 번 재시도할지를 제어합니다. 기본값은 `3`입니다(총 4번의 시도). [폴백 제공자(fallback providers)](/user-guide/features/fallback-providers)가 구성되어 있고 오류를 더 빠르게 페일오버(fail over)하고 싶다면 이 값을 `0`으로 낮추십시오. 이렇게 하면 불안정한 엔드포인트에 대해 재시도를 반복하지 않고 주 제공자의 첫 일시적 오류 발생 즉시 폴백으로 넘어갑니다.

### API 타임아웃

Hermes는 스트리밍을 위한 별도의 타임아웃 레이어와 비-스트리밍(non-streaming) 호출을 위한 지연 감지기(stale detector)를 가지고 있습니다. 지연 감지기들은 암시적 기본값으로 남겨두었을 때에만 로컬 제공자에 대해 자동으로 조정됩니다.

| 타임아웃 | 기본값 | 로컬 제공자 | Config / env |
|---------|---------|----------------|--------------|
| 소켓 읽기 타임아웃 | 120s | 1800초로 자동 상향됨 | `HERMES_STREAM_READ_TIMEOUT` |
| 지연 스트림(Stale stream) 감지 | 180s | 자동 비활성화됨 | `HERMES_STREAM_STALE_TIMEOUT` |
| 지연 비-스트림(non-stream) 감지 | 300s | 기본 설정에서는 자동 비활성화됨 | `providers.<id>.stale_timeout_seconds` 또는 `HERMES_API_CALL_STALE_TIMEOUT` |
| API 호출 (비-스트리밍) | 1800s | 변경되지 않음 | `providers.<id>.request_timeout_seconds` / `timeout_seconds` 또는 `HERMES_API_TIMEOUT` |

**소켓 읽기 타임아웃(socket read timeout)**은 제공자로부터 데이터의 다음 청크(chunk)를 위해 httpx가 대기하는 시간을 제어합니다. 로컬 LLM은 대용량 컨텍스트 환경에서 첫 토큰을 생성하기 전 prefill(사전 채우기) 과정에 몇 분이 걸릴 수 있기 때문에, Hermes는 로컬 엔드포인트 감지 시 이 값을 30분으로 올립니다. 만약 명시적으로 `HERMES_STREAM_READ_TIMEOUT`을 지정했다면, 엔드포인트 종류와 상관없이 항상 그 값을 사용합니다.

**지연 스트림 감지기(stale stream detection)**는 SSE 생존 핑(keep-alive pings)은 받지만 실제 콘텐츠가 없는 연결을 죽입니다. 이 기능은 로컬 제공자의 경우 prefill 도중 생존 핑을 보내지 않기 때문에 완전히 비활성화됩니다.

**지연 비-스트림 감지기(stale non-stream detection)**는 너무 오래 응답을 생성하지 않는 비스트리밍 호출을 죽입니다. 기본적으로 Hermes는 긴 prefill 중 거짓 양성(false positives)을 피하기 위해 로컬 엔드포인트에서 이 기능을 비활성화합니다. 만약 명시적으로 `providers.<id>.stale_timeout_seconds`, `providers.<id>.models.<model>.stale_timeout_seconds`, 또는 `HERMES_API_CALL_STALE_TIMEOUT`을 설정했다면, 로컬 엔드포인트에서도 해당 명시적 값을 존중합니다.

## 컨텍스트 압박 경고 (Context Pressure Warnings)

반복 예산 압박과는 별개로, 컨텍스트 압박은 대화가 **압축 임계값(compaction threshold)**(이전 메시지를 요약하기 위해 컨텍스트 압축이 발생하는 시점)에 얼마나 가까운지를 추적합니다. 이는 대화가 얼마나 길어지고 있는지 사용자와 에이전트 모두가 이해하는 데 도움이 됩니다.

| 진행도 | 레벨 | 발생하는 일 |
|----------|-------|-------------|
| 임계값에 **≥ 60%** | 정보 | CLI는 청록색 진행률 표시줄을 표시합니다. 게이트웨이는 정보 알림을 보냅니다. |
| 임계값에 **≥ 85%** | 경고 | CLI는 굵은 노란색 막대를 표시합니다. 게이트웨이는 압축이 임박했다고 경고합니다. |

CLI에서 컨텍스트 압박은 도구 출력 피드의 진행률 표시줄(progress bar)로 나타납니다:

```
  ◐ context ████████████░░░░░░░░ 62% to compaction  48k threshold (50%) · approaching compaction
```

메시징 플랫폼에서는 일반 텍스트 알림이 전송됩니다:

```
◐ Context: ████████████░░░░░░░░ 62% to compaction (threshold: 50% of window).
```

자동 압축이 비활성화된 경우 경고는 컨텍스트가 잘릴 수 있음을 알려줍니다.

컨텍스트 압박은 자동입니다 — 별도의 구성이 필요하지 않습니다. 이것은 순수하게 사용자를 위한 알림으로만 작동하며 메시지 스트림을 수정하거나 모델의 컨텍스트에 어떤 것도 주입하지 않습니다.

## 자격 증명 풀 전략 (Credential Pool Strategies)

동일한 제공자에 대해 여러 API 키나 OAuth 토큰이 있는 경우 교체(rotation) 전략을 구성할 수 있습니다:

```yaml
credential_pool_strategies:
  openrouter: round_robin    # 여러 키를 고르게 순환
  anthropic: least_used      # 항상 가장 덜 사용된 키를 선택
```

옵션: `fill_first` (기본값), `round_robin`, `least_used`, `random`. 전체 문서는 [자격 증명 풀 (Credential Pools)](/user-guide/features/credential-pools)을 참조하세요.

## 프롬프트 캐싱 (Prompt caching)

Hermes는 활성 제공자가 프롬프트 캐싱을 지원하는 경우 교차 세션 프롬프트 캐싱을 자동으로 켭니다 — 사용자 구성은 필요 없습니다.

**native Anthropic**, **OpenRouter**, **Nous Portal**의 Claude에 대해 Hermes는 시스템 프롬프트 및 스킬 블록에 1시간 TTL(`ttl: "1h"`)을 적용한 `cache_control` 중단점(breakpoints)을 첨부합니다. 매 한 시간의 첫 번째 전송은 전체 입력 요금을 지불합니다; 동일한 시간 내에 다른 `hermes` 세션이나 파생된 하위 에이전트 간의 후속 전송은 할인된 캐시된 읽기 속도로 캐시를 가져옵니다. 즉, 시스템 프롬프트, 로드된 스킬 콘텐츠, 그리고 내용이 긴 include(포함 파일)의 앞부분이 첫 1시간 동안 `hermes` 세션과 여러 파생 하위 에이전트에서 재사용된다는 의미입니다.

Qwen Cloud (Alibaba DashScope) 업스트림은 캐시 TTL을 5분으로 제한하므로, Hermes는 5분 길이의 중단점 TTL을 사용합니다. 다른 타사(third-party) Claude 경로(AWS Bedrock, Azure Foundry)는 제공자 자체 캐싱 기본값으로 폴백합니다. xAI Grok은 별도의 세션 고정 대화-ID 메커니즘을 사용합니다 — [xAI 프롬프트 캐싱](/integrations/providers#xai-grok--responses-api--prompt-caching)을 확인하세요.

이를 비활성화하는 옵션은 없습니다. 캐싱은 항상 켜져 있으며 시스템 프롬프트 하나만으로도 입력 토큰 수의 의미 있는 비율을 차지하므로 한 턴(turn) 대화라도 비용을 절약합니다.

## 보조 모델 (Auxiliary Models)

Hermes는 이미지 분석, 웹 페이지 요약, 브라우저 스크린샷 분석, 세션 제목 생성 및 컨텍스트 압축과 같은 보조(side) 작업을 위해 "보조(auxiliary)" 모델을 사용합니다. 기본적으로 (`auxiliary.*.provider: "auto"`) Hermes는 모든 보조 작업을 **주 채팅 모델** (당신이 `hermes model`에서 선택한 동일한 제공자/모델)로 라우팅합니다. 시작하기 위해 아무것도 구성할 필요는 없지만, 값비싼 추론 모델(Opus, MiniMax M2.7 등)에서는 보조 작업이 의미 있는 비용을 추가한다는 점에 유의하세요. 주 모델과 관계없이 저렴하고 빠른 부가 작업을 원한다면 `auxiliary.<task>.provider` 및 `auxiliary.<task>.model`을 명시적으로 설정하세요 (예: 비전 및 웹 추출을 위해 OpenRouter의 Gemini Flash 사용).

:::note "auto"가 주 모델을 사용하는 이유
초기 빌드에서는 어그리게이터(OpenRouter, Nous Portal) 사용자를 저렴한 제공자 측 기본값으로 분할했습니다. 그러나 그것은 뜻밖의 문제였습니다 — 어그리게이터 구독 비용을 지불한 사용자가 다른 모델에서 보조 트래픽을 처리하는 것을 보게 될 것이기 때문입니다. 이제 `auto`는 모든 사람의 주 모델을 사용하며 `config.yaml`에서의 작업별 재정의가 여전히 우선합니다 (아래 [전체 보조 구성 참조](#전체-보조-구성-참조-full-auxiliary-config-reference) 확인).
:::

### 보조 모델 대화형으로 구성하기

YAML을 직접 편집하는 대신 `hermes model`을 실행하고 메뉴에서 **"Configure auxiliary models(보조 모델 구성)"**를 선택하세요. 대화형 작업별 선택기가 표시됩니다:

```
$ hermes model
→ Configure auxiliary models

[ ] vision               currently: auto / main model
[ ] web_extract          currently: auto / main model
[ ] title_generation     currently: openrouter / google/gemini-3-flash-preview
[ ] compression          currently: auto / main model
[ ] approval             currently: auto / main model
[ ] triage_specifier     currently: auto / main model
[ ] kanban_decomposer    currently: auto / main model
[ ] profile_describer    currently: auto / main model
```

작업을 선택하고, 제공자를 선택한 다음(OAuth 인증은 브라우저를 엽니다; API 키 제공자는 프롬프트를 띄웁니다), 모델을 선택하세요. 변경 사항은 `config.yaml`의 `auxiliary.<task>.*`에 유지됩니다. 주 모델 선택기와 동일한 기제(machinery)이며 — 추가로 배울 구문이 없습니다.

### 비디오 튜토리얼

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/NoF-YajElIM"
    title="Hermes Agent — Auxiliary Models Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

### 보편적 구성 패턴

Hermes의 모든 모델 슬롯 (보조 작업, 압축, 폴백)은 동일한 세 개의 노브를 사용합니다:

| 키 | 작동 방식 | 기본값 |
|-----|-------------|---------|
| `provider` | 인증 및 라우팅에 사용할 제공자 | `"auto"` |
| `model` | 요청할 모델 | 제공자의 기본값 |
| `base_url` | 사용자 정의 OpenAI 호환 엔드포인트 (제공자 덮어씀) | 설정 안 됨 |

`base_url`이 설정되면 Hermes는 제공자를 무시하고 해당 엔드포인트를 직접 호출합니다 (인증을 위해 `api_key` 또는 `OPENAI_API_KEY` 사용). `provider`만 설정된 경우 Hermes는 해당 제공자의 내장 인증 및 기본 URL을 사용합니다.

보조 작업을 위해 사용 가능한 제공자: `auto`, `main`, 그리고 [제공자 레지스트리](/reference/environment-variables) 내의 모든 제공자 — `openrouter`, `nous`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `gemini`, `google-gemini-cli`, `qwen-oauth`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `deepseek`, `nvidia`, `xai`, `xai-oauth`, `ollama-cloud`, `alibaba`, `bedrock`, `huggingface`, `arcee`, `xiaomi`, `kilocode`, `opencode-zen`, `opencode-go`, `azure-foundry` — 또는 `custom_providers` 리스트 내의 이름을 가진 사용자 정의 제공자 (예: `provider: "beans"`).

:::tip MiniMax OAuth
`minimax-oauth`는 브라우저 OAuth(API 키 필요 없음)를 통해 로그인합니다. `hermes model`을 실행하고 **MiniMax (OAuth)**를 선택하여 인증하세요. 보조 작업은 자동으로 `MiniMax-M2.7-highspeed`를 사용합니다. [MiniMax OAuth 가이드](../guides/minimax-oauth.md)를 확인하세요.
:::

:::tip xAI Grok OAuth
`xai-oauth`는 브라우저 OAuth(API 키 필요 없음)를 통해 SuperGrok 및 X Premium+ 구독자를 위해 로그인합니다. `hermes model`을 실행하고 **xAI Grok OAuth (SuperGrok / Premium+)**를 선택하여 인증하세요. 채팅, 보조 작업, TTS, 이미지 생성, 비디오 생성, 트랜스크립션 등 직접적인 xAI 작업에는 동일한 OAuth 토큰이 재사용됩니다. [xAI Grok OAuth 가이드](../guides/xai-grok-oauth.md)를 참조하고, Hermes가 원격 호스트에 있다면 [OAuth over SSH / Remote Hosts](../guides/oauth-over-ssh.md)를 확인하세요.
:::

:::warning `"main"`은 보조 작업에만 사용됩니다.
`"main"` 제공자 옵션은 "내 주 에이전트가 사용하는 어떤 제공자든 사용"을 의미합니다 — 이 값은 오직 `auxiliary:`, `compression:`, 그리고 기본 폴백 항목(`fallback_providers:` 또는 레거시 `fallback_model:`) 내에서만 유효합니다. 최상위 `model.provider` 설정의 값으로는 유효하지 **않습니다**. 커스텀 OpenAI 호환 엔드포인트를 사용할 경우 `model:` 섹션에서 `provider: custom`을 설정하십시오. 모든 주 모델 제공자 옵션은 [AI 제공자](/integrations/providers)를 확인하세요.
:::

### 전체 보조 구성 참조 (Full auxiliary config reference)

```yaml
auxiliary:
  # 이미지 분석 (vision_analyze 도구 + 브라우저 스크린샷)
  vision:
    provider: "auto"           # "auto", "openrouter", "nous", "codex", "main" 등
    model: ""                  # 예: "openai/gpt-4o", "google/gemini-2.5-flash"
    base_url: ""               # 커스텀 OpenAI 호환 엔드포인트 (provider 무시)
    api_key: ""                # base_url용 API 키 (OPENAI_API_KEY로 폴백됨)
    timeout: 120               # 초 — LLM API 호출 타임아웃; 비전 페이로드에는 넉넉한 타임아웃 필요
    download_timeout: 30       # 초 — 이미지 HTTP 다운로드; 연결이 느릴 시 증가

  # 웹 페이지 요약 + 브라우저 페이지 텍스트 추출
  web_extract:
    provider: "auto"
    model: ""                  # 예: "google/gemini-2.5-flash"
    base_url: ""
    api_key: ""
    timeout: 360               # 초 (6분) — 시도당 LLM 요약

  # 위험한 명령어 승인 분류기
  approval:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30                # 초

  # 컨텍스트 압축 타임아웃 (compression.* 구성과 별개)
  compression:
    timeout: 120               # 초 — 압축은 긴 대화를 요약하므로 더 많은 시간이 필요

  # 스킬 허브 — 스킬 매칭 및 검색
  skills_hub:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # MCP 도구 라우팅
  mcp:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # 칸반 분류기 구분자 (Kanban triage specifier) — `hermes kanban specify <id>` (또는
  # 대시보드의 Triage 열 카드에 있는 ✨ Specify 버튼)은
  # 한 줄 요약을 구체적인 사양으로 확장하고 작업을
  # `todo`로 승격시키기 위해 이 슬롯을 사용합니다. 저렴하고 빠른 모델이 여기에 적합합니다. 사양
  # 확장은 길이가 짧고 추론 깊이가 필요하지 않습니다.
  triage_specifier:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
```

:::tip
각 보조 작업은 설정 가능한 `timeout` (단위: 초)을 가집니다. 기본값: vision 120초, web_extract 360초, approval 30초, compression 120초. 보조 작업에 느린 로컬 모델을 사용하는 경우 이 수치를 늘리세요. Vision에는 HTTP 이미지 다운로드를 위한 별도의 `download_timeout`(기본값 30초)도 있습니다 — 인터넷 환경이 느리거나 자체 호스팅 이미지 서버일 때 이 값을 증가시키세요.
:::

:::info
컨텍스트 압축에는 임계값을 위한 자체 `compression:` 블록과 모델/제공자 설정을 위한 `auxiliary.compression:` 블록이 있습니다 — 위에 있는 [컨텍스트 압축 (Context Compression)](#컨텍스트-압축-context-compression)을 참조하세요. 주 폴백 체인은 최상위 `fallback_providers:` 리스트를 사용합니다 — [폴백 제공자 (Fallback Providers)](/integrations/providers#fallback-providers)를 참조하세요. 세 곳 모두 동일한 provider/model/base_url 패턴을 따릅니다.
:::

### 보조 작업에 대한 OpenRouter 라우팅 및 Pareto Code

보조 작업이 OpenRouter(명시적으로, 혹은 메인 에이전트가 OpenRouter인 상태에서 `provider: "main"`을 통해)로 지정될 때, 메인 에이전트의 `provider_routing`과 `openrouter.min_coding_score` 설정은 **전파되지 않습니다** — 의도적으로 각 보조 작업은 독립적입니다. 보조 작업에 대해 OpenRouter의 제공자 선호도를 지정하거나 [Pareto Code 라우터](/integrations/providers#openrouter-pareto-code-router)를 사용하려면, `extra_body`를 통해 작업별로 설정해야 합니다:

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openrouter/pareto-code         # 이 작업에 Pareto Code 라우터 사용
    extra_body:
      provider:                            # OpenRouter 제공자 라우팅 선호
        order: [anthropic, google]         # 이 순서로 제공자 시도
        sort: throughput                   # 또는 "price" | "latency"
        # only: [anthropic]                # 특정 제공자로만 제한
        # ignore: [deepinfra]              # 특정 제공자 무시
      plugins:                             # OpenRouter Pareto Code 라우터 제어
        - id: pareto-router
          min_coding_score: 0.5            # 0.0–1.0; 높을수록 강력한 코더
```

이 형태는 채팅 자동 완성 요청 본문에서 OpenRouter가 수용하는 것과 동일합니다. Hermes는 모든 `extra_body`를 그대로 전달하므로 [openrouter.ai/docs](https://openrouter.ai/docs)에 문서화된 다른 OpenRouter 요청 본문 필드도 동일하게 작동합니다.

### 비전(Vision) 모델 변경하기

이미지 분석을 위해 Gemini Flash 대신 GPT-4o를 사용하려면:

```yaml
auxiliary:
  vision:
    model: "openai/gpt-4o"
```

또는 환경 변수를 통해 (`~/.hermes/.env`에서):

```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
```

### 제공자(Provider) 옵션

이 옵션들은 귀하의 주 `model.provider` 설정이 아닌 **보조 작업 구성**(`auxiliary:`, `compression:`)과 기본 폴백 항목(`fallback_providers:` 또는 레거시 `fallback_model:`)에만 적용됩니다.

| 제공자 | 설명 | 요구 사항 |
|----------|-------------|-------------|
| `"auto"` | 이용 가능한 최고 (기본값). Vision은 OpenRouter → Nous → Codex 순으로 시도. | — |
| `"openrouter"` | 강제로 OpenRouter 사용 — 모든 모델(Gemini, GPT-4o, Claude 등)로 라우팅. | `OPENROUTER_API_KEY` |
| `"nous"` | 강제로 Nous Portal 사용. | `hermes auth` |
| `"codex"` | 강제로 Codex OAuth (ChatGPT 계정) 사용. 비전 지원 (gpt-5.3-codex). | `hermes model` → Codex |
| `"minimax-oauth"` | 강제로 MiniMax OAuth 사용 (브라우저 로그인, API 키 필요 없음). 보조 작업에는 MiniMax-M2.7-highspeed 사용. | `hermes model` → MiniMax (OAuth) |
| `"xai-oauth"` | 강제로 xAI Grok OAuth 사용 (SuperGrok 또는 X Premium+ 구독자용 브라우저 로그인, API 키 필요 없음). 동일한 OAuth 토큰을 채팅, TTS, 이미지, 동영상, 스크립트 작성에 사용. | `hermes model` → xAI Grok OAuth (SuperGrok / Premium+) |
| `"main"` | 활성 커스텀/메인 엔드포인트 사용. `OPENAI_BASE_URL` + `OPENAI_API_KEY`에서 비롯될 수 있거나, `hermes model` / `config.yaml`을 통해 저장된 사용자 정의 엔드포인트에서 비롯될 수 있습니다. OpenAI, 로컬 모델, 혹은 임의의 OpenAI-호환 API와 동작. **보조 작업에서만 사용 가능 — `model.provider`로 사용할 수 없습니다.** | 커스텀 엔드포인트 자격증명 + 기본 URL |

기본 라우터를 무시하고 보조 작업이 직접 모델을 호출하려 할 때 주 제공자 카탈로그의 API 기반 제공자도 이 설정에서 지원됩니다. `gmi`는 `GMI_API_KEY`가 구성된 뒤 유효합니다:

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

GMI의 보조 라우팅을 설정하려면 GMI의 `/v1/models` 엔드포인트가 반환하는 모델 ID를 그대로 사용하세요.

### 일반적인 설정

**직접 커스텀 엔드포인트 사용하기** (로컬/자체 호스팅 API에서 `provider: "main"`보다 명시적임):
```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url`은 `provider`보다 우선 순위를 가집니다. 그래서 이것은 특정 엔드포인트로 보조 작업을 라우팅하는 가장 확실한 방법입니다. 직접 엔드포인트를 오버라이드할 때, Hermes는 설정된 `api_key`를 사용하거나 `OPENAI_API_KEY`로 롤백합니다; `OPENROUTER_API_KEY`를 사용자 지정 엔드포인트에 다시 쓰지 않습니다.

**비전 작업에 OpenAI API 키 사용하기:**
```yaml
# ~/.hermes/.env 파일 내:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # 혹은 더 저렴한 "gpt-4o-mini"
```

**비전 작업에 OpenRouter 사용하기** (어떤 모델로든 라우팅):
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # 혹은 "google/gemini-2.5-flash" 등.
```

**Codex OAuth 사용하기** (ChatGPT Pro/Plus 계정 — API 키 불필요):
```yaml
auxiliary:
  vision:
    provider: "codex"     # ChatGPT OAuth 토큰을 사용합니다.
    # 모델은 기본적으로 비전을 지원하는 gpt-5.3-codex가 됩니다.
```

**MiniMax OAuth 사용하기** (브라우저 로그인, API 키 불필요):
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
이 과정을 자동으로 수행하려면 `hermes model`을 실행하여 **MiniMax (OAuth)**를 선택하여 로그인하세요. 중국 리전의 경우 base URL은 `https://api.minimaxi.com/anthropic`입니다. 과정에 대한 전체 설명은 [MiniMax OAuth 가이드](../guides/minimax-oauth.md)를 참조하십시오.

**로컬/자체 호스팅 모델 사용하기:**
```yaml
auxiliary:
  vision:
    provider: "main"      # 여러분의 주 사용자 정의 엔드포인트를 사용합니다.
    model: "my-local-model"
```

`provider: "main"`은 Hermes가 일반적인 채팅에서 사용하는 제공자를 그대로 사용합니다. (명명된 커스텀 제공자(예: `beans`), `openrouter`와 같은 내장 제공자, 혹은 구형 `OPENAI_BASE_URL` 엔드포인트 모두 해당)

:::tip
만약 여러분의 주 모델 제공자로 Codex OAuth를 사용한다면 비전 작업은 추가 설정 없이 자동으로 작동합니다. Codex는 비전 관련 자동-감지 대상(chain)에 포함되어 있습니다.
:::

:::warning
**비전에는 다중 모달(multimodal) 모델이 필요합니다.** `provider: "main"`으로 설정한 경우 여러분의 엔드포인트가 다중 모달/비전을 지원하는지 확인하십시오. 그렇지 않으면 이미지 분석에 실패하게 됩니다.
:::

### 환경 변수 (레거시)

보조 모델은 환경 변수를 통해서도 구성할 수 있습니다. 그러나 `config.yaml`의 사용이 권장됩니다 — 관리가 용이할뿐더러 `base_url`과 `api_key` 같은 모든 선택형 속성을 지원하기 때문입니다.

| 설정 | 환경 변수 |
|---------|---------------------|
| 비전 제공자 | `AUXILIARY_VISION_PROVIDER` |
| 비전 모델 | `AUXILIARY_VISION_MODEL` |
| 비전 엔드포인트 | `AUXILIARY_VISION_BASE_URL` |
| 비전 API 키 | `AUXILIARY_VISION_API_KEY` |
| 웹 추출 제공자 | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| 웹 추출 모델 | `AUXILIARY_WEB_EXTRACT_MODEL` |
| 웹 추출 엔드포인트 | `AUXILIARY_WEB_EXTRACT_BASE_URL` |
| 웹 추출 API 키 | `AUXILIARY_WEB_EXTRACT_API_KEY` |

압축과 폴백 모델 설정은 config.yaml 전용 옵션입니다.

:::tip
현재 보조 모델 설정을 확인하려면 `hermes config` 명령을 실행하세요. 덮어씌워진 설정(Overrides)은 오직 기본값과 다를 때만 노출됩니다.
:::

## 추론 노력 (Reasoning Effort)

응답 전에 모델이 "생각"하는 수준을 지정합니다:

```yaml
agent:
  reasoning_effort: ""   # 비워두면 = medium (기본값). 옵션: none, minimal, low, medium, high, xhigh (max)
```

지정되지 않은 경우(기본값) 추론 노력의 기본 수준은 "medium"이 됩니다. 대부분의 일반적인 작업에 적합한 균형잡힌 값입니다. 값을 변경하면 기존 값을 무시하며, 높은 추론 노력을 쓸수록 복잡한 문제에 대하여 긍정적인 성과를 내지만 더 긴 지연 시간과 높은 토큰 비용이 발생합니다.

런타임에 `/reasoning` 명령어로 추론 수준을 변경할 수도 있습니다:

```
/reasoning           # 현재 노력 수준 및 표시 상태 보기
/reasoning high      # 추론 노력을 높음으로 설정
/reasoning none      # 추론 노력 사용 안 함
/reasoning show      # 응답 위에 모델이 추론한 결과를 표시
/reasoning hide      # 모델 추론 결과 숨김
```

## 도구-사용 강제화 (Tool-Use Enforcement)

어떤 모델들은 도구 호출 대신 텍스트로 수행하려는 의도를 설명하는 경우가 있습니다 (예: 도구 호출이 아닌 "테스트를 실행합니다..." 같은 문구 작성). 도구 사용 강제화 기능은 모델이 실제로 도구를 사용할 수 있도록 유도하기 위해 지침을 시스템 프롬프트에 주입합니다.

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| 값 | 동작 |
|-------|----------|
| `"auto"` (기본값) | `gpt`, `codex`, `gemini`, `gemma`, `grok`에 일치하는 모델에 활성화. 나머지(Claude, DeepSeek, Qwen 등)에는 비활성화. |
| `true` | 모델 이름과 상관없이 항시 사용. 만약 기존 모델이 도구를 사용하지 않고 의도만 표현한다면 이 값을 설정. |
| `false` | 모델 이름과 상관없이 항상 비활성화. |
| `["gpt", "codex", "qwen", "llama"]` | 목록 내의 하위 문자열을 갖는 모델 이름에서만 허용. (대소문자 구별 안 함) |

### 무엇이 추가되나요?

사용이 활성화된 경우, 총 세 개의 안내 지침이 시스템 프롬프트에 추가될 수 있습니다:

1. **보편적 도구 사용 가이드** (일치하는 모든 모델) — 모델이 의도를 설명하지 말고 곧바로 도구를 호출하도록 가르치며 작업이 끝날 때까지 턴을 마치지 않게 돕습니다.

2. **OpenAI 환경용 실행 규칙** (GPT 및 Codex 모델만) — GPT만의 특수한 문제를 다룹니다: 미완의 부분적인 결과에 안주하여 작업을 종료하거나, 선행 조사를 빼먹거나, 도구 사용 대신 환각 작용에 빠지거나 검증 과정 없이 "완료"라고 선언하는 문제를 해결.

3. **Google 운영 안내** (Gemini 및 Gemma 모델만) — 간결성, 절대 경로, 병렬 호출, 검증 후 변경 패턴에 대해 학습.

이 기능은 오로지 시스템 프롬프트 내에만 주입되며, 사용자 입장에서는 투명합니다. 만약 (Claude와 같이) 도구를 안정적으로 사용하는 모델들은 굳이 이 지침이 필요하지 않으므로, `"auto"` 설정은 그들을 제외시킵니다.

### 기능을 활성화할 때

기본 자동(`"auto"`) 목록에 없는 모델을 사용하는 동안 "수행할 작업"을 직접 이행하지 않고 단순히 설명하기만 할 때, `tool_use_enforcement: true`로 지정하거나 리스트에 모델의 문자열을 덧붙이세요:

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## TTS 구성

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts"
  speed: 1.0                    # 전체 재생속도 배율 (모든 제공자에 대한 롤백)
  edge:
    voice: "en-US-AriaNeural"   # 322개의 음성 지원, 74개의 다국어 지원
    speed: 1.0                  # 재생 속도 배율 (백분율, 1.5 → +50%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    speed: 1.0                  # 속도 조정 계수 (API 특성상 0.25에서 4.0 사이로만 동작)
    base_url: "https://api.openai.com/v1"  # 호환되는 커스텀 OpenAI TTS용으로 재지정 가능
  minimax:
    speed: 1.0                  # 음성 속도 배율
    # base_url: ""              # 선택 사항: OpenAI 호환 TTS 엔드포인트 용도
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - 중성적/보통 목소리 (기본)
  gemini:
    model: "gemini-2.5-flash-preview-tts"   # 또는 gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30가지 음성: Zephyr, Puck, Kore, Enceladus 등.
  xai:
    voice_id: "eve"             # xAI TTS 음성
    language: "en"              # ISO 639-1 기반
    sample_rate: 24000
    bit_rate: 128000            # MP3 비트 레이트
    # base_url: "https://api.x.ai/v1"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

이는 `text_to_speech` 도구와 음성 모드 대답(`/voice tts` (CLI 또는 메시징 게이트웨이 내에서)) 양쪽을 모두 관할합니다.

**속도 옵션 계층 구조:** 제공자별 특정 속도(예: `tts.edge.speed`) → 글로벌 `tts.speed` → 기본값 `1.0`. 글로벌 값 `tts.speed`를 설정하면 모든 제공자에게 획일적인 재생 속도가 적용되며 개별 단위 옵션으로 제공자별로 세부 조정할 수 있습니다.

## 디스플레이 (Display Settings)

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # 메시징 게이트웨이에서 /verbose 슬래시 명령어 활성화
  platforms: {}           # 플랫폼별 디스플레이 재정의 (아래 항목 참고)
  tool_progress_overrides: {}  # 더 이상 사용 안 함 (DEPRECATED) — display.platforms를 사용할 것
  interim_assistant_messages: true  # 게이트웨이: 턴 도중 어시스턴트의 자연어 업데이트를 새 메시지로 분리 발송
  skin: default           # 내장된 또는 사용자 지정 CLI 스킨 (참조: user-guide/features/skins)
  personality: "kawaii"  # 레거시 장식용. 요약문에 이전에 쓰임
  compact: false          # 축소 뷰 출력 (공백 축소)
  resume_display: full    # full (이전 대화를 보여줌) | minimal (짧은 요약 한 줄로 제한)
  bell_on_complete: false # 답변 작성 완료시 터미널 벨 소리 활성화 (오래 걸리는 연산에 도움)
  show_reasoning: false   # 응답 상단에 추론 내용을 명기. 런타임시(/reasoning show|hide)와 같음
  streaming: false        # 응답이 오는대로 바로 실시간 표시
  show_cost: false        # CLI 하단 상태바에 예측 $ 비용 보여줌
  timestamps: false       # CLI / TUI에서 사용자와 에이전트의 턴이 나타날 때 [HH:MM] 꼴의 스탬프 명시
  tool_preview_length: 0  # 도구 실행 시 보여지는 커맨드의 허용 문자열 최대. 0=제한 없음(다 보여줌)
  runtime_footer:         # 게이트웨이: 응답 끝자락에 상태정보를 보여주기.
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  file_mutation_verifier: true    # 이번 턴 안에 실패한 write_file/patch 시도가 있었으면 확인 안내문을 하단에 추가.
  language: en            # 정적 UI 텍스트 출력의 번역. en | zh | zh-hant | ja | de | es | fr | tr | uk | af | ko | it | ga | pt | ru | hu
```

### 파일 변조 검증 (File-mutation verifier)

`display.file_mutation_verifier`가 `true`(기본값)로 설정되면 해당 턴 동안 `write_file` 또는 `patch` 호출이 실패하고 성공적인 시도로 뒤덮어쓰기 되지 않았다면 Hermes는 최종 응답 하단에 간단한 경고를 하나 출력합니다. 병렬 처리 중 실패한 쓰기를 모른 채 작업이 마쳤다 단정 짓는 모델의 허풍을 캐치하며 매번 `git status`를 쳐볼 귀찮음을 줄여줍니다.

바닥글 예시:

```
⚠️ File-mutation verifier: 3 file(s) were NOT modified this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm.
  • concepts/automatic-organization.md — [patch] Could not find match for old_string
  • concepts/lora.md — [patch] Could not find match for old_string
  • concepts/rag-pipeline.md — [patch] Could not find match for old_string
```

바닥글을 안 보려면 `file_mutation_verifier: false`(또는 `HERMES_FILE_MUTATION_VERIFIER=0`)로 지정하세요. 턴 마지막까지도 유효한 실패기록이 남을 때만 바닥글이 나타납니다. 턴 안에 에이전트가 재시도하여 수정에 성공한다면 해당 항목은 알람에 등재되지 않습니다.

### 정적 UI 번역용 설정 (UI language for static messages)

`display.language` 설정은 제한된 몇 가지 정적 메시지 묶음만 교체합니다 — CLI 상의 권한 승인 명령문, 특정 슬래시 커맨드의 반환 대답 따위들(e.g., restart 팁, approval 만기, 목표(goal) 클리어)입니다. 에이전트의 본질적 회답, 도구 로그, 도구들의 출력 텍스트 및 상세 에러, 슬래시 명령어 자체의 설명을 번역하지는 않습니다. 에이전트와 다른 언어로 대화하길 바라면 당신이 보내는 시스템 프롬프트 메시지에 이를 말해 주면 됩니다.

지원하는 옵션 값: `en` (영어), `zh` (중국어 간체), `ja` (일본어), `de` (독일어), `es` (스페인어), `fr` (프랑스어), `tr` (튀르키예어), `uk` (우크라이나어). 불명확하거나 지원하지 않는 옵션은 영어로 롤백됩니다.

세션 한정으로 환경변수 `HERMES_LANGUAGE`를 주입할 수도 있으며, 설정된 YAML 속성을 재정의합니다.

```yaml
display:
  language: zh   # 중국어로 CLI 질문 출력 (approval prompt)
```

| 모드 | 관측 대상 |
|------|-------------|
| `off` | 무음 — 오직 최후의 답변만 보여줌. |
| `new` | 도구가 교체될 시에만 로그 표출. |
| `all` | 매 도구 호출을 짧은 축약 본으로 전시. (기본값) |
| `verbose` | 인수, 출력 로그, 디버그 데이터의 전체 나열. |

CLI상에선 `/verbose` 명령으로 모드를 전환할 수 있습니다. Telegram 등 메시징 플랫폼에서 `/verbose`를 이용하기 원한다면 위에 기술된 `display` 메뉴에 `tool_progress_command: true`를 적으십시오.

### 실행 환경 상태 표시 바닥글 (Runtime-metadata footer (게이트웨이 전용))

만약 `display.runtime_footer.enabled: true` 이라면, Hermes는 메시징 플랫폼 턴의 **마지막** 응답 아래에 상태 정보 바닥글을 함께 표출해 줍니다 — CLI 하단 상태바(모델 이름, 문맥 %, 작업폴더, 처리 소요 시간, 토큰수, 비용 등)에서 보이는 것과 같은 정보입니다. 기본값은 비활성화 상태입니다. 만약 당신이나 당신의 워크 팀이 봇의 활동 제반 사항을 매 대화마다 체크하길 소망한다면 옵트-인(opt in) 하십시오.

```yaml
display:
  runtime_footer:
    enabled: true
    fields: ["model", "context_pct", "cwd"]   # 선택 가능 인자들: model, context_pct, cwd, duration, tokens, cost
```

모든 세션 내에서 슬래시 커맨드인 `/footer`를 활용해 표시 유무를 온/오프시킬 수도 있습니다.

Telegram/Discord/Slack 응답에 달린 바닥글의 예시:

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

한 턴의 **마지막** 회신 메시지에만 기입됩니다. 중간 턴 메시지들에는 노출되지 않습니다.

### 플랫폼별 디스플레이 재정의 (Per-platform progress overrides)

플랫폼마다 요구되는 정보량(verbosity)이 상이합니다. 가령, Signal의 경우 발송 메시지의 사후 편집을 지원하지 않기 때문에 매 프로그레스마다 계속 새로운 채팅 메시지를 보내 매우 소란해질 우려가 큽니다. 플랫폼별 특성에 맞추려면 `display.platforms`를 쓰세요:

```yaml
display:
  tool_progress: all          # 글로벌 환경 기본설정
  platforms:
    signal:
      tool_progress: 'off'    # Signal 환경에서의 업데이트 표출 제거
    telegram:
      tool_progress: verbose  # Telegram 환경에서의 장황한 업데이트 표출
    slack:
      tool_progress: 'off'    # 다수가 쓰는 공유 Slack 공간에서의 업데이트 표출 제거
```

여기 속하지 않은 플랫폼은 글로벌 환경 변수 `tool_progress`의 값을 승계받습니다. 유효한 옵션은: `telegram`, `discord`, `slack`, `signal`, `whatsapp`, `matrix`, `mattermost`, `email`, `sms`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`. 레거시 키워드인 `display.tool_progress_overrides`도 하위 호환성을 위해 지원되나 사용이 배제될 예정(deprecated)이며 첫 가동 시 `display.platforms`로 마이그레이션이 진행됩니다.

`interim_assistant_messages`는 게이트웨이 전용입니다. 이 값이 켜져 있을 때 Hermes가 턴 사이의 어시스턴트 말을 별개로 채팅창에 분리해 줍니다. 이것은 `tool_progress`와 무관하며 게이트웨이의 스트리밍 옵션을 요구하지 않습니다.

## 개인정보 보호 (Privacy)

```yaml
privacy:
  redact_pii: false  # LLM 컨텍스트에서 PII 삭제 (게이트웨이 전용)
```

`redact_pii`가 `true`일 경우, 지원되는 플랫폼에서 시스템 프롬프트가 LLM에 넘어가기 전 개인 식별 정보를 게이트웨이가 가립니다:

| 필드 | 조치 |
|-------|-----------|
| 휴대폰 번호 (WhatsApp/Signal 사용자 ID 등) | 해시 처리됨 (`user_<12자-sha256>`) |
| 사용자 ID | 해시 처리됨 (`user_<12자-sha256>`) |
| 채팅 ID | 고유 넘버 부분 해시, 플랫폼 식별자는 남김 (`telegram:<해시>`) |
| 홈 채널 ID | 고유 넘버 부분 해시 |
| 사용자 본명 / 가명 등 | **영향 없음** (본인이 정하고 모두에게 열려있으므로) |

**지원 대상:** WhatsApp, Signal, Telegram. Discord와 Slack은 자체 멘션 시스템(`<@user_id>`)이 원본 그대로의 시스템 ID를 사용해야만 돌아가기 때문에 예외 처리됩니다.

해시는 일관적(deterministic)입니다. 동일인은 언제나 같은 암호형으로 마킹되기에 그룹 톡 내에서도 모델은 각자 누가 누군지 정확하게 지목할 수 있습니다.

## Speech-to-Text (STT)

```yaml
stt:
  provider: "local"            # "local" | "groq" | "openai" | "mistral"
  local:
    model: "base"              # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe
  # model: "whisper-1"         # 여전히 존중받는 예전의 폴백 키
```

제공자 특성:

- `local`은 여러분의 로컬 시스템 안에 구축된 `faster-whisper`를 활용합니다. `pip install faster-whisper`로 별도 설치가 요구됩니다.
- `groq`는 Groq의 호환 엔드포인트를 쓰며 `GROQ_API_KEY`를 참고합니다.
- `openai`는 OpenAI의 API를 쓰며 `VOICE_TOOLS_OPENAI_KEY`를 봅니다.

기본적으로 요청 제공자가 막혔을 때, Hermes는 다음 순번으로 폴백을 시도합니다: `local` → `groq` → `openai`.

Groq와 OpenAI는 환경 변수를 통한 세부 오버라이드 지정이 됩니다:

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```

## 음성 모드 (Voice Mode (CLI용))

```yaml
voice:
  record_key: "ctrl+b"         # CLI용 녹음 단축 버튼
  max_recording_seconds: 120    # 너무 긴 오디오에 대처하기 위한 강제 정지 제한
  auto_tts: false               # /voice on 일 때, 로봇의 회답 음성 자동 재생을 킬 것인지 유무
  beep_enabled: true            # 녹음 전후로 확인용 짧은 삑 소리
  silence_threshold: 200        # 발화 종료 판단 한계점
  silence_duration: 3.0         # 말이 끝난 뒤 이 초(s)수 후에 레코딩 스탑
```

`/voice on` 을 치면 마이크가 켜지고 `record_key`에 따라 활성/비활성이 작동하며 `/voice tts` 로 목소리를 듣습니다. 상세 기능 및 기타 게이트웨이 플랫폼에서의 설정은 [Voice Mode](/user-guide/features/voice-mode) 페이지에 기술되어 있습니다.

## 스트리밍 (Streaming)

응답 전체가 한 번에 완성될 때까지 멍하니 대기하는 대신에 터미널 혹은 메시지 플랫폼에 전송되는 토큰을 즉시 수신해 실시간으로 송출합니다.

### CLI 스트리밍

```yaml
display:
  streaming: true         # 도착하는 토큰들을 곧장 터미널에 띄움
  show_reasoning: true    # 추론 내용도 함께 화면에 갱신 표시
```

이 옵션이 허용되면 진행 상자의 내부에서 토큰이 날아올 때마다 하나씩 출력물이 붙어 들어갑니다. (도구 호출은 원래와 같이 무음/조용히 발생). 만일 당신의 백엔드 제공자가 실시간 스트리밍 송출을 배제한다면, 조용하게 일반 응답으로 대처됩니다.

### 메시지 게이트웨이 (Telegram, Discord, Slack)

```yaml
streaming:
  enabled: true           # 메시지를 수시로 교체/편집하며 표시
  transport: edit         # "edit" (지속 교체형) 또는 "off"
  edit_interval: 0.3      # 메시지 변환 간격(초)
  buffer_threshold: 40    # 교체 발송이 작동될 단어수 경계치
  cursor: " ▉"            # 수신 중이라는 사실을 묘사할 커서 기호
  fresh_final_after_seconds: 60   # (Telegram) 너무 오래된 메시지를 교체하지 않고 새로 알림; 항상 기존 것을 수정하게 두려면 0
```

활성화되면, 첫 토큰이 전송되었을 때 곧장 메시지가 나타나고 그 메시지를 점진적으로 덮어씌워 갱신해나가는 방식으로 작동합니다. 그러나 만약 (Signal, Email, Home Assistant 처럼) 사후 교체를 원천 배제한 플랫폼일 경우 첫 토큰 전달이 실패할 시 자동으로 이를 간파하고 그 이후 세션부터 쏟아지는 문장들 없이 잠잠해집니다.

실시간으로 메시지 내용을 바꿔 끼우지 않고 턴 사이마다 중간 과정을 독립된 별개 톡으로 출력하게 하려면 `display.interim_assistant_messages: true` 를 사용하면 됩니다.

**용량 초과 분절:** 너무 문장이 길어져 플랫폼 고유의 메시지 규격 상한(약 4096문자)에 부딪히면 현재까지의 결과가 종료 처리되고, 그 뒤의 남은 내용은 다음 새로운 톡 박스로 분절 연결됩니다.

**새로운 알림 교체 (Telegram의 경우):** 텔레그램은 메시지가 갱신, 편집되더라도 애초 작성된 처음 시간을 그대로 유지합니다. 길고 긴 답변이 이뤄지다 보면 실제로는 방금 끝마쳤음에도 불구하고 한참 전(첫 톡이 이뤄진 시간)의 메시지로 보입니다. `fresh_final_after_seconds > 0`(기본값 `60`) 옵션을 부여하면 완성본을 아예 새로운 메시지로 다시 보냅니다. (이전 미리보기 메시지는 최선의 노력으로 삭제합니다) 완료 시간이 시각적으로 갱신됩니다. 짧은 요약/응답들은 변함 없이 인-플레이스 편집으로 덮어씁니다. `0`으로 두면 이런 신규 교체 작업 없이 항상 그 자리를 다시 덮어씁니다.

:::note
스트리밍은 초기에 비활성화되어 있습니다. 스트리밍 UX를 체감하려면 `~/.hermes/config.yaml` 에서 켜주세요.
:::

## 그룹 채널의 격리성 (Group Chat Session Isolation)

여러 사람이 공용으로 대화하는 방에서 모든 대화 내역을 하나의 방/객체에 귀속시킬지 혹은 매 참가자별로 방을 분리할지 설정:

```yaml
group_sessions_per_user: true  # true = 그룹방 내에서도 화자별 개별 격리, false = 방 안에선 통짜로 한 개의 공용 세션 사용
```

- `true`는 기본 권장 사항입니다. Discord 채널, Telegram 그룹, Slack 채널 따위의 여러 명이 모인 장소라면 플랫폼이 개별 ID를 제공할 때마다 각각 다른 세션으로 찢어 관리합니다.
- `false`는 오래전 방식입니다. 채널 전체의 채팅이 협업/공유를 위한 것이라 여길 때 쓰이지만 다중 유저의 대화 내용이 난입하고 비용 공유 및 간섭 현상이 발생합니다.
- 물론, 1:1 메시지는 여태껏 그래왔듯 DM 번호에 따라 개별 적용됩니다.
- 스레드 공간은 외부 본채널 공간과 단절 격리됩니다. `true`인 경우, 스레드 내부에서도 화자별로 분리됩니다.

추가적인 거동 사항들은 [Sessions](/user-guide/sessions)와 [Discord 가이드](/user-guide/messaging/discord)에 있습니다.

## 인증 안 된 DM 제어 (Unauthorized DM Behavior)

초대받지 않은 제3자가 Hermes 봇에게 직접 메시지를 걸 때 조치할 행동:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` 가 기본입니다. 대화를 거부하면서 1회용 등록용 코드(pairing code)를 그에게 안내해 줍니다.
- `ignore` 은 무단 사용자를 소리 없이 무시해버립니다.
- 플랫폼 파트(whatsapp 등)에서 기본 글로벌 옵션을 거부할 수 있으니 한 번에 허용하는 상황과 일부 기기를 통제하는 옵션을 혼용할 수 있습니다.

## 퀵 커맨드 (Quick Commands)

로컬의 시스템 명령을 가동하거나 다른 명령어에 단축/가명을 붙여 쓰는 용도입니다. LLM 소모 토큰이 발생하지 않고, 메시징 플랫폼 (Telegram, Discord) 등에서 간단히 핑 상태나 기기 조작을 지시할 때 뛰어납니다.

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  disk:
    type: exec
    command: df -h /
  update:
    type: exec
    command: cd ~/.hermes/hermes-agent && git pull && pip install -e .
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

이용 예: `/status`, `/disk`, `/update`, `/gpu`, `/restart` 등을 치면. `exec` 류는 로컬 PC에서 즉시 구동되고 화면으로 쏴줍니다 — LLM을 전혀 부르지 않으므로 토큰 비용 또한 지불하지 않습니다. `alias`류는 목표물이었던 명령어들을 바꿔서 부릅니다.

- **30초 타임아웃** — 오래된 명령들은 끊어버리고 에러로 간주
- **우선순위** — 스킬 커맨드보다도 제일 먼저 찾아보게 되므로 같은 이름의 스킬들을 장악할 수도 있음
- **자동완성** — 퀵 커맨드들은 디스패치 타임에 해소되므로, 슬래시 커맨드의 추천 리스트 창에 들어있지는 않습니다.
- **분류 타입** — `exec`와 `alias`를 지원하며 기타 입력은 버그를 일으킵니다.
- **모든 곳에서 지원** — CLI, Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant.

단순 텍스트 매크로는 이 기능에서 쓰일 수 없습니다. 장황한 반복 작업을 위해서는 스킬이나 슬래시 명령어 별칭으로 지정하세요.

## 인간적인 딜레이 (Human Delay)

채팅 플랫폼 상에서 진짜 사람이 입력하는 듯한 속도감을 선사:

```yaml
human_delay:
  mode: "off"                  # off | natural | custom
  min_ms: 800                  # 가장 짧은 대기 허용 (custom 모드 한정)
  max_ms: 2500                 # 가장 늦은 대기 허용 (custom 모드 한정)
```

## 코드 실행 (Code Execution)

`execute_code` 도구를 관리:

```yaml
code_execution:
  mode: project                # project (기본) | strict
  timeout: 300                 # 최대 제한 시간 (초)
  max_tool_calls: 50           # 코드 내에서의 허용 도구 호출 제한 수
```

**`mode`** 는 실행 중의 파이썬 인터프리터 주소와 작업 공간 디렉토리를 지휘합니다:

- **`project`** (기본) — 현재 활성화 중인 가상환경(venv/conda 등) 인터프리터를 써서 대화 내에 지목된 디렉토리에서 진행됩니다. 소속 패키지(`pandas`, `torch` 등)와 타겟이 되는 프로젝트의 상대 경로 파일(`.env`, `./data.csv`) 들을 자연스레 `terminal()`의 실행 궤와 동일하게 이어받습니다.
- **`strict`** — 외부 임시 공간 내에서 `sys.executable`(Hermes 본연의 내장 경로)만을 가동시킵니다. 높은 실행 순도성을 가져오지만 외부 환경과 절대 경로가 매치되지 않습니다.

환경 변수 스크러빙 (각종 `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, `*_CREDENTIAL`, `*_PASSWD`, `*_AUTH` 따위의 노출 차단) 및 도구 허용 화이트리스트는 양 측에서 모두 안전하게 기능합니다 — 어느 쪽 모드이건 보안 강도는 동일합니다.

## 웹 서치 백엔드 설정 (Web Search Backends)

`web_search` 및 `web_extract` 도구는 다섯 가지의 외부 제공자를 허용합니다. `config.yaml` 안이나 `hermes tools` 명령을 써서 조절하세요:

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | exa

  # 또는 각 개별 목적 단위로 따로따로 할당할 수도 있습니다:
  search_backend: "searxng"
  extract_backend: "firecrawl"
```

| 백엔드 | 환경 변수 명 | Search (검색) | Extract (추출) |
|---------|---------|--------|---------|
| **Firecrawl** (기본값) | `FIRECRAWL_API_KEY` | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ |

**백엔드 분류/배분 조건:** 만약 `web.backend` 변수가 비워져있다면 등록되어 있는 각 API 키를 보고 자율적으로 가려냅니다. 오로지 `SEARXNG_URL` 혼자 존재하면 SearXNG가 구동됩니다. 반면 `EXA_API_KEY`만 있다면 Exa, `TAVILY_API_KEY`만 있으면 Tavily, `PARALLEL_API_KEY`라면 Parallel, 그 무엇도 매칭되지 않는다면 Firecrawl을 고릅니다.

**SearXNG** 는 70여 개 검색 엔진을 포괄하는 무료이면서, 본인이 직접 호스팅하고, 사생활 추적을 차단하는 메타 검색 장비입니다. API 키가 전혀 필요 없으며, 보유 중인 주소 인스턴스(예: `http://localhost:8080`)만 `SEARXNG_URL`에 박아주시면 됩니다. 단점은 오로지 '서치'만 해낸다는 것이며 `web_extract` 기능은 별도로 (`web.extract_backend` 로 배분) 분리 배당해 줘야 합니다. Docker 설치에 관한 사항은 [웹서치 구축 지침서](/user-guide/features/web-search)를 확인하세요.

**직접 호스팅하는 Firecrawl:** 여러분 본인 인스턴스 주소를 `FIRECRAWL_API_URL`으로 맞춰놓으세요. 커스텀 지정이 이뤄졌다면 API 키는 선택형이 됩니다. (참고: 서버 파트에서 인증 절차를 피하려면 `USE_DB_AUTHENTICATION=***` 세팅)

**Parallel 탐색 모드 조절:** `PARALLEL_SEARCH_MODE`를 변경하면 검색의 깊이가 달라집니다 — `fast`, `one-shot`, 혹은 `agentic` (기본: `agentic`).

**Exa:** `~/.hermes/.env` 속에서 `EXA_API_KEY`를 넣고 사용합니다. 카테고리별 필터(`company`, `research paper`, `news`, `people`, `personal site`, `pdf`) 기능 및 도메인/일자 기반 소거 기능이 특장점입니다.

## 브라우저 모드 (Browser)

자동 웹 브라우징 탐색 환경 지침:

```yaml
browser:
  inactivity_timeout: 120        # 사용 없는 정지된 세션을 자동 종료(초)
  command_timeout: 30             # 브라우저 명령(스크린샷, 경로 이동 등) 대기 초
  record_sessions: false         # 웹 이동 행적을 WebM 비디오화하여 ~/.hermes/browser_recordings/ 에 저장
  # CDP 재지정 (옵션 사항) — 부여될 시 헤드리스 백엔드 말고 당신이 켜둔
  # Chromium 파생 브라우저(로컬 접속. /browser connect)로 직통 꽂습니다.
  cdp_url: ""
  # 다이얼로그 관리 — 기본 CDP 환경(Browserbase나 로컬 Chromium 계열
  # 의 /browser connect 구동)에서 마주치는 알럿창/컨펌창/프롬프트 따위를 어찌 통과할지 제어.
  # 만약 Camofox와 같은 기본 로컬 에이전트 모드에서는 무시됩니다.
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # must_respond 옵션의 응답 실패 유효 만기시간 (초)
  camofox:
    managed_persistence: false   # 참인 경우 재시작에도 Camofox 세션이 쿠키/로그인 정보를 기억함
    user_id: ""                  # 선택 사항: 외부에서 지정하는 Camofox의 userId
    session_key: ""              # 선택 사항: Hermes가 브라우저 탭 생성 시 첨부하는 특정 키
    adopt_existing_tab: false    # 새 탭을 만들지 않고 기존에 있는 탭을 재활용함
```

**다이얼로그 돌파 방법:**

- `must_respond` (기본 설정) — 팝업 창을 마주친 사실을 적발한 뒤 `browser_snapshot.pending_dialogs` 안에 위치시킵니다. 그 후 에이전트가 `browser_dialog(action=...)` 명령으로 스스로 직접 해결할 때까지 기다립니다. 만약 `dialog_timeout_s` 초과 시까지 무시된 상황이라면 스레드가 영구 스턴/마비되는 대참사를 방어하고자 그 창을 무시(auto-dismissed)시켜버립니다.
- `auto_dismiss` — 창이 등장하자마자 그냥 거절(dismiss)로 무마시킵니다. 대신에 사후 `browser_snapshot.recent_dialogs` 내부의 로그에서 `closed_by="auto_policy"` 명목으로 그 창을 본 적이 있었다고 알려는 줍니다.
- `auto_accept` — 등장 즉시 수락/허용(accept). 창이 넘어가며 공격적인 `beforeunload` 프롬프트들을 통과하는 데 이롭습니다.

기타 다이얼로그 흐름 파악을 원하면 [브라우저 기능 페이지](./features/browser.md#browser_dialog) 항목을 살피세요.

브라우저 환경은 다수 제공자 지원 구조입니다. Browserbase, Browser Use 그리고 로컬 환경의 Chromium CDP 세팅 안내 등은 [브라우저 설명란](/user-guide/features/browser)에 나열되어 있습니다.

## 타임존 (Timezone)

서버 로컬 지역 설정을 무시하고 강제로 IANA 양식으로 타임존을 치환합니다. 로그 찍기, 크론 작업, 시스템 프롬프트의 시간 주입 값에 영향을 줍니다.

```yaml
timezone: "America/New_York"   # IANA 타임존 (기본: "" = 서버 내부 시간)
```

어떤 IANA 아이디 값이든 넣을 수 있습니다. (예: `America/New_York`, `Europe/London`, `Asia/Kolkata`, `UTC`). 빈칸이거나 없으면 원래 서버 시간.

## Discord

메시지 게이트웨이 내의 Discord-전용 설정:

```yaml
discord:
  require_mention: true          # 서버 오픈 채널에서는 @mention 을 받아야만 답변
  free_response_channels: ""     # @mention 없이도 답변을 할 특혜 채널 ID(쉼표로 다수 작성)
  auto_thread: true              # 멘션을 받을 시 자동으로 하위 스레드를 개설함
```

- `require_mention` — `true`일 경우(기본), 봇이 오픈된 채널 내부에서는 `@BotName`으로 자신을 불렀을 때만 답변을 시작합니다. DM 에서는 이 규칙 없이 항시 답변.
- `free_response_channels` — 채널 내의 ID들을 쉼표로 나열한 곳. 이 안에서는 호출이 없더라도 알아서 답변.
- `auto_thread` — `true`일 경우(기본), 대화 채널 환경에서 봇 호출이 발생하면 그 방을 소란스럽게 만들지 않기 위해 자동 분기되어 파생 스레드로 입장합니다. (Slack 스레딩과 닮음).

## 보안 및 통제 (Security)

실행하기 전의 사전 스캔, 정보 차단 등:

```yaml
security:
  redact_secrets: true           # 터미널 출력 및 로그에서 API 키 패턴 등을 덮어 가림 (기본으로 켜짐)
  tirith_enabled: true           # 터미널 명령어 보안 스캐너 Tirith 작동
  tirith_path: "tirith"          # tirith 위치 경로 지정 (기본값: $PATH의 "tirith")
  tirith_timeout: 5              # tirith 답변 대기 한계 시간(초)
  tirith_fail_open: true         # tirith가 반응 없거나 맛이 간 상태여도 터미널 명령을 승인하고 통과시킬지.
  website_blocklist:             # 웹사이트 블랙리스트 안내서 (아래 설명 참조)
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets` — `true` 일 때 API, 토큰 등 기밀 사항으로 보이는 데이터 패턴이 대화록과 시스템 로그로 진입하기 전 도구 단계에서 미리 덮어 지워버립니다. **기본값은 참(on)입니다**. 자신이 특별히 에러 수리를 위해 날것의 문자열 패턴을 봐야 할 때만 임시로 `false` 하십시오.
- `tirith_enabled` — `true` 이면 터미널 내부로 실행 코드가 가동되기 직전 파괴적이고 심각한 지시가 없는지 [Tirith](https://github.com/sheeki03/tirith)가 사전 조사합니다.
- `tirith_path` — tirith 실행 경로. 표준적이지 않은 위치에 있다면 지정하십시오.
- `tirith_timeout` — tirith 검사 기다림 한도(초). 초과 시 다음 명령어로 속행.
- `tirith_fail_open` — `true`(기본)일 때, tirith 자체를 쓸 수 없거나 불능이라도 실행 코드를 허용. 반면 `false`일 때는 tirith의 보안 검사 허가가 무조건 떨어져야만 실행되도록 통제.

## 웹사이트 블랙리스트 차단 (Website Blocklist)

지정된 웹사이트에 대해서는 에이전트의 웹 검색 혹은 브라우저 도구가 무효화되도록 금지합니다:

```yaml
security:
  website_blocklist:
    enabled: false               # 차단기 스위치 온/오프 (기본: false)
    domains:                     # 금지 패턴 등록부
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:                # 따로 관리하는 외부 문서 파일 연결부
      - "/etc/hermes/blocked-sites.txt"
```

활성화 시, 이 금지된 웹 도메인과 부합하는 URL을 띄우는 순간 웹/브라우저 도구는 실행하기도 전 차단 및 폐기됩니다. 이는 `web_search`, `web_extract`, `browser_navigate` 등을 아울러 모든 URL 도구에 작용합니다.

도메인 조건 규칙 지원:
- 완벽한 일치: `admin.example.com`
- 하위 도메인 전체 (와일드카드): `*.internal.company.com` (내부망 전부 막기)
- TLD 전체: `*.local`

연결된 `shared_files` 외부 문서들도 한 줄에 하나의 금지 도메인을 갖춰야 합니다. (빈 칸이나 `#` 주석은 통과됨). 만약 그 문서를 찾지 못하거나 못 읽는 상황이 와도 경고만 띄울 뿐이지 웹 도구를 정지시키진 않습니다.

보안 규칙 변경 정책은 30초마다 갱신(cache)되므로 프로그램 재시작 없이도 비교적 신속히 적용됩니다.

## 스마트 승인 대기 (Smart Approvals)

위험성이 도사린 파괴적 명령어를 Hermes가 어떻게 취급할지 배분합니다:

```yaml
approvals:
  mode: manual   # manual | smart | off
```

| 모드 | 작동 |
|------|----------|
| `manual` (기본값) | 위험 판정된 커맨드를 실행하기 전 사용자에게 허가를 구합니다. CLI에선 직접 대화형 승인 창이 뜹니다. 원격 메시지에선 보류 대기 명령이 떨어집니다. |
| `smart` | 그 위험 커맨드가 실질적으로 큰 위협인지 한 번 더 보조 LLM에게 판별받습니다. 낮고 가벼운 위험으로 추려지면 세션 단위 기록과 함께 무인 자동 승인됩니다. 허나 정말 심각한 위험이라 판가름 나면 다시 사용자에게 허가를 구하기 위해 전가합니다. |
| `off` | 일체의 묻지도 따지지도 않고 터미널 허가를 통과합니다. `HERMES_YOLO_MODE=true` 와 같은 뜻. **크게 경계하고 사용할 것.** |

Smart 모드는 쉴 새 없는 승인 알림으로 인한 사용자의 피로감을 낮추는 데 지대합니다. 에이전트가 가벼운 작업은 능수능란하게 헤쳐 나가는 자율성을 주면서도 정말 시스템이 망가지는 커맨드는 확실하게 잡아냅니다.

:::warning
`approvals.mode: off`는 명령어 실행의 모든 보안 벽을 걷어냅니다. 이는 철저히 믿을 수 있거나 파괴되어도 무방한 샌드박스에서만 쓰기 바랍니다.
:::

## 체크포인트 백업 복구 (Checkpoints)

파괴적인 작업 이전에 자동으로 백업 스냅샷을 땁니다. 세부 사항은 [체크포인트 및 롤백 가이드](/user-guide/checkpoints-and-rollback)에.

```yaml
checkpoints:
  enabled: false                 # 자동 생성 옵션 (혹은 켜기: hermes chat --checkpoints). 기본: false.
  max_snapshots: 20              # 디렉토리 당 백업할 수 있는 스냅샷 총 개수 상한선 (기본: 20)
```

## 하위 에이전트 분업 및 위임 (Delegation)

위임 도구를 위해 하위 에이전트의 특성을 부여합니다:

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # 모델 교체 오버라이드 (공백= 부모 상속)
  # provider: "openrouter"                  # 제공자 교체 (공백= 부모 상속)
  # base_url: "http://localhost:1234/v1"    # 직통 OpenAI-호환 엔드포인트 주소 (위 제공자 세팅보다 우위)
  # api_key: "local-key"                    # 직통 base_url 을 위한 키 (OPENAI_API_KEY 로 폴백 받음)
  # api_mode: ""                            # 엔드포인트에서 쓸 통신 구조 방식: "chat_completions", "codex_responses" 혹은 "anthropic_messages". 빈칸(기본) = URL에서 추론.
  max_concurrent_children: 3                # 병렬 동시처리 배당 수 (최소 1, 끝없음). 변수 DELEGATION_MAX_CONCURRENT_CHILDREN로도 지정됨.
  max_spawn_depth: 1                        # 위임 분기 구조의 깊이 제한 (최소 1, 끝없음). 1 = 평면적 1대1 (기본): 자식은 더 이상 타인에게 하달 불가. 2 = 자식이 또 다른 손자 에이전트에게 하달 가능. 3+ = 그 아래로 깊이 허용.
  orchestrator_enabled: true                # 분업 총괄 스위치 오프. false일 경우 위임받는 모든 자식 에이전트는 무조건 단순 작업자(leaf)로서 굴려지고 더 이상의 분업 분기(depth)를 생성할 수 없음.
```

**하위 에이전트의 provider:model 오버라이딩:** 기본적으로 하위 에이전트는 본체 에이전트의 제공자와 모델을 상속받습니다. 만약 좁은 범위에 그치는 자잘한 소작업에는 값싸고 빠른 모델로, 본체에는 비싸고 섬세한 모델을 쓸 생각이라면 이 `delegation.provider`와 `delegation.model`로 궤도를 트세요.

**커스텀 엔드포인트 통제:** 아예 외부 엔드포인트로 노선을 타겠다면, `delegation.base_url`, `delegation.api_key`, `delegation.model`로 조율하세요. 이 값들이 `delegation.provider` 설정 위에 군림하게 됩니다. `delegation.api_key`가 배제되면 오로지 `OPENAI_API_KEY` 값만 의존하게 됩니다.

**통신 규약/프로토콜 (`api_mode`):** Hermes는 알아서 `delegation.base_url`의 값만 가지고 통신 프로토콜을 추론합니다 (예: `/anthropic` → `anthropic_messages` 도출; Codex / native Anthropic / Kimi-coding 같은 호스트네임도 원래 하던 대로 탐지함). 하지만 예측이 힘든 곳 — 예를 들어 Azure AI Foundry, MiniMax, Zhipu GLM, 혹은 LiteLLM 처럼 속은 Anthropic-계열의 형태면서 앞단은 Proxy 처리를 해 둔 곳들의 경우 — 직접 `delegation.api_mode`에다가 `chat_completions`나 `codex_responses`, `anthropic_messages` 중 하나를 집어넣으세요. 빈 공간(기본)으로 두면 그냥 알아서 추론합니다.

하위 위임자들도 기존과 똑같은 신원 증명 경로를 탑니다. `openrouter`, `nous`, `copilot`, `zai`, `kimi-coding`, `minimax`, `minimax-cn` 등 기존 제공자를 모두 수용합니다. 제공자가 맞춰지면 그에 적법한 베이스 URL, 키, 통신 규약까지 모든 연결점이 완성되므로 추가로 수동 세팅할 곳이 없습니다.

**우선순위 역학 관계:** (가장 강력) 설정 내 `delegation.base_url` → 설정 내 `delegation.provider` → 본체의 제공자(상속). 그 다음, 설정 내 `delegation.model` → 본체의 모델(상속). 단, `provider` 건너뛰고 `model`만 교체했다면 부모와 같은 제공자 내에서 모델 체급만 바꿔치는 역할(예: OpenRouter 내에서 급만 바꾸기)을 합니다.

**너비와 깊이 제한선:** `max_concurrent_children` (기본값 `3`, 최소 1, 무한정)는 동시 다발 처리(배치 런)를 몇 마리까지 수용할 건지를 정합니다. 또한 `DELEGATION_MAX_CONCURRENT_CHILDREN` 변수로도 제어됩니다. 이 한도량을 넘어선 작업 배분이 모델에서 발생하면 조용히 뒷열을 자르지 않고, 오히려 작업이 거부되며 에러를 고지받게 됩니다. `max_spawn_depth`는 하부 명령 트리 구조가 얼마나 깊어질지 제어합니다 (최소 1, 무한정). 기본값인 `1`에선 모두가 1대 1 분배 방식이며: 자식들은 그 하위 자식(손자)을 낳을 권한이 박탈된 채, 평직원(`leaf`)으로 취급되어 `role="orchestrator"` 속성이 부여되어도 조용히 무시됩니다. 값을 `2`로 올리면 부모로부터 위임받은 자식이 `orchestrator` 속성을 부여받고 또다시 아랫사람(손자)에게 일거리를 패스(하달)할 수 있습니다. 3단계면 3, 더 깊은 단계는 더 늘려주면 됩니다. 모든 하부 배치는 비용 배수에 합산됩니다 — 만일 `max_spawn_depth: 3` & `max_concurrent_children: 3`이면 평직원(`leaf`) 27명(3×3×3)의 동시다발 토큰 요금을 소모할지도 모릅니다. 분배 및 파생 활용안에 관한 심도 깊은 이야기는 [Subagent Delegation → Depth Limit and Nested Orchestration](features/delegation.md#depth-limit-and-nested-orchestration)를 참고하세요.

## 보완 질문 타이머 (Clarify)

질문 응답에 있어 시간 연장 통제:

```yaml
clarify:
  timeout: 120                 # 명확화를 위해 사용자 답변을 대기하는 허용 한계 초
```

## 작업 환경 내 규칙 파일 참조 (Context Files: SOUL.md, AGENTS.md)

Hermes는 외부 규칙 탐색에 크게 2가지 범위를 탐색합니다:

| 문서 | 목적성 | 탐색 범위 |
|------|---------|-------|
| `SOUL.md` | **에이전트의 영혼(성격)** — 봇 자체가 누구인지(시스템 프롬프트의 1순위) | `~/.hermes/SOUL.md` 혹은 `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | 당해 프로젝트의 최상위 수칙 (절대적 우선권) | 파일의 최상위 git 루트로 역주행 |
| `AGENTS.md` | 프로젝트 지침, 코딩 컨벤션 약속 | 재귀적으로 하위 디렉토리 파생 탑색 |
| `CLAUDE.md` | (Claude Code의 유산. 같이 읽어드림) | 지금 서 있는 작업 디렉토리 한정 |
| `.cursorrules` | (Cursor IDE 룰 파일. 같이 읽어드림) | 지금 서 있는 작업 디렉토리 한정 |
| `.cursor/rules/*.mdc` | (Cursor 룰 파일들. 같이 읽어드림) | 지금 서 있는 작업 디렉토리 한정 |

- **SOUL.md**는 이 봇의 가장 주체적인 자아입니다. 시스템 프롬프트의 무조건 1순위 슬롯에 위치하여 내장된 기본 정체성을 짓밟고 완전히 새 자아를 구가합니다. 에이전트를 바꾸고 싶다면 이것을 수정하십시오.
- 만일 SOUL.md를 못 찾거나 내용이 비었다면, 기본으로 하드코딩된 원론적 정체성을 복구합니다.
- **프로젝트 내부 지시 문서의 서열** — 단 하나의 최고 지시문만을 인준합니다 (선착순 승자독식): `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. 반면 SOUL.md는 언제나 이것과는 별개로 무조건 읽힙니다.
- **AGENTS.md**는 상하위 수직 통합적입니다: 하위 폴더에 또 AGENTS.md가 있다면 양쪽 의견을 모두 수렴/병합합니다.
- Hermes는 구동 시 최초에 `SOUL.md`가 없으면 빈 스켈레톤 상태의 `SOUL.md`를 자동으로 생성합니다.
- 이렇게 수거된 규칙 문서들은 최대 20,000자 용량까지만 스마트하게 깎여 수용됩니다.

더 깊은 읽을거리:
- [성격 & SOUL.md](/user-guide/features/personality)
- [컨텍스트 파일들](/user-guide/features/context-files)

## 현재 작업 디렉토리 (Working Directory)

| 환경 | 기본값 |
|---------|---------|
| **CLI (`hermes`)** | 명령어를 발동한 당시 위치한 바로 그 디렉토리 |
| **메시지 게이트웨이** | `~/.hermes/config.yaml` 안의 `terminal.cwd`를 따름; 비워있으면 유저 홈 디렉토리 `~` |
| **Docker / Singularity / Modal / SSH** | 컨테이너 또는 원격지의 타겟 유저 홈 디렉토리 |

작업 디렉토리 덮어쓰기:
```yaml
# ~/.hermes/config.yaml 내부:
terminal:
  cwd: /home/myuser/projects
```

참고: 과거 `~/.hermes/.env` 시절 쓰던 `MESSAGING_CWD` 나 직통의 `TERMINAL_CWD` 환경 변수는 하위 호환성을 위한 잔재(fallback)로 격하되었습니다. 가급적 새 방식인 `terminal.cwd`를 사용하십시오.
