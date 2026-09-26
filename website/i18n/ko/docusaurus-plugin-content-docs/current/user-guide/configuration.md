---
sidebar_position: 2
title: "구성"
description: "Hermes Agent 구성 — config.yaml, 제공자, 모델, API 키 등"
---

# 구성

모든 설정은 쉽게 접근할 수 있도록 `~/.hermes/` 디렉터리에 저장됩니다.

:::tip 작동하는 `config.yaml`을 만드는 가장 쉬운 방법
`hermes setup --portal`을 실행하세요. OAuth 한 번으로 모델 제공자와 네 가지 Tool Gateway 도구를 모두 설정할 수 있으므로 YAML을 직접 편집할 필요가 없습니다. Portal 구독자는 토큰 기준으로 요금이 부과되는 제공자를 10% 할인받을 수도 있습니다. [Nous Portal](/integrations/nous-portal)을 참고하세요.
:::

## 디렉터리 구조

```text
~/.hermes/
├── config.yaml     # Settings (model, terminal, TTS, compression, etc.)
├── .env            # API keys and secrets
├── auth.json       # OAuth provider credentials (Nous Portal, etc.)
├── SOUL.md         # Primary agent identity (slot #1 in system prompt)
├── memories/       # Persistent memory (MEMORY.md, USER.md)
├── skills/         # Agent-created skills (managed via skill_manage tool)
├── cron/           # Scheduled jobs
├── sessions/       # Gateway sessions
└── logs/           # Logs (errors.log, gateway.log — secrets auto-redacted)
```

## 구성 관리

```bash
hermes config              # View current configuration
hermes config edit         # Open config.yaml in your editor
hermes config get KEY      # Print a resolved value
hermes config set KEY VAL  # Set a specific value
hermes config unset KEY    # Remove a user-set value
hermes config check        # Check for missing options (after updates)
hermes config migrate      # Interactively add missing options

# Examples:
hermes config get model
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config unset terminal.backend
hermes config set OPENROUTER_API_KEY sk-or-...  # Saves to .env
```

:::tip
`hermes config set` 명령은 값을 올바른 파일로 자동 분류합니다. API 키는 `.env`에 저장되고, 나머지는 모두 `config.yaml`에 저장됩니다.
:::

## 구성 우선순위

설정은 다음 순서(우선순위가 높은 순)로 결정됩니다.

1. **CLI 인수** — 예: `hermes chat --model anthropic/claude-sonnet-4` (호출별 재정의)
2. **`~/.hermes/config.yaml`** — 모든 비밀 정보가 아닌 설정의 기본 구성 파일
3. **`~/.hermes/.env`** — 환경 변수의 대체 수단이며, 비밀 정보(API 키, 토큰, 비밀번호)에는 **필수**
4. **내장 기본값** — 다른 설정이 없을 때 사용하는 하드코딩된 안전한 기본값

:::info 일반적인 기준
비밀 정보(API 키, 봇 토큰, 비밀번호)는 `.env`에 넣습니다. 그 외의 모든 항목(모델, 터미널 백엔드, 압축 설정, 메모리 제한, 도구 모음)은 `config.yaml`에 넣습니다. 두 곳 모두 설정된 경우 비밀 정보가 아닌 설정에는 `config.yaml`이 우선합니다.
:::

:::tip 조직 배포
관리자는 시스템 수준의 관리 디렉터리를 통해 일반 사용자가 재정의할 수 없는 특정 구성 및 비밀 값을 고정할 수 있습니다. [Managed Scope](/user-guide/managed-scope)를 참고하세요.
:::

## 런타임 제한

장시간 실행되는 Hermes 서버 표면(게이트웨이 및
`hermes serve --isolated` 포함)은 운영 체제가 지원하는 경우 시작할 때 구성된
`RLIMIT_NOFILE` 소프트 제한을 적용합니다.

```yaml
runtime:
  nofile_soft_limit: 4096
```

기본값은 `4096`입니다. Hermes는 목표값을 운영 체제의 하드 제한 이하로 조정하며, 이미 더 높은 소프트 제한을 가진 프로세스의 제한을 절대 낮추지 않습니다. 조정을 비활성화하려면 값을 `0`, `false` 또는 `null`로 설정하세요. Windows 및 제한을 변경할 수 없는 샌드박스에서는 제한을 변경하지 않고 시작을 계속합니다.

## 환경 변수 치환

`${VAR_NAME}` 구문을 사용하면 `config.yaml`에서 환경 변수를 참조할 수 있습니다.

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

하나의 값에 여러 참조를 사용할 수도 있습니다. 예: `url: "${HOST}:${PORT}"`. 참조한 변수가 설정되지 않은 경우 자리 표시자는 그대로 유지되며(`${UNDEFINED_VAR}`가 그대로 남음) 경고가 기록됩니다. `$VAR`처럼 단독으로 사용하는 형식은 확장되지 않습니다.

Cursor 스타일의 SecretRef 구문도 허용됩니다. `${env:VAR_NAME}`은 `${VAR_NAME}`과 정확히 동일하게 해석됩니다(`env:` 접두사가 제거됨). 따라서 Cursor / Claude 구성에서 복사한 MCP 또는 제공자 snippet을 `config.yaml`과 `mcp_servers` 블록에서 수정 없이 사용할 수 있습니다. 다른 SecretRef 소스(`${file:...}`, `${vault:...}`, `${bitwarden:...}`)는 인라인으로 해석되지 않습니다. 외부 비밀 백엔드는 시작 시 `secrets:` 블록을 통해 값을 환경에 주입하므로, 대신 `${env:NAME}`으로 참조하세요. 알 수 없는 접두사는 한 번 경고한 뒤 그대로 유지됩니다.

AI 제공자 설정(OpenRouter, Anthropic, Copilot, 사용자 지정 엔드포인트, 자체 호스팅 LLM, 대체 모델 등)은 [AI Providers](/integrations/providers)를 참고하세요.

### 제공자 시간 제한

제공자 전체 요청 시간 제한에는 `providers.<id>.request_timeout_seconds`를, 모델별 재정의에는 `providers.<id>.models.<model>.timeout_seconds`를 설정할 수 있습니다. 이 설정은 모든 전송 방식(OpenAI-wire, 네이티브 Anthropic, Anthropic-compatible)의 기본 턴 클라이언트, 대체 체인, 인증 정보 교체 후 재구성, 그리고 (OpenAI-wire의 경우) 요청별 timeout 키워드 인수에 적용됩니다. 따라서 구성된 값이 기존 `HERMES_API_TIMEOUT` 환경 변수보다 우선합니다.

비스트리밍 stale-call 감지기의 시간 제한에는 `providers.<id>.stale_timeout_seconds`를, 모델별 재정의에는 `providers.<id>.models.<model>.stale_timeout_seconds`를 설정할 수도 있습니다. 이 값은 기존 `HERMES_API_CALL_STALE_TIMEOUT` 환경 변수보다 우선합니다.

이 값을 설정하지 않으면 기존 기본값(`HERMES_API_TIMEOUT=1800`s, `HERMES_API_CALL_STALE_TIMEOUT=90`s, 네이티브 Anthropic 900s)이 유지됩니다. 비스트리밍 stale 감지기는 설정을 명시하지 않은 로컬 엔드포인트에서 자동으로 비활성화되며, 매우 큰 컨텍스트에서는 더 큰 값으로 확장될 수 있습니다. 현재 AWS Bedrock에는 연결되지 않습니다(`bedrock_converse`와 AnthropicBedrock SDK 경로 모두 자체 시간 제한 구성을 사용하는 boto3를 사용함). [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example)의 주석 처리된 예시를 참고하세요.

## 업데이트 동작

`hermes update` 설정은 `config.yaml`의 `updates` 아래에 있습니다.

```yaml
updates:
  pre_update_backup: quick       # quick (state snapshot, default) | full (snapshot + HERMES_HOME zip) | off
  backup_keep: 5                 # Keep this many full pre-update backup zips
  non_interactive_local_changes: stash  # stash | discard
```

`pre_update_backup`은 업데이트 전 안전을 위한 단일 설정입니다. `quick`(기본값)은 중요 상태 파일(페어링 데이터, cron 작업, 구성, 인증 정보)을 `state-snapshots/`에 저장합니다(1GiB를 초과하는 파일은 건너뜀). `full`은 여기에 더해 `HERMES_HOME` 전체를 `backups/`에 zip으로 보관하며, 홈 디렉터리가 크면 몇 분이 더 걸릴 수 있습니다. `off`는 두 작업을 모두 비활성화합니다. 기존 불리언 값도 지원됩니다(`true` → `full`, `false` → `off`).

Git 설치의 경우 Hermes는 업데이트 브랜치를 체크아웃하거나 pull하기 전에 수정된 추적 파일과 추적되지 않는 파일을 자동으로 stash합니다. 대화형 터미널 업데이트에서는 해당 stash를 복원하기 전에 확인을 요청합니다. 비대화형 업데이트(데스크톱/채팅 앱, 게이트웨이 또는 `--yes`)는 `updates.non_interactive_local_changes`를 사용합니다. `stash`는 pull 성공 후 로컬 소스 수정을 복원하고, `discard`는 pull 성공 후 업데이트로 생성된 stash를 삭제합니다. 로컬 소스 수정이 절대 유지되면 안 되는 관리형 설치에서만 `discard`를 사용하세요.

이 stash 단계에 앞서 Hermes는 npm install/build 과정에서 남은 추적된 `package-lock.json` 차이도 복원합니다. 업데이트하기 전에 의도한 lockfile 수정 사항을 커밋하거나 수동으로 stash하세요.

## 터미널 백엔드 구성

Hermes는 일곱 가지 터미널 백엔드를 지원합니다. 각 백엔드는 에이전트의 셸 명령이 실제로 실행되는 위치를 결정합니다. 로컬 컴퓨터, Docker 컨테이너, SSH를 통한 원격 서버, Modal 클라우드 샌드박스(직접 또는 Nous 관리 게이트웨이 경유), Daytona workspace, Vercel Sandbox, Singularity/Apptainer 컨테이너 중 하나입니다.

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | vercel_sandbox | singularity
  cwd: "."          # Gateway/cron working directory (CLI always uses launch dir)
  font_family: ""   # Desktop terminal font; e.g. "MesloLGS NF"
  timeout: 180      # Per-command timeout in seconds
  home_mode: auto   # auto | real | profile — subprocess HOME policy
  env_passthrough: []  # Env var names to forward to sandboxed execution (terminal + execute_code)
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Container image for Singularity backend
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Container image for Modal backend
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Container image for Daytona backend
```

`terminal.font_family`는 Hermes Desktop에 내장된 터미널의 글꼴을 제어합니다. 로컬에 설치된 단일 글꼴 패밀리 이름(예: `MesloLGS NF`) 또는 CSS 글꼴 스택을 사용할 수 있습니다. Hermes는 번들된 JetBrains Mono 스택을 대체 글꼴로 추가하며, 값이 비어 있으면 기본값을 유지합니다. **Settings → Appearance → Terminal Font**에서 동일한 프로필 범위 설정을 편집할 수도 있습니다. Google Fonts를 다운로드하거나 시스템 글꼴 권한을 부여할 필요가 없습니다.

Modal, Daytona, Vercel Sandbox와 같은 클라우드 샌드박스에서 `container_persistent: true`는 샌드박스를 다시 만들 때 파일 시스템 상태를 보존하도록 Hermes가 시도한다는 뜻입니다. 이후에도 동일한 실행 중인 샌드박스, PID 공간 또는 백그라운드 프로세스가 계속 실행 중임을 보장하지는 않습니다.

### 백엔드 개요

| 백엔드 | 명령 실행 위치 | 격리 | 적합한 용도 |
|---------|-------------------|-----------|----------|
| **local** | 사용자의 컴퓨터에서 직접 실행 | 없음 | 개발, 개인 사용 |
| **docker** | 단일 영구 Docker 컨테이너(세션, `/new`, 하위 에이전트 간 공유) | 완전 격리(namespaces, cap-drop) | 안전한 샌드박싱, CI/CD |
| **ssh** | SSH를 통한 원격 서버 | 네트워크 경계 | 원격 개발, 강력한 하드웨어 |
| **modal** | Modal 클라우드 샌드박스 | 완전 격리(클라우드 VM) | 일회성 클라우드 컴퓨팅, 평가 |
| **daytona** | Daytona workspace | 완전 격리(클라우드 컨테이너) | 관리형 클라우드 개발 환경 |
| **vercel_sandbox** | Vercel Sandbox | 완전 격리(클라우드 microVM) | 스냅샷 기반 파일 시스템 영속성을 지원하는 클라우드 실행 |
| **singularity** | Singularity/Apptainer 컨테이너 | Namespaces(--containall) | HPC 클러스터, 공유 컴퓨터 |

### 로컬 백엔드

기본값입니다. 명령은 격리 없이 사용자의 컴퓨터에서 직접 실행됩니다. 특별한 설정이 필요하지 않습니다.

```yaml
terminal:
  backend: local
```

기본적으로 로컬 도구 subprocess는 실제 OS 사용자 `HOME`을 유지합니다. 이를 통해 `git`, `ssh`, `gh`, `az`, `npm`, Claude Code, Codex와 같은 외부 CLI가 일반 셸에서 이미 사용 중인 인증 정보와 구성을 찾을 수 있습니다. Hermes 상태는 `HERMES_HOME`을 통해 여전히 프로필 범위로 관리됩니다. `HOME`은 프로필의 구성, 메모리, 세션 또는 스킬을 선택하는 데 사용되지 않습니다.

Hermes는 시스템 전체의 `HOME`, 셸 시작 파일 또는 운영 체제 계정 홈을 변경하지 않습니다. 이 설정은 `terminal`, 백그라운드 터미널 프로세스, `execute_code`, ACP helper process와 같은 도구를 통해 Hermes가 실행하는 subprocess에 전달되는 환경만 제어합니다.

#### `terminal.home_mode`

| 모드 | 호스트 설치 | 컨테이너 | 절충점 |
|---|---|---|---|
| `auto` | 실제 OS 사용자 `HOME` 유지 | `{HERMES_HOME}/home` 사용 | 권장 기본값. 호스트 CLI는 계속 작동하고 컨테이너 상태는 유지됩니다. |
| `real` | 실제 OS 사용자 `HOME` 강제 | 표시되는 경우 실제 OS 사용자 `HOME` 강제 | 상위 프로세스가 실수로 `HOME`을 프로필 홈으로 시작한 경우 유용합니다. |
| `profile` | 존재하는 경우 `{HERMES_HOME}/home` 사용 | 존재하는 경우 `{HERMES_HOME}/home` 사용 | 엄격한 프로필별 CLI 구성 격리. 하지만 프로필 홈에서 초기화하거나 연결하지 않으면 일반적인 `~/.ssh`, `~/.gitconfig`, `~/.azure`, `~/.config/gh`, Claude/Codex 인증, npm 상태 등을 사용할 수 없습니다. |

기본값의 단점은 호스트 프로필이 `~` 아래의 일반 사용자 수준 CLI 인증 정보/구성을 공유한다는 것입니다. 별도의 git ID, SSH 키, GitHub CLI 로그인, npm 구성 또는 클라우드 CLI 로그인이 필요한 프로필은 `home_mode: profile`을 사용하고 해당 프로필 홈에서 이러한 도구를 의도적으로 초기화하세요.

프로필별 도구 구성의 엄격한 격리를 원한다면 다음과 같이 설정하세요.

```yaml
terminal:
  home_mode: profile
```

이 모드에서 도구 subprocess는 `{HERMES_HOME}/home`을 `HOME`으로 사용합니다. Hermes는 실제 사용자 홈을 찾아야 하는 스크립트가 사용할 수 있도록 `HERMES_REAL_HOME`도 설정합니다. 컨테이너 백엔드는 `auto` 모드에서도 `{HERMES_HOME}/home`을 계속 사용합니다. 이 디렉터리가 영구 Hermes 데이터 볼륨에 있기 때문입니다.

프로필 상태와 실제 사용자 홈을 구분해야 하는 스크립트는 Hermes 데이터에는 `HERMES_HOME`을, 계정 홈에는 `HERMES_REAL_HOME`을 우선 사용해야 합니다.

```python
from pathlib import Path
import os

hermes_home = Path(os.environ["HERMES_HOME"])
real_home = Path(os.environ.get("HERMES_REAL_HOME", os.environ["HOME"]))
```

:::warning
에이전트는 사용자 계정과 동일한 파일 시스템 접근 권한을 가집니다. 사용하지 않을 도구는 `hermes tools`로 비활성화하거나, 샌드박싱을 위해 Docker로 전환하세요.
:::
### Docker 백엔드

보안 강화(모든 capability 삭제, 권한 상승 방지, PID 제한)가 적용된 Docker 컨테이너 내부에서 명령을 실행합니다.

**Hermes 프로세스 간에 공유되는 단일 영속 컨테이너.** Hermes는 처음 사용할 때 수명이 긴 컨테이너 하나를 시작하고, 이후 모든 terminal, file, `execute_code` 호출을 동일한 컨테이너에 대한 `docker exec`로 라우팅합니다. 세션 간, `/new`, `/reset`, `delegate_task` 하위 에이전트 간에도 마찬가지입니다. 작업 디렉터리 변경, 설치된 패키지, `/workspace`의 파일, **백그라운드 프로세스**는 한 번의 도구 호출에서 다음 호출로, 한 Hermes 프로세스에서 다음 프로세스로 모두 이어집니다. TUI 세션을 닫거나 `/quit`를 실행하거나 새 `hermes` 호출을 시작해도 컨테이너는 계속 실행되며, 다음 Hermes 프로세스가 레이블 조회를 통해 재사용합니다. 정확한 규칙은 아래 **컨테이너 수명 주기**를 참고하세요.

**세션별 격리 모드 (`container_persistent: false`).** Docker 백엔드에서 `container_persistent: false`를 설정하면 세션마다 컨테이너 하나가 사용됩니다. 각 채팅(데스크톱 앱 세션, gateway 대화, TUI 세션)은 첫 terminal/file 호출 시 고유한 새 샌드박스를 만들고, 세션이 닫히거나 `lifetime_seconds` 동안 유휴 상태가 지나면 제거합니다. 세션 간에 파일 시스템 상태, 마운트, 백그라운드 프로세스는 전달되지 않습니다. `docker_mount_cwd_to_workspace: true`이면 해당 세션에 연결된 workspace만 `/workspace`에 마운트됩니다. 연결된 디렉터리가 없는 새 세션은 이전 세션의 마운트를 이어받지 않고 빈 workspace를 사용합니다. `delegate_task` 하위 에이전트는 부모 세션의 컨테이너를 공유합니다. 이 모드는 대화 간 샌드박스가 보안 경계여야 할 때 사용하고, 위에 설명한 세션 간 장기 실행 컨테이너를 원한다면 기본값인 `true`를 유지하세요.

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # Mount launch dir into /workspace
  docker_run_as_host_user: false   # See "Running container as host user" below
  docker_forward_env:              # Host env vars to forward into container
    - "GITHUB_TOKEN"
  docker_env:                      # Literal env vars to inject (KEY=value)
    DEBUG: "1"
    PYTHONUNBUFFERED: "1"
  docker_volumes:                  # Host directory mounts
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro for read-only
  docker_extra_args:               # Extra flags appended verbatim to `docker run`
    - "--gpus=all"
    - "--network=host"
  docker_network: true             # false = air-gap the container (--network=none)

  # Resource limits
  container_cpu: 1                 # CPU cores (0 = unlimited)
  container_memory: 5120           # MB (0 = unlimited)
  container_disk: 51200            # MB (requires overlay2 on XFS+pquota)
  container_persistent: true       # true = persist /workspace + /root, shared container; false = fresh container per session (see below)

  # Cross-process container reuse (defaults match the "one long-lived
  # container shared across sessions" contract — see Container lifecycle).
  docker_persist_across_processes: true   # Reuse container across Hermes restarts
  docker_orphan_reaper: true              # Sweep abandoned Exited containers at startup

  # Cross-backend lifecycle settings (apply to docker as well)
  timeout: 180                     # Per-command timeout in seconds
  lifetime_seconds: 300            # Idle-reaper window; also feeds 2× orphan-reaper threshold
```

**`docker_env`**와 **`docker_forward_env`**: 전자는 설정에서 지정한 리터럴 `KEY=value` 쌍을 주입합니다(값은 `config.yaml`에 저장되거나 `TERMINAL_DOCKER_ENV='{"DEBUG":"1"}'` JSON dict로 전달됩니다). 후자는 셸 또는 `~/.hermes/.env`의 값을 전달하므로 실제 secret이 설정 파일에 나타나지 않습니다. token에는 `docker_forward_env`를, 컨테이너에 필요한 정적 설정에는 `docker_env`를 사용하세요.

**`terminal.docker_extra_args`**(`TERMINAL_DOCKER_EXTRA_ARGS='["--gpus=all"]'`로도 재정의 가능)를 사용하면 Hermes가 일급 키로 제공하지 않는 임의의 `docker run` flag를 전달할 수 있습니다. 예를 들어 `--gpus`, `--network`, `--add-host`, 대체 `--security-opt` 재정의 등이 있습니다. 각 항목은 문자열이어야 하며, 목록은 조립된 `docker run` 호출의 마지막에 추가되므로 필요하면 Hermes의 기본값을 재정의할 수 있습니다. 샌드박스 강화 설정과 충돌하는 flag(capability 삭제, `--user`, workspace bind mount 등)는 격리를 알림 없이 약화시킬 수 있으므로 신중하게 사용하세요.

**`terminal.docker_network`**(기본값 `true`; env: `TERMINAL_DOCKER_NETWORK`) — `false`로 설정하면 `--network=none`으로 샌드박스 컨테이너를 실행해 agent 명령의 모든 네트워크 egress를 차단합니다. 이는 `terminal`, `execute_code`, file 도구가 사용하는 실행 컨테이너에 적용됩니다. 컨테이너는 Hermes 프로세스 간에 유지되므로, 기존 네트워크 연결 컨테이너가 있는 상태에서 `false`로 변경하면 해당 컨테이너를 제거하고 새 air-gapped 컨테이너를 시작합니다(경고가 기록됨). 그 안에서 실행 중인 백그라운드 프로세스는 사라집니다. `docker_extra_args`를 통해 `--network=none`을 전달하기보다 이 키를 사용하세요.

**요구 사항:** Docker Desktop 또는 Docker Engine이 설치되어 실행 중이어야 합니다. Hermes는 `$PATH`와 일반적인 macOS 설치 경로(`/usr/local/bin/docker`, `/opt/homebrew/bin/docker`, Docker Desktop 앱 번들)를 확인합니다. Podman도 기본 지원됩니다. Docker와 Podman이 모두 설치되어 있을 때는 `HERMES_DOCKER_BINARY=podman`(또는 전체 경로)을 설정해 강제로 사용할 수 있습니다.

#### 컨테이너 수명 주기

Hermes가 관리하는 모든 컨테이너에는 후속 프로세스와 orphan reaper가 컨테이너를 식별할 수 있도록 세 개의 label이 지정됩니다.

- `hermes-agent=1` — Hermes가 관리하는 컨테이너임을 표시
- `hermes-task-id=<sanitized task_id>` — task별 재사용 조회의 키
- `hermes-profile=<sanitized profile name>` — 현재 Hermes profile에 재사용과 수명 정리를 한정

시작할 때 Hermes는 `docker ps --filter label=hermes-task-id=<id> --filter label=hermes-profile=<profile>`을 실행하고, **기존 컨테이너를 찾으면 연결합니다.** 컨테이너가 `exited` 상태라면(예: Docker daemon 재시작 후) `docker start`로 다시 시작해 재사용합니다. 파일 시스템 상태와 설치된 패키지는 유지되지만, 컨테이너 내부의 백그라운드 프로세스는 유지되지 않습니다.

Hermes 프로세스가 종료될 때(`/quit`, TUI 세션 종료, gateway 종료, 심지어 SIGKILL도 포함) 기본 모드에서는 정리 경로가 **컨테이너에 아무 작업도 하지 않습니다.** 컨테이너는 계속 실행되며, 다음 Hermes 프로세스는 label 조회를 통해 수 밀리초 안에 연결합니다. 이것이 "세션 간 하나의 장기 실행 컨테이너" 계약에 필요한 동작입니다. 백그라운드 프로세스(npm watcher, 개발 서버, 장시간 실행되는 pytest)가 세션 간에 유지되는 유일한 방법이기도 합니다.

**다음 경우에만 컨테이너를 종료합니다(중지 후 `docker rm -f`).**

| 트리거 | 발생 시점 |
|---|---|
| `docker_persist_across_processes: false` | 명시적인 프로세스별 격리. 각 `cleanup()`이 `stop` + `rm -f`를 수행합니다. issue #20561 이전 동작과 동일합니다. |
| 유휴 정리기(`lifetime_seconds`, 기본값 300초) | env가 `persist_across_processes=false`인 경우에만 해당합니다. Persist 모드 env에서는 no-op이 되며 컨테이너는 유휴 정리를 통과해 유지됩니다. |
| 다음 시작 시 orphan reaper | 현재 profile에 한정하여 `2 × lifetime_seconds`(기본값 600초 = 10분)보다 오래된 **Exited** Hermes label 컨테이너를 정리합니다. **실행 중인 컨테이너에는 절대 손대지 않습니다** — 형제 프로세스 안전성 때문입니다. 비활성화하려면 `docker_orphan_reaper: false`를 설정하세요. |
| 직접적인 사용자 작업 | `docker rm -f`, `docker system prune`, Docker Desktop 재시작. `--restart=always`를 설정하지 않으므로 호스트가 재부팅되면 컨테이너는 `Exited` 상태가 됩니다(CoW layer는 유지되어 다음 시작 시 재사용되지만 백그라운드 프로세스는 사라짐). |

알아두면 좋은 예외 상황:

- **컨테이너 내부 PID 1의 OOM kill**은 컨테이너를 `Exited` 상태로 전환합니다. 다음 재사용 시 `docker start`로 다시 시작하며, 파일 시스템 상태는 유지되지만 백그라운드 프로세스는 유지되지 않습니다.
- **profile 전환**은 컨테이너를 서로 격리합니다 — `hermes-profile=work` label이 있는 컨테이너는 `hermes-profile=research`에서 실행 중인 Hermes 프로세스에 보이지 않습니다. orphan reaper도 profile 범위로 제한되므로 다른 profile의 컨테이너를 실수로 정리하지 않지만, 해당 profile로 Hermes를 다시 시작하기 전까지 자동으로 정리되지도 않습니다.

`delegate_task(tasks=[...])`를 통해 생성된 병렬 하위 에이전트는 이 컨테이너를 공유합니다 — 동시에 `cd`를 실행하거나 env를 변경하거나 같은 경로에 쓰면 충돌합니다. 하위 에이전트에 격리된 샌드박스가 필요하면 `register_task_env_overrides()`를 통해 task별 image 재정의를 등록해야 합니다. RL 및 benchmark 환경(TerminalBench2, HermesSweEnv 등)은 task별 Docker image에 대해 이를 자동으로 수행합니다.

**보안 강화:**
- `DAC_OVERRIDE`, `CHOWN`, `FOWNER`만 다시 추가하고 `--cap-drop ALL` 적용
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- `/tmp`(512MB), `/var/tmp`(256MB), `/run`(64MB)에 크기가 제한된 tmpfs

**Credential 전달:** `docker_forward_env`에 나열된 env var는 먼저 셸 환경에서, 다음으로 `~/.hermes/.env`에서 확인됩니다. Skill은 `required_environment_variables`를 선언할 수도 있으며, 이 값은 자동으로 병합됩니다.

#### 환경 변수 재정의

`terminal:` 아래의 모든 키에는 `TERMINAL_<KEY_UPPERCASE>` 형식의 env-var 재정의가 있습니다. Docker 백엔드에서 가장 유용한 항목은 다음과 같습니다.

| Env var | 매핑 대상 | 참고 |
|---|---|---|
| `TERMINAL_DOCKER_IMAGE` | `docker_image` | 기본 image |
| `TERMINAL_DOCKER_FORWARD_ENV` | `docker_forward_env` | JSON array: `'["GITHUB_TOKEN","OPENAI_API_KEY"]'` |
| `TERMINAL_DOCKER_ENV` | `docker_env` | JSON dict: `'{"DEBUG":"1"}'` |
| `TERMINAL_DOCKER_VOLUMES` | `docker_volumes` | JSON array: `"host:container[:ro]"` 문자열 |
| `TERMINAL_DOCKER_EXTRA_ARGS` | `docker_extra_args` | JSON array |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | `docker_mount_cwd_to_workspace` | `true` / `false` |
| `TERMINAL_DOCKER_RUN_AS_HOST_USER` | `docker_run_as_host_user` | `true` / `false` |
| `TERMINAL_DOCKER_NETWORK` | `docker_network` | `true` / `false` — 기본값 `true`; `false` = `--network=none` |
| `TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES` | `docker_persist_across_processes` | `true` / `false` — 기본값 `true` |
| `TERMINAL_DOCKER_ORPHAN_REAPER` | `docker_orphan_reaper` | `true` / `false` — 기본값 `true` |
| `TERMINAL_CONTAINER_CPU` | `container_cpu` | CPU 코어 |
| `TERMINAL_CONTAINER_MEMORY` | `container_memory` | MB |
| `TERMINAL_CONTAINER_DISK` | `container_disk` | MB |
| `TERMINAL_CONTAINER_PERSISTENT` | `container_persistent` | `docker_persist_across_processes`와 별개로 bind-mount workspace 디렉터리를 제어하는 `true` / `false` |
| `TERMINAL_LIFETIME_SECONDS` | `lifetime_seconds` | 유휴 정리기 기간 |
| `TERMINAL_TIMEOUT` | `timeout` | 명령별 timeout |
| `HERMES_DOCKER_BINARY` | _none_ | 특정 docker/podman binary 경로 강제 |

### SSH 백엔드

SSH를 통해 원격 서버에서 명령을 실행합니다. 연결 재사용을 위해 ControlMaster를 사용하며(5분 유휴 keepalive), 기본적으로 영속 셸이 활성화되어 있습니다 — 상태(cwd, env var)는 명령 간에 유지됩니다.

```yaml
terminal:
  backend: ssh
  persistent_shell: true           # Keep a long-lived bash session (default: true)
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
| `TERMINAL_SSH_KEY` | (system default) | SSH private key 경로 |
| `TERMINAL_SSH_PERSISTENT` | `true` | 영속 셸 활성화 |

**작동 방식:** 초기화 시 `BatchMode=yes` 및 `StrictHostKeyChecking=accept-new`으로 연결합니다. 영속 셸은 원격 호스트에서 단일 `bash -l` 프로세스를 계속 실행하며, 임시 파일을 통해 통신합니다. `stdin_data` 또는 `sudo`가 필요한 명령은 자동으로 일회성 모드로 전환됩니다.

### Modal 백엔드

[Modal](https://modal.com) cloud sandbox에서 명령을 실행합니다. 각 task에는 CPU, 메모리, 디스크를 구성할 수 있는 격리된 VM이 할당됩니다. 파일 시스템은 세션 간에 snapshot/restore할 수 있습니다.

```yaml
terminal:
  backend: modal
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB (5GB)
  container_disk: 51200            # MB (50GB)
  container_persistent: true       # Snapshot/restore filesystem
```

**필수 사항:** `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` 환경 변수를 사용하거나 `~/.modal.toml` 설정 파일을 사용해야 합니다.

**지속성:** 활성화하면 정리 시 샌드박스 파일 시스템을 snapshot하고 다음 세션에서 복원합니다. Snapshot은 `~/.hermes/modal_snapshots.json`에서 추적합니다. 이를 통해 파일 시스템 상태는 보존되지만 실행 중인 프로세스, PID 공간, 백그라운드 작업은 보존되지 않습니다.

**Credential 파일:** `~/.hermes/`에서 자동으로 마운트되며 각 명령 전에 동기화됩니다.
### Daytona 백엔드

[Daytona](https://daytona.io)에서 관리하는 작업 공간에서 명령을 실행합니다. 지속성을 지원하므로 중지했다가 재개할 수 있습니다.

```yaml
terminal:
  backend: daytona
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB → converted to GiB
  container_disk: 10240            # MB → converted to GiB (max 10 GiB)
  container_persistent: true       # Stop/resume instead of delete
```

**필수:** `DAYTONA_API_KEY` 환경 변수.

**지속성:** 지속성이 활성화되면 샌드박스가 정리될 때 삭제되지 않고 중지되며, 다음 세션에서 재개됩니다. 샌드박스 이름은 `hermes-{task_id}` 패턴을 따릅니다.

**디스크 제한:** Daytona는 최대 10 GiB를 적용합니다. 10 GiB를 초과하는 요청은 경고와 함께 제한됩니다.

### Vercel 샌드박스 백엔드

[Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) 클라우드 마이크로 VM에서 명령을 실행합니다. Hermes는 일반 터미널 및 파일 도구 표면을 사용하며, Vercel 전용 모델용 도구는 없습니다.

```yaml
terminal:
  backend: vercel_sandbox
  vercel_runtime: node24          # node24 | node22 | python3.13
  cwd: /vercel/sandbox            # default workspace root
  container_persistent: true      # Snapshot/restore filesystem
  container_disk: 51200           # Shared default only; custom disk is unsupported
```

**필수 설치:** 선택적 SDK 추가 기능을 설치합니다.

```bash
pip install 'hermes-agent[vercel]'
```

**필수 인증:** `VERCEL_TOKEN`, `VERCEL_PROJECT_ID`, `VERCEL_TEAM_ID` 세 가지를 모두 사용해 액세스 토큰 인증을 구성합니다. Render, Railway, Docker 및 유사한 호스트에서 배포와 일반적인 장기 실행 Hermes 프로세스에 지원되는 설정입니다.

일회성 로컬 개발에는 수명이 짧은 Vercel OIDC 토큰도 사용할 수 있습니다.

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

연결된 Vercel 프로젝트 디렉터리에서는 프로젝트 이름을 생략할 수 있습니다.

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

OIDC 토큰은 수명이 짧으므로 문서화된 배포 경로로 사용해서는 안 됩니다.

**런타임:** `terminal.vercel_runtime`은 `node24`, `node22`, `python3.13`을 지원합니다. 설정하지 않으면 Hermes는 `node24`를 기본값으로 사용합니다.

**지속성:** `container_persistent: true`이면 Hermes는 정리 중에 샌드박스 파일 시스템의 스냅샷을 저장하고, 동일한 작업의 후속 샌드박스를 해당 스냅샷에서 복원합니다. 스냅샷에는 샌드박스에 복사된 Hermes 동기화 자격 증명, 스킬 및 캐시 파일이 포함될 수 있습니다. 이는 파일 시스템 상태만 보존하며, 실행 중인 샌드박스의 ID, PID 공간 또는 백그라운드 프로세스는 보존하지 않습니다.

**백그라운드 명령:** `terminal(background=true)`는 Hermes의 일반적인 로컬이 아닌 백그라운드 프로세스 흐름을 사용합니다. 샌드박스가 실행 중인 동안 일반 프로세스 도구를 통해 프로세스를 생성하고, 폴링하고, 대기하고, 로그를 확인하고, 종료할 수 있습니다. Hermes는 정리 또는 재시작 후 Vercel에서 분리된 프로세스를 복구하는 기능을 기본 제공하지 않습니다.

**디스크 크기:** Vercel Sandbox는 현재 Hermes의 `container_disk` 리소스 설정을 지원하지 않습니다. `container_disk`를 설정하지 않거나 공유 기본값인 `51200`으로 두십시오. 기본값이 아닌 값은 조용히 무시되지 않고 진단 및 백엔드 생성을 실패하게 합니다.

### Singularity/Apptainer 백엔드

Docker를 사용할 수 없는 HPC 클러스터 및 공유 머신을 위해 설계된 [Singularity/Apptainer](https://apptainer.org) 컨테이너에서 명령을 실행합니다.

```yaml
terminal:
  backend: singularity
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB
  container_persistent: true       # Writable overlay persists across sessions
```

**요구 사항:** `$PATH`에 `apptainer` 또는 `singularity` 바이너리가 있어야 합니다.

**이미지 처리:** Docker URL(`docker://...`)은 자동으로 SIF 파일로 변환되어 캐시됩니다. 기존 `.sif` 파일은 그대로 사용합니다.

**스크래치 디렉터리:** 다음 순서로 확인합니다: `TERMINAL_SCRATCH_DIR` → `TERMINAL_SANDBOX_DIR/singularity` → `/scratch/$USER/hermes-agent` (HPC 관례) → `~/.hermes/sandboxes/singularity`.

**격리:** 호스트 홈 디렉터리를 마운트하지 않고 전체 네임스페이스 격리를 적용하기 위해 `--containall --no-home`을 사용합니다.

### 일반적인 터미널 백엔드 문제

터미널 명령이 즉시 실패하거나 터미널 도구가 비활성화되었다고 표시되는 경우:

- **로컬** — 특별한 요구 사항이 없습니다. 시작할 때 가장 안전한 기본값입니다.
- **Docker** — `docker version`을 실행해 Docker가 작동하는지 확인합니다. 실패하면 Docker를 수정하거나 `hermes config set terminal.backend local`을 실행합니다.
- **SSH** — `TERMINAL_SSH_HOST`와 `TERMINAL_SSH_USER`를 모두 설정해야 합니다. 둘 중 하나라도 없으면 Hermes가 명확한 오류를 기록합니다.
- **Modal** — `MODAL_TOKEN_ID` 환경 변수 또는 `~/.modal.toml`이 필요합니다. 확인하려면 `hermes doctor`를 실행합니다.
- **Daytona** — `DAYTONA_API_KEY`가 필요합니다. Daytona SDK가 서버 URL 설정을 처리합니다.
- **Singularity** — `$PATH`에 `apptainer` 또는 `singularity`가 필요합니다. HPC 환경에서 일반적으로 사용됩니다.

확실하지 않다면 먼저 `hermes config set terminal.backend local`로 터미널 백엔드를 `local`로 되돌리고 명령이 실행되는지 확인합니다.

### 원격에서 호스트로 상태 동기화

**SSH**, **Modal**, **Daytona** 백엔드의 경우 Hermes는 세션 중 `~/.hermes/` 상태(자격 증명 파일, 스킬, 캐시)를 원격 샌드박스로 전송하고, 종료 시 변경된 상태 파일을 원래 호스트 위치로 **동기화합니다**. 처음 전송된 내용과 다른 파일(콘텐츠 해시로 비교)은 원래 위치에 적용되며, 동기화된 디렉터리 아래의 새 원격 파일(예: 에이전트가 원격에서 생성한 스킬)은 해당 호스트 경로에 대응하도록 매핑됩니다. 업로드 전용 자격 증명 파일은 호스트에서 절대 덮어쓰지 않습니다.

- 동기화는 백오프와 함께 최대 3회 재시도하며, 2 GiB보다 큰 원격 아카이브의 압축 해제를 거부합니다.
- Docker와 Singularity는 바인드 마운트(호스트 파일 시스템을 실시간으로 확인)를 사용하므로 이 동기화가 필요하지 않습니다.
- 이는 Hermes 상태(`~/.hermes/`)에 적용되며, 샌드박스 내부의 임의 작업 트리 파일에는 적용되지 않습니다 — 중요한 산출물은 명시적으로 호스트 밖으로 복사해야 합니다(예: `scp`, `modal volume put`).

### Docker 볼륨 마운트

Docker 볼륨을 사용하면 호스트 디렉터리를 컨테이너와 공유할 수 있습니다. 각 항목은 표준 Docker `-v` 구문을 사용합니다.

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # Read-write (default)
    - "/home/user/datasets:/data:ro"              # Read-only
    - "/home/user/.hermes/cache/documents:/output" # Gateway-visible exports
```

이는 다음에 유용합니다:
- **에이전트에 파일 제공** (데이터 세트, 구성, 참조 코드)
- **산출물 수신** (생성된 보고서)
- **공유 작업 공간** (에이전트와 호스트가 모두 파일에 액세스)

메시징 게이트웨이를 사용하고 생성된 파일을 전송하려면 `/home/user/.hermes/cache/documents:/output`처럼 호스트에서 접근 가능한 전용 내보내기 마운트를 우선 사용하고 `MEDIA:/...`를 지정합니다.

- 컨테이너 안에서 `/output/...`에 파일을 작성합니다.
- `MEDIA:`에는 호스트 경로를 지정합니다(예: `MEDIA:/home/user/.hermes/cache/documents/report.txt`).
- 정확히 동일한 경로가 호스트의 게이트웨이 프로세스에도 존재하지 않는 한 `/workspace/...` 또는 `/output/...`을 지정하지 마십시오.

:::warning
YAML의 중복 키는 앞선 값을 조용히 덮어씁니다. 이미 `docker_volumes:` 블록이 있다면 나중에 또 다른 `docker_volumes:` 키를 추가하지 말고 새 마운트를 같은 목록에 병합하십시오.
:::

환경 변수로 설정할 수도 있습니다: `TERMINAL_DOCKER_VOLUMES='["/host:/container"]'` (JSON 배열).

### Docker 자격 증명 전달

기본적으로 Docker 터미널 세션은 임의의 호스트 자격 증명을 상속하지 않습니다. 컨테이너 안에서 특정 토큰이 필요하면 `terminal.docker_forward_env`에 추가합니다.

```yaml
terminal:
  backend: docker
  docker_forward_env:
    - "GITHUB_TOKEN"
    - "NPM_TOKEN"
```

Hermes는 나열된 각 변수를 먼저 현재 셸에서 확인하고, 저장된 경우 `hermes config set`을 통해 `~/.hermes/.env`로 대체 확인합니다.

:::warning
`docker_forward_env`에 나열된 항목은 컨테이너 안에서 실행되는 명령에 표시됩니다. 터미널 세션에 노출해도 괜찮은 자격 증명만 전달하십시오.
:::

### 호스트 사용자로 컨테이너 실행

기본적으로 Docker 컨테이너는 `root`(UID 0)로 실행됩니다. `/workspace` 또는 다른 바인드 마운트 안에서 생성된 파일은 호스트에서 root 소유가 되므로, 호스트 편집기에서 편집하려면 세션 후 `sudo chown`을 실행해야 합니다. `terminal.docker_run_as_host_user` 플래그가 이를 해결합니다.

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # default: false
```

활성화하면 Hermes는 `docker run` 명령에 `--user $(id -u):$(id -g)`를 추가하므로 바인드 마운트된 디렉터리(`/workspace`, `/root`, `docker_volumes`의 모든 항목)에 작성된 파일은 root가 아니라 호스트 사용자 소유가 됩니다. 그 대신 컨테이너에서 더 이상 `apt install`을 실행하거나 `/root/.npm`처럼 root 소유 경로에 쓸 수 없습니다 — 두 가지가 모두 필요하다면 `HOME`이 root가 아닌 사용자가 소유한 기본 이미지를 사용하거나 이미지 빌드 시 필요한 도구를 추가하십시오.

기존 동작과의 호환성을 위해 이 값을 `false`(기본값)로 두십시오. 주 작업이 마운트된 호스트 파일 편집이고 `sudo chown -R` 실행이 번거롭다면 켜십시오.

### 선택 사항: 시작 디렉터리를 `/workspace`에 마운트

Docker 샌드박스는 기본적으로 격리된 상태를 유지합니다. 명시적으로 선택하지 않는 한 Hermes는 현재 호스트 작업 디렉터리를 컨테이너에 전달하지 않습니다.

`config.yaml`에서 활성화합니다.

```yaml
terminal:
  backend: docker
  docker_mount_cwd_to_workspace: true
```

활성화하면:
- `~/projects/my-app`에서 Hermes를 실행할 경우 해당 호스트 디렉터리가 `/workspace`에 바인드 마운트됩니다.
- Docker 백엔드는 `/workspace`에서 시작합니다.
- 파일 도구와 터미널 명령 모두 동일한 마운트된 프로젝트를 확인합니다.

비활성화하면 `docker_volumes`를 통해 명시적으로 마운트하지 않는 한 `/workspace`는 샌드박스가 소유합니다.

보안상의 절충:
- `false`는 샌드박스 경계를 유지합니다.
- `true`는 샌드박스가 Hermes를 실행한 디렉터리에 직접 액세스하도록 합니다.

컨테이너가 호스트의 실시간 파일에서 작업하기를 의도적으로 원할 때만 옵트인하십시오.

### 영속 셸

기본적으로 각 터미널 명령은 별도의 하위 프로세스에서 실행되므로 작업 디렉터리, 환경 변수 및 셸 변수가 명령 사이에 초기화됩니다. **영속 셸**을 활성화하면 하나의 장기 실행 bash 프로세스가 `execute()` 호출 사이에 계속 유지되어 명령 사이의 상태가 보존됩니다.

이는 명령마다 연결을 설정하는 오버헤드도 제거하는 **SSH 백엔드**에서 특히 유용합니다. **SSH에서는 기본적으로 활성화**되고 로컬 백엔드에서는 비활성화됩니다.

```yaml
terminal:
  persistent_shell: true   # default — enables persistent shell for SSH
```

비활성화하려면:

```bash
hermes config set terminal.persistent_shell false
```

**명령 사이에 유지되는 항목:**
- 작업 디렉터리 (`cd /tmp`는 다음 명령에도 적용됨)
- 내보낸 환경 변수 (`export FOO=bar`)
- 셸 변수 (`MY_VAR=hello`)

**우선순위:**

| Level | Variable | Default |
|-------|----------|---------|
| Config | `terminal.persistent_shell` | `true` |
| SSH 재정의 | `TERMINAL_SSH_PERSISTENT` | 구성 값을 따름 |
| Local override | `TERMINAL_LOCAL_PERSISTENT` | `false` |

백엔드별 환경 변수가 가장 높은 우선순위를 가집니다. 로컬 백엔드에서도 영속 셸을 사용하려면:

```bash
export TERMINAL_LOCAL_PERSISTENT=true
```

:::note
`stdin_data` 또는 sudo가 필요한 명령은 영속 셸의 stdin이 이미 IPC 프로토콜에서 사용 중이므로 자동으로 일회성 모드로 대체됩니다.
:::

각 백엔드에 대한 자세한 내용은 [코드 실행](features/code-execution.md) 및 [README의 터미널 섹션](features/tools.md)을 참조하십시오.

## 스킬 설정

스킬은 SKILL.md 프런트매터를 통해 자체 구성 설정을 선언할 수 있습니다. 이러한 값은 비밀이 아닌 값(경로, 기본 설정, 도메인 설정)이며 `config.yaml`의 `skills.config` 네임스페이스 아래에 저장됩니다.

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # Example — each skill defines its own keys
```

**스킬 설정의 작동 방식:**

- `hermes config migrate`는 활성화된 모든 스킬을 검사하고, 구성되지 않은 설정을 찾아 입력을 요청합니다.
- `hermes config show`는 "스킬 설정" 아래에 모든 스킬 설정과 해당 스킬을 표시합니다.
- 스킬이 로드되면 확인된 구성 값이 자동으로 스킬 컨텍스트에 주입됩니다.

**값을 수동으로 설정:**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

자체 스킬에서 구성 설정을 선언하는 자세한 방법은 [스킬 만들기 — 구성 설정](/developer-guide/creating-skills#config-settings-configyaml)을 참조하십시오.
### 에이전트가 만든 스킬 쓰기 보호

에이전트가 `skill_manage`를 사용해 스킬을 만들거나, 편집하거나, 패치하거나, 삭제할 때 Hermes는 선택적으로 새 스킬 또는 업데이트된 스킬 콘텐츠에서 위험한 키워드 패턴(자격 증명 탈취, 명백한 프롬프트 인젝션, 데이터 유출 지침)을 검사할 수 있습니다. 스캐너는 기본적으로 꺼져 있습니다. `~/.ssh/`를 합법적으로 다루거나 `$OPENAI_API_KEY`를 언급하는 실제 에이전트 워크플로가 휴리스틱에 너무 자주 걸렸기 때문입니다. 에이전트가 만든 스킬 쓰기에서 검사를 다시 활성화하려면 다음과 같이 설정하세요.

```yaml
skills:
  guard_agent_created: true   # default: false
```

활성화하면 플래그가 지정된 쓰기 작업에 대해 스캐너의 판단 근거와 함께 승인 요청이 표시됩니다. 승인된 쓰기는 반영되고, 거부된 쓰기는 설명과 함께 반환됩니다.

### 스킬 쓰기 승인

이 설정은 콘텐츠 스캐너와 별개로 모든 에이전트 스킬 쓰기(생성 / 편집 / 패치 / 삭제 / 지원 파일)를 사용자의 명시적 승인 뒤에 둘 수 있습니다.

```yaml
skills:
  write_approval: false   # false = write freely (default) | true = stage every write for review
```

활성화하면 스킬 쓰기는 `/skills pending` → `/skills diff <id>` → `/skills approve <id>` 또는 `/skills reject <id>`를 통해 CLI나 메시징 플랫폼에서 검토할 수 있도록 `~/.hermes/pending/skills/` 아래에 대기 상태로 저장됩니다. 런타임에는 `/skills approval on|off`로 전환할 수 있습니다. 전체 안내는 [스킬 쓰기 승인 게이팅](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval)을 참고하세요.

## 메모리 구성

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
  write_approval: false     # true = require approval before any memory write
```

`memory.write_approval: true`로 설정하면 메모리를 저장하기 전에 승인이 필요합니다. 대화형 CLI에서는 입력창에 승인 요청이 표시되고, 메시징 세션과 백그라운드 자기 개선 검토에서는 저장할 내용을 `/memory pending`에 대기시킨 후 `/memory approve <id>` / `/memory reject <id>`로 검토합니다. 런타임에는 `/memory approval on|off`로 전환할 수 있습니다. 자세한 내용은 [메모리 저장 제어](/user-guide/features/memory#controlling-memory-writes-write_approval)를 참고하세요.

## 컨텍스트 파일 자르기

head/tail 자르기를 적용하기 전에 Hermes가 각 자동 컨텍스트 파일에서 불러오는 콘텐츠의 양을 제어합니다. 이는 `SOUL.md`, `.hermes.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`처럼 시스템 프롬프트에 주입되는 파일에 적용됩니다. `read_file` 도구에는 영향을 주지 않습니다.

```yaml
context_file_max_chars: null  # default — dynamic cap scaled to the model's context window (floor 20K, ceiling 500K chars)
```

동적 동작 대신 고정 상한을 사용하려면 양의 정수로 설정하세요.

```yaml
context_file_max_chars: 25000
```

## 파일 읽기 안전성

단일 `read_file` 호출이 반환할 수 있는 콘텐츠의 양을 제어합니다. 읽기 결과가 이 한도를 초과하면, 에이전트가 더 작은 범위를 읽도록 `offset`과 `limit`을 사용하라는 오류와 함께 요청이 거부됩니다. 이렇게 하면 축소되지 않은 JS 번들이나 대용량 데이터 파일 하나 때문에 컨텍스트 창이 가득 차는 것을 방지할 수 있습니다.

```yaml
file_read_max_chars: 100000  # default — ~25-35K tokens
```

컨텍스트 창이 큰 모델을 사용하며 대용량 파일을 자주 읽는다면 값을 늘리세요. 컨텍스트가 작은 모델에서는 효율적인 읽기를 위해 값을 낮추세요.

```yaml
# Large context model (200K+)
file_read_max_chars: 200000

# Small local model (16K context)
file_read_max_chars: 30000
```

에이전트는 파일이 변경되지 않은 경우 동일한 파일 영역을 자동으로 중복 제거합니다. 같은 파일 영역을 두 번 읽으면 다시 전송하는 대신 가벼운 스텁이 반환됩니다. 이 동작은 컨텍스트 압축 후 초기화되므로, 콘텐츠가 요약되어 사라진 뒤에는 에이전트가 파일을 다시 읽을 수 있습니다.

## 도구 출력 자르기 한도

원시 도구 출력이 잘리기 전에 적용되는 세 가지 관련 상한을 제어합니다.

```yaml
tool_output:
  max_bytes: 50000        # terminal output cap (chars)
  max_lines: 2000         # read_file pagination cap
  max_line_length: 2000   # per-line cap in read_file's line-numbered view
```

- **`max_bytes`** — `terminal` 명령이 이보다 많은 문자를 stdout/stderr에 합쳐 출력하면 Hermes는 앞 40%와 뒤 60%를 유지하고 그 사이에 `[OUTPUT TRUNCATED]` 알림을 삽입합니다. 기본값은 `50000`이며, 일반적인 토크나이저 기준 약 12~15K 토큰입니다.
- **`max_lines`** — 단일 `read_file` 호출의 `limit` 매개변수 상한입니다. 한 번의 읽기로 컨텍스트 창이 가득 차지 않도록 요청값이 이 한도를 넘으면 잘립니다. 기본값은 `2000`입니다.
- **`max_line_length`** — `read_file`이 줄 번호가 포함된 보기로 출력할 때 각 줄에 적용되는 상한입니다. 이보다 긴 줄은 지정한 문자 수까지 자른 뒤 `... [truncated]`를 덧붙입니다.

원시 도구 출력을 더 많이 처리할 수 있는 컨텍스트 창이 큰 모델에서는 한도를 늘리세요. 컨텍스트가 작은 모델에서는 출력 크기를 줄이기 위해 낮추세요.

```yaml
# Large context model (200K+)
tool_output:
  max_bytes: 150000
  max_lines: 5000

# Small local model (16K context)
tool_output:
  max_bytes: 20000
  max_lines: 500
```

## 전역 도구 세트 비활성화

모든 CLI와 모든 게이트웨이 플랫폼에서 특정 도구 세트를 한 곳에서 억제하려면 `agent.disabled_toolsets` 아래에 이름을 나열하세요.

```yaml
agent:
  disabled_toolsets:
    - memory       # hide memory tools + MEMORY_GUIDANCE injection
    - web          # no web_search / web_extract anywhere
```

이는 플랫폼별 도구 구성(`hermes tools`로 저장됨)보다 나중에 적용되므로, 여기에 나열된 도구 세트는 플랫폼의 저장된 구성에 포함되어 있어도 항상 제거됩니다. `hermes tools` 화면에서 15개가 넘는 플랫폼 행을 편집하는 대신 "어디서나 X 끄기"를 위한 단일 스위치로 사용할 수 있습니다.

목록을 비워 두거나 키를 생략하면 아무 동작도 하지 않습니다.

## Git 작업 트리 격리

동일한 저장소에서 여러 에이전트를 병렬로 실행할 때 격리된 git 작업 트리를 사용하도록 설정합니다.

```yaml
worktree: true    # Always create a worktree (same as hermes -w)
# worktree: false # Default — only when -w flag is passed
```

활성화하면 각 CLI 세션은 `.worktrees/` 아래에 자체 브랜치를 가진 새 작업 트리를 만듭니다. 에이전트는 서로 간섭하지 않고 편집, 커밋, 푸시, PR 생성을 수행할 수 있습니다. 깨끗한 작업 트리는 종료 시 제거되고, 변경 사항이 있는 작업 트리는 수동 복구를 위해 남겨집니다.

기본적으로 새 작업 트리는 로컬 클론의 오래된 `HEAD`가 아니라 새로 가져온 원격 최신 커밋에서 분기합니다. 현재 브랜치에 upstream이 있으면 그 upstream을, 없으면 원격 기본 브랜치를 기준으로 삼습니다. 이렇게 하면 PR의 기준이 실제 변경 사항에 맞게 유지됩니다. `worktree_sync: false`로 설정하면 로컬 `HEAD`에서 분기하도록 바꿀 수 있습니다. 오프라인 상태이거나 클론의 정확한 현재 상태를 기준으로 삼고 싶을 때 유용합니다. 원격에 연결할 수 없으면 자동으로 로컬 `HEAD`를 사용합니다.

```yaml
worktree_sync: true    # Default — branch from the fetched remote tip
# worktree_sync: false # Branch from local HEAD (offline / pinned base)
```

`.worktreeinclude`를 저장소 루트에 두어 gitignore된 파일을 작업 트리에 복사할 수도 있습니다.

```
# .worktreeinclude
.env
.venv/
node_modules/
```

## 컨텍스트 압축

Hermes는 모델 컨텍스트 창에 맞추기 위해 긴 대화를 자동으로 압축합니다. 압축 요약기는 별도의 LLM 호출로 실행되며, 어떤 프로바이더나 엔드포인트를 사용하도록 지정할 수 있습니다.

모든 압축 설정은 `config.yaml`에 저장됩니다(환경 변수는 사용하지 않음).

### 전체 참조

```yaml
compression:
  enabled: true                                     # Toggle compression on/off
  progress_notices: false                           # Opt-in: deliver routine compression progress notices to chat platforms — see below
  threshold: 0.50                                   # Compress at this % of context limit
  threshold_tokens: null                            # Absolute token cap (optional) — takes lower of ratio vs absolute
  target_ratio: 0.20                                # Fraction of threshold to preserve as recent tail
  protect_last_n: 20                                # Min recent messages to keep uncompressed
  protect_first_n: 3                                # Non-system head messages pinned across compactions (0 = pin nothing)
  in_place: true                                    # Compact on the same session id (no rotation) — see below
  idle_compact_after_seconds: 0                     # Opt-in idle compaction (0 = disabled) — see below
  hygiene_hard_message_limit: 5000                  # Gateway safety valve — see below
  hygiene_timeout_seconds: 30                       # Max seconds of NO summary-model output before hygiene compression is cut off
  hygiene_total_ceiling_seconds: 600                # Absolute cap on the hygiene wait even while tokens are still streaming
  hygiene_failure_cooldown_seconds: 300             # First rung of the per-session hygiene-failure backoff (x1/x3/x9, capped at 1h)
  context_timeout_seconds: 120                      # Inactivity budget for in-agent compress_context (loop /compress / preflight) — see below
  context_total_ceiling_seconds: 600                # Absolute cap on the *pre-commit* in-agent compress_context wait even while tokens are still streaming (an already-started SessionDB commit is never abandoned; overruns are logged + surfaced)
  proactive_prune_tokens: 0                         # Opt-in tokens trigger for the no-LLM tool-result prune (0 = off; see below)
  proactive_prune_min_result_chars: 8000            # Prune's summarize pass only touches tool results larger than this (clamped >= 200)
  proactive_prune_min_reclaim_tokens: 4096          # Prune only commits when it reclaims at least this many tokens (0 = commit any)

# The summarization model/provider is configured under auxiliary:
auxiliary:
  compression:
    model: ""                                       # Empty = use main chat model. Override with e.g. "google/gemini-3-flash-preview" for cheaper/faster compression.
    provider: "auto"                                # Provider: "auto", "openrouter", "nous", "codex", "main", etc.
    base_url: null                                  # Custom OpenAI-compatible endpoint (overrides provider)
```

:::info 레거시 구성 마이그레이션
`compression.summary_model`, `compression.summary_provider`, `compression.summary_base_url`가 포함된 이전 구성은 처음 로드할 때 `auxiliary.compression.*`으로 자동 마이그레이션됩니다(구성 버전 17). 수동 작업은 필요하지 않습니다.
:::

`progress_notices`(기본값 `false`)는 **일반적인** 압축 진행 상태를 채팅 플랫폼(Telegram, Discord, Slack 등)에 전달할지 제어합니다. 자동 압축은 채팅 화면에서 조용히 실행되고 서버 측 로그만 남기는 것이 기본 동작입니다. `progress_notices: true`로 설정하면 채팅 플랫폼에서 일반적인 수명 주기를 확인할 수 있습니다. 여기에는 "Compacting context…" 시작 알림, 사전 확인/사전 API 압축 트리거, 유휴 압축, 재시도 진행 상태("Compressed 30 → 12 messages, retrying…"), "Context compaction complete" 알림이 포함됩니다. 이 게이트는 압축 상태에만 적용되며, 관련 없는 운영 잡음(보조 모델 오류, 프로바이더 속도 제한/재시도 메시지)은 어느 쪽이든 계속 억제됩니다. 압축 **실패** 알림과 수동 `/compress` 피드백은 이 설정과 관계없이 항상 표시됩니다. 실행 중인 게이트웨이에서 이 값을 편집하면 다음 메시지부터 적용됩니다.

`hygiene_hard_message_limit`은 게이트웨이 전용 **사전 압축 안전장치**입니다. API 호출이 너무 큰 세션에서 계속 연결 해제될 때 발생하는 악순환을 끊기 위해 존재합니다. 게이트웨이는 토큰 사용량 데이터를 받지 못하므로 토큰 기반 임계값이 작동하지 않고, 대화 기록은 계속 커져 연결 해제가 더 심해질 수 있습니다. 이 개수 기반 하한은 API 오류와 관계없이 항상 알 수 있는 메시지 수만으로 작동하여 압축을 강제하고 세션을 복구합니다. 기본값은 `5000`이며, 압축이 이 값보다 훨씬 전에 토큰 임계값에 도달하므로 대규모 컨텍스트(1M 이상) 모델이 짧은 턴을 수천 번 수행하는 경우를 포함한 일반 세션보다 훨씬 큽니다. 특수한 플랫폼에서는 더 높이고, 더 적극적인 압축을 원하면 낮추세요. 실행 중인 게이트웨이에서 이 값을 편집하면 다음 메시지부터 적용됩니다(아래 참조).

`hygiene_timeout_seconds`는 이 에이전트 실행 전 압축 단계에 적용되는 게이트웨이의 **비활성 예산**이며, 전체 벽시계 시간 상한이 아닙니다. 압축 요약 호출은 모델에서 스트리밍되며, 도착하는 각 토큰은 진행 상황으로 계산됩니다. 따라서 느린 추론 모델도 계속 생성 중이면 자체 기한을 계속 연장하므로, 느리지만 정상인 요약 모델이 생성 중간에 끊기지 않습니다. 요약 모델이 이 시간 동안 **아무 출력도 생성하지 않을 때만**(백엔드 중단, 멈춘 연결, 응답이 없는 프로바이더) 게이트웨이가 사용자에게 경고하고 압축 없이 수신 메시지를 계속 처리하며, 멈춘 것처럼 보이는 대신 세션별 임시 실패 쿨다운을 기록합니다.

`hygiene_total_ceiling_seconds`(기본값 `600`)은 토큰이 계속 이동 중이어도 전체 대기 시간을 제한하므로, 비정상적으로 느린 스트리밍이 턴을 무기한 붙잡지 못합니다. 이 값은 최소한 `hygiene_timeout_seconds` 이상으로 제한됩니다.

`hygiene_failure_cooldown_seconds`는 위생 압축이 시간 초과되거나 중단된 후 세션별 쿨다운을 제어합니다. 쿨다운 중에는 게이트웨이가 같은 초과 크기 세션에 대한 반복적인 위생 압축 시도를 건너뛰므로, 수신 메시지가 매번 동일하게 고장 난 보조 백엔드에서 차단되지 않습니다. `/compress`, `/reset` 또는 이후 정상적인 턴을 통해 세션을 복구할 수 있습니다.

이 값은 고정된 간격이 아니라 점진적으로 증가하는 단계의 **첫 단계**입니다. 같은 세션에서 연속으로 실패하면 이 값의 `1x`, `3x`, `9x`를 기다리며, 최대 한 시간으로 제한됩니다. 요약 모델이 영구적으로 고장 난 세션도 고정된 간격으로 계속 재시도하지 않고 백오프하며, 실제로 대화 기록을 줄인 실행은 첫 단계로 재설정됩니다. 증가는 세션별·프로세스 로컬로 적용되므로 게이트웨이를 다시 시작하면 첫 단계로 재설정되지만, 쿨다운 기한 자체는 유지됩니다.

`context_timeout_seconds`(기본값 `120`)는 에이전트 내부의 `compress_context`에도 동일하게 적용되는 **비활성 예산**입니다. 대화 루프, 사전 확인 압축, 수동 `/compress`에서 멈춘 요약 모델이 세션을 무기한 지연시키지 못하게 합니다. 스트리밍되는 요약 토큰은 대기 시간을 연장하고, 응답이 없는 작업자만 중단됩니다. 시간 초과 시 Hermes는 압축을 건너뛰고 기존 메시지를 유지하며 사용자에게 경고합니다. 비활성화하려면 `0`으로 설정하세요. 게이트웨이 세션 위생은 자체 `hygiene_timeout_seconds` 경로를 사용하며 이중으로 감싸지지 않습니다.

`context_total_ceiling_seconds`(기본값 `600`)은 토큰이 계속 이동 중이어도 에이전트 내부 **커밋 전** 대기(요약/스트리밍 단계)를 제한합니다. 이 값은 최소한 `context_timeout_seconds` 이상으로 제한됩니다. 정확한 보장 사항은 다음과 같습니다. **요약 단계는 이 상한으로 제한되며, 커밋 단계가 이를 초과하면 로그에 기록되고 사용자에게 표시됩니다.** 작업자가 압축 커밋 펜스에 진입하고 SessionDB 변경이 진행 중이면 대화 기록이 불일치할 위험이 있으므로 커밋을 중간에 포기하지 않습니다. 대신 대기는 더 이상 조용히 진행되지 않습니다. 커밋이 상한을 넘으면 Hermes는 초과 시간을 기록하고(반복 시 WARNING에서 ERROR로 상승), 사용자에게 보이는 경고 채널을 통해 한 번만 경고를 보낸 뒤 커밋이 완료될 때까지 제한된 단위로 계속 기다립니다.

`protect_first_n`은 모든 압축에서 시스템이 아닌 앞부분 메시지 중 몇 개를 고정할지 제어합니다. 기본값은 `3`이며, 시작 부분의 사용자/어시스턴트 교환이 모든 요약기 실행에서 살아남아 원래 목표가 계속 보이게 합니다. 시작 턴이 더 이상 관련되지 않는 장기 롤링 압축 세션에서는 `protect_first_n: 0`으로 설정해 시스템 프롬프트와 요약, 최근 꼬리 부분만 고정할 수 있습니다. 이 설정과 관계없이 시스템 프롬프트 자체는 항상 보존됩니다.

`in_place`(기본값 `true`)는 압축이 실행될 때 세션 ID를 어떻게 처리할지 제어합니다. `true`이면 압축이 세션 ID를 변경하지 않고 메시지 목록을 다시 작성하며 시스템 프롬프트를 재구성합니다. 대화는 전체 수명 동안 하나의 영구 ID를 유지하므로 세션 목록에 `parent_session_id` 체인이나 `name #2` / `#3` 번호가 생기지 않습니다. 압축은 비파괴적입니다. 현재 컨텍스트는 압축되지만 압축 전 턴은 같은 ID 아래에서 소프트 아카이브되고(비활성/압축으로 표시됨) 삭제되지 않으므로 `session_search`로 계속 검색하고 복구할 수 있습니다. 훅은 `session:compress` 이벤트의 `in_place` 필드를 통해 모드를 확인합니다. `in_place: false`로 설정하면 각 압축이 이전 세션에 연결된 새 세션 ID로 교체되는 레거시 동작을 복원합니다.

`threshold_tokens`는 압축 트리거에 사용할 선택적 **절대 토큰 상한**을 설정합니다. 설정하면 비율 기반 `threshold`와 이 절대 개수 중 더 낮은 값에서 압축이 실행되므로, 현재 활성 모델과 관계없이 사용자가 선호한 토큰 수보다 늦게 압축이 실행되지 않습니다. 이 기능은 컨텍스트 창이 서로 다른 모델(예: 1M → 400K) 사이를 전환할 때 절대 트리거 지점이 바뀌는 문제를 해결합니다. 상한은 모델의 컨텍스트 길이로 제한되므로 모델이 지원하는 것보다 높게 설정해도 안전하며, 이 경우 비율 기반 임계값이 사용됩니다. 기본값은 `null`(비활성화 — 비율 기반 임계값만 사용)입니다. 이 상한은 모델 전환 및 폴백 활성화 후에도 유지됩니다.

`idle_compact_after_seconds`는 크기 기반 `threshold`를 보완하는 **선택적 시간 기반** 트리거입니다. 기본값은 `0`(비활성화)입니다. 0보다 크게 설정하면 일정 시간 이상 유휴 상태였던 세션이 첫 응답 전에 누적된 기록을 미리 압축합니다. 따라서 오랫동안 유지된 스레드(예: 몇 시간 후 다시 연 Telegram 대화)가 이후 각 턴마다 오래된 전체 컨텍스트를 다시 읽지 않아도 됩니다. 컨텍스트가 이미 압축 후 목표(`threshold × target_ratio`) 이하인 경우에는 실행되지 않으며, 모든 자동 압축과 동일한 실패 쿨다운, 과도한 반복 방지, 세션별 잠금 보호를 따릅니다. 예: `idle_compact_after_seconds: 1800`은 30분 유휴 후 압축합니다.

`proactive_prune_tokens`는 `threshold`와 독립적으로 실행되는 결정론적 무-LLM 도구 결과 정리를 활성화합니다. 대규모 창 모델에서는 `threshold` 압축(창의 약 50%)이 거의 실행되지 않으므로, 부피가 큰 도구 출력(터미널 덤프, 파일 읽기, 웹 추출)이 기록에 남아 이후 모든 턴마다 다시 전송됩니다. 재전송되는 기록이 `proactive_prune_tokens`(기본값 `0` = 비활성화; 활성화하려면 `48000`을 시도)보다 많으면 정리 과정에서 동일한 결과를 중복 제거하고, 오래된 대형 결과를 요약하며, 큰 도구 호출 인수를 자릅니다. 이때 가장 최근의 `protect_last_n` 메시지는 보호하고 모델은 호출하지 않습니다. 전체 출력은 세션 저장소에서 계속 복구할 수 있습니다. `proactive_prune_min_result_chars`(기본값 `8000`, 최소 `200`으로 제한)는 이보다 작은 도구 결과를 수정하지 않도록 합니다. `proactive_prune_min_reclaim_tokens`(기본값 `4096`)은 최소한 그만큼의 토큰을 회수할 때만 정리를 커밋합니다. 커밋된 정리는 이미 전송된 기록을 다시 작성하고 프로바이더의 프롬프트 캐시 접두사를 무효화하므로, 이 게이트는 모든 도구 반복마다 캐시가 깨지는 대신 압축 경계처럼 의미 있는 한 번의 중단으로 분산되도록 합니다. 이 기능은 내장 `compressor` 엔진에서만 실행되며 다른 컨텍스트 엔진에서는 아무 동작도 하지 않습니다.

:::tip 압축 및 컨텍스트 길이의 게이트웨이 핫 리로드
최근 릴리스부터 실행 중인 게이트웨이에서 `config.yaml`의 `model.context_length` 또는 `compression.*` 키를 편집하면 다음 메시지부터 적용됩니다. 게이트웨이를 다시 시작하거나 `/reset`을 실행하거나 세션을 교체할 필요가 없습니다. 캐시된 에이전트 시그니처에 이 키들이 포함되므로, 변경 사항을 감지하면 게이트웨이가 에이전트를 투명하게 다시 빌드합니다. API 키와 도구/스킬 구성에는 여전히 일반적인 리로드 경로가 필요합니다.
:::
### 일반적인 설정

**기본값(자동 감지) — 별도 설정이 필요하지 않습니다:**
```yaml
compression:
  enabled: true
  threshold: 0.50
```
주 제공자와 주 모델을 사용합니다. 작업별로 재정의할 수도 있습니다(예: 더 저렴한 모델에서 압축을 사용하려면 `auxiliary.compression.provider: openrouter` + `model: google/gemini-2.5-flash`).

**특정 제공자 강제 지정** (OAuth 또는 API 키 기반):
```yaml
auxiliary:
  compression:
    provider: nous
    model: gemini-3-flash
```
`nous`, `openrouter`, `codex`, `anthropic`, `main` 등 어떤 제공자에서도 사용할 수 있습니다.

**사용자 지정 엔드포인트** (자체 호스팅, Ollama, zai, DeepSeek 등):
```yaml
auxiliary:
  compression:
    model: glm-4.7
    base_url: https://api.z.ai/api/coding/paas/v4
```
사용자 지정 OpenAI 호환 엔드포인트를 지정합니다. 인증에는 `OPENAI_API_KEY`를 사용합니다.

### 세 가지 설정의 상호 작용

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | 결과 |
|---------------------|---------------------|--------|
| `auto` (기본값) | 설정하지 않음 | 사용 가능한 최적의 제공자를 자동 감지 |
| `nous` / `openrouter` / 기타 | 설정하지 않음 | 해당 제공자를 강제하고 그 인증 방식 사용 |
| 모든 값 | 설정함 | 사용자 지정 엔드포인트를 직접 사용 (제공자 설정은 무시) |

:::warning 요약 모델 컨텍스트 길이 요구 사항
요약 모델은 **주 에이전트 모델의 컨텍스트 창 이상**을 지원해야 합니다. 압축기는 대화의 중간 부분 전체를 요약 모델로 전송하므로, 해당 모델의 컨텍스트 창이 주 모델보다 작으면 요약 호출이 컨텍스트 길이 오류로 실패합니다. 이 경우 중간 대화가 요약 없이 **삭제되어**, 대화 컨텍스트가 조용히 손실됩니다. 모델을 재정의한다면 해당 모델의 컨텍스트 길이가 주 모델 이상인지 확인하세요.
:::

## 게이트웨이 턴 리스 타임아웃

게이트웨이는 확인된 세션 ID를 기준으로 턴을 직렬화하므로, 두 라우팅 키가 동일한 대화 기록을 동시에 불러오거나 기록할 수 없습니다. 일반 에이전트 비활성 타임아웃과는 별도로 최대 리스 대기 시간을 설정할 수 있습니다.

```yaml
agent:
  gateway_turn_lease_timeout: 1800
```

이 예산이 만료될 때 다른 턴이 세션 리스를 계속 보유하고 있으면 Hermes는 안전하게 실패합니다. 즉, 대화 기록을 불러오거나 대기 중인 메시지에 대해 모델을 실행하지 않습니다. 사용자는 거부 알림을 받고 메시지를 다시 보내야 합니다. 내구성 있는 순서 보장과 멱등성이 없는 상태에서 자동으로 메시지를 다시 큐에 넣으면 메시지가 두 번 처리될 수 있으므로, Hermes는 메시지를 자동으로 재큐잉하지 않습니다. 0 이하의 값은 기본값인 1800초를 사용합니다.

## 세션 정지 감시

게이트웨이는 알림만 보내는 정지 감시 기능(`agent.session_stall_timeout`, 기본값 `300`초, `0` = 비활성화)을 실행합니다. 사용 중인 세션에 **수신 대기 중인 후속 메시지**가 있고 에이전트의 공유 활동 시계가 이 시간 이상 유휴 상태이면, 게이트웨이는 WARNING을 기록하고 사용자에게 다음 일회성 알림을 보냅니다.

```
⚠️ Agent session appears stalled (last activity N min ago). Try /new to reset.
```

동작 방식:

- **알림만 보냅니다.** 감시 기능은 턴을 종료하지 않습니다. 장시간 비활성 후 실행을 취소하는 `agent.gateway_timeout`과 대비됩니다. 정지 알림은 에이전트가 멈춘 것처럼 보인다는 사실만 알려 주므로, 사용자가 (`/new`, `/stop`을 사용하거나 계속 기다리는 등) 대응을 결정할 수 있습니다.
- **정지 에피소드마다 한 번만 알립니다.** 수신 대기 메시지가 처리되거나 활동이 재개되면 래치가 해제되므로, 복구된 세션이 다시 정지하면 또 알림을 보냅니다.
- 진행 상태는 공유 활동 스냅샷(도구 호출, API 스트림 진행, 압축 하트비트)에서만 가져옵니다. 수신 대기 메시지는 알림 게이트일 뿐 진행 시계가 아닙니다.

```yaml
agent:
  session_stall_timeout: 300   # seconds; 0 disables the watchdog
```

## 재연결 주의 단계 상승

플랫폼 어댑터가 연결에 실패하면(네트워크 중단, 폐기된 봇 토큰, 손상된 사이드카 등) 게이트웨이는 상한이 있는 지수 백오프로 무기한 재시도합니다. 재시도는 멈추지 않으므로 일시적인 중단은 운영자의 조치 없이도 항상 복구됩니다. 단점은 영구적인 실패(폐기된 Telegram 토큰, 누락된 Discord privileged intents)가 "재시도 중"이라는 일시적인 문제와 똑같이 보인다는 것입니다.

영구적인 실패를 드러내기 위해 두 가지 메커니즘을 사용합니다.

- **종료 분류.** 예외 *유형*만으로 자체 복구가 불가능하다는 것을 증명할 수 있는 실패 — 거부되거나 폐기된 토큰(`telegram_auth_error`, `discord_auth_error`, `email_auth_error`), 누락된 privileged intents(`discord_intents_required`), 종속 항목을 설치할 수 없는 Photon 사이드카(`SIDECAR_DEPS_MISSING`) 또는 누락된 노드 바이너리(`SIDECAR_NODE_MISSING`) — 는 재시도 큐에 들어가지 않고 치명적 실패로 표시됩니다. 분류는 유형만 엄격하게 기준으로 삼으며, 모호한 오류는 항상 재시도를 계속합니다.
- **주의 필요 단계 상승.** `agent.reconnect_attention_after`(기본값 `7200`초 = 2시간)를 초과하여 플랫폼이 계속 재시도 큐에 있으면(값이 `0`이면 비활성화), 게이트웨이 런타임 상태(`hermes status`)에 `needs_attention: true`와 `retrying_since` 타임스탬프가 추가되고 WARNING 로그가 기록됩니다. 재시도 동작은 변경되지 않으며, 이는 회로 차단기가 아니라 신호입니다. 성공적으로 재연결되면 플래그가 해제됩니다.

```yaml
agent:
  reconnect_attention_after: 7200   # seconds; 0 disables the escalation flag
```

## 게이트웨이 에이전트 캐시

게이트웨이는 세션마다 에이전트 하나를 유지하므로, 매 턴 시스템 프롬프트를 다시 만드는 대신 대화에서 캐시된 프롬프트 접두사를 재사용합니다. 이 캐시된 에이전트는 세션의 전체 대화 기록도 보유하며 도구 출력까지 포함합니다. 도구 호출이 100번인 세션에서는 수십 메가바이트에 달할 수 있습니다. 따라서 여러 플랫폼을 처리하는 바쁜 게이트웨이에서는 이 캐시가 프로세스 메모리를 가장 많이 사용하는 단일 요소가 됩니다.

```yaml
agent:
  agent_cache:
    max_size: 128            # LRU entry cap
    idle_ttl_secs: 3600      # evict an agent idle this long
    memory_high_mb: auto     # anon-RSS budget; number, "auto", or 0/off
    max_evictions_per_pass: 16
    protect_recent: 8
```

`max_size`와 `idle_ttl_secs`는 각각 개수와 시간으로 캐시의 크기를 제한합니다. 어느 설정도 캐시가 보유한 바이트 수는 알 수 없으므로, `memory_high_mb`가 세 번째 제한을 추가합니다. 게이트웨이 자체의 익명 상주 메모리가 예산을 넘으면 가장 오래 사용되지 않은 대화 기록을 제거하고, 다음 턴에 저장된 세션에서 다시 불러옵니다. 게이트웨이가 다른 서비스와 메모리를 두고 경쟁한다면 값을 낮추고, 모든 프롬프트를 계속 준비된 상태로 유지하려면 값을 높이거나(`0`으로 설정하면 해당 정리 작업을 끌 수 있음) 하세요.

현재 턴을 처리 중인 세션, 가장 최근에 사용된 `protect_recent`개 세션, 그리고 대화 기록을 디스크에 기록하는 작업이 끝나지 않은 세션은 절대 제거하지 않습니다. 제거 시 측정된 RSS와 제거된 세션을 WARNING 수준으로 기록합니다.

```
Agent cache pressure: anon RSS 6802MB over budget 6656MB — evicting 5 LRU session(s): ...
```

## 컨텍스트 엔진

컨텍스트 엔진은 대화가 모델의 토큰 한도에 가까워질 때 대화를 관리하는 방식을 제어합니다. 기본 제공 `compressor` 엔진은 손실 요약을 사용합니다([컨텍스트 압축](/developer-guide/context-compression-and-caching) 참고). 플러그인 엔진은 이를 대체 전략으로 바꿀 수 있습니다.

```yaml
context:
  engine: "compressor"    # default — built-in lossy summarization
```

플러그인 엔진(예: 손실 없는 컨텍스트 관리를 위한 LCM)을 사용하려면 다음과 같이 설정합니다.

```yaml
context:
  engine: "lcm"          # must match the plugin's name
```

플러그인 엔진은 **절대 자동으로 활성화되지 않습니다.** 플러그인 이름을 `context.engine`에 명시적으로 설정해야 합니다. 사용 가능한 엔진은 `hermes plugins` → Provider Plugins → Context Engine에서 찾아 선택할 수 있습니다.

메모리 플러그인에 대한 동일한 단일 선택 시스템은 [메모리 제공자](/user-guide/features/memory-providers)를 참고하세요.

## 반복 예산

에이전트가 많은 도구 호출이 필요한 복잡한 작업을 수행하면 반복 예산(기본값: 500턴)을 모두 사용할 수 있습니다. Hermes는 **작업 중간에 압박 경고를 삽입하지 않습니다.** 이전 빌드에서는 예산의 70%/90%에 모델에 경고를 보냈지만, 이로 인해 모델이 복잡한 작업을 너무 일찍 포기하여 2026년 4월에 제거되었습니다.

대신 예산이 실제로 소진되면(500/500), Hermes는 모델에 작업을 마무리하라는 메시지를 한 번 삽입하고 최종 응답을 전달할 수 있도록 단 한 번의 **유예 호출**을 허용합니다. 유예 호출에서도 텍스트가 생성되지 않으면 에이전트에게 지금까지 수행한 작업을 요약하도록 요청합니다.

```yaml
agent:
  max_turns: 500               # Max iterations per conversation turn (default: 500)
  api_max_retries: 3           # Retries per provider before fallback engages (default: 3)
```

반복 예산이 완전히 소진되면 CLI는 사용자에게 다음 알림을 표시합니다. `⚠ Iteration budget reached (500/500) — response may be incomplete`

`agent.api_max_retries`는 대체 제공자로 전환하기 **전에** Hermes가 일시적인 오류(속도 제한, 연결 끊김, 5xx)로 제공자 API 호출을 재시도하는 횟수를 제어합니다. 기본값은 `3`이며 총 네 번 시도합니다. [대체 제공자](/user-guide/features/fallback-providers)를 설정했고 불안정한 엔드포인트에서 더 빠르게 장애 조치를 수행하려면 이 값을 `0`으로 낮추세요. 그러면 주 제공자에서 첫 번째 일시적 오류가 발생하는 즉시 재시도를 계속하는 대신 대체 제공자로 넘깁니다.

## 중지 시 검증(코딩 검증)

활성화하면 에이전트가 작업 공간의 코드를 수정했지만 새로운 검증 증거(통과한 테스트 실행, 빌드, 린트 등)를 남기지 않은 턴에서는 Hermes가 최종 답변을 받아들이지 않습니다. 대신 검증하거나 검증할 수 없는 이유를 설명하도록 요청하는 합성 후속 메시지를 삽입합니다. 문서/Markdown/skill만 수정한 경우에는 작동하지 않으며, 루프에도 상한이 있어 에이전트가 영원히 갇히지 않습니다.

```yaml
agent:
  verify_on_stop: false        # true | false | "auto" (surface-aware: on for CLI/TUI/desktop, off for messaging)
  verify_guidance: true        # Append creative-UI / clean-diff guidance to the missing-evidence nudge
  max_verify_nudges: 3         # Cap on consecutive continue nudges per turn (built-in + pre_verify hooks)
  coding_instructions: ""      # Standing project-wide coding rules appended to the coding brief
```

`verify_on_stop`은 어디서나 켜는 `true`, 끄는 `false`(기본값), 또는 `"auto"`(이전의 표면 인식 동작: CLI, TUI, desktop 같은 대화형 코딩 표면과 프로그래밍 방식 호출에서는 켜고 Telegram/Discord 같은 메시징 표면에서는 끔)를 허용합니다. 기본값은 어디서나 꺼져 있습니다. 새로 설치하면 `false`로 제공되고 기존 설치의 설정 마이그레이션도 이를 껐으므로, 활성화하려면 명시적으로 선택해야 합니다. 설정된 경우 `HERMES_VERIFY_ON_STOP` 환경 변수가 구성 값을 재정의합니다.

같은 지점에서 사용자/플러그인 정책 게이트를 적용하여 자체 검사를 계속 실행하게 하려면 [`pre_verify` 훅](/user-guide/features/hooks#pre_verify)을 참고하세요.

## 지속 목표(`/goal`)

지속 목표가 활성화되면 Hermes는 각 assistant 응답이 목표를 충족하는지 판단합니다. 충족하지 않으면 동일한 세션에 계속 진행하라는 프롬프트를 다시 전달하고, 목표가 완료되거나 턴 예산이 소진되거나 사용자가 일시 중지/삭제할 때까지 작업을 계속합니다. 턴 예산이 실제 안전장치입니다. 판단 실패는 **계속 진행하는 방향으로 실패를 허용**하므로, 불안정한 판단기가 진행을 막지 않습니다.

```yaml
goals:
  max_turns: 20   # Max continuation turns before Hermes auto-pauses the goal (default: 20)
```

`max_turns`는 목표가 계속 진행을 유도할 수 있는 턴 수를 제한합니다. 한도에 도달하면 Hermes가 목표를 자동으로 일시 중지하고 사용자에게 `/goal resume`을 요청합니다. 이는 판단기의 거짓 음성(실제로는 목표가 완료되었지만 판단기가 계속하라고 하는 경우)과 모호하거나 달성할 수 없는 목표에 대한 무제한 모델 비용을 방지합니다. 전체 기능은 [목표](/user-guide/features/goals)를 참고하세요.

### API 타임아웃

Hermes에는 스트리밍을 위한 별도의 타임아웃 계층과 비스트리밍 호출을 위한 오래된 응답 감지기가 있습니다. 오래된 응답 감지기는 암시적 기본값을 그대로 둔 경우에만 로컬 제공자에 맞춰 자동 조정됩니다.

| 타임아웃 | 기본값 | 로컬 제공자 | 설정 / 환경 변수 |
|---------|---------|----------------|--------------|
| 소켓 읽기 타임아웃 | 120s | 1800s로 자동 상향 | `HERMES_STREAM_READ_TIMEOUT` |
| 오래된 스트림 감지 | 180s | 900s 상한으로 상향 (`agent.local_stream_stale_timeout`) | `HERMES_STREAM_STALE_TIMEOUT` |
| 오래된 비스트리밍 감지 | 90s | 암시적 설정인 경우 자동 비활성화 | `providers.<id>.stale_timeout_seconds` 또는 `HERMES_API_CALL_STALE_TIMEOUT` |
| API 호출(비스트리밍) | 1800s | 변경 없음 | `providers.<id>.request_timeout_seconds` / `timeout_seconds` 또는 `HERMES_API_TIMEOUT` |

**소켓 읽기 타임아웃**은 제공자로부터 다음 데이터 청크를 기다리는 동안 httpx가 대기하는 시간을 제어합니다. 로컬 LLM은 큰 컨텍스트를 프리필하는 동안 첫 토큰을 생성하기 전에 수 분이 걸릴 수 있으므로, Hermes는 로컬 엔드포인트를 감지하면 이 값을 30분으로 높입니다. `HERMES_STREAM_READ_TIMEOUT`을 명시적으로 설정하면 엔드포인트 감지와 관계없이 항상 그 값을 사용합니다.

**오래된 스트림 감지**는 SSE 연결 유지 핑은 받지만 실제 콘텐츠는 받지 못하는 연결을 종료합니다. 프리필 중 연결 유지 핑을 보내지 않는 로컬 제공자의 경우 기본값을 기본 180초 대신 유한한 900초 상한으로 높입니다. `agent.local_stream_stale_timeout` 또는 `HERMES_LOCAL_STREAM_STALE_TIMEOUT` 환경 변수로 설정할 수 있습니다.

**오래된 비스트리밍 감지**는 너무 오래 응답을 생성하지 않는 비스트리밍 호출을 종료합니다. 기본적으로 Hermes는 로컬 엔드포인트에서 이 기능을 비활성화하여 긴 프리필 중 잘못된 감지를 방지합니다. `providers.<id>.stale_timeout_seconds`, `providers.<id>.models.<model>.stale_timeout_seconds` 또는 `HERMES_API_CALL_STALE_TIMEOUT`을 명시적으로 설정하면 로컬 엔드포인트에서도 해당 명시적 값을 적용합니다.

이 예산은 cron 작업과 위임된 하위 에이전트가 인라인으로 실행하는 호출을 포함한 모든 비스트리밍 호출에 적용됩니다. 요청을 수락한 뒤 아무 응답도 보내지 않는 제공자(연결은 열린 상태이고 바이트나 오류가 없는 경우)는 훨씬 긴 소켓 읽기 타임아웃까지(또는 무인 cron 실행의 경우 외부에서 프로세스를 종료할 때까지) 멈춰 있지 않고, 오래된 응답 타임아웃에 도달하면 중단된 뒤 재시도됩니다.
## 컨텍스트 압력 경고

반복 횟수 예산과 별개로, 컨텍스트 압력은 대화가 압축 임계값에 얼마나 가까워졌는지를 추적합니다. 압축 임계값에 도달하면 이전 컨텍스트가 요약됩니다.

| 진행률 | 수준 | 동작 |
|----------|-------|-------------|
| **임계값의 60% 이상** | 정보 | CLI에 청록색 진행률 표시줄이 나타나고, 게이트웨이가 안내 알림을 보냅니다 |
| **임계값의 85% 이상** | 경고 | 굵은 노란색 진행률 표시줄이 나타나고, 게이트웨이가 압축이 임박했음을 알립니다 |

CLI에서 컨텍스트 압력은 도구 출력 피드에 진행률 표시줄로 나타납니다.

```
  ◐ context ████████████░░░░░░░░ 62% to compaction  48k threshold (50%) · approaching compaction
```

메시징 플랫폼에서는 일반 텍스트 알림이 전송됩니다.

```
◐ Context: ████████████░░░░░░░░ 62% to compaction (threshold: 50% of window).
```

자동 압축이 비활성화되어 있으면 컨텍스트가 잘릴 수 있다는 경고가 표시됩니다.

컨텍스트 압력은 자동으로 처리되므로 별도의 설정이 필요하지 않습니다. 오직 사용자에게 알림을 표시하기 위해 작동하며, 메시지 스트림을 수정하거나 모델의 컨텍스트에 무언가를 삽입하지 않습니다.

## 자격 증명 풀 전략

동일한 제공업체에 여러 API 키 또는 OAuth 토큰이 있는 경우 순환 전략을 설정할 수 있습니다.

```yaml
credential_pool_strategies:
  openrouter: round_robin    # cycle through keys evenly
  anthropic: least_used      # always pick the least-used key
```

옵션은 `fill_first`(기본값), `round_robin`, `least_used`, `random`입니다. 자세한 문서는 [자격 증명 풀](/user-guide/features/credential-pools)을 참조하세요.

## 프롬프트 캐싱

활성 제공업체가 지원하는 경우 Hermes는 세션 간 프롬프트 캐싱을 자동으로 활성화합니다. 사용자가 설정할 필요는 없습니다.

**native Anthropic**, **OpenRouter**, **Nous Portal**에서 Claude를 사용할 때 Hermes는 시스템 프롬프트와 스킬 블록에 1시간 TTL(`ttl: "1h"`)의 `cache_control` 중단점을 추가합니다. 새로운 시간이 시작된 후 첫 전송에는 입력 요금이 전액 부과되고, 같은 시간 내 어느 세션에서든 이후 전송은 할인된 캐시 읽기 요금으로 캐시에서 가져옵니다. 따라서 시스템 프롬프트, 로드된 스킬 콘텐츠, 그리고 긴 컨텍스트 포함 내용의 앞부분은 첫 1시간 동안 `hermes` 세션과 분기된 하위 에이전트에서 재사용됩니다.

Qwen Cloud(Alibaba DashScope) 업스트림은 캐시 TTL을 5분으로 제한하므로 Hermes도 그곳에서는 5분 중단점 TTL을 사용합니다. 그 밖의 서드파티 경유 Claude 경로(AWS Bedrock, Azure Foundry)는 제공업체 자체 캐싱 기본값으로 대체됩니다. xAI Grok은 별도의 세션 고정 conversation-id 메커니즘을 사용합니다. [xAI 프롬프트 캐싱](/integrations/providers#xai-grok--responses-api--prompt-caching)을 참조하세요.

이를 비활성화하는 설정은 없습니다. 캐싱은 항상 켜져 있으며, 시스템 프롬프트만으로도 입력 토큰 수에서 상당한 비중을 차지하므로 단일 턴 대화에서도 비용을 절약합니다.

명시적으로 조정할 수 있는 항목은 Hermes가 Anthropic 스타일 중단점에 요청하는 캐시 TTL 등급입니다.

```yaml
prompt_caching:
  cache_ttl: "5m"   # "5m" or "1h" (Anthropic-supported tiers); other values are ignored
```

`cache_ttl`은 Hermes가 native Anthropic API, OpenRouter, Nous Portal을 통해 Claude에 추가하는 중단점 TTL을 선택합니다. Anthropic이 지원하는 두 등급(`"5m"`, `"1h"`)만 적용되며, 다른 값은 무시됩니다. 자체 제한이 있는 제공업체(예: 최대 5분인 Qwen Cloud)는 업스트림이 허용하는 값으로 계속 제한됩니다.

## 보조 모델

Hermes는 이미지 분석, 웹 페이지 요약, 브라우저 스크린샷 분석, 세션 제목 생성, 컨텍스트 압축과 같은 부가 작업에 "보조" 모델을 사용합니다. 기본값(`auxiliary.*.provider: "auto"`)에서는 모든 보조 작업을 **주 채팅 모델**로 라우팅합니다. 즉, `hermes model`에서 선택한 것과 같은 제공업체/모델을 사용합니다. 시작을 위해 별도로 설정할 필요는 없지만, 고가의 추론 모델(Opus, MiniMax M2.7 등)에서는 보조 작업이 상당한 비용을 추가할 수 있다는 점에 유의하세요. 주 모델과 관계없이 저렴하고 빠른 보조 작업을 원한다면 `auxiliary.<task>.provider`와 `auxiliary.<task>.model`을 명시적으로 설정하세요(예: 비전 및 웹 추출에 OpenRouter의 Gemini Flash 사용).

:::note "auto"가 주 모델을 사용하는 이유
이전 빌드에서는 집계기 사용자를(OpenRouter, Nous Portal) 제공업체 측의 저렴한 기본값으로 분리했습니다. 이는 예상 밖의 동작이었습니다. 집계기 구독료를 낸 사용자가 보조 트래픽을 처리하는 다른 모델을 보게 되었기 때문입니다. 이제 `auto`는 모든 사용자에게 주 모델을 사용하며, `config.yaml`의 작업별 재정의가 여전히 우선합니다(아래 [전체 보조 설정 참조](#full-auxiliary-config-reference) 참조).
:::

### 보조 모델을 대화형으로 설정하기

YAML을 직접 편집하는 대신 `hermes model`을 실행하고 메뉴에서 **"Configure auxiliary models"**를 선택하세요. 작업별 대화형 선택기가 표시됩니다.

```
$ hermes model
→ Configure auxiliary models

[ ] vision               currently: auto / main model
[ ] web_extract          currently: auto / main model
[ ] title_generation     currently: openrouter / google/gemini-3-flash-preview
[ ] tts_audio_tags       currently: auto / main model
[ ] compression          currently: auto / main model
[ ] approval             currently: auto / main model
[ ] triage_specifier     currently: auto / main model
[ ] kanban_decomposer    currently: auto / main model
[ ] profile_describer    currently: auto / main model
```

작업을 선택하고 제공업체를 고른 다음(OAuth 흐름은 브라우저를 열고, API 키 제공업체는 입력을 요청합니다) 모델을 선택하세요. 변경 사항은 `config.yaml`의 `auxiliary.<task>.*`에 저장됩니다. 주 모델 선택기와 동일한 방식이므로 새로 배울 문법이 없습니다.

첫 번째 교환 이후 Hermes가 제목을 자동 생성하지 않게 하려면 `auxiliary.title_generation.enabled: false`로 설정하세요. 수동 제목은 `/title` 및 `hermes sessions rename`을 통해 계속 사용할 수 있습니다.

### 스트림 전용 엔드포인트

일부 OpenAI 호환 엔드포인트는 비스트리밍 채팅 요청을 아예 거부합니다(예: Tencent Copilot은 HTTP 400 `"Non-stream chat request is currently not supported"`를 반환합니다). 대화형 채팅은 이미 스트리밍하지만, 보조 작업(제목 생성, 압축, 웹 추출)은 비스트리밍 호출을 사용하므로 매번 실패합니다. Hermes는 `copilot.tencent.com`을 항상 스트림 전용으로 처리합니다. 그 밖의 이러한 엔드포인트는 `auxiliary.stream_only_base_urls` 아래에 URL 부분 문자열을 나열하세요.

```yaml
auxiliary:
  stream_only_base_urls:
    - "my-stream-only-proxy.example.com"
```

일치하는 보조 호출은 `stream=True`로 전송되고 청크(도구 호출 델타 포함)는 클라이언트 측에서 집계됩니다. 다른 엔드포인트의 동작은 변경되지 않습니다.

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

### 범용 설정 패턴

Hermes의 모든 모델 슬롯(보조 작업, 압축, 대체 모델)은 동일한 세 가지 설정을 사용합니다.

| 키 | 기능 | 기본값 |
|-----|-------------|---------|
| `provider` | 인증과 라우팅에 사용할 제공업체 | `"auto"` |
| `model` | 요청할 모델 | 제공업체의 기본값 |
| `base_url` | 사용자 지정 OpenAI 호환 엔드포인트(제공업체 재정의) | 설정되지 않음 |

보조 작업 블록은 `reasoning_effort` 설정도 추가로 허용합니다.

| 키 | 기능 | 기본값 |
|-----|-------------|---------|
| `reasoning_effort` | 해당 작업의 LLM 호출에서 사고 수준: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra` | 설정되지 않음(제공업체 기본값) |

이는 전역 `agent.reasoning_effort`의 작업별 대응 설정입니다. 주 모델이 고가의 추론 모델이더라도 주 채팅 동작은 건드리지 않고 압축을 `low`로, 비전을 `none`으로 실행하여 보조 작업의 지연 시간과 비용을 줄일 수 있습니다. 이 설정은 세 가지 보조 전송 형식(chat completions, Codex Responses, Anthropic Messages) 모두에서 모든 보조 작업 블록(`vision`, `web_extract`, `compression`, `title_generation`, `curator`, `background_review`, ...)에 적용됩니다. 같은 작업에 명시적인 `extra_body.reasoning`이 있으면 간편 설정 대신 해당 값이 적용됩니다.

MoA는 유일한 예외입니다. Mixture-of-Agents의 추론 깊이는 `moa_reference`/`moa_aggregator` 보조 블록이 아니라 MoA 프리셋의 슬롯별 설정(`moa.presets.<name>.reference_models[].reasoning_effort` / `aggregator.reasoning_effort`)으로 구성합니다. [Mixture of Agents](/user-guide/features/mixture-of-agents)를 참조하세요.

```yaml
auxiliary:
  compression:
    reasoning_effort: "low"    # summaries don't need deep thinking
  vision:
    reasoning_effort: "none"   # disable thinking for image description
```

`base_url`을 설정하면 Hermes는 제공업체를 무시하고 해당 엔드포인트를 직접 호출합니다(`api_key` 또는 인증용 `OPENAI_API_KEY` 사용). `provider`만 설정하면 Hermes는 해당 제공업체의 기본 인증과 base URL을 사용합니다.

보조 작업에 사용할 수 있는 제공업체는 `auto`, `main`, 그리고 [제공업체 레지스트리](/reference/environment-variables)에 있는 모든 제공업체입니다. — `openrouter`, `nous`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `gemini`, `qwen-oauth`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `deepseek`, `nvidia`, `xai`, `xai-oauth`, `ollama-cloud`, `alibaba`, `bedrock`, `huggingface`, `arcee`, `xiaomi`, `kilocode`, `opencode-zen`, `opencode-go`, `ai-gateway`, `azure-foundry` — 또는 `providers:` 딕셔너리에 정의한 이름 있는 사용자 지정 제공업체(예: `provider: "beans"`)입니다.

:::tip MiniMax OAuth
`minimax-oauth`는 브라우저 OAuth로 로그인하므로 API 키가 필요하지 않습니다. `hermes model`을 실행하고 **MiniMax (OAuth)**를 선택해 인증하세요. 보조 작업은 자동으로 `MiniMax-M2.7-highspeed`를 사용합니다. [MiniMax OAuth 가이드](../guides/minimax-oauth.md)를 참조하세요.
:::

:::tip xAI Grok OAuth
`xai-oauth`는 SuperGrok 및 X Premium+ 구독자를 위해 브라우저 OAuth로 로그인하므로 API 키가 필요하지 않습니다. `hermes model`을 실행하고 **xAI Grok OAuth (SuperGrok / Premium+)**를 선택해 인증하세요. 동일한 OAuth 토큰이 xAI로 직접 연결되는 모든 표면(채팅, 보조 작업, TTS, 이미지 생성, 동영상 생성, 전사)에 재사용됩니다. [xAI Grok OAuth 가이드](../guides/xai-grok-oauth.md)를 참조하고, Hermes가 원격 호스트에서 실행 중이라면 [SSH를 통한 OAuth / 원격 호스트](../guides/oauth-over-ssh.md)도 참조하세요.
:::

:::warning `"main"`은 보조 작업 전용입니다
`"main"` 제공업체 옵션은 "내 주 에이전트가 사용하는 제공업체를 사용하라"는 뜻이며 `auxiliary:`, `compression:`, 기본 대체 항목(`fallback_providers:` 또는 레거시 `fallback_model:`) 내부에서만 유효합니다. 최상위 `model.provider` 설정에서는 유효한 값이 아닙니다. 사용자 지정 OpenAI 호환 엔드포인트를 사용한다면 `model:` 섹션에서 `provider: custom`을 설정하세요. 주 모델 제공업체의 전체 목록은 [AI 제공업체](/integrations/providers)를 참조하세요.
:::

### 전체 보조 설정 참조

```yaml
auxiliary:
  # Image analysis (vision_analyze tool + browser screenshots)
  vision:
    provider: "auto"           # "auto", "openrouter", "nous", "codex", "main", etc.
    model: ""                  # e.g. "openai/gpt-4o", "google/gemini-2.5-flash"
    base_url: ""               # Custom OpenAI-compatible endpoint (overrides provider)
    api_key: ""                # API key for base_url (falls back to OPENAI_API_KEY)
    timeout: 120               # seconds — LLM API call timeout; vision payloads need generous timeout
    download_timeout: 30       # seconds — image HTTP download; increase for slow connections
    max_concurrency: 8         # max concurrent image encode/resize bursts across the process
                               # (default: host CPU core count, no ceiling) — bounds only the
                               # CPU-bound encode step so a video-frame fan-out can't saturate
                               # every core and starve the event loop; LLM calls stay fully
                               # concurrent. Minimum 1; values < 1 are ignored.

  # Web page summarization + browser page text extraction
  web_extract:
    provider: "auto"
    model: ""                  # e.g. "google/gemini-2.5-flash"
    base_url: ""
    api_key: ""
    timeout: 360               # seconds (6min) — per-attempt LLM summarization

  # Dangerous command approval classifier
  approval:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30                # seconds

  # Gemini 3.1 TTS hidden audio-tag insertion
  tts_audio_tags:
    provider: "auto"
    model: ""                  # empty = main chat model
    base_url: ""
    api_key: ""
    timeout: 30

  # Context compression timeout (separate from compression.* config)
  compression:
    timeout: 120               # seconds — compression summarizes long conversations, needs more time
    # fallback_chain:           # Optional — providers to try on rate-limit / connectivity failure
    #   - provider: nous
    #     model: deepseek/deepseek-chat
    #   - provider: openrouter
    #     model: google/gemini-2.5-flash
    #     base_url: ""
    #     api_key: ""
    # max_concurrency: 2       # Optional: cap simultaneous compression LLM calls so
                               # multiple sessions don't pile retries on a degraded provider

  # Auto-generated session titles. Empty language follows the conversation;
  # set e.g. "English" or "Japanese" to pin titles to one language.
  title_generation:
    enabled: true              # set false to disable auto-title generation
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    language: ""

  # Skills hub — skill matching and search
  skills_hub:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # MCP tool dispatch
  mcp:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # Auto-generated short session titles after the first exchange
  title_generation:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    # max_concurrency: 2       # Optional: cap simultaneous title-generation calls

  # Kanban triage specifier — `hermes kanban specify <id>` (or the
  # dashboard's ✨ Specify button on Triage-column cards) uses this
  # slot to expand a one-liner into a concrete spec and promote the
  # task to `todo`. Cheap fast models work well here; spec expansion
  # is short and doesn't need reasoning depth.
  triage_specifier:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
```

:::tip
각 보조 작업에는 설정 가능한 `timeout`(초 단위)이 있습니다. 기본값은 vision 120초, web_extract 360초, approval 30초, compression 120초입니다. 보조 작업에 느린 로컬 모델을 사용한다면 이 값을 늘리세요. Vision에는 HTTP 이미지 다운로드를 위한 별도의 `download_timeout`(기본값 30초)도 있습니다. 느린 연결이나 자체 호스팅 이미지 서버를 사용한다면 이 값도 늘리세요.
:::

:::info
컨텍스트 압축에는 임계값을 위한 자체 `compression:` 블록과 모델/제공업체 설정을 위한 `auxiliary.compression:` 블록이 있습니다. 위의 [컨텍스트 압축](#context-compression)을 참조하세요. 기본 대체 체인은 최상위 `fallback_providers:` 목록을 사용합니다. [대체 제공업체](/integrations/providers#fallback-providers)를 참조하세요. 세 설정 모두 동일한 provider/model/base_url 패턴을 따릅니다.
:::
### 보조 작업별 대체 체인

각 보조 작업은 선택적으로 `fallback_chain` — Hermes가 기본 보조 제공업체가 속도 제한, 연결 문제 또는 결제 제한으로 실패할 때 시도하는 제공업체/모델 항목 목록 — 을 정의할 수 있습니다:

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openai/gpt-4o-mini
    fallback_chain:
      - provider: nous
        model: deepseek/deepseek-chat
      - provider: openrouter
        model: google/gemini-2.5-flash
```

기본 보조 제공업체(`openrouter` / `openai/gpt-4o-mini`)가 속도 제한, 연결 시간 초과 또는 결제 필요 오류를 반환하면 Hermes는 `fallback_chain`을 순서대로 순회합니다. 이미 실패한 제공업체와 일치하는 항목은 건너뛰고, 하나가 성공하거나 체인이 소진될 때까지 남은 항목을 시도합니다.

모든 대체 항목이 실패하면 최종 안전망으로 주 에이전트 모델로 돌아갑니다.

각 항목은 다른 보조 작업 설정과 동일한 세 가지 옵션을 지원합니다:

| 키 | 설명 |
|-----|-------------|
| `provider` | 제공업체 이름(`nous`, `openrouter`, `anthropic`, `gemini`, `main` 등) |
| `model` | 해당 제공업체의 모델 이름 |
| `base_url` | (선택 사항) 사용자 지정 OpenAI 호환 엔드포인트 |

`fallback_chain`은 모든 보조 작업 — `compression`, `vision`, `web_extract`, `approval`, `skills_hub`, `mcp` 등 — 에 사용할 수 있습니다.

### 보조 동시성 제한

`max_concurrency`는 전체 프로세스에서 `compression`, `title_generation` 같은 보조 작업에 대해 동시에 실행 중인 LLM 호출 수를 제한합니다. `auxiliary.vision.max_concurrency`는 제외됩니다. 이 설정은 LLM 요청이 아니라 vision의 CPU 바운드 이미지 인코딩/크기 조정 작업자만 제어하기 때문입니다. 다음과 같은 경우에 특히 유용합니다.

- 많은 세션이 동시에 백그라운드 작업을 생성할 수 있는 경우(Discord/Telegram 채널, 여러 터미널)
- 제공업체가 속도 제한을 받거나 장애를 겪고 있어 재시도가 폭주를 키울 수 있는 경우

기본값은 제한 없음입니다. 일반적인 안전 상한은 `2`입니다.

```yaml
auxiliary:
  title_generation:
    max_concurrency: 2
  compression:
    max_concurrency: 2
```

세마포어는 재시도와 대체를 포함한 전체 호출을 감싸므로, 느린 호출 하나는 제한에 한 번만 계산됩니다.

### 보조 작업의 OpenRouter 라우팅 및 Pareto Code

보조 작업이 OpenRouter로 확인되는 경우(명시적으로 지정했거나 주 에이전트가 OpenRouter를 사용할 때 `provider: "main"`을 통해 지정한 경우), 주 에이전트의 `provider_routing` 및 `openrouter.min_coding_score` 설정은 설계상 전달되지 않습니다. 각 보조 작업은 독립적이기 때문입니다. 특정 보조 작업에 OpenRouter 제공업체 선호도를 설정하거나 [Pareto Code 라우터](/integrations/providers#openrouter-pareto-code-router)를 사용하려면 작업별로 `extra_body`에 지정하세요.

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openrouter/pareto-code         # use the Pareto Code router for this task
    extra_body:
      provider:                            # OpenRouter provider routing prefs
        order: [anthropic, google]         # try these providers in order
        sort: throughput                   # or "price" | "latency"
        # only: [anthropic]                # restrict to a specific provider
        # ignore: [deepinfra]              # exclude specific providers
      plugins:                             # OpenRouter Pareto Code router knob
        - id: pareto-router
          min_coding_score: 0.5            # 0.0–1.0; higher = stronger coders
```

이 구조는 OpenRouter가 채팅 완성 요청 본문에서 허용하는 형식과 동일합니다. Hermes는 전체 `extra_body`를 그대로 전달하므로, [openrouter.ai/docs](https://openrouter.ai/docs)에 문서화된 다른 OpenRouter 요청 본문 필드도 같은 방식으로 작동합니다.

### Vision 모델 변경

이미지 분석에 Gemini Flash 대신 GPT-4o를 사용하려면 다음과 같이 설정하세요.

```yaml
auxiliary:
  vision:
    model: "openai/gpt-4o"
```

또는 환경 변수로 설정할 수 있습니다(`~/.hermes/.env`).

```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
```

### 제공업체 옵션

이 옵션은 **보조 작업 설정**(`auxiliary:`, `compression:`)과 기본 대체 항목(`fallback_providers:` 또는 레거시 `fallback_model:`)에 적용되며, 기본 `model.provider` 설정에는 적용되지 않습니다.

| 제공업체 | 설명 | 요구 사항 |
|----------|-------------|-------------|
| `"auto"` | 사용 가능한 최적의 제공업체(기본값). Vision은 OpenRouter → Nous → Codex 순으로 시도합니다. | — |
| `"openrouter"` | OpenRouter를 강제합니다 — 모든 모델(Gemini, GPT-4o, Claude 등)로 라우팅합니다. | `OPENROUTER_API_KEY` |
| `"nous"` | Nous Portal을 강제합니다. | `hermes auth` |
| `"codex"` | Codex OAuth(ChatGPT 계정)를 강제합니다. Vision을 지원합니다(gpt-5.3-codex). | `hermes model` → ChatGPT 또는 Codex Subscription |
| `"minimax-oauth"` | MiniMax OAuth(브라우저 로그인, API 키 불필요)를 강제합니다. 보조 작업에 MiniMax-M2.7-highspeed를 사용합니다. | `hermes model` → MiniMax (OAuth) |
| `"xai-oauth"` | xAI Grok OAuth(SuperGrok 또는 X Premium+ 구독자를 위한 브라우저 로그인, API 키 불필요)를 강제합니다. 동일한 OAuth 토큰이 채팅, TTS, 이미지, 동영상 및 전사를 모두 지원합니다. | `hermes model` → xAI Grok OAuth (SuperGrok / Premium+) |
| `"main"` | 활성화된 사용자 지정/기본 엔드포인트를 사용합니다. `OPENAI_BASE_URL` + `OPENAI_API_KEY`에서 가져오거나 `hermes model` / `config.yaml`을 통해 저장한 사용자 지정 엔드포인트일 수 있습니다. OpenAI, 로컬 모델 또는 OpenAI 호환 API와 함께 작동합니다. **보조 작업 전용이며 `model.provider`에는 유효하지 않습니다.** | 사용자 지정 엔드포인트 자격 증명 + 기본 URL |

기본 라우터를 우회하여 보조 작업에 사용하려는 경우, 기본 제공업체 카탈로그의 API 키 기반 제공업체도 사용할 수 있습니다. 예를 들어 `GMI_API_KEY`가 구성되어 있으면 `gmi`를 사용할 수 있고, `FIREWORKS_API_KEY`가 구성되어 있으면 `fireworks`를 사용할 수 있습니다.

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

GMI 보조 라우팅에는 GMI의 `/v1/models` 엔드포인트가 반환하는 정확한 모델 ID를 사용하세요. Fireworks 모델 ID는 `accounts/fireworks/models/glm-5p2`처럼 제공업체 고유의 슬래시 형식을 사용합니다.

### 일반적인 설정

**직접 사용자 지정 엔드포인트 사용**(`provider: "main"`보다 로컬/자체 호스팅 API에 더 명확함):
```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url`은 `provider`보다 우선하므로, 특정 엔드포인트로 보조 작업을 라우팅하는 가장 명시적인 방법입니다. 직접 엔드포인트를 재정의할 때 Hermes는 구성된 `api_key`를 사용하거나 `OPENAI_API_KEY`로 대체하며, 사용자 지정 엔드포인트에 `OPENROUTER_API_KEY`를 재사용하지 않습니다.

**Vision에 OpenAI API 키 사용:**
```yaml
# In ~/.hermes/.env:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # or "gpt-4o-mini" for cheaper
```

**Vision에 OpenRouter 사용**(모든 모델로 라우팅):
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # or "google/gemini-2.5-flash", etc.
```

**Codex OAuth 사용**(ChatGPT Pro/Plus 계정 — API 키 불필요):
```yaml
auxiliary:
  vision:
    provider: "codex"     # uses your ChatGPT OAuth token
    # model defaults to gpt-5.3-codex (supports vision)
```

**MiniMax OAuth 사용**(브라우저 로그인, API 키 불필요):
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
`hermes model`을 실행하고 **MiniMax (OAuth)**를 선택하여 로그인하면 자동으로 설정됩니다. 중국 리전에서는 기본 URL이 `https://api.minimaxi.com/anthropic`이 됩니다. 전체 안내는 [MiniMax OAuth 가이드](../guides/minimax-oauth.md)를 참고하세요.

**로컬/자체 호스팅 모델 사용:**
```yaml
auxiliary:
  vision:
    provider: "main"      # uses your active custom endpoint
    model: "my-local-model"
```

`provider: "main"`은 일반 채팅에 Hermes가 사용하는 제공업체를 그대로 사용합니다 — 이름이 지정된 사용자 지정 제공업체(예: `beans`), OpenRouter 같은 기본 제공업체 또는 레거시 `OPENAI_BASE_URL` 엔드포인트일 수 있습니다.

:::tip
Codex OAuth를 기본 모델 제공업체로 사용하는 경우 Vision은 자동으로 작동하므로 추가 설정이 필요하지 않습니다. Codex는 Vision 자동 감지 체인에 포함되어 있습니다.
:::

:::warning
**Vision에는 멀티모달 모델이 필요합니다.** `provider: "main"`을 설정했다면 엔드포인트가 멀티모달/Vision을 지원하는지 확인하세요. 그렇지 않으면 이미지 분석이 실패합니다.
:::

### 환경 변수(레거시)

보조 모델은 환경 변수로도 구성할 수 있습니다. 그러나 `config.yaml`이 권장 방법입니다. 관리하기 더 쉽고 `base_url`, `api_key`를 포함한 모든 옵션을 지원하기 때문입니다.

| 설정 | 환경 변수 |
|---------|---------------------|
| Vision 제공업체 | `AUXILIARY_VISION_PROVIDER` |
| Vision 모델 | `AUXILIARY_VISION_MODEL` |
| Vision 엔드포인트 | `AUXILIARY_VISION_BASE_URL` |
| Vision API 키 | `AUXILIARY_VISION_API_KEY` |
| 웹 추출 제공업체 | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| 웹 추출 모델 | `AUXILIARY_WEB_EXTRACT_MODEL` |
| 웹 추출 엔드포인트 | `AUXILIARY_WEB_EXTRACT_BASE_URL` |
| 웹 추출 API 키 | `AUXILIARY_WEB_EXTRACT_API_KEY` |

압축 및 대체 모델 설정은 `config.yaml`에서만 지정할 수 있습니다.

:::tip
`hermes config`를 실행하면 현재 보조 모델 설정을 확인할 수 있습니다. 기본값과 다른 경우에만 재정의 항목이 표시됩니다.
:::

## 추론 강도

모델이 응답하기 전에 수행하는 "사고"의 양을 제어합니다.

```yaml
agent:
  reasoning_effort: ""   # empty = medium. Options: none, minimal, low, medium, high, xhigh, max, ultra
```

설정하지 않으면(기본값) 추론 강도는 대부분의 작업에 적합한 균형 잡힌 수준인 `medium`으로 설정됩니다. 값을 지정하면 이를 재정의합니다. 추론 강도가 높을수록 복잡한 작업에서 더 나은 결과를 얻지만 토큰 사용량과 지연 시간이 증가합니다.

:::note OpenRouter를 통한 적응형 사고 모델(Claude 4.6+, Fable/Mythos 계열)
이 모델들은 *적응형* 사고를 사용하며 일반적인 `reasoning.effort` 필드를 허용하지 않습니다 — OpenRouter가 이를 무시합니다. Hermes는 대신 `reasoning_effort`를 OpenRouter의 `verbosity` 매개변수로 투명하게 라우팅합니다(이는 Anthropic의 `output_config.effort`에 매핑됨). 따라서 선택한 모델이 지원하는 수준에서 동일한 강도 조절 기능을 계속 사용할 수 있습니다. `none`(또는 설정하지 않음)은 모델 자체의 적응형 기본값을 사용합니다. 네이티브 Anthropic 제공업체는 이미 강도를 직접 제어하므로 영향을 받지 않습니다.
:::

:::note OpenRouter 모델 및 지원되는 강도 수준
OpenRouter를 통해 라우팅되는 다른 모델의 경우 Hermes는 실시간 모델 카탈로그의 추론 메타데이터(`supported_parameters` + 모델별 `reasoning.supported_efforts`)를 읽어 추론 제어를 전송할지 결정하고, 요청한 강도를 실제 라우트가 지원하는 가장 가까운 수준으로 제한합니다(항상 아래쪽으로만 조정 — 예를 들어 `high`에서 끝나는 라우트에서는 `ultra`가 `high`가 되며, 조용히 상향되지 않음). 새로운 추론 지원 공급업체는 Hermes 업데이트를 기다리지 않아도 자동으로 작동합니다. 카탈로그에 접근할 수 없거나 모델이 목록에 없으면 Hermes는 내장 모델 계열 목록으로 대체하고 요청한 강도를 변경 없이 전달합니다.
:::

런타임에 `/reasoning` 명령으로 추론 강도를 변경할 수도 있습니다.

```
/reasoning                # Show current effort level and display state
/reasoning high           # Set reasoning effort to high (this session only)
/reasoning high --global  # Set effort and persist to config.yaml
/reasoning none           # Disable reasoning (this session only)
/reasoning show           # Show model thinking above each response
/reasoning hide           # Hide model thinking
```

강도 변경은 기본적으로 세션 범위에 적용됩니다. 새 수준을 `agent.reasoning_effort` 기본값으로 저장하려면 `--global`을 추가하세요.

#### 모델별 추론 재정의

모델마다 서로 다른 추론 강도 수준을 설정할 수 있습니다. 복잡한 모델에는 높은 추론을 사용하고 더 빠른 모델에는 중간 수준을 사용하려는 경우에 유용합니다.

```yaml
agent:
  reasoning_effort: "medium"       # global default
  reasoning_overrides:
    "openrouter/anthropic/claude-opus-4.5": "xhigh"
    "openai/gpt-5": "low"
    "claude-sonnet-4.6": "high"    # bare model name also works
```

키 매칭은 **표기 방식에 관대합니다** — 합리적인 표기라면 모두 일치합니다.
- `claude-opus-4.5`, `claude-opus-4-5`, `claude-opus.4.5`(점과 대시는 서로 바꿔 사용할 수 있음)
- `anthropic/claude-opus-4.5`, `openrouter/anthropic/claude-opus-4.5`(제공업체 접두사는 선택 사항)
- 변형보다 정확히 일치하는 항목이 우선합니다.

:::note
`reasoning_overrides` 키에는 `hermes config set` 지원이 없습니다 — YAML 파일을 직접 편집하세요. 모델 이름에는 점이 포함되는 경우가 많고(예: `claude-opus-4.5`), 이는 CLI의 점으로 구분된 키 문법과 충돌하기 때문입니다.
:::

**해결 우선순위:**

1. 세션 범위 `/reasoning --session` 재정의(게이트웨이 전용)
2. `agent.reasoning_overrides`의 모델별 재정의(표기 방식에 관대함)
3. 전역 `agent.reasoning_effort`
4. 제공업체 기본값

재정의는 CLI 시작, 메시징 게이트웨이, Desktop/TUI, cron 작업, `/model`을 통한 세션 중 모델 전환, 대체 모델 활성화 등 모든 곳에 자동으로 적용됩니다.
## 도구 사용 적용

일부 모델은 도구 호출 의도 대신 도구 호출을 하겠다는 설명을 텍스트로 작성하는 경우가 있습니다(예: "터미널에서 이 명령을 실행하겠습니다"). 도구 사용 적용 기능은 모델이 즉시 도구를 호출하도록 유도하는 시스템 프롬프트 지침을 주입합니다.

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| 값 | 설명 |
|-------|----------|
| `"auto"` (기본값) | `gpt`, `codex`, `gemini`, `gemma`, `grok`, `glm`, `qwen`, `deepseek`와 일치하는 모델에서 활성화됩니다. 다른 모델(예: Claude)에서는 비활성화됩니다. |
| `true` | 모델과 관계없이 항상 활성화됩니다. 도구를 안정적으로 사용하는 모델이 아닌 경우 유용합니다. |
| `false` | 항상 비활성화됩니다. |
| `["gpt", "codex", "qwen", "llama"]` | 모델 이름에 나열된 문자열이 포함된 경우에만 활성화됩니다(대소문자 무시). |

### 주입되는 내용

활성화되면 시스템 프롬프트에 세 가지 지침 계층이 추가됩니다.

1. **일반 도구 사용 적용**(일치하는 모든 모델) — 모델이 즉시 도구를 호출하고, 작업을 설명하는 데 그치지 않으며, 미래에 작업하겠다는 약속으로 응답을 끝내지 않도록 지시합니다.

2. **OpenAI 실행 규율**(GPT, Codex, Grok 모델) — 부분 작업을 포기하거나, 필요한 사전 조회를 건너뛰거나, 도구를 사용하지 않고 환각하거나, 검증 없이 완료를 선언하는 등 GPT 특유의 실패 양상을 추가로 다룹니다.

3. **Google 운영 지침**(Gemini 및 Gemma 모델만 해당) — 간결성, 절대 경로, 병렬 도구 호출, 편집 전 검증을 강조합니다.

이 지침은 사용자에게 투명하게 표시되며 시스템 프롬프트에만 영향을 줍니다. Claude처럼 이미 도구를 안정적으로 사용하는 모델에는 필요하지 않으므로 `"auto"`에서는 제외됩니다.

### 활성화 시점

기본 `auto` 목록에 없는 모델을 사용하면서 모델이 자주 실제 도구 호출 대신 "무엇을 하겠다"고 설명한다면 `tool_use_enforcement: true`로 설정하거나 모델 문자열을 목록에 추가하세요.

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## 도구 루프 보호 장치

Hermes는 에이전트가 생산적이지 않은 도구 호출 루프에 빠졌는지 감지합니다. 여기에는 같은 도구 호출의 반복 실패, 같은 도구의 반복 실패, 진행 없이 동일한 결과를 반환하는 멱등 호출이 포함됩니다. 기본적으로 Hermes는 도구 결과에 **경고**를 주입해 모델이 스스로 수정하도록 하며, CLI/TUI를 지켜보는 사람이 개입할 수 있도록 강제 중단은 하지 않습니다.

게이트웨이 또는 서버를 자동으로 운영하는 배포에서는 하드 스톱을 활성화하여, 멈춘 에이전트가 반복 호출로 반복 횟수 예산을 소모하지 않고 회로 차단되도록 하세요.

```yaml
tool_loop_guardrails:
  warnings_enabled: true       # inject warnings into tool results (default: true)
  hard_stop_enabled: false     # also BLOCK the call past the hard-stop threshold (default: false)
  warn_after:
    exact_failure: 2           # identical failing call repeated N times
    same_tool_failure: 3       # same tool failing N times (different args)
    idempotent_no_progress: 2  # same result, no progress, N times
  hard_stop_after:
    exact_failure: 5
    same_tool_failure: 8
    idempotent_no_progress: 5
  loop_caps:
    max_web_searches: 50       # max web_search calls per turn (0 = unlimited)
    max_subagents: 50          # max subagents spawned per turn (0 = unlimited)
```

대화형 세션에는 사람이 개입할 수 있으므로 `hard_stop_enabled`의 기본값은 `false`입니다. 무인 배포(게이트웨이, cron, kanban worker)에서는 반복 호출이 경고만 표시하는 것이 아니라 차단되도록 `true`로 설정하세요. [Docker / 무인 배포](docker.md)도 참고하세요.

### 턴별 폭주 루프 상한

실패 기반 임계값과 별도로 `loop_caps`는 단일 에이전트 루프(턴)에서 수행할 수 있는 `web_search` 호출 및 서브에이전트 생성 횟수의 상한을 설정합니다. 카운터는 매 턴 시작 시 초기화되므로 정상적인 다중 턴 세션이 고갈되지 않습니다. 하지만 단일 턴이 무제한 검색 또는 위임 루프로 치닫는 경우에는 중지됩니다. 이 상한은 항상 활성화되며 `hard_stop_enabled`와 관계없이 적용됩니다. 한 턴에 수십 번 웹 검색을 하거나 수십 개의 서브에이전트를 생성하는 것은 이미 비정상적이므로 기본값은 낮게 설정되어 있습니다. 상한에 도달하면 해당 도구 호출이 설명 메시지와 함께 차단되고, 턴은 남은 예산을 소모하지 않고 정상적으로 종료됩니다. 상한을 완전히 비활성화하려면 값을 `0`으로 설정하세요.

하나의 `delegate_task` 배치는 각 작업을 `max_subagents`에 포함합니다(작업 3개 배치는 3개를 사용). 따라서 상한은 `delegate_task` 호출 횟수가 아니라 실제로 생성된 서브에이전트를 추적합니다.

이는 Claude Code의 세션별 WebSearch 및 서브에이전트 상한(v2.1.212)을 본뜬 것으로, 해당 상한도 기본값이 200이며 `/clear`에서 초기화됩니다.

## TTS 구성

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts" | "kittentts" | "piper" | "deepinfra"
  speed: 1.0                    # Global speed multiplier (fallback for all providers)
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
    speed: 1.0                  # Speed multiplier (converted to rate percentage, e.g. 1.5 → +50%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    speed: 1.0                  # Speed multiplier (clamped to 0.25–4.0 by the API)
    base_url: "https://api.openai.com/v1"  # Override for OpenAI-compatible TTS endpoints
  minimax:
    speed: 1.0                  # Speech speed multiplier
    # base_url: ""              # Optional: override for OpenAI-compatible TTS endpoints
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral (default)
  gemini:
    model: "gemini-2.5-flash-preview-tts"   # or gemini-3.1-flash-tts-preview
    voice: "Kore"               # 30 prebuilt voices: Zephyr, Puck, Kore, Enceladus, etc.
    audio_tags: false           # Hidden Gemini 3.1 TTS audio-tag insertion
    persona_prompt_file: ""      # Optional Markdown/text file with Gemini voice direction
  xai:
    voice_id: "eve"             # xAI TTS voice
    language: "en"              # ISO 639-1
    sample_rate: 24000
    bit_rate: 128000            # MP3 bitrate
    # base_url: "https://api.x.ai/v1"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

이는 `text_to_speech` 도구와 음성 모드의 음성 응답(CLI 또는 메시징 게이트웨이에서의 `/voice tts`)을 모두 제어합니다.

**속도 대체 계층:** 공급자별 속도(예: `tts.edge.speed`) → 전역 `tts.speed` → 기본값 `1.0`. 모든 공급자에 동일한 속도를 적용하려면 전역 `tts.speed`를 설정하고, 세밀하게 조정하려면 공급자별로 재정의하세요.

## 표시 설정

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # Enable /verbose slash command in messaging gateway
  focus_view: false       # CLI focus view (/focus) — reduced output, display-only
  platforms: {}           # Per-platform display overrides (see below)
  interim_assistant_messages: true  # Gateway: send natural mid-turn assistant updates as separate messages
  show_commentary: true   # Codex models: deliver commentary-channel progress narration as visible mid-turn updates
  skin: default           # Built-in or custom CLI skin (see user-guide/features/skins)
  personality: ""         # Legacy cosmetic field still surfaced in some summaries
  compact: false          # Compact output mode (less whitespace)
  cli_multiline_shortcuts: true  # CLI: Ctrl+J, \ + Enter, and supported Shift+Enter insert newlines (false = legacy c-j submit fallback)
  resume_display: full    # full (show previous messages on resume) | minimal (one-liner only)
  bell_on_complete: false # Play terminal bell when agent finishes (great for long tasks)
  show_reasoning: true    # Show model reasoning/thinking above each response (default: true; toggle with /reasoning show|hide)
  streaming: false        # Stream tokens to terminal as they arrive (real-time output)
  show_cost: false        # Show estimated $ cost in the CLI status bar
  timestamps: false       # When true, prefixes user and assistant labels with timestamps in the CLI / TUI transcript
  timestamp_format: "%H:%M"  # strftime format for those timestamps (e.g. "%b-%d %H:%M" for month-day)
  tool_preview_length: 0  # Max chars for tool call previews (0 = no limit, show full paths/commands)
  turn_summary: true      # CLI only: print a one-line post-turn accounting footer after each interactive turn
  spinner_token_flow: true # CLI only: append live cumulative turn tokens to the spinner timer
  runtime_footer:         # Gateway: append a runtime-context footer to final replies
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  file_mutation_verifier: true    # Append an advisory footer when write_file/patch calls failed this turn
  credits_notices: true   # Nous credits status-bar notices (usage bands, grant-spent, depleted). false = silence them; /usage still works
  cli_rebuild_scrollback_on_redraw: false  # Classic CLI: also wipe terminal scrollback (CSI 3J) on /redraw / Ctrl+L / width-change resize recovery. Enable when a terminal/tmux stack stamps stale prompt chrome into scrollback on maximize/restore.
  language: en            # UI language for static messages (approval prompts, some gateway replies). en | zh | zh-hant | ja | de | es | fr | tr | uk | af | ko | it | ga | pt | ru | hu
```

### 턴 요약 및 스피너 토큰 흐름

`display.turn_summary`(기본값 `true`)는 각 **대화형 CLI** 턴이 실제로 수행한 작업을 요약하는 흐릿한 결산 한 줄을 출력합니다.

```
⋯ 12.4s · edited 2 files +18 -3 · read 4 files · ran 3 commands
```

집계는 CLI가 이미 수신하는 도구 진행 피드에서 확인하므로 추가 비용이 들지 않습니다. 세부 사항은 다음과 같습니다.

- 실제 턴 소요 시간(1분이 지나면 `2m05s`)을 표시합니다.
- 도구 호출은 동사별로 묶이며(`edited`, `read`, `ran`, `searched` 등) 올바른 단복수 형태를 사용합니다. 엄선된 동사가 없는 플러그인/MCP 도구는 `called N tools`로 합쳐집니다.
- `+X -Y` 줄 차이는 도구 결과에 이미 diff가 보고된 경우(현재는 `patch`)에만 표시됩니다. Hermes는 줄 수를 계산하기 위해 git을 실행하지 않으므로 `write_file` 편집은 줄 차이 없이 집계됩니다.
- **실패한 도구 호출은 집계하지 않습니다** — 거부된 쓰기 작업은 성공한 편집으로 표시되지 않습니다(보완 경고는 [파일 변경 검증기](#file-mutation-verifier) 참고).
- 긴 턴은 동사 세그먼트 네 개와 `+N more` 꼬리로 제한되어 줄바꿈이 발생하지 않습니다.
- 도구 호출이 없는 빠른 턴은 아무것도 출력하지 않습니다.

`display.spinner_token_flow`(기본값 `true`)는 스피너의 실시간 타이머에 해당 턴에서 누적된 출력 토큰 수를 추가합니다.

```
  ⚡ Reading cli.py  (  2.3s · ↓ 1.2k tok)
```

수치는 턴별로 계산됩니다(세션 총량은 턴 시작 시 기준값으로 설정). 턴의 각 API 호출에서 사용량이 보고될 때마다 업데이트됩니다. 첫 사용량 보고가 도착하기 전에는 아무것도 표시되지 않으므로 오해의 소지가 있는 `↓ 0 tok`이 나타나지 않습니다.

두 키 모두 표시 전용이며 CLI 전용입니다. quiet 모드, `display.tool_progress`가 `off`인 경우, 단일 쿼리/`-Q` 배치 실행, 게이트웨이/메시징 화면에서는 표시되지 않습니다(게이트웨이/메시징 화면은 대신 `display.runtime_footer`를 사용합니다). 끄려면 각 키를 `false`로 설정하세요.

### 파일 변경 검증기

`display.file_mutation_verifier`가 `true`(기본값)이면, 해당 턴에 `write_file` 또는 `patch` 호출이 실패했고 같은 경로에 대한 성공적인 쓰기로 대체되지 않은 경우 Hermes가 어시스턴트의 최종 응답에 한 줄의 안내를 추가합니다. 이는 매번 `git status`를 직접 실행하지 않아도 "병렬 패치 묶음 중 절반이 조용히 실패했는데 모델이 성공했다고 요약하는" 유형의 과장 주장을 감지합니다.

예시 푸터:

```
⚠️ File-mutation verifier: 3 file(s) were NOT modified this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm.
  • concepts/automatic-organization.md — [patch] Could not find match for old_string
  • concepts/lora.md — [patch] Could not find match for old_string
  • concepts/rag-pipeline.md — [patch] Could not find match for old_string
```

`file_mutation_verifier: false`(또는 `HERMES_FILE_MUTATION_VERIFIER=0`)로 설정하면 푸터가 표시되지 않습니다. 검증기는 턴 종료 시 실제 실패가 남아 있을 때만 작동합니다. 같은 턴에 실패한 패치를 재시도해 해당 파일에 성공하면 해당 파일에는 푸터가 표시되지 않습니다.

**모델의 요약보다 검증기를 신뢰하세요.** 이 푸터는 위에 어시스턴트가 완료했다고 말했더라도 목록에 있는 파일이 디스크에서 수정되지 않았다는 의미입니다. 일반적인 원인은 다음과 같습니다.

- **쓰기 거부** — 경로가 자격 증명 거부 목록에 있거나 `HERMES_WRITE_SAFE_ROOT` 외부에 있습니다(파일 쓰기 안전성은 [파일 쓰기 안전성](./security.md#file-write-safety) 참고).
- **패치 불일치** — `old_string`이 파일의 내용과 일치하지 않습니다.
- **구문 게이트** — 후보 내용이 쓰기 전에 JSON/YAML/TOML 검증에 실패했습니다.

쓰기가 차단된 경우의 예시 푸터:

```
⚠️ File-mutation verifier: 2 file(s) were NOT modified this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm.
  • ~/.hermes/cron/jobs.json — [patch] Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (/path/to/project)
  • ~/.hermes/scripts/monitor.py — [write_file] Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (/path/to/project)
```

Hermes 상태(예: `~/.hermes/` 아래의 cron 작업, 스킬, 스크립트)에 쓰기가 실패하면 환경에 `HERMES_WRITE_SAFE_ROOT`가 설정되어 있는지 확인하세요. cron 변경에는 `jobs.json`을 직접 패치하는 대신 `cronjob` 도구 또는 `hermes cron edit`를 사용하세요.
### 정적 메시지의 UI 언어

`display.language` 설정은 일부 정적 사용자 대면 메시지, 즉 CLI 승인 프롬프트와 일부 게이트웨이 슬래시 명령 응답(예: 재시작 드레이닝 알림, "승인 만료", "목표 삭제")을 번역합니다. 에이전트 응답, 로그 줄, 도구 출력, 오류 트레이스백, 슬래시 명령 설명은 번역하지 않으며 영어로 유지됩니다. 에이전트가 다른 언어로 응답하게 하려면 프롬프트나 시스템 메시지에서 해당 언어를 지정하세요.

지원되는 값: `en` (기본값), `zh` (간체 중국어), `zh-hant` (번체 중국어), `ja` (일본어), `de` (독일어), `es` (스페인어), `fr` (프랑스어), `tr` (터키어), `uk` (우크라이나어), `af` (아프리칸스어), `ko` (한국어), `it` (이탈리아어), `ga` (아일랜드어), `pt` (포르투갈어), `ru` (러시아어), `hu` (헝가리어). 알 수 없는 값은 영어로 대체됩니다.

세션별로 `HERMES_LANGUAGE` 환경 변수를 사용해 이 값을 설정할 수도 있으며, 이 변수는 구성 값보다 우선합니다.

```yaml
display:
  language: zh   # CLI approval prompts appear in Chinese
```

| 모드 | 표시되는 내용 |
|------|-------------|
| `off` | 무음 — 최종 응답만 표시 |
| `new` | 도구가 변경될 때만 도구 표시기 표시 |
| `all` | 짧은 미리보기와 함께 모든 도구 호출 표시(기본값) |
| `verbose` | 전체 인수, 결과, 디버그 로그 표시 |

CLI에서는 `/verbose`로 이 모드들을 순환할 수 있습니다. 메시징 플랫폼(Telegram, Discord, Slack 등)에서 `/verbose`를 사용하려면 위의 `display` 섹션에서 `tool_progress_command: true`로 설정하세요. 그러면 명령이 모드를 순환하고 구성을 저장합니다.

도구 진행 표시를 사용하려면 진행 업데이트를 안전하게 표시할 수 있는 게이트웨이 어댑터가 필요합니다. Signal을 포함해 메시지 편집을 지원하지 않는 플랫폼은 `/verbose`가 `off`가 아닌 모드를 저장하더라도 도구 진행 말풍선을 표시하지 않습니다.

### 포커스 뷰(`/focus`, CLI + TUI)

`display.focus_view: true`는 답변만 보고 진행 과정을 모두 확인할 필요가 없을 때 사용하는 축약 출력 모드인 **포커스 뷰**를 활성화합니다. 이는 별도의 억제 경로가 아니라 동일한 `tool_progress` 기능 위에 얹힌 얇은 계층입니다.

- 활성화하면 `tool_progress`를 `off`로 고정하고 이전 모드를 `display.focus_saved_tool_progress`에 저장합니다.
- `/focus off`를 사용하면 해당 모드를 정확히 복원하므로 `/verbose verbose` 설정도 원래대로 돌아옵니다.
- 각 완료된 턴의 끝에는 흐린 복구 줄이 표시됩니다 — `⋯ 7 tool lines hidden · /focus off to show` — 이 줄은 *포커스 전* 모드를 기준으로 집계되므로, 이미 숨겨 두었던 줄까지 숨겼다고 잘못 표시하지 않습니다.
- 축약 모드가 보이지 않는 상태가 되지 않도록 프롬프트 툴킷 CLI와 Ink TUI 모두 상태 표시줄에 영구적인 `◉ focus` 배지를 표시합니다.
- 포커스가 켜진 상태에서 `/verbose`를 순환하면 모드가 `/verbose`로 다시 넘어가고 배지가 지워집니다.

포커스 뷰는 **표시 전용**입니다. 대화 기록, 시스템 프롬프트, 도구 스키마 또는 요청 페이로드를 절대 수정하지 않습니다. 숨겨진 세부 정보는 화면에서만 억제되고 삭제되지 않으며, 프롬프트 캐싱은 전혀 영향을 받지 않습니다.

### 런타임 메타데이터 푸터(게이트웨이 전용)

`display.runtime_footer.enabled: true`이면 Hermes가 각 게이트웨이 턴의 **최종** 메시지에 작은 런타임 컨텍스트 푸터를 덧붙입니다. 현재 푸터에는 모델, 컨텍스트 윈도 비율, 현재 작업 디렉터리가 표시될 수 있습니다. 기본값은 꺼져 있으며, 모든 응답에 이 출처 정보를 포함하려는 팀은 게이트웨이별로 켤 수 있습니다.

```yaml
display:
  runtime_footer:
    enabled: true
    fields: ["model", "context_pct", "cwd"]   # order shown; drop any to hide
```

지원되는 필드:

| 필드 | 렌더링 내용 | 예시 |
| --- | --- | --- |
| `model` | 공급자 접두사를 제거한 모델 ID | `gpt-5.4` |
| `context_pct` | 마지막 호출의 컨텍스트 점유율(백분율) | `5%` |
| `latency` | 턴의 실제 경과 시간 | `22s`, `1m05s` |
| `cwd` | 홈 디렉터리 기준 작업 디렉터리 | `~` |

기본 필드 집합은 `["model", "context_pct", "cwd"]`입니다. `latency`는 선택 사항이며, 사용하려면 `fields`에 추가하세요. 데이터를 사용할 수 없는 필드는 빈 슬롯을 표시하는 대신 조용히 건너뜁니다.

`/footer` 슬래시 명령은 모든 세션에서 런타임에 이 기능을 전환합니다.

Telegram/Discord/Slack 응답에 추가되는 푸터 예시:

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

턴의 **최종** 메시지에만 푸터가 추가되며, 중간 업데이트는 그대로 깔끔하게 유지됩니다.

### 플랫폼별 진행 표시 재정의

플랫폼마다 적합한 상세 수준이 다릅니다. `display.platforms`를 사용해 플랫폼별 모드를 설정하세요.

```yaml
display:
  tool_progress: all          # global default
  platforms:
    signal:
      tool_progress: 'off'    # Signal cannot currently display tool-progress bubbles
    telegram:
      tool_progress: verbose  # detailed progress on Telegram
    slack:
      tool_progress: 'off'    # quiet in shared Slack workspace
```

재정의가 없는 플랫폼은 전역 `tool_progress` 값으로 대체됩니다. 유효한 플랫폼 키: `telegram`, `discord`, `slack`, `signal`, `whatsapp`, `matrix`, `mattermost`, `email`, `sms`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`. 이전 버전의 `display.tool_progress_overrides` 키도 하위 호환성을 위해 계속 로드되지만 더 이상 사용되지 않으며, 처음 로드될 때 `display.platforms`로 마이그레이션됩니다.

Signal은 플랫폼별로 설정을 저장할 수 있으므로 유효한 플랫폼 키로 등록되어 있지만, 현재 Signal 어댑터는 전송한 메시지를 편집할 수 없고 도구 진행 말풍선도 렌더링하지 않습니다. Signal의 `tool_progress`는 `off`로 유지하세요. 각 도구 호출을 실시간으로 확인하려면 CLI나 메시지 편집이 가능한 메시징 플랫폼을 사용하세요.

`interim_assistant_messages`는 게이트웨이 전용입니다. 활성화하면 Hermes가 턴 중간에 완료된 어시스턴트 업데이트를 별도의 채팅 메시지로 보냅니다. 이는 `tool_progress`와 독립적이며 게이트웨이 스트리밍이 필요하지 않습니다.

`show_commentary`(기본값 `true`)는 Codex Responses 모델의 commentary 채널을 제어합니다. 이 채널은 이러한 모델이 비공개 추론과 함께 생성하는 다듬어진 진행 서술입니다. 활성화하면 완료된 각 commentary 메시지가 턴 중간에 표시되는 업데이트로 전달됩니다(게이트웨이에서는 `interim_assistant_messages`도 필요). 추가 서술이 번거롭다면 `false`로 설정하세요. 그러면 commentary가 reasoning 채널로 대체되고 `show_reasoning`이 활성화된 경우에만 표시됩니다.

## 개인정보 보호

```yaml
privacy:
  redact_pii: false  # Strip PII from LLM context (gateway only)
```

`redact_pii`가 `true`이면 게이트웨이는 지원되는 플랫폼에서 LLM으로 보내기 전에 시스템 프롬프트에서 개인 식별 정보를 삭제합니다.

| 필드 | 처리 방식 |
|-------|-----------|
| 전화번호(WhatsApp/Signal의 사용자 ID) | `user_<12-char-sha256>`로 해시 |
| 사용자 ID | `user_<12-char-sha256>`로 해시 |
| 채팅 ID | 숫자 부분을 해시하고 플랫폼 접두사는 유지(`telegram:<hash>`) |
| 홈 채널 ID | 숫자 부분을 해시 |
| 사용자 이름/사용자명 | **영향 없음**(사용자가 선택하며 공개적으로 표시됨) |

**플랫폼 지원:** 삭제는 WhatsApp, Signal, Telegram에 적용됩니다. Discord와 Slack은 멘션 시스템(`<@user_id>`)에서 LLM 컨텍스트에 실제 ID가 필요하므로 제외됩니다.

해시는 결정적입니다. 같은 사용자는 항상 같은 해시로 매핑되므로 모델은 그룹 채팅에서 여전히 사용자를 구분할 수 있습니다. 라우팅과 전달에는 내부적으로 원래 값이 사용됩니다.

## 음성-텍스트 변환(STT)

```yaml
stt:
  enabled: true                # Auto-transcribe inbound voice messages (default: true)
  echo_transcripts: true       # Post raw transcripts back to the chat as 🎙️ "..." (default: true)
  provider: "local"            # "local" | "groq" | "openai" | "mistral" | "xai" | "elevenlabs" | "deepinfra" | ...
  language: "en"               # GLOBAL language hint for every provider (per-provider language wins); set "" for auto-detect
  cloud_trim_silence: true     # trim long pauses with ffmpeg before uploading to a cloud provider (default: true)
  cloud_trim_threshold_db: -40 # audio quieter than this counts as silence
  cloud_trim_keep_ms: 300      # how much of each pause survives the trim (keeps natural pacing)
  # prompt: "Hermes, Teknium, Nous Research, kanban"   # Static vocabulary hint (see below)
  local:
    model: "base"              # tiny, base, small, medium, large-v3
    language: ""               # per-provider override of stt.language
    initial_prompt: ""         # optional whisper prompt to bias vocabulary/script (e.g. Simplified Chinese)
    vad: true                  # Silero VAD filter (default on) — silence never reaches whisper; false = raw behavior (music/ambient)
    vad_min_silence_ms: 500    # min silence (ms) that splits speech chunks when vad is on
    no_speech_prob_threshold: 0.6  # drop a segment only when no_speech_prob > this...
    logprob_threshold: -1.0        # ...AND avg_logprob < this (both must hit — quiet real speech survives)
    unload_after_idle_seconds: 0   # 0=never unload (default); e.g. 300 = release the model after 5min idle
  groq:
    language: ""               # per-provider override of stt.language
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe | gpt-transcribe
    language: ""               # per-provider override of stt.language
  # model: "whisper-1"         # Legacy fallback key still respected
```

언어 해석은 **모든** STT 공급자(local, groq, openai, mistral, xai, elevenlabs, deepinfra, command 공급자 및 플러그인)에서 동일합니다: `stt.<provider>.language` → `stt.language` → `HERMES_LOCAL_STT_LANGUAGE` 환경 변수 → 공급자 자동 감지. **기본값은 `stt.language: "en"`**입니다. Whisper의 자동 감지는 짧거나 억양이 있는 클립을 자주 잘못 식별하며, 이 때문에 음성 메모가 잘못된 언어로 전사되는 문제가 발생합니다. 영어가 아닌 언어를 사용하는 사람은 `stt.language`를 자신의 언어 코드(예: `"es"`, `"zh"`, `"uk"`)로 한 번 설정하세요. 다국어 사용을 위해 자동 감지를 복원하려면 `""`로 설정합니다.

게이트웨이가 에이전트를 위해 음성 메모를 전사하되 원시 전사 내용을 채팅에 게시하지 않아야 한다면(예: 고객 대상 WhatsApp 봇) `stt.echo_transcripts: false`로 설정하세요.

공급자별 동작:

- `local`은 컴퓨터에서 실행되는 `faster-whisper`를 사용합니다. `pip install faster-whisper`로 별도 설치하세요. 무음 환각 방지는 기본적으로 켜져 있습니다. Silero VAD 필터가 무음/잡음이 Whisper에 도달하지 않도록 하고, 교차 창 조건화는 비활성화되며, 모델 자체가 음성이 아닐 가능성이 높다고 표시한 세그먼트 중 신뢰도도 낮은 세그먼트는 삭제됩니다. 비음성 오디오(음악, 주변 소리)를 원시 동작으로 전사하려면 `stt.local.vad: false`로 설정하세요. 지연 시간이 짧은 전사를 위해 음성 메시지 사이에도 모델이 메모리에 로드된 상태로 유지됩니다. 유휴 상태일 때 모델을 자동으로 해제하려면 `stt.local.unload_after_idle_seconds`(예: 5분은 `300`)를 설정하세요. 이렇게 하면 CUDA 호스트에서 GPU 메모리가 해제됩니다(로컬 LLM이 GPU를 공유할 때 가장 큰 이점). CPU에서는 프로세스가 메모리를 다시 사용할 수 있게 되지만, 다른 작업에 해당 공간이 필요해질 때까지 OS에 보이는 메모리 사용량이 줄어들지 않을 수 있습니다. 다음 음성 메시지가 오면 모델이 투명하게 다시 로드됩니다.
- `groq`는 Groq의 Whisper 호환 엔드포인트를 사용하며 `GROQ_API_KEY`를 읽습니다. 자동 감지를 건너뛰고 지연 시간을 줄이려면 `stt.groq.language`(또는 전역 `HERMES_LOCAL_STT_LANGUAGE` 환경 변수)를 전달하세요.
- `openai`는 OpenAI 음성 API를 사용하며 `VOICE_TOOLS_OPENAI_KEY`를 읽습니다.

클라우드 공급자(groq, openai, mistral, xai, elevenlabs, deepinfra)는 기본적으로 `ffmpeg`가 설치되어 있으면 **업로드 전 무음 제거**를 적용합니다. 음성 메모의 긴 무음 구간은 파일 업로드 전에 클라이언트 측에서 축약되며, 각 구간의 `cloud_trim_keep_ms`만큼은 유지되어 자연스러운 속도가 보존됩니다. 오디오가 짧아지면 업로드가 빨라지고 오디오 분당 과금이 줄어들며 원격 모델의 무음 환각도 감소합니다. 12초보다 짧은 클립은 무음 제거를 완전히 건너뜁니다(이 경우 절약 효과가 없고 여러 공급자가 요청당 최소 요금을 부과하기 때문입니다). 무음 제거는 최선의 노력 방식입니다. ffmpeg가 없거나, 제거에 실패하거나, 클립 대부분이 무음이거나, 제거로 절약되는 시간이 약 10% 미만이면 원본 파일을 수정 없이 업로드합니다. 원본을 항상 업로드하려면(예: 클라우드 공급자를 통해 음악이나 주변 소리를 전사할 때) `stt.cloud_trim_silence: false`로 설정하세요. command 유형 및 플러그인 공급자는 제거된 오디오를 사용하지 않습니다.

요청한 공급자를 사용할 수 없으면 Hermes는 다음 순서로 자동 대체합니다: `local` → `groq` → `openai`.

Groq 및 OpenAI 모델 재정의는 환경 변수로 지정합니다.

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```
### 전사

`stt.prompt`는 Whisper 계열 STT 백엔드가 잘못 인식할 수 있는 고유명사, 제품명, 전문 용어를 위한 선택적 정적 힌트입니다:

```yaml
stt:
  provider: "local"
  prompt: "Hermes, Teknium, Nous Research, kanban, Ollama"
```

**구성.** 구성 값이 기본값입니다. 플러그인이 [`pre_transcription`](/user-guide/features/hooks#pre_transcription) 훅을 등록하면 그 위에 변경 사항을 적용하며, 필드별로 마지막 작성자가 승리합니다. 여러 플러그인의 힌트는 결정론적으로 결합됩니다. 플러그인 검색은 플러그인 ID의 정렬 순서로 플러그인을 로드하고, 각 플러그인의 콜백은 등록된 순서대로 실행되므로 동일한 플러그인 집합은 항상 동일한 최종 프롬프트를 만듭니다. 훅이 `prompt`에 빈 문자열을 반환하면 해당 요청의 구성 프롬프트가 지워집니다. 훅은 `language`와 `model`도 재정의할 수 있으며, `file_path`는 읽기 전용이므로 변경하려는 시도는 로그에 기록된 뒤 무시됩니다. 등록된 훅이 없고 `stt.prompt`도 설정되지 않았다면 전송되는 요청은 이전 릴리스와 동일합니다.

**프로바이더 지원.**

| 프로바이더 | 프롬프트 매개변수 | 동작 |
|----------|-----------------|----------|
| `local` (faster-whisper) | `initial_prompt` | 변경 없이 로컬 모델로 전달 |
| `openai` | `prompt` | 전사 요청에 변경 없이 전달 |
| `groq` | `prompt` | 전사 요청에 변경 없이 전달 |
| `mistral` | `prompt` | 전사 요청에 변경 없이 전달 |
| `deepinfra` | `prompt` | OpenAI 호환 경로를 통해 변경 없이 전달 |
| `xai` | 지원되지 않음 | DEBUG에 기록하고 프롬프트 없이 요청 진행 |
| `elevenlabs` | 지원되지 않음 | DEBUG에 기록하고 프롬프트 없이 요청 진행 |
| `local_command` | 지원되지 않음 | DEBUG에 기록하고 프롬프트 없이 요청 진행 |
| `stt.providers.<name>` with `type: command` | 지원되지 않음 | DEBUG에 기록하고 프롬프트 없이 요청 진행 |
| 플러그인이 등록한 프로바이더 | `transcribe(**extra)` kwargs의 `prompt` | 프롬프트가 설정된 경우에만 전송되므로, 이 키가 없던 프로바이더는 호출이 변경되지 않음 |

**길이.** Whisper 계열 모델은 프롬프트의 마지막 약 224개 토큰만 조건으로 사용합니다. Whisper 계열 백엔드(`local`, `openai`, `groq`, `deepinfra`)의 경우 Hermes가 클라이언트 측에서 이 제한을 적용합니다. 마지막 프롬프트가 너무 길면 로그에 경고를 기록하고 끝부분으로 잘라내므로, 프롬프트 길이 때문에 요청이 오류로 끝나지 않습니다. 다른 백엔드(`mistral`, 플러그인 프로바이더)는 프롬프트를 변경 없이 받아 자체 검증을 담당합니다. 어느 경우든 힌트는 짧고 구체적으로 작성하세요.

:::warning 프롬프트는 오디오와 함께 업로드됩니다
최종 프롬프트는 오디오 파일과 함께 구성된 STT 프로바이더로 전송됩니다. `stt.prompt`와 `pre_transcription` 훅이 반환하는 값 어디에도 비밀 정보나 세션에서 파생된 컨텍스트를 넣지 마세요. 특히 프로바이더가 로컬 `faster-whisper`가 아닌 호스팅 API인 경우 더욱 주의해야 합니다.
:::

## 음성 모드 (CLI)

```yaml
voice:
  record_key: "ctrl+b"         # Push-to-talk key inside the CLI
  max_recording_seconds: 120    # Hard stop for long recordings
  auto_tts: false               # Enable spoken replies automatically when /voice on
  beep_enabled: true            # Play record start/stop beeps in CLI voice mode
  beep_volume: 0.3              # Beep amplitude (0.0-1.0); raise it on quiet systems / headphones
  silence_threshold: 200        # RMS threshold for speech detection
  silence_duration: 3.0         # Seconds of silence before auto-stop
```

CLI에서 `/voice on`을 사용해 마이크 모드를 활성화하고, `record_key`로 녹음을 시작하거나 중지하며, `/voice tts`로 음성 응답을 전환하세요. 전체 설정 방법과 플랫폼별 동작은 [음성 모드](/user-guide/features/voice-mode)를 참조하세요.

## 스트리밍

전체 응답을 기다리는 대신 토큰이 도착하는 대로 터미널이나 메시징 플랫폼으로 스트리밍합니다.

### CLI 스트리밍

```yaml
display:
  streaming: true         # Stream tokens to terminal in real-time
  show_reasoning: true    # Also stream reasoning/thinking tokens (optional)
```

활성화하면 응답이 스트리밍 상자 안에 토큰 단위로 표시됩니다. 도구 호출은 계속 조용히 수집됩니다. 프로바이더가 스트리밍을 지원하지 않으면 자동으로 일반 표시 방식으로 대체됩니다.

### 게이트웨이 스트리밍 (Telegram, Discord, Slack)

```yaml
streaming:
  enabled: true           # Enable progressive message editing (default: false)
  transport: auto         # "auto" (default) | "edit" (progressive message editing) | "off"
  edit_interval: 0.8      # Seconds between message edits (default: 0.8)
  buffer_threshold: 24    # Characters before forcing an edit flush (default: 24)
  cursor: " ▉"            # Cursor shown during streaming
  fresh_final_after_seconds: 0    # Opt in to fresh final (Telegram) when preview is this old
```

활성화하면 봇은 첫 토큰에서 메시지를 보낸 다음, 더 많은 토큰이 도착할 때마다 메시지를 점진적으로 수정합니다. 메시지 수정을 지원하지 않는 플랫폼(Signal, Email, Home Assistant)은 첫 시도에서 자동으로 감지되며, 메시지가 쏟아지지 않도록 해당 세션의 스트리밍이 정상적으로 비활성화됩니다.

점진적인 토큰 수정 없이 턴 중간에 별도의 자연스러운 어시스턴트 업데이트를 보내려면 `display.interim_assistant_messages: true`를 설정하세요.

**오버플로 처리:** 스트리밍된 텍스트가 플랫폼의 메시지 길이 제한(약 4096자)을 초과하면 현재 메시지를 완료하고 새 메시지를 자동으로 시작합니다.

**새 최종 메시지 (Telegram):** Telegram의 `editMessageText`는 원본 메시지의 타임스탬프를 유지하므로, 스트리밍 응답이 오래 실행되면 완료 후에도 첫 토큰의 타임스탬프가 남습니다. `fresh_final_after_seconds > 0`으로 설정하면 오래된 미리보기를 가능한 경우 미리보기 삭제와 함께 새로운 최종 메시지로 전달하도록 선택할 수 있습니다. 기본값은 `0`이며, 스트리밍 응답을 항상 원래 위치에서 완료합니다. 따라서 두 작업을 모두 표시하는 클라이언트에서 잠시 중복 메시지가 나타났다가 삭제되는 현상을 피할 수 있습니다.

:::note 플랫폼별 스트리밍 기본값
기본 `streaming.enabled` 스위치는 기본적으로 `false`이므로 전환하기 전에는 아무것도 스트리밍되지 않습니다. 활성화한 뒤에는 플랫폼별로 스트리밍이 결정됩니다. Telegram은 `display.platforms.telegram.streaming: true`(스트리밍)로 제공되고 Discord는 `display.platforms.discord.streaming: false`(스트리밍하지 않음)로 제공됩니다. 따라서 스트리밍을 활성화하면 Telegram은 즉시 스트리밍하고, Discord는 토글을 변경할 때까지 전체 메시지 응답을 유지합니다. 대시보드의 **Channels** 토글에서 또는 `~/.hermes/config.yaml`을 직접 수정해 플랫폼별 스위치를 조정할 수 있습니다.
:::

## 그룹 채팅 세션 격리

CLI, TUI/대시보드, 메시징 게이트웨이 전반에서 동시에 열어 둘 수 있는 채팅 세션 수를 제한합니다:

```yaml
max_concurrent_sessions: null  # null/0 = unlimited; positive integer = active session cap
```

슬롯은 채팅 창을 열 때가 아니라 세션이 **첫 번째 턴**을 실행할 때 할당됩니다. 메시지를 보낼 때까지 채팅을 열거나, 재개하거나, 다시 연결하는 데는 비용이 없습니다. 따라서 유휴 상태인 데스크톱 탭(그리고 불안정한 웹소켓을 백그라운드에서 재개하는 트리거)이 이 제한을 공유하는 메시징 게이트웨이의 슬롯을 고갈시키지 않습니다.

제한에 도달하면 Hermes는 어떤 표면이 슬롯을 점유하고 있는지 표시하는 직접적인 제한 메시지를 반환합니다. 기존 활성 세션은 평소와 동일하게 동작합니다. 현재 슬롯 사용량과 모든 점유자를 확인하려면 `hermes status`를 실행하세요.

표준 키는 최상위 `max_concurrent_sessions`입니다. Hermes는 `gateway.max_concurrent_sessions`도 대체 키로 허용하지만 두 키가 모두 설정된 경우 최상위 키가 우선합니다.

제한은 로컬 런타임 임대 파일로 적용되며 최선의 노력 방식입니다. 사용자가 고립되지 않도록 레지스트리를 읽거나 잠글 수 없으면 Hermes는 제한을 적용하지 않고 계속 진행합니다. 이 기능은 단일 호스트/프로필 런타임을 위한 것이며, 여러 머신에서 공유하는 `$HERMES_HOME` 마운트를 위한 것이 아닙니다.

공유 채팅이 방마다 하나의 대화를 유지할지, 참가자마다 하나의 대화를 유지할지 제어합니다:

```yaml
group_sessions_per_user: true  # true = per-user isolation in groups/channels, false = one shared session per chat
```

- `true`가 기본값이며 권장 설정입니다. Discord 채널, Telegram 그룹, Slack 채널 및 이와 유사한 공유 컨텍스트에서는 플랫폼이 사용자 ID를 제공할 때 각 발신자에게 별도의 세션이 할당됩니다.
- `false`로 설정하면 이전의 공유 방 동작으로 되돌아갑니다. Hermes가 채널을 하나의 협업 대화처럼 취급하기를 명시적으로 원하는 경우 유용할 수 있지만, 사용자들이 컨텍스트, 토큰 비용, 인터럽트 상태를 공유하게 됩니다.
- 다이렉트 메시지에는 영향을 주지 않습니다. Hermes는 평소처럼 채팅/DM ID를 기준으로 DM 세션을 구분합니다.
- 스레드는 어느 설정에서나 부모 채널과 격리됩니다. `true`인 경우 스레드 안에서도 참가자마다 자체 세션을 사용합니다.

동작 세부 사항과 예시는 [세션](/user-guide/sessions) 및 [Discord 가이드](/user-guide/messaging/discord)를 참조하세요.

## 인증되지 않은 DM 동작

알 수 없는 사용자가 다이렉트 메시지를 보낼 때 Hermes가 수행할 동작을 제어합니다:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair`는 채팅 방식 DM 플랫폼의 기본값입니다. Hermes는 액세스를 거부하지만 DM으로 일회성 페어링 코드를 회신합니다.
- `ignore`는 인증되지 않은 DM을 조용히 삭제합니다.
- 이메일의 기본값은 `ignore`입니다. 받은 편지함에 관련 없는 읽지 않은 메일이 포함될 수 있기 때문입니다. `platforms.email.unauthorized_dm_behavior: pair`를 설정하면 예외적으로 페어링을 사용할 수 있습니다.
- 플랫폼 섹션은 전역 기본값을 덮어쓰므로, 전반적으로 페어링을 활성화하면서 특정 플랫폼만 더 조용하게 만들 수 있습니다.

## 빠른 명령

LLM을 호출하지 않고 셸 명령을 실행하거나, 슬래시 명령 하나를 다른 명령의 별칭으로 지정하는 사용자 지정 명령을 정의합니다. 실행형 빠른 명령은 토큰을 전혀 사용하지 않으며 메시징 플랫폼(Telegram, Discord 등)에서 빠른 서버 확인이나 유틸리티 스크립트에 유용합니다.

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
    command: cd ~/.hermes/hermes-agent && git pull && uv pip install -e .
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

사용법: CLI 또는 모든 메시징 플랫폼에서 `/status`, `/disk`, `/update`, `/gpu`, `/restart`를 입력하세요. `exec` 명령은 호스트에서 로컬로 실행되고 출력을 직접 반환합니다. LLM 호출이나 토큰 소비는 없습니다. `alias` 명령은 설정된 슬래시 명령 대상에 맞게 다시 작성됩니다.

- **30초 시간 제한** — 오래 실행되는 명령은 오류 메시지와 함께 종료됩니다.
- **우선순위** — 빠른 명령은 스킬 명령보다 먼저 확인되므로 스킬 이름을 재정의할 수 있습니다.
- **자동 완성** — 빠른 명령은 디스패치 시점에 확인되며 기본 제공 슬래시 명령 자동 완성 표에는 표시되지 않습니다.
- **유형** — 지원되는 유형은 `exec`와 `alias`입니다. 다른 유형은 오류를 표시합니다.
- **어디서나 작동** — CLI, Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant

문자열만으로 구성된 프롬프트 단축키는 유효한 빠른 명령이 아닙니다. 재사용 가능한 프롬프트 워크플로의 경우 스킬을 만들거나 기존 슬래시 명령의 별칭을 만드세요.

## 사람과 같은 지연

메시징 플랫폼에서 사람과 비슷한 응답 속도를 시뮬레이션합니다:

```yaml
human_delay:
  mode: "off"                  # off | natural | custom
  min_ms: 800                  # Minimum delay (custom mode)
  max_ms: 2500                 # Maximum delay (custom mode)
```

## 코드 실행

`execute_code` 도구를 구성합니다:

```yaml
code_execution:
  mode: project                # project (default) | strict
  timeout: 300                 # Max execution time in seconds
  max_tool_calls: 50           # Max tool calls within code execution
```

`mode`는 스크립트의 작업 디렉터리와 Python 인터프리터를 제어합니다:

- **`project`** (기본값) — 스크립트는 세션의 작업 디렉터리와 활성 virtualenv/conda 환경의 Python으로 실행됩니다. 프로젝트 의존성(`pandas`, `torch`, 프로젝트 패키지)과 상대 경로(`.env`, `./data.csv`)가 자연스럽게 해석되어 `terminal()`이 보는 환경과 일치합니다.
- **`strict`** — 스크립트는 임시 스테이징 디렉터리에서 `sys.executable`(Hermes 자체 Python)로 실행됩니다. 재현성은 가장 높지만 프로젝트 의존성과 상대 경로는 해석되지 않습니다.

환경 정리(`*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, `*_CREDENTIAL`, `*_PASSWD`, `*_AUTH` 제거)와 도구 허용 목록은 두 모드에 동일하게 적용됩니다. 모드를 바꿔도 보안 태세는 달라지지 않습니다.
## 웹 검색 백엔드

`web_search` 및 `web_extract` 도구는 다섯 가지 백엔드 제공자를 지원합니다. `config.yaml`에서 또는 `hermes tools`를 통해 백엔드를 구성하세요.

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | exa

  # Or use per-capability keys to mix providers (e.g. free search + paid extract):
  search_backend: "searxng"
  extract_backend: "firecrawl"
```

| 백엔드 | 환경 변수 | 검색 | 추출 |
|---------|---------|---------|---------|
| **Firecrawl** (기본값) | `FIRECRAWL_API_KEY` | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ |

**백엔드 선택:** `web.backend`가 설정되지 않으면 사용 가능한 API 키에서 백엔드를 자동으로 감지합니다. `SEARXNG_URL`만 설정되어 있으면 SearXNG를 사용합니다. `EXA_API_KEY`만 설정되어 있으면 Exa를 사용합니다. `TAVILY_API_KEY`만 설정되어 있으면 Tavily를 사용합니다. `PARALLEL_API_KEY`만 설정되어 있으면 Parallel을 사용합니다. 그 외에는 Firecrawl이 기본값입니다.

**SearXNG**는 70개 이상의 검색 엔진을 조회하는 무료 자체 호스팅 개인정보 보호 메타 검색 엔진입니다. API 키는 필요하지 않으며, 인스턴스의 `SEARXNG_URL`만 설정하면 됩니다(예: `http://localhost:8080`). SearXNG는 검색만 지원하므로 `web_extract`에는 별도의 추출 제공자가 필요합니다(`web.extract_backend` 설정). Docker 설정 방법은 [웹 검색 설정 가이드](/user-guide/features/web-search)를 참조하세요.

**자체 호스팅 Firecrawl:** 자체 인스턴스를 가리키도록 `FIRECRAWL_API_URL`을 설정하세요. 사용자 지정 URL을 설정하면 API 키는 선택 사항이 됩니다(인증을 비활성화하려면 서버에서 `USE_DB_AUTHENTICATION=***`을 설정하세요).

**Parallel 검색 모드:** 검색 동작을 제어하려면 `PARALLEL_SEARCH_MODE`를 설정하세요 — `fast`, `one-shot`, `agentic`(기본값: `agentic`) 중 하나입니다.

**Exa:** `~/.hermes/.env`에 `EXA_API_KEY`를 설정하세요. `category` 필터링(`company`, `research paper`, `news`, `people`, `personal site`, `pdf`)과 도메인/날짜 필터를 지원합니다.

## 브라우저

브라우저 자동화 동작을 구성합니다.

```yaml
browser:
  inactivity_timeout: 120        # Seconds before auto-closing idle sessions
  command_timeout: 30             # Timeout in seconds for browser commands (screenshot, navigate, etc.)
  record_sessions: false         # Auto-record browser sessions as WebM videos to ~/.hermes/browser_recordings/
  # Optional CDP override — when set, Hermes attaches directly to your own
  # Chromium-family browser (via /browser connect) rather than starting a headless browser.
  cdp_url: ""
  # Dialog supervisor — controls how native JS dialogs (alert / confirm / prompt)
  # are handled when a CDP backend is attached (Browserbase, local Chromium-family
  # browser via /browser connect). Ignored on Camofox and default local agent-browser mode.
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # Safety auto-dismiss under must_respond (seconds)
  camofox:
    managed_persistence: false   # When true, Camofox sessions persist cookies/logins across restarts
    user_id: ""                  # Optional externally managed Camofox userId
    session_key: ""              # Optional session key sent when Hermes creates a tab
    adopt_existing_tab: false    # Reuse an existing tab for this identity before creating one
```

**대화상자 정책:**

- `must_respond`(기본값) — 대화상자를 캡처하고 `browser_snapshot.pending_dialogs`에 표시한 다음, 에이전트가 `browser_dialog(action=...)`을 호출할 때까지 기다립니다. `dialog_timeout_s`초 동안 응답이 없으면 페이지의 JS 스레드가 영원히 멈추지 않도록 대화상자를 자동으로 닫습니다.
- `auto_dismiss` — 캡처한 뒤 즉시 닫습니다. 이후에도 에이전트는 `closed_by="auto_policy"`가 설정된 대화상자 기록을 `browser_snapshot.recent_dialogs`에서 확인할 수 있습니다.
- `auto_accept` — 캡처한 뒤 즉시 수락합니다. 공격적인 `beforeunload` 프롬프트가 있는 페이지에 유용합니다.

브라우저 도구 세트는 여러 제공자를 지원합니다. Browserbase, Browser Use, 로컬 Chromium 계열 CDP 설정에 관한 자세한 내용은 [브라우저 기능 페이지](/user-guide/features/browser)를 참조하세요.

## 시간대

서버 로컬 시간대를 IANA 시간대 문자열로 재정의합니다. 로그의 타임스탬프, cron 예약, 시스템 프롬프트에 주입되는 시간에 영향을 줍니다.

```yaml
timezone: "America/New_York"   # IANA timezone (default: "" = server-local time)
```

지원되는 값은 모든 IANA 시간대 식별자입니다(예: `America/New_York`, `Europe/London`, `Asia/Kolkata`, `UTC`). 서버 로컬 시간을 사용하려면 비워 두거나 생략하세요.

## Discord

메시징 게이트웨이의 Discord 전용 동작을 구성합니다.

```yaml
discord:
  require_mention: true          # Require @mention to respond in server channels
  free_response_channels: ""     # Comma-separated channel IDs where bot responds without @mention
  auto_thread: true              # Auto-create threads on @mention in channels
```

- `require_mention` — `true`(기본값)이면 봇은 `@BotName`으로 멘션된 경우에만 서버 채널에서 응답합니다. DM에서는 멘션 없이도 항상 작동합니다.
- `free_response_channels` — 멘션 없이 모든 메시지에 봇이 응답하는 채널 ID를 쉼표로 구분한 목록입니다.
- `auto_thread` — `true`(기본값)이면 채널에서 멘션할 때 대화용 스레드를 자동으로 만들어 채널을 깔끔하게 유지합니다(Slack 스레드와 유사).

## 보안

실행 전 보안 검사 및 비밀 정보 삭제:

```yaml
security:
  redact_secrets: true           # Redact API key patterns in tool output and logs (on by default)
  tirith_enabled: true           # Enable Tirith security scanning for terminal commands
  tirith_path: "tirith"          # Path to tirith binary (default: "tirith" in $PATH)
  tirith_timeout: 5              # Seconds to wait for tirith scan before timing out
  tirith_fail_open: true         # Allow command execution if tirith is unavailable
  website_blocklist:             # See Website Blocklist section below
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets` — `true`이면 도구 출력과 로그에 들어가기 전에 API 키, 토큰, 비밀번호처럼 보이는 패턴을 자동으로 감지하고 삭제합니다. **기본적으로 활성화되어 있습니다**. 디버깅 또는 삭제 기능 개발을 위해 원시 자격 증명 형태의 문자열이 꼭 필요한 경우에만 명시적으로 `false`로 설정하세요.
- `tirith_enabled` — `true`이면 터미널 명령이 실행되기 전에 [Tirith](https://github.com/sheeki03/tirith)로 검사하여 위험할 수 있는 작업을 탐지합니다.
- `tirith_path` — Tirith 바이너리의 경로입니다. 표준 위치가 아닌 곳에 Tirith가 설치되어 있으면 설정하세요.
- `tirith_timeout` — Tirith 검사를 기다리는 최대 시간(초)입니다. 검사 시간이 초과되면 명령을 계속 진행합니다.
- `tirith_fail_open` — `true`(기본값)이면 Tirith를 사용할 수 없거나 실패해도 명령 실행을 허용합니다. Tirith가 확인할 수 없을 때 명령을 차단하려면 `false`로 설정하세요.

## 웹사이트 차단 목록

에이전트의 웹 및 브라우저 도구가 접근하지 못하도록 특정 도메인을 차단합니다.

```yaml
security:
  website_blocklist:
    enabled: false               # Enable URL blocking (default: false)
    domains:                     # List of blocked domain patterns
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:                # Load additional rules from external files
      - "/etc/hermes/blocked-sites.txt"
```

활성화되면 도메인 패턴과 일치하는 모든 URL은 웹 또는 브라우저 도구가 실행되기 전에 거부됩니다. 이는 `web_search`, `web_extract`, `browser_navigate` 및 URL에 접근하는 모든 도구에 적용됩니다.

도메인 규칙은 다음을 지원합니다.

- 정확한 도메인: `admin.example.com`
- 와일드카드 하위 도메인: `*.internal.company.com`(모든 하위 도메인 차단)
- TLD 와일드카드: `*.local`

공유 파일에는 한 줄에 하나의 도메인 규칙을 작성합니다(빈 줄과 `#` 주석은 무시됨). 파일이 없거나 읽을 수 없으면 경고가 로그에 기록되지만 다른 웹 도구가 비활성화되지는 않습니다.

정책은 30초 동안 캐시되므로 구성을 변경하면 재시작 없이 빠르게 적용됩니다. 전체 대화상자 처리 흐름은 [브라우저 기능 페이지](./features/browser.md#browser_dialog)를 참고하세요.

## 스마트 승인

Hermes가 잠재적으로 위험한 명령을 처리하는 방식을 제어합니다.

```yaml
approvals:
  mode: smart   # smart | manual | off
```

| 모드 | 동작 |
|------|----------|
| `smart`(기본값) | 보조 LLM을 사용해 표시된 명령이 실제로 위험한지 평가합니다. 위험이 낮은 명령은 해당 명령에 한해 자동 승인합니다. 실제로 위험한 명령은 거부하고, 판단이 불확실하면 사용자에게 확인을 요청합니다. |
| `manual` | 표시된 명령을 실행하기 전에 사용자에게 묻습니다. CLI에서는 대화형 승인 대화상자를 표시하고, 메시징에서는 보류 중인 승인 요청을 대기열에 추가합니다. |
| `off` | 모든 승인 검사를 건너뜁니다. `HERMES_YOLO_MODE=true`와 같습니다. **주의해서 사용하세요.** |

스마트 모드는 안전한 작업에 대해 에이전트가 더 자율적으로 작업하도록 하면서 실제로 위험한 작업은 계속 잡아내므로 승인 피로를 줄이는 데 특히 유용합니다.

:::warning
`approvals.mode: off`로 설정하면 터미널 명령에 대한 모든 안전 검사가 비활성화됩니다. 신뢰할 수 있고 샌드박스 처리된 환경에서만 사용하세요.
:::

### 거부 회로 차단기

`approvals.denial_breaker_threshold`(기본값 `3`)는 스마트 승인 검토자가 계속 거부하는 명령의 변형을 에이전트가 재시도하지 않도록 보호합니다. 재시도할 때마다 추가 guardian LLM 호출이 발생합니다. 세션에서 그 횟수만큼 연속 거부되면 거부 메시지가, 작업을 중단하고 차단된 작업을 보고한 뒤 수동으로 실행하거나 `/approve`를 사용하라고 지시하는 강제 중지 안내로 바뀝니다. 승인이 이루어지면 횟수가 초기화됩니다. 비활성화하려면 `0`으로 설정하세요.

```yaml
approvals:
  denial_breaker_threshold: 3   # 0 disables the breaker
```

### 거부 규칙

`approvals.deny`는 `--yolo`, `/yolo` 또는 `mode: off`에서도 일치하는 터미널 명령을 무조건 차단하는 glob 패턴 목록입니다. 내장된 강력 차단 목록에 대응하는 사용자가 편집할 수 있는 설정입니다.

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
```

패턴은 대소문자를 구분하지 않는 fnmatch glob이며 YAML에서 따옴표로 감싸야 합니다(맨 앞의 `*`를 따옴표 없이 사용하면 파싱 오류가 발생합니다). 자세한 내용은 [보안 — 사용자 정의 거부 규칙](/user-guide/security#user-defined-deny-rules-approvalsdeny)을 참조하세요.

### 사용자 지정 스마트 승인 정책

`approvals.smart_policy`를 사용하면 스마트 승인 검토자의 지침에 자체 규칙을 추가할 수 있습니다. 설정하면 해당 텍스트가 guardian LLM의 시스템 프롬프트(신뢰할 수 있는 채널 — 신뢰할 수 없는 명령 텍스트와 절대 함께 전달되지 않음)에 추가되므로 코드를 수정하지 않고도 환경에 맞게 판단을 엄격하게 하거나 완화할 수 있습니다.

```yaml
approvals:
  smart_policy: |
    Always ESCALATE commands that modify anything under /etc.
    APPROVE docker compose restarts in ~/deploys — they are routine here.
```


## 체크포인트

위험한 파일 작업 전에 파일 시스템 스냅샷을 자동으로 생성합니다. 자세한 내용은 [체크포인트 및 롤백](/user-guide/checkpoints-and-rollback)을 참조하세요.

```yaml
checkpoints:
  enabled: false                 # Enable automatic checkpoints (also: hermes chat --checkpoints). Default: false (opt-in).
  max_snapshots: 20              # Max checkpoints to keep per directory (default: 20)
```


## 위임

delegate 도구의 서브에이전트 동작을 구성합니다.

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # Override model (empty = inherit parent)
  # provider: "openrouter"                  # Override provider (empty = inherit parent)
  # base_url: "http://localhost:1234/v1"    # Direct OpenAI-compatible endpoint (takes precedence over provider)
  # api_key: "local-key"                    # API key for base_url (falls back to OPENAI_API_KEY)
  # api_mode: ""                            # Wire protocol for base_url: "chat_completions", "codex_responses", or "anthropic_messages". Empty = auto-detect from URL (e.g. /anthropic suffix → anthropic_messages). Set explicitly for non-standard endpoints the heuristic can't detect.
  max_concurrent_children: 3                # Parallel children per batch (floor 1, no ceiling). Also via DELEGATION_MAX_CONCURRENT_CHILDREN env var.
  worktree_isolation: false                 # Give each child its own git worktree branched from HEAD (local backend + git repos only; inspired by Muse Code). See Subagent Delegation → Worktree Isolation.
  max_spawn_depth: 1                        # Delegation tree depth cap (1-3, clamped). 1 = flat (default): parent spawns leaves that cannot delegate. 2 = orchestrator children can spawn leaf grandchildren. 3 = three levels.
  orchestrator_enabled: true                # Global kill switch. When false, role="orchestrator" is ignored and every child is forced to leaf regardless of max_spawn_depth.
```

**서브에이전트 제공자:모델 재정의:** 기본적으로 서브에이전트는 부모 에이전트의 제공자와 모델을 상속합니다. `delegation.provider` 및 `delegation.model`을 설정하면 서브에이전트를 다른 제공자:모델 조합으로 라우팅할 수 있습니다. 예를 들어 주 에이전트는 고가의 추론 모델을 사용하면서 좁은 범위의 하위 작업에는 저렴하고 빠른 모델을 사용할 수 있습니다.

**직접 엔드포인트 재정의:** `delegation.base_url`을 설정하면 `delegation.provider`보다 우선합니다. `delegation.api_key`를 생략하면 `OPENAI_API_KEY`로 대체됩니다.

**와이어 프로토콜(`api_mode`):** Hermes는 `delegation.base_url`에서 와이어 프로토콜을 자동 감지합니다(예: `/anthropic`으로 끝나는 경로는 `anthropic_messages`). 휴리스틱으로 분류할 수 없는 엔드포인트(예: Azure AI Foundry, MiniMax, Zhipu GLM 또는 Anthropic 형태의 백엔드를 제공하는 LiteLLM 프록시)는 `delegation.api_mode`를 `chat_completions`, `codex_responses`, `anthropic_messages` 중 하나로 명시적으로 설정하세요. 기본값인 빈 값으로 두면 자동 감지를 유지합니다.

위임 제공자는 CLI/게이트웨이 시작 시와 동일한 자격 증명 확인 방식을 사용합니다. 지원되는 제공자는 모두 사용할 수 있습니다: `openrouter`, `nous`, `copilot`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`. 제공자를 설정하면 시스템이 올바른 기본 URL, API 키, API 모드를 자동으로 확인하므로 자격 증명을 수동으로 연결할 필요가 없습니다.

**우선순위:** 구성의 `delegation.base_url` → 구성의 `delegation.provider` → 부모 제공자(상속) 순입니다. 구성의 `delegation.model` → 부모 모델(상속) 순입니다. `provider` 없이 `model`만 설정하면 부모의 자격 증명을 유지하면서 모델 이름만 변경합니다(OpenRouter처럼 동일한 제공자 내에서 모델을 전환할 때 유용).

**너비와 깊이:** `max_concurrent_children`는 배치당 병렬로 실행되는 서브에이전트 수를 제한합니다(기본값 `3`, 최솟값 1, 상한 없음). `DELEGATION_MAX_CONCURRENT_CHILDREN` 환경 변수를 통해서도 설정할 수 있습니다. 모델이 제한보다 긴 `tasks` 배열을 제출하면 `delegate_task`는 조용히 잘라내는 대신 제한을 설명하는 도구 오류를 반환합니다. `max_spawn_depth`는 위임 트리의 깊이를 제어합니다(1~3으로 제한). 기본값 `1`에서는 위임이 평면적이므로 자식이 손자 에이전트를 생성할 수 없으며, `role="orchestrator"`를 전달해도 조용히 `leaf`로 강등됩니다. 오케스트레이터 자식이 리프 손자 에이전트를 생성하도록 하려면 `2`, 3단계 트리에는 `3`으로 높이세요. 비용은 곱셈으로 증가하므로 `max_spawn_depth: 3` 및 `max_concurrent_children: 3`에서는 트리에 최대 3×3×3 = 27개의 병렬 리프 에이전트가 도달할 수 있습니다. 사용 패턴은 [서브에이전트 위임 → 깊이 제한 및 중첩 오케스트레이션](features/delegation.md#depth-limit-and-nested-orchestration)을 참조하세요.
## 명확화

명확화 질문에 대한 응답을 게이트웨이가 얼마나 오래 기다릴지 설정합니다. 표준 키는 `agent.clarify_timeout`(기본값 `3600`초)이며, 레거시 최상위 키인 `clarify.timeout`도 명시적으로 설정하면 계속 사용할 수 있습니다.

```yaml
agent:
  clarify_timeout: 3600        # Seconds to wait for user clarification response (0 or less = unlimited)
```

## 컨텍스트 파일(SOUL.md, AGENTS.md)

Hermes는 서로 다른 두 가지 컨텍스트 범위를 사용합니다.

| 파일 | 용도 | 범위 |
|------|------|------|
| `SOUL.md` | **기본 에이전트 정체성** — 시스템 프롬프트의 1번 슬롯에서 에이전트가 누구인지 정의합니다 | `~/.hermes/SOUL.md` 또는 `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | 프로젝트별 지침(최우선) | Git 루트까지 탐색 |
| `AGENTS.md` | 프로젝트별 지침, 코딩 규칙 | 재귀적으로 디렉터리를 탐색 |
| `CLAUDE.md` | Claude Code 컨텍스트 파일(또한 감지됨) | 작업 디렉터리만 |
| `.cursorrules` | Cursor 규칙 파일(또한 감지됨) | 작업 디렉터리만 |
| `.cursor/rules/*.mdc` | Cursor 규칙 파일(또한 감지됨) | 작업 디렉터리만 |

`SOUL.md`는 에이전트의 기본 정체성입니다. 시스템 프롬프트의 1번 슬롯을 차지하며, 내장된 기본 정체성을 완전히 대체합니다. `SOUL.md`가 없거나 비어 있거나 로드할 수 없으면 Hermes는 내장된 기본 정체성으로 돌아갑니다.

프로젝트 컨텍스트 파일은 우선순위 체계를 사용합니다. `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules` 순서에서 가장 먼저 일치하는 한 가지 유형만 로드됩니다. `SOUL.md`는 이와 별개로 항상 로드됩니다.

`AGENTS.md`는 계층적으로 적용됩니다. 하위 디렉터리에도 `AGENTS.md`가 있으면 모두 결합됩니다.

Hermes는 기본값인 `context_file_max_chars` 20,000자로 제한하여 모든 로드된 컨텍스트 파일을 캡처하며, 스마트한 잘라내기를 적용합니다.

다음도 참고하세요.

- [개성과 SOUL.md](/user-guide/features/personality)
- [컨텍스트 파일](/user-guide/features/context-files)

## 작업 디렉터리

| 컨텍스트 | 기본값 |
|---------|---------|
| **CLI(`hermes`)** | 명령을 실행한 현재 디렉터리 |
| **메시징 게이트웨이** | `~/.hermes/config.yaml`의 `terminal.cwd`; 설정되지 않으면 홈 디렉터리 `~` |
| **Docker / Singularity / Modal / SSH** | 컨테이너 또는 원격 시스템 내부의 사용자 홈 디렉터리 |

작업 디렉터리를 재정의하려면 다음과 같이 설정합니다.

```yaml
# In ~/.hermes/config.yaml:
terminal:
  cwd: /home/myuser/projects
```

`~/.hermes/.env`의 `MESSAGING_CWD`와 직접 지정한 `TERMINAL_CWD` 항목은 레거시 호환성을 위한 대체 수단입니다. 새 구성에서는 `terminal.cwd`를 사용해야 합니다.

## 네트워크

아웃바운드 HTTP 연결을 위한 연결성 우회 설정입니다.

```yaml
network:
  force_ipv4: false   # Force IPv4 for outbound connections (default: false)
```

`force_ipv4` — IPv6가 손상되었거나 연결할 수 없는 서버에서는 Python이 먼저 AAAA 레코드를 확인합니다. 그러면 IPv4로 대체하기 전에 전체 TCP 타임아웃 동안 멈출 수 있습니다. 이 값을 `true`로 설정하면 IPv6를 완전히 건너뛰고 IPv4로 직접 연결합니다.

## 온보딩

첫 접속 온보딩 힌트와 구조화된 프로필 작성 제안에 대한 설정입니다.

```yaml
onboarding:
  profile_build: "ask"   # "ask" (default) | "off"
  seen: {}               # internal latch — leave empty
```

- `profile_build` — 최초의 게이트웨이 메시지에서 제공되는 프로필 작성 경로를 제어합니다. 기본값인 `"ask"`는 사용자 프로필 작성을 제안합니다. 이 제안은 **옵트인 및 동의 기반**으로 제공되며, 에이전트는 조회 전에 먼저 묻고 연결된 계정을 몰래 읽지 않습니다. `"off"`로 설정하면 일반적인 소개만 표시됩니다. 이 제안은 최대 한 번만 표시됩니다.
- `seen` — 내부 상태입니다. Hermes는 여기에 표시한 각 힌트를 기록하여 다시 표시하지 않습니다. 프로필 작성 제안도 표시된 뒤 여기에 기록됩니다. 직접 수정하지 마세요. 모든 힌트를 다시 보려면 `onboarding` 섹션 전체를 삭제하세요.

## 대시보드

[웹 대시보드](/user-guide/features/web-dashboard)의 시각적 테마, 공개 URL, 인증 공급자를 설정합니다. 인증 공급자(OAuth, 기본 비밀번호, drain)는 웹 대시보드 페이지에서 자세히 설명하며, 여기서는 `config.yaml`의 형태를 설명합니다.

```yaml
dashboard:
  theme: "default"            # "default" | "midnight" | "ember" | "mono" | "cyberpunk" | "rose"
  show_token_analytics: false # Re-enable the (local-estimate-only) token/cost analytics surfaces
  public_url: ""              # Full public authority for OAuth redirect_uri (env: HERMES_DASHBOARD_PUBLIC_URL)
  oauth:                      # Portal OAuth gate (engaged with --host and not --insecure)
    client_id: ""             # agent:{instance_id} — Portal provisions this
    portal_url: ""            # blank → plugin default (production Portal)
  basic_auth:                 # Self-hosted username/password gate (dashboard_auth/basic plugin)
    username: ""              # blank → plugin no-op
    password_hash: ""         # scrypt$... (preferred — no plaintext at rest)
    password: ""              # plaintext fallback (hashed in-memory at load)
    secret: ""                # token-signing key; blank → random per-process
    session_ttl_seconds: 0    # 0 → plugin default (12h)
  drain_auth:                 # Drain-control service-credential gate (dashboard_auth/drain plugin)
    scope: "drain"            # capability label on the verified principal
    min_secret_chars: 43      # entropy bar (url-safe-b64 chars; 43 ≈ 256 bits)
```

- `theme` — 대시보드의 시각적 테마입니다.
- `show_token_analytics` — 기본적으로 꺼져 있습니다. Analytics 페이지와 토큰/비용 수치는 **로컬 하한 추정치**입니다(보조 호출, 재시도, 폴백, 캐시 쓰기는 포함하지 않음). 따라서 공급자 청구액보다 훨씬 낮게 표시될 수 있습니다. 실제 청구액을 나타내는 값이 아님을 이해한 경우에만 `true`로 설정하세요.
- `public_url` — 설정하면 OAuth `redirect_uri`를 구성할 전체 권한(스킴 + 호스트 + 선택적 경로 접두사)이 됩니다. `X-Forwarded-*` 헤더를 안정적으로 전달하지 않는 리버스 프록시 뒤에 배포할 때 설정하세요. 비워 두면 프록시 헤더를 재구성하여 사용합니다.
- `oauth` / `basic_auth` / `drain_auth` — 번들로 제공되는 dashboard-auth 플러그인이 읽는 인증 공급자 설정입니다. drain 시크릿 자체는 여기에 설정하지 않으며, `HERMES_DASHBOARD_DRAIN_SECRET` 환경 변수로 제공합니다. 전체 인증 설정은 [웹 대시보드](/user-guide/features/web-dashboard)를 참고하세요.
