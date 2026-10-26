---
sidebar_position: 2
---

# 프로필: 여러 에이전트 실행하기

같은 컴퓨터에서 여러 개의 독립적인 Hermes 에이전트를 실행하세요. 각 에이전트는 자체 구성, API 키, 메모리, 세션, 스킬, 게이트웨이 상태를 가집니다.

## 프로필이란 무엇인가요?

프로필은 별도의 Hermes 홈 디렉터리입니다. 각 프로필에는 자체 `config.yaml`, `.env`, `SOUL.md`, 메모리, 세션, 스킬, cron 작업 및 상태 데이터베이스가 들어 있는 디렉터리가 있습니다. 프로필을 사용하면 코딩 보조, 개인 봇, 연구 에이전트 등 서로 다른 목적의 에이전트를 실행하면서 Hermes 상태가 서로 섞이지 않게 할 수 있습니다.

:::caution 각 에이전트에 고유한 프로필을 제공하세요
두 에이전트 프로세스가 같은 프로필(동일한 Hermes 홈)을 가리키도록 하지 마세요. 두 프로세스 모두 자동으로 메모리를 기록하고, 세션이 시작될 때 서로의 기록을 시스템 프롬프트에 불러옵니다. 따라서 두 작성자가 한 홈에서 상태를 계속 누적하면, 사용자가 구성한 범위를 벗어날 때까지 서로의 상태를 키우게 됩니다. 프로필은 바로 이런 상황을 방지하기 위해 존재합니다. 공유 메모리가 필요한 에이전트는 대신 [외부 메모리 제공자](/user-guide/features/memory-providers)를 사용해야 합니다.
:::

프로필을 만들면 자동으로 자체 명령이 됩니다. `coder`라는 프로필을 만들면 즉시 `coder chat`, `coder setup`, `coder gateway start` 등을 사용할 수 있습니다.

## 빠른 시작

```bash
hermes profile create coder       # creates profile + "coder" command alias
coder setup                       # configure API keys and model
coder chat                        # start chatting
```

이제 `coder`는 자체 구성, 메모리, 상태를 가진 독립적인 Hermes 프로필입니다.

## 프로필 만들기

:::tip
가장 빠른 설정 방법: 새 프로필 안에서 `hermes setup --portal`을 실행하면 모델과 도구를 한 번에 연결할 수 있습니다. [Nous Portal](/integrations/nous-portal)을 참고하세요.
:::

### 빈 프로필

```bash
hermes profile create mybot
```

번들된 스킬이 미리 채워진 새 프로필을 만듭니다. API 키, 모델, 게이트웨이 토큰을 구성하려면 `mybot setup`을 실행하세요.

이 프로필을 kanban 작업자로 사용하거나 kanban 오케스트레이터가 이 프로필로 작업을 라우팅하도록 하려면, 오케스트레이터가 프로필의 용도를 알 수 있도록 만들 때 `--description "<role>"`을 전달하세요.

```bash
hermes profile create researcher --description "Reads source code and external docs, writes findings."
```

나중에 `hermes profile describe`를 사용해 설명을 설정하거나 자동 생성할 수도 있습니다. 전체 라우팅 모델은 [Kanban 가이드](./features/kanban#auto-vs-manual-orchestration)를 참고하세요.

### 구성만 복제하기 (`--clone`)

```bash
hermes profile create work --clone
```

현재 프로필의 `config.yaml`, `.env`, `SOUL.md`, 스킬을 새 프로필로 복사합니다. API 키, 모델, 기능은 같지만 세션과 메모리는 새로 시작합니다. 다른 API 키를 사용하려면 `~/.hermes/profiles/work/.env`를, 다른 성격을 사용하려면 `~/.hermes/profiles/work/SOUL.md`를 편집하세요.

### 모든 항목 복제하기 (`--clone-all`)

```bash
hermes profile create backup --clone-all
```

**모든 항목**을 복사합니다. 구성, API 키, 성격, 모든 메모리, 스킬, cron 작업, 플러그인이 포함됩니다. 프로필별 기록(세션 기록, `state.db`, `backups/`, `state-snapshots/`, `checkpoints/`)은 원본 프로필에 속하며 수십 GB까지 커질 수 있으므로 제외됩니다. 기록까지 포함한 전체 백업에는 `hermes profile export` 또는 `hermes backup`을 사용하세요.

### 특정 프로필에서 복제하기

```bash
hermes profile create work --clone-from coder
```

`--clone-from <source>`는 원본 프로필을 직접 선택하며 구성/스킬/SOUL 복제를 암시합니다. 원본 프로필을 완전히 복사하려면 `--clone-all`과 함께 사용하세요.

```bash
hermes profile create work-backup --clone-from coder --clone-all
```

:::tip Honcho 메모리 + 프로필
Honcho가 활성화되어 있으면 복제 작업이 새 프로필 전용 AI 피어를 자동으로 만들면서 동일한 사용자 작업 공간을 공유합니다. 각 프로필은 자체 관찰 내용과 정체성을 구축합니다. 자세한 내용은 [Honcho -- 다중 에이전트 / 프로필](./features/memory-providers.md#honcho)을 참고하세요.
:::

## 프로필 사용하기

### 명령 별칭

모든 프로필은 `~/.local/bin/<name>`에 명령 별칭을 자동으로 생성합니다.

```bash
coder chat                    # chat with the coder agent
coder setup                   # configure coder's settings
coder gateway start           # start coder's gateway
coder doctor                  # check coder's health
coder skills list             # list coder's skills
coder config set model.default anthropic/claude-sonnet-4
```

이 별칭은 모든 hermes 하위 명령에서 작동합니다. 내부적으로는 `hermes -p <name>`을 실행하는 것과 같습니다.

### `-p` 플래그

어떤 명령에서든 프로필을 명시적으로 지정할 수도 있습니다.

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # works in any position
```

### 고정 기본값 (`hermes profile use`)

```bash
hermes profile use coder
hermes chat                   # now targets coder
hermes tools                  # configures coder's tools
hermes profile use default    # switch back
```

기본값을 설정하면 일반 `hermes` 명령이 해당 프로필을 대상으로 실행됩니다. `kubectl config use-context`와 비슷합니다.

### 현재 위치 확인하기

CLI에는 항상 활성 프로필이 표시됩니다.

- **프롬프트**: `❯` 대신 `coder ❯`
- **배너**: 시작할 때 `Profile: coder` 표시
- **`hermes profile`**: 현재 프로필 이름, 경로, 모델, 게이트웨이 상태 표시

## 프로필과 작업 공간 및 샌드박싱의 차이

프로필은 작업 공간이나 샌드박스와 혼동하기 쉽지만 서로 다릅니다.

- **프로필**은 Hermes에 자체 상태 디렉터리(`config.yaml`, `.env`, `SOUL.md`, 세션, 메모리, 로그, cron 작업, 게이트웨이 상태)를 제공합니다.
- **작업 공간** 또는 **작업 디렉터리**는 터미널 명령이 시작되는 위치입니다. 이는 `terminal.cwd`로 별도로 제어합니다.
- **샌드박스**는 파일 시스템 접근을 제한합니다. 프로필은 에이전트를 샌드박싱하지 않습니다.

기본 `local` 터미널 백엔드에서는 에이전트가 여전히 사용자 계정과 동일한 파일 시스템 접근 권한을 가집니다. 프로필을 사용해도 프로필 디렉터리 밖의 폴더에 접근하는 것을 막을 수 없습니다.

프로필이 특정 프로젝트 폴더에서 시작하게 하려면 해당 프로필의 `config.yaml`에서 명시적인 절대 `terminal.cwd`를 설정하세요.

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

로컬 백엔드에서 `cwd: "."`를 사용하면 "프로필 디렉터리"가 아니라 "Hermes를 시작한 디렉터리"를 의미합니다.

또한 다음 사항에 유의하세요.

- `SOUL.md`는 모델을 안내할 수 있지만 작업 공간 경계를 강제하지는 않습니다.
- `SOUL.md`의 변경 사항은 새 세션에서 정상적으로 적용됩니다. 기존 세션은 여전히 이전 프롬프트 상태를 사용하고 있을 수 있습니다.
- 모델에게 "현재 어느 디렉터리에 있나요?"라고 묻는 것은 신뢰할 수 있는 격리 테스트가 아닙니다. 도구의 시작 디렉터리를 예측 가능하게 하려면 `terminal.cwd`를 명시적으로 설정하세요.

## 게이트웨이 실행하기

각 프로필은 자체 봇 토큰을 사용하는 별도의 프로세스로 자체 게이트웨이를 실행합니다.

```bash
coder gateway start           # starts coder's gateway
assistant gateway start       # starts assistant's gateway (separate process)
```

### 다른 봇 토큰

각 프로필에는 자체 `.env` 파일이 있습니다. 각 프로필에 서로 다른 Telegram/Discord/Slack 봇 토큰을 구성하세요.

```bash
# Edit coder's tokens
nano ~/.hermes/profiles/coder/.env

# Edit assistant's tokens
nano ~/.hermes/profiles/assistant/.env
```

### 안전성: 토큰 잠금

두 프로필이 실수로 같은 봇 토큰을 사용하면 두 번째 게이트웨이가 충돌하는 프로필을 명확히 표시하며 차단됩니다. Telegram, Discord, Slack, WhatsApp, Signal에서 지원됩니다.

### 영구 서비스

```bash
coder gateway install         # creates hermes-gateway-coder systemd/launchd service
assistant gateway install     # creates hermes-gateway-assistant service
```

각 프로필은 고유한 서비스 이름을 가집니다. 서비스는 독립적으로 실행됩니다.

:::note 공식 Docker 이미지 내부
프로필별 게이트웨이는 [s6-overlay](https://github.com/just-containers/s6-overlay)(컨테이너의 PID 1)가 관리하므로 `hermes profile create <name>`이 `/run/service/gateway-<name>/`에 s6 서비스 슬롯을 자동으로 등록합니다. `hermes -p <name> gateway start/stop/restart`는 일반 프로세스를 생성하는 대신 `s6-svc`로 전달됩니다. 따라서 충돌이 자동으로 다시 시작되고 `docker restart`를 해도 이전에 실행 중이던 게이트웨이 집합이 유지됩니다. 자세한 내용은 [프로필별 게이트웨이 감독](/user-guide/docker#per-profile-gateway-supervision)을 참고하세요.
:::

## 프로필 구성하기

각 프로필에는 다음 항목이 자체적으로 있습니다.

- **`config.yaml`** — 모델, 제공자, 도구 세트, 모든 설정
- **`.env`** — API 키, 봇 토큰
- **`SOUL.md`** — 성격과 지침

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

이 프로필이 기본적으로 특정 프로젝트에서 작동하게 하려면 자체 `terminal.cwd`도 설정하세요.

```bash
coder config set terminal.cwd /absolute/path/to/project
```

### 대시보드에서

[웹 대시보드](features/web-dashboard.md#managing-multiple-profiles)는 프로필별 대시보드 없이도 사이드바의 프로필 전환기를 통해 **모든** 프로필의 구성, API 키, 스킬, MCP, 모델을 관리할 수 있는 컴퓨터 수준의 화면입니다. `coder dashboard`는 `coder` 프로필을 미리 선택한 상태로 컴퓨터 대시보드로 연결됩니다. 대시보드의 Chat 탭도 프로필 전환기를 따르며, 선택한 프로필의 홈에서 대화를 시작합니다.

참고: 대시보드 프로필 페이지의 "활성 상태로 설정"은 **향후 CLI/게이트웨이 실행**의 고정 기본값(`hermes profile use`와 동일)입니다. 대시보드에서 프로필을 편집하려면 프로필 전환기를 사용하세요.

## 업데이트

`hermes update`는 코드를 한 번 가져오고(공유됨) 새로 번들된 스킬을 **모든** 프로필에 자동으로 동기화합니다.

```bash
hermes update
# → Code updated (12 commits)
# → Skills synced: default (up to date), coder (+2 new), assistant (+2 new)
```

사용자가 수정한 스킬은 절대 덮어쓰지 않습니다.

## 프로필 관리하기

```bash
hermes profile list           # show all profiles with status
hermes profile show coder     # detailed info for one profile
hermes profile rename coder dev-bot   # rename (updates alias + service)
hermes profile export coder   # pack into coder.tar.gz (shareable; keys stripped)
hermes profile import coder.tar.gz   # install an archive as a new profile
```

채팅에서는 같은 기능을 `/export`와 `/import`로 사용할 수 있으며, 데스크톱 앱에서는 **⌘K → Export/Import profile…**로 사용할 수 있습니다. [프로필 공유하기](#sharing-a-profile)를 참고하세요.

## 프로필 삭제하기

```bash
hermes profile delete coder
```

게이트웨이를 중지하고, systemd/launchd 서비스를 제거하고, 명령 별칭을 제거한 뒤 모든 프로필 데이터를 삭제합니다. 확인을 위해 프로필 이름을 입력하라는 메시지가 표시됩니다.

확인을 건너뛰려면 `--yes`를 사용하세요. `hermes profile delete coder --yes`

:::note
기본 프로필(`~/.hermes`)은 삭제할 수 없습니다. 모든 항목을 제거하려면 `hermes uninstall`을 사용하세요.
:::

## 탭 완성

```bash
# Bash
eval "$(hermes completion bash)"

# Zsh
eval "$(hermes completion zsh)"
```

영구적으로 완성 기능을 사용하려면 `~/.bashrc` 또는 `~/.zshrc`에 해당 줄을 추가하세요. `-p` 뒤의 프로필 이름, 프로필 하위 명령, 최상위 명령을 완성합니다.

## 작동 방식

프로필은 `HERMES_HOME` 환경 변수를 사용합니다. `coder chat`을 실행하면 래퍼 스크립트가 Hermes를 시작하기 전에 `HERMES_HOME=~/.hermes/profiles/coder`를 설정합니다. 코드베이스의 119개가 넘는 파일이 `get_hermes_home()`을 통해 경로를 확인하므로 Hermes 상태(구성, 세션, 메모리, 스킬, 상태 데이터베이스, 게이트웨이 PID, 로그, cron 작업)가 자동으로 프로필 디렉터리에 한정됩니다.

이는 터미널 작업 디렉터리와는 별개입니다. 도구 실행은 자동으로 `HERMES_HOME`에서 시작하지 않고 `terminal.cwd`(`cwd: "."`인 로컬 백엔드에서는 실행 디렉터리)에서 시작합니다.

호스트 설치에서는 기존 CLI 자격 증명이 프로필 간에 `~`에서 계속 작동하도록 도구 하위 프로세스가 기본적으로 실제 OS 사용자 `HOME`을 유지합니다. 컨테이너 백엔드는 영구 도구 상태에 `{HERMES_HOME}/home`을 계속 사용하며, 엄격한 프로필별 도구 구성이 필요한 호스트 사용자는 `terminal.home_mode: profile`을 선택할 수 있습니다.

따라서 쉽게 혼동할 수 있는 두 가지가 있습니다.

- `HERMES_HOME`은 프로필 경계입니다. Hermes 구성, `.env`, 메모리, 세션, 스킬, 로그, cron 작업, 게이트웨이 상태 및 기타 Hermes 데이터를 제어합니다.
- `HOME`은 외부 CLI가 기대하는 운영 체제/사용자 홈입니다. 호스트 설치에서 Hermes는 기본적으로 이를 실제 사용자 홈으로 유지하므로 `git`, `ssh`, `gh`, `az`, `npm`, Claude Code, Codex 같은 도구가 일반 셸에서 사용하는 것과 동일한 자격 증명을 찾습니다.

이 때문에 호스트 프로필은 기본적으로 일반 사용자 수준의 CLI 상태를 공유합니다. 프로필별로 별도의 CLI ID가 필요하면 해당 프로필의 `config.yaml`에서 `terminal.home_mode: profile`을 설정하세요. 이 모드에서 Hermes는 `HOME={HERMES_HOME}/home`으로 도구 하위 프로세스를 실행합니다. 그런 다음 프로필 홈 안에서 프로필별 `~/.ssh`, `~/.gitconfig`, `~/.config/gh`, 클라우드 CLI 인증, Claude/Codex 인증, npm 상태 및 유사한 파일을 초기화하거나 연결해야 합니다.

또한 Hermes는 `home_mode: profile`이 활성화되어 있을 때 스크립트가 실제 계정 홈을 찾을 수 있도록 하위 프로세스에 `HERMES_REAL_HOME`을 제공합니다.

기본 프로필은 그저 `~/.hermes` 자체입니다. 마이그레이션은 필요하지 않으며 기존 설치는 동일하게 작동합니다.

## 프로필 공유하기

한 컴퓨터에서 만든 프로필을 다른 곳으로 옮길 수 있습니다. 자신의 워크스테이션, 팀원의 노트북, 또는 커뮤니티로 공유할 수 있습니다. 방법은 두 가지입니다.

**파일 보내기.** `/export`는 프로필을 하나의 `.tar.gz` 파일로 묶습니다. 스킬, 메모리, 페르소나, cron, 플러그인, 그리고 (데스크톱에서는) 테마와 레이아웃이 포함됩니다. API 키는 제거됩니다. 받는 사람은 `/import`를 실행합니다.

```bash
# In chat, run /export, hand over the file, and they run /import on it
hermes profile export coder
hermes profile import ./coder.tar.gz --name coder
```

**배포판 게시하기.** 프로필을 **git 저장소**로 패키징하면 받는 사람이 한 번의 명령으로 설치하고 나중에 버전이 지정된 업데이트를 가져올 수 있습니다. SOUL, 구성, 스킬, cron 작업, MCP 연결을 포함하며, 자격 증명, 메모리, 세션은 컴퓨터별로 유지됩니다.

```bash
# Install a whole agent from a git repo
hermes profile install github.com/you/research-bot --alias

# Update later when the author ships a new version (keeps your memories + .env)
hermes profile update research-bot
```

한 번만 전달하거나 이전할 에이전트에는 내보내기 파일을, 계속 배포할 에이전트에는 배포판을 사용하세요. 두 방식의 비교 표, 작성, 게시, 업데이트 의미론, 보안 모델은 **[프로필 배포판: 에이전트 전체 공유하기](./profile-distributions.md)**를 참고하세요.
