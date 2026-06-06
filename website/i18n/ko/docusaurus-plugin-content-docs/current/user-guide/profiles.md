---
sidebar_position: 2
---

# 프로필: 여러 에이전트 실행 (Profiles: Running Multiple Agents)

동일한 장비에서 각기 다른 구성(config), API 키, 기억(memory), 세션, 스킬 및 게이트웨이 상태를 가진 독립적인 여러 Hermes 에이전트를 실행해 보세요.

## 프로필이란? (What are profiles?)

프로필은 분리된 Hermes 홈 디렉토리입니다. 각 프로필은 자체 `config.yaml`, `.env`, `SOUL.md`, 기억, 세션, 스킬, 크론 작업(cron jobs) 및 상태 데이터베이스가 포함된 개별 디렉토리를 갖습니다. 프로필을 사용하면 코딩 어시스턴트, 개인 봇, 리서치 에이전트 등 다양한 목적에 맞게 Hermes의 상태를 섞지 않고 별도의 에이전트로 실행할 수 있습니다.

프로필을 생성하면 자동으로 해당 프로필 이름이 명령어 자체가 됩니다. 예를 들어 `coder`라는 프로필을 생성하면 즉시 `coder chat`, `coder setup`, `coder gateway start` 등을 사용할 수 있습니다.

## 빠른 시작 (Quick start)

```bash
hermes profile create coder       # 프로필 생성 + "coder" 명령어 별칭 생성
coder setup                       # API 키 및 모델 구성
coder chat                        # 대화 시작
```

이것이 전부입니다. 이제 `coder`는 고유한 구성, 기억, 상태를 가진 자체 Hermes 프로필이 되었습니다.

## 프로필 생성 (Creating a profile)

:::tip
가장 빠른 설정: 새 프로필 내에서 `hermes setup --portal`을 실행하면 모델과 도구를 한 번에 연결할 수 있습니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

### 빈 프로필 (Blank profile)

```bash
hermes profile create mybot
```

내장(bundled) 스킬이 기본적으로 포함된 새 프로필을 생성합니다. `mybot setup`을 실행하여 API 키, 모델, 게이트웨이 토큰을 구성하세요.

이 프로필을 칸반 워커(kanban worker)로 사용하거나 칸반 오케스트레이터(kanban orchestrator)가 이 프로필로 작업을 라우팅하도록 하려면, 생성 시 오케스트레이터가 이 봇의 특기를 알 수 있도록 `--description "<역할>"`을 전달하세요:

```bash
hermes profile create researcher --description "소스 코드와 외부 문서를 읽고, 조사 결과를 작성합니다."
```

나중에 `hermes profile describe`를 사용하여 설명을 설정하거나 자동 생성할 수도 있습니다 — 전체 라우팅 모델에 대해서는 [칸반 가이드](./features/kanban#auto-vs-manual-orchestration)를 참조하세요.

### 구성만 복제하기 (`--clone`)

```bash
hermes profile create work --clone
```

현재 프로필의 `config.yaml`, `.env`, `SOUL.md`를 새 프로필로 복사합니다. API 키와 모델은 동일하지만, 세션과 기억은 비어 있습니다. 다른 API 키를 설정하려면 `~/.hermes/profiles/work/.env`를 수정하거나 다른 성격(personality)을 원하면 `~/.hermes/profiles/work/SOUL.md`를 수정하세요.

### 모든 것 복제하기 (`--clone-all`)

```bash
hermes profile create backup --clone-all
```

구성, API 키, 성격, 모든 기억, 전체 세션 기록, 스킬, 크론 작업, 플러그인 등 **모든 것**을 복사합니다. 완전한 스냅샷입니다. 백업용으로 쓰거나 이미 컨텍스트가 풍부한 에이전트를 포크(forking)할 때 유용합니다.

### 특정 프로필에서 복제하기

```bash
hermes profile create work --clone --clone-from coder
```

:::tip Honcho 메모리 + 프로필
Honcho가 활성화된 상태에서 `--clone`을 사용하면, 동일한 사용자 작업 공간(workspace)을 공유하면서도 새 프로필을 위한 전용 AI 피어(peer)가 자동으로 생성됩니다. 각 프로필은 자신만의 관찰(observations)과 정체성을 형성합니다. 자세한 내용은 [Honcho -- 다중 에이전트 / 프로필](./features/memory-providers.md#honcho)을 참조하세요.
:::

## 프로필 사용하기 (Using profiles)

### 명령어 별칭 (Command aliases)

모든 프로필은 자동으로 `~/.local/bin/<이름>` 위치에 명령어 별칭을 갖습니다:

```bash
coder chat                    # coder 에이전트와 대화
coder setup                   # coder의 설정 구성
coder gateway start           # coder의 게이트웨이 시작
coder doctor                  # coder의 상태 확인
coder skills list             # coder의 스킬 목록 확인
coder config set model.default anthropic/claude-sonnet-4
```

별칭은 내부적으로 `hermes -p <이름>`으로 작동하기 때문에 모든 hermes 하위 명령어와 함께 사용할 수 있습니다.

### `-p` 플래그

모든 명령어에서 특정 프로필을 명시적으로 타겟팅할 수도 있습니다:

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # 위치에 관계없이 작동
```

### 기본값 고정하기 (`hermes profile use`)

```bash
hermes profile use coder
hermes chat                   # 이제 coder 프로필을 타겟팅합니다
hermes tools                  # coder의 도구를 구성합니다
hermes profile use default    # 기본 프로필로 다시 전환합니다
```

기본값을 설정하여 인자 없는 일반 `hermes` 명령어가 해당 프로필을 타겟팅하도록 합니다. 마치 `kubectl config use-context`와 같습니다.

### 현재 위치 확인 (Knowing where you are)

CLI는 항상 어느 프로필이 활성화되어 있는지 보여줍니다:

- **프롬프트**: `❯` 대신 `coder ❯`
- **배너**: 시작 시 `Profile: coder` 표시
- **`hermes profile`**: 현재 프로필 이름, 경로, 모델, 게이트웨이 상태 표시

## 프로필 vs 작업 공간 vs 샌드박싱 (Profiles vs workspaces vs sandboxing)

프로필은 종종 작업 공간(workspace)이나 샌드박스(sandbox)와 혼동되지만, 완전히 다른 개념입니다:

- **프로필(profile)** 은 Hermes에게 자체 상태 디렉토리를 제공합니다: `config.yaml`, `.env`, `SOUL.md`, 세션, 메모리, 로그, 크론 작업 및 게이트웨이 상태.
- **작업 공간(workspace)** 또는 **작업 디렉토리(working directory)** 는 터미널 명령어가 시작되는 곳입니다. 이는 `terminal.cwd`를 통해 별도로 제어됩니다.
- **샌드박스(sandbox)** 는 파일 시스템 접근을 제한하는 것을 말합니다. 프로필은 에이전트를 샌드박싱하지 **않습니다.**

기본 `local` 터미널 백엔드에서, 에이전트는 여전히 사용자 계정과 동일한 파일 시스템 접근 권한을 갖습니다. 프로필이 있다고 해서 프로필 디렉토리 외부 폴더에 접근하는 것을 막을 수는 없습니다.

프로필이 특정 프로젝트 폴더에서 시작되길 원한다면, 해당 프로필의 `config.yaml`에 절대 경로로 명시적인 `terminal.cwd`를 설정하세요:

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

로컬 백엔드에서 `cwd: "."`를 사용하는 것은 "프로필 디렉토리"가 아니라 "Hermes가 실행된 디렉토리"를 의미합니다.

또한 다음 사항에 주의하세요:

- `SOUL.md`는 모델을 안내할 수는 있지만, 작업 공간의 경계를 강제할 수는 없습니다.
- `SOUL.md`의 변경 사항은 새 세션에서만 깔끔하게 적용됩니다. 기존 세션은 여전히 이전 프롬프트 상태를 사용하고 있을 수 있습니다.
- 모델에게 "지금 어느 디렉토리에 있어?"라고 묻는 것은 격리 상태를 확인하는 신뢰할 수 있는 방법이 아닙니다. 도구를 위한 예측 가능한 시작 디렉토리가 필요하다면 `terminal.cwd`를 명시적으로 설정하세요.

## 게이트웨이 실행 (Running gateways)

각 프로필은 자체 봇 토큰을 사용하는 별도의 프로세스로 자체 게이트웨이를 실행합니다:

```bash
coder gateway start           # coder의 게이트웨이 시작
assistant gateway start       # assistant의 게이트웨이 시작 (별도 프로세스)
```

### 다른 봇 토큰 사용 (Different bot tokens)

각 프로필은 자체 `.env` 파일을 갖습니다. 각각에 다른 Telegram/Discord/Slack 봇 토큰을 구성하세요:

```bash
# coder의 토큰 편집
nano ~/.hermes/profiles/coder/.env

# assistant의 토큰 편집
nano ~/.hermes/profiles/assistant/.env
```

### 안전 기능: 토큰 잠금 (Safety: token locks)

실수로 두 프로필이 동일한 봇 토큰을 사용하는 경우, 두 번째 게이트웨이는 충돌하는 프로필 이름을 명시하는 오류 메시지와 함께 차단됩니다. Telegram, Discord, Slack, WhatsApp, Signal에 대해 지원됩니다.

### 영구 서비스 (Persistent services)

```bash
coder gateway install         # hermes-gateway-coder systemd/launchd 서비스 생성
assistant gateway install     # hermes-gateway-assistant 서비스 생성
```

각 프로필은 자신만의 서비스 이름을 갖습니다. 이들은 독립적으로 실행됩니다.

:::note 공식 Docker 이미지 내부
프로필별 게이트웨이는 [s6-overlay](https://github.com/just-containers/s6-overlay)(컨테이너의 PID 1)에 의해 관리되므로 `hermes profile create <name>`은 `/run/service/gateway-<name>/`에 자동으로 s6 서비스 슬롯을 등록합니다. `hermes -p <name> gateway start/stop/restart`는 단독 프로세스를 생성하는 대신 `s6-svc`로 작업을 디스패치합니다 — 크래시 발생 시 자동 재시작되며 `docker restart` 시 이전에 실행 중이던 게이트웨이 세트를 유지합니다. 자세한 내용은 [프로필별 게이트웨이 관리](/user-guide/docker#per-profile-gateway-supervision)를 참조하세요.
:::

## 프로필 구성 (Configuring profiles)

각 프로필은 다음 파일을 개별적으로 보유합니다:

- **`config.yaml`** — 모델, 제공자, 도구 세트 및 모든 설정
- **`.env`** — API 키, 봇 토큰
- **`SOUL.md`** — 성격 및 지침

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

이 프로필이 기본적으로 특정 프로젝트에서 작업하도록 하려면 자체 `terminal.cwd`를 설정하세요:

```bash
coder config set terminal.cwd /absolute/path/to/project
```

## 업데이트 (Updating)

`hermes update`는 코드를 한 번(공유됨) 가져오고, 새로 제공되는 내장 스킬을 **모든** 프로필에 자동으로 동기화합니다:

```bash
hermes update
# → 코드 업데이트 완료 (12 commits)
# → 스킬 동기화됨: default (최신 상태), coder (새로운 항목 +2), assistant (새로운 항목 +2)
```

사용자가 수정한 스킬은 덮어쓰이지 않습니다.

## 프로필 관리 (Managing profiles)

```bash
hermes profile list           # 상태와 함께 모든 프로필 표시
hermes profile show coder     # 단일 프로필에 대한 세부 정보 표시
hermes profile rename coder dev-bot   # 이름 변경 (별칭 + 서비스 업데이트)
hermes profile export coder   # coder.tar.gz로 내보내기
hermes profile import coder.tar.gz   # 압축 파일에서 가져오기
```

## 프로필 삭제 (Deleting a profile)

```bash
hermes profile delete coder
```

이렇게 하면 게이트웨이가 중지되고, systemd/launchd 서비스가 제거되며, 명령어 별칭이 제거되고, 모든 프로필 데이터가 삭제됩니다. 확인을 위해 프로필 이름을 입력하라는 메시지가 표시됩니다.

확인 절차를 건너뛰려면 `--yes`를 사용하세요: `hermes profile delete coder --yes`

:::note
기본 프로필(`~/.hermes`)은 삭제할 수 없습니다. 모든 것을 제거하려면 `hermes uninstall`을 사용하세요.
:::

## 탭 자동 완성 (Tab completion)

```bash
# Bash
eval "$(hermes completion bash)"

# Zsh
eval "$(hermes completion zsh)"
```

해당 줄을 `~/.bashrc` 또는 `~/.zshrc`에 추가하여 지속적인 자동 완성을 설정하세요. `-p` 뒤의 프로필 이름, 프로필 하위 명령어, 최상위 명령어를 자동 완성합니다.

## 작동 방식 (How it works)

프로필은 `HERMES_HOME` 환경 변수를 사용합니다. `coder chat`을 실행하면 래퍼 스크립트가 hermes를 시작하기 전에 `HERMES_HOME=~/.hermes/profiles/coder`를 설정합니다. 코드베이스 내의 119개 이상의 파일이 `get_hermes_home()`을 통해 경로를 확인하기 때문에, Hermes 상태(구성, 세션, 메모리, 스킬, 상태 데이터베이스, 게이트웨이 PID, 로그, 크론 작업)는 자동으로 해당 프로필의 디렉토리로 범위가 지정됩니다.

이는 터미널 작업 디렉토리와는 별개입니다. 도구 실행은 `HERMES_HOME`이 아니라 `terminal.cwd` (또는 로컬 백엔드에서 `cwd: "."`일 경우 실행 디렉토리)에서 시작됩니다.

기본 프로필은 단순히 `~/.hermes` 자체입니다. 마이그레이션이 필요 없으며 기존 설치 환경에서도 동일하게 작동합니다.

## 배포판으로 프로필 공유하기 (Sharing profiles as distributions)

한 컴퓨터에서 만든 프로필을 **git 저장소**로 패키징하여 다른 컴퓨터(자신의 워크스테이션, 팀원의 노트북 또는 커뮤니티 사용자의 환경)에 단일 명령어로 설치할 수 있습니다. 공유 패키지에는 SOUL, 구성(config), 스킬, 크론 작업 및 MCP 연결 정보가 포함됩니다. 자격 증명(credentials), 기억, 세션은 기기별로 유지됩니다.

```bash
# git 저장소에서 전체 에이전트 설치
hermes profile install github.com/you/research-bot --alias

# 제작자가 새 버전을 출시했을 때 나중에 업데이트 (기억 + .env 유지)
hermes profile update research-bot
```

작성, 게시, 업데이트 시맨틱, 보안 모델 및 활용 사례에 대한 전체 가이드는 **[프로필 배포판: 전체 에이전트 공유](./profile-distributions.md)** 를 참조하세요.
