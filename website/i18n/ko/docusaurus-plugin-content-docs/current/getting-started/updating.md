---
sidebar_position: 3
title: "업데이트 및 제거"
description: "Hermes Agent를 최신 버전으로 업데이트하거나 제거하는 방법"
---

# 업데이트 및 제거

## 업데이트

다음 한 줄 명령으로 최신 버전으로 업데이트할 수 있습니다.

```bash
hermes update
```

이 명령은 `main`에서 최신 코드를 가져오고, 종속성을 업데이트하며, 마지막 업데이트 이후 추가된 새 옵션을 구성하라는 메시지를 표시합니다.

:::tip
`hermes update`는 새 구성 옵션을 자동으로 감지하고 추가하라는 메시지를 표시합니다. 이 메시지를 건너뛰었다면 `hermes config check`를 직접 실행해 누락된 옵션을 확인한 다음, `hermes config migrate`를 실행해 대화형으로 추가할 수 있습니다.
:::

### 업데이트 중에 수행되는 작업

`hermes update`를 실행하면 다음 단계가 수행됩니다.

1. **업데이트 전 스냅샷** — 기본적으로 가벼운 상태 스냅샷이 저장됩니다(페어링 데이터, cron 작업, `config.yaml`, `.env`, `auth.json` 및 런타임에 수정되는 기타 상태 파일이 포함됩니다. 단, 개별 파일이 1GiB를 초과하면 건너뛰므로 대용량 세션 DB 때문에 업데이트가 느려지지 않습니다). `updates.pre_update_backup`으로 제어하며(`quick`이 기본값이고, `full`은 모든 `HERMES_HOME`을 zip으로 보관하며, `off`는 비활성화), [스냅샷 및 롤백](../user-guide/checkpoints-and-rollback.md)에 설명된 스냅샷 복원 절차를 통해 복구할 수 있습니다.
2. **Git pull** — `main` 브랜치에서 최신 코드를 가져오고 서브모듈을 업데이트합니다.
3. **pull 후 구문 검증 및 자동 롤백** — pull 후 Hermes는 시작 시 모든 `hermes` 호출에서 가져오는 핵심 파일 9개의 컴파일을 수행합니다. 구문 분석에 실패하는 파일이 하나라도 있으면(예: 고립된 병합 충돌 마커나 실수로 잘린 파일) Hermes는 `git reset --hard <pre-pull-sha>`를 실행해 설치를 되돌리므로 셸을 계속 시작할 수 있습니다. 업스트림 수정 사항이 반영된 후 `hermes update`를 다시 실행하세요.
4. **종속성 설치** — 새 종속성이나 변경된 종속성을 가져오기 위해 `uv pip install -e ".[all]"`을 실행합니다.
5. **구성 마이그레이션** — 현재 버전 이후 추가된 새 구성 옵션을 감지하고 설정하라는 메시지를 표시합니다.
6. **게이트웨이 자동 재시작** — 업데이트가 완료되면 실행 중인 게이트웨이가 새 코드의 효과를 즉시 적용하도록 새로 고쳐집니다. 서비스가 관리하는 게이트웨이(Linux의 systemd, macOS의 launchd)는 서비스 관리자를 통해 재시작됩니다. 수동 게이트웨이는 Hermes가 실행 중인 PID를 프로필에 연결할 수 있으면 자동으로 다시 실행됩니다.

### 기본 브랜치가 아닌 브랜치로 업데이트하기: `--branch`

기본적으로 `hermes update`는 `origin/main`을 추적합니다. `--branch <name>`을 전달하면 다른 브랜치를 기준으로 업데이트할 수 있습니다. QA 채널, 기능 브랜치 또는 릴리스 후보 테스트에 유용합니다.

```bash
hermes update --branch release-candidate
hermes update --check --branch experimental   # preview behindness only
```

로컬 체크아웃이 다른 브랜치에 있다면 Hermes는 커밋되지 않은 작업을 자동으로 stash하고, HEAD를 대상 브랜치로 전환한 다음 pull합니다. 로컬에 없는 브랜치는 `origin/<name>`에서 자동으로 추적합니다(`git checkout -B <name> origin/<name>`). 어디에도 존재하지 않는 브랜치는 정상적으로 실패하며, 종료 전에 stash된 변경 사항이 복원되므로 이상한 상태에 남겨지지 않습니다. `main` 브랜치가 아닌 경우 `main` 전용 fork 업스트림 동기화 로직은 자동으로 건너뜁니다.

### 비대화형 업데이트에서의 로컬 변경 사항

터미널에서 `hermes update`를 실행하면 Hermes는 커밋되지 않은 소스 트리 변경 사항을 stash하고 pull한 다음, 변경 사항을 복원할지 **묻습니다**. 이는 지금까지와 동일합니다. 대화형 업데이트에는 아무런 변화가 없습니다.

업데이트가 **터미널 없이** 실행되는 경우(데스크톱/채팅 앱의 "Update" 버튼 또는 게이트웨이에서 트리거된 업데이트)에는 응답할 메시지가 없습니다. `updates.non_interactive_local_changes` 설정이 stash된 변경 사항의 처리 방식을 결정합니다.

```yaml
# ~/.hermes/config.yaml
updates:
  non_interactive_local_changes: stash   # default: keep + auto-restore
  # non_interactive_local_changes: discard  # throw local source edits away
```

- `stash`(기본값) — 자동으로 stash하고 pull한 다음, 업데이트된 코드 위에 변경 사항을 자동으로 복원합니다. 아무것도 잃지 않으며, 복원 중 충돌이 발생하면 수동 복구를 위해 git stash에 보존됩니다.
- `discard` — 자동으로 stash한 뒤 pull 후 stash를 삭제하므로 업데이트가 항상 깨끗한 트리에 적용됩니다. Hermes 소스의 로컬 편집 내용을 보존할 의도가 전혀 없는 컴퓨터에서만 사용하세요. 이 옵션은 stash를 삭제하며(`git reset --hard` + `git clean -fd`가 아님), `node_modules`, `venv`, 빌드 출력물과 같은 무시된 경로에는 절대 손대지 않습니다.

데스크톱 앱에서는 **Settings → Advanced → In-App Update Local Changes**에서 설정합니다.

### 미리 보기 전용: `hermes update --check`

pull하기 전에 업데이트가 있는지 확인하고 싶다면 `hermes update --check`를 실행하세요. 이 명령은 `origin/main`과 비교하기 위해 fetch하지만 커밋은 변경하지 않습니다. 파일은 수정되지 않고 게이트웨이도 재시작되지 않습니다. "업데이트가 있는가"를 기준으로 실행 여부를 결정하는 스크립트와 cron 작업에 유용합니다.

### 전체 업데이트 전 백업: `--backup`

중요한 프로필(프로덕션 게이트웨이, 공유 팀 설치)의 경우 `HERMES_HOME` 전체를 pull 전에 백업하도록 선택할 수 있습니다(config, auth, 세션, 스킬, 페어링 포함).

```bash
hermes update --backup
```

또는 모든 실행에서 이를 기본값으로 지정할 수 있습니다.

```yaml
# ~/.hermes/config.yaml
updates:
  pre_update_backup: full
```

`updates.pre_update_backup`은 세 가지 모드를 사용하는 단일 설정입니다. `quick`(기본값 — 위에서 설명한 가벼운 상태 스냅샷), `full`(빠른 스냅샷에 전체 `HERMES_HOME` zip을 추가하며, 큰 홈 디렉터리에서는 몇 분이 더 걸릴 수 있음), `off`(업데이트 전에 백업하지 않음 — 한 번의 실행에서는 `--no-backup`도 동일)입니다. 기존의 불리언 값도 계속 사용할 수 있습니다. `true`는 `full`, `false`는 `off`를 의미합니다.

:::tip 새 컴퓨터로 이동하려는 경우인가요?
업데이트 백업은 현재 컴퓨터에서 진행하는 업데이트를 보호합니다. 전체 설정을 다른 하드웨어로 이전하는 경우에는 대신 `hermes backup` + `hermes import`를 사용하세요. [Hermes를 다른 컴퓨터로 내보내기](/reference/faq#exporting-hermes-to-another-machine) 및 [`hermes backup`과 `hermes profile export` 비교](/reference/faq#hermes-backup-vs-hermes-profile-export)를 참조하세요.
:::

### Windows: 다른 `hermes.exe`가 실행 중인 경우

Windows에서 `hermes update`는 venv의 진입점 실행 파일을 열어 둔 다른 `hermes.exe` 프로세스를 감지하면 실행을 거부합니다. 가장 흔한 경우는 Hermes Desktop 앱이 생성한 백엔드, 다른 터미널에서 열려 있는 `hermes` REPL 또는 실행 중인 게이트웨이입니다.

```
$ hermes update
✗ Another hermes.exe is running:
    PID 12345  hermes.exe

  Updating now would fail to overwrite ...\venv\Scripts\hermes.exe because
  Windows blocks REPLACE on a running executable.

  Close Hermes Desktop, exit any open `hermes` REPLs, and
  stop the gateway (`hermes gateway stop`) before retrying.
  Override with `hermes update --force` if you've already
  confirmed those processes will not write to the venv.
```

나열된 프로세스를 닫고 다시 실행하세요. 동시 실행 중인 프로세스가 방해하지 않을 것이라고 확신한다면(드물며, 보통 바이러스 백신 shim이 잘못 식별된 경우에만 유용) `--force`를 전달해 검사를 건너뛸 수 있습니다. 이 경우에도 업데이트 프로그램은 `.exe` 이름 변경을 지수 백오프로 계속 재시도하며, 잠금이 풀리지 않으면 `MoveFileEx(MOVEFILE_DELAY_UNTIL_REBOOT)`를 통해 다음 재부팅 시 교체하도록 예약하므로 업데이트를 완료할 수 있습니다.

이와 별개로, Python 인터프리터에서 실행 중인 프로세스가 하나라도 있으면 venv를 건드리지 않는 두 번째 보호 장치가 작동합니다(Desktop 앱의 백엔드, 게이트웨이, Python REPL 등). 이러한 프로세스는 네이티브 확장 파일(`.pyd`)을 잠그며, 액세스 거부 오류로 인해 종속성 동기화가 중간에 종료되면 설치가 버전 사이의 상태에 고립됩니다. 이 보호 장치는 `--force`로 우회할 수 없습니다. 감지된 프로세스가 오탐이라고 확신한다면 명시적인 `hermes update --force-venv`를 사용하세요.

#### Windows venv 재생성은 트랜잭션 방식으로 수행됩니다

Windows 설치 프로그램이 기존 `venv`를 재생성해야 하는 경우, 먼저 이전 디렉터리를 고유한 `venv.stale.*` 이름으로 이동한 다음 대체 디렉터리를 만들고 검증합니다. 새 디렉터리에서 종속성 설치가 완료되고 기준 import가 통과한 후에만 이전 트리를 삭제합니다. 그때까지 이전 트리는 롤백 소스이며 `venv.pending-backup`에 기록됩니다.

이동을 완료할 수 없으면 설치 프로그램은 중지되고 기존 `venv`는 그대로 유지됩니다. `uv`가 실패하거나 인터프리터를 만들지 않고 성공을 보고하면 부분적으로 생성된 대체 디렉터리는 `venv.failed.*`로 이동되고 이전 venv가 복원됩니다. 따라서 설치에 실패한 후에도 상태 및 차단 검사를 사용할 수 있습니다.

다른 프로세스가 여전히 파일 핸들을 소유하고 있으면 `venv.stale.*` 또는 `venv.failed.*` 디렉터리가 남을 수 있습니다. 설치를 사용 중인 Hermes Desktop, 게이트웨이 및 Python 프로세스를 닫은 다음 설치/업데이트를 다시 시도하세요. 남겨진 디렉터리는 재생성이 성공한 후 가능한 범위에서 자동으로 정리됩니다.

예상 출력은 다음과 같습니다.

```
$ hermes update
Updating Hermes Agent...
📥 Pulling latest code...
Already up to date.  (or: Updating abc1234..def5678)
📦 Updating dependencies...
✅ Dependencies updated
🔍 Checking for new config options...
✅ Config is up to date  (or: Found 2 new options — running migration...)
🔄 Restarting gateways...
✅ Gateway restarted
✅ Hermes Agent updated successfully!
```

### 업데이트 후 권장 검증

`hermes update`가 주요 업데이트 경로를 처리하지만, 간단한 검증을 수행하면 모든 변경 사항이 문제없이 적용되었는지 확인할 수 있습니다.

1. `git status --short` — 트리가 예상치 않게 변경되어 있다면 계속하기 전에 확인합니다.
2. `hermes doctor` — config, 종속성 및 서비스 상태를 확인합니다.
3. `hermes --version` — 버전이 예상대로 올라갔는지 확인합니다.
4. 게이트웨이를 사용하는 경우: `hermes gateway status`
5. `doctor`가 npm audit 문제를 보고하면 표시된 디렉터리에서 `npm audit fix`를 실행합니다.

:::warning 업데이트 후 작업 트리가 변경된 경우
`hermes update` 후 `git status --short`에 예상하지 못한 변경 사항이 표시되면 중지하고 계속하기 전에 확인하세요. 이는 대개 로컬 수정 사항이 업데이트된 코드 위에 다시 적용되었거나 종속성 단계에서 lockfile이 새로 고쳐졌다는 의미입니다.
:::

### 업데이트 중 터미널 연결이 끊긴 경우

`hermes update`는 실수로 터미널을 잃어버리는 상황에 대비해 스스로를 보호합니다.

- 업데이트는 `SIGHUP`을 무시하므로 SSH 세션이나 터미널 창을 닫아도 설치 도중 프로세스가 더 이상 종료되지 않습니다. `pip` 및 `git` 자식 프로세스도 이 보호를 상속하므로 연결이 끊겨 Python 환경이 반쯤 설치된 상태로 남지 않습니다.
- 업데이트가 실행되는 동안 모든 출력은 `~/.hermes/logs/update.log`에도 기록됩니다. 터미널이 사라지면 다시 연결해 로그를 확인하고 업데이트가 완료되었는지, 게이트웨이 재시작이 성공했는지 확인하세요.

```bash
tail -f ~/.hermes/logs/update.log
```

- `Ctrl-C`(SIGINT)와 시스템 종료(SIGTERM)는 계속 적용됩니다. 이는 실수가 아니라 의도적인 취소이기 때문입니다.

이제 터미널 연결이 끊겨도 `hermes update`를 유지하기 위해 `screen`이나 `tmux`로 감쌀 필요가 없습니다.

### 현재 버전 확인

```bash
hermes version
```

[GitHub 릴리스 페이지](https://github.com/NousResearch/hermes-agent/releases)에서 최신 릴리스와 비교하세요.

### 메시징 플랫폼에서 업데이트하기

Telegram, Discord, Slack, WhatsApp 또는 Teams에서 다음을 보내 직접 업데이트할 수도 있습니다.

```
/update
```

이 명령은 최신 코드를 가져오고 종속성을 업데이트하며 실행 중인 게이트웨이를 재시작합니다. 재시작하는 동안 봇이 잠시 오프라인 상태가 되었다가(보통 5–15초) 다시 작동합니다.

### 수동 업데이트

수동으로 설치한 경우(빠른 설치 프로그램을 사용하지 않은 경우):

```bash
cd /path/to/hermes-agent
# Activate the venv you created during install (outside the source tree)
export VIRTUAL_ENV="$HOME/.hermes/venvs/hermes-dev"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Pull latest code
git pull origin main

# Reinstall (picks up new dependencies)
uv pip install -e ".[all]"

# Check for new config options
hermes config check
hermes config migrate   # Interactively add any missing options
```

### 롤백 방법

업데이트로 문제가 발생하면 이전 버전으로 롤백할 수 있습니다.

```bash
cd /path/to/hermes-agent

# List recent versions
git log --oneline -10

# Roll back to a specific commit
git checkout <commit-hash>
uv pip install -e ".[all]"

# Restart the gateway if running
hermes gateway restart
```

특정 릴리스 태그로 롤백하려면 이전 태그를 지정하세요(예: `v2026.5.16`과 같은 최근 릴리스 또는 `git tag --sort=-version:refname`에서 확인한 이전 태그).

```bash
git checkout vX.Y.Z
uv pip install -e ".[all]"
```

:::warning
롤백하면 새 옵션이 추가된 경우 config와 호환되지 않을 수 있습니다. 롤백한 후 `hermes config check`를 실행하고, 오류가 발생하면 `config.yaml`에서 인식할 수 없는 옵션을 제거하세요.
:::

### Nix 사용자 참고 사항

Nix는 더 이상 명시적으로 지원되는 설치 경로가 아니며(최선의 노력으로만 지원됨) [Nix 설정](./nix-setup.md)을 참조하세요. Nix flake를 통해 설치했다면 업데이트는 Nix 패키지 관리자를 통해 관리됩니다.

```bash
# Update the flake input
nix flake update hermes-agent

# Or rebuild with the latest
nix profile upgrade hermes-agent
```

Nix 설치는 변경할 수 없으므로 롤백은 Nix의 generation 시스템으로 처리됩니다.

```bash
nix profile rollback
```

자세한 내용은 [Nix 설정](./nix-setup.md)을 참조하세요.

---

## 제거

```bash
hermes uninstall
```

제거 프로그램은 나중에 다시 설치할 수 있도록 구성 파일(`~/.hermes/`)을 보관할지 선택할 수 있게 합니다.

### 수동 제거

```bash
rm -f ~/.local/bin/hermes
rm -rf /path/to/hermes-agent
rm -rf ~/.hermes            # Optional — keep if you plan to reinstall
```

:::info
게이트웨이를 시스템 서비스로 설치했다면 먼저 중지하고 비활성화하세요.
```bash
hermes gateway stop
# Linux: systemctl --user disable hermes-gateway
# macOS: launchctl remove ai.hermes.gateway
```
:::
