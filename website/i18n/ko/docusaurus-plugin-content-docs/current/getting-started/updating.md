---
sidebar_position: 3
title: "업데이트 및 삭제"
description: "How to update Hermes Agent to the latest version or uninstall it"
---

# 업데이트 및 삭제

## 업데이트

### Git 설치 버전

단 한 줄의 명령어로 최신 버전으로 업데이트할 수 있습니다:

```bash
hermes update
```

이 명령어는 `main` 브랜치에서 최신 코드를 가져오고, 의존성을 업데이트하며, 지난 업데이트 이후 새롭게 추가된 설정 옵션이 있는 경우 이를 구성하도록 안내합니다.

### pip 설치 버전

PyPI 릴리즈는 `main` 브랜치의 모든 커밋이 아닌 **태그가 지정된 버전**(주요 및 마이너 릴리즈)을 추적합니다. 다음과 같이 업데이트를 확인하고 업그레이드할 수 있습니다:

```bash
hermes update --check    # PyPI에 더 새로운 릴리즈가 있는지 확인
hermes update            # pip install --upgrade hermes-agent 실행
```

또는 수동으로 진행할 수도 있습니다:

```bash
pip install --upgrade hermes-agent    # 또는: uv pip install --upgrade hermes-agent
```

:::tip
`hermes update`는 새로운 설정 옵션을 자동으로 감지하고 추가할지 묻는 메시지를 표시합니다. 이 프롬프트를 건너뛰었다면, 수동으로 `hermes config check`를 실행하여 누락된 옵션을 확인한 다음, `hermes config migrate`를 실행하여 대화식으로 추가할 수 있습니다.
:::

### 업데이트 과정 (Git 설치 버전)

`hermes update`를 실행하면 다음과 같은 단계가 진행됩니다:

1. **페어링 데이터 스냅샷(Pairing-data snapshot)** — 업데이트 전의 경량 상태 스냅샷을 저장합니다 (`~/.hermes/pairing/`, Feishu 댓글 규칙 및 런타임에 수정되는 기타 상태 파일 포함). 이 스냅샷은 [스냅샷 및 롤백](../user-guide/checkpoints-and-rollback.md)에 설명된 스냅샷 복구 흐름을 통해 복구하거나, Hermes가 `~/.hermes/` 디렉터리 옆에 생성한 가장 최근의 퀵 스냅샷 zip 압축 파일을 해제하여 복구할 수 있습니다.
2. **Git pull** — `main` 브랜치에서 최신 코드를 가져오고 서브모듈을 업데이트합니다.
3. **가져오기 후 구문 검증 및 자동 롤백(Post-pull syntax validation + auto-rollback)** — 코드를 가져온 후, Hermes는 매 `hermes` 실행 시 시작할 때 임포트하는 8개의 주요 핵심 파일을 컴파일합니다. 만약 구문 분석에 실패하는 경우(예: 해결되지 않은 병합 갈등 마커, 실수로 잘린 파일 등), Hermes는 `git reset --hard <pre-pull-sha>`를 실행하여 설치 상태를 롤백함으로써 쉘 환경이 정상 작동하도록 유지합니다. 업스트림(upstream) 수정 사항이 반영된 후 `hermes update`를 다시 실행하십시오.
4. **의존성 설치** — `uv pip install -e ".[all]"`을 실행하여 새로 추가되거나 변경된 의존성을 반영합니다.
5. **설정 마이그레이션(Config migration)** — 현재 버전 이후에 추가된 새로운 설정 옵션을 감지하고 이를 설정하도록 안내합니다.
6. **게이트웨이 자동 재시작** — 업데이트가 완료되면 실행 중인 게이트웨이를 새로고침하여 새로운 코드가 즉시 적용되도록 합니다. 서비스로 관리되는 게이트웨이(Linux의 systemd, macOS의 launchd)는 서비스 관리자를 통해 재시작됩니다. 수동으로 실행된 게이트웨이는 Hermes가 실행 중인 PID를 프로필에 매핑할 수 있는 경우 자동으로 다시 시작됩니다.

### 기본 브랜치가 아닌 다른 브랜치로 업데이트하기: `--branch`

기본적으로 `hermes update`는 `origin/main`을 추적합니다. QA 채널, 기능(feature) 브랜치 또는 릴리즈 후보(release candidate) 테스트와 같이 다른 브랜치를 기준으로 업데이트하려면 `--branch <이름>`을 전달하십시오:

```bash
hermes update --branch release-candidate
hermes update --check --branch experimental   # 변경 사항 유무만 미리보기
```

로컬 저장소가 다른 브랜치에 있는 경우, Hermes는 커밋되지 않은 로컬 작업 내용을 자동으로 스태시(stash)하고, HEAD를 대상 브랜치로 전환한 후 코드를 가져옵니다. 로컬에 존재하지 않는 브랜치는 `origin/<이름>`에서 자동으로 추적하도록 설정됩니다 (`git checkout -B <이름> origin/<이름>`). 어디에도 존재하지 않는 브랜치인 경우 깔끔하게 실패하며, 종료 전에 스태시된 변경 사항이 복원되므로 이상한 상태에 방치되지 않습니다. `main` 브랜치 전용인 포크 업스트림 동기화(fork-upstream sync) 로직은 `main`이 아닌 브랜치에서는 자동으로 건너뜁니다.

### 비대화식 업데이트 시 로컬 변경 사항 처리

터미널에서 `hermes update`를 실행하면, Hermes는 커밋되지 않은 소스 트리 변경 사항을 스태시(stash)하고 코드를 가져온 후, 이전과 동일하게 이를 복원할지 **여부를 묻습니다**. 대화식 업데이트의 작동 방식은 동일합니다.

데스크톱/채팅 앱의 "업데이트" 버튼이나 게이트웨이에 의해 트리거되는 등 **터미널 없이(비대화식으로)** 업데이트가 실행되는 경우 응답할 프롬프트가 없습니다. 이 경우 `updates.non_interactive_local_changes` 설정에 따라 스태시된 변경 사항의 처리 방식이 결정됩니다:

```yaml
# ~/.hermes/config.yaml
updates:
  non_interactive_local_changes: stash   # 기본값: 보존 및 자동 복구
  # non_interactive_local_changes: discard  # 로컬 소스 변경 사항 버리기
```

- `stash` (기본값) — 변경 사항을 자동으로 스태시하고 코드를 가져온 후, 업데이트된 코드 위에 변경 사항을 자동으로 복구합니다. 데이터는 유실되지 않으며, 복구 중 충돌(conflict)이 발생하면 수동 복구를 위해 git 스태시에 보존됩니다.
- `discard` — 자동으로 스태시하고 풀이 끝난 후 해당 스태시를 삭제하여 항상 깨끗한 소스 트리 상태로 업데이트가 진행되도록 합니다. Hermes 소스 코드에 가한 로컬 수정을 유지할 필요가 전혀 없는 환경에서만 이 옵션을 사용하십시오. 이 방식은 스태시만 삭제하는 것(`git reset --hard` 및 `git clean -fd`가 아님)이므로 `node_modules`, `venv` 및 빌드 출력물 같이 git에서 무시되는(ignored) 경로는 건드리지 않습니다.

데스크톱 앱에서는 이 설정을 **Settings → Advanced → In-App Update Local Changes**에서 변경할 수 있습니다.

### 업데이트 미리보기만 수행: `hermes update --check`

코드를 가져오기 전에 업데이트 가능 여부를 확인하고 싶으신가요? `hermes update --check`를 실행하십시오. Git 설치 버전의 경우 `origin/main`과 커밋을 페치(fetch)하고 비교하며, pip 설치 버전의 경우 PyPI에서 최신 릴리즈를 조회합니다. 이 명령으로는 어떤 파일도 수정되지 않고 게이트웨이도 재시작되지 않습니다. "업데이트 확인" 여부를 판별하는 스크립트나 크론(cron) 작업에서 유용하게 사용할 수 있습니다.

### 업데이트 전 전체 백업: `--backup`

중요한 프로필(프로덕션 게이트웨이, 팀 공유 설치 환경 등)을 운영 중인 경우, 코드를 가져오기 전에 `HERMES_HOME` (config, auth, sessions, skills, pairing 등) 전체를 백업하도록 선택할 수 있습니다:

```bash
hermes update --backup
```

또는 매 업데이트마다 기본적으로 백업이 수행되도록 설정할 수도 있습니다:

```yaml
# ~/.hermes/config.yaml
updates:
  pre_update_backup: true
```

이전 빌드에서는 `--backup`이 항상 실행되는 기본 동작이었으나, 데이터 크기가 큰 환경에서는 업데이트 시간이 몇 분씩 추가되는 문제가 있어 현재는 선택 옵션으로 변경되었습니다. 단, 앞서 언급한 경량 페어링 데이터 스냅샷은 조건 없이 항상 실행됩니다.

### Windows: 다른 `hermes.exe` 프로세스가 실행 중인 경우

Windows 환경에서는 가상 환경(venv)의 엔트리 포인트 실행 파일을 점유하고 있는 다른 `hermes.exe` 프로세스가 감지되면 `hermes update` 실행이 거부됩니다. 가장 흔한 원인은 Hermes 데스크톱 앱이 실행한 백엔드, 다른 터미널에서 열려 있는 `hermes` REPL, 또는 실행 중인 게이트웨이 등입니다:

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

목록에 표시된 프로세스를 종료하고 다시 시도하십시오. 동시 실행 중인 프로세스가 업데이트를 방해하지 않을 것이라고 확신하는 경우(드문 경우로, 보통 백신 프로그램 오탐지와 같은 경우에만 해당됨) `--force`를 전달하여 이 확인 과정을 건너뛸 수 있습니다. 이 경우 업데이터는 지수 백오프(exponential backoff) 방식으로 `.exe` 파일 이름 변경을 재시도하며, 잠금이 풀리지 않는 경우 `MoveFileEx(MOVEFILE_DELAY_UNTIL_REBOOT)`를 통해 다음 부팅 시 교체되도록 예약하여 업데이트를 완료합니다.

정상적으로 업데이트가 완료되었을 때의 예상 출력은 다음과 같습니다:

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

### 권장되는 업데이트 후 검증 작업

`hermes update`가 주요 업데이트 과정을 처리하지만, 다음 단계를 통해 모든 항목이 정상적으로 적용되었는지 빠르게 확인할 수 있습니다:

1. `git status --short` — 작업 트리가 예상치 못하게 지저분한(dirty) 상태라면, 계속 진행하기 전에 변경 사항을 확인하십시오.
2. `hermes doctor` — 설정, 의존성 및 서비스 상태를 검증합니다.
3. `hermes --version` — 버전이 예상대로 업데이트되었는지 확인합니다.
4. 게이트웨이를 사용하는 경우: `hermes gateway status`를 실행합니다.
5. `doctor` 명령어가 npm audit 관련 이슈를 보고하면, 해당 디렉터리에서 `npm audit fix`를 실행합니다.

:::warning 업데이트 후 더티 워킹 트리(Dirty working tree) 발생 시
`hermes update`를 실행한 후 `git status --short` 결과에 예상치 못한 변경 사항이 표시되면, 계속 진행하기 전에 작업을 멈추고 검토하십시오. 이는 보통 업데이트된 코드 위에 로컬 수정 사항이 다시 적용되었거나, 의존성 설치 과정에서 락파일(lockfile)이 갱신되었음을 의미합니다.
:::

### 업데이트 도중 터미널 연결이 끊긴 경우

`hermes update`는 터미널 연결이 예상치 못하게 끊어지는 상황으로부터 자신을 보호합니다:

- 업데이트 작업은 `SIGHUP` 시그널을 무시하므로, SSH 세션이나 터미널 창을 닫아도 설치가 도중에 취소되지 않습니다. `pip`와 `git` 자식 프로세스 역시 이 보호 설정을 상속받으므로, 연결이 끊겼다고 해서 Python 환경이 불완전하게 설치된 채로 방치되지 않습니다.
- 업데이트가 실행되는 동안 모든 출력이 `~/.hermes/logs/update.log` 파일에 동시에 기록됩니다. 터미널 창이 사라진 경우, 다시 연결하여 로그를 확인해 업데이트가 정상적으로 완료되었는지, 게이트웨이가 재시작되었는지 확인할 수 있습니다:

```bash
tail -f ~/.hermes/logs/update.log
```

- `Ctrl-C` (SIGINT) 및 시스템 종료 (SIGTERM) 시그널은 정상적으로 처리됩니다. 이는 사고가 아닌 사용자가 의도한 명시적인 취소 요청이기 때문입니다.

더 이상 터미널 유실에 대비해 `screen`이나 `tmux` 안에서 `hermes update`를 실행할 필요가 없습니다.

### 현재 버전 확인하기

```bash
hermes version
```

[GitHub 릴리즈 페이지](https://github.com/NousResearch/hermes-agent/releases)에서 최신 릴리즈 버전과 비교해 보십시오.

### 메시징 플랫폼에서 업데이트하기

Telegram, Discord, Slack, WhatsApp, Teams 등의 메시징 채널에서도 직접 다음과 같은 명령을 전송하여 업데이트할 수 있습니다:

```
/update
```

이 명령은 최신 코드를 가져오고 의존성을 업데이트하며, 실행 중인 게이트웨이를 재시작합니다. 재시작되는 동안 봇은 일시적으로 오프라인 상태가 되며(보통 5~15초 소요), 이후 다시 온라인 상태로 복귀합니다.

### 수동 업데이트

(빠른 설치 스크립트를 사용하지 않고) 수동으로 설치한 경우:

```bash
cd /path/to/hermes-agent
export VIRTUAL_ENV="$(pwd)/venv"

# 최신 코드 가져오기
git pull origin main

# 재설치 (새로운 의존성 반영)
uv pip install -e ".[all]"

# 새로운 설정 옵션 확인
hermes config check
hermes config migrate   # 누락된 옵션을 대화식으로 추가
```

### 롤백 방법

업데이트 후 문제가 발생한 경우 이전 버전으로 되돌릴 수(롤백) 있습니다:

```bash
cd /path/to/hermes-agent

# 최근 버전 목록 확인
git log --oneline -10

# 특정 커밋으로 롤백
git checkout <commit-hash>
uv pip install -e ".[all]"

# 실행 중인 경우 게이트웨이 재시작
hermes gateway restart
```

특정 릴리즈 태그로 되돌리려면 다음과 같이 실행하십시오 (이전 태그로 대체하십시오. 예: 최근 릴리즈인 `v2026.5.16` 또는 `git tag --sort=-version:refname`으로 확인한 이전 태그):

```bash
git checkout vX.Y.Z
uv pip install -e ".[all]"
```

:::warning
롤백 시 새로운 옵션이 추가되었던 경우 설정 호환성 문제가 발생할 수 있습니다. 롤백 후 `hermes config check`를 실행하고, 에러가 발생하면 `config.yaml`에서 인식되지 않는 옵션을 제거하십시오.
:::

### Nix 사용자 참고 사항

Nix flake를 통해 설치한 경우, Nix 패키지 매니저를 통해 업데이트를 관리합니다:

```bash
# flake 입력 업데이트
nix flake update hermes-agent

# 또는 최신 버전으로 빌드
nix profile upgrade hermes-agent
```

Nix 설치 환경은 불변(immutable)이므로, Nix의 세대(generation) 시스템을 통해 롤백이 처리됩니다:

```bash
nix profile rollback
```

자세한 내용은 [Nix 설정](./nix-setup.md)을 참조하십시오.

---

## 삭제

### Git 설치 버전

```bash
hermes uninstall
```

삭제 도구 실행 시, 향후 재설치를 대비하여 설정 파일(`~/.hermes/`)을 유지할지 묻는 옵션이 제공됩니다.

### pip 설치 버전

```bash
pip uninstall hermes-agent
rm -rf ~/.hermes            # 선택 사항 — 재설치할 계획이 있다면 유지
```

### 수동 삭제

```bash
rm -f ~/.local/bin/hermes
rm -rf /path/to/hermes-agent
rm -rf ~/.hermes            # 선택 사항 — 재설치할 계획이 있다면 유지
```

:::info
게이트웨이를 시스템 서비스로 설치한 경우, 먼저 서비스를 중지하고 비활성화하십시오:
```bash
hermes gateway stop
# Linux: systemctl --user disable hermes-gateway
# macOS: launchctl remove ai.hermes.gateway
```
:::
