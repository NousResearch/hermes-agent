---
sidebar_position: 4
---

# 다중 프로필 게이트웨이 실행 (Running Many Gateways at Once)

여러 [프로필](./profiles.md)을 — 각각 고유한 봇 토큰, 세션 및 메모리를 갖춘 — 단일 머신에서 관리되는 서비스로 동시에 운영할 수 있습니다. 이 페이지에서는 여러 게이트웨이를 동시에 시작하고, 프로필 전반의 로그를 확인하고, 호스트의 절전 모드를 방지하며, 일반적인 launchd/systemd 문제에서 복구하는 등의 운영 문제를 다룹니다.

하나의 Hermes 에이전트만 실행하는 경우 이 페이지는 필요하지 않습니다. 기본 사항은 [프로필](./profiles.md)을 참조하세요.

## 언제 사용해야 할까요?

동시에 온라인 상태여야 하는 두 개 이상의 Hermes 에이전트가 있을 때 이 설정이 필요합니다. 일반적인 이유는 다음과 같습니다:

- 하나의 Telegram 봇에는 개인 비서를, 다른 하나에는 코딩 에이전트를 두는 경우
- 가족 구성원당 하나의 에이전트 또는 Slack 작업 공간당 하나의 에이전트
- 동일한 구성의 샌드박스 + 프로덕션 인스턴스
- 각각 분리된 메모리와 스킬을 가진 연구 에이전트 + 작문 에이전트 + 크론 기반 봇

모든 프로필은 이미 플랫폼별 LaunchAgent(`ai.hermes.gateway-<name>.plist`) 또는 systemd 사용자 서비스(`hermes-gateway-<name>.service`)를 얻습니다. 이 가이드는 이들을 집합적으로 관리하기 위한 패턴을 추가합니다.

## 빠른 시작

```bash
# 프로필 생성 (한 번만 수행)
hermes profile create coder
hermes profile create personal-bot
hermes profile create research

# 각각 구성
coder setup
personal-bot setup
research setup

# 각각의 게이트웨이를 관리형 서비스로 설치
coder gateway install
personal-bot gateway install
research gateway install

# 모두 시작
coder gateway start
personal-bot gateway start
research gateway start
```

이게 전부입니다 — 세 개의 독립적인 에이전트가 각각 자체 프로세스에서 실행되며, 충돌 시 또는 사용자 로그인 시 자동으로 다시 시작됩니다.

## 모든 게이트웨이를 한 번에 시작, 중지 또는 재시작하기

CLI는 단일 프로필 수명 주기 명령을 제공합니다. 모든 프로필에 걸쳐 작업을 수행하려면 쉘 루프 안에 래핑하세요. 아래의 스니펫을 `~/.local/bin/hermes-gateways`에 넣고 `chmod +x`를 적용하세요:

```sh
#!/bin/sh
set -eu

# 프로필을 생성/삭제함에 따라 여기에 프로필 이름을 추가하거나 제거하세요.
profiles="default coder personal-bot research"

usage() {
  echo "Usage: hermes-gateways {start|stop|restart|status|list}"
}

run_for_profile() {
  profile="$1"
  action="$2"
  if [ "$profile" = "default" ]; then
    hermes gateway "$action"
  else
    hermes -p "$profile" gateway "$action"
  fi
}

action="${1:-}"
case "$action" in
  start|stop|restart|status)
    for profile in $profiles; do
      echo "==> $action $profile"
      run_for_profile "$profile" "$action"
    done
    ;;
  list)
    hermes gateway list
    ;;
  *)
    usage
    exit 2
    ;;
esac
```

그런 다음:

```bash
hermes-gateways start      # 구성된 모든 프로필 시작
hermes-gateways stop       # 구성된 모든 프로필 중지
hermes-gateways restart    # 모두 재시작
hermes-gateways status     # 모든 프로필의 상태 확인
hermes-gateways list       # `hermes gateway list`에 위임
```

:::tip
`default` 프로필은 `hermes -p default gateway <action>`이 아니라 `hermes gateway <action>` (`-p` 없음)으로 지정됩니다. 위의 래퍼는 두 가지 형태를 모두 처리합니다.
:::

## 단일 프로필 관리

모든 프로필이 설치하는 단축 명령:

```bash
coder gateway run        # 포그라운드 (중지하려면 Ctrl-C)
coder gateway start      # 관리형 서비스 시작
coder gateway stop       # 관리형 서비스 중지
coder gateway restart    # 재시작
coder gateway status     # 상태
coder gateway install    # LaunchAgent / systemd 유닛 생성
coder gateway uninstall  # 서비스 파일 제거
```

이들은 `hermes -p coder gateway <action>`과 동일합니다. 프로필 별칭이 `PATH`에 없거나 스크립트에서 프로필을 동적으로 타겟팅할 때 유용합니다.

## 서비스 파일

각 프로필은 고유한 이름으로 자체 서비스를 설치하므로 설치가 충돌하지 않습니다:

| 플랫폼 | 경로                                                              |
| -------- | ----------------------------------------------------------------- |
| macOS    | `~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist`        |
| Linux    | `~/.config/systemd/user/hermes-gateway-<profile>.service`         |

기본 프로필은 과거 이름(`ai.hermes.gateway.plist` / `hermes-gateway.service`)을 유지합니다.

## 로그 보기

각 프로필은 자체 로그 파일에 기록합니다:

```bash
# 기본 프로필
tail -f ~/.hermes/logs/gateway.log
tail -f ~/.hermes/logs/gateway.error.log

# 이름이 지정된 프로필
tail -f ~/.hermes/profiles/<name>/logs/gateway.log
tail -f ~/.hermes/profiles/<name>/logs/gateway.error.log
```

모든 프로필의 로그를 동시에 스트리밍하려면:

```bash
tail -f ~/.hermes/logs/gateway.log ~/.hermes/profiles/*/logs/gateway.log
```

CLI에는 구조화된 로그 뷰어도 있습니다:

```bash
hermes logs --tail              # 기본 프로필 팔로우
hermes -p coder logs --tail     # 특정 프로필 팔로우
hermes logs --help              # 필터, 수준, JSON 출력 확인
```

## 실제로 실행 중인 항목 식별

```bash
hermes profile list             # 프로필 + 모델 + 게이트웨이 상태
hermes-gateways status          # 모든 프로필의 전체 상태
launchctl list | grep hermes    # macOS — PID 및 레이블
systemctl --user list-units 'hermes-gateway-*'   # Linux — 유닛
```

## 구성 편집

모든 프로필은 자체 디렉토리 내에 구성을 보관합니다:

```
~/.hermes/profiles/<name>/
├── .env              # API 키, 봇 토큰 (chmod 600)
├── config.yaml       # 모델, 제공자, 도구 세트, 게이트웨이 설정
└── SOUL.md           # 성격 / 시스템 프롬프트
```

기본 프로필은 동일한 세 개의 파일과 함께 `~/.hermes/`를 직접 사용합니다.

편집기나 CLI를 통해 이를 편집할 수 있습니다:

```bash
hermes config set model.model anthropic/claude-sonnet-4    # 기본 프로필
coder config set model.model openai/gpt-5                  # 이름이 지정된 프로필
```

`.env` 또는 `config.yaml`을 편집한 후, 영향을 받는 게이트웨이를 재시작하세요:

```bash
coder gateway restart
# 또는 모든 게이트웨이에 대해:
hermes-gateways restart
```

## 호스트 깨어 있게 유지하기

게이트웨이 프로세스는 하루 종일 실행될 수 있지만, 유휴 상태일 때 운영 체제는 여전히 절전 모드로 전환하려고 시도합니다. 두 가지 패턴이 있습니다:

### macOS — `caffeinate`

`caffeinate`는 macOS에 내장되어 있으며 실행되는 동안 절전 모드를 방지합니다. 설치가 필요 없습니다.

```bash
caffeinate -dis                    # 디스플레이, 유휴 및 시스템 절전 차단
caffeinate -dis -t 28800           # 동일, 8시간 후 자동 종료
caffeinate -i -w $(cat ~/.hermes/gateway.pid) &   # 기본 게이트웨이가 실행되는 동안 깨어있게 함

# 영구적: 백그라운드에서 실행하고 잊기
nohup caffeinate -dis >/dev/null 2>&1 &
disown

# 검사 / 중지
pmset -g assertions | grep -iE 'caffeinate|prevent|user is active'
pkill caffeinate
```

| 플래그 | 효과 |
| ------ | ------------------------------------------------- |
| `-d`   | 디스플레이 절전 차단 |
| `-i`   | 유휴 시스템 절전 차단 (기본값) |
| `-m`   | 디스크 절전 차단 |
| `-s`   | 시스템 절전 차단 (AC 전원이 연결된 Mac 전용) |
| `-u`   | 사용자 활동 시뮬레이션 (화면 잠금 방지) |
| `-t N` | `N`초 후 자동 종료 |
| `-w P` | PID `P`가 종료될 때 종료 |

:::warning 덮개를 닫으면 여전히 Mac이 절전 모드로 들어갑니다
`caffeinate`는 MacBook의 하드웨어 기반 덮개 닫기 절전 모드를 재정의할 수 없습니다. 덮개를 닫은 상태에서 작동하려면 에너지 절약 / 배터리 기본 설정을 변경하거나 타사 도구를 사용하세요.
:::

### Linux — `systemd-inhibit` 또는 `loginctl`

```bash
# 명령이 실행되는 동안 일시 중단 방지
systemd-inhibit --what=idle:sleep --who=hermes --why="gateways running" \
  sleep infinity &

# 로그아웃 후에도 사용자 서비스가 계속 실행되도록 허용 (권장)
sudo loginctl enable-linger "$USER"
```

Linger(유지)를 활성화하면 `hermes-gateway-<profile>.service`를 포함한 systemd 사용자 유닛이 SSH 연결 해제 및 재부팅 후에도 계속 실행됩니다.

## 토큰 충돌 안전장치

각 프로필은 플랫폼별로 고유한 봇 토큰을 사용해야 합니다. 두 프로필이 텔레그램, 디스코드, 슬랙, 왓츠앱 또는 시그널 토큰을 공유하는 경우, 두 번째 게이트웨이는 충돌하는 프로필의 이름을 명시한 오류와 함께 시작을 거부합니다.

감사(audit)하려면:

```bash
grep -H 'TELEGRAM_BOT_TOKEN\|DISCORD_BOT_TOKEN' \
     ~/.hermes/.env ~/.hermes/profiles/*/.env
```

## 코드 업데이트

`hermes update`는 최신 코드를 한 번 가져와 새 번들 스킬을 모든 프로필에 동기화합니다:

```bash
hermes update
hermes-gateways restart
```

사용자가 수정한 스킬은 절대 덮어쓰여지지 않습니다.

## 문제 해결

### "Could not find service in domain for user gui: 501"

이전에 `hermes gateway stop`을 수행한 후 `hermes gateway start`를 실행했습니다. CLI의 `stop`은 완전한 `launchctl unload`를 수행하여 launchd의 레지스트리에서 서비스를 제거합니다. CLI는 `start` 시에 이 특정 오류를 포착하고 자동으로 plist를 다시 로드합니다 (`↻ launchd job was unloaded; reloading service definition`). 서비스는 정상적으로 시작됩니다. 고칠 것은 없습니다.

### 충돌 후 오래된 PID가 남은 경우

프로필 게이트웨이가 `not running`으로 표시되지만 프로세스가 여전히 살아있는 경우:

```bash
ps -ef | grep "hermes_cli.*-p <profile>"
cat ~/.hermes/profiles/<profile>/gateway.pid
kill -TERM <pid>          # 우아한 종료
kill -KILL <pid>          # 몇 초 후에도 실패하면 강제 종료
<profile> gateway start
```

### 한 서비스의 하드 리셋 강제 실행

```bash
# macOS
launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist
launchctl load   ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist

# Linux
systemctl --user restart hermes-gateway-<profile>.service
```

### 상태 점검 (Health check)

```bash
hermes doctor                  # 기본 프로필
hermes -p <profile> doctor     # 특정 프로필
```
