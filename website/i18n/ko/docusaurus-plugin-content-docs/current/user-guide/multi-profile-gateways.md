---
sidebar_position: 4
---

# 여러 게이트웨이를 동시에 실행하기

여러 [프로필](./profiles.md)을 관리형 서비스로 운영하세요. 각 프로필은 하나의 머신에서 고유한 봇 토큰, 세션, 메모리를 사용합니다. 이 페이지에서는 모든 프로필을 함께 시작하고, 여러 프로필의 로그를 확인하고, 호스트가 절전 모드로 전환되지 않게 하며, 흔히 발생하는 launchd/systemd 실행 문제에서 복구하는 방법을 설명합니다.

Hermes 에이전트를 하나만 실행한다면 이 페이지는 필요하지 않습니다 — 기본 사항은 [프로필](./profiles.md)을 참고하세요. 여러 인스턴스가 *서로 다른* 머신에 있고 하나의 데스크톱 앱에서 동시에 연결하려는 경우에는 [여러 Hermes 인스턴스에 데스크톱 연결하기](./multi-connection-desktop.md)를 참고하세요.

## 이 기능을 사용할 때

두 개 이상의 Hermes 에이전트를 동시에 온라인 상태로 유지하려는 경우 이 구성을 사용하세요. 일반적인 이유는 다음과 같습니다.

- 한 Telegram 봇은 개인 비서로, 다른 봇은 코딩 에이전트로 사용
- 가족 구성원마다 에이전트 하나씩 사용하거나 Slack 워크스페이스마다 하나씩 사용
- 동일한 구성의 샌드박스 인스턴스와 프로덕션 인스턴스
- 메모리와 스킬을 각각 격리한 연구 에이전트 + 글쓰기 에이전트 + cron 기반 봇

모든 프로필에는 이미 플랫폼별 LaunchAgent
(`ai.hermes.gateway-<name>.plist`) 또는 systemd 사용자 서비스
(`hermes-gateway-<name>.service`)가 각각 제공됩니다. 이 가이드에서는 이를 일괄적으로 관리하는 패턴을 추가로 설명합니다.

## 빠른 시작

```bash
# Create profiles (once)
hermes profile create coder
hermes profile create personal-bot
hermes profile create research

# Configure each
coder setup
personal-bot setup
research setup

# Install each gateway as a managed service
coder gateway install
personal-bot gateway install
research gateway install

# Start them all
coder gateway start
personal-bot gateway start
research gateway start
```

이제 끝입니다 — 서로 독립된 세 에이전트가 각각 별도의 프로세스로 실행되며, 충돌하거나 사용자가 로그인할 때 자동으로 다시 시작됩니다.

## 대안: 모든 프로필에 하나의 게이트웨이 사용(멀티플렉싱)

위 모델은 **프로필마다 프로세스 하나**를 실행합니다. 이것이 기본값이며 대부분의 구성에 적합합니다. 하지만 프로필이 많은 호스트나 프로필마다 프로세스 하나를 운영하기 부담스러운 컨테이너 배포 환경에서는 대신 **하나의 멀티플렉싱 게이트웨이**를 실행할 수 있습니다. 기본 프로필의 게이트웨이가 유일한 인바운드 프로세스가 되어 해당 머신의 모든 프로필에 대한 메시지를 처리합니다.

이 기능은 **선택 사항**이며 기본적으로 **꺼져 있습니다**. 꺼져 있을 때는 이 페이지의 내용이 아무것도 바꾸지 않으며 아래 동작은 모두 비활성 상태입니다.

### 멀티플렉싱을 선택할 때

- N개의 supervisor 유닛, N개의 포트, N개의 PID 파일을 관리하기 부담스러운 컨테이너/VPS 배포
- 각자 완전한 프로세스를 실행할 필요가 없는 저트래픽 프로필이 많은 경우
- 시작, 모니터링, 재시작할 대상을 하나로 통합하려는 경우

프로필 간에 프로세스 수준의 강한 격리(별도의 메모리 사용량, 독립적인 충돌 영역, 다른 프로필에 영향을 주지 않고 하나의 프로필만 재시작할 수 있는 기능)가 필요하다면 프로필마다 프로세스 하나를 유지하세요.

### 활성화 방법

**기본 프로필**(멀티플렉서를 소유함)에 플래그를 설정하고 해당 게이트웨이를 재시작하세요.

```bash
hermes config set gateway.multiplex_profiles true
hermes gateway restart
```

기본 프로필의 `~/.hermes/config.yaml`에 다음과 같이 설정해도 됩니다.

```yaml
gateway:
  multiplex_profiles: true
```

(편의를 위해 최상위 수준의 `multiplex_profiles: true`도 허용됩니다.) 다음 시작 시 기본 게이트웨이는 모든 프로필을 열거하고, 각 프로필의 고유 자격 증명으로 활성화된 플랫폼을 시작하며, 각 인바운드 메시지를 해당 프로필로 라우팅합니다. 각 턴마다 라우팅된 프로필의 구성, 스킬, 메모리, SOUL, **provider 키**를 확인하며 자격 증명은 프로필 간에 공유되지 않습니다.

보조 프로필에는 `hermes gateway start`를 실행하지 **않습니다** — 기본 게이트웨이가 이를 처리합니다. 아래 계약 변경 사항을 참고하세요.

### 멀티플렉싱이 켜졌을 때 달라지는 점

플래그를 활성화하면 몇 가지 동작이 달라집니다. 플래그를 끄는 즉시 이 변경 사항은 모두 되돌아갑니다.

#### 1. 보조 프로필은 자체 게이트웨이를 시작하면 안 됩니다

멀티플렉서가 실행 중일 때 이름이 지정된 프로필에서 `hermes gateway start` / `run`을 실행하면 멀티플렉서를 안내하는 **치명적 오류**가 발생합니다.

```
The default gateway is running as a profile multiplexer and already serves
profile 'coder'. ...
```

멀티플렉서는 유일한 인바운드 프로세스입니다. 두 번째 프로필 게이트웨이를 실행하면 해당 프로필의 플랫폼에 중복으로 바인딩됩니다. 멀티플렉서가 실행 중인 동안 별도 프로세스를 의도적으로 실행하려는 경우에만 `--force`를 전달하세요(권장하지 않음). 따라서 이 페이지 앞부분의 프로필 간 수명 주기 래퍼 스크립트는 멀티플렉스 모드에서 사용하지 않으며 — 기본 게이트웨이만 관리합니다.

#### 2. HTTP 인바운드 플랫폼은 `/p/<profile>/` URL 접두사로 접근합니다

보조 프로필의 Webhook(및 기타 HTTP 인바운드) 트래픽은 두 번째 포트가 아니라 프로필 접두사가 붙은 기본 리스너로 들어옵니다.

```
# default profile
POST http://host:8644/webhooks/<route>
# the "coder" profile, same listener
POST http://host:8644/p/coder/webhooks/<route>
```

접두사에 알 수 없거나 구성되지 않은 프로필을 지정하면 `404`가 반환됩니다. 하나의 공유 리스너가 이 방식으로 모든 프로필을 처리하므로 **보조 프로필은 포트 바인딩 플랫폼을 자체적으로 활성화하면 안 됩니다** — 그렇게 하면 구성 오류가 발생하여 해당 보조 프로필 전체가 건너뛰고, 기본 프로필과 다른 정상 프로필은 계속 실행됩니다. 경고에는 건너뛴 프로필과 충돌하는 모든 플랫폼이 표시됩니다.

```
Skipping secondary profile 'coder' due to port-binding config error: Profile
'coder' enables port-binding platform(s) webhook, but gateway.multiplex_profiles
is on. ... Remove these platform entries from profile 'coder's config.yaml or
configure them only on the default profile.
```

이 규칙이 적용되는 포트 바인딩 플랫폼은 `webhook`, `api_server`,
`msgraph_webhook`, `feishu`, `wecom_callback`, `bluebubbles`, `sms`,
`whatsapp_cloud`, `line`입니다. 이들 중 어떤 것이든 **기본 프로필에서만** 구성하세요. 모든 프로필은 `/p/<profile>/` 접두사를 통해 접근할 수 있습니다.

인증은 URL에 지정된 프로필을 따릅니다. 접두사가 없는 엔드포인트는 기본 리스너의 기존 자격 증명을 계속 사용합니다.

- `/p/coder/...` API-server 요청은 `~/.hermes/profiles/coder/.env`의 `API_SERVER_KEY`를 사용해야 하며, 기본 리스너 키는 거부됩니다.
- `coder`를 대상으로 하는 webhook 라우트는 기본 프로필의 `config.yaml`에서 기존 라우트별 `secret` 옆에 `profile: coder`를 선언해야 합니다. 그러면 해당 secret은 `/p/coder/webhooks/<route>`에서만 허용되고 다른 모든 프로필 접두사에서는 거부됩니다.
- `profile`이 없는 webhook 라우트는 기본 프로필 라우트로 남으며 이름이 지정된 프로필 접두사로는 접근할 수 없습니다.

보조 프로필 구성에서는 포트 바인딩 플랫폼을 비활성화한 상태로 유지하세요. 공유 리스너와 라우트 정의는 기본 프로필에 남고, 프로필 바인딩은 인증된 각 webhook 라우트를 어느 프로필에서 실행할 수 있는지 제어합니다. 대상 프로필에 `API_SERVER_KEY`가 없으면 이름이 지정된 API 요청은 안전하게 거부됩니다.

공유 리스너 충돌만 프로필을 건너뛴 상태로 저하시킵니다. 보안 구성 오류는 치명적입니다. 예를 들어 `GATEWAY_ALLOW_ALL_USERS` 또는 플랫폼별 allow-all 활성화 없이 `open` 자체 정책 플랫폼을 사용하면 안전하지 않은 프로필을 조용히 제거하지 않고 게이트웨이 시작을 중단합니다.

#### 3. 자격 증명별 플랫폼은 프로필마다 자체 토큰이 필요합니다

폴링/연결 플랫폼(Telegram, Discord, Slack, Matrix, Signal, …)은 멀티플렉싱에서도 정상적으로 작동하지만, 이를 활성화하는 각 프로필은 **자체 봇 토큰**을 제공해야 합니다 — 동일한 토큰을 두 프로필에서 동시에 폴링할 수 없습니다. 두 프로필이 동일한 `(platform, token)`을 구성하면 시작이 두 프로필의 이름을 표시하며 즉시 실패합니다([토큰 충돌 안전성](#token-conflict-safety) 참고 — 규칙은 그대로이며 이제 하나의 프로세스 내부에서 적용됩니다).

#### 4. 세션 키는 프로필별로 네임스페이스가 지정됩니다

각 프로필의 세션은 `agent:<profile>:…` 네임스페이스 아래에 저장되므로 동일한 플랫폼/채팅을 사용하는 두 프로필도 공유 세션 저장소에서 충돌하지 않습니다. **기본** 프로필은 기존의 `agent:main:…` 네임스페이스를 바이트 단위로 그대로 유지하므로 기존 기본 프로필 세션에는 영향을 주지 않습니다 — 마이그레이션도, 유실된 기록도 없습니다.

#### 5. PID/잠금 하나와 상태 화면 하나

프로세스 수준 PID와 잠금은 하나뿐입니다(기본 홈 아래의 멀티플렉서). `hermes status`는 멀티플렉서와 멀티플렉서가 처리하는 프로필을 표시하고, `hermes status -p <name>`은 하나의 프로필만 보여줍니다. 각 프로필은 여전히 자체 홈 아래에 `runtime_status.json`을 기록하므로 기존의 프로필별 리더도 계속 작동합니다.

#### 변경되지 않는 사항

프로필별 `.env` 자격 증명 격리는 유지되며, 오히려 더 엄격해집니다. 프로필의 키는 자체 범위에서 확인되며 공유 환경으로 합쳐지지 않습니다(따라서 MCP 서버와 Kanban worker 같은 하위 프로세스도 자신의 프로필 secret만 볼 수 있습니다). Kanban, 프로필 범위 스킬/메모리/SOUL, 모델 라우팅은 별도의 게이트웨이를 사용할 때와 정확히 동일하게 프로필별로 동작합니다.

### 선택한 프로필 제공하기

기본적으로 `gateway.multiplex_profiles: true`는 호스트의 유효한 이름 지정 프로필을 모두 제공합니다. 관련 없는 프로필은 설치된 상태로 유지하되 해당 어댑터나 cron 작업을 시작하지 않으려면 `gateway.multiplex_profile_allowlist`를 설정하세요.

```yaml
gateway:
  multiplex_profiles: true
  multiplex_profile_allowlist:
    - worker
    - guest
```

기본 프로필은 항상 제공되므로 목록에 넣을 필요가 없습니다. allowlist를 설정하지 않으면 기존의 전체 제공 동작이 유지되고, 빈 목록을 설정하면 기본 프로필만 제공됩니다. 이름은 정규화되고 중복 제거됩니다. 유효하지 않은 목록 항목이나 설치되지 않은 이름은 경고와 함께 건너뜁니다. 목록이 아닌 잘못된 값은 안전하게 기본 프로필만 제공하는 동작으로 처리됩니다.

결과적으로 제공되는 프로필 집합은 `/p/<profile>/` API 및 webhook 접두사, 런타임 상태, 프로필 라우트 적격성, 프로세스 내 cron 스케줄러가 틱할 프로필도 제어합니다. allowlist 밖의 이름 지정 프로필은 자체 독립 게이트웨이를 계속 실행할 수 있습니다.

### 공유 봇 채팅을 프로필로 라우팅하기(`profile_routes`)

멀티플렉싱은 자격 증명(각 프로필의 자체 봇 토큰) 또는 URL 접두사(HTTP 플랫폼의 `/p/<profile>/`)별로 프로필을 선택합니다. 여러 커뮤니티가 **하나의** 봇 토큰을 공유하는 경우 — 예를 들어 하나의 Discord 봇이 여러 guild를 서비스하는 경우 — `gateway.profile_routes`를 사용하여 특정 guild/채널/스레드를 서로 다른 프로필로 추가 라우팅할 수 있습니다.

```yaml
gateway:
  multiplex_profiles: true
  profile_routes:
    # An entire Discord server → one profile
    - name: acme-server
      platform: discord
      guild_id: "1234567890"
      profile: acme

    # One channel in that server → a different profile
    - name: acme-support
      platform: discord
      guild_id: "1234567890"
      chat_id: "9876543210"
      profile: acme-support

    # A Telegram group (no guild concept — chat_id only)
    - name: tg-group
      platform: telegram
      chat_id: "-1001234567890"
      profile: tg-profile
```

라우트는 가장 구체적인 것부터 매칭됩니다(`thread_id` > `chat_id` > `guild_id`). 선언된 모든 필드가 충족되어야 하며(AND), 채널을 기준으로 지정된 라우트는 상위 채널이 해당 채널인 스레드/포럼 게시물에도 일치합니다. 어떤 라우트에도 일치하지 않는 메시지는 기본/활성 프로필에 남습니다. 라우팅된 프로필에는 위에서 설명한 전체 프로필별 격리(구성, 스킬, 메모리, 자격 증명, 세션 네임스페이스)가 적용됩니다. 라우팅은 Discord뿐 아니라 모든 플랫폼 어댑터에서 작동합니다.

`profile_routes`에는 `gateway.multiplex_profiles: true`가 필요하며 멀티플렉싱이 꺼져 있으면 라우트가 무시됩니다. 명시적인 라우트가 일치했지만 대상 프로필이 설치되지 않았거나 `multiplex_profile_allowlist` 밖에 있으면 게이트웨이는 해당 인그레스를 거부하고 라우트와 대상을 로그에 기록합니다. 기본 프로필로 실행하지는 않습니다. 어떤 라우트에도 일치하지 않는 트래픽은 기존의 기본 프로필 동작을 유지합니다.

## 모든 게이트웨이를 한 번에 시작, 중지 또는 재시작하기

CLI에는 단일 프로필 수명 주기 명령이 제공됩니다. 모든 프로필에 작업을 수행하려면 이를 셸 루프로 감싸세요. 아래 스니펫을 `~/.local/bin/hermes-gateways`에 저장하고 `chmod +x`를 실행하세요.

```sh
#!/bin/sh
set -eu

# Add or remove profile names here as you create / delete profiles.
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
hermes-gateways start      # start every configured profile
hermes-gateways stop       # stop every configured profile
hermes-gateways restart    # restart all
hermes-gateways status     # status across all
hermes-gateways list       # delegates to `hermes gateway list`
```

:::tip
`default` 프로필은 `hermes -p default gateway <action>`이 아니라 `hermes gateway <action>`(`-p` 없음)을 대상으로 합니다. 위 래퍼가 두 형식을 모두 처리합니다.
:::

## 하나의 프로필 관리하기

모든 프로필에 설치되는 단축 명령:

```bash
coder gateway run        # foreground (Ctrl-C to stop)
coder gateway start      # start the managed service
coder gateway stop       # stop the managed service
coder gateway restart    # restart
coder gateway status     # status
coder gateway install    # create the LaunchAgent / systemd unit
coder gateway uninstall  # remove the service file
```

이는 `hermes -p coder gateway <action>`과 동일합니다 — 프로필 별칭이 `PATH`에 없거나 스크립트에서 동적으로 프로필을 지정할 때 유용합니다.

## 서비스 파일

각 프로필은 고유한 이름의 자체 서비스를 설치하므로 설치 항목이 서로 충돌하지 않습니다.

| 플랫폼 | 경로 |
| -------- | ----------------------------------------------------------------- |
| macOS    | `~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist`        |
| Linux    | `~/.config/systemd/user/hermes-gateway-<profile>.service`         |

기본 프로필은 기존 이름인 `ai.hermes.gateway.plist` / `hermes-gateway.service`를 유지합니다.

## 로그 확인하기

각 프로필은 자체 로그 파일에 기록합니다.

```bash
# Default profile
tail -f ~/.hermes/logs/gateway.log
tail -f ~/.hermes/logs/gateway.error.log

# Named profile
tail -f ~/.hermes/profiles/<name>/logs/gateway.log
tail -f ~/.hermes/profiles/<name>/logs/gateway.error.log
```

모든 프로필의 로그를 동시에 스트리밍하려면:

```bash
tail -f ~/.hermes/logs/gateway.log ~/.hermes/profiles/*/logs/gateway.log
```

CLI에는 구조화된 로그 뷰어도 있습니다.

```bash
hermes logs -f                  # follow default profile
hermes -p coder logs -f         # follow one profile
hermes logs --help              # filters, levels, JSON output
```

## 실제로 실행 중인 항목 확인하기

```bash
hermes profile list             # profiles + model + gateway state
hermes-gateways status          # full status across every profile
launchctl list | grep hermes    # macOS — PIDs and labels
systemctl --user list-units 'hermes-gateway-*'   # Linux — units
```

## 구성 편집하기

모든 프로필은 자체 디렉터리에 구성을 보관합니다.

```
~/.hermes/profiles/<name>/
├── .env              # API keys, bot tokens (chmod 600)
├── config.yaml       # model, provider, toolsets, gateway settings
└── SOUL.md           # personality / system prompt
```

기본 프로필은 `~/.hermes/`를 직접 사용하며 동일한 세 파일을 둡니다.

아무 편집기나 CLI를 사용하여 편집하세요.

```bash
hermes config set model.model anthropic/claude-sonnet-4    # default profile
coder config set model.model openai/gpt-5                  # named profile
```

`.env` 또는 `config.yaml`을 편집한 후 영향을 받는 게이트웨이를 재시작하세요.

```bash
coder gateway restart
# or, for everything:
hermes-gateways restart
```

## 호스트를 깨어 있는 상태로 유지하기

게이트웨이 프로세스는 하루 종일 실행할 수 있지만 운영체제는 유휴 상태에서 여전히 절전 모드로 전환하려고 합니다. 두 가지 패턴이 있습니다.

### macOS — `caffeinate`

`caffeinate`는 macOS에 내장되어 있으며 실행 중인 동안 절전 모드를 방지합니다. 설치할 필요가 없습니다.

```bash
caffeinate -dis                    # block display, idle, and system sleep
caffeinate -dis -t 28800           # same, auto-exit after 8 hours
caffeinate -i -w $(cat ~/.hermes/gateway.pid) &   # awake while default gateway runs

# Persistent: run in background and forget
nohup caffeinate -dis >/dev/null 2>&1 &
disown

# Inspect / stop
pmset -g assertions | grep -iE 'caffeinate|prevent|user is active'
pkill caffeinate
```

| 플래그   | 효과                                            |
| ------ | ------------------------------------------------- |
| `-d`   | 디스플레이 절전 차단                               |
| `-i`   | 유휴 시스템 절전 차단(기본값)                 |
| `-m`   | 디스크 절전 차단                                  |
| `-s`   | 시스템 절전 차단(AC 전원 연결 Mac만 해당)         |
| `-u`   | 사용자 활동을 시뮬레이션(화면 잠금 방지)     |
| `-t N` | `N`초 후 자동 종료                       |
| `-w P` | PID `P`가 종료되면 종료                           |

:::warning 덮개를 닫으면 여전히 Mac이 절전 모드로 전환됩니다
`caffeinate`는 MacBook의 하드웨어로 발생하는 덮개 닫힘 절전을 무시할 수 없습니다. 덮개를 닫은 상태로 작동하려면 에너지 절약/배터리 환경설정을 변경하거나 타사 도구를 사용하세요.
:::

### Linux — `systemd-inhibit` 또는 `loginctl`

```bash
# Inhibit suspend while a command runs
systemd-inhibit --what=idle:sleep --who=hermes --why="gateways running" \
  sleep infinity &

# Allow user services to keep running after logout (recommended)
sudo loginctl enable-linger "$USER"
```

linger를 활성화하면 systemd 사용자 유닛(`hermes-gateway-<profile>.service` 포함)이 SSH 연결이 끊기거나 재부팅된 후에도 계속 실행됩니다.

## 토큰 충돌 안전성

각 프로필은 플랫폼별로 고유한 봇 토큰을 사용해야 합니다. 두 프로필이 Telegram, Discord, Slack, WhatsApp 또는 Signal 토큰을 공유하면 두 번째 게이트웨이는 충돌한 프로필의 이름을 표시하며 시작을 거부합니다.

감사하려면:

```bash
grep -H 'TELEGRAM_BOT_TOKEN\|DISCORD_BOT_TOKEN' \
     ~/.hermes/.env ~/.hermes/profiles/*/.env
```

## 코드 업데이트

`hermes update`는 최신 코드를 한 번 가져온 뒤 새로 번들된 스킬을 모든 프로필에 동기화합니다.

```bash
hermes update
hermes-gateways restart
```

사용자가 수정한 스킬은 절대 덮어쓰지 않습니다.

## 문제 해결

### "Could not find service in domain for user gui: 501"

이전 `hermes gateway stop` 이후 `hermes gateway start`를 실행했습니다. CLI의 `stop`은 전체 `launchctl unload`를 수행하여 launchd 레지스트리에서 서비스를 제거합니다. CLI는 `start`에서 이 특정 오류를 감지하고 plist를 자동으로 다시 로드합니다(`↻ launchd job was unloaded; reloading service definition`). 서비스가 정상적으로 시작됩니다. 수정할 사항은 없습니다.

### 충돌 후 오래된 PID

프로필의 게이트웨이가 `not running`으로 표시되지만 프로세스가 여전히 실행 중인 경우:

```bash
ps -ef | grep "hermes_cli.*-p <profile>"
cat ~/.hermes/profiles/<profile>/gateway.pid
kill -TERM <pid>          # graceful
kill -KILL <pid>          # if that fails after a few seconds
<profile> gateway start
```

### 한 서비스 강제 초기화

```bash
# macOS
launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist
launchctl load   ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist

# Linux
systemctl --user restart hermes-gateway-<profile>.service
```

### 상태 점검

```bash
hermes doctor                  # default profile
hermes -p <profile> doctor     # one profile
```
