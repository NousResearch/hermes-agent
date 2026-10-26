---
sidebar_position: 12
title: "스크립트 출력을 메시징 플랫폼으로 파이프하기"
description: "`hermes send`를 사용해 셸 스크립트, cron 작업, CI 훅 또는 모니터링 데몬의 텍스트를 Telegram, Discord, Slack, Signal 및 기타 플랫폼으로 전송합니다."
---

# 스크립트 출력을 메시징 플랫폼으로 파이프하기

`hermes send`는 Hermes에 이미 구성된 모든 메시징 플랫폼으로 메시지를
전송하는 작고 스크립트 친화적인 CLI입니다. 알림을 위한 플랫폼 간
`curl`이라고 생각하면 됩니다. 실행 중인 gateway도, LLM도 필요 없으며,
각 스크립트에 봇 토큰을 다시 붙여 넣을 필요도 없습니다.

다음과 같은 용도로 사용할 수 있습니다.

- 시스템 모니터링(메모리, 디스크, GPU 온도, 장시간 실행 작업 완료)
- CI/CD 알림(배포 완료, 테스트 실패)
- 결과를 알려야 하는 cron 스크립트
- 터미널에서 빠르게 한 번 보내는 메시지
- 모든 도구의 출력을 어디로든 파이프하기 (`make | hermes send --to slack:#builds`)

이 명령은 `hermes gateway`가 이미 사용하는 것과 동일한 자격 증명과
플랫폼 어댑터를 재사용하므로, 유지 관리해야 할 두 번째 설정 영역이
생기지 않습니다.

---

## 빠른 시작

```bash
# Plain text to the home channel for a platform
hermes send --to telegram "deploy finished"

# Pipe in stdout from anything
echo "RAM 92%" | hermes send --to telegram:-1001234567890

# Send a file
hermes send --to discord:#ops --file /tmp/report.md

# Attach a subject/header line
hermes send --to slack:#eng --subject "[CI] build.log" --file build.log

# Thread target (Telegram topic, Discord thread)
hermes send --to telegram:-1001234567890:17585 "threaded reply"

# List every configured target
hermes send --list

# Filter by platform
hermes send --list telegram
```

---

## 인자 레퍼런스

| 플래그 | 설명 |
|------|-------------|
| `-t, --to TARGET` | 대상. [대상 형식](#target-formats)을 참조하세요. |
| `message` (위치 인자) | 메시지 텍스트. `--file` 또는 stdin에서 읽으려면 생략합니다. |
| `-f, --file PATH` | 파일에서 본문을 읽습니다. `--file -`는 stdin을 강제합니다. |
| `-s, --subject LINE` | 본문 앞에 헤더/제목 줄을 추가합니다. |
| `-l, --list` | 사용 가능한 대상을 나열합니다. 선택적 위치 인자로 플랫폼을 필터링할 수 있습니다. |
| `-q, --quiet` | 성공 시 stdout에 출력하지 않습니다(종료 코드만 출력 — 스크립트에 적합). |
| `--json` | 전송 결과의 원시 JSON을 출력합니다. |
| `-h, --help` | 내장 도움말 텍스트를 표시합니다. |

### 대상 형식

| 형식 | 예시 | 의미 |
|------|------|---------|
| `platform` | `telegram` | 해당 플랫폼에 구성된 홈 채널로 전송 |
| `platform:chat_id` | `telegram:-1001234567890` | 특정 숫자 chat / 그룹 / 사용자 |
| `platform:chat_id:thread_id` | `telegram:-1001234567890:17585` | 특정 스레드 또는 Telegram 포럼 토픽 |
| `platform:#channel` | `discord:#ops` | 사람이 읽기 쉬운 채널 이름(채널 디렉터리에서 확인) |
| `platform:+E164` | `signal:+15551234567` | 전화번호를 대상으로 하는 플랫폼: Signal, SMS, WhatsApp |

Hermes가 어댑터를 제공하는 모든 플랫폼을 대상으로 사용할 수 있습니다.
`telegram`, `discord`, `slack`, `signal`, `sms`, `whatsapp`, `matrix`,
`mattermost`, `feishu`, `dingtalk`, `wecom`, `weixin`, `email` 및
그 외 플랫폼이 해당합니다.

### 종료 코드

| 코드 | 의미 |
|------|---------|
| `0` | 전송(또는 목록 조회) 성공 |
| `1` | 플랫폼 수준에서 전송 실패(인증, 권한, 네트워크) |
| `2` | 사용법 / 인자 / 설정 오류 |

종료 코드는 표준 Unix 규칙을 따르므로, 스크립트에서 `curl`이나 `grep`과
동일한 방식으로 분기할 수 있습니다.

---

## 메시지 본문 결정 순서

`hermes send`는 다음 순서로 메시지 본문을 결정합니다.

1. **위치 인자** — `hermes send --to telegram "hi"`
2. **`--file PATH`** — `hermes send --to telegram --file msg.txt`
3. **파이프로 연결된 stdin** — `echo hi | hermes send --to telegram`

stdin이 TTY인 경우(파이프가 없는 경우) Hermes는 입력을 기다리지 않습니다.
대신 명확한 사용법 오류가 표시됩니다. 실수로 본문을 생략했을 때
스크립트가 멈추지 않도록 하기 위한 동작입니다.

---

## 실제 예시

### 모니터링: 메모리 / 디스크 알림

watchdog에 있는 임시 `curl https://api.telegram.org/...` 호출을 다음과
같은 이식성 높은 한 줄로 바꿀 수 있습니다.

```bash
#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  hermes send --to telegram --subject "⚠ MEMORY WARNING" \
    "RAM ${ram_pct}% on $(hostname)"
fi
```

`hermes send`는 Hermes 설정을 재사용하므로, Hermes가 설치된 모든 호스트에서
동일한 스크립트가 작동합니다. 각 머신의 환경 변수로 봇 토큰을 수동으로
내보낼 필요가 없습니다.

:::tip gateway 자체에 알림을 보내지 마세요
gateway 자체에 문제가 생길 때 발생할 수 있는 watchdog(OOM 알림,
디스크 가득 참 알림)의 경우 `hermes send` 대신 최소한의 `curl` 호출을
계속 사용하세요. 시스템이 과부하되어 Python 인터프리터를 로드할 수
없더라도 알림은 전송되기를 원할 것입니다.
:::

### CI / CD: 빌드 및 테스트 결과

```bash
# In .github/workflows/deploy.yml or any CI script
if ./scripts/deploy.sh; then
  hermes send --to slack:#deploys "✅ ${CI_COMMIT_SHA:0:7} deployed"
else
  tail -n 100 deploy.log | hermes send \
    --to slack:#deploys --subject "❌ deploy failed"
  exit 1
fi
```

### Cron: 일일 보고서

```bash
# Crontab entry
0 9 * * * /usr/local/bin/generate-metrics.sh \
  | /home/me/.hermes/bin/hermes send \
      --to telegram --subject "Daily metrics $(date +%Y-%m-%d)"
```

### 장시간 실행 작업: 완료 시 알림

```bash
./train.py --epochs 200 && \
  hermes send --to telegram "training done" || \
  hermes send --to telegram "training failed (exit $?)"
```

### `--json` 및 `--quiet`를 사용한 스크립팅

```bash
# Hard-fail a script if delivery fails; don't clutter logs on success
hermes send --to telegram --quiet "keepalive" || {
  echo "Telegram delivery failed" >&2
  exit 1
}

# Capture the message ID for later editing / threading
msg_id=$(hermes send --to discord:#ops --json "build started" \
  | jq -r .message_id)
```

---

## `hermes send`를 사용하려면 gateway가 실행 중이어야 하나요?

**대부분의 경우 그렇지 않습니다.** 봇 토큰을 사용하는 플랫폼(Telegram,
Discord, Slack, Signal, SMS, WhatsApp Cloud API 및 대부분의 기타 플랫폼)의
경우 `hermes send`는 `~/.hermes/.env`와 `~/.hermes/config.yaml`의 자격
증명을 사용해 플랫폼의 REST 엔드포인트를 직접 호출합니다. 메시지가
전송되는 즉시 종료되는 독립 실행형 하위 프로세스입니다.

지속적인 어댑터 연결에 의존하는 **플러그인 플랫폼**(예: 장시간 실행되는
WebSocket을 유지하는 사용자 지정 플러그인)에는 실행 중인 gateway가
필요합니다. 이 경우 gateway를 가리키는 명확한 오류가 표시됩니다.
`hermes gateway start`로 시작한 뒤 다시 시도하세요.

---

## 대상 나열 및 검색

특정 채널로 전송하기 전에 사용 가능한 대상을 확인할 수 있습니다.

```bash
# Every target across every configured platform
hermes send --list

# Just Telegram targets
hermes send --list telegram

# Machine-readable
hermes send --list --json
```

목록은 `~/.hermes/channel_directory.json`에서 생성되며, gateway가 실행
중인 동안 몇 분마다 갱신합니다. "no channels discovered yet"가 표시되면
gateway를 한 번 시작하여(`hermes gateway start`) 캐시를 채우세요.

사람이 읽기 쉬운 이름(`discord:#ops`, `slack:#engineering`)은 전송 시점에
이 캐시를 기준으로 확인되므로 숫자 ID를 외울 필요가 없습니다.

---

## 다른 접근 방식과 비교

| 접근 방식 | 멀티 플랫폼 | Hermes 자격 증명 재사용 | gateway 필요 | 가장 적합한 용도 |
|----------|------------|---------------------|--------------|----------------|
| `hermes send` | ✅ | ✅ | 없음 (봇 토큰) | 아래의 모든 용도 |
| 각 플랫폼에 직접 `curl` | 스크립트마다 별도 구성 | 수동 | 없음 | 핵심 watchdog |
| `--deliver`를 사용하는 `cron` 작업 | ✅ | ✅ | 없음 | 예약된 에이전트 작업 |

`hermes send`는 의도적으로 가장 단순한 표면으로 만들어졌습니다. 에이전트가
무슨 말을 할지 결정해야 한다면 cron 작업을 예약하세요. 에이전트의 최종
응답이 구성된 `deliver:` 대상으로 자동 전송됩니다(에이전트가 더 이상
직접 메시지를 보내지 않습니다). LLM이 생성한 콘텐츠를 예약 실행해야
한다면 `cronjob(action='create', prompt=...)`에 `deliver='telegram:...'`을
사용하세요. 원시 문자열만 파이프하면 된다면 `hermes send`를 사용하세요.

---

## 관련 문서

- [Cron으로 무엇이든 자동화하기](/guides/automate-with-cron) — 출력이 모든 플랫폼으로 자동 전송되는 예약 작업.
- [Gateway 내부 구조](/developer-guide/gateway-internals) — cron 전송과 `hermes send`가 공유하는 전송 라우터.
- [메시징 플랫폼 설정](/user-guide/messaging/) — 각 플랫폼의 초기 설정.
