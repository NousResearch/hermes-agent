---
sidebar_position: 12
title: "메시징 플랫폼으로 스크립트 출력 파이핑"
description: "셸 스크립트, 크론 작업, CI 훅(hook), 또는 모니터링 데몬의 텍스트를 `hermes send`를 사용하여 Telegram, Discord, Slack, Signal 및 기타 플랫폼으로 전송합니다."
---

# 메시징 플랫폼으로 스크립트 출력 파이핑(Piping)

`hermes send`는 Hermes에 이미 구성된 모든 메시징 플랫폼으로 메시지를 푸시하는 스크립트 작성 가능한 작은 CLI 도구입니다. 알림을 위한 크로스 플랫폼 `curl`이라고 생각하시면 됩니다 — 게이트웨이를 실행할 필요가 없고, LLM도 필요 없으며, 각 스크립트마다 봇 토큰을 매번 다시 붙여넣을 필요가 없습니다.

다음과 같은 용도로 사용하세요:

- 시스템 모니터링 (메모리, 디스크, GPU 온도, 장기 실행 작업 완료 등)
- CI/CD 알림 (배포 완료, 테스트 실패 등)
- 결과를 알려주는 크론 스크립트
- 터미널에서 보내는 간단한 일회성 메시지
- 모든 도구의 출력을 원하는 곳으로 파이핑 (`make | hermes send --to slack:#builds`)

이 명령어는 `hermes gateway`가 이미 사용 중인 자격 증명과 플랫폼 어댑터를 재사용하므로, 유지 관리해야 할 두 번째 구성 요소가 없습니다.

---

## 빠른 시작

```bash
# 특정 플랫폼의 홈 채널로 일반 텍스트 전송
hermes send --to telegram "deploy finished"

# 어떤 것이든 stdout에서 파이핑
echo "RAM 92%" | hermes send --to telegram:-1001234567890

# 파일 전송
hermes send --to discord:#ops --file /tmp/report.md

# 제목/헤더 줄 추가
hermes send --to slack:#eng --subject "[CI] build.log" --file build.log

# 특정 스레드 대상 (Telegram 토픽, Discord 스레드)
hermes send --to telegram:-1001234567890:17585 "threaded reply"

# 구성된 모든 대상 목록 표시
hermes send --list

# 플랫폼별 필터링
hermes send --list telegram
```

---

## 인수 레퍼런스

| 플래그 | 설명 |
|------|-------------|
| `-t, --to TARGET` | 목적지. [대상(Target) 형식](#대상-형식) 참조. |
| `message` (위치 인수) | 메시지 텍스트. `--file`이나 stdin에서 읽으려면 생략. |
| `-f, --file PATH` | 파일에서 본문을 읽어옵니다. `--file -`는 stdin 사용을 강제합니다. |
| `-s, --subject LINE` | 본문 앞에 헤더/제목 줄을 덧붙입니다. |
| `-l, --list` | 사용 가능한 대상을 나열합니다. 선택적으로 위치 인수를 통해 플랫폼 필터링 가능. |
| `-q, --quiet` | 성공 시 stdout 출력 없음 (종료 코드만 반환 — 스크립트에 이상적). |
| `--json` | 전송 결과를 처리되지 않은 JSON으로 내보냅니다. |
| `-h, --help` | 내장 도움말 텍스트를 표시합니다. |

### 대상 형식

| 형식 | 예시 | 의미 |
|--------|---------|---------|
| `platform` | `telegram` | 플랫폼의 구성된 홈 채널로 전송 |
| `platform:chat_id` | `telegram:-1001234567890` | 특정 숫자 형식의 채팅 / 그룹 / 사용자 |
| `platform:chat_id:thread_id` | `telegram:-1001234567890:17585` | 특정 스레드 또는 Telegram 포럼 토픽 |
| `platform:#channel` | `discord:#ops` | 사람이 읽을 수 있는 친숙한 채널 이름 (채널 디렉터리에 대해 분석됨) |
| `platform:+E164` | `signal:+15551234567` | 전화번호 기반 플랫폼: Signal, SMS, WhatsApp |

Hermes에서 어댑터를 제공하는 모든 플랫폼은 대상(target)으로 사용될 수 있습니다:
`telegram`, `discord`, `slack`, `signal`, `sms`, `whatsapp`, `matrix`, `mattermost`, `feishu`, `dingtalk`, `wecom`, `weixin`, `email` 등.

### 종료 코드

| 코드 | 의미 |
|------|---------|
| `0` | 전송(또는 나열) 성공 |
| `1` | 플랫폼 수준에서 전송 실패 (인증, 권한, 네트워크 등) |
| `2` | 사용법 / 인수 / 구성 오류 |

종료 코드는 표준 유닉스 규칙을 따르므로 여러분의 스크립트는 `curl`이나 `grep`에서와 같은 방식으로 조건 분기를 할 수 있습니다.

---

## 메시지 본문 해결(Resolution) 방식

`hermes send`는 다음 순서로 메시지 본문을 해결합니다:

1. **위치 인수** — `hermes send --to telegram "hi"`
2. **`--file PATH`** — `hermes send --to telegram --file msg.txt`
3. **파이핑된 stdin** — `echo hi | hermes send --to telegram`

stdin이 TTY(파이프 아님)인 경우, Hermes는 입력을 기다리지 **않고** 명확한 사용법 오류를 반환합니다. 이는 사용자가 본문을 실수로 누락했을 때 스크립트가 멈추지(hanging) 않게 해줍니다.

---

## 실제 사용 사례

### 모니터링: 메모리 / 디스크 알림

감시 스크립트에서 단발성 `curl https://api.telegram.org/...` 호출을 이식 가능한 한 줄의 명령어로 대체하세요:

```bash
#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  hermes send --to telegram --subject "⚠ MEMORY WARNING" \
    "RAM ${ram_pct}% on $(hostname)"
fi
```

`hermes send`는 Hermes 구성을 재사용하므로, 이 스크립트는 Hermes가 설치된 모든 호스트에서 작동합니다 — 각 시스템의 환경 변수에 수동으로 봇 토큰을 내보낼 필요가 없습니다.

:::tip 게이트웨이 자체에 대한 알림은 주의하세요
게이트웨이 자체가 과부하 상태일 때 (OOM 알림, 디스크 가득 참 알림) 실행될 수 있는 감시 스크립트의 경우, `hermes send` 대신 최소한의 `curl` 호출을 유지하는 것이 좋습니다. 시스템 스래싱(thrashing)으로 인해 Python 인터프리터를 로드할 수 없는 상황에서도 여전히 알림이 전송되기를 원할 것이기 때문입니다.
:::

### CI / CD: 빌드 및 테스트 결과

```bash
# .github/workflows/deploy.yml 또는 기타 CI 스크립트에서
if ./scripts/deploy.sh; then
  hermes send --to slack:#deploys "✅ ${CI_COMMIT_SHA:0:7} deployed"
else
  tail -n 100 deploy.log | hermes send \
    --to slack:#deploys --subject "❌ deploy failed"
  exit 1
fi
```

### 크론: 일일 보고서

```bash
# 크론탭(crontab) 항목
0 9 * * * /usr/local/bin/generate-metrics.sh \
  | /home/me/.hermes/bin/hermes send \
      --to telegram --subject "Daily metrics $(date +%Y-%m-%d)"
```

### 장기 실행 작업: 완료 시 알림(Ping)

```bash
./train.py --epochs 200 && \
  hermes send --to telegram "training done" || \
  hermes send --to telegram "training failed (exit $?)"
```

### `--json`과 `--quiet`을 활용한 스크립팅

```bash
# 전송 실패 시 스크립트를 하드 페일 처리; 성공 시에는 로그를 깔끔하게 유지
hermes send --to telegram --quiet "keepalive" || {
  echo "Telegram delivery failed" >&2
  exit 1
}

# 추후 편집 / 스레딩을 위해 메시지 ID 캡처
msg_id=$(hermes send --to discord:#ops --json "build started" \
  | jq -r .message_id)
```

---

## `hermes send`는 실행 중인 게이트웨이가 필요한가요?

**대체로 아닙니다.** 봇 토큰을 기반으로 하는 모든 플랫폼(Telegram, Discord, Slack, Signal, SMS, WhatsApp Cloud API 등 대부분의 플랫폼)의 경우, `hermes send`는 `~/.hermes/.env` 및 `~/.hermes/config.yaml`의 자격 증명을 사용하여 해당 플랫폼의 REST 엔드포인트를 직접 호출합니다. 이는 메시지가 전송되는 즉시 종료되는 독립된 하위 프로세스입니다.

실행 중인 게이트웨이가 필요한 경우는 지속적인 어댑터 연결에 의존하는 **플러그인 플랫폼**(예: 수명이 긴 WebSocket을 열어두는 사용자 정의 플러그인)뿐입니다. 이 경우 게이트웨이를 가리키는 명확한 오류가 표시되므로, `hermes gateway start`로 게이트웨이를 시작한 후 다시 시도하세요.

---

## 대상 목록 나열 및 검색

특정 채널에 메시지를 보내기 전에, 사용할 수 있는 채널 목록을 확인할 수 있습니다:

```bash
# 구성된 모든 플랫폼의 전체 대상
hermes send --list

# Telegram 대상만 나열
hermes send --list telegram

# 기계가 읽을 수 있는 형식
hermes send --list --json
```

목록은 게이트웨이가 실행되는 동안 몇 분마다 새로 고침하는 `~/.hermes/channel_directory.json`에서 생성됩니다. 만약 "no channels discovered yet(아직 발견된 채널이 없습니다)"라는 메시지가 나타나면, 게이트웨이를 한 번 실행하여(`hermes gateway start`) 캐시를 채울 수 있도록 하세요.

사람이 읽기 쉬운 이름(`discord:#ops`, `slack:#engineering`)은 전송 시 캐시를 통해 분석되므로, 숫자 ID를 외울 필요가 없습니다.

---

## 다른 방법론과의 비교

| 방법 | 다중 플랫폼 지원 | Hermes 자격 증명 재사용 | 게이트웨이 필요 여부 | 최적의 용도 |
|----------|----------------|---------------------|---------------|----------|
| `hermes send` | ✅ | ✅ | 아니오 (봇 토큰) | 아래 모든 경우 |
| 각 플랫폼에 직접 `curl` 호출 | 스크립트를 각각 따로 작성 | 수동 처리 | 아니오 | 중요 감시 시스템(watchdogs) |
| `--deliver`와 함께 사용되는 `cron` 작업 | ✅ | ✅ | 아니오 | 예약된 에이전트 작업 |
| `send_message` 에이전트 도구 | ✅ | ✅ | 아니오 | 에이전트 루프 내부 |

`hermes send`는 의도적으로 가장 단순한 형태를 취합니다. 만약 에이전트가 어떤 내용을 보낼지 결정해야 한다면, 채팅이나 크론 작업 내부에서 `send_message` 도구를 사용하세요. LLM이 생성한 콘텐츠를 정기적으로 실행해야 한다면, `cronjob(action='create', prompt=...)`와 `deliver='telegram:...'`을 함께 사용하세요. 그리고 원본 텍스트를 그대로 보내기만 하면 된다면, `hermes send`를 선택하세요.

---

## 참고 항목

- **[크론을 사용한 자동화 가이드](/guides/automate-with-cron)** — 출력 결과를 플랫폼으로 자동 전달하는 예약 작업
- **[게이트웨이 내부(Internals)](/developer-guide/gateway-internals)** — `hermes send`가 크론 전달과 공유하는 배달 라우터
- **[메시징 플랫폼 설정](/user-guide/messaging/)** — 각 플랫폼에 대한 일회성 구성 가이드
