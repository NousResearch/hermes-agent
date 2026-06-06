---
title: "Teams 미팅 파이프라인 운영 (Operate the Teams Meeting Pipeline)"
description: "Microsoft Teams 미팅 파이프라인을 위한 런북, 출시 체크리스트 및 운영자 워크시트"
---

# Teams 미팅 파이프라인 운영

이 가이드는 [Teams Meetings](/user-guide/messaging/teams-meetings)에서 이미 기능을 활성화한 이후에 사용합니다.

이 페이지는 다음 내용을 다룹니다:
- 운영자 CLI 흐름
- 정기적인 구독 유지 관리
- 실패 분류(Failure triage)
- 출시 점검(Go-live checks)
- 출시(Rollout) 워크시트

## 핵심 운영자 명령어 (Core Operator Commands)

### 구성 스냅샷 유효성 검사

```bash
hermes teams-pipeline validate
```

구성을 변경한 후 이 명령어를 가장 먼저 사용하세요.

### 토큰 상태 검사

```bash
hermes teams-pipeline token-health
hermes teams-pipeline token-health --force-refresh
```

오래된 인증 상태(stale auth state)가 의심될 때 `--force-refresh`를 사용하세요.

### 구독 검사

```bash
hermes teams-pipeline subscriptions
```

### 만료 임박 구독 갱신

```bash
hermes teams-pipeline maintain-subscriptions
hermes teams-pipeline maintain-subscriptions --dry-run
```

### 구독 갱신 자동화 (프로덕션 환경 필수 사항)

**Microsoft Graph 구독은 최대 72시간 내에 만료됩니다.** 이를 갱신하는 조치가 없다면, 미팅 알림은 3일 후 아무 경고 없이 중단되고 파이프라인이 "고장난" 것처럼 보이게 됩니다. 이는 모든 Graph 지원 통합에서 발생하는 가장 큰 1순위 운영 실패 유형입니다.

반드시 일정에 따라 `maintain-subscriptions`를 실행해야 합니다. 다음 세 가지 옵션 중 하나를 선택하세요:

#### 옵션 1: Hermes 크론 (이미 Hermes 게이트웨이를 실행 중인 경우 권장)

Hermes는 내장된 크론 스케줄러를 제공합니다. `--no-agent` 모드는 LLM을 사용하는 대신 스크립트를 작업으로 실행하며, `--script`는 `~/.hermes/scripts/` 아래의 파일을 가리켜야 합니다. 먼저 스크립트를 생성하세요:

```bash
mkdir -p ~/.hermes/scripts
cat > ~/.hermes/scripts/maintain-teams-subscriptions.sh <<'EOF'
#!/usr/bin/env bash
exec hermes teams-pipeline maintain-subscriptions
EOF
chmod +x ~/.hermes/scripts/maintain-teams-subscriptions.sh
```

그런 다음 12시간마다 실행되는 스크립트 전용 크론 작업을 등록합니다 (72시간 만료 창에 대해 6배의 여유 공간을 제공합니다):

```bash
hermes cron create "0 */12 * * *" \
  --name "teams-pipeline-maintain-subscriptions" \
  --no-agent \
  --script maintain-teams-subscriptions.sh \
  --deliver local
```

등록되었는지 확인하고 다음 실행 시간을 검사하세요:

```bash
hermes cron list
hermes cron status        # 스케줄러 상태
```

#### 옵션 2: systemd 타이머 (Linux 프로덕션 배포에 권장)

`/etc/systemd/system/hermes-teams-pipeline-maintain.service` 생성:

```ini
[Unit]
Description=Hermes Teams pipeline subscription maintenance
After=network-online.target

[Service]
Type=oneshot
User=hermes
EnvironmentFile=/etc/hermes/env
ExecStart=/usr/local/bin/hermes teams-pipeline maintain-subscriptions
```

그리고 `/etc/systemd/system/hermes-teams-pipeline-maintain.timer` 생성:

```ini
[Unit]
Description=Run Hermes Teams pipeline subscription maintenance every 12 hours

[Timer]
OnBootSec=5min
OnUnitActiveSec=12h
Persistent=true

[Install]
WantedBy=timers.target
```

활성화:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-teams-pipeline-maintain.timer
systemctl list-timers hermes-teams-pipeline-maintain.timer
```

#### 옵션 3: 일반 crontab

```cron
0 */12 * * * /usr/local/bin/hermes teams-pipeline maintain-subscriptions >> /var/log/hermes/teams-pipeline-maintain.log 2>&1
```

cron 환경에 `MSGRAPH_*` 자격 증명이 있는지 확인하세요. 가장 간단한 수정 방법은: crontab이 호출하는 래퍼 스크립트의 맨 위에서 `~/.hermes/.env`를 source 하는 것입니다.

#### 갱신이 작동하는지 확인

일정을 설정한 후, 첫 번째 예약된 실행이 끝난 뒤 갱신 활동을 확인하세요:

```bash
hermes teams-pipeline subscriptions   # expirationDateTime이 연장되었는지 확인해야 함
hermes teams-pipeline maintain-subscriptions --dry-run   # 대부분의 경우 "0 expiring soon"이 표시되어야 함
```

Graph 웹훅이 정확히 ~72시간 후에 미스터리하게 "작동을 멈추는" 것을 본다면 가장 먼저 확인해야 할 사항입니다: 갱신 작업이 실제로 실행되었는가?

### 최근 작업 검사

```bash
hermes teams-pipeline list
hermes teams-pipeline list --status failed
hermes teams-pipeline show <job-id>
```

### 저장된 작업 다시 재생 (Replay)

```bash
hermes teams-pipeline run <job-id>
```

### 미팅 아티팩트 가져오기(fetch) 시뮬레이션 (Dry-run)

```bash
hermes teams-pipeline fetch --meeting-id <meeting-id>
hermes teams-pipeline fetch --join-web-url "<join-url>"
```

## 정기 런북 (Routine Runbook)

### 첫 설정 후

순서대로 다음을 실행하세요:

```bash
hermes teams-pipeline validate
hermes teams-pipeline token-health --force-refresh
hermes teams-pipeline subscriptions
```

그런 다음 실제 미팅 이벤트를 트리거하거나 기다렸다가 다음을 확인하세요:

```bash
hermes teams-pipeline list
hermes teams-pipeline show <job-id>
```

### 일일 또는 정기 점검

- `hermes teams-pipeline maintain-subscriptions --dry-run` 실행
- `hermes teams-pipeline list --status failed` 검사
- Teams 전송 대상이 여전히 올바른 채팅인지 또는 채널인지 확인

### 웹훅 URL 또는 전송 대상을 변경하기 전

- 공개 알림 URL 또는 Teams 대상 구성 업데이트
- `hermes teams-pipeline validate` 실행
- 영향을 받는 구독 갱신 또는 재생성
- 새로운 이벤트가 예상되는 수신처(sink)에 도착하는지 확인

## 실패 분류 (Failure Triage)

### 작업이 생성되지 않음

확인 사항:
- `msgraph_webhook`이 활성화되어 있는지
- 공개 알림 URL이 `/msgraph/webhook`을 가리키는지
- 구독의 클라이언트 상태가 `MSGRAPH_WEBHOOK_CLIENT_STATE`와 일치하는지
- 원격으로 구독이 여전히 존재하고 만료되지 않았는지

### 작업이 재시도 상태에 머물거나 요약 전 실패함

확인 사항:
- 트랜스크립트(대화록) 권한 및 사용 가능성
- 녹화 권한 및 아티팩트 사용 가능성
- 녹화 폴백(fallback)이 활성화된 경우 `ffmpeg` 사용 가능성
- Graph 토큰 상태

### 요약이 생성되지만 Teams로 전송되지 않음

확인 사항:
- `platforms.teams.enabled: true`
- `delivery_mode`
- 웹훅 모드의 경우 `incoming_webhook_url`
- Graph 모드의 경우 `chat_id` 또는 `team_id`와 `channel_id`
- Graph 포스팅이 사용되는 경우 Teams 인증 구성

### 중복되거나 예상치 못한 재생 (Replays)

확인 사항:
- `hermes teams-pipeline run`으로 작업을 수동으로 재생했는지
- 해당 미팅에 대해 이미 싱크(sink) 기록이 존재하는지
- 로컬 구성에서 의도적으로 재전송(resend) 경로를 활성화했는지

## 출시 체크리스트 (Go-Live Checklist)

- [ ] Graph 자격 증명이 존재하고 정확함
- [ ] `msgraph_webhook`이 활성화되어 있고 퍼블릭 인터넷에서 접근 가능함
- [ ] `MSGRAPH_WEBHOOK_CLIENT_STATE`가 설정되어 있고 구독과 일치함
- [ ] 트랜스크립트 구독이 생성됨
- [ ] STT 폴백이 필요한 경우 녹화 구독이 생성됨
- [ ] 녹화 폴백이 활성화된 경우 `ffmpeg`가 설치됨
- [ ] Teams 아웃바운드 전송 대상이 구성되고 검증됨
- [ ] Notion 및 Linear 싱크는 실제로 필요한 경우에만 구성됨
- [ ] `hermes teams-pipeline validate`가 정상적인(OK) 스냅샷을 반환함
- [ ] `hermes teams-pipeline token-health --force-refresh`가 성공함
- [ ] **`maintain-subscriptions`가 예약되어 있음** (Hermes 크론, systemd 타이머 또는 crontab — [구독 갱신 자동화](#구독-갱신-자동화-프로덕션-환경-필수-사항) 참조). 이것이 없으면 Graph 구독은 72시간 이내에 조용히 만료됩니다.
- [ ] 실제 엔드투엔드(end-to-end) 미팅 이벤트가 저장된 작업을 성공적으로 생성함
- [ ] 하나 이상의 요약이 의도한 전송 수신처(sink)에 도달함

## 전송 모드 (Delivery-Mode) 결정 가이드

| 모드 | 사용해야 할 때 | 절충 사항(Tradeoff) |
|------|----------|----------|
| `incoming_webhook` | Teams에 단순한 포스팅만 필요할 때 | 설정이 가장 간단하지만 제어력이 떨어짐 |
| `graph` | Graph를 통한 채널 또는 채팅 포스팅이 필요할 때 | 제어력이 높지만 인증 및 대상 설정이 더 필요함 |

## 운영자 워크시트

출시 전에 이 양식을 작성하세요:

| 항목 | 값 |
|------|-------|
| 공개 알림 URL (Public notification URL) | |
| Graph 테넌트 ID (Graph tenant ID) | |
| Graph 클라이언트 ID (Graph client ID) | |
| 웹훅 클라이언트 상태 (Webhook client state) | |
| 트랜스크립트 리소스 구독 (Transcript resource subscription) | |
| 녹화 리소스 구독 (Recording resource subscription) | |
| Teams 전송 모드 (Teams delivery mode) | |
| Teams 채팅 ID 또는 팀/채널 (Teams chat ID or team/channel) | |
| Notion 데이터베이스 ID (Notion database ID) | |
| Linear 팀 ID (Linear team ID) | |
| 저장 경로 재정의 (필요한 경우) (Store path override) | |
| 일일 점검 담당자 (Owner for daily checks) | |

## 변경 사항 검토 워크시트 (Change Review Worksheet)

배포 환경을 변경하기 전에 사용하세요:

| 질문 | 답변 |
|----------|--------|
| 공개 웹훅 URL을 변경하고 있습니까? | |
| Graph 자격 증명을 교체하고 있습니까? | |
| Teams 전송 모드를 변경하고 있습니까? | |
| 새로운 Teams 채팅이나 채널로 이동하고 있습니까? | |
| 구독을 다시 생성하거나 갱신해야 합니까? | |
| 완전히 새로운 엔드투엔드 검증 실행이 필요합니까? | |

## 관련 문서 (Related Docs)

- [Teams 미팅 설정 (Teams Meetings setup)](/user-guide/messaging/teams-meetings)
- [Microsoft Teams 봇 설정 (Microsoft Teams bot setup)](/user-guide/messaging/teams)
