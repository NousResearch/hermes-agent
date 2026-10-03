---
title: "Teams 회의 파이프라인 운영"
description: "Microsoft Teams 회의 파이프라인을 위한 런북, go-live 체크리스트 및 운영자 워크시트"
---

# Teams 회의 파이프라인 운영

[Teams Meetings](/user-guide/messaging/teams-meetings)에서 기능을 이미 활성화한 후 이 가이드를 사용하세요.

이 페이지에서는 다음을 다룹니다.
- 운영자 CLI 흐름
- 정기 구독 유지 관리
- 장애 분류
- go-live 점검
- 롤아웃 워크시트

## 핵심 운영자 명령

### 설정 스냅샷 검증

```bash
hermes teams-pipeline validate
```

설정을 변경한 후에는 항상 먼저 실행하세요.

### 토큰 상태 검사

```bash
hermes teams-pipeline token-health
hermes teams-pipeline token-health --force-refresh
```

인증 상태가 오래되었다고 의심되면 `--force-refresh`를 사용하세요.

### 구독 검사

```bash
hermes teams-pipeline subscriptions
```

### 만료가 임박한 구독 갱신

```bash
hermes teams-pipeline maintain-subscriptions
hermes teams-pipeline maintain-subscriptions --dry-run
```

### 구독 갱신 자동화(프로덕션 필수)

**Microsoft Graph 구독은 최대 72시간 후 만료됩니다.** 갱신하는 작업이 없으면 3일 후 회의 알림이 조용히 중단되고 파이프라인이 "고장 난" 것처럼 보입니다. 이는 Graph 기반 통합에서 가장 흔한 운영 장애 원인입니다.

`maintain-subscriptions`를 반드시 일정에 따라 실행해야 합니다. 다음 세 가지 옵션 중 하나를 선택하세요.

#### 옵션 1: Hermes cron(Hermes gateway를 이미 실행 중인 경우 권장)

Hermes에는 내장 cron 스케줄러가 포함되어 있습니다. `--no-agent` 모드는 작업으로 스크립트를 실행하며(LLM을 사용하지 않음), `--script`는 `~/.hermes/scripts/` 아래의 파일을 가리켜야 합니다. 먼저 스크립트를 만드세요.

```bash
mkdir -p ~/.hermes/scripts
cat > ~/.hermes/scripts/maintain-teams-subscriptions.sh <<'EOF'
#!/usr/bin/env bash
exec hermes teams-pipeline maintain-subscriptions
EOF
chmod +x ~/.hermes/scripts/maintain-teams-subscriptions.sh
```

12시간마다 실행되는 스크립트 전용 cron 작업을 등록하세요(72시간 만료 창에 대해 6배의 여유를 제공합니다).

```bash
hermes cron create "0 */12 * * *" \
  --name "teams-pipeline-maintain-subscriptions" \
  --no-agent \
  --script maintain-teams-subscriptions.sh \
  --deliver local
```

등록되었는지 확인하고 다음 실행 시간을 검사하세요.

```bash
hermes cron list
hermes cron status        # scheduler status
```

#### 옵션 2: systemd timer(Linux 프로덕션 배포에 권장)

`/etc/systemd/system/hermes-teams-pipeline-maintain.service`를 만드세요.

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

그리고 `/etc/systemd/system/hermes-teams-pipeline-maintain.timer`를 만드세요.

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

활성화하세요.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-teams-pipeline-maintain.timer
systemctl list-timers hermes-teams-pipeline-maintain.timer
```

#### 옵션 3: 일반 crontab

```cron
0 */12 * * * /usr/local/bin/hermes teams-pipeline maintain-subscriptions >> /var/log/hermes/teams-pipeline-maintain.log 2>&1
```

cron 환경에 `MSGRAPH_*` 자격 증명이 있는지 확인하세요. 가장 간단한 방법은 crontab이 호출하는 래퍼 스크립트 상단에서 `~/.hermes/.env`를 source하는 것입니다.

#### 갱신 작동 여부 확인

일정을 설정한 후 첫 번째 예약 실행이 끝나면 갱신 활동을 확인하세요.

```bash
hermes teams-pipeline subscriptions   # should show expirationDateTime advanced
hermes teams-pipeline maintain-subscriptions --dry-run   # should show "0 expiring soon" most of the time
```

Graph 웹훅이 정확히 약 72시간 후에 신기하게 "작동을 멈추는" 경우, 가장 먼저 확인할 것은 이것입니다. 갱신 작업이 실제로 실행되었나요?

### 최근 작업 검사

```bash
hermes teams-pipeline list
hermes teams-pipeline list --status failed
hermes teams-pipeline show <job-id>
```

### 저장된 작업 재생

```bash
hermes teams-pipeline run <job-id>
```

### 회의 아티팩트 가져오기 드라이런

```bash
hermes teams-pipeline fetch --meeting-id <meeting-id>
hermes teams-pipeline fetch --join-web-url "<join-url>"
```

## 정기 런북

### 최초 설정 후

다음 순서로 실행하세요.

```bash
hermes teams-pipeline validate
hermes teams-pipeline token-health --force-refresh
hermes teams-pipeline subscriptions
```

그런 다음 실제 회의 이벤트를 발생시키거나 기다린 후 다음을 확인하세요.

```bash
hermes teams-pipeline list
hermes teams-pipeline show <job-id>
```

### 일일 또는 주기적 점검

- `hermes teams-pipeline maintain-subscriptions --dry-run` 실행
- `hermes teams-pipeline list --status failed` 검사
- Teams 전달 대상이 여전히 올바른 채팅 또는 채널인지 확인

### 웹훅 URL 또는 전달 대상을 변경하기 전

- 공개 알림 URL 또는 Teams 대상 설정 업데이트
- `hermes teams-pipeline validate` 실행
- 영향을 받는 구독 갱신 또는 재생성
- 새 이벤트가 예상한 수신처에 도착하는지 확인

## 장애 분류

### 작업이 생성되지 않음

다음을 확인하세요.
- `msgraph_webhook`이 활성화되어 있음
- 공개 알림 URL이 `/msgraph/webhook`을 가리킴
- 구독의 client state가 `MSGRAPH_WEBHOOK_CLIENT_STATE`와 일치함
- 원격에 구독이 여전히 존재하며 만료되지 않음

### 작업이 재시도 상태에 머물거나 요약 전에 실패함

다음을 확인하세요.
- 트랜스크립트 권한 및 사용 가능 여부
- 녹화 권한 및 아티팩트 사용 가능 여부
- 녹화 폴백이 활성화된 경우 `ffmpeg` 사용 가능 여부
- Graph 토큰 상태

### 요약은 생성되지만 Teams로 전달되지 않음

다음을 확인하세요.
- `platforms.teams.enabled: true`
- `delivery_mode`
- 웹훅 모드의 `incoming_webhook_url`
- Graph 모드의 `chat_id` 또는 `team_id`와 `channel_id`
- Graph 게시를 사용하는 경우 Teams 인증 설정

### 중복되거나 예상치 못한 재생

다음을 확인하세요.
- `hermes teams-pipeline run`으로 작업을 수동 재생했는지
- 해당 회의의 sink 레코드가 이미 존재하는지
- 로컬 설정에서 재전송 경로를 의도적으로 활성화했는지

## Go-live 체크리스트

- [ ] Graph 자격 증명이 존재하며 올바름
- [ ] `msgraph_webhook`이 활성화되어 있고 공개 인터넷에서 접근 가능함
- [ ] `MSGRAPH_WEBHOOK_CLIENT_STATE`가 설정되어 있으며 구독과 일치함
- [ ] 트랜스크립트 구독이 생성됨
- [ ] STT 폴백이 필요한 경우 녹화 구독이 생성됨
- [ ] 녹화 폴백이 활성화된 경우 `ffmpeg`가 설치됨
- [ ] Teams 아웃바운드 전달 대상이 설정되고 검증됨
- [ ] 실제로 필요한 경우에만 Notion 및 Linear sink가 설정됨
- [ ] `hermes teams-pipeline validate`가 OK 스냅샷을 반환함
- [ ] `hermes teams-pipeline token-health --force-refresh`가 성공함
- [ ] **`maintain-subscriptions`가 일정에 등록됨**(Hermes cron, systemd timer 또는 crontab — [구독 갱신 자동화](#automating-subscription-renewal-required-for-production) 참조). 그렇지 않으면 Graph 구독이 72시간 이내에 조용히 만료됩니다.
- [ ] 실제 종단 간 회의 이벤트가 저장된 작업을 생성함
- [ ] 하나 이상의 요약이 의도한 전달 sink에 도달함

## 전달 모드 결정 가이드

| 모드 | 다음과 같은 경우 사용 | 트레이드오프 |
|------|----------|----------|
| `incoming_webhook` | Teams에 간단히 게시하기만 하면 되는 경우 | 설정이 가장 간단하고 제어력이 낮음 |
| `graph` | Graph를 통해 채널 또는 채팅에 게시해야 하는 경우 | 제어력이 높지만 인증 및 대상 설정이 더 필요함 |

## 운영자 워크시트

롤아웃 전에 작성하세요.

| 항목 | 값 |
|------|-------|
| 공개 알림 URL | |
| Graph 테넌트 ID | |
| Graph 클라이언트 ID | |
| 웹훅 클라이언트 상태 | |
| 트랜스크립트 리소스 구독 | |
| 녹화 리소스 구독 | |
| Teams 전달 모드 | |
| Teams 채팅 ID 또는 팀/채널 | |
| Notion 데이터베이스 ID | |
| Linear 팀 ID | |
| 저장소 경로 재정의(있는 경우) | |
| 일일 점검 담당자 | |

## 변경 검토 워크시트

배포를 변경하기 전에 사용하세요.

| 질문 | 답변 |
|----------|--------|
| 공개 웹훅 URL을 변경하나요? | |
| Graph 자격 증명을 교체하나요? | |
| Teams 전달 모드를 변경하나요? | |
| 새 Teams 채팅 또는 채널로 이동하나요? | |
| 구독을 재생성하거나 갱신해야 하나요? | |
| 새로운 종단 간 검증 실행이 필요한가요? | |

## 관련 문서

- [Teams Meetings 설정](/user-guide/messaging/teams-meetings)
- [Microsoft Teams 봇 설정](/user-guide/messaging/teams)
