---
sidebar_position: 6
title: "Teams 화상 회의"
description: "Microsoft Graph 웹훅을 사용한 Microsoft Teams 화상 회의 요약 파이프라인 설정"
---

# Microsoft Teams 화상 회의

Hermes가 Microsoft Graph 회의 이벤트를 수집하고, 트랜스크립트(자막)를 우선 가져오며, 필요시 녹화본 및 STT(Speech-to-Text)로 폴백(fallback)한 뒤, 구조화된 요약본을 다운스트림 싱크로 전달하도록 하려면 Teams 회의 파이프라인을 사용하세요.

사전 요구 사항: 기본적인 봇/자격 증명 설정은 [Microsoft Teams](./teams.md) 문서를 참조하세요.

> `hermes gateway setup`을 실행하고 **Teams Meetings**를 선택하면 안내에 따라 설정할 수 있습니다.

이 페이지에서는 다음과 같은 설정 및 활성화에 중점을 둡니다:
- Graph 자격 증명 (Credentials)
- 웹훅 리스너 (Webhook listener) 구성
- Teams 전송(Delivery) 모드
- 파이프라인 구성 형태

실제 운영(day-2 operations), 라이브 배포 전 확인 사항(go-live checks), 운영자 워크시트(operator worksheet)에 대한 내용은 전용 가이드인 [Teams 화상 회의 파이프라인 운영 가이드](/guides/operate-teams-meeting-pipeline)를 참조하세요.

## 이 기능의 역할

파이프라인은 다음과 같은 작업을 수행합니다:
1. Microsoft Graph 웹훅 이벤트를 수신합니다.
2. 회의를 식별하고 트랜스크립트 결과물을 최우선으로 처리합니다.
3. 사용할 수 있는 트랜스크립트가 없는 경우, 녹화본 다운로드 및 STT로 예외 처리(fallback)합니다.
4. 내구성 있는(durable) 작업 상태 및 싱크(sink) 기록을 로컬에 저장합니다.
5. 요약본을 Notion, Linear, 그리고 Microsoft Teams로 기록할 수 있습니다.

운영자 작업은 CLI 환경 내에서 이루어집니다 (플러그인 `teams_pipeline`이 `teams-pipeline` 하위 명령어를 등록합니다. — `hermes plugins enable teams_pipeline`를 실행하거나 `config.yaml`에 `plugins.enabled: [teams_pipeline]`을 설정하여 활성화하세요):

```bash
hermes teams-pipeline validate
hermes teams-pipeline list
hermes teams-pipeline maintain-subscriptions
```

## 사전 요구 사항

회의 파이프라인을 활성화하기 전에 다음 사항이 준비되어야 합니다:

- 정상적으로 작동하는 Hermes 설치 환경
- Teams로 나가는 아웃바운드 메시지 발송이 필요한 경우 기존 [Microsoft Teams 봇 설정](/user-guide/messaging/teams) 완료
- 구독(Subscribe)할 회의 리소스에 대해 필요한 권한이 부여된 Microsoft Graph 애플리케이션 자격 증명
- Microsoft Graph가 웹훅 전송을 위해 호출할 수 있는 퍼블릭 HTTPS URL
- 녹음/녹화 기록과 STT로 예외 처리(fallback)할 경우 `ffmpeg` 설치 완료

## 1단계: Microsoft Graph 자격 증명 추가

`~/.hermes/.env` 파일에 Graph 앱 전용 자격 증명(app-only credentials)을 추가합니다:

```bash
MSGRAPH_TENANT_ID=<tenant-id>
MSGRAPH_CLIENT_ID=<client-id>
MSGRAPH_CLIENT_SECRET=<client-secret>
```

이 자격 증명은 다음과 같은 목적으로 사용됩니다:
- Graph 클라이언트의 기반 작업
- 구독 유지 관리 명령어들
- 회의 식별 및 결과물(artifacts) 획득
- 별도의 Teams 엑세스 토큰이 제공되지 않은 경우 Graph 기반의 Teams 아웃바운드 메시지 전달

## 2단계: Graph 웹훅 리스너 활성화

웹훅 리스너는 `msgraph_webhook`이라는 이름의 게이트웨이 플랫폼입니다. 최소한 리스너를 활성화하고 클라이언트 상태(client state) 값을 설정해야 합니다:

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_HOST=127.0.0.1
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<random-shared-secret>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

리스너가 노출하는 엔드포인트:
- Graph 알림을 위한 `/msgraph/webhook`
- 단순 상태 확인을 위한 `/health`

퍼블릭 HTTPS 엔드포인트를 이 리스너로 라우팅해야 합니다. 예를 들어, 퍼블릭 도메인이 `https://ops.example.com` 이라면 Graph 알림 URL은 대개 다음과 같습니다:

```text
https://ops.example.com/msgraph/webhook
```

## 3단계: Teams 전달(Delivery) 및 파이프라인 동작 구성

회의 파이프라인은 런타임 구성을 기존 `teams` 플랫폼 항목에서 읽어옵니다. 파이프라인에 특화된 설정 값은 `teams.extra.meeting_pipeline` 아래에 존재합니다. Teams 아웃바운드 전달 기능은 기존의 Teams 플랫폼 구성 영역(surface) 내에서 동작합니다.

`~/.hermes/config.yaml` 예시:

```yaml
platforms:
  msgraph_webhook:
    enabled: true
    extra:
      host: 127.0.0.1
      port: 8646
      client_state: "replace-me"
      accepted_resources:
        - "communications/onlineMeetings"

  teams:
    enabled: true
    extra:
      client_id: "your-teams-client-id"
      client_secret: "your-teams-client-secret"
      tenant_id: "your-teams-tenant-id"

      # 아웃바운드 요약본 전달 설정
      delivery_mode: "graph" # 또는 incoming_webhook
      team_id: "team-id"
      channel_id: "channel-id"
      # incoming_webhook_url: "https://..."

      meeting_pipeline:
        transcript_min_chars: 80
        transcript_required: false
        transcription_fallback: true
        ffmpeg_extract_audio: true
        notion:
          enabled: false
        linear:
          enabled: false
```

리스너를 `0.0.0.0`과 같이 루프백(loopback) 방식이 아닌 호스트로 바인딩한다면 `allowed_source_cidrs`에 Microsoft 웹훅 외부 통신 대역을 설정해야 합니다. 루프백 바인딩(`127.0.0.1` / `::1`)은 개발용 터널과 로컬 리버스 프록시 설정을 위해 의도된 형태입니다.

## Teams 전달(Delivery) 모드

이 파이프라인은 기존 Teams 플러그인 내에서 두 가지 Teams 요약본 전달 모드를 지원합니다.

### `incoming_webhook`

Graph를 거쳐 채널 메시지를 직접 생성하지 않고 Teams 내의 웹훅을 통해 단순 전송할 때 사용합니다.

필수 구성:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "incoming_webhook"
      incoming_webhook_url: "https://..."
```

### `graph`

Hermes가 Microsoft Graph를 통해 Teams 채팅방이나 채널로 요약본을 등록하길 원할 때 사용합니다.

지원되는 타겟(targets):
- `chat_id`
- `team_id` + `channel_id`
- `team_id` + 기존 Teams 플랫폼을 위한 `home_channel` 대체 타겟

예시:

```yaml
platforms:
  teams:
    enabled: true
    extra:
      delivery_mode: "graph"
      team_id: "team-id"
      channel_id: "channel-id"
```

## 4단계: 게이트웨이 시작하기

구성을 마친 후 평소처럼 Hermes를 시작하세요:

```bash
hermes gateway run
```

Docker 환경에서 Hermes를 실행 중이라면 사용하는 배포 방식과 동일하게 게이트웨이를 시작하면 됩니다.

리스너 확인:

```bash
curl http://localhost:8646/health
```

## 5단계: Graph 구독 생성

플러그인 CLI를 사용하여 구독 항목을 생성하고 점검합니다.

예시:

```bash
hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllTranscripts \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"

hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllRecordings \
  --notification-url https://ops.example.com/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"
```

:::warning Graph 구독은 72시간 후 만료됩니다

Microsoft Graph는 웹훅 구독을 최대 72시간으로 제한하며 자동으로 갱신해주지 않습니다. 운영 환경 적용(go-live) 전에 반드시 `hermes teams-pipeline maintain-subscriptions` 명령어를 예약(schedule)해야 합니다. 이를 지키지 않으면 수동으로 구독을 갱신하더라도 3일 뒤에는 어떠한 알림도 발생하지 않고 조용히 기능이 멈춰버립니다. 예약 옵션(Hermes cron, systemd 타이머, 기본 crontab 등 3가지)은 운영자 런북 중 [구독 자동 갱신](/guides/operate-teams-meeting-pipeline#automating-subscription-renewal-required-for-production) 항목을 참조하세요.

:::

구독 유지 관리와 세부 운영 흐름에 대해 더 알아보려면 다음 문서를 확인하세요: [Teams 화상 회의 파이프라인 운영 가이드](/guides/operate-teams-meeting-pipeline).

## 유효성 검사 (Validation)

내장된 유효성 검사 스냅샷을 실행합니다:

```bash
hermes teams-pipeline validate
```

유용한 동반 명령어:

```bash
hermes teams-pipeline token-health
hermes teams-pipeline subscriptions
```

## 문제 해결 (Troubleshooting)

| 문제점 | 확인 사항 |
|---------|---------------|
| Graph 웹훅 유효성 검사 실패 | 사용된 public URL이 정확하고 접근 가능한 상태인지, 그리고 Graph가 `/msgraph/webhook` 경로를 정확히 호출하고 있는지 확인합니다. |
| `hermes teams-pipeline list`에 작업이 나타나지 않음 | `msgraph_webhook` 설정이 활성화되어 있는지, 또한 구독 항목들이 올바른 notification URL을 바라보고 있는지 점검합니다. |
| 트랜스크립트 최우선(Transcript-first) 처리 방식이 항상 실패함 | 트랜스크립트 리소스에 대한 Graph 권한을 점검하고 회의 결과물에 실제로 트랜스크립트가 존재하는지 확인합니다. |
| 녹화/녹음 대체(Fallback) 수단 실패 | `ffmpeg`가 설치되어 있는지, 그리고 Graph 앱이 녹화본(recording artifacts)에 엑세스 가능한지 점검합니다. |
| Teams 요약본 발송 실패 | `delivery_mode`, 전송 타겟(target IDs) 및 Teams 권한 설정을 재확인합니다. |

## 관련 문서

- [Microsoft Teams 봇 설정](/user-guide/messaging/teams)
- [Teams 화상 회의 파이프라인 운영 가이드](/guides/operate-teams-meeting-pipeline)
