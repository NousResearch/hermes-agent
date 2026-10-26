---
sidebar_position: 6
title: "Teams 회의"
description: "Microsoft Graph 웹훅을 사용해 Microsoft Teams 회의 요약 파이프라인 설정하기"
---

# Microsoft Teams 회의

Microsoft Graph 회의 이벤트를 Hermes로 수집하고, 먼저 트랜스크립트를 가져온 뒤 필요하면 녹화본과 STT로 대체 처리하며, 구조화된 요약을 다운스트림 싱크로 전달하려면 Teams 회의 파이프라인을 사용하세요.

사전 요구 사항은 기반 봇/자격 증명 설정을 설명하는 [Microsoft Teams](./teams.md)를 참조하세요.

> 안내형 설정 과정을 사용하려면 `hermes gateway setup`을 실행하고 **Teams Meetings**를 선택하세요.

이 페이지에서는 설정과 활성화에 초점을 맞춥니다:
- Graph 자격 증명
- 웹훅 리스너 구성
- Teams 전달 모드
- 파이프라인 구성 형태

2일 차 운영, 가동 점검, 운영자 워크시트는 전용 가이드인 [Teams 회의 파이프라인 운영](/guides/operate-teams-meeting-pipeline)을 사용하세요.

## 이 기능이 하는 일

파이프라인은 다음 작업을 수행합니다:
1. Microsoft Graph 웹훅 이벤트를 수신합니다.
2. 회의를 확인하고 트랜스크립트 아티팩트를 우선 처리합니다.
3. 사용할 수 있는 트랜스크립트가 없으면 녹화본 다운로드와 STT로 대체 처리합니다.
4. 내구성 있는 작업 상태와 싱크 레코드를 로컬에 저장합니다.
5. Notion, Linear, Microsoft Teams에 요약을 기록할 수 있습니다.

운영자 작업은 CLI에 유지됩니다 (`teams-pipeline` 서브커맨드는 `teams_pipeline` 플러그인이 등록합니다. `hermes plugins enable teams_pipeline`으로 활성화하거나 `config.yaml`에서 `plugins.enabled: [teams_pipeline]`을 설정하세요):

```bash
hermes teams-pipeline validate
hermes teams-pipeline list
hermes teams-pipeline maintain-subscriptions
```

## 사전 요구 사항

회의 파이프라인을 활성화하기 전에 다음을 준비하세요:

- 정상적으로 설치된 Hermes
- Teams 아웃바운드 전달을 사용하려는 경우 기존의 [Microsoft Teams 봇 설정](/user-guide/messaging/teams)
- 구독하려는 회의 리소스에 필요한 권한이 부여된 Microsoft Graph 애플리케이션 자격 증명
- Microsoft Graph가 웹훅 전달을 위해 호출할 수 있는 공개 HTTPS URL
- 녹화본과 STT 대체 처리를 사용하려는 경우 설치된 `ffmpeg`

## 1단계: Microsoft Graph 자격 증명 추가

Graph 앱 전용 자격 증명을 `~/.hermes/.env`에 추가하세요:

```bash
MSGRAPH_TENANT_ID=<tenant-id>
MSGRAPH_CLIENT_ID=<client-id>
MSGRAPH_CLIENT_SECRET=<client-secret>
```

이 자격 증명은 다음 작업에 사용됩니다:
- Graph 클라이언트 기반 기능
- 구독 유지 관리 명령
- 회의 확인 및 아티팩트 가져오기
- 전용 Teams 액세스 토큰을 제공하지 않은 경우 Graph 기반 Teams 아웃바운드 전달

## 2단계: Graph 웹훅 리스너 활성화

웹훅 리스너는 `msgraph_webhook`이라는 게이트웨이 플랫폼입니다. 최소한 이를 활성화하고 클라이언트 상태 값을 설정하세요:

```bash
MSGRAPH_WEBHOOK_ENABLED=true
MSGRAPH_WEBHOOK_PORT=8646
MSGRAPH_WEBHOOK_CLIENT_STATE=<random-shared-secret>
MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES=communications/onlineMeetings
```

바인드 호스트는 `config.yaml`에 있는 플랫폼의 `extra.host`에서 읽습니다 (`MSGRAPH_WEBHOOK_HOST` 환경 변수는 없습니다. [웹훅 리스너 참조 문서](msgraph-webhook.md)를 참고하세요).

리스너는 다음 엔드포인트를 제공합니다:
- Graph 알림용 `/msgraph/webhook`
- 간단한 상태 확인용 `/health`

공개 HTTPS 엔드포인트를 이 리스너로 라우팅해야 합니다. 예를 들어 공개 도메인이 `https://ops.example.com`이라면 Graph 알림 URL은 일반적으로 다음과 같습니다:

```text
https://ops.example.com/msgraph/webhook
```

## 3단계: Teams 전달 및 파이프라인 동작 구성

회의 파이프라인은 기존 `teams` 플랫폼 항목에서 런타임 구성을 읽습니다. 파이프라인 전용 설정은 `teams.extra.meeting_pipeline` 아래에 있습니다. Teams 아웃바운드 전달은 일반 Teams 플랫폼 구성 영역을 그대로 사용합니다.

예시 `~/.hermes/config.yaml`:

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

      # outbound summary delivery
      delivery_mode: "graph" # or incoming_webhook
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

리스너를 `0.0.0.0`처럼 루프백이 아닌 호스트에 바인드하면 Microsoft의 웹훅 이그레스 범위로 `allowed_source_cidrs`도 설정해야 합니다. 루프백 바인드(`127.0.0.1` / `::1`)는 개발 터널 및 로컬 리버스 프록시 설정에 사용하도록 설계되었습니다.

## Teams 전달 모드

파이프라인은 기존 Teams 플러그인 안에서 두 가지 Teams 요약 전달 모드를 지원합니다.

### `incoming_webhook`

Graph를 통한 채널 메시지 생성 없이 간단한 웹훅 게시 방식으로 Teams에 게시하려면 사용하세요.

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

Microsoft Graph를 통해 Teams 채팅 또는 채널에 요약을 게시하려면 사용하세요.

지원 대상:
- `chat_id`
- `team_id` + `channel_id`
- 기존 Teams 플랫폼의 `team_id` + `home_channel` 대체 대상

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

## 4단계: 게이트웨이 시작

구성을 업데이트한 뒤 평소처럼 Hermes를 시작하세요:

```bash
hermes gateway run
```

Hermes를 Docker에서 실행한다면 배포 환경에서 이미 사용하는 것과 동일한 방식으로 게이트웨이를 시작하세요.

리스너를 확인하세요:

```bash
curl http://localhost:8646/health
```

## 5단계: Graph 구독 생성

플러그인 CLI를 사용해 구독을 생성하고 검사하세요.

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

Microsoft Graph는 웹훅 구독 기간을 72시간으로 제한하며 자동으로 갱신하지 않습니다. 수동으로 구독을 생성한 뒤 3일이 지나면 알림이 조용히 중단되므로, 가동 전에 `hermes teams-pipeline maintain-subscriptions`를 반드시 예약해야 합니다. 운영자 런북의 [구독 갱신 자동화](/guides/operate-teams-meeting-pipeline#automating-subscription-renewal-required-for-production)를 참고하세요. 세 가지 방법(Hermes cron, systemd timer, 일반 crontab)이 있습니다.

:::

구독 유지 관리와 2일 차 운영 흐름은 가이드 [Teams 회의 파이프라인 운영](/guides/operate-teams-meeting-pipeline)에서 계속 확인하세요.

## 검증

내장 검증 스냅샷을 실행하세요:

```bash
hermes teams-pipeline validate
```

유용한 보조 점검:

```bash
hermes teams-pipeline token-health
hermes teams-pipeline subscriptions
```

## 문제 해결

| 문제 | 확인할 항목 |
|---------|---------------|
| Graph 웹훅 검증 실패 | 공개 URL이 올바르고 접근 가능한지, Graph가 정확한 `/msgraph/webhook` 경로를 호출하는지 확인하세요. |
| `hermes teams-pipeline list`에 작업이 나타나지 않음 | `msgraph_webhook`이 활성화되어 있고 구독이 올바른 알림 URL을 가리키는지 확인하세요. |
| 트랜스크립트 우선 처리가 항상 성공하지 않음 | 트랜스크립트 리소스에 대한 Graph 권한과 해당 회의에 트랜스크립트 아티팩트가 존재하는지 확인하세요. |
| 녹화본 대체 처리 실패 | `ffmpeg`가 설치되어 있고 Graph 앱이 녹화본 아티팩트에 접근할 수 있는지 확인하세요. |
| Teams 요약 전달 실패 | `delivery_mode`, 대상 ID, Teams 인증 구성을 다시 확인하세요. |

## 관련 문서

- [Microsoft Teams 봇 설정](/user-guide/messaging/teams)
- [Teams 회의 파이프라인 운영](/guides/operate-teams-meeting-pipeline)
