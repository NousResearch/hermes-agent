---
title: "Teams 회의 파이프라인 — Teams 회의 요약, 작업 재실행, Graph 구독"
sidebar_label: "Teams 회의 파이프라인"
description: "Teams 회의 요약, 작업 재실행, Graph 구독"
---

{/* 이 페이지는 스킬의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Teams 회의 파이프라인

Teams 회의 요약, 작업 재실행, Graph 구독.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨 (기본 설치) |
| 경로 | `skills/productivity/teams-meeting-pipeline` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent + Teknium |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Teams`, `Microsoft Graph`, `Meetings`, `Productivity`, `Operations` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되어 있을 때 에이전트가 보는 지침이 바로 이것입니다.
:::

# Teams 회의 파이프라인

사용자가 Microsoft Teams 회의 요약, 대화 기록, 녹화, 작업 항목, Graph 구독 또는 Teams 회의 파이프라인에 관한 운영상의 질문을 할 때마다 이 스킬을 사용하세요. 어떤 언어로든 작동합니다. 아래 트리거는 예시일 뿐, 전체 목록이 아닙니다.

운영자가 사용하는 모든 기능은 터미널 도구로 실행하는 `hermes teams-pipeline` 하위 명령입니다. 이 파이프라인에는 새로운 모델 도구가 없습니다. 인터페이스는 CLI입니다.

## 이 스킬을 사용하는 경우

사용자가 다음을 요청합니다:
- Teams 회의를 요약하거나 작업 항목을 추출하거나 회의 노트를 가져오기
- 파이프라인 상태를 확인하거나 저장된 회의 작업을 검사하거나 최근 회의를 보기
- 실패했거나 새 요약이 필요한 저장된 작업을 재생하거나 다시 실행하기
- env 또는 config를 변경한 후 Microsoft Graph 설정을 검증하기
- "회의 요약이 도착하지 않음" 또는 "새 회의가 수집되지 않음" 문제 해결하기
- Graph 웹훅 구독을 관리하기 (생성, 갱신, 삭제, 검사)
- 자동 구독 갱신 설정하기 (아래 주의사항 참고)

다국어 트리거 예시 (전체 목록이 아님):
- 영어: "summarize the Teams meeting", "pipeline status", "replay job X"
- 터키어: "Teams meeting özetle", "action item çıkar", "toplantı notu", "pipeline durumu", "replay job"

## 사전 요구 사항

파이프라인을 사용하기 전에 다음 값이 `${HERMES_HOME:-~/.hermes}/.env`에 설정되어 있는지 확인하세요:

```bash
MSGRAPH_TENANT_ID=...
MSGRAPH_CLIENT_ID=...
MSGRAPH_CLIENT_SECRET=...
```

누락된 값이 있으면 `/docs/guides/microsoft-graph-app-registration`의 Azure 앱 등록 가이드를 안내하세요. 파이프라인이 작동하려면 관리자 동의를 받은 Graph 애플리케이션 권한이 있는 Azure AD 앱 등록이 필요합니다.

## 명령어 참고

### 상태 및 검사 (여기서 시작)

```bash
hermes teams-pipeline validate              # config snapshot — run first after any change
hermes teams-pipeline token-health          # Graph token status
hermes teams-pipeline token-health --force-refresh   # force a fresh token acquisition
hermes teams-pipeline list                  # recent meeting jobs
hermes teams-pipeline list --status failed  # only failed jobs
hermes teams-pipeline show <job-id>         # full detail of one job
hermes teams-pipeline subscriptions         # current Graph webhook subscriptions
```

### 다시 실행 / 디버깅

```bash
hermes teams-pipeline run <job-id>          # replay a stored job (re-summarize, re-deliver)
hermes teams-pipeline fetch --meeting-id <id>   # dry-run: resolve meeting + transcript without persisting
hermes teams-pipeline fetch --join-web-url "<url>"   # dry-run by join URL
```

### 구독 관리

```bash
hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllTranscripts \
  --notification-url https://<your-public-host>/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"

hermes teams-pipeline renew-subscription <sub-id> --expiration <iso-8601>
hermes teams-pipeline delete-subscription <sub-id>
hermes teams-pipeline maintain-subscriptions            # renew near-expiry ones
hermes teams-pipeline maintain-subscriptions --dry-run  # show what would be renewed
```

## 일반적인 요청에 대한 의사결정 트리

- 사용자가 "오늘 회의 요약을 왜 받지 못했나요?"라고 묻는 경우 → 먼저 `list --status failed`를 실행한 다음, 관련 행에 대해 `show <job-id>`를 실행하세요. 작업이 아예 없다면 `subscriptions`를 확인하세요. 웹훅이 만료되었을 수 있습니다 (아래 주의사항 참고).
- 사용자가 "설정이 작동하나요?"라고 묻는 경우 → `validate`, `token-health`, `subscriptions` 순서로 실행하세요. 세 가지가 모두 통과하면 테스트 회의를 요청하고 `list`에서 새 행을 확인하세요.
- 사용자가 "회의 X의 요약을 다시 실행해 주세요"라고 묻는 경우 → `list`로 작업 ID를 찾고 `run <job-id>`로 재생하세요. 다시 실패하면 `show <job-id>`로 오류를 검사하고 `fetch --meeting-id`로 아티팩트 확인을 시험 실행하세요.
- 사용자가 "회의 X를 파이프라인에 추가해 주세요"라고 묻는 경우 → 일반적으로 그렇게 하지 않습니다. 파이프라인은 회의별이 아니라 구독 기반입니다. 특정 과거 회의를 요약하려는 경우 `fetch`로 대화 기록을 가져온 다음 작업이 생성된 후 `run`을 사용하세요.

## 중요한 주의사항: Graph 구독은 72시간 후 만료됩니다

Microsoft Graph는 웹훅 구독을 최대 72시간으로 제한하며 **자동으로 갱신하지 않습니다**. `maintain-subscriptions`가 예약되어 있지 않으면, 수동으로 구독을 생성한 뒤 3일이 지나면 회의 알림이 아무런 표시 없이 도착하지 않게 됩니다.

사용자가 "어제까지는 파이프라인이 작동했는데 오늘은 아무것도 도착하지 않아요"라고 말하는 경우:
1. `hermes teams-pipeline subscriptions`를 실행하세요. 비어 있거나 모든 항목의 `expirationDateTime`이 과거라면 이것이 원인입니다.
2. 위에 표시된 `subscribe`로 다시 생성하세요.
3. **즉시 자동 갱신을 설정하세요**. `hermes cron add`, systemd 타이머 또는 일반 crontab을 사용할 수 있습니다. 운영자 런북 `/docs/guides/operate-teams-meeting-pipeline#automating-subscription-renewal-required-for-production`에 세 가지 방법이 모두 나와 있습니다. 12시간 간격이면 72시간 제한에 대해 6배의 여유가 있어 안전합니다.

## 기타 주의사항

- **대화 기록을 아직 사용할 수 없습니다.** Teams는 회의가 끝난 후 대화 기록 아티팩트를 생성하는 데 시간이 걸립니다. 방금 끝난 회의에서 `fetch --meeting-id`를 실행하면 빈 결과가 반환될 수 있습니다. 2~5분 기다렸다가 다시 시도하거나 Graph 웹훅이 자연스럽게 수집을 처리하도록 두세요.
- **전달 모드가 일치하지 않습니다.** 요약이 생성되었지만 (`list`에 성공으로 표시됨) Teams에 아무것도 도착하지 않는다면 `platforms.teams.extra.delivery_mode`와 일치하는 대상 설정 (`incoming_webhook_url` 또는 `chat_id` 또는 `team_id`+`channel_id`)을 확인하세요. 작성기는 config.yaml 또는 `TEAMS_*` env 값에서 이를 읽습니다.
- **Graph 앱 권한.** 토큰은 정상적으로 발급되지만 (`token-health` 통과) 권한을 추가한 후 Graph API 호출이 401/403을 반환한다면 관리자 동의를 다시 부여하지 않았기 때문일 수 있습니다. 사용자가 Azure 포털에서 앱 등록을 다시 열고 "Grant admin consent"를 다시 클릭하도록 안내하세요.

## 관련 문서

이 스킬의 범위를 넘어 더 자세한 내용이 필요할 때는 다음을 안내하세요:
- Azure 앱 등록 안내: `/docs/guides/microsoft-graph-app-registration`
- 전체 파이프라인 설정: `/docs/user-guide/messaging/teams-meetings`
- 운영자 런북 (갱신 자동화, 문제 해결, 출시 준비 체크리스트): `/docs/guides/operate-teams-meeting-pipeline`
- 웹훅 리스너 설정: `/docs/user-guide/messaging/msgraph-webhook`
