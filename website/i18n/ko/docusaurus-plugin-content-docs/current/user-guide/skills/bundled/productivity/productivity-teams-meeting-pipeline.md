---
title: "Teams Meeting Pipeline"
sidebar_label: "Teams Meeting Pipeline"
description: "Hermes CLI를 통해 Teams 미팅 요약 파이프라인 운영 — 미팅 요약, 파이프라인 상태 확인, 작업 재실행, Microsoft Graph 구독 관리"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Teams Meeting Pipeline

Hermes CLI를 통해 Teams 미팅 요약 파이프라인을 운영합니다 — 미팅 요약, 파이프라인 상태 확인, 작업(jobs) 재실행, Microsoft Graph 구독 관리.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/productivity/teams-meeting-pipeline` |
| Version | `1.1.0` |
| Author | Hermes Agent + Teknium |
| License | MIT |
| Tags | `Teams`, `Microsoft Graph`, `Meetings`, `Productivity`, `Operations` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Teams Meeting Pipeline (Teams 미팅 파이프라인)

사용자가 Microsoft Teams 미팅 요약, 전사(transcripts), 녹화본, 액션 아이템, Graph 구독 또는 Teams 미팅 파이프라인에 대한 기타 운영 관련 질문을 할 때마다 이 스킬을 사용하세요. 어떤 언어로든 작동합니다 — 아래의 트리거는 전체 목록이 아닌 예시일 뿐입니다.

운영자와 관련된 모든 작업은 터미널 도구를 통해 실행되는 `hermes teams-pipeline` 하위 명령어입니다. 이 파이프라인을 위한 새로운 모델 도구는 없습니다 — CLI가 겉으로 드러나는 인터페이스(surface)입니다.

## 이 스킬의 사용 시기

사용자가 다음을 요청할 때:
- Teams 미팅 요약 / 액션 아이템 추출 / 미팅 노트 가져오기
- 파이프라인 상태 확인, 저장된 미팅 작업(job) 검사, 또는 최근 미팅 보기
- 실패했거나 새로운 요약이 필요한 저장된 작업을 재실행 / 다시 실행(replay)
- 환경 변수나 구성(config)을 변경한 후 Microsoft Graph 설정 유효성 검사
- "미팅 요약이 도착하지 않음" 또는 "새로운 미팅이 수집되지 않음" 문제 해결
- Graph 웹훅 구독 관리 (생성, 갱신, 삭제, 검사)
- 자동 구독 갱신 설정 (아래의 주의사항 참고)

다국어 트리거 예시 (일부만 포함):
- 영어: "summarize the Teams meeting", "pipeline status", "replay job X"
- 튀르키예어: "Teams meeting özetle", "action item çıkar", "toplantı notu", "pipeline durumu", "replay job"

## 사전 요구 사항

파이프라인을 사용하기 전에 `~/.hermes/.env`에 다음이 설정되어 있는지 확인하세요:

```bash
MSGRAPH_TENANT_ID=...
MSGRAPH_CLIENT_ID=...
MSGRAPH_CLIENT_SECRET=...
```

누락된 항목이 있는 경우 `/docs/guides/microsoft-graph-app-registration`의 Azure 앱 등록 가이드로 사용자를 안내하세요 — 파이프라인이 작동하려면 관리자 동의를 받은 Graph 애플리케이션 권한이 있는 Azure AD 앱 등록이 필요합니다.

## 명령어 참조 (Command reference)

### 상태 및 검사 (여기서 시작)

```bash
hermes teams-pipeline validate              # 구성 스냅샷 — 변경 후 가장 먼저 실행
hermes teams-pipeline token-health          # Graph 토큰 상태
hermes teams-pipeline token-health --force-refresh   # 토큰 강제 새로 고침
hermes teams-pipeline list                  # 최근 미팅 작업들
hermes teams-pipeline list --status failed  # 실패한 작업만 보기
hermes teams-pipeline show <job-id>         # 한 작업에 대한 전체 세부 정보
hermes teams-pipeline subscriptions         # 현재 Graph 웹훅 구독
```

### 재실행 / 디버깅 (Re-running / debugging)

```bash
hermes teams-pipeline run <job-id>          # 저장된 작업 다시 실행(replay) (다시 요약, 다시 전달)
hermes teams-pipeline fetch --meeting-id <id>   # 예행 연습(dry-run): 저장하지 않고 미팅 + 전사본(transcript) 해석
hermes teams-pipeline fetch --join-web-url "<url>"   # 참석 URL로 예행 연습
```

### 구독 관리 (Subscription management)

```bash
hermes teams-pipeline subscribe \
  --resource communications/onlineMeetings/getAllTranscripts \
  --notification-url https://<your-public-host>/msgraph/webhook \
  --client-state "$MSGRAPH_WEBHOOK_CLIENT_STATE"

hermes teams-pipeline renew-subscription <sub-id> --expiration <iso-8601>
hermes teams-pipeline delete-subscription <sub-id>
hermes teams-pipeline maintain-subscriptions            # 만료가 임박한 것들 갱신
hermes teams-pipeline maintain-subscriptions --dry-run  # 무엇이 갱신될지 보여주기
```

## 일반적인 요청에 대한 의사결정 트리

- 사용자가 "오늘 회의 요약을 왜 못 받았나요?"라고 묻는 경우 → 먼저 `list --status failed`로 시작한 다음 관련 행에 대해 `show <job-id>`를 실행하세요. 작업이 아예 존재하지 않는다면 `subscriptions`를 확인하세요 — 웹훅이 만료되었을 수 있습니다 (아래 주의사항 참조).
- 사용자가 "설정이 잘 작동하나요?"라고 묻는 경우 → `validate`, 그 다음 `token-health`, 그 다음 `subscriptions`를 실행하세요. 세 가지가 모두 통과하면 테스트 회의를 요청하고 `list`에서 새 행을 확인하세요.
- 사용자가 "회의 X에 대한 요약 다시 실행해 줘"라고 요청하는 경우 → `list`를 통해 작업 ID를 찾고, `run <job-id>`로 재실행(replay)하세요. 만약 또 실패한다면, `show <job-id>`로 오류를 검사하고 `fetch --meeting-id`로 아티팩트 해상도를 예행 연습(dry-run) 해보세요.
- 사용자가 "파이프라인에 회의 X를 추가해 줘"라고 요청하는 경우 → 보통 이렇게 하지 않습니다 — 파이프라인은 구독 중심이며 개별 미팅 단위가 아닙니다. 만약 특정 과거 회의를 요약하기 원한다면 `fetch`를 사용하여 전사본을 가져오고 작업이 생성된 후 `run`을 실행하세요.

## 치명적인 주의사항: Graph 구독은 72시간 내에 만료됩니다

Microsoft Graph는 웹훅 구독을 72시간으로 제한하며 **자동으로 갱신하지 않습니다**. `maintain-subscriptions`가 예약되어 있지 않으면 수동으로 구독을 생성한 지 3일 후에 미팅 알림 도착이 소리 없이 멈춥니다.

사용자가 "어제는 파이프라인이 작동했는데 오늘은 아무것도 오지 않는다"라고 보고할 때:
1. `hermes teams-pipeline subscriptions`를 실행하세요 — 비어 있거나 모든 항목에 과거 시점의 `expirationDateTime`이 표시된다면 그것이 원인입니다.
2. 위에서 보여준 대로 `subscribe`로 다시 생성하세요.
3. `hermes cron add`, systemd 타이머, 또는 기본 crontab을 통해 **자동 갱신을 즉시 설정**하세요. `/docs/guides/operate-teams-meeting-pipeline#automating-subscription-renewal-required-for-production`의 운영자 런북에 세 가지 옵션이 모두 있습니다. 12시간 간격이 안전합니다 (72시간 제한에 대해 6배의 여유 공간).

## 기타 주의사항 (Other pitfalls)

- **전사본(Transcript)을 아직 사용할 수 없음.** Teams는 미팅이 종료된 후 전사 아티팩트를 생성하는 데 시간이 좀 걸립니다. 방금 끝난 미팅에 대해 `fetch --meeting-id`를 실행하면 비어 있을 수 있습니다. 2-5분 정도 기다렸다가 다시 시도하거나, Graph 웹훅이 수집을 자연스럽게 유도하도록 둡니다.
- **전송 모드(Delivery mode) 불일치.** 요약이 생성되었지만(`list`에 성공으로 표시됨) Teams에 아무것도 도착하지 않은 경우, `platforms.teams.extra.delivery_mode`와 일치하는 타겟 구성(`incoming_webhook_url` 또는 `chat_id` 또는 `team_id`+`channel_id`)을 확인하세요. writer는 config.yaml 또는 `TEAMS_*` 환경 변수에서 이를 읽습니다.
- **Graph 앱 권한.** 토큰 획득은 정상적으로 되지만(`token-health` 통과) 권한은 추가되었으나 관리자 동의가 다시 승인되지 않은 경우 Graph API 호출이 401/403을 반환합니다. 사용자가 Azure Portal의 앱 등록을 다시 방문하여 "Grant admin consent"를 다시 클릭하도록 안내하세요.

## 관련 문서 (Related docs)

이 스킬에서 다루는 내용보다 더 깊은 정보가 필요한 경우 사용자를 다음으로 안내하세요:
- Azure 앱 등록 안내: `/docs/guides/microsoft-graph-app-registration`
- 전체 파이프라인 설정: `/docs/user-guide/messaging/teams-meetings`
- 운영자 런북(갱신 자동화, 문제 해결, 라이브 전 체크리스트): `/docs/guides/operate-teams-meeting-pipeline`
- 웹훅 리스너 설정: `/docs/user-guide/messaging/msgraph-webhook`
