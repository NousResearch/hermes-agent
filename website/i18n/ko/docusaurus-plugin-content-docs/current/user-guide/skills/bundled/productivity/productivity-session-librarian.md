---
title: "Session Librarian — 프롬프트로 세션 정리: 찾기, 이름 변경, 보관, 정리"
sidebar_label: "Session Librarian"
description: "프롬프트로 세션 정리: 찾기, 이름 변경, 보관, 정리"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Session Librarian

프롬프트로 세션 정리: 찾기, 이름 변경, 보관, 정리.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 번들 포함(기본 설치) |
| 경로 | `skills/productivity/session-librarian` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent + Teknium |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Sessions`, `Organization`, `Cleanup`, `Library`, `Productivity` |
| 관련 스킬 | [`weekly-review-planning`](/docs/user-guide/skills/bundled/productivity/productivity-weekly-review-planning) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Session Librarian

사용자의 세션 라이브러리를 대화 방식으로 관리합니다. 주제와 관련된 과거 세션을 찾고, 결정한 내용을 요약하고, 의미 있는 이름으로 변경하고, 작업을 병렬 세션으로 나누며, 오래된 세션을 보관하거나 삭제할 후보로 제안할 수 있습니다. 예를 들어 *"Q3 가격 책정에 관한 세션을 찾고, 유용한 것은 남기고, 중복된 것은 정리해 줘"*처럼 자연어로 요청하면 됩니다.

Perplexity Computer의 프롬프트 기반 세션 관리에서 영감을 받았습니다(2026년 8월). 에이전트가 사용자의 세션 라이브러리를 시작하고, 정리하고, 관리하며, 무엇이든 변경하기 전에 항상 계획을 보여 줍니다.

## 사용 시점

- "X에 관한 세션이 뭐가 있지?" / "X에 대해 무엇을 결정했지?"
- "이 세션들의 이름을 의미 있게 바꿔 줘."
- "내 세션 라이브러리를 정리해 줘" / "오래된 세션을 보관해 줘."
- "이 세션을 Y에 초점을 맞춘 후속 세션으로 분기해 줘."
- "이 작업을 티켓별 세션 하나씩으로 나눠 줘"(아래의 병렬 작업 흐름 참조).

## 두 가지 표면

| 작업 | 표면 |
|---|---|
| 주제로 세션 찾기, 내용 읽기, 결정 사항 요약 | `session_search` 도구(메시지 저장소의 FTS5) |
| 메타데이터(나이, 소스, 비용, 토큰, 작업공간)로 나열/필터링 | 터미널을 통한 `hermes sessions list` / `stats` |
| 이름 변경 | `hermes sessions rename <session_id> <title...>` |
| 일괄 소프트 숨김(되돌릴 수 있음) | `hermes sessions archive <filters>` |
| 삭제(파괴적 작업) | `hermes sessions delete` / `hermes sessions prune <filters>` |
| 중요한 항목을 삭제하기 전에 내보내기 | `hermes sessions export --session-id <id> --format md` |
| 새 위치에서 작업 계속하기 | `/branch`(현재 세션 분기) 또는 새 세션을 시작하고 요약을 인용 |

## 절차

① **탐색.** 주제 키워드와 함께 `session_search(query=..., limit=5-10)`을 사용하고, 표현을 바꿔 가며 검색합니다(기능 이름, 증상, 프로젝트 이름). 메타데이터를 훑어볼 때(예: "Telegram에서 60일보다 오래된 세션")는 대신 `hermes sessions list --source telegram --limit 50`을 사용합니다.

② **세션별 요약.** 탐색 결과의 `bookend_start`(목표), 일치 구간, `bookend_end`(해결 내용)만으로 대개 충분합니다. 사용자가 결정 사항을 자세히 요청한 경우에만 전체 세션(`session_search(session_id=...)`)을 출력합니다. 각각을 다음과 같이 보고합니다: 링크(`@session:` 형식) — 한 줄 목표 — 한 줄 결과.

③ **실행 전 계획 수립(변경 작업에는 필수).** 먼저 계획 표를 제시합니다. 어떤 세션의 이름을 무엇으로 바꿀지, 어떤 세션을 보관할지, 어떤 세션을 왜 삭제 후보로 제안하는지(어떤 보존 세션의 중복인지, 오래됐는지, 비어 있는지)를 표시합니다. 사용자의 진행 승인을 기다립니다. 예외: 사용자가 명시적으로 지시한 단일 이름 변경은 바로 실행할 수 있습니다.

④ **가장 안전한 기본 동작으로 실행.**
- `delete`/`prune`보다 `archive`(되돌릴 수 있는 소프트 숨김)를 우선합니다.
- 파괴적 명령은 항상 먼저 `--dry-run`으로 실행하고 결과를 보여 준 다음, 확인을 받은 뒤 `--yes`로 다시 실행합니다.
- 의미 있는 내용이 있는 항목을 삭제하기 전에 백업으로 `hermes sessions export --format md`를 제안합니다.

⑤ **보고.** 이름 변경 결과, 보관한 세션(개수와 되돌리는 방법: 보관된 세션은 DB에 남아 있으며 `--include-archived`로 나열할 수 있음), 내보낸 항목, 건너뛴 항목과 그 이유를 보고합니다.

## 병렬 작업 흐름

"티켓별 세션 하나씩으로 나눠 각각 조사하고 결과를 보고해 줘"와 같은 요청에서는 다른 실행 중인 세션을 직접 조작하지 마세요. 작업 흐름마다 하나씩 `delegate_task`를 사용하세요. 각 하위 에이전트는 자동으로 별도 세션에서 실행되며, 이후 결과를 종합합니다. 각 위임 작업의 대화 기록도 나중에 `session_search`로 검색할 수 있다고 알려 주세요.

## 주의 사항

- 이 대화에서 명시적으로 확인받은 뒤가 아니면 절대 삭제하지 마세요. "정리해 줘"라는 포괄적인 요청은 삭제가 아니라 제안할 권한을 의미합니다.
- `session_search`는 내용만 찾고 메타데이터는 찾지 않습니다. 나이/비용/소스 필터는 CLI에 있습니다. 요청에 두 조건이 섞여 있으면 둘을 함께 사용하세요(예: "가격 책정에 관한 오래된 세션").
- 제목은 `/resume <title>`에서 식별자로 사용됩니다. 제목을 변경할 때는 짧고 고유하며 접두사로 구분하기 쉽게 유지하고, 기존 제목과 충돌하면 사용자에게 경고하세요.
- 보관은 삭제가 아닙니다. 기본 목록에서만 세션을 숨깁니다. 무엇을 했는지 보관인지 삭제인지 명확히 말하세요.
- 교차 프로필 세션 링크(`@session:<profile>/<id>`)는 다른 프로필에서 읽기 전용입니다. 관리 명령은 현재 프로필의 DB에 적용됩니다.

## 검증

정리 작업 후 탐색 쿼리와 `hermes sessions list`를 다시 실행해 라이브러리가 계획대로 반영되었는지 확인합니다(보존된 세션은 새 이름으로 존재하고, 보관된 세션은 기본 목록에서 사라졌는지 확인).
