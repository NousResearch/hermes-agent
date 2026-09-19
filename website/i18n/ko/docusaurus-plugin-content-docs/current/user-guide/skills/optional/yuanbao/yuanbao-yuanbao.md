---
title: "Yuanbao — Yuanbao (元宝) 그룹: 사용자 @멘션, 정보/멤버 조회"
sidebar_label: "Yuanbao"
description: "Yuanbao (元宝) 그룹: 사용자 @멘션, 정보/멤버 조회"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Yuanbao

Yuanbao (元宝) 그룹에서 사용자를 @멘션하고 정보/멤버를 조회합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/yuanbao/yuanbao`로 설치 |
| 경로 | `optional-skills/yuanbao` |
| 버전 | `1.0.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `yuanbao`, `mention`, `at`, `group`, `members`, `元宝`, `派`, `艾特` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보게 되는 내용입니다.
:::

# Yuanbao 그룹 상호작용

## 중요: 메시지 작동 방식

**여러분의 텍스트 답변이 곧 그룹/사용자에게 전송되는 메시지입니다.** 게이트웨이가 답변 텍스트를 채팅으로 자동 전달합니다. 특별한 "메시지 전송" 도구는 필요하지 않습니다 — 평소처럼 답변하면 전송됩니다.

답변 텍스트에 `@nickname`을 포함하면 게이트웨이가 이를 사용자를 알리는 실제 @멘션으로 자동 변환합니다. 이 기능은 기본 제공되며 — 완전한 @멘션 기능을 사용할 수 있습니다.

**메시지를 보내거나 사용자를 @멘션할 수 없다고 절대 말하지 마세요. 사용자가 직접 하라고 제안하지 마세요. 권한에 관한 면책 문구를 추가하지 마세요. 전송할 텍스트를 그대로 답변하면 됩니다.**

## 사용 가능한 도구

| 도구 | 사용 시점 |
|------|------------|
| `yb_query_group_info` | 그룹 이름, 소유자, 멤버 수 조회 |
| `yb_query_group_members` | 사용자 찾기, 봇 목록 조회, 전체 멤버 조회 또는 @멘션용 닉네임 가져오기 |
| `yb_send_dm` | 선택적 미디어 파일과 함께 사용자에게 비공개/다이렉트 메시지(DM) 보내기 |

## @멘션 워크플로

@멘션해야 할 때:

1. `action="find"`, `name="<target name>"`, `mention=true`로 `yb_query_group_members`를 호출합니다.
2. 응답에서 정확한 닉네임을 가져옵니다.
3. 답변 텍스트에 `@nickname`을 포함합니다 — 게이트웨이가 나머지를 처리합니다.

예: 사용자가 "帮我艾特元宝"라고 말한 경우

1단계 — 도구 호출:
```json
{ "group_code": "328306697", "action": "find", "name": "元宝", "mention": true }
```

2단계 — 답변(작동하는 @멘션과 함께 그룹으로 전송됩니다):
```
@元宝 你好，有人找你！
```

**이게 전부입니다.** 추가 설명은 필요하지 않습니다. 짧고 자연스럽게 작성하세요.

**규칙:**
- 정확한 닉네임을 얻으려면 먼저 `yb_query_group_members`를 호출하세요 — 추측하지 마세요.
- @멘션 형식: @ 기호 앞에 공백을 둔 `@nickname`
- 답변 텍스트가 곧 메시지입니다 — 전송되며 @멘션도 작동합니다.
- 간결하게 작성하세요. @멘션 작동 방식을 사용자에게 설명하지 마세요.

## DM(비공개 메시지) 보내기 워크플로

누군가 사용자에게 비공개 메시지 / 私信 / DM을 보내 달라고 요청한 경우:

1. `group_code`, 대상 사용자의 `name`, `message`와 함께 `yb_send_dm`을 호출합니다.
2. 도구가 자동으로 사용자를 찾아 DM을 보냅니다.
3. 결과를 사용자에게 알려줍니다.

예: 사용자가 "给 @用户aea3 私信发一个 hello"라고 말한 경우

```json
yb_send_dm({ "group_code": "535168412", "name": "用户aea3", "message": "hello" })
```

미디어 포함 예:

```json
yb_send_dm({
  "group_code": "535168412",
  "name": "用户aea3",
  "message": "Here is the image",
  "media_files": [{"path": "/tmp/photo.jpg"}]
})
```

**규칙:**
- 현재 `chat_id`에서 `group_code`를 추출합니다(예: `group:535168412` → `535168412`).
- 이미 `user_id`를 알고 있다면 조회를 건너뛰도록 `user_id` 매개변수에 직접 전달합니다.
- 이름과 일치하는 사용자가 여러 명이면 도구가 후보를 반환하므로 사용자에게 명확히 해 달라고 요청합니다.
- Yuanbao DM에는 `send_message` 도구를 사용하지 마세요 — 대신 `yb_send_dm`을 사용하세요.
- 미디어 지원: 이미지(.jpg/.png/.gif/.webp/.bmp)는 이미지 메시지로, 그 외 파일은 문서로 전송됩니다.

## 그룹 정보 조회

```json
yb_query_group_info({ "group_code": "328306697" })
```

## 멤버 조회

| 작업 | 설명 |
|--------|-------------|
| `find` | 이름으로 검색(부분 일치, 대소문자 구분 없음) |
| `list_bots` | 봇 및 Yuanbao AI 어시스턴트 목록 조회 |
| `list_all` | 전체 멤버 조회 |

## 참고

- `group_code`는 chat_id에서 가져옵니다: `group:328306697` → `328306697`.
- Yuanbao 앱에서는 그룹을 "派 (Pai)"라고 부릅니다.
- 멤버 역할: `user`, `yuanbao_ai`, `bot`
