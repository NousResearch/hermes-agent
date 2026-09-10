---
title: "Imessage — macOS에서 imsg CLI로 iMessage/SMS 보내고 받기"
sidebar_label: "Imessage"
description: "macOS에서 imsg CLI로 iMessage/SMS 보내고 받기"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Imessage

macOS에서 imsg CLI를 통해 iMessage/SMS를 보내고 받습니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들 포함 (기본 설치) |
| 경로 | `skills/apple/imessage` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | macos |
| 태그 | `iMessage`, `SMS`, `messaging`, `macOS`, `Apple` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# iMessage

`imsg`를 사용해 macOS Messages.app에서 iMessage/SMS를 읽고 보냅니다.

## 사전 요구 사항

- **Messages.app에 로그인된 macOS**
- 설치: `brew install steipete/tap/imsg`
- 터미널에 전체 디스크 접근 권한 부여 (시스템 설정 → 개인정보 보호 및 보안 → 전체 디스크 접근 권한)
- 메시지가 표시되면 Messages.app에 자동화 권한 부여

## 사용 시점

- 사용자가 iMessage 또는 문자 메시지 전송을 요청할 때
- iMessage 대화 기록 읽기
- 최근 Messages.app 대화 확인
- 전화번호 또는 Apple ID로 보내기

## 사용하지 않을 때

- Telegram/Discord/Slack/WhatsApp 메시지 → 해당 게이트웨이 채널 사용
- 그룹 채팅 관리 (멤버 추가/삭제) → 지원되지 않음
- 대량/일괄 메시지 전송 → 항상 먼저 사용자에게 확인

## 빠른 참조

### 채팅 목록 보기

```bash
imsg chats --limit 10 --json
```

### 기록 보기

```bash
# By chat ID
imsg history --chat-id 1 --limit 20 --json

# With attachments info
imsg history --chat-id 1 --limit 20 --attachments --json
```

### 메시지 보내기

```bash
# Text only
imsg send --to "+14155551212" --text "Hello!"

# With attachment
imsg send --to "+14155551212" --text "Check this out" --file /path/to/image.jpg

# Force iMessage or SMS
imsg send --to "+14155551212" --text "Hi" --service imessage
imsg send --to "+14155551212" --text "Hi" --service sms
```

### 새 메시지 감시

```bash
imsg watch --chat-id 1 --attachments
```

## 서비스 옵션

- `--service imessage` — iMessage 강제 (수신자에게 iMessage가 필요함)
- `--service sms` — SMS 강제 (초록색 말풍선)
- `--service auto` — Messages.app이 결정하도록 함 (기본값)

## 규칙

1. **보내기 전에 항상 수신자와 메시지 내용을 확인**
2. **명시적인 사용자 승인 없이 알 수 없는 번호로 절대 보내지 않기**
3. **첨부하기 전에 파일 경로가 존재하는지 확인**
4. **스팸 전송 금지** — 스스로 전송 속도 제한

## 예시 워크플로

사용자: "엄마한테 늦는다고 문자해 줘"

```bash
# 1. Find mom's chat
imsg chats --limit 20 --json | jq '.[] | select(.displayName | contains("Mom"))'

# 2. Confirm with user: "Found Mom at +1555123456. Send 'I'll be late' via iMessage?"

# 3. Send after confirmation
imsg send --to "+1555123456" --text "I'll be late"
```
