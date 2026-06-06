---
title: "Imessage — macOS에서 imsg CLI를 통해 iMessage/SMS 보내기 및 받기"
sidebar_label: "Imessage"
description: "macOS에서 imsg CLI를 통해 iMessage/SMS 보내기 및 받기"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Imessage

macOS에서 imsg CLI를 통해 iMessage/SMS 보내기 및 받기.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/apple/imessage` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | macos |
| 태그 | `iMessage`, `SMS`, `messaging`, `macOS`, `Apple` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# iMessage

`imsg`를 사용하여 macOS Messages.app을 통해 iMessage/SMS를 읽고 보냅니다.

## 전제 조건

- Messages.app이 로그인된 **macOS**
- 설치: `brew install steipete/tap/imsg`
- 터미널에 전체 디스크 접근 권한 부여 (시스템 설정 → 개인정보 보호 및 보안 → 전체 디스크 접근 권한)
- 메시지가 표시될 때 Messages.app에 대한 자동화 권한 부여

## 사용 시기

- 사용자가 iMessage 또는 문자 메시지 전송을 요청할 때
- iMessage 대화 기록을 읽을 때
- 최근 Messages.app 채팅을 확인할 때
- 전화번호나 Apple ID로 전송할 때

## 사용하지 말아야 할 시기

- Telegram/Discord/Slack/WhatsApp 메시지 → 적절한 게이트웨이 채널을 사용하세요
- 그룹 채팅 관리(멤버 추가/제거) → 지원되지 않음
- 대량/단체 메시지 전송 → 항상 먼저 사용자와 확인하세요

## 빠른 참조

### 채팅 목록

```bash
imsg chats --limit 10 --json
```

### 기록 보기

```bash
# 채팅 ID로 보기
imsg history --chat-id 1 --limit 20 --json

# 첨부 파일 정보 포함
imsg history --chat-id 1 --limit 20 --attachments --json
```

### 메시지 보내기

```bash
# 텍스트만 전송
imsg send --to "+14155551212" --text "안녕하세요!"

# 첨부 파일과 함께 전송
imsg send --to "+14155551212" --text "이거 한 번 보세요" --file /path/to/image.jpg

# iMessage 또는 SMS 강제 지정
imsg send --to "+14155551212" --text "안녕" --service imessage
imsg send --to "+14155551212" --text "안녕" --service sms
```

### 새 메시지 감시

```bash
imsg watch --chat-id 1 --attachments
```

## 서비스 옵션

- `--service imessage` — iMessage 강제 지정 (수신자에게 iMessage가 있어야 함)
- `--service sms` — SMS 강제 지정 (초록색 말풍선)
- `--service auto` — Messages.app이 결정하도록 함 (기본값)

## 규칙

1. 전송하기 전에 **항상 수신자와 메시지 내용을 확인**하세요
2. 사용자의 명시적인 승인 없이 **모르는 번호로 절대 보내지 마세요**
3. 첨부하기 전에 **파일 경로가 존재하는지 확인**하세요
4. **스팸을 보내지 마세요** — 스스로 전송 빈도를 제한(rate-limit)하세요

## 워크플로우 예시

사용자: "엄마에게 늦을 거라고 문자 보내줘"

```bash
# 1. 엄마와의 채팅 찾기
imsg chats --limit 20 --json | jq '.[] | select(.displayName | contains("엄마"))'

# 2. 사용자에게 확인: "엄마를 +1555123456 번호로 찾았습니다. iMessage를 통해 '늦을 거예요'라고 보낼까요?"

# 3. 확인 후 전송
imsg send --to "+1555123456" --text "늦을 거예요"
```
