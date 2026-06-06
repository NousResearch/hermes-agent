---
title: "Agentmail — AgentMail을 통해 에이전트에게 전용 이메일 받은편지함 제공"
sidebar_label: "Agentmail"
description: "AgentMail을 통해 에이전트에게 전용 이메일 받은편지함 제공"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Agentmail

AgentMail을 통해 에이전트에게 전용 이메일 받은편지함을 제공합니다. 에이전트 소유의 이메일 주소(예: hermes-agent@agentmail.to)를 사용하여 이메일을 자율적으로 주고받고 관리할 수 있습니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/email/agentmail` 명령어로 설치 |
| 경로 | `optional-skills/email/agentmail` |
| 버전 | `1.0.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `email`, `communication`, `agentmail`, `mcp` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# AgentMail — 에이전트 소유 이메일 받은편지함

## 요구 사항

- **AgentMail API 키** (필수) — https://console.agentmail.to 에서 가입 (무료 등급: 받은편지함 3개, 월 3,000개 이메일; 유료 요금제는 월 $20부터)
- Node.js 18+ (MCP 서버용)

## 사용 시기
다음 작업이 필요할 때 이 스킬을 사용하세요:
- 에이전트에게 전용 이메일 주소를 제공할 때
- 에이전트를 대신하여 자율적으로 이메일을 보낼 때
- 수신 이메일을 받고 읽을 때
- 이메일 스레드와 대화를 관리할 때
- 서비스에 가입하거나 이메일을 통해 인증할 때
- 이메일을 통해 다른 에이전트나 인간과 소통할 때

이 스킬은 사용자의 개인 이메일을 읽기 위한 것이 아닙니다(개인 이메일에는 himalaya나 Gmail을 사용하세요).
AgentMail은 에이전트에게 고유한 신원과 받은편지함을 제공합니다.

## 설정

### 1. API 키 발급받기
- https://console.agentmail.to 로 이동합니다.
- 계정을 만들고 API 키(`am_`으로 시작)를 생성합니다.

### 2. MCP 서버 구성
`~/.hermes/config.yaml`에 다음 내용을 추가하세요 (실제 키를 붙여넣으세요 — MCP 환경 변수는 .env에서 확장되지 않습니다):
```yaml
mcp_servers:
  agentmail:
    command: "npx"
    args: ["-y", "agentmail-mcp"]
    env:
      AGENTMAIL_API_KEY: "am_your_key_here"
```

### 3. Hermes 재시작
```bash
hermes
```
이제 11개의 모든 AgentMail 도구를 자동으로 사용할 수 있습니다.

## 사용 가능한 도구 (MCP를 통해)

| 도구 | 설명 |
|------|-------------|
| `list_inboxes` | 모든 에이전트 받은편지함 나열 |
| `get_inbox` | 특정 받은편지함의 세부 정보 가져오기 |
| `create_inbox` | 새 받은편지함 만들기 (실제 이메일 주소를 받음) |
| `delete_inbox` | 받은편지함 삭제 |
| `list_threads` | 받은편지함의 이메일 스레드 나열 |
| `get_thread` | 특정 이메일 스레드 가져오기 |
| `send_message` | 새 이메일 보내기 |
| `reply_to_message` | 기존 이메일에 답장하기 |
| `forward_message` | 이메일 전달하기 |
| `update_message` | 메시지 라벨/상태 업데이트 |
| `get_attachment` | 이메일 첨부파일 다운로드 |

## 절차

### 받은편지함 생성 및 이메일 보내기
1. 전용 받은편지함 만들기:
   - 사용자 이름과 함께 `create_inbox` 사용 (예: `hermes-agent`)
   - 에이전트는 `hermes-agent@agentmail.to` 주소를 받습니다.
2. 이메일 전송:
   - `inbox_id`, `to`, `subject`, `text`와 함께 `send_message` 사용
3. 답장 확인:
   - 수신된 대화를 보려면 `list_threads` 사용
   - 특정 스레드를 읽으려면 `get_thread` 사용

### 수신 이메일 확인
1. `list_inboxes`를 사용하여 받은편지함 ID를 찾습니다.
2. 받은편지함 ID와 함께 `list_threads`를 사용하여 대화를 봅니다.
3. `get_thread`를 사용하여 스레드와 해당 메시지들을 읽습니다.

### 이메일에 답장하기
1. `get_thread`로 스레드를 가져옵니다.
2. 메시지 ID와 답장 내용(text)과 함께 `reply_to_message`를 사용합니다.

## 워크플로우 예시

**서비스 가입하기:**
```
1. create_inbox (username: "signup-bot")
2. 받은편지함 주소를 사용하여 서비스에 등록
3. 인증 이메일을 확인하기 위해 list_threads 실행
4. 인증 코드를 읽기 위해 get_thread 실행
```

**에이전트가 사람에게 연락하기 (Outreach):**
```
1. create_inbox (username: "hermes-outreach")
2. send_message (to: user@example.com, subject: "Hello", text: "...")
3. 답장을 확인하기 위해 list_threads 실행
```

## 주의 사항 (Pitfalls)
- 무료 등급은 받은편지함 3개, 월 3,000개 이메일로 제한됩니다.
- 무료 등급에서는 `@agentmail.to` 도메인에서 이메일이 발송됩니다 (유료 요금제에서는 사용자 지정 도메인 지원).
- MCP 서버를 실행하려면 Node.js(18+)가 필요합니다 (`npx -y agentmail-mcp`).
- `mcp` 파이썬 패키지를 설치해야 합니다: `pip install mcp`
- 실시간 수신 이메일(웹훅) 기능을 사용하려면 공개 서버가 필요합니다 — 개인용으로는 cronjob을 이용한 `list_threads` 폴링을 대신 사용하세요.

## 검증 (Verification)
설정 후 다음 명령어로 테스트하세요:
```
hermes --toolsets mcp -q "Create an AgentMail inbox called test-agent and tell me its email address"
```
반환된 새로운 받은편지함 주소를 확인할 수 있어야 합니다.

## 참조
- AgentMail 문서: https://docs.agentmail.to/
- AgentMail 콘솔: https://console.agentmail.to
- AgentMail MCP 저장소: https://github.com/agentmail-to/agentmail-mcp
- 가격: https://www.agentmail.to/pricing
