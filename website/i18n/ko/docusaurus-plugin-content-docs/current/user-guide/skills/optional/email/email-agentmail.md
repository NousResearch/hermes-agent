---
title: "Agentmail — 에이전트 전용 받은편지함: 이메일 송수신"
sidebar_label: "Agentmail"
description: "에이전트에게 자체 받은편지함을 제공하여 이메일을 송수신합니다"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Agentmail

에이전트에게 자체 받은편지함을 제공하여 이메일을 송수신합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/email/agentmail`로 설치 |
| 경로 | `optional-skills/email/agentmail` |
| 버전 | `1.0.0` |
| 작성자 | teyrebaz33, Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `email`, `communication`, `agentmail`, `mcp` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보게 되는 내용입니다.
:::

# AgentMail — 에이전트 소유 이메일 받은편지함

## 요구 사항

- **AgentMail API 키**(필수) — https://console.agentmail.to에서 가입합니다(무료 요금제: 받은편지함 3개, 월 3,000개 이메일; 유료 요금제는 월 $20부터).
- Node.js 18 이상(MCP 서버용)

## 사용 시점
다음이 필요할 때 이 스킬을 사용합니다.
- 에이전트에게 전용 이메일 주소 제공
- 에이전트를 대신하여 자율적으로 이메일 전송
- 수신 이메일 수신 및 읽기
- 이메일 스레드 및 대화 관리
- 이메일을 통한 서비스 가입 또는 인증
- 이메일로 다른 에이전트나 사람과 소통

사용자의 개인 이메일을 읽는 용도가 아닙니다(이 경우 himalaya 또는 Gmail 사용).
AgentMail은 에이전트에게 자체 신원과 받은편지함을 제공합니다.

## 설정

### 1. API 키 받기
- https://console.agentmail.to로 이동합니다.
- 계정을 만들고 API 키를 생성합니다(`am_`으로 시작).

### 2. MCP 서버 구성
`~/.hermes/config.yaml`에 추가합니다(실제 키를 붙여 넣으세요 — MCP 환경 변수는 `.env`에서 확장되지 않습니다).
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
이제 11개의 AgentMail 도구를 모두 자동으로 사용할 수 있습니다.

## 사용 가능한 도구(MCP를 통해)

| 도구 | 설명 |
|-------------|-------------|
| `list_inboxes` | 모든 에이전트 받은편지함 목록 조회 |
| `get_inbox` | 특정 받은편지함의 세부 정보 조회 |
| `create_inbox` | 새 받은편지함 생성(실제 이메일 주소를 받음) |
| `delete_inbox` | 받은편지함 삭제 |
| `list_threads` | 받은편지함의 이메일 스레드 목록 조회 |
| `get_thread` | 특정 이메일 스레드 조회 |
| `send_message` | 새 이메일 전송 |
| `reply_to_message` | 기존 이메일에 답장 |
| `forward_message` | 이메일 전달 |
| `update_message` | 메시지 라벨/상태 업데이트 |
| `get_attachment` | 이메일 첨부 파일 다운로드 |

## 절차

### 받은편지함을 만들고 이메일 보내기
1. 전용 받은편지함을 만듭니다.
   - 사용자 이름(예: `hermes-agent`)으로 `create_inbox`를 사용합니다.
   - 에이전트가 `hermes-agent@agentmail.to` 주소를 받습니다.
2. 이메일을 보냅니다.
   - `inbox_id`, `to`, `subject`, `text`와 함께 `send_message`를 사용합니다.
3. 답장을 확인합니다.
   - 수신 대화를 확인하려면 `list_threads`를 사용합니다.
   - 특정 스레드와 메시지를 읽으려면 `get_thread`를 사용합니다.

### 수신 이메일 확인
1. `list_inboxes`를 사용하여 받은편지함 ID를 찾습니다.
2. 받은편지함 ID와 함께 `list_threads`를 사용하여 대화를 확인합니다.
3. 특정 스레드를 읽으려면 `get_thread`를 사용합니다.

### 이메일에 답장하기
1. `get_thread`로 스레드를 가져옵니다.
2. 메시지 ID와 답장 텍스트를 사용하여 `reply_to_message`를 호출합니다.

## 예시 워크플로

**서비스 가입:**
```
1. create_inbox (username: "signup-bot")
2. Use the inbox address to register on the service
3. list_threads to check for verification email
4. get_thread to read the verification code
```

**에이전트에서 사람에게 연락:**
```
1. create_inbox (username: "hermes-outreach")
2. send_message (to: user@example.com, subject: "Hello", text: "...")
3. list_threads to check for replies
```

## 주의 사항
- 무료 요금제는 받은편지함 3개 및 월 3,000개 이메일로 제한됩니다.
- 무료 요금제의 이메일은 `@agentmail.to` 도메인에서 발송됩니다(사용자 지정 도메인은 유료 요금제에서 지원).
- MCP 서버에는 Node.js(18 이상)가 필요합니다(`npx -y agentmail-mcp`).
- `mcp` Python 패키지를 설치해야 합니다: `pip install mcp`
- 실시간 수신 이메일(웹훅)에는 공개 서버가 필요합니다 — 개인 사용에는 대신 cronjob으로 `list_threads`를 폴링하세요.

## 검증
설정 후 다음 명령으로 테스트합니다.
```
hermes --toolsets mcp -q "Create an AgentMail inbox called test-agent and tell me its email address"
```
새 받은편지함 주소가 반환되어야 합니다.

## 참고 자료
- AgentMail 문서: https://docs.agentmail.to/
- AgentMail 콘솔: https://console.agentmail.to
- AgentMail MCP 저장소: https://github.com/agentmail-to/agentmail-mcp
- 가격: https://www.agentmail.to/pricing
