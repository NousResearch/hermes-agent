---
title: "Himalaya — Himalaya CLI: 터미널에서 IMAP/SMTP 이메일 사용"
sidebar_label: "Himalaya"
description: "Himalaya CLI: 터미널에서 사용하는 IMAP/SMTP 이메일"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Himalaya

Himalaya CLI: 터미널에서 사용하는 IMAP/SMTP 이메일.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함 (기본 설치됨) |
| 경로 | `skills/email/himalaya` |
| 버전 | `1.1.0` |
| 작성자 | community |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Email`, `IMAP`, `SMTP`, `CLI`, `Communication` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Himalaya 이메일 CLI

Himalaya는 IMAP, SMTP, Notmuch 또는 Sendmail 백엔드를 사용해 터미널에서 이메일을 관리할 수 있는 CLI 이메일 클라이언트입니다.

이 스킬은 Hermes 이메일 게이트웨이 어댑터와 별개입니다. 게이트웨이 어댑터는 Hermes에 내장된 IMAP/SMTP 어댑터를 사용해 사람들이 에이전트에 이메일을 보낼 수 있도록 하며, 이 스킬은 에이전트가 터미널 도구를 통해 메일함을 조작할 수 있도록 하고 외부 `himalaya` CLI가 필요합니다.

## 참고 자료

- `references/configuration.md` (설정 파일 설정 + IMAP/SMTP 인증)
- `references/message-composition.md` (이메일 작성용 MML 문법)

## 사전 요구 사항

1. Himalaya CLI 설치 (`himalaya --version`으로 확인)
2. `~/.config/himalaya/config.toml`에 설정 파일이 있어야 함
3. IMAP/SMTP 자격 증명 설정 (비밀번호는 안전하게 저장)

### 설치

```bash
# Pre-built binary (Linux/macOS — recommended)
curl -sSL https://raw.githubusercontent.com/pimalaya/himalaya/master/install.sh | PREFIX=~/.local sh

# macOS via Homebrew
brew install himalaya

# Or via cargo (any platform with Rust)
cargo install himalaya --locked
```

## 설정 구성

대화형 마법사를 실행해 계정을 설정합니다.

```bash
himalaya account configure
```

또는 `~/.config/himalaya/config.toml`을 직접 만듭니다.

```toml
[accounts.personal]
email = "you@example.com"
display-name = "Your Name"
default = true

backend.type = "imap"
backend.host = "imap.example.com"
backend.port = 993
backend.encryption.type = "tls"
backend.login = "you@example.com"
backend.auth.type = "password"
backend.auth.cmd = "pass show email/imap"  # or use keyring

message.send.backend.type = "smtp"
message.send.backend.host = "smtp.example.com"
message.send.backend.port = 587
message.send.backend.encryption.type = "start-tls"
message.send.backend.login = "you@example.com"
message.send.backend.auth.type = "password"
message.send.backend.auth.cmd = "pass show email/smtp"

# Folder aliases (himalaya v1.2.0+ syntax). Required whenever the
# server's folder names don't match himalaya's canonical names
# (inbox/sent/drafts/trash). Gmail is the common case — see
# `references/configuration.md` for the `[Gmail]/Sent Mail` mapping.
folder.aliases.inbox = "INBOX"
folder.aliases.sent = "Sent"
folder.aliases.drafts = "Drafts"
folder.aliases.trash = "Trash"
```

> **별칭 문법에 관한 주의 사항.** v1.2.0 이전 문서에서는
> `[accounts.NAME.folder.alias]` 하위 섹션(단수형 `alias`)을 사용했습니다.
> v1.2.0은 이 형식을 조용히 무시합니다. TOML은 정상적으로 파싱되지만 별칭 확인자가 이를 읽지 않으므로 모든 조회가 표준 이름으로 넘어갑니다. Gmail에서는 SMTP 전송이 성공한 **후** 보낸 편지함 저장이 실패하고 `himalaya message send`가 0이 아닌 상태 코드로 종료됩니다. 이 종료 코드에 따라 재시도하는 호출자(에이전트, 스크립트, 사용자)는 SMTP를 포함한 전체 전송을 다시 실행하게 되어 수신자에게 이메일이 중복으로 전송됩니다. 항상 `[accounts.NAME]` 바로 아래에 `folder.aliases.X`(복수형, 점 표기 키)를 사용하세요.

## Hermes 통합 참고 사항

- **읽기, 목록 조회, 검색, 이동, 삭제**는 모두 터미널 도구를 통해 직접 작동합니다.
- **작성/답장/전달** — 안정성을 위해 파이프 입력(`cat << EOF | himalaya template send`)을 권장합니다. 대화형 `$EDITOR` 모드는 `pty=true` + 백그라운드 + 프로세스 도구로 작동하지만, 편집기와 해당 편집기의 명령을 알고 있어야 합니다.
- 구조화된 출력은 프로그래밍 방식으로 더 쉽게 파싱할 수 있도록 `--output json`을 사용하세요.
- `himalaya account configure` 마법사는 대화형 입력이 필요하므로 PTY 모드를 사용하세요: `terminal(command="himalaya account configure", pty=true)`

## 일반적인 작업

### 폴더 목록 조회

```bash
himalaya folder list
```

### 이메일 목록 조회

받은 편지함(INBOX)의 이메일을 조회합니다(기본값).

```bash
himalaya envelope list
```

특정 폴더의 이메일을 조회합니다.

```bash
himalaya envelope list --folder "Sent"
```

페이지 매김을 사용해 조회합니다.

```bash
himalaya envelope list --page 1 --page-size 20
```

### 이메일 검색

```bash
himalaya envelope list from john@example.com subject meeting
```

### 이메일 읽기

ID로 이메일을 읽습니다(일반 텍스트 표시).

```bash
himalaya message read 42
```

원시 MIME을 내보냅니다.

```bash
himalaya message export 42 --full
```

### 이메일 답장

Hermes에서 비대화형으로 답장하려면 원본 메시지를 읽고 답장을 작성한 뒤 파이프로 전달합니다.

```bash
# Get the reply template, edit it, and send
himalaya template reply 42 | sed 's/^$/\nYour reply text here\n/' | himalaya template send
```

또는 답장을 직접 작성합니다.

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: sender@example.com
Subject: Re: Original Subject
In-Reply-To: <original-message-id>

Your reply here.
EOF
```

전체 답장(대화형 — `$EDITOR`가 필요하므로 위의 템플릿 방식을 대신 사용하세요).

```bash
himalaya message reply 42 --all
```

### 이메일 전달

```bash
# Get forward template and pipe with modifications
himalaya template forward 42 | sed 's/^To:.*/To: newrecipient@example.com/' | himalaya template send
```

### 새 이메일 작성

**비대화형(Hermes에서 사용)** — 표준 입력을 통해 메시지를 파이프로 전달합니다.

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: recipient@example.com
Subject: Test Message

Hello from Himalaya!
EOF
```

또는 헤더 플래그를 사용합니다.

```bash
himalaya message write -H "To:recipient@example.com" -H "Subject:Test" "Message body here"
```

참고: 파이프 입력 없이 `himalaya message write`를 실행하면 `$EDITOR`가 열립니다. `pty=true` + 백그라운드 모드에서 작동하지만, 파이프 방식이 더 간단하고 안정적입니다.

### 이메일 이동/복사

폴더로 이동합니다(대상 폴더가 먼저 오고 그다음에 메시지 ID가 옵니다).

```bash
himalaya message move "Archive" 42
```

폴더로 복사합니다(대상 폴더가 먼저 오고 그다음에 메시지 ID가 옵니다).

```bash
himalaya message copy "Important" 42
```

### 이메일 삭제

```bash
himalaya message delete 42
```

### 플래그 관리

플래그 추가:

```bash
himalaya flag add 42 --flag seen
```

플래그 제거:

```bash
himalaya flag remove 42 --flag seen
```

## 여러 계정

계정 목록을 조회합니다.

```bash
himalaya account list
```

특정 계정을 사용합니다.

```bash
himalaya --account work envelope list
```

## 첨부 파일

메시지에서 첨부 파일을 저장합니다.

```bash
himalaya attachment download 42
```

특정 디렉터리에 저장합니다.

```bash
himalaya attachment download 42 --downloads-dir ~/Downloads
```

## 출력 형식

대부분의 명령은 구조화된 출력을 위해 `--output`을 지원합니다.

```bash
himalaya envelope list --output json
himalaya envelope list --output plain
```

## 디버깅

디버그 로깅을 활성화합니다.

```bash
RUST_LOG=debug himalaya envelope list
```

백트레이스를 포함한 전체 추적:

```bash
RUST_LOG=trace RUST_BACKTRACE=1 himalaya envelope list
```

## 팁

- 자세한 사용법은 `himalaya --help` 또는 `himalaya <command> --help`를 사용하세요.
- 메시지 ID는 현재 폴더를 기준으로 하므로 폴더를 변경한 뒤 목록을 다시 조회하세요.
- 첨부 파일이 있는 서식 있는 이메일을 작성하려면 MML 문법을 사용하세요(`references/message-composition.md` 참고).
- `pass`, 시스템 키링 또는 비밀번호를 출력하는 명령을 사용해 비밀번호를 안전하게 저장하세요.
