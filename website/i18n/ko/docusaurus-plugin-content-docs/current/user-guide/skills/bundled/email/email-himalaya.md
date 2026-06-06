---
title: "Himalaya — Himalaya CLI: IMAP/SMTP email from terminal"
sidebar_label: "Himalaya"
description: "Himalaya CLI: IMAP/SMTP email from terminal"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Himalaya

Himalaya CLI: 터미널에서 IMAP/SMTP 이메일 사용.

## 스킬 메타데이터

| | |
|---|---|
| Source | 번들 (기본 설치) |
| Path | `skills/email/himalaya` |
| Version | `1.1.0` |
| Author | community |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Email`, `IMAP`, `SMTP`, `CLI`, `Communication` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Himalaya Email CLI

Himalaya는 IMAP, SMTP, Notmuch 또는 Sendmail 백엔드를 사용하여 터미널에서 이메일을 관리할 수 있게 해주는 CLI 이메일 클라이언트입니다.

## 참고 자료

- `references/configuration.md` (설정 파일 셋업 + IMAP/SMTP 인증)
- `references/message-composition.md` (이메일 작성을 위한 MML 구문)

## 사전 요구 사항

1. Himalaya CLI 설치 (`himalaya --version`으로 확인)
2. `~/.config/himalaya/config.toml`에 설정 파일 존재
3. IMAP/SMTP 자격 증명 설정 (비밀번호는 안전하게 저장됨)

### 설치

```bash
# 미리 빌드된 바이너리 (Linux/macOS — 권장)
curl -sSL https://raw.githubusercontent.com/pimalaya/himalaya/master/install.sh | PREFIX=~/.local sh

# Homebrew를 통한 macOS 설치
brew install himalaya

# 또는 cargo를 통한 설치 (Rust가 있는 모든 플랫폼)
cargo install himalaya --locked
```

## 설정 구성

계정을 설정하려면 대화형 마법사를 실행하세요:

```bash
himalaya account configure
```

또는 수동으로 `~/.config/himalaya/config.toml`을 생성하세요:

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
backend.auth.cmd = "pass show email/imap"  # 또는 키링 사용

message.send.backend.type = "smtp"
message.send.backend.host = "smtp.example.com"
message.send.backend.port = 587
message.send.backend.encryption.type = "start-tls"
message.send.backend.login = "you@example.com"
message.send.backend.auth.type = "password"
message.send.backend.auth.cmd = "pass show email/smtp"

# 폴더 별칭 (himalaya v1.2.0+ 구문). 서버의 폴더 이름이
# himalaya의 표준 이름(inbox/sent/drafts/trash)과 일치하지
# 않을 때마다 필요합니다. Gmail이 일반적인 경우입니다 —
# `[Gmail]/Sent Mail` 매핑에 대해서는 `references/configuration.md`를 참조하세요.
folder.aliases.inbox = "INBOX"
folder.aliases.sent = "Sent"
folder.aliases.drafts = "Drafts"
folder.aliases.trash = "Trash"
```

> **별칭 구문에 대한 주의 사항.** v1.2.0 이전 문서에서는
> `[accounts.NAME.folder.alias]` 하위 섹션(단수형 `alias`)을 사용했습니다.
> v1.2.0은 이 형태를 조용히 무시합니다 — TOML 파싱은 잘 되지만
> 별칭 리졸버가 이를 읽지 않으므로 모든 조회가 표준 이름으로 귀결됩니다.
> Gmail에서는 이로 인해 SMTP 전송이 성공한 *후에* 보낸편지함(Sent) 저장이 실패하고,
> `himalaya message send`가 0이 아닌 코드(non-zero exit)로 종료됩니다.
> 해당 종료 코드로 인해 재시도하는 호출자(에이전트, 스크립트, 사용자)는
> SMTP를 포함한 전체 전송을 다시 실행하여 수신자에게 중복 이메일을 생성합니다.
> 항상 `folder.aliases.X` (복수형, 점으로 구분된 키, `[accounts.NAME]` 바로 아래)를 사용하세요.

## Hermes 통합 참고 사항

- **읽기, 목록 조회, 검색, 이동, 삭제**는 모두 터미널 도구를 통해 직접 작동합니다.
- **작성/답장/전달** — 안정성을 위해 파이프 입력(`cat << EOF | himalaya template send`)을 권장합니다. 대화형 `$EDITOR` 모드는 `pty=true` + 백그라운드 + 프로세스 도구를 통해 작동하지만, 에디터와 그 명령어를 알아야 합니다.
- 프로그래밍 방식으로 파싱하기 쉬운 구조화된 출력을 얻으려면 `--output json`을 사용하세요.
- `himalaya account configure` 마법사는 대화형 입력이 필요하므로 PTY 모드를 사용하세요: `terminal(command="himalaya account configure", pty=true)`

## 일반적인 작업

### 폴더 목록 보기

```bash
himalaya folder list
```

### 이메일 목록 보기

INBOX(기본값)의 이메일 목록 보기:

```bash
himalaya envelope list
```

특정 폴더의 이메일 목록 보기:

```bash
himalaya envelope list --folder "Sent"
```

페이지 매기기와 함께 목록 보기:

```bash
himalaya envelope list --page 1 --page-size 20
```

### 이메일 검색

```bash
himalaya envelope list from john@example.com subject meeting
```

### 이메일 읽기

ID로 이메일 읽기 (일반 텍스트 표시):

```bash
himalaya message read 42
```

원시 MIME 내보내기:

```bash
himalaya message export 42 --full
```

### 이메일 답장

Hermes에서 비대화형으로 답장하려면 원본 메시지를 읽고 답장을 작성한 다음 파이프로 전달하세요:

```bash
# 답장 템플릿을 가져와 편집한 후 전송
himalaya template reply 42 | sed 's/^$/\n여기에 답장 텍스트 입력\n/' | himalaya template send
```

또는 수동으로 답장을 구성하세요:

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: sender@example.com
Subject: Re: Original Subject
In-Reply-To: <original-message-id>

여기에 답장 내용 입력.
EOF
```

전체 답장 (대화형 — `$EDITOR`가 필요하므로 대신 위의 템플릿 방식을 사용하세요):

```bash
himalaya message reply 42 --all
```

### 이메일 전달 (Forward)

```bash
# 전달 템플릿을 가져와 수정 사항과 함께 파이프로 전달
himalaya template forward 42 | sed 's/^To:.*/To: newrecipient@example.com/' | himalaya template send
```

### 새 이메일 작성

**비대화형 (Hermes에서 이 방법 사용)** — 메시지를 stdin을 통해 파이프로 전달하세요:

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: recipient@example.com
Subject: Test Message

Hello from Himalaya!
EOF
```

또는 헤더 플래그 사용:

```bash
himalaya message write -H "To:recipient@example.com" -H "Subject:Test" "여기에 메시지 본문 입력"
```

참고: 파이프 입력 없이 `himalaya message write`를 실행하면 `$EDITOR`가 열립니다. 이는 `pty=true` + 백그라운드 모드에서 작동하지만, 파이프 방식이 더 간단하고 안정적입니다.

### 이메일 이동/복사

폴더로 이동:

```bash
himalaya message move 42 "Archive"
```

폴더로 복사:

```bash
himalaya message copy 42 "Important"
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

## 다중 계정

계정 목록 보기:

```bash
himalaya account list
```

특정 계정 사용:

```bash
himalaya --account work envelope list
```

## 첨부 파일

메시지에서 첨부 파일 저장:

```bash
himalaya attachment download 42
```

특정 디렉토리에 저장:

```bash
himalaya attachment download 42 --dir ~/Downloads
```

## 출력 형식

대부분의 명령어는 구조화된 출력을 위해 `--output`을 지원합니다:

```bash
himalaya envelope list --output json
himalaya envelope list --output plain
```

## 디버깅

디버그 로깅 활성화:

```bash
RUST_LOG=debug himalaya envelope list
```

백트레이스가 포함된 전체 추적:

```bash
RUST_LOG=trace RUST_BACKTRACE=1 himalaya envelope list
```

## 팁

- 자세한 사용법은 `himalaya --help` 또는 `himalaya <command> --help`를 사용하세요.
- 메시지 ID는 현재 폴더에 상대적입니다. 폴더를 변경한 후 다시 목록을 조회하세요.
- 첨부 파일이 포함된 리치 이메일을 작성하려면 MML 구문을 사용하세요 (`references/message-composition.md` 참조).
- `pass`, 시스템 키링 또는 암호를 출력하는 명령을 사용하여 암호를 안전하게 저장하세요.
