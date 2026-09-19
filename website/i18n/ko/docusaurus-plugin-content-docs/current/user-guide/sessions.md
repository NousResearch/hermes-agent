---
sidebar_position: 7
title: "세션"
description: "세션 지속성, 재개, 검색, 관리 및 플랫폼별 세션 추적"
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# 세션

Hermes Agent는 모든 대화를 세션으로 자동 저장합니다. 세션을 사용하면 대화를 재개하고, 여러 세션을 검색하며, 전체 대화 기록을 관리할 수 있습니다.

## 세션 작동 방식

CLI, Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Teams 또는 다른 메시징 플랫폼에서 시작한 모든 대화는 전체 메시지 기록과 함께 세션으로 저장됩니다. 세션은 다음 위치에서 추적됩니다.

1. **SQLite 데이터베이스** (`~/.hermes/state.db`) — FTS5 전문 검색을 지원하는 구조화된 세션 메타데이터와 전체 메시지 기록

SQLite 데이터베이스에는 다음 정보가 저장됩니다.
- 세션 ID, 소스 플랫폼, 사용자 ID
- **세션 제목** (고유하며 사람이 읽을 수 있는 이름)
- 모델 이름과 구성
- 시스템 프롬프트 스냅샷
- 전체 메시지 기록 (역할, 콘텐츠, 도구 호출, 도구 결과)
- 토큰 수 (입력/출력)
- 타임스탬프 (started_at, ended_at)
- 부모 세션 ID (압축으로 세션이 분할된 경우)

### 컨텍스트에 포함되는 항목

Hermes는 대화를 재개할 수 있도록 세션 기록을 저장하지만, 지금까지 처리한 모든 바이트를 매번 다시 전송하지는 않습니다. 각 턴마다 모델은 선택된 시스템 프롬프트, 현재 대화 창, 그리고 Hermes가 해당 턴에 명시적으로 주입한 콘텐츠를 봅니다.

미디어 첨부 파일은 턴 단위 입력으로 처리됩니다.

- 이미지는 다음 모델 호출에 네이티브 방식으로 첨부되거나, 활성 모델이 네이티브 비전을 지원하지 않을 때 텍스트 설명으로 미리 분석될 수 있습니다.
- 음성-텍스트 변환이 구성되어 있으면 오디오가 텍스트로 변환됩니다.
- 텍스트 문서는 추출된 텍스트가 포함될 수 있으며, 다른 문서 유형은 대개 저장된 로컬 경로와 짧은 메모로 표현됩니다.
- 첨부 파일 경로와 추출되거나 파생된 텍스트는 대화 기록에 나타날 수 있지만, 원본 이미지·오디오·바이너리 파일의 바이트가 이후 프롬프트에 반복해서 복사되지는 않습니다.

예를 들어 사용자가 이미지를 보내고 Hermes에게 밈을 만들어 달라고 요청하면, Hermes는 비전 기능으로 이미지를 한 번 살펴보고 이미지 처리 스크립트를 실행할 수 있습니다. 이후 턴에는 원본 JPEG가 컨텍스트로 자동 전달되지 않습니다. 대신 사용자의 요청, 짧은 이미지 설명, 로컬 캐시 경로 또는 최종 어시스턴트 응답처럼 대화에 기록된 내용만 전달됩니다.

컨텍스트가 커지는 가장 흔한 원인은 미디어 파일 자체가 아닙니다. 붙여 넣은 대화 기록, 전체 로그, 큰 도구 출력, 긴 diff, 반복되는 상태 보고서, 자세한 증명 덤프 같은 장황한 텍스트가 원인입니다. 큰 산출물을 채팅에 복사하기보다 요약, 파일 경로, 필요한 부분만 발췌한 내용, 도구 기반 조회를 사용하세요.

:::tip
세션이 길어지면 `/compress`를 사용하고, 새 스레드가 필요하면 `/new`를 사용하세요. 저장소에서 오래된 종료 세션을 삭제하려는 경우에만 `hermes sessions prune`을 사용하세요. `state.db`가 단순히 커진 경우에는 먼저 파괴적이지 않은 옵션을 사용하세요. `hermes sessions optimize`는 세션 데이터를 건드리지 않고 FTS5 인덱스 세그먼트를 병합하고 데이터베이스를 VACUUM합니다. 압축은 활성 컨텍스트를 줄이는 기능이며, 개인정보 삭제 기능이 아닙니다.
`/new`에 이름을 전달하면 (예: `/new payments-refactor`) 새 세션의 초기 제목을 미리 설정할 수 있습니다. 나중에 `/resume <name>` 또는 `/sessions` 선택기에서 세션을 찾을 때 유용합니다.
:::

### 세션 소스

각 세션에는 소스 플랫폼 태그가 붙습니다.

| 소스 | 설명 |
|--------|-------------|
| `cli` | 대화형 CLI (`hermes` 또는 `hermes chat`) |
| `telegram` | Telegram 메신저 |
| `discord` | Discord 서버/DM |
| `slack` | Slack 워크스페이스 |
| `whatsapp` | WhatsApp 메신저 |
| `signal` | Signal 메신저 |
| `matrix` | Matrix 방 및 DM |
| `mattermost` | Mattermost 채널 |
| `email` | 이메일 (IMAP/SMTP) |
| `sms` | Twilio를 통한 SMS |
| `dingtalk` | DingTalk 메신저 |
| `feishu` | Feishu/Lark 메신저 |
| `wecom` | WeCom (WeChat Work) |
| `weixin` | Weixin (개인 WeChat) |
| `bluebubbles` | BlueBubbles macOS 서버를 통한 Apple iMessage |
| `qqbot` | 공식 API v2를 통한 QQ Bot (Tencent QQ) |
| `homeassistant` | Home Assistant 대화 |
| `webhook` | 수신 웹훅 |
| `api-server` | API 서버 요청 |
| `acp` | ACP 편집기 통합 |
| `cron` | 예약된 cron 작업 |
| `batch` | 배치 처리 실행 |

## CLI 세션 재개

`--continue` 또는 `--resume`을 사용해 CLI에서 이전 대화를 재개할 수 있습니다.

### 마지막 세션 계속하기

```bash
# Resume the most recent CLI session
hermes --continue
hermes -c

# Or with the chat subcommand
hermes chat --continue
hermes chat -c
```

SQLite 데이터베이스에서 가장 최근의 `cli` 세션을 조회하고 전체 대화 기록을 불러옵니다.

### 이름으로 재개하기

세션에 제목을 지정했다면 (아래 [세션 이름 지정](#session-naming) 참고) 이름으로 세션을 재개할 수 있습니다.

```bash
# Resume a named session
hermes -c "my project"

# If there are lineage variants (my project, my project #2, my project #3),
# this automatically resumes the most recent one
hermes -c "my project"   # → resumes "my project #3"
```

### 특정 세션 재개

```bash
# Resume a specific session by ID
hermes --resume 20250305_091523_a1b2c3d4
hermes -r 20250305_091523_a1b2c3d4

# Resume by title
hermes --resume "refactoring auth"

# Resume the most recent session — same lookup as -c
hermes --resume latest

# Or with the chat subcommand
hermes chat --resume 20250305_091523_a1b2c3d4
```

CLI 세션을 종료하면 세션 ID가 표시되며, `hermes sessions list`로도 찾을 수 있습니다.

:::note
`latest`는 `--resume`에서 예약된 키워드입니다. 제목이 실제로 "latest"인 세션도 ID 또는 `-c latest`(제목 일치)를 통해 접근할 수 있습니다.
:::

### 특정 디렉터리에서 재개하기

시작하거나 재개하기 전에 `--in <dir>`을 전달하면 해당 디렉터리로 이동합니다. `--resume latest`(또는 `-c`)와 함께 사용하면 해당 디렉터리의 워크스페이스에 속한 가장 최근 세션이 선택되므로, 먼저 `cd`하거나 세션 ID를 기억할 필요가 없습니다.

```bash
# Resume the latest session that belongs to ./my-project
hermes --resume latest --in ./my-project

# Works with the TUI too
hermes --tui --resume latest --in ./my-project
```

`--in`은 세션을 해당 디렉터리에 고정하기도 합니다. 재개한 세션의 기록된 작업 디렉터리는 복원되지 않습니다 (`--no-restore-cwd`를 전달한 것과 같습니다).

### 재개 시 작업 디렉터리 복원

CLI 세션을 재개하면 세션에 기록된 작업 디렉터리(저장소 루트 또는 프로젝트 디렉터리)로 다시 `cd`하므로, 대화가 원래 속해 있던 워크스페이스에서 이어집니다. 현재 위치에 그대로 머물고 싶다면 `--no-restore-cwd`를 전달하세요.

```bash
hermes --resume 20250305_091523_a1b2c3 --no-restore-cwd
```

`↪ restored workspace dir: …` 줄이 전환을 알려 줍니다. 복원에 실패해도 세션 재개 자체는 중단되지 않습니다.

### 워크스페이스별 세션 필터링

`hermes sessions list`에 `--workspace <needle>`을 전달하면 워크스페이스 키(저장소 루트, 없으면 cwd)가 일치하는 세션만 표시됩니다. 경로 부분 문자열 또는 디렉터리의 정확한 basename으로 일치시킵니다.

```bash
hermes sessions list --workspace my-project
hermes sessions list --workspace ~/code/hermes-agent
```

### 재개 시 대화 요약

세션을 재개하면 입력 프롬프트 전에 Hermes가 스타일이 적용된 패널에 이전 대화의 간결한 요약을 표시합니다.

<img className="docs-terminal-figure" src={useBaseUrl('/img/docs/session-recap.svg')} alt="Hermes 세션을 재개할 때 표시되는 이전 대화 요약 패널의 양식화된 미리보기." />
<p className="docs-figure-caption">재개 모드에서는 최근 사용자 및 어시스턴트 턴이 포함된 간결한 요약 패널을 표시한 뒤 실시간 프롬프트로 돌아갑니다.</p>

요약 패널은 다음을 수행합니다.
- **사용자 메시지**(금색 `●`)와 **어시스턴트 응답**(초록색 `◆`)을 표시합니다.
- 긴 메시지를 **잘라냅니다**(사용자는 300자, 어시스턴트는 200자/3줄).
- **도구 호출을** 도구 이름과 함께 개수로 축약합니다(예: `[3 tool calls: terminal, web_search]`).
- 시스템 메시지, 도구 결과 및 내부 추론을 **숨깁니다**.
- 마지막 10개 교환으로 **제한**하고 `"... N earlier messages ..."` 표시를 사용합니다.
- 활성 대화와 구분되도록 **어두운 스타일**을 사용합니다.

요약을 비활성화하고 최소 한 줄만 표시하는 기존 동작을 유지하려면 `~/.hermes/config.yaml`에 다음을 설정하세요.

```yaml
display:
  resume_display: minimal   # default: full
```

:::tip
세션 ID 형식은 `YYYYMMDD_HHMMSS_<hex>`입니다. CLI/TUI 세션은 6자 hex 접미사(예: `20250305_091523_a1b2c3`), 게이트웨이 세션은 8자 접미사(예: `20250305_091523_a1b2c3d4`)를 사용합니다. ID(전체 또는 고유 접두사)나 제목으로 재개할 수 있으며, `-c`와 `-r` 모두에서 작동합니다.
:::

## 플랫폼 간 인계

CLI 세션에서 `/handoff <platform>`을 사용하면 현재 대화를 메시징 플랫폼의 홈 채널로 전달합니다. 에이전트는 CLI에서 중단한 정확한 지점(동일한 세션 ID, 전체 역할 인식 대화 기록과 모든 도구 호출)에서 이어갑니다.

```bash
# Inside a CLI session
/handoff telegram
```

동작 과정:

1. CLI가 `<platform>`이 활성화되어 있고 홈 채널이 설정되어 있는지 확인합니다(설정하려면 대상 채팅에서 `/sethome`을 한 번 실행하세요).
2. CLI가 세션을 대기 상태로 표시하고 **게이트웨이를 블로킹 폴링**합니다. 에이전트가 턴을 처리 중이면 거부하므로 현재 응답이 끝날 때까지 기다려야 합니다.
3. 게이트웨이 감시자가 인계를 확보하고 대상 어댑터에 새 스레드를 요청합니다.
   - **Telegram** — 새 포럼 토픽을 엽니다(채팅에서 Bot API 9.4+ 토픽 모드가 활성화된 DM 토픽 또는 포럼 슈퍼그룹 토픽).
   - **Discord** — 홈 텍스트 채널 아래에 1440분 후 자동 보관되는 스레드를 만듭니다.
   - **Slack** — 시드 메시지를 게시하고 해당 메시지의 `ts`를 스레드 앵커로 사용합니다.
   - **WhatsApp / Signal / Matrix / SMS** — 네이티브 스레드가 없으므로 홈 채널로 직접 폴백합니다.
4. 게이트웨이가 대상 키를 기존 CLI 세션 ID에 다시 바인딩한 다음, 에이전트에게 확인 및 요약을 요청하는 합성 사용자 턴을 생성합니다. 응답은 새 스레드에 도착합니다.
5. 게이트웨이가 성공을 확인하면 CLI가 `/resume` 힌트를 출력하고 정상적으로 종료합니다.

   ```
   ↻ Handoff complete. The session is now active on telegram.
     Resume it on this CLI later with: /resume my-session-title
   ```

6. 그 시점부터 대화는 플랫폼에서 진행됩니다. 새 스레드에서 답장하면 해당 채널에서 권한이 있는 모든 사용자가 동일한 세션을 공유합니다. 이후 스레드에서 실제 사용자 메시지가 오면 `user_id` 없이 스레드 세션 키가 지정되므로 자연스럽게 합류합니다.

**CLI로 다시 재개하기:** 데스크톱으로 돌아오려면 `/resume <title>`을 실행하거나 셸에서 `hermes -r "<title>"`을 실행하세요. 그러면 플랫폼에서 진행하던 지점부터 이어집니다.

**실패 모드:**
- 홈 채널이 구성되지 않음 → CLI가 `/sethome` 힌트와 함께 거부합니다.
- 플랫폼이 활성화되지 않았거나 게이트웨이가 실행 중이 아님 → CLI가 60초 후 명확한 메시지와 함께 시간 초과되며 CLI 세션은 그대로 유지됩니다.
- 스레드 생성 실패(권한, 토픽 모드 비활성화) → 홈 채널로 직접 폴백하면서 인계를 완료합니다. 스레드 격리는 없지만 인계 자체는 작동합니다.
- `adapter.send` 실패(속도 제한, 일시적 API 오류) → 인계가 이유와 함께 실패로 표시되고 행이 삭제되므로 다시 시도할 수 있습니다.

**알아 두면 좋은 제한:** 스레드를 지원하지 않는 플랫폼의 다중 사용자 그룹 홈 채널에서는 합성 턴이 DM 방식 세션으로 키 지정됩니다. 일반적인 설정인 자기 자신에게 보내는 DM 홈 채널에서는 작동하지만, 실제로 공유되는 그룹 채팅에는 적합하지 않습니다. Telegram / Discord / Slack에서는 스레딩이 지원되며, 이는 가장 일반적인 경우이므로 대부분의 설정에서는 이 제한을 만나지 않습니다.

## 세션 이름 지정

세션을 쉽게 찾고 재개할 수 있도록 사람이 읽을 수 있는 제목을 지정하세요.

### 자동 생성 제목

Hermes는 첫 번째 교환이 끝난 후 각 세션에 대해 짧고 설명적인 제목(3~7단어)을 자동으로 생성합니다. 이 작업은 빠른 보조 모델을 사용하는 백그라운드 스레드에서 실행되므로 지연이 발생하지 않습니다. `hermes sessions list` 또는 `hermes sessions browse`로 세션을 탐색할 때 자동 생성된 제목을 볼 수 있습니다.

세션당 한 번만 자동 제목이 생성되며, 수동으로 제목을 설정한 경우에는 건너뜁니다.

### 수동으로 제목 설정

모든 채팅 세션(CLI 또는 게이트웨이) 안에서 `/title` 슬래시 명령을 사용하세요.

```
/title my research project
```

제목은 즉시 적용됩니다. 아직 데이터베이스에 세션이 생성되지 않은 경우(예: 첫 메시지를 보내기 전에 `/title` 실행), 제목은 대기열에 들어갔다가 세션이 시작되면 적용됩니다.

명령줄에서 기존 세션의 이름을 바꿀 수도 있습니다.

```bash
hermes sessions rename 20250305_091523_a1b2c3d4 "refactoring auth module"
```

### 제목 규칙

- **고유성** — 두 세션이 같은 제목을 공유할 수 없습니다.
- **최대 100자** — 목록 출력을 깔끔하게 유지합니다.
- **정제됨** — 제어 문자, 0 너비 문자, RTL 재정의 문자가 자동으로 제거됩니다.
- **일반 Unicode 허용** — 이모지, CJK, 악센트 문자를 모두 사용할 수 있습니다.

### 압축 시 자동 계보 생성

세션의 컨텍스트가 압축되면(`/compress`를 통한 수동 압축 또는 자동 압축) Hermes는 새 연속 세션을 만듭니다. 원래 세션에 제목이 있었다면 새 세션은 자동으로 번호가 붙은 제목을 받습니다.

```
"my project" → "my project #2" → "my project #3"
```

이름으로 재개하면(`hermes -c "my project"`) 해당 계보에서 가장 최근 세션이 자동으로 선택됩니다.

### 메시징 플랫폼의 /title

`/title` 명령은 모든 게이트웨이 플랫폼(Telegram, Discord, Slack, WhatsApp)에서 작동합니다.

- `/title My Research` — 세션 제목 설정
- `/title` — 현재 제목 표시

## 세션 관리 명령

Hermes는 `hermes sessions`를 통해 전체 세션 관리 명령을 제공합니다.

### 세션 목록 보기

```bash
# List recent sessions (default: last 20)
hermes sessions list

# Filter by platform
hermes sessions list --source telegram

# Show more sessions
hermes sessions list --limit 50
```

세션에 제목이 있으면 출력에 제목, 미리보기, 상대 타임스탬프가 표시됩니다.

```
Title                  Preview                                  Last Active   ID
────────────────────────────────────────────────────────────────────────────────────────────────
refactoring auth       Help me refactor the auth module please   2h ago        20250305_091523_a
my project #3          Can you check the test failures?          yesterday     20250304_143022_e
—                      What's the weather in Las Vegas?          3d ago        20250303_101500_f
```

제목이 있는 세션이 없으면 더 단순한 형식을 사용합니다.

```
Preview                                            Last Active   Src    ID
──────────────────────────────────────────────────────────────────────────────────────
Help me refactor the auth module please             2h ago        cli    20250305_091523_a
What's the weather in Las Vegas?                    3d ago        tele   20250303_101500_f
```

### 세션 내보내기

`hermes sessions export`는 `--format`으로 선택하는 모든 내보내기 형식을 위한 단일 인터페이스입니다.

| 형식 | 출력 | 용도 |
|--------|--------|------------|
| `jsonl` (기본값) | 세션당 JSON 객체 하나 | 백업, 컴퓨터 간 왕복 처리 |
| `md` / `qmd` | 세션당 Markdown/Quarto 파일 하나 + 매니페스트 | 읽기 쉬운 보관 파일, 노트 |
| `html` | 자체 완결형 단일 페이지(여러 세션이면 사이드바) | 공유, 탐색 |
| `trace` | Claude Code JSONL | HF Agent Trace Viewer, `--upload` |

또한 프롬프트만 표시하는 뷰(`jsonl` 또는 `md`)를 위해 `--only user-prompts`를 사용할 수 있습니다.

모든 형식은 동일한 선택 옵션을 공유합니다. 하나의 세션에는 `--session-id`를 사용하고, 일괄 처리에는 전체 `prune`/`archive` 필터 집합을 사용합니다 — `--older-than` / `--newer-than` / `--before` / `--after`(`5h`/`2d`/`1w` 같은 기간, 숫자만 있는 일수 또는 ISO 타임스탬프), `--source`, `--title`, `--model`, `--provider`, `--cwd`, `--min/--max-messages`, `--min/--max-tokens`, `--min/--max-cost`, `--min/--max-tool-calls`, `--user`, `--chat-id`, `--chat-type`, `--branch`, `--end-reason`을 사용할 수 있습니다. `--dry-run`은 파일을 쓰지 않고 일치 집합을 미리 보여 줍니다. `--redact`는 어떤 형식에서든 내보낸 콘텐츠의 비밀(API 키, 토큰, 자격 증명)을 삭제합니다. 공유할 계획이 있는 콘텐츠에는 권장됩니다. 참고: 일괄 필터는 *종료된* 세션에 일치하며, 필터 없이 `export`하면 활성 세션을 포함한 모든 세션을 덤프합니다.

#### JSONL (기본값)

```bash
# Export all sessions to a JSONL file
hermes sessions export backup.jsonl

# Export sessions from a specific platform
hermes sessions export telegram-history.jsonl --source telegram

# Export a single session
hermes sessions export session.jsonl --session-id 20250305_091523_a1b2c3d4

# Redact API keys/tokens/credentials from the exported content
hermes sessions export backup.jsonl --redact
```

내보낸 파일에는 줄마다 전체 세션 메타데이터와 모든 메시지가 포함된 JSON 객체 하나가 들어 있습니다.

#### HTML

`--format html`은 원격 의존성이 없는 자체 완결형 HTML 파일 하나를 작성합니다. 스타일이 적용된 메시지 버블, 접을 수 있는 도구 출력, 그리고 여러 세션을 내보낼 때 세션 간 전환을 위한 사이드바가 포함됩니다.

```bash
# One session as a standalone HTML page
hermes sessions export --format html --session-id 20250305_091523_a1b2c3d4 transcript.html

# All Telegram sessions from the last week in one file, secrets redacted
hermes sessions export --format html --newer-than 1w --source telegram --redact archive.html
```

#### 프롬프트만

`--only user-prompts`는 작성한 프롬프트만 내보냅니다. 어시스턴트 응답, 도구 출력, 시스템 컨텍스트는 제외됩니다. 프롬프트 라이브러리를 만들거나 자신이 요청한 내용을 검토할 때 유용합니다.

```bash
# One JSONL record per prompt (session id, index, timestamp, text)
hermes sessions export prompts.jsonl --session-id 20250305_091523_a1b2c3d4 --only user-prompts

# Markdown, straight to stdout
hermes sessions export - --session-id 20250305_091523_a1b2c3d4 --only user-prompts --format md
```

`--format jsonl`(기본값) 또는 `md`와 함께 사용할 수 있고, 일괄 내보내기에는 동일한 필터가 적용되며, `--redact`와 조합할 수 있습니다.

#### 트레이스 (HF Agent Trace Viewer)

`--format trace`는 Claude Code JSONL을 출력합니다. Hugging Face Hub가 [Agent Trace Viewer](https://huggingface.co/docs/hub/agent-traces)용으로 자동 인식하는 대화 기록 형식입니다. 로컬에 쓰거나 `--upload`를 추가해 자신만의 비공개 `hermes-traces` 데이터셋으로 업로드할 수 있습니다(`HF_TOKEN`을 읽음).

```bash
# Trace of the most recent session, to stdout
hermes sessions export --format trace

# One session to a local trace file
hermes sessions export --format trace --session-id 20250305_091523_a1b2c3d4 trace.jsonl

# Upload straight to your private HF traces dataset
hermes sessions export --format trace --session-id 20250305_091523_a1b2c3d4 --upload
```

트레이스 내보내기는 컴퓨터 밖으로 나가는 것을 전제로 기본적으로 비밀이 삭제됩니다. 수동 검토 후 `--no-redact`로 해제할 수 있습니다. `--upload`는 `--public`을 사용하지 않는 한 비공개입니다. 필터를 사용한 일괄 트레이스 내보내기는 세션마다 `<id>.trace.jsonl` 파일 하나를 작성합니다.

#### Markdown / QMD

오래된 세션을 숨기거나 삭제하기 전에 읽기 쉬운 파일 기반 보관본을 만들려면 `--format md` 또는 `--format qmd`를 전달하세요. Markdown/QMD 내보내기는 기본적으로 디렉터리(`~/.hermes/session-exports`)에 세션당 파일 하나를 작성합니다.

```bash
# Export one session to Markdown
hermes sessions export --format md --session-id 20250305_091523_a1b2c3d4

# Export a compression lineage as one logical document
hermes sessions export --format md --session-id 20250305_091523_a1b2c3d4 --lineage logical

# Preview ended sessions older than 90 days without writing files
hermes sessions export --format md --older-than 90 --dry-run

# Export ended Telegram sessions older than 2 weeks to QMD files
hermes sessions export --format qmd --older-than 2w --source telegram

# Export long Claude sessions, secrets redacted
hermes sessions export --format md --model sonnet --min-messages 50 --redact

# Only after verification, export and delete one explicitly named session
hermes sessions export --format md --session-id 20250305_091523_a1b2c3d4 --delete-after-verified --yes
```

Markdown/QMD 내보내기는 내보낸 세션마다 `.md` 또는 `.qmd` 파일 하나와 파일 경로, 메시지 수, 계보 ID, SHA-256이 담긴 `manifest.jsonl`을 작성합니다. 일괄 내보내기에는 하나 이상의 필터가 필요하며, 필터 없는 일괄 내보내기는 거부됩니다. `--delete-after-verified`는 의도적으로 `--session-id`에만 제한되며 `--yes`가 필요합니다. 부모 세션을 삭제하면 위임/하위 에이전트 세션도 삭제되므로, 이 모드는 삭제 전에 각 위임 세션을 별도 파일로 내보내고 확인합니다. 내보내는 동안 위임 세션 집합이 변경되면 삭제가 거부됩니다. `--redact`는 기록하기 전에 메시지 콘텐츠와 도구 출력에서 비밀(API 키, 토큰, 자격 증명)을 삭제합니다. 공유할 내보내기에는 권장됩니다.

### 세션 삭제

```bash
# Delete a specific session (with confirmation)
hermes sessions delete 20250305_091523_a1b2c3d4

# Delete without confirmation
hermes sessions delete 20250305_091523_a1b2c3d4 --yes
```

### 세션 이름 변경

```bash
# Set or change a session's title
hermes sessions rename 20250305_091523_a1b2c3d4 "debugging auth flow"

# Multi-word titles don't need quotes in the CLI
hermes sessions rename 20250305_091523_a1b2c3d4 debugging auth flow
```

제목이 다른 세션에서 이미 사용 중이면 오류가 표시됩니다.

### 오래된 세션 정리

```bash
# Delete ended sessions inactive for 90 days (default)
hermes sessions prune

# Custom age threshold — bare numbers are days
hermes sessions prune --older-than 30

# Durations work too: 5h, 30m, 2d, 1w
hermes sessions prune --older-than 12h

# Delete only a specific time window (e.g. a batch of test sessions
# created in the last 5 hours)
hermes sessions prune --newer-than 5h

# Explicit window with absolute timestamps
hermes sessions prune --after "2026-07-05 09:00" --before "2026-07-05 14:30"

# Only prune sessions from a specific platform (all ages — any filter
# disables the implicit 90-day default)
hermes sessions prune --source telegram
hermes sessions prune --source cron --older-than 60   # add a time flag to narrow

# More filters — all AND together
hermes sessions prune --newer-than 5h --title "smoke test"   # title substring
hermes sessions prune --older-than 30 --max-messages 3        # tiny sessions
hermes sessions prune --cwd ~/scratch --end-reason done       # by cwd / end reason
hermes sessions prune --model gpt-5 --older-than 1w           # by model (substring)
hermes sessions prune --provider openrouter --older-than 60   # by billing provider
hermes sessions prune --branch feature/old-experiment         # by git branch
hermes sessions prune --user 12345678 --chat-type group       # by messaging origin
hermes sessions prune --max-tokens 500 --older-than 7         # by token usage
hermes sessions prune --max-cost 0.01 --max-tool-calls 0      # cheap, tool-less runs

# Preview what would be deleted, without deleting anything
hermes sessions prune --newer-than 5h --dry-run

# Skip confirmation
hermes sessions prune --older-than 30 --yes
```

시간 값(`--older-than`, `--newer-than`, `--before`, `--after`)은 기간(`5h`, `30m`, `2d`, `1w`), 숫자만 있는 일수 또는 ISO 타임스탬프(`2026-07-05`, `2026-07-05 14:30`)를 허용합니다. `--older-than`/`--before`는 상한을 설정하고, `--newer-than`/`--after`는 하한을 설정합니다. `--older-than`/`--newer-than` 쌍은 최신 메시지 활동을 사용하며, 빈 세션은 세션 시작 시각으로 대체합니다. `--before`/`--after`는 세션 시작 시각을 명시적으로 사용합니다. 어느 쌍이든 조합해 범위를 만들 수 있습니다.

속성 필터: `--source`(플랫폼, 정확히 일치), `--title` / `--model` / `--branch`(대소문자를 구분하지 않는 부분 문자열), `--provider`(청구 제공자, 정확히 일치), `--end-reason`, `--user`, `--chat-id`, `--chat-type`(정확히 일치), `--cwd`(경로 접두사), 그리고 숫자 범위 `--min/--max-messages`, `--min/--max-tokens`(입력+출력), `--min/--max-cost`(USD, 실제 값이 없으면 추정값), `--min/--max-tool-calls`가 있습니다. 필터를 하나라도 사용하면 암묵적인 90일 기본값이 해제됩니다. 따라서 `hermes sessions prune --source cron` 또는 `--model gpt-4o`는 모든 기간에 일치하므로 범위를 좁히려면 시간 플래그를 추가하세요. 완전히 인자가 없는 `hermes sessions prune`만 90일 기준을 유지합니다. `--yes`가 없는 모든 실행은 확인을 요청하기 전에 일치하는 세션 수와 가장 오래된 세션 및 가장 최근 세션을 표시합니다.

보관된 세션은 기본적으로 건너뜁니다. 보관된 세션도 삭제하려면 `--include-archived`를 전달하세요.

:::info
정리는 **종료된** 세션만 삭제합니다(명시적으로 종료되었거나 자동으로 초기화된 세션). 활성 세션은 절대 정리되지 않습니다.
:::

### 세션 일괄 보관

삭제하지 않고 목록에서 세션을 숨기고 싶다면 `hermes sessions archive`가 `prune`과 동일한 필터를 사용해 일치하는 세션을 대신 소프트 숨김 처리합니다. 단일 세션을 Desktop/Dashboard UI에서 보관할 때와 동일한 보관 플래그를 설정하며, 메시지와 검색은 그대로 유지됩니다.

```bash
# Archive everything from the last 5 hours (e.g. 75 CI smoke-test sessions)
hermes sessions archive --newer-than 5h

# Archive by title substring, preview first
hermes sessions archive --title "dry run" --dry-run
hermes sessions archive --title "dry run" --yes
```

하나 이상의 필터가 필요합니다. 인자 없는 `hermes sessions archive`는 전체 기록을 보관하는 것을 거부합니다. 보관된 세션은 `hermes sessions list`와 `/resume`에서 숨겨지지만 데이터베이스에는 남으며, Desktop/Dashboard 세션 목록에서 보관을 해제할 수 있습니다.

### 세션 통계

```bash
hermes sessions stats
```

출력:

```
Total sessions: 142
Total messages: 3847
  cli: 89 sessions
  telegram: 38 sessions
  discord: 15 sessions
Database size: 12.4 MB
```

더 심층적인 분석(토큰 사용량, 비용 추정, 도구별 분류, 활동 패턴)에는 [`hermes insights`](/reference/cli-commands#hermes-insights)를 사용하세요.

### 고립된 게이트웨이 세션 복구

게이트웨이를 다시 시작한 뒤 대화가 "시간을 거슬러 올라가" 최근 메시지가 전혀 없었던 것처럼 며칠 전 주제를 재개한다면, 실시간 대화가 라우팅 ID를 잃은 세션 행에 고립되었을 수 있습니다(이 문제 유형은 v0.21 세션 연속성 작업에서 수정되었으며, 현재 버전은 생성 단계에서 방지하고 런타임에 자체 복구합니다).

`hermes sessions repair-routing`은 라우팅 ID가 없는 메시지 포함 세션 행을 찾아, 증거가 모호하지 않은 경우에만 각 행을 이어지는 대화에 다시 연결합니다.

```bash
# Report only — shows each orphan, the proposed adoption, and the evidence
hermes sessions repair-routing

# Perform the adoptions (stop the gateway first — a running gateway holds
# the old routing in memory and would write it back over the repair)
hermes sessions repair-routing --apply

# Widen/narrow the contiguity window (default 900 seconds)
hermes sessions repair-routing --max-gap-seconds 300
```

증거 규칙:

- **계보** — 고아의 `parent_session_id`가 동일한 플랫폼의 키 지정 행을 가리킵니다(기록된 사실이므로 시간 창이 적용되지 않음).
- **연속성** — 동일한 플랫폼의 키 지정 행 중 정확히 하나가 고아의 시작 시점 기준 창 안에서 활동을 멈췄습니다.

모호한 경우(후보 선행 행이 두 개이거나, 두 고아가 같은 선행 행을 주장하는 경우)는 이유와 함께 보고되고 변경되지 않습니다. 잘못 채택하면 한 대화가 다른 채팅에 이어 붙을 수 있기 때문입니다. 대체된 행은 `superseded_by_repair`로 폐기되므로 재시작 복구가 이를 다시 되살릴 수 없습니다.

복구는 의도적으로 **자동화되지 않습니다**. 채팅이 그 이후 두 번째 기록을 쌓았다면 어느 스레드를 이어갈지 결정하는 것은 사용자입니다. 어느 쪽이든 고립된 대화는 `/resume`과 세션 검색을 통해 계속 읽을 수 있으며, 복구가 변경하는 것은 라우팅뿐입니다. 먼저 백업하세요.
(`cp ~/.hermes/state.db ~/.hermes/state.db.bak`)


## 세션 검색 도구

에이전트에는 SQLite의 FTS5 엔진을 사용해 과거 모든 대화에서 전문 검색을 수행하고, 검색된 세션을 스크롤할 수 있는 내장 `session_search` 도구가 있습니다. LLM을 호출하지 않으며 요약을 생성하는 대신 데이터베이스의 실제 메시지 뷰를 반환합니다.

### 네 가지 호출 형태

도구는 어떤 인자를 설정했는지에 따라 원하는 작업을 추론합니다. `mode` 매개변수는 없습니다.

**1. 검색 — `query` 전달:**

```python
session_search(query="auth refactor", limit=3)
```

FTS5를 실행하고 세션 계보별로 중복을 제거한 뒤 상위 N개 세션을 반환합니다. 검색은 기본적으로 적응형 세부 정보를 사용합니다. 가장 높은 순위의 결과에는 전체 컨텍스트 창과 앞뒤 메시지가 포함되고, 순위가 낮은 결과는 간결하게 유지됩니다. `detail="full"`을 전달하면 모든 결과를 완전히 불러옵니다.

각 결과에는 다음이 포함됩니다.

- `session_id`, `title`, `when`, `source`
- `snippet` — FTS5로 일치 부분이 강조된 발췌문
- `detail` — `full` 또는 `compact`
- `bookend_start` / `bookend_end` — 전체 결과에서는 처음/마지막 사용자+어시스턴트 메시지 3개, 간결한 결과에서는 빈 목록
- `messages` — 전체 결과에서는 FTS5 일치 주변의 ±5개 메시지, 간결한 결과에서는 표시된 앵커 메시지만 포함
- `match_message_id`, `messages_before`, `messages_after`

상위 결과는 목표 → 일치 → 해결 과정을 즉시 재구성합니다. 다른 간결한 결과가 더 유망해 보이면 해당 세션 및 메시지 ID를 스크롤 형태에 사용하세요. 실제 세션 데이터베이스에서 일반적인 소요 시간은 수십 밀리초입니다.

**2. 스크롤 — `session_id` + `around_message_id` 전달:**

```python
session_search(session_id="20260510_174648_805cc2", around_message_id=590803, window=10)
```

앵커를 중심으로 ±`window`개의 메시지 창을 반환합니다. FTS5와 앞뒤 메시지는 사용하지 않고 해당 조각만 반환합니다. 기본 ±5 창보다 더 많은 컨텍스트가 필요할 때 검색 호출 후 사용하세요.

- **앞으로** 스크롤하려면 `messages[-1].id`를 `around_message_id`로 다시 전달합니다.
- **뒤로** 스크롤하려면 `messages[0].id`를 `around_message_id`로 다시 전달합니다.
- 경계 메시지는 방향을 잡는 표식으로 두 창에 모두 나타납니다.
- `messages_before` 또는 `messages_after`가 `window`보다 작으면 세션의 시작 또는 끝에 도달한 것입니다.

일반적인 소요 시간은 스크롤 호출당 1~2ms입니다.

**3. 읽기 — 앵커 없이 `session_id` 전달:**

```python
session_search(session_id="20260510_174648_805cc2")
```

전체 세션을 반환하거나, 큰 세션에는 시작/끝을 제한한 뷰를 반환합니다. `@session:<profile>/<id>` 링크를 해석할 때도 이 형태를 사용합니다.

**4. 탐색 — 인자 없음:**

```python
session_search()
```

최근 세션을 시간순으로 반환합니다(제목, 미리보기, 타임스탬프). 주제를 지정하지 않고 사용자가 "무엇을 작업하고 있었지?"라고 물을 때 유용합니다.

### FTS5 쿼리 구문

키워드 모드는 표준 FTS5 쿼리 구문을 지원합니다.

- 단순 키워드: `docker deployment` (FTS5 기본값은 AND)
- 구문: `"exact phrase"`
- 불리언: `docker OR kubernetes`, `python NOT java`
- 접두사: `deploy*`

### 선택적 매개변수

- `sort` — FTS5 순위 위에 적용할 `newest` 또는 `oldest`입니다. 관련성만으로 정렬하려면 생략하세요(기본값이며 탐색형 회상에 적합). "X를 어디까지 했지?"에는 `newest`, "X가 어떻게 시작됐지?"에는 `oldest`를 사용하세요.
- `detail` — `adaptive`(기본값)는 검색 결과 중 상위 결과만 완전히 불러오고, `full`은 모든 검색 결과를 완전히 불러옵니다.
- `role_filter` — 포함할 역할을 쉼표로 구분합니다. 검색 기본값은 `user,assistant`입니다(도구 출력은 대개 잡음). 도구 출력도 포함하려면 `user,assistant,tool`을, 도구 출력만 검색하려면 `tool`을 전달하세요.

### 사용 시점

에이전트는 세션 검색을 자동으로 사용하도록 프롬프트가 지정되어 있습니다.

> *"사용자가 과거 대화의 내용을 언급하거나 관련 과거 컨텍스트가 있다고 생각되면, 다시 설명해 달라고 요청하기 전에 session_search를 사용해 기억해 내세요."*

일반적인 트리거는 다음과 같습니다. "전에 이걸 했지", "기억나?", "지난번에", "내가 말했듯이" 또는 현재 창에 없는 프로젝트/사람/개념에 대한 모든 언급입니다.

## 플랫폼별 세션 추적

### 게이트웨이 세션

메시징 플랫폼에서 세션은 메시지 소스로부터 결정론적으로 만든 세션 키로 지정됩니다.

| 채팅 유형 | 기본 키 형식 | 동작 |
|-----------|--------------------|----------|
| Telegram DM | `agent:main:telegram:dm:<chat_id>` | DM 채팅마다 하나의 세션 |
| Discord DM | `agent:main:discord:dm:<chat_id>` | DM 채팅마다 하나의 세션 |
| WhatsApp DM | `agent:main:whatsapp:dm:<canonical_identifier>` | 사용자마다 하나의 세션 (매핑이 있으면 LID/전화번호 별칭을 하나의 ID로 통합) |
| 그룹 채팅 | `agent:main:<platform>:group:<chat_id>:<user_id>` | 플랫폼이 사용자 ID를 제공하면 그룹 안에서 사용자별 세션 |
| 그룹 스레드/토픽 | `agent:main:<platform>:group:<chat_id>:<thread_id>` | 모든 스레드 참여자가 공유하는 세션 (기본값). `thread_sessions_per_user: true`이면 사용자별 |
| 채널 | `agent:main:<platform>:channel:<chat_id>:<user_id>` | 플랫폼이 사용자 ID를 제공하면 채널 안에서 사용자별 세션 |

공유 채팅에서 Hermes가 참여자 식별자를 얻지 못하면 해당 방에서 하나의 공유 세션으로 폴백합니다.

### 공유 그룹 세션과 격리 그룹 세션

기본적으로 Hermes는 `config.yaml`에서 `group_sessions_per_user: true`를 사용합니다. 즉:

- Alice와 Bob은 같은 Discord 채널에서 Hermes와 대화해도 대화 기록을 공유하지 않습니다.
- 한 사용자의 도구가 많이 필요한 긴 작업이 다른 사용자의 컨텍스트 창을 오염시키지 않습니다.
- 실행 중인 에이전트 키가 격리된 세션 키와 일치하므로 중단 처리도 사용자별로 유지됩니다.

대신 공유되는 하나의 "방 두뇌"를 원한다면 다음을 설정하세요.

```yaml
group_sessions_per_user: false
```

그러면 그룹/채널이 방마다 하나의 공유 세션으로 되돌아갑니다. 공유 대화 컨텍스트는 유지되지만 토큰 비용, 중단 상태, 컨텍스트 증가도 공유됩니다.

### 세션 초기화 정책

**기본적으로 게이트웨이 세션은 자동으로 초기화되지 않습니다** (`mode: none`). `config.yaml`의 `session_reset` 섹션에서 자동 초기화를 선택할 수 있습니다.

- **none** — 자동 초기화 안 함(기본값; `/reset`과 압축으로 컨텍스트 관리)
- **idle** — N분 동안 활동이 없으면 초기화
- **daily** — 매일 특정 시각에 초기화
- **both** — 유휴 또는 일일 조건 중 먼저 도달한 시점에 초기화

세션이 자동 초기화되기 전에 에이전트에는 대화에서 중요한 메모리나 스킬을 저장할 한 턴이 주어집니다.

활성 백그라운드 프로세스가 있는 세션은 정책과 관계없이 절대 자동 초기화되지 않습니다.

### 충돌 및 재시작 후 연속성

게이트웨이 채팅은 명시적으로 `/new`(또는 `/reset`)를 실행할 때까지 계속 압축되는 **하나의 연속 세션**으로 설계되었습니다. 이 동작은 게이트웨이 충돌, 재시작, 업데이트 후에도 유지됩니다.

- 세션 ID(라우팅 키, 채팅, 출처)는 세션 행이 생성될 때와 모든 생성 경로(`/new`, 첫 메시지, `/branch` 하위 세션)에서 **원자적으로** 기록됩니다. 이 기록이 실패하면 다음 턴의 라우팅 갱신이 행을 자동으로 복구합니다.
- 재시작 후 게이트웨이는 각 채팅을 **실제 활동이 가장 최근인** 세션에 다시 연결합니다. 오래되고 오래된 행이 실제로 진행 중이던 대화보다 우선되는 일은 없습니다.
- 복구는 `/new` 경계를 **존중**합니다. 채팅의 가장 최근 이벤트가 의도적인 초기화라면 복구는 새로 시작하며, 초기화 이전의 오래된 세션을 되살리지 않습니다. 복구된 세션도 실제 유휴 시간을 유지하므로, 선택한 유휴/일일 초기화 정책이 모든 복구 세션을 새 세션으로 취급하지 않고 올바르게 적용됩니다.


## 저장 위치

| 항목 | 경로 | 설명 |
|------|------|-------------|
| SQLite 데이터베이스 | `~/.hermes/state.db` | FTS5를 사용하는 모든 세션 메타데이터 + 메시지 |
| 게이트웨이 메시지 | `~/.hermes/state.db` | 모든 세션 메시지의 정식 SQLite 저장소 |
| 게이트웨이 라우팅 인덱스 | `~/.hermes/state.db`의 `gateway_routing` 테이블 | 세션 키를 활성 세션 ID에 매핑(출처 메타데이터, 만료 플래그) |
| 레거시 라우팅 미러 | `~/.hermes/sessions/sessions.json` | `gateway.write_sessions_json: true`(기본값)일 때 작성되는 라우팅 인덱스의 이전 버전 호환 미러 |

SQLite 데이터베이스는 동시 읽기와 단일 쓰기를 위해 WAL 모드를 사용하며, 이는 게이트웨이의 멀티플랫폼 아키텍처에 적합합니다.

:::warning `sessions.json`은 세션 목록이 아닙니다
게이트웨이 라우팅 인덱스는 `state.db` 안의 `gateway_routing` 테이블에 있습니다. `~/.hermes/sessions/sessions.json`은 이전 버전과의 호환성을 위해 유지되는 **레거시 미러**입니다(`gateway.write_sessions_json: false`로 비활성화). 이 파일은 메시징 세션 키(`agent:main:<platform>:...`)를 활성 세션 ID에 매핑합니다.
이 파일에는 게이트웨이/메시징 항목만 포함되므로, 메시징 플랫폼을 실행하면 해당 항목만 보입니다(예: `agent:main:whatsapp:dm:...`).

이는 **정상적인 동작**이며 CLI 세션이 누락되었다는 뜻이 아닙니다. `hermes sessions list`, `/sessions`, 대시보드는 모두 모든 세션(CLI, TUI, 게이트웨이)을 포함하는 `state.db`를 읽습니다. `~/.hermes/sessions/saved/*.json` 아래의 `/save` 스냅샷은 편의상 만든 내보내기 파일이지 인덱스가 아닙니다.

CLI 세션이 실제로 `hermes sessions list`에 나타나지 않는다면 원인은 해당 실행 중 `state.db`에 세션이 기록되지 않았기 때문입니다. `hermes sessions repair`를 실행하고 CLI 시작 시 `⚠ Session store unavailable` 경고가 나타나는지 확인하세요. 이 경고는 해당 실행에서 SQLite 지속성이 실패했다는 뜻입니다.
:::

:::note 레거시 JSONL 대화 기록
`state.db`가 정식 저장소가 되기 전에 생성된 세션에는 `~/.hermes/sessions/` 아래에 남은 `*.jsonl` 파일이 있을 수 있습니다. Hermes는 더 이상 이 파일을 쓰거나 읽지 않습니다. 대응하는 세션이 `state.db`에 있는지 확인한 후 안전하게 삭제할 수 있습니다.
:::

### 데이터베이스 스키마

`state.db`의 주요 테이블:

- **sessions** — 세션 메타데이터(id, source, user_id, model, title, 타임스탬프, 토큰 수). 제목에는 고유 인덱스가 있습니다(NULL 제목은 허용되며, NULL이 아닌 제목만 고유해야 함).
- **messages** — 전체 메시지 기록(role, content, tool_calls, tool_name, token_count)
- **messages_fts** — 메시지 콘텐츠 전문 검색을 위한 FTS5 가상 테이블

## 세션 만료 및 정리

### 자동 정리

- 게이트웨이 세션은 구성된 초기화 정책에 따라 자동 초기화됩니다.
- 초기화 전에 에이전트가 만료되는 세션의 메모리와 스킬을 저장합니다.
- 선택적 자동 정리: `sessions.auto_prune`이 `true`이면 CLI/게이트웨이 시작 시 `sessions.retention_days`(기본값 90) 동안 비활성 상태인 종료 세션을 정리합니다.
- 실제로 행을 삭제한 정리 후에는 마지막 성공적인 `VACUUM` 이후 최소 `sessions.min_vacuum_interval_days`(기본값 30)가 지났을 때 디스크 공간을 회수하기 위해 `state.db`를 `VACUUM`합니다(SQLite는 일반 DELETE만으로 파일 크기를 줄이지 않음).
- 정리는 `sessions.min_interval_hours`(기본값 24)마다 최대 한 번 실행됩니다. 마지막 실행 타임스탬프는 `state.db` 자체에 기록되므로 동일한 `HERMES_HOME`의 모든 Hermes 프로세스가 공유합니다.

기본값은 **꺼짐**입니다. 세션 기록은 `session_search` 회상에 유용하며, 조용히 삭제하면 사용자를 놀라게 할 수 있기 때문입니다. `~/.hermes/config.yaml`에서 활성화하세요.

```yaml
sessions:
  auto_prune: true          # opt in — default is false
  retention_days: 90        # keep ended sessions active within this window
  vacuum_after_prune: true  # reclaim disk space after a pruning sweep
  min_vacuum_interval_days: 30 # don't rewrite the DB more often than this
  min_interval_hours: 24    # don't re-run the sweep more often than this
```

활성 세션은 기간과 관계없이 절대 자동 정리되지 않습니다. 종료 세션의 기간은 가장 최근 메시지부터 계산되므로, 오래 지속된 대화를 최근에 사용했다면 보존 기간이 시작된 시점이 오래되었다는 이유만으로 삭제되지 않습니다.

### 수동 정리

```bash
# Prune sessions older than 90 days
hermes sessions prune

# Delete a specific session
hermes sessions delete <session_id>

# Export before pruning (backup)
hermes sessions export backup.jsonl
hermes sessions prune --older-than 30 --yes
```

:::tip
데이터베이스는 느리게 커집니다(일반적으로 수백 세션에 10~15MB). 세션 기록은 과거 대화에 대한 `session_search` 회상을 지원하므로 자동 정리는 비활성화되어 제공됩니다. `state.db`가 성능에 실제로 영향을 줄 정도로 무거운 게이트웨이/cron 작업을 실행한다면 활성화하세요(관찰된 실패 사례: 세션 약 1000개에서 384MB의 state.db가 FTS5 삽입과 `/resume` 목록 표시를 느리게 함). 자동 스윕을 켜지 않고 일회성으로 정리하려면 `hermes sessions prune`을 사용하세요.
:::
