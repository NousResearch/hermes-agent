---
sidebar_position: 7
title: "세션 (Sessions)"
description: "세션 영속성(persistence), 재개, 검색, 관리 및 플랫폼별 세션 추적"
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# 세션 (Sessions)

Hermes Agent는 모든 대화를 자동으로 세션으로 저장합니다. 세션을 통해 대화 재개, 여러 세션에 걸친 검색 및 전체 대화 기록 관리가 가능합니다.

## 세션 작동 방식

CLI, Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Teams 또는 기타 어떤 메시징 플랫폼에서 온 대화이든 간에, 모든 대화는 전체 메시지 기록과 함께 세션으로 저장됩니다. 세션은 다음 위치에서 추적됩니다:

1. **SQLite 데이터베이스** (`~/.hermes/state.db`) — FTS5 전체 텍스트 검색을 지원하는 구조화된 세션 메타데이터와 전체 메시지 기록

SQLite 데이터베이스 저장 항목:
- 세션 ID, 소스 플랫폼, 사용자 ID
- **세션 제목** (고유하고 사람이 읽을 수 있는 이름)
- 모델 이름 및 구성
- 시스템 프롬프트 스냅샷
- 전체 메시지 기록 (역할, 내용, 도구 호출, 도구 결과)
- 토큰 수 (입력/출력)
- 타임스탬프 (started_at, ended_at)
- 상위 세션 ID (압축에 의해 트리거된 세션 분할용)

### 컨텍스트에 포함되는 것

Hermes는 대화를 재개하기 위해 세션 기록을 저장하지만, 자신이 처리했던 모든 바이트를 계속해서 다시 전송하지는 않습니다. 매 턴마다 모델은 선택된 시스템 프롬프트, 현재 대화 창, 그리고 Hermes가 해당 턴에 명시적으로 주입한 콘텐츠를 봅니다.

미디어 첨부 파일은 턴 범위의(turn-scoped) 입력으로 처리됩니다:

- 이미지는 다음 모델 호출에 네이티브로 첨부되거나, 활성 모델이 네이티브 비전을 지원하지 않는 경우 텍스트 설명으로 사전 분석될 수 있습니다.
- 오디오는 음성-텍스트 변환(speech-to-text)이 구성된 경우 텍스트로 기록됩니다.
- 텍스트 문서의 경우 추출된 텍스트가 포함될 수 있습니다. 다른 문서 유형은 일반적으로 저장된 로컬 경로와 짧은 메모로 표시됩니다.
- 첨부 파일 경로와 추출/파생된 텍스트는 대화 기록에 나타날 수 있지만, 원본 이미지, 오디오 또는 바이너리 파일 바이트가 미래의 프롬프트에 반복적으로 복사되지는 않습니다.

예를 들어, 사용자가 이미지를 보내고 Hermes에게 밈(meme)을 만들라고 요청하면, Hermes는 비전 모델을 사용하여 이미지를 한 번 검사하고 이미지 처리 스크립트를 실행할 수 있습니다. 미래의 턴은 컨텍스트에서 원본 JPEG를 자동으로 가지고 다니지 않습니다. 이들은 사용자의 요청, 짧은 이미지 설명, 로컬 캐시 경로 또는 최종 어시스턴트 응답과 같이 대화에 쓰여진 내용만 포함합니다.

컨텍스트 증가의 가장 일반적인 원인은 미디어 파일 자체가 아닙니다. 장황한 텍스트가 그 원인입니다: 붙여넣은 대화 내용, 전체 로그, 큰 도구 출력, 긴 차이점(diff), 반복되는 상태 보고서 및 자세한 증명(proof) 덤프. 커다란 결과물을 채팅에 복사하는 대신 요약, 파일 경로, 초점을 맞춘 발췌문 및 도구 기반 조회를 우선시하세요.

:::tip
세션이 길어지면 `/compress`를, 새로운 스레드에는 `/new`를 사용하고, 예전의 종료된 세션을 저장소에서 삭제하려는 경우에만 `hermes sessions prune`을 사용하세요. 압축은 활성 컨텍스트를 줄이지만, 개인정보 보호를 위한 삭제는 아닙니다.
새 세션의 초기 제목을 미리 설정하려면 `/new`에 이름을 전달하세요(예: `/new payments-refactor`). 이는 나중에 `/resume <name>`을 사용하거나 `/sessions` 피커에서 찾을 때 유용합니다.
:::

### 세션 출처 (Session Sources)

각 세션에는 해당 소스 플랫폼이 태그되어 있습니다:

| 소스 | 설명 |
|--------|-------------|
| `cli` | 대화형 CLI (`hermes` 또는 `hermes chat`) |
| `telegram` | Telegram 메신저 |
| `discord` | Discord 서버/DM |
| `slack` | Slack 작업 공간(workspace) |
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
| `cron` | 예약된 크론 작업 |
| `batch` | 일괄(Batch) 처리 실행 |

## CLI 세션 재개

`--continue` 또는 `--resume`을 사용하여 CLI에서 이전 대화를 재개합니다:

### 마지막 세션 계속하기

```bash
# 가장 최근의 CLI 세션 재개
hermes --continue
hermes -c

# 또는 chat 하위 명령어와 함께
hermes chat --continue
hermes chat -c
```

이렇게 하면 SQLite 데이터베이스에서 가장 최근의 `cli` 세션을 찾고 전체 대화 기록을 로드합니다.

### 이름으로 재개

세션에 제목을 지정한 경우(아래의 [세션 이름 지정](#세션-이름-지정-session-naming) 참조) 이름으로 재개할 수 있습니다:

```bash
# 이름이 지정된 세션 재개
hermes -c "my project"

# 리니지 변형(lineage variants)이 있는 경우 (my project, my project #2, my project #3),
# 자동으로 가장 최근의 세션을 재개합니다.
hermes -c "my project"   # → "my project #3" 재개
```

### 특정 세션 재개

```bash
# ID로 특정 세션 재개
hermes --resume 20250305_091523_a1b2c3d4
hermes -r 20250305_091523_a1b2c3d4

# 제목으로 재개
hermes --resume "refactoring auth"

# 또는 chat 하위 명령어와 함께
hermes chat --resume 20250305_091523_a1b2c3d4
```

세션 ID는 CLI 세션을 종료할 때 표시되며 `hermes sessions list`로 찾을 수 있습니다.

### 재개 시 대화 요약 (Recap)

세션을 재개할 때 Hermes는 입력 프롬프트가 나타나기 전에 스타일이 지정된 패널에 이전 대화에 대한 간결한 요약을 표시합니다:

<img className="docs-terminal-figure" src={useBaseUrl('/img/docs/session-recap.svg')} alt="Hermes 세션을 재개할 때 표시되는 이전 대화(Previous Conversation) 요약 패널의 시각화 프리뷰" />
<p className="docs-figure-caption">재개(Resume) 모드에서는 활성 프롬프트로 돌아가기 전에 최근 사용자 및 어시스턴트 턴이 포함된 간결한 요약 패널을 표시합니다.</p>

요약(recap)의 특징:
- **사용자 메시지**(금색 `●`) 및 **어시스턴트 응답**(녹색 `◆`) 표시
- 긴 메시지 **자르기** (사용자의 경우 300자, 어시스턴트의 경우 200자 / 3줄)
- **도구 호출 축소** 도구 이름과 개수로 축소 표시 (예: `[3 tool calls: terminal, web_search]`)
- 시스템 메시지, 도구 결과 및 내부 추론 **숨기기**
- "... N earlier messages ..." 표시기를 사용하여 최근 10번의 대화로 **제한**
- 활성 대화와 구별하기 위해 **흐릿한(dim) 스타일** 사용

요약 기능을 비활성화하고 최소한의 한 줄 동작을 유지하려면 `~/.hermes/config.yaml`에 다음과 같이 설정하세요:

```yaml
display:
  resume_display: minimal   # 기본값: full
```

:::tip
세션 ID는 `YYYYMMDD_HHMMSS_<hex>` 형식을 따릅니다 — CLI/TUI 세션은 6자리 접미사(예: `20250305_091523_a1b2c3`)를 사용하고 게이트웨이 세션은 8자리 접미사(예: `20250305_091523_a1b2c3d4`)를 사용합니다. ID(전체 또는 고유 접두사) 또는 제목으로 재개할 수 있으며 두 가지 모두 `-c` 및 `-r`과 작동합니다.
:::

## 플랫폼 간 전환 (Cross-Platform Handoff)

CLI 세션에서 `/handoff <platform>`을 사용하여 현재 진행 중인 대화를 메시징 플랫폼의 홈 채널로 전송할 수 있습니다. 에이전트는 동일한 세션 ID, 모든 역할이 인식된 대화 스크립트, 도구 호출 등 CLI가 중단한 바로 그 지점에서 정확히 다시 시작합니다.

```bash
# CLI 세션 내부
/handoff telegram
```

발생하는 일:

1. CLI는 `<platform>`이 활성화되어 있고 홈 채널이 설정되어 있는지 검증합니다(설정하려면 대상 채팅에서 `/sethome`을 한 번 실행하세요).
2. CLI는 세션을 보류(pending) 상태로 표시하고 **게이트웨이를 블록-폴링(block-polls)합니다.** 에이전트가 턴(turn)을 진행 중인 경우 거부합니다 — 먼저 현재 응답이 완료될 때까지 기다리세요.
3. 게이트웨이 감시자(watcher)가 전환을 청구하고 대상 어댑터에 새로운 스레드를 요청합니다:
   - **Telegram** — 새로운 포럼 주제를 엽니다(채팅에서 Bot API 9.4+ 토픽 모드가 활성화된 경우 DM 토픽, 또는 포럼 슈퍼그룹 토픽).
   - **Discord** — 홈 텍스트 채널 아래에 1440분 자동 보관 스레드를 만듭니다.
   - **Slack** — 시드 메시지를 게시하고 해당 `ts`를 스레드 앵커로 사용합니다.
   - **WhatsApp / Signal / Matrix / SMS** — 네이티브 스레드가 없으므로 홈 채널로 직접 폴백(fallback)합니다.
4. 게이트웨이는 대상 키를 기존 CLI 세션 ID에 다시 바인딩한 다음, 에이전트에게 확인 및 요약을 요청하는 합성(synthetic) 사용자 턴을 만듭니다. 응답은 새로운 스레드에 도착합니다.
5. 게이트웨이가 성공을 확인하면 CLI는 `/resume` 힌트를 인쇄하고 깔끔하게 종료됩니다:

   ```
   ↻ Handoff complete. The session is now active on telegram.
     Resume it on this CLI later with: /resume my-session-title
   ```

6. 이 시점부터 대화는 플랫폼에서 유지됩니다. 새 스레드에 답장하세요. 해당 채널에서 권한이 부여된 모든 사람은 동일한 세션을 공유하며, 스레드 세션은 `user_id` 없이 키를 생성하기 때문에 스레드의 후속 사용자 메시지는 매끄럽게 연결됩니다.

**CLI로 다시 전환하기(Resume):** 데스크톱으로 돌아오고 싶을 때 `/resume <title>`(또는 쉘에서 `hermes -r "<title>"`)을 실행하기만 하면 플랫폼에서 중단된 위치부터 다시 시작합니다.

**실패 모드(Failure modes):**
- 홈 채널이 구성되지 않음 → CLI가 `/sethome` 힌트와 함께 거부합니다.
- 플랫폼이 활성화되지 않음 / 게이트웨이가 실행 중이 아님 → CLI는 60초 후에 명확한 메시지와 함께 시간 초과되며 CLI 세션은 그대로 유지됩니다.
- 스레드 생성 실패(권한, 토픽 모드 꺼짐) → 홈 채널로 직접 폴백하며 여전히 완료됩니다; 스레드 격리는 없지만 전환 자체는 작동합니다.
- `adapter.send` 실패(속도 제한, 일시적인 API 오류) → 전환이 이유와 함께 실패로 표시됩니다; 행이 지워지므로 다시 시도할 수 있습니다.

**알아두면 좋은 제한 사항:** 다중 사용자 그룹 홈 채널이 있는, 스레드를 지원하지 않는 플랫폼의 경우, 합성 턴은 DM 스타일 세션으로 키를 생성합니다. 이는 본인과의 DM 홈 채널(일반적인 설정)에서는 작동하지만, 진정으로 공유되는 그룹 채팅에는 이상적이지 않습니다. 스레딩은 Telegram / Discord / Slack (압도적으로 흔한 경우)을 포괄하므로 대부분의 설정은 이 문제에 직면하지 않습니다.

## 세션 이름 지정 (Session Naming)

쉽게 찾고 재개할 수 있도록 세션에 사람이 읽을 수 있는 제목을 지정하세요.

### 자동 생성된 제목

Hermes는 첫 번째 교환 후 각 세션에 대해 짧은 설명 제목(3~7개 단어)을 자동으로 생성합니다. 이것은 빠른 보조 모델을 사용하여 백그라운드 스레드에서 실행되므로 지연 시간을 늘리지 않습니다. `hermes sessions list` 또는 `hermes sessions browse`로 세션을 찾아볼 때 자동 생성된 제목이 표시됩니다.

자동 제목 지정은 세션당 한 번만 실행되며 수동으로 제목을 이미 설정한 경우 건너뜁니다.

### 제목 수동으로 설정

(CLI 또는 게이트웨이의) 채팅 세션 내에서 `/title` 슬래시 명령어를 사용하세요:

```
/title my research project
```

제목이 즉시 적용됩니다. 아직 데이터베이스에 세션이 생성되지 않은 경우(예: 첫 번째 메시지를 보내기 전에 `/title`을 실행하는 경우) 대기열에 추가되었다가 세션이 시작되면 적용됩니다.

명령줄에서 기존 세션의 이름을 바꿀 수도 있습니다:

```bash
hermes sessions rename 20250305_091523_a1b2c3d4 "refactoring auth module"
```

### 제목 규칙

- **고유성** — 두 세션이 같은 제목을 공유할 수 없습니다.
- **최대 100자** — 목록 출력을 깔끔하게 유지합니다.
- **삭제(Sanitized)** — 제어 문자, 폭 없는 공백 문자, RTL 오버라이드 문자는 자동으로 제거됩니다.
- **일반 유니코드 사용 가능** — 이모지, CJK(한중일 문자), 악센트가 있는 문자가 모두 작동합니다.

### 압축 시 자동 리니지 (Auto-Lineage on Compression)

세션의 컨텍스트가 압축되면(수동으로 `/compress`를 통하거나 자동으로) Hermes는 새로운 연속(continuation) 세션을 만듭니다. 원본에 제목이 있었다면 새 세션에는 자동으로 번호가 매겨진 제목이 부여됩니다:

```
"my project" → "my project #2" → "my project #3"
```

이름으로 재개(`hermes -c "my project"`)할 때 리니지(lineage)에서 가장 최근 세션을 자동으로 선택합니다.

### 메시징 플랫폼에서의 /title

`/title` 명령어는 모든 게이트웨이 플랫폼(Telegram, Discord, Slack, WhatsApp)에서 작동합니다:

- `/title My Research` — 세션 제목 설정
- `/title` — 현재 제목 표시

## 세션 관리 명령

Hermes는 `hermes sessions`를 통해 전체 세션 관리 명령 세트를 제공합니다:

### 세션 목록 보기

```bash
# 최근 세션 목록 표시 (기본값: 최근 20개)
hermes sessions list

# 플랫폼별 필터링
hermes sessions list --source telegram

# 더 많은 세션 표시
hermes sessions list --limit 50
```

세션에 제목이 있는 경우 출력에는 제목, 미리보기 및 상대적 타임스탬프가 표시됩니다:

```
Title                  Preview                                  Last Active   ID
────────────────────────────────────────────────────────────────────────────────────────────────
refactoring auth       Help me refactor the auth module please   2h ago        20250305_091523_a
my project #3          Can you check the test failures?          yesterday     20250304_143022_e
—                      What's the weather in Las Vegas?          3d ago        20250303_101500_f
```

어떤 세션도 제목이 없는 경우 더 간단한 형식이 사용됩니다:

```
Preview                                            Last Active   Src    ID
──────────────────────────────────────────────────────────────────────────────────────
Help me refactor the auth module please             2h ago        cli    20250305_091523_a
What's the weather in Las Vegas?                    3d ago        tele   20250303_101500_f
```

### 세션 내보내기

```bash
# 모든 세션을 JSONL 파일로 내보내기
hermes sessions export backup.jsonl

# 특정 플랫폼의 세션 내보내기
hermes sessions export telegram-history.jsonl --source telegram

# 단일 세션 내보내기
hermes sessions export session.jsonl --session-id 20250305_091523_a1b2c3d4
```

내보낸 파일은 한 줄당 하나의 JSON 객체를 포함하며 전체 세션 메타데이터와 모든 메시지가 포함되어 있습니다.

### 세션 삭제하기

```bash
# 특정 세션 삭제 (확인 필요)
hermes sessions delete 20250305_091523_a1b2c3d4

# 확인 없이 삭제
hermes sessions delete 20250305_091523_a1b2c3d4 --yes
```

### 세션 이름 바꾸기

```bash
# 세션의 제목 설정 또는 변경
hermes sessions rename 20250305_091523_a1b2c3d4 "debugging auth flow"

# 여러 단어로 된 제목은 CLI에서 따옴표가 필요하지 않습니다.
hermes sessions rename 20250305_091523_a1b2c3d4 debugging auth flow
```

다른 세션에서 제목을 이미 사용하고 있으면 오류가 표시됩니다.

### 오래된 세션 정리하기 (Prune)

```bash
# 90일이 지난 종료된 세션 삭제 (기본값)
hermes sessions prune

# 사용자 정의 연령 임계값
hermes sessions prune --older-than 30

# 특정 플랫폼에서만 세션 정리
hermes sessions prune --source telegram --older-than 60

# 확인 건너뛰기
hermes sessions prune --older-than 30 --yes
```

:::info
Prune(정리)는 **종료된** 세션(명시적으로 종료되었거나 자동으로 재설정된 세션)만 삭제합니다. 활성 세션은 절대로 정리되지 않습니다.
:::

### 세션 통계

```bash
hermes sessions stats
```

출력 예시:

```
Total sessions: 142
Total messages: 3847
  cli: 89 sessions
  telegram: 38 sessions
  discord: 15 sessions
Database size: 12.4 MB
```

토큰 사용량, 비용 추정, 도구 분석 및 활동 패턴 등 더 깊은 분석을 보려면 [`hermes insights`](/reference/cli-commands#hermes-insights)를 사용하세요.

## 세션 검색 도구 (Session Search Tool)

에이전트에는 SQLite의 FTS5 엔진을 사용하여 과거의 모든 대화에 걸쳐 전체 텍스트 검색을 수행하는 내장형 `session_search` 도구가 있으며, 이를 통해 에이전트가 찾은 모든 세션을 스크롤할 수 있습니다. LLM 호출, 요약, 자르기(truncation)가 필요 없습니다. 모든 형태의 요청은 DB에서 실제 메시지를 반환합니다.

### 세 가지 호출 형태

이 도구는 사용자가 설정한 인수를 기반으로 사용자가 원하는 것을 유추합니다. `mode` 매개변수는 없습니다.

**1. 검색 (Discovery) — `query` 전달:**

```python
session_search(query="auth refactor", limit=3)
```

FTS5를 실행하고, 세션 리니지별로 중복(hit)을 제거하며, 상위 N개 세션을 반환합니다. 각 결과는 다음을 전달합니다:

- `session_id`, `title`, `when`, `source`
- `snippet` — FTS5 하이라이트된 일치 부분 발췌
- `bookend_start` — 세션의 처음 3개 사용자+어시스턴트 메시지 (목표/시작)
- `messages` — FTS5 일치 전후의 ±5개 메시지 (문맥 안에서의 발견인 앵커 메시지 표시)
- `bookend_end` — 세션의 마지막 3개 사용자+어시스턴트 메시지 (해결/결정)
- `match_message_id`, `messages_before`, `messages_after`

북엔드(bookends)와 윈도우(window)를 결합하여 전체 스크립트 비용을 지불하지 않고도 '목표 → 일치 → 해결'을 재구성합니다. 일반적인 실행 시간: 실제 세션 DB에서 15–50ms.

**2. 스크롤 (Scroll) — `session_id` + `around_message_id` 전달:**

```python
session_search(session_id="20260510_174648_805cc2", around_message_id=590803, window=10)
```

앵커를 중심으로 ±`window`개의 메시지 창을 반환합니다. FTS5도 북엔드도 없습니다 — 그저 분할된 조각(slice)일 뿐입니다. ±5 기본 창보다 더 많은 컨텍스트가 필요한 경우, 검색(discovery) 호출 후 사용하세요.

- **앞으로** 스크롤하려면: `messages[-1].id`를 `around_message_id`로 다시 전달합니다.
- **뒤로** 스크롤하려면: `messages[0].id`를 `around_message_id`로 다시 전달합니다.
- 경계(boundary) 메시지는 위치 마커로서 양쪽 창 모두에 나타납니다.
- `messages_before` 또는 `messages_after`가 `window`보다 작으면 세션의 시작이나 끝에 있는 것입니다.

일반적인 실행 시간: 스크롤 호출당 1–2ms.

**3. 탐색 (Browse) — 인수 없음:**

```python
session_search()
```

최근 세션을 시간순으로 반환합니다(제목, 미리보기, 타임스탬프). 사용자가 주제를 지정하지 않고 "내가 무슨 작업을 하고 있었지"라고 물을 때 유용합니다.

### FTS5 쿼리 구문

키워드 모드는 표준 FTS5 쿼리 구문을 지원합니다:

- 단순 키워드: `docker deployment` (FTS5는 기본적으로 AND 조건입니다)
- 구문(Phrase): `"exact phrase"`
- 불리언(Boolean): `docker OR kubernetes`, `python NOT java`
- 접두사: `deploy*`

### 선택적 매개변수

- `sort` — FTS5 순위에 더해 `newest` (최신순) 또는 `oldest` (오래된 순) 정렬. 관련성만을 기준으로 정렬하려면 생략하세요(기본값; 탐색적 회상에 적합). "X를 어디까지 했지" 같은 질문에는 `newest`를 사용하고, "X가 어떻게 시작되었지" 같은 질문에는 `oldest`를 사용하세요.
- `role_filter` — 포함할 역할을 쉼표로 구분하여 전달. 검색(Discovery) 기본값은 `user,assistant`입니다 (도구 출력은 대개 노이즈입니다). 도구 동작을 디버깅하기 위해 도구 출력을 포함하려면 `user,assistant,tool`을 전달하거나, 도구 출력만 검색하려면 `tool`을 전달하세요.

### 언제 사용되나요?

에이전트는 자동으로 세션 검색을 사용하도록 프롬프트됩니다:

> *"사용자가 과거 대화의 무언가를 참조하거나 관련 이전 컨텍스트가 존재한다고 의심되는 경우, 사용자에게 다시 말해 달라고 요청하기 전에 session_search를 사용하여 이를 회상하세요."*

일반적인 트리거: "전에 이걸 했었지", "기억나니", "지난번에", "내가 언급했듯이", 또는 현재 창에 없는 프로젝트/사람/개념에 대한 모든 참조.

## 플랫폼별 세션 추적

### 게이트웨이 세션

메시징 플랫폼에서 세션은 메시지 소스에서 빌드된 결정론적(deterministic) 세션 키에 의해 키가 생성됩니다:

| 채팅 유형 | 기본 키 형식 | 동작 |
|-----------|--------------------|----------|
| Telegram DM | `agent:main:telegram:dm:<chat_id>` | DM 채팅당 하나의 세션 |
| Discord DM | `agent:main:discord:dm:<chat_id>` | DM 채팅당 하나의 세션 |
| WhatsApp DM | `agent:main:whatsapp:dm:<canonical_identifier>` | DM 사용자당 하나의 세션 (매핑이 존재할 때 LID/전화번호 별칭은 하나의 식별자로 병합됨) |
| 그룹 채팅 | `agent:main:<platform>:group:<chat_id>:<user_id>` | 플랫폼이 사용자 ID를 노출할 때 그룹 내 사용자당 |
| 그룹 스레드/토픽 | `agent:main:<platform>:group:<chat_id>:<thread_id>` | 모든 스레드 참가자를 위한 공유 세션(기본값). `thread_sessions_per_user: true`일 때는 사용자별. |
| 채널 | `agent:main:<platform>:channel:<chat_id>:<user_id>` | 플랫폼이 사용자 ID를 노출할 때 채널 내 사용자당 |

Hermes가 공유 채팅의 참가자 식별자를 가져올 수 없는 경우, 해당 방에 대해 공유된 세션 1개로 폴백합니다.

### 공유 그룹 세션 vs 격리된 그룹 세션

기본적으로 Hermes는 `config.yaml`에서 `group_sessions_per_user: true`를 사용합니다. 이는 다음을 의미합니다:

- Alice와 Bob은 기록(transcript)을 공유하지 않고 동일한 Discord 채널에서 Hermes와 이야기할 수 있습니다.
- 도구를 많이 사용하는 한 사용자의 긴 작업이 다른 사용자의 컨텍스트 창을 오염시키지 않습니다.
- 실행 중인 에이전트 키가 격리된 세션 키와 일치하기 때문에 인터럽트(interrupt) 처리도 사용자별로 유지됩니다.

대신 하나의 공유된 "채팅방 뇌(room brain)"를 원한다면 다음을 설정하세요:

```yaml
group_sessions_per_user: false
```

그러면 그룹/채널이 방마다 하나의 단일 공유 세션으로 되돌아갑니다. 이는 공유된 대화 컨텍스트를 보존하지만 토큰 비용, 인터럽트 상태 및 컨텍스트 증가도 함께 공유하게 됩니다.

### 세션 재설정 정책 (Session Reset Policies)

게이트웨이 세션은 구성 가능한 정책에 따라 자동으로 재설정됩니다:

- **idle (유휴)** — N분의 비활동 후 재설정
- **daily (매일)** — 매일 특정 시간에 재설정
- **both (둘 다)** — (유휴 또는 매일 중) 먼저 도달하는 조건에서 재설정
- **none (없음)** — 자동 재설정 안 함

세션이 자동 재설정되기 전에, 에이전트는 대화에서 중요한 메모리나 스킬을 저장할 턴(turn)을 갖게 됩니다.

**활성 백그라운드 프로세스**가 있는 세션은 정책에 관계없이 절대 자동 재설정되지 않습니다.

## 저장소 위치

| 항목 | 경로 | 설명 |
|------|------|-------------|
| SQLite 데이터베이스 | `~/.hermes/state.db` | 모든 세션 메타데이터 + FTS5가 있는 메시지 |
| 게이트웨이 메시지    | `~/.hermes/state.db`   | SQLite — 모든 세션 메시지에 대한 표준(canonical) 저장소 |
| 게이트웨이 라우팅 인덱스 | `~/.hermes/sessions/sessions.json` | 세션 키를 활성 세션 ID(출처 메타데이터, 만료 플래그)에 매핑 |

SQLite 데이터베이스는 동시 읽기 작업과 단일 쓰기 작업에 WAL 모드를 사용하며, 이는 게이트웨이의 다중 플랫폼 아키텍처에 매우 적합합니다.

:::note 과거의 JSONL 기록 (Legacy JSONL transcripts)
state.db가 표준(canonical) 저장소가 되기 전에 생성된 세션은 `~/.hermes/sessions/`에 남겨진 `*.jsonl` 파일이 있을 수 있습니다. 이것들은 더 이상 Hermes에 의해 기록되거나 읽히지 않습니다. state.db에 해당 세션이 존재하는지 확인한 후 삭제해도 안전합니다.
:::

### 데이터베이스 스키마

`state.db`의 주요 테이블:

- **sessions** — 세션 메타데이터 (id, source, user_id, model, title, timestamps, token counts). 제목에는 고유(unique) 인덱스가 있습니다 (NULL 제목은 허용되며, 비어 있지 않은 제목만 고유해야 함).
- **messages** — 전체 메시지 기록 (role, content, tool_calls, tool_name, token_count)
- **messages_fts** — 메시지 내용 전반의 전체 텍스트 검색을 위한 FTS5 가상 테이블

## 세션 만료 및 정리 (Session Expiry and Cleanup)

### 자동 정리 (Automatic Cleanup)

- 게이트웨이 세션은 구성된 재설정 정책에 따라 자동으로 재설정됩니다.
- 재설정하기 전, 에이전트는 만료되는 세션에서 메모리와 스킬을 저장합니다.
- 옵트인(Opt-in) 자동 정리: `sessions.auto_prune`이 `true`인 경우 CLI/게이트웨이 시작 시 `sessions.retention_days` (기본값 90)보다 오래된 종료된 세션이 정리됩니다.
- 실제로 행이 제거된 정리 후에는 디스크 공간을 회수하기 위해 `state.db`에 `VACUUM`이 수행됩니다 (SQLite는 단순한 DELETE로는 파일 크기를 줄이지 않음).
- 정리는 `sessions.min_interval_hours` (기본값 24)마다 최대 한 번 실행됩니다; 마지막 실행 타임스탬프는 `state.db` 자체 내부에서 추적되므로 동일한 `HERMES_HOME`에 있는 모든 Hermes 프로세스에서 공유됩니다.

기본값은 **꺼짐(off)**입니다 — 세션 기록은 `session_search`의 회상(recall)에 귀중하며 조용히 삭제하면 사용자가 놀랄 수 있습니다. `~/.hermes/config.yaml`에서 활성화할 수 있습니다:

```yaml
sessions:
  auto_prune: true          # 옵트인 — 기본값은 false입니다.
  retention_days: 90        # 이 날짜 동안 종료된 세션 유지
  vacuum_after_prune: true  # 정리 수행(sweep) 후 디스크 공간 확보
  min_interval_hours: 24    # 이 시간보다 더 자주 정리 수행을 실행하지 않음
```

활성 세션은 수명(age)에 관계없이 절대로 자동 정리되지 않습니다.

### 수동 정리 (Manual Cleanup)

```bash
# 90일이 지난 세션 정리
hermes sessions prune

# 특정 세션 삭제
hermes sessions delete <session_id>

# 정리 전 내보내기 (백업)
hermes sessions export backup.jsonl
hermes sessions prune --older-than 30 --yes
```

:::tip
데이터베이스는 천천히 증가하며 (일반적인 경우: 수백 개의 세션에 대해 10-15 MB), 세션 기록은 과거 대화의 `session_search` 회상을 구동하므로 자동 정리는 비활성화되어 제공됩니다. 무거운 게이트웨이/크론 워크로드를 실행하여 `state.db`가 성능에 의미 있는 영향을 미칠 경우에 활성화하세요 (관찰된 오류 유형: ~1000개의 세션이 있는 384 MB의 state.db가 FTS5 삽입 및 `/resume` 목록 작성을 늦춤). 자동 정리를 켜지 않고 일회성 정리를 수행하려면 `hermes sessions prune`을 사용하세요.
:::
