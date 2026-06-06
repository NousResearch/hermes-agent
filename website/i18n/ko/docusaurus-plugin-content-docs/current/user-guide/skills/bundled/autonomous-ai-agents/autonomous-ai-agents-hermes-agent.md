---
title: "Hermes Agent — Configure, extend, or contribute to Hermes Agent"
sidebar_label: "Hermes Agent"
description: "Configure, extend, or contribute to Hermes Agent"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Agent

Hermes Agent의 구성, 확장 및 기여 방법.

## 스킬 메타데이터

| | |
|---|---|
| Source | 번들 (기본 설치) |
| Path | `skills/autonomous-ai-agents/hermes-agent` |
| Version | `2.1.0` |
| Author | Hermes Agent + Teknium |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `hermes`, `setup`, `configuration`, `multi-agent`, `spawning`, `cli`, `gateway`, `development` |
| Related skills | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Hermes Agent

Hermes Agent는 터미널, 메시징 플랫폼, IDE에서 실행되는 Nous Research의 오픈 소스 AI 에이전트 프레임워크입니다. 이 에이전트는 도구 호출(tool calling)을 통해 시스템과 상호 작용하는 자율 코딩 및 작업 실행 에이전트인 Claude Code (Anthropic), Codex (OpenAI), OpenClaw와 같은 범주에 속합니다. Hermes는 모든 LLM 제공자(OpenRouter, Anthropic, OpenAI, DeepSeek, 로컬 모델 등 15개 이상)와 함께 작동하며 Linux, macOS, WSL에서 실행됩니다.

Hermes의 차별점:

- **스킬(Skills)을 통한 자기 개선** — Hermes는 재사용 가능한 절차를 스킬로 저장하여 경험으로부터 배웁니다. 복잡한 문제를 해결하거나, 워크플로우를 발견하거나, 교정받았을 때 그 지식을 스킬 문서로 유지하여 향후 세션에 로드할 수 있습니다. 스킬은 시간이 지남에 따라 축적되어 에이전트가 여러분의 특정 작업과 환경에 더 능숙해지도록 만듭니다.
- **세션을 넘나드는 영구 메모리** — 사용자가 누구인지, 취향은 어떤지, 환경 세부 사항, 그리고 배운 교훈을 기억합니다. 플러그형 메모리 백엔드(기본 내장, Honcho, Mem0 등)를 통해 메모리 작동 방식을 선택할 수 있습니다.
- **다중 플랫폼 게이트웨이** — 단순한 채팅이 아닌 전체 도구 접근 권한을 가지고 Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Email 등 10개 이상의 다른 플랫폼에서 동일한 에이전트를 실행할 수 있습니다.
- **제공자(Provider)에 종속되지 않음** — 다른 것은 변경하지 않고도 워크플로우 중간에 모델과 제공자를 교체할 수 있습니다. 자격 증명 풀(Credential pools)은 여러 API 키를 자동으로 순환시킵니다.
- **프로필(Profiles)** — 격리된 구성, 세션, 스킬 및 메모리를 갖춘 독립적인 Hermes 인스턴스를 여러 개 실행할 수 있습니다.
- **확장성** — 플러그인, MCP 서버, 커스텀 도구, 웹훅 트리거, cron 스케줄링 및 완전한 Python 생태계를 지원합니다.

사람들은 소프트웨어 개발, 연구, 시스템 관리, 데이터 분석, 콘텐츠 제작, 홈 오토메이션 및 영구적인 컨텍스트와 전체 시스템 접근 권한이 있는 AI 에이전트가 필요한 기타 모든 작업에 Hermes를 사용합니다.

**이 스킬은 Hermes Agent를 효과적으로 사용하는 데 도움을 줍니다** — 설치, 기능 구성, 추가 에이전트 인스턴스 생성, 문제 해결, 올바른 명령어 및 설정 찾기, 그리고 시스템을 확장하거나 기여해야 할 때 시스템의 작동 방식을 이해하도록 돕습니다.

**문서:** https://hermes-agent.nousresearch.com/docs/

## 빠른 시작

```bash
# 설치
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

# 대화형 채팅 (기본값)
hermes

# 단일 쿼리
hermes chat -q "What is the capital of France?"

# 설정 마법사
hermes setup

# 모델/제공자 변경
hermes model

# 상태 점검(Health check)
hermes doctor
```

---

## CLI 참조 가이드

### 전역 플래그 (Global Flags)

```
hermes [flags] [command]

  --version, -V             버전 표시
  --resume, -r SESSION      ID 또는 제목으로 세션 재개
  --continue, -c [NAME]     이름으로 또는 가장 최근 세션 재개
  --worktree, -w            격리된 git worktree 모드 (병렬 에이전트)
  --skills, -s SKILL        스킬 미리 로드 (쉼표로 구분하거나 반복)
  --profile, -p NAME        이름이 지정된 프로필 사용
  --yolo                    위험한 명령어 승인 단계 건너뛰기
  --pass-session-id         시스템 프롬프트에 세션 ID 포함
```

하위 명령어가 없으면 기본적으로 `chat`으로 실행됩니다.

### 채팅 (Chat)

```
hermes chat [flags]
  -q, --query TEXT          비대화형 단일 쿼리
  -m, --model MODEL         모델 (예: anthropic/claude-sonnet-4)
  -t, --toolsets LIST       쉼표로 구분된 도구 세트
  --provider PROVIDER       특정 제공자 강제 지정 (openrouter, anthropic, nous 등)
  -v, --verbose             상세 출력
  -Q, --quiet               배너, 스피너, 도구 미리보기 숨김
  --checkpoints             파일 시스템 체크포인트 활성화 (/rollback 용)
  --source TAG              세션 소스 태그 (기본값: cli)
```

### 구성 (Configuration)

```
hermes setup [section]      대화형 마법사 (model|terminal|gateway|tools|agent)
hermes model                대화형 모델/제공자 선택기
hermes config               현재 설정 보기
hermes config edit          $EDITOR에서 config.yaml 열기
hermes config set KEY VAL   설정값 지정
hermes config path          config.yaml 경로 출력
hermes config env-path      .env 경로 출력
hermes config check         누락되거나 오래된 구성 확인
hermes config migrate       새 옵션으로 구성 업데이트
hermes auth                 대화형 자격 증명 관리자
hermes auth add PROVIDER    OAuth 또는 API 키 자격 증명 추가 (예: nous, openai-codex, qwen-oauth)
hermes auth list            저장된 자격 증명 목록
hermes auth remove PROVIDER 저장된 자격 증명 제거
hermes doctor [--fix]       종속성 및 구성 확인
hermes status [--all]       컴포넌트 상태 표시
```

### 도구 & 스킬 (Tools & Skills)

```
hermes tools                대화형 도구 활성화/비활성화 (curses UI)
hermes tools list           모든 도구 및 상태 표시
hermes tools enable NAME    도구 세트 활성화
hermes tools disable NAME   도구 세트 비활성화

hermes skills list          설치된 스킬 목록
hermes skills search QUERY  스킬 허브(hub) 검색
hermes skills install ID    스킬 설치 (ID는 허브 식별자 또는 직접적인 https://…/SKILL.md URL일 수 있음; 메타데이터에 이름이 없는 경우 --name을 전달하여 재정의 가능)
hermes skills inspect ID    설치 전 미리보기
hermes skills config        플랫폼별 스킬 활성화/비활성화
hermes skills check         업데이트 확인
hermes skills update        오래된 스킬 업데이트
hermes skills uninstall N   허브 스킬 제거
hermes skills publish PATH  레지스트리에 게시
hermes skills browse        사용 가능한 모든 스킬 찾아보기
hermes skills tap add REPO  스킬 소스로 GitHub 레포지토리 추가
```

### MCP 서버

```
hermes mcp serve            Hermes를 MCP 서버로 실행
hermes mcp add NAME         MCP 서버 추가 (--url 또는 --command)
hermes mcp remove NAME      MCP 서버 제거
hermes mcp list             구성된 서버 목록
hermes mcp test NAME        연결 테스트
hermes mcp configure NAME   도구 선택 토글
```

기본 내장 MCP 클라이언트가 어떻게 서버(stdio/HTTP)를 연결하고, 자동으로 도구를 발견하고, 일급(first-class) 도구로 노출시키는지, 그리고 카탈로그 설치(`hermes mcp install <name>`)에 대한 내용은 `skill_view(name="hermes-agent", file_path="references/native-mcp.md")`를 참고하세요.

### 게이트웨이 (메시징 플랫폼)

```
hermes gateway run          게이트웨이를 포그라운드에서 시작
hermes gateway install      백그라운드 서비스로 설치
hermes gateway start/stop   서비스 제어
hermes gateway restart      서비스 재시작
hermes gateway status       상태 확인
hermes gateway setup        플랫폼 구성
```

지원 플랫폼: Telegram, Discord, Slack, WhatsApp, Signal, Email, SMS, Matrix, Mattermost, Home Assistant, DingTalk, Feishu, WeCom, BlueBubbles (iMessage), Weixin (WeChat), API Server, Webhooks. Open WebUI는 API Server 어댑터를 통해 연결합니다.

플랫폼 문서: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/

### 세션 (Sessions)

```
hermes sessions list        최근 세션 목록
hermes sessions browse      대화형 선택기
hermes sessions export OUT  JSONL로 내보내기
hermes sessions rename ID T 세션 이름 변경
hermes sessions delete ID   세션 삭제
hermes sessions prune       오래된 세션 정리 (--older-than N days)
hermes sessions stats       세션 저장소 통계
```

### 크론 작업 (Cron Jobs)

```
hermes cron list            작업 목록 (--all 사용 시 비활성화된 것도 표시)
hermes cron create SCHED    생성: '30m', 'every 2h', '0 9 * * *'
hermes cron edit ID         일정, 프롬프트, 전달 방법 편집
hermes cron pause/resume ID 작업 상태 제어
hermes cron run ID          다음 주기에 트리거
hermes cron remove ID       작업 삭제
hermes cron status          스케줄러 상태
```

### 웹훅 (Webhooks)

```
hermes webhook subscribe N  /webhooks/<name> 경로 생성
hermes webhook list         구독 목록
hermes webhook remove NAME  구독 제거
hermes webhook test NAME    테스트 POST 전송
```

전체 설정, 라우팅 구성, 페이로드 템플릿화, 이벤트 기반 에이전트 실행 패턴에 대해서는 `skill_view(name="hermes-agent", file_path="references/webhooks.md")`를 참조하세요.

### 프로필 (Profiles)

```
hermes profile list         모든 프로필 목록
hermes profile create NAME  생성 (--clone, --clone-all, --clone-from)
hermes profile use NAME     기본 프로필로 고정
hermes profile delete NAME  프로필 삭제
hermes profile show NAME    세부 정보 표시
hermes profile alias NAME   래퍼(wrapper) 스크립트 관리
hermes profile rename A B   프로필 이름 변경
hermes profile export NAME  tar.gz로 내보내기
hermes profile import FILE  아카이브에서 가져오기
```

### 자격 증명 풀 (Credential Pools)

```
hermes auth add             대화형 자격 증명 마법사
hermes auth list [PROVIDER] 풀에 있는 자격 증명 목록
hermes auth remove P INDEX  제공자 + 인덱스로 제거
hermes auth reset PROVIDER  고갈(exhaustion) 상태 초기화
```

### 기타

```
hermes insights [--days N]  사용량 분석
hermes update               최신 버전으로 업데이트
hermes pairing list/approve/revoke  DM 인증 권한
hermes plugins list/install/remove  플러그인 관리
hermes honcho setup/status  Honcho 메모리 통합 (honcho 플러그인 필요)
hermes memory setup/status/off  메모리 제공자 구성
hermes completion bash|zsh  쉘 자동 완성
hermes acp                  ACP 서버 (IDE 통합)
hermes claw migrate         OpenClaw에서 마이그레이션
hermes uninstall            Hermes 제거
```

---

## 슬래시 명령어 (세션 중)

대화형 채팅 세션 중에 이 명령어들을 입력하세요. 새로운 명령어들이 자주 추가됩니다; 만약 아래 내용이 오래된 것 같다면 세션 내에서 `/help`를 실행하여 공식 목록을 확인하거나 [라이브 슬래시 명령어 레퍼런스](https://hermes-agent.nousresearch.com/docs/reference/slash-commands)를 참조하세요.
모든 것의 기준이 되는 레지스트리는 `hermes_cli/commands.py`이며 자동 완성, Telegram 메뉴, Slack 매핑, `/help` 등 모든 소비자가 이를 기반으로 합니다.

### 세션 제어
```
/new (/reset)        새 세션 시작
/clear               화면 지우기 + 새 세션 시작 (CLI)
/retry               마지막 메시지 재전송
/undo                마지막 대화 삭제
/title [name]        세션 이름 지정
/compress            수동으로 컨텍스트 압축
/stop                백그라운드 프로세스 종료
/rollback [N]        파일 시스템 체크포인트 복원
/snapshot [sub]      Hermes 구성/상태의 스냅샷 생성 또는 복원 (CLI)
/background <prompt> 백그라운드에서 프롬프트 실행
/queue <prompt>      다음 턴을 위한 대기열 추가
/steer <prompt>      실행을 중단하지 않고 다음 도구 호출 후 메시지 삽입
/agents (/tasks)     활성 에이전트 및 실행 중인 작업 표시
/resume [name]       이름이 지정된 세션 재개
/goal [text|sub]     Hermes가 달성할 때까지 여러 턴에 걸쳐 작업할 목표 설정
                     (하위 명령어: status, pause, resume, clear)
/redraw              전체 UI 다시 그리기 강제 실행 (CLI)
```

### 구성
```
/config              구성 표시 (CLI)
/model [name]        모델 표시 또는 변경
/personality [name]  성격(personality) 설정
/reasoning [level]   추론 설정 (none|minimal|low|medium|high|xhigh|show|hide)
/verbose             순환: off → new → all → verbose
/voice [on|off|tts]  음성 모드
/yolo                승인 우해(bypass) 토글
/busy [sub]          Hermes가 작업하는 동안 Enter 키의 동작 제어 (CLI)
                     (하위 명령어: queue, steer, interrupt, status)
/indicator [style]   TUI 작업 중(busy) 표시 스타일 선택 (CLI)
                     (styles: kaomoji, emoji, unicode, ascii)
/footer [on|off]     최종 답변에 게이트웨이 런타임 메타데이터 푸터 토글
/skin [name]         테마 변경 (CLI)
/statusbar           상태 표시줄 토글 (CLI)
```

### 도구 & 스킬
```
/tools               도구 관리 (CLI)
/toolsets            도구 세트 목록 (CLI)
/skills              스킬 검색/설치 (CLI)
/skill <name>        세션에 스킬 로드
/reload-skills       추가/제거된 스킬을 위해 ~/.hermes/skills/ 다시 스캔
/reload              실행 중인 세션에 .env 변수 다시 로드 (CLI)
/reload-mcp          MCP 서버 다시 로드
/cron                크론 작업 관리 (CLI)
/curator [sub]       백그라운드 스킬 유지 관리 (status, run, pin, archive, …)
/kanban [sub]        다중 프로필 협업 보드 (tasks, links, comments)
/plugins             플러그인 목록 (CLI)
```

### 게이트웨이
```
/approve             대기 중인 명령어 승인 (게이트웨이)
/deny                대기 중인 명령어 거부 (게이트웨이)
/restart             게이트웨이 재시작 (게이트웨이)
/sethome             현재 채팅을 홈 채널로 설정 (게이트웨이)
/update              Hermes를 최신으로 업데이트 (게이트웨이)
/topic [sub]         Telegram DM 주제 세션 활성화 또는 검사 (게이트웨이)
/platforms (/gateway) 플랫폼 연결 상태 표시 (게이트웨이)
```

### 유틸리티
```
/branch (/fork)      현재 세션 분기(Branch)
/fast                우선순위/빠른 처리 토글
/browser             CDP 브라우저 연결 열기
/history             대화 기록 표시 (CLI)
/save                대화를 파일로 저장 (CLI)
/copy [N]            어시스턴트의 마지막 응답을 클립보드에 복사 (CLI)
/paste               클립보드 이미지 첨부 (CLI)
/image               로컬 이미지 파일 첨부 (CLI)
```

### 정보
```
/help                명령어 표시
/commands [page]     모든 명령어 찾아보기 (게이트웨이)
/usage               토큰 사용량
/insights [days]     사용량 분석
/gquota              Google Gemini Code Assist 할당량 사용 표시 (CLI)
/status              세션 정보 (게이트웨이)
/profile             활성 프로필 정보
/debug               디버그 보고서(시스템 정보 + 로그) 업로드 및 공유 링크 받기
```

### 종료
```
/quit (/exit, /q)    CLI 종료
```

---

## 주요 경로 & 설정

```
~/.hermes/config.yaml       메인 구성 파일
~/.hermes/.env              API 키 및 비밀 정보
$HERMES_HOME/skills/        설치된 스킬
~/.hermes/sessions/         게이트웨이 라우팅 인덱스, 요청 덤프, *.jsonl 트랜스크립트 (sessions.write_json_snapshots: true 인 경우 선택적인 세션별 JSON 스냅샷)
~/.hermes/state.db          공식 세션 저장소 (SQLite + FTS5)
~/.hermes/logs/             게이트웨이 및 오류 로그
~/.hermes/auth.json         OAuth 토큰 및 자격 증명 풀
~/.hermes/hermes-agent/     소스 코드 (git 설치의 경우)
```

프로필은 동일한 레이아웃으로 `~/.hermes/profiles/<name>/`을 사용합니다.

### 설정 섹션

`hermes config edit` 또는 `hermes config set section.key value`를 사용하여 편집합니다.

| 섹션 | 핵심 옵션 |
|---------|-------------|
| `model` | `default`, `provider`, `base_url`, `api_key`, `context_length` |
| `agent` | `max_turns` (90), `tool_use_enforcement` |
| `terminal` | `backend` (local/docker/ssh/modal), `cwd`, `timeout` (180) |
| `compression` | `enabled`, `threshold` (0.50), `target_ratio` (0.20) |
| `display` | `skin`, `tool_progress`, `show_reasoning`, `show_cost` |
| `stt` | `enabled`, `provider` (local/groq/openai/mistral) |
| `tts` | `provider` (edge/elevenlabs/openai/minimax/mistral/neutts) |
| `memory` | `memory_enabled`, `user_profile_enabled`, `provider` |
| `security` | `tirith_enabled`, `website_blocklist` |
| `delegation` | `model`, `provider`, `base_url`, `api_key`, `max_iterations` (50), `reasoning_effort` |
| `checkpoints` | `enabled`, `max_snapshots` (50) |

전체 구성 참조: https://hermes-agent.nousresearch.com/docs/user-guide/configuration

### 제공자 (Providers)

20개 이상의 제공자를 지원합니다. `hermes model` 또는 `hermes setup`으로 설정합니다.

| 제공자 | 인증 | 주요 환경 변수 |
|----------|------|-------------|
| OpenRouter | API 키 | `OPENROUTER_API_KEY` |
| Anthropic | API 키 | `ANTHROPIC_API_KEY` |
| Nous Portal | OAuth | `hermes auth` |
| OpenAI Codex | OAuth | `hermes auth` |
| GitHub Copilot | 토큰 | `COPILOT_GITHUB_TOKEN` |
| Google Gemini | API 키 | `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY` |
| DeepSeek | API 키 | `DEEPSEEK_API_KEY` |
| xAI / Grok | API 키 | `XAI_API_KEY` |
| Hugging Face | 토큰 | `HF_TOKEN` |
| Z.AI / GLM | API 키 | `GLM_API_KEY` |
| MiniMax | API 키 | `MINIMAX_API_KEY` |
| MiniMax CN | API 키 | `MINIMAX_CN_API_KEY` |
| Kimi / Moonshot | API 키 | `KIMI_API_KEY` |
| Alibaba / DashScope | API 키 | `DASHSCOPE_API_KEY` |
| Xiaomi MiMo | API 키 | `XIAOMI_API_KEY` |
| Kilo Code | API 키 | `KILOCODE_API_KEY` |
| OpenCode Zen | API 키 | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | API 키 | `OPENCODE_GO_API_KEY` |
| Qwen OAuth | OAuth | `hermes auth add qwen-oauth` |
| Custom endpoint | Config | config.yaml의 `model.base_url` + `model.api_key` |
| GitHub Copilot ACP | 외부 | `COPILOT_CLI_PATH` 또는 Copilot CLI |

전체 제공자 문서: https://hermes-agent.nousresearch.com/docs/integrations/providers

### 도구 세트 (Toolsets)

`hermes tools` (대화형) 또는 `hermes tools enable/disable NAME`을 통해 활성화/비활성화합니다.

| 도구 세트 | 제공하는 기능 |
|---------|-----------------|
| `web` | 웹 검색 및 콘텐츠 추출 |
| `search` | 웹 검색 전용 (`web`의 하위 집합) |
| `browser` | 브라우저 자동화 (Browserbase, Camofox, 또는 로컬 Chromium) |
| `terminal` | 쉘 명령어 및 프로세스 관리 |
| `file` | 파일 읽기/쓰기/검색/패치 |
| `code_execution` | 샌드박스 처리된 Python 실행 |
| `vision` | 이미지 분석 |
| `image_gen` | AI 이미지 생성 |
| `video` | 비디오 분석 및 생성 |
| `tts` | 텍스트 음성 변환 (Text-to-speech) |
| `skills` | 스킬 탐색 및 관리 |
| `memory` | 세션을 넘나드는 영구 메모리 |
| `session_search` | 과거 대화 검색 |
| `delegation` | 하위 에이전트 작업 위임 |
| `cronjob` | 예약된 작업 관리 |
| `clarify` | 사용자에게 명확한 질문 요구 |
| `messaging` | 크로스 플랫폼 메시지 전송 |
| `todo` | 세션 내 작업 계획 및 추적 |
| `kanban` | 다중 에이전트 작업 대기열 도구 (워커로 제한됨) |
| `debugging` | 추가 인트로스펙션(introspection)/디버그 도구 (기본적으로 꺼짐) |
| `safe` | 잠긴 세션을 위한 최소한의 위험성이 낮은 도구 세트 |
| `spotify` | Spotify 재생 및 재생목록 제어 |
| `homeassistant` | 스마트 홈 제어 (기본적으로 꺼짐) |
| `discord` | Discord 통합 도구 |
| `discord_admin` | Discord 관리/모더레이션 도구 |
| `feishu_doc` | Feishu (Lark) 문서 도구 |
| `feishu_drive` | Feishu (Lark) 드라이브 도구 |
| `yuanbao` | Yuanbao 통합 도구 |
| `rl` | 강화 학습 도구 (기본적으로 꺼짐) |
| `moa` | Mixture of Agents (기본적으로 꺼짐) |

전체 열거는 `toolsets.py`의 `TOOLSETS` 딕셔너리에 있으며, 대부분의 플랫폼이 상속받는 기본 번들은 `_HERMES_CORE_TOOLS`입니다.

도구 변경은 `/reset` (새 세션)에서 적용됩니다. 프롬프트 캐싱을 보존하기 위해 대화 도중에는 적용되지 않습니다.

---

## 보안 및 개인 정보 보호 토글

"Hermes가 내 출력 / 도구 호출 / 명령에 왜 X를 수행합니까?"와 관련된 일반적인 토글과 이를 변경하기 위한 정확한 명령어입니다. 이것들은 시작 시 한 번 읽히므로 대부분 새 세션(채팅에서 `/reset` 또는 새로운 `hermes` 호출 시작)이 필요합니다.

### 도구 출력에서 비밀 정보 삭제 (Secret redaction)

비밀 정보 삭제는 **기본적으로 켜져 있습니다** — 도구 출력 (터미널 stdout, `read_file`, 웹 콘텐츠, 하위 에이전트 요약 등)이 대화 컨텍스트 및 로그에 들어가기 전에 API 키, 토큰 및 비밀 정보로 보이는 문자열이 있는지 스캔합니다. 일반적인 사용 시에는 활성화된 상태로 유지하세요:

```bash
hermes config set security.redact_secrets true       # 전역적으로 활성화 상태 유지
```

**재시작 필요.** `security.redact_secrets`는 가져오기(import) 시점에 스냅샷으로 저장됩니다. 도구 호출에서 `export HERMES_REDACT_SECRETS=false`를 통해 세션 도중에 전환하더라도 실행 중인 프로세스에는 적용되지 **않습니다**. 터미널의 설정에서 변경한 다음 새 세션을 시작해야 합니다. 이는 LLM이 작업 중간에 이 토글을 스스로 끄는 것을 방지하기 위한 의도적인 설계입니다.

디버깅 또는 삭제기(redactor) 개발을 위해 원시 자격 증명과 같은 문자열이 의도적으로 필요한 경우에만 비활성화하세요:
```bash
hermes config set security.redact_secrets false
```

### 게이트웨이 메시지에서 PII(개인 식별 정보) 삭제

비밀 정보 삭제와는 별개입니다. 이 기능이 활성화되면 게이트웨이는 모델에 도달하기 전에 사용자 ID를 해시 처리하고 세션 컨텍스트에서 전화번호를 제거합니다:

```bash
hermes config set privacy.redact_pii true    # 활성화
hermes config set privacy.redact_pii false   # 비활성화 (기본값)
```

### 명령어 승인 프롬프트

기본적으로 (`approvals.mode: manual`), 파괴적인 것으로 표시된 쉘 명령(`rm -rf`, `git reset --hard` 등)을 실행하기 전에 Hermes는 사용자에게 묻습니다. 모드는 다음과 같습니다:

- `manual` — 항상 묻기 (기본값)
- `smart` — 위험도가 낮은 명령은 보조 LLM을 사용하여 자동 승인하고, 위험도가 높은 경우에만 묻기
- `off` — 모든 승인 프롬프트 건너뛰기 (`--yolo`와 동일)

```bash
hermes config set approvals.mode smart       # 권장하는 중간 지점
hermes config set approvals.mode off         # 모든 것을 우회 (권장하지 않음)
```

구성을 변경하지 않고 호출당 우회하는 방법:
- `hermes --yolo …`
- `export HERMES_YOLO_MODE=1`

참고: YOLO / `approvals.mode: off` 설정은 비밀 정보 삭제(secret redaction)를 끄지 않습니다. 둘은 서로 독립적입니다.

### 쉘 훅 허용 목록 (Shell hooks allowlist)

일부 쉘 훅 통합은 실행되기 전에 명시적인 허용 목록(allowlisting)이 필요합니다. 훅이 처음 실행되려고 할 때 대화형으로 프롬프트를 표시하여 `~/.hermes/shell-hooks-allowlist.json`에서 관리됩니다.

### 웹/브라우저/이미지 생성 도구 비활성화

모델이 네트워크나 미디어 도구에 접근하지 못하게 하려면, `hermes tools`를 열고 플랫폼별로 설정을 변경하세요. 다음 세션(`/reset`)부터 적용됩니다. 위 도구 & 스킬 섹션을 참조하세요.

---

## 음성 및 트랜스크립션 (Voice & Transcription)

### STT (음성 → 텍스트)

메시징 플랫폼에서 보낸 음성 메시지는 자동으로 텍스트로 변환됩니다.

제공자 우선순위 (자동 감지):
1. **로컬 faster-whisper** — 무료, API 키 불필요: `pip install faster-whisper`
2. **Groq Whisper** — 무료 티어: `GROQ_API_KEY` 설정
3. **OpenAI Whisper** — 유료: `VOICE_TOOLS_OPENAI_KEY` 설정
4. **Mistral Voxtral** — `MISTRAL_API_KEY` 설정

구성:
```yaml
stt:
  enabled: true
  provider: local        # local, groq, openai, mistral
  local:
    model: base          # tiny, base, small, medium, large-v3
```

### TTS (텍스트 → 음성)

| 제공자 | 환경 변수 | 무료 여부 |
|----------|---------|-------|
| Edge TTS | 없음 | 예 (기본값) |
| ElevenLabs | `ELEVENLABS_API_KEY` | 무료 티어 |
| OpenAI | `VOICE_TOOLS_OPENAI_KEY` | 유료 |
| MiniMax | `MINIMAX_API_KEY` | 유료 |
| Mistral (Voxtral) | `MISTRAL_API_KEY` | 유료 |
| NeuTTS (로컬) | 없음 (`pip install neutts[all]` + `espeak-ng`) | 무료 |

음성 명령어: `/voice on` (음성 대 음성), `/voice tts` (항상 음성), `/voice off`.

---

## 추가 Hermes 인스턴스 스폰(Spawning)

별도의 세션, 도구 및 환경을 갖춘 완전히 독립적인 하위 프로세스로 추가 Hermes 프로세스를 실행합니다.

### `delegate_task`와의 사용 시기 비교

| | `delegate_task` | `hermes` 프로세스 생성 |
|-|-----------------|--------------------------|
| 격리(Isolation) | 분리된 대화, 프로세스는 공유 | 완전히 독립된 프로세스 |
| 기간 | 분 단위 (부모의 루프에 종속됨) | 시간/일 단위 |
| 도구 권한 | 부모 도구의 하위 집합 | 전체 도구 권한 |
| 대화형(Interactive) | 아니오 | 예 (PTY 모드) |
| 사용 사례 | 빠르고 병렬적인 하위 작업 | 장기적 자율 미션 |

### 일회성(One-Shot) 모드

```
terminal(command="hermes chat -q 'Research GRPO papers and write summary to ~/research/grpo.md'", timeout=300)

# 장기 작업을 위한 백그라운드 실행:
terminal(command="hermes chat -q 'Set up CI/CD for ~/myapp'", background=true)
```

### 대화형 PTY 모드 (tmux 사용)

Hermes는 실제 터미널이 필요한 `prompt_toolkit`을 사용합니다. 대화형 스폰(spawning)을 위해 tmux를 사용하세요:

```
# 시작
terminal(command="tmux new-session -d -s agent1 -x 120 -y 40 'hermes'", timeout=10)

# 시작 대기 후 메시지 전송
terminal(command="sleep 8 && tmux send-keys -t agent1 'Build a FastAPI auth service' Enter", timeout=15)

# 출력 읽기
terminal(command="sleep 20 && tmux capture-pane -t agent1 -p", timeout=5)

# 후속 메시지 전송
terminal(command="tmux send-keys -t agent1 'Add rate limiting middleware' Enter", timeout=5)

# 종료
terminal(command="tmux send-keys -t agent1 '/exit' Enter && sleep 2 && tmux kill-session -t agent1", timeout=10)
```

### 다중 에이전트 협업

```
# 에이전트 A: 백엔드
terminal(command="tmux new-session -d -s backend -x 120 -y 40 'hermes -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t backend 'Build REST API for user management' Enter", timeout=15)

# 에이전트 B: 프론트엔드
terminal(command="tmux new-session -d -s frontend -x 120 -y 40 'hermes -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t frontend 'Build React dashboard for user management' Enter", timeout=15)

# 진행 확인 후 서로에게 컨텍스트 릴레이
terminal(command="tmux capture-pane -t backend -p | tail -30", timeout=5)
terminal(command="tmux send-keys -t frontend 'Here is the API schema from the backend agent: ...' Enter", timeout=5)
```

### 세션 재개 (Session Resume)

```
# 가장 최근 세션 재개
terminal(command="tmux new-session -d -s resumed 'hermes --continue'", timeout=10)

# 특정 세션 재개
terminal(command="tmux new-session -d -s resumed 'hermes --resume 20260225_143052_a1b2c3'", timeout=10)
```

### 팁

- **빠른 하위 작업에는 `delegate_task`를 우선 사용하세요** — 전체 프로세스를 생성하는 것보다 오버헤드가 적습니다.
- 코드를 편집하는 에이전트를 스폰할 때는 **`-w` (worktree 모드)** 를 사용하세요 — git 충돌을 방지합니다.
- 일회성 모드의 경우 **timeout을 설정하세요** — 복잡한 작업은 5-10분이 걸릴 수 있습니다.
- "Fire-and-forget" 작업에는 **`hermes chat -q`를 사용하세요** — PTY가 필요 없습니다.
- 대화형 세션에는 **tmux를 사용하세요** — 원시 PTY 모드는 `prompt_toolkit`과 `\r` 대 `\n` 문제가 있습니다.
- **예약된 작업**의 경우 스폰하는 대신 `cronjob` 도구를 사용하세요 — 전달과 재시도를 처리해 줍니다.

---

## 내구성(Durable) & 백그라운드 시스템

메인 대화 루프와 함께 네 가지 시스템이 실행됩니다. 여기서는 빠른 참조만 제공하며 전체 개발자 노트는 `AGENTS.md`에, 사용자 대상 문서는 `website/docs/user-guide/features/` 아래에 있습니다.

### 위임 (`delegate_task`)

동기식 하위 에이전트 생성 — 부모 에이전트는 하위 에이전트의 요약을 기다린 후 자신의 루프를 계속 진행합니다. 격리된 컨텍스트 + 터미널 세션을 가집니다.

- **단일:** `delegate_task(goal, context, toolsets)`.
- **일괄 처리(Batch):** `delegate_task(tasks=[{goal, ...}, ...])` 하위 에이전트를 병렬로 실행하며 `delegation.max_concurrent_children` (기본값 3)의 제한을 받습니다.
- **역할(Roles):** `leaf` (기본값; 재위임 불가) 대 `orchestrator` (`delegation.max_spawn_depth`의 제한 내에서 자체 워커를 생성할 수 있음).
- **내구성 없음(Not durable).** 부모가 중단되면 자식도 취소됩니다. 해당 턴(turn)보다 더 오래 지속되어야 하는 작업의 경우 `cronjob` 또는 `terminal(background=True, notify_on_complete=True)`를 사용하세요.

구성: `config.yaml`의 `delegation.*`.

### 크론 (예약 작업)

내구성이 있는 스케줄러 — `cron/jobs.py` + `cron/scheduler.py`. `cronjob` 도구, `hermes cron` CLI (`list`, `add`, `edit`, `pause`, `resume`, `run`, `remove`) 또는 `/cron` 슬래시 명령어로 구동할 수 있습니다.

- **스케줄:** 지속 시간 (`"30m"`, `"2h"`), "every" 구문 (`"every monday 9am"`), 5개 필드 크론 (`"0 9 * * *"`), 또는 ISO 타임스탬프.
- **작업별 옵션:** `skills`, `model`/`provider` 재정의, `script` (실행 전 데이터 수집; `no_agent=True` 설정 시 스크립트가 전체 작업이 됨), `context_from` (작업 A의 출력을 작업 B로 연결), `workdir` (해당 디렉토리의 `AGENTS.md` / `CLAUDE.md`를 로드하여 실행), 멀티 플랫폼 전달(delivery).
- **불변 원칙(Invariants):** 실행당 3분의 하드 인터럽트 제한, `.tick.lock` 파일로 프로세스 간 중복 틱 방지, 크론 세션은 기본적으로 `skip_memory=True` 전달, 그리고 크론의 전달은 타겟 게이트웨이 세션에 미러링되지 않고 헤더/푸터로 묶여 전송됩니다 (역할 교대를 유지하기 위함).

사용자 문서: https://hermes-agent.nousresearch.com/docs/user-guide/features/cron

### 큐레이터 (스킬 라이프사이클)

에이전트가 생성한 스킬의 백그라운드 유지 관리입니다. 사용량을 추적하고, 유휴(idle) 스킬을 오래된(stale) 스킬로 표시하고, 오래된 스킬을 보관소(archive)로 보내고, 실행 전 tar.gz 백업을 유지하여 데이터가 유실되지 않게 합니다.

- **CLI:** `hermes curator <verb>` — `status`, `run`, `pause`, `resume`, `pin`, `unpin`, `archive`, `restore`, `prune`, `backup`, `rollback`.
- **슬래시 명령어:** `/curator <subcommand>`는 CLI의 기능을 그대로 반영합니다.
- **범위:** `created_by: "agent"` 출처를 가진 스킬만 건드립니다. 번들 및 허브(hub)에서 설치된 스킬은 건드리지 않습니다. **절대 삭제하지 않습니다** — 가장 파괴적인 작업은 보관(archive)입니다. 고정된(Pinned) 스킬은 모든 자동 전환 및 LLM 리뷰 과정에서 면제됩니다.
- **텔레메트리(원격 분석):** `~/.hermes/skills/.usage.json`의 사이드카 파일에서 스킬별 `use_count`, `view_count`, `patch_count`, `last_activity_at`, `state`, `pinned` 정보를 관리합니다.

구성: `curator.*` (`enabled`, `interval_hours`, `min_idle_hours`, `stale_after_days`, `archive_after_days`, `backup.*`).
사용자 문서: https://hermes-agent.nousresearch.com/docs/user-guide/features/curator

### 칸반 (다중 에이전트 작업 대기열)

다중 프로필 / 다중 워커 협업을 위한 내구성이 있는 SQLite 보드입니다. 사용자는 `hermes kanban <verb>`를 통해 제어합니다; 디스패처에 의해 생성된 워커들은 `HERMES_KANBAN_TASK`로 제한된 전용 `kanban_*` 도구 세트를 보게 되며, 오케스트레이터 프로필은 더 넓은 범위의 `kanban` 도구 세트를 사용할 수 있습니다. 일반 세션에서는 설정을 하지 않는 한 `kanban_*` 스키마가 전혀 나타나지 않습니다.

- **CLI 동사 (일반적):** `init`, `create`, `list` (`ls` 별칭), `show`, `assign`, `link`, `unlink`, `comment`, `complete`, `block`, `unblock`, `archive`, `tail`. 덜 일반적인 것: `watch`, `stats`, `runs`, `log`, `dispatch`, `daemon`, `gc`.
- **워커/오케스트레이터 도구 세트:** `kanban_show`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`; 디스패처가 생성한 작업 외부에서 명시적으로 `kanban` 도구 세트를 활성화한 프로필도 보드 라우팅을 위해 `kanban_list`와 `kanban_unblock`을 갖습니다.
- **디스패처(Dispatcher)** 는 기본적으로 게이트웨이 내부에서 실행됩니다 (`kanban.dispatch_in_gateway: true`) — 오래된 권한을 회수하고, 준비된 작업을 승격시키고, 원자적으로 권한을 주장하고, 할당된 프로필을 생성합니다. 연속 생성 실패 횟수가 `failure_limit` (기본값 2; `kanban.failure_limit` 또는 작업별 `max_retries`로 구성 가능)에 도달하면 작업을 자동으로 차단합니다.
- **격리:** 보드(board)는 단단한 경계입니다 (워커들은 환경 변수에 `HERMES_KANBAN_BOARD`가 고정됨); 테넌트(tenant)는 작업 공간 경로 + 메모리 키 격리를 위한 보드 내의 부드러운 네임스페이스입니다.

사용자 문서: https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban

---

## Windows 특유의 주의 사항 (Quirks)

Hermes는 Windows(PowerShell, cmd, Windows Terminal, git-bash mintty, VS Code 통합 터미널)에서 네이티브로 실행됩니다. 대부분 잘 작동하지만 Win32와 POSIX 간의 몇 가지 차이점으로 인해 문제가 발생하기도 합니다 — 다음 사람(또는 다음 세션)이 처음부터 다시 문제를 찾지 않도록 문제를 발견할 때마다 여기에 문서화하세요.

### 입력 / 키 바인딩 (Keybindings)

**Alt+Enter는 줄 바꿈을 삽입하지 않습니다.** Windows Terminal은 터미널 계층에서 전체 화면 토글을 위해 Alt+Enter를 가로챕니다 — 따라서 이 키 입력은 `prompt_toolkit`에 도달하지 못합니다. 대신 **Ctrl+Enter**를 사용하세요. Windows Terminal은 Ctrl+Enter를 일반 Enter(`c-m` / CR)와 구별되는 LF(`c-j`)로 전달하며, CLI는 `win32`에서만 `c-j`를 줄 바꿈 삽입에 바인딩합니다 ( `_bind_prompt_submit_keys` + `cli.py`의 Windows 전용 `c-j` 바인딩 참조).
부작용: 원시 Ctrl+J 키 입력도 Windows에서 줄 바꿈을 삽입합니다 — Windows Terminal은 Win32 콘솔 API 계층에서 Ctrl+Enter와 Ctrl+J를 동일한 키 코드로 통합하기 때문에 이는 불가피합니다. Windows에서 Ctrl+J에 대해 상충되는 바인딩이 없었기 때문에 이것은 무해한 부작용입니다.

mintty / git-bash의 경우 옵션 → 키(Options → Keys)에서 Alt+Fn 단축키를 비활성화하지 않는 한 (Alt+Enter시 전체 화면으로) 동일하게 동작합니다. 그냥 Ctrl+Enter를 사용하는 것이 더 쉽습니다.

**키 바인딩 진단.** 저장소 루트에서 `python scripts/keystroke_diagnostic.py`를 실행하여 현재 터미널에서 `prompt_toolkit`이 각 키 입력을 어떻게 식별하는지 정확하게 확인하세요. 이는 "Shift+Enter가 별도의 키로 인식되는가?" (거의 그렇지 않음 — 대부분의 터미널이 일반 Enter로 통합함) 또는 "내 터미널이 Ctrl+Enter에 대해 어떤 바이트 시퀀스를 보내는가?"와 같은 질문에 답해줍니다. Ctrl+Enter = c-j 라는 사실이 이렇게 확인되었습니다.

### 구성 / 파일

**첫 실행 시 HTTP 400 "No models provided" 에러.** `config.yaml`이 UTF-8 BOM과 함께 저장되었습니다 (Windows 앱이 작성할 때 일반적임). BOM이 없는 UTF-8로 다시 저장하세요. `hermes config edit`은 BOM 없이 저장합니다; 메모장에서의 수동 편집이 대개 원인입니다.

### `execute_code` / 샌드박스 (Sandbox)

**WinError 10106** ("요청한 서비스 제공자를 로드하거나 초기화할 수 없습니다.") 샌드박스의 하위 프로세스 오류 — `AF_INET` 소켓을 생성할 수 없으므로 루프백(loopback)-TCP RPC 폴백(fallback)이 `connect()` 이전에 실패합니다. 근본적인 원인은 대개 망가진 Winsock LSP가 **아닙니다**; 원인은 Hermes 자체의 환경 변수 스크러버(scrubber)가 하위 환경에서 `SYSTEMROOT` / `WINDIR` / `COMSPEC`을 삭제했기 때문입니다. Python의 `socket` 모듈은 `mswsock.dll`을 찾기 위해 `SYSTEMROOT`를 필요로 합니다. `tools/code_execution_tool.py`의 `_WINDOWS_ESSENTIAL_ENV_VARS` 허용 목록(allowlist)을 통해 해결되었습니다. 여전히 이 문제가 발생한다면, `execute_code` 블록 내부에서 `os.environ`을 에코(echo)하여 `SYSTEMROOT`가 설정되었는지 확인하세요. 전체 진단 방법은 `references/execute-code-sandbox-env-windows.md`에 있습니다.

### 테스트 / 기여 (Testing / Contributing)

**`scripts/run_tests.sh`가 Windows에서는 그대로 작동하지 않습니다** — POSIX venv 레이아웃(`.venv/bin/activate`)을 찾기 때문입니다. `venv/Scripts/`에 설치된 Hermes의 venv에는 `pip`이나 `pytest`가 없습니다 (설치 크기를 줄이기 위해 제거됨).
해결 방법: 시스템의 Python 3.11 유저 사이트에 `pytest + pytest-xdist + pyyaml`을 설치한 다음, `PYTHONPATH`를 설정하고 pytest를 직접 호출하세요:

```bash
"/c/Program Files/Python311/python" -m pip install --user pytest pytest-xdist pyyaml
export PYTHONPATH="$(pwd)"
"/c/Program Files/Python311/python" -m pytest tests/foo/test_bar.py -v --tb=short -n 0
```

`-n 4`가 아닌 `-n 0`을 사용하세요 — `pyproject.toml`의 기본 `addopts`에 이미 `-n`이 포함되어 있고, 래퍼(wrapper)의 CI 동등성 보장(parity guarantees)은 POSIX 환경 외부에는 적용되지 않습니다.

**POSIX 전용 테스트에는 건너뛰기 조건(skip guards)이 필요합니다.** 코드베이스에 이미 있는 일반적인 마커(markers)들:
- 심볼릭 링크(Symlinks) — Windows에서는 상승된 권한이 필요함
- `0o600` 파일 모드 — 기본적으로 NTFS에서는 POSIX 모드 비트가 강제되지 않음
- `signal.SIGALRM` — Unix 전용 (`tests/conftest.py::_enforce_test_timeout` 참조)
- Winsock / Windows 전용 회귀(regressions) — `@pytest.mark.skipif(sys.platform != "win32", ...)`

기존 코드베이스와 일관성을 유지하기 위해 기존의 건너뛰기 패턴 스타일(`sys.platform == "win32"` 또는 `sys.platform.startswith("win")`)을 사용하세요.

### 경로 / 파일 시스템

**줄 바꿈(Line endings).** Git이 `LF will be replaced by CRLF the next time Git touches it`이라고 경고할 수 있습니다. 이는 외관상의 문제입니다 — 저장소의 `.gitattributes`가 이를 정규화합니다. 커밋된 POSIX 줄 바꿈 파일을 에디터가 CRLF로 자동 변환하지 않도록 하세요.

**슬래시(Forward slashes)는 거의 모든 곳에서 작동합니다.** `C:/Users/...`는 모든 Hermes 도구와 대부분의 Windows API에서 허용됩니다. 코드와 로그에서는 가급적 슬래시를 사용하세요 — 이렇게 하면 bash에서 백슬래시를 이스케이프(shell-escaping)하는 것을 피할 수 있습니다.

---

## 문제 해결 (Troubleshooting)

### 음성(Voice)이 작동하지 않음
1. config.yaml에서 `stt.enabled: true` 인지 확인하세요.
2. 제공자 확인: `pip install faster-whisper`를 하거나 API 키를 설정하세요.
3. 게이트웨이의 경우: `/restart`. CLI의 경우: 종료 후 다시 시작하세요.

### 도구를 사용할 수 없음
1. `hermes tools` — 사용 중인 플랫폼에 해당 도구 세트가 활성화되어 있는지 확인하세요.
2. 일부 도구는 환경 변수가 필요합니다 (`.env` 확인).
3. 도구를 활성화한 후에는 `/reset`을 수행하세요.

### 모델/제공자 이슈
1. `hermes doctor` — 구성 및 종속성(dependencies)을 확인하세요.
2. `hermes auth` — OAuth 제공자를 다시 인증하세요 (또는 `hermes auth add <provider>`).
3. `.env`에 올바른 API 키가 있는지 확인하세요.
4. **Copilot 403**: `gh auth login`의 토큰은 Copilot API에서 동작하지 **않습니다**. 반드시 `hermes model` → GitHub Copilot을 통해 Copilot 전용 OAuth 디바이스 코드(device code) 흐름을 사용해야 합니다.

### 변경 사항이 적용되지 않음
- **도구/스킬:** `/reset`을 입력하여 업데이트된 도구 세트로 새 세션을 시작하세요.
- **구성(Config) 변경:** 게이트웨이의 경우: `/restart`. CLI의 경우: 종료 후 다시 시작하세요.
- **코드 변경:** CLI 또는 게이트웨이 프로세스를 재시작하세요.

### 스킬이 나타나지 않음
1. `hermes skills list` — 설치 여부 확인.
2. `hermes skills config` — 플랫폼별 활성화 여부 확인.
3. 명시적으로 로드: `/skill name` 또는 `hermes -s name`.

### 게이트웨이 이슈
먼저 로그를 확인하세요:
```bash
grep -i "failed to send\|error" ~/.hermes/logs/gateway.log | tail -20
```

일반적인 게이트웨이 문제:
- **SSH 로그아웃 시 게이트웨이 종료**: Linger(남아있기) 활성화: `sudo loginctl enable-linger $USER`
- **WSL2 닫을 때 게이트웨이 종료**: WSL2에서 systemd 서비스가 작동하려면 `/etc/wsl.conf`에 `systemd=true`가 있어야 합니다. 이 설정이 없으면 게이트웨이는 `nohup`으로 돌아가며(fall back) 세션이 닫힐 때 종료됩니다.
- **게이트웨이 크래시(충돌) 루프**: 실패 상태 초기화: `systemctl --user reset-failed hermes-gateway`

### 플랫폼 특정 문제
- **Discord 봇이 조용할 때**: 봇 설정의 Privileged Gateway Intents에서 **Message Content Intent**를 반드시 활성화해야 합니다.
- **Slack 봇이 DM에서만 작동할 때**: `message.channels` 이벤트를 구독해야 합니다. 그렇지 않으면 봇이 공개 채널을 무시합니다.
- **Windows 특정 문제들** (`Alt+Enter` 줄 바꿈, WinError 10106, UTF-8 BOM 설정, 테스트 스위트, 줄 바꿈 문자): 위의 전용 **Windows 특유의 주의 사항(Quirks)** 섹션을 참조하세요.

### 보조 모델(Auxiliary models)이 작동하지 않음
`auxiliary` 작업(vision, compression, session_search)이 아무 메시지 없이 실패한다면, `auto` 제공자가 백엔드를 찾지 못하는 것입니다. `OPENROUTER_API_KEY`나 `GOOGLE_API_KEY`를 설정하거나, 각 보조 작업의 제공자를 명시적으로 구성하세요:
```bash
hermes config set auxiliary.vision.provider <your_provider>
hermes config set auxiliary.vision.model <model_name>
```

---

## 정보가 있는 위치 (Where to Find Things)

| 찾고 있는 것 | 위치 (Location) |
|----------------|----------|
| 구성 옵션 | `hermes config edit` 또는 [Configuration docs](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) |
| 사용 가능한 도구 | `hermes tools list` 또는 [Tools reference](https://hermes-agent.nousresearch.com/docs/reference/tools-reference) |
| 슬래시 명령어 | 세션 중 `/help` 또는 [Slash commands reference](https://hermes-agent.nousresearch.com/docs/reference/slash-commands) |
| 스킬 카탈로그 | `hermes skills browse` 또는 [Skills catalog](https://hermes-agent.nousresearch.com/docs/reference/skills-catalog) |
| 제공자 설정 | `hermes model` 또는 [Providers guide](https://hermes-agent.nousresearch.com/docs/integrations/providers) |
| 플랫폼 설정 | `hermes gateway setup` 또는 [Messaging docs](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/) |
| MCP 서버 | `hermes mcp list` 또는 [MCP guide](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) |
| 프로필(Profiles) | `hermes profile list` 또는 [Profiles docs](https://hermes-agent.nousresearch.com/docs/user-guide/profiles) |
| 크론 작업 | `hermes cron list` 또는 [Cron docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) |
| 메모리 | `hermes memory status` 또는 [Memory docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) |
| 환경 변수 | `hermes config env-path` 또는 [Env vars reference](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) |
| CLI 명령어 | `hermes --help` 또는 [CLI reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) |
| 게이트웨이 로그 | `~/.hermes/logs/gateway.log` |
| 세션 파일 | `hermes sessions browse` (state.db를 읽음) |
| 소스 코드 | `~/.hermes/hermes-agent/` |

---

## 기여자(Contributor) 빠른 참조

이따금 기여하는 사람과 PR 작성자를 위한 안내입니다. 전체 개발자 문서: https://hermes-agent.nousresearch.com/docs/developer-guide/

### 프로젝트 구조 (Project Layout)

<!-- ascii-guard-ignore -->
```
hermes-agent/
├── run_agent.py          # AIAgent — 메인 대화 루프
├── model_tools.py        # 도구 탐색 및 디스패치
├── toolsets.py           # 도구 세트 정의
├── cli.py                # 대화형 CLI (HermesCLI)
├── hermes_state.py       # SQLite 세션 저장소
├── agent/                # 프롬프트 빌더, 컨텍스트 압축, 메모리, 모델 라우팅, 자격 증명 풀링, 스킬 디스패치
├── hermes_cli/           # CLI 하위 명령어, 구성, 셋업, 명령어
│   ├── commands.py       # 슬래시 명령어 레지스트리 (CommandDef)
│   ├── config.py         # DEFAULT_CONFIG, 환경 변수 정의
│   └── main.py           # CLI 진입점 및 argparse
├── tools/                # 도구당 하나의 파일
│   └── registry.py       # 중앙 도구 레지스트리
├── gateway/              # 메시징 게이트웨이
│   └── platforms/        # 플랫폼 어댑터 (telegram, discord 등)
├── cron/                 # 작업 스케줄러
├── tests/                # 약 3000개의 pytest 테스트
└── website/              # Docusaurus 문서 사이트
```
<!-- ascii-guard-ignore-end -->

구성: `~/.hermes/config.yaml` (설정), `~/.hermes/.env` (API 키).

### 도구 추가하기 (파일 3개)

**1. `tools/your_tool.py` 파일 생성:**
```python
import json, os
from tools.registry import registry

def check_requirements() -> bool:
    return bool(os.getenv("EXAMPLE_API_KEY"))

def example_tool(param: str, task_id: str = None) -> str:
    return json.dumps({"success": True, "data": "..."})

registry.register(
    name="example_tool",
    toolset="example",
    schema={"name": "example_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: example_tool(
        param=args.get("param", ""), task_id=kw.get("task_id")),
    check_fn=check_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

**2. `toolsets.py` 에 추가** → `_HERMES_CORE_TOOLS` 목록.

자동 발견(Auto-discovery): 최상위 레벨에 `registry.register()` 호출이 있는 모든 `tools/*.py` 파일은 자동으로 가져오기(import) 되므로 수동으로 목록을 작성할 필요가 없습니다.

모든 핸들러는 JSON 문자열을 반환해야 합니다. 하드코딩된 `~/.hermes` 대신 경로를 얻기 위해 `get_hermes_home()`을 사용하세요.

### 슬래시 명령어 추가하기

1. `hermes_cli/commands.py`의 `COMMAND_REGISTRY`에 `CommandDef` 추가
2. `cli.py` → `process_command()`에 핸들러 추가
3. (선택 사항) `gateway/run.py`에 게이트웨이 핸들러 추가

모든 소비자(도움말 텍스트, 자동 완성, Telegram 메뉴, Slack 매핑)는 중앙 레지스트리에서 자동으로 파생됩니다.

### 에이전트 루프 (하이 레벨)

```
run_conversation():
  1. 시스템 프롬프트 작성
  2. 반복 횟수가 최대치(max) 미만인 동안 루프 실행:
     a. LLM 호출 (OpenAI 형식의 메시지 + 도구 스키마)
     b. tool_calls가 있으면 → handle_function_call()을 통해 각각 디스패치 → 결과 추가 → 계속
     c. 텍스트 응답이 오면 → 반환(return)
  3. 토큰 제한에 가까워지면 컨텍스트 압축이 자동으로 트리거됨
```

### 테스트 (Testing)

```bash
python -m pytest tests/ -o 'addopts=' -q   # 전체 테스트 스위트
python -m pytest tests/tools/ -q            # 특정 영역
```

- 테스트는 `HERMES_HOME`을 임시 디렉토리로 자동 리디렉션합니다 — 실제 `~/.hermes/`를 절대 건드리지 않습니다.
- 변경 사항을 푸시(push)하기 전에 전체 테스트 스위트를 실행하세요.
- 기본적으로 적용된(baked-in) pytest 플래그를 지우려면 `-o 'addopts='`를 사용하세요.

**Windows 기여자:** `scripts/run_tests.sh`는 현재 POSIX venvs (`.venv/bin/activate` / `venv/bin/activate`)를 찾도록 되어있어서 레이아웃이 `venv/Scripts/activate` + `python.exe`인 Windows에서는 오류가 발생합니다. `venv/Scripts/`에 설치된 Hermes의 venv에는 엔드유저의 설치 크기를 줄이기 위해 `pip`이나 `pytest`도 없습니다. 해결 방법: 시스템의 Python 3.11 유저 사이트에 pytest + pytest-xdist + pyyaml을 설치(`/c/Program Files/Python311/python -m pip install --user pytest pytest-xdist pyyaml`)한 다음 직접 테스트를 실행하세요:

```bash
export PYTHONPATH="$(pwd)"
"/c/Program Files/Python311/python" -m pytest tests/tools/test_foo.py -v --tb=short -n 0
```

`pyproject.toml`의 기본 `addopts`에 이미 `-n`이 포함되어 있고, 래퍼(wrapper)의 CI 동등성 스토리가 POSIX 외 환경에는 적용되지 않으므로 (`-n 4`가 아닌) `-n 0`을 사용하세요.

**크로스 플랫폼 테스트 검증(guards):** POSIX 전용 시스템 호출(syscalls)을 사용하는 테스트에는 건너뛰기 마커(skip marker)가 필요합니다. 코드베이스에 이미 존재하는 일반적인 예시:
- 심볼릭 링크(Symlink) 생성 → `@pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require elevated privileges on Windows")` (`tests/cron/test_cron_script.py` 참조)
- POSIX 파일 모드 (0o600 등) → `@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows")` (`tests/hermes_cli/test_auth_toctou_file_modes.py` 참조)
- `signal.SIGALRM` → Unix 전용 (`tests/conftest.py::_enforce_test_timeout` 참조)
- 실제(Live) Winsock / Windows 전용 회귀 테스트 → `@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific regression")`

테스트 대상 코드가 `platform.system()` / `platform.release()` / `platform.mac_ver()`도 호출하는 경우 **`sys.platform`을 몽키패칭(Monkeypatching)하는 것만으로는 충분하지 않습니다**. 이 함수들은 실제 OS를 독립적으로 다시 읽기 때문에 Windows 러너에서 `sys.platform = "linux"`로 설정한 테스트라도 여전히 `platform.system() == "Windows"`를 보고 Windows 분기(branch)를 타게 됩니다. 세 가지를 모두 함께 패치하세요:

```python
monkeypatch.setattr(sys, "platform", "linux")
monkeypatch.setattr(platform, "system", lambda: "Linux")
monkeypatch.setattr(platform, "release", lambda: "6.8.0-generic")
```

실제 작동 예시는 `tests/agent/test_prompt_builder.py::TestEnvironmentHints`를 참조하세요.

### 시스템 프롬프트의 실행 환경 블록 확장

호스트 OS, 사용자 홈 디렉토리, 현재 작업 디렉토리(cwd), 터미널 백엔드, 그리고 쉘 (Windows의 경우 bash 대 PowerShell)에 대한 사실적인 가이드는 `agent/prompt_builder.py::build_environment_hints()`에서 내보냅니다. WSL 힌트 및 백엔드별 탐색(probe) 로직도 이곳에 있습니다. 규칙(convention)은 다음과 같습니다:

- **로컬 터미널 백엔드** → 호스트 정보 (OS, `$HOME`, cwd) + Windows 전용 참고 사항 (호스트 이름 ≠ 사용자 이름, `terminal` 도구는 PowerShell이 아닌 bash를 사용함)을 출력합니다.
- **원격 터미널 백엔드** (`docker, singularity, modal, daytona, ssh, managed_modal` 등 `_REMOTE_TERMINAL_BACKENDS`에 있는 모든 것) → 호스트 정보를 완전히 **숨기고** 백엔드에 대해서만 설명합니다. 백엔드 내부에서는 `tools.environments.get_environment(...).execute(...)`를 통해 실시간 `uname`/`whoami`/`pwd` 탐색이 실행되며 프로세스당 `_BACKEND_PROBE_CACHE`에 캐시되고, 만약 탐색이 타임아웃되면 정적 폴백(fallback)을 사용합니다.
- **프롬프트 작성을 위한 핵심 사실:** `TERMINAL_ENV != "local"`일 때, 모든 파일 도구(`read_file`, `write_file`, `patch`, `search_files`)는 호스트가 아닌 백엔드 컨테이너 내부에서 실행됩니다. 에이전트가 건드릴 수 없으므로 이 경우 시스템 프롬프트는 절대로 호스트에 대해 설명해서는 안 됩니다.

전체 설계 노트, 방출되는 정확한 문자열 및 테스트 시 주의 사항은 `references/prompt-builder-environment-hints.md`에 있습니다.

**리팩터링 안전성 패턴 (POSIX-equivalence guard):** 인라인(inline) 로직을 Windows/플랫폼 전용 동작을 추가하는 헬퍼 함수로 추출할 때, 테스트 파일에 예전 코드와 동일한 `_legacy_<name>` 오라클 함수를 유지하고 이를 기준으로 파라미터화된 차이점(parametrize-diff) 검사를 하세요. 예시: `tests/tools/test_code_execution_windows_env.py::TestPosixEquivalence`. 이렇게 하면 POSIX 동작이 비트 단위까지 완벽하게 동일하다는 불변성을 유지할 수 있고, 향후 발생할 수 있는 변화(drift)를 명확한 차이(diff)와 함께 크게 실패하게 만듭니다.

### 커밋 규칙 (Commit Conventions)

```
type: 간결한 제목 (concise subject line)

선택적인 본문 (Optional body).
```

유형(Types): `fix:`, `feat:`, `refactor:`, `docs:`, `chore:`

### 핵심 규칙 (Key Rules)

- **절대로 프롬프트 캐싱을 깨뜨리지 마세요** — 대화 중에 컨텍스트, 도구 또는 시스템 프롬프트를 변경하지 마세요.
- **메시지 역할(role) 교대** — 어시스턴트 메시지 두 개나 사용자 메시지 두 개가 연속으로 오면 안 됩니다.
- 프로필 사용 환경을 안전하게 지원하기 위해 모든 경로에 대해 `hermes_constants`의 `get_hermes_home()`을 사용하세요.
- 설정값은 `config.yaml`에, 비밀 정보는 `.env`에 넣습니다.
- 새로운 도구는 요구 사항이 충족될 때만 표시되도록 `check_fn`이 필요합니다.
