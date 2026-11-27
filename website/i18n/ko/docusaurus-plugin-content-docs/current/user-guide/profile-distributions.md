---
sidebar_position: 3
---

# 프로필 배포판: 에이전트 전체를 공유하기

**프로필 배포판**은 성격, 스킬, cron 작업, MCP 연결, 설정을 포함한 완전한 Hermes 에이전트를 git 저장소로 패키징합니다. 저장소에 접근할 수 있는 사람은 누구나 한 번의 명령으로 에이전트 전체를 설치하고, 제자리에서 업데이트하면서 자신의 메모리, 세션, API 키는 그대로 유지할 수 있습니다.

[프로필](./profiles.md)이 로컬 에이전트라면, 배포판은 공유할 수 있게 만든 해당 에이전트입니다.

## 프로필을 공유하는 두 가지 방법

Hermes에는 공유 경로가 두 가지 있으며, 각각 서로 다른 질문에 답합니다. 배포판은 지속적인 공유를 위한 방법이고, 내보내기 파일은 빠른 공유를 위한 방법입니다.

| | **배포판** (git 저장소) | **내보내기 파일** (`.tar.gz`) |
|---|---|---|
| 공유 방법 | `hermes profile install <repo>` | 파일 전송 — 채팅, AirDrop, USB, 이메일 |
| 수신자에게 필요한 것 | git 및 저장소 접근 권한 | 파일 |
| 업데이트 | `hermes profile update`로 새 버전을 가져옴 | 파일을 다시 전송 |
| 버전 관리 | 태그, 브랜치, 커밋 SHA | 없음 — 특정 시점의 스냅샷 |
| 작성자의 설정 비용 | `distribution.yaml` + `.gitignore` + 저장소 | 없음 — 한 번의 명령 |
| 포함 항목 | SOUL, 설정, 스킬, cron, MCP, 플러그인 | 동일한 항목 **및** 데스크톱 테마와 레이아웃 |
| 생성 방법 | `hermes profile install` / `update` | `/export` 및 `/import`, 또는 `hermes profile export` / `import` |

에이전트를 계속 개선할 제품으로 만들고 다른 사람들이 변경 사항을 따라오길 원한다면 **배포판**을 선택하세요. 검토된 팀 내부 에이전트, 커뮤니티 릴리스, 다섯 대의 컴퓨터에 배포할 동일한 에이전트가 그 예입니다.

지금 당장 누군가에게 자신의 설정을 전달하거나 새 노트북으로 옮기려는 경우에는 **내보내기 파일**을 선택하세요. 저장소나 매니페스트가 필요 없습니다. 채팅에서 `/export`를 실행해 파일을 전달하면 상대방은 `/import`를 실행하면 됩니다. [프로필 파일 내보내기 및 가져오기](#export-and-import-a-profile-file)를 참고하세요.

두 방법은 서로 배타적이지 않습니다. 많은 작성자가 프로필을 직접 사용해 보고 `/export`로 동료에게 2차 의견을 요청한 다음, 버전 관리할 가치가 생기면 배포판으로 공개합니다.

## 이것이 의미하는 것

배포판이 없던 때에는 Hermes 에이전트를 공유하려면 다음을 보내야 했습니다.

1. SOUL.md
2. 설치해야 할 스킬 목록
3. 비밀 정보를 제외한 config.yaml
4. 연결한 MCP 서버에 대한 설명
5. 예약한 cron 작업
6. 설정해야 할 env var에 대한 안내

…그리고 상대방이 올바르게 조립하기를 바랐습니다. 버전을 올리거나 버그를 수정할 때마다 인계를 반복해야 했습니다.

배포판을 사용하면 이 모든 것이 하나의 git 저장소에 들어갑니다.

```
my-research-agent/
├── distribution.yaml    # manifest: name, version, env-var requirements
├── SOUL.md              # the agent's personality / system prompt
├── config.yaml          # model, temperature, reasoning, tool defaults
├── skills/              # bundled skills that come with the agent
├── cron/                # scheduled tasks the agent runs
└── mcp.json             # MCP servers the agent connects to
```

수신자는 다음을 실행합니다.

```bash
hermes profile install github.com/you/my-research-agent --alias
```

…그러면 에이전트 전체를 갖게 됩니다. 자신의 API 키(`.env.EXAMPLE` → `.env`)를 입력한 뒤 `my-research-agent chat`을 실행하거나 Telegram / Discord / Slack / 기타 모든 게이트웨이 플랫폼을 통해 에이전트에 접근할 수 있습니다. 새 버전을 푸시하면 `hermes profile update my-research-agent`를 실행해 변경 사항을 가져오면 됩니다. 메모리와 세션은 그대로 유지됩니다.

## 왜 git인가요?

tarball, HTTP 아카이브, 사용자 지정 형식을 검토했지만 git보다 나은 방법은 없었습니다.

- **작성자에게 빌드 단계가 없습니다.** GitHub에 푸시하면 소비자가 설치합니다. “이것을 패키징하고, 저것을 업로드하고, 인덱스를 갱신하는” 과정이 없습니다.
- **태그, 브랜치, 커밋이 이미 버전 관리 시스템입니다.** 다른 도구에서 “패키징 + 릴리스 업로드”를 하는 일을 태그 푸시가 대신합니다.
- **업데이트는 fetch입니다.** 전체 아카이브를 다시 다운로드하지 않습니다.
- **투명합니다.** 사용자는 저장소를 둘러보고, 버전 간 차이를 읽고, 이슈를 열고, 커스터마이즈를 위해 포크할 수 있습니다.
- **비공개 저장소도 추가 비용 없이 작동합니다.** SSH 키, `git credential` 헬퍼, GitHub CLI에 저장된 자격 증명 등 터미널에 이미 설정된 인증이 투명하게 적용됩니다.
- **재현성은 커밋 SHA입니다.** pip와 npm이 기록하는 것과 같습니다.

단점은 수신자에게 git이 설치되어 있어야 한다는 점입니다. 2026년에 Hermes를 실행하는 모든 컴퓨터에는 이미 git이 설치되어 있을 것입니다.

## 언제 배포판을 사용해야 하나요?

잘 맞는 경우:

- **전문 에이전트를 공유하려는 경우** — 규정 준수 모니터, 코드 리뷰어, 연구 보조, 고객 지원 봇을 팀이나 커뮤니티와 공유할 때.
- **동일한 에이전트를 여러 컴퓨터에 배포하려는 경우** — 매번 파일을 수동으로 복사하고 싶지 않을 때.
- **에이전트를 반복 개선하는 경우** — 수신자가 한 번의 명령으로 새 버전을 받을 수 있게 하려 할 때.
- **에이전트를 제품으로 만드는 경우** — 다른 사람들이 출발점으로 사용할, 명확한 기본값과 선별된 스킬, 조정된 프롬프트를 제공할 때.

맞지 않는 경우:

- **지금 당장 한 번만 자신의 설정을 전달하려는 경우.** 배포판에는 저장소, 매니페스트, `.gitignore`가 필요합니다. `/export`에는 아무것도 필요하지 않습니다 — [프로필 파일 내보내기 및 가져오기](#export-and-import-a-profile-file)를 참고하세요. 프로필을 백업하거나 새 컴퓨터로 옮길 때도 마찬가지입니다.
- **데스크톱 테마와 레이아웃을 공유하려는 경우.** 배포판은 에이전트 — SOUL, 설정, 스킬, cron, MCP, 플러그인 — 을 담습니다. 데스크톱 앱에서 만든 내보내기에는 외형도 담깁니다. 스킨, 라이트/다크 모드, 사용자 지정 테마, 레일 색상, 창 레이아웃이 포함됩니다.
- **에이전트와 함께 API 키도 공유하려는 경우.** `auth.json`과 `.env`는 배포판에서 의도적으로 제외됩니다. 설치하는 사람마다 자신의 자격 증명을 가져옵니다. (내보내기 파일도 자격 증명을 제거합니다.)
- **메모리 / 세션 / 대화 기록을 공유하려는 경우.** 이는 배포판 콘텐츠가 아닌 사용자 데이터입니다. 절대 전송되지 않습니다. (내보내기 파일은 여기서 다릅니다 — 파일을 보내기 전에 [내보내기에 포함되는 내용](#what-an-export-file-contains)을 읽으세요.)

:::caution
**Hermes는 git을 제어하지 않습니다.** 이 페이지에서 설명하는 파일 제외는 누군가 `hermes profile install` 또는 `hermes profile update`를 실행할 때 **설치 프로그램**이 적용합니다. `git add`나 `git commit`을 실행할 때 적용되는 것이 아닙니다.
:::

## 생명주기: 작성자에서 설치, 업데이트까지

다음은 처음부터 끝까지의 전체 흐름입니다. 관심 있는 쪽을 선택하세요.

---

## 작성자를 위한 안내: 배포판 공개하기

### 1단계 — 작동하는 프로필에서 시작하기

다른 프로필과 마찬가지로 에이전트를 만들고 다듬습니다.

```bash
hermes profile create research-bot
research-bot setup                    # configure model, API keys
# Edit ~/.hermes/profiles/research-bot/SOUL.md
# Install skills, wire up MCP servers, schedule cron jobs, etc.
research-bot chat                     # dogfood until it feels right
```

### 2단계 — `distribution.yaml` 추가하기

`~/.hermes/profiles/research-bot/distribution.yaml`을 만드세요.

```yaml
name: research-bot
version: 1.0.0
description: "Autonomous research assistant with arXiv and web tools"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"

# Tell installers which env vars the agent needs. These are checked against
# the installer's shell and existing .env file so they don't get nagged
# about keys they already have configured.
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API key (for model access)"
    required: true
  - name: SERPAPI_KEY
    description: "SerpAPI key for web search"
    required: false
    default: ""
```

이것이 매니페스트의 전부입니다. `name`을 제외한 모든 필드에는 합리적인 기본값이 있습니다.

### 3단계 — 첫 커밋 전에 `.gitignore` 만들기

:::warning
`git init` 또는 `git add`를 실행하기 **전에** 이 작업을 하세요. 프로필과 대화했거나, 설정을 실행했거나, 그 밖의 방식으로 프로필을 사용했다면 이제 디렉터리에 전송해서는 안 되는 파일이 들어 있습니다: `.env`, `auth.json`, `memories/`, `sessions/`, `state.db*`, `logs/` 등입니다.
:::

최소한 다음 내용으로 `~/.hermes/profiles/research-bot/.gitignore`를 만드세요.

```gitignore
# Credentials & secrets — NEVER commit
auth.json
.env
.env.EXAMPLE    # generated by install, not authorship domain

# Runtime databases & state
state.db
state.db-shm
state.db-wal
hermes_state.db
response_store.db
response_store.db-shm
response_store.db-wal
gateway.pid
gateway_state.json
processes.json
auth.lock
active_profile
.update_check

# User data — NEVER commit
memories/
sessions/
logs/
plans/
workspace/
home/

# Caches & generated artifacts
image_cache/
audio_cache/
document_cache/
browser_screenshots/
cache/

# Infrastructure (should not be in profile dir, but safe to exclude)
hermes-agent/
.worktrees/
profiles/
bin/
node_modules/

# User customization namespace — your local overrides
local/

# Checkpoints & backups (can be huge)
checkpoints/
sandboxes/
backups/

# Logs
errors.log
.hermes_history
```

이는 설치 프로그램이 마지막 단계에서 제거하는 [강제 제외 경로](#whats-not-in-a-distribution-ever)를 그대로 반영합니다. 저장소에서 제외하고 싶은 다른 항목(임시 파일, 대형 자산, 로컬 전용 스킬 등)도 `.gitignore`에 추가하세요.

### 4단계 — git 저장소에 푸시하기

```bash
cd ~/.hermes/profiles/research-bot
git init
git add .
git commit -m "v1.0.0"
git remote add origin git@github.com:you/research-bot.git
git tag v1.0.0
git push -u origin main --tags
```

이제 저장소가 배포판입니다. 접근 권한이 있는 사람은 누구나 설치할 수 있습니다.

:::note
작성자가 실수로 저장소에 포함했더라도 설치 프로그램은 [강제 제외 경로](#whats-not-in-a-distribution-ever)를 추가로 제거합니다. 하지만 이는 설치 프로그램만 보호할 뿐 작성자는 보호하지 않습니다.
:::

### 5단계 — 버전이 지정된 릴리스에 태그 달기

에이전트가 안정적인 지점에 도달할 때마다 버전을 올리고 태그를 다세요.

```bash
# Edit distribution.yaml: version: 1.1.0
git add distribution.yaml SOUL.md skills/
git commit -m "v1.1.0: tighter research SOUL, add arxiv skill"
git tag v1.1.0
git push --tags
```

`hermes profile update research-bot`을 실행하는 수신자는 최신 버전을 가져옵니다.

### 저장소의 모습

완전히 작성된 배포판은 다음과 같습니다.

```
research-bot/
├── .gitignore                   # excludes secrets & user data (see Step 3)
├── distribution.yaml            # required
├── SOUL.md                      # strongly recommended
├── config.yaml                  # model, provider, tool defaults
├── mcp.json                     # MCP server connections
├── skills/
│   ├── arxiv-search/SKILL.md
│   ├── paper-summarization/SKILL.md
│   └── citation-lookup/SKILL.md
├── cron/
│   └── weekly-digest.json       # scheduled tasks
└── README.md                    # human-facing description (optional)
```

### 배포판 소유 항목과 사용자 소유 항목

설치 프로그램이 새 버전으로 업데이트할 때 일부 항목은 교체되고(작성자 영역), 일부 항목은 그대로 유지됩니다(설치자 영역). 기본값은 다음과 같습니다.

| 범주 | 경로 | 업데이트 시 |
|---|---|---|
| **배포판 소유** | `SOUL.md`, `config.yaml`, `mcp.json`, `skills/`, `cron/`, `distribution.yaml` | 새 복제본으로 교체 |
| **설정 재정의** | `config.yaml` | 실제로는 기본적으로 보존 — 설치 프로그램이 모델이나 프로바이더를 조정했을 수 있음. 재설정하려면 업데이트 시 `--force-config` 전달 |
| **사용자 소유** | `memories/`, `sessions/`, `state.db*`, `auth.json`, `.env`, `logs/`, `workspace/`, `plans/`, `home/`, `*_cache/`, `local/` | 절대 건드리지 않음 |

매니페스트에서 배포판 소유 목록을 재정의할 수 있습니다.

```yaml
distribution_owned:
  - SOUL.md
  - skills/research/            # only my research skills; other installed skills stay
  - cron/digest.json
```

생략하면 위의 기본값이 적용됩니다. 대부분의 배포판에는 이 설정이 적합합니다.

---

## 설치자를 위한 안내: 배포판 사용하기

### 설치

```bash
hermes profile install github.com/you/research-bot --alias
```

진행되는 작업은 다음과 같습니다.

1. 저장소를 임시 디렉터리에 복제합니다.
2. `distribution.yaml`을 읽고 매니페스트(이름, 버전, 설명, 작성자, 필수 env var)를 표시합니다.
3. 각 필수 env var를 셸 환경과 대상 프로필의 기존 `.env`에서 확인합니다. 무엇을 설정해야 하는지 정확히 알 수 있도록 각각을 `✓ set` 또는 `needs setting`으로 표시합니다.
4. 확인을 요청합니다. 건너뛰려면 `-y` / `--yes`를 전달하세요.
5. 배포판 소유 파일을 `~/.hermes/profiles/research-bot/`(또는 매니페스트의 `name`이 결정하는 경로)에 복사합니다. 작성자가 실수로 저장소에 남겨 둔 경우에도 [강제 제외 경로](#whats-not-in-a-distribution-ever)는 복사 중 제거됩니다.
6. 필수 키가 주석 처리된 `.env.EXAMPLE`을 작성합니다 — 이를 `.env`로 복사하고 값을 입력하세요.
7. `--alias`를 사용하면 `research-bot chat`을 직접 실행할 수 있도록 래퍼를 만듭니다.

### 소스 유형

모든 git URL을 사용할 수 있습니다.

```bash
# GitHub shorthand
hermes profile install github.com/you/research-bot

# Full HTTPS
hermes profile install https://github.com/you/research-bot.git

# SSH
hermes profile install git@github.com:you/research-bot.git

# Self-hosted, GitLab, Gitea, Forgejo — any Git host
hermes profile install https://git.example.com/team/research-bot.git

# Private repo using your configured git auth
hermes profile install git@github.com:your-org/internal-bot.git

# Local directory during development (no git push needed)
hermes profile install ~/my-profile-in-progress/
```

### 프로필 이름 재정의

동일한 배포판을 서로 다른 프로필 이름으로 사용하려는 두 사용자를 생각해 봅시다.

```bash
# Alice
hermes profile install github.com/acme/support-bot --name support-us --alias
# Bob (same distribution, different local name)
hermes profile install github.com/acme/support-bot --name support-eu --alias
```

### env var 입력

설치가 끝나면 에이전트 프로필에 `.env.EXAMPLE`이 들어 있습니다.

```
# Environment variables required by this Hermes distribution.
# Copy to `.env` and fill in your own values before running.

# OpenAI API key (for model access)
# (required)
OPENAI_API_KEY=

# SerpAPI key for web search
# (optional)
# SERPAPI_KEY=
```

복사하세요.

```bash
cp ~/.hermes/profiles/research-bot/.env.EXAMPLE ~/.hermes/profiles/research-bot/.env
# Edit .env, paste your real keys
```

셸 환경에 이미 있던 필수 키(예: `~/.zshrc`에서 내보낸 `OPENAI_API_KEY`)는 설치 중 `✓ set`으로 표시됩니다. `.env`에 중복해서 넣을 필요가 없습니다.

### 설치한 항목 확인

```bash
hermes profile info research-bot
```

다음과 같이 표시됩니다.

```
Distribution: research-bot
Version:      1.0.0
Description:  Autonomous research assistant with arXiv and web tools
Author:       Your Name
Requires:     Hermes >=0.12.0
Source:       https://github.com/you/research-bot
Installed:    2026-05-08T17:04:32+00:00

Environment variables:
  OPENAI_API_KEY (required) — OpenAI API key (for model access)
  SERPAPI_KEY (optional) — SerpAPI key for web search
```

`hermes profile list`에는 `Distribution` 열도 표시됩니다. 저장소에서 가져온 프로필과 직접 만든 프로필을 한눈에 구분할 수 있습니다.

```
 Profile          Model                        Gateway      Alias        Distribution
 ───────────────    ───────────────────────────    ───────────    ───────────    ────────────────────
 ◆default         claude-sonnet-4              stopped      —            —
  coder           gpt-5                        stopped      coder        —
  research-bot    claude-opus-4                stopped      research-bot research-bot@1.0.0
  telemetry       claude-sonnet-4              running      telemetry    telemetry@2.3.1
```

### 업데이트

```bash
hermes profile update research-bot
```

진행되는 작업은 다음과 같습니다.

1. 기록된 소스 URL에서 저장소를 다시 복제합니다.
2. 배포판 소유 파일(SOUL, 스킬, cron, mcp.json)을 교체합니다.
3. **config.yaml은 보존합니다** — 모델, temperature 또는 기타 설정을 조정했을 수 있기 때문입니다. 덮어쓰려면 `--force-config`를 전달하세요.
4. **사용자 데이터는 절대 건드리지 않습니다.** 메모리, 세션, auth, `.env`, 로그, 상태가 그대로 유지됩니다.

전체 아카이브를 다시 다운로드하지 않습니다. 설정에 대한 로컬 변경 사항을 덮어쓰지 않습니다. 대화 기록을 삭제하지 않습니다.

### 제거

```bash
hermes profile delete research-bot
```

삭제 확인을 요청하기 전에 삭제 프롬프트에 배포판 정보가 표시됩니다.

```
Profile: research-bot
Path:    ~/.hermes/profiles/research-bot
Model:   claude-opus-4 (anthropic)
Skills:  12
Distribution: research-bot@1.0.0
Installed from: https://github.com/you/research-bot

This will permanently delete:
  • All config, API keys, memories, sessions, skills, cron jobs
  • Command alias (~/.local/bin/research-bot)

Type 'research-bot' to confirm:
```

따라서 에이전트의 출처를 모르거나 다시 설치할 수 없는 상태에서 실수로 에이전트를 삭제하는 일이 없습니다.

---

## 사용 사례와 패턴

### 개인 사용: 여러 컴퓨터에서 하나의 에이전트 동기화

노트북에서 연구 보조 에이전트를 만들었고 워크스테이션에서도 같은 에이전트를 사용하려고 합니다.

```bash
# Laptop — create .gitignore first (see "For authors" Step 3), then:
cd ~/.hermes/profiles/research-bot
git init && git add . && git status   # confirm no secrets staged
git commit -m "initial"
git remote add origin git@github.com:you/research-bot.git
git push -u origin main

# Workstation
hermes profile install github.com/you/research-bot --alias
# Fill in .env. Done.
```

노트북에서 변경한 내용(`git commit && push`)은 워크스테이션에서 `hermes profile update research-bot`으로 가져옵니다. 메모리는 컴퓨터별로 유지됩니다. 노트북은 자신의 대화를 기억하고 워크스테이션은 자신의 대화를 기억하므로 서로 충돌하지 않습니다.

### 팀 사용: 검토된 내부 에이전트 배포

엔지니어링 팀에서 특정 SOUL과 스킬, 모든 PR을 처리하는 cron이 포함된 공용 PR 리뷰 봇을 사용하려고 합니다.

```bash
# Engineering lead — create .gitignore first (see "For authors" Step 3), then:
cd ~/.hermes/profiles/pr-reviewer
# ... build and tune ...
git init && git add . && git status   # confirm no secrets staged
git commit -m "v1.0 PR reviewer"
git tag v1.0.0
git push -u origin main --tags    # push to your company's internal Git host

# Each engineer
hermes profile install git@github.com:your-org/pr-reviewer.git --alias
# Fill in .env with their own API key (billed to them), .env.EXAMPLE points at what's required
pr-reviewer chat
```

팀 리더가 v1.1(더 나은 SOUL, 새 스킬)을 배포하면 엔지니어들은 `hermes profile update pr-reviewer`를 실행해 몇 분 안에 새 버전을 사용할 수 있습니다.

### 커뮤니티 사용: 공개 에이전트 게시

새로운 것을 만들었습니다. 예를 들면 “Polymarket 트레이더”, “학술 논문 요약기”, “Minecraft 서버 운영 보조” 같은 것입니다. 이를 공유하려 합니다.

```bash
# You — create .gitignore first (see "For authors" Step 3), then:
cd ~/.hermes/profiles/polymarket-trader
# Write a solid README.md at the repo root — GitHub shows it on the repo page
git init && git add . && git status   # confirm no secrets staged
git commit -m "v1.0"
git tag v1.0.0
# Publish to a public GitHub repo
git remote add origin https://github.com/you/hermes-polymarket-trader.git
git push -u origin main --tags

# Anyone
hermes profile install github.com/you/hermes-polymarket-trader --alias
```

설치 명령을 트윗하세요. 사용해 본 사람들은 이슈와 PR을 보내고, 커스터마이즈하려는 사람은 포크할 수 있습니다. 모두가 이미 알고 있는 동일한 git 워크플로를 사용합니다.

### 제품 사용: 명확한 방향성을 가진 에이전트 출시

Hermes를 기반으로 무언가를 만들었습니다. 규정 준수 모니터링 도구, 고객 지원 스택, 특정 분야 연구 플랫폼 등이 될 수 있습니다. 이를 제품으로 배포하려 합니다.

```yaml
# distribution.yaml
name: telemetry-harness
version: 2.3.1
description: "Compliance telemetry harness — monitors and reviews regulated workflows"
hermes_requires: ">=0.13.0"
author: "Acme Compliance Inc."
license: "Commercial"

env_requires:
  - name: ACME_API_KEY
    description: "Your Acme Compliance license key (email support@acme.com)"
    required: true
  - name: OPENAI_API_KEY
    description: "OpenAI API key for model access"
    required: true
  - name: GRAPHITI_MCP_URL
    description: "URL for your Graphiti knowledge graph instance"
    required: false
    default: "http://127.0.0.1:8000/sse"
```

고객은 한 번의 명령으로 설치할 수 있습니다. 설치 미리보기에는 준비해야 할 키가 정확히 표시되고, 새 릴리스에 태그를 다는 즉시 업데이트가 배포됩니다. 고객의 규정 준수 데이터(`memories/`, `sessions/`)는 고객의 컴퓨터 밖으로 나가지 않습니다.

### 임시 사용: 공유 인프라에서 일회성 스크립트 실행

운영 책임자라고 가정해 봅시다. 올바른 도구와 MCP 연결이 포함된 미리 작성된 SOUL로 운영 장애를 진단하는 임시 에이전트를 만들고, 다음 일주일 동안 당직 엔지니어 세 명의 노트북에서 실행하려 합니다.

```bash
# You — create .gitignore first (see "For authors" Step 3), then:
# Build the profile, commit, push a private repo
git push -u origin main

# Each on-call
hermes profile install git@github.com:your-org/incident-2026-q2.git --alias

# Incident resolved — tear it down
hermes profile delete incident-2026-q2
```

설치와 삭제를 반복하는 과정이 일회용으로 사용하기에 충분히 간편합니다.

---

## 레시피

### 특정 버전에 고정하기

:::note
git ref 고정(` + "`#v1.2.0`" + `)은 계획되어 있지만 초기 릴리스에는 포함되지 않았습니다. 현재 설치는 기본 브랜치를 추적합니다. 설치된 버전은 `hermes profile info <name>`으로 확인하고, 준비가 될 때까지 업데이트를 보류하세요.
:::

### 현재 버전과 최신 버전 비교하기

```bash
# Your installed version
hermes profile info research-bot | grep Version

# Latest upstream (without installing)
git ls-remote --tags https://github.com/you/research-bot | tail -5
```

### 업데이트 중 로컬 설정 커스터마이즈 유지하기

기본 업데이트 동작은 이미 이를 지원합니다. `config.yaml`은 보존됩니다. 안전을 위해 배포판이 소유하지 않는 파일에 로컬 변경 사항을 기록하세요.

```yaml
# ~/.hermes/profiles/research-bot/local/my-overrides.yaml
# (distribution never touches local/)
```

…그리고 필요에 따라 `config.yaml` 또는 SOUL에서 이를 참조하세요.

### 완전히 새로 설치하기

```bash
# Nuke and re-install from scratch (loses memories/sessions too)
hermes profile delete research-bot --yes
hermes profile install github.com/you/research-bot --alias

# Update to current main but reset config.yaml to the distribution's default
hermes profile update research-bot --force-config --yes
```

### 포크하고 커스터마이즈하기

표준 git 워크플로를 사용하면 됩니다. 배포판도 저장소일 뿐입니다.

```bash
# Fork the repo on GitHub, then install your fork
hermes profile install github.com/yourname/forked-research-bot --alias

# Iterate locally in ~/.hermes/profiles/forked-research-bot/
# Edit SOUL.md, commit, push to your fork
# Upstream changes: pull them into your fork the usual way
```

### 푸시 전에 배포판 테스트하기

작성자의 컴퓨터에서 다음을 실행합니다.

```bash
# Install from a local directory (no git push needed)
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test --alias

# Tweak, delete, re-install until it's right
hermes profile delete research-bot-test --yes
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test
```

---

## 프로필 파일 내보내기 및 가져오기

버전 관리가 필요하지 않다면 저장소를 건너뛰세요. `/export`는 프로필을 하나의 `.tar.gz`로 패키징하고, `/import`는 반대편에서 압축을 풀어 새 프로필로 만듭니다. 자격 증명은 내보낼 때 제거됩니다.

### 내보내기

CLI, TUI 또는 데스크톱 채팅에서 실행합니다.

```
/export                          # the active profile → <name>.tar.gz
/export research-bot             # a named profile
/export research-bot -o ~/Desktop/research-bot.tar.gz
```

또는 셸에서 동일한 기능을 실행할 수 있습니다.

```bash
hermes profile export research-bot
hermes profile export research-bot -o ./research-bot.tar.gz
```

**데스크톱 앱**에는 세 가지 진입점이 있으며, 모두 운영체제의 저장 대화상자로 연결됩니다.

- **⌘K → 프로필 내보내기…**
- 사이드바 레일에서 프로필 사각형을 마우스 오른쪽 버튼으로 클릭 → **프로필 내보내기…**
- 레일의 **+** 옆 가져오기 버튼으로 반대 방향 작업 수행

데스크톱 내보내기에는 CLI에 없는 파일 하나가 추가됩니다. `desktop.json`에는 스킨, 라이트/다크 모드, 스킨에 필요한 사용자 지정 테마 정의, 프로필의 레일 색상, 창 레이아웃이 들어 있습니다. 따라서 데스크톱에서 공유한 프로필은 단지 같은 방식으로 동작하는 것뿐 아니라, 내 것과 같은 모습으로 도착합니다.

### 가져오기

```
/import ~/Downloads/research-bot.tar.gz
/import ~/Downloads/research-bot.tar.gz --name research-bot-2
```

```bash
hermes profile import ./research-bot.tar.gz
hermes profile import ./research-bot.tar.gz --name research-bot-2
```

`--name`을 전달하지 않으면 아카이브에서 프로필 이름을 추론합니다. 기존 프로필 위로 가져오기는 거부됩니다. 먼저 기존 프로필의 이름을 바꾸거나 삭제하세요. 이름이 기존 명령과 충돌하지 않을 때는 셸 래퍼(`research-bot` → `hermes -p research-bot`)가 생성됩니다.

데스크톱 앱에서 가져오면 `desktop.json` 오버레이도 적용되고, 새 채팅이 열린 새 프로필로 이동합니다. 데스크톱에서 만든 아카이브를 CLI에서 가져와도 괜찮습니다. 오버레이 파일은 디스크에 함께 남아 있다가 다음에 데스크톱에서 해당 프로필을 열 때 적용됩니다.

:::note
`default`로 가져올 수 없습니다. 이 이름은 기본 루트 프로필(`~/.hermes`)에 사용됩니다. `--name something-else`를 전달하세요.
:::

### 내보내기 파일에 포함되는 내용

두 프로필 유형 모두에서 항상 제외되는 항목은 `auth.json`과 `.env`입니다. API 키는 절대 컴퓨터 밖으로 나가지 않습니다.

**기본 프로필**(`~/.hermes`)은 허용 목록을 통해 내보내집니다. 따라서 홈 디렉터리에 놓인 관련 없는 파일이 함께 포함되지 않습니다. 허용 목록에는 `config.yaml`, `SOUL.md`, `MEMORY.md`, `USER.md`, `todo.json`, `system_prompt.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `skills/`, `plugins/`, `cron/`, `scripts/`, `sessions/`, `memories/`, `knowledge/`, `preferences/`, 그리고 데스크톱에서 준비한 경우의 `desktop.json`이 있습니다.

**이름이 지정된 프로필**(`~/.hermes/profiles/<name>`)은 `auth.json` / `.env`를 제외하고 디렉터리 전체를 복사합니다. 범위가 더 넓으므로 프로필에 `state.db`, 로그 또는 캐시가 있다면 아카이브에도 포함되어 파일이 커집니다.

:::caution 아카이브를 보내기 전에 읽어 보세요
내보내기는 선별된 릴리스가 아니라 프로필의 스냅샷입니다. 배포판과 달리 **`memories/`, `sessions/`, `USER.md`가 포함될 수 있습니다.** 또한 스킬, 메모리 또는 페르소나에 직접 작성한 개인 정보가 있는지 검사하지 않습니다. 자격 증명은 파일 이름으로 필터링되지만, 콘텐츠는 필터링되지 않습니다.

다른 사람과 공유하기 전에 내부 항목을 나열하세요.

```bash
tar -tzf research-bot.tar.gz | less
```

넘기고 싶지 않은 대화 기록이 포함되어 있다면 대신 [배포판](#for-authors-publishing-a-distribution)을 공개하세요. 배포판에는 메모리나 세션이 절대 포함되지 않습니다.
:::

## 배포판에 절대 포함되지 않는 것

작성자가 실수로 포함했더라도 설치 프로그램은 다음 경로를 강제로 제외합니다. 설정으로 이 동작을 재정의할 수 없습니다. 이 안전 장치는 회귀 테스트된 불변 조건입니다.

- `auth.json` — OAuth 토큰, 플랫폼 자격 증명
- `.env` — API 키, 비밀 정보
- `memories/` — 대화 메모리
- `sessions/` — 대화 기록
- `state.db`, `state.db-shm`, `state.db-wal` — 세션 메타데이터
- `logs/` — 에이전트 및 오류 로그
- `workspace/` — 생성된 작업 파일
- `plans/` — 임시 계획
- `home/` — Docker 백엔드에서 사용자의 홈 마운트
- `*_cache/` — 이미지 / 오디오 / 문서 캐시
- `local/` — 사용자가 예약한 커스터마이즈 네임스페이스

설치자가 배포판을 복제하면 이러한 항목은 프로필 디렉터리에 복사되지 않습니다. 업데이트해도 기존 사본은 그대로 유지됩니다. 동일한 배포판을 다섯 대의 컴퓨터에 설치했다면 이 데이터도 컴퓨터별로 다섯 개의 격리된 집합으로 존재합니다.

:::caution
이 제외는 **설치자의 컴퓨터에서 설치 / 업데이트할 때** 실행됩니다. 작성자가 민감하거나 불필요한 파일을 커밋하는 것을 막지는 **않습니다**. 작성자는 [`.gitignore`](#step-3--create-a-gitignore-before-the-first-commit)를 사용해 비밀 정보가 저장소에 들어가지 않도록 해야 합니다.
:::

## 보안과 신뢰

프로필 배포판은 기본적으로 서명되지 않습니다. 다음을 신뢰해야 합니다.

- **git 호스트**(GitHub / GitLab / 기타 호스트)가 작성자가 푸시한 바이트를 그대로 제공한다는 것.
- **작성자**가 악성 SOUL, 스킬 또는 cron 작업을 포함하지 않는다는 것.

배포판의 cron 작업은 **자동으로 예약되지 않습니다**. 설치 프로그램은 `hermes -p <name> cron list`를 출력하며, 사용자가 명시적으로 활성화해야 합니다. SOUL.md와 스킬은 프로필과 대화를 시작하는 즉시 활성화되므로, 모르는 사람이 만든 배포판을 설치한다면 첫 실행 전에 내용을 읽으세요.

대략적으로 말해 배포판 설치는 브라우저 확장 프로그램이나 VS Code 확장 프로그램 설치와 비슷합니다. 마찰은 적고 권한은 크므로 출처를 신뢰해야 합니다. 회사 내부 배포판에는 비공개 저장소와 평소의 git 인증을 사용하세요. 새로 설정할 것은 없습니다.

향후 버전에는 서명, 해석된 커밋 SHA가 포함된 잠금 파일(`.distribution-lock.yaml`), 업데이트 적용 전에 차이를 출력하는 `--dry-run` 플래그가 추가될 수 있습니다. 아직 이 중 어느 것도 제공되지 않습니다.

## 내부 동작

구현 세부 정보, 정확한 CLI 동작 및 모든 플래그는 [프로필 명령어 참조](../reference/profile-commands.md#distribution-commands)를 참고하세요.

요약하면 다음과 같습니다.

- `install`, `update`, `info`는 별도의 명령 트리가 아니라 `hermes profile` 안에 있습니다.
- 매니페스트 형식은 필수 스키마가 아주 작은(`name` 하나) YAML입니다.
- 설치 프로그램은 로컬 `git` 바이너리를 사용해 복제하므로, 셸에서 이미 처리하는 인증(SSH 키, 자격 증명 헬퍼)이 투명하게 작동합니다.
- 복제 후 `.git/`이 제거됩니다. 설치된 프로필 자체가 git 체크아웃이 되지 않으므로 “이런, 실수로 `.env`를 배포판의 git 기록에 커밋했네”와 같은 사고를 피할 수 있습니다.
- 예약된 프로필 이름(`hermes`, `test`, `tmp`, `root`, `sudo`)은 흔한 바이너리와의 충돌을 막기 위해 설치 시 거부됩니다.

## 함께 보기

- [프로필: 여러 에이전트 실행하기](./profiles.md) — 기본 개념
- [프로필 명령어 참조](../reference/profile-commands.md) — 모든 플래그와 옵션
- [`hermes profile export` / `import`](../reference/profile-commands.md#hermes-profile-export) — [내보내기 파일](#export-and-import-a-profile-file)의 CLI 형식
- [슬래시 명령어 참조](../reference/slash-commands.md) — `/export`, `/import` 및 채팅에서 사용하는 모든 명령어
- [Hermes에서 SOUL 사용하기](../guides/use-soul-with-hermes.md) — 성격 작성
- [성격 및 SOUL](./features/personality.md) — SOUL이 에이전트에 포함되는 방식
- [스킬 카탈로그](../reference/skills-catalog.md) — 번들로 제공할 수 있는 스킬
