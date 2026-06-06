---
sidebar_position: 3
---

# 프로필 배포 (Profile Distributions): 전체 에이전트 공유하기

**프로필 배포(profile distribution)**는 성격(personality), 스킬, 크론(cron) 작업, MCP 연결, 구성을 하나의 git 저장소로 패키징하여 완전한 Hermes 에이전트로 만듭니다. 저장소에 접근할 수 있는 사람은 누구나 명령어 하나로 전체 에이전트를 설치하고, 현재 위치에서 업데이트하며, 자체 메모리, 세션 및 API 키를 그대로 유지할 수 있습니다.

[프로필(profile)](./profiles.md)이 로컬 에이전트라면, 배포(distribution)는 공유 가능하게 만든 에이전트입니다.

## 이것이 의미하는 것

배포 기능 이전에는 Hermes 에이전트를 공유하려면 누군가에게 다음을 보내야 했습니다:

1. 사용자의 SOUL.md
2. 설치할 스킬 목록
3. 비밀 정보가 제외된 config.yaml
4. 연결한 MCP 서버에 대한 설명
5. 예약한 크론 작업
6. 설정할 환경 변수에 대한 지침

…그리고 그들이 올바르게 조립하기를 바라야 했습니다. 버전 변경이나 버그 수정이 있을 때마다 이 전달 과정을 반복해야 했습니다.

배포 기능을 사용하면 이 모든 것이 하나의 git 저장소에 존재합니다:

```
my-research-agent/
├── distribution.yaml    # 매니페스트: 이름, 버전, 환경 변수 요구 사항
├── SOUL.md              # 에이전트의 성격 / 시스템 프롬프트
├── config.yaml          # 모델, 온도, 추론, 도구 기본값
├── skills/              # 에이전트와 함께 제공되는 번들 스킬
├── cron/                # 에이전트가 실행하는 예약된 작업
└── mcp.json             # 에이전트가 연결되는 MCP 서버
```

수신자는 다음을 실행합니다:

```bash
hermes profile install github.com/you/my-research-agent --alias
```

…그러면 그들은 전체 에이전트를 갖게 됩니다. 그들은 자신의 API 키를 채워 넣고(`.env.EXAMPLE` → `.env`), `my-research-agent chat`을 실행하거나 Telegram / Discord / Slack / 모든 게이트웨이 플랫폼을 통해 에이전트와 대화할 수 있습니다. 새 버전을 푸시하면 수신자는 `hermes profile update my-research-agent`를 실행하여 변경 사항을 가져오며, 그들의 메모리와 세션은 그대로 유지됩니다.

## 왜 git인가요?

우리는 tarball, HTTP 아카이브, 사용자 지정 형식 등을 고려했지만, 어떤 것도 git을 능가하지 못했습니다:

- **작성자를 위한 빌드 단계 제로.** GitHub에 푸시하면 소비자가 설치합니다. "이것을 압축하고, 저것을 업로드하고, 인덱스를 업데이트하는" 반복이 없습니다.
- **태그, 브랜치, 커밋이 이미 버전 관리 시스템입니다.** 태그 푸시가 다른 도구에서 "릴리스 압축 + 업로드"가 하는 일을 대신합니다.
- **업데이트는 fetch입니다.** 전체 아카이브를 다시 다운로드하지 않습니다.
- **투명함.** 사용자는 저장소를 탐색하고, 버전 간의 차이점을 읽고, 해당 버전에 대한 이슈를 열거나, 사용자 지정을 위해 포크(fork)할 수 있습니다.
- **프라이빗 저장소가 무료로 작동합니다.** SSH 키, `git credential` 헬퍼, GitHub CLI 저장 자격 증명 등 이미 터미널에 설정된 모든 인증이 투명하게 적용됩니다.
- **재현성은 커밋 SHA입니다.** pip 및 npm이 기록하는 것과 같습니다.

단점은 수신자에게 git이 설치되어 있어야 한다는 점입니다. 2026년에 Hermes를 실행하는 시스템이라면 이는 이미 충족된 조건입니다.

## 배포를 언제 사용해야 할까요?

적합한 경우:

- **특수화된 에이전트를 공유할 때** — 규정 준수 모니터, 코드 리뷰어, 연구 보조원, 고객 지원 봇 등 — 팀이나 커뮤니티와 공유할 때.
- **동일한 에이전트를 여러 머신에 배포할 때** 매번 수동으로 파일을 복사하고 싶지 않을 때.
- **에이전트를 반복해서 개선할 때** 수신자가 명령어 하나로 새 버전을 가져오길 원할 때.
- **에이전트를 제품으로 구축할 때** — 다른 사람들이 시작점으로 사용해야 하는 선별된 기본값, 엄선된 스킬, 조정된 프롬프트를 포함할 때.

적합하지 않은 경우:

- **자신의 머신에서 프로필을 백업하고 싶을 때.** [`hermes profile export` / `import`](../reference/profile-commands.md#hermes-profile-export)를 사용하세요 — 이것이 그 기능의 목적입니다.
- **에이전트와 함께 API 키를 공유하고 싶을 때.** `auth.json`과 `.env`는 배포에서 의도적으로 제외됩니다. 각 설치자는 자신의 자격 증명을 가져옵니다.
- **메모리 / 세션 / 대화 기록을 공유하고 싶을 때.** 이것은 배포 콘텐츠가 아닌 사용자 데이터입니다. 절대 배포되지 않습니다.

## 라이프사이클: 작성자부터 설치자, 업데이트까지

아래는 전체 엔드투엔드(end-to-end) 흐름입니다. 관심 있는 쪽을 확인하세요.

---

## 작성자용: 배포 생성 및 게시

### 1단계 — 작동하는 프로필에서 시작

다른 프로필과 마찬가지로 에이전트를 구축하고 개선합니다:

```bash
hermes profile create research-bot
research-bot setup                    # 모델, API 키 구성
# ~/.hermes/profiles/research-bot/SOUL.md 편집
# 스킬 설치, MCP 서버 연결, 크론 작업 예약 등
research-bot chat                     # 올바르게 작동할 때까지 직접 사용하며 테스트(dogfood)
```

### 2단계 — `distribution.yaml` 추가

`~/.hermes/profiles/research-bot/distribution.yaml`을 생성합니다:

```yaml
name: research-bot
version: 1.0.0
description: "arXiv 및 웹 도구가 있는 자율 연구 보조원"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"

# 에이전트에 필요한 환경 변수를 설치자에게 알립니다.
# 이미 구성된 키에 대해 알림을 받지 않도록
# 설치자의 쉘 및 기존 .env 파일과 대조하여 확인합니다.
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API 키 (모델 액세스용)"
    required: true
  - name: SERPAPI_KEY
    description: "웹 검색을 위한 SerpAPI 키"
    required: false
    default: ""
```

이것이 전체 매니페스트입니다. `name`을 제외한 모든 필드에는 합리적인 기본값이 있습니다.

### 3단계 — git 저장소로 푸시

```bash
cd ~/.hermes/profiles/research-bot
git init
git add .
git commit -m "v1.0.0"
git remote add origin git@github.com:you/research-bot.git
git tag v1.0.0
git push -u origin main --tags
```

이제 저장소는 배포판입니다. 액세스 권한이 있는 사람은 누구나 설치할 수 있습니다.

:::note
git 저장소에는 **배포에서 이미 제외된 항목을 제외한 프로필 디렉토리의 모든 항목**이 포함됩니다: `auth.json`, `.env`, `memories/`, `sessions/`, `state.db*`, `logs/`, `workspace/`, `*_cache/`, `local/`. 이들은 컴퓨터에 남아 있습니다. 추가 경로를 제외하려는 경우 `.gitignore`를 추가할 수도 있습니다.
:::

### 4단계 — 버전 릴리스 태그

에이전트가 안정적인 지점에 도달할 때마다 버전을 올리고 태그를 지정합니다:

```bash
# distribution.yaml 편집: version: 1.1.0
git add distribution.yaml SOUL.md skills/
git commit -m "v1.1.0: tighter research SOUL, add arxiv skill"
git tag v1.1.0
git push --tags
```

`hermes profile update research-bot`을 실행하는 수신자는 최신 버전을 가져오게 됩니다.

### 저장소의 모습

완성된 배포판:

```
research-bot/
├── distribution.yaml            # 필수
├── SOUL.md                      # 강력 권장
├── config.yaml                  # 모델, 제공자, 도구 기본값
├── mcp.json                     # MCP 서버 연결
├── skills/
│   ├── arxiv-search/SKILL.md
│   ├── paper-summarization/SKILL.md
│   └── citation-lookup/SKILL.md
├── cron/
│   └── weekly-digest.json       # 예약된 작업
└── README.md                    # 사용자를 위한 설명 (선택 사항)
```

### 배포 소유(Distribution-owned) vs 사용자 소유(User-owned)

설치자가 새 버전으로 업데이트하면 일부는 교체되고(작성자 도메인), 일부는 그대로 유지됩니다(설치자 도메인). 기본값:

| 범주 | 경로 | 업데이트 시 |
|---|---|---|
| **배포 소유** | `SOUL.md`, `config.yaml`, `mcp.json`, `skills/`, `cron/`, `distribution.yaml` | 새 복제본(clone)으로 교체됨 |
| **구성 재정의** | `config.yaml` | 기본적으로 보존됨 — 설치자가 모델이나 제공자를 조정했을 수 있음. 재설정하려면 업데이트 시 `--force-config` 전달. |
| **사용자 소유** | `memories/`, `sessions/`, `state.db*`, `auth.json`, `.env`, `logs/`, `workspace/`, `plans/`, `home/`, `*_cache/`, `local/` | 절대 건드리지 않음 |

매니페스트에서 배포 소유 목록을 재정의할 수 있습니다:

```yaml
distribution_owned:
  - SOUL.md
  - skills/research/            # 내 연구 스킬만; 설치된 다른 스킬은 유지됨
  - cron/digest.json
```

생략하면 위의 기본값이 적용되며, 이는 대부분의 배포판이 원하는 것입니다.

---

## 설치자용: 배포판 사용하기

### 설치

```bash
hermes profile install github.com/you/research-bot --alias
```

발생하는 일:

1. 임시 디렉토리로 저장소를 복제합니다.
2. `distribution.yaml`을 읽고 매니페스트(이름, 버전, 설명, 작성자, 필요한 환경 변수)를 표시합니다.
3. 대상 프로필의 기존 `.env`와 쉘 환경에 대해 각각 필요한 환경 변수를 확인합니다. 정확히 무엇을 구성해야 하는지 알 수 있도록 각각 `✓ set` 또는 `needs setting`으로 표시합니다.
4. 확인을 요청합니다. 건너뛰려면 `-y` / `--yes`를 전달하세요.
5. 배포 소유 파일을 `~/.hermes/profiles/research-bot/` (또는 매니페스트의 `name`이 확인되는 위치)에 복사합니다.
6. 필수 키가 주석 처리된 `.env.EXAMPLE`을 작성합니다 — `.env`에 복사하고 채워 넣으세요.
7. `--alias`를 사용하면 래퍼가 생성되므로 `research-bot chat`을 직접 실행할 수 있습니다.

### 소스 유형

모든 git URL이 작동합니다:

```bash
# GitHub 약칭
hermes profile install github.com/you/research-bot

# 전체 HTTPS
hermes profile install https://github.com/you/research-bot.git

# SSH
hermes profile install git@github.com:you/research-bot.git

# 자체 호스팅, GitLab, Gitea, Forgejo — 모든 Git 호스트
hermes profile install https://git.example.com/team/research-bot.git

# 구성된 git 인증을 사용하는 프라이빗 저장소
hermes profile install git@github.com:your-org/internal-bot.git

# 개발 중 로컬 디렉토리 (git 푸시 필요 없음)
hermes profile install ~/my-profile-in-progress/
```

### 프로필 이름 재정의

두 명의 사용자가 동일한 배포판을 다른 프로필 이름으로 원하는 경우:

```bash
# Alice
hermes profile install github.com/acme/support-bot --name support-us --alias
# Bob (동일한 배포판, 다른 로컬 이름)
hermes profile install github.com/acme/support-bot --name support-eu --alias
```

### 환경 변수 채우기

설치 후 에이전트의 프로필에는 `.env.EXAMPLE`이 포함됩니다:

```
# 이 Hermes 배포판에 필요한 환경 변수입니다.
# 실행하기 전에 `.env`에 복사하고 자체 값을 입력하세요.

# OpenAI API 키 (모델 액세스용)
# (필수)
OPENAI_API_KEY=

# 웹 검색을 위한 SerpAPI 키
# (선택)
# SERPAPI_KEY=
```

복사합니다:

```bash
cp ~/.hermes/profiles/research-bot/.env.EXAMPLE ~/.hermes/profiles/research-bot/.env
# .env를 편집하여 실제 키를 붙여넣습니다.
```

쉘 환경에 이미 있던 필수 키(예: `~/.zshrc`에서 `export`된 `OPENAI_API_KEY`)는 설치 중 `✓ set`으로 표시됩니다 — `.env`에 중복해서 넣을 필요가 없습니다.

### 설치된 내용 확인

```bash
hermes profile info research-bot
```

다음과 같이 표시됩니다:

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

`hermes profile list`에는 `Distribution` 열도 표시되어 저장소에서 가져온 프로필과 수동으로 구축한 프로필을 한 눈에 확인할 수 있습니다:

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

발생하는 일:

1. 기록된 소스 URL에서 저장소를 다시 복제합니다.
2. 배포 소유 파일(SOUL, 스킬, 크론, mcp.json)을 교체합니다.
3. `config.yaml`은 **유지**됩니다 — 모델, 온도 또는 다른 설정을 변경했을 수 있습니다. 덮어쓰려면 `--force-config`를 전달하세요.
4. 사용자 데이터는 **절대 건드리지 않습니다**: 메모리, 세션, 인증, `.env`, 로그, 상태.

전체 아카이브를 다시 다운로드하지 않습니다. 로컬 구성 변경을 덮어쓰지 않습니다. 대화 기록을 삭제하지 않습니다.

### 삭제

```bash
hermes profile delete research-bot
```

삭제 프롬프트는 확인을 요청하기 전에 배포 정보를 보여줍니다:

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

따라서 에이전트가 어디서 왔는지 모르거나 다시 설치할 수 없는 상태에서 실수로 에이전트를 삭제하는 일은 없습니다.

---

## 사용 사례 및 패턴

### 개인: 여러 머신에서 하나의 에이전트 동기화

노트북에서 연구 보조원을 만들었습니다. 워크스테이션에서도 동일한 에이전트를 원합니다.

```bash
# 노트북
cd ~/.hermes/profiles/research-bot
git init && git add . && git commit -m "initial"
git remote add origin git@github.com:you/research-bot.git
git push -u origin main

# 워크스테이션
hermes profile install github.com/you/research-bot --alias
# .env 채우기. 완료.
```

노트북에서의 모든 반복 작업(`git commit && push`)은 `hermes profile update research-bot`을 통해 워크스테이션으로 끌려옵니다. 메모리는 머신별로 유지됩니다. 노트북은 자체 대화를 기억하고 워크스테이션은 자체 대화를 기억하며, 서로 충돌하지 않습니다.

### 팀: 검토된 내부 에이전트 배포

엔지니어링 팀은 특정한 SOUL, 특정한 스킬, 그리고 모든 PR을 검토하는 크론이 있는 공유된 PR 검토 봇을 원합니다.

```bash
# 엔지니어링 리드
cd ~/.hermes/profiles/pr-reviewer
# ... 빌드 및 튜닝 ...
git init && git add . && git commit -m "v1.0 PR reviewer"
git tag v1.0.0
git push -u origin main --tags    # 회사 내부 Git 호스트에 푸시

# 각 엔지니어
hermes profile install git@github.com:your-org/pr-reviewer.git --alias
# .env를 각자의 API 키(본인에게 청구됨)로 채움, .env.EXAMPLE이 필요한 것을 알려줌
pr-reviewer chat
```

리드가 v1.1(더 나은 SOUL, 새 스킬)을 배포하면 엔지니어는 `hermes profile update pr-reviewer`를 실행하고 몇 분 안에 모든 사람이 새 버전을 사용할 수 있게 됩니다.

### 커뮤니티: 공개 에이전트 게시

새로운 무언가를 만들었습니다 — 어쩌면 "Polymarket 트레이더"나 "학술 논문 요약기" 또는 "Minecraft 서버 관리 도우미"일 수도 있습니다. 당신은 이것을 공유하고 싶습니다.

```bash
# 자신
cd ~/.hermes/profiles/polymarket-trader
# 저장소 루트에 충실한 README.md 작성 — GitHub 저장소 페이지에 표시됨
git init && git add . && git commit -m "v1.0"
git tag v1.0.0
# 공개 GitHub 저장소에 게시
git remote add origin https://github.com/you/hermes-polymarket-trader.git
git push -u origin main --tags

# 누구나
hermes profile install github.com/you/hermes-polymarket-trader --alias
```

설치 명령을 트윗합니다. 시도해 본 사람들이 이슈와 PR을 보냅니다. 누군가 사용자 정의를 원하면 그들은 포크합니다 — 누구나 이미 알고 있는 동일한 git 워크플로입니다.

### 제품: 오피니언이 반영된(Opinionated) 에이전트 출시

당신은 Hermes 기반 위에서 무언가를 구축했습니다 — 어쩌면 규정 준수 모니터링 하네스, 고객 지원 스택, 도메인별 리서치 플랫폼일 수 있습니다. 제품으로 배포하고 싶습니다.

```yaml
# distribution.yaml
name: telemetry-harness
version: 2.3.1
description: "규정 준수 원격 측정 하네스 — 규제 대상 워크플로 모니터링 및 검토"
hermes_requires: ">=0.13.0"
author: "Acme Compliance Inc."
license: "Commercial"

env_requires:
  - name: ACME_API_KEY
    description: "Acme Compliance 라이선스 키 (support@acme.com에 이메일 문의)"
    required: true
  - name: OPENAI_API_KEY
    description: "모델 액세스를 위한 OpenAI API 키"
    required: true
  - name: GRAPHITI_MCP_URL
    description: "Graphiti 지식 그래프 인스턴스 URL"
    required: false
    default: "http://127.0.0.1:8000/sse"
```

고객은 단일 명령으로 설치합니다. 설치 미리보기는 준비해야 할 키를 정확하게 알려주고, 새 릴리스 태그를 지정하는 순간 업데이트가 적용되며, 그들의 규정 준수 데이터(`memories/`, `sessions/`)는 그들의 컴퓨터를 벗어나지 않습니다.

### 일회성: 공유 인프라에서의 일회성 스크립트

당신은 운영(Ops) 리드입니다. 올바른 도구와 MCP 연결을 갖춘 통조림 같은 SOUL을 가진 임시 에이전트가 프로덕션 장애를 진단하고 다음 주 동안 세 명의 당직 엔지니어의 노트북에서 실행되기를 원합니다.

```bash
# 자신
# 프로필을 빌드, 커밋, 비공개 저장소에 푸시
git push -u origin main

# 각 당직 엔지니어
hermes profile install git@github.com:your-org/incident-2026-q2.git --alias

# 인시던트 해결 — 삭제
hermes profile delete incident-2026-q2
```

설치 및 삭제 주기가 쓰레기통에 버릴 수 있을 만큼 저렴합니다.

---

## 레시피

### 특정 버전에 고정하기

:::note
Git 참조 고정(예: `#v1.2.0`)은 계획되어 있지만 초기 릴리스에는 포함되지 않습니다. 설치는 현재 기본 브랜치를 추적합니다. `hermes profile info <name>`을 통해 설치된 버전을 추적하고 준비가 될 때까지 업데이트를 보류하세요.
:::

### 최신 버전과 설치된 버전 비교 확인하기

```bash
# 설치된 버전
hermes profile info research-bot | grep Version

# 최신 업스트림 (설치하지 않고 확인)
git ls-remote --tags https://github.com/you/research-bot | tail -5
```

### 업데이트를 통해 로컬 구성 맞춤 설정 유지하기

기본 업데이트 동작은 이미 이 작업을 수행합니다. `config.yaml`은 보존됩니다. 안전을 위해 배포판이 소유하지 않은 파일에 로컬 변경 사항을 기록하세요:

```yaml
# ~/.hermes/profiles/research-bot/local/my-overrides.yaml
# (배포판은 local/을 절대 건드리지 않음)
```

…그리고 필요에 따라 `config.yaml`이나 SOUL에서 이를 참조하세요.

### 클린(Clean) 재설치 강제 수행

```bash
# 모두 지우고 처음부터 다시 설치 (메모리/세션도 손실됨)
hermes profile delete research-bot --yes
hermes profile install github.com/you/research-bot --alias

# 현재 main으로 업데이트하지만 config.yaml을 배포판의 기본값으로 재설정
hermes profile update research-bot --force-config --yes
```

### 포크 및 맞춤 설정(Customize)

표준 git 워크플로 — 배포판은 결국 저장소일 뿐입니다:

```bash
# GitHub에서 저장소를 포크한 다음, 자신의 포크를 설치합니다.
hermes profile install github.com/yourname/forked-research-bot --alias

# ~/.hermes/profiles/forked-research-bot/ 에서 로컬로 작업
# SOUL.md 편집, 커밋, 포크에 푸시
# 업스트림 변경 사항: 일반적인 방법으로 포크에 가져옵니다.
```

### 푸시하기 전에 배포판 테스트하기

작성자의 컴퓨터에서:

```bash
# 로컬 디렉토리에서 설치 (git 푸시 필요 없음)
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test --alias

# 올바르게 될 때까지 조정, 삭제, 재설치
hermes profile delete research-bot-test --yes
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test
```

---

## 배포판에 포함되지 않는 것 (절대 포함 안 됨)

작성자가 실수로 이 경로들을 제공하더라도 설치 프로그램은 이를 강제로 배제합니다. 이 기능을 재정의할 수 있는 구성 옵션은 없으며 안전 가드는 회귀 테스트되는 불변 원칙(invariant)입니다.

- `auth.json` — OAuth 토큰, 플랫폼 자격 증명
- `.env` — API 키, 비밀 정보
- `memories/` — 대화 메모리
- `sessions/` — 대화 기록
- `state.db`, `state.db-shm`, `state.db-wal` — 세션 메타데이터
- `logs/` — 에이전트 및 오류 로그
- `workspace/` — 생성된 작업 파일
- `plans/` — 초안 계획(scratch plans)
- `home/` — Docker 백엔드의 사용자 홈 마운트
- `*_cache/` — 이미지 / 오디오 / 문서 캐시
- `local/` — 사용자 예약 커스터마이징 네임스페이스

배포판을 복제하면 이런 항목들은 단순히 존재하지 않습니다. 업데이트할 때 제자리에 유지됩니다. 동일한 배포판을 다섯 대의 머신에 설치했다면, 머신당 하나씩 총 다섯 개의 독립된 데이터 집합을 가지게 됩니다.

## 보안 및 신뢰

프로필 배포판은 기본적으로 서명되지 않습니다. 당신은 다음을 신뢰하는 것입니다:

- **git 호스트** (GitHub / GitLab 등)가 작성자가 푸시한 바이트를 제공할 것임을 신뢰합니다.
- **작성자**가 악성 SOUL, 스킬 또는 크론 작업을 배포하지 않을 것임을 신뢰합니다.

배포판의 크론 작업은 **자동으로 예약되지 않습니다** — 설치 프로그램이 `hermes -p <name> cron list`를 인쇄하고 당신이 직접 명시적으로 활성화해야 합니다. 그러나 SOUL.md와 스킬은 프로필과 대화를 시작하는 즉시 활성화되므로 모르는 사람에게서 설치하는 경우 첫 실행 전에 내용을 확인하세요.

대략적인 비유: 배포판 설치는 브라우저 확장 프로그램이나 VS Code 확장 프로그램을 설치하는 것과 같습니다. 마찰이 적고, 능력이 뛰어나며, 소스를 신뢰해야 합니다. 사내 배포판의 경우 프라이빗 저장소와 일반적인 git 인증을 사용하세요 — 새로 구성할 것이 없습니다.

향후 버전에는 서명, 확인된 커밋 SHA가 포함된 잠금 파일(`.distribution-lock.yaml`), 업데이트를 적용하기 전에 차이점을 인쇄하는 `--dry-run` 플래그가 추가될 수 있습니다. 현재 출시된 기능은 아닙니다.

## 내부 동작 (Under the hood)

구현 세부 정보, 정확한 CLI 동작 및 모든 플래그는 [프로필 명령 참조](../reference/profile-commands.md#distribution-commands)를 확인하세요.

요약:

- `install`, `update`, `info`는 별도의 병렬 명령 트리가 아니라 `hermes profile` 내부에 있습니다.
- 매니페스트 형식은 아주 작은 필수 스키마(`name`만)를 가진 YAML입니다.
- 설치 프로그램은 복제를 위해 로컬 `git` 바이너리를 사용하므로 쉘이 이미 처리하는 모든 인증(SSH 키, 자격 증명 헬퍼)이 투명하게 작동합니다.
- 복제 후 `.git/`은 제거됩니다 — 설치된 프로필 자체는 git 체크아웃이 아니므로, 실수로 자신의 `.env`를 배포판의 git 기록에 커밋하는 실수를 방지합니다.
- 예약된 프로필 이름(`hermes`, `test`, `tmp`, `root`, `sudo`)은 일반 바이너리와의 충돌을 피하기 위해 설치 시 거부됩니다.

## 함께 보기

- [프로필: 여러 에이전트 실행](./profiles.md) — 기본 개념
- [프로필 명령 참조](../reference/profile-commands.md) — 모든 플래그, 모든 옵션
- [`hermes profile export` / `import`](../reference/profile-commands.md#hermes-profile-export) — 로컬 백업 / 복원 (배포 기능 아님)
- [Hermes와 함께 SOUL 사용하기](../guides/use-soul-with-hermes.md) — 성격(personality) 작성
- [성격 및 SOUL](./features/personality.md) — SOUL이 에이전트에 어떻게 적용되는지
- [스킬 카탈로그](../reference/skills-catalog.md) — 번들로 제공할 수 있는 스킬들
