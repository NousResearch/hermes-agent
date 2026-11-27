---
sidebar_position: 7
---

# 프로필 명령어 참조

이 페이지에서는 [Hermes 프로필](../user-guide/profiles.md)과 관련된 모든 명령어를 다룹니다. 일반 CLI 명령어는 [CLI 명령어 참조](./cli-commands.md)를 참고하세요.

## `hermes profile`

```bash
hermes profile <subcommand>
```

프로필을 관리하는 최상위 명령어입니다. 하위 명령어 없이 `hermes profile`을 실행하면 도움말이 표시됩니다.

| 하위 명령어 | 설명 |
|------------|-------------|
| `list` | 모든 프로필을 나열합니다. |
| `use` | 활성(기본) 프로필을 설정합니다. |
| `create` | 새 프로필을 만듭니다. |
| `describe` | 프로필 설명을 읽거나 설정합니다(kanban 오케스트레이터가 라우팅에 사용). |
| `delete` | 프로필을 삭제합니다. |
| `show` | 프로필의 세부 정보를 표시합니다. |
| `alias` | 프로필의 셸 별칭을 다시 생성합니다. |
| `rename` | 프로필 이름을 변경합니다. |
| `export` | 프로필을 tar.gz 아카이브로 내보냅니다. |
| `import` | tar.gz 아카이브에서 프로필을 가져옵니다. |
| `install` | git URL 또는 로컬 디렉터리에서 프로필 배포판을 설치합니다. [프로필 배포판](../user-guide/profile-distributions.md)을 참고하세요. |
| `update` | 배포판으로 관리되는 프로필을 다시 가져오고 번들을 다시 적용합니다. |
| `info` | 프로필의 배포판 메타데이터(출처 URL, 커밋, 마지막 업데이트)를 표시합니다. |

## `hermes profile list`

```bash
hermes profile list
```

모든 프로필을 나열합니다. 현재 활성 프로필에는 `*`가 표시됩니다.

**예시:**

```bash
$ hermes profile list
  default
* work
  dev
  personal
```

옵션이 없습니다.

## `hermes profile use`

```bash
hermes profile use <name>
```

`<name>`을 활성 프로필로 설정합니다. 이후 실행하는 모든 `hermes` 명령어(`-p` 없이)는 이 프로필을 사용합니다.

| 인수 | 설명 |
|----------|-------------|
| `<name>` | 활성화할 프로필 이름입니다. 기본 프로필로 돌아가려면 `default`를 사용하세요. |

**예시:**

```bash
hermes profile use work
hermes profile use default
```

## `hermes profile create`

```bash
hermes profile create <name> [options]
```

새 프로필을 만듭니다.

| 인수 / 옵션 | 설명 |
|-------------------|-------------|
| `<name>` | 새 프로필의 이름입니다. 유효한 디렉터리 이름(영숫자, 하이픈, 밑줄)이어야 합니다. |
| `--clone` | 현재 프로필의 `config.yaml`, `.env`, `SOUL.md` 및 스킬을 복사합니다. |
| `--clone-all` | 모든 항목(구성, 메모리, 스킬, cron, 플러그인)을 현재 프로필에서 복사합니다. 프로필별 기록인 세션, `state.db`, 백업, state-snapshots, 체크포인트는 제외됩니다. |
| `--clone-from <profile>` | 현재 프로필 대신 지정한 프로필에서 구성/스킬/SOUL을 복제합니다. `--clone-all`과 함께 사용하지 않는 한 `--clone`을 암시합니다. |
| `--no-alias` | 래퍼 스크립트 생성을 건너뜁니다. |
| `--description "<text>"` | 이 프로필이 어떤 작업에 적합한지 설명하는 한두 문장입니다. kanban 오케스트레이터가 프로필 이름만이 아니라 역할을 기준으로 작업을 라우팅하는 데 사용합니다. 생략한 뒤 `hermes profile describe`로 나중에 추가할 수 있습니다. `<profile_dir>/profile.yaml`에 저장됩니다. |
| `--no-skills` | 번들 스킬이 하나도 활성화되지 않은 **빈** 프로필을 만듭니다. 향후 `hermes update`가 번들 스킬을 다시 심지 않도록 프로필에 `.no-bundled-skills` 마커를 기록하며, `--clone`, `--clone-from`, `--clone-all`과 함께 사용할 수 없습니다(어차피 스킬이 복사되기 때문). 전체 스킬 카탈로그를 상속하지 않아야 하는 좁은 오케스트레이터 프로필이나 샌드박스 프로필에 유용합니다. 이미 생성된 프로필(기본 `~/.hermes` 포함)에서 이 설정을 전환하려면 `hermes skills opt-out` / `hermes skills opt-in`을 사용하세요. |

프로필을 만든다고 해서 해당 프로필 디렉터리가 터미널 명령어의 기본 프로젝트/작업 공간 디렉터리가 되지는 않습니다. 프로필을 특정 프로젝트에서 시작하려면 해당 프로필의 `config.yaml`에서 `terminal.cwd`를 설정하세요.

**예시:**

```bash
# Blank profile — needs full setup
hermes profile create mybot

# Clone config only from current profile
hermes profile create work --clone

# Clone everything from current profile
hermes profile create backup --clone-all

# Clone config from a specific profile
hermes profile create work2 --clone-from work

# Clone everything from a specific profile
hermes profile create work2-backup --clone-from work --clone-all
```

## `hermes profile describe`

```bash
hermes profile describe [<name>] [options]
```

프로필 설명을 읽거나 설정합니다. 설명은 프로필 이름만 보고 추측하는 대신 각 프로필이 잘하는 일을 기준으로 작업을 라우팅하기 위해 kanban 오케스트레이터가 사용합니다. `<profile_dir>/profile.yaml`에 저장되므로 재부팅 후에도 유지되며 게이트웨이와 공유됩니다.

플래그 없이 실행하면 현재 설명을 출력합니다(비어 있을 경우 `(no description set for '<name>')`).

| 인수 / 옵션 | 설명 |
|-------------|-------------|
| `<name>` | 설명할 프로필입니다. `--all --auto`를 사용하는 경우가 아니면 필수입니다. |
| `--text "<text>"` | 설명을 이 정확한 텍스트(사용자가 작성한 내용)로 설정합니다. 기존 설명을 덮어씁니다. |
| `--auto` | 프로필에 설치된 스킬, 구성된 모델, 이름을 바탕으로 보조 LLM을 통해 1~2문장 설명을 자동 생성합니다. `config.yaml`의 `auxiliary.profile_describer`에서 모델을 구성하세요. 자동 생성된 설명에는 `description_auto: true`가 표시되므로 대시보드에서 검토 대상으로 표시할 수 있습니다. |
| `--overwrite` | `--auto`와 함께 사용하면 사용자가 작성한 설명도 교체합니다(기본값은 설명을 명시적으로 설정한 프로필을 건너뜀). |
| `--all` | `--auto`와 함께 사용하면 설명이 없는 모든 프로필을 순회합니다. |

**예시:**

```bash
# Read the current description
hermes profile describe researcher

# Set it explicitly
hermes profile describe researcher --text "Reads source code and writes findings."

# Let the LLM generate one
hermes profile describe researcher --auto

# Fill in descriptions for every profile that doesn't have one
hermes profile describe --all --auto
```

## `hermes profile delete`

```bash
hermes profile delete <name> [options]
```

프로필을 삭제하고 셸 별칭을 제거합니다.

| 인수 / 옵션 | 설명 |
|-------------|-------------|
| `<name>` | 삭제할 프로필입니다. |
| `--yes`, `-y` | 확인 프롬프트를 건너뜁니다. |

**예시:**

```bash
hermes profile delete mybot
hermes profile delete mybot --yes
```

:::warning
이 작업은 모든 구성, 메모리, 세션 및 스킬을 포함한 프로필의 전체 디렉터리를 영구적으로 삭제합니다. `default` 프로필(`~/.hermes`)은 삭제할 수 없습니다. 모든 항목을 제거하려면 `hermes uninstall`을 사용하세요.
:::

## `hermes profile show`

```bash
hermes profile show <name>
```

홈 디렉터리, 구성된 모델, 게이트웨이 상태, 스킬 수 및 구성 파일 상태를 포함한 프로필 세부 정보를 표시합니다.

이는 터미널 작업 디렉터리가 아니라 프로필의 Hermes 홈 디렉터리를 표시합니다. 터미널 명령어는 `terminal.cwd`에서 시작합니다(로컬 백엔드에서 `cwd: "."`인 경우에는 실행 디렉터리에서 시작).

| 인수 | 설명 |
|----------|-------------|
| `<name>` | 검사할 프로필입니다. |

**예시:**

```bash
$ hermes profile show work
Profile: work
Path:    ~/.hermes/profiles/work
Model:   anthropic/claude-sonnet-4 (anthropic)
Gateway: stopped
Skills:  12
.env:    exists
SOUL.md: exists
Alias:   ~/.local/bin/work
```

## `hermes profile alias`

```bash
hermes profile alias <name> [options]
```

`~/.local/bin/<name>`에 셸 별칭 스크립트를 다시 생성합니다. 별칭을 실수로 삭제했거나 Hermes 설치를 옮긴 후 업데이트해야 할 때 유용합니다.

| 인수 / 옵션 | 설명 |
|-------------|-------------|
| `<name>` | 별칭을 만들거나 업데이트할 프로필입니다. |
| `--remove` | 생성하는 대신 래퍼 스크립트를 제거합니다. |
| `--name <alias>` | 사용자 지정 별칭 이름입니다(기본값: 프로필 이름). |

**예시:**

```bash
hermes profile alias work
# Creates/updates ~/.local/bin/work

hermes profile alias work --name mywork
# Creates ~/.local/bin/mywork

hermes profile alias work --remove
# Removes the wrapper script
```

## `hermes profile rename`

```bash
hermes profile rename <old-name> <new-name>
```

프로필 이름을 변경합니다. 디렉터리와 셸 별칭이 업데이트됩니다.

| 인수 | 설명 |
|----------|-------------|
| `<old-name>` | 현재 프로필 이름입니다. |
| `<new-name>` | 새 프로필 이름입니다. |

**예시:**

```bash
hermes profile rename mybot assistant
# ~/.hermes/profiles/mybot → ~/.hermes/profiles/assistant
# ~/.local/bin/mybot → ~/.local/bin/assistant
```

## `hermes profile export`

```bash
hermes profile export <name> [options]
```

프로필을 압축된 tar.gz 아카이브로 내보냅니다. 백업하거나 다른 컴퓨터로 옮기거나 다른 사람에게 전달할 수 있는 이식 가능한 스냅샷입니다. `auth.json`과 `.env`는 항상 제외됩니다.

채팅에서는 [`/export`](./slash-commands.md)로도 사용할 수 있으며, 데스크톱 앱에서는 **⌘K → 프로필 내보내기…** 또는 프로필 사각형의 오른쪽 클릭 메뉴를 통해 사용할 수 있습니다. 데스크톱 내보내기는 추가로 `desktop.json`(스킨, 라이트/다크 모드, 사용자 지정 테마, 레일 색상, 창 레이아웃)을 아카이브에 스테이징합니다.

| 인수 / 옵션 | 설명 |
|-------------------|-------------|
| `<name>` | 내보낼 프로필입니다. |
| `-o`, `--output <path>` | 출력 파일 경로입니다(기본값: `<name>.tar.gz`). |

**예시:**

```bash
hermes profile export work
# Creates work.tar.gz in the current directory

hermes profile export work -o ./work-2026-03-29.tar.gz
```

아카이브에 정확히 무엇이 들어가며 다른 사람에게 보내기 전에 무엇을 확인해야 하는지는 [프로필 파일 내보내기 및 가져오기](../user-guide/profile-distributions.md#export-and-import-a-profile-file)를 참고하세요.

## `hermes profile import`

```bash
hermes profile import <archive> [options]
```

tar.gz 아카이브에서 새 프로필로 가져옵니다. 기존 프로필을 덮어쓰는 작업은 거부되며 `default`(내장 루트 프로필)로 가져올 수 없습니다. 두 경우 모두 `--name`을 전달하세요. 이름이 기존 명령어와 충돌하지 않으면 셸 래퍼가 생성됩니다.

채팅에서는 [`/import`](./slash-commands.md)로도 사용할 수 있으며, 데스크톱 앱에서는 **⌘K → 프로필 가져오기…** 또는 프로필 레일의 **+** 옆에 있는 가져오기 버튼을 통해 사용할 수 있습니다. 데스크톱 가져오기는 포함된 `desktop.json` 오버레이(테마, 레이아웃)도 적용하고 새 프로필로 전환합니다.

| 인수 / 옵션 | 설명 |
|-------------------|-------------|
| `<archive>` | 가져올 tar.gz 아카이브의 경로입니다. |
| `--name <name>` | 가져온 프로필의 이름입니다(기본값: 아카이브에서 추론). |

**예시:**

```bash
hermes profile import ./work-2026-03-29.tar.gz
# Infers profile name from the archive

hermes profile import ./work-2026-03-29.tar.gz --name work-restored
```

## 배포판 명령어

:::tip
**배포판이 처음인가요?** [프로필 배포판 사용자 가이드](../user-guide/profile-distributions.md)부터 시작하세요. 이 가이드에서는 이유, 시점 및 방법을 전체 예시와 함께 설명합니다. 아래 섹션은 원하는 작업을 알고 있을 때 참고하는 간단한 CLI 참조입니다.
:::

배포판은 프로필을 **git 저장소**로 게시되는 공유 가능하고 버전이 지정된 아티팩트로 바꿉니다. 수신자는 단일 명령어로 배포판을 설치하고 나중에 로컬 메모리, 세션 또는 자격 증명을 건드리지 않고 제자리에서 업데이트할 수 있습니다.

`auth.json`과 `.env`는 배포판에 절대 포함되지 않으며 설치하는 사용자의 컴퓨터에 그대로 남습니다.

수신자의 사용자 데이터(메모리, 세션, 인증 정보, `.env`에 직접 적용한 변경 사항)는 최초 설치와 이후 업데이트에서 항상 보존됩니다.

:::info
프로필을 공유하는 두 가지 방법은 서로 보완됩니다. `hermes profile export` / `import`(채팅에서는 `/export` 및 `/import`)는 **단일 파일**을 생성합니다. 저장소나 매니페스트가 필요 없으며 데스크톱 내보내기에는 테마와 레이아웃도 포함됩니다. 배포판(`install` / `update` / `info`)은 프로필을 **git 저장소**로 게시하므로 수신자가 나중에 버전이 지정된 업데이트를 가져올 수 있습니다. 백업과 복원은 내보내기 파일의 또 다른 용도입니다. [프로필을 공유하는 두 가지 방법](../user-guide/profile-distributions.md#two-ways-to-share-a-profile)을 참고하세요.
:::

### `hermes profile install`

```bash
hermes profile install <source> [--name <name>] [--alias] [--force] [--yes]
```

git URL 또는 로컬 디렉터리에서 프로필 배포판을 설치합니다.

| 옵션 | 설명 |
|--------|-------------|
| `<source>` | Git URL(`github.com/user/repo`, `https://...`, `git@...`, `ssh://`, `git://`) 또는 루트에 `distribution.yaml`이 있는 로컬 디렉터리입니다. |
| `--name NAME` | 매니페스트의 프로필 이름을 재정의합니다. |
| `--alias` | 셸 래퍼도 생성합니다(예: `telemetry` → `hermes -p telemetry`). |
| `--force` | 같은 이름의 기존 프로필을 덮어씁니다. 사용자 데이터는 여전히 보존됩니다. |
| `-y`, `--yes` | 매니페스트 미리보기 확인 프롬프트를 건너뜁니다. |

설치 프로그램은 매니페스트를 표시하고, 필요한 환경 변수를 나열하며, 확인을 요청하기 전에 cron 작업에 대해 경고합니다. 필요한 환경 변수는 `.env.EXAMPLE` 파일에 기록되며, 이 파일을 `.env`로 복사한 뒤 값을 입력해야 합니다.

**예시:**

```bash
# Install from a GitHub repo (shorthand)
hermes profile install github.com/kyle/telemetry-distribution --alias

# Install from a full HTTPS git URL
hermes profile install https://github.com/kyle/telemetry-distribution.git

# Install from SSH
hermes profile install git@github.com:kyle/telemetry-distribution.git

# Install from a local directory during development
hermes profile install ./telemetry/
```

### `hermes profile update`

```bash
hermes profile update <name> [--force-config] [--yes]
```

기록된 출처에서 배포판을 다시 복제하고 업데이트를 적용합니다. 배포판이 소유한 파일(SOUL.md, skills/, cron/, mcp.json)은 덮어쓰지만 사용자 데이터(메모리, 세션, 인증 정보, .env)는 절대 건드리지 않습니다.

로컬 재정의를 유지하기 위해 기본적으로 `config.yaml`은 보존됩니다. 배포판에 포함된 구성으로 재설정하려면 `--force-config`를 전달하세요.

### `hermes profile info`

```bash
hermes profile info <name>
```

프로필의 배포판 매니페스트(이름, 버전, 필요한 Hermes 버전, 작성자, 환경 변수 요구 사항, 출처 URL/경로, 배포판을 마지막으로 `install` 또는 `update`한 시점에 기록된 `Installed:` 타임스탬프)를 출력합니다. 공유 프로필을 설치하기 전에 필요한 항목을 확인하고, "이 프로필은 6개월 전에 설치되었으며 업데이트되지 않았다"는 상황을 파악하는 데 유용합니다.

`hermes profile list`는 `Distribution` 열에도 배포판 이름과 버전을 표시하며, `hermes profile show <name>` / `delete <name>`는 출처 URL을 표시하므로 어떤 프로필이 git 저장소에서 왔고 어떤 프로필이 로컬에서 생성되었는지 한눈에 알 수 있습니다.

### 비공개 배포판

비공개 git 저장소는 별도의 구성 없이 배포판 출처로 사용할 수 있습니다. 설치 과정에서는 일반 `git` 바이너리를 셸로 실행하므로 셸에 이미 설정한 인증 방식(SSH 키, `git credential` 헬퍼, GitHub CLI에 저장된 HTTPS 자격 증명)이 투명하게 적용됩니다.

```bash
# Uses your SSH key, the same as any other `git clone`
hermes profile install git@github.com:your-org/internal-assistant.git

# Uses your git credential helper
hermes profile install https://github.com/your-org/internal-assistant.git
```

설치 중 터미널에서 복제 과정이 대화형으로 자격 증명을 요청하면 해당 프롬프트가 그대로 전달됩니다. 먼저 같은 저장소에 대해 평소처럼 `git clone`을 실행할 때 사용할 인증을 설정한 다음 설치하세요.

### 배포판 매니페스트(`distribution.yaml`)

모든 배포판은 저장소 루트에 `distribution.yaml`을 둡니다.

```yaml
name: telemetry
version: 0.1.0
description: "Compliance monitoring harness"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API key"
    required: true
  - name: GRAPHITI_MCP_URL
    description: "Memory graph URL"
    required: false
    default: "http://127.0.0.1:8000/sse"
distribution_owned:   # optional; defaults to SOUL.md, config.yaml,
                      #   mcp.json, skills/, cron/, distribution.yaml
  - SOUL.md
  - skills/compliance/
  - cron/
```

`hermes_requires`는 `>=`, `<=`, `==`, `!=`, `>`, `<` 또는 단독 버전을 지원합니다(단독 버전은 `>=`로 처리됨). 현재 Hermes 버전이 사양을 충족하지 않으면 설치가 명확한 오류와 함께 실패합니다.

`distribution_owned`는 선택 사항입니다. 설정하면 업데이트 시 해당 경로만 교체되며 프로필의 나머지 항목은 사용자 소유로 유지됩니다. 생략하면 위의 기본값이 적용됩니다.

### 배포판 게시

배포판 작성은 git push만으로 이루어집니다.

1. 프로필 디렉터리에 최소한 `name`과 `version`을 포함하는 `distribution.yaml`을 만듭니다.
2. git 저장소를 초기화하거나 기존 저장소를 사용한 뒤 GitHub / GitLab / Hermes가 복제할 수 있는 호스트로 푸시합니다.
3. 수신자에게 `hermes profile install <your-repo-url>`을 실행하라고 안내합니다.

버전이 지정된 릴리스에는 git 태그를 사용하세요. `HEAD`를 복제하는 수신자는 최신 상태를 받으며, 매니페스트의 `version:`을 올려 언제든지 버전을 증가시킬 수 있습니다.

## `hermes -p` / `hermes --profile`

```bash
hermes -p <name> <command> [options]
hermes --profile <name> <command> [options]
```

고정된 기본 프로필을 변경하지 않고 특정 프로필에서 어떤 Hermes 명령어든 실행할 수 있게 하는 전역 플래그입니다. 명령어가 실행되는 동안 활성 프로필을 재정의합니다.

| 옵션 | 설명 |
|-------------|-------------|
| `-p <name>`, `--profile <name>` | 이 명령어에 사용할 프로필입니다. |

**예시:**

```bash
hermes -p work chat -q "Check the server status"
hermes --profile dev gateway start
hermes -p personal skills list
hermes -p work config edit
```

## `hermes completion`

```bash
hermes completion <shell>
```

셸 자동 완성 스크립트를 생성합니다. 프로필 이름과 프로필 하위 명령어에 대한 자동 완성이 포함됩니다.

| 인수 | 설명 |
|-------------|-------------|
| `<shell>` | 자동 완성을 생성할 셸입니다: `bash`, `zsh` 또는 `fish`. |

**예시:**

```bash
# Install completions
hermes completion bash >> ~/.bashrc
hermes completion zsh >> ~/.zshrc
hermes completion fish > ~/.config/fish/completions/hermes.fish

# Reload shell
source ~/.bashrc
```

설치 후 탭 자동 완성이 다음 항목에 대해 작동합니다:
- `hermes profile <TAB>` — 하위 명령어(list, use, create 등)
- `hermes profile use <TAB>` — 프로필 이름
- `hermes -p <TAB>` — 프로필 이름

## 함께 보기

- [프로필 사용자 가이드](../user-guide/profiles.md)
- [CLI 명령어 참조](./cli-commands.md)
- [FAQ — 프로필 섹션](./faq.md#profiles)
