---
sidebar_position: 4
title: "기여하기"
description: "Hermes Agent에 기여하는 방법 — 개발 환경 설정, 코드 스타일, PR 절차"
---

# 기여하기

Hermes Agent에 기여해 주셔서 감사합니다! 이 가이드에서는 개발 환경 설정, 코드베이스 이해, PR 병합 절차를 다룹니다.

## 기여 우선순위

기여의 우선순위는 다음과 같습니다.

1. **버그 수정** — 충돌, 잘못된 동작, 데이터 손실
2. **멀티 플랫폼 호환성** — macOS, 다양한 Linux 배포판, WSL2
3. **보안 강화** — 셸 인젝션, 프롬프트 인젝션, 경로 순회
4. **성능 및 견고성** — 재시도 로직, 오류 처리, 우아한 성능 저하
5. **새 스킬** — 폭넓게 유용한 스킬([스킬 만들기](creating-skills.md) 참조)
6. **새 도구** — 거의 필요하지 않음. 대부분의 기능은 스킬이어야 합니다.
7. **문서** — 수정, 명확화, 새로운 예시

## 일반적인 기여 경로

- Hermes 코어를 수정하지 않고 사용자 지정/로컬 도구를 만들려면? [Hermes 플러그인 만들기](../developer-guide/plugins/index.md)부터 시작하세요.
- Hermes 자체에 새 내장 코어 도구를 만들려면? [도구 추가하기](./adding-tools.md)부터 시작하세요.
- 새 스킬을 만들려면? [스킬 만들기](./creating-skills.md)부터 시작하세요.
- 새로운 추론 제공자를 만들려면? [제공자 추가하기](./adding-providers.md)부터 시작하세요.

## 개발 환경 설정

### 사전 요구 사항

| 요구 사항          | 참고                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------- |
| **Git**              | `git-lfs` 확장이 설치되어 있어야 함                                                         |
| **Python 3.11–3.13** | 없으면 uv가 설치함                                                                 |
| **uv**               | 빠른 Python 패키지 관리자([설치](https://docs.astral.sh/uv/))                           |
| **Node.js 26+**      | 선택 사항 — 브라우저 도구와 WhatsApp 브리지에 필요(루트 `package.json`의 engines와 일치) |

### 표준 설치 프로그램으로 설치

대부분의 기여자에게 가장 좋은 개발 부트스트랩 방법은 사용자가 사용하는 경로와 동일합니다. 표준 설치 프로그램을 실행한 다음 설치 프로그램이 복제한 저장소에서 작업하세요.
설치 프로그램은 Hermes venv를 만들고, `hermes` 명령을 연결하며, `hermes update`를 위한 설치 방법을 기록하고, 전체 git 프로젝트를 `$HERMES_HOME/hermes-agent`(일반적으로 `~/.hermes/hermes-agent`)에 복제합니다. 이렇게 하면 개발 환경이 CLI, 업데이터, 지연 의존성 설치 프로그램, 게이트웨이 및 문서가 전제로 하는 동일한 레이아웃을 사용하게 됩니다.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"

# Add dev/test extras on top of the standard install.
uv pip install -e ".[all,dev]"

# Optional: browser tools / docs site dependencies.
npm install
```

그런 다음 해당 체크아웃에서 브랜치를 만들고 테스트를 실행합니다.

```bash
git checkout -b fix/description
scripts/run_tests.sh
```

완전히 격리된 Hermes 인스턴스를 실행할 수도 있습니다(일회용 HERMES_HOME, 별도의 Electron userData, 단일 인스턴스 잠금을 피하기 위한 별도의 Electron 앱 이름).

```bash
scripts/dev-sandbox.sh python -m hermes_cli.main
scripts/dev-sandbox.sh --persistent python -m hermes_cli.main desktop  # state survives restarts, but lives in the worktree :)
```

### 수동 복제 대안

Hermes가 관리하는 설치 레이아웃을 의도적으로 사용하지 않으려는 경우에만 이 방법을 사용하세요(예: 컨테이너나 CI 작업의 일회용 복제본). 이 방법으로 설치했다면 이 venv의 `hermes` 진입점을 실행해야 합니다. 시스템의 `python3 -m hermes_cli.main`을 실행하면 관련 없는 시스템 Python 패키지를 가져올 수 있습니다.

복제한 소스 트리 **외부에** venv를 만드세요. 에이전트가 자체 체크아웃에 대해 실행하는 상대 경로 명령(`rm -rf venv`, `uv venv venv` 등)으로 트리 내부에 있는 venv가 삭제될 수 있으며, 실행 중인 런타임이 조용히 파괴됩니다. 트리 외부에 두면 작업 공간의 상대 경로가 해당 venv로 해석되지 않습니다.

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Create venv with Python 3.11, OUTSIDE the source tree
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
export VIRTUAL_ENV="$HOME/.hermes/venvs/hermes-dev"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Install with all extras (messaging, cron, CLI menus, dev tools)
uv pip install -e ".[all,dev]"

# Optional: browser tools
npm install
```

### 개발 환경 설정

```bash
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env

# Add at minimum an LLM provider key:
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' >> ~/.hermes/.env
```

### 실행

```bash
# The standard installer already put `hermes` on PATH.
hermes doctor
hermes chat -q "Hello"
```

수동 복제 대안을 사용했다면 체크아웃에서 `./hermes`를 실행하거나 이 복제본의 venv를 명시적으로 심볼릭 링크하세요.

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes
```

### 테스트 실행

```bash
scripts/run_tests.sh
```

## 코드 스타일

- **PEP 8**을 따르되 실용적인 예외를 허용합니다(엄격한 줄 길이 적용 없음).
- **주석:** 명확하지 않은 의도, 트레이드오프 또는 API 특이점을 설명할 때만 작성합니다.
- **오류 처리:** 구체적인 예외를 처리합니다. 예상하지 못한 오류에는 `exc_info=True`와 함께 `logger.warning()`/`logger.error()`를 사용합니다.
- **멀티 플랫폼:** Unix라고 가정하지 마세요(아래 참조).
- **프로필에 안전한 경로:** `~/.hermes`를 하드코딩하지 마세요 — 코드 경로에는 `hermes_constants`의 `get_hermes_home()`을, 사용자에게 표시하는 메시지에는 `display_hermes_home()`을 사용하세요. 전체 규칙은 [AGENTS.md](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md#profiles-multi-instance-support)를 참조하세요.

## 멀티 플랫폼 호환성

**[플랫폼 지원](../getting-started/platform-support.md)**을 참조하세요. 네이티브 Windows는 셸 명령에 [Git for Windows](https://git-scm.com/download/win)의 Git Bash를 사용합니다. 일부 기능은 POSIX 커널 기본 요소가 필요하므로 제한됩니다. 대시보드의 내장 PTY 터미널 창(`/chat` 탭)은 POSIX PTY(Linux, macOS 또는 WSL2)가 필요합니다. Windows 환경 중심으로 개발한다면 푸시하기 전에 Windows 취약점 린트(`scripts/check-windows-footguns.py`)를 실행하세요.

코드를 기여할 때 다음 규칙을 지키세요.

- **보호되지 않은 `signal.SIGKILL` 참조를 추가하지 마세요.** Windows에는 이 상수가 정의되어 있지 않습니다. `gateway.status.terminate_pid(pid, force=True)`(Windows에서는 `taskkill /T /F`, POSIX에서는 SIGKILL을 사용하는 중앙화된 기본 요소)를 사용하거나, `getattr(signal, "SIGKILL", signal.SIGTERM)`으로 대체하세요.
- `os.kill(pid, 0)` 프로브에서는 `ProcessLookupError`와 함께 `OSError`도 처리하세요. 이미 종료된 PID에 대해 Windows는 `ProcessLookupError` 대신 `OSError`(WinError 87, "parameter is incorrect")를 발생시킵니다.
- **터미널을 POSIX 의미 체계로 강제하지 마세요.** `os.setsid`, `os.killpg`, `os.getpgid`, `os.fork`는 Windows에서 예외를 발생시키므로 `if sys.platform != "win32":` 또는 `if os.name != "nt":`로 보호하세요.
- 파일은 명시적인 `encoding="utf-8"`으로 여세요. Windows의 Python 기본값은 시스템 로캘(대개 cp1252)이므로 라틴 문자 이외의 문자가 깨지거나 오류가 발생할 수 있습니다.
- `pathlib.Path`/`os.path.join`을 사용하세요 — `/`를 직접 이어 붙이지 마세요. OS가 반환하는 문자열에는 영향이 적지만, 서브프로세스에 전달할 문자열을 구성할 때 중요합니다.

주요 패턴:

### 1. 파일 인코딩

일부 환경에서는 `.env` 파일이 UTF-8이 아닌 인코딩으로 저장될 수 있습니다.

```python
try:
    load_dotenv(env_path)
except UnicodeDecodeError:
    load_dotenv(env_path, encoding="latin-1")
```

### 2. 프로세스 관리

`os.setsid()`, `os.killpg()` 및 시그널 처리는 플랫폼마다 다릅니다.

```python
import platform
if platform.system() != "Windows":
    kwargs["preexec_fn"] = os.setsid
```

### 3. 경로 구분자

문자열 연결 대신 `pathlib.Path`를 사용하세요.

## 보안 고려 사항

Hermes는 터미널에 접근할 수 있습니다. 보안이 중요합니다.

### 기존 보호 기능

| 계층                           | 구현                                                                      |
| ------------------------------- | -------------------------------------------------------------------------- |
| **Sudo 비밀번호 전달**          | 셸 인젝션 방지를 위해 `shlex.quote()` 사용                                |
| **위험한 명령 감지**            | 사용자 승인 흐름이 있는 `tools/approval.py`의 정규식 패턴                 |
| **Cron 프롬프트 인젝션**        | 명령 재정의 패턴을 차단하는 스캐너                                        |
| **쓰기 거부 목록**              | 심볼릭 링크 우회를 방지하기 위해 `os.path.realpath()`로 보호 경로 해석   |
| **스킬 가드**                   | 허브에서 설치한 스킬을 위한 보안 스캐너                                   |
| **코드 실행 샌드박스**          | 자식 프로세스에서 API 키 제거                                            |
| **컨테이너 강화**               | Docker: 모든 기능(capability) 제거, 권한 상승 금지, PID 제한             |

### 보안에 민감한 코드 기여

- 사용자 입력을 셸 명령에 삽입할 때는 항상 `shlex.quote()`를 사용하세요.
- 접근 제어 확인 전에 `os.path.realpath()`로 심볼릭 링크를 해석하세요.
- 비밀 정보를 로그에 기록하지 마세요.
- 도구 실행 주변에서 광범위한 예외를 처리하세요.
- 파일 경로나 프로세스를 변경하는 경우 모든 플랫폼에서 테스트하세요.

## 풀 리퀘스트 절차

### 브랜치 이름

```
fix/description        # Bug fixes
feat/description       # New features
docs/description       # Documentation
test/description       # Tests
refactor/description   # Code restructuring
```

### 제출 전 확인 사항

1. **테스트 실행:** CI와 동일한 조건으로 `scripts/run_tests.sh`를 실행합니다. 래퍼를 사용할 수 없거나 래퍼 외부에서 의도적으로 디버깅하는 경우에만 직접 `python -m pytest ...`를 사용합니다.
2. **수동 테스트:** `hermes`를 실행하고 변경한 코드 경로를 사용해 봅니다.
3. **멀티 플랫폼 영향 확인:** macOS, Linux, WSL2 및 네이티브 Windows를 고려합니다. 파일 I/O, 프로세스 관리, 터미널 처리, 서브프로세스 또는 시그널을 수정했다면 `scripts/check-windows-footguns.py`를 실행합니다.
4. **PR을 집중된 상태로 유지:** 하나의 논리적 변경만 포함하는 PR을 만듭니다.

### PR 설명

다음 내용을 포함하세요.

- 변경한 내용과 **이유**
- **테스트 방법**
- **테스트한 플랫폼**
- 관련 이슈 참조

### 커밋 메시지

[Conventional Commits](https://www.conventionalcommits.org/)를 사용합니다.

```
<type>(<scope>): <description>
```

| 유형       | 용도                       |
| ---------- | -------------------------- |
| `fix`      | 버그 수정                  |
| `feat`     | 새 기능                    |
| `docs`     | 문서                       |
| `test`     | 테스트                     |
| `refactor` | 코드 구조 재정리           |
| `chore`    | 빌드, CI, 의존성 업데이트  |

범위: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`

예시:

```
fix(cli): prevent crash in save_config_value when model is a string
feat(gateway): add WhatsApp multi-user session isolation
fix(security): prevent shell injection in sudo password piping
```

### 저장소 로컬 검토 체크리스트: `.agents/checks/*.md`

Hermes를 기반으로 구축되거나 Hermes가 검토하는 프로젝트는 저장소 내부 `.agents/checks/`에 변경 사항과 일치하는 검토자 체크리스트를 둘 수 있습니다. 각 파일은 해당 영역을 수정하는 변경 사항을 검토하기 전에 에이전트가 불러오는 집중형 일반 Markdown 체크리스트입니다.

```
.agents/
  checks/
    security.md        # e.g. "grep the diff for shell interpolation; check subprocess calls quote args"
    migrations.md      # e.g. "every schema change ships a backfill and a rollback note"
    public-api.md      # e.g. "exported signatures changed? flag for semver review"
```

이 체크리스트가 효과적으로 작동하도록 하는 규칙:

- 파일당 하나의 관심사만 다루고, 파일 이름은 관심사에 맞춥니다. 작은 파일은 전체를 읽고, 하나의 거대한 `checklist.md`는 훑어봅니다.
- 바람직한 사항이 아니라 검증 가능한 행동을 체크 항목으로 작성합니다("X를 실행하고 Y를 확인").
- 상단에 트리거를 명시합니다 — 어떤 경로나 변경 유형에 적용되는지 표시하여 에이전트(또는 사람)가 관련 없는 체크리스트를 빠르게 건너뛸 수 있게 합니다.
- 코드와 체크하는 규칙을 함께 발전시킬 수 있도록 해당 코드 옆 버전 관리에 보관합니다. 규칙을 변경하는 PR은 같은 diff에서 체크리스트도 변경합니다.

`.agents/checks/`가 있는 저장소에서 PR을 검토하도록 Hermes에 요청할 때는 먼저 관련 체크리스트를 읽고 그에 따라 보고하도록 지시하세요. 이렇게 하면 일반적인 검토 프롬프트로는 놓치기 쉬운 프로젝트별 기준을 적용할 수 있습니다.

## 이슈 보고

- [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)를 사용하세요.
- 운영체제, Python 버전, Hermes 버전(`hermes version`), 전체 오류 트레이스백을 포함하세요.
- 재현 단계를 포함하세요.
- 중복 이슈를 만들기 전에 기존 이슈를 확인하세요.
- 보안 취약점은 비공개로 신고해 주세요.

## 커뮤니티

- **Discord**: [discord.gg/NousResearch](https://discord.gg/NousResearch)
- **GitHub Discussions**: 설계 제안 및 아키텍처 논의
- **Skills Hub**: 전문 스킬을 업로드하고 커뮤니티와 공유

## 라이선스

기여함으로써 귀하는 자신의 기여물이 [MIT 라이선스](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE)에 따라 라이선스된다는 데 동의합니다.
