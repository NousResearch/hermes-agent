---
sidebar_position: 4
title: "Contributing"
description: "Hermes 에이전트에 기여하는 방법 - 개발 설정, 코드 스타일, PR 프로세스"
---

# 기여하기

Hermes 에이전트에 기여해 주셔서 감사합니다! 이 가이드는 개발 환경 설정, 코드베이스 이해, 그리고 PR을 병합하는 과정을 다룹니다.

## 기여 우선순위

다음과 같은 순서로 기여를 가치 있게 평가합니다:

1. **버그 수정** — 충돌, 잘못된 동작, 데이터 손실
2. **크로스 플랫폼 호환성** — macOS, 다양한 Linux 배포판, WSL2
3. **보안 강화** — 셸 주입(shell injection), 프롬프트 주입, 경로 탐색(path traversal)
4. **성능 및 견고성** — 재시도 로직, 오류 처리, 우아한 성능 저하(graceful degradation)
5. **새로운 스킬** — 광범위하게 유용한 스킬 ([스킬 생성하기](creating-skills.md) 참조)
6. **새로운 도구** — 거의 필요하지 않음; 대부분의 기능은 스킬이어야 함
7. **문서화** — 수정, 명확한 설명, 새로운 예제

## 일반적인 기여 경로

- Hermes 핵심을 수정하지 않고 커스텀/로컬 도구를 만들고 싶으신가요? [Hermes 플러그인 빌드하기](../guides/build-a-hermes-plugin.md)로 시작하세요.
- Hermes 자체를 위한 새로운 내장 핵심 도구를 만들고 싶으신가요? [도구 추가하기](./adding-tools.md)로 시작하세요.
- 새로운 스킬을 만들고 싶으신가요? [스킬 생성하기](./creating-skills.md)로 시작하세요.
- 새로운 추론 프로바이더를 만들고 싶으신가요? [프로바이더 추가하기](./adding-providers.md)로 시작하세요.

## 개발 설정

### 사전 요구 사항

| 요구 사항 | 참고 |
|-------------|-------|
| **Git** | `git-lfs` 확장이 설치되어 있어야 합니다 |
| **Python 3.11+** | 없는 경우 uv가 설치해 줍니다 |
| **uv** | 빠른 Python 패키지 관리자 ([설치](https://docs.astral.sh/uv/)) |
| **Node.js 20+** | 선택 사항 — 브라우저 도구 및 WhatsApp 브리지에 필요 (루트 `package.json` 엔진과 일치) |

### 복제 및 설치

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Python 3.11로 venv 생성
uv venv venv --python 3.11
export VIRTUAL_ENV="$(pwd)/venv"

# 모든 extras와 함께 설치 (메시징, 크론, CLI 메뉴, 개발 도구)
uv pip install -e ".[all,dev]"

# 선택 사항: 브라우저 도구
npm install
```

### 개발용 구성

```bash
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env

# 최소한 LLM 프로바이더 키 추가:
echo 'OPENROUTER_API_KEY=sk-or-v1-your-key' >> ~/.hermes/.env
```

### 실행

```bash
# 전역 접근을 위한 심볼릭 링크
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes

# 확인
hermes doctor
hermes chat -q "Hello"
```

### 테스트 실행

```bash
pytest tests/ -v
```

## 코드 스타일

- **PEP 8**: 실용적인 예외 허용 (엄격한 줄 길이 강제 없음)
- **주석**: 명확하지 않은 의도, 트레이드오프 또는 API의 특이점을 설명할 때만 사용
- **오류 처리**: 특정 예외 포착. 예상치 못한 오류의 경우 `exc_info=True`와 함께 `logger.warning()`/`logger.error()` 사용
- **크로스 플랫폼**: Unix라고 단정하지 마세요 (아래 참조)
- **프로필 안전 경로**: 절대 `~/.hermes`를 하드코딩하지 마세요 — 코드 경로에는 `hermes_constants`의 `get_hermes_home()`을 사용하고 사용자 대상 메시지에는 `display_hermes_home()`을 사용하세요. 전체 규칙은 [AGENTS.md](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md#profiles-multi-instance-support)를 참조하세요.

## 크로스 플랫폼 호환성

Hermes는 **Linux, macOS, WSL2 및 네이티브 Windows(PowerShell 설치를 통한)**를 공식적으로 지원합니다. 네이티브 Windows는 셸 명령어에 대해 Git Bash([Git for Windows](https://git-scm.com/download/win)에서 제공)를 사용합니다. 몇 가지 기능은 POSIX 커널 원형(primitives)을 필요로 하며 제한되어 있습니다. 대시보드의 내장 PTY 터미널 창(`/chat` 탭)은 WSL2 전용입니다. Windows 중심의 개발을 하는 경우, 푸시하기 전에 Windows의 위험 요소 검사(`scripts/check-windows-footguns.py`)를 실행하세요.

코드를 기여할 때 다음 규칙을 염두에 두세요:

- **방어되지 않은 `signal.SIGKILL` 참조를 추가하지 마세요.** Windows에서는 정의되어 있지 않습니다. `gateway.status.terminate_pid(pid, force=True)`(Windows에서는 `taskkill /T /F`, POSIX에서는 SIGKILL을 수행하는 중앙 집중식 원형)를 통해 라우팅하거나 `getattr(signal, "SIGKILL", signal.SIGTERM)`으로 폴백하세요.
- **`os.kill(pid, 0)` 프로브에서 `ProcessLookupError`와 함께 `OSError`를 잡으세요.** Windows는 이미 사라진 PID에 대해 `ProcessLookupError` 대신 `OSError`(WinError 87, "parameter is incorrect")를 발생시킵니다.
- **터미널에 POSIX 의미론(semantics)을 강요하지 마세요.** `os.setsid`, `os.killpg`, `os.getpgid`, `os.fork`는 모두 Windows에서 오류를 발생시킵니다 — `if sys.platform != "win32":` 또는 `if os.name != "nt":`로 방어하세요.
- **명시적 `encoding="utf-8"`로 파일을 엽니다.** Windows의 Python 기본값은 시스템 로케일(종종 cp1252)이며, 이는 라틴어가 아닌 텍스트에서 글자가 깨지거나 충돌을 일으킵니다.
- **`pathlib.Path` / `os.path.join`을 사용하세요 — 절대 `/`로 수동 연결하지 마세요.** 이는 OS가 우리에게 돌려주는 문자열에는 덜 중요하지만, 우리가 하위 프로세스에 전달하기 위해 구성하는 문자열에는 더 중요합니다.

주요 패턴:

### 1. `termios`와 `fcntl`은 Unix 전용입니다

항상 `ImportError`와 `NotImplementedError`를 모두 잡으세요:

```python
try:
    from simple_term_menu import TerminalMenu
    menu = TerminalMenu(options)
    idx = menu.show()
except (ImportError, NotImplementedError):
    # 폴백: 번호 매기기 메뉴
    for i, opt in enumerate(options):
        print(f"  {i+1}. {opt}")
    idx = int(input("Choice: ")) - 1
```

### 2. 파일 인코딩

일부 환경에서는 UTF-8이 아닌 인코딩으로 `.env` 파일을 저장할 수 있습니다:

```python
try:
    load_dotenv(env_path)
except UnicodeDecodeError:
    load_dotenv(env_path, encoding="latin-1")
```

### 3. 프로세스 관리

`os.setsid()`, `os.killpg()` 및 신호 처리는 플랫폼마다 다릅니다:

```python
import platform
if platform.system() != "Windows":
    kwargs["preexec_fn"] = os.setsid
```

### 4. 경로 구분자

`/`를 사용한 문자열 연결 대신 `pathlib.Path`를 사용하세요.

## 보안 고려 사항

Hermes는 터미널에 접근할 수 있습니다. 보안은 중요합니다.

### 기존 보호 조치

| 계층 | 구현 |
|-------|---------------|
| **Sudo 비밀번호 파이핑** | 셸 주입을 방지하기 위해 `shlex.quote()` 사용 |
| **위험한 명령어 감지** | 사용자 승인 흐름과 함께 `tools/approval.py`의 정규식 패턴 |
| **크론 프롬프트 주입** | 스캐너가 명령어 재정의 패턴 차단 |
| **쓰기 거부 목록** | 심볼릭 링크 우회를 방지하기 위해 `os.path.realpath()`를 통해 확인된 보호된 경로 |
| **스킬 보호** | 허브에 설치된 스킬에 대한 보안 스캐너 |
| **코드 실행 샌드박스** | 자식 프로세스는 API 키가 제거된 상태로 실행 |
| **컨테이너 강화** | Docker: 모든 기능 삭제, 권한 상승 불가, PID 제한 |

### 보안에 민감한 코드 기여하기

- 셸 명령어에 사용자 입력을 삽입할 때는 항상 `shlex.quote()`를 사용하세요
- 접근 제어 확인 전에 `os.path.realpath()`로 심볼릭 링크를 해결하세요
- 비밀(secrets)을 로그에 남기지 마세요
- 도구 실행과 관련된 광범위한 예외를 잡으세요
- 파일 경로나 프로세스를 건드리는 변경 사항이 있다면 모든 플랫폼에서 테스트하세요

## 풀 리퀘스트(PR) 프로세스

### 브랜치 명명 규칙

```
fix/description        # 버그 수정
feat/description       # 새로운 기능
docs/description       # 문서
test/description       # 테스트
refactor/description   # 코드 재구조화
```

### 제출하기 전에

1. **테스트 실행**: `pytest tests/ -v`
2. **수동 테스트**: `hermes`를 실행하고 변경한 코드 경로를 연습해 보세요
3. **크로스 플랫폼 영향 확인**: macOS와 다양한 Linux 배포판을 고려하세요
4. **PR은 한 가지에 집중**: PR당 하나의 논리적 변경 사항

### PR 설명

다음을 포함하세요:
- **무엇이** 왜 **변경**되었는지
- **어떻게 테스트**하는지
- 테스트한 **플랫폼**
- 관련된 모든 이슈 참조

### 커밋 메시지

우리는 [Conventional Commits](https://www.conventionalcommits.org/ko/v1.0.0/)를 사용합니다:

```
<type>(<scope>): <description>
```

| 유형(Type) | 용도 |
|------|---------|
| `fix` | 버그 수정 |
| `feat` | 새로운 기능 |
| `docs` | 문서 |
| `test` | 테스트 |
| `refactor` | 코드 재구조화 |
| `chore` | 빌드, CI, 종속성 업데이트 |

스코프(Scopes): `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`

예시:
```
fix(cli): prevent crash in save_config_value when model is a string
feat(gateway): add WhatsApp multi-user session isolation
fix(security): prevent shell injection in sudo password piping
```

## 문제 보고 (Reporting Issues)

- [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)를 사용하세요
- 포함할 내용: OS, Python 버전, Hermes 버전 (`hermes version`), 전체 오류 추적(traceback)
- 재현 단계를 포함하세요
- 중복을 방지하기 위해 이슈를 생성하기 전에 기존 이슈를 확인하세요
- 보안 취약점의 경우 비공개로 보고해 주세요

## 커뮤니티

- **Discord**: [discord.gg/NousResearch](https://discord.gg/NousResearch)
- **GitHub Discussions**: 설계 제안 및 아키텍처 토론
- **Skills Hub**: 특화된 스킬을 업로드하고 커뮤니티와 공유하세요

## 라이선스

기여함으로써 귀하의 기여가 [MIT License](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE)에 따라 라이선스가 부여된다는 것에 동의하게 됩니다.
