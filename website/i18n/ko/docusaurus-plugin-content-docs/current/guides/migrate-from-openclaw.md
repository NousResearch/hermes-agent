---
sidebar_position: 10
title: "OpenClaw에서 마이그레이션"
description: "OpenClaw 사용자를 위한 마이그레이션 가이드 — 구성 번역, 도구 매핑 및 아키텍처 차이점"
---

# OpenClaw에서 마이그레이션하기

OpenClaw 런타임에서 Hermes Agent로 마이그레이션하시는 것을 환영합니다! Hermes는 단일 파일 스크립트였던 원래의 OpenClaw 설계를 완전한 프로덕션급 에이전트 시스템으로 발전시킨 공식 후속 프로젝트입니다.

이 가이드는 기존의 `openclaw.json` 구성을 Hermes의 `config.yaml`로 변환하고 주요 아키텍처의 차이점을 이해하는 데 도움을 줍니다.

---

## 1분 요약

- **새로운 이름, 새로운 CLI:** `openclaw start` 대신, 이제 `hermes chat` (CLI의 경우) 또는 `hermes gateway` (메시징의 경우)를 실행합니다.
- **새로운 구성:** 모든 설정은 더 이상 `~/.openclaw`가 아닌 `~/.hermes/config.yaml`에 있습니다. (환경 변수는 `~/.hermes/.env`에 있습니다).
- **프로바이더 분리:** OpenClaw는 모델과 메시징 프로바이더를 단일 구성 파일에 섞어서 사용했습니다. Hermes는 이를 명확하게 분리합니다.
- **도구 아키텍처:** 플러그인은 이제 표준화된 서명(signature)을 가진 Python 클래스로 사용됩니다.
- **메모리 지속성:** 내장된 `MEMORY.md` 외에도 이제 PostgreSQL/Redis 메모리 백엔드를 연결할 수 있습니다.

**빠른 변환 도구:** 수동으로 매핑하고 싶지 않다면, 내장된 마이그레이션 명령어를 실행하세요:

```bash
hermes auth import-openclaw
```
이 명령어는 기존의 `~/.openclaw` 디렉토리를 읽고 설정을 새로운 Hermes 형식으로 포팅(port)합니다.

---

## 구성 파일 (Config) 매핑

OpenClaw의 플랫(flat) JSON 구성 체계는 Hermes에서 모듈화된 YAML 구성 체계로 재구성되었습니다.

### 모델 구성

| OpenClaw (`openclaw.json`) | Hermes (`config.yaml`) | 변경 사항 요약 |
|---|---|---|
| `"provider": "anthropic"` | `model.provider: anthropic` | 중첩된 `model` 블록 내부로 이동 |
| `"model": "claude-3-opus-20240229"` | `model.default: claude-3-opus-20240229` | 이름을 `model`에서 `default`로 변경 |
| `"context_window": 200000` | `model.context_length: 200000` | 이름을 `context_window`에서 `context_length`로 변경 |
| `"temperature": 0.7` | `model.temperature: 0.7` | `model` 블록 내부로 이동 |
| `"system_prompt_extension": "..."` | `agent.custom_instructions: "..."` | `agent` 블록으로 이동 및 이름 변경 |

**OpenClaw 예시:**
```json
{
  "provider": "anthropic",
  "model": "claude-3-opus-20240229",
  "temperature": 0.7
}
```

**Hermes 예시:**
```yaml
model:
  provider: anthropic
  default: claude-3-opus-20240229
  temperature: 0.7
```

### 터미널 및 파일 시스템

| OpenClaw (`openclaw.json`) | Hermes (`config.yaml`) | 변경 사항 요약 |
|---|---|---|
| `"sandbox": "docker"` | `terminal.backend: docker` | `terminal` 하위 시스템 내부로 이동 |
| `"docker_image": "ubuntu:latest"` | `terminal.docker_image: "ubuntu:latest"` | 이름을 `docker_image`로 변경 |
| `"workspace_dir": "/path/to/work"` | `terminal.cwd: "/path/to/work"` | `cwd` (Current Working Directory)로 매핑 |
| `"allowed_paths": ["/tmp"]` | `security.allowed_paths: ["/tmp"]` | 전용 `security` 블록으로 이동 |

### 도구 (Tools) 및 통합 (Integrations)

| OpenClaw (`openclaw.json`) | Hermes (`config.yaml`) | 변경 사항 요약 |
|---|---|---|
| `"enable_web_search": true` | 도구는 이제 기본적으로 활성화됨 | `hermes tools`를 통해 비활성화할 수 있습니다 |
| `"browser_type": "playwright"` | `browser.backend: browser-use` | 새로운 브라우저 백엔드. Playwright는 더 이상 직접 사용되지 않습니다 |
| `"tts_engine": "elevenlabs"` | `tts.provider: elevenlabs` | `tts` 블록 내부로 이동 |
| `"stt_engine": "whisper"` | `voice.stt_provider: openai` | `voice` 블록으로 이름 변경 |

---

## 환경 변수 (.env)

API 키 관리는 기본적으로 동일하게 유지되지만 파일 위치가 변경되었습니다.

- **이전:** `~/.openclaw/.env`
- **신규:** `~/.hermes/.env`

API 키 환경 변수의 이름은 대부분 동일하게 유지됩니다 (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` 등).

몇 가지 중요한 변경 사항이 있습니다:
- `CLAW_DISCORD_TOKEN` ➔ `DISCORD_BOT_TOKEN`
- `CLAW_TELEGRAM_TOKEN` ➔ `TELEGRAM_BOT_TOKEN`
- `CLAW_SLACK_TOKEN` ➔ `SLACK_BOT_TOKEN`

---

## 메시징 플랫폼 및 게이트웨이

OpenClaw는 설정 파일에 직접 메시징 구성을 가졌습니다. Hermes에서는 메시징 봇이 **Gateway**라는 별도의 프로세스로 실행됩니다.

**OpenClaw를 사용하여 Discord 봇 실행하기:**
```bash
openclaw start --discord
```

**Hermes를 사용하여 Discord 봇 실행하기:**
1. `~/.hermes/config.yaml`에 활성화:
```yaml
gateway:
  platforms:
    discord:
      enabled: true
```
2. 게이트웨이 실행:
```bash
hermes gateway
```

이 접근 방식을 통해 Hermes는 단일 프로세스에서 여러 메시징 플랫폼(Telegram, Discord, Slack)에 동시에 연결할 수 있습니다. 각 플랫폼에 대한 개별 설정 지침은 [메시징 개요](/user-guide/messaging) 가이드를 참조하세요.

---

## 사용자 권한 및 보안

OpenClaw의 단순한 `allowed_users` 목록은 보다 강력한 권한 부여 시스템으로 대체되었습니다.

**OpenClaw:**
```json
{
  "allowed_users": ["123456789"]
}
```

**Hermes:**
- 플랫폼별 허용 목록은 이제 `~/.hermes/.env`에 구성됩니다 (예: `TELEGRAM_ALLOWED_USERS=123456789`).
- 위험한 명령어 승인은 `config.yaml`의 `approvals` 블록에서 구성됩니다.

```yaml
approvals:
  mode: manual  # manual, smart, off
  timeout: 60
```
자세한 내용은 [보안](/user-guide/security) 가이드를 참조하세요.

---

## 도구(Tools) 및 플러그인(Plugins) 차이점

사용자 지정 도구를 작성한 경우 형식을 조정해야 합니다.

### OpenClaw 도구 예시
```python
def my_custom_tool(arg1: str):
    """도구 설명입니다.
    
    Args:
        arg1: 첫 번째 인수 설명
    """
    return f"Result: {arg1}"

CLAW_TOOLS = [my_custom_tool]
```

### Hermes 플러그인(스킬) 예시
Hermes는 스킬이라는 개념을 사용합니다. 스킬은 사용자 정의 도구와 지침을 하나의 재사용 가능한 모듈로 결합합니다. 스킬은 `~/.hermes/plugins/` 디렉토리에 마크다운 형식(`SKILL.md`)으로 저장되거나 진입점(entrypoints)을 포함하는 Python 패키지로 생성될 수 있습니다.

도구를 패키징하는 방법에 대한 자세한 지침은 [플러그인 구축](/guides/build-a-hermes-plugin) 및 [플러그인](/user-guide/features/plugins) 문서를 참조하세요. Hermes의 도구 아키텍처는 타입 검사, 문서화 및 보안 샌드박싱에 더 엄격한 패턴을 사용합니다.

---

## 메모리 및 컨텍스트 (Memory & Context)

OpenClaw의 메모리 시스템(단일 JSON 파일)은 두 가지 새로운 시스템으로 발전했습니다:

1. **상태 기반 (State-based):** 에이전트가 직접 제어하는 세션 간 메모리를 위한 `MEMORY.md` 및 `USER.md`
2. **프로바이더 기반 (Provider-based):** 외부 벡터 데이터베이스 및 지식 그래프에 연결되는 [메모리 프로바이더](/user-guide/features/memory-providers) 블록

**OpenClaw:**
```json
{
  "memory_file": "/path/to/memory.json"
}
```

**Hermes:**
`MEMORY.md`와 `USER.md`는 자동으로 관리됩니다. 고급 메모리의 경우 구성 파일에 프로바이더를 추가하세요:

```yaml
memory:
  provider: supermemory
```

---

## 단계별 마이그레이션 권장 사항

1. **Hermes 설치:** [빠른 시작](/getting-started/quickstart)의 지침을 따릅니다.
2. **자동 가져오기 실행:** `hermes auth import-openclaw` 명령어를 실행하여 가능한 한 많은 설정을 자동 변환합니다.
3. **`.env` 파일 확인:** `~/.openclaw/.env`에서 누락된 API 키를 `~/.hermes/.env`로 복사합니다. (디스코드/슬랙/텔레그램의 변수 이름이 변경되었는지 확인하세요).
4. **구성 수정:** `hermes config edit` 명령어를 실행하여 새로운 구성을 확인하고, 필요한 경우 조정합니다.
5. **테스트:** `hermes chat`을 실행하여 로컬 CLI 모드가 작동하는지 확인합니다.
6. **메시징 테스트 (선택 사항):** 봇을 실행 중이었다면 `hermes gateway`를 실행하여 봇이 제대로 연결되고 응답하는지 확인합니다.

문제 해결 및 추가 옵션에 대해서는 [구성 가이드](/user-guide/configuration)를 참조하세요.
