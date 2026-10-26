---
sidebar_position: 10
title: "OpenClaw에서 마이그레이션"
description: "OpenClaw / Clawdbot 설정을 Hermes Agent로 마이그레이션하는 전체 가이드 — 무엇이 마이그레이션되는지, 설정이 어떻게 매핑되는지, 마이그레이션 후 무엇을 확인해야 하는지를 설명합니다."
---

# OpenClaw에서 마이그레이션

`hermes claw migrate`는 OpenClaw(또는 기존 Clawdbot/Moldbot) 설정을 Hermes로 가져옵니다. 이 가이드에서는 정확히 무엇이 마이그레이션되는지, 설정 키 매핑, 마이그레이션 후 확인할 사항을 다룹니다.

:::note
**Claude Code** 또는 **OpenAI Codex CLI**에서 가져오시나요? [`hermes import-agent`](../user-guide/import-from-other-agents.md)를 사용하세요.
:::

:::tip
OpenClaw 설정이 여러 프로바이더를 사용했다면 `hermes setup --portal`이 이를 하나의 OAuth로 통합합니다 — 한 번의 로그인으로 300개 이상의 모델과 Tool Gateway를 이용할 수 있습니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 빠른 시작

```bash
# Preview then migrate (always shows a preview first, then asks to confirm)
hermes claw migrate

# Preview only, no changes
hermes claw migrate --dry-run

# Full migration including API keys, skip confirmation
hermes claw migrate --preset full --migrate-secrets --yes
```

마이그레이션은 변경하기 전에 가져올 항목의 전체 미리 보기를 항상 표시합니다. 목록을 검토한 다음 계속 진행할지 확인하세요.

기본적으로 `~/.openclaw/`에서 읽습니다. 기존 `~/.clawdbot/` 또는 `~/.moltbot/` 디렉터리도 자동으로 감지합니다. 기존 설정 파일 이름(`clawdbot.json`, `moltbot.json`)도 마찬가지입니다.

## 옵션

| 옵션 | 설명 |
|--------|-------------|
| `--dry-run` | 미리 보기만 수행 — 마이그레이션될 항목을 표시한 후 중지합니다. |
| `--preset <name>` | `full`(호환되는 모든 설정) 또는 `user-data`(인프라 설정 제외). 두 프리셋 모두 기본적으로 시크릿을 가져오지 않습니다 — 명시적으로 `--migrate-secrets`를 전달하세요. |
| `--overwrite` | 충돌 시 기존 Hermes 파일을 덮어씁니다(기본값: 계획에 충돌이 있으면 적용을 거부합니다). |
| `--migrate-secrets` | API 키를 포함합니다. `--preset full`에서도 필요합니다 — 어떤 프리셋도 시크릿을 자동으로 가져오지 않습니다. |
| `--no-backup` | 마이그레이션 전 `~/.hermes/`의 zip 스냅샷을 생략합니다(기본적으로 적용 전에 단일 복원 지점 아카이브가 `~/.hermes/backups/pre-migration-*.zip`에 기록되며, `hermes import`로 복원할 수 있습니다). |
| `--source <path>` | 사용자 지정 OpenClaw 디렉터리입니다. |
| `--workspace-target <path>` | `AGENTS.md`를 배치할 위치입니다. |
| `--skill-conflict <mode>` | `skip`(기본값), `overwrite` 또는 `rename`입니다. |
| `--yes` | 미리 보기 후 확인 요청을 생략합니다. |

## 마이그레이션되는 항목

### 페르소나, 메모리 및 지침

| 항목 | OpenClaw 소스 | Hermes 대상 | 참고 |
|------|----------------|-------------------|-------|
| 페르소나 | `workspace/SOUL.md` | `~/.hermes/SOUL.md` | 직접 복사 |
| 워크스페이스 지침 | `workspace/AGENTS.md` | `--workspace-target`의 `AGENTS.md` | `--workspace-target` 플래그 필요 |
| 장기 메모리 | `workspace/MEMORY.md` | `~/.hermes/memories/MEMORY.md` | 항목으로 파싱하고 기존 항목과 병합한 뒤 중복을 제거합니다. `§` 구분자를 사용합니다. |
| 사용자 프로필 | `workspace/USER.md` | `~/.hermes/memories/USER.md` | 메모리와 동일한 항목 병합 로직을 사용합니다. |
| 일일 메모리 파일 | `workspace/memory/*.md` | `~/.hermes/memories/MEMORY.md` | 모든 일일 파일을 주 메모리에 병합합니다. |

워크스페이스 파일은 대체 경로로 `workspace.default/` 및 `workspace-main/`에서도 확인합니다(OpenClaw는 최근 버전에서 `workspace/`를 `workspace-main/`으로 이름을 바꾸었으며, 멀티 에이전트 설정에서는 `workspace-{agentId}`를 사용합니다).

### 스킬 (4개 소스)

| 소스 | OpenClaw 위치 | Hermes 대상 |
|---|---|---|
| 워크스페이스 스킬 | `workspace/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 관리형/공유 스킬 | `~/.openclaw/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 개인 크로스 프로젝트 스킬 | `~/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 프로젝트 수준 공유 스킬 | `workspace/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |

스킬 충돌은 `--skill-conflict`로 처리합니다. `skip`은 기존 Hermes 스킬을 유지하고, `overwrite`는 이를 교체하며, `rename`은 `-imported` 사본을 생성합니다.

### 모델 및 프로바이더 설정

| 항목 | OpenClaw 설정 경로 | Hermes 대상 | 참고 |
|------|---------------------|-------------------|-------|
| 기본 모델 | `agents.defaults.model` | `config.yaml` → `model` | 문자열 또는 `{primary, fallbacks}` 객체일 수 있습니다. |
| 사용자 지정 프로바이더 | `models.providers.*` | `config.yaml` → `custom_providers` (다음 `hermes update` 설정 마이그레이션 시 표준 `providers:` 딕셔너리로 자동 마이그레이션됨) | `baseUrl`, `apiType`/`api`를 매핑하며, 짧은 값(`openai`, `anthropic`)과 하이픈으로 연결된 값(`openai-completions`, `anthropic-messages`, `google-generative-ai`)을 모두 처리합니다. |
| 프로바이더 API 키 | `models.providers.*.apiKey` | `~/.hermes/.env` | `--migrate-secrets`가 필요합니다. 아래 [API 키 확인](#api-key-resolution)을 참조하세요. |

### 에이전트 동작

| 항목 | OpenClaw 설정 경로 | Hermes 설정 경로 | 매핑 |
|------|---------------------|-------------------|---------|
| 최대 턴 수 | `agents.defaults.timeoutSeconds` | `agent.max_turns` | `timeoutSeconds / 10`, 최대 200 |
| 상세 모드 | `agents.defaults.verboseDefault` | `agent.verbose` | "off" / "on" / "full" |
| 추론 강도 | `agents.defaults.thinkingDefault` | `agent.reasoning_effort` | "always"/"high"/"xhigh" → "high", "auto"/"medium"/"adaptive" → "medium", "off"/"low"/"none"/"minimal" → "low" |
| 압축 | `agents.defaults.compaction.mode` | `compression.enabled` | "off" → false, 그 외 값 → true |
| 압축 모델 | `agents.defaults.compaction.model` | `compression.summary_model` | 문자열을 직접 복사합니다. |
| 사람 지연 | `agents.defaults.humanDelay.mode` | `human_delay.mode` | "natural" / "custom" / "off" |
| 사람 지연 시간 | `agents.defaults.humanDelay.minMs` / `.maxMs` | `human_delay.min_ms` / `.max_ms` | 직접 복사 |
| 시간대 | `agents.defaults.userTimezone` | `timezone` | 문자열을 직접 복사합니다. |
| 실행 제한 시간 | `tools.exec.timeoutSec` | `terminal.timeout` | 직접 복사(`timeout`이 아니라 `timeoutSec` 필드) |
| Docker 샌드박스 | `agents.defaults.sandbox.backend` | `terminal.backend` | "docker" → "docker" |
| Docker 이미지 | `agents.defaults.sandbox.docker.image` | `terminal.docker_image` | 직접 복사 |

### 세션 초기화 정책

| OpenClaw 설정 경로 | Hermes 설정 경로 | 참고 |
|---------------------|-------------------|-------|
| `session.reset.mode` | `session_reset.mode` | "daily", "idle" 또는 둘 다 |
| `session.reset.atHour` | `session_reset.at_hour` | 매일 초기화할 시간(0–23) |
| `session.reset.idleMinutes` | `session_reset.idle_minutes` | 비활성 시간(분) |

참고: OpenClaw에는 `session.resetTriggers`(예: `["daily", "idle"]`와 같은 단순 문자열 배열)도 있습니다. 구조화된 `session.reset`이 없으면 마이그레이션은 `resetTriggers`를 바탕으로 추론합니다.

### MCP 서버

| OpenClaw 필드 | Hermes 필드 | 참고 |
|----------------|-------------|-------|
| `mcp.servers.*.command` | `mcp_servers.*.command` | Stdio 전송 |
| `mcp.servers.*.args` | `mcp_servers.*.args` | |
| `mcp.servers.*.env` | `mcp_servers.*.env` | |
| `mcp.servers.*.cwd` | `mcp_servers.*.cwd` | |
| `mcp.servers.*.url` | `mcp_servers.*.url` | HTTP/SSE 전송 |
| `mcp.servers.*.tools.include` | `mcp_servers.*.tools.include` | 도구 필터링 |
| `mcp.servers.*.tools.exclude` | `mcp_servers.*.tools.exclude` | |

### TTS(텍스트 음성 변환)

TTS 설정은 우선순위에 따라 **두** OpenClaw 설정 위치에서 읽습니다.

1. `messages.tts.providers.{provider}.*` (표준 위치)
2. 최상위 `talk.providers.{provider}.*` (대체 위치)
3. 기존 플랫 키 `messages.tts.{provider}.*` (가장 오래된 형식)

| 항목 | Hermes 대상 |
|------|-------------------|
| 프로바이더 이름 | `config.yaml` → `tts.provider` |
| ElevenLabs 음성 ID | `config.yaml` → `tts.elevenlabs.voice_id` |
| ElevenLabs 모델 ID | `config.yaml` → `tts.elevenlabs.model_id` |
| OpenAI 모델 | `config.yaml` → `tts.openai.model` |
| OpenAI 음성 | `config.yaml` → `tts.openai.voice` |
| Edge TTS 음성 | `config.yaml` → `tts.edge.voice` (OpenClaw는 "edge"를 "microsoft"로 이름을 바꾸었지만 둘 다 인식됩니다) |
| TTS 자산 | `~/.hermes/tts/` (파일 복사) |

### 메시징 플랫폼

| 플랫폼 | OpenClaw 설정 경로 | Hermes `.env` 변수 | 참고 |
|----------|--------------------------|----------------------|-------|
| Telegram | `channels.telegram.botToken` 또는 `.accounts.default.botToken` | `TELEGRAM_BOT_TOKEN` | 토큰은 문자열 또는 [SecretRef](#secretref-handling)일 수 있습니다. 플랫 레이아웃과 accounts 레이아웃을 모두 지원합니다. |
| Telegram | `credentials/telegram-default-allowFrom.json` | `TELEGRAM_ALLOWED_USERS` | `allowFrom[]` 배열을 쉼표로 결합합니다. |
| Discord | `channels.discord.token` 또는 `.accounts.default.token` | `DISCORD_BOT_TOKEN` | |
| Discord | `channels.discord.allowFrom` 또는 `.accounts.default.allowFrom` | `DISCORD_ALLOWED_USERS` | |
| Slack | `channels.slack.botToken` 또는 `.accounts.default.botToken` | `SLACK_BOT_TOKEN` | |
| Slack | `channels.slack.appToken` 또는 `.accounts.default.appToken` | `SLACK_APP_TOKEN` | |
| Slack | `channels.slack.allowFrom` 또는 `.accounts.default.allowFrom` | `SLACK_ALLOWED_USERS` | |
| WhatsApp | `channels.whatsapp.allowFrom` 또는 `.accounts.default.allowFrom` | `WHATSAPP_ALLOWED_USERS` | Baileys QR 페어링으로 인증 — 마이그레이션 후 다시 페어링해야 합니다. |
| Signal | `channels.signal.account` 또는 `.accounts.default.account` | `SIGNAL_ACCOUNT` | |
| Signal | `channels.signal.httpUrl` 또는 `.accounts.default.httpUrl` | `SIGNAL_HTTP_URL` | |
| Signal | `channels.signal.allowFrom` 또는 `.accounts.default.allowFrom` | `SIGNAL_ALLOWED_USERS` | |
| Matrix | `channels.matrix.accessToken` 또는 `.accounts.default.accessToken` | `MATRIX_ACCESS_TOKEN` | `accessToken`을 사용합니다(`botToken`이 아님). |
| Mattermost | `channels.mattermost.botToken` 또는 `.accounts.default.botToken` | `MATTERMOST_BOT_TOKEN` | |

### 기타 설정

| 항목 | OpenClaw 경로 | Hermes 경로 | 참고 |
|------|-------------|-------------|-------|
| 승인 모드 | `approvals.exec.mode` | `config.yaml` → `approvals.mode` | "auto"→"off", "always"→"manual", "smart"→"smart" |
| 명령어 허용 목록 | `exec-approvals.json` | `config.yaml` → `command_allowlist` | 패턴을 병합하고 중복을 제거합니다. |
| 브라우저 CDP URL | `browser.cdpUrl` | `config.yaml` → `browser.cdp_url` | |
| 브라우저 헤드리스 | `browser.headless` | `config.yaml` → `browser.headless` | |
| Brave 검색 키 | `tools.web.search.brave.apiKey` | `.env` → `BRAVE_API_KEY` | `--migrate-secrets`가 필요합니다. |
| Gateway 인증 토큰 | `gateway.auth.token` | `.env` → `HERMES_GATEWAY_TOKEN` | `--migrate-secrets`가 필요합니다. |
| 작업 디렉터리 | `agents.defaults.workspace` | `config.yaml` → `terminal.cwd` | 기존 마이그레이션에서는 호환성을 위해 `MESSAGING_CWD`가 계속 생성될 수 있습니다. |

### 보관됨(직접적인 Hermes 대응 항목 없음)

다음 항목은 수동 검토를 위해 `~/.hermes/migration/openclaw/<timestamp>/archive/`에 저장됩니다.

| 항목 | 아카이브 파일 | Hermes에서 다시 만드는 방법 |
|------|-------------|--------------------------|
| `IDENTITY.md` | `archive/workspace/IDENTITY.md` | `SOUL.md`에 병합 |
| `TOOLS.md` | `archive/workspace/TOOLS.md` | Hermes에는 기본 제공 도구 지침이 있습니다. |
| `HEARTBEAT.md` | `archive/workspace/HEARTBEAT.md` | 주기적인 작업에는 cron 작업 사용 |
| `BOOTSTRAP.md` | `archive/workspace/BOOTSTRAP.md` | 컨텍스트 파일 또는 스킬 사용 |
| Cron 작업 | `archive/cron-config.json` | `hermes cron create`로 다시 생성 |
| 플러그인 | `archive/plugins-config.json` | [플러그인 가이드](/user-guide/features/hooks) 참조 |
| Hooks/webhooks | `archive/hooks-config.json` | `hermes webhook` 또는 gateway hooks 사용 |
| 메모리 백엔드 | `archive/memory-backend-config.json` | `hermes honcho`를 통해 설정 |
| 스킬 레지스트리 | `archive/skills-registry-config.json` | `hermes skills config` 사용 |
| UI/아이덴티티 | `archive/ui-identity-config.json` | `/skin` 명령어 사용 |
| 로깅 | `archive/logging-diagnostics-config.json` | `config.yaml`의 logging 섹션에서 설정 |
| 멀티 에이전트 목록 | `archive/agents-list.json` | Hermes 프로필 사용 |
| 채널 바인딩 | `archive/bindings.json` | 플랫폼별로 수동 설정 |
| 복잡한 채널 | `archive/channels-deep-config.json` | 플랫폼 설정을 수동으로 구성 |

## API 키 확인

`--migrate-secrets`가 활성화되면 다음 **네 가지 소스**에서 우선순위에 따라 API 키를 수집합니다.

1. **설정 값** — `openclaw.json`의 `models.providers.*.apiKey` 및 TTS 프로바이더 키
2. **환경 파일** — `~/.openclaw/.env`(`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` 등의 키)
3. **설정 env 하위 객체** — `openclaw.json` → `"env"` 또는 `"env"."vars"`(일부 설정은 별도의 `.env` 파일 대신 여기에 키를 저장합니다)
4. **인증 프로필** — `~/.openclaw/agents/main/agent/auth-profiles.json`(에이전트별 자격 증명)

설정 값이 우선합니다. 이후 각 소스는 아직 채워지지 않은 값을 보완합니다.

### 지원되는 키 대상

`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `ZAI_API_KEY`, `MINIMAX_API_KEY`, `ELEVENLABS_API_KEY`, `TELEGRAM_BOT_TOKEN`, `VOICE_TOOLS_OPENAI_KEY`

이 허용 목록에 없는 키는 절대 복사되지 않습니다.

## SecretRef 처리

토큰 및 API 키에 대한 OpenClaw 설정 값은 다음 세 가지 형식일 수 있습니다.

```json
// Plain string
"channels": { "telegram": { "botToken": "123456:ABC-DEF..." } }

// Environment template
"channels": { "telegram": { "botToken": "${TELEGRAM_BOT_TOKEN}" } }

// SecretRef object
"channels": { "telegram": { "botToken": { "source": "env", "id": "TELEGRAM_BOT_TOKEN" } } }
```

마이그레이션은 세 형식을 모두 확인합니다. 환경 템플릿과 `source: "env"`인 SecretRef 객체의 경우 `~/.openclaw/.env`와 `openclaw.json`의 env 하위 객체에서 값을 조회합니다. `source: "file"` 또는 `source: "exec"`인 SecretRef 객체는 자동으로 확인할 수 없습니다 — 마이그레이션에서 경고하며, 해당 값은 `hermes config set`을 통해 Hermes에 수동으로 추가해야 합니다.

## 마이그레이션 후

1. **마이그레이션 보고서 확인** — 완료 시 마이그레이션, 건너뜀, 충돌 항목의 수가 출력됩니다.

2. **보관된 파일 검토** — `~/.hermes/migration/openclaw/<timestamp>/archive/`의 모든 항목은 수동 확인이 필요합니다.

3. **새 세션 시작** — 가져온 스킬과 메모리 항목은 현재 세션이 아니라 새 세션에서 적용됩니다.

4. **API 키 확인** — `hermes status`를 실행하여 프로바이더 인증을 확인합니다.

5. **메시징 테스트** — 플랫폼 토큰을 마이그레이션했다면 gateway를 다시 시작합니다: `systemctl --user restart hermes-gateway`

6. **세션 정책 확인** — `hermes config show`를 실행하여 `session_reset` 값이 예상과 일치하는지 확인합니다.

7. **WhatsApp 다시 페어링** — WhatsApp은 토큰 마이그레이션이 아니라 QR 코드 페어링(Baileys)을 사용합니다. 페어링하려면 `hermes whatsapp`을 실행하세요.

8. **아카이브 정리** — 모든 것이 정상적으로 작동하는지 확인한 후 `hermes claw cleanup`을 실행하여 남은 OpenClaw 디렉터리 이름을 `.pre-migration/`으로 변경합니다(상태 혼동 방지).

## 문제 해결

### "OpenClaw directory not found"

마이그레이션은 `~/.openclaw/`, `~/.clawdbot/`, `~/.moltbot/` 순서로 확인합니다. 설치 위치가 다른 경우 `--source /path/to/your/openclaw`를 사용하세요.

### "No provider API keys found"

OpenClaw 버전에 따라 키가 여러 위치에 저장될 수 있습니다. `models.providers.*.apiKey` 아래의 `openclaw.json`에 인라인으로 저장되거나, `~/.openclaw/.env`, `openclaw.json`의 `"env"` 하위 객체 또는 `agents/main/agent/auth-profiles.json`에 있을 수 있습니다. 마이그레이션은 네 위치를 모두 확인합니다. 키가 `source: "file"` 또는 `source: "exec"` SecretRef를 사용하는 경우 자동으로 확인할 수 없습니다 — `hermes config set`을 통해 추가하세요.

### 마이그레이션 후 스킬이 표시되지 않음

가져온 스킬은 `~/.hermes/skills/openclaw-imports/`에 저장됩니다. 스킬을 적용하려면 새 세션을 시작하거나 `/skills`를 실행하여 로드되었는지 확인하세요.

### TTS 음성이 마이그레이션되지 않음

OpenClaw는 TTS 설정을 `messages.tts.providers.*`와 최상위 `talk` 설정, 두 곳에 저장합니다. 마이그레이션은 두 위치를 모두 확인합니다. OpenClaw UI를 통해 음성 ID를 설정했다면(다른 경로에 저장됨) 수동으로 설정해야 할 수 있습니다: `hermes config set tts.elevenlabs.voice_id YOUR_VOICE_ID`.
