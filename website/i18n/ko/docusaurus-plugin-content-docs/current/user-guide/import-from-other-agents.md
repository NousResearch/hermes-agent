---
sidebar_position: 9
title: "다른 에이전트에서 가져오기"
description: "Claude Code(~/.claude) 또는 OpenAI Codex CLI(~/.codex) 설정(지침, 허용 목록, MCP 서버, 스킬, 메모리)을 한 번에 Hermes로 가져옵니다."
---

# 다른 에이전트에서 가져오기

`hermes import-agent`는 기존 **Claude Code** 또는 **OpenAI Codex CLI** 설정을 한 번에 Hermes로 가져옵니다. [`hermes claw migrate`](../guides/migrate-from-openclaw.md)와 같은 미리 보기 우선 패턴을 따릅니다. 아무것도 기록하기 전에 항목별 계획을 항상 확인할 수 있으며, `--dry-run`은 디스크에 절대 접근하지 않습니다.

```bash
hermes import-agent                    # auto-detect ~/.claude or ~/.codex
hermes import-agent claude-code        # import from ~/.claude
hermes import-agent codex              # import from ~/.codex
hermes import-agent claude-code --dry-run          # preview only
hermes import-agent codex --source /path/to/.codex # custom location
hermes import-agent claude-code --overwrite --yes  # replace conflicts, skip prompts
```

## 가져오는 항목

### Claude Code(`~/.claude`)

| Claude Code | Hermes |
|---|---|
| `CLAUDE.md`(전역 지침) | `~/.hermes/memories/MEMORY.md`의 메모리 항목 |
| `settings.json` → `permissions.allow`(`Bash(...)` 규칙) | `config.yaml`의 `command_allowlist` |
| `settings.json` → `permissions.deny`(`Bash(...)` 규칙) | `config.yaml`의 `approvals.deny` |
| `mcpServers`(`~/.claude.json` 및 `settings.json`에서) | `config.yaml`의 `mcp_servers` |
| `skills/<name>/`(`SKILL.md`가 있는 디렉터리) | `~/.hermes/skills/claude-code-imports/<name>/` |
| `commands/*.md`(슬래시 명령) | 참고와 함께 건너뜀 — 스킬로 변환하세요 |

Claude의 `Bash(npm run test:*)` 접두사 규칙은 `npm run test*` 글롭으로 변환됩니다. `Bash`가 아닌 권한 규칙(`Read(...)`, `WebFetch` 등)은 Claude 전용 도구를 제어하므로 매핑되지 않은 항목으로 보고되며 가져오지 않습니다.

### Codex CLI(`~/.codex`)

| Codex CLI | Hermes |
|---|---|
| `AGENTS.md`(전역 지침) | `~/.hermes/memories/MEMORY.md`의 메모리 항목 |
| `config.toml` → `[mcp_servers.*]` | `config.yaml`의 `mcp_servers` |
| `memories/*.md` | `~/.hermes/memories/MEMORY.md`의 메모리 항목 |
| `skills/<name>/`(`SKILL.md`가 있는 디렉터리) | `~/.hermes/skills/codex-imports/<name>/` |

## 절대 가져오지 않는 항목

**API 키와 자격 증명.** 자격 증명 파일(`~/.claude/.credentials.json`, `~/.codex/auth.json`)은 절대 읽지 않습니다. 또한 비밀처럼 보이는 이름의 MCP 서버 환경 변수 또는 헤더(`*_TOKEN`, `*_API_KEY`, `Authorization` 등)는 제거한 뒤 보고서에 나열하므로, 필요하면 의도적으로 다시 추가할 수 있습니다. `hermes setup`으로 공급자를 구성하거나 `~/.hermes/.env`에 비밀을 추가하세요.

## 동작 참고 사항

- **항상 먼저 미리 봅니다.** 명령은 적용하기 전에 전체 계획을 출력합니다. 대화형이 아닌 세션에서는 `--yes`를 전달하지 않으면 미리 보기 단계에서 멈춥니다.
- **대체가 아니라 병합입니다.** 메모리 항목은 기존 `MEMORY.md`와 비교해 중복을 제거하고, 허용 목록/거부 목록 패턴은 기존 `config.yaml` 항목과 병합합니다.
- **충돌은 기본적으로 건너뜁니다.** Hermes에 이미 존재하는 MCP 서버 또는 스킬은 충돌로 보고됩니다. 대체하려면 `--overwrite`를 전달하세요.
- **형식이 잘못된 파일이 실행을 중단시키지 않습니다.** 손상된 `settings.json` 또는 `config.toml`은 보고서의 항목별 오류가 되며, 나머지 항목은 계속 가져옵니다.
- OpenClaw에서 가져오는 경우에는 [`hermes claw migrate`](../guides/migrate-from-openclaw.md)를 사용하세요.
