---
title: "코드베이스 소유권 맵"
description: "각 하위 시스템에 속한 디렉터리와 해당 하위 시스템의 올바른 문서 진입점"
---

# 코드베이스 소유권 맵

Hermes는 대규모 저장소이며, 대부분의 기여는 정확히 하나의 하위 시스템을 대상으로 합니다. 이 페이지는 각 하위 시스템을 소스 디렉터리 및 변경 전에 읽어야 할 문서 진입점과 연결해 보여 줍니다. 적절한 시작 문서, 변경 위치, 올바른 테스트 디렉터리를 찾는 데 사용하세요(테스트는 소스를 미러링합니다. `tools/`의 코드는 `tests/tools/`에서 테스트하고, 플러그인은 `tests/plugins/<type>/`에서 테스트하는 식입니다).

| 하위 시스템 | 소스 디렉터리 | 문서 진입점 |
|-----------|-------------------|------------------|
| 에이전트 코어(루프, 전송, 압축) | `agent/`, `run_agent.py` | [에이전트 루프](agent-loop.md), [컨텍스트 압축 및 캐싱](context-compression-and-caching.md) |
| 프롬프트 조립 | `agent/prompt_builder.py`, `agent/system_prompt.py` | [프롬프트 조립](prompt-assembly.md) |
| 모델 공급자 및 전송 | `agent/transports/`, `plugins/model-providers/`, `hermes_cli/models.py` | [공급자 추가](adding-providers.md), [모델 공급자 플러그인](model-provider-plugin.md), [공급자 런타임](provider-runtime.md) |
| 기본 제공 도구 | `tools/` | [도구 추가](adding-tools.md), [도구 런타임](tools-runtime.md) |
| 메시징 게이트웨이 | `gateway/`, `plugins/platforms/` | [게이트웨이 내부 구조](gateway-internals.md), [플랫폼 어댑터 추가](adding-platform-adapters.md) |
| CLI | `hermes_cli/` | [CLI 확장](extending-the-cli.md) |
| 플러그인 시스템 | `plugins/` | [Hermes 플러그인 빌드](plugins/index.md) |
| 스킬(번들 및 선택 사항) | `skills/`, `optional-skills/` | [스킬 만들기](creating-skills.md) |
| Cron / 예약 작업 | `cron/` | [Cron 내부 구조](cron-internals.md) |
| 세션 저장소 | `hermes_state.py` | [세션 저장소](session-storage.md) |
| 브라우저 스택 | `tools/browser_tool.py`, `tools/browser_supervisor.py`, `tools/browser_cdp_tool.py` | [브라우저 Supervisor](browser-supervisor.md) |
| 송신 방화벽 | `agent/proxy_sources/iron_proxy.py` | [송신 내부 구조](egress-internals.md) |
| ACP(IDE 통합) | `acp_adapter/` | [ACP 내부 구조](acp-internals.md) |
| 데스크톱 앱 | `apps/desktop/` | [데스크톱 플러그인 SDK](desktop-plugin-sdk.md), [Worktree UI 개발](worktree-ui-dev.md) |
| TUI | `ui-tui/`, `tui_gateway/` | [Worktree UI 개발](worktree-ui-dev.md) |
| 문서 사이트 | `website/` | [기여하기](contributing.md) |
| 테스트 | `tests/`, `tests-js/` | [기여하기 → 제출 전 확인](contributing.md#before-submitting) |

이 맵에서 몇 가지 규칙을 도출할 수 있습니다.

- **변경 사항은 해당 하위 시스템 안에 두세요.** 핵심 파일을 수정해야 하는 플러그인은 설계상의 문제 신호입니다. 대신 일반적인 플러그인 표면을 확장하세요(저장소의 `AGENTS.md`에 있는 기여 지침 참조).
- **수정하는 모든 소스 디렉터리에 대응하는 미러 테스트 디렉터리를 실행하세요.** `plugins/platforms/telegram/`을 변경했다면 우연히 생각해 낸 테스트 파일 하나만이 아니라 `tests/plugins/platforms/`가 통과해야 합니다.
- **두 하위 시스템이 관련되면 더 좁은 범위의 하위 시스템이 변경을 소유합니다.** 에이전트 코어의 분기문보다 어댑터나 플러그인에 수정 사항을 두세요. 코어는 좁은 허리(narrow waist)이며, 코어에 추가되는 모든 항목은 모든 API 호출에서 비용이 발생합니다.
