---
sidebar_label: "플러그인 빌드"
slug: /developer-guide/plugins
title: "Hermes 플러그인 빌드"
description: "도구, 훅, 데이터 파일, 스킬을 포함한 완전한 Hermes 플러그인을 빌드하는 단계별 가이드"
---

# Hermes 플러그인 빌드

이 가이드는 처음부터 완전한 Hermes 플러그인을 빌드하는 과정을 안내합니다. 이 과정을 마치면 여러 도구, 수명 주기 훅, 함께 제공되는 데이터 파일, 번들된 스킬을 갖춘 작동하는 플러그인을 만들게 됩니다. 즉, 플러그인 시스템이 지원하는 모든 기능을 사용할 수 있습니다.

:::info 어떤 가이드가 필요한지 잘 모르겠나요?
Hermes에는 서로 다른 플러그인 인터페이스가 여러 가지 있습니다. 일부는 Python `register_*` API를 사용하고, 다른 일부는 설정 기반이거나 디렉터리를 그대로 넣는 방식입니다. 먼저 이 표를 사용하세요.

| 추가하려는 항목 | 읽을 문서 |
|---|---|
| 사용자 지정 도구, 훅, 슬래시 명령, 스킬 또는 CLI 하위 명령 | **이 가이드** (일반 플러그인 인터페이스) |
| **네이티브 데스크톱 앱** 확장 (패널, 페이지, 상태 표시줄, 팔레트, 테마) | [Desktop Plugin SDK](/developer-guide/desktop-plugin-sdk) |
| **웹 대시보드** 확장 (탭, 셸 슬롯, 테마) | [대시보드 확장](/user-guide/features/extending-the-dashboard) |
| **LLM / 추론 백엔드** (새 프로바이더) | [모델 프로바이더 플러그인](/developer-guide/model-provider-plugin) |
| **게이트웨이 채널** (Discord/Telegram/IRC/Teams 등) | [플랫폼 어댑터 추가](/developer-guide/adding-platform-adapters) |
| **메모리 백엔드** (Honcho/Mem0/Supermemory 등) | [메모리 프로바이더 플러그인](/developer-guide/memory-provider-plugin) |
| **컨텍스트 압축 엔진** | [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin) |
| **이미지 생성 백엔드** | [이미지 생성 프로바이더 플러그인](/developer-guide/image-gen-provider-plugin) |
| **동영상 생성 백엔드** | [동영상 생성 프로바이더 플러그인](/developer-guide/video-gen-provider-plugin) |
| **웹 검색 / 추출 백엔드** | [웹 검색 프로바이더 플러그인](/developer-guide/web-search-provider-plugin) |
| **클라우드 브라우저 백엔드** (Browserbase 스타일 CDP 세션 프로바이더) | [브라우저 프로바이더 플러그인](/developer-guide/browser-provider-plugin) |
| **시크릿 관리자 백엔드** (vault / 비밀번호 관리자 / OS 키 저장소) | [시크릿 소스 플러그인](/developer-guide/secret-source-plugin) |
| **대시보드 OIDC/auth 프로바이더** | [웹 대시보드 — 사용자 지정 프로바이더](/user-guide/features/web-dashboard#custom-providers) — `ctx.register_dashboard_auth_provider()` |
| **TTS 백엔드** (Piper, VoxCPM, Kokoro, 음성 복제 등 모든 CLI) | [TTS 사용자 지정 명령 프로바이더](/user-guide/features/tts#custom-command-providers) — 설정 기반이며 Python이 필요하지 않음 |
| **STT 백엔드** (사용자 지정 whisper / ASR CLI) | [음성 메시지 전사](/user-guide/features/tts#voice-message-transcription-stt) — `HERMES_LOCAL_STT_COMMAND`를 argv 토큰화 템플릿으로 설정 |
| **MCP를 통한 외부 도구** (파일 시스템, GitHub, Linear, 모든 MCP 서버) | [MCP](/user-guide/features/mcp) — `config.yaml`에 `mcp_servers.<name>` 선언 |
| **게이트웨이 이벤트 훅** (시작, 세션 이벤트, 명령에서 실행) | [이벤트 훅](/user-guide/features/hooks#gateway-event-hooks) — `~/.hermes/hooks/<name>/`에 `HOOK.yaml` + `handler.py` 배치 |
| **셸 훅** (이벤트에서 셸 명령 실행) | [셸 훅](/user-guide/features/hooks#shell-hooks) — `config.yaml`의 `hooks:` 아래에 선언 |
| **추가 스킬 소스** (사용자 지정 GitHub 저장소, 비공개 스킬 인덱스) | [스킬](/user-guide/features/skills) — `hermes skills tap add <repo>` · [탭 게시](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| 일급 **코어** 추론 프로바이더 (플러그인이 아님) | [프로바이더 추가](/developer-guide/adding-providers) |

설정 기반(TTS, STT, MCP, 셸 훅) 및 디렉터리 투입 방식(게이트웨이 훅)을 포함해 모든 확장 지점을 한눈에 볼 수 있는 전체 [플러그인 가능 인터페이스 표](/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)도 참고하세요.
:::

:::caution 서드파티 제품 플러그인은 코어 트리가 아니라 독립형으로 제공됩니다
**다른 사람의 제품이나 프로젝트**를 통합하는 플러그인(관측성/메트릭 백엔드, 벤더 SaaS 커넥터, 분석 대시보드, 유료 서비스 연동)은 `NousResearch/hermes-agent`에 병합하지 않고 **독립적인 플러그인 저장소**로 빌드하고 배포합니다. 사용자는 `~/.hermes/plugins/` 또는 pip 엔트리 포인트를 통해 설치하며, 독립 저장소에서도 이 가이드의 모든 내용은 동일하게 작동합니다. 이는 품질 기준이 아니라 결합 및 유지 관리에 관한 결정입니다(코어는 빠르게 변하고 해당 백엔드는 우리가 소유하지 않기 때문입니다). 플러그인이 훌륭하더라도 자체 저장소에 속할 수 있습니다. Nous Research Discord의 `#plugins-skills-and-skins` 채널에서 홍보하세요. 정책은 [CONTRIBUTING.md](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)를 참고하세요.
:::

## Portable Agent Plugins v1 패키지

Hermes는 Agent Plugins v1.0.0 형식을 대상으로 하는 디렉터리 패키지도 설치하고 로드할 수 있습니다. 이는 Hermes가 이미 소유한 이식 가능한 구성 요소를 위한 호환성 어댑터입니다. 네이티브 `plugin.yaml`과 `register(ctx)` 플러그인을 대체하지는 않습니다.

```text
my-portable-plugin/
├── plugin.json
├── skills/
│   └── summarize/
│       ├── SKILL.md
│       └── references/
└── mcp.json
```

일반적인 워크플로를 통해 이식 가능한 패키지를 설치하고 활성화합니다.

```bash
hermes plugins install owner/repository --no-enable
hermes plugins list
hermes plugins enable <plugin-name>
```

이식 가능한 패키지는 명시적으로 활성화하지 않는 한 설치 후 비활성화됩니다. 활성화된 패키지는 즉시 사용할 수 있는 `skills/*/SKILL.md` 디렉터리와 루트 `mcp.json`의 stdio MCP 서버를 제공할 수 있습니다. 스킬은 읽기 전용이며 네임스페이스가 지정되고 `skills_list`와 `skill_view`를 통해 로드됩니다. MCP 명령은 셸을 통하지 않고, 별도의 인수 목록과 함께 하나의 실행 파일 토큰으로 전달됩니다. 전체 정규화된 스킬 이름을 확인하려면 `skills_list`를 사용하세요. 이식 가능한 스킬 네임스페이스는 검색된 플러그인 키에서 파생된 `agent-plugin-<slug>-<hash>` 형식을 결정론적으로 사용하므로 정제된 이름이 충돌할 수 없습니다.

Hermes는 `plugin.json`, Agent Skills frontmatter, 고정된 구성 요소 위치, `mcp.json`, 해석된 경로, 심볼릭 링크 포함 여부를 로컬에서 검증합니다. 패키지를 로드하는 동안 JSON 스키마를 가져오지는 않습니다. 유효한 형제 구성 요소를 계속 로드할 수 있다면 잘못된 스킬 또는 MCP 항목은 해당 경계에서 건너뜁니다. `PLUGIN_ROOT`는 해석된 패키지 루트를 가리킵니다. `PLUGIN_DATA`는 Hermes가 관리하는 프로필 범위의 쓰기 가능한 디렉터리를 가리킵니다. 이식 가능한 MCP의 `env`에 선언된 값은 패키지 데이터로, 시크릿 저장 메커니즘이 아닙니다. 자격 증명을 `mcp.json`에 넣지 마세요.

현재 이식 가능한 하위 집합은 stdio 및 Streamable HTTP MCP 항목을 지원합니다. 이식 가능한 `streamable-http` 항목은 Hermes의 기존 네이티브 원격 MCP 클라이언트(URL 기반 `mcp_servers` 설정을 구동하는 동일한 런타임)를 통해 라우팅되며, v1 경계 규칙이 적용됩니다. URL은 사용자 정보나 fragment가 없는 절대 http(s) URL이어야 하고, 일반 HTTP는 `localhost`/루프백 호스트에만 허용되며, 교차 출처 리디렉션을 거쳐서는 설정된 헤더를 절대 전달하지 않습니다. 레거시 `sse` 항목은 보고된 후 건너뜁니다. Agent Plugins v1은 신뢰, 권한, 출처 정보 또는 샌드박스를 정의하지 않습니다. 패키지를 활성화하면 해당 지침과 로컬 실행 파일에 다른 설치된 Hermes 플러그인과 동일한 완전 신뢰 상태가 부여됩니다.

[렌더링된 사양](https://agent-plugins.org/specification)은 현재 v1.0.0을 Working Draft로 표시하는 반면, [버전이 지정된 사양 저장소](https://github.com/agentplugins/agent-plugins-spec/blob/main/spec/1.0.0.md)는 이를 Published로 기록합니다. Hermes는 변경될 수 있는 어느 상태 레이블도 아닌 정식 v1.0.0 스키마 식별자와 규범적 텍스트를 기준으로 동작을 결정합니다. 이는 Agent Plugins 전체 적합성을 주장하는 것이 아니라, 명시적으로 지원되는 하위 집합입니다.

## 네이티브 플러그인 호환성 계약

네이티브 `plugin.yaml`과 `register(ctx)` 플러그인은 하나의 전역 플러그인 API 번호가 아니라 동작으로 보호됩니다. Hermes는 `PLUGIN_API_VERSION`을 노출하지 않고, manifest 전체에 `api:` 일치를 요구하지 않으며, 관련 없는 값에 API 버전을 연결하지 않습니다. 문서화된 동작을 사용하는 플러그인은 일반적인 Hermes 업그레이드 후에도 계속 작동해야 합니다.

호환성 규칙은 다음과 같습니다.

- **추가 방식으로 발전시키세요.** 문서화된 `PluginContext` 메서드는 제거하거나 이름을 바꾸지 않습니다. 새 매개변수는 선택 사항이고 기본값을 가지며 키워드 전용이어야 합니다. 기존 반환 필드는 제거하거나 조용히 타입을 바꾸지 않습니다.
- **훅 payload는 키워드 payload입니다.** 새 훅 데이터는 기존 필드의 의미나 위치를 바꾸지 않고 키워드 필드로만 추가합니다. Hermes는 콜백 시그니처를 검사합니다. 레거시 콜백은 자신이 선언한 필드를 받고, `**kwargs`가 있는 콜백은 현재의 전체 payload를 받습니다. 새 플러그인은 `**kwargs`를 허용해야 시그니처를 다시 바꾸지 않고 추가 데이터에 선택적으로 참여할 수 있습니다.
- **Manifest에는 추가 항목을 허용합니다.** 알 수 없는 `plugin.yaml` 필드는 무시됩니다. 따라서 플러그인 코드 자체가 지원되는 런타임 동작만 사용한다면, 이전 Hermes 릴리스도 최신 릴리스에서 도입된 메타데이터가 manifest에 포함된 플러그인을 로드할 수 있습니다.
- **프로바이더 인터페이스는 기본값을 통해 확장됩니다.** 새 프로바이더 메서드에는 기본 구현이 있습니다. 새 콜백 컨텍스트는 선택 사항이며, 시그니처 검사 결과 프로바이더가 이를 받는 경우에만 전달됩니다. 추상 메서드를 추가하거나 인수를 무조건 전달하려면 일괄 시그니처 변경이 아니라 마이그레이션 기간이 필요합니다.
- **경계를 넘는 계약에는 버전을 지정하세요.** capability가 wire payload 또는 지속 형식을 정의한다면 자체 스키마 버전을 가질 수 있습니다(예: observer payload 또는 secret-source state). 해당 로컬 스키마 안에서는 필드를 추가 방식으로 유지하세요. 지속된 플러그인 상태와 설정은 계속 읽을 수 있어야 하며, 그렇지 않다면 명시적인 마이그레이션을 제공해야 합니다. 이전 형식으로 작성된 재개 세션도 여전히 재생할 수 있어야 합니다. 관련 없는 콜백이나 컨텍스트 값에 버전 리터럴을 추가하지 마세요.

### 지원 중단 정책

문서화된 네이티브 플러그인 동작은 다음 조건을 모두 충족하는 경우에만 지원 중단할 수 있습니다.

1. 플러그인 가이드와 릴리스 노트에 대체 방법 및 마이그레이션 지침을 제공합니다.
2. 프로세스당 최대 한 번만 경고를 내보내며, 대체 방법과 가장 이른 제거 릴리스를 명시합니다.
3. 이후 최소 두 번의 마이너 릴리스 동안 기존 동작을 지원합니다.
4. 해당 기간 내내 레거시 경로와 대체 경로 모두에 대해 동작 기반 호환성 검사를 제공합니다.

기간이 끝난 후 제거할 때는 지속된 데이터 또는 재개 가능한 세션에 필요한 마이그레이션도 포함해야 합니다. 실제로는 제거보다 추가적인 별칭과 어댑터를 선호합니다.

Hermes는 격리된 `HERMES_HOME`에서 검색된 고정 외부 플러그인 fixture를 사용해 이 계약을 적용합니다. 이러한 테스트는 `PluginManager`를 통해 플러그인을 로드하고 호출하며, 내부 심볼 목록이나 소스 코드 형태가 아니라 실제 등록 및 콜백 결과를 검증합니다.

## 만들게 될 것

다음 두 도구를 갖춘 **계산기** 플러그인입니다.
- `calculate` — 수학 표현식 평가 (`2**16`, `sqrt(144)`, `pi * 5**2`)
- `unit_convert` — 단위 간 변환 (`100 F → 37.78 C`, `5 km → 3.11 mi`)

또한 모든 도구 호출을 기록하는 훅과 번들된 스킬 파일도 포함합니다.

## 1단계: 플러그인 디렉터리 만들기

디렉터리를 만들고 2단계를 계속 진행합니다.

```bash
mkdir -p ~/.hermes/plugins/calculator
cd ~/.hermes/plugins/calculator
```

### Plugin Doctor로 검증

`hermes plugins doctor [path-or-id]`는 Hermes 자체에서 사용하는 것과 동일한 디렉터리 검색, manifest 파서, 네임스페이스 import, `register(ctx)`, 훅 레지스트리 및 도구 레지스트리를 실행합니다. 잘못된 훅 이름, `**kwargs`를 허용하지 않는 콜백, 등록 실패, 선언된 도구/훅과 등록된 도구/훅 사이의 불일치를 보고합니다. 오류 발생 시 0이 아닌 값으로 종료하려면 `--ci`를 전달하세요.

```bash
hermes plugins doctor . --ci
```

Doctor는 임시 `HERMES_HOME`을 사용하고 검사 후 플러그인 등록 상태를 복원하며, 등록 중 우발적인 네트워크 접근을 포착하기 위해 직접적인 Python 소켓 연결을 차단합니다. 이는 샌드박스가 아닙니다. 플러그인 코드는 현재 사용자의 권한으로 여전히 프로세스 내부에서 실행되고 하위 프로세스를 생성할 수 있으므로, import할 만큼 신뢰할 수 있는 코드에 대해서만 Doctor를 실행하세요.
## 2단계: 매니페스트 작성

`plugin.yaml`을 생성합니다.

```yaml
name: calculator
version: 1.0.0
description: Math calculator — evaluate expressions and convert units
provides_tools:
  - calculate
  - unit_convert
provides_hooks:
  - post_tool_call
```

이렇게 하면 Hermes에 다음을 알립니다. “calculator라는 플러그인이며, 도구와 훅을 제공합니다.” `provides_tools`와 `provides_hooks` 필드는 플러그인이 등록하는 항목의 목록입니다.

추가할 수 있는 선택적 필드:
```yaml
author: Your Name
requires_env:          # gate loading on env vars; prompted during install
  - SOME_API_KEY       # simple format — plugin disabled if missing
  - name: OTHER_KEY    # rich format — shows description/url during install
    description: "Key for the Other service"
    url: "https://other.com/keys"
    secret: true
capabilities:          # privileged host surfaces you request (consent flow)
  - tools.override     # replace built-in tools (needs user consent)
  - llm.model_override # choose the model for host-owned LLM calls
```

### capability 선언

플러그인에 권한이 필요한 호스트 표면(기본 제공 도구 재정의, `ctx.llm` 호출에 사용할 모델 선택 등)이 있다면 `capabilities:`에 선언합니다. 설치 또는 활성화 시 사용자에게 목록이 표시되고 한 번 동의하면 됩니다. 이후 버전에서 capability가 추가되면 추가된 항목에 대해서만 업데이트 과정에서 다시 묻습니다. 선언되지 않았거나 동의하지 않은 capability는 단순히 꺼진 상태가 됩니다(fail closed). 따라서 **사용하기 전에 확인하고 graceful degradation을 적용해야 합니다**.

```python
def register(ctx):
    if ctx.has_capability("tools.override"):
        ctx.register_tool(..., override=True)
    else:
        ctx.register_tool(...)   # register under a non-conflicting name
```

알려진 capability id: `tools.override`, `llm.provider_override`,
`llm.model_override`, `llm.agent_id_override`, `llm.profile_override`,
`llm.task_override` (정식 레지스트리는 `hermes_cli/plugin_capabilities.py` 참조). 알 수 없는 id는 무시됩니다. 이전의 capability별 설정 키(`plugins.entries.<id>.allow_tool_override`, …)도 계속 작동하지만 deprecated 상태입니다. 사용자가 하나의 감사 가능한 동의 화면에서 처리할 수 있도록 capability를 선언하세요. Capability는 동의 및 감사 기능이며, **샌드박스가 아닙니다**. 호스트 API 표면만 제어합니다.

**Pip으로 배포되는 플러그인**은 설치 후 `plugin.yaml` 디렉터리가 존재하지 않으므로, 배포 메타데이터의 `hermes_agent.plugin_capabilities` entry-point 그룹을 통해 capability를 선언합니다. 각 선언의 이름은 `<plugin-id>.<capability-id>`이며, `hermes_agent.plugins` entry point와 동일한 객체를 가리킵니다.

```toml
[project.entry-points."hermes_agent.plugins"]
calculator = "my_pkg:register"

[project.entry-points."hermes_agent.plugin_capabilities"]
"calculator.tools.override" = "my_pkg:register"
```

Hermes는 코드를 import하지 않고 설치된 메타데이터에서 이를 읽습니다. 따라서 pip 설치에서도 `hermes plugins capabilities`와 동의 흐름이 정확하게 유지됩니다.

### 매니페스트 v2 레퍼런스

`plugin.yaml`은 추가적인 **v2 스키마**(#64165)도 지원합니다. 모든 필드는 선택 사항입니다. `manifest_version`이 없는 매니페스트는 v1 매니페스트로 간주되며 영구적으로 완전히 지원됩니다. 알 수 없는 필드는 로딩을 중단하지 않고 경고와 함께 무시됩니다(순방향 호환성). Hermes가 이해하는 버전보다 큰 `manifest_version`도 경고와 함께 로드됩니다.

| 필드 | 유형 | 의미 |
|---|---|---|
| `manifest_version` | int | 매니페스트 **파일 형식** 버전. 누락 시 `1`. 현재 최댓값: `2`. `api_version`과 독립적입니다. |
| `api_version` | int | 플러그인이 대상으로 하는 런타임 **플러그인 API 세대**(`ctx` 표면 / 훅 시그니처). `manifest_version`과 의도적으로 별개의 축입니다. `api_version: 1` 플러그인도 v2 매니페스트를 사용할 수 있습니다. |
| `requires_plugins` | list | 플러그인 간 의존성: 선택적 `version_range: ">=1.0,<2"`와 함께 `- id: other-plugin` 형식으로 지정합니다. **권고 사항**입니다. 의존성이 없으면 명확한 경고가 기록되지만 플러그인은 계속 로드됩니다. 런타임에 `ctx.has_plugin("other-plugin")`으로 확인하세요. 로드 **순서**는 이 간선을 따릅니다. A가 B를 필요로 하면 B의 `register()`가 A보다 먼저 실행됩니다(위상 정렬, 알파벳 순서로 동률 해소; 순환이 있으면 경고 후 알파벳 순서로 대체). |
| `python_dependencies` | str의 list | 선언된 pip 요구 사항(예: `"requests>=2.0,<3"`). **선언 경계만 제공합니다**. Hermes는 이를 검증하고 `hermes plugins install` / `hermes plugins doctor`에서 누락된 항목을 `pip install` 힌트와 함께 표시하지만, **절대 자동 설치하지 않습니다**. 상한을 고정하세요. |
| `config_schema` | mapping | `plugins.entries.<id>.settings` 아래 키에 대한 JSON Schema와 유사한 설명: `api_url: {type: str, default: "", description: "...", required: false}`. 로드 시 검증되며, 불일치하면 예상 유형과 키 이름을 포함한 실행 가능한 경고를 기록합니다. 로드 실패로 처리하지는 않습니다. 유형: `str`, `int`, `float`, `bool`, `list`, `dict`(JSON Schema 별칭도 포함). |
| `license` | str | SPDX 스타일 라이선스 id(예: `MIT`). |
| `homepage` | str | 프로젝트 URL. |
| `tags` | str의 list | 자유 형식의 검색 태그(예: `[gateway, telegram]`). |

```yaml
# plugin.yaml — manifest v2 example
name: my-plugin
version: 1.2.0
manifest_version: 2
api_version: 1
license: MIT
homepage: https://github.com/owner/my-plugin
tags: [gateway, demo]
requires_plugins:
  - id: other-plugin
    version_range: ">=1.0,<2"
python_dependencies:
  - "somepkg>=1.0,<2"     # surfaced, never auto-installed
config_schema:
  api_url: {type: str, default: "", description: "Service endpoint"}
```

:::note pip 종속성 격리는 추후 지원 예정
`python_dependencies`는 의도적으로 선언 후 표시만 수행합니다. Hermes의 공유 venv에 임의의 패키지를 설치하는 것은 충돌 및 공급망 공격 표면을 만들기 때문에, 설치 경계의 격리 설계(호스트 잠금 파일을 기준으로 한 constraints-file 설치, 플러그인별 vendored 디렉터리, 또는 충돌 감지 후 거부)는 명시적으로 후속 작업으로 미뤄졌습니다. [#64165](https://github.com/NousResearch/hermes-agent/issues/64165)의 2차 라운드 리뷰와 [#15220](https://github.com/NousResearch/hermes-agent/issues/15220)을 참조하세요. 플러그인 팩(#64166)은 이러한 v2 필드를 기반으로 합니다.
:::

## 3단계: 도구 스키마 작성

`schemas.py`를 생성합니다. LLM이 도구를 호출할 시점을 판단할 때 읽는 내용입니다.

```python
"""Tool schemas — what the LLM sees."""

CALCULATE = {
    "name": "calculate",
    "description": (
        "Evaluate a mathematical expression and return the result. "
        "Supports arithmetic (+, -, *, /, **), functions (sqrt, sin, cos, "
        "log, abs, round, floor, ceil), and constants (pi, e). "
        "Use this for any math the user asks about."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate (e.g., '2**10', 'sqrt(144)')",
            },
        },
        "required": ["expression"],
    },
}

UNIT_CONVERT = {
    "name": "unit_convert",
    "description": (
        "Convert a value between units. Supports length (m, km, mi, ft, in), "
        "weight (kg, lb, oz, g), temperature (C, F, K), data (B, KB, MB, GB, TB), "
        "and time (s, min, hr, day)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The numeric value to convert",
            },
            "from_unit": {
                "type": "string",
                "description": "Source unit (e.g., 'km', 'lb', 'F', 'GB')",
            },
            "to_unit": {
                "type": "string",
                "description": "Target unit (e.g., 'mi', 'kg', 'C', 'MB')",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    },
}
```

**스키마가 중요한 이유:** `description` 필드를 보고 LLM이 도구 사용 시점을 판단합니다. 도구가 무엇을 하고 언제 사용해야 하는지 구체적으로 작성하세요. `parameters`는 LLM이 전달하는 인수를 정의합니다.

## 4단계: 도구 핸들러 작성

`tools.py`를 생성합니다. LLM이 도구를 호출할 때 실제로 실행되는 코드입니다.

```python
"""Tool handlers — the code that runs when the LLM calls each tool."""

import json
import math

# Safe globals for expression evaluation — no file/network access
_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log2": math.log2, "log10": math.log10,
    "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "e": math.e,
    "factorial": math.factorial,
}


def calculate(args: dict, **kwargs) -> str:
    """Evaluate a math expression safely.

    Rules for handlers:
    1. Receive args (dict) — the parameters the LLM passed
    2. Do the work
    3. Return a JSON string — ALWAYS, even on error
    4. Accept **kwargs for forward compatibility
    """
    expression = args.get("expression", "").strip()
    if not expression:
        return json.dumps({"error": "No expression provided"})

    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        return json.dumps({"expression": expression, "result": result})
    except ZeroDivisionError:
        return json.dumps({"expression": expression, "error": "Division by zero"})
    except Exception as e:
        return json.dumps({"expression": expression, "error": f"Invalid: {e}"})


# Conversion tables — values are in base units
_LENGTH = {"m": 1, "km": 1000, "mi": 1609.34, "ft": 0.3048, "in": 0.0254, "cm": 0.01}
_WEIGHT = {"kg": 1, "g": 0.001, "lb": 0.453592, "oz": 0.0283495}
_DATA = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_TIME = {"s": 1, "ms": 0.001, "min": 60, "hr": 3600, "day": 86400}


def _convert_temp(value, from_u, to_u):
    # Normalize to Celsius
    c = {"F": (value - 32) * 5/9, "K": value - 273.15}.get(from_u, value)
    # Convert to target
    return {"F": c * 9/5 + 32, "K": c + 273.15}.get(to_u, c)


def unit_convert(args: dict, **kwargs) -> str:
    """Convert between units."""
    value = args.get("value")
    from_unit = args.get("from_unit", "").strip()
    to_unit = args.get("to_unit", "").strip()

    if value is None or not from_unit or not to_unit:
        return json.dumps({"error": "Need value, from_unit, and to_unit"})

    try:
        # Temperature
        if from_unit.upper() in {"C","F","K"} and to_unit.upper() in {"C","F","K"}:
            result = _convert_temp(float(value), from_unit.upper(), to_unit.upper())
            return json.dumps({"input": f"{value} {from_unit}", "result": round(result, 4),
                             "output": f"{round(result, 4)} {to_unit}"})

        # Ratio-based conversions
        for table in (_LENGTH, _WEIGHT, _DATA, _TIME):
            lc = {k.lower(): v for k, v in table.items()}
            if from_unit.lower() in lc and to_unit.lower() in lc:
                result = float(value) * lc[from_unit.lower()] / lc[to_unit.lower()]
                return json.dumps({"input": f"{value} {from_unit}",
                                 "result": round(result, 6),
                                 "output": f"{round(result, 6)} {to_unit}"})

        return json.dumps({"error": f"Cannot convert {from_unit} → {to_unit}"})
    except Exception as e:
        return json.dumps({"error": f"Conversion failed: {e}"})
```

**핸들러의 핵심 규칙:**
1. **시그니처:** `def my_handler(args: dict, **kwargs) -> str`
2. **반환값:** 항상 JSON 문자열이어야 합니다. 성공과 오류 모두 해당합니다.
3. **예외를 발생시키지 않기:** 모든 예외를 포착하고 대신 오류 JSON을 반환합니다.
4. **`**kwargs` 허용:** Hermes는 향후 추가 컨텍스트를 전달할 수 있습니다.

## 5단계: 등록 작성

`__init__.py`를 생성합니다. 이 파일이 스키마와 핸들러를 연결합니다.

```python
"""Calculator plugin — registration."""

import logging

from . import schemas, tools

logger = logging.getLogger(__name__)

# Track tool usage via hooks
_call_log = []

def _on_post_tool_call(tool_name, args, result, task_id, **kwargs):
    """Hook: runs after every tool call (not just ours)."""
    _call_log.append({"tool": tool_name, "session": task_id})
    if len(_call_log) > 100:
        _call_log.pop(0)
    logger.debug("Tool called: %s (session %s)", tool_name, task_id)


def register(ctx):
    """Wire schemas to handlers and register hooks."""
    ctx.register_tool(name="calculate",    toolset="calculator",
                      schema=schemas.CALCULATE,    handler=tools.calculate)
    ctx.register_tool(name="unit_convert", toolset="calculator",
                      schema=schemas.UNIT_CONVERT, handler=tools.unit_convert)

    # This hook fires for ALL tool calls, not just ours
    ctx.register_hook("post_tool_call", _on_post_tool_call)
```

**`register()`가 하는 일:**
- 시작 시 정확히 한 번 호출됩니다.
- `ctx.register_tool()`은 도구를 레지스트리에 넣으며, 모델이 즉시 도구를 볼 수 있습니다.
- `ctx.register_hook()`은 수명 주기 이벤트를 구독합니다.
- `ctx.register_cli_command()`는 CLI 하위 명령(예: `hermes my-plugin <subcommand>`)을 등록합니다.
- `ctx.register_command()`는 세션 내 슬래시 명령(예: CLI / gateway 채팅에서 `/myplugin <args>`)을 등록합니다. 자세한 내용은 아래 [슬래시 명령 등록](#register-slash-commands)을 참조하세요.
- `ctx.dispatch_tool(name, arguments)`는 부모 에이전트의 컨텍스트(승인, 자격 증명, task_id)를 자동으로 연결해 다른 도구(기본 제공 도구 또는 다른 플러그인의 도구)를 호출합니다. 모델이 직접 호출한 것처럼 `terminal`, `read_file` 또는 다른 도구를 실행해야 하는 슬래시 명령 핸들러에서 유용합니다.
- `ctx.get_config()` / `ctx.set_config()`는 이 플러그인의 설정 네임스페이스에만 접근합니다. `ctx.state`는 활성 프로필 아래에 플러그인이 소유한 런타임 데이터를 저장합니다.
- 이 함수가 충돌하면 플러그인은 비활성화되지만 Hermes는 정상적으로 계속 실행됩니다.

**`dispatch_tool` 예시 — 도구를 실행하는 슬래시 명령:**

```python
def handle_scan(ctx, raw_args: str):
    """Implement /scan by invoking the terminal tool through the registry."""
    result = ctx.dispatch_tool("terminal", {"command": f"find . -name '{raw_args}'"})
    return result  # returned to the caller's chat UI

def register(ctx):
    # Handlers receive a single raw_args string; close over ctx via a lambda.
    ctx.register_command(
        "scan",
        lambda raw: handle_scan(ctx, raw),
        description="Find files matching a glob",
    )
```

디스패치된 도구는 일반적인 승인, 비식별화, 예산 파이프라인을 거칩니다. 이는 해당 파이프라인을 우회하는 지름길이 아니라 실제 도구 호출입니다.
### 설정 및 런타임 상태 저장

사용자에게 표시되는 동작에는 플러그인 기준 설정 키를 사용하세요. Hermes는 이를 `plugins.entries.<plugin-id>.settings` 아래에서 해석하며 전역 경로, 플러그인 간 경로, 경로 순회 경로는 거부합니다:

```python
def register(ctx):
    endpoint = ctx.get_config("endpoint", default="https://example.invalid")
    retries = ctx.get_config("retry.attempts", default=3)

    ctx.set_config("endpoint", endpoint)
    ctx.set_config("retry.attempts", retries)
```

런타임 장부를 `config.yaml`에 저장하는 대신, 플러그인이 소유하는 커서, 캐시, 중복 제거 데이터에는 `ctx.state`를 사용하세요:

```python
def register(ctx):
    cursor = ctx.state.get("cursor", default={"page": 0})
    ctx.state.set("cursor", {"page": cursor["page"] + 1})
```

상태는 프로필 범위로 관리되고, 원자적으로 교체되며, 동시 작성에 안전하고 플러그인당 10MiB로 제한됩니다. 이식 가능한 패키지는 `PLUGIN_DATA`와 같은 디렉터리를 공유하고, 네이티브 플러그인은 충돌 가능성이 낮고 Windows에서 안전한 네임스페이스를 받습니다. 기존 상태가 손상된 경우에는 이를 알리고 보존합니다.

설정과 상태의 소유자는 서로 다릅니다. 설정은 `config.yaml`에 저장되는 사용자가 볼 수 있는 동작이고, 상태는 `<HERMES_HOME>/plugin-data/` 아래에 저장되는 플러그인 소유 런타임 데이터입니다. 어느 API도 다른 플러그인의 네임스페이스를 노출하지 않습니다.

## 6단계: 테스트

Hermes를 시작합니다:

```bash
hermes
```

배너의 도구 목록에 `calculator: calculate, unit_convert`가 표시되어야 합니다.

다음 프롬프트를 시도해 보세요:
```
What's 2 to the power of 16?
Convert 100 fahrenheit to celsius
What's the square root of 2 times pi?
How many gigabytes is 1.5 terabytes?
```

플러그인 상태를 확인합니다:
```
/plugins
```

출력:
```
Plugins (1):
  ✓ calculator v1.0.0 (2 tools, 1 hooks)
```

### 플러그인 검색 디버깅

플러그인이 나타나지 않거나 나타나지만 로드되지 않는 경우, `HERMES_PLUGINS_DEBUG=1`을 설정하면 stderr에 자세한 검색 로그가 출력됩니다:

```bash
HERMES_PLUGINS_DEBUG=1 hermes plugins list
```

각 플러그인 소스(번들, 사용자, 프로젝트, 진입점)에 대해 다음 내용을 확인할 수 있습니다.

- 검색한 디렉터리와 각 디렉터리에서 찾은 매니페스트 수
- 매니페스트별: 해석된 키, 이름, 종류, 소스, 디스크상의 경로
- 건너뛴 이유: `disabled via config`, `not enabled in config`, `exclusive plugin`, `no plugin.yaml, depth cap reached`
- 로드 시: 가져오는 플러그인과 `register(ctx)`가 등록한 항목(도구, 훅, 슬래시 명령, CLI 명령)의 한 줄 요약
- 파싱 실패 시: 예외의 전체 트레이스백(YAML 스캐너 오류 등)
- `register()` 실패 시: 예외를 발생시킨 `__init__.py`의 해당 줄을 가리키는 전체 트레이스백

동일한 로그는 항상 `~/.hermes/logs/agent.log`에도 기록됩니다. 환경 변수가 설정된 경우 WARNING 수준에는 실패만, DEBUG 수준에는 모든 내용이 기록됩니다. 환경 변수를 설정한 상태로 실행할 수 없다면(예: 게이트웨이 내부에서 실행하는 경우) 대신 로그 파일을 추적하세요:

```bash
hermes logs --level WARNING | grep -i plugin
```

플러그인이 나타나지 않는 일반적인 이유는 다음과 같습니다.

- **설정에서 활성화되지 않음** — 플러그인은 명시적으로 활성화해야 합니다. `hermes plugins enable <name>`을 실행하세요(`<name>`은 `plugins list` 출력에 표시되는 이름이며, 중첩된 레이아웃에서는 `<category>/<plugin>` 형식일 수 있습니다).
- **디렉터리 레이아웃이 잘못됨:** 네이티브 패키지는 `~/.hermes/plugins/<plugin-name>/plugin.yaml`(평면 구조) 또는 한 단계의 카테고리를 사용합니다. 이식 가능한 패키지는 같은 위치에 루트 `plugin.json`을 사용합니다. 그보다 깊은 위치는 무시됩니다.
- **`__init__.py`가 없음:** 네이티브 패키지에는 `plugin.yaml`과 `register(ctx)` 함수가 있는 `__init__.py`가 모두 필요합니다. 이식 가능한 패키지는 Python을 가져오지 않으므로 `__init__.py`가 필요하지 않습니다.
- **`kind`가 잘못됨** — 게이트웨이 어댑터는 매니페스트에 `kind: platform`이 필요합니다. 메모리 제공자는 `kind: exclusive`로 자동 감지되며 `plugins.enabled` 대신 `memory.provider` 설정을 통해 라우팅됩니다.

## 플러그인의 최종 구조

```
~/.hermes/plugins/calculator/
├── plugin.yaml      # "I'm calculator, I provide tools and hooks"
├── __init__.py      # Wiring: schemas → handlers, register hooks
├── schemas.py       # What the LLM reads (descriptions + parameter specs)
└── tools.py         # What runs (calculate, unit_convert functions)
```

네 개의 파일로 명확하게 분리합니다:
- **매니페스트**는 플러그인의 정체와 제공 항목을 선언합니다.
- **스키마**는 LLM용 도구를 설명합니다.
- **핸들러**는 실제 로직을 구현합니다.
- **등록**은 모든 요소를 연결합니다.

## 플러그인으로 할 수 있는 다른 일

### 데이터 파일 제공

플러그인 디렉터리에 원하는 파일을 넣고 가져오기 시점에 읽을 수 있습니다:

```python
# In tools.py or __init__.py
from pathlib import Path

_PLUGIN_DIR = Path(__file__).parent
_DATA_FILE = _PLUGIN_DIR / "data" / "languages.yaml"

with open(_DATA_FILE) as f:
    _DATA = yaml.safe_load(f)
```

### 스킬 번들링

플러그인은 에이전트가 `skill_view("plugin:skill")`을 통해 로드하는 스킬 파일을 제공할 수 있습니다. `__init__.py`에서 등록하세요:

```
~/.hermes/plugins/my-plugin/
├── __init__.py
├── plugin.yaml
└── skills/
    ├── my-workflow/
    │   └── SKILL.md
    └── my-checklist/
        └── SKILL.md
```

```python
from pathlib import Path

def register(ctx):
    skills_dir = Path(__file__).parent / "skills"
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            ctx.register_skill(child.name, skill_md)
```

이제 에이전트는 네임스페이스가 붙은 이름으로 스킬을 로드할 수 있습니다:

```python
skill_view("my-plugin:my-workflow")   # → plugin's version
skill_view("my-workflow")              # → built-in version (unchanged)
```

**주요 속성:**
- 플러그인 스킬은 **읽기 전용**입니다. `~/.hermes/skills/`에 들어가지 않으며 `skill_manage`를 통해 편집할 수 없습니다.
- 플러그인 스킬은 시스템 프롬프트의 `<available_skills>` 색인에 **나열되지 않습니다**. 명시적으로 로드해야 합니다.
- 일반 스킬 이름은 영향을 받지 않습니다. 네임스페이스가 내장 스킬과의 충돌을 방지합니다.
- 에이전트가 플러그인 스킬을 로드하면 동일한 플러그인에 속한 형제 스킬을 나열하는 번들 컨텍스트 배너가 앞에 추가됩니다.

:::tip 레거시 패턴
스킬을 `~/.hermes/skills/`에 복사하는 기존 `shutil.copy2` 패턴도 여전히 작동하지만, 내장 스킬과 이름이 충돌할 위험이 있습니다. 새 플러그인에는 `ctx.register_skill()`을 사용하세요.
:::

### 환경 변수에 따른 활성화

플러그인에 API 키가 필요한 경우:

```yaml
# plugin.yaml — simple format (backwards-compatible)
requires_env:
  - WEATHER_API_KEY
```

`WEATHER_API_KEY`가 설정되지 않으면 플러그인은 명확한 메시지와 함께 비활성화됩니다. 충돌도, 에이전트 오류도 발생하지 않으며 단지 "Plugin weather disabled (missing: WEATHER_API_KEY)"가 표시됩니다.

사용자가 `hermes plugins install`을 실행하면 누락된 `requires_env` 변수에 대해 **대화형으로 입력을 요청받습니다**. 값은 자동으로 `.env`에 저장됩니다.

더 나은 설치 환경을 위해 설명과 가입 URL이 포함된 확장 형식을 사용하세요:

```yaml
# plugin.yaml — rich format
requires_env:
  - name: WEATHER_API_KEY
    description: "API key for OpenWeather"
    url: "https://openweathermap.org/api"
    secret: true
```

| 필드 | 필수 | 설명 |
|-------|----------|-------------|
| `name` | 예 | 환경 변수 이름 |
| `description` | 아니요 | 설치 프롬프트에서 사용자에게 표시됨 |
| `url` | 아니요 | 자격 증명을 얻을 수 있는 곳 |
| `secret` | 아니요 | `true`이면 입력을 숨김(비밀번호 필드와 같음) |

두 형식은 같은 목록에서 섞어 사용할 수 있습니다. 이미 설정된 변수는 조용히 건너뜁니다.

### 선택적 Python 의존성 지연 설치

모든 사용자가 설치해 두지는 않을 SDK(벤더 SDK, 대형 ML 라이브러리, 플랫폼별 패키지 등)를 플러그인이 감싸는 경우 모듈 최상위에서 이를 `import`하지 마세요. 도구 핸들러 내부에서 `tools.lazy_deps.ensure(...)` 헬퍼를 사용하면 Hermes가 처음 사용할 때 패키지를 설치하며, 이때 사용자의 `security.allow_lazy_installs` 설정으로 허용 여부를 확인합니다.

```python
# tools.py
from tools.lazy_deps import ensure, FeatureUnavailable

def my_tool_handler(args, **kwargs):
    try:
        ensure("my-plugin.my-backend")   # key must be in LAZY_DEPS
    except FeatureUnavailable as exc:
        return {"error": str(exc)}

    import my_backend_sdk   # safe now
    ...
```

`tools/lazy_deps.py`의 보안 모델에는 다음 두 규칙이 있습니다.

| 규칙 | 이유 |
|---|---|
| 기능 키가 트리 내부의 `LAZY_DEPS` 허용 목록에 있어야 함 | 악성 설정이 Hermes를 유도해 임의의 패키지를 설치하지 못하도록 합니다. Hermes 자체가 제공하는 사양만 대상이 될 수 있습니다. |
| 사양은 PyPI 이름만 사용함 | `--index-url`, `git+https://`, `file:` 경로는 사용할 수 없습니다. 허용 목록 항목 내부에서 PEP 440(`"my-sdk>=1.2,<2"`)으로 버전을 고정하세요. |

pip로 배포되는 서드파티 플러그인은 자체 `pyproject.toml`의 `[project.optional-dependencies]` 추가 기능으로 선택적 의존성을 선언하고 사용자에게 `pip install your-plugin[backend]`를 실행하도록 안내하세요. 이 경로는 `lazy_deps`를 거치지 않습니다. 지연 설치 방식은 모든 설치 환경에 하드 의존성을 포함하면 기본 Hermes 용량이 커지는 **번들** 플러그인에 가장 유용합니다.

전역에서 `security.allow_lazy_installs: false`가 설정되면 `ensure()`는 즉시 복구 방법 안내와 함께 `FeatureUnavailable`을 발생시킵니다. 플러그인은 이를 잡아 우아하게 성능을 낮춰야 합니다(도구 루프를 충돌시키지 말고 오류 결과를 반환하세요).



### 스레드 안전 지연 싱글턴

플러그인은 첫 사용 시 모듈 수준 변수에 고비용 객체(SDK 클라이언트, HTTP 세션, 연결 풀 등)를 생성해 캐시하는 경우가 많습니다:

```python
_client = None

def get_client():
    global _client
    if _client is not None:
        return _client
    _client = ExpensiveClient(...)   # ← TOCTOU race
    return _client
```

이는 위험한 함정입니다. Hermes는 한 프로세스에서 여러 스레드(위임된 도구 호출, 백그라운드 워커, 자체 개선 fork)를 실행하므로, `_client`가 설정되기 전에 두 스레드가 `get_client()`에 진입할 수 있습니다. **둘 다** `is not None` 검사를 통과하고, **둘 다** 비용이 큰 객체 생성을 실행하며, 두 번째 쓰기가 첫 번째 결과를 덮어씁니다. 그 결과 먼저 생성했지만 사용되지 않은 쪽이 연 연결, 파일 핸들, 백그라운드 스레드 등의 리소스가 유출됩니다.

락을 직접 작성하지 마세요. `plugins/plugin_utils.py`의 헬퍼를 사용하세요:

```python
from plugins.plugin_utils import lazy_singleton, SingletonSlot

# Zero-arg accessor → decorate it:
@lazy_singleton
def get_client():
    return ExpensiveClient(load_config())   # runs exactly once

client = get_client()    # safe across threads
get_client.reset()       # drop the instance (tests / teardown)


# Accessor that takes a build argument → use a slot:
_slot: SingletonSlot = SingletonSlot()

def get_client(config=None):
    return _slot.get(lambda: ExpensiveClient(resolve(config)))

def reset_client():
    _slot.reset()
```

두 구현 모두 이중 확인 잠금을 사용해 동시에 발생하는 최초 호출을 직렬화하고 팩토리를 최대 한 번만 실행합니다. 팩토리에서 예외가 발생하면 아무것도 캐시되지 않으며 다음 호출에서 다시 시도합니다. `plugins/memory/honcho/client.py`의 honcho 메모리 플러그인이 참조 구현입니다.

> 경험칙: `global _something` 다음에 `is None` 검사와 객체 생성을 작성할 때마다 이 헬퍼 중 하나를 사용하세요.



### 조건부 도구 사용 가능 여부

선택적 라이브러리에 의존하는 도구에는 다음과 같이 작성합니다:

```python
ctx.register_tool(
    name="my_tool",
    schema={...},
    handler=my_handler,
    check_fn=lambda: _has_optional_lib(),  # False = tool hidden from model
)
```

### 내장 도구 재정의

내장 도구를 자체 구현으로 교체하려면(예: 기본 브라우저 도구를 headed-Chrome CDP 백엔드로 바꾸거나 `web_search`를 사용자 지정 사내 색인으로 교체하려면) `override=True`를 전달하세요:

```python
def register(ctx):
    ctx.register_tool(
        name="browser_navigate",             # same name as the built-in
        toolset="plugin_my_browser",         # your own toolset namespace
        schema={...},
        handler=my_custom_navigate,
        override=True,                       # explicit opt-in
    )
```

`override=True`가 없으면 레지스트리는 다른 도구 세트의 기존 도구를 가리는 등록을 거부합니다. 이를 통해 실수로 덮어쓰는 일을 방지합니다. **내장** 도구를 재정의하려면 추가로 운영자가 `config.yaml`에서 `plugins.entries.<plugin_id>.allow_tool_override: true`를 설정해 명시적으로 동의해야 합니다. 이 게이트가 없으면 `register_tool(override=True)`가 `PluginToolOverrideError`를 발생시킵니다. 재정의 내역은 감사할 수 있도록 `~/.hermes/logs/agent.log`에 기록됩니다. 플러그인은 내장 도구보다 나중에 로드되므로 등록 순서가 올바릅니다. 즉, 플러그인의 핸들러가 내장 핸들러를 대체합니다.

**번들에 포함되지 않은 플러그인에도 운영자 승인이 필요합니다.** Hermes core와 함께 제공되지 않는 모든 플러그인(사용자, 프로젝트 또는 pip 소스)이 기존 내장 도구에 `override=True`를 적용하려면 `config.yaml`에서 플러그인별로 추가 동의를 받아야 합니다:

```yaml
plugins:
  entries:
    my-plugin:                    # the plugin's registry key from `hermes plugins list`
      allow_tool_override: true
```

이 승인이 없으면 `ctx.register_tool(..., override=True)`가 `PluginToolOverrideError`를 발생시킵니다. `register()` 예외는 로더가 포착하므로 플러그인은 비활성화되고 Hermes는 계속 실행됩니다. 이 게이트가 있는 이유는 활성화된 플러그인이 `shell_exec`나 `write_file` 같은 권한 있는 내장 도구를 조용히 교체해 모델이 해당 도구로 라우팅하는 모든 작업을 가로챌 수 있기 때문입니다. 번들 플러그인은 예외입니다. 해당 재정의는 메인테이너의 결정입니다. 설정을 로드할 수 없으면 게이트는 기본적으로 차단됩니다.

일반적으로 이 키를 직접 편집할 일은 없습니다. `hermes plugins enable <name>`은 번들에 포함되지 않은 플러그인을 활성화할 때 이 기능을 허용할지 묻고(기본값은 허용하지 않음), `--allow-tool-override` / `--no-allow-tool-override` 플래그를 사용하면 스크립트 설치 시 프롬프트를 건너뜁니다. 동일한 승인으로 `deregister()`도 제한됩니다. 승인이 없으면 플러그인은 자신이 소유하지 않은 도구를 제거할 수 없으며, 그렇지 않으면 재정의 검사를 우회하는 방법이 될 수 있습니다.
### 여러 훅 등록

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", before_any_tool)
    ctx.register_hook("post_tool_call", after_any_tool)
    ctx.register_hook("pre_llm_call", inject_memory)
    ctx.register_hook("on_session_start", on_new_session)
    ctx.register_hook("on_session_end", on_session_end)
```

### 훅 레퍼런스

각 훅은 전체 내용이 **[Event Hooks reference](/user-guide/features/hooks#plugin-hooks)**에 문서화되어 있습니다. 콜백 시그니처, 매개변수 표, 각 훅이 실행되는 정확한 시점과 예시는 해당 문서를 참조하세요. 아래는 요약입니다.

| 훅 | 실행 시점 | 콜백 시그니처 | 반환값 |
|------|-----------|----------------|---------|
| [`pre_tool_call`](/user-guide/features/hooks#pre_tool_call) | 모든 도구가 실행되기 전 | `tool_name: str, args: dict, task_id: str` | 선택적 지시: `{"action": "block", "message": ...}`는 호출을 거부하고, `{"action": "approve", "message": ...}`는 사람 승인 게이트로 전달합니다 |
| [`post_tool_call`](/user-guide/features/hooks#post_tool_call) | 모든 도구가 반환된 후 | `tool_name: str, args: dict, result: str, task_id: str, duration_ms: int` | 무시됨 |
| [`pre_llm_call`](/user-guide/features/hooks#pre_llm_call) | 턴마다 한 번, 도구 호출 루프 전에 | `session_id: str, user_message: str, conversation_history: list, is_first_turn: bool, model: str, platform: str` | [컨텍스트 주입](#pre_llm_call-context-injection) |
| [`post_llm_call`](/user-guide/features/hooks#post_llm_call) | 턴마다 한 번, 도구 호출 루프 후 (성공한 턴에 한함) | `session_id: str, user_message: str, assistant_response: str, conversation_history: list, model: str, platform: str` | 무시됨 |
| `pre_api_request` | 각 원시 제공자 API 요청 전 (모델이 도구를 호출하는 턴에는 여러 번) | `session_id: str, model: str, provider: str, base_url: str, api_mode: str, api_call_count: int, message_count: int, tool_count: int, approx_input_tokens: int, max_tokens: int, request: dict` | 무시됨 |
| `post_api_request` | 각 원시 제공자 API 요청 반환 후 | `pre_api_request` 필드와 `api_duration: float, finish_reason: str, response_model: str \| None, usage: dict, response: dict, assistant_content_chars: int, assistant_tool_call_count: int` | 무시됨 |
| `api_request_error` | 제공자 API 호출에서 예외가 발생했을 때 | 상관관계 필드와 `status_code: int \| None, retry_count: int \| None, max_retries: int \| None, retryable: bool \| None, reason: str \| None, error: dict, request: dict` | 무시됨 |
| [`on_session_start`](/user-guide/features/hooks#on_session_start) | 새 세션이 생성될 때 (첫 턴에 한함) | `session_id: str, model: str, platform: str` | 무시됨 |
| [`on_session_end`](/user-guide/features/hooks#on_session_end) | 모든 `run_conversation` 호출 및 CLI 종료 시 | `session_id: str, completed: bool, interrupted: bool, model: str, platform: str` | 무시됨 |
| [`on_session_finalize`](/user-guide/features/hooks#on_session_finalize) | CLI/게이트웨이가 활성 세션을 정리할 때 | `session_id: str \| None, platform: str` | 무시됨 |
| [`on_session_reset`](/user-guide/features/hooks#on_session_reset) | 게이트웨이가 새 세션 키(`/new`, `/reset`)로 교체할 때 | `session_id: str, platform: str` | 무시됨 |
| [`gateway_platform_event`](/user-guide/features/hooks#gateway_platform_event) | 인증된 플랫폼 네이티브 이벤트가 게이트웨이 경계에서 정규화될 때 (현재 Telegram 반응) | `platform: str, event_type: str, payload: dict` | 무시됨 |
| `kanban_task_claimed` | 칸반 작업을 할당받을 때 (작업자 생성 전 디스패처 프로세스) | `task_id: str, board: str \| None, assignee: str \| None, run_id: int \| None, profile_name: str` | 무시됨 |
| `kanban_task_completed` | 칸반 작업이 완료될 때 (작업자 프로세스) | `task_id, board, assignee, run_id, profile_name, summary: str \| None` | 무시됨 |
| `kanban_task_blocked` | 칸반 작업이 차단될 때 (작업자 프로세스) | `task_id, board, assignee, run_id, profile_name, reason: str \| None` | 무시됨 |

대부분의 훅은 fire-and-forget 관찰자이며 반환값은 무시됩니다. 예외는 `pre_llm_call`과 `pre_tool_call`입니다. 전자는 컨텍스트를 대화에 주입할 수 있고, 후자는 차단/승인 지시를 반환할 수 있습니다.

모든 콜백은 향후 호환성을 위해 `**kwargs`를 받아야 합니다. 훅 콜백에서 오류가 발생하면 기록한 뒤 건너뜁니다. 다른 훅과 에이전트는 정상적으로 계속 실행됩니다.

칸반 수명 주기 훅은 보드 DB 변경이 커밋된 **후** 실행되므로 콜백은 항상 영속화된 상태를 확인할 수 있으며 SQLite 쓰기 잠금을 점유하지 않습니다. 칸반 작업자는 별도의 `hermes -p <profile> chat -q` 하위 프로세스로 실행됩니다. 따라서 `kanban_task_claimed`는 **디스패처** 프로세스에서 실행되고 `kanban_task_completed` / `kanban_task_blocked`는 **작업자** 프로세스에서 실행됩니다. 모든 전환을 중앙에서 관찰하려면 디스패처에 훅을 연결하고, 작업별 세션 내 컨텍스트가 필요하면 작업자에 연결하세요.

**API 요청 훅**은 턴별 `pre_llm_call` / `post_llm_call` 쌍보다 한 단계 아래에서 원시 제공자 요청을 관찰합니다. 도구를 호출하는 한 턴은 여러 API 요청을 만들며, 이 훅은 각 요청 전후에 실행됩니다. 이 훅은 관찰 가능성 플러그인(트레이싱, 비용 계산, 지연 시간 대시보드)을 위한 것입니다. `request`와 `response` kwargs는 제공자 페이로드를 정제하고 크기를 제한한 JSON 뷰입니다(민감한 키는 삭제하고 긴 문자열은 잘라내며 SDK 객체는 정규화). `usage`는 일반 토큰 요약 딕셔너리입니다. 모든 페이로드에는 `turn_id`, `api_request_id`, `task_id`, `session_id`, `api_call_count`라는 상관관계 필드가 포함되므로 플러그인에서 요청, 도구 호출, 턴을 서로 연결할 수 있습니다. 제공자 호출에서 예외가 발생하면 `api_request_error`가 실행되며 `status_code`, `retry_count` / `max_retries`, `retryable`, `reason`, `type`과 `message`를 담은 `error` 딕셔너리가 추가됩니다.

### `pre_llm_call` 컨텍스트 주입

반환값이 의미를 갖는 유일한 훅입니다. `pre_llm_call` 콜백이 `"context"` 키가 있는 딕셔너리(또는 일반 문자열)를 반환하면 Hermes는 해당 텍스트를 **현재 턴의 사용자 메시지**에 주입합니다. 이는 메모리 플러그인, RAG 통합, 가드레일 및 모델에 추가 컨텍스트를 제공해야 하는 모든 플러그인이 사용하는 메커니즘입니다.

#### 반환 형식

```python
# Dict with context key
return {"context": "Recalled memories:\n- User prefers dark mode\n- Last project: hermes-agent"}

# Plain string (equivalent to the dict form above)
return "Recalled memories:\n- User prefers dark mode"

# Return None or don't return → no injection (observer-only)
return None
```

`"context"` 키가 있는 0이 아닌 반환값 또는 비어 있지 않은 일반 문자열은 모두 수집되어 현재 턴의 사용자 메시지에 추가됩니다.

#### 초과 컨텍스트 외부 저장

훅별 컨텍스트는 기본적으로 `10,000`자로 제한됩니다. 제한을 초과하는 내용은 `$HERMES_HOME/hook_outputs/<session_id>/<uuid>.txt`에 기록되고, 저장된 경로와 함께 앞부분/뒷부분 미리 보기로 대체됩니다. 모델은 실제로 필요할 때 `read_file` 또는 `terminal`을 통해 전체 내용을 읽을 수 있습니다. 이렇게 하면 플러그인 오류로 컨텍스트가 계속 이어지는 모든 턴의 프롬프트를 부풀리고 프롬프트 캐시 접두사를 초과하는 일을 막을 수 있습니다. `config.yaml`에서 조정할 수 있습니다.

```yaml
hooks:
  output_spill:
    enabled: true          # default: true
    max_chars: 10000       # default; set higher to opt out of spilling
    preview_head: 500      # chars shown at the top of the preview
    preview_tail: 500      # chars shown at the bottom of the preview
    # directory: null      # default: $HERMES_HOME/hook_outputs
```

#### 주입 작동 방식

주입된 컨텍스트는 시스템 프롬프트가 아니라 **사용자 메시지**에 추가됩니다. 이는 의도적인 설계입니다.

- **프롬프트 캐시 보존** — 시스템 프롬프트는 턴마다 동일하게 유지됩니다. Anthropic과 OpenRouter는 시스템 프롬프트 접두사를 캐시하므로 안정적으로 유지하면 여러 턴 대화에서 입력 토큰 비용을 75% 이상 절약할 수 있습니다. 플러그인이 시스템 프롬프트를 수정하면 모든 턴에서 캐시 미스가 발생합니다.
- **임시성** — 주입은 API 호출 시점에만 발생합니다. 대화 기록의 원래 사용자 메시지는 변경되지 않으며 세션 데이터베이스에도 저장되지 않습니다.
- **시스템 프롬프트는 Hermes의 영역** — 모델별 지침, 도구 적용 규칙, 성격 지침, 캐시된 스킬 콘텐츠가 들어 있습니다. 플러그인은 에이전트의 핵심 지침을 변경하지 않고 사용자 입력과 나란히 컨텍스트를 제공합니다.

#### 예시: 메모리 조회 플러그인

```python
"""Memory plugin — recalls relevant context from a vector store."""

import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall_context(session_id, user_message, is_first_turn, **kwargs):
    """Called before each LLM turn. Returns recalled memories."""
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None  # nothing to inject

        text = "Recalled context from previous sessions:\n"
        text += "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None  # fail silently, don't break the agent

def register(ctx):
    ctx.register_hook("pre_llm_call", recall_context)
```

#### 예시: 가드레일 플러그인

```python
"""Guardrails plugin — enforces content policies."""

POLICY = """You MUST follow these content policies for this session:
- Never generate code that accesses the filesystem outside the working directory
- Always warn before executing destructive operations
- Refuse requests involving personal data extraction"""

def inject_guardrails(**kwargs):
    """Injects policy text into every turn."""
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", inject_guardrails)
```

#### 예시: 관찰자 전용 훅 (주입하지 않음)

```python
"""Analytics plugin — tracks turn metadata without injecting context."""

import logging
logger = logging.getLogger(__name__)

def log_turn(session_id, user_message, model, is_first_turn, **kwargs):
    """Fires before each LLM call. Returns None — no context injected."""
    logger.info("Turn: session=%s model=%s first=%s msg_len=%d",
                session_id, model, is_first_turn, len(user_message or ""))
    # No return → no injection

def register(ctx):
    ctx.register_hook("pre_llm_call", log_turn)
```

#### 여러 플러그인이 컨텍스트를 반환하는 경우

여러 플러그인이 `pre_llm_call`에서 컨텍스트를 반환하면 각 출력은 두 개의 줄바꿈으로 연결되어 사용자 메시지에 함께 추가됩니다. 순서는 플러그인 검색 순서(플러그인 디렉터리 이름의 알파벳순)를 따릅니다.

### 미들웨어: 동작 변경

훅은 에이전트 루프를 관찰합니다(문서에 설명된 몇 가지 조정 형식은 예외). **미들웨어는 동작을 변경합니다**. 요청 미들웨어는 다운스트림에서 처리되기 전에 유효 페이로드를 다시 작성하고, 실행 미들웨어는 실제 호출을 감쌉니다. 동일한 `register(ctx)` 진입점에서 등록합니다.

```python
def cap_find_output(tool_name, args, **kwargs):
    """Rewrite terminal find commands to cap their output."""
    command = args.get("command", "")
    if tool_name == "terminal" and command.startswith("find "):
        return {
            "args": {**args, "command": command + " | head -100"},
            "source": "my-plugin",
            "reason": "cap find output",
        }
    return None  # leave the call unchanged

def register(ctx):
    ctx.register_middleware("tool_request", cap_find_output)
```

허용되는 종류의 표준 목록은 `hermes_cli/middleware.py`의 `VALID_MIDDLEWARE`입니다.

| 종류 | 수신 항목 | 반환 계약 |
|------|----------|-----------|
| `tool_request` | `tool_name`, `args`, `original_args`, 컨텍스트 kwargs | `{"args": {...}}`를 반환하면 훅, 가드레일, 승인 및 실행에서 유효 도구 인수로 사용할 값을 바꿉니다. 호출을 변경하지 않으려면 `None`을 반환합니다. |
| `llm_request` | `request`, `original_request`, 컨텍스트 kwargs | `{"request": {...}}`를 반환하면 Hermes가 전송하기 전에 유효 제공자 kwargs를 바꿉니다. |
| `tool_execution` | 페이로드와 `next_call` | 도구 실행을 감쌉니다. 다운스트림 체인을 실행하려면 `next_call(payload)`을 정확히 한 번 호출하고 결과를 반환합니다(또는 건너뛰어 단락시킵니다). |
| `llm_execution` | 페이로드와 `next_call` | 제공자 호출을 감싸는 동일한 형식입니다. |

**실제로 중요한 규칙:**

- 요청 미들웨어 체인에서는 각 콜백이 이전 콜백이 다시 작성한 페이로드를 받지만 `original_args` / `original_request`에는 항상 미들웨어 적용 전 복사본이 들어 있습니다. 콜백 사이에서 페이로드가 복사되므로 자유롭게 수정할 수 있습니다.
- 반환 딕셔너리에 `source`, `reason`, `name` 문자열을 포함할 수 있습니다. 이 값은 미들웨어 추적에 기록되며 다운스트림 관찰자 훅은 `middleware_trace` kwarg로 받습니다.
- 실행 미들웨어의 `next_call`은 **한 번만 사용할 수 있습니다**. 두 번 호출하면 제공자나 도구가 다시 실행되므로 예외가 발생합니다.
- 미들웨어 콜백에서 예외가 발생하면 기록하고 건너뛰며 체인은 계속됩니다. `next_call` 이후 발생한 다운스트림 오류는 그대로 전파됩니다. 미들웨어는 기본 런타임 경로를 중단할 수 없습니다.
- 미들웨어 페이로드에는 관찰자 텔레메트리 필드와 함께 `middleware_schema_version`(`hermes.middleware.v1`)이 포함됩니다.
- 알 수 없는 종류는 실패시키지 않고 경고와 함께 등록됩니다. 따라서 최신 Hermes를 대상으로 작성된 플러그인도 구버전에서 계속 로드됩니다.
### CLI 명령 등록

플러그인은 자체 `hermes <plugin>` 하위 명령 트리를 추가할 수 있습니다:

```python
def _my_command(args):
    """Handler for hermes my-plugin <subcommand>."""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("All good!")
    elif sub == "config":
        print("Current config: ...")
    else:
        print("Usage: hermes my-plugin <status|config>")

def _setup_argparse(subparser):
    """Build the argparse tree for hermes my-plugin."""
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show plugin status")
    subs.add_parser("config", help="Show plugin config")
    subparser.set_defaults(func=_my_command)

def register(ctx):
    ctx.register_tool(...)
    ctx.register_cli_command(
        name="my-plugin",
        help="Manage my plugin",
        setup_fn=_setup_argparse,
        handler_fn=_my_command,
    )
```

등록이 완료되면 사용자는 `hermes my-plugin status`, `hermes my-plugin config` 등을 실행할 수 있습니다.

**메모리 제공자 플러그인**은 대신 규칙 기반 방식을 사용합니다. 플러그인의 `cli.py` 파일에 `register_cli(subparser)` 함수를 추가하면 됩니다. 메모리 플러그인 검색 시스템이 자동으로 찾아내므로 `ctx.register_cli_command()`를 호출할 필요가 없습니다. 자세한 내용은 [메모리 제공자 플러그인 가이드](/developer-guide/memory-provider-plugin#adding-cli-commands)를 참고하세요.

**활성 제공자 게이팅:** 메모리 플러그인의 CLI 명령은 해당 제공자가 설정의 활성 `memory.provider`일 때만 나타납니다. 사용자가 제공자를 설정하지 않았다면 해당 플러그인의 CLI 명령이 도움말 출력을 불필요하게 채우지 않습니다.

### 슬래시 명령 등록

플러그인은 세션 내 슬래시 명령, 즉 대화 중 사용자가 입력하는 명령(예: `/lcm status` 또는 `/ping`)을 등록할 수 있습니다. 이 명령은 CLI와 게이트웨이(Telegram, Discord 등)에서 모두 작동합니다.

```python
def _handle_status(raw_args: str) -> str:
    """Handler for /mystatus — called with everything after the command name."""
    if raw_args.strip() == "help":
        return "Usage: /mystatus [help|check]"
    return "Plugin status: all systems nominal"

def register(ctx):
    ctx.register_command(
        "mystatus",
        handler=_handle_status,
        description="Show plugin status",
    )
```

등록이 완료되면 모든 세션에서 `/mystatus`를 입력할 수 있습니다. 이 명령은 자동 완성, `/help` 출력, Telegram 봇 메뉴에 표시됩니다.

**시그니처:** `ctx.register_command(name: str, handler: Callable, description: str = "", args_hint: str = "")`

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `name` | `str` | 앞에 슬래시가 붙지 않은 명령 이름(예: `"lcm"`, `"mystatus"`) |
| `handler` | `Callable[[str], str \| None]` | 원시 인수 문자열을 전달받습니다. `async`일 수도 있습니다. |
| `description` | `str` | `/help`, 자동 완성, Telegram 봇 메뉴에 표시됩니다. |

**`register_cli_command()`와의 주요 차이:**

| | `register_command()` | `register_cli_command()` |
|---|---|---|
| 호출 방식 | 세션에서 `/name` | 터미널에서 `hermes name` |
| 작동 위치 | CLI 세션, Telegram, Discord 등 | 터미널만 |
| 핸들러가 받는 값 | 원시 args 문자열 | argparse `Namespace` |
| 사용 사례 | 진단, 상태 확인, 빠른 작업 | 복잡한 하위 명령 트리, 설정 마법사 |

**충돌 방지:** 플러그인이 기본 제공 명령(`help`, `model`, `new` 등)과 충돌하는 이름을 등록하려 하면 로그 경고와 함께 등록이 조용히 거부됩니다. 기본 제공 명령이 항상 우선합니다.

**비동기 핸들러:** 게이트웨이 디스패치는 비동기 핸들러를 자동으로 감지하고 기다리므로 동기 함수와 비동기 함수 중 어느 쪽이든 사용할 수 있습니다.

```python
async def _handle_check(raw_args: str) -> str:
    result = await some_async_operation()
    return f"Check result: {result}"

def register(ctx):
    ctx.register_command("check", handler=_handle_check, description="Run async check")
```

### 슬래시 명령에서 도구 디스패치

도구를 조율해야 하는 슬래시 명령 핸들러(예: `delegate_task`로 서브에이전트를 생성하거나 `file_edit`를 호출하는 경우)는 프레임워크 내부에 직접 접근하지 말고 `ctx.dispatch_tool()`을 사용해야 합니다. 상위 에이전트의 컨텍스트(워크스페이스 힌트, 스피너, 모델 상속)는 자동으로 연결됩니다.

```python
def register(ctx):
    def _handle_deliver(raw_args: str):
        result = ctx.dispatch_tool(
            "delegate_task",
            {
                "goal": raw_args,
                "toolsets": ["terminal", "file", "web"],
            },
        )
        return result

    ctx.register_command(
        "deliver",
        handler=_handle_deliver,
        description="Delegate a goal to a subagent",
    )
```

**시그니처:** `ctx.dispatch_tool(name: str, args: dict, *, parent_agent=None) -> str`

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `name` | `str` | 도구 레지스트리에 등록된 도구 이름(예: `"delegate_task"`, `"file_edit"`) |
| `args` | `dict` | 모델이 보낼 때와 같은 형태의 도구 인수 |
| `parent_agent` | `Agent \| None` | 선택적 재정의 값. 생략하면 현재 CLI 에이전트에서 확인합니다(또는 게이트웨이 모드에서 정상적으로 기능이 축소됩니다). |

**런타임 동작:**

- **CLI 모드:** `parent_agent`는 활성 CLI 에이전트에서 확인되므로 워크스페이스 힌트, 스피너, 모델 선택이 예상대로 상속됩니다.
- **게이트웨이 모드:** CLI 에이전트가 없으므로 도구가 정상적으로 기능을 축소합니다. 워크스페이스는 설정된 터미널 작업 디렉터리에서 읽고 스피너는 표시하지 않습니다.
- **명시적 재정의:** 호출자가 `parent_agent=`를 명시적으로 전달하면 해당 값이 그대로 사용되며 덮어쓰지 않습니다.

이는 플러그인 명령에서 도구를 디스패치하기 위한 공개된 안정 인터페이스입니다. 플러그인은 `ctx._cli_ref.agent` 또는 이와 유사한 비공개 상태에 접근해서는 안 됩니다.

### 훅 내부에서 동작 수행(프로필 + 도구)

`ctx._cli_ref`는 **대화형 CLI** 세션에서만 채워집니다. 게이트웨이, 비대화형 `hermes chat -q` 실행, **kanban이 생성한 워커 세션**에서는 `None`입니다. 따라서 `_cli_ref`를 거치는 플러그인 로직은 정확히 이런 컨텍스트에서 조용히 아무 동작도 하지 않습니다. 훅에 필요한 기능은 세션과 무관한 다음 두 가지 안정적인 API로 처리할 수 있습니다.

- **`ctx.profile_name`** — 활성 프로필 이름(예: `"default"`, 또는 kanban 워커의 담당자 프로필)입니다. `HERMES_HOME`에서 파생되므로 `_cli_ref` 의존성 없이 어디서나 작동합니다.
- **`ctx.dispatch_tool(name, args)`** — `kanban_*` 도구, `delegate_task`, `terminal`, `read_file` 등을 포함한 등록된 모든 도구(기본 제공 또는 플러그인)를 호출합니다. 훅 콜백이 실행되는 프로세스와 관계없이 작동합니다.

이 둘을 함께 사용하면 kanban 수명 주기 훅이 프레임워크 내부에 접근하지 않고 전환을 감지한 뒤 보드에서 작업을 수행할 수 있습니다.

```python
def register(ctx):
    def on_blocked(*, task_id, reason=None, **kw):
        # Runs in the worker process; ctx._cli_ref is None here.
        ctx.dispatch_tool("kanban_comment", {
            "task_id": task_id,
            "comment": f"[{ctx.profile_name}] auto-noted block: {reason}",
        })
    ctx.register_hook("kanban_task_blocked", on_blocked)
```

전체 `hermes <subcommand>`(예: `hermes kanban show`)를 실행하려면 `ctx.dispatch_tool("terminal", {"command": "hermes kanban show ..."})`를 통해 `terminal` 도구를 사용해 셸을 실행하세요. 헤드리스 워커 세션에는 프로세스 내 슬래시 명령 브리지가 없으며, 훅에서 Hermes를 구동하는 데 지원되는 방법은 도구를 사용하는 것입니다.

### Slack Block Kit 버튼 클릭 처리

대화형 요소(버튼, 오버플로 메뉴, 날짜 선택기 등)가 포함된 Block Kit 메시지를 게시하는 플러그인은 `slack_bolt.AsyncApp`을 몽키 패치하지 않고 Slack 어댑터에 직접 클릭 핸들러를 등록할 수 있습니다.

```python
def register(ctx):
    async def _on_approve(ack, body, action):
        # ack within 3 seconds — slack_bolt requirement.
        await ack()
        # body["channel"]["id"], body["user"]["id"], body["message"]["ts"]
        # action["action_id"], action["value"]
        sweep_id = (action.get("value") or "").split("|", 1)[-1]
        # ...do the deterministic work, then post a follow-up.

    ctx.register_slack_action_handler("inbox_sweep_approve", _on_approve)
```

**시그니처:** `ctx.register_slack_action_handler(action_id, callback) -> None`

| 매개변수 | 타입 | 설명 |
|-----------|------|-------------|
| `action_id` | `str \| re.Pattern \| dict` | `slack_bolt.App.action()`이 허용하는 값입니다. 리터럴 `action_id`, 여러 ID와 일치하는 컴파일된 정규식, 또는 `{"action_id": "...", "block_id": "..."}`와 같은 제약 조건 딕셔너리일 수 있습니다. |
| `callback` | `async callable` | `slack_bolt` 규약에 따라 `(ack, body, action)`을 받습니다. |

**런타임 동작:**

- 핸들러는 플러그인 로드 시점에 큐에 들어가고 Slack 플랫폼이 연결될 때 어댑터의 `slack_bolt.AsyncApp`에 연결됩니다.
- 각 콜백은 방어적으로 래핑됩니다. 핸들러에서 예외가 발생하면 게이트웨이가 오류를 기록하고 Slack이 재시도하지 않도록 최선의 방법으로 클릭을 확인합니다.
- 표준 `slack_bolt` 규칙이 적용됩니다. 3초 이내에 `await ack()`를 호출한 다음 시간이 오래 걸리는 작업을 수행하세요.
- 여러 워크스페이스를 배포한 경우 연결된 모든 워크스페이스의 클릭에 핸들러가 실행됩니다. 동작 범위를 지정해야 한다면 `body["team"]["id"]`를 사용하세요.

플러그인이 Slack 상호작용에 참여하는 공개 방법은 이것입니다. 기존 플러그인은 `SlackAdapter.connect`를 패치할 수 있지만, 이제는 이 API를 사용하는 것이 좋습니다.

:::tip
이 가이드는 **일반 플러그인**(도구, 훅, 슬래시 명령, CLI 명령)을 다룹니다. 아래 섹션에서는 각 특수 플러그인 유형의 작성 패턴을 개략적으로 설명하며, 각 섹션에서 필드 참조와 예제가 포함된 전체 가이드로 연결합니다.
:::

## 특수 플러그인 유형

Hermes에는 일반 표면 외에 다섯 가지 특수 플러그인 유형이 있습니다. 각 유형은 번들된 경우 `plugins/<category>/<name>/` 아래에, 사용자가 설치한 경우 `~/.hermes/plugins/<category>/<name>/` 아래에 디렉터리로 제공됩니다. 카테고리마다 계약이 다르므로 필요한 유형을 선택한 뒤 해당 전체 가이드를 읽으세요.

### 모델 제공자 플러그인 — LLM 백엔드 추가

`plugins/model-providers/<name>/`에 프로필을 추가합니다.

```python
# plugins/model-providers/acme/__init__.py
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="acme",
    aliases=("acme-inference",),
    display_name="Acme Inference",
    env_vars=("ACME_API_KEY", "ACME_BASE_URL"),
    base_url="https://api.acme.example.com/v1",
    auth_type="api_key",
    default_aux_model="acme-small-fast",
    fallback_models=("acme-large-v3", "acme-medium-v3"),
))
```

```yaml
# plugins/model-providers/acme/plugin.yaml
name: acme-provider
kind: model-provider
version: 1.0.0
description: Acme Inference — OpenAI-compatible direct API
```

`get_provider_profile()` 또는 `list_providers()`가 처음 호출될 때 지연 검색됩니다. `auth.py`, `config.py`, `doctor.py`, `models.py`, `runtime_provider.py`, 그리고 `chat_completions` 전송 계층이 자동으로 연결됩니다. 사용자 플러그인은 이름으로 번들 플러그인을 재정의합니다.

**전체 가이드:** [모델 제공자 플러그인](/developer-guide/model-provider-plugin) — 필드 참조, 재정의 가능한 훅(`prepare_messages`, `build_extra_body`, `build_api_kwargs_extras`, `fetch_models`), api_mode 선택, 인증 유형, 테스트.

### 플랫폼 플러그인 — 게이트웨이 채널 추가

`plugins/platforms/<name>/`에 어댑터를 추가합니다.

```python
# plugins/platforms/myplatform/adapter.py
from gateway.platforms.base import BasePlatformAdapter

class MyPlatformAdapter(BasePlatformAdapter):
    async def connect(self): ...
    async def send(self, chat_id, text): ...
    async def disconnect(self): ...

def check_requirements():
    import os
    return bool(os.environ.get("MYPLATFORM_TOKEN"))

def _env_enablement():
    import os
    tok = os.getenv("MYPLATFORM_TOKEN", "").strip()
    if not tok:
        return None
    return {"token": tok}

def register(ctx):
    ctx.register_platform(
        name="myplatform",
        label="MyPlatform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        required_env=["MYPLATFORM_TOKEN"],
        # Auto-populate PlatformConfig.extra from env so env-only setups
        # show up in `hermes gateway status` without SDK instantiation.
        env_enablement_fn=_env_enablement,
        # Opt in to cron delivery: `deliver=myplatform` routes to this var.
        cron_deliver_env_var="MYPLATFORM_HOME_CHANNEL",
        emoji="💬",
        platform_hint="You are chatting via MyPlatform. Keep responses concise.",
    )
```

```yaml
# plugins/platforms/myplatform/plugin.yaml
name: myplatform-platform
label: MyPlatform
kind: platform
version: 1.0.0
description: MyPlatform gateway adapter
requires_env:
  - name: MYPLATFORM_TOKEN
    description: "Bot token from the MyPlatform console"
    password: true
optional_env:
  - name: MYPLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery"
    password: false
```

**전체 가이드:** [플랫폼 어댑터 추가](/developer-guide/adding-platform-adapters) — 전체 `BasePlatformAdapter` 계약, 메시지 라우팅, 인증 게이팅, 설정 마법사 통합. 표준 라이브러리만 사용하는 작동 예제는 `plugins/platforms/irc/`를 참고하세요.

### 메모리 제공자 플러그인 — 세션 간 지식 백엔드 추가

`MemoryProvider` 구현을 `plugins/memory/<name>/`에 추가합니다.

```python
# plugins/memory/my-memory/__init__.py
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-memory"

    def is_available(self) -> bool:
        import os
        return bool(os.environ.get("MY_MEMORY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

    def sync_turn(self, user_content, assistant_content, *,
                  session_id="", messages=None) -> None:
        ...

    def prefetch(self, query, *, session_id="") -> str:
        ...

    def get_tool_schemas(self) -> list[dict]:
        return []   # required @abstractmethod — see full guide

def register(ctx):
    ctx.register_memory_provider(MyMemoryProvider())
```

메모리 제공자는 하나만 선택할 수 있으며, `config.yaml`의 `memory.provider`로 활성 제공자를 지정합니다.

**전체 가이드:** [메모리 제공자 플러그인](/developer-guide/memory-provider-plugin) — 전체 `MemoryProvider` ABC, 스레딩 계약, 프로필 격리, `cli.py`를 통한 CLI 명령 등록

### 컨텍스트 엔진 플러그인 — 컨텍스트 압축기 교체

```python
# plugins/context_engine/my-engine/__init__.py
from agent.context_engine import ContextEngine

class MyContextEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "my-engine"

    def update_from_response(self, usage) -> None: ...
    def should_compress(self, prompt_tokens: int = None) -> bool: ...
    def compress(self, messages, current_tokens=None, focus_topic=None,
                 force=False, memory_context="") -> list: ...

def register(ctx):
    ctx.register_context_engine(MyContextEngine())
```

컨텍스트 엔진은 하나만 선택할 수 있으며, `config.yaml`의 `context.engine`으로 지정합니다.

**전체 가이드:** [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin).

### 이미지 생성 백엔드

제공자를 `plugins/image_gen/<name>/`에 추가합니다.

```python
# plugins/image_gen/my-imggen/__init__.py
from agent.image_gen_provider import ImageGenProvider

class MyImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "my-imggen"

    def is_available(self) -> bool: ...
    def generate(self, prompt: str, aspect_ratio="landscape", **kwargs) -> dict:
        # returns success_response(...) / error_response(...)
        ...

def register(ctx):
    ctx.register_image_gen_provider(MyImageGenProvider())
```

```yaml
# plugins/image_gen/my-imggen/plugin.yaml
name: my-imggen
kind: backend
version: 1.0.0
description: Custom image generation backend
```

**전체 가이드:** [이미지 생성 제공자 플러그인](/developer-guide/image-gen-provider-plugin) — 전체 `ImageGenProvider` ABC, `list_models()` / `get_setup_schema()` 메타데이터, `success_response()`/`error_response()` 헬퍼, base64와 URL 출력, 사용자 재정의, pip 배포

**참고 예시:** `plugins/image_gen/openai/` (OpenAI SDK를 통한 DALL-E / GPT-Image), `plugins/image_gen/openai-codex/`, `plugins/image_gen/xai/` (Grok 이미지 생성)

## Python이 아닌 확장 표면

Hermes는 Python 플러그인이 아닌 확장도 지원합니다. 이러한 확장은 [플러그인 인터페이스 표](/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)에 나와 있으며, 아래에서는 각 작성 방식을 간략히 설명합니다.

### MCP 서버 — 외부 도구 등록

Model Context Protocol(MCP) 서버는 Python 플러그인 없이도 자체 도구를 Hermes에 등록합니다. `~/.hermes/config.yaml`에 선언합니다.

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    timeout: 120

  linear:
    url: "https://mcp.linear.app/sse"
    auth:
      type: "oauth"
```

Hermes는 시작할 때 각 서버에 연결하고, 서버의 도구를 나열한 다음 기본 제공 도구와 함께 등록합니다. LLM은 이를 다른 도구와 완전히 동일하게 인식합니다. **전체 가이드:** [MCP](/user-guide/features/mcp).

### 게이트웨이 이벤트 훅 — 수명 주기 이벤트에서 실행

매니페스트와 핸들러를 `~/.hermes/hooks/<name>/`에 추가합니다.

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Send a push notification when a long task finishes
events:
  - agent:end
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
async def handle(event_type: str, context: dict) -> None:
    if context.get("duration_seconds", 0) > 120:
        # send notification …
        pass
```

이벤트에는 `gateway:startup`, `session:start`, `session:end`, `session:reset`, `agent:start`, `agent:step`, `agent:end`, 와일드카드 `command:*`가 포함됩니다. 훅의 오류는 포착되어 로그에 기록되며, 메인 파이프라인을 차단하지 않습니다.

**전체 가이드:** [게이트웨이 이벤트 훅](/user-guide/features/hooks#gateway-event-hooks).

### 셸 훅 — 도구 호출 시 셸 명령 실행

도구가 실행될 때 스크립트(알림, 감사 로그, 데스크톱 알림, 자동 포매터)를 실행하려는 것뿐이라면 Python 없이 `config.yaml`에서 셸 훅을 사용합니다.

```yaml
hooks:
  - event: post_tool_call
    command: "notify-send 'Tool ran: {tool_name}'"
    when:
      tools: [terminal, patch, write_file]
```

Python 플러그인 훅과 동일한 모든 이벤트(`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`, `pre_gateway_dispatch`)와 `pre_tool_call` 차단 결정에 사용할 구조화된 JSON 출력을 지원합니다.

**전체 가이드:** [셸 훅](/user-guide/features/hooks#shell-hooks).

### 스킬 소스 — 사용자 지정 스킬 레지스트리 추가

스킬 GitHub 저장소를 관리하거나 기본 제공 소스 외의 커뮤니티 색인에서 가져오려면 **tap**으로 추가합니다.

```bash
hermes skills tap add myorg/skills-repo
hermes skills search my-workflow --source myorg/skills-repo
hermes skills install myorg/skills-repo/my-workflow
```

사용자 지정 tap을 게시하는 방법은 `skills/<skill-name>/SKILL.md` 디렉터리가 있는 GitHub 저장소를 만드는 것뿐이며, 서버나 레지스트리 가입은 필요하지 않습니다.

**전체 가이드:** [Skills Hub](/user-guide/features/skills#skills-hub) · [사용자 지정 tap 게시](/user-guide/features/skills#publishing-a-custom-skill-tap) (저장소 레이아웃, 최소 예시, 기본값이 아닌 경로, 신뢰 수준).

### 명령 템플릿을 통한 TTS / STT

오디오나 텍스트를 읽고 쓰는 CLI는 무엇이든 `config.yaml`을 통해 연결할 수 있으며, Python 코드는 필요하지 않습니다.

```yaml
tts:
  provider: voxcpm
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      voice_compatible: true
```

STT의 경우 `HERMES_LOCAL_STT_COMMAND`에 argv 토큰으로 분리된 템플릿을 지정합니다. 암시적인 셸 해석 없이 실행되므로, 신뢰할 수 있는 로컬 명령에 셸 문법이 필요하면 명시적으로 `sh -c`, `cmd /c` 또는 PowerShell로 감쌉니다. 지원되는 플레이스홀더는 `{input_path}`, `{output_path}`, `{format}`, `{voice}`, `{model}`, `{speed}`(TTS), `{input_path}`, `{output_dir}`, `{language}`, `{model}`(STT)입니다. 경로와 상호 작용하는 CLI는 자동으로 플러그인으로 취급됩니다.

**전체 가이드:** [사용자 지정 TTS 명령 제공자](/user-guide/features/tts#custom-command-providers) · [STT](/user-guide/features/tts#voice-message-transcription-stt).

## pip를 통한 배포

플러그인을 공개적으로 공유하려면 Python 패키지에 엔트리 포인트를 추가합니다.

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-plugin = "my_plugin_package"
```

```bash
pip install hermes-plugin-calculator
# Plugin auto-discovered on next hermes startup
```

## NixOS용 배포

:::warning Nix는 더 이상 명시적으로 지원되지 않음
Nix/NixOS는 더 이상 명시적으로 지원되는 설치 경로가 아니며(최선의 노력으로만 지원) — [Nix 설정](/getting-started/nix-setup)을 참조하세요. 이 절은 이미 NixOS에 배포한 사용자를 위해 유지됩니다.
:::

`pyproject.toml`에 엔트리 포인트를 제공하면 NixOS 사용자가 선언적으로 플러그인을 설치할 수 있습니다.

**엔트리 포인트 플러그인** (배포에 권장):
```nix
# User's configuration.nix
services.hermes-agent.extraPythonPackages = [
  (pkgs.python312Packages.buildPythonPackage {
    pname = "my-plugin";
    version = "1.0.0";
    src = pkgs.fetchFromGitHub {
      owner = "you";
      repo = "hermes-my-plugin";
      rev = "v1.0.0";
      hash = "sha256-...";  # nix-prefetch-url --unpack
    };
    format = "pyproject";
    build-system = [ pkgs.python312Packages.setuptools ];
  })
];
```

**디렉터리 플러그인** (`pyproject.toml` 불필요):
```nix
services.hermes-agent.extraPlugins = [
  (pkgs.fetchFromGitHub {
    owner = "you";
    repo = "hermes-my-plugin";
    rev = "v1.0.0";
    hash = "sha256-...";
  })
];
```

충돌 검사와 오버레이 사용을 비롯한 전체 문서는 [Nix 설정 가이드](/getting-started/nix-setup#plugins)를 참조하세요.

## 흔한 실수

**핸들러가 JSON 문자열을 반환하지 않음:**
```python
# Wrong — returns a dict
def handler(args, **kwargs):
    return {"result": 42}

# Right — returns a JSON string
def handler(args, **kwargs):
    return json.dumps({"result": 42})
```

**핸들러 시그니처에 `**kwargs`가 없음:**
```python
# Wrong — will break if Hermes passes extra context
def handler(args):
    ...

# Right
def handler(args, **kwargs):
    ...
```

**핸들러가 예외를 발생시킴:**
```python
# Wrong — exception propagates, tool call fails
def handler(args, **kwargs):
    result = 1 / int(args["value"])  # ZeroDivisionError!
    return json.dumps({"result": result})

# Right — catch and return error JSON
def handler(args, **kwargs):
    try:
        result = 1 / int(args.get("value", 0))
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

**스키마 설명이 너무 모호함:**
```python
# Bad — model doesn't know when to use it
"description": "Does stuff"

# Good — model knows exactly when and how
"description": "Evaluate a mathematical expression. Use for arithmetic, trig, logarithms. Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e."
```
