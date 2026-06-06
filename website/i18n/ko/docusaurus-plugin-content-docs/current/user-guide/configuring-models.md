---
sidebar_position: 3
---

# 모델 구성하기 (Configuring Models)

Hermes는 두 가지 종류의 모델 슬롯을 사용합니다:

- **주 모델 (Main model)** — 에이전트가 사고하는 데 사용하는 모델. 모든 사용자 메시지, 모든 도구 호출 루프, 모든 스트리밍 응답이 이 모델을 거칩니다.
- **보조 모델 (Auxiliary models)** — 에이전트가 오프로드하는 더 작은 부가 작업용 모델. 컨텍스트 압축, 비전(이미지 분석), 웹 페이지 요약, 승인 채점, MCP 도구 라우팅, 세션 제목 생성, 스킬 검색 등이 포함됩니다. 각각 고유한 슬롯을 가지며 독립적으로 재정의할 수 있습니다.

이 페이지에서는 대시보드에서 이 두 가지를 구성하는 방법을 다룹니다. 구성 파일이나 CLI를 선호하는 경우 하단의 [대체 방법 (Alternative methods)](#대체-방법-alternative-methods)으로 이동하세요.

:::tip 가장 빠른 방법: Nous Portal
[Nous Portal](/user-guide/features/tool-gateway)은 하나의 구독으로 300개 이상의 모델을 제공합니다. 새로 설치한 경우 `hermes setup --portal`을 실행하여 한 번의 명령으로 로그인하고 Nous를 제공자로 설정하세요. 연결된 내용은 `hermes portal info`로 검사할 수 있습니다.

- Portal 구독자는 **토큰 청구 제공자에 대해 10% 할인**도 받습니다.
:::

:::note `model:` 스키마 — 빈 문자열 vs 매핑
새로 설치한 기본 구성에는 `model: ""`("아직 구성되지 않음"을 의미하는 빈 문자열 구분 기호)이 있습니다. `hermes setup` 또는 `hermes model`을 처음 실행하면 이 키는 이 페이지 전체와 [`profiles.md`](./profiles.md) / [`configuration.md`](./configuration.md)에 표시된 모양인 `provider`, `default`, `base_url`, `api_mode` 하위 키가 있는 매핑(mapping)으로 적절하게 업그레이드됩니다. `config.yaml`에 빈 문자열이 표시되면 `hermes model`을 실행하거나 대시보드에서 **Change(변경)**를 클릭하면 Hermes가 딕셔너리 형태로 작성해 줍니다.
:::

## 모델 (Models) 페이지

대시보드를 열고 사이드바에서 **Models**를 클릭하세요. 두 개의 섹션이 있습니다:

1. **모델 설정 (Model Settings)** — 모델을 슬롯에 할당하는 상단 패널.
2. **사용 분석 (Usage analytics)** — 선택한 기간 동안 세션을 실행한 모든 모델을 표시하는 순위표 카드로 토큰 수, 비용 및 기능 배지를 표시합니다.

![Models page overview](/img/docs/dashboard-models/overview.png)

상단 카드는 **Model Settings** 패널입니다. 주(Main) 행은 에이전트가 새 세션을 위해 시작할 모델을 항상 보여줍니다. 피커(picker)를 열려면 **Change**를 클릭하세요.

## 주 모델 설정

주 모델 행에서 **Change**를 클릭하세요:

![Model picker dialog](/img/docs/dashboard-models/picker-dialog.png)

피커에는 두 개의 열이 있습니다:

- **왼쪽** — 인증된 제공자. 설정한 제공자(API 키 설정됨, OAuth 인증됨 또는 사용자 지정 엔드포인트로 정의됨)만 여기에 표시됩니다. 제공자가 누락된 경우 **Keys**로 이동하여 자격 증명을 추가하세요.
- **오른쪽** — 선택한 제공자의 선별된 모델 목록. 이는 해당 제공자에 대해 Hermes가 권장하는 에이전트용 모델이며, (OpenRouter에서 TTS, 이미지 생성기 및 리랭커를 포함한 400개 이상의 모델이 있는) 처리되지 않은 `/models` 덤프가 아닙니다.

제공자 이름, 슬러그(slug) 또는 모델 ID로 좁히려면 필터 상자에 입력하세요.

모델을 선택하고 **Switch**를 누르면 Hermes가 `~/.hermes/config.yaml`의 `model` 섹션에 이를 작성합니다. **이것은 새 세션에만 적용됩니다** — 이미 열려 있는 채팅 탭은 시작했던 모델로 계속 실행됩니다. 현재 채팅을 즉시 변경(hot-swap)하려면 그 안에서 `/model` 슬래시 명령어를 사용하세요.

## 보조 모델 설정

8개의 작업 슬롯을 보려면 **Show auxiliary**를 클릭하세요:

![Auxiliary panel expanded](/img/docs/dashboard-models/auxiliary-expanded.png)

모든 보조 작업의 기본값은 `auto`입니다 — 즉, Hermes가 해당 작업에도 주 모델을 사용함을 의미합니다. 부가 작업을 위해 더 저렴하거나 빠른 모델을 원할 때 특정 작업을 재정의하세요.

### 일반적인 재정의 패턴

| 작업 | 재정의할 때 |
|---|---|
| **Title Gen (제목 생성)** | 거의 항상. $0.10/M 플래시 모델도 Opus만큼 세션 제목을 잘 작성합니다. 기본 구성은 OpenRouter에서 이것을 `google/gemini-3-flash-preview`로 설정합니다. |
| **Vision (비전)** | 주 모델이 비전 지원이 부족할 때. `google/gemini-2.5-flash` 또는 `gpt-4o-mini`를 가리키게 하세요. |
| **Compression (압축)** | 컨텍스트를 요약하기 위해서만 Opus/M2.7의 추론 토큰을 낭비하고 있을 때. 빠른 채팅 모델이 비용의 1/50로 그 작업을 수행합니다. |
| **Approval (승인)** | `approval_mode: smart`인 경우 — 빠르고 저렴한 모델(haiku, flash, gpt-5-mini)이 위험이 낮은 명령을 자동 승인할지 결정합니다. 여기에 비싼 모델을 사용하는 것은 낭비입니다. |
| **Web Extract (웹 추출)** | `web_extract`를 많이 사용할 때. 압축과 같은 논리입니다 — 요약에는 고도화된 추론이 필요하지 않습니다. |
| **Skills Hub (스킬 허브)** | `hermes skills search`가 이를 사용합니다. 일반적으로 `auto`로 두는 것이 좋습니다. |
| **MCP** | MCP 도구 라우팅. 일반적으로 `auto`로 두는 것이 좋습니다. |

### 작업별 재정의

보조 행에서 **Change**를 클릭하세요. 동일한 피커가 열리고 동일한 동작을 합니다 — 제공자 + 모델을 선택하고 Switch를 누릅니다. 행이 업데이트되어 `auto (use main model)` 대신 `provider · model`을 표시합니다.

### 모두 자동으로 재설정 (Reset all to auto)

너무 많이 수정해서 처음부터 다시 시작하고 싶다면 보조 섹션 상단에 있는 **Reset all to auto**를 클릭하세요. 모든 슬롯이 다시 주 모델을 사용하게 됩니다.

## "Use as" 단축키

페이지의 모든 모델 카드에는 **Use as** 드롭다운이 있습니다. 이것은 빠른 경로입니다 — 분석에서 보이는 모델을 선택하고 **Use as**를 클릭하여 클릭 한 번으로 메인 슬롯이나 특정 보조 작업에 할당할 수 있습니다:

![Use as dropdown](/img/docs/dashboard-models/use-as-dropdown.png)

드롭다운 항목:

- **Main model** — 주 행에서 Change를 클릭하는 것과 같습니다.
- **All auxiliary tasks** — 이 모델을 8개의 보조 슬롯 모두에 한 번에 할당합니다. 모든 부가 작업을 저렴한 플래시 모델에서 수행하고 싶을 때 유용합니다.
- **Individual task options** — Vision, Web Extract, Compression 등. 각 작업에 현재 할당된 모델은 `current`로 표시됩니다.

카드가 현재 무언가에 할당되어 있으면 `main` 또는 `aux · <task>` 배지가 지정되므로 과거 모델 중 어느 것이 어디에 연결되어 있는지 한눈에 볼 수 있습니다.

## `config.yaml`에 기록되는 내용

대시보드를 통해 저장하면 Hermes가 `~/.hermes/config.yaml`에 작성합니다:

**주 모델:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # 제공자 전환 시 지워짐
  api_mode: chat_completions
```

**보조 재정의 (예: gemini-flash에서의 비전):**
```yaml
auxiliary:
  vision:
    provider: openrouter
    model: google/gemini-2.5-flash
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
    download_timeout: 30
```

**자동 설정된 보조 모델 (기본값):**
```yaml
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    # ... 변경되지 않은 다른 필드들
```

`model: ''`과 함께 지정된 `provider: auto`는 Hermes에게 해당 작업에 주 모델을 사용하라고 지시합니다.

## 언제 적용되나요?

- **CLI** (`hermes chat`): 다음 `hermes chat` 실행 시.
- **게이트웨이** (Telegram, Discord, Slack 등): 다음 *새* 세션 시. 기존 세션은 해당 모델을 유지합니다. 모든 세션이 변경 사항을 적용하도록 강제하려면 게이트웨이를 재시작하세요(`hermes gateway restart`).
- **대시보드 채팅 탭** (`/chat`): 다음 새 PTY 시. 현재 열려 있는 채팅은 해당 모델을 유지합니다 — 현재 세션을 즉시 변경(hot-swap)하려면 그 안에서 `/model`을 사용하세요.

변경 사항은 실행 중인 세션의 프롬프트 캐시를 무효화하지 않습니다. 이는 의도적입니다. 세션 내부에서 주 모델을 교체하려면 캐시 재설정(시스템 프롬프트에는 모델별 콘텐츠가 포함됨)이 필요하며, 우리는 이를 채팅 내 명시적인 `/model` 슬래시 명령어에만 유보합니다.

## 문제 해결 (Troubleshooting)

### 피커에 "No authenticated providers (인증된 제공자 없음)"이 표시됨

Hermes는 작동하는 자격 증명이 있는 경우에만 제공자를 나열합니다. 사이드바에서 **Keys**를 확인하세요 — API 키, 성공적인 OAuth 또는 사용자 지정 엔드포인트 URL 중 하나가 표시되어야 합니다. 원하는 제공자가 없으면 `hermes setup`을 실행하여 연결하거나 **Keys**로 이동하여 환경 변수를 추가하세요.

### 실행 중인 채팅에서 주 모델이 변경되지 않음

정상적인 현상입니다. 대시보드는 새 세션이 읽을 `config.yaml`을 작성합니다. 현재 열려 있는 채팅은 실시간 에이전트 프로세스입니다 — 이는 시작될 때의 모델을 그대로 유지합니다. 해당 특정 세션을 즉시 교체하려면 채팅 내부에서 `/model <name>`을 사용하세요.

### 보조 재정의가 "적용되지 않음"

확인해야 할 세 가지:

1. **새 세션을 시작했나요?** 기존 채팅은 구성을 다시 읽지 않습니다.
2. **`provider`가 `auto`가 아닌 다른 것으로 설정되어 있나요?** 필드에 `auto`가 표시되면 작업은 여전히 주 모델을 사용하고 있는 것입니다. **Change**를 클릭하고 실제 제공자를 선택하세요.
3. **제공자가 인증되었나요?** 작업에 `minimax`를 할당했지만 MiniMax API 키가 없는 경우 해당 작업은 openrouter 기본값으로 돌아가고 `agent.log`에 경고를 기록합니다.

### 모델을 선택했는데 Hermes가 제공자를 전환함

OpenRouter(또는 모든 어그리게이터)에서 단일 모델 이름은 먼저 어그리게이터 *내부*에서 확인됩니다. 따라서 OpenRouter의 `claude-sonnet-4`는 OpenRouter 인증을 유지하면서 `anthropic/claude-sonnet-4.6`이 됩니다. 하지만 기본 Anthropic 인증에서 `claude-sonnet-4`를 입력하면 `claude-sonnet-4-6`으로 유지됩니다. 예기치 않은 제공자 전환이 표시되면 현재 제공자가 예상하는 것인지 확인하세요 — 피커는 대화 상자 상단에 항상 현재 주 모델을 표시합니다.

## 대체 방법 (Alternative methods)

### CLI 슬래시 명령어

모든 `hermes chat` 세션 내부:

```
/model gpt-5.4 --provider openrouter             # 세션 전용
/model gpt-5.4 --provider openrouter --global    # config.yaml에도 유지됨
```

`--global`은 대시보드의 **Change** 버튼이 수행하는 것과 동일한 작업을 수행하며, 게다가 실행 중인 세션을 즉석에서 교체합니다.

### 사용자 지정 별칭 (Custom aliases)

자주 찾는 모델에 대한 고유한 짧은 이름을 정의한 다음, CLI 또는 메시징 플랫폼에서 `/model <alias>`를 사용하세요. 두 가지 동등한 형식이 있습니다 — 워크플로에 맞는 것을 선택하세요.

**Canonical (최상위 `model_aliases:`)** — 제공자 + base_url에 대한 모든 제어:

```yaml
# ~/.hermes/config.yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
```

**짧은 문자열 형식 (`model.aliases.<name>: provider/model`)** — `hermes config set`은 스칼라 값만 쓰므로 쉘에서 편리하지만, 사용자 지정 `base_url`을 가질 수는 없습니다:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

두 경로 모두 동일한 로더(`hermes_cli/model_switch.py`)로 공급됩니다. `model_aliases:`에 선언된 항목이 동일한 이름의 `model.aliases:` 항목보다 우선합니다.

그 후 채팅에서 `/model fav` 또는 `/model grok`를 사용합니다. 사용자 별칭은 기본으로 내장된 짧은 이름(`sonnet`, `kimi`, `opus` 등)을 가립니다. 전체 참조는 [사용자 지정 모델 별칭](/reference/slash-commands#custom-model-aliases)을 확인하세요.

### `hermes model` 하위 명령어

```bash
hermes model            # 대화형 제공자 + 모델 피커 (기본값을 전환하는 표준적인 방법)
```

`hermes model`은 제공자를 선택하고, 인증하고(OAuth 흐름은 브라우저를 엽니다; API 키 제공자는 키를 묻습니다), 해당 제공자의 선별된 카탈로그에서 특정 모델을 선택하는 과정을 안내합니다. 선택한 항목은 `~/.hermes/config.yaml`의 `model.provider` 및 `model.model`에 작성됩니다.

피커를 실행하지 않고 제공자/모델을 나열하려면 대시보드나 아래의 REST 엔드포인트를 사용하세요. CLI가 지금 당장 실제로 사용할 항목을 검사하려면: `hermes config show | grep '^model\.'` 및 `hermes status`.

### 직접 구성 편집

`~/.hermes/config.yaml`을 편집하고 그것을 읽는 프로세스를 재시작하세요. 전체 스키마는 [구성 참조(Configuration reference)](./configuration.md)를 확인하세요.

### REST API

대시보드는 세 가지 엔드포인트를 사용합니다. 스크립팅에 유용합니다:

```bash
# 인증된 제공자 + 선별된 모델 목록 나열
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# 현재 메인 + 보조 할당 읽기
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# 주 모델 설정
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# 단일 보조 작업 재정의
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# 하나의 모델을 모든 보조 작업에 할당
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# 모든 보조 작업을 자동으로 재설정
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

세션 토큰은 시작 시 대시보드 HTML에 주입되며 서버가 다시 시작될 때마다 변경(rotate)됩니다. 실행 중인 대시보드에 대해 스크립팅하는 경우 브라우저 개발자 도구(`window.__HERMES_SESSION_TOKEN__`)에서 토큰을 가져오세요.
