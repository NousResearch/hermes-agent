---
sidebar_position: 11
title: "펫 (Petdex 마스코트)"
description: "CLI, TUI, 데스크톱 앱에서 에이전트 활동에 반응하는 애니메이션 마스코트를 입양합니다"
---

# 펫

Hermes는 에이전트가 하는 일(대기, 도구 실행, 사고, 완료, 실패)에 반응하는
작은 마스코트 스프라이트인 애니메이션 **펫**을 **CLI**, **TUI**, **데스크톱 앱**에서
표시할 수 있습니다. 펫은 공개
[petdex](https://github.com/crafter-station/petdex) 갤러리에서 제공됩니다.

펫은 순전히 장식용입니다. **프롬프트 캐싱, 토큰 또는 에이전트 동작에 아무런
영향을 주지 않으며**, 스프라이트는 표시 전용입니다. 이 기능은 **기본적으로
꺼져** 있으며, 펫을 설치하고 선택할 때까지 동작하지 않습니다.

## 작동 방식

- 펫은 프로필의 `pets/` 디렉터리(`<HERMES_HOME>/pets/<slug>/`)에 설치되므로 각
  [프로필](../profiles.md)이 자체 펫을 관리합니다.
- 펫을 선택하면 `display.pet.slug` 및 `display.pet.enabled`가 `config.yaml`에
  기록됩니다. 비밀이나 env var로 저장되는 값은 없습니다.
- 각 화면은 이미 추적 중인 활동을 감시하고 이를 6가지 애니메이션 상태 중 하나에
  매핑합니다. 모든 화면이 동일하게 동작하도록 매핑은 한 곳에서 관리합니다:

  | 에이전트 활동 | 펫 상태 |
  | --- | --- |
  | 도구 또는 턴이 방금 실패함 | `failed` |
  | 계획이 완료됨(모든 todo 완료) | `jump` (축하) |
  | 턴이 문제없이 완료됨 | `wave` |
  | 도구를 실행 중임 | `run` |
  | 모델이 사고하거나 읽는 중임 | `review` |
  | 턴이 진행 중임(상태를 특정할 수 없음) | `run` |
  | 사용자의 입력을 기다리는 중임(명확화/승인 프롬프트가 열려 있음) | `waiting` (이전 8행 시트에서는 `idle`로 대체) |
  | 아무 일도 일어나지 않음 | `idle` |

## 렌더링

터미널(CLI/TUI)에서 Hermes는 터미널이 그래픽 프로토콜(**kitty**, **Ghostty**,
**WezTerm**, **iTerm2** 또는 **sixel**)을 지원하면 스프라이트를 원본 품질로
렌더링합니다. 그렇지 않으면 자동으로 트루컬러 Unicode **하프 블록** 렌더링으로
대체합니다. 파이프나 리디렉션 내부(TTY 없음)에서는 터미널 렌더링이 의도적으로
비활성화됩니다.

데스크톱 앱에서는 캔버스에 펫을 떠 있는 스프라이트로 그리며, **설정 → 모양**에서
켜고 끌 수 있습니다.

## 빠른 시작 (CLI)

```bash
# Browse the gallery (filter by substring)
hermes pets list
hermes pets list cat

# Install a pet and make it active in one step
hermes pets install boba --select

# Preview / animate it in your terminal (Ctrl+C to stop)
hermes pets show

# Check your setup
hermes pets doctor
```

## `hermes pets` 명령

| 목표 | 명령 |
| --- | --- |
| 갤러리 둘러보기 | `hermes pets list [query] [--limit N]` |
| 설치된 펫 목록 보기 | `hermes pets list --installed` |
| 펫 설치 | `hermes pets install <slug> [--select] [--force]` |
| 활성 펫 설정 | `hermes pets select [slug]` (slug를 생략하면 선택기 표시) |
| 모든 곳에서 펫 크기 조정 | `hermes pets scale <factor>` (예: `0.5`, 0.1–3.0 범위로 제한) |
| 미리 보기/애니메이션 | `hermes pets show [slug] [--state <s>] [--cycle] [--once] [--mode <m>] [--scale <f>]` |
| 펫 비활성화 | `hermes pets off` |
| 설치된 펫 제거 | `hermes pets remove <slug>` |
| 설정 진단 | `hermes pets doctor` |

`hermes pets show` 플래그:

- `--state` — 단일 상태 재생(`idle`, `wave`, `run`, `failed`, `review`,
  `jump`).
- `--cycle` — 모든 상태를 순환합니다.
- `--once` — 반복하지 않고 한 번만 재생합니다.
- `--mode` — 렌더링 프로토콜 재정의(`kitty`, `iterm`, `sixel`,
  `unicode`, `auto`).
- `--scale` — 화면 배율 재정의(`0` = config 사용).

## `/pet` 슬래시 명령

CLI와 TUI에서는 세션을 나가지 않고도 펫을 관리할 수 있습니다:

- `/pet` — 펫을 켜거나 끕니다(활성 펫이 없으면 설치된 첫 번째 펫을 입양합니다).
- `/pet list` — 갤러리를 둘러봅니다.
- `/pet scale <factor>` — 모든 곳에서 펫 크기를 조정합니다(예: `/pet scale 0.5`).
- `/pet <slug>` — 특정 펫을 입양합니다.
- `/pet off` — 펫을 비활성화합니다.

TUI에서는 `/pet list`가 대화형 선택기 오버레이를 열고, 데스크톱 앱에서는
Cmd+K 펫 팔레트를 엽니다.

## 펫 생성(`/hatch`)

갤러리의 미리 만들어진 펫을 설치하는 것 외에도 Hermes는 텍스트 설명을 바탕으로
완전히 새로운 펫을 **생성**할 수 있습니다. 자체 AI 스프라이트 생성 파이프라인을
사용합니다.

- CLI/TUI: `/hatch <description>`(`/generate-pet` 별칭) 또는 `hermes pets` → 생성 흐름.
- 데스크톱 앱: Pokédex 스타일의 **생성** UI — 애니메이션 알, 부화 FX, 초안 선택기.

생성 방식(비용이 제한된 2단계 흐름):

1. **기본 초안** — 저렴한 프롬프트 전용 방식으로 "이 펫은 어떻게 생겨야 할까"에
   대한 여러 변형을 생성합니다. 하나를 선택하거나 리믹스/재시도로 새 라운드를
   시작할 수 있습니다.
2. **부화** — 선택한 기본 초안을 참조 이미지로 사용해 Hermes의 각 상태(대기,
   사고, 도구 사용 등)에 맞는 애니메이션 행을 하나씩 생성합니다. 이 행은
   결정론적으로 프레임으로 나뉘고 표준 petdex/Codex 아틀라스(192×208 셀의
   8×9 그리드)로 패킹됩니다. 완성된 스프라이트 시트는 보관할 수 있으며
   `petdex submit`할 수도 있습니다.

### 이미지 백엔드

생성에는 활성 [이미지 생성 공급자](/user-guide/features/image-generation)가
사용되지만, 각 애니메이션 행이 기본 이미지와 동일한 캐릭터를 유지하려면
**참조 이미지 기반 생성**이 필요합니다. 참조를 지원하는 백엔드는 **Nous Portal**,
**OpenRouter**, **OpenAI**(`gpt-image-2`) 및 **Krea**입니다. OpenRouter/Nous는
기본적으로 품질 우선 모델 체인을 사용합니다.

- 해상도 순서는 Nous Portal → OpenAI → OpenRouter를 우선합니다.
- 참조 이미지를 지원하는 백엔드가 구성되지 않았으면 `hermes tools` → 이미지
  생성으로 이동하라는 실행 가능한 오류가 표시됩니다. (기존 갤러리 펫을
  설치하거나 입양하는 데는 이미지 백엔드가 필요하지 않습니다.)
- `HERMES_PET_IMAGE_PROVIDER` env var로 백엔드를 재정의할 수 있습니다(예:
  `HERMES_PET_IMAGE_PROVIDER=openrouter`).

## 데스크톱 앱

데스크톱 앱에서는 두 가지 방법으로 펫을 관리할 수 있습니다:

- **Cmd+K → "Pets…"** — 키보드에서 벗어나지 않고 펫을 둘러보고, 검색하고,
  입양하고, 켜고 끕니다(테마 선택기와 동일한 방식).
- **설정 → 모양** — 드래그하는 동안 떠 있는 마스코트의 크기를 조정하는
  **크기 슬라이더**가 있는 동일한 갤러리입니다.

두 방법 모두 떠 있는 마스코트를 그 자리에서 입양·전환·크기 조정합니다. 크기
변경은 즉시 적용되고, 새 펫을 입양하면 잠시 후 활성화됩니다.

### 배회

설정 → 모양에는 **배회** 토글이 있습니다. 켜면 에이전트가 대기 중일 때 펫이
창 안을 스스로 돌아다닙니다. 표면을 걷고, 멈추고, 장소 사이를 뛰어다닙니다.
배회는 펫이 창 안에 있고 활성 상태이며 에이전트가 쉬고 있을 때만 실행됩니다.
에이전트가 주도하는 상태(작업 중, 축하 중)가 되면 즉시 그 상태가 우선합니다.
토글은 기본적으로 꺼져 있으며 재시작 후에도 유지됩니다.

### Alt+휠 크기 조정

**Alt**를 누른 채 펫 위에서 마우스 휠을 스크롤하면 앱 창 안과 분리된 오버레이
모두에서 그 자리의 펫 크기를 조정할 수 있습니다. 오버레이는 커서 위치를
기준으로 확대되며 결과 배율이 저장되므로, 재시작 후에도 유지되고 앱 안의 펫과
동기화됩니다.

### 분위기 반응

에이전트에게 "good bot", "thank you", "ily", `<3` 또는 하트 이모지처럼 좋은
말을 해 보세요. 펫이 떠다니는 하트(데스크톱) 또는 하트 플래시(CLI/TUI)로
반응합니다. 감지는 각 사용자 메시지에서 로컬로 일치시키는 엄선된 토큰 없는
어휘를 사용하며 모델을 호출하지 않습니다. 일반적인 긍정 감정이 아니라
에이전트를 향한 애정과 감사에 반응합니다. CLI 펫, TUI, 데스크톱의 떠 있는 펫,
분리된 오버레이 등 모든 화면이 동일한 신호에 반응합니다.

### 분리된 오버레이

떠 있는 펫을 **Shift-클릭**하면 투명하고 항상 위에 표시되는 별도의 데스크톱
창으로 분리됩니다. 분리된 상태에서는 Hermes가 최소화되어 있어도(Codex 방식)
계속 표시되므로, 펫을 힐끗 보는 것만으로도 에이전트가 무엇을 하는지 알 수
있습니다.

분리한 뒤의 제스처:

| 제스처 | 동작 |
| --- | --- |
| **드래그** | 앱 바깥을 포함해 화면 어디로든 펫을 이동합니다. 위치와 안/밖 상태는 재시작 후에도 유지됩니다. |
| **한 번 클릭** | 앱을 화면 앞으로 가져오지 않고 최근 세션에 프롬프트를 보낼 수 있는 미니 작성기를 엽니다. |
| **두 번 클릭** | 앱 창을 전환합니다. 앞에 표시 중이면 최소화하고, 숨겨져 있으면 복원합니다. |
| **Shift-클릭** | 펫을 창 안으로 다시 가져옵니다. |
| **메일 아이콘** | 자리를 비운 동안 턴이 완료된 경우에만 나타납니다. 클릭하면 가장 최근 스레드에서 앱을 열고 읽음으로 표시합니다. |

분리된 펫에만 **말풍선**(`working…`, `thinking…`, `your turn`, …)이 표시됩니다.
창 안에서는 앱 자체가 화면이므로 펫은 조용히 있습니다.

오버레이는 앱 안의 펫을 그대로 보여 주는 순수한 퍼펫입니다. 별도의 게이트웨이
연결을 가지지 않으며 도크나 앱 전환기에 나타나지 않습니다.

## 구성

모든 설정은 `config.yaml`의 `display.pet` 아래에 있습니다:

```yaml
display:
  pet:
    enabled: false        # master on/off (true once you select a pet)
    slug: ""              # active pet; empty = first installed
    render_mode: auto      # auto | kitty | iterm | sixel | unicode | off
    scale: 0.33           # master size knob (relative to native 192x208 frames)
    unicode_cols: 0       # hard override for terminal width (0 = derive from scale)
```

- **`scale`**은 하나의 마스터 크기 조절 값입니다. 하나의 숫자로 모든 화면의
  크기를 줄입니다. 데스크톱 캔버스는 이 값을 픽셀 배율로 사용하고, CLI/TUI는
  이 값에서 터미널 열 너비를 계산합니다. 하프 블록 대체 렌더링은 가독성을 위한
  하한을 적용합니다. 너무 작게 줄이면 형태를 알아보기 어려워지므로 트루 픽셀
  kitty/GUI 렌더링만큼 작아질 수 없습니다. 따라서 같은 `scale`이라도 kitty에서는
  선명하게 보이고 하프 블록에서는 하한이 적용됩니다.
- **`render_mode: auto`**는 kitty/iTerm2/sixel을 감지하고 Unicode 하프 블록으로
  대체합니다. 특정 프로토콜을 강제하려면 명시적으로 설정하고, 데스크톱에서는
  펫을 유지한 채 터미널 렌더링을 끄려면 `off`로 설정합니다.
- **`unicode_cols`**는 `scale`과 독립적으로 터미널 열 너비를 고정합니다.
  `scale`에서 너비를 계산하려면 `0`으로 둡니다.

## 문제 해결

`hermes pets doctor`를 실행하면 다음을 보고합니다:

- 펫 디렉터리와 설치된 펫 목록
- `display.pet.enabled`, `display.pet.slug`, 확인된 활성 펫
- 구성된 `render_mode`, 감지된 터미널 그래픽 프로토콜, TTY에 적용되는 유효 모드
- Pillow(스프라이트 디코딩에 사용)를 import할 수 있는지 여부

펫이 설치되고 선택되었으며 활성화되어 있고 Pillow를 사용할 수 있으면
`✓ ready`를 출력합니다.

자주 발생하는 문제:

- 펫은 **설치와 선택을 모두** 마쳤을 때만 표시됩니다(`enabled: true`).
- 파이프/리디렉션 내부(TTY 없음)에서는 터미널 렌더링이 의도적으로 비활성화됩니다.
- petdex npm CLI는 `~/.codex/pets`에 설치하지만 Hermes는 프로필별
  `<HERMES_HOME>/pets/`를 사용합니다. Hermes에서는 `hermes pets`로 설치하세요.

## 함께 보기

- [`hermes-agent` 스킬](../skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md)은
  에이전트가 요청에 따라 펫을 설치하고 전환하도록 합니다(`references/petdex.md` 참조).
