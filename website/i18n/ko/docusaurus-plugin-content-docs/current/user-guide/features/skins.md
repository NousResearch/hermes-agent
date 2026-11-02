---
sidebar_position: 10
title: "스킨 및 테마"
description: "기본 제공 스킨과 사용자 정의 스킨으로 Hermes CLI를 꾸밉니다"
---

# 스킨 및 테마

스킨은 Hermes CLI의 **시각적 표현**을 제어합니다. 배너 색상, 스피너 얼굴과 동사, 응답 상자 레이블, 브랜딩 텍스트, 도구 활동 접두사가 이에 포함됩니다.

대화 스타일과 시각적 스타일은 서로 다른 개념입니다.

- **개성**은 에이전트의 말투와 표현을 바꿉니다.
- **스킨**은 CLI의 외관을 바꿉니다.

## 스킨 변경

```bash
/skin                # show the current skin and list available skins
/skin ares           # switch to a built-in skin
/skin mytheme        # switch to a custom skin from ~/.hermes/skins/mytheme.yaml
```

또는 `~/.hermes/config.yaml`에서 기본 스킨을 설정할 수 있습니다.

```yaml
display:
  skin: default
```

## 기본 제공 스킨

| 스킨 | 설명 | 에이전트 브랜딩 | 시각적 특징 |
|------|-------------|------------------|------------------|
| `default` | 클래식 Hermes — 금색과 귀여운 스타일 | `Hermes Agent` | 따뜻한 금색 테두리, 콘실크 색상 텍스트, 스피너의 귀여운 얼굴. 익숙한 카두세우스 배너. 깔끔하고 친근합니다. |
| `ares` | 전쟁의 신 테마 — 진홍색과 청동색 | `Ares Agent` | 청동색 포인트가 있는 짙은 진홍색 테두리. 공격적인 스피너 동사("단조하기", "행군하기", "강철 담금질하기"). 커스텀 검과 방패 ASCII 아트 배너. |
| `mono` | 흑백 스타일 — 깔끔한 회색조 | `Hermes Agent` | 모든 요소가 회색이며 색상이 없습니다. 테두리는 `#555555`, 텍스트는 `#c9d1d9`입니다. 최소한의 터미널 환경이나 화면 녹화에 적합합니다. |
| `slate` | 차가운 파랑 — 개발자 중심 | `Hermes Agent` | 로열 블루 테두리(`#4169e1`)와 부드러운 파란색 텍스트. 차분하고 전문적입니다. 커스텀 스피너 없이 기본 얼굴을 사용합니다. |
| `daylight` | 어두운 텍스트와 시원한 파란색 포인트를 사용하는 밝은 터미널용 테마 | `Hermes Agent` | 흰색 또는 밝은 터미널을 위해 설계되었습니다. 파란색 테두리의 어두운 슬레이트 텍스트, 옅은 상태 표면, 밝은 터미널 프로필에서도 읽기 쉬운 밝은 완료 메뉴를 제공합니다. |
| `warm-lightmode` | 밝은 터미널 배경을 위한 따뜻한 갈색/금색 텍스트 | `Hermes Agent` | 밝은 터미널에 어울리는 따뜻한 양피지 색조입니다. 짙은 갈색 텍스트와 새들 브라운 포인트, 크림색 상태 표면을 사용합니다. 차가운 `daylight` 테마의 흙내음 나는 대안입니다. |
| `poseidon` | 바다의 신 테마 — 짙은 파랑과 바다 거품색 | `Poseidon Agent` | 짙은 파랑에서 바다 거품색으로 이어지는 그라데이션. 바다를 테마로 한 스피너("해류 그리기", "깊이 측심하기"). 삼지창 ASCII 아트 배너. |
| `sisyphus` | 시시포스 테마 — 끈기를 담은 절제된 회색조 | `Sisyphus Agent` | 강한 대비의 밝은 회색. 바위를 테마로 한 스피너("오르막으로 밀기", "바위 다시 설정하기", "순환 견디기"). 바위와 언덕 ASCII 아트 배너. |
| `charizard` | 화산 테마 — 번트 오렌지와 불씨 | `Charizard Agent` | 따뜻한 번트 오렌지에서 불씨 색으로 이어지는 그라데이션. 불을 테마로 한 스피너("기류에 진입하기", "연소량 측정하기"). 용 실루엣 ASCII 아트 배너. |

## 구성 가능한 키 전체 목록

### 색상(`colors:`)

CLI 전반의 모든 색상 값을 제어합니다. 값은 16진수 색상 문자열입니다.

| 키 | 설명 | 기본값(`default` 스킨) |
|-----|-------------|--------------------------|
| `banner_border` | 시작 배너를 둘러싼 패널 테두리 | `#CD7F32` (청동색) |
| `banner_title` | 배너의 제목 텍스트 색상 | `#FFD700` (금색) |
| `banner_accent` | 배너의 섹션 헤더(Available Tools 등) | `#FFBF00` (호박색) |
| `banner_dim` | 배너의 흐린 텍스트(구분선, 보조 레이블) | `#B8860B` (짙은 골든로드색) |
| `banner_text` | 배너의 본문 텍스트(도구 이름, 스킬 이름) | `#FFF8DC` (콘실크색) |
| `ui_accent` | 일반 UI 포인트 색상(강조 표시, 활성 요소) | `#FFBF00` |
| `ui_label` | UI 레이블과 태그 | `#DAA520` (골든로드색) |
| `ui_ok` | 성공 표시(체크 표시, 완료) | `#4caf50` (녹색) |
| `ui_error` | 오류 표시(실패, 차단됨) | `#ef5350` (빨간색) |
| `ui_warn` | 경고 표시(주의, 승인 프롬프트) | `#ffa726` (주황색) |
| `prompt` | 대화형 프롬프트 텍스트 색상 | `#FFF8DC` |
| `input_rule` | 입력 영역 위의 가로선 | `#CD7F32` |
| `response_border` | 에이전트 응답 상자 주변의 테두리(ANSI 이스케이프) | `#FFD700` |
| `session_label` | 세션 레이블 색상 | `#DAA520` |
| `session_border` | 세션 ID의 흐린 테두리 색상 | `#8B8682` |
| `status_bar_bg` | TUI 상태/사용량 표시줄의 배경색 | `#1a1a2e` |
| `voice_status_bg` | 음성 모드 상태 배지의 배경색 | `#1a1a2e` |
| `selection_bg` | TUI 마우스 선택 강조 표시의 배경색. 설정하지 않으면 `completion_menu_current_bg`로 대체됩니다. | `#3a3a55` |
| `completion_menu_bg` | 자동 완성 메뉴 목록의 배경색 | `#1a1a2e` |
| `completion_menu_current_bg` | 활성 자동 완성 행의 배경색 | `#333355` |
| `completion_menu_meta_bg` | 자동 완성 메타 열의 배경색 | `#1a1a2e` |
| `completion_menu_meta_current_bg` | 활성 자동 완성 메타 열의 배경색 | `#333355` |

### 스피너(`spinner:`)

API 응답을 기다리는 동안 표시되는 애니메이션 스피너를 제어합니다.

| 키 | 유형 | 설명 | 예시 |
|-----|------|-------------|---------|
| `waiting_faces` | 문자열 목록 | API 응답을 기다리는 동안 순환하는 얼굴 | `["(⚔)", "(⛨)", "(▲)"]` |
| `thinking_faces` | 문자열 목록 | 모델이 추론하는 동안 순환하는 얼굴 | `["(⚔)", "(⌁)", "(<>)"]` |
| `thinking_verbs` | 문자열 목록 | 스피너 메시지에 표시되는 동사 | `["forging", "plotting", "hammering plans"]` |
| `wings` | [왼쪽, 오른쪽] 쌍의 목록 | 스피너 양옆을 장식하는 괄호 | `[["⟪⚔", "⚔⟫"], ["⟪▲", "▲⟫"]]` |

스피너 값이 비어 있으면(`default`와 `mono`처럼) `display.py`의 하드코딩된 기본값이 사용됩니다.

### 브랜딩(`branding:`)

CLI 인터페이스 전반에서 사용되는 텍스트 문자열입니다.

| 키 | 설명 | 기본값 |
|-----|-------------|---------|
| `agent_name` | 배너 제목과 상태 표시 영역에 표시되는 이름 | `Hermes Agent` |
| `welcome` | CLI 시작 시 표시되는 환영 메시지 | `Welcome to Hermes Agent! Type your message or /help for commands.` |
| `goodbye` | 종료할 때 표시되는 메시지 | `Goodbye! ⚕` |
| `response_label` | 응답 상자 헤더의 레이블 | ` ⚕ Hermes ` |
| `prompt_symbol` | 사용자 입력 프롬프트 앞의 기호(일반 토큰이며 렌더러가 뒤에 공백을 추가함) | `❯` |
| `help_header` | `/help` 명령 출력의 헤더 텍스트 | `(^_^)? Available Commands` |

### 기타 최상위 키

| 키 | 유형 | 설명 | 기본값 |
|-----|-------------|-------------|---------|
| `tool_prefix` | 문자열 | CLI의 도구 출력 줄 앞에 붙는 문자 | `┊` |
| `tool_emojis` | dict | 스피너와 진행 표시를 위한 도구별 이모지 재정의(`{tool_name: emoji}`) | `{}` |
| `banner_logo` | 문자열 | Rich 마크업 ASCII 아트 로고(기본 HERMES_AGENT 배너 대체) | `""` |
| `banner_hero` | 문자열 | Rich 마크업 히어로 아트(기본 카두세우스 아트 대체) | `""` |

## 사용자 정의 스킨

`~/.hermes/skins/` 아래에 YAML 파일을 만듭니다. 사용자 스킨은 기본 제공 `default` 스킨에서 누락된 값을 상속하므로, 변경하려는 키만 지정하면 됩니다.

### 전체 사용자 정의 스킨 YAML 템플릿

```yaml
# ~/.hermes/skins/mytheme.yaml
# Complete skin template — all keys shown. Delete any you don't need;
# missing values automatically inherit from the 'default' skin.

name: mytheme
description: My custom theme

colors:
  banner_border: "#CD7F32"
  banner_title: "#FFD700"
  banner_accent: "#FFBF00"
  banner_dim: "#B8860B"
  banner_text: "#FFF8DC"
  ui_accent: "#FFBF00"
  ui_label: "#4dd0e1"
  ui_ok: "#4caf50"
  ui_error: "#ef5350"
  ui_warn: "#ffa726"
  prompt: "#FFF8DC"
  input_rule: "#CD7F32"
  response_border: "#FFD700"
  session_label: "#DAA520"
  session_border: "#8B8682"
  status_bar_bg: "#1a1a2e"
  voice_status_bg: "#1a1a2e"
  selection_bg: "#333355"
  completion_menu_bg: "#1a1a2e"
  completion_menu_current_bg: "#333355"
  completion_menu_meta_bg: "#1a1a2e"
  completion_menu_meta_current_bg: "#333355"

spinner:
  waiting_faces:
    - "(⚔)"
    - "(⛨)"
    - "(▲)"
  thinking_faces:
    - "(⚔)"
    - "(⌁)"
    - "(<>)"
  thinking_verbs:
    - "processing"
    - "analyzing"
    - "computing"
    - "evaluating"
  wings:
    - ["⟪⚡", "⚡⟫"]
    - ["⟪●", "●⟫"]

branding:
  agent_name: "My Agent"
  welcome: "Welcome to My Agent! Type your message or /help for commands."
  goodbye: "See you later! ⚡"
  response_label: " ⚡ My Agent "
  prompt_symbol: "⚡"
  help_header: "(⚡) Available Commands"

tool_prefix: "┊"

# Per-tool emoji overrides (optional)
tool_emojis:
  terminal: "⚔"
  web_search: "🔮"
  read_file: "📄"

# Custom ASCII art banners (optional, Rich markup supported)
# banner_logo: |
#   [bold #FFD700] MY AGENT [/]
# banner_hero: |
#   [#FFD700]  Custom art here  [/]
```

### 최소 사용자 정의 스킨 예시

모든 항목은 `default`에서 상속되므로, 최소한의 스킨은 다른 부분만 변경하면 됩니다.

```yaml
name: cyberpunk
description: Neon terminal theme

colors:
  banner_border: "#FF00FF"
  banner_title: "#00FFFF"
  banner_accent: "#FF1493"

spinner:
  thinking_verbs: ["jacking in", "decrypting", "uploading"]
  wings:
    - ["⟨⚡", "⚡⟩"]

branding:
  agent_name: "Cyber Agent"
  response_label: " ⚡ Cyber "

tool_prefix: "▏"
```

## Hermes Mod — 시각적 스킨 편집기

[Hermes Mod](https://github.com/cocktailpeanut/hermes-mod)는 스킨을 시각적으로 만들고 관리할 수 있는 커뮤니티 제작 웹 UI입니다. YAML을 손으로 작성하는 대신, 클릭 몇 번으로 편집하고 실시간 미리 보기를 확인할 수 있습니다.

![Hermes Mod 스킨 편집기](https://raw.githubusercontent.com/cocktailpeanut/hermes-mod/master/nous.png)

**주요 기능:**

- 기본 제공 스킨과 사용자 정의 스킨을 모두 나열합니다.
- 모든 Hermes 스킨 필드(색상, 스피너, 브랜딩, 도구 접두사, 도구 이모지)가 포함된 시각적 편집기에서 원하는 스킨을 엽니다.
- 텍스트 프롬프트로 `banner_logo` 텍스트 아트를 생성합니다.
- 업로드한 이미지(PNG, JPG, GIF, WEBP)를 여러 렌더링 스타일(점자, ASCII 램프, 블록, 점)의 `banner_hero` ASCII 아트로 변환합니다.
- `~/.hermes/skins/`에 직접 저장합니다.
- `~/.hermes/config.yaml`을 업데이트하여 스킨을 활성화합니다.
- 생성된 YAML과 실시간 미리 보기를 표시합니다.

### 설치

**옵션 1 — Pinokio(원클릭):**

[pinokio.computer](https://pinokio.computer)에서 찾아 한 번의 클릭으로 설치합니다.

**옵션 2 — npx(터미널에서 가장 빠른 방법):**

```bash
npx -y hermes-mod
```

**옵션 3 — 수동 설치:**

```bash
git clone https://github.com/cocktailpeanut/hermes-mod.git
cd hermes-mod/app
npm install
npm start
```

### 사용법

1. 앱을 시작합니다(Pinokio 또는 터미널을 통해).
2. **Skin Studio**를 엽니다.
3. 편집할 기본 제공 스킨 또는 사용자 정의 스킨을 선택합니다.
4. 텍스트에서 로고를 생성하거나 이미지를 업로드해 히어로 아트를 만듭니다. 렌더링 스타일과 너비를 선택합니다.
5. 색상, 스피너, 브랜딩 및 기타 필드를 편집합니다.
6. **Save**를 클릭하여 스킨 YAML을 `~/.hermes/skins/`에 기록합니다.
7. **Activate**를 클릭하여 현재 스킨으로 설정합니다(`config.yaml`의 `display.skin`을 업데이트합니다).

Hermes Mod는 `HERMES_HOME` 환경 변수를 인식하므로 [프로필](/user-guide/profiles)에서도 사용할 수 있습니다.

## 운영 참고 사항

- 기본 제공 스킨은 `hermes_cli/skin_engine.py`에서 불러옵니다.
- 알 수 없는 스킨은 자동으로 `default`로 대체됩니다.
- `/skin`은 현재 세션의 활성 CLI 테마를 즉시 업데이트합니다.
- `~/.hermes/skins/`의 사용자 스킨은 이름이 같은 기본 제공 스킨보다 우선합니다.
- `/skin`을 통한 스킨 변경은 현재 세션에만 적용됩니다. 스킨을 영구적인 기본값으로 설정하려면 `config.yaml`에 지정합니다.
- `banner_logo`와 `banner_hero` 필드는 색상이 있는 ASCII 아트를 위한 Rich 콘솔 마크업(예: `[bold #FF0000]text[/]`)을 지원합니다.
