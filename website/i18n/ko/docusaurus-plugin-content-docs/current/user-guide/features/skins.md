---
sidebar_position: 10
title: "스킨 & 테마 (Skins & Themes)"
description: "기본 제공 스킨 및 사용자 정의 스킨으로 Hermes CLI 맞춤 설정하기"
---

# 스킨 & 테마 (Skins & Themes)

스킨은 Hermes CLI의 **시각적 표현**을 제어합니다. 배너 색상, 스피너의 표정과 동사, 응답 상자 라벨, 브랜드 텍스트, 도구 활동 접두사 등을 설정할 수 있습니다.

대화 스타일과 시각적 스타일은 별개의 개념입니다:

- **성격 (Personality)**은 에이전트의 어조와 단어 선택을 변경합니다.
- **스킨 (Skin)**은 CLI의 외관을 변경합니다.

## 스킨 변경

```bash
/skin                # 현재 스킨을 표시하고 사용 가능한 스킨 목록을 나열합니다.
/skin ares           # 기본 제공 스킨으로 전환합니다.
/skin mytheme        # ~/.hermes/skins/mytheme.yaml의 사용자 정의 스킨으로 전환합니다.
```

또는 `~/.hermes/config.yaml`에서 기본 스킨을 설정할 수 있습니다:

```yaml
display:
  skin: default
```

## 기본 제공 스킨

| 스킨 (Skin) | 설명 (Description) | 에이전트 브랜드 (Agent branding) | 시각적 특징 (Visual character) |
|------|-------------|----------------|------------------|
| `default` | 클래식 Hermes — 골드 및 카와이(kawaii) | `Hermes Agent` | 따뜻한 골드 테두리, cornsilk 색상 텍스트, 스피너의 카와이 표정. 친숙한 카두세우스 배너. 깔끔하고 매력적입니다. |
| `ares` | 전쟁의 신 테마 — 진홍색과 청동색 | `Ares Agent` | 청동색 포인트가 있는 깊은 진홍색 테두리. 공격적인 스피너 동사("forging", "marching", "tempering steel"). 사용자 정의 검과 방패 ASCII 아트 배너. |
| `mono` | 모노크롬 — 깔끔한 흑백 | `Hermes Agent` | 모두 회색 — 색상이 없습니다. 테두리는 `#555555`, 텍스트는 `#c9d1d9`입니다. 미니멀한 터미널 설정이나 화면 녹화에 이상적입니다. |
| `slate` | 쿨 블루 — 개발자 친화적 | `Hermes Agent` | 로열 블루 테두리(`#4169e1`), 부드러운 파란색 텍스트. 차분하고 전문적입니다. 사용자 정의 스피너가 없으며 기본 표정을 사용합니다. |
| `daylight` | 어두운 텍스트와 시원한 파란색 포인트가 있는 밝은 터미널용 라이트 테마 | `Hermes Agent` | 흰색 또는 밝은 터미널에 맞게 설계되었습니다. 파란색 테두리의 어두운 슬레이트 텍스트, 옅은 상태 표면, 밝은 터미널 프로필에서도 읽기 쉬운 밝은 완료 메뉴. |
| `warm-lightmode` | 밝은 터미널 배경을 위한 따뜻한 갈색/골드 텍스트 | `Hermes Agent` | 밝은 터미널을 위한 따뜻한 양피지 색조. 새들 브라운 포인트가 있는 짙은 갈색 텍스트, 크림색 상태 표면. 차가운 daylight 테마의 흙빛 대안입니다. |
| `poseidon` | 바다의 신 테마 — 딥 블루와 씨폼(seafoam) | `Poseidon Agent` | 딥 블루에서 씨폼으로 이어지는 그라데이션. 바다 테마 스피너("charting currents", "sounding the depth"). 트라이던트 ASCII 아트 배너. |
| `sisyphus` | 시시포스 테마 — 끈기가 돋보이는 엄격한 흑백 | `Sisyphus Agent` | 강한 대비를 이루는 밝은 회색. 바위 테마 스피너("pushing uphill", "resetting the boulder", "enduring the loop"). 바위와 언덕 ASCII 아트 배너. |
| `charizard` | 화산 테마 — 번트 오렌지와 잉걸불 | `Charizard Agent` | 따뜻한 번트 오렌지에서 잉걸불로 이어지는 그라데이션. 불 테마 스피너("banking into the draft", "measuring burn"). 용 실루엣 ASCII 아트 배너. |

## 구성 가능한 전체 키 목록

### 색상 (`colors:`)

CLI 전체의 모든 색상 값을 제어합니다. 값은 16진수(Hex) 색상 문자열입니다.

| 키 (Key) | 설명 (Description) | 기본값 (`default` 스킨) |
|-----|-------------|--------------------------|
| `banner_border` | 시작 배너 주변의 패널 테두리 | `#CD7F32` (청동) |
| `banner_title` | 배너의 제목 텍스트 색상 | `#FFD700` (골드) |
| `banner_accent` | 배너의 섹션 헤더 (Available Tools 등) | `#FFBF00` (호박색) |
| `banner_dim` | 배너의 흐린 텍스트 (구분선, 보조 라벨) | `#B8860B` (다크 골든로드) |
| `banner_text` | 배너의 본문 텍스트 (도구 이름, 기술 이름) | `#FFF8DC` (콘실크) |
| `ui_accent` | 일반 UI 강조 색상 (하이라이트, 활성 요소) | `#FFBF00` |
| `ui_label` | UI 라벨 및 태그 | `#4dd0e1` (청록색) |
| `ui_ok` | 성공 표시기 (체크 표시, 완료) | `#4caf50` (녹색) |
| `ui_error` | 오류 표시기 (실패, 차단됨) | `#ef5350` (빨간색) |
| `ui_warn` | 경고 표시기 (주의, 승인 프롬프트) | `#ffa726` (주황색) |
| `prompt` | 대화형 프롬프트 텍스트 색상 | `#FFF8DC` |
| `input_rule` | 입력 영역 위의 수평선 | `#CD7F32` |
| `response_border` | 에이전트 응답 상자 주변의 테두리 (ANSI 이스케이프) | `#FFD700` |
| `session_label` | 세션 라벨 색상 | `#DAA520` |
| `session_border` | 세션 ID 흐린 테두리 색상 | `#8B8682` |
| `status_bar_bg` | TUI 상태 / 사용량 바의 배경 색상 | `#1a1a2e` |
| `voice_status_bg` | 음성 모드 상태 배지의 배경 색상 | `#1a1a2e` |
| `selection_bg` | TUI 마우스 선택 하이라이터의 배경 색상. 설정되지 않은 경우 `completion_menu_current_bg`로 대체됩니다. | `#333355` |
| `completion_menu_bg` | 완료 메뉴 목록의 배경 색상 | `#1a1a2e` |
| `completion_menu_current_bg` | 활성 완료 행의 배경 색상 | `#333355` |
| `completion_menu_meta_bg` | 완료 메타 열의 배경 색상 | `#1a1a2e` |
| `completion_menu_meta_current_bg` | 활성 완료 메타 열의 배경 색상 | `#333355` |

### 스피너 (`spinner:`)

API 응답을 기다리는 동안 표시되는 애니메이션 스피너를 제어합니다.

| 키 (Key) | 유형 (Type) | 설명 (Description) | 예시 (Example) |
|-----|------|-------------|---------|
| `waiting_faces` | 문자열 목록 | API 응답을 기다리는 동안 순환되는 표정 | `["(⚔)", "(⛨)", "(▲)"]` |
| `thinking_faces` | 문자열 목록 | 모델 추론 중 순환되는 표정 | `["(⚔)", "(⌁)", "(<>)"]` |
| `thinking_verbs` | 문자열 목록 | 스피너 메시지에 표시되는 동사 | `["forging", "plotting", "hammering plans"]` |
| `wings` | [왼쪽, 오른쪽] 쌍 목록 | 스피너 주변의 장식용 괄호 | `[["⟪⚔", "⚔⟫"], ["⟪▲", "▲⟫"]]` |

스피너 값이 비어 있으면(`default` 및 `mono`와 같이) `display.py`에 하드코딩된 기본값이 사용됩니다.

### 브랜드 (`branding:`)

CLI 인터페이스 전체에서 사용되는 텍스트 문자열입니다.

| 키 (Key) | 설명 (Description) | 기본값 (Default) |
|-----|-------------|---------|
| `agent_name` | 배너 제목 및 상태 표시에 표시되는 이름 | `Hermes Agent` |
| `welcome` | CLI 시작 시 표시되는 환영 메시지 | `Welcome to Hermes Agent! Type your message or /help for commands.` |
| `goodbye` | 종료 시 표시되는 메시지 | `Goodbye! ⚕` |
| `response_label` | 응답 상자 헤더의 라벨 | ` ⚕ Hermes ` |
| `prompt_symbol` | 사용자 입력 프롬프트 앞의 기호 (단일 토큰, 렌더러가 후행 공백을 추가함) | `❯` |
| `help_header` | `/help` 명령 출력의 헤더 텍스트 | `(^_^)? Available Commands` |

### 기타 최상위 키

| 키 (Key) | 유형 (Type) | 설명 (Description) | 기본값 (Default) |
|-----|------|-------------|---------|
| `tool_prefix` | 문자열 | CLI에서 도구 출력 줄 앞에 접두사로 붙는 문자 | `┊` |
| `tool_emojis` | 딕셔너리 | 스피너 및 진행률에 대한 도구별 이모지 재정의 (`{tool_name: emoji}`) | `{}` |
| `banner_logo` | 문자열 | 풍부한 마크업 ASCII 아트 로고 (기본 HERMES_AGENT 배너를 대체함) | `""` |
| `banner_hero` | 문자열 | 풍부한 마크업 히어로 아트 (기본 카두세우스 아트를 대체함) | `""` |

## 사용자 정의 스킨

`~/.hermes/skins/` 아래에 YAML 파일을 생성합니다. 사용자 스킨은 기본 `default` 스킨에서 누락된 값을 상속하므로, 변경하려는 키만 지정하면 됩니다.

### 전체 사용자 정의 스킨 YAML 템플릿

```yaml
# ~/.hermes/skins/mytheme.yaml
# 전체 스킨 템플릿 — 모든 키가 표시됩니다. 필요 없는 키는 삭제하십시오.
# 누락된 값은 'default' 스킨에서 자동으로 상속됩니다.

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

# 도구별 이모지 재정의 (선택 사항)
tool_emojis:
  terminal: "⚔"
  web_search: "🔮"
  read_file: "📄"

# 사용자 정의 ASCII 아트 배너 (선택 사항, Rich 마크업 지원)
# banner_logo: |
#   [bold #FFD700] MY AGENT [/]
# banner_hero: |
#   [#FFD700]  Custom art here  [/]
```

### 최소 사용자 정의 스킨 예시

모든 것이 `default`에서 상속되므로, 최소 스킨은 다른 점만 변경하면 됩니다:

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

[Hermes Mod](https://github.com/cocktailpeanut/hermes-mod)는 스킨을 시각적으로 생성하고 관리하기 위해 커뮤니티에서 구축한 웹 UI입니다. 수동으로 YAML을 작성하는 대신 라이브 미리보기가 포함된 포인트 앤 클릭 편집기를 얻을 수 있습니다.

![Hermes Mod 스킨 편집기](https://raw.githubusercontent.com/cocktailpeanut/hermes-mod/master/nous.png)

**기능:**

- 모든 기본 제공 및 사용자 정의 스킨 나열
- 모든 스킨을 Hermes 스킨 필드(색상, 스피너, 브랜드, 도구 접두사, 도구 이모지)가 포함된 시각적 편집기로 엽니다.
- 텍스트 프롬프트에서 `banner_logo` 텍스트 아트를 생성합니다.
- 업로드된 이미지(PNG, JPG, GIF, WEBP)를 여러 렌더링 스타일(점자, ASCII 램프, 블록, 점)이 있는 `banner_hero` ASCII 아트로 변환합니다.
- `~/.hermes/skins/`에 직접 저장
- `~/.hermes/config.yaml`을 업데이트하여 스킨 활성화
- 생성된 YAML과 라이브 미리보기 표시

### 설치

**옵션 1 — Pinokio (원클릭):**

[pinokio.computer](https://pinokio.computer)에서 찾아 한 번의 클릭으로 설치합니다.

**옵션 2 — npx (터미널에서 가장 빠름):**

```bash
npx -y hermes-mod
```

**옵션 3 — 수동:**

```bash
git clone https://github.com/cocktailpeanut/hermes-mod.git
cd hermes-mod/app
npm install
npm start
```

### 사용법

1. 앱을 시작합니다 (Pinokio 또는 터미널을 통해).
2. **Skin Studio**를 엽니다.
3. 편집할 기본 제공 또는 사용자 정의 스킨을 선택합니다.
4. 텍스트에서 로고를 생성하거나 히어로 아트용 이미지를 업로드합니다. 렌더링 스타일과 너비를 선택합니다.
5. 색상, 스피너, 브랜드 및 기타 필드를 편집합니다.
6. **Save**를 클릭하여 스킨 YAML을 `~/.hermes/skins/`에 기록합니다.
7. **Activate**를 클릭하여 현재 스킨으로 설정합니다 (`config.yaml`의 `display.skin`을 업데이트함).

Hermes Mod는 `HERMES_HOME` 환경 변수를 존중하므로 [프로필](/user-guide/profiles)과도 잘 작동합니다.

## 작동 참고 사항

- 기본 제공 스킨은 `hermes_cli/skin_engine.py`에서 로드됩니다.
- 알 수 없는 스킨은 자동으로 `default`로 대체됩니다.
- `/skin`은 현재 세션에 대해 즉시 활성 CLI 테마를 업데이트합니다.
- `~/.hermes/skins/`의 사용자 스킨은 같은 이름의 기본 제공 스킨보다 우선순위가 높습니다.
- `/skin`을 통한 스킨 변경은 세션 전용입니다. 스킨을 영구 기본값으로 만들려면 `config.yaml`에서 설정하십시오.
- `banner_logo` 및 `banner_hero` 필드는 컬러 ASCII 아트를 위해 Rich 콘솔 마크업(예: `[bold #FF0000]text[/]`)을 지원합니다.
