---
title: "밈 생성 — Pillow 텍스트 오버레이로 템플릿에서 밈 PNG 만들기"
sidebar_label: "밈 생성"
description: "Pillow 텍스트 오버레이로 템플릿에서 밈 PNG 만들기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 skill의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 수정하세요. */}

# 밈 생성

Pillow 텍스트 오버레이로 템플릿에서 밈 PNG를 만듭니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/meme-generation`으로 설치 |
| 경로 | `optional-skills/creative/meme-generation` |
| 버전 | `2.0.0` |
| 작성자 | adanaleycio |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `creative`, `memes`, `humor`, `images` |
| 관련 스킬 | [`ascii-art`](/docs/user-guide/skills/bundled/creative/creative-ascii-art) |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 스킬 정의 전문입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# 밈 생성

주제에서 실제 밈 이미지를 생성합니다. 템플릿을 고르고, 캡션을 작성하고, 텍스트 오버레이가 적용된 실제 `.png` 파일을 렌더링합니다.

## 사용 시점

- 사용자가 밈을 만들어 달라고 하거나 생성해 달라고 요청할 때
- 사용자가 특정 주제, 상황 또는 불만에 관한 밈을 원할 때
- 사용자가 "meme this" 또는 이와 비슷한 표현을 사용할 때

## 사용 가능한 템플릿

스크립트는 이름 또는 ID로 **약 100개의 인기 imgflip 템플릿**을 모두 지원하며, 텍스트 위치를 수작업으로 조정한 엄선 템플릿 10개도 지원합니다.

### 엄선 템플릿 (사용자 지정 텍스트 배치)

| ID | 이름 | 필드 | 적합한 용도 |
|----|------|------|----------|
| `this-is-fine` | This is Fine | top, bottom | 혼란, 부정 |
| `drake` | Drake Hotline Bling | reject, approve | 거부/선호 |
| `distracted-boyfriend` | Distracted Boyfriend | distraction, current, person | 유혹, 우선순위 변경 |
| `two-buttons` | Two Buttons | left, right, person | 불가능한 선택 |
| `expanding-brain` | Expanding Brain | 4 levels | 점점 커지는 아이러니 |
| `change-my-mind` | Change My Mind | statement | 논쟁적인 주장 |
| `woman-yelling-at-cat` | Woman Yelling at Cat | woman, cat | 논쟁 |
| `one-does-not-simply` | One Does Not Simply | top, bottom | 겉보기보다 어려운 일 |
| `grus-plan` | Gru's Plan | step1-3, realization | 실패로 돌아가는 계획 |
| `batman-slapping-robin` | Batman Slapping Robin | robin, batman | 나쁜 아이디어 차단 |

### 동적 템플릿 (imgflip API에서 가져옴)

엄선 목록에 없는 템플릿도 이름 또는 imgflip ID로 사용할 수 있습니다. 이 템플릿에는 스마트 기본 텍스트 배치가 적용됩니다(필드 2개면 위/아래, 3개 이상이면 균등 배치). 다음 명령으로 검색하세요.
```bash
python "$SKILL_DIR/scripts/generate_meme.py" --search "disaster"
```

## 절차

### 모드 1: 클래식 템플릿 (기본값)

1. 사용자의 주제를 읽고 핵심 구도(혼란, 딜레마, 선호, 아이러니 등)를 파악합니다.
2. 가장 잘 맞는 템플릿을 고릅니다. `--search`와 함께 "적합한 용도" 열을 사용하거나 검색하세요.
3. 각 필드에 짧은 캡션을 작성합니다(필드당 최대 8~12단어, 짧을수록 좋습니다).
4. 스킬의 스크립트 디렉터리를 찾습니다.
   ```
   SKILL_DIR=$(dirname "$(find ~/.hermes/skills -path '*/meme-generation/SKILL.md' 2>/dev/null | head -1)")
   ```
5. 생성기를 실행합니다.
   ```bash
   python "$SKILL_DIR/scripts/generate_meme.py" <template_id> /tmp/meme.png "caption 1" "caption 2" ...
   ```
6. `MEDIA:/tmp/meme.png`로 이미지를 반환합니다.

### 모드 2: 사용자 지정 AI 이미지 (`image_generate`를 사용할 수 있을 때)

클래식 템플릿에 맞는 것이 없거나 독창적인 것을 원할 때 사용하세요.

1. 먼저 캡션을 작성합니다.
2. `image_generate`를 사용해 밈 콘셉트에 맞는 장면을 만듭니다. 이미지 프롬프트에는 텍스트를 절대 포함하지 마세요 — 텍스트는 스크립트가 추가합니다. 시각적 장면만 설명하세요.
3. `image_generate` 결과 URL에서 생성된 이미지 경로를 찾습니다. 필요하면 로컬 경로로 다운로드합니다.
4. `--image`로 스크립트를 실행하고 모드를 선택해 텍스트를 오버레이합니다.
   - **오버레이** (이미지에 텍스트를 직접 표시, 검은색 테두리가 있는 흰색 텍스트):
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png /tmp/meme.png "top text" "bottom text"
     ```
   - **막대** (위/아래에 흰색 텍스트가 있는 검은 막대 — 더 깔끔하고 항상 읽기 쉬움):
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png --bars /tmp/meme.png "top text" "bottom text"
     ```
   이미지가 복잡하거나 세부 묘사가 많아 위에 텍스트를 표시하기 어려울 때는 `--bars`를 사용하세요.
5. **비전으로 확인합니다** (`vision_analyze`를 사용할 수 있는 경우): 결과가 보기 좋은지 확인합니다.
   ```
   vision_analyze(image_url="/tmp/meme.png", question="Is the text legible and well-positioned? Does the meme work visually?")
   ```
   비전 모델이 문제(읽기 어려운 텍스트, 나쁜 배치 등)를 지적하면 다른 모드로 전환하거나(오버레이와 막대 사이 전환) 장면을 다시 생성해 보세요.
6. `MEDIA:/tmp/meme.png`로 이미지를 반환합니다.

## 예시

**"새벽 2시에 프로덕션 디버깅":**
```bash
python generate_meme.py this-is-fine /tmp/meme.png "SERVERS ARE ON FIRE" "This is fine"
```

**"잠과 에피소드 하나 더 보기 중에서 선택하기":**
```bash
python generate_meme.py drake /tmp/meme.png "Getting 8 hours of sleep" "One more episode at 3 AM"
```

**"월요일 아침의 단계":**
```bash
python generate_meme.py expanding-brain /tmp/meme.png "Setting an alarm" "Setting 5 alarms" "Sleeping through all alarms" "Working from bed"
```

## 템플릿 목록 보기

사용 가능한 모든 템플릿을 보려면 다음을 실행하세요.
```bash
python generate_meme.py --list
```

## 주의할 점

- 캡션은 짧게 유지하세요. 텍스트가 길면 밈이 보기 좋지 않습니다.
- 텍스트 인수의 개수를 템플릿의 필드 개수에 맞추세요.
- 단순히 주제가 아니라 농담의 구조에 맞는 템플릿을 고르세요.
- 혐오성, 모욕적 또는 특정 개인을 겨냥한 콘텐츠를 생성하지 마세요.
- 첫 다운로드 후 스크립트는 템플릿 이미지를 `scripts/.cache/`에 캐시합니다.

## 확인

다음 조건을 만족하면 출력이 올바른 것입니다.
- 출력 경로에 `.png` 파일이 생성되었습니다.
- 템플릿 위에서 텍스트(검은색 테두리가 있는 흰색)가 읽기 쉽습니다.
- 농담이 잘 전달됩니다 — 캡션이 템플릿의 의도된 구조에 맞습니다.
- 파일을 `MEDIA:` 경로로 전달할 수 있습니다.
