---
title: "밈 생성 (Meme Generation) — 템플릿을 선택하고 Pillow로 텍스트를 오버레이하여 실제 밈 이미지 생성"
sidebar_label: "밈 생성 (Meme Generation)"
description: "템플릿을 선택하고 Pillow로 텍스트를 오버레이하여 실제 밈 이미지 생성"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# 밈 생성 (Meme Generation)

템플릿을 선택하고 Pillow로 텍스트를 오버레이하여 실제 밈 이미지를 생성합니다. 실제 .png 밈 파일을 생성합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/meme-generation` 명령어로 설치 |
| 경로 | `optional-skills/creative/meme-generation` |
| 버전 | `2.0.0` |
| 작성자 | adanaleycio |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `creative`, `memes`, `humor`, `images` |
| 관련 스킬 | [`ascii-art`](/docs/user-guide/skills/bundled/creative/creative-ascii-art), `generative-widgets` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# 밈 생성 (Meme Generation)

주제에서 실제 밈 이미지를 생성합니다. 템플릿을 선택하고, 캡션을 작성하고, 텍스트가 오버레이된 실제 .png 파일을 렌더링합니다.

## 사용 시기

- 사용자가 밈을 만들거나 생성해달라고 요청할 때
- 사용자가 특정 주제, 상황, 또는 불만에 대한 밈을 원할 때
- 사용자가 "이걸 밈으로 만들어줘" 또는 이와 유사한 말을 할 때

## 사용 가능한 템플릿

이 스크립트는 이름이나 ID를 통해 **약 100개의 인기 있는 imgflip 템플릿 중 하나**를 지원하며, 수동으로 텍스트 위치를 조정한 10개의 엄선된 템플릿도 추가로 지원합니다.

### 엄선된 템플릿 (사용자 지정 텍스트 배치)

| ID | 이름 | 필드 | 최적의 용도 |
|----|------|--------|----------|
| `this-is-fine` | This is Fine | top, bottom | 혼돈, 현실 부정 |
| `drake` | Drake Hotline Bling | reject, approve | 거절/선호 |
| `distracted-boyfriend` | Distracted Boyfriend | distraction, current, person | 유혹, 우선순위 변경 |
| `two-buttons` | Two Buttons | left, right, person | 불가능한 선택 |
| `expanding-brain` | Expanding Brain | 4 levels | 커지는 아이러니 |
| `change-my-mind` | Change My Mind | statement | 논란의 여지가 있는 의견 (Hot takes) |
| `woman-yelling-at-cat` | Woman Yelling at Cat | woman, cat | 말다툼 |
| `one-does-not-simply` | One Does Not Simply | top, bottom | 겉보기에만 쉬운 일 |
| `grus-plan` | Gru's Plan | step1-3, realization | 역효과를 낳는 계획 |
| `batman-slapping-robin` | Batman Slapping Robin | robin, batman | 나쁜 아이디어 차단 |

### 동적 템플릿 (imgflip API 사용)

엄선된 목록에 없는 템플릿은 이름이나 imgflip ID로 사용할 수 있습니다. 이들은 스마트 기본 텍스트 위치를 사용합니다 (2필드의 경우 상단/하단, 3개 이상인 경우 균등 간격). 다음 명령어로 검색하세요:
```bash
python "$SKILL_DIR/scripts/generate_meme.py" --search "disaster"
```

## 절차

### 모드 1: 클래식 템플릿 (기본값)

1. 사용자의 주제를 읽고 핵심 역학(혼돈, 딜레마, 선호, 아이러니 등)을 파악합니다.
2. 가장 잘 맞는 템플릿을 선택합니다. "최적의 용도" 열을 사용하거나 `--search`로 검색합니다.
3. 각 필드에 대한 짧은 캡션을 작성합니다 (필드당 최대 8-12단어, 짧을수록 좋습니다).
4. 스킬의 스크립트 디렉토리를 찾습니다:
   ```
   SKILL_DIR=$(dirname "$(find ~/.hermes/skills -path '*/meme-generation/SKILL.md' 2>/dev/null | head -1)")
   ```
5. 생성기를 실행합니다:
   ```bash
   python "$SKILL_DIR/scripts/generate_meme.py" <template_id> /tmp/meme.png "caption 1" "caption 2" ...
   ```
6. `MEDIA:/tmp/meme.png`를 사용하여 이미지를 반환합니다.

### 모드 2: 사용자 지정 AI 이미지 (`image_generate` 사용 가능 시)

클래식 템플릿이 맞지 않거나, 사용자가 독창적인 것을 원할 때 이 모드를 사용합니다.

1. 먼저 캡션을 작성합니다.
2. `image_generate`를 사용하여 밈 컨셉과 일치하는 장면을 만듭니다. 이미지 프롬프트에 텍스트를 포함하지 마십시오 — 텍스트는 스크립트에 의해 추가됩니다. 시각적 장면만 설명하세요.
3. image_generate 결과 URL에서 생성된 이미지 경로를 찾습니다. 필요한 경우 로컬 경로로 다운로드합니다.
4. `--image`와 함께 스크립트를 실행하여 텍스트를 오버레이할 모드를 선택합니다:
   - **오버레이 (Overlay)** (이미지 위에 텍스트를 직접 표시, 검은색 윤곽선이 있는 흰색 텍스트):
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png /tmp/meme.png "top text" "bottom text"
     ```
   - **막대 (Bars)** (텍스트를 읽기 쉽도록 위/아래에 검은색 막대와 흰색 텍스트 추가):
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png --bars /tmp/meme.png "top text" "bottom text"
     ```
   이미지가 복잡/자세해서 텍스트를 그 위에 올리면 읽기 어려울 때 `--bars`를 사용합니다.
5. **비전으로 확인** (`vision_analyze` 사용 가능 시): 결과가 좋아 보이는지 확인합니다:
   ```
   vision_analyze(image_url="/tmp/meme.png", question="Is the text legible and well-positioned? Does the meme work visually?")
   ```
   비전 모델이 문제(텍스트 읽기 어려움, 잘못된 배치 등)를 발견하면 다른 모드(오버레이와 막대 사이 전환)를 시도하거나 장면을 다시 생성합니다.
6. `MEDIA:/tmp/meme.png`를 사용하여 이미지를 반환합니다.

## 예시

**"새벽 2시에 프로덕션 디버깅하기":**
```bash
python generate_meme.py this-is-fine /tmp/meme.png "SERVERS ARE ON FIRE" "This is fine"
```

**"수면과 한 에피소드 더 보기 사이에서 선택하기":**
```bash
python generate_meme.py drake /tmp/meme.png "Getting 8 hours of sleep" "One more episode at 3 AM"
```

**"월요일 아침의 단계들":**
```bash
python generate_meme.py expanding-brain /tmp/meme.png "Setting an alarm" "Setting 5 alarms" "Sleeping through all alarms" "Working from bed"
```

## 템플릿 나열

사용 가능한 모든 템플릿을 보려면:
```bash
python generate_meme.py --list
```

## 주의 사항 (Pitfalls)

- 캡션은 짧게 유지하세요. 텍스트가 긴 밈은 보기에 좋지 않습니다.
- 텍스트 인수 개수를 템플릿의 필드 수와 일치시키세요.
- 단순히 주제에 맞는 것이 아니라, 농담의 구조에 맞는 템플릿을 선택하세요.
- 혐오스럽거나, 모욕적이거나, 개인을 표적으로 삼는 콘텐츠를 생성하지 마세요.
- 스크립트는 첫 번째 다운로드 후 템플릿 이미지를 `scripts/.cache/`에 캐시합니다.

## 검증 (Verification)

다음과 같은 경우 결과가 올바른 것입니다:
- 출력 경로에 .png 파일이 생성되었습니다.
- 텍스트가 템플릿에서 읽기 쉽습니다(검은색 윤곽선이 있는 흰색).
- 농담이 성공적입니다 — 캡션이 템플릿의 의도된 구조와 일치합니다.
- 파일을 MEDIA: 경로를 통해 전달할 수 있습니다.
