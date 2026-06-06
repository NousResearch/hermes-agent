---
title: "Memento Flashcards — 간격 반복 플래시카드 시스템"
sidebar_label: "Memento Flashcards"
description: "간격 반복 플래시카드 시스템"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Memento Flashcards

간격 반복(Spaced-repetition) 플래시카드 시스템입니다. 사실이나 텍스트에서 카드를 생성하고, 에이전트가 채점하는 자유 텍스트(free-text) 답변을 사용하여 플래시카드와 대화할 수 있으며, YouTube 자막에서 퀴즈를 생성하고, 적응형 일정으로 복습 기한이 된 카드를 학습하고, 덱(deck)을 CSV로 내보내기/가져오기 할 수 있습니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택적(Optional) — `hermes skills install official/productivity/memento-flashcards` 명령어로 설치 |
| 경로 | `optional-skills/productivity/memento-flashcards` |
| 버전 | `1.0.0` |
| 작성자 | Memento AI |
| 라이선스 | MIT |
| 플랫폼 | macos, linux |
| 태그 | `Education`, `Flashcards`, `Spaced Repetition`, `Learning`, `Quiz`, `YouTube` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되어 있을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Memento Flashcards — 간격 반복 플래시카드 스킬

## 개요

Memento는 간격 반복 스케줄링 기능을 갖춘 로컬 파일 기반의 플래시카드 시스템을 제공합니다.
사용자는 자유 텍스트로 답을 말하면 에이전트가 응답을 채점하고 다음 복습을 예약하는 방식으로 플래시카드와 대화하며 학습할 수 있습니다.
사용자가 다음과 같은 작업을 원할 때 이 스킬을 사용하세요:

- **사실 기억하기** — 어떤 진술이든 문답(Q/A) 형식의 플래시카드로 만들기
- **간격 반복으로 학습하기** — 에이전트가 채점하는 자유 텍스트 답변과 적응형 주기를 통해 복습 기한이 된 카드 복습하기
- **YouTube 비디오에서 퀴즈 풀기** — 자막을 가져와 5개의 질문으로 구성된 퀴즈 생성하기
- **덱(Deck) 관리하기** — 카드를 컬렉션으로 구성, CSV 내보내기/가져오기

모든 카드 데이터는 단일 JSON 파일에 저장됩니다. 외부 API 키가 필요하지 않습니다 — 에이전트인 당신이 직접 플래시카드 콘텐츠와 퀴즈 질문을 생성합니다.

Memento Flashcards에 대한 사용자 대면 응답 스타일:
- 일반 텍스트(plain text)만 사용합니다. 사용자에게 응답할 때 마크다운 포맷팅을 사용하지 마세요.
- 복습 및 퀴즈 피드백은 짧고 중립적으로 유지합니다. 과도한 칭찬, 격려 또는 긴 설명은 피하세요.

## 사용 시기

사용자가 다음을 원할 때 이 스킬을 사용하세요:
- 나중에 복습하기 위해 사실을 플래시카드로 저장
- 간격 반복 시스템을 통해 복습 기한이 된 카드 복습
- YouTube 비디오 자막에서 퀴즈 생성
- 플래시카드 데이터를 가져오기, 내보내기, 확인, 삭제

일반적인 질의응답(Q&A), 코딩 도움말 또는 암기와 관련 없는 작업에는 이 스킬을 사용하지 마세요.

## 빠른 참조

| 사용자 의도 | 액션 |
|---|---|
| "X를 기억해줘" / "이걸 플래시카드로 저장해줘" | Q/A 카드를 생성하고, `memento_cards.py add` 호출 |
| 플래시카드를 언급하지 않고 사실(fact)을 보낼 때 | "이 내용을 Memento 플래시카드로 저장할까요?"라고 묻기 — 승인 시에만 생성 |
| "플래시카드를 만들어줘" | 질문(Q), 답변(A), 컬렉션을 물어보고, `memento_cards.py add` 호출 |
| "내 카드 복습할래" | `memento_cards.py due` 호출, 카드를 하나씩 제시 |
| "[YouTube URL] 퀴즈 내줘" | `youtube_quiz.py fetch VIDEO_ID` 호출, 질문 5개 생성 후 `memento_cards.py add-quiz` 호출 |
| "내 카드 내보내기 해줘" | `memento_cards.py export --output PATH` 호출 |
| "CSV에서 카드 가져와줘" | `memento_cards.py import --file PATH --collection NAME` 호출 |
| "내 통계 보여줘" | `memento_cards.py stats` 호출 |
| "카드 지워줘" | `memento_cards.py delete --id ID` 호출 |
| "컬렉션 지워줘" | `memento_cards.py delete-collection --collection NAME` 호출 |

## 카드 저장소

카드는 다음 경로의 JSON 파일에 저장됩니다:

```
~/.hermes/skills/productivity/memento-flashcards/data/cards.json
```

**이 파일을 절대 직접 편집하지 마세요.** 항상 `memento_cards.py` 하위 명령어를 사용하세요. 스크립트는 손상을 방지하기 위해 원자적 쓰기(임시 파일에 쓴 후 이름 변경)를 처리합니다.

파일은 처음 사용할 때 자동으로 생성됩니다.

## 절차

### 사실(Facts)에서 카드 생성

### 활성화 규칙

모든 사실적 진술이 플래시카드가 되어야 하는 것은 아닙니다. 다음의 3단계 확인을 사용하세요:

1. **명시적 의도** — 사용자가 "memento", "flashcard(플래시카드)", "remember this(이거 기억해)", "save this card(이 카드 저장해)", "add a card(카드 추가해)" 등 플래시카드를 명확히 요청하는 표현을 사용할 때 → 확인 없이 **바로 카드를 생성합니다**.
2. **암시적 의도** — 사용자가 플래시카드 언급 없이 사실적인 진술을 보낼 때 (예: "빛의 속도는 299,792 km/s야") → **먼저 물어봅니다**: "이 내용을 Memento 플래시카드로 저장할까요?" 사용자가 동의할 때만 카드를 생성합니다.
3. **의도 없음** — 메시지가 코딩 작업, 질문, 지시, 일반적인 대화이거나 암기할 사실이 명백히 아닐 때 → **이 스킬을 전혀 활성화하지 마세요**. 다른 스킬이나 기본 동작이 처리하도록 둡니다.

활성화가 확정되면 (1단계는 즉시, 2단계는 승인 후), 플래시카드를 생성합니다:

**1단계:** 진술을 Q/A 쌍으로 바꿉니다. 내부적으로 다음 형식을 사용하세요:

```
Turn the factual statement into a front-back pair.
Return exactly two lines:
Q: <question text>
A: <answer text>

Statement: "{statement}"
```

규칙:
- 질문은 핵심 사실에 대한 기억을 테스트해야 합니다.
- 답변은 간결하고 직접적이어야 합니다.

**2단계:** 스크립트를 호출하여 카드를 저장합니다:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add \
  --question "What year did World War 2 end?" \
  --answer "1945" \
  --collection "History"
```

사용자가 컬렉션을 지정하지 않으면 `"General"`을 기본값으로 사용합니다.

스크립트는 생성된 카드를 확인하는 JSON을 출력합니다.

### 수동 카드 생성

사용자가 명시적으로 플래시카드 생성을 요청할 때, 다음을 물어봅니다:
1. 질문 (카드 앞면)
2. 답변 (카드 뒷면)
3. 컬렉션 이름 (선택 사항 — 기본값은 `"General"`)

그런 다음 위와 같이 `memento_cards.py add`를 호출합니다.

### 복습 기한이 된 카드 복습하기

사용자가 복습을 원할 때, 기한이 된 모든 카드를 가져옵니다:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

이는 `next_review_at <= now`인 카드의 JSON 배열을 반환합니다. 컬렉션 필터가 필요한 경우:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due --collection "History"
```

**복습 흐름 (자유 텍스트 채점):**

다음은 당신이 따라야 할 정확한 상호작용 패턴의 예시입니다. 사용자가 대답하면, 당신이 그것을 채점하고 정답을 알려준 뒤, 카드의 등급(rate)을 매깁니다.

**상호작용 예시:**

> **에이전트:** 베를린 장벽이 무너진 해는 언제입니까?
>
> **사용자:** 1991
>
> **에이전트:** 약간 다릅니다. 정답: 1989. 다음 복습은 내일입니다.
> *(에이전트가 `memento_cards.py rate --id ABC --rating hard --user-answer "1991"`를 호출함)*
>
> 다음 질문: 달에 처음으로 걸은 사람은 누구입니까?

**규칙:**

1. 질문만 보여줍니다. 사용자가 대답할 때까지 기다리세요.
2. 사용자의 대답을 받은 후, 예상 정답과 비교하여 채점합니다:
   - **correct** → 사용자 핵심 사실을 맞춤 (표현이 다르더라도)
   - **partial** → 방향은 맞으나 핵심 세부 사항이 누락됨
   - **incorrect** → 틀렸거나 주제에서 벗어남
3. **사용자에게 반드시 정답과 그들이 어떻게 했는지 알려주어야 합니다.** 짧고 일반 텍스트(plain-text)로 유지하세요. 다음 형식을 사용합니다:
   - correct: "정답입니다. 정답: &#123;answer&#125;. 7일 후에 다음 복습을 진행합니다."
   - partial: "거의 맞았습니다. 정답: &#123;answer&#125;. &#123;놓친 부분&#125;. 3일 후에 다음 복습을 진행합니다."
   - incorrect: "약간 다릅니다. 정답: &#123;answer&#125;. 내일 다음 복습을 진행합니다."
4. 그런 다음 rate 명령을 호출합니다: correct→easy, partial→good, incorrect→hard.
5. 그런 다음 다음 질문을 보여줍니다.

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```

**3단계를 절대로 건너뛰지 마세요.** 당신이 넘어가기 전에 사용자는 항상 정답과 피드백을 보아야 합니다.

복습 기한이 된 카드가 없다면 사용자에게 알리세요: "지금은 복습할 카드가 없습니다. 나중에 다시 확인해 주세요!"

**카드 은퇴(Retire) 오버라이드:** 언제든지 사용자가 "이 카드 그만할래(retire this card)"라고 말하면 복습에서 영구적으로 제외할 수 있습니다. 이를 위해 `--rating retire`를 사용합니다.

### 간격 반복 알고리즘 (Spaced Repetition Algorithm)

매긴 등급이 다음 복습 간격을 결정합니다:

| 등급 (Rating) | 간격 | ease_streak (연속 정답) | 상태 변화 |
|---|---|---|---|
| **hard** | +1일 | 0으로 리셋 | learning 유지 |
| **good** | +3일 | 0으로 리셋 | learning 유지 |
| **easy** | +7일 | +1 | ease_streak >= 3 인 경우 → retired |
| **retire** | 영구적 | 0으로 리셋 | → retired |

- **learning (학습 중)**: 카드가 로테이션에 활성화되어 있습니다.
- **retired (은퇴함)**: 카드가 복습에 나타나지 않습니다. (사용자가 마스터했거나 수동으로 제외함)
- "easy" 등급을 3번 연속으로 받으면 카드가 자동으로 은퇴(retire) 처리됩니다.

### YouTube 퀴즈 생성

사용자가 YouTube URL을 보내고 퀴즈를 원할 때:

**1단계:** URL에서 비디오 ID를 추출합니다 (예: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`에서 `dQw4w9WgXcQ`).

**2단계:** 자막을 가져옵니다:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/youtube_quiz.py fetch VIDEO_ID
```

이는 `{"title": "...", "transcript": "..."}` 또는 오류를 반환합니다.

스크립트가 `missing_dependency`를 보고하면, 사용자에게 다음을 설치하도록 안내합니다:
```bash
pip install youtube-transcript-api
```

**3단계:** 자막에서 5개의 퀴즈 질문을 생성합니다. 다음 규칙을 사용하세요:

```
You are creating a 5-question quiz for a podcast episode.
Return ONLY a JSON array with exactly 5 objects.
Each object must contain keys 'question' and 'answer'.

Selection criteria:
- Prioritize important, surprising, or foundational facts.
- Skip filler, obvious details, and facts that require heavy context.
- Never return true/false questions.
- Never ask only for a date.

Question rules:
- Each question must test exactly one discrete fact.
- Use clear, unambiguous wording.
- Prefer What, Who, How many, Which.
- Avoid open-ended Describe or Explain prompts.

Answer rules:
- Each answer must be under 240 characters.
- Lead with the answer itself, not preamble.
- Add only minimal clarifying detail if needed.
```

자막의 처음 15,000자를 컨텍스트로 사용합니다. (LLM인 당신이 직접 질문을 생성합니다.)

**4단계:** 출력이 각각 비어있지 않은 `question`과 `answer` 문자열을 가지는 정확히 5개의 항목으로 구성된 유효한 JSON인지 검증합니다. 검증에 실패하면 한 번 재시도합니다.

**5단계:** 퀴즈 카드를 저장합니다:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add-quiz \
  --video-id "VIDEO_ID" \
  --questions '[{"question":"...","answer":"..."},...]' \
  --collection "Quiz - Episode Title"
```

스크립트는 `video_id`를 통해 중복을 제거합니다 — 해당 비디오에 대한 카드가 이미 존재하는 경우 생성 작업을 건너뛰고 기존 카드를 보고합니다.

**6단계:** 자유 텍스트 채점 흐름과 동일하게 질문을 한 번에 하나씩 제시합니다:
1. "질문 1/5: ..."를 보여주고 사용자의 대답을 기다립니다. 절대 정답이나 정답을 암시하는 힌트를 포함하지 마세요.
2. 사용자가 자신의 언어로 대답할 때까지 기다립니다.
3. 채점 프롬프트를 사용하여 그들의 대답을 채점합니다 ("복습 기한이 된 카드 복습하기" 섹션 참조).
4. **중요: 다른 작업을 수행하기 전에 반드시 피드백과 함께 사용자에게 응답해야 합니다.** 등급, 정답, 그리고 카드의 다음 복습 기한을 보여주세요. 아무 말 없이 다음 질문으로 넘어가지 마세요. 짧고 일반 텍스트로 유지하세요. 예: "약간 다릅니다. 정답: &#123;answer&#125;. 내일 다음 복습을 진행합니다."
5. **피드백을 보여준 후**, rate 명령을 호출하고 동일한 메시지에서 다음 질문을 보여줍니다:
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```
6. 반복합니다. 다음 질문 전에 모든 대답은 반드시 시각적인 피드백을 받아야 합니다.

### CSV 내보내기/가져오기

**내보내기 (Export):**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py export \
  --output ~/flashcards.csv
```

3개 열(column)의 CSV 파일을 생성합니다: `question,answer,collection` (헤더 행 없음).

**가져오기 (Import):**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py import \
  --file ~/flashcards.csv \
  --collection "Imported"
```

question, answer, 그리고 선택적으로 collection(3번째 열)을 포함하는 CSV를 읽습니다. 컬렉션 열이 누락된 경우 `--collection` 인수를 사용합니다.

### 통계 (Statistics)

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
```

다음을 포함하는 JSON을 반환합니다:
- `total`: 전체 카드 수
- `learning`: 활성화된 학습 로테이션에 있는 카드
- `retired`: 마스터한 카드
- `due_now`: 지금 복습 기한이 된 카드
- `collections`: 컬렉션 이름별 분석

## 주의 사항 (Pitfalls)

- **`cards.json`을 직접 편집하지 마세요** — 손상을 방지하기 위해 항상 스크립트의 하위 명령어를 사용하세요.
- **자막 실패** — 일부 YouTube 비디오에는 영어 자막이 없거나 자막 기능이 비활성화되어 있습니다. 사용자에게 알리고 다른 비디오를 제안하세요.
- **선택적 종속성** — `youtube_quiz.py`는 `youtube-transcript-api`가 필요합니다. 누락된 경우 사용자에게 `pip install youtube-transcript-api`를 실행하라고 안내하세요.
- **대규모 가져오기** — 수천 행의 CSV 가져오기는 잘 작동하지만 JSON 출력이 매우 길어질 수 있습니다. 사용자에게 결과를 요약하여 알려주세요.
- **비디오 ID 추출** — `youtube.com/watch?v=ID`와 `youtu.be/ID` URL 형식을 모두 지원해야 합니다.

## 검증 (Verification)

도우미(helper) 스크립트를 직접 검증합니다:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add --question "Capital of France?" --answer "Paris" --collection "General"
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

저장소 체크아웃(repo checkout) 환경에서 테스트하는 경우 다음을 실행하세요:

```bash
pytest tests/skills/test_memento_cards.py tests/skills/test_youtube_quiz.py -q
```

에이전트 수준 검증:
- 복습을 시작하고 피드백이 일반 텍스트인지, 짧은지, 다음 카드 전에 항상 정답을 포함하는지 확인합니다.
- YouTube 퀴즈 흐름을 실행하고 다음 질문 전에 각 대답이 시각적인 피드백을 받는지 확인합니다.
