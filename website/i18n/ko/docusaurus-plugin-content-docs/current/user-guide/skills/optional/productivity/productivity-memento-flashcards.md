---
title: "Memento Flashcards — 간격 반복 플래시카드: 생성, 복습, 퀴즈, 내보내기"
sidebar_label: "Memento Flashcards"
description: "간격 반복 플래시카드: 생성, 복습, 퀴즈, 내보내기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Memento Flashcards

간격 반복 플래시카드: 생성, 복습, 퀴즈, 내보내기.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/productivity/memento-flashcards`로 설치 |
| 경로 | `optional-skills/productivity/memento-flashcards` |
| 버전 | `1.0.0` |
| 작성자 | Memento AI |
| 라이선스 | MIT |
| 플랫폼 | macos, linux |
| 태그 | `Education`, `Flashcards`, `Spaced Repetition`, `Learning`, `Quiz`, `YouTube` |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 활성화될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Memento Flashcards — 간격 반복 플래시카드 스킬

## 개요

Memento는 간격 반복 일정 관리 기능을 갖춘 로컬 파일 기반 플래시카드 시스템입니다.
사용자는 플래시카드에 자유 형식 텍스트로 답하고, 에이전트가 답변을 채점한 뒤 다음 복습 일정을 정하도록 할 수 있습니다.
다음과 같은 경우에 사용하세요.

- **사실 기억하기** — 어떤 진술이든 Q/A 플래시카드로 변환
- **간격 반복으로 공부하기** — 적응형 간격과 에이전트가 채점하는 자유 형식 답변으로 복습할 카드 확인
- **YouTube 동영상으로 퀴즈 풀기** — 자막을 가져와 5문제 퀴즈 생성
- **덱 관리하기** — 카드를 컬렉션으로 정리하고 CSV로 내보내기/가져오기

모든 카드 데이터는 하나의 JSON 파일에 저장됩니다. 외부 API 키는 필요하지 않습니다. 플래시카드 내용과 퀴즈 질문은 에이전트가 직접 생성합니다.

Memento Flashcards의 사용자 응답 스타일:
- 일반 텍스트만 사용하세요. 사용자에게 답할 때 Markdown 서식을 사용하지 마세요.
- 복습 및 퀴즈 피드백은 짧고 중립적으로 유지하세요. 과도한 칭찬, 격려, 긴 설명은 피하세요.

## 사용 시점

다음과 같은 경우 이 스킬을 사용하세요.
- 나중에 복습할 수 있도록 사실을 플래시카드로 저장하려는 경우
- 간격 반복으로 복습 기한이 된 카드를 복습하려는 경우
- YouTube 동영상 자막으로 퀴즈를 생성하려는 경우
- 플래시카드 데이터를 가져오거나, 내보내거나, 확인하거나, 삭제하려는 경우

일반적인 Q&A, 코딩 도움, 또는 기억과 관련 없는 작업에는 이 스킬을 사용하지 마세요.

## 빠른 참조

| 사용자 의도 | 작업 |
|---|---|
| "Remember that X" / "save this as a flashcard" | Q/A 카드를 생성하고 `memento_cards.py add` 호출 |
| 플래시카드를 언급하지 않고 사실을 보냄 | "Want me to save this as a Memento flashcard?"라고 질문 — 확인된 경우에만 생성 |
| "Create a flashcard" | 질문, 답변, 컬렉션을 묻고 `memento_cards.py add` 호출 |
| "Review my cards" | `memento_cards.py due` 호출 후 카드를 하나씩 제시 |
| "Quiz me on [YouTube URL]" | `youtube_quiz.py fetch VIDEO_ID` 호출, 5개 질문 생성, `memento_cards.py add-quiz` 호출 |
| "Export my cards" | `memento_cards.py export --output PATH` 호출 |
| "Import cards from CSV" | `memento_cards.py import --file PATH --collection NAME` 호출 |
| "Show my stats" | `memento_cards.py stats` 호출 |
| "Delete a card" | `memento_cards.py delete --id ID` 호출 |
| "Delete a collection" | `memento_cards.py delete-collection --collection NAME` 호출 |

## 카드 저장

카드는 다음 JSON 파일에 저장됩니다.

```
~/.hermes/skills/productivity/memento-flashcards/data/cards.json
```

**이 파일을 직접 편집하지 마세요.** 항상 `memento_cards.py` 하위 명령을 사용하세요. 스크립트는 손상을 방지하기 위해 원자적 쓰기(임시 파일에 쓴 후 이름 변경)를 처리합니다.

파일은 처음 사용할 때 자동으로 생성됩니다.

## 절차

### 사실에서 카드 생성

### 활성화 규칙

모든 사실 진술을 플래시카드로 만들 필요는 없습니다. 다음 3단계 확인을 사용하세요.

1. **명시적 의도** — 사용자가 "memento", "flashcard", "remember this", "save this card", "add a card" 또는 플래시카드를 명확히 요청하는 유사 표현을 사용함 → **확인 없이 바로 카드 생성**
2. **암시적 의도** — 사용자가 플래시카드를 언급하지 않고 사실 진술을 보냄(예: "The speed of light is 299,792 km/s") → 먼저 **"Want me to save this as a Memento flashcard?"라고 질문**: 사용자가 확인한 경우에만 카드 생성
3. **의도 없음** — 코딩 작업, 질문, 지시, 일반적인 대화 또는 외울 사실이 아닌 것이 명확한 모든 메시지 → **이 스킬을 활성화하지 마세요.** 다른 스킬이나 기본 동작이 처리하도록 하세요.

활성화가 확인되면(1단계는 즉시, 2단계는 확인 후) 플래시카드를 생성하세요.

**1단계:** 진술을 Q/A 쌍으로 변환합니다. 내부적으로 다음 형식을 사용하세요.

```
Turn the factual statement into a front-back pair.
Return exactly two lines:
Q: <question text>
A: <answer text>

Statement: "{statement}"
```

규칙:
- 질문은 핵심 사실의 기억을 테스트해야 합니다.
- 답변은 간결하고 직접적이어야 합니다.

**2단계:** 스크립트를 호출하여 카드를 저장합니다.

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add \
  --question "What year did World War 2 end?" \
  --answer "1945" \
  --collection "History"
```

사용자가 컬렉션을 지정하지 않으면 기본값으로 `"General"`을 사용하세요.

### 수동 카드 생성

사용자가 플래시카드 생성을 명시적으로 요청하면 다음을 물어보세요.
1. 질문(카드 앞면)
2. 답변(카드 뒷면)
3. 컬렉션 이름(선택 사항 — 기본값은 `"General"`)

그런 다음 위와 같이 `memento_cards.py add`를 호출하세요.

### 복습 기한 카드 복습

복습하려는 경우 기한이 된 카드를 모두 가져옵니다.

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

이 명령은 `next_review_at <= now`인 카드의 JSON 배열을 반환합니다. 컬렉션 필터가 필요한 경우:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due --collection "History"
```

**복습 흐름(자유 형식 채점):**

다음은 반드시 따라야 하는 정확한 상호작용 패턴의 예시입니다. 사용자가 답하면 채점하고, 정답을 알려준 다음 카드의 등급을 매깁니다.

**상호작용 예시:**

> **에이전트:** 베를린 장벽이 무너진 해는 언제인가요?
>
> **사용자:** 1991년
>
> **에이전트:** 아쉽습니다. 베를린 장벽이 무너진 해는 1989년입니다. 다음 복습은 내일입니다.
> *(에이전트 호출: memento_cards.py rate --id ABC --rating hard --user-answer "1991")*
>
> 다음 질문: 달에 처음 걸어간 사람은 누구인가요?

**규칙:**

1. 질문만 보여 주세요. 사용자가 답할 때까지 기다리세요.
2. 답변을 받은 뒤 예상 답변과 비교하여 채점하세요.
   - **정답** → 표현이 달라도 핵심 사실을 맞혔습니다.
   - **부분 정답** → 방향은 맞지만 핵심 세부 사항이 빠졌습니다.
   - **오답** → 틀렸거나 주제에서 벗어났습니다.
3. **반드시 정답과 결과를 사용자에게 알려야 합니다.** 짧고 일반 텍스트로 유지하세요. 다음 형식을 사용하세요.
   - 정답: "Correct. Answer: &#123;answer&#125;. Next review in 7 days."
   - 부분 정답: "Close. Answer: &#123;answer&#125;. &#123;what they missed&#125;. Next review in 3 days."
   - 오답: "Not quite. Answer: &#123;answer&#125;. Next review tomorrow."
4. 그다음 등급 명령을 호출하세요. 정답→easy, 부분 정답→good, 오답→hard.
5. 다음 질문을 보여 주세요.

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```

**3단계를 절대 건너뛰지 마세요.** 다음 카드로 넘어가기 전에 사용자가 항상 정답과 피드백을 볼 수 있어야 합니다.

기한이 된 카드가 없으면 다음과 같이 말하세요: "No cards due for review right now. Check back later!"

**폐기 재정의:** 사용자는 언제든 "retire this card"라고 말해 카드를 복습에서 영구적으로 제거할 수 있습니다. 이때 `--rating retire`를 사용하세요.

### 간격 반복 알고리즘

등급에 따라 다음 복습 간격이 결정됩니다.

| 등급 | 간격 | ease_streak | 상태 변경 |
|---|---|---|---|
| **hard** | +1일 | 0으로 초기화 | 학습 상태 유지 |
| **good** | +3일 | 0으로 초기화 | 학습 상태 유지 |
| **easy** | +7일 | +1 | ease_streak >= 3이면 → 폐기 |
| **retire** | 영구 | 0으로 초기화 | → 폐기 |

- **learning**: 카드가 현재 복습 순환에 있음
- **retired**: 카드가 복습에 나타나지 않음(사용자가 숙달했거나 수동으로 폐기함)
- "easy" 등급이 연속 3번이면 카드가 자동으로 폐기됩니다.

### YouTube 퀴즈 생성

사용자가 YouTube URL을 보내 퀴즈를 요청하면 다음과 같이 하세요.

**1단계:** URL에서 동영상 ID를 추출합니다(예: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`에서 `dQw4w9WgXcQ`).

**2단계:** 자막을 가져옵니다.

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/youtube_quiz.py fetch VIDEO_ID
```

이 명령은 `{"title": "...", "transcript": "..."}` 또는 오류를 반환합니다.

스크립트가 `missing_dependency`를 보고하면 다음 설치 명령을 안내하세요.
```bash
pip install youtube-transcript-api
```

**3단계:** 자막에서 퀴즈 질문 5개를 생성합니다. 다음 규칙을 사용하세요.

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

자막의 처음 15,000자를 문맥으로 사용하세요. 질문은 직접 생성합니다(LLM).

**4단계:** 출력이 정확히 5개 항목을 가진 유효한 JSON인지, 각 항목에 비어 있지 않은 `question` 및 `answer` 문자열이 있는지 검증합니다. 검증에 실패하면 한 번 다시 시도하세요.

**5단계:** 퀴즈 카드를 저장합니다.

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add-quiz \
  --video-id "VIDEO_ID" \
  --questions '[{"question":"...","answer":"..."},...]' \
  --collection "Quiz - Episode Title"
```

스크립트는 `video_id`를 기준으로 중복을 제거합니다. 해당 동영상의 카드가 이미 있으면 생성을 건너뛰고 기존 카드를 보고합니다.

**6단계:** 동일한 자유 형식 채점 흐름으로 질문을 하나씩 제시합니다.
1. "Question 1/5: ..."를 표시하고 사용자의 답변을 기다립니다. 정답이나 정답을 유추할 힌트를 절대 포함하지 마세요.
2. 사용자가 자신의 말로 답할 때까지 기다립니다.
3. "복습 기한 카드 복습" 섹션의 채점 프롬프트를 사용하여 답변을 채점합니다.
4. **중요: 다른 일을 하기 전에 반드시 사용자에게 피드백으로 답해야 합니다.** 등급, 정답, 카드의 다음 복습 시점을 표시하세요. 다음 질문으로 조용히 넘어가지 마세요. 짧고 일반 텍스트로 유지하세요. 예: "Not quite. Answer: &#123;answer&#125;. Next review tomorrow."
5. **피드백을 표시한 후** 등급 명령을 호출하고 같은 메시지에서 다음 질문을 보여 주세요.
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```
6. 반복합니다. 모든 답변에는 반드시 다음 질문 전에 눈에 보이는 피드백이 있어야 합니다.

### CSV 내보내기/가져오기

**내보내기:**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py export \
  --output ~/flashcards.csv
```

3열 CSV인 `question,answer,collection`을 생성합니다(헤더 행 없음).

**가져오기:**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py import \
  --file ~/flashcards.csv \
  --collection "Imported"
```

`question`, `answer`, 선택적으로 `collection`(3열)이 있는 CSV를 읽습니다. 컬렉션 열이 없으면 `--collection` 인수를 사용합니다.

### 통계

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
```

다음 항목이 포함된 JSON을 반환합니다.
- `total`: 전체 카드 수
- `learning`: 활성 순환 중인 카드
- `retired`: 숙달된 카드
- `due_now`: 현재 복습 기한인 카드
- `collections`: 컬렉션 이름별 내역

## 주의 사항

- **`cards.json`을 직접 편집하지 마세요** — 손상을 방지하려면 항상 스크립트 하위 명령을 사용하세요.
- **자막 가져오기 실패** — 일부 YouTube 동영상에는 영어 자막이 없거나 자막이 비활성화되어 있습니다. 사용자에게 알리고 다른 동영상을 제안하세요.
- **선택적 종속성** — `youtube_quiz.py`에는 `youtube-transcript-api`가 필요합니다. 없는 경우 `pip install youtube-transcript-api`를 실행하라고 안내하세요.
- **대량 가져오기** — 수천 행의 CSV 가져오기도 문제없이 작동하지만 JSON 출력이 장황할 수 있으므로 결과를 요약하세요.
- **동영상 ID 추출** — `youtube.com/watch?v=ID` 및 `youtu.be/ID` URL 형식을 모두 지원하세요.

## 검증

도우미 스크립트를 직접 검증하세요.

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add --question "Capital of France?" --answer "Paris" --collection "General"
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

저장소 체크아웃에서 테스트하는 경우 다음을 실행하세요.

```bash
pytest tests/skills/test_memento_cards.py tests/skills/test_youtube_quiz.py -q
```

에이전트 수준 검증:
- 복습을 시작하여 피드백이 일반 텍스트이고, 짧으며, 다음 카드 전에 항상 정답을 포함하는지 확인하세요.
- YouTube 퀴즈 흐름을 실행하여 각 답변에 다음 질문 전에 눈에 보이는 피드백이 제공되는지 확인하세요.
