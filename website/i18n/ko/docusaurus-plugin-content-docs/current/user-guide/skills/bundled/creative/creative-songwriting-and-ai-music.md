---
title: "Songwriting And Ai Music — 작곡 기술 및 Suno AI 음악 프롬프트"
sidebar_label: "Songwriting And Ai Music"
description: "작곡 기술 및 Suno AI 음악 프롬프트"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Songwriting And Ai Music

작곡 기술 및 Suno AI 음악 프롬프트입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/creative/songwriting-and-ai-music` |
| 플랫폼 | linux, macos, windows |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬을 트리거할 때 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 명령으로 보는 내용입니다.
:::

# 작곡 & AI 음악 생성

여기에 있는 모든 것은 가이드라인일 뿐 규칙이 아닙니다. 예술은 목적을 위해 규칙을 깹니다.
노래에 도움이 되는 것은 사용하고, 그렇지 않은 것은 무시하세요.

---

## 1. 곡 구조 (하나를 선택하거나 직접 만드세요)

일반적인 뼈대 — 필요에 따라 혼합하거나 수정하거나 버리세요:

```
ABABCB  Verse/Chorus/Verse/Chorus/Bridge/Chorus    (대부분의 팝/록)
AABA    Verse/Verse/Bridge/Verse (refrain-based)   (재즈 스탠다드, 발라드)
ABAB    Verse/Chorus 교대                          (단순하고 직관적)
AAA     Verse/Verse/Verse (코러스가 없는 유절 형식)  (포크, 스토리텔링)
```

여섯 가지 구성 요소:
- Intro      — 분위기를 설정하고 청취자를 끌어들입니다.
- Verse      — 이야기, 세부 사항, 세계관을 구축합니다.
- Pre-Chorus — 클라이맥스 전의 긴장감을 끌어올립니다(선택 사항).
- Chorus     — 감정의 핵심이자 사람들이 기억하는 부분입니다.
- Bridge     — 관점이나 조(key)가 바뀌는 전환점입니다.
- Outro      — 마무리로, 앞의 내용을 반복하거나 반전을 줍니다.

이 모든 것이 필요하지는 않습니다. 일부 훌륭한 노래는 진화하는 단 하나의 섹션으로 이루어져 있습니다. 구조는 감정을 위해 존재하는 것이지, 그 반대가 아닙니다.

---

## 2. 운율(Rhyme), 박자(Meter), 그리고 소리

운율 유형 (엄격한 것에서 느슨한 것 순):
- Perfect (완전 운율): lean/mean
- Family (유사 운율): crate/braid
- Assonance (모음 운율): had/glass (모음은 같고 끝자음이 다름)
- Consonance (자음 운율): scene/when (모음은 다르고 끝자음이 유사함)
- Near/slant (불완전 운율): 얽매이지 않고 연결성만 암시할 정도

이들을 섞으세요. 모두 완전 운율만 사용하면 동요처럼 들릴 수 있습니다.
모두 불완전 운율만 사용하면 성의 없어 보일 수 있습니다. 적절한 혼합에 생명력이 있습니다.

내부 운율 (INTERNAL RHYME): 끝뿐만 아니라 줄 안에서 운을 맞추는 것.
  "We pruned the lies from bleeding trees / Distilled the storm
   from entropy" — "lies/flies", "trees/entropy"가 내부적 메아리를 만듭니다.

박자 (METER): 강세가 있는 음절과 없는 음절의 리듬.
- 평행한 줄 사이의 음절 수를 맞추면 부르기 쉬워집니다.
- 총 음절 수보다 **강세가 있는** 음절이 더 중요합니다.
- 소리 내어 읽어보세요. 더듬거리게 된다면 박자를 수정해야 합니다.
- 의도적으로 박자를 깨면 강조나 놀라움을 줄 수 있습니다.

---

## 3. 감정선과 다이내믹

노래를 평탄한 길이 아니라 하나의 여정으로 생각하세요.

에너지 매핑 (대략적인 아이디어이며 처방은 아님):
  Intro: 2-3  |  Verse: 5-6  |  Pre-Chorus: 7
  Chorus: 8-9  |  Bridge: 다양함  |  Final Chorus: 9-10

가장 강력한 다이내믹 트릭: **대비(CONTRAST)**.
- 소리치기 전의 속삭임이 그냥 소리치는 것보다 더 강렬합니다.
- 빽빽하기 전에 드문드문하게. 빠르기 전에 느리게. 높기 전에 낮게.
- 드랍(drop)은 빌드업이 있기 때문에 효과가 있습니다.
- 침묵도 하나의 악기입니다.

"속삭임에서 포효로, 다시 속삭임으로" — 친밀하게 시작해서 최대의 힘으로 키우고, 다시 취약한 상태로 벗겨냅니다. 발라드, 서사곡, 앤섬(anthem)에 모두 통용됩니다.

---

## 4. 효과적인 가사 쓰기

보여주되 말하지 마세요 (SHOW, DON'T TELL - 대부분의 경우):
- "나는 슬펐다" = 평면적
- "네 후드티가 아직 문 옆 후크에 걸려 있어" = 생생함
- 하지만 때로는 "내 목숨을 바치겠다"라고 솔직하게 말하는 것이 힘이 될 때도 있습니다.

훅 (THE HOOK):
- 사람들이 기억하고, 흥얼거리고, 반복하는 구절
- 주로 제목이나 핵심 문구
- 멜로디 + 가사 + 감정이 모두 일치할 때 가장 효과적입니다.
- 가장 강하게 와닿는 곳(주로 코러스의 첫 줄이나 마지막 줄)에 배치하세요.

운율학 (PROSODY) — 가사와 음악이 서로를 지지하는 것:
- 안정적인 감정(해결, 평화)은 안정적인 멜로디, 완전 운율, 해결된 코드로 짝을 이룹니다.
- 불안정한 감정(갈망, 의심)은 헤매는 멜로디, 불완전 운율, 해결되지 않은 코드로 짝을 이룹니다.
- Verse 멜로디는 보통 낮게 깔리고, Chorus는 더 높이 올라갑니다.
- 하지만 노래에 도움이 된다면 이를 뒤집어도 좋습니다.

피해야 할 것 (의도적인 것이 아니라면):
- 무의식적으로 쓰는 클리셰 (맥락 없이 쓰는 "황금 같은 마음")
- 운율을 맞추기 위해 억지로 어순 바꾸기 ("요다 화법")
- 모든 섹션에 같은 에너지 사용하기 (평면적인 다이내믹)
- 초안을 신성시하기 — 수정이 곧 창조입니다.

---

## 5. 패러디와 각색

기존 노래에 새 가사를 써서 다시 쓸 때:

뼈대 (THE SKELETON): 원곡의 구조를 먼저 파악하세요.
- 줄당 음절 수를 세세요.
- 운율 체계(ABAB, AABB 등)를 표시하세요.
- 어느 음절에 강세(STRESSED)가 들어가는지 확인하세요.
- 음을 길게 끄는(held/sustained) 곳이 어디인지 주목하세요.

새 단어 맞추기 (FITTING NEW WORDS):
- 강세가 있는 음절을 원곡의 같은 박자에 맞추세요.
- 전체 음절 수는 강세가 없는 음절 1-2개 정도는 유동적일 수 있습니다.
- 길게 끄는 음에서는 원곡의 **모음 소리**에 맞추려고 노력하세요.
  (원곡이 "LOOOVE"처럼 '우' 모음을 끈다면, "LIFE"보다는 "FOOOD"가 더 잘 맞습니다.)
- 주요 위치에서 1음절 단어를 교체하면 리듬이 그대로 유지됩니다.
  (Crime -> Code, Snake -> Noose)
- 원곡 위에 새 단어를 불러보세요 — 더듬거린다면 수정하세요.

컨셉 (CONCEPT):
- 곡 전체를 끌고 갈 만큼 강력한 컨셉을 고르세요.
- 제목/훅에서 시작해서 밖으로 확장해 나가세요.
- 날것의 소재(말장난, 구절, 이미지)를 **먼저** 많이 생성한 다음, 가장 좋은 것을 구조에 끼워 넣으세요.
- 특정 위치에 특정 줄이 필요하다면, 운율 체계를 거꾸로 역설계해서 설정하세요.

일부 원본 남기기 (KEEP SOME ORIGINALS):
원본의 몇 줄이나 구조를 그대로 남겨두면 인지도를 높이고 청중이 연결감을 느끼게 할 수 있습니다.

---

## 6. Suno AI 프롬프트 엔지니어링

### Style/Genre Description (스타일/장르 설명) 필드

공식 (필요에 따라 적용):
  장르 + 무드 + 시대 + 악기 + 보컬 스타일 + 프로덕션 + 다이내믹

```
나쁨:  "슬픈 록 음악"
좋음: "Cinematic orchestral spy thriller, 1960s Cold War era, smoky
       sultry female vocalist, big band jazz, brass section with
       trumpets and french horns, sweeping strings, minor key,
       vintage analog warmth"
```

장르뿐만 아니라 **여정**을 묘사하세요:
```
"Begins as a haunting whisper over sparse piano. Gradually layers
 in muted brass. Builds through the chorus with full orchestra.
 Second verse erupts with raw belting intensity. Outro strips back
 to a lone piano and a fragile whisper fading to silence."
```

팁:
- V4.5 이상은 Style 필드에서 최대 1,000자까지 지원합니다 — 최대한 활용하세요.
- 아티스트 이름이나 상표를 사용하지 마세요. 대신 사운드를 묘사하세요.
  "James Bond style" 대신 "1960s Cold War spy thriller brass"
  "Nirvana-style" 대신 "90s grunge"
- 선호하는 것이 있다면 BPM과 조(key)를 지정하세요.
- 원하지 않는 것을 제외하려면 Exclude Styles 필드를 사용하세요.
- "bossa nova trap", "Appalachian gothic", "chiptune jazz" 같이 예상치 못한 장르 조합이 좋은 결과를 낼 수 있습니다.
- 단순한 성별이 아니라 보컬 **페르소나**를 구축하세요:
  "A weathered torch singer with a smoky alto, slight rasp,
   who starts vulnerable and builds to devastating power"

### 메타태그 (Lyrics 필드 내 괄호 [ ] 안에 배치)

구조:
  [Intro] [Verse] [Verse 1] [Pre-Chorus] [Chorus]
  [Post-Chorus] [Hook] [Bridge] [Interlude]
  [Instrumental] [Instrumental Break] [Guitar Solo]
  [Breakdown] [Build-up] [Outro] [Silence] [End]

보컬 퍼포먼스:
  [Whispered] [Spoken Word] [Belted] [Falsetto] [Powerful]
  [Soulful] [Raspy] [Breathy] [Smooth] [Gritty]
  [Staccato] [Legato] [Vibrato] [Melismatic]
  [Harmonies] [Choir] [Harmonized Chorus]

다이내믹:
  [High Energy] [Low Energy] [Building Energy] [Explosive]
  [Emotional Climax] [Gradual swell] [Orchestral swell]
  [Quiet arrangement] [Falling tension] [Slow Down]

성별:
  [Female Vocals] [Male Vocals]

분위기:
  [Melancholic] [Euphoric] [Nostalgic] [Aggressive]
  [Dreamy] [Intimate] [Dark Atmosphere]

효과음(SFX):
  [Vinyl Crackle] [Rain] [Applause] [Static] [Thunder]

스타일 필드와 가사 **모두**에 태그를 넣어 강조하세요.
섹션당 태그는 최대 5-8개로 유지하세요 — 너무 많으면 AI가 혼란스러워합니다.
모순되는 태그를 같이 쓰지 마세요 (같은 섹션에 [Calm] + [Aggressive]).

### Custom Mode (커스텀 모드)
- 진지한 작업을 할 때는 항상 커스텀 모드를 사용하세요 (Style과 Lyrics 분리).
- 가사 필드 제한: 약 3,000자 (약 40-60줄).
- 구조적 태그를 항상 추가하세요 — 태그가 없으면 Suno는 감정선이 없는 평면적인 verse/chorus/verse를 기본으로 합니다.

---

## 7. AI 가수를 위한 발음 팁

AI 보컬리스트는 글을 읽는 것이 아니라 발음합니다. 그들을 도와주세요:

발음대로 스펠링 쓰기:
- **소리 나는 대로** 단어 철자를 쓰세요: "through" -> "thru"
- 고유 명사는 실패율이 가장 높습니다 — 일찍 테스트하세요.
- "Nous" -> "Noose" (정확한 발음을 유도)
- 하이픈을 사용하여 음절을 안내하세요: "Re-search", "bio-engineering"

전달력 제어 (DELIVERY CONTROL):
- ALL CAPS (대문자) = 더 크고 강렬하게
- 모음 연장: "lo-o-o-ove" = 소리 길게 끌기/멜리스마
- 말줄임표: "I... need... you" = 극적인 멈춤
- 하이픈 연장: "ne-e-ed" = 감정적인 끌림

항상 해야 할 것:
- 숫자는 문자로 풀어 쓰세요: "24/7" -> "twenty four seven"
- 약어는 띄어 쓰세요: "AI" -> "A I" 또는 "A-I"
- 고유 명사나 특이한 단어는 먼저 30초짜리 짧은 클립으로 테스트하세요.
- 일단 생성되면 발음이 고정됩니다 — 생성 **전**에 가사에서 수정하세요.

---

## 8. 워크플로우

1. 컨셉/훅을 먼저 쓰세요 — 감정의 핵심은 무엇인가요?
2. 각색하는 경우 원본 구조(음절, 운율, 강세)를 매핑하세요.
3. 원시 소재를 생성하세요 — 구조를 잡기 전에 자유롭게 브레인스토밍하세요.
4. 가사 초안을 구조에 맞게 작성하세요.
5. 소리 내어 읽거나 불러보세요 — 더듬거리는 부분을 찾고 박자를 고치세요.
6. Suno 스타일 설명을 만드세요 — 다이내믹한 여정을 그려주세요.
7. 퍼포먼스 디렉션을 위해 가사에 메타태그를 추가하세요.
8. 최소 3-5개의 변형을 생성하세요 — 녹음 테이크처럼 다루세요.
9. 가장 좋은 것을 고르고, Extend/Continue 기능을 사용해 좋은 섹션을 기반으로 계속 만들어가세요.
10. 우연히 훌륭한 결과가 나오면 그것을 살리세요.

예상하기: 1개의 좋은 결과를 얻기 위해 대략 3-5번의 생성이 필요합니다. 수정은 정상입니다.
Extend 시 스타일이 변질될 수 있습니다 — 연장할 때 장르/분위기를 다시 명시하세요.

---

## 9. 얻은 교훈 (Lessons Learned)

- 스타일 필드에서 단순 장르 나열보다 다이내믹한 곡선을 묘사하는 것이 훨씬 중요합니다. "Whisper to roar to whisper"는 Suno에게 퍼포먼스 지도를 제공합니다.
- 패러디 시 원본의 일부 가사를 그대로 두면 인지도와 감정적인 무게를 더해줍니다 — 청중이 원본의 향수를 느끼게 합니다.
- 노래의 Bridge 위치는 이미지를 변형할 수 있는 곳입니다. 감정적 기능(반사, 전환, 계시)은 유지하면서 원본의 구체적인 레퍼런스를 주제의 은유로 바꾸세요.
- 훅이나 태그에서 한 음절 단어를 바꾸는 것이 의미를 바꾸면서도 리듬을 유지하는 가장 깔끔한 방법입니다.
- 스타일 필드에서 강력한 보컬 페르소나 묘사는 개별 메타태그보다 더 큰 차이를 만듭니다.
- 규칙에 얽매이지 마세요. 어떤 줄이 박자를 깨더라도 더 강렬하게 와닿는다면 그대로 두세요. 느낌이 가장 중요합니다. 기교는 예술을 위해 존재하는 것이지, 그 반대가 아닙니다.
