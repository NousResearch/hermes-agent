---
title: "Fitness Nutrition — 헬스장 운동 플래너 및 영양 트래커"
sidebar_label: "Fitness Nutrition"
description: "헬스장 운동 플래너 및 영양 트래커"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Fitness Nutrition

헬스장 운동 플래너 및 영양 트래커입니다. wger를 통해 근육, 장비 또는 카테고리별로 690개 이상의 운동을 검색할 수 있습니다. USDA FoodData Central을 통해 380,000개 이상의 식품에 대한 매크로(다량 영양소)와 칼로리를 조회할 수 있습니다. BMI, TDEE, 1RM(1-rep max), 매크로 분할, 체지방을 계산합니다. pip 설치가 필요 없는 순수 Python으로 작성되었습니다. 근육량을 늘리거나, 체중을 감량하거나, 단지 더 나은 식습관을 가지려는 모든 사람을 위해 만들어졌습니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | 선택 사항 — `hermes skills install official/health/fitness-nutrition`으로 설치합니다. |
| Path | `optional-skills/health/fitness-nutrition` |
| Version | `1.0.0` |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `health`, `fitness`, `nutrition`, `gym`, `workout`, `diet`, `exercise` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Fitness & Nutrition

전문 피트니스 코치 및 스포츠 영양사 스킬입니다. 두 가지 데이터 소스와 오프라인 계산기를 제공하여 헬스장 이용자에게 필요한 모든 것을 한 곳에서 제공합니다.

**데이터 소스 (모두 무료, pip 의존성 없음):**

- **wger** (https://wger.de/api/v2/) — 근육, 장비, 이미지가 포함된 690개 이상의 운동이 있는 개방형 운동 데이터베이스입니다. 공개 엔드포인트는 인증이 전혀 필요하지 않습니다.
- **USDA FoodData Central** (https://api.nal.usda.gov/fdc/v1/) — 380,000개 이상의 식품이 있는 미국 정부 영양 데이터베이스입니다. `DEMO_KEY`는 즉시 작동하며, 더 높은 한도를 원하면 무료로 가입할 수 있습니다.

**오프라인 계산기 (순수 표준 라이브러리 Python):**

- BMI, TDEE (Mifflin-St Jeor), 1RM (Epley/Brzycki/Lombardi), 매크로 분할, 체지방률 (US Navy method)

---

## 사용 시기

사용자가 다음과 같은 내용을 물어볼 때 이 스킬을 트리거하세요:
- 운동, 워크아웃, 헬스장 루틴, 근육군, 운동 분할
- 식품 매크로, 칼로리, 단백질 함량, 식단 계획, 칼로리 계산
- 신체 조성: BMI, 체지방, TDEE, 잉여/결핍 칼로리
- 1RM 추정치, 훈련 비율, 점진적 과부하
- 커팅, 벌킹, 유지를 위한 매크로 비율

---

## 절차

### 운동 조회 (wger API)

모든 wger 공개 엔드포인트는 JSON을 반환하며 인증이 필요하지 않습니다. 운동 쿼리에는 항상 `format=json` 및 `language=2` (영어)를 추가하세요.

**1단계 — 사용자가 원하는 것 파악:**

- 근육별 → `/api/v2/exercise/?muscles={id}&language=2&status=2&format=json` 사용
- 카테고리별 → `/api/v2/exercise/?category={id}&language=2&status=2&format=json` 사용
- 장비별 → `/api/v2/exercise/?equipment={id}&language=2&status=2&format=json` 사용
- 이름별 → `/api/v2/exercise/search/?term={query}&language=english&format=json` 사용
- 전체 세부 정보 → `/api/v2/exerciseinfo/{exercise_id}/?format=json` 사용

**2단계 — 참조 ID (추가 API 호출을 방지하기 위함):**

운동 카테고리:

| ID | 카테고리 |
|----|-------------|
| 8  | 팔 (Arms)        |
| 9  | 다리 (Legs)        |
| 10 | 복근 (Abs)         |
| 11 | 가슴 (Chest)       |
| 12 | 등 (Back)        |
| 13 | 어깨 (Shoulders)   |
| 14 | 종아리 (Calves)      |
| 15 | 유산소 (Cardio)      |

근육:

| ID | 근육                    | ID | 근육                  |
|----|---------------------------|----|-------------------------|
| 1  | 상완이두근 (Biceps brachii)            | 2  | 전면 삼각근 (Anterior deltoid)        |
| 3  | 전거근 (Serratus anterior)         | 4  | 대흉근 (Pectoralis major)        |
| 5  | 외복사근 (Obliquus externus)         | 6  | 비복근 (Gastrocnemius)           |
| 7  | 복직근 (Rectus abdominis)          | 8  | 대둔근 (Gluteus maximus)         |
| 9  | 승모근 (Trapezius)                 | 10 | 대퇴사두근 (Quadriceps femoris)      |
| 11 | 대퇴이두근 (Biceps femoris)            | 12 | 광배근 (Latissimus dorsi)        |
| 13 | 상완근 (Brachialis)                | 14 | 상완삼두근 (Triceps brachii)         |
| 15 | 가자미근 (Soleus)                    |    |                         |

장비:

| ID | 장비      |
|----|----------------|
| 1  | 바벨 (Barbell)        |
| 3  | 덤벨 (Dumbbell)       |
| 4  | 짐볼/매트 (Gym mat)        |
| 5  | 스위스 볼 (Swiss Ball)     |
| 6  | 풀업 바 (Pull-up bar)    |
| 7  | 없음/맨몸 (none/bodyweight) |
| 8  | 벤치 (Bench)          |
| 9  | 인클라인 벤치 (Incline bench)  |
| 10 | 케틀벨 (Kettlebell)     |

**3단계 — 결과 가져오기 및 표시:**

```bash
# 이름으로 운동 검색
QUERY="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$QUERY")
curl -s "https://wger.de/api/v2/exercise/search/?term=${ENCODED}&language=english&format=json" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
for s in data.get('suggestions',[])[:10]:
    d=s.get('data',{})
    print(f\"  ID {d.get('id','?'):>4} | {d.get('name','N/A'):<35} | Category: {d.get('category','N/A')}\")
"
```

```bash
# 특정 운동에 대한 전체 세부 정보 가져오기
EXERCISE_ID="$1"
curl -s "https://wger.de/api/v2/exerciseinfo/${EXERCISE_ID}/?format=json" \
  | python3 -c "
import json,sys,html,re
data=json.load(sys.stdin)
trans=[t for t in data.get('translations',[]) if t.get('language')==2]
t=trans[0] if trans else data.get('translations',[{}])[0]
desc=re.sub('<[^>]+>','',html.unescape(t.get('description','N/A')))
print(f\"Exercise  : {t.get('name','N/A')}\")
print(f\"Category  : {data.get('category',{}).get('name','N/A')}\")
print(f\"Primary   : {', '.join(m.get('name_en','') for m in data.get('muscles',[])) or 'N/A'}\")
print(f\"Secondary : {', '.join(m.get('name_en','') for m in data.get('muscles_secondary',[])) or 'none'}\")
print(f\"Equipment : {', '.join(e.get('name','') for e in data.get('equipment',[])) or 'bodyweight'}\")
print(f\"How to    : {desc[:500]}\")
imgs=data.get('images',[])
if imgs: print(f\"Image     : {imgs[0].get('image','')}\")
"
```

```bash
# 근육, 카테고리 또는 장비별로 필터링하여 운동 목록 가져오기
# 필요에 따라 필터 결합: ?muscles=4&equipment=1&language=2&status=2
FILTER="$1"  # 예: "muscles=4" 또는 "category=11" 또는 "equipment=3"
curl -s "https://wger.de/api/v2/exercise/?${FILTER}&language=2&status=2&limit=20&format=json" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
print(f'Found {data.get(\"count\",0)} exercises.')
for ex in data.get('results',[]):
    print(f\"  ID {ex['id']:>4} | muscles: {ex.get('muscles',[])} | equipment: {ex.get('equipment',[])}\")
"
```

### 영양 조회 (USDA FoodData Central)

설정된 경우 `USDA_API_KEY` 환경 변수를 사용하고, 그렇지 않으면 `DEMO_KEY`로 대체합니다.
DEMO_KEY = 시간당 30회 요청. 무료 가입 키 = 시간당 1,000회 요청.

```bash
# 이름으로 식품 검색
FOOD="$1"
API_KEY="${USDA_API_KEY:-DEMO_KEY}"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$FOOD")
curl -s "https://api.nal.usda.gov/fdc/v1/foods/search?api_key=${API_KEY}&query=${ENCODED}&pageSize=5&dataType=Foundation,SR%20Legacy" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
foods=data.get('foods',[])
if not foods: print('No foods found.'); sys.exit()
for f in foods:
    n={x['nutrientName']:x.get('value','?') for x in f.get('foodNutrients',[])}
    cal=n.get('Energy','?'); prot=n.get('Protein','?')
    fat=n.get('Total lipid (fat)','?'); carb=n.get('Carbohydrate, by difference','?')
    print(f\"{f.get('description','N/A')}\")
    print(f\"  Per 100g: {cal} kcal | {prot}g protein | {fat}g fat | {carb}g carbs\")
    print(f\"  FDC ID: {f.get('fdcId','N/A')}\")
    print()
"
```

```bash
# FDC ID별 자세한 영양 프로필
FDC_ID="$1"
API_KEY="${USDA_API_KEY:-DEMO_KEY}"
curl -s "https://api.nal.usda.gov/fdc/v1/food/${FDC_ID}?api_key=${API_KEY}" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f\"Food: {d.get('description','N/A')}\")
print(f\"{'Nutrient':<40} {'Amount':>8} {'Unit'}\")
print('-'*56)
for x in sorted(d.get('foodNutrients',[]),key=lambda x:x.get('nutrient',{}).get('rank',9999)):
    nut=x.get('nutrient',{}); amt=x.get('amount',0)
    if amt and float(amt)>0:
        print(f\"  {nut.get('name',''):<38} {amt:>8} {nut.get('unitName','')}\")
"
```

### 오프라인 계산기

일괄 작업에는 `scripts/`에 있는 헬퍼 스크립트를 사용하거나,
단일 계산의 경우 인라인으로 실행하세요:

- `python3 scripts/body_calc.py bmi <weight_kg> <height_cm>`
- `python3 scripts/body_calc.py tdee <weight_kg> <height_cm> <age> <M|F> <activity 1-5>`
- `python3 scripts/body_calc.py 1rm <weight> <reps>`
- `python3 scripts/body_calc.py macros <tdee_kcal> <cut|maintain|bulk>`
- `python3 scripts/body_calc.py bodyfat <M|F> <neck_cm> <waist_cm> [hip_cm] <height_cm>`

각 공식의 과학적 배경은 `references/FORMULAS.md`를 참조하세요.

---

## 주의 사항

- wger 운동 엔드포인트는 **기본적으로 모든 언어를 반환합니다** — 영어를 얻으려면 항상 `language=2`를 추가하세요.
- wger에는 **검증되지 않은 사용자 제출 항목**이 포함되어 있습니다 — 승인된 운동만 얻으려면 `status=2`를 추가하세요.
- USDA `DEMO_KEY`는 **시간당 30회 요청**으로 제한됩니다 — 일괄 요청 사이에 `sleep 2`를 추가하거나 무료 키를 받으세요.
- USDA 데이터는 **100g 기준**입니다 — 사용자에게 실제 제공량에 맞게 조정하도록 알려주세요.
- BMI는 근육과 지방을 구분하지 않습니다 — 근육질인 사람의 높은 BMI가 반드시 건강하지 않다는 것을 의미하지는 않습니다.
- 체지방 공식은 **추정치**입니다 (±3-5%) — 정확한 측정을 위해 DEXA 스캔을 권장하세요.
- 1RM 공식은 10회 이상 반복 시 정확도가 떨어집니다 — 최상의 추정을 위해 3-5회 세트를 사용하세요.
- wger의 `exercise/search` 엔드포인트는 매개변수 이름으로 `query`가 아닌 `term`을 사용합니다.

---

## 검증

운동 검색 실행 후: 결과에 운동 이름, 근육군 및 장비가 포함되어 있는지 확인합니다.
영양 조회 후: 100g당 매크로가 kcal, 단백질, 지방, 탄수화물과 함께 반환되는지 확인합니다.
계산기 사용 후: 출력 결과를 검토합니다 (예: 성인의 TDEE는 대부분 1500-3500이어야 함).

---

## 빠른 참조

| 작업 | 소스 | 엔드포인트 |
|------|--------|----------|
| 이름으로 운동 검색 | wger | `GET /api/v2/exercise/search/?term=&language=english` |
| 운동 세부 정보 | wger | `GET /api/v2/exerciseinfo/{id}/` |
| 근육별 필터 | wger | `GET /api/v2/exercise/?muscles={id}&language=2&status=2` |
| 장비별 필터 | wger | `GET /api/v2/exercise/?equipment={id}&language=2&status=2` |
| 카테고리 목록 | wger | `GET /api/v2/exercisecategory/` |
| 근육 목록 | wger | `GET /api/v2/muscle/` |
| 식품 검색 | USDA | `GET /fdc/v1/foods/search?query=&dataType=Foundation,SR Legacy` |
| 식품 세부 정보 | USDA | `GET /fdc/v1/food/{fdcId}` |
| BMI / TDEE / 1RM / 매크로 | 오프라인 | `python3 scripts/body_calc.py` |
