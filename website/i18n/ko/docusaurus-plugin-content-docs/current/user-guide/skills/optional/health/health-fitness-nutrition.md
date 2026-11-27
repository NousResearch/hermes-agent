---
title: "피트니스 영양 — wger/USDA를 통한 운동 계획, 매크로 및 신체 지표"
sidebar_label: "피트니스 영양"
description: "wger/USDA를 통한 운동 계획, 매크로 및 신체 지표"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 피트니스 영양

wger/USDA를 통한 운동 계획, 매크로 및 신체 지표.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/health/fitness-nutrition`으로 설치 |
| 경로 | `optional-skills/health/fitness-nutrition` |
| 버전 | `1.0.0` |
| 작성자 | Hailey Marshall (haileymarshall), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `health`, `fitness`, `nutrition`, `gym`, `workout`, `diet`, `exercise` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 지침으로 보게 되는 내용입니다.
:::

# 피트니스 및 영양

전문 피트니스 코치이자 스포츠 영양사 스킬입니다. 두 가지 데이터 소스와
오프라인 계산기를 활용해 헬스장 이용자에게 필요한 모든 것을 한곳에서 제공합니다.

**데이터 소스 (모두 무료, pip 의존성 없음):**

- **wger** (https://wger.de/api/v2/) — 근육, 장비, 이미지 정보가 포함된 690개 이상의 운동을 제공하는 공개 운동 데이터베이스입니다. 공개 엔드포인트에는 인증이 전혀 필요하지 않습니다.
- **USDA FoodData Central** (https://api.nal.usda.gov/fdc/v1/) — 미국 정부의 영양 데이터베이스로, 380,000개 이상의 식품을 제공합니다. `DEMO_KEY`는 즉시 작동하며, 더 높은 한도를 사용하려면 무료 가입이 필요합니다.

**오프라인 계산기 (순수 stdlib Python):**

- BMI, TDEE (Mifflin-St Jeor), 1회 최대 중량 (Epley/Brzycki/Lombardi), 매크로 분배, 체지방률 (미 해군 방식)

---

## 사용 시점

사용자가 다음에 관해 질문할 때 이 스킬을 활성화합니다:
- 운동, 워크아웃, 헬스장 루틴, 근육군, 운동 분할
- 식품 매크로, 칼로리, 단백질 함량, 식단 계획, 칼로리 계산
- 체성분: BMI, 체지방률, TDEE, 칼로리 흑자/적자
- 1회 최대 중량 추정, 훈련 비율, 점진적 과부하
- 감량, 증량 또는 유지에 필요한 매크로 비율

---

## 절차

### 운동 조회 (wger API)

모든 wger 공개 엔드포인트는 JSON을 반환하며 인증이 필요하지 않습니다. 운동 쿼리에는 항상
`format=json`과 `language=2`(영어)를 추가합니다.

**1단계 — 사용자가 원하는 것을 파악합니다:**

- 근육별 → `/api/v2/exercise/?muscles={id}&language=2&status=2&format=json` 사용
- 카테고리별 → `/api/v2/exercise/?category={id}&language=2&status=2&format=json` 사용
- 장비별 → `/api/v2/exercise/?equipment={id}&language=2&status=2&format=json` 사용
- 이름별 → `/api/v2/exercise/search/?term={query}&language=english&format=json` 사용
- 전체 상세 정보 → `/api/v2/exerciseinfo/{exercise_id}/?format=json` 사용

**2단계 — 참조 ID (추가 API 호출이 필요하지 않도록 제공):**

운동 카테고리:

| ID | 카테고리 |
|----|-------------|
| 8  | 팔 |
| 9  | 다리 |
| 10 | 복근 |
| 11 | 가슴 |
| 12 | 등 |
| 13 | 어깨 |
| 14 | 종아리 |
| 15 | 유산소 |

근육:

| ID | 근육                    | ID | 근육                  |
|----|---------------------------|----|-------------------------|
| 1  | 상완이두근            | 2  | 전면 삼각근        |
| 3  | 전거근              | 4  | 대흉근        |
| 5  | 외복사근            | 6  | 비복근           |
| 7  | 복직근          | 8  | 대둔근         |
| 9  | 승모근            | 10 | 대퇴사두근      |
| 11 | 대퇴이두근           | 12 | 광배근        |
| 13 | 상완근                | 14 | 상완삼두근         |
| 15 | 가자미근                    |    |                         |

장비:

| ID | 장비      |
|----|----------------|
| 1  | 바벨        |
| 3  | 덤벨       |
| 4  | 운동 매트         |
| 5  | 스위스 볼     |
| 6  | 풀업 바    |
| 7  | 없음 (맨몸) |
| 8  | 벤치          |
| 9  | 인클라인 벤치  |
| 10 | 케틀벨     |

**3단계 — 결과를 가져와 제시합니다:**

```bash
# Search exercises by name
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
# Get full details for a specific exercise
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
# List exercises filtering by muscle, category, or equipment
# Combine filters as needed: ?muscles=4&equipment=1&language=2&status=2
FILTER="$1"  # e.g. "muscles=4" or "category=11" or "equipment=3"
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

설정되어 있으면 `USDA_API_KEY` 환경 변수를 사용하고, 그렇지 않으면 `DEMO_KEY`로 대체합니다.
DEMO_KEY = 시간당 30회 요청. 무료 가입 키 = 시간당 1,000회 요청.

```bash
# Search foods by name
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
# Detailed nutrient profile by FDC ID
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

일괄 작업에는 `scripts/`의 도우미 스크립트를 사용하고,
단일 계산에는 인라인으로 실행합니다:

- `python3 scripts/body_calc.py bmi <weight_kg> <height_cm>`
- `python3 scripts/body_calc.py tdee <weight_kg> <height_cm> <age> <M|F> <activity 1-5>`
- `python3 scripts/body_calc.py 1rm <weight> <reps>`
- `python3 scripts/body_calc.py macros <tdee_kcal> <cut|maintain|bulk>`
- `python3 scripts/body_calc.py bodyfat <M|F> <neck_cm> <waist_cm> [hip_cm] <height_cm>`

각 공식의 과학적 근거는 `references/FORMULAS.md`를 참고하세요.

---

## 주의 사항

- wger 운동 엔드포인트는 기본적으로 **모든 언어**를 반환합니다 — 영어로 받으려면 항상 `language=2`를 추가하세요.
- wger에는 **검증되지 않은 사용자 제출 항목**이 포함됩니다 — 승인된 운동만 받으려면 `status=2`를 추가하세요.
- USDA `DEMO_KEY`는 **시간당 30회 요청**으로 제한됩니다 — 일괄 요청 사이에 `sleep 2`를 추가하거나 무료 키를 발급받으세요.
- USDA 데이터는 **100g 기준**입니다 — 사용자에게 실제 섭취량에 맞게 환산하도록 안내하세요.
- BMI는 근육과 지방을 구분하지 않습니다 — 근육량이 많은 사람의 높은 BMI가 반드시 건강하지 않다는 뜻은 아닙니다.
- 체지방 공식은 **추정치**(±3-5%)입니다 — 정밀도가 필요하면 DEXA 스캔을 권장하세요.
- 1RM 공식은 10회를 초과하면 정확도가 떨어집니다 — 최상의 추정을 위해 3-5회 세트를 사용하세요.
- wger의 `exercise/search` 엔드포인트는 매개변수 이름으로 `query`가 아닌 `term`을 사용합니다.

---

## 검증

운동 검색 후: 결과에 운동 이름, 근육군 및 장비가 포함되는지 확인하세요.
영양 조회 후: 100g 기준 매크로가 kcal, 단백질, 지방, 탄수화물과 함께 반환되는지 확인하세요.
계산기 실행 후: 결과의 타당성을 점검하세요 (예: 대부분 성인의 TDEE는 1500-3500이어야 합니다).

---

## 빠른 참조

| 작업 | 소스 | 엔드포인트 |
|------|--------|----------|
| 이름으로 운동 검색 | wger | `GET /api/v2/exercise/search/?term=&language=english` |
| 운동 상세 정보 | wger | `GET /api/v2/exerciseinfo/{id}/` |
| 근육별 필터링 | wger | `GET /api/v2/exercise/?muscles={id}&language=2&status=2` |
| 장비별 필터링 | wger | `GET /api/v2/exercise/?equipment={id}&language=2&status=2` |
| 카테고리 목록 | wger | `GET /api/v2/exercisecategory/` |
| 근육 목록 | wger | `GET /api/v2/muscle/` |
| 식품 검색 | USDA | `GET /fdc/v1/foods/search?query=&dataType=Foundation,SR Legacy` |
| 식품 상세 정보 | USDA | `GET /fdc/v1/food/{fdcId}` |
| BMI / TDEE / 1RM / 매크로 | 오프라인 | `python3 scripts/body_calc.py` |
