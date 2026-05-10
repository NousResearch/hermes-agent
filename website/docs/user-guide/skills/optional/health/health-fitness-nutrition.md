---
title: "Fitness Nutrition — Gym workout planner and nutrition tracker"
sidebar_label: "Fitness Nutrition"
description: "Gym workout planner and nutrition tracker"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Fitness Nutrition

Gym workout planner and nutrition tracker. Search 690+ exercises by muscle, equipment, or category via wger. Look up macros and calories for 380,000+ foods via USDA FoodData Central. Compute BMI, TDEE, one-rep max, macro splits, and body fat — pure Python, no pip installs. Built for anyone chasing gains, cutting weight, or just trying to eat better.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/health/fitness-nutrition` |
| Path | `optional-skills/health/fitness-nutrition` |
| Version | `1.0.0` |
| License | MIT |
| Tags | `health`, `fitness`, `nutrition`, `gym`, `workout`, `diet`, `exercise` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Fitness & Nutrition

Expert fitness coach and sports nutritionist skill. Two data sources
plus offline calculators — everything a gym-goer needs in one place.

**Data sources (all free, no pip dependencies):**

- **wger** (https://wger.de/api/v2/) — open exercise database, 690+ exercises with muscles, equipment, images. Public endpoints need zero authentication.
- **USDA FoodData Central** (https://api.nal.usda.gov/fdc/v1/) — US government nutrition database, 380,000+ foods. `DEMO_KEY` works instantly; free signup for higher limits.

**Offline calculators (pure stdlib Python):**

- BMI, TDEE (Mifflin-St Jeor), one-rep max (Epley/Brzycki/Lombardi), macro splits, body fat % (US Navy method)

---

## When to Use

Trigger this skill when the user asks about:
- Exercises, workouts, gym routines, muscle groups, workout splits
- Food macros, calories, protein content, meal planning, calorie counting
- Body composition: BMI, body fat, TDEE, caloric surplus/deficit
- One-rep max estimates, training percentages, progressive overload
- Macro ratios for cutting, bulking, or maintenance

---

## Procedure

### Exercise Lookup (wger API)

All wger public endpoints return JSON and require no auth. Always add
`format=json` and `language=2` (English) to exercise queries.

**Step 1 — Identify what the user wants:**

- By muscle → use `/api/v2/exercise/?muscles={id}&language=2&status=2&format=json`
- By category → use `/api/v2/exercise/?category={id}&language=2&status=2&format=json`
- By equipment → use `/api/v2/exercise/?equipment={id}&language=2&status=2&format=json`
- By name → use `/api/v2/exercise/search/?term={query}&language=english&format=json`
- Full details → use `/api/v2/exerciseinfo/{exercise_id}/?format=json`

**Step 2 — Reference IDs (so you don't need extra API calls):**

Exercise categories:

| ID | Category    |
|----|-------------|
| 8  | Arms        |
| 9  | Legs        |
| 10 | Abs         |
| 11 | Chest       |
| 12 | Back        |
| 13 | Shoulders   |
| 14 | Calves      |
| 15 | Cardio      |

Muscles:

| ID | Muscle                    | ID | Muscle                  |
|----|---------------------------|----|-------------------------|
| 1  | Biceps brachii            | 2  | Anterior deltoid        |
| 3  | Serratus anterior         | 4  | Pectoralis major        |
| 5  | Obliquus externus         | 6  | Gastrocnemius           |
| 7  | Rectus abdominis          | 8  | Gluteus maximus         |
| 9  | Trapezius                 | 10 | Quadriceps femoris      |
| 11 | Biceps femoris            | 12 | Latissimus dorsi        |
| 13 | Brachialis                | 14 | Triceps brachii         |
| 15 | Soleus                    |    |                         |

Equipment:

| ID | Equipment      |
|----|----------------|
| 1  | Barbell        |
| 3  | Dumbbell       |
| 4  | Gym mat        |
| 5  | Swiss Ball     |
| 6  | Pull-up bar    |
| 7  | none (bodyweight) |
| 8  | Bench          |
| 9  | Incline bench  |
| 10 | Kettlebell     |

**Step 3 — Fetch and present results:**

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

### Nutrition Lookup (USDA FoodData Central)

Uses `USDA_API_KEY` env var if set, otherwise falls back to `DEMO_KEY`.
DEMO_KEY = 30 requests/hour. Free signup key = 1,000 requests/hour.

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

### Offline Calculators

Use the helper scripts in `scripts/` for batch operations,
or run inline for single calculations:

- `python3 scripts/body_calc.py bmi <weight_kg> <height_cm>`
- `python3 scripts/body_calc.py tdee <weight_kg> <height_cm> <age> <M|F> <activity 1-5>`
- `python3 scripts/body_calc.py 1rm <weight> <reps>`
- `python3 scripts/body_calc.py macros <tdee_kcal> <cut|maintain|bulk>`
- `python3 scripts/body_calc.py bodyfat <M|F> <neck_cm> <waist_cm> [hip_cm] <height_cm>`

See `references/FORMULAS.md` for the science behind each formula.

---

## Pitfalls

- wger exercise endpoint returns **all languages by default** — always add `language=2` for English
- wger includes **unverified user submissions** — add `status=2` to only get approved exercises
- USDA `DEMO_KEY` has **30 req/hour** — add `sleep 2` between batch requests or get a free key
- USDA data is **per 100g** — remind users to scale to their actual portion size
- BMI does not distinguish muscle from fat — high BMI in muscular people is not necessarily unhealthy
- Body fat formulas are **estimates** (±3-5%) — recommend DEXA scans for precision
- 1RM formulas lose accuracy above 10 reps — use sets of 3-5 for best estimates
- wger's `exercise/search` endpoint uses `term` not `query` as the parameter name

---

## Verification

After running exercise search: confirm results include exercise names, muscle groups, and equipment.
After nutrition lookup: confirm per-100g macros are returned with kcal, protein, fat, carbs.
After calculators: sanity-check outputs (e.g. TDEE should be 1500-3500 for most adults).

---

## Quick Reference

| Task | Source | Endpoint |
|------|--------|----------|
| Search exercises by name | wger | `GET /api/v2/exercise/search/?term=&language=english` |
| Exercise details | wger | `GET /api/v2/exerciseinfo/{id}/` |
| Filter by muscle | wger | `GET /api/v2/exercise/?muscles={id}&language=2&status=2` |
| Filter by equipment | wger | `GET /api/v2/exercise/?equipment={id}&language=2&status=2` |
| List categories | wger | `GET /api/v2/exercisecategory/` |
| List muscles | wger | `GET /api/v2/muscle/` |
| Search foods | USDA | `GET /fdc/v1/foods/search?query=&dataType=Foundation,SR Legacy` |
| Food details | USDA | `GET /fdc/v1/food/{fdcId}` |
| BMI / TDEE / 1RM / macros | offline | `python3 scripts/body_calc.py` |
