---
title: "健身营养 — 健身房训练计划与营养追踪"
sidebar_label: "健身营养"
description: "健身房训练计划与营养追踪"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 健身营养

健身房训练计划与营养追踪。通过 wger 按肌肉、器械或类别搜索 690+ 个动作。通过 USDA FoodData Central 查询 380,000+ 种食物的宏量营养素和热量。纯 Python 计算 BMI、TDEE、单次最大重量（one-rep max）、宏量分配和体脂率——无需 pip 安装。适合增肌、减脂或只是想吃得更健康的用户。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/health/fitness-nutrition` 安装 |
| 路径 | `optional-skills/health/fitness-nutrition` |
| 版本 | `1.0.0` |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `health`, `fitness`, `nutrition`, `gym`, `workout`, `diet`, `exercise` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# 健身与营养

专业健身教练与运动营养师 skill。两个数据源加上离线计算器——健身者所需的一切尽在其中。

**数据源（全部免费，无 pip 依赖）：**

- **wger** (https://wger.de/api/v2/) — 开放动作数据库，690+ 个动作，含肌肉、器械、图片信息。公开端点无需任何认证。
- **USDA FoodData Central** (https://api.nal.usda.gov/fdc/v1/) — 美国政府营养数据库，380,000+ 种食物。`DEMO_KEY` 可立即使用；免费注册可获得更高请求限额。

**离线计算器（纯标准库 Python）：**

- BMI、TDEE（Mifflin-St Jeor 公式）、单次最大重量（Epley/Brzycki/Lombardi 公式）、宏量分配、体脂率（美国海军方法）

---

## 使用时机

当用户询问以下内容时触发此 skill：
- 动作、训练、健身计划、肌肉群、训练分化
- 食物宏量、热量、蛋白质含量、饮食计划、热量计算
- 身体成分：BMI、体脂率、TDEE、热量盈余/赤字
- 单次最大重量估算、训练百分比、渐进超负荷
- 减脂、增肌或维持期的宏量比例

---

## 操作流程

### 动作查询（wger API）

所有 wger 公开端点返回 JSON，无需认证。动作查询始终添加 `format=json` 和 `language=2`（英语）。

**第一步 — 确认用户需求：**

- 按肌肉 → 使用 `/api/v2/exercise/?muscles={id}&language=2&status=2&format=json`
- 按类别 → 使用 `/api/v2/exercise/?category={id}&language=2&status=2&format=json`
- 按器械 → 使用 `/api/v2/exercise/?equipment={id}&language=2&status=2&format=json`
- 按名称 → 使用 `/api/v2/exercise/search/?term={query}&language=english&format=json`
- 完整详情 → 使用 `/api/v2/exerciseinfo/{exercise_id}/?format=json`

**第二步 — 参考 ID（避免额外 API 调用）：**

动作类别：

| ID | 类别        |
|----|-------------|
| 8  | Arms        |
| 9  | Legs        |
| 10 | Abs         |
| 11 | Chest       |
| 12 | Back        |
| 13 | Shoulders   |
| 14 | Calves      |
| 15 | Cardio      |

肌肉：

| ID | 肌肉                      | ID | 肌肉                    |
|----|---------------------------|----|-------------------------|
| 1  | Biceps brachii            | 2  | Anterior deltoid        |
| 3  | Serratus anterior         | 4  | Pectoralis major        |
| 5  | Obliquus externus         | 6  | Gastrocnemius           |
| 7  | Rectus abdominis          | 8  | Gluteus maximus         |
| 9  | Trapezius                 | 10 | Quadriceps femoris      |
| 11 | Biceps femoris            | 12 | Latissimus dorsi        |
| 13 | Brachialis                | 14 | Triceps brachii         |
| 15 | Soleus                    |    |                         |

器械：

| ID | 器械           |
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

**第三步 — 获取并展示结果：**

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

### 营养查询（USDA FoodData Central）

优先使用 `USDA_API_KEY` 环境变量，否则回退到 `DEMO_KEY`。
DEMO_KEY = 每小时 30 次请求。免费注册密钥 = 每小时 1,000 次请求。

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

### 离线计算器

对批量操作使用 `scripts/` 中的辅助脚本，或内联运行单次计算：

- `python3 scripts/body_calc.py bmi <weight_kg> <height_cm>`
- `python3 scripts/body_calc.py tdee <weight_kg> <height_cm> <age> <M|F> <activity 1-5>`
- `python3 scripts/body_calc.py 1rm <weight> <reps>`
- `python3 scripts/body_calc.py macros <tdee_kcal> <cut|maintain|bulk>`
- `python3 scripts/body_calc.py bodyfat <M|F> <neck_cm> <waist_cm> [hip_cm] <height_cm>`

各公式的科学依据详见 `references/FORMULAS.md`。

---

## 注意事项

- wger 动作端点默认返回**所有语言**——始终添加 `language=2` 以获取英语内容
- wger 包含**未经验证的用户提交内容**——添加 `status=2` 仅获取已审核动作
- USDA `DEMO_KEY` 限制**每小时 30 次请求**——批量请求之间添加 `sleep 2`，或申请免费密钥
- USDA 数据基于 **每 100g**——提醒用户按实际份量换算
- BMI 无法区分肌肉与脂肪——肌肉量大的人 BMI 偏高不一定不健康
- 体脂率公式为**估算值**（误差 ±3-5%）——精确测量建议使用 DEXA 扫描
- 单次最大重量公式在超过 10 次重复时准确性下降——建议使用 3-5 次重复组进行估算
- wger 的 `exercise/search` 端点参数名为 `term` 而非 `query`

---

## 验证

运行动作搜索后：确认结果包含动作名称、肌肉群和器械信息。
营养查询后：确认返回每 100g 的宏量数据，包含 kcal、蛋白质、脂肪、碳水化合物。
计算器运行后：对输出进行合理性检查（例如，大多数成年人的 TDEE 应在 1500-3500 之间）。

---

## 快速参考

| 任务 | 数据源 | 端点 |
|------|--------|----------|
| 按名称搜索动作 | wger | `GET /api/v2/exercise/search/?term=&language=english` |
| 动作详情 | wger | `GET /api/v2/exerciseinfo/{id}/` |
| 按肌肉筛选 | wger | `GET /api/v2/exercise/?muscles={id}&language=2&status=2` |
| 按器械筛选 | wger | `GET /api/v2/exercise/?equipment={id}&language=2&status=2` |
| 列出类别 | wger | `GET /api/v2/exercisecategory/` |
| 列出肌肉 | wger | `GET /api/v2/muscle/` |
| 搜索食物 | USDA | `GET /fdc/v1/foods/search?query=&dataType=Foundation,SR Legacy` |
| 食物详情 | USDA | `GET /fdc/v1/food/{fdcId}` |
| BMI / TDEE / 单次最大重量 / 宏量 | 离线 | `python3 scripts/body_calc.py` |