---
title: "健身营养 — 健身房锻炼计划和营养追踪器"
sidebar_label: "健身营养"
description: "健身房锻炼计划和营养追踪器"
---

{/* 此页面由 website/scripts/generate-skill-docs.py 根据 skill 的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# 健身营养

健身房锻炼计划和营养追踪器。通过 wger 按肌肉、器械或类别搜索 690+ 锻炼。通过 USDA FoodData Central 查询 380,000+ 食物的宏量和卡路里。计算 BMI、TDEE、单次最大重复次数、宏量分配和体脂 — 纯 Python，无需 pip 安装。专为追求增肌、减脂或只是想吃得更好的人打造。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/health/fitness-nutrition` 安装 |
| 路径 | `optional-skills/health/fitness-nutrition` |
| 版本 | `1.0.0` |
| 许可证 | MIT |
| 标签 | `health`、`fitness`、`nutrition`、`gym`、`workout`、`diet`、`exercise` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# 健身与营养

专家健身教练和运动营养师技能。两个数据源
加离线计算器 — 健身房爱好者需要的一切在一处。

**数据源（全部免费，无需 pip 依赖）：**

- **wger**（https://wger.de/api/v2/） — 开放的锻炼数据库，690+ 锻炼，包含肌肉、器械、图片。公共端点无需认证。
- **USDA FoodData Central**（https://api.nal.usda.gov/fdc/v1/） — 美国政府营养数据库，380,000+ 食物。`DEMO_KEY` 可立即使用；免费注册可获得更高限制。

**离线计算器（纯 stdlib Python）：**

- BMI、TDEE（Mifflin-St Jeor）、单次最大重复次数（Epley/Brzycki/Lombardi）、宏量分配、体脂%（美国海军方法）

---

## 使用场景

当用户询问以下内容时触发此技能：
- 锻炼、锻炼计划、健身房训练、肌肉群、锻炼分割
- 食物宏量、卡路里、蛋白质含量、膳食计划、卡路里计数
- 身体成分：BMI、体脂、TDEE、热量盈余/赤字
- 单次最大重复次数估算、训练百分比、渐进超负荷
- 减脂、增肌或维持的宏量比例

---

## 步骤

### 锻炼查询（wger API）

所有 wger 公共端点返回 JSON，无需认证。始终添加
`format=json` 和 `language=2`（英语）到锻炼查询。

**步骤 1 — 确定用户想要什么：**

- 按肌肉 → 使用 `/api/v2/exercise/?muscles={id}&language=2&status=2&format=json`
- 按类别 → 使用 `/api/v2/exercise/?category={id}&language=2&status=2&format=json`
- 按器械 → 使用 `/api/v2/exercise/?equipment={id}&language=2&status=2&format=json`
- 按名称 → 使用 `/api/v2/exercise/search/?term={query}&language=english&format=json`
- 完整详情 → 使用 `/api/v2/exerciseinfo/{exercise_id}/?format=json`

**步骤 2 — 参考 ID（这样您不需要额外 API 调用）：**

锻炼类别：

| ID | 类别    |
|----|-------------|
| 8  | 手臂        |
| 9  | 腿部        |
| 10 | 腹部         |
| 11 | 胸部         |
| 12 | 背部         |
| 13 | 肩部         |
| 14 | 小腿      |
| 15 | 有氧运动      |

肌肉：

| ID | 肌肉                    | ID | 肌肉                  |
|----|---------------------------|----|-------------------------|
| 1  | 肱二头肌            | 2  | 前三角肌        |
| 3  | 前锯肌           | 4  | 胸大肌        |
| 5  | 外斜肌         | 6  | 腓肠肌           |
| 7  | 腹直肌         | 8  | 臀大肌         |
| 9  | 斜方肌             | 10 | 股四头肌          |
| 11 | 股二头肌           | 12 | 背阔肌        |
| 13 | 肱肌           | 14 | 肱三头肌         |
| 15 | 比目鱼肌                 |    |                         |

器械：

| ID | 器械      |
|----|----------------|
| 1  | 杠铃        |
| 3  | 哑铃       |
| 4  | 健身房垫        |
| 5  | 瑞士球     |
| 6  | 引体向上杆    |
| 7  | 无（自重） |
| 8  | 卧推凳          |
| 9  | 倾斜凳  |
| 10 | 壶铃     |

**步骤 3 — 获取并呈现结果：**

```bash
# 按名称搜索锻炼
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
# 获取特定锻炼的完整详情
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
# 列出按肌肉、类别或器械过滤的锻炼
# 按需组合过滤器：?muscles=4&equipment=1&language=2&status=2
FILTER="$1"  # 例如 "muscles=4" 或 "category=11" 或 "equipment=3"
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

使用 `USDA_API_KEY` 环境变量（如果设置），否则回退到 `DEMO_KEY`。
DEMO_KEY = 每小时 30 次请求。免费注册密钥 = 每小时 1,000 次请求。

```bash
# 按名称搜索食物
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
# 按 FDC ID 获取详细营养成分
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

使用 `scripts/` 中的辅助脚本进行批量操作，
或内联运行进行单个计算：

- `python3 scripts/body_calc.py bmi <weight_kg> <height_cm>`
- `python3 scripts/body_calc.py tdee <weight_kg> <height_cm> <age> <M|F> <activity 1-5>`
- `python3 scripts/body_calc.py 1rm <weight> <reps>`
- `python3 scripts/body_calc.py macros <tdee_kcal> <cut|maintain|bulk>`
- `python3 scripts/body_calc.py bodyfat <M|F> <neck_cm> <waist_cm> [hip_cm] <height_cm>`

参见 `references/FORMULAS.md` 了解每个公式背后的科学原理。

---

## 陷阱

- wger 锻炼端点**默认返回所有语言** — 始终添加 `language=2` 以获取英语
- wger 包含**未经验证的用户提交** — 添加 `status=2` 仅获取已批准的锻炼
- USDA `DEMO_KEY` 有**每小时 30 次请求** — 在批量请求之间添加 `sleep 2` 或获取免费密钥
- USDA 数据是**每 100g** — 提醒用户按其实际份量缩放
- BMI 不能区分肌肉和脂肪 — 肌肉型人的高 BMI 不一定不健康
- 体脂公式是**估算值**（±3-5%） — 建议 DEXA 扫描以获得精确度
- 1RM 公式在超过 10 次时失去准确性 — 使用 3-5 次组以获得最佳估算
- wger 的 `exercise/search` 端点使用 `term` 而非 `query` 作为参数名

---

## 验证

运行锻炼搜索后：确认结果包含锻炼名称、肌肉群和器械。
运行营养查询后：确认返回每 100g 宏量（kcal、蛋白质、脂肪、碳水）。
运行计算器后：理智检查输出（例如，TDEE 对于大多数成年人应为 1500-3500）。

---

## 快速参考

| 任务 | 来源 | 端点 |
|------|--------|----------|
| 按名称搜索锻炼 | wger | `GET /api/v2/exercise/search/?term=&language=english` |
| 锻炼详情 | wger | `GET /api/v2/exerciseinfo/{id}/` |
| 按肌肉过滤 | wger | `GET /api/v2/exercise/?muscles={id}&language=2&status=2` |
| 按器械过滤 | wger | `GET /api/v2/exercise/?equipment={id}&language=2&status=2` |
| 列出类别 | wger | `GET /api/v2/exercisecategory/` |
| 列出肌肉 | wger | `GET /api/v2/muscle/` |
| 搜索食物 | USDA | `GET /fdc/v1/foods/search?query=&dataType=Foundation,SR Legacy` |
| 食物详情 | USDA | `GET /fdc/v1/food/{fdcId}` |
| BMI / TDEE / 1RM / 宏量 | 离线 | `python3 scripts/body_calc.py` |
