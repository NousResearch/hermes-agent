---
title: "Maps — 通过 OpenStreetMap/OSRM 进行地理编码、POI、路线、时区查询"
sidebar_label: "Maps"
description: "通过 OpenStreetMap/OSRM 进行地理编码、POI、路线、时区查询"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Maps

通过 OpenStreetMap/OSRM 进行地理编码、POI、路线、时区查询。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/maps` |
| 版本 | `1.2.0` |
| 作者 | Mibayy |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `maps`, `geocoding`, `places`, `routing`, `distance`, `directions`, `nearby`, `location`, `openstreetmap`, `nominatim`, `overpass`, `osrm` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Maps Skill

使用免费开放数据源的位置智能工具。8 个命令，44 个 POI（兴趣点）分类，零依赖（仅 Python 标准库），无需 API 密钥。

数据来源：OpenStreetMap/Nominatim、Overpass API、OSRM、TimeAPI.io。

本 skill 取代了旧版 `find-nearby` skill —— find-nearby 的所有功能均由下方的 `nearby` 命令覆盖，支持相同的 `--near "<place>"` 快捷方式和多分类查询。

## 使用场景

- 用户发送 Telegram 位置图钉（消息中包含经纬度）→ `nearby`
- 用户需要某地名的坐标 → `search`
- 用户有坐标并想获取地址 → `reverse`
- 用户询问附近的餐厅、医院、药店、酒店等 → `nearby`
- 用户需要驾车/步行/骑行距离或行程时间 → `distance`
- 用户需要两地之间的逐步导航 → `directions`
- 用户需要某位置的时区信息 → `timezone`
- 用户需要在某地理区域内搜索 POI → `area` + `bbox`

## 前置条件

Python 3.8+（仅标准库，无需 pip 安装）。

脚本路径：`~/.hermes/skills/maps/scripts/maps_client.py`

## 命令

```bash
MAPS=~/.hermes/skills/maps/scripts/maps_client.py
```

### search — 地理编码地名

```bash
python3 $MAPS search "Eiffel Tower"
python3 $MAPS search "1600 Pennsylvania Ave, Washington DC"
```

返回：纬度、经度、显示名称、类型、边界框、重要性评分。

### reverse — 坐标转地址

```bash
python3 $MAPS reverse 48.8584 2.2945
```

返回：完整地址分解（街道、城市、州/省、国家、邮政编码）。

### nearby — 按分类查找地点

```bash
# 按坐标（例如来自 Telegram 位置图钉）
python3 $MAPS nearby 48.8584 2.2945 restaurant --limit 10
python3 $MAPS nearby 40.7128 -74.0060 hospital --radius 2000

# 按地址/城市/邮编/地标 —— --near 自动进行地理编码
python3 $MAPS nearby --near "Times Square, New York" --category cafe
python3 $MAPS nearby --near "90210" --category pharmacy

# 多个分类合并为一次查询
python3 $MAPS nearby --near "downtown austin" --category restaurant --category bar --limit 10
```

46 个分类：restaurant、cafe、bar、hospital、pharmacy、hotel、guest_house、
camp_site、supermarket、atm、gas_station、parking、museum、park、school、
university、bank、police、fire_station、library、airport、train_station、
bus_stop、church、mosque、synagogue、dentist、doctor、cinema、theatre、gym、
swimming_pool、post_office、convenience_store、bakery、bookshop、laundry、
car_wash、car_rental、bicycle_rental、taxi、veterinary、zoo、playground、
stadium、nightclub。

每条结果包含：`name`、`address`、`lat`/`lon`、`distance_m`、
`maps_url`（可点击的 Google Maps 链接）、`directions_url`（从搜索点出发的 Google Maps 导航链接），以及可用时的扩展标签 ——
`cuisine`、`hours`（营业时间）、`phone`、`website`。

### distance — 行程距离与时间

```bash
python3 $MAPS distance "Paris" --to "Lyon"
python3 $MAPS distance "New York" --to "Boston" --mode driving
python3 $MAPS distance "Big Ben" --to "Tower Bridge" --mode walking
```

模式：driving（驾车，默认）、walking（步行）、cycling（骑行）。返回道路距离、行程时长及直线距离以供对比。

### directions — 逐步导航

```bash
python3 $MAPS directions "Eiffel Tower" --to "Louvre Museum" --mode walking
python3 $MAPS directions "JFK Airport" --to "Times Square" --mode driving
```

返回带编号的步骤，包含指令、距离、时长、道路名称及操作类型（转弯、出发、到达等）。

### timezone — 坐标对应时区

```bash
python3 $MAPS timezone 48.8584 2.2945
python3 $MAPS timezone 35.6762 139.6503
```

返回时区名称、UTC 偏移量及当前本地时间。

### area — 地点的边界框与面积

```bash
python3 $MAPS area "Manhattan, New York"
python3 $MAPS area "London"
```

返回边界框坐标、宽度/高度（千米）及近似面积。可作为 bbox 命令的输入使用。

### bbox — 在边界框内搜索

```bash
python3 $MAPS bbox 40.75 -74.00 40.77 -73.98 restaurant --limit 20
```

在地理矩形区域内查找 POI。可先使用 `area` 命令获取命名地点的边界框坐标。

## 处理 Telegram 位置图钉

当用户发送位置图钉时，消息中包含 `latitude:` 和 `longitude:` 字段。提取这些字段并直接传入 `nearby`：

```bash
# 用户在 36.17, -115.14 发送了图钉并询问"附近有哪些咖啡馆"
python3 $MAPS nearby 36.17 -115.14 cafe --radius 1500
```

以编号列表形式呈现结果，包含名称、距离及 `maps_url` 字段，使用户在聊天中获得可点击链接。对于"现在是否营业？"的问题，检查 `hours` 字段；若缺失或不明确，请通过 `web_search` 核实，因为 OSM 营业时间由社区维护，不一定是最新的。

## 工作流示例

**"查找斗兽场附近的意大利餐厅"：**
1. `nearby --near "Colosseum Rome" --category restaurant --radius 500`
   —— 一条命令，自动地理编码

**"用户发送了位置图钉，附近有什么？"：**
1. 从 Telegram 消息中提取经纬度
2. `nearby LAT LON cafe --radius 1500`

**"如何从酒店步行到会议中心？"：**
1. `directions "Hotel Name" --to "Conference Center" --mode walking`

**"西雅图市中心有哪些餐厅？"：**
1. `area "Downtown Seattle"` → 获取边界框
2. `bbox S W N E restaurant --limit 30`

## 注意事项

- Nominatim 服务条款：最多 1 次请求/秒（脚本自动处理）
- `nearby` 需要经纬度或 `--near "<address>"` —— 二者必须提供其一
- OSRM 路线规划在欧洲和北美覆盖最佳
- Overpass API 在高峰时段可能较慢；脚本会自动在镜像站之间切换（overpass-api.de → overpass.kumi.systems）
- `distance` 和 `directions` 使用 `--to` 标志指定目的地（非位置参数）
- 若单独使用邮政编码在全球范围内结果模糊，请附上国家/州信息

## 验证

```bash
python3 ~/.hermes/skills/maps/scripts/maps_client.py search "Statue of Liberty"
# 应返回纬度约 40.689，经度约 -74.044

python3 ~/.hermes/skills/maps/scripts/maps_client.py nearby --near "Times Square" --category restaurant --limit 3
# 应返回 Times Square 约 500 米范围内的餐厅列表
```