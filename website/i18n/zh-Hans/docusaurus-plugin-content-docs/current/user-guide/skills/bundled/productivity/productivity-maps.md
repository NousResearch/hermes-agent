---
title: "Maps — 通过 OpenStreetMap/OSRM 进行地理编码、POI、路线、时区"
sidebar_label: "Maps"
description: "通过 OpenStreetMap/OSRM 进行地理编码、POI、路线、时区"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Maps

通过 OpenStreetMap/OSRM 进行地理编码、POI、路线、时区。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/maps` |
| 版本 | `1.2.0` |
| 作者 | Mibayy |
| 许可证 | MIT |
| 标签 | `地图`, `地理编码`, `地点`, `路线规划`, `距离`, `导航`, `附近`, `位置`, `openstreetmap`, `nominatim`, `overpass`, `osrm` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# 地图技能

使用免费、开放数据源的位置智能。8 个命令、44 个 POI 类别、零依赖（仅 Python 标准库）、无需 API 密钥。

数据源：OpenStreetMap/Nominatim、Overpass API、OSRM、TimeAPI.io。

此技能取代了旧的 `find-nearby` 技能 — find-nearby 的所有功能都由下面的 `nearby` 命令覆盖，具有相同的 `--near "<地点>"` 快捷方式和多类别支持。

## 何时使用

- 用户发送 Telegram 位置图钉（消息中包含纬度/经度）→ `nearby`
- 用户需要地点名称的坐标 → `search`
- 用户有坐标并想要地址 → `reverse`
- 用户询问附近的餐厅、医院、药房、酒店等 → `nearby`
- 用户想要驾车/步行/骑行距离或出行时间 → `distance`
- 用户想要两个地点之间的逐步导航 → `directions`
- 用户想要某个位置的时区信息 → `timezone`
- 用户想要在地理区域内搜索 POI → `area` + `bbox`

## 前提条件

Python 3.8+（仅标准库 — 无需 pip 安装）。

脚本路径：`~/.hermes/skills/maps/scripts/maps_client.py`

## 命令

```bash
MAPS=~/.hermes/skills/maps/scripts/maps_client.py
```

### search — 地理编码地点名称

```bash
python3 $MAPS search "Eiffel Tower"
python3 $MAPS search "1600 Pennsylvania Ave, Washington DC"
```

返回：纬度、经度、显示名称、类型、边界框、重要性分数。

### reverse — 坐标转地址

```bash
python3 $MAPS reverse 48.8584 2.2945
```

返回：完整地址分解（街道、城市、州、国家、邮编）。

### nearby — 按类别查找地点

```bash
# 按坐标（例如来自 Telegram 位置图钉）
python3 $MAPS nearby 48.8584 2.2945 restaurant --limit 10
python3 $MAPS nearby 40.7128 -74.0060 hospital --radius 2000

# 按地址 / 城市 / 邮编 / 地标 — --near 自动地理编码
python3 $MAPS nearby --near "Times Square, New York" --category cafe
python3 $MAPS nearby --near "90210" --category pharmacy

# 多个类别合并到一个查询
python3 $MAPS nearby --near "downtown austin" --category restaurant --category bar --limit 10
```

46 个类别：restaurant、cafe、bar、hospital、pharmacy、hotel、guest_house、camp_site、supermarket、atm、gas_station、parking、museum、park、school、university、bank、police、fire_station、library、airport、train_station、bus_stop、church、mosque、synagogue、dentist、doctor、cinema、theatre、gym、swimming_pool、post_office、convenience_store、bakery、bookshop、laundry、car_wash、car_rental、bicycle_rental、taxi、veterinary、zoo、playground、stadium、nightclub。

每个结果包括：`name`、`address`、`lat`/`lon`、`distance_m`、`maps_url`（可点击的 Google Maps 链接）、`directions_url`（从搜索点的 Google Maps 导航）和可用时的推广标签 — `cuisine`、`hours`（opening_hours）、`phone`、`website`。

### distance — 出行距离和时间

```bash
python3 $MAPS distance "Paris" --to "Lyon"
python3 $MAPS distance "New York" --to "Boston" --mode driving
python3 $MAPS distance "Big Ben" --to "Tower Bridge" --mode walking
```

模式：driving（默认）、walking、cycling。返回道路距离、时长和直线距离用于比较。

### directions — 逐步导航

```bash
python3 $MAPS directions "Eiffel Tower" --to "Louvre Museum" --mode walking
python3 $MAPS directions "JFK Airport" --to "Times Square" --mode driving
```

返回带编号的步骤，包括指令、距离、时长、道路名称和操作类型（转弯、出发、到达等）。

### timezone — 坐标的时区

```bash
python3 $MAPS timezone 48.8584 2.2945
python3 $MAPS timezone 35.6762 139.6503
```

返回时区名称、UTC 偏移和当前本地时间。

### area — 地点的边界框和面积

```bash
python3 $MAPS area "Manhattan, New York"
python3 $MAPS area "London"
```

返回边界框坐标、宽/高（公里）和近似面积。可用作 bbox 命令的输入。

### bbox — 在边界框内搜索

```bash
python3 $MAPS bbox 40.75 -74.00 40.77 -73.98 restaurant --limit 20
```

在地理矩形内查找 POI。先使用 `area` 获取命名地点的边界框坐标。

## 使用 Telegram 位置图钉

当用户发送位置图钉时，消息包含 `latitude:` 和 `longitude:` 字段。提取这些并直接传给 `nearby`：

```bash
# 用户在 36.17, -115.14 发送了图钉并询问"查找附近的咖啡馆"
python3 $MAPS nearby 36.17 -115.14 cafe --radius 1500
```

将结果呈现为编号列表，包含名称、距离和 `maps_url` 字段，以便用户在聊天中获得可点击的链接。对于"现在营业吗？"的问题，检查 `hours` 字段；如果缺失或不清楚，使用 `web_search` 验证，因为 OSM 营业时间是社区维护的，不总是最新的。

## 工作流示例

**"查找罗马斗兽场附近的意大利餐厅"：**
1. `nearby --near "Colosseum Rome" --category restaurant --radius 500`
   — 一个命令，自动地理编码

**"他们发送的位置图钉附近有什么？"：**
1. 从 Telegram 消息中提取纬度/经度
2. `nearby LAT LON cafe --radius 1500`

**"从酒店步行到会议中心怎么走？"：**
1. `directions "酒店名称" --to "会议中心" --mode walking`

**"西雅图市中心有哪些餐厅？"：**
1. `area "Downtown Seattle"` → 获取边界框
2. `bbox S W N E restaurant --limit 30`

## 注意事项

- Nominatim 服务条款：最多 1 请求/秒（脚本自动处理）
- `nearby` 需要纬度/经度或 `--near "<地址>"` — 两者必须提供一个
- OSRM 路由覆盖范围在欧洲和北美最佳
- Overpass API 在高峰时段可能较慢；脚本会自动在镜像之间回退（overpass-api.de → overpass.kumi.systems）
- `distance` 和 `directions` 使用 `--to` 标志指定目的地（非位置参数）
- 如果仅邮编给出模糊的全局结果，请包含国家/州

## 验证

```bash
python3 ~/.hermes/skills/maps/scripts/maps_client.py search "Statue of Liberty"
# 应返回纬度 ~40.689，经度 ~-74.044

python3 ~/.hermes/skills/maps/scripts/maps_client.py nearby --near "Times Square" --category restaurant --limit 3
# 应返回时代广场约 500m 范围内的餐厅列表
```
