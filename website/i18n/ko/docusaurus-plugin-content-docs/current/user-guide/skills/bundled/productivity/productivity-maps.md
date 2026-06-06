---
title: "Maps — OpenStreetMap/OSRM을 통한 지오코딩, POI, 경로, 시간대"
sidebar_label: "Maps"
description: "OpenStreetMap/OSRM을 통한 지오코딩, POI, 경로, 시간대"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Maps

OpenStreetMap/OSRM을 통한 지오코딩, POI, 경로, 시간대.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/productivity/maps` |
| Version | `1.2.0` |
| Author | Mibayy |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `maps`, `geocoding`, `places`, `routing`, `distance`, `directions`, `nearby`, `location`, `openstreetmap`, `nominatim`, `overpass`, `osrm` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Maps 스킬 (Maps Skill)

무료 오픈 데이터 소스를 사용한 위치 인텔리전스입니다. 8개의 명령어, 44개의 POI(관심 장소) 카테고리, 의존성 없음(Python 표준 라이브러리만 사용), API 키가 필요하지 않습니다.

데이터 소스: OpenStreetMap/Nominatim, Overpass API, OSRM, TimeAPI.io.

이 스킬은 기존 `find-nearby` 스킬을 대체합니다 — find-nearby의 모든 기능은 동일한 `--near "<place>"` 단축키와 다중 카테고리 지원을 갖춘 아래의 `nearby` 명령어로 처리됩니다.

## 사용 시기

- 사용자가 텔레그램 위치 핀(메시지에 위도/경도 포함)을 보낼 때 → `nearby`
- 사용자가 장소 이름의 좌표를 원할 때 → `search`
- 사용자가 좌표를 가지고 있고 주소를 원할 때 → `reverse`
- 사용자가 주변 레스토랑, 병원, 약국, 호텔 등을 요청할 때 → `nearby`
- 사용자가 운전/도보/자전거 거리 또는 이동 시간을 원할 때 → `distance`
- 사용자가 두 장소 사이의 턴바이턴 경로 안내를 원할 때 → `directions`
- 사용자가 특정 위치의 시간대 정보를 원할 때 → `timezone`
- 사용자가 지리적 영역 내에서 POI를 검색하고자 할 때 → `area` + `bbox`

## 사전 요구 사항

Python 3.8 이상 (표준 라이브러리만 필요 — pip 설치가 필요 없음).

스크립트 경로: `~/.hermes/skills/maps/scripts/maps_client.py`

## 명령어 (Commands)

```bash
MAPS=~/.hermes/skills/maps/scripts/maps_client.py
```

### search — 장소 이름 지오코딩

```bash
python3 $MAPS search "Eiffel Tower"
python3 $MAPS search "1600 Pennsylvania Ave, Washington DC"
```

반환값: 위도, 경도, 표시 이름, 유형, 경계 상자(bounding box), 중요도 점수.

### reverse — 좌표를 주소로 변환

```bash
python3 $MAPS reverse 48.8584 2.2945
```

반환값: 전체 주소 세부정보 (거리, 도시, 주, 국가, 우편번호).

### nearby — 카테고리별 장소 찾기

```bash
# 좌표로 찾기 (예: 텔레그램 위치 핀에서)
python3 $MAPS nearby 48.8584 2.2945 restaurant --limit 10
python3 $MAPS nearby 40.7128 -74.0060 hospital --radius 2000

# 주소 / 도시 / 우편번호 / 랜드마크로 찾기 — --near는 자동 지오코딩 됨
python3 $MAPS nearby --near "Times Square, New York" --category cafe
python3 $MAPS nearby --near "90210" --category pharmacy

# 여러 카테고리를 하나의 쿼리로 병합
python3 $MAPS nearby --near "downtown austin" --category restaurant --category bar --limit 10
```

46개 카테고리: restaurant, cafe, bar, hospital, pharmacy, hotel, guest_house, camp_site, supermarket, atm, gas_station, parking, museum, park, school, university, bank, police, fire_station, library, airport, train_station, bus_stop, church, mosque, synagogue, dentist, doctor, cinema, theatre, gym, swimming_pool, post_office, convenience_store, bakery, bookshop, laundry, car_wash, car_rental, bicycle_rental, taxi, veterinary, zoo, playground, stadium, nightclub.

각 결과에는 다음이 포함됩니다: `name`, `address`, `lat`/`lon`, `distance_m`, `maps_url` (클릭 가능한 Google 지도 링크), `directions_url` (검색 지점으로부터의 Google 지도 경로 안내), 그리고 사용 가능한 경우 프로모션 태그들 — `cuisine`, `hours` (영업시간), `phone`, `website`.

### distance — 이동 거리 및 시간

```bash
python3 $MAPS distance "Paris" --to "Lyon"
python3 $MAPS distance "New York" --to "Boston" --mode driving
python3 $MAPS distance "Big Ben" --to "Tower Bridge" --mode walking
```

모드(Modes): driving (기본값), walking, cycling. 도로 기준 거리, 소요 시간, 그리고 비교를 위한 직선 거리를 반환합니다.

### directions — 턴바이턴 내비게이션

```bash
python3 $MAPS directions "Eiffel Tower" --to "Louvre Museum" --mode walking
python3 $MAPS directions "JFK Airport" --to "Times Square" --mode driving
```

지시사항, 거리, 소요 시간, 도로 이름 및 이동 유형(회전, 출발, 도착 등)이 포함된 번호가 매겨진 단계들을 반환합니다.

### timezone — 좌표에 대한 시간대

```bash
python3 $MAPS timezone 48.8584 2.2945
python3 $MAPS timezone 35.6762 139.6503
```

시간대 이름, UTC 오프셋 및 현재 현지 시간을 반환합니다.

### area — 장소에 대한 경계 상자 및 면적

```bash
python3 $MAPS area "Manhattan, New York"
python3 $MAPS area "London"
```

경계 상자 좌표, 폭/높이(km), 대략적인 면적을 반환합니다. bbox 명령어의 입력으로 유용합니다.

### bbox — 경계 상자 내 검색

```bash
python3 $MAPS bbox 40.75 -74.00 40.77 -73.98 restaurant --limit 20
```

지리적 직사각형 내에서 POI를 찾습니다. 이름이 지정된 장소의 경계 상자 좌표를 얻으려면 먼저 `area`를 사용하세요.

## 텔레그램 위치 핀 처리하기

사용자가 위치 핀을 보낼 때 메시지에는 `latitude:` 및 `longitude:` 필드가 포함되어 있습니다. 이 값을 추출하여 `nearby`에 직접 전달하세요:

```bash
# 사용자가 36.17, -115.14에 핀을 보내고 "근처 카페 찾아줘"라고 물었을 때
python3 $MAPS nearby 36.17 -115.14 cafe --radius 1500
```

결과를 이름, 거리 및 `maps_url` 필드가 포함된 번호 매기기 목록으로 표시하여 사용자가 채팅에서 탭하여 열 수 있는 링크를 제공하세요. "지금 열려 있나요?"라는 질문에는 `hours` 필드를 확인하세요; 만약 누락되었거나 불분명한 경우, OSM 영업시간은 커뮤니티에서 유지관리되며 항상 최신 상태가 아닐 수 있으므로 `web_search`를 사용하여 확인하세요.

## 워크플로우 예시

**"콜로세움 근처 이탈리안 레스토랑 찾아줘":**
1. `nearby --near "Colosseum Rome" --category restaurant --radius 500`
   — 명령어 1개, 자동 지오코딩 됨

**"이 사람들이 보낸 위치 핀 근처에 뭐가 있어?":**
1. 텔레그램 메시지에서 위도/경도 추출
2. `nearby LAT LON cafe --radius 1500`

**"호텔에서 컨퍼런스 센터까지 어떻게 걸어가?":**
1. `directions "Hotel Name" --to "Conference Center" --mode walking`

**"시애틀 다운타운에 있는 레스토랑은?":**
1. `area "Downtown Seattle"` → 경계 상자 얻기
2. `bbox S W N E restaurant --limit 30`

## 주의사항 (Pitfalls)

- Nominatim 서비스 약관(ToS): 최대 1 요청/초 (스크립트에서 자동으로 처리됨)
- `nearby`는 위도/경도 또는 `--near "<address>"`를 필요로 합니다 — 둘 중 하나가 필요합니다.
- OSRM 경로 커버리지는 유럽과 북미에서 가장 우수합니다.
- Overpass API는 피크 시간대에 느릴 수 있습니다; 스크립트는 미러 서버 간에 자동으로 폴백합니다 (overpass-api.de → overpass.kumi.systems)
- `distance`와 `directions`는 목적지에 위치 인자 대신 `--to` 플래그를 사용합니다.
- 우편번호만으로 전 세계적으로 모호한 결과가 나오는 경우 국가/주를 포함하세요.

## 검증 (Verification)

```bash
python3 ~/.hermes/skills/maps/scripts/maps_client.py search "Statue of Liberty"
# 반환값: lat ~40.689, lon ~-74.044

python3 ~/.hermes/skills/maps/scripts/maps_client.py nearby --near "Times Square" --category restaurant --limit 3
# Times Square 주변 ~500m 이내의 레스토랑 목록을 반환해야 함
```
