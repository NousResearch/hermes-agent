---
title: "Maps — OpenStreetMap/OSRM을 통한 지오코딩, POI, 경로, 시간대"
sidebar_label: "Maps"
description: "OpenStreetMap/OSRM을 통한 지오코딩, POI, 경로, 시간대"
---

{/* 이 페이지는 스킬의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Maps

OpenStreetMap/OSRM을 통한 지오코딩, POI, 경로, 시간대.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨 (기본 설치) |
| 경로 | `skills/productivity/maps` |
| 버전 | `1.2.0` |
| 작성자 | Mibayy |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `maps`, `geocoding`, `places`, `routing`, `distance`, `directions`, `nearby`, `location`, `openstreetmap`, `nominatim`, `overpass`, `osrm` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되어 있을 때 에이전트가 보는 지침이 바로 이것입니다.
:::

# Maps 스킬

무료 공개 데이터 소스를 활용한 위치 정보입니다. 명령어 8개, POI 카테고리 44개, 종속성 0개 (Python 표준 라이브러리만 사용), API 키가 필요하지 않습니다.

데이터 소스: OpenStreetMap/Nominatim, Overpass API, OSRM, TimeAPI.io.

이 스킬은 기존 `find-nearby` 스킬을 대체합니다. 아래의 `nearby` 명령어가 동일한 `--near "<place>"` 단축 문법과 다중 카테고리 지원을 제공하여 `find-nearby`의 모든 기능을 포함합니다.

## 사용 시점

- 사용자가 Telegram 위치 핀을 보냄 (메시지에 위도/경도가 포함됨) → `nearby`
- 장소 이름의 좌표를 원함 → `search`
- 좌표가 있고 주소를 원함 → `reverse`
- 주변의 레스토랑, 병원, 약국, 호텔 등을 요청함 → `nearby`
- 운전/도보/자전거 거리 또는 이동 시간을 원함 → `distance`
- 두 장소 사이의 단계별 길 안내를 원함 → `directions`
- 위치의 시간대 정보를 원함 → `timezone`
- 지리적 영역 안에서 POI를 검색하려 함 → `area` + `bbox`

## 사전 요구 사항

Python 3.8+ (표준 라이브러리만 사용 — pip 설치 불필요).

스크립트 경로: `~/.hermes/skills/maps/scripts/maps_client.py`

## 명령어

```bash
MAPS=~/.hermes/skills/maps/scripts/maps_client.py
```

### search — 장소 이름 지오코딩

```bash
python3 $MAPS search "Eiffel Tower"
python3 $MAPS search "1600 Pennsylvania Ave, Washington DC"
```

반환값: 위도, 경도, 표시 이름, 유형, 경계 상자, 중요도 점수.

### reverse — 좌표를 주소로 변환

```bash
python3 $MAPS reverse 48.8584 2.2945
```

반환값: 전체 주소 구성 요소 (도로, 도시, 주, 국가, 우편번호).

### nearby — 카테고리별 장소 찾기

```bash
# By coordinates (from a Telegram location pin, for example)
python3 $MAPS nearby 48.8584 2.2945 restaurant --limit 10
python3 $MAPS nearby 40.7128 -74.0060 hospital --radius 2000

# By address / city / zip / landmark — --near auto-geocodes
python3 $MAPS nearby --near "Times Square, New York" --category cafe
python3 $MAPS nearby --near "90210" --category pharmacy

# Multiple categories merged into one query
python3 $MAPS nearby --near "downtown austin" --category restaurant --category bar --limit 10
```

카테고리는 46개입니다: restaurant, cafe, bar, hospital, pharmacy, hotel, guest_house,
camp_site, supermarket, atm, gas_station, parking, museum, park, school,
university, bank, police, fire_station, library, airport, train_station,
bus_stop, church, mosque, synagogue, dentist, doctor, cinema, theatre, gym,
swimming_pool, post_office, convenience_store, bakery, bookshop, laundry,
car_wash, car_rental, bicycle_rental, taxi, veterinary, zoo, playground,
stadium, nightclub.

각 결과에는 `name`, `address`, `lat`/`lon`, `distance_m`,
`maps_url` (클릭 가능한 Google Maps 링크), `directions_url` (검색 지점에서 시작하는 Google Maps 길찾기 링크), 그리고 사용 가능한 경우 다음의 홍보 태그가 포함됩니다 — `cuisine`, `hours` (opening_hours), `phone`, `website`.

### distance — 이동 거리와 시간

```bash
python3 $MAPS distance "Paris" --to "Lyon"
python3 $MAPS distance "New York" --to "Boston" --mode driving
python3 $MAPS distance "Big Ben" --to "Tower Bridge" --mode walking
```

모드: driving (기본값), walking, cycling. 비교를 위해 도로 거리, 소요 시간, 직선거리를 반환합니다.

### directions — 단계별 길 안내

```bash
python3 $MAPS directions "Eiffel Tower" --to "Louvre Museum" --mode walking
python3 $MAPS directions "JFK Airport" --to "Times Square" --mode driving
```

반환값: 안내, 거리, 소요 시간, 도로 이름, 이동 유형 (turn, depart, arrive 등)이 포함된 번호 매겨진 단계.

### timezone — 좌표의 시간대

```bash
python3 $MAPS timezone 48.8584 2.2945
python3 $MAPS timezone 35.6762 139.6503
```

반환값: 시간대 이름, UTC 오프셋, 현재 현지 시간.

### area — 장소의 경계 상자 및 영역

```bash
python3 $MAPS area "Manhattan, New York"
python3 $MAPS area "London"
```

반환값: 경계 상자 좌표, km 단위 너비/높이, 대략적인 면적. `bbox` 명령어의 입력으로 유용합니다.

### bbox — 경계 상자 안에서 검색

```bash
python3 $MAPS bbox 40.75 -74.00 40.77 -73.98 restaurant --limit 20
```

지리적 직사각형 안의 POI를 찾습니다. 장소 이름의 경계 상자 좌표를 얻으려면 먼저 `area`를 사용하세요.

## Telegram 위치 핀 활용

사용자가 위치 핀을 보내면 메시지에 `latitude:` 및 `longitude:` 필드가 포함됩니다. 해당 값을 추출하여 `nearby`에 그대로 전달하세요:

```bash
# User sent a pin at 36.17, -115.14 and asked "find cafes nearby"
python3 $MAPS nearby 36.17 -115.14 cafe --radius 1500
```

결과는 이름, 거리, `maps_url` 필드가 포함된 번호 매겨진 목록으로 제시하여 사용자가 채팅에서 탭 한 번으로 링크를 열 수 있게 하세요. "지금 영업하나요?"라는 질문에는 `hours` 필드를 확인하세요. 없거나 명확하지 않으면 OSM의 영업시간은 커뮤니티가 관리하며 항상 최신이 아닐 수 있으므로 `web_search`로 확인하세요.

## 워크플로 예시

**"콜로세움 근처의 이탈리아 레스토랑을 찾아줘":**
1. `nearby --near "Colosseum Rome" --category restaurant --radius 500`
   — 한 번의 명령으로 자동 지오코딩

**"그들이 보낸 이 위치 핀 근처에는 무엇이 있나요?":**
1. Telegram 메시지에서 위도/경도를 추출
2. `nearby LAT LON cafe --radius 1500`

**"호텔에서 컨퍼런스 센터까지 어떻게 걸어가나요?":**
1. `directions "Hotel Name" --to "Conference Center" --mode walking`

**"시애틀 다운타운에는 어떤 레스토랑이 있나요?":**
1. `area "Downtown Seattle"` → 경계 상자 가져오기
2. `bbox S W N E restaurant --limit 30`

## 주의사항

- Nominatim 서비스 약관: 초당 최대 1개 요청 (스크립트가 자동으로 처리)
- `nearby`에는 위도/경도 또는 `--near "<address>"`가 필요합니다. 둘 중 하나를 제공해야 합니다.
- OSRM 경로 안내 범위는 유럽과 북미에서 가장 좋습니다.
- Overpass API는 사용량이 많은 시간대에 느릴 수 있습니다. 스크립트가 미러 간에 자동으로 대체합니다 (overpass-api.de → overpass.kumi.systems).
- `distance`와 `directions`는 목적지에 `--to` 플래그를 사용합니다 (위치 인수 아님).
- 우편번호만 입력하면 전 세계적으로 결과가 모호할 수 있으므로 국가/주를 포함하세요.

## 검증

```bash
python3 ~/.hermes/skills/maps/scripts/maps_client.py search "Statue of Liberty"
# Should return lat ~40.689, lon ~-74.044

python3 ~/.hermes/skills/maps/scripts/maps_client.py nearby --near "Times Square" --category restaurant --limit 3
# Should return a list of restaurants within ~500m of Times Square
```
