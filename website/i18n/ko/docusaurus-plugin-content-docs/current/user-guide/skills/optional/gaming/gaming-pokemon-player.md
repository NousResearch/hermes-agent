---
title: "Pokemon Player — 헤드리스 에뮬레이터와 RAM 읽기로 포켓몬 플레이"
sidebar_label: "Pokemon Player"
description: "헤드리스 에뮬레이터와 RAM 읽기로 포켓몬 플레이"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pokemon Player

헤드리스 에뮬레이터와 RAM 읽기로 포켓몬을 플레이합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/gaming/pokemon-player`로 설치 |
| 경로 | `optional-skills/gaming/pokemon-player` |
| 버전 | `1.0.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Pokemon Player

`pokemon-agent` 패키지를 사용한 헤드리스 에뮬레이션으로 포켓몬 게임을 플레이합니다.

## 사용 시점
- 사용자가 "play pokemon", "start pokemon", "pokemon game"이라고 말할 때
- 사용자가 Pokemon Red, Blue, Yellow, FireRed 등을 물어볼 때
- 사용자가 AI가 포켓몬을 플레이하는 모습을 보고 싶어 할 때
- 사용자가 ROM 파일(`.gb`, `.gbc`, `.gba`)을 언급할 때

## 시작 절차

### 1. 최초 설정(clone, venv, install)
저장소는 GitHub의 NousResearch/pokemon-agent입니다. 저장소를 clone한 다음 Python 3.10 이상의 가상 환경을 설정합니다. 속도가 빠른 `uv`를 사용해 venv를 만들고 `pyboy` extra와 함께 패키지를 editable 모드로 설치합니다. `uv`를 사용할 수 없으면 `python3 -m venv` + `pip`으로 대체합니다.

이미 체크아웃된 저장소가 있고(예: `~/pokemon-agent`에 venv가 준비되어 있는 경우) 다시 clone하지 말고 해당 디렉터리로 이동해 `.venv/bin/activate`를 source합니다.

ROM 파일도 필요합니다. 사용자의 ROM 파일을 요청합니다(이전 설정에서 체크아웃 내부의 `roms/pokemon_red.gb`에 이미 있을 수도 있습니다).
ROM 파일을 절대 다운로드하거나 제공하지 마세요 — 항상 사용자에게 요청해야 합니다.

### 2. 게임 서버 시작
venv를 활성화한 상태로 pokemon-agent 디렉터리 안에서 `--rom`으로 ROM을 지정하고 `--port 9876`을 사용해 `pokemon-agent serve`를 실행합니다.
`&`를 사용해 백그라운드에서 실행합니다.
저장된 게임을 재개하려면 저장 이름과 함께 `--load-state`를 추가합니다.
시작 후 4초를 기다린 다음 GET /health로 확인합니다.

### 3. 사용자가 볼 수 있도록 라이브 대시보드 설정
사용자가 브라우저에서 대시보드를 볼 수 있도록 localhost.run을 통한 SSH reverse tunnel을 사용합니다. keyless localhost.run 엔드포인트에서 로컬 포트 9876을 원격 포트 80으로 전달하며 ssh로 연결합니다(`ssh -R 80:localhost:9876 ssh://nokey@localhost.run`). 출력을 로그 파일로 리디렉션하고 10초를 기다린 다음 로그에서 .lhr.life URL을 grep합니다. 사용자에게 `/dashboard/`를 덧붙인 URL을 제공합니다.
터널 URL은 실행할 때마다 바뀝니다 — 재시작했다면 사용자에게 새 URL을 제공합니다.

## 저장 및 불러오기

### 저장할 때
- 게임플레이 15~20턴마다
- 체육관 배틀, 라이벌 조우 또는 위험한 전투 전에는 항상
- 새로운 마을이나 던전에 들어가기 전
- 확신이 없는 행동을 하기 전

### 저장 방법
설명적인 이름으로 POST /save를 호출합니다. 좋은 예:
`before_brock`, `route1_start`, `mt_moon_entrance`, `got_cut`

### 불러오기 방법
저장 이름과 함께 POST /load를 호출합니다.

### 사용 가능한 저장 목록
GET /saves는 저장된 모든 상태를 반환합니다.

### 서버 시작 시 불러오기
서버를 시작할 때 `--load-state` 플래그를 사용해 저장을 자동으로 불러옵니다.
이는 서버 시작 후 API를 통해 불러오는 것보다 빠릅니다.

## 게임플레이 루프

### 1단계: OBSERVE — 상태를 확인하고 스크린샷을 찍기
GET /state로 위치, HP, 배틀, 대화 상태를 확인합니다.
GET /screenshot으로 스크린샷을 찍어 `/tmp/pokemon.png`에 저장한 다음 `vision_analyze`를 사용합니다.
항상 둘 다 수행하세요 — RAM 상태는 수치를 제공하고, vision은 공간 인식을 제공합니다.

### 2단계: ORIENT
- 화면에 대화/텍스트가 표시됨 → 진행
- 배틀 중임 → 싸우거나 도망
- 파티가 다침 → 포켓몬센터로 이동
- 목표 근처임 → 신중하게 이동

### 3단계: DECIDE
우선순위: 대화 > 배틀 > 회복 > 스토리 목표 > 훈련 > 탐험

### 4단계: ACT — 최대 2~4걸음 이동한 후 다시 확인
짧은 action 목록(10~15개가 아니라 2~4개)과 함께 POST /action을 호출합니다.

### 5단계: VERIFY — 모든 이동 시퀀스 후 스크린샷
스크린샷을 찍고 `vision_analyze`를 사용해 의도한 위치로 이동했는지 확인합니다.
이 단계가 가장 중요합니다. vision이 없으면 반드시 길을 잃습니다.

### 6단계: `PKM:` 접두사와 함께 진행 상황을 메모리에 기록

### 7단계: 주기적으로 저장

## 액션 참고
- press_a — 확인, 대화, 선택
- press_b — 취소, 닫기 메뉴
- press_start — 게임 메뉴 열기
- walk_up/down/left/right — 한 칸 이동
- hold_b_N — N프레임 동안 B 누르기(텍스트를 빠르게 넘길 때 사용)
- wait_60 — 약 1초 대기(60프레임)
- a_until_dialog_end — 대화가 끝날 때까지 A를 반복해서 누르기

## 경험에서 얻은 핵심 팁

### VISION을 끊임없이 사용하세요
- 이동 2~4걸음마다 스크린샷을 찍습니다.
- RAM 상태는 위치와 HP는 알려주지만 주변에 무엇이 있는지는 알려주지 않습니다.
- 절벽, 울타리, 표지판, 건물 문, NPC는 스크린샷으로만 볼 수 있습니다.
- vision 모델에 구체적으로 질문합니다: "내 북쪽 한 칸에는 무엇이 있나요?"
- 막혔을 때는 무작위 방향을 시도하기 전에 항상 스크린샷을 찍습니다.

### 워프 전환에는 추가 대기 시간이 필요합니다
문이나 계단을 통과하면 맵 전환 중 화면이 검게 페이드됩니다. 전환이 완료될 때까지 반드시 기다려야 합니다. 문/계단 워프 후 `wait_60` 액션을 2~3개 추가합니다. 기다리지 않으면 위치가 오래된 상태로 읽혀서 아직 이전 맵에 있다고 생각하게 됩니다.

### 건물 출구 함정
건물에서 나오면 문 바로 앞에 나타납니다. 북쪽으로 걸으면 곧바로 다시 들어가게 됩니다. 의도한 방향으로 진행하기 전에 왼쪽이나 오른쪽으로 2칸 항상 비켜서 이동합니다.

### 대화 처리
1세대 텍스트는 글자 단위로 천천히 스크롤됩니다. 대화를 빠르게 넘기려면 B를 120프레임 동안 누른 다음 A를 누릅니다. 필요하면 반복합니다. B를 누르고 있으면 텍스트가 최고 속도로 표시됩니다. 그런 다음 A를 눌러 다음 줄로 넘어갑니다.
`a_until_dialog_end` 액션은 RAM 대화 플래그를 확인하지만 모든 텍스트 상태를 포착하지는 못합니다. 대화가 멈춘 것 같으면 수동으로 `hold_b` + `press_a` 패턴을 사용하고 스크린샷으로 확인합니다.

### 절벽은 일방통행입니다
절벽(작은 낭떠러지)은 아래쪽(남쪽)으로만 뛰어내릴 수 있고, 위쪽(북쪽)으로 올라갈 수는 없습니다. 북쪽으로 가다가 절벽에 막히면 왼쪽이나 오른쪽으로 가서 주변을 돌아갈 틈을 찾아야 합니다. vision을 사용해 틈이 어느 방향에 있는지 확인합니다. vision 모델에 명시적으로 물어보세요.

### 이동 전략
- 한 번에 2~4걸음 이동한 다음 스크린샷으로 위치를 확인합니다.
- 새로운 지역에 들어가면 즉시 스크린샷을 찍어 방향을 파악합니다.
- vision 모델에 "[목적지]로 가려면 어느 방향인가요?"라고 묻습니다.
- 3번 이상 시도해도 막히면 스크린샷을 찍고 완전히 다시 판단합니다.
- 10~15번 이동을 연달아 하지 마세요 — 지나치거나 막힐 수 있습니다.

### 야생 배틀에서 도망치기
배틀 메뉴에서 RUN은 오른쪽 아래에 있습니다. 기본 커서 위치(FIGHT, 왼쪽 위)에서 이동하려면 아래쪽을 누른 다음 오른쪽을 눌러 커서를 RUN으로 옮기고 A를 누릅니다. 텍스트/애니메이션을 빠르게 넘기려면 `hold_b`로 감쌉니다.

### 배틀(FIGHT)
배틀 메뉴에서 FIGHT는 왼쪽 위(기본 커서 위치)에 있습니다. A를 눌러 기술 선택으로 들어가고, 다시 A를 눌러 첫 번째 기술을 사용합니다. 공격 애니메이션과 텍스트를 빠르게 넘기려면 B를 누르고 있습니다.

## 배틀 전략

### 의사결정 트리
1. 잡고 싶은가? → 약화시킨 다음 Poke Ball을 던집니다.
2. 필요 없는 야생 포켓몬인가? → RUN
3. 타입 상성이 유리한가? → 효과가 굉장한 기술을 사용합니다.
4. 유리하지 않은가? → 가장 강한 STAB 기술을 사용합니다.
5. HP가 낮은가? → 교체하거나 Potion을 사용합니다.

### 1세대 타입 상성표(주요 상성)
- Water는 Fire, Ground, Rock에 강함
- Fire는 Grass, Bug, Ice에 강함
- Grass는 Water, Ground, Rock에 강함
- Electric은 Water, Flying에 강함
- Ground는 Fire, Electric, Rock, Poison에 강함
- Psychic은 Fighting, Poison에 강함(1세대에서는 압도적!)

### 1세대 특징
- 특수 능력치는 특수 기술의 공격과 방어를 모두 담당
- Psychic 타입은 매우 강함(Ghost 기술에 버그가 있음)
- 크리티컬 히트는 Speed 능력치에 기반
- Wrap/Bind는 상대가 행동하지 못하게 함
- Focus Energy 버그: 크리티컬 확률을 올리는 대신 낮춤

## 메모리 규칙
| 접두사 | 용도 | 예시 |
|--------|---------|---------|
| PKM:OBJECTIVE | 현재 목표 | Viridian Mart에서 Parcel 받기 |
| PKM:MAP | 이동 지식 | Viridian: mart는 북동쪽 |
| PKM:STRATEGY | 배틀/팀 계획 | Misty 전에 Grass 타입 필요 |
| PKM:PROGRESS | 마일스톤 추적 | 라이벌을 이기고 Viridian으로 이동 중 |
| PKM:STUCK | 막힌 상황 | y=28의 절벽은 오른쪽으로 돌아가기 |
| PKM:TEAM | 팀 메모 | Squirtle Lv6, Tackle + Tail Whip |

## 진행 마일스톤
- 스타터 선택
- Viridian Mart에서 Parcel을 전달하고 Pokedex 받기
- Boulder Badge — Brock (Rock) → Water/Grass 사용
- Cascade Badge — Misty (Water) → Grass/Electric 사용
- Thunder Badge — Lt. Surge (Electric) → Ground 사용
- Rainbow Badge — Erika (Grass) → Fire/Ice/Flying 사용
- Soul Badge — Koga (Poison) → Ground/Psychic 사용
- Marsh Badge — Sabrina (Psychic) → 가장 어려운 체육관
- Volcano Badge — Blaine (Fire) → Water/Ground 사용
- Earth Badge — Giovanni (Ground) → Water/Grass/Ice 사용
- Elite Four → 챔피언!

## 플레이 중지
1. POST /save를 통해 설명적인 이름으로 게임을 저장합니다.
2. `PKM:PROGRESS`로 메모리를 업데이트합니다.
3. 사용자에게 다음과 같이 알립니다: "게임이 [name]으로 저장되었습니다! 계속하려면 'play pokemon'이라고 말하세요."
4. 서버와 터널의 백그라운드 프로세스를 종료합니다.

## 주의 사항
- ROM 파일을 절대 다운로드하거나 제공하지 마세요.
- vision으로 확인하지 않고 4~5개가 넘는 액션을 보내지 마세요.
- 건물에서 나온 후 북쪽으로 가기 전에 항상 옆으로 비켜 이동합니다.
- 문/계단 워프 후에는 항상 `wait_60` x2~3을 추가합니다.
- RAM을 통한 대화 감지는 신뢰할 수 없습니다 — 스크린샷으로 확인합니다.
- 위험한 조우 전에 저장합니다.
- 터널 URL은 재시작할 때마다 바뀝니다.
