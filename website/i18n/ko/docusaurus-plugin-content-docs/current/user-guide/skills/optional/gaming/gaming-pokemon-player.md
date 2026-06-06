---
title: "Pokemon Player — 헤드리스 에뮬레이터 + RAM 읽기를 통한 포켓몬 플레이"
sidebar_label: "Pokemon Player"
description: "헤드리스 에뮬레이터 + RAM 읽기를 통한 포켓몬 플레이"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pokemon Player

헤드리스 에뮬레이터와 RAM 읽기를 통해 포켓몬을 플레이합니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/gaming/pokemon-player` |
| Path | `optional-skills/gaming/pokemon-player` |
| Platforms | linux, macos, windows |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Pokemon Player

`pokemon-agent` 패키지를 사용한 헤드리스 에뮬레이션을 통해 포켓몬 게임을 플레이합니다.

## When to Use
- 사용자가 "포켓몬 플레이해 줘", "포켓몬 시작", "포켓몬 게임"이라고 말할 때
- 사용자가 포켓몬 레드, 블루, 옐로우, 파이어레드 등에 대해 물어볼 때
- 사용자가 AI가 포켓몬을 플레이하는 것을 보고 싶어 할 때
- 사용자가 ROM 파일(.gb, .gbc, .gba)을 언급할 때

## Startup Procedure

### 1. First-time setup (clone, venv, install)
리포지토리는 GitHub의 NousResearch/pokemon-agent에 있습니다. 복제한 다음 Python 3.10+ 가상 환경을 설정합니다. 가상 환경을 만들고 pyboy extra와 함께 패키지를 편집 모드로 설치하려면 uv(속도 때문에 권장됨)를 사용합니다. uv를 사용할 수 없는 경우 python3 -m venv + pip를 폴백으로 사용합니다.

이 머신에서는 이미 /home/teknium/pokemon-agent에 준비된 가상 환경과 함께 설정되어 있습니다. 해당 디렉토리로 이동한 후 `source .venv/bin/activate`를 실행하기만 하면 됩니다.

ROM 파일도 필요합니다. 사용자에게 ROM 파일을 요청하세요. 이 머신의 경우 해당 디렉토리 안의 roms/pokemon_red.gb에 존재합니다.
절대로 ROM 파일을 다운로드하거나 제공하지 마십시오. 항상 사용자에게 요청하십시오.

### 2. Start the game server
활성화된 가상 환경이 있는 pokemon-agent 디렉토리 내부에서 ROM을 가리키는 `--rom`과 `--port 9876`으로 `pokemon-agent serve`를 실행합니다. `&`를 사용하여 백그라운드에서 실행합니다.
저장된 게임에서 재개하려면 저장 이름과 함께 `--load-state`를 추가합니다.
시작 후 4초를 기다린 다음 GET /health로 확인합니다.

### 3. Set up live dashboard for user to watch
사용자가 브라우저에서 대시보드를 볼 수 있도록 localhost.run을 통한 SSH 역방향 터널을 사용합니다. 로컬 포트 9876을 nokey@localhost.run의 원격 포트 80으로 전달하여 ssh로 연결합니다. 출력을 로그 파일로 리디렉션하고 10초를 기다린 다음 로그에서 .lhr.life URL을 grep합니다. 사용자에게 끝에 /dashboard/가 추가된 URL을 제공합니다.
터널 URL은 매번 변경됩니다. 다시 시작할 경우 사용자에게 새 URL을 제공하세요.

## Save and Load

### When to save
- 게임 플레이의 매 15-20턴마다
- 체육관 배틀, 라이벌과의 조우, 또는 위험한 전투 전에는 **항상**
- 새로운 마을이나 던전에 들어가기 전
- 당신이 확신할 수 없는 어떤 행동을 하기 전

### How to save
설명적인 이름과 함께 POST /save. 좋은 예:
before_brock, route1_start, mt_moon_entrance, got_cut

### How to load
저장 이름과 함께 POST /load.

### List available saves
GET /saves는 모든 저장된 상태를 반환합니다.

### Loading on server startup
서버를 시작할 때 `--load-state` 플래그를 사용하여 저장을 자동 로드합니다. 이는 시작 후 API를 통해 로드하는 것보다 빠릅니다.

## The Gameplay Loop

### Step 1: OBSERVE — check state AND take a screenshot
위치, HP, 전투, 대화 확인을 위해 GET /state를 사용합니다.
GET /screenshot을 하고 /tmp/pokemon.png로 저장한 다음, vision_analyze를 사용합니다.
항상 이 둘을 모두 수행하세요 — RAM 상태는 수치를 제공하고 비전은 공간 인식을 제공합니다.

### Step 2: ORIENT
- 화면의 대화/텍스트 → 넘기기
- 전투 중 → 싸우거나 도망치기
- 파티원이 다침 → 포켓몬 센터로 향하기
- 목표에 가까움 → 조심스럽게 탐색하기

### Step 3: DECIDE
우선순위: 대화 > 전투 > 치료 > 스토리 목표 > 훈련 > 탐험

### Step 4: ACT — move 2-4 steps max, then re-check
짧은 행동 목록과 함께 POST /action을 사용합니다 (10-15개가 아닌 2-4개의 행동).

### Step 5: VERIFY — screenshot after every move sequence
스크린샷을 찍고 vision_analyze를 사용하여 의도한 대로 이동했는지 확인합니다. 이것이 **가장 중요한 단계**입니다. 비전이 없으면 무조건 길을 잃게 됩니다.

### Step 6: RECORD progress to memory with PKM: prefix

### Step 7: SAVE periodically

## Action Reference
- press_a — 확인, 말하기, 선택
- press_b — 취소, 메뉴 닫기
- press_start — 게임 메뉴 열기
- walk_up/down/left/right — 한 타일 이동
- hold_b_N — N 프레임 동안 B 누르기 (텍스트를 빠르게 넘길 때 사용)
- wait_60 — 약 1초(60 프레임) 대기
- a_until_dialog_end — 대화가 끝날 때까지 A를 반복해서 누르기

## Critical Tips from Experience

### USE VISION CONSTANTLY
- 이동을 2-4번 할 때마다 스크린샷을 찍습니다.
- RAM 상태는 위치와 HP를 알려주지만 주위에 무엇이 있는지는 알려주지 않습니다.
- 턱, 울타리, 표지판, 건물 문, NPC — 스크린샷을 통해서만 볼 수 있습니다.
- 비전 모델에게 "내 캐릭터의 북쪽 한 타일 위에 무엇이 있습니까?"와 같이 구체적인 질문을 하세요.
- 막혔을 때는 무작위 방향으로 시도하기 전에 항상 스크린샷을 찍으세요.

### Warp Transitions Need Extra Wait Time
문이나 계단을 통과할 때, 맵 전환 중에 화면이 검게 페이드 아웃됩니다. 완료될 때까지 반드시 기다려야 합니다. 문이나 계단을 통한 전환 후에는 2-3개의 wait_60 행동을 추가하세요. 기다리지 않으면 위치가 갱신되지 않은 것으로 읽혀서 여전히 이전 맵에 있다고 생각하게 될 것입니다.

### Building Exit Trap
건물을 나올 때 문 바로 **앞**에 나타납니다. 북쪽으로 걸어가면 바로 다시 안으로 들어가게 됩니다. 항상 먼저 왼쪽이나 오른쪽으로 2타일을 걸어 옆으로 비켜선 다음, 원래 가려던 방향으로 진행하세요.

### Dialog Handling
1세대 텍스트는 글자 단위로 느리게 스크롤됩니다. 대화를 빠르게 넘기려면 120 프레임 동안 B를 누른 다음 A를 누릅니다. 필요한 만큼 반복합니다. B를 누르고 있으면 텍스트가 최대 속도로 표시됩니다. 그런 다음 A를 눌러 다음 줄로 넘깁니다.
a_until_dialog_end 행동은 RAM의 대화 플래그를 확인하지만, 이 플래그는 모든 텍스트 상태를 포착하지는 못합니다. 대화가 막힌 것 같으면 수동으로 hold_b + press_a 패턴을 대신 사용하고 스크린샷으로 확인하세요.

### Ledges Are One-Way
턱(작은 절벽 가장자리)은 아래(남쪽)로만 뛰어내릴 수 있고 위(북쪽)로는 결코 오를 수 없습니다. 북쪽으로 향할 때 턱에 막히면, 왼쪽이나 오른쪽으로 가서 틈을 찾아야 합니다. 비전을 사용하여 틈이 어느 방향에 있는지 파악하세요. 비전 모델에게 명시적으로 물어보세요.

### Navigation Strategy
- 한 번에 2-4걸음씩 움직인 다음, 스크린샷을 찍어 위치를 확인합니다.
- 새로운 지역에 들어갈 때는 즉시 스크린샷을 찍어 방향을 잡습니다.
- 비전 모델에게 "[목적지]로 가려면 어느 방향입니까?"라고 묻습니다.
- 3번 이상 시도해도 막히면, 스크린샷을 찍고 상황을 완전히 재평가합니다.
- 10-15개의 이동을 연속으로 스팸처럼 날리지 마세요. 목적지를 지나치거나 갇히게 됩니다.

### Running from Wild Battles
전투 메뉴에서 RUN은 오른쪽 아래에 있습니다. 기본 커서 위치(FIGHT, 왼쪽 위)에서 도달하려면 아래쪽을 누른 다음 오른쪽을 눌러 커서를 RUN으로 이동시키고 A를 누릅니다. 텍스트/애니메이션을 빠르게 넘기기 위해 hold_b로 감쌉니다.

### Battling (FIGHT)
전투 메뉴에서 FIGHT는 왼쪽 위(기본 커서 위치)에 있습니다.
기술 선택으로 들어가려면 A를 누르고, 첫 번째 기술을 사용하려면 다시 A를 누릅니다.
그런 다음 공격 애니메이션과 텍스트를 빠르게 넘기기 위해 B를 누르고 있습니다.

## Battle Strategy

### Decision Tree
1. 잡고 싶은가? → 체력을 약화시킨 다음 몬스터볼을 던집니다.
2. 필요 없는 야생 포켓몬인가? → RUN(도망치기)
3. 상성에서 우위인가? → 효과가 굉장한 기술을 사용합니다.
4. 상성 우위가 없는가? → 자속(STAB) 보정을 받는 가장 강한 기술을 사용합니다.
5. HP가 낮은가? → 교체하거나 상처약을 사용합니다.

### Gen 1 Type Chart (key matchups)
- 물 타입은 불꽃, 땅, 바위 타입에 강합니다.
- 불꽃 타입은 풀, 벌레, 얼음 타입에 강합니다.
- 풀 타입은 물, 땅, 바위 타입에 강합니다.
- 전기 타입은 물, 비행 타입에 강합니다.
- 땅 타입은 불꽃, 전기, 바위, 독 타입에 강합니다.
- 에스퍼 타입은 격투, 독 타입에 강합니다. (1세대에서 지배적입니다!)

### Gen 1 Quirks
- 특수(Special) 능력치 = 특수 공격 기술에 대한 공격 **그리고** 방어 능력치
- 에스퍼 타입이 지나치게 강합니다. (고스트 타입 공격에 버그가 있음)
- 치명타(Critical hits) 확률은 스피드 능력치에 기반합니다.
- 김밥말이(Wrap)/조이기(Bind)는 상대방이 행동하지 못하게 막습니다.
- 기충전(Focus Energy) 버그: 치명타 확률을 높이는 대신 감소시킵니다.

## Memory Conventions
| Prefix | Purpose | Example |
|--------|---------|---------|
| PKM:OBJECTIVE | 현재 목표 | 상록시티 프렌들리숍에서 소포 받기 |
| PKM:MAP | 내비게이션 지식 | 상록시티: 프렌들리숍은 북동쪽 |
| PKM:STRATEGY | 전투/팀 계획 | 이슬이와 싸우기 전에 풀 타입 포켓몬 필요 |
| PKM:PROGRESS | 이정표 추적 | 라이벌 이기고 상록시티로 향하는 중 |
| PKM:STUCK | 막힌 상황 | y=28에 턱이 있음. 우회하려면 오른쪽으로 갈 것 |
| PKM:TEAM | 팀 메모 | 꼬부기 Lv6, 몸통박치기 + 꼬리흔들기 |

## Progression Milestones
- 스타팅 포켓몬 선택
- 상록시티 프렌들리숍의 소포를 배달하고 도감을 받기
- 회색배지 — 웅 (바위) → 물/풀 속성 사용
- 블루배지 — 이슬 (물) → 풀/전기 속성 사용
- 갈색배지 — 마티스 (전기) → 땅 속성 사용
- 무지개배지 — 민화 (풀) → 불꽃/얼음/비행 속성 사용
- 연분홍배지 — 독수 (독) → 땅/에스퍼 속성 사용
- 노랑배지 — 초련 (에스퍼) → 가장 어려운 체육관
- 진홍배지 — 강연 (불꽃) → 물/땅 속성 사용
- 그린배지 — 비주기 (땅) → 물/풀/얼음 속성 사용
- 사천왕 → 챔피언!

## Stopping Play
1. POST /save를 통해 설명이 포함된 이름으로 게임을 저장합니다.
2. PKM:PROGRESS로 메모리를 업데이트합니다.
3. 사용자에게 알립니다: "게임이 [이름](으)로 저장되었습니다! 이어서 하려면 '포켓몬 플레이해 줘'라고 말해주세요."
4. 서버와 터널의 백그라운드 프로세스를 종료(kill)합니다.

## Pitfalls
- **절대로** ROM 파일을 다운로드하거나 제공하지 마십시오.
- 비전(스크린샷)으로 확인하지 않고 4-5개 이상의 동작을 보내지 마십시오.
- 북쪽으로 가기 전에 건물을 나선 직후 항상 옆으로 이동하세요.
- 문이나 계단을 통한 워프 후에는 항상 wait_60을 2-3회 추가하세요.
- RAM을 통한 대화 감지는 신뢰할 수 없습니다 — 스크린샷으로 확인하십시오.
- 위험한 조우(전투) **전에** 저장하십시오.
- 터널 URL은 다시 시작할 때마다 변경됩니다.
