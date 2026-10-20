---
title: "Neuroskill Bci — NeuroSkill의 실시간 BCI 인지 및 기분 상태 사용"
sidebar_label: "Neuroskill Bci"
description: "NeuroSkill의 실시간 BCI 인지 및 기분 상태 사용"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 skill의 SKILL.md에서 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Neuroskill Bci

NeuroSkill의 실시간 BCI 인지 및 기분 상태를 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| 원본 | 선택 사항 — `hermes skills install official/health/neuroskill-bci`로 설치 |
| 경로 | `optional-skills/health/neuroskill-bci` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent + Nous Research |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `BCI`, `neurofeedback`, `health`, `focus`, `EEG`, `cognitive-state`, `biometrics`, `neuroskill` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침입니다.
:::

# NeuroSkill BCI 통합

실행 중인 [NeuroSkill](https://neuroskill.com/) 인스턴스에 Hermes를 연결하여 BCI 웨어러블에서 실시간 뇌 및 신체 지표를 읽습니다. 이를 통해 인지 상태를 고려한 응답을 제공하고, 개입을 제안하며, 시간에 따른 정신 능력을 추적할 수 있습니다.

> **⚠️ 연구용으로만 사용** — NeuroSkill은 오픈 소스 연구 도구입니다. 의료 기기가 아니며 FDA, CE 또는 어떤 규제 기관의 승인도 받지 않았습니다. 이 지표를 임상 진단이나 치료에 절대 사용하지 마세요.

전체 지표 참조는 `references/metrics.md`, 개입 프로토콜은 `references/protocols.md`, WebSocket/HTTP API는 `references/api.md`를 참조하세요.

---

## 사전 요구 사항

- **Node.js 20+** 설치됨 (`node --version`)
- 연결된 BCI 장치와 함께 **NeuroSkill 데스크톱 앱** 실행 중
- **BCI 하드웨어**: Muse 2, Muse S 또는 OpenBCI (BLE를 통한 4채널 EEG + PPG + IMU)
- `npx neuroskill status`가 오류 없이 데이터를 반환함

### 설정 확인
```bash
node --version                    # Must be 20+
npx neuroskill status             # Full system snapshot
npx neuroskill status --json      # Machine-parseable JSON
```

`npx neuroskill status`가 오류를 반환하면 사용자에게 다음을 안내하세요.
- NeuroSkill 데스크톱 앱이 열려 있는지 확인하세요.
- BCI 장치가 켜져 있고 Bluetooth를 통해 연결되어 있는지 확인하세요.
- 신호 품질을 확인하세요 — NeuroSkill에서 전극별 녹색 표시(≥0.7)를 확인하세요.
- `command not found`가 표시되면 Node.js 20+를 설치하세요.

---

## CLI 참조: `npx neuroskill <command>`

모든 명령은 `--json`(가공되지 않은 JSON, 파이프 안전) 및 `--full`(사람이 읽을 수 있는 요약 + JSON)을 지원합니다.

| 명령 | 설명 |
|---------|-------------|
| `status` | 전체 시스템 스냅샷: 장치, 점수, 대역, 비율, 수면, 기록 |
| `session [N]` | 첫 번째/두 번째 절반의 추세가 포함된 단일 세션 분석 (0=가장 최근) |
| `sessions` | 모든 날짜에 걸쳐 기록된 세션 목록 |
| `search` | 신경적으로 유사한 과거 순간의 ANN 유사도 검색 |
| `compare` | 지표 변화량과 추세 분석을 포함한 A/B 세션 비교 |
| `sleep [N]` | 분석을 포함한 수면 단계 분류 (Wake/N1/N2/N3/REM) |
| `label "text"` | 현재 순간에 타임스탬프 주석 생성 |
| `search-labels "query"` | 과거 라벨에 대한 의미 벡터 검색 |
| `interactive "query"` | 교차 모달 4계층 그래프 검색 (text → EXG → labels) |
| `listen` | 실시간 이벤트 스트리밍 (기본 5초, `--seconds N` 설정) |
| `umap` | 세션 임베딩의 3D UMAP 투영 |
| `calibrate` | 보정 창을 열고 프로필 시작 |
| `timer` | 집중 타이머 실행 (Pomodoro/Deep Work/Short Focus 프리셋) |
| `notify "title" "body"` | NeuroSkill 앱을 통해 OS 알림 전송 |
| `raw '{json}'` | 서버로 가공되지 않은 JSON 전달 |

### 전역 플래그
| 플래그 | 설명 |
|-------------|-------------|
| `--json` | 가공되지 않은 JSON 출력 (ANSI 없음, 파이프 안전) |
| `--full` | 사람이 읽을 수 있는 요약 + 색상화된 JSON |
| `--port <N>` | 서버 포트 재정의 (기본값: 자동 검색, 일반적으로 8375) |
| `--ws` | WebSocket 전송 강제 |
| `--http` | HTTP 전송 강제 |
| `--k <N>` | 최근접 이웃 개수 (search, search-labels) |
| `--seconds <N>` | listen 기간 (기본값: 5) |
| `--trends` | 세션별 지표 추세 표시 (sessions) |
| `--dot` | Graphviz DOT 출력 (interactive) |

---

## 1. 현재 상태 확인

### 실시간 지표 가져오기
```bash
npx neuroskill status --json
```

**항상 `--json`을 사용하세요.** 안정적인 파싱을 위해 필요합니다. 기본 출력은 색상으로 표시된 사람이 읽을 수 있는 텍스트입니다.

### 응답의 주요 필드

`scores` 객체에는 모든 실시간 지표가 포함됩니다(별도 표기가 없는 한 0–1 척도).

```jsonc
{
  "scores": {
    "focus": 0.70,           // β / (α + θ) — sustained attention
    "relaxation": 0.40,      // α / (β + θ) — calm wakefulness
    "engagement": 0.60,      // active mental investment
    "meditation": 0.52,      // alpha + stillness + HRV coherence
    "mood": 0.55,            // composite from FAA, TAR, BAR
    "cognitive_load": 0.33,  // frontal θ / temporal α · f(FAA, TBR)
    "drowsiness": 0.10,      // TAR + TBR + falling spectral centroid
    "hr": 68.2,              // heart rate in bpm (from PPG)
    "snr": 14.3,             // signal-to-noise ratio in dB
    "stillness": 0.88,       // 0–1; 1 = perfectly still
    "faa": 0.042,            // Frontal Alpha Asymmetry (+ = approach)
    "tar": 0.56,             // Theta/Alpha Ratio
    "bar": 0.53,             // Beta/Alpha Ratio
    "tbr": 1.06,             // Theta/Beta Ratio (ADHD proxy)
    "apf": 10.1,             // Alpha Peak Frequency in Hz
    "coherence": 0.614,      // inter-hemispheric coherence
    "bands": {
      "rel_delta": 0.28, "rel_theta": 0.18,
      "rel_alpha": 0.32, "rel_beta": 0.17, "rel_gamma": 0.05
    }
  }
}
```

그 외에도 `device`(상태, 배터리, 펌웨어), `signal_quality`(전극별 0–1), `session`(기간, 에포크), `embeddings`, `labels`, `sleep` 요약 및 `history`가 포함됩니다.

### 출력 해석

JSON을 파싱하고 지표를 자연어로 변환하세요. 원시 숫자만 보고하지 말고 항상 의미를 함께 설명하세요.

**해야 할 예:**
> "현재 집중력이 0.70으로 탄탄합니다 — 몰입 상태에 해당하는 수준입니다. 심박수는 68bpm으로 안정적이고 FAA가 양수여서 접근 동기가 양호함을 시사합니다. 복잡한 일을 처리하기에 좋은 때입니다."

**하지 말아야 할 예:**
> "집중력: 0.70, 이완: 0.40, HR: 68"

주요 해석 임계값(전체 가이드는 `references/metrics.md` 참조):
- **Focus > 0.70** → 몰입 상태 수준, 이 상태를 보호하세요.
- **Focus &lt; 0.40** → 휴식 또는 프로토콜을 제안하세요.
- **Drowsiness > 0.60** → 피로 경고, 미세 수면 위험
- **Relaxation &lt; 0.30** → 스트레스 개입 필요
- **Cognitive Load > 0.70 sustained** → 생각 비우기 또는 휴식
- **TBR > 1.5** → 세타 우세, 실행 통제력 감소
- **FAA &lt; 0** → 철회/부정적 정서 — FAA 재균형 고려
- **SNR &lt; 3 dB** → 신뢰할 수 없는 신호, 전극 위치 조정을 제안하세요.

---

## 2. 세션 분석

### 단일 세션 분석
```bash
npx neuroskill session --json         # most recent session
npx neuroskill session 1 --json       # previous session
npx neuroskill session 0 --json | jq '{focus: .metrics.focus, trend: .trends.focus}'
```

첫 번째 절반과 두 번째 절반의 추세(`"up"`, `"down"`, `"flat"`)를 포함한 전체 지표를 반환합니다. 이를 사용하여 세션이 어떻게 변화했는지 설명하세요.

> "집중력이 0.64에서 시작해 마지막에는 0.76까지 올랐습니다 — 뚜렷한 상승 추세입니다. 인지 부하는 0.38에서 0.28로 감소했으며, 이는 적응하면서 작업이 더 자동화되었음을 시사합니다."

### 모든 세션 나열
```bash
npx neuroskill sessions --json
npx neuroskill sessions --trends      # show per-session metric trends
```

---

## 3. 과거 검색

### 신경 유사도 검색
```bash
npx neuroskill search --json                    # auto: last session, k=5
npx neuroskill search --k 10 --json             # 10 nearest neighbors
npx neuroskill search --start <UTC> --end <UTC> --json
```

128-D ZUNA 임베딩에 대한 HNSW 근사 최근접 이웃 검색을 사용하여 신경적으로 유사한 과거의 순간을 찾습니다. 거리 통계, 시간적 분포(하루 중 시간) 및 가장 많이 일치하는 날짜를 반환합니다.

다음과 같은 요청에 사용하세요.
- "마지막으로 지금과 비슷한 상태였던 때는 언제야?"
- "집중력이 가장 좋았던 세션을 찾아줘"
- "오후에 보통 언제 무너져?"

### 의미 라벨 검색
```bash
npx neuroskill search-labels "deep focus" --k 10 --json
npx neuroskill search-labels "stress" --json | jq '[.results[].EXG_metrics.tbr]'
```

벡터 임베딩(Xenova/bge-small-en-v1.5)을 사용하여 라벨 텍스트를 검색합니다. 라벨이 지정된 시점의 관련 EXG 지표와 함께 일치하는 라벨을 반환합니다.

### 교차 모달 그래프 검색
```bash
npx neuroskill interactive "deep focus" --json
npx neuroskill interactive "deep focus" --dot | dot -Tsvg > graph.svg
```

4계층 그래프: query → text labels → EXG points → nearby labels. 조정하려면 `--k-text`, `--k-EXG`, `--reach <minutes>`를 사용하세요.

---

## 4. 세션 비교
```bash
npx neuroskill compare --json                   # auto: last 2 sessions
npx neuroskill compare --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

약 50개 지표에 대해 절대 변화량, 백분율 변화량 및 방향이 포함된 지표 변화량을 반환합니다. 또한 `insights.improved[]` 및 `insights.declined[]` 배열, 두 세션의 수면 단계 및 UMAP 작업 ID도 포함합니다.

비교는 맥락과 함께 해석하세요 — 단순한 변화량이 아니라 추세를 언급하세요.
> "어제는 오전 10시와 오후 2시에 두 번의 강한 집중 블록이 있었습니다. 오늘은 오전 11시쯤 시작한 한 블록이 아직 진행 중입니다. 오늘 전반적인 몰입도는 더 높지만 스트레스 급증이 더 많았고, 스트레스 지수는 15% 상승했으며 FAA가 음수가 되는 경우가 더 잦았습니다."

```bash
# Sort metrics by improvement percentage
npx neuroskill compare --json | jq '.insights.deltas | to_entries | sort_by(.value.pct) | reverse'
```

---

## 5. 수면 데이터
```bash
npx neuroskill sleep --json                     # last 24 hours
npx neuroskill sleep 0 --json                   # most recent sleep session
npx neuroskill sleep --start <UTC> --end <UTC> --json
```

분석과 함께 에포크별 수면 단계(5초 단위 창)를 반환합니다.
- **단계 코드**: 0=Wake, 1=N1, 2=N2, 3=N3 (deep), 4=REM
- **분석**: efficiency_pct, onset_latency_min, rem_latency_min, bout counts
- **건강한 목표**: N3 15–25%, REM 20–25%, efficiency >85%, onset &lt;20 min

```bash
npx neuroskill sleep --json | jq '.summary | {n3: .n3_epochs, rem: .rem_epochs}'
npx neuroskill sleep --json | jq '.analysis.efficiency_pct'
```

사용자가 수면, 피로 또는 회복을 언급할 때 사용하세요.

---

## 6. 순간 라벨링
```bash
npx neuroskill label "breakthrough"
npx neuroskill label "studying algorithms"
npx neuroskill label "post-meditation"
npx neuroskill label --json "focus block start"   # returns label_id
```

다음과 같은 경우 순간에 자동으로 라벨을 지정하세요.
- 사용자가 돌파구나 통찰을 말할 때
- 사용자가 새로운 작업 유형을 시작할 때(예: "코드 리뷰로 전환")
- 사용자가 중요한 프로토콜을 완료할 때
- 사용자가 현재 순간을 표시해 달라고 요청할 때
- 주목할 만한 상태 전환이 발생할 때(몰입 진입/이탈)

라벨은 데이터베이스에 저장되며 나중에 `search-labels` 및 `interactive` 명령으로 검색할 수 있도록 색인됩니다.

---

## 7. 실시간 스트리밍
```bash
npx neuroskill listen --seconds 30 --json
npx neuroskill listen --seconds 5 --json | jq '[.[] | select(.event == "scores")]'
```

지정된 기간 동안 실시간 WebSocket 이벤트(EXG, PPG, IMU, scores, labels)를 스트리밍합니다. WebSocket 연결이 필요하며(`--http`에서는 사용할 수 없음), 지정된 기간 동안 라이브 WebSocket 이벤트를 스트리밍합니다.

프로토콜 중 지속적인 모니터링을 하거나 실시간으로 지표 변화를 관찰할 때 사용하세요.

---

## 8. UMAP 시각화
```bash
npx neuroskill umap --json                      # auto: last 2 sessions
npx neuroskill umap --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

ZUNA 임베딩의 GPU 가속 3D UMAP 투영입니다. `separation_score`는 두 세션이 신경적으로 얼마나 구별되는지를 나타냅니다.
- **> 1.5** → 세션이 신경적으로 구별됨(서로 다른 뇌 상태)
- **&lt; 0.5** → 두 세션에서 유사한 뇌 상태

---

## 9. 선제적 상태 인식

### 세션 시작 확인
세션 시작 시 사용자가 장치를 착용 중이라고 언급하거나 자신의 상태를 물으면 선택적으로 상태 확인을 실행하세요.
```bash
npx neuroskill status --json
```

간단한 상태 요약을 삽입하세요.
> "빠른 확인: 집중력이 0.62로 올라오고 있고, 이완 상태는 0.55로 좋으며, FAA는 양수입니다 — 접근 동기가 활성화되어 있습니다. 순조로운 시작으로 보입니다."

### 상태를 선제적으로 언급할 때

다음과 같은 경우에만 인지 상태를 언급하세요.
- 사용자가 명시적으로 물을 때("내 상태가 어때?", "집중력을 확인해줘")
- 사용자가 집중의 어려움, 스트레스 또는 피로를 말할 때
- 중요한 임계값을 넘었을 때(졸림 > 0.70, 집중력 &lt; 0.30 지속)
- 사용자가 인지적으로 부담이 큰 일을 하려 하며 준비 상태를 물을 때

몰입 상태를 지표로 방해하지 마세요. 집중력이 > 0.75이면 세션을 보호하세요 — 침묵이 올바른 응답입니다.

---

## 10. 프로토콜 제안

지표가 필요성을 나타내면 `references/protocols.md`의 프로토콜을 제안하세요. 시작하기 전에 항상 물어보세요 — 몰입 상태를 절대 방해하지 마세요.

> "지난 15분 동안 집중력이 떨어지고 TBR이 1.5를 넘어서 상승하고 있습니다 — 세타 우세와 정신적 피로의 징후입니다. Theta-Beta Neurofeedback Anchor를 함께 해볼까요? 리듬 있는 숫자 세기와 호흡을 사용해 세타를 억제하고 베타를 높이는 90초 운동입니다."

주요 트리거:
- **Focus &lt; 0.40, TBR > 1.5** → Theta-Beta Neurofeedback Anchor 또는 Box Breathing
- **Relaxation &lt; 0.30, stress_index high** → Cardiac Coherence 또는 4-7-8 Breathing
- **Cognitive Load > 0.70 sustained** → Cognitive Load Offload (생각 비우기)
- **Drowsiness > 0.60** → Ultradian Reset 또는 Wake Reset
- **FAA &lt; 0 (negative)** → FAA Rebalancing
- **Flow State (focus > 0.75, engagement > 0.70)** → 방해하지 마세요.
- **High stillness + headache_index** → Neck Release Sequence
- **Low RMSSD (&lt; 25ms)** → Vagal Toning

---

## 11. 추가 도구

### 집중 타이머
```bash
npx neuroskill timer --json
```
Pomodoro(25/5), Deep Work(50/10) 또는 Short Focus(15/5) 프리셋으로 집중 타이머 창을 실행합니다.

### 보정
```bash
npx neuroskill calibrate
npx neuroskill calibrate --profile "Eyes Open"
```
보정 창을 엽니다. 신호 품질이 낮거나 사용자가 개인 기준선을 설정하려 할 때 유용합니다.

### OS 알림
```bash
npx neuroskill notify "Break Time" "Your focus has been declining for 20 minutes"
```

### 가공되지 않은 JSON 전달
```bash
npx neuroskill raw '{"command":"status"}' --json
```
아직 CLI 하위 명령으로 매핑되지 않은 모든 서버 명령에 사용합니다.

---

## 오류 처리

| 오류 | 가능한 원인 | 해결 방법 |
|-------|-------------|-----|
| `npx neuroskill status` hangs | NeuroSkill 앱이 실행되지 않음 | NeuroSkill 데스크톱 앱 열기 |
| `device.state: "disconnected"` | BCI 장치가 연결되지 않음 | Bluetooth와 장치 배터리 확인 |
| 모든 점수가 0을 반환함 | 전극 접촉 불량 | 헤드밴드 위치 조정, 전극을 적시기 |
| `signal_quality` 값이 &lt; 0.7 | 전극이 느슨함 | 착용 상태 조정, 전극 접촉부 청소 |
| SNR &lt; 3 dB | 신호에 잡음이 많음 | 머리 움직임 최소화, 환경 확인 |
| `command not found: npx` | Node.js가 설치되지 않음 | Node.js 20+ 설치 |

---

## 상호작용 예시

**"지금 내 상태가 어때?"**
```bash
npx neuroskill status --json
```
→ 집중력, 이완, 기분 및 주목할 만한 비율(FAA, TBR)을 언급하며 점수를 자연스럽게 해석하세요. 지표가 필요성을 나타낼 때만 행동을 제안하세요.

**"집중이 안 돼"**
```bash
npx neuroskill status --json
```
→ 지표가 이를 확인하는지(높은 세타, 낮은 베타, 상승하는 TBR, 높은 졸림) 확인하세요.
→ 확인되면 `references/protocols.md`에서 적절한 프로토콜을 제안하세요.
→ 지표가 정상으로 보이면 문제는 신경학적 원인보다 동기 부여의 문제일 수 있습니다.

**"오늘 집중력을 어제와 비교해줘"**
```bash
npx neuroskill compare --json
```
→ 단순한 숫자가 아니라 추세를 해석하세요. 개선된 점, 저하된 점 및 가능한 원인을 언급하세요.

**"마지막으로 몰입 상태였던 때가 언제야?"**
```bash
npx neuroskill search-labels "flow" --json
npx neuroskill search --json
```
→ 타임스탬프, 관련 지표 및 사용자가 무엇을 하고 있었는지(라벨에서)를 보고하세요.

**"잠은 어땠어?"**
```bash
npx neuroskill sleep --json
```
→ 수면 구조(N3%, REM%, 효율)를 보고하고 건강한 목표와 비교하며, 문제가 있으면 언급하세요(각성 에포크가 많음, REM이 낮음).

**"이 순간을 표시해줘 — 방금 돌파구를 찾았어"**
```bash
npx neuroskill label "breakthrough"
```
→ 라벨이 저장되었다고 확인하세요. 현재 지표를 기록하기 위해 선택적으로 언급할 수 있습니다.

---

## 참조

- [NeuroSkill 논문 — arXiv:2603.03212](https://arxiv.org/abs/2603.03212) (Kosmyna & Hauptmann, MIT Media Lab)
- [NeuroSkill 데스크톱 앱](https://github.com/NeuroSkill-com/skill) (GPLv3)
- [NeuroLoop CLI Companion](https://github.com/NeuroSkill-com/neuroloop) (GPLv3)
- [MIT Media Lab 프로젝트](https://www.media.mit.edu/projects/neuroskill/overview/)
