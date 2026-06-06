---
title: "Neuroskill Bci"
sidebar_label: "Neuroskill Bci"
description: "실행 중인 NeuroSkill 인스턴스에 연결하여 사용자의 실시간 인지 및 감정 상태(집중, 이완, 기분, 인지 부하, 졸음 등)를 응답에 반영합니다."
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Neuroskill Bci

실행 중인 NeuroSkill 인스턴스에 연결하여 사용자의 실시간 인지 및 감정 상태(집중도, 이완도, 기분, 인지 부하, 졸음, 심박수, HRV, 수면 단계 및 40개 이상의 파생 EXG 점수)를 응답에 반영합니다. BCI 웨어러블(Muse 2/S 또는 OpenBCI)과 로컬에서 실행 중인 NeuroSkill 데스크톱 앱이 필요합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | 선택 사항 — `hermes skills install official/health/neuroskill-bci` 로 설치 |
| Path | `optional-skills/health/neuroskill-bci` |
| Version | `1.0.0` |
| Author | Hermes Agent + Nous Research |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `BCI`, `neurofeedback`, `health`, `focus`, `EEG`, `cognitive-state`, `biometrics`, `neuroskill` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# NeuroSkill BCI 통합

Hermes를 실행 중인 [NeuroSkill](https://neuroskill.com/) 인스턴스에 연결하여 BCI 웨어러블에서 실시간 뇌 및 신체 지표를 읽습니다. 이를 사용하여 인지 상태를 인식한 응답을 제공하고, 개입(intervention)을 제안하며, 시간에 따른 정신적 성과를 추적할 수 있습니다.

> **⚠️ 연구 목적으로만 사용** — NeuroSkill은 오픈 소스 연구 도구입니다. 의료 기기가 아니며 FDA, CE 또는 기타 규제 기관의 승인을 받지 않았습니다. 이 지표를 임상 진단이나 치료용으로 절대 사용하지 마십시오.

전체 지표 참조는 `references/metrics.md`를, 개입 프로토콜은 `references/protocols.md`를, WebSocket/HTTP API는 `references/api.md`를 참조하세요.

---

## 사전 요구 사항

- **Node.js 20 이상** 설치됨 (`node --version`)
- 연결된 BCI 기기로 실행 중인 **NeuroSkill 데스크톱 앱**
- **BCI 하드웨어**: Muse 2, Muse S 또는 OpenBCI (BLE를 통한 4채널 EEG + PPG + IMU)
- `npx neuroskill status`가 오류 없이 데이터를 반환함

### 설정 확인
```bash
node --version                    # 20 이상이어야 함
npx neuroskill status             # 전체 시스템 스냅샷
npx neuroskill status --json      # 기계 판독 가능한 JSON
```

만약 `npx neuroskill status`가 오류를 반환하면 사용자에게 다음을 안내하세요:
- NeuroSkill 데스크톱 앱이 열려 있는지 확인하세요.
- BCI 기기의 전원이 켜져 있고 블루투스로 연결되어 있는지 확인하세요.
- 신호 품질을 확인하세요 — NeuroSkill의 녹색 표시기(전극당 0.7 이상)
- `command not found`가 나타나면, Node.js 20 이상을 설치하세요.

---

## CLI 참조: `npx neuroskill <command>`

모든 명령어는 `--json`(파이프-세이프 원시 JSON) 및 `--full`(사람이 읽을 수 있는 요약 + JSON)을 지원합니다.

| 명령어 | 설명 |
|---------|-------------|
| `status` | 전체 시스템 스냅샷: 기기, 점수, 대역, 비율, 수면, 기록 |
| `session [N]` | 전/후반 추세가 포함된 단일 세션 분석 (0=최근) |
| `sessions` | 모든 날짜에 걸쳐 기록된 모든 세션 목록 |
| `search` | 신경적으로 유사한 과거 순간에 대한 ANN 유사도 검색 |
| `compare` | 지표 델타 및 추세 분석을 포함한 A/B 세션 비교 |
| `sleep [N]` | 분석이 포함된 수면 단계 분류 (Wake/N1/N2/N3/REM) |
| `label "text"` | 현재 순간에 타임스탬프가 지정된 주석(라벨) 생성 |
| `search-labels "query"` | 과거 라벨에 대한 의미론적(semantic) 벡터 검색 |
| `interactive "query"` | 교차 모달 4계층 그래프 검색 (텍스트 → EXG → 라벨) |
| `listen` | 실시간 이벤트 스트리밍 (기본 5초, `--seconds N`으로 설정) |
| `umap` | 세션 임베딩의 3D UMAP 프로젝션 |
| `calibrate` | 캘리브레이션 창을 열고 프로필 시작 |
| `timer` | 집중 타이머 실행 (Pomodoro/Deep Work/Short Focus 프리셋) |
| `notify "title" "body"` | NeuroSkill 앱을 통해 OS 알림 전송 |
| `raw '{json}'` | 서버로의 원시 JSON 패스스루 |

### 전역 플래그
| 플래그 | 설명 |
|------|-------------|
| `--json` | 원시 JSON 출력 (ANSI 없음, 파이프-세이프) |
| `--full` | 사람이 읽을 수 있는 요약 + 색상이 지정된 JSON |
| `--port <N>` | 서버 포트 재정의 (기본값: 자동 검색, 주로 8375) |
| `--ws` | WebSocket 전송 강제 |
| `--http` | HTTP 전송 강제 |
| `--k <N>` | 최근접 이웃 수 (search, search-labels) |
| `--seconds <N>` | listen 지속 시간 (기본값: 5) |
| `--trends` | 세션별 지표 추세 표시 (sessions) |
| `--dot` | Graphviz DOT 출력 (interactive) |

---

## 1. 현재 상태 확인하기

### 실시간 지표 가져오기
```bash
npx neuroskill status --json
```

안정적인 파싱을 위해 **항상 `--json`을 사용하세요**. 기본 출력은 사람이 읽을 수 있도록 색상이 지정된 텍스트입니다.

### 응답의 주요 필드

`scores` 객체에는 모든 실시간 지표가 포함되어 있습니다(별도 표기가 없는 한 0–1 척도):

```jsonc
{
  "scores": {
    "focus": 0.70,           // β / (α + θ) — sustained attention (지속적 주의력)
    "relaxation": 0.40,      // α / (β + θ) — calm wakefulness (차분한 각성 상태)
    "engagement": 0.60,      // active mental investment (적극적인 정신적 투자)
    "meditation": 0.52,      // alpha + stillness + HRV coherence (알파 파 + 정지 상태 + HRV 결합도)
    "mood": 0.55,            // composite from FAA, TAR, BAR (FAA, TAR, BAR에서 합성)
    "cognitive_load": 0.33,  // frontal θ / temporal α · f(FAA, TBR) (전두엽 θ / 측두엽 α · f(FAA, TBR))
    "drowsiness": 0.10,      // TAR + TBR + falling spectral centroid (TAR + TBR + 떨어지는 스펙트럼 중심)
    "hr": 68.2,              // heart rate in bpm (from PPG) (bpm 단위의 심박수 (PPG 기반))
    "snr": 14.3,             // signal-to-noise ratio in dB (dB 단위의 신호 대 잡음비)
    "stillness": 0.88,       // 0–1; 1 = perfectly still (0-1; 1 = 완벽히 정지 상태)
    "faa": 0.042,            // Frontal Alpha Asymmetry (+ = approach) (전두엽 알파 비대칭성 (+ = 접근))
    "tar": 0.56,             // Theta/Alpha Ratio (세타/알파 비율)
    "bar": 0.53,             // Beta/Alpha Ratio (베타/알파 비율)
    "tbr": 1.06,             // Theta/Beta Ratio (ADHD proxy) (세타/베타 비율 (ADHD 프록시))
    "apf": 10.1,             // Alpha Peak Frequency in Hz (Hz 단위의 알파 피크 주파수)
    "coherence": 0.614,      // inter-hemispheric coherence (반구 간 일관성)
    "bands": {
      "rel_delta": 0.28, "rel_theta": 0.18,
      "rel_alpha": 0.32, "rel_beta": 0.17, "rel_gamma": 0.05
    }
  }
}
```

추가 포함 항목: `device` (상태, 배터리, 펌웨어), `signal_quality` (전극당 0–1), `session` (지속 시간, 에포크), `embeddings`, `labels`, `sleep` 요약, 그리고 `history`.

### 출력 해석하기

JSON을 파싱하고 지표를 자연어로 번역합니다. 절대로 원시 숫자만 보고하지 마세요. 항상 그 의미를 함께 전달하세요:

**권장 사항 (DO):**
> "현재 집중도가 0.70으로 꽤 탄탄합니다 — 몰입(flow) 상태 영역이네요. 심박수는 68bpm으로 안정적이고, FAA도 양수라서 좋은 접근 동기를 보여주고 있습니다. 복잡한 일을 처리하기에 아주 좋은 시간입니다."

**금지 사항 (DON'T):**
> "집중: 0.70, 이완: 0.40, HR: 68"

주요 해석 기준값 (전체 가이드는 `references/metrics.md` 참조):
- **Focus > 0.70** → 몰입 상태 영역, 이를 유지하도록 보호할 것
- **Focus < 0.40** → 휴식이나 프로토콜 제안
- **Drowsiness > 0.60** → 피로 경고, 미세 수면(micro-sleep) 위험
- **Relaxation < 0.30** → 스트레스 개입 필요
- **Cognitive Load > 0.70 지속됨** → 마인드 덤프 또는 휴식 필요
- **TBR > 1.5** → 세타파 우세, 실행 통제력 감소
- **FAA < 0** → 위축/부정적 감정 — FAA 재균형 고려
- **SNR < 3 dB** → 신뢰할 수 없는 신호, 전극 위치 재조정 제안

---

## 2. 세션 분석

### 단일 세션 분석
```bash
npx neuroskill session --json         # 최근 세션
npx neuroskill session 1 --json       # 이전 세션
npx neuroskill session 0 --json | jq '{focus: .metrics.focus, trend: .trends.focus}'
```

**전반부 대비 후반부 추세**(`"up"`, `"down"`, `"flat"`)를 포함한 전체 지표를 반환합니다. 이를 사용하여 세션이 어떻게 전개되었는지 설명하세요:

> "집중도가 0.64에서 시작해 끝날 즈음 0.76으로 올랐습니다 — 뚜렷한 상승 추세네요. 인지 부하는 0.38에서 0.28로 떨어졌는데, 이는 자리를 잡아가면서 작업이 더 자동화(능숙화)되었음을 시사합니다."

### 모든 세션 목록
```bash
npx neuroskill sessions --json
npx neuroskill sessions --trends      # 세션별 지표 추세 표시
```

---

## 3. 과거 검색

### 신경 유사도 검색
```bash
npx neuroskill search --json                    # 자동: 마지막 세션, k=5
npx neuroskill search --k 10 --json             # 가장 가까운 이웃 10개
npx neuroskill search --start <UTC> --end <UTC> --json
```

128차원 ZUNA 임베딩에 대한 HNSW 근사 최근접 이웃 검색을 사용하여 신경적으로 유사한 과거의 순간들을 찾습니다. 거리 통계, 시간적 분포(하루 중 시간대), 그리고 일치하는 상위 날짜들을 반환합니다.

사용자가 다음과 같이 질문할 때 이 기능을 사용하세요:
- "마지막으로 이런 상태였던 적이 언제야?"
- "내 최고 집중 세션을 찾아줘"
- "오후에 주로 언제쯤 텐션이 떨어져?"

### 의미론적(Semantic) 라벨 검색
```bash
npx neuroskill search-labels "deep focus" --k 10 --json
npx neuroskill search-labels "stress" --json | jq '[.results[].EXG_metrics.tbr]'
```

벡터 임베딩(Xenova/bge-small-en-v1.5)을 사용하여 라벨 텍스트를 검색합니다. 일치하는 라벨과 라벨링 당시의 관련 EXG 지표를 반환합니다.

### 교차 모달 그래프 검색
```bash
npx neuroskill interactive "deep focus" --json
npx neuroskill interactive "deep focus" --dot | dot -Tsvg > graph.svg
```

4계층 그래프: 쿼리 → 텍스트 라벨 → EXG 포인트 → 주변 라벨. 조정하려면 `--k-text`, `--k-EXG`, `--reach <minutes>`를 사용하세요.

---

## 4. 세션 비교
```bash
npx neuroskill compare --json                   # 자동: 마지막 2개 세션
npx neuroskill compare --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

약 50개의 지표에 대해 절대 변화량, 백분율 변화량 및 방향이 포함된 지표 델타를 반환합니다. 또한 `insights.improved[]` 및 `insights.declined[]` 배열, 두 세션의 수면 단계, 그리고 UMAP 작업 ID를 포함합니다.

맥락에 맞게 비교 결과를 해석하세요 — 델타(변화량) 수치만이 아니라 추세를 언급하세요:
> "어제는 두 번의 강한 집중 블록(오전 10시와 오후 2시)이 있었습니다. 오늘은 오전 11시쯤 시작된 하나의 집중 블록이 계속 진행 중이네요. 오늘은 전반적인 몰입도는 더 높지만, 스트레스가 치솟는 횟수가 더 많았습니다 — 스트레스 지수가 15% 올랐고, FAA가 음수로 떨어지는 빈도가 더 잦았습니다."

```bash
# 개선율(백분율)을 기준으로 지표 정렬
npx neuroskill compare --json | jq '.insights.deltas | to_entries | sort_by(.value.pct) | reverse'
```

---

## 5. 수면 데이터
```bash
npx neuroskill sleep --json                     # 지난 24시간
npx neuroskill sleep 0 --json                   # 최근 수면 세션
npx neuroskill sleep --start <UTC> --end <UTC> --json
```

다음 분석과 함께 에포크(5초 단위 구간)별 수면 단계를 반환합니다:
- **단계 코드**: 0=Wake(각성), 1=N1, 2=N2, 3=N3 (깊은 수면), 4=REM
- **분석**: efficiency_pct(효율 비율), onset_latency_min(입면 지연시간), rem_latency_min(REM 지연시간), bout counts(발생 횟수)
- **건강한 목표치**: N3 15–25%, REM 20–25%, 수면 효율 >85%, 입면 시간 &lt;20분

```bash
npx neuroskill sleep --json | jq '.summary | {n3: .n3_epochs, rem: .rem_epochs}'
npx neuroskill sleep --json | jq '.analysis.efficiency_pct'
```

사용자가 수면, 피로 또는 회복을 언급할 때 이 기능을 사용하세요.

---

## 6. 순간 라벨링하기
```bash
npx neuroskill label "breakthrough"
npx neuroskill label "studying algorithms"
npx neuroskill label "post-meditation"
npx neuroskill label --json "focus block start"   # label_id 반환
```

다음과 같은 경우 순간을 자동으로 라벨링합니다:
- 사용자가 돌파구나 통찰을 얻었다고 보고할 때
- 사용자가 새로운 작업 유형을 시작할 때 (예: "코드 리뷰로 전환 중")
- 사용자가 중요한 프로토콜을 완료했을 때
- 사용자가 현재 순간을 표시해 달라고 요청할 때
- 눈에 띄는 상태 전환이 발생할 때 (몰입 상태로 진입/빠져나옴)

라벨은 데이터베이스에 저장되며, 나중에 `search-labels` 및 `interactive` 명령을 통해 검색할 수 있도록 인덱싱됩니다.

---

## 7. 실시간 스트리밍
```bash
npx neuroskill listen --seconds 30 --json
npx neuroskill listen --seconds 5 --json | jq '[.[] | select(.event == "scores")]'
```

지정된 기간 동안 실시간 WebSocket 이벤트(EXG, PPG, IMU, 점수, 라벨)를 스트리밍합니다. WebSocket 연결이 필요합니다(`--http`로는 사용 불가).

지속적인 모니터링 시나리오나 프로토콜 진행 중 실시간으로 지표 변화를 관찰할 때 이 기능을 사용하세요.

---

## 8. UMAP 시각화
```bash
npx neuroskill umap --json                      # 자동: 마지막 2개 세션
npx neuroskill umap --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

ZUNA 임베딩에 대한 GPU 가속 3D UMAP 프로젝션입니다. `separation_score`는 두 세션이 신경학적으로 얼마나 뚜렷이 구별되는지 나타냅니다:
- **> 1.5** → 세션들이 신경학적으로 뚜렷이 구별됨 (서로 다른 뇌 상태)
- **< 0.5** → 두 세션 전반에 걸쳐 유사한 뇌 상태

---

## 9. 선제적 상태 인식

### 세션 시작 확인
세션 시작 시, 사용자가 기기를 착용하고 있다고 언급하거나 자신의 상태에 대해 묻는 경우 상황에 따라 상태 확인을 실행합니다:
```bash
npx neuroskill status --json
```

간략한 상태 요약을 추가합니다:
> "잠깐 상태를 확인해 보겠습니다: 집중도가 0.62로 형성되고 있고, 이완도는 0.55로 양호하며, FAA가 양수여서 접근 동기가 활성화되어 있습니다. 순조로운 출발인 것 같네요."

### 언제 선제적으로 상태를 언급해야 하는가

다음과 같은 경우에**만** 인지 상태를 언급하세요:
- 사용자가 명시적으로 질문할 때 ("나 지금 어떤 것 같아?", "내 집중력 좀 확인해 줘")
- 사용자가 집중력 저하, 스트레스 또는 피로를 호소할 때
- 중요한 기준값을 초과했을 때 (졸음 > 0.70, 집중도 < 0.30 지속 등)
- 사용자가 인지적으로 힘든 작업을 시작하려 하며 준비 상태를 물어볼 때

몰입 상태를 방해하며 지표를 보고**하지 마세요**. 집중도가 0.75를 넘으면 해당 세션을 보호하세요 — 이럴 땐 침묵하는 것이 올바른 대응입니다.

---

## 10. 프로토콜 제안하기

지표가 필요성을 나타낼 때 `references/protocols.md`에서 프로토콜을 제안하세요. 시작하기 전에 항상 먼저 물어보세요 — 몰입 상태를 절대 방해하지 마세요:

> "지난 15분 동안 집중력이 떨어지고 TBR(세타/베타 비율)이 1.5를 넘어서고 있습니다. 이는 세타파 우세 및 정신적 피로의 신호입니다. 세타-베타 뉴로피드백 앵커(Theta-Beta Neurofeedback Anchor)를 함께 진행해 볼까요? 리드미컬한 카운팅과 호흡을 사용해 세타파를 억제하고 베타파를 높이는 90초짜리 연습입니다."

주요 트리거:
- **Focus < 0.40, TBR > 1.5** → 세타-베타 뉴로피드백 앵커 또는 박스 호흡(Box Breathing)
- **Relaxation < 0.30, 스트레스 지수 높음** → 심장 결합(Cardiac Coherence) 또는 4-7-8 호흡
- **Cognitive Load > 0.70 지속됨** → 인지 부하 해소(마인드 덤프)
- **Drowsiness > 0.60** → 울트라디언 리셋(Ultradian Reset) 또는 웨이크 리셋(Wake Reset)
- **FAA < 0 (음수)** → FAA 재균형(FAA Rebalancing)
- **몰입 상태 (focus > 0.75, engagement > 0.70)** → 절대 방해하지 말 것
- **High stillness(높은 정지 상태) + headache_index(두통 지수)** → 목 릴리스 시퀀스(Neck Release Sequence)
- **Low RMSSD (< 25ms)** → 미주 신경 토닝(Vagal Toning)

---

## 11. 추가 도구

### 집중 타이머
```bash
npx neuroskill timer --json
```
Pomodoro(25/5), Deep Work(50/10), 또는 Short Focus(15/5) 프리셋과 함께 집중 타이머 창을 실행합니다.

### 캘리브레이션
```bash
npx neuroskill calibrate
npx neuroskill calibrate --profile "Eyes Open"
```
캘리브레이션 창을 엽니다. 신호 품질이 나쁘거나 사용자가 개인화된 기준선을 설정하고 싶을 때 유용합니다.

### OS 알림
```bash
npx neuroskill notify "Break Time" "Your focus has been declining for 20 minutes"
```

### 원시 JSON 패스스루
```bash
npx neuroskill raw '{"command":"status"}' --json
```
아직 CLI 하위 명령어에 매핑되지 않은 서버 명령어를 위해 사용합니다.

---

## 오류 처리

| 오류 | 예상 원인 | 해결 방법 |
|-------|-------------|-----|
| `npx neuroskill status` 응답 없음 | NeuroSkill 앱이 실행되지 않음 | NeuroSkill 데스크톱 앱 열기 |
| `device.state: "disconnected"` | BCI 기기가 연결되지 않음 | 블루투스, 기기 배터리 확인 |
| 모든 점수가 0으로 반환됨 | 전극 접촉 불량 | 헤드밴드 위치 재조정, 전극 적시기 |
| `signal_quality` 값이 0.7 미만 | 전극 헐거움 | 착용 상태 조절, 전극 접촉부 청소 |
| SNR < 3 dB | 노이즈가 많은 신호 | 머리 움직임 최소화, 주변 환경 확인 |
| `command not found: npx` | Node.js 미설치 | Node.js 20 이상 설치 |

---

## 상호작용 예시

**"나 지금 상태가 어때?"**
```bash
npx neuroskill status --json
```
→ 집중도, 이완도, 기분, 주목할 만한 비율(FAA, TBR)을 언급하며 점수를 자연스럽게 해석합니다. 지표상으로 필요성이 나타날 때만 조치를 제안합니다.

**"집중이 안 돼"**
```bash
npx neuroskill status --json
```
→ 지표상으로도 확인되는지 점검합니다 (높은 세타, 낮은 베타, 상승하는 TBR, 높은 졸음 수치).
→ 확인될 경우, `references/protocols.md`에서 적절한 프로토콜을 제안합니다.
→ 지표상 문제가 없다면, 신경학적인 이유라기보다는 동기부여의 문제일 수 있습니다.

**"어제랑 오늘 집중도를 비교해 줘"**
```bash
npx neuroskill compare --json
```
→ 단순한 숫자뿐 아니라 추세를 해석합니다. 무엇이 개선되었고 무엇이 나빠졌는지, 그리고 가능한 원인을 언급합니다.

**"내가 마지막으로 몰입 상태였던 게 언제야?"**
```bash
npx neuroskill search-labels "flow" --json
npx neuroskill search --json
```
→ 타임스탬프, 관련된 지표, 그리고 (라벨을 바탕으로) 사용자가 당시 무엇을 하고 있었는지 보고합니다.

**"나 잠은 잘 잤어?"**
```bash
npx neuroskill sleep --json
```
→ 수면 구조(N3%, REM%, 수면 효율)를 보고하고, 건강한 목표치와 비교하며, 문제가 될 만한 부분(높은 각성 에포크 빈도, 낮은 REM 등)을 지적합니다.

**"지금 이 순간을 기록해 줘 — 방금 엄청난 아이디어가 떠올랐어"**
```bash
npx neuroskill label "breakthrough"
```
→ 라벨이 저장되었음을 확인해 줍니다. 현재 상태를 기억할 수 있도록 현재 지표를 덧붙여 주면 좋습니다.

---

## 참고 자료

- [NeuroSkill Paper — arXiv:2603.03212](https://arxiv.org/abs/2603.03212) (Kosmyna & Hauptmann, MIT Media Lab)
- [NeuroSkill Desktop App](https://github.com/NeuroSkill-com/skill) (GPLv3)
- [NeuroLoop CLI Companion](https://github.com/NeuroSkill-com/neuroloop) (GPLv3)
- [MIT Media Lab Project](https://www.media.mit.edu/projects/neuroskill/overview/)
