---
title: "Research Paper Writing — Autoreason 논문 작성: 문헌 검토부터 제출 준비까지"
sidebar_label: "Research Paper Writing"
description: "Autoreason 논문 작성: 문헌 검토부터 제출 준비까지"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Research Paper Writing

Autoreason 논문 작성 파이프라인. 문헌 검토, 실험 실행, 초안 작성 및 템플릿 관리를 다룹니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치됨) |
| 경로 | `skills/research/research-paper-writing` |
| 버전 | `2.0.0` |
| 작성자 | Orchestra Research + Hermes Agent |
| 플랫폼 | linux, macos, windows |
| 태그 | `research`, `writing`, `machine-learning`, `latex`, `academic`, `autoreason` |
| 관련 스킬 | [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv), [`diagramming`](/docs/user-guide/skills/bundled/productivity/productivity-diagramming), [`data-science`](/docs/user-guide/skills/bundled/dev/dev-data-science) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Autoreason Research Paper Writing

이 스킬은 고품질의 머신러닝 연구 논문을 처음부터 끝까지 실행하고 작성하기 위한 방법론을 정의합니다. 백엔드에서 9페이지 논문을 작성하거나(SakanaAI의 AI-Scientist와 유사), 인간 연구자를 위한 반복적인 부조종사(co-pilot) 역할을 하는 헤드리스 자동 파이프라인으로 작동하도록 설계되었습니다.

문헌 검토부터 실험, 초안 작성, 자체 검토, 제출 준비까지 연구 수명 주기의 모든 단계를 다룹니다.

## 핵심 방법론: Autoreason 루프 (The Autoreason Loop)

단일 패스 생성(single-pass generation)은 연구 작업에 실패합니다. 이 스킬은 모든 단계에서 **Autoreason Loop**를 강제합니다:

1. **초안 작성/실행 (Draft/Execute)**: 실험 실행, 증명 작성 또는 섹션 초안 작성
2. **평가 (Evaluate)**: 엄격한 규칙 세트를 기반으로 비판
3. **통합 (Synthesize)**: 가장 좋은 요소를 유지하며 수정
4. **결정 (Decide)**: 합격(pass)할지, 1단계로 루프를 돌지, 아니면 차단(block)하고 사람의 의견을 구할지 결정

이것은 텍스트 생성뿐만 아니라 실험에도 적용됩니다. 실험이 실패하면 버그를 수정하고 다시 시도하세요. 결과가 부정적이면 가설을 수정하세요. 실험이 3번 이상 실패하거나 부정적인 결과를 산출하면, 해당 방향을 차단하고 사람의 의견을 구하세요(headless 모드인 경우 분석으로 방향을 전환하세요).

## 전체 파이프라인 개요

논문 작성 파이프라인은 8단계로 나뉩니다:

| 단계 | 목표 | 주요 도구/스킬 |
|-------|------|----------------|
| **0: 프로젝트 설정 (Project Setup)** | 저장소, 템플릿, 목표 설정 | `terminal`, `write_file` |
| **1: 문헌 검토 (Literature Review)** | 선행 연구 파악, 기준(baselines) 식별 | `arxiv` 스킬, Semantic Scholar API |
| **2: 방법론 설계 (Method Design)** | 알고리즘 공식화, 아키텍처 | `diagramming` 스킬, `execute_code` |
| **3: 실험 계획 (Experiment Planning)** | 평가 프로토콜, 지표(metrics) 정의 | `write_file`, 사용자 의견(user input) |
| **4: 실행 및 분석 (Execution & Analysis)** | 코드 실행, 결과 로그, 통계 | `process`, `terminal`, `data-science` |
| **5: 초안 작성 (Drafting)** | LaTeX 템플릿을 완성된 논문으로 변환 | `delegate_task`, `write_file` |
| **6: 자체 검토 (Self-Review)** | 앙상블 프롬프팅을 사용한 시뮬레이션된 동료 평가 | 여러 모델에서 `delegate_task` |
| **7: 제출 준비 (Submission Prep)** | 린팅, 익명화, 템플릿 변환 | `chktex`, `terminal` |
| **8: 사후 작업 (Post-Acceptance)** | 카메라 레디(camera-ready), arXiv, 포스터, 홍보 | `terminal`, `write_file` |

---

## Phase 0: Setup and Environment

**목표**: 재현성, LaTeX 템플릿 관리 및 조직을 위한 기반을 구축합니다.

### Step 0.1: Directory Structure

이 구조는 코드, 데이터 및 논문 초안을 깨끗하게 분리합니다. 새 프로젝트를 시작할 때 이 계층 구조를 만드세요:

```
project_root/
  README.md                 # 프로젝트 목표, 실행 지침
  requirements.txt          # 또는 환경 설정 파일
  src/                      # 핵심 메소드 구현
    model.py
    data.py
    utils.py
  scripts/                  # 실행 가능한 실험 스크립트
    run_baselines.sh
    train.py
    evaluate.py
  results/                  # 원시 출력, 모델 체크포인트
    logs/
    figures/                # 플롯 생성을 위해 코드가 저장하는 곳
  paper/                    # LaTeX 작업 공간
    main.tex
    references.bib
    figures/                # paper.tex에 포함된 컴파일된 그림
  context/                  # (선택 사항) 에이전트용 문서
    literature_notes.md
    experiment_log.md       # 에이전트 컨텍스트용
```

### Step 0.2: Target Venue Selection

어떤 코드를 작성하기 전에 사용자에게 **대상 학술대회(target venue)**(예: NeurIPS, ICLR, ICML, ACL)를 결정하도록 요청하세요. 이는 다음을 결정합니다:
- 사용할 LaTeX 템플릿 (`templates/` 디렉토리 참조)
- 페이지 제한 (일반적으로 참조 제외 8-9 페이지)
- 평가 기준 (예: ICML은 이론에 강함, ICLR은 표현 학습에 강함)
- 필수 섹션 (예: NeurIPS의 Broader Impacts, ACL의 Limitations)

*참고: 대상 학술대회가 없으면 기본적으로 가장 유연한 NeurIPS 템플릿을 사용하세요.*

### Step 0.3: The "One-Sentence Contribution"

논문은 하나의 명확한 주장(claim)을 증명해야 합니다. 실험을 시작하기 전에 이 주장을 단일 문장으로 작성하세요.

**좋은 예**: "Direct Preference Optimization (DPO) aligns language models to human preferences without requiring a separate reward model or RL optimization loop, achieving comparable performance to PPO with greater stability."
**나쁜 예**: "We explore different ways to fine-tune language models using human preferences and introduce a new method."

이 문장을 `context/contribution.md` 또는 `MEMORY.md`에 작성하세요. 모든 실험은 이 주장을 뒷받침해야 합니다.

---

## Phase 1: Literature Review & Baselines

**목표**: 관련 연구를 파악하고, 방법론적 차이를 파악하며, 복제할 최첨단(SOTA) 기준을 식별합니다.

### Step 1.1: Citation Discovery

`arxiv` 스킬과 Semantic Scholar API를 사용하여 관련 논문을 찾습니다. 단순한 키워드 검색을 넘어 인용 그래프(citation graph)를 따라가세요.

**검색 팁**:
1. 핵심 최신 논문을 찾습니다(예: "Llama 3 technical report").
2. 해당 논문의 arXiv ID 또는 DOI를 사용합니다.
3. Semantic Scholar API를 쿼리하여 *누가 이 논문을 인용했는지*(후속 연구)와 *이 논문이 누구를 인용했는지*(기초 연구)를 찾습니다.

```python
# 에이전트 실행 코드 예시 (Semantic Scholar API)
import requests
paper_id = "arXiv:2307.09288" # Llama 2
# Llama 2를 인용한 영향력 있는 논문 가져오기
url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=title,authors,year,citationCount,influentialCitationCount&limit=10"
response = requests.get(url).json()
# 영향력을 기준으로 정렬
sorted_papers = sorted(response.get('data', []), key=lambda x: x['citingPaper']['influentialCitationCount'], reverse=True)
```

### Step 1.2: Reading and Summarizing

전체 PDF를 읽는 것은 에이전트 컨텍스트 창을 불필요하게 낭비합니다.
1. 초록(abstracts)부터 시작합니다 (`arxiv` 스킬 사용).
2. 가장 관련된 5-10개의 논문에 대해서만 전체 PDF 내용을 추출합니다 (`web_extract` 사용).
3. **중요**: 각각의 관련 논문에 대해 문헌 노트(literature notes)에 2-3 문장 요약을 작성합니다. "이 논문은 X를 수행합니다. 우리의 접근 방식인 Y와는 Z 측면에서 다릅니다."

### Step 1.3: Baseline Identification

논문이 채택되려면 공정하고 강력한 기준(baselines)과 비교해야 합니다. 문헌 검토를 통해 다음을 식별해야 합니다:
- **Heuristic/Simple Baseline**: 가장 어리석고 작동하는 방식 (예: 텍스트에 대한 BM25, 무작위 샘플링). 종종 놀라울 정도로 이기기 어렵습니다.
- **SOTA Baseline**: 최근에 최고 수준의 성능을 발표한 기존 방법.
- **Ablation Baselines**: 제안하는 방법에서 주요 기여를 제거한 변형 (예: "우리 방법(가중치 적용 없음)").

### Step 1.4: Managing references.bib

논문을 찾을 때마다 지속적으로 `.bib` 파일을 업데이트하세요. 수동 생성을 시도하지 마세요 — `arxiv` 스킬(BibTeX 생성 스크립트 포함)을 사용하거나 CrossRef/Semantic Scholar를 쿼리하세요.

```python
# Semantic Scholar DOI를 사용한 신뢰할 수 있는 BibTeX 가져오기
import requests
doi = "10.48550/arXiv.2305.18290" # DPO paper
bibtex = requests.get(f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex").text
with open("paper/references.bib", "a") as f:
    f.write(bibtex + "\n")
```

---

## Phase 2: Method Design

**목표**: 공식화(formalisms), 아키텍처 및 알고리즘 의사 코드(pseudocode)를 완성합니다.

### Step 2.1: Mathematical Formulation

코딩 전에 수학을 명확히 하세요. 입력/출력 공간, 목적 함수, 제약 조건을 정의합니다.
*에이전트 가이드라인*: 노트를 작성할 때 LaTeX 수학 모드(`$E_{x \sim P}[f(x)]$`)를 사용하세요. 나중에 섹션을 초안할 때 바로 복사할 수 있습니다.

### Step 2.2: Algorithm Pseudocode

구현을 안내할 `algorithm2e` 또는 `algorithmicx` 형식의 LaTeX 의사 코드를 작성하세요.
의사 코드는 실제 코드와 종종 차이가 나기 때문에, 실험(Phase 4) 후 논문 초안 작성(Phase 5) 직전에 의사 코드를 *업데이트*하세요.

### Step 2.3: Architecture Diagram (Optional but Recommended)

아키텍처가 관련된 경우 다이어그램을 설계하세요.
- 복잡한 모델의 경우: `diagramming` 스킬을 사용하여 Excalidraw 기반 아키텍처 다이어그램을 생성합니다.
- 간단한 흐름의 경우: `tikz` 라이브러리를 사용하여 LaTeX 노트를 초안합니다 (예제는 Step 5.4 참조).

---

## Phase 3: Experiment Planning

**목표**: 기여 주장을 증명(또는 반증)하는 실험 스위트(experiment suite)를 정의합니다.

### Step 3.1: Hypothesis Mapping

실행하는 모든 실험은 주장을 테스트해야 합니다. 계획 문서(`context/experiment_plan.md`)를 작성하세요:

| 실험 | 테스트할 가설 | 성공의 의미 | 실패의 의미 |
|------------|----------------------|----------------|----------------|
| SOTA 비교 | 우리 방법이 X 벤치마크에서 Y를 능가함 | 주 기여(기본 주장) 검증됨 | 방법이 경쟁력이 없음 (종료 또는 피벗) |
| Ablation: No-Z | Z 구성요소가 성능 향상의 원인임 | 설계 선택이 정당함 | 방법이 불필요하게 복잡함 (Z 제거) |
| Scaling Law | 데이터 크기에 따라 성능이 확장됨 | 방법에 미래 가치가 있음 | 데이터 제한 환경에 국한됨 |

### Step 3.2: Metrics Definition

추적할 지표를 정의합니다. Accuracy, F1과 같은 일차 지표(primary metrics)와 FLOPs, 지연 시간(latency), VRAM 사용량과 같은 이차 지표(secondary metrics)를 포함하세요. 학회 리뷰어들은 계산 효율성 주장을 매우 면밀히 검토합니다.

### Step 3.3: The Empty Tables

결과를 얻기 전에 테이블 *구조*를 설계하세요(이른바 "비어 있는 테이블(Empty Tables)" 원칙). 채워야 할 행과 열이 무엇인지 알면 실험 코드를 작성할 때 무엇을 출력해야 하는지 알 수 있습니다.

```latex
% 비어 있는 대상 테이블 초안
\begin{table}[]
\begin{tabular}{lccc}
\toprule
Method & Accuracy $\uparrow$ & Latency (ms) $\downarrow$ & Memory (GB) $\downarrow$ \\
\midrule
BM25 (Baseline) & ? & ? & ? \\
DPR (SOTA) & ? & ? & ? \\
\textbf{Ours (Full)} & ? & ? & ? \\
\textbf{Ours (w/o X)} & ? & ? & ? \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Phase 4: Execution & Analysis

**목표**: 코드를 실행하고, 출력을 캡처하고, 결과를 논문 작성에 사용할 수 있도록 집계합니다. 에이전트는 여기서 실험을 자율적으로 모니터링해야 합니다.

### Step 4.1: Writing the Code

이 스킬 범위 밖의 도구(예: `subagent-driven-development`, 표준 코딩 스킬)를 사용하여 실험 코드를 작성하세요.
**핵심 요구 사항**:
- 모든 코드는 특정 시드(seed)와 함께 실행 가능해야 합니다.
- 스크립트는 터미널에 데이터를 출력하는 것뿐만 아니라 나중에 파싱할 수 있도록 측정 항목을 JSON/CSV 파일(예: `results/run_baseline_seed42.json`)에 저장해야 합니다.

### Step 4.2: Running the Experiments

단기 실행 스크립트의 경우 `terminal` 또는 `execute_code`를 사용하세요.
장기 실행 훈련 작업의 경우:
1. `process("start", ...)`을 사용하거나 백그라운드에서 스크립트를 시작합니다 (`nohup python train.py > logs/train.log 2>&1 &`).
2. `cronjob` 도구를 설정하여 주기적으로 로그를 모니터링하고 실험이 실패하거나 완료되면 알려줍니다.
   *예시 프롬프트: "매시간마다 `tail -n 20 logs/train.log`를 확인해. loss가 NaN이거나 스크립트가 충돌하면 즉시 경고해. 그렇지 않으면 [SILENT]로 응답해."*

### Step 4.3: Handling Negative Results (The Autoreason Loop in Action)

실험 결과가 좋지 않을 때 에이전트는 어떻게 대처해야 할까요?
1. **버그인가요?** `data-science` 도구를 사용하여 분포를 확인하세요. loss 곡선이 정상인가요? 지표 계산이 정확한가요?
2. **초매개변수(Hyperparameter) 문제인가요?** 작은 그리드 서치(grid search)를 제안합니다.
3. **진정한 부정적 결과인가요?** 가설(Phase 3.1)을 재평가하세요. 방법이 실제로 기준보다 성능이 떨어질 수 있습니다.

*진정한 부정적 결과*가 나타나면:
- 방법을 폐기하지 말고 사용자에게 알려 변경(pivot) 방향(새로운 아키텍처, 다른 작업, 분석 논문으로 변경 등)을 논의하세요.
- 머신러닝 논문은 반드시 기존 SOTA를 이길 필요는 없습니다. (예: "우리 방법이 SOTA보다 성능이 떨어지지만 추론 비용이 10배 저렴합니다" 또는 "이 인기 있는 기술이 특정 조건 하에서 체계적으로 실패하는 이유에 대한 분석" 등도 훌륭한 논문이 될 수 있습니다).

### Step 4.4: Statistical Significance

단일 시드(seed) 실행은 주요 학회에서 더 이상 허용되지 않습니다(LLM API 평가 등 비용이 많이 드는 경우는 제외).
최소 3개(권장 5개)의 시드를 실행하고 평균(mean) 및 표준편차(std dev) 또는 표준오차(std err)를 보고하세요.
에이전트는 `execute_code`를 사용하여 이러한 집계를 직접 수행하고 테이블을 준비해야 합니다.

```python
# 에이전트 분석 스크립트 예시
import json, numpy as np
import scipy.stats as stats

# 기준 vs 제안 방법 결과 로드
baseline = [85.2, 84.8, 85.5, 84.9, 85.1]
ours = [86.5, 86.1, 87.0, 86.4, 86.7]

print(f"Baseline: {np.mean(baseline):.2f} \pm {np.std(baseline):.2f}")
print(f"Ours:     {np.mean(ours):.2f} \pm {np.std(ours):.2f}")

# 통계적 유의성(T-test)
t_stat, p_val = stats.ttest_ind(baseline, ours)
print(f"p-value: {p_val:.4f} (Significant if < 0.05)")
```

### Step 4.5: Creating Figures

논문 품질의 플롯을 만듭니다 (`matplotlib` 사용 시 `scienceplots` 라이브러리를 적극 권장합니다).
결과를 벡터 형식(`.pdf`)으로 저장하세요. 논문 그림에 `.png`나 `.jpg`를 사용하지 마세요(현미경 사진 등 실제 사진 제외).

### Step 4.6: The Experiment Log Bridge

이것은 파이프라인에서 **가장 중요한 단계** 중 하나입니다.
초안 작성으로 넘어가기 전에 `context/experiment_log.md`를 생성하세요. 이 파일은 모든 원시 출력, JSON 결과 및 수백 줄의 스크립트를 초안 작성 시 에이전트가 처리할 수 있는 요약으로 "압축(compress)"합니다.

이 로그에는 다음 내용이 포함되어야 합니다:
- 빈 테이블(Step 3.3)을 실제 데이터(평균 및 분산)로 채운 내용
- P-value 및 통계적 중요성(significance) 설명
- 그림 목록(각 .pdf 경로와 그것이 보여주는 내용에 대한 1-2문장 설명)
- 관찰된 주요 결과 (예: "아블레이션 Z가 성능을 15% 저하시키므로 핵심 구성 요소임을 확인했습니다.")

**왜 이 작업을 하나요?** 초안 작성 시 에이전트에게 50개의 JSON 파일과 복잡한 로그를 제공하면 컨텍스트 한도를 초과하거나 모델이 혼란스러워합니다. 요약된 `experiment_log.md`를 제공하면 명확하고 일관된 초안 작성을 보장합니다.

---

## Phase 5: Drafting the Paper

**목표**: 수집된 연구(Phase 1), 아키텍처(Phase 2) 및 실험 결과(Phase 4)를 대상 학술대회 양식에 맞는 단일 응집력 있는 논문으로 전환합니다.

### Step 5.0: Context Management for Agents
50개가 넘는 실험 파일, 여러 결과 디렉토리 및 광범위한 문헌 노트가 있는 논문 프로젝트는 에이전트의 컨텍스트 윈도우를 쉽게 초과할 수 있습니다. 이를 주도적으로 관리하세요:

**초안 작성 작업당 컨텍스트에 로드할 항목:**

| 초안 작성 작업 | 컨텍스트에 로드 | 로드 금지 |
|---------------|------------------|-------------|
| 서론 작성 | `experiment_log.md`, 기여(contribution) 요약문, 가장 관련성 높은 논문 초록 5-10개 | 원시 결과 JSON, 전체 실험 스크립트, 모든 문헌 노트 |
| 방법론 작성 | 실험 구성(configs), 의사 코드, 아키텍처 설명 | 원시 로그, 다른 실험의 결과 |
| 결과 작성 | `experiment_log.md`, 결과 요약 테이블, 그림(figure) 목록 | 전체 분석 스크립트, 중간 데이터 |
| 관련 연구 작성 | 정리된 인용 노트(Step 1.4 출력), .bib 파일 | 실험 파일, 원시 PDF |
| 수정(Revision) 패스 | 전체 논문 초안, 특정 리뷰어 의견 | 그 외 모든 것 |

**원칙:**
- **`experiment_log.md`가 기본 컨텍스트 다리입니다** — 원시 데이터 파일을 로드하지 않고도 작성에 필요한 모든 것을 요약합니다(Step 4.6 참조).
- 작업을 위임할 때 **한 번에 한 섹션의 컨텍스트만 로드**하세요. 방법론 초안을 작성하는 하위 에이전트는 문헌 검토 노트가 필요하지 않습니다.
- **요약하고, 원시 파일을 포함하지 마세요.** 200줄의 결과 JSON의 경우 10줄의 요약 테이블을 로드하세요. 50페이지 분량의 관련 논문의 경우 5문장의 초록 + 관련성에 대한 2줄의 노트를 로드하세요.
- **매우 큰 프로젝트의 경우**: 사전에 압축된 요약이 있는 `context/` 디렉토리를 만드세요:
  ```
  context/
    contribution.md          # 1 문장
    experiment_summary.md    # 핵심 결과 테이블 (experiment_log.md에서)
    literature_map.md        # 정리된 인용 노트
    figure_inventory.md      # 설명이 포함된 그림 목록
  ```

### The Narrative Principle (서사 원칙)

**가장 중요한 핵심 통찰력**: 논문은 실험의 모음이 아닙니다 — 증거로 뒷받침되는 하나의 명확한 기여를 담은 이야기입니다.

성공적인 모든 ML 논문은 Neel Nanda가 "서사(the narrative)"라고 부르는 것을 중심으로 합니다: 독자가 관심을 갖는 시사점(takeaway)이 있는 짧고, 엄격하며, 증거에 기반한 기술적인 이야기.

**세 가지 기둥 (서론의 끝에서 아주 명확하게 드러나야 합니다):**

| 기둥 | 설명 | 테스트 |
|--------|-------------|------|
| **The What (무엇을)** | 1-3개의 구체적이고 새로운 주장 | 한 문장으로 말할 수 있나요? |
| **The Why (왜)** | 엄격한 경험적 증거 | 실험이 당신의 가설을 대안과 구별해주나요? |
| **The So What (그래서 뭐)** | 독자가 관심을 가져야 하는 이유 | 인정받는 커뮤니티의 문제와 연결되나요? |

**기여를 한 문장으로 말할 수 없다면, 아직 논문이 완성된 것이 아닙니다.**

### The Sources Behind This Guidance (이 지침의 출처)

이 스킬은 최고 수준의 학술대회에서 폭넓게 논문을 발표한 연구자들의 작성 철학을 종합한 것입니다. 작성 철학 레이어는 원래 [Orchestra Research](https://github.com/orchestra-research)에서 `ml-paper-writing` 스킬로 컴파일되었습니다.

| 출처 | 주요 기여 | 링크 |
|--------|-----------------|------|
| **Neel Nanda** (Google DeepMind) | 서사 원칙, What/Why/So What 프레임워크 | [How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers) |
| **Sebastian Farquhar** (DeepMind) | 5문장 초록 공식 | [How to Write ML Papers](https://sebastianfarquhar.com/on-research/2024/11/04/how_to_write_ml_papers/) |
| **Gopen & Swan** | 독자 기대치의 7가지 원칙 | [Science of Scientific Writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf) |
| **Zachary Lipton** | 단어 선택, 모호한 표현(hedging) 제거 | [Heuristics for Scientific Writing](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/) |
| **Jacob Steinhardt** (UC Berkeley) | 정확성, 일관된 용어 | [Writing Tips](https://bounded-regret.ghost.io/) |
| **Ethan Perez** (Anthropic) | 미시적 수준의 명확성 팁 | [Easy Paper Writing Tips](https://ethanperez.net/easy-paper-writing-tips/) |
| **Andrej Karpathy** | 단일 기여에 초점 맞추기 | 각종 강의 |

**이러한 내용에 대한 심층 분석을 보려면 다음을 참조하세요:**
- [references/writing-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/writing-guide.md) — 예시와 함께 전체 설명
- [references/sources.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/sources.md) — 전체 참고 문헌

### Time Allocation (시간 분배)

다음 각각에 대해 대략 **동일한 시간**을 할애하세요:
1. 초록 (Abstract)
2. 서론 (Introduction)
3. 그림 (Figures)
4. 그 외 나머지 부분 합친 것

**왜일까요?** 대부분의 리뷰어는 방법론(methods)에 도달하기 전에 판단을 내립니다. 독자는 당신의 논문을 다음과 같은 순서로 만납니다: 제목 → 초록 → 서론 → 그림 → 어쩌면 그 나머지.

### Writing Workflow (작성 워크플로우)

```
논문 작성 체크리스트:
- [ ] Step 1: 한 문장 기여(contribution) 정의
- [ ] Step 2: Figure 1 초안 작성 (핵심 아이디어 또는 가장 매력적인 결과)
- [ ] Step 3: 초록 초안 작성 (5문장 공식)
- [ ] Step 4: 서론 초안 작성 (최대 1-1.5 페이지)
- [ ] Step 5: 방법론 초안 작성
- [ ] Step 6: 실험 및 결과 초안 작성
- [ ] Step 7: 관련 연구 초안 작성
- [ ] Step 8: 결론 및 토의 초안 작성
- [ ] Step 9: 한계점 초안 작성 (모든 학술대회에서 필수)
- [ ] Step 10: 부록 계획 (증명, 추가 실험, 세부 정보)
- [ ] Step 11: 논문 체크리스트 완료
- [ ] Step 12: 최종 검토
```

### Two-Pass Refinement Pattern (2단계 정제 패턴)

AI 에이전트와 함께 초안을 작성할 때는 **두 번의 패스(two-pass)** 접근 방식(SakanaAI의 AI-Scientist 파이프라인에서 효과가 입증됨)을 사용하세요:

**패스 1 — 작성 + 섹션별 즉각적인 정제:**
각 섹션에 대해 전체 초안을 작성한 다음 동일한 컨텍스트 내에서 즉시 정제합니다. 섹션이 생생할 때 국지적인 문제(명확성, 흐름, 완전성)를 잡습니다.

**패스 2 — 전체 논문 컨텍스트를 사용한 전역 정제:**
모든 섹션의 초안이 작성되면 전체 논문에 대한 인식을 바탕으로 각 섹션을 다시 검토합니다. 다른 섹션과 관련된 문제(중복, 일관성 없는 용어, 서사 흐름, 한 섹션에서 약속한 내용을 다른 섹션에서 제공하지 않는 문제)를 잡습니다.

```
두 번째 패스 정제 프롬프트 (섹션별):
"전체 논문의 컨텍스트에서 [섹션]을 검토하세요.
- 논문의 나머지 부분과 잘 어울립니까? 다른 섹션과 중복되는 부분이 있습니까?
- 용어가 서론 및 방법론과 일치합니까?
- 메시지를 약화시키지 않고 생략할 수 있는 부분이 있습니까?
- 이전 섹션에서 다음 섹션으로 서사가 잘 흘러갑니까?
최소한의 타겟팅된 편집만 수행하세요. 처음부터 다시 작성하지 마세요."
```

### LaTeX Error Checklist (LaTeX 오류 체크리스트)

모든 정제 프롬프트에 이 체크리스트를 추가하세요. LLM이 LaTeX를 작성할 때 가장 흔히 범하는 오류입니다:

```
LaTeX 품질 체크리스트 (모든 편집 후 확인):
- [ ] 닫히지 않은 수학 기호 없음 ($ 기호 짝 맞춤)
- [ ] 존재하는 그림/표만 참조 (\ref가 \label과 일치)
- [ ] 조작된 인용 없음 (\cite가 .bib의 항목과 일치)
- [ ] 모든 \begin{env}에는 일치하는 \end{env}가 있음 (특히 그림, 표, 알고리즘)
- [ ] HTML 오염 없음 (\end{figure} 대신 </end{figure}> 사용 금지)
- [ ] 수식 모드 외부에서 이스케이프되지 않은 밑줄(_) 없음 (텍스트에서는 \_ 사용)
- [ ] 중복된 \label 정의 없음
- [ ] 중복된 섹션 헤더 없음
- [ ] 텍스트의 숫자가 실제 실험 결과와 일치
- [ ] 모든 그림에 캡션과 레이블이 있음
- [ ] overfull hbox 경고를 유발하는 너무 긴 줄이 없음
```

### Step 5.0: Title (제목)

제목은 논문에서 가장 많이 읽히는 요소입니다. 누군가가 초록을 클릭해 읽어볼지 여부를 결정합니다.

**좋은 제목**:
- 기여나 발견을 명시: "Autoreason: When Iterative LLM Refinement Works and Why It Fails"
- 놀라운 결과를 강조: "Scaling Data-Constrained Language Models" (할 수 있다는 것을 암시)
- 방법론의 이름 + 기능: "DPO: Direct Preference Optimization of Language Models"

**나쁜 제목**:
- 너무 포괄적임: "An Approach to Improving Language Model Outputs"
- 너무 김: 대략 15단어를 넘는 모든 것
- 전문 용어만 사용: "Asymptotic Convergence of Iterative Stochastic Policy Refinement" (이건 누굴 위한 건가요?)

**규칙**:
- 방법론의 이름이 있다면 포함하세요 (인용 가능성을 위해)
- 리뷰어들이 검색할 만한 핵심 키워드 1-2개를 포함하세요
- 콜론(:)의 양쪽 모두 의미를 지니지 않는 한 사용을 피하세요
- 테스트: 리뷰어가 제목만 보고도 분야와 기여를 알 수 있을까요?

### Step 5.1: Abstract (5-Sentence Formula) (초록 - 5문장 공식)

Sebastian Farquhar (DeepMind)의 공식:

```
1. 당신이 이룬 것: "우리는 ~를 소개합니다...", "우리는 ~를 증명합니다...", "우리는 ~를 시연합니다..."
2. 이것이 왜 어렵고 중요한지
3. 어떻게 수행하는지 (발견 가능성을 위한 전문가 키워드 포함)
4. 어떤 증거가 있는지
5. 당신의 가장 놀라운 수치/결과
```

"대규모 언어 모델은 놀라운 성공을 거두었습니다..."와 같은 포괄적인 오프닝은 **삭제**하세요.

### Step 5.2: Figure 1 (그림 1)

Figure 1은 대부분의 독자가 두 번째로(초록 다음으로) 보는 항목입니다. 서론을 작성하기 전에 초안을 작성하세요 — 핵심 아이디어를 명확히 하도록 유도합니다.

| Figure 1 유형 | 사용 시기 | 예시 |
|---------------|-------------|---------|
| **Method diagram (방법론 다이어그램)** | 새로운 아키텍처 또는 파이프라인 | 시스템을 보여주는 TikZ 플로우차트 |
| **Results teaser (결과 티저)** | 하나의 설득력 있는 결과가 전체 이야기를 말해줄 때 | 막대 차트: 선명한 차이가 있는 "우리 것 vs 기준(baselines)" |
| **Problem illustration (문제 설명)** | 문제가 직관적이지 않을 때 | 당신이 수정한 실패 모드를 보여주는 전/후 |
| **Conceptual diagram (개념 다이어그램)** | 추상적인 기여를 시각적으로 보여줘야 할 때 | 방법론의 특성을 나타내는 2x2 매트릭스 |

**규칙**: Figure 1은 어떤 텍스트도 읽지 않고 이해할 수 있어야 합니다. 캡션만으로도 핵심 아이디어를 전달해야 합니다. 목적에 맞게 색상을 사용하세요 — 단지 꾸미기 위해 사용하지 마세요.

### Step 5.3: Introduction (1-1.5 pages max) (서론)

반드시 포함해야 할 항목:
- 명확한 문제 진술(problem statement)
- 접근 방식에 대한 간략한 개요
- 2-4개의 주요 기여 요약 (2단 편집 형식에서 각각 최대 1-2줄)
- 방법론(Methods)은 2-3페이지쯤에서 시작해야 합니다.

### Step 5.4: Methods (방법론)

재구현(reimplementation)이 가능하게 하세요:
- 개념적 개요 또는 의사 코드
- 모든 초매개변수(hyperparameters) 나열
- 재현에 충분한 아키텍처 세부 정보
- 최종 설계 결정을 제시하세요; 아블레이션(ablations)은 실험(experiments) 부분에 넣습니다.

### Step 5.5: Experiments & Results (실험 및 결과)

각 실험에 대해 명시적으로 기술하세요:
- **어떤 주장을 뒷받침하는지**
- 주요 기여(main contribution)와 어떻게 연결되는지
- 무엇을 관찰해야 하는지: "파란색 선은 X를 보여주며, 이는 Y를 증명합니다"

요구 사항:
- 방법론과 함께 오차 막대(Error bars) 명시 (표준편차 vs 표준오차)
- 초매개변수 탐색 범위
- 컴퓨팅 인프라 (GPU 유형, 총 소요 시간)
- 시드(Seed) 설정 방법

### Step 5.6: Related Work (관련 연구)

논문별로 나열하지 말고, 방법론적으로(methodologically) 정리하세요. 관대하게 인용하세요 — 리뷰어들이 관련 논문의 저자일 가능성이 높습니다.

### Step 5.7: Limitations (REQUIRED) (한계점 - 필수)

모든 주요 학술대회에서 이를 요구합니다. 정직함이 도움이 됩니다:
- 리뷰어들은 솔직한 한계점 인정에 불이익을 주지 않도록 지시받습니다.
- 약점을 먼저 식별하여 비판을 선제적으로 차단하세요.
- 한계점이 왜 핵심 주장을 훼손하지 않는지 설명하세요.

### Step 5.8: Conclusion & Discussion (결론 및 토의)

**결론** (필수, 0.5-1 페이지):
- 단일 문장으로 기여(contribution)를 재진술하세요 (초록과는 다른 표현 사용)
- 핵심 발견 요약 (목록 형태가 아닌 2-3 문장)
- 시사점(Implications): 이것이 이 분야에 어떤 의미를 갖나요?
- 향후 과제(Future work): 2-3개의 구체적인 다음 단계 (모호한 "X를 향후 과제로 남깁니다"는 피하세요)

**토의** (선택 사항, 때때로 결론과 결합됨):
- 즉각적인 결과를 넘어선 더 넓은 의미의 시사점
- 다른 하위 분야(subfields)와의 연결
- 방법론이 효과가 있는 경우와 없는 경우에 대한 정직한 평가
- 실제 배포 시 고려 사항

결론에서 새로운 결과나 주장을 도입하지 **마세요**.

### Step 5.9: Appendix Strategy (부록 전략)

부록은 주요 학술대회에서 무제한이며 재현성을 위해 필수적입니다. 구조:

| 부록 섹션 | 들어갈 내용 |
|-----------------|---------------|
| **Proofs & Derivations (증명 및 도출)** | 본문에 넣기에는 너무 긴 전체 증명. 본문에는 "부록 A의 증명"이라고 정리를 명시할 수 있습니다. |
| **Additional Experiments (추가 실험)** | 아블레이션, 확장(scaling) 곡선, 데이터 세트별 분석, 초매개변수 민감도 |
| **Implementation Details (구현 세부 정보)** | 전체 초매개변수 테이블, 훈련 세부 정보, 하드웨어 사양, 무작위 시드(random seeds) |
| **Dataset Documentation (데이터 세트 문서화)** | 데이터 수집 과정, 주석(annotation) 가이드라인, 라이선스, 전처리 |
| **Prompts & Templates (프롬프트 및 템플릿)** | 사용된 정확한 프롬프트 (LLM 기반 방법론의 경우), 평가 템플릿 |
| **Human Evaluation (인간 평가)** | 주석 인터페이스 스크린샷, 주석자에게 제공된 지침, IRB 세부 정보 |
| **Additional Figures (추가 그림)** | 작업별 분석, 궤적(trajectory) 시각화, 실패 사례 예시 |

**규칙**:
- 주요 논문은 그 자체로 완결성이 있어야 합니다 — 리뷰어에게 부록을 읽도록 요구할 수 없습니다.
- 중요한 증거를 부록에만 두지 마세요.
- 상호 참조하세요: 단순히 "부록 참조"가 아니라 "전체 결과는 부록 B의 표 5 참조"와 같이 명시하세요.
- `\appendix` 명령을 사용한 다음 `\section{A: Proofs}` 등을 사용하세요.

### Page Budget Management (페이지 제한 관리)

페이지 제한을 초과한 경우:

| 축소 전략 | 절약 | 위험성 |
|-------------|-------|------|
| 증명을 부록으로 이동 | 0.5-2 페이지 | 낮음 — 표준 관행 |
| 관련 연구 압축 | 0.5-1 페이지 | 중간 — 핵심 인용을 누락할 수 있음 |
| 표와 하위 그림(subfigures) 결합 | 0.25-0.5 페이지 | 낮음 — 가독성이 향상되는 경우가 많음 |
| `\vspace{-Xpt}` 신중하게 사용 | 0.1-0.3 페이지 | 교묘하면 낮음, 티가 많이 나면 높음 |
| 정성적인(qualitative) 예시 제거 | 0.5-1 페이지 | 중간 — 리뷰어들은 예시를 좋아함 |
| 그림 크기 줄이기 | 0.25-0.5 페이지 | 높음 — 그림은 계속 읽을 수 있어야 함 |

**절대 하지 말 것**: 폰트 크기 줄이기, 여백 변경하기, 필수 섹션(한계점, 넓은 영향(broader impact)) 제거하기, 본문에 `\small`/`\footnotesize` 사용하기.

### Step 5.10: Ethics & Broader Impact Statement (윤리 및 넓은 영향 성명)

대부분의 학술대회에서 윤리/넓은 영향 성명(ethics/broader impact statement)을 요구하거나 강력히 권장합니다. 이는 단순한 형식적인 문구가 아닙니다 — 리뷰어들이 이를 읽고 채택 거부(desk rejection)를 유발할 수 있는 윤리적 문제를 제기할 수 있습니다.

**포함해야 할 사항:**

| 구성 요소 | 내용 | 요구하는 학회 |
|-----------|---------|-------------|
| **Positive societal impact (긍정적인 사회적 영향)** | 귀하의 연구가 사회에 어떤 이익을 주는지 | NeurIPS, ICML |
| **Potential negative impact (잠재적인 부정적 영향)** | 오용 위험, 이중 사용 우려, 실패 모드 | NeurIPS, ICML |
| **Fairness & bias (공정성 및 편향)** | 당신의 방법론/데이터에 알려진 편향이 있는가? | 모든 학회 (암묵적으로) |
| **Environmental impact (환경적 영향)** | 대규모 훈련의 경우 컴퓨팅 탄소 발자국 | ICML, 갈수록 NeurIPS에서도 요구 |
| **Privacy (개인정보보호)** | 귀하의 연구가 개인 데이터 처리를 사용하거나 가능하게 하는가? | ACL, NeurIPS |
| **LLM disclosure (LLM 공개)** | 작성이나 실험에 AI가 사용되었는가? | ICLR (의무적), ACL |

**성명 작성:**

```latex
\section*{Broader Impact Statement}
% NeurIPS/ICML: 결론 다음, 페이지 제한에 포함되지 않음

% 1. 긍정적인 응용 (1-2 문장)
이 연구는 [특정 집단]에게 혜택을 줄 수 있는 [특정 응용 분야]를 가능하게 합니다.

% 2. 위험과 완화 (1-3 문장, 구체적으로)
[방법론/모델]은 [특정 위험]에 오용될 가능성이 있습니다. 우리는 
[예를 들어, X 크기 이상의 모델 가중치만 배포, 안전 필터 포함, 실패 모드 문서화와 같은 특정 완화 방법]
를 통해 이를 완화합니다.

% 3. 영향 주장의 한계점 (1 문장)
우리의 평가는 [특정 분야]로 제한되며; 보다 폭넓은 배포를 위해서는 
[특정 추가 연구]가 필요할 것입니다.
```

**흔한 실수:**
- "우리는 어떠한 부정적 영향도 예견하지 않습니다"라고 쓰는 것 (거의 사실이 아님 — 리뷰어들은 이를 불신합니다)
- 모호함: 어떻게 오용될 수 있는지 명시하지 않고 "이것은 오용될 수 있습니다"라고 말하는 것
- 대규모 연구에서 컴퓨팅 비용을 무시하는 것
- 요구하는 학회에서 LLM 사용을 공개하는 것을 잊는 것

**컴퓨팅 탄소 발자국** (훈련이 많은 논문의 경우):
```python
# ML CO2 Impact 도구 방법론을 사용한 추정치
gpu_hours = 1000  # 총 GPU 시간
gpu_tdp_watts = 400  # 예: A100 = 400W
pue = 1.1  # Power Usage Effectiveness (데이터 센터 오버헤드)
carbon_intensity = 0.429  # kg CO2/kWh (미국 평균; 지역에 따라 다름)

energy_kwh = (gpu_hours * gpu_tdp_watts * pue) / 1000
carbon_kg = energy_kwh * carbon_intensity
print(f"Energy: {energy_kwh:.0f} kWh, Carbon: {carbon_kg:.0f} kg CO2eq")
```

### Step 5.11: Datasheets & Model Cards (해당하는 경우)

논문이 **새로운 데이터 세트**를 소개하거나 **모델을 배포**하는 경우, 구조화된 문서화를 포함하세요. 리뷰어들은 점점 더 이를 기대하고 있으며, NeurIPS Datasets & Benchmarks 트랙에서는 이를 요구합니다.

**데이터 세트를 위한 데이터 시트 (Datasheets for Datasets)** (Gebru et al., 2021) — 부록에 포함:

```
데이터 세트 문서화 (부록):
- Motivation (동기): 왜 이 데이터 세트가 만들어졌나요? 어떤 작업을 지원하나요?
- Composition (구성): 인스턴스는 무엇인가요? 몇 개나 있나요? 어떤 데이터 유형인가요?
- Collection (수집): 데이터는 어떻게 수집되었나요? 출처는 어디인가요?
- Preprocessing (전처리): 어떤 정제/필터링이 적용되었나요?
- Distribution (배포): 데이터 세트는 어떻게 배포되나요? 어떤 라이선스 하에 배포되나요?
- Maintenance (유지 관리): 누가 유지 관리하나요? 문제는 어떻게 보고하나요?
- Ethical considerations (윤리적 고려 사항): 개인 데이터가 포함되어 있나요? 동의를 얻었나요?
  피해 가능성이 있나요? 알려진 편향이 있나요?
```

**모델 카드 (Model Cards)** (Mitchell et al., 2019) — 모델 배포를 위해 부록에 포함:

```
모델 카드 (부록):
- Model details (모델 세부 정보): 아키텍처, 훈련 데이터, 훈련 과정
- Intended use (의도된 용도): 주요 사용 사례, 범위 외 사용
- Metrics (지표): 평가 지표 및 벤치마크 결과
- Ethical considerations (윤리적 고려 사항): 알려진 편향, 공정성 평가
- Limitations (한계점): 알려진 실패 모드, 모델의 성능이 떨어지는 영역
```

### Writing Style (작성 스타일)

**문장 단위의 명확성 (Gopen & Swan's 7 Principles):**

| 원칙 | 규칙 |
|-----------|------|
| 주어-동사 근접 (Subject-verb proximity) | 주어와 동사를 가깝게 유지 |
| 강조 위치 (Stress position) | 강조할 내용을 문장 끝에 배치 |
| 주제 위치 (Topic position) | 맥락을 먼저 제시하고, 새로운 정보를 나중에 |
| 헌 것 먼저 새 것 나중에 (Old before new) | 친숙한 정보 → 생소한 정보 |
| 1단위 1기능 (One unit, one function) | 각 문단은 하나의 요점을 전달 |
| 동작은 동사로 (Action in verb) | 명사화(nominalizations) 대신 동사 사용 |
| 새 정보 전 맥락 제시 (Context before new) | 제시하기 전에 무대 설정 |

**단어 선택 (Lipton, Steinhardt):**
- 구체적으로 쓰세요: "performance" 대신 "accuracy"
- 모호한 표현 피하기: 정말 불확실한 경우가 아니면 "may" 사용 금지
- 전체적으로 일관된 용어 사용
- 점진적인 어휘 사용 피하기: "combine"보다는 "develop"

**예시가 포함된 전체 작성 가이드**: [references/writing-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/writing-guide.md) 참조

### Using LaTeX Templates (LaTeX 템플릿 사용하기)

**항상 템플릿 디렉토리 전체를 먼저 복사한 다음, 그 안에서 작성하세요.**

```
템플릿 설정 체크리스트:
- [ ] Step 1: 템플릿 디렉토리 전체를 새 프로젝트로 복사
- [ ] Step 2: 템플릿이 변경 없이(as-is) 컴파일되는지 확인
- [ ] Step 3: 템플릿의 예시 내용을 읽고 구조 파악
- [ ] Step 4: 예시 내용을 섹션별로 바꾸기
- [ ] Step 5: 템플릿 매크로 사용 (preamble의 \newcommand 정의 확인)
- [ ] Step 6: 템플릿 찌꺼기(artifacts)는 맨 마지막에 정리
```

**Step 1: 템플릿 전체 복사**

```bash
cp -r templates/neurips2025/ ~/papers/my-paper/
cd ~/papers/my-paper/
ls -la  # main.tex, neurips.sty, Makefile 등이 보여야 합니다.
```

`.tex` 파일만 복사하지 말고 전체 디렉토리를 복사하세요. 템플릿에는 스타일 파일(`.sty`), 참고문헌 스타일(`.bst`), 예시 콘텐츠, Makefile이 포함되어 있습니다.

**Step 2: 먼저 템플릿 컴파일 확인**

어떤 변경도 하기 전에:
```bash
latexmk -pdf main.tex
# 또는 수동으로: pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

수정되지 않은 템플릿이 컴파일되지 않으면 그 문제부터 해결하세요 (보통 TeX 패키지가 누락된 경우입니다 — `tlmgr install <package>`를 통해 설치하세요).

**Step 3: 템플릿 콘텐츠를 참고용으로 유지**

예시 콘텐츠를 즉시 삭제하지 마세요. 주석 처리하고 서식 지정 참고용으로 사용하세요:
```latex
% 템플릿 예시 (참고용으로 유지):
% \begin{figure}[t]
%   \centering
%   \includegraphics[width=0.8\linewidth]{example-image}
%   \caption{Template shows caption style}
% \end{figure}

% 당신의 실제 그림:
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{your-figure.pdf}
  \caption{같은 스타일을 따르는 당신의 캡션.}
\end{figure}
```

**Step 4: 섹션별로 내용 교체**

체계적으로 작업하세요: 제목/저자 → 초록 → 서론 → 방법론 → 실험 → 관련 연구 → 결론 → 참고문헌 → 부록. 각 섹션을 수정한 후 컴파일하세요.

**Step 5: 템플릿 매크로 사용**

```latex
\newcommand{\method}{YourMethodName}  % 일관된 방법론 명명
\newcommand{\eg}{e.g.,\xspace}        % 올바른 약어 표현
\newcommand{\ie}{i.e.,\xspace}
```

### Template Pitfalls (템플릿 주의 사항)

| 함정 | 문제 | 해결책 |
|---------|---------|----------|
| `.tex` 파일만 복사 | `.sty`가 누락되어 컴파일되지 않음 | 디렉토리 전체를 복사 |
| `.sty` 파일 수정 | 학술대회 서식 규칙 위반 | 스타일 파일은 절대 편집하지 말 것 |
| 마구잡이 패키지 추가 | 충돌 발생, 템플릿 깨짐 | 필요한 경우에만 추가 |
| 템플릿 내용을 일찍 삭제 | 서식 참고 자료 손실 | 완료될 때까지 주석으로 유지 |
| 자주 컴파일하지 않음 | 오류 누적 | 각 섹션 작성 후 컴파일 |
| 그림에 래스터(Raster) PNG 사용 | 논문에서 흐릿하게 보임 | 항상 `savefig('fig.pdf')`를 통해 벡터 PDF 사용 |

### Quick Template Reference (빠른 템플릿 참조)

| 학술대회 | 메인 파일 | 스타일 파일 | 페이지 제한 |
|------------|-----------|------------|------------|
| NeurIPS 2025 | `main.tex` | `neurips.sty` | 9 페이지 |
| ICML 2026 | `example_paper.tex` | `icml2026.sty` | 8 페이지 |
| ICLR 2026 | `iclr2026_conference.tex` | `iclr2026_conference.sty` | 9 페이지 |
| ACL 2025 | `acl_latex.tex` | `acl.sty` | 8 페이지 (long) |
| AAAI 2026 | `aaai2026-unified-template.tex` | `aaai2026.sty` | 7 페이지 |
| COLM 2025 | `colm2025_conference.tex` | `colm2025_conference.sty` | 9 페이지 |

**공통**: 이중 눈가림(Double-blind), 참고문헌은 분량에 포함 안됨, 부록 무제한, LaTeX 필수.

템플릿은 `templates/` 디렉토리에 있습니다. (VS Code, CLI, Overleaf, 기타 IDE 설정 방법은 [templates/README.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/templates/README.md) 참조)

### Tables and Figures (표와 그림)

**표(Tables)** — 전문가다운 서식을 위해 `booktabs`를 사용하세요:

```latex
\usepackage{booktabs}
\begin{tabular}{lcc}
\toprule
Method & Accuracy $\uparrow$ & Latency $\downarrow$ \\
\midrule
Baseline & 85.2 & 45ms \\
\textbf{Ours} & \textbf{92.1} & 38ms \\
\bottomrule
\end{tabular}
```

규칙:
- 지표별로 가장 좋은 값을 굵게 표시(Bold)
- 방향 기호 포함 ($\uparrow$ 높을수록 좋음, $\downarrow$ 낮을수록 좋음)
- 숫자 열은 오른쪽 정렬
- 일관된 소수점 정밀도

**그림(Figures)**:
- 모든 플롯과 다이어그램은 **벡터 그래픽(Vector graphics)** (PDF, EPS) 사용 — `plt.savefig('fig.pdf')`
- 사진의 경우에만 **래스터(Raster)** (PNG 600 DPI) 사용
- **색맹 친화적인 색상 팔레트(Colorblind-safe palettes)** (Okabe-Ito 또는 Paul Tol)
- **흑백(Grayscale) 가독성** 확인 (남성의 8%는 색각 이상이 있습니다)
- **그림 안에 제목을 넣지 마세요** — 캡션이 이 역할을 수행합니다
- **자체적으로 완결된 캡션** — 독자가 본문을 읽지 않고도 이해할 수 있어야 합니다

### Conference Resubmission (학술대회 재제출)

학술대회 간 변환 방법은 Phase 7 (제출 준비)을 참조하세요 — 여기에는 전체 변환 워크플로우, 페이지 변경 테이블 및 거부 후 지침이 포함되어 있습니다.

### Professional LaTeX Preamble (전문적인 LaTeX 프리앰블)

전문적인 품질을 위해 모든 논문에 다음 패키지를 추가하세요. 이는 모든 주요 학술대회 스타일 파일과 호환됩니다:

```latex
% --- 전문 패키지 (학술대회 스타일 파일 뒤에 추가) ---

% Typography
\usepackage{microtype}              % 미세 타이포그래피 개선(돌출, 확장)
                                     % 텍스트를 눈에 띄게 깔끔하게 만들어줍니다 — 항상 포함하세요

% Tables
\usepackage{booktabs}               % 전문적인 표 테두리 (\toprule, \midrule, \bottomrule)
\usepackage{siunitx}                % 일관된 숫자 서식, 소수점 정렬
                                     % 사용법: \num{12345} → 12,345; \SI{3.5}{GHz} → 3.5 GHz
                                     % 표 정렬: 소수점 정렬 숫자를 위한 S 열(column) 유형

% Figures
\usepackage{graphicx}               % 그래픽 포함 (\includegraphics)
\usepackage{subcaption}             % (a), (b), (c) 라벨이 있는 하위 그림
                                     % 사용법: \begin{subfigure}{0.48\textwidth} ... \end{subfigure}

% Diagrams and Algorithms
\usepackage{tikz}                   % 프로그래밍 가능한 벡터 다이어그램
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, calc, fit, backgrounds}
\usepackage[ruled,vlined]{algorithm2e}  % 전문적인 의사 코드(pseudocode)
                                     % 대안: 템플릿에 포함된 경우 \usepackage{algorithmicx}

% Cross-references
\usepackage{cleveref}               % 스마트 참조: \cref{fig:x} → "Figure 1"
                                     % 반드시 hyperref "이후에" 로드해야 합니다
                                     % 다루는 항목: 그림, 표, 섹션, 방정식, 알고리즘

% Math (일반적으로 학회 .sty에 포함되지만 확인하세요)
\usepackage{amsmath,amssymb}        % AMS 수학 환경 및 기호
\usepackage{mathtools}              % amsmath 확장 (dcases, coloneqq 등)

% Colors (그림과 다이어그램을 위함)
\usepackage{xcolor}                 % 색상 관리
% Okabe-Ito 색맹 친화 팔레트:
\definecolor{okblue}{HTML}{0072B2}
\definecolor{okorange}{HTML}{E69F00}
\definecolor{okgreen}{HTML}{009E73}
\definecolor{okred}{HTML}{D55E00}
\definecolor{okpurple}{HTML}{CC79A7}
\definecolor{okcyan}{HTML}{56B4E9}
\definecolor{okyellow}{HTML}{F0E442}
```

**참고사항:**
- `microtype`은 시각적 품질에 가장 큰 영향을 미치는 단일 패키지입니다. 서브픽셀 단위로 문자 간격을 조정합니다. 항상 포함하세요.
- `siunitx`는 `S` 열 유형을 통해 표에서 소수점 정렬을 처리하여 수동 간격 조정을 없애줍니다.
- `cleveref`는 반드시 `hyperref` **이후에** 로드해야 합니다. 대부분의 학회 .sty 파일이 hyperref를 로드하므로 cleveref를 마지막에 두세요.
- 학회 템플릿이 이러한 패키지(특히 `algorithm`, `amsmath`, `graphicx`)를 이미 로드하는지 확인하세요. 이중 로드하지 마세요.

### siunitx Table Alignment (siunitx 표 정렬)

`siunitx`는 숫자가 많은 표를 눈에 띄게 더 읽기 쉽게 만들어줍니다:

```latex
\begin{tabular}{l S[table-format=2.1] S[table-format=2.1] S[table-format=2.1]}
\toprule
Method & {Accuracy $\uparrow$} & {F1 $\uparrow$} & {Latency (ms) $\downarrow$} \\
\midrule
Baseline         & 85.2  & 83.7  & 45.3 \\
Ablation (no X)  & 87.1  & 85.4  & 42.1 \\
\textbf{Ours}    & \textbf{92.1} & \textbf{90.8} & \textbf{38.7} \\
\bottomrule
\end{tabular}
```

`S` 열(column) 유형은 소수점을 기준으로 자동 정렬합니다. 중괄호 `{}` 안에 있는 헤더는 이 정렬을 피해갑니다.

### Subfigures (하위 그림)

나란히 있는 그림의 표준 패턴:

```latex
\begin{figure}[t]
  \centering
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig_results_a.pdf}
    \caption{Dataset A의 결과.}
    \label{fig:results-a}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig_results_b.pdf}
    \caption{Dataset B의 결과.}
    \label{fig:results-b}
  \end{subfigure}
  \caption{두 개의 데이터 세트에 걸친 제안 방법의 비교. (a)는 확장(scaling)
  행동을 보여주며 (b)는 아블레이션(ablation) 결과를 보여줍니다. 둘 다 5개의 무작위 시드를 사용합니다.}
  \label{fig:results}
\end{figure}
```

`\cref{fig:results}` → "Figure 1", `\cref{fig:results-a}` → "Figure 1a"를 사용하세요.

### Pseudocode with algorithm2e (algorithm2e를 활용한 의사코드)

```latex
\begin{algorithm}[t]
\caption{판사 패널(Judge Panel)이 있는 반복적 개선}
\label{alg:method}
\KwIn{작업 $T$, 모델 $M$, 판사 $J_1 \ldots J_n$, 수렴 임계값 $k$}
\KwOut{최종 출력 $A^*$}
$A \gets M(T)$ \tcp*{초기 생성}
$\text{streak} \gets 0$\;
\While{$\text{streak} < k$}{
  $C \gets \text{Critic}(A, T)$ \tcp*{약점 파악}
  $B \gets M(T, C)$ \tcp*{비판을 반영한 수정 버전}
  $AB \gets \text{Synthesize}(A, B)$ \tcp*{가장 좋은 요소 통합}
  \ForEach{judge $J_i$}{
    $\text{rank}_i \gets J_i(\text{shuffle}(A, B, AB))$ \tcp*{블라인드 순위 매기기}
  }
  $\text{winner} \gets \text{BordaCount}(\text{ranks})$\;
  \eIf{$\text{winner} = A$}{
    $\text{streak} \gets \text{streak} + 1$\;
  }{
    $A \gets \text{winner}$; $\text{streak} \gets 0$\;
  }
}
\Return{$A$}\;
\end{algorithm}
```

### TikZ Diagram Patterns (TikZ 다이어그램 패턴)

TikZ는 ML 논문의 방법론 다이어그램을 그리는 표준입니다. 일반적인 패턴:

**Pipeline/Flow Diagram (파이프라인/흐름 다이어그램)** (ML 논문에서 가장 흔함):

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[
  node distance=1.8cm,
  box/.style={rectangle, draw, rounded corners, minimum height=1cm, 
              minimum width=2cm, align=center, font=\small},
  arrow/.style={-{Stealth[length=3mm]}, thick},
]
  \node[box, fill=okcyan!20] (input) {Input\\$x$};
  \node[box, fill=okblue!20, right of=input] (encoder) {Encoder\\$f_\theta$};
  \node[box, fill=okgreen!20, right of=encoder] (latent) {Latent\\$z$};
  \node[box, fill=okorange!20, right of=latent] (decoder) {Decoder\\$g_\phi$};
  \node[box, fill=okred!20, right of=decoder] (output) {Output\\$\hat{x}$};
  
  \draw[arrow] (input) -- (encoder);
  \draw[arrow] (encoder) -- (latent);
  \draw[arrow] (latent) -- (decoder);
  \draw[arrow] (decoder) -- (output);
\end{tikzpicture}
\caption{아키텍처 개요. 인코더는 입력 $x$를 잠재(latent) 
표현 $z$로 매핑하고, 디코더는 이를 재구성합니다.}
\label{fig:architecture}
\end{figure}
```

**Comparison/Matrix Diagram (비교/행렬 다이어그램)** (방법론의 변형을 보여줄 때):

```latex
\begin{tikzpicture}[
  cell/.style={rectangle, draw, minimum width=2.5cm, minimum height=1cm, 
               align=center, font=\small},
  header/.style={cell, fill=gray!20, font=\small\bfseries},
]
  % Headers
  \node[header] at (0, 0) {Method};
  \node[header] at (3, 0) {Converges?};
  \node[header] at (6, 0) {Quality?};
  % Rows
  \node[cell] at (0, -1) {Single Pass};
  \node[cell, fill=okgreen!15] at (3, -1) {N/A};
  \node[cell, fill=okorange!15] at (6, -1) {Baseline};
  \node[cell] at (0, -2) {Critique+Revise};
  \node[cell, fill=okred!15] at (3, -2) {No};
  \node[cell, fill=okred!15] at (6, -2) {Degrades};
  \node[cell] at (0, -3) {Ours};
  \node[cell, fill=okgreen!15] at (3, -3) {Yes ($k$=2)};
  \node[cell, fill=okgreen!15] at (6, -3) {Improves};
\end{tikzpicture}
```

**Iterative Loop Diagram (반복 루프 다이어그램)** (피드백이 있는 방법론의 경우):

```latex
\begin{tikzpicture}[
  node distance=2cm,
  box/.style={rectangle, draw, rounded corners, minimum height=0.8cm, 
              minimum width=1.8cm, align=center, font=\small},
  arrow/.style={-{Stealth[length=3mm]}, thick},
  label/.style={font=\scriptsize, midway, above},
]
  \node[box, fill=okblue!20] (gen) {Generator};
  \node[box, fill=okred!20, right=2.5cm of gen] (critic) {Critic};
  \node[box, fill=okgreen!20, below=1.5cm of $(gen)!0.5!(critic)$] (judge) {Judge Panel};
  
  \draw[arrow] (gen) -- node[label] {output $A$} (critic);
  \draw[arrow] (critic) -- node[label, right] {critique $C$} (judge);
  \draw[arrow] (judge) -| node[label, left, pos=0.3] {winner} (gen);
\end{tikzpicture}
```

### latexdiff for Revision Tracking (버전 추적을 위한 latexdiff)

이의 제기(Rebuttal) 시 필수 — 버전 간 변경 사항을 보여주는 마크업 PDF를 생성합니다:

```bash
# 설치
# macOS: brew install latexdiff (또는 TeX Live에 포함됨)
# Linux: sudo apt install latexdiff

# diff 생성
latexdiff paper_v1.tex paper_v2.tex > paper_diff.tex
pdflatex paper_diff.tex

# 여러 파일 프로젝트의 경우 (\input{} 또는 \include{} 사용 시)
latexdiff --flatten paper_v1.tex paper_v2.tex > paper_diff.tex
```

이 코드는 삭제된 텍스트는 빨간색 취소선으로, 추가된 텍스트는 파란색으로 표시된 PDF를 생성합니다 — 이의 제기 부록의 표준 형식입니다.

### SciencePlots for matplotlib (matplotlib를 위한 SciencePlots)

출판 수준 품질의 플롯을 위해 설치 및 사용하세요:

```bash
pip install SciencePlots
```

```python
import matplotlib.pyplot as plt
import scienceplots  # 스타일 등록

# science 스타일 사용 (IEEE 스타일, 깔끔함)
with plt.style.context(['science', 'no-latex']):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # 1단 폭
    ax.plot(x, y, label='Ours', color='#0072B2')
    ax.plot(x, y2, label='Baseline', color='#D55E00', linestyle='--')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.savefig('paper/fig_results.pdf', bbox_inches='tight')

# 사용 가능한 스타일: 'science', 'ieee', 'nature', 'science+ieee'
# 플롯을 생성하는 머신에 LaTeX이 설치되지 않은 경우 'no-latex' 추가
```

**표준 그림 크기** (2단(two-column) 형식의 경우):
- 1단 (Single column): `figsize=(3.5, 2.5)` — 1단에 맞음
- 2단 (Double column): `figsize=(7.0, 3.0)` — 양쪽 단 모두에 걸침
- 정사각형 (Square): `figsize=(3.5, 3.5)` — 히트맵, 혼동 행렬(confusion matrices)용

---

## Phase 6: Self-Review & Revision

**목표**: 제출 전에 리뷰 과정을 모방해봅니다. 약점을 조기에 발견합니다.

### Step 6.1: Simulate Reviews (Ensemble Pattern) (리뷰 시뮬레이션 - 앙상블 패턴)

여러 가지 관점에서 리뷰를 생성하세요. 자동화된 연구 파이프라인(특히 SakanaAI의 AI-Scientist)에서 얻은 핵심 통찰력은 **메타 리뷰어(meta-reviewer)가 포함된 앙상블 리뷰가 단일 리뷰 패스보다 훨씬 더 정확하게 조정된(calibrated) 피드백을 생성한다는 것입니다.**

**Step 1: N개의 독립적인 리뷰 생성** (N=3-5)

다양한 모델 또는 온도(temperature) 설정을 사용하세요. 각 리뷰어는 다른 리뷰는 보지 않고 오로지 논문만 봅니다. **부정적 편향을 기본값으로 하세요** — LLM은 평가 시 긍정 편향을 보인다는 문서화된 특징이 있습니다.

```
당신은 [학술대회 이름]의 전문가 리뷰어입니다. 당신은 비판적이고 철저합니다.
논문에 약점이 있거나 주장에 대해 확신이 서지 않는다면 그것을 명확히 지적하고
점수에 반영하세요. 유리하게 좋게만 해석해주지 마세요.

공식 리뷰어 가이드라인에 따라 이 논문을 평가하세요. 평가 항목:

1. 타당성 (Soundness) (주장이 잘 뒷받침되는가? 기준(baselines)이 공정하고 강력한가?)
2. 명확성 (Clarity) (논문이 잘 쓰였는가? 전문가가 이를 재현할 수 있는가?)
3. 중요성 (Significance) (이것이 커뮤니티에 중요한가?)
4. 독창성 (Originality) (단순한 증분적 결합이 아닌 새로운 통찰인가?)

리뷰를 구조화된 JSON으로 제공하세요:
{
  "summary": "2-3 문장 요약",
  "strengths": ["강점 1", "강점 2", ...],
  "weaknesses": ["약점 1 (가장 심각한 것)", "약점 2", ...],
  "questions": ["저자에게 묻는 질문 1", ...],
  "missing_references": ["인용해야 할 논문", ...],
  "soundness": 1-4,
  "presentation": 1-4,
  "contribution": 1-4,
  "overall": 1-10,
  "confidence": 1-5
}
```

**Step 2: Meta-review (메타 리뷰 - Area Chair 집계)**

N개의 리뷰를 모두 메타 리뷰어에게 입력합니다:

```
당신은 [학술대회 이름]의 Area Chair입니다. 한 논문에 대해 [N]개의 
독립적인 리뷰를 받았습니다. 당신의 업무는 다음과 같습니다:

1. 리뷰어들 사이의 공통된 강점과 약점을 파악합니다.
2. 논문을 직접 검토하여 의견 불일치를 해결합니다.
3. 종합된 판단을 나타내는 메타 리뷰(meta-review)를 작성합니다.
4. 모든 리뷰에 걸친 평균 숫자 점수를 사용합니다.

보수적인 태도를 취하세요: 만약 어떤 약점이 심각한지에 대해 리뷰어들 간에 의견이 
일치하지 않는다면, 저자가 해결할 때까지 그것을 심각한 것으로 간주하세요.

리뷰들:
[review_1]
[review_2]
...
```
**Step 3: Reflection loop (성찰 루프)** (선택 사항, 2-3 라운드)

각 리뷰어는 메타 리뷰를 확인한 후 자신의 리뷰를 세부 조정할 수 있습니다. 조기 종료(early termination) 조건을 사용하세요: 리뷰어가 "I am done(완료함)"(더 이상 변경 사항 없음)이라고 응답하면 반복을 중지합니다.

**리뷰를 위한 모델 선택**: 논문 작성 시 더 저렴한 모델을 사용했더라도, 리뷰는 사용 가능한 가장 강력한 모델을 사용하는 것이 가장 좋습니다. 리뷰어 모델은 작성 모델과 독립적으로 선택해야 합니다.

**Few-shot calibration (소수 샷 보정)**: 가능하다면 대상 학술대회에서 실제 게재된 1~2개의 리뷰를 예시로 포함시키세요. 이는 점수 조정(calibration)을 극적으로 향상시킵니다. 예시 리뷰는 [references/reviewer-guidelines.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/reviewer-guidelines.md)를 참조하세요.

### Step 6.1b: Visual Review Pass (VLM) (시각적 검토 패스)

텍스트 전용 리뷰는 전체적인 문제 부류, 즉 그림의 품질, 레이아웃 문제, 시각적 일관성 등을 놓칩니다. 시각적 모델(VLM)에 대한 액세스 권한이 있는 경우 컴파일된 PDF에 대해 별도의 **시각적 검토(visual review)**를 실행하세요:

```
당신은 이 연구 논문 PDF의 시각적 프레젠테이션을 검토하고 있습니다.
다음을 확인하세요:
1. 그림의 품질: 플롯을 읽을 수 있는가? 레이블이 읽기 쉬운가? 색상을 구별할 수 있는가?
2. 그림-캡션 일치 여부: 각 캡션이 해당하는 그림을 정확하게 묘사하는가?
3. 레이아웃 문제: 홀로 남겨진 섹션 헤더, 어색한 페이지 나누기, 관련 참조에서 너무 멀리 떨어진 그림 등
4. 표 서식 지정: 열 정렬 상태, 일관 일관된 소수점 정밀도, 가장 좋은 결과 굵게 표시
5. 시각적 일관성: 모든 그림에 걸쳐 동일한 색 구성표, 일관된 글꼴 크기
6. 흑백 가독성: 흑백으로 인쇄되어도 그림을 이해할 수 있는가?

각 문제에 대해 페이지 번호와 정확한 위치를 명시하세요.
```

이는 텍스트 기반 리뷰로는 잡아낼 수 없는 문제들, 예를 들어 읽을 수 없는 축 라벨이 있는 플롯, 처음 참조된 위치에서 3페이지나 떨어진 그림, 그림 2와 그림 5 사이의 일관성 없는 색상 팔레트, 단(column) 너비보다 명확히 더 넓은 표 등을 잡아냅니다.

### Step 6.1c: Claim Verification Pass (주장 검증 패스)

시뮬레이션된 리뷰 후 별도의 검증 패스(verification pass)를 실행하세요. 리뷰어가 놓칠 수 있는 사실적 오류를 잡아냅니다:

```
주장 검증 프로토콜(Claim Verification Protocol):
1. 논문에서 모든 사실적 주장(수치, 비교, 추세) 추출
2. 각 주장에 대해 그것을 뒷받침하는 특정 실험/결과 추적
3. 논문의 수치가 실제 결과 파일의 수치와 일치하는지 확인
4. 추적 가능한 출처가 없는 주장은 [VERIFY]로 플래그 지정
```

에이전트 기반 워크플로우의 경우: 오직 논문 텍스트와 원시 결과 파일만 제공받는 **새로운 하위 에이전트**에게 검증을 위임하세요. 새로운 컨텍스트는 확증 편향(confirmation bias)을 방지합니다 — 검증 에이전트는 결과가 어때야 했는지 "기억"하지 못합니다.

### Step 6.2: Prioritize Feedback (피드백 우선순위 지정)

리뷰를 수집한 후 다음과 같이 분류하세요:

| 우선순위 | 조치 |
|----------|--------|
| **Critical (매우 중요)** (기술적 결함, 기준(baseline) 누락) | 반드시 수정해야 함. 새로운 실험 필요할 수 있음 → Phase 2로 복귀 |
| **High (높음)** (명확성 문제, 아블레이션 누락) | 이번 수정본에서 고쳐야 함 |
| **Medium (중간)** (사소한 작성 문제, 추가 실험) | 시간이 허락하면 수정 |
| **Low (낮음)** (스타일 선호도, 부차적인 제안) | 향후 작업(future work)을 위한 메모로 남김 |

### Step 6.3: Revision Cycle (수정 사이클)

각 Critical/High 문제에 대해:
1. 영향받는 특정 섹션(들) 식별
2. 수정 초안 작성
3. 수정으로 인해 다른 주장이 손상되지 않는지 확인
4. 논문 업데이트
5. 리뷰어의 우려 사항에 대해 다시 확인(Re-check)

### Step 6.4: Rebuttal Writing (이의 제기(Rebuttal) 작성)

제출 후 실제 리뷰에 답변할 때, 이의 제기는 논문 수정과는 또 다른 독특한 스킬입니다:

**형식**: 항목별로 (Point-by-point). 각 리뷰어의 우려 사항에 대해:
```
> R1-W1: "The paper lacks comparison with Method X. (논문에 Method X와의 비교가 빠져 있습니다.)"

We thank the reviewer for this suggestion. We have added a comparison with 
Method X in Table 3 (revised). Our method outperforms X by 3.2pp on [metric] 
(p<0.05). We note that X requires 2x our compute budget.
(이러한 제안을 해주신 리뷰어께 감사드립니다. 개정된 표 3에 Method X와의 비교를 
추가했습니다. 우리 방법론은 [지표]에서 X보다 3.2%p 우수합니다 (p<0.05). 
참고로 X는 우리 컴퓨팅 예산의 2배를 요구합니다.)
```

**규칙**:
- 모든 우려 사항에 답하세요 — 리뷰어는 당신이 하나라도 건너뛰면 바로 알아차립니다.
- 가장 강력한 답변부터 먼저 쓰세요.
- 간결하고 직접적으로 쓰세요 — 리뷰어는 수십 개의 이의 제기 문서를 읽습니다.
- 이의 제기 기간 중 추가 실험을 진행했다면 새로운 결과를 포함하세요.
- 결코 방어적이거나, 아무리 약한 비판이라도 무시하는 태도를 취하지 마세요.
- 버전 변경 사항을 표시하는 마크업 PDF 생성을 위해 `latexdiff`를 사용하세요 (Professional LaTeX Tooling 섹션 참조).
- (일반적인 칭찬이 아닌) 구체적이고 실행 가능한 피드백을 제공해 준 리뷰어에게 감사함을 표하세요.

**절대 하지 말아야 할 것**: 증거 없이 "저희는 정중히 동의하지 않습니다(We respectfully disagree)"라고 하는 것. 설명 없이 "이것은 범위를 벗어납니다(This is out of scope)"라고 하는 것. 강점에만 응답하여 약점을 외면하는 것.

### Step 6.5: Paper Evolution Tracking (논문 진화 추적)

주요 단계마다 스냅샷을 저장하세요:
```
paper/
  paper.tex                    # 현재 작업 중인 버전
  paper_v1_first_draft.tex     # 첫 번째 완성 초안
  paper_v2_post_review.tex     # 리뷰 시뮬레이션 이후
  paper_v3_pre_submission.tex  # 제출 전 최종본
  paper_v4_camera_ready.tex    # 논문 채택 후의 카메라 레디(camera-ready) 최종본
```

---

## Phase 7: Submission Preparation (제출 준비)

**목표**: 최종 점검, 형식 조정 및 제출.

### Step 7.1: Conference Checklist (학술대회 체크리스트)

모든 학회에는 필수 체크리스트가 있습니다. 신중하게 완료하세요 — 불완전한 체크리스트는 채택 거부(desk rejection)를 초래할 수 있습니다.

다음을 위해 [references/checklists.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/checklists.md)를 참조하세요:
- NeurIPS 16개 항목 논문 체크리스트
- ICML 넓은 영향(broader impact) + 재현성
- ICLR LLM 공개 정책
- ACL 필수 한계점(limitations) 섹션
- 공통 제출 전 체크리스트

### Step 7.2: Anonymization Checklist (익명화 체크리스트)

이중 눈가림(Double-blind) 리뷰는 리뷰어가 논문을 쓴 사람이 누구인지 알 수 없음을 의미합니다. 다음 **모든 항목**을 확인하세요:

```
익명화 체크리스트 (Anonymization Checklist):
- [ ] PDF 어디에도 저자 이름이나 소속이 없어야 함
- [ ] 감사의 글(Acknowledgments) 섹션이 없어야 함 (채택 후 추가)
- [ ] 자기 논문 인용(Self-citations) 시 3인칭 사용: "We previously showed [1]..."이 아닌 "Smith et al. [1] showed..."
- [ ] 개인 저장소(personal repos)를 가리키는 GitHub/GitLab URL 없음
- [ ] 코드 링크에는 Anonymous GitHub (https://anonymous.4open.science/) 사용
- [ ] 그림에 기관 로고나 식별자(identifiers) 없음
- [ ] 저자 이름을 포함하는 파일 메타데이터 없음 (PDF 속성 확인)
- [ ] "our previous work" 또는 "in our earlier paper" 등의 표현 없음
- [ ] 데이터 세트 이름에 기관이 노출되지 않음 (필요시 이름 변경)
- [ ] 부록(Supplementary materials)에 식별 정보가 포함되지 않음
```

**흔한 실수**: 보조 코드에 있는 Git 커밋 메시지가 노출되는 경우, 기관 도구의 워터마크가 그림에 있는 경우, 이전 초안에서 감사의 글을 지우지 않고 남겨둔 경우, 익명 기간 전에 arXiv 프리프린트(preprint)를 게시하는 경우.

### Step 7.3: Formatting Verification (서식 확인)

```
제출 전 서식 점검:
- [ ] 페이지 제한 준수 (참고문헌과 부록 제외)
- [ ] 모든 그림은 벡터(PDF) 또는 고해상도 래스터(600 DPI PNG)
- [ ] 모든 그림은 흑백으로도 읽을 수 있음
- [ ] 모든 표는 booktabs 사용
- [ ] 참고문헌이 올바르게 컴파일됨 (인용에 "?" 없음)
- [ ] 중요한 영역에 overfull hboxes 없음
- [ ] 부록이 명확하게 라벨링되고 분리됨
- [ ] 필수 섹션이 존재함 (한계점, 넓은 영향 등)
```

### Step 7.4: Pre-Compilation Validation (컴파일 전 검증)

`pdflatex`를 실행하기 **전에** 이러한 자동 점검을 실행하세요. 이 과정에서 오류를 잡아내는 것이 컴파일러 출력을 디버깅하는 것보다 빠릅니다.

```bash
# 1. chktex로 린팅 (흔한 LaTeX 실수를 잡아냄)
# 시끄러운 경고 억제: -n2 (문장 끝), -n24 (괄호), -n13 (문장 간), -n1 (명령어 종료)
chktex main.tex -q -n2 -n24 -n13 -n1

# 2. .bib에 모든 인용이 존재하는지 확인
# .tex에서 \cite{...} 추출 후 각각 .bib와 대조
python3 -c "
import re
tex = open('main.tex').read()
bib = open('references.bib').read()
cites = set(re.findall(r'\\\\cite[tp]?{([^}]+)}', tex))
for cite_group in cites:
    for cite in cite_group.split(','):
        cite = cite.strip()
        if cite and cite not in bib:
            print(f'WARNING: \\\\cite{{{cite}}} not found in references.bib')
"

# 3. 참조된 모든 그림이 디스크에 존재하는지 확인
python3 -c "
import re, os
tex = open('main.tex').read()
figs = re.findall(r'\\\\includegraphics(?:\[.*?\])?{([^}]+)}', tex)
for fig in figs:
    if not os.path.exists(fig):
        print(f'WARNING: Figure file not found: {fig}')
"

# 4. 중복된 \label 정의 확인
python3 -c "
import re
from collections import Counter
tex = open('main.tex').read()
labels = re.findall(r'\\\\label{([^}]+)}', tex)
dupes = {k: v for k, v in Counter(labels).items() if v > 1}
for label, count in dupes.items():
    print(f'WARNING: Duplicate label: {label} (appears {count} times)')
"
```

진행하기 전에 경고를 수정하세요. 에이전트 기반 워크플로우의 경우: chktex 출력을 최소한의 수정을 지시하는 명령과 함께 다시 에이전트에게 공급하세요.

### Step 7.5: Final Compilation (최종 컴파일)

```bash
# 클린 빌드
rm -f *.aux *.bbl *.blg *.log *.out *.pdf
latexmk -pdf main.tex

# 또는 수동 (상호 참조를 위해 세 번의 pdflatex + bibtex 실행)
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# 출력이 존재하고 내용이 있는지 확인
ls -la main.pdf
```

**컴파일에 실패할 경우**: 첫 번째 오류를 확인하기 위해 `.log` 파일을 분석하세요. 일반적인 해결책:
- "Undefined control sequence" → 패키지 누락 또는 명령어 오타
- "Missing $ inserted" → 수학 모드 밖의 수학 기호
- "File not found" → 잘못된 그림 경로 또는 .sty 파일 누락
- "Citation undefined" → .bib 항목이 누락되었거나 bibtex 실행되지 않음

### Step 7.6: Conference-Specific Requirements (학회별 요구사항)

| 학회 | 특별 요구사항 |
|-------|---------------------|
| **NeurIPS** | 부록에 논문 체크리스트, 채택 시 일반인 요약(lay summary) |
| **ICML** | Broader Impact Statement (결론 뒤, 제한 분량 불포함) |
| **ICLR** | LLM 사용 공개 필수, 상호 리뷰(reciprocal reviewing) 동의 |
| **ACL** | 필수 Limitations 섹션, 책임 있는 NLP 체크리스트 |
| **AAAI** | 엄격한 스타일 파일 — 절대 수정 금지 |
| **COLM** | 언어 모델 커뮤니티에 맞게 기여도를 조율할 것 |

### Step 7.7: Conference Resubmission & Format Conversion (학회 재제출 및 서식 변환)

학회 간에 서식을 변환할 때, **절대 LaTeX 프리앰블(preambles)을 복사해서 붙여넣지 마세요**:

```bash
# 1. 대상 템플릿으로 새롭게 시작
cp -r templates/icml2026/ new_submission/

# 2. 내용(content) 섹션만 복사 (프리앰블 제외)
#    - 초록 텍스트, 섹션 내용, 그림, 표, bib 항목 등

# 3. 페이지 제한 분량 조정
# 4. 학회 특화 필수 섹션 추가
# 5. 참고문헌 업데이트
```

| 변환 방향 | 페이지 변동 | 핵심 조정 사항 |
|-----------|-------------|-----------------|
| NeurIPS → ICML | 9 → 8 | 1페이지 축소, Broader Impact 추가 |
| ICML → ICLR | 8 → 9 | 실험 확장, LLM 공개(disclosure) 추가 |
| NeurIPS → ACL | 9 → 8 | NLP 규칙에 맞게 재구성, Limitations 추가 |
| ICLR → AAAI | 9 → 7 | 대폭 축소, 스타일 엄격히 준수 |
| 아무 학회 → COLM | (다양) → 9 | 언어 모델 초점에 맞게 재조정 |

분량을 줄일 때: 증명(proofs)을 부록으로 이동, 관련 연구를 요약, 표 결합, 하위 그림(subfigures) 사용.
분량을 늘릴 때: 아블레이션 추가, 한계점 확장, 추가적인 기준 모델(baselines) 포함, 질적 예시(qualitative examples) 추가.

**거부 후**: 새로운 제출본에서 리뷰어의 의견을 반영하되 (블라인드 리뷰이므로) '변경 사항(changes)' 섹션을 추가하거나 이전 리뷰를 언급하지는 마세요.

### Step 7.8: Camera-Ready Preparation (Post-Acceptance) (카메라 레디 준비 - 채택 후)

논문이 채택된 후, 최종본(camera-ready)을 준비합니다:

```
Camera-Ready 체크리스트:
- [ ] 익명성 해제(De-anonymize): 저자명, 소속, 이메일 추가
- [ ] 감사의 글(Acknowledgments) 추가 (연구비, 컴퓨팅 지원, 유용한 피드백 등)
- [ ] 공개 코드/데이터 URL 추가 (익명 말고 실제 GitHub)
- [ ] 메타 리뷰어의 필수 수정(mandatory revisions) 사항 반영
- [ ] 템플릿을 카메라 레디 모드로 변경 (해당하는 경우 - 예: AAAI \anon → \camera)
- [ ] 학회 요구 시 저작권 고지 추가
- [ ] 텍스트 내 "익명(anonymous)" 자리 표시자 모두 업데이트
- [ ] 최종 PDF가 깔끔하게 컴파일되는지 확인
- [ ] 카메라 레디 버전의 페이지 제한 분량 확인 (제출본과 다를 때가 있음)
- [ ] 보충 자료 (코드, 데이터, 부록) 학회 포털에 업로드
```

### Step 7.9: arXiv & Preprint Strategy (arXiv 및 프리프린트 전략)

arXiv 게시는 ML 분야에서 일반적이지만 시기와 익명성에 대한 주의사항이 있습니다.

**시기 결정 트리:**

| 상황 | 권장사항 |
|-----------|---------------|
| 이중 눈가림 학회(NeurIPS, ICML, ACL)에 제출 | 제출 마감일 **이후** arXiv에 올리기. 마감 전 업로드하면 익명성 정책을 위반할 수 있음 (엄격함 정도는 다름). |
| ICLR 제출 | 제출 전에 올려도 명시적으로 허용. 단, 논문 본문 내에 저자명 기입 금지. |
| 이미 arXiv에 있고, 다른 학회로 제출 | 대부분 학회에서 허용. 그러나 **리뷰 기간 중** 리뷰 내용을 반영하여 버전을 업데이트하지 말 것. |
| 워크샵 논문 | 보통 단일 블라인드(single-blind)나 공개이므로 언제든 올려도 됨. |
| 우선권(priority) 확보 | 연구 선취(scooping)가 걱정되면 즉시 게시 - 단 익명성을 포기하는 대가를 치러야 함. |

**arXiv 카테고리 선택** (ML/AI 논문):

| 카테고리 | 분류 코드 | 대상 분야 |
|----------|------|----------|
| Machine Learning | `cs.LG` | 범용 ML 방법론 |
| Computation and Language | `cs.CL` | NLP, 언어 모델 |
| Artificial Intelligence | `cs.AI` | 추론, 계획, 에이전트 |
| Computer Vision | `cs.CV` | 비전 모델 |
| Information Retrieval | `cs.IR` | 검색, 추천 |

**주 카테고리 + 1-2개의 크로스 리스트 카테고리 기입.** 카테고리가 많을수록 노출도가 올라가지만 진정으로 관련 있는 카테고리만 크로스 리스트 하세요.

**버전 관리 전략:**
- **v1**: 초기 제출본 (학회 제출본과 일치)
- **v2**: 채택 후 카메라 레디 교정본 (초록에 "accepted at [Venue]" 문구 추가)
- 리뷰어의 피드백에 명확히 반응한 변경 사항을 리뷰 기간 동안 v2로 게시하지 마세요

```bash
# arXiv에 논문 제목이 이미 있는지 확인
# (제목 결정 전에 확인용)
pip install arxiv
python -c "
import arxiv
results = list(arxiv.Search(query='ti:\"Your Exact Title\"', max_results=5).results())
print(f'Found {len(results)} matches')
for r in results: print(f'  {r.title} ({r.published.year})')
"
```

### Step 7.10: Research Code Packaging (연구 코드 패키징)

깔끔하고 실행 가능한 코드 배포는 인용 횟수와 리뷰어의 신뢰도를 크게 높여줍니다. 카메라 레디 제출과 함께 코드를 패키징하세요.

**저장소 구조:**

```
your-method/
  README.md              # 설정, 사용법, 재현 지침
  requirements.txt       # 또는 conda용 environment.yml
  setup.py               # pip 설치용 패키지일 경우
  LICENSE                # 연구용으로 MIT 또는 Apache 2.0 권장
  configs/               # 실험 설정 파일들
  src/                   # 핵심 방법론 구현
  scripts/               # 훈련, 평가, 분석 스크립트
    train.py
    evaluate.py
    reproduce_table1.sh  # 주요 결과마다 하나의 스크립트
  data/                  # 소규모 데이터 또는 다운로드 스크립트
    download_data.sh
  results/               # 검증을 위한 예상 출력값
```

**연구 코드용 README 템플릿:**

```markdown
# [Paper Title]

"[Paper Title]" (Venue Year) 의 공식 구현 코드입니다.

## Setup
[환경 설정을 위한 정확한 명령어]

## Reproduction
Table 1 재현하기: `bash scripts/reproduce_table1.sh`
Figure 2 재현하기: `python scripts/make_figure2.py`

## Citation
[BibTeX 항목]
```

**배포 전 체크리스트:**
```
- [ ] 새로 클론한 상태에서 코드가 정상 실행됨 (완전 새 환경이나 Docker에서 테스트)
- [ ] 모든 종속성이 특정 버전으로 고정됨(pinned)
- [ ] 하드코딩된 절대 경로가 없음
- [ ] API 키, 자격 증명, 개인 데이터가 저장소에 없음
- [ ] README에 설정, 재현, 인용 정보가 포함됨
- [ ] LICENSE 파일 있음 (재사용 극대화를 위해 MIT 또는 Apache 2.0)
- [ ] 예상되는 오차 범위 내에서 결과를 재현할 수 있음
- [ ] .gitignore가 데이터 파일, 체크포인트, 로그를 제외시킴
```

**제출용 익명 코드** (채택 전):
```bash
# 이중 눈가림 리뷰를 위해 Anonymous GitHub 사용
# https://anonymous.4open.science/
# 저장소 업로드 → 익명 URL 획득 → 논문에 해당 URL 삽입
```

---

## Phase 8: Post-Acceptance Deliverables (채택 후 작업물)

**목표**: 프레젠테이션 자료 및 커뮤니티 참여를 통해 채택된 논문의 영향력을 극대화합니다.

### Step 8.1: Conference Poster (학술대회 포스터)

대부분의 학회에서는 포스터 세션을 요구합니다. 포스터 디자인 원칙:

| 요소 | 가이드라인 |
|---------|-----------|
| **Size(크기)** | 학회 요구사항 확인 (일반적으로 24"x36" 또는 A0 세로/가로) |
| **Content(내용)** | 제목, 저자, 1문장 요약, 방법론 그림, 2-3개 핵심 결과, 결론 |
| **Flow(흐름)** | 왼쪽 위에서 오른쪽 아래로 (Z 패턴) 또는 열(column) 방식 |
| **Text(글)** | 제목은 3미터, 본문은 1미터 거리에서 보여야 함. 긴 문단 없이 짧은 요점만. |
| **Figures(그림)** | 고해상도 그림 재사용. 핵심 결과는 크게 배치. |

**도구**: LaTeX (`beamerposter` 패키지), PowerPoint/Keynote, Figma, Canva.

**제작**: 학회 2주 전에는 인쇄 주문을 넣으세요. 패브릭 포스터가 여행 시 휴대가 편합니다. 많은 학회에서 가상/디지털 포스터도 지원합니다.

### Step 8.2: Conference Talk / Spotlight (학회 발표 / 스포트라이트)

구두 발표(Oral)나 스포트라이트(Spotlight)에 배정된 경우:

| 발표 유형 | 소요 시간 | 내용 |
|-----------|----------|---------|
| **Spotlight (스포트라이트)** | 5분 | 문제, 접근법, 하나의 핵심 결과. 정확히 5분에 맞춰 연습할 것. |
| **Oral (구두 발표)** | 15-20분 | 전체 내용: 문제, 접근법, 핵심 결과, 아블레이션, 한계점. |
| **Workshop talk (워크샵 발표)** | 10-15분 | 워크샵 청중에 맞게 조정 — 배경지식 설명이 더 필요할 수 있음. |

**슬라이드 디자인 규칙:**
- 한 슬라이드에 하나의 아이디어만
- 글자는 최소화 — 화면에 다 띄우지 말고 입으로 설명하세요
- 핵심 그림에는 애니메이션을 넣어 단계별로 이해시키기
- 마지막에 "시사점(takeaway)" 슬라이드 포함 (한 문장 정렬)
- 예상 질문에 대비한 백업 슬라이드 준비

### Step 8.3: Blog Post / Social Media (블로그 포스트 / 소셜 미디어)

접근하기 쉬운 요약은 영향력을 크게 향상시킵니다:

- **Twitter/X 스레드**: 5-8개의 트윗. 방법론이 아닌 결과부터 시작. Figure 1과 핵심 결과 그림 포함.
- **블로그 포스트**: 800-1500 단어. 리뷰어가 아닌 일반 ML 실무자를 위해 작성. 형식주의(formalism)를 버리고, 직관(intuition)과 실용적인 시사점을 강조.
- **프로젝트 페이지**: 초록, 그림, 데모, 코드 링크, BibTeX이 포함된 HTML 페이지. GitHub Pages 활용.

**타이밍**: 학회 논문집(proceedings)이나 arXiv 카메라 레디 버전이 게시된 후 1-2일 이내에 업로드.

---

## Workshop & Short Papers (워크샵 및 단편 논문)

워크샵 논문과 단편 논문(예: ACL short papers, Findings 논문)도 같은 파이프라인을 따르지만 제약 조건과 기대치가 다릅니다.

### Workshop Papers (워크샵 논문)

| 특징 | 워크샵 | 메인 학회 |
|----------|----------|-----------------|
| **페이지 분량** | 4-6 페이지 (보통) | 7-9 페이지 |
| **리뷰 기준** | 완성도에 대한 허들이 낮음 | 반드시 완결성 있고 철저해야 함 |
| **리뷰 방식** | 주로 단일 블라인드나 가벼운 리뷰 | 이중 눈가림(Double-blind), 엄격함 |
| **주요 평가요소** | 흥미로운 아이디어, 초기 결과, 입장문(position) | 강력한 기준(baselines)을 갖춘 완전한 실증적 이야기 |
| **arXiv** | 언제든 게시 가능 | 시기가 중요함 (arXiv 전략 참조) |
| **기여도 컷** | 새로운 방향, 의미 있는 부정적 결과, 진행 중인 작업 | 강력한 증거가 뒷받침되는 중요한 발전 |

**워크샵을 노려야 할 때:**
- 정식 논문을 쓰기 전, 피드백이 필요한 초기 아이디어
- 8페이지 이상을 할당할 만큼은 아니지만 의미 있는 부정적 결과(negative result)
- 시의적절한 주제에 대한 입장문(position piece)이나 의견
- 복제(Replication) 연구 또는 재현성 보고서

### ACL Short Papers & Findings (ACL 단편 및 Findings)

ACL 학회는 독특한 제출 형식을 가집니다:

| 유형 | 페이지 | 기대치 |
|------|-------|-----------------|
| **Long paper** | 8 | 완결된 연구, 강력한 기준(baselines), 아블레이션 |
| **Short paper** | 4 | 집중적인 기여: 하나의 명확한 포인트와 증거 |
| **Findings** | 8 | 메인 학회에는 아깝게 떨어졌지만 탄탄한 연구 |

**단편 논문(Short paper) 전략**: **단 하나**의 주장을 골라 철저하게 뒷받침하세요. 긴 논문을 4페이지로 압축하려 하지 말고, 아예 다른 집중도 높은 논문을 쓰세요.

---

## Paper Types Beyond Empirical ML (경험적 ML 이외의 논문 유형)

위의 주요 파이프라인은 경험적(empirical) ML 논문을 대상으로 합니다. 다른 논문 유형은 다른 구조와 증거 기준을 요구합니다. 각 유형에 대한 자세한 안내는 [references/paper-types.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/paper-types.md)를 참조하세요.

### Theory Papers (이론 논문)

**구조**: 서론(Introduction) → 예비 지식(Preliminaries - 정의, 기호) → 주요 결과(Main Results - 정리) → 증명 스케치(Proof Sketches) → 토의(Discussion) → 전체 증명(부록)

**경험적 논문과의 주요 차이점:**
- 기여가 실험 수치가 아니라, 정리(theorem), 경계(bound), 또는 불가능성 결과(impossibility result)입니다
- 방법론(Methods) 섹션이 "예비 지식"과 "주요 결과"로 대체됨
- 실험이 아니라 증명이 곧 증거입니다 (물론 이론에 대한 실증적 검증이 추가되면 논문이 강력해짐)
- 본문에는 증명 스케치를, 전체 증명은 부록에 넣는 것이 표준 관행입니다

**증명 작성 원칙:**
- 모든 가정을 명시하여 정리를 공식적으로 기술하세요
- 공식 증명 전에 직관(intuition)을 제공하세요 ("핵심 통찰력은...")
- 증명 스케치는 0.5-1페이지 내에 주요 아이디어를 전달해야 합니다
- `\begin{proof}...\end{proof}` 환경을 사용하세요
- 가정에 번호를 매기고 정리에서 "Assumption 1-3 하에..." 식으로 참조하세요

### Survey / Tutorial Papers (서베이 / 튜토리얼 논문)

**구조**: 서론(Introduction) → 분류 체계/구성(Taxonomy / Organization) → 상세 논의(Detailed Coverage) → 미해결 문제(Open Problems) → 결론(Conclusion)

**주요 차이점:**
- 새로운 방법론이 아니라 논문들의 조직화, 통합, 미해결 문제 식별이 기여입니다
- 범위 내에서는 철저하고 포괄적이어야 합니다 (리뷰어들이 누락된 인용을 찾을 것임)
- 명확한 분류 체계(taxonomy) 또는 조직화 프레임워크가 필수적입니다
- 개별 논문에서는 알 수 없는 논문들 간의 연결성을 보여주어 가치를 창출합니다
- 추천 학술지: TMLR (서베이 트랙), JMLR, Foundations and Trends in ML, ACM Computing Surveys

### Benchmark Papers (벤치마크 논문)

**구조**: 서론(Introduction) → 작업 정의(Task Definition) → 데이터 세트 구축(Dataset Construction) → 기준 모델 평가(Baseline Evaluation) → 분석(Analysis) → 예상 용도 및 한계점(Intended Use & Limitations)

**주요 차이점:**
- 기여가 벤치마크 자체입니다 — 기존 평가의 진정한 틈을 메워야 합니다
- 데이터 세트 문서화가 선택이 아닌 필수입니다 (Datasheets 참조, Step 5.11)
- 벤치마크가 어렵다는 점을 증명해야 합니다 (기준 모델들이 이미 100% 점수를 달성하지 못함을 보여야 함)
- 벤치마크가 실제로 주장하는 바를 측정한다는 것(구인 타당도, construct validity)을 증명해야 합니다
- 추천 학회: NeurIPS Datasets & Benchmarks 트랙, ACL (resource papers), LREC-COLING

### Position Papers (포지션/입장 논문)

**구조**: 서론(Introduction) → 배경(Background) → 주장/논제(Thesis / Argument) → 뒷받침 증거(Supporting Evidence) → 반론(Counterarguments) → 시사점(Implications)

**주요 차이점:**
- 단순한 결과가 아니라 주장(argument)이 기여입니다
- 반론을 진지하게 다루어야 합니다
- 증거는 경험적, 이론적이거나 또는 논리적 분석일 수 있습니다
- 추천 학회: ICML (포지션 트랙), 워크샵, TMLR

---

## Hermes Agent Integration (Hermes 에이전트 통합)

이 스킬은 Hermes 에이전트용으로 설계되었습니다. 전체 연구 수명 주기를 위해 Hermes 도구, 위임(delegation), 스케줄링, 메모리 기능을 활용합니다.

### Related Skills (관련 스킬)

특정 단계를 위해 이 스킬을 다른 Hermes 스킬과 결합하세요:

| 스킬 | 사용 시기 | 로드 방법 |
|-------|-------------|-------------|
| **arxiv** | Phase 1 (문헌 검토): arXiv 검색, BibTeX 생성, Semantic Scholar를 통한 관련 논문 찾기 | `skill_view("arxiv")` |
| **subagent-driven-development** | Phase 5 (초안 작성): 병렬 섹션 작성과 2단계 리뷰 (사양 준수 확인 후 품질 확인) | `skill_view("subagent-driven-development")` |
| **plan** | Phase 0 (설정): 실행 전 구조화된 계획 수립. `.hermes/plans/` 에 작성 | `skill_view("plan")` |
| **qmd** | Phase 1 (문헌): 하이브리드 BM25+벡터 검색으로 로컬 지식 기반(노트, 기록, 문서) 검색 | 설치: `skill_manage("install", "qmd")` |
| **diagramming** | Phase 4-5: Excalidraw 기반 그림과 아키텍처 다이어그램 생성 | `skill_view("diagramming")` |
| **data-science** | Phase 4 (분석): 대화형 분석과 시각화를 위한 Jupyter 라이브 커널 | `skill_view("data-science")` |

**이 스킬은 `ml-paper-writing`을 대체합니다** — `ml-paper-writing`의 모든 내용에 전체 실험/분석 파이프라인 및 autoreason 방법론이 추가되었습니다.

### Hermes Tools Reference (Hermes 도구 참조)

| 도구 | 이 파이프라인에서의 용도 |
|------|----------------------|
| **`terminal`** | LaTeX 컴파일 (`latexmk -pdf`), git 조작, 실험 실행 (`nohup python run.py &`), 프로세스 확인 |
| **`process`** | 백그라운드 실험 관리: `process("start", ...)`, `process("poll", pid)`, `process("log", pid)`, `process("kill", pid)` |
| **`execute_code`** | 인용 확인, 통계 분석, 데이터 집계를 위해 Python 실행. RPC를 통한 도구 접근 가능. |
| **`read_file`** / **`write_file`** / **`patch`** | 논문 편집, 실험 스크립트, 결과 파일 처리. 대규모 .tex 파일의 부분 편집은 `patch` 활용. |
| **`web_search`** | 문헌 검색: `web_search("transformer attention mechanism 2024")` |
| **`web_extract`** | 논문 내용 가져오기, 인용 확인: `web_extract("https://arxiv.org/abs/2303.17651")` |
| **`delegate_task`** | **병렬 섹션 초안 작성** — 각 섹션마다 독립된 하위 에이전트(subagents) 생성. 또는 동시 인용 확인 작업용. |
| **`todo`** | 여러 세션에 걸친 주요 상태 추적기. 각 Phase 전환 시마다 업데이트. |
| **`memory`** | 세션 간 주요 결정 유지: 기여(contribution) 방향, 목표 학회, 리뷰어 피드백 등. |
| **`cronjob`** | 실험 모니터링, 마감일 카운트다운, 자동 arXiv 확인 스케줄링. |
| **`clarify`** | 블로킹 상황(목표 학회, 기여 방향 설정 등)에서 사용자에게 구체적인 질문 요청. |
| **`send_message`** | 사용자가 채팅 중이 아니더라도 실험 완료나 초안 준비가 완료되면 알림 발송. |

### Tool Usage Patterns (도구 사용 패턴)

**실험 모니터링** (가장 흔함):
```
terminal("ps aux | grep <pattern>")
→ terminal("tail -30 <logfile>")
→ terminal("ls results/")
→ execute_code("analyze results JSON, compute metrics")
→ terminal("git add -A && git commit -m '<descriptive message>' && git push")
→ send_message("Experiment complete: <summary>")
```

**병렬 섹션 초안 작성** (위임(delegation) 활용):
```
delegate_task("이 실험 스크립트와 설정 파일들을 바탕으로 방법론(Methods) 섹션을 작성하라.
  포함할 내용: 의사코드, 모든 초매개변수, 재현 가능한 세부 아키텍처 정보.
  neurips2025 템플릿 양식에 맞춰 LaTeX으로 작성하라.")

delegate_task("관련 연구(Related Work) 섹션을 작성하라. web_search와 web_extract를 사용해
  논문을 찾고 Semantic Scholar를 통해 모든 인용을 검증하라. 방법론 기준으로 분류하라.")

delegate_task("실험(Experiments) 섹션을 작성하라. results/ 폴더 안의 모든 결과 파일을 읽고,
  각 실험이 어떤 주장을 뒷받침하는지 명시하라. 오차 막대와 유의성 지표를 포함하라.")
```

각 위임 작업은 공유된 문맥이 없는 **완전 새 하위 에이전트**로 실행되므로, 필요한 모든 정보를 프롬프트에 제공하세요. 출력을 모아서 통합합니다.

**인용 검증** (`execute_code` 활용):
```python
# execute_code 내에서:
from semanticscholar import SemanticScholar
import requests

sch = SemanticScholar()
results = sch.search_paper("attention mechanism transformers", limit=5)
for paper in results:
    doi = paper.externalIds.get('DOI', 'N/A')
    if doi != 'N/A':
        bibtex = requests.get(f"https://doi.org/{doi}", 
                              headers={"Accept": "application/x-bibtex"}).text
        print(bibtex)
```

### State Management with `memory` and `todo` (메모리 및 Todo를 활용한 상태 관리)

**`memory` 도구** — 핵심 결정사항 보존 (크기 제한: MEMORY.md의 경우 약 2200자):

```
memory("add", "논문: autoreason. 학회: NeurIPS 2025 (9페이지).
  기여점: 생성-평가 격차가 클 때 구조화된 세부조정(refinement)이 효과가 있다.
  핵심 결과: Haiku 42/42, Sonnet 3/5, S4.6 constrained 2/3.
  상태: Phase 5 — 방법론(Methods) 섹션 작성 중.")
```

중요한 결정이나 Phase를 전환한 후에 메모리를 업데이트하세요. 이는 여러 세션에 걸쳐 지속됩니다.

**`todo` 도구** — 세부적인 진행 상황 추적:

```
todo("add", "Sonnet 4.6에 대한 제한된 작업(constrained task) 실험 설계")
todo("add", "Haiku 기준 모델 비교 실행")
todo("add", "방법론 섹션 초안 작성")
todo("update", id=3, status="in_progress")
todo("update", id=1, status="completed")
```

**세션 시작 프로토콜:**
```
1. todo("list")                           # 현재 작업 목록 확인
2. memory("read")                         # 주요 결정사항 불러오기
3. terminal("git log --oneline -10")      # 최근 커밋 내역 확인
4. terminal("ps aux | grep python")       # 실행 중인 실험 확인
5. terminal("ls results/ | tail -20")     # 새 결과물 확인
6. 사용자에게 상태를 보고하고 방향 요청
```

### Cron Monitoring with `cronjob` (cronjob을 활용한 모니터링)

주기적인 실험 점검에 `cronjob` 도구 사용:

```
cronjob("create", {
  "schedule": "*/30 * * * *",  # 30분마다
  "prompt": "실험 상태 확인:
    1. ps aux | grep run_experiment
    2. tail -30 logs/experiment_haiku.log
    3. ls results/haiku_baselines/
    4. 완료되었다면: 결과 읽기, Borda 점수 계산,
       git add -A && git commit -m 'Add Haiku results' && git push
    5. 보고: 결과 표, 핵심 발견, 다음 단계
    6. 이전과 바뀐 게 없다면: [SILENT] 로만 응답할 것"
})
```

**[SILENT] 프로토콜**: 마지막 점검 이후 아무 것도 변하지 않았다면 오직 `[SILENT]` 로만 응답하세요. 이렇게 하면 사용자에게 알림이 전송되는 것을 억제합니다. 알릴 만한 진짜 변화가 있을 때만 보고하세요.

**마감일 추적**:
```
cronjob("create", {
  "schedule": "0 9 * * *",  # 매일 오전 9시
  "prompt": "NeurIPS 2025 마감일: May 22. 오늘은 {date} 입니다.
    남은 일수: {compute}.
    todo 목록 확인 — 계획대로 잘 진행되고 있습니까?
    만약 7일 미만 남았다면: 미완료 작업에 대해 사용자에게 경고하세요."
})
```

### Communication Patterns (커뮤니케이션 패턴)

**사용자에게 알림을 보낼 때** (`send_message` 또는 직접 답변을 통해):
- 일련의 실험 배치(batch)가 완료되었을 때 (결과 표와 함께)
- 결정이 필요한 예상치 못한 발견이나 오류가 발생했을 때
- 리뷰 받을 준비가 된 초안 섹션이 있을 때
- 작업은 덜 끝났는데 마감일이 임박했을 때

**알림을 보내지 말아야 할 때:**
- 실험이 아직 진행 중이고 새로운 결과가 없을 때 → `[SILENT]`
- 변화가 없는 일상적인 모니터링일 때 → `[SILENT]`
- 확인받지 않아도 되는 중간 단계일 때

**보고 형식** — 항상 구조화된 데이터를 포함하세요:
```
## 실험: <이름>
상태: Complete / Running / Failed

| 작업 | Method A | Method B | Method C |
|------|---------|---------|---------|
| Task 1 | 85.2 | 82.1 | **89.4** |

핵심 발견: <한 문장>
다음 단계: <다음에 일어날 일>
```

### Decision Points Requiring Human Input (인간의 결정이 필요한 지점)

정말로 멈춰 있을 때만 구체적인 질문에 `clarify`를 사용하세요:

| 결정 항목 | 물어볼 시점 |
|----------|-------------|
| 목표 학회 | 논문 시작 전 (분량 제한과 톤 매너에 영향 미침) |
| 기여도 방향(framing) | 타당한 방향이 여러 개 존재할 때 |
| 실험 우선순위 | 시간이 부족한데 TODO에 실험이 너무 많을 때 |
| 제출 준비 여부 | 최종 제출 직전 |

**물어보지 말아야 할 것** (스스로 결정하고 표시만 해두세요):
- 단어 선택, 섹션 순서 배열
- 어떤 특정 결과를 눈에 띄게 할 것인가
- 인용 문헌의 완결성 (일단 찾은 것으로 초안을 쓰고, 빈 공간은 메모로 남길 실 것)

---

## Reviewer Evaluation Criteria (리뷰어 평가 기준)

리뷰어들이 무엇을 보는지 이해하면 노력의 방향을 잡기 쉽습니다:

| 기준 | 확인하는 내용 |
|-----------|----------------|
| **품질 (Quality)** | 기술적 타당성, 주장을 뒷받침하는 근거, 공정한 기준(baselines) |
| **명확성 (Clarity)** | 명확한 글, 전문가가 재현할 수 있는지, 일관성 있는 수학 기호 표기 |
| **중요도 (Significance)** | 커뮤니티 파급력, 학문적 이해를 발전시키는지 여부 |
| **독창성 (Originality)** | 새로운 통찰력 (반드시 새로운 방법론이어야 할 필요는 없음) |

**채점 (NeurIPS 6점 척도):**
- 6: Strong Accept — 획기적이고 결함이 없음
- 5: Accept — 기술적으로 탄탄하고 영향력이 큼
- 4: Borderline Accept — 탄탄하지만 평가가 다소 제한적임
- 3: Borderline Reject — 강점보다 약점이 큼
- 2: Reject — 기술적 결함이 있음
- 1: Strong Reject — 이미 알려진 결과이거나 윤리적 문제가 있음

상세한 가이드라인, 자주 제기되는 문제점, 그리고 반박(rebuttal) 전략은 [references/reviewer-guidelines.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/reviewer-guidelines.md)를 참조하세요.

---

## Common Issues and Solutions (일반적인 문제 및 해결책)

| 문제점 | 해결책 |
|-------|----------|
| 초록이 너무 일반적임 | 어떤 ML 논문에도 붙을 수 있는 첫 문장이라면 삭제하세요. 당신만의 특정 기여(contribution)로 바로 시작하세요. |
| 서론이 1.5페이지 초과 | 배경 지식은 '관련 연구(Related Work)'로 빼세요. 기여(contribution) 목록을 앞쪽으로 당기세요. |
| 실험 파트에 주장이 명확하지 않음 | 각 실험 전에 "이 실험은 [특정 주장]을 검증합니다..."라고 명시하세요. |
| 리뷰어가 논문을 따라가기 어렵다고 함 | 이정표(signposting)를 추가하고, 일관된 용어를 쓰며, 그림 캡션만 읽어도 이해되게 만드세요. |
| 통계적 유의성이 누락됨 | 오차 막대, 실행(run) 횟수, 통계적 테스트, 신뢰 구간을 추가하세요. |
| 실험이 범위(scope)를 벗어나 늘어남 | 모든 실험은 특정 주장에 맵핑되어야 합니다. 그렇지 않은 실험은 과감히 삭제하세요. |
| 논문이 거절되어 재제출해야 함 | Phase 7의 학회 재제출을 참조하세요. 이전에 받은 리뷰를 직접 언급하지 말고 문제점만 논문에 고쳐서 제출하세요. |
| 넓은 영향(Broader impact) 성명이 없음 | Step 5.10을 보세요. 대부분 학회가 필수로 요구합니다. "어떤 부정적 영향도 없다"는 주장은 거의 믿어주지 않습니다. |
| 인간 평가(Human eval)가 부실하다고 비판받음 | Step 2.5 및 [references/human-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/human-evaluation.md) 참조. 일치도 지표(agreement metrics), 평가자 상세 정보, 보상 내역을 명시하세요. |
| 리뷰어가 재현성을 의심함 | 코드를 배포하고(Step 7.9), 모든 초매개변수를 문서화하며, 시드(seeds)와 컴퓨팅 제원을 명시하세요. |
| 이론 논문에 직관(intuition)이 없음 | 공식 증명 전에 쉬운 말로 푼 증명 스케치를 추가하세요. [references/paper-types.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/paper-types.md) 참조. |
| 결과가 부정적/무의미함(null) | 부정적 결과 처리에 대한 Phase 4.3을 보세요. 워크샵, TMLR을 고려하거나 '분석(analysis)' 논문으로 방향을 트는 것을 고려하세요. |

---

## Reference Documents (참고 문서)

| 문서 | 내용 |
|----------|----------|
| [references/writing-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/writing-guide.md) | Gopen & Swan의 7원칙, Perez의 미시적 팁, Lipton의 단어 선택, Steinhardt의 정확성, 그림 디자인 |
| [references/citation-workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/citation-workflow.md) | 인용 API, Python 코드, CitationManager 클래스, BibTeX 관리 |
| [references/checklists.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/checklists.md) | NeurIPS 16개 항목, ICML, ICLR, ACL 요구사항, 통합 제출 전 체크리스트 |
| [references/reviewer-guidelines.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/reviewer-guidelines.md) | 평가 기준, 점수 매기기, 흔히 지적되는 문제들, 반박(rebuttal) 템플릿 |
| [references/sources.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/sources.md) | 모든 작성 가이드, 학회 지침, API에 대한 전체 참고 문헌 |
| [references/experiment-patterns.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/experiment-patterns.md) | 실험 설계 패턴, 평가 프로토콜, 모니터링, 오류 복구 |
| [references/autoreason-methodology.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/autoreason-methodology.md) | Autoreason 루프, 전략 선택, 모델 가이드, 프롬프트, 범위 제약, Borda 점수 체계 |
| [references/human-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/human-evaluation.md) | 인간 평가 설계, 주석 가이드라인, 일치도 지표, 크라우드소싱 품질 관리(QC), IRB 가이드라인 |
| [references/paper-types.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/paper-types.md) | 이론 논문 (증명 작성, 정리 구조), 서베이 논문, 벤치마크 논문, 포지션 논문 |

### LaTeX Templates (LaTeX 템플릿)

`templates/` 디렉토리에 위치: **NeurIPS 2025**, **ICML 2026**, **ICLR 2026**, **ACL**, **AAAI 2026**, **COLM 2025**.

컴파일 안내는 [templates/README.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/templates/README.md)를 확인하세요.

### Key External Sources (주요 외부 출처)

**작성 철학 (Writing Philosophy):**
- [Neel Nanda: How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)
- [Sebastian Farquhar: How to Write ML Papers](https://sebastianfarquhar.com/on-research/2024/11/04/how_to_write_ml_papers/)
- [Gopen & Swan: Science of Scientific Writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- [Lipton: Heuristics for Scientific Writing](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/)
- [Perez: Easy Paper Writing Tips](https://ethanperez.net/easy-paper-writing-tips/)

**API:** [Semantic Scholar](https://api.semanticscholar.org/api-docs/) | [CrossRef](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | [arXiv](https://info.arxiv.org/help/api/basics.html)

**학회 (Venues):** [NeurIPS](https://neurips.cc/Conferences/2025/PaperInformation/StyleFiles) | [ICML](https://icml.cc/Conferences/2025/AuthorInstructions) | [ICLR](https://iclr.cc/Conferences/2026/AuthorGuide) | [ACL](https://github.com/acl-org/acl-style-files)
