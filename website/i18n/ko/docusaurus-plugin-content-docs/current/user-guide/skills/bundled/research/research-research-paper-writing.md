---
title: "연구 논문 작성 — NeurIPS/ICML/ICLR용 ML 논문 작성: 설계→제출"
sidebar_label: "연구 논문 작성"
description: "NeurIPS/ICML/ICLR용 ML 논문 작성: 설계→제출"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# 연구 논문 작성

NeurIPS/ICML/ICLR용 ML 논문을 작성합니다: 설계→제출.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨 (기본 설치) |
| 경로 | `skills/research/research-paper-writing` |
| 버전 | `1.1.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `semanticscholar`, `arxiv`, `habanero`, `requests`, `scipy`, `numpy`, `matplotlib`, `SciencePlots` |
| 플랫폼 | linux, macos |
| 태그 | `Research`, `Paper Writing`, `Experiments`, `ML`, `AI`, `NeurIPS`, `ICML`, `ICLR`, `ACL`, `AAAI`, `COLM`, `LaTeX`, `Citations`, `Statistical Analysis` |
| 관련 스킬 | [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv), [`subagent-driven-development`](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development), [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 활성 상태에서 에이전트가 보게 되는 지침입니다.
:::

# 연구 논문 작성 파이프라인

**NeurIPS, ICML, ICLR, ACL, AAAI, COLM**을 대상으로 출판 가능한 수준의 ML/AI 연구 논문을 작성하기 위한 엔드투엔드 파이프라인입니다. 이 스킬은 연구의 전체 수명 주기인 실험 설계, 실행, 모니터링, 분석, 논문 작성, 검토, 수정, 제출을 다룹니다.

이것은 **선형 파이프라인이 아니라** 반복 루프입니다. 결과가 새로운 실험을 촉발하고, 검토가 추가 분석을 촉발합니다. 에이전트는 이러한 피드백 루프를 처리해야 합니다.

<!-- ascii-guard-ignore -->
<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────────────┐
│                    RESEARCH PAPER PIPELINE                  │
│                                                             │
│  Phase 0: Project Setup ──► Phase 1: Literature Review      │
│       │                          │                          │
│       ▼                          ▼                          │
│  Phase 2: Experiment     Phase 5: Paper Drafting ◄──┐      │
│       Design                     │                   │      │
│       │                          ▼                   │      │
│       ▼                    Phase 6: Self-Review      │      │
│  Phase 3: Execution &           & Revision ──────────┘      │
│       Monitoring                 │                          │
│       │                          ▼                          │
│       ▼                    Phase 7: Submission               │
│  Phase 4: Analysis ─────► (feeds back to Phase 2 or 5)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->
<!-- ascii-guard-ignore-end -->

---

## 이 스킬을 사용할 때

다음과 같은 경우 이 스킬을 사용합니다:
- **새 연구 논문을 시작할 때** — 기존 코드베이스나 아이디어에서 출발하는 경우
- **논문의 주장을 뒷받침할 실험을 설계하고 실행할 때**
- **연구 논문의 어떤 섹션이든 작성하거나 수정할 때**
- **특정 학회나 워크숍에 제출을 준비할 때**
- **추가 실험이나 수정으로 리뷰에 답변할 때**
- **학회 형식 간에 논문을 변환할 때**
- **비경험적 논문을 작성할 때** — 이론, 서베이, 벤치마크 또는 포지션 논문 ([경험적 ML 이외의 논문 유형](#paper-types-beyond-empirical-ml) 참고)
- **NLP, HCI 또는 얼라인먼트 연구를 위한 사람 대상 평가를 설계할 때**
- **승인 후 산출물을 준비할 때** — 포스터, 발표 자료, 코드 공개

## 핵심 철학

1. **선제적으로 행동하세요.** 질문이 아니라 완성된 초안을 제공하세요. 과학자는 바쁩니다 — 먼저 구체적인 결과물을 만들어 반응할 수 있게 한 다음 반복해서 개선하세요.
2. **인용을 절대 지어내지 마세요.** AI가 생성한 인용은 약 40%의 오류율을 보입니다. 항상 프로그래밍 방식으로 가져오세요. 검증할 수 없는 인용은 `[CITATION NEEDED]`로 표시하세요.
3. **논문은 실험 모음이 아니라 이야기입니다.** 모든 논문에는 한 문장으로 명확하게 말할 수 있는 하나의 기여가 필요합니다. 그렇게 말할 수 없다면 논문은 아직 준비되지 않은 것입니다.
4. **실험은 주장을 뒷받침합니다.** 모든 실험은 어떤 주장을 뒷받침하는지 명시해야 합니다. 논문의 서사와 연결되지 않는 실험은 절대 실행하지 마세요.
5. **일찍 커밋하고 자주 커밋하세요.** 완료된 각 실험 배치와 논문 초안 업데이트마다 설명적인 메시지로 커밋하세요. Git 로그는 실험의 이력입니다.

### 선제적 작업과 협업

**기본값: 먼저 초안을 작성하고, 초안과 함께 질문하세요.**

| 확신 수준 | 행동 |
|---------|--------|
| **높음** (명확한 저장소, 분명한 기여) | 전체 초안을 작성하고 전달한 뒤 피드백을 반영 |
| **중간** (일부 모호함) | 불확실성을 표시한 초안을 작성하고 계속 진행 |
| **낮음** (주요 미지수) | `clarify`를 통해 1~2개의 핵심 질문을 한 다음 초안 작성 |

| 섹션 | 자율적으로 초안 작성? | 초안과 함께 표시할 내용 |
|---------|---------------------|-------------------|
| 초록 | 예 | "기여를 X로 정리했습니다 — 필요하면 조정하세요" |
| 서론 | 예 | "문제 Y를 강조했습니다 — 맞는지 확인하세요" |
| 방법론 | 예 | "세부 사항 A, B, C를 포함했습니다 — 빠진 부분을 추가하세요" |
| 실험 | 예 | "결과 1, 2, 3을 강조했습니다 — 필요하면 순서를 바꾸세요" |
| 관련 연구 | 예 | "논문 X, Y, Z를 인용했습니다 — 빠진 것이 있으면 추가하세요" |

**다음과 같은 경우에만** 입력을 요청하며 작업을 막으세요: 대상 학회가 불분명하거나, 서로 모순되는 프레이밍이 여러 개 있거나, 결과가 불완전해 보이거나, 먼저 검토해 달라는 명시적인 요청이 있는 경우.

---

## 0단계: 프로젝트 설정

**목표**: 작업 공간을 구축하고, 기존 작업을 파악하며, 기여를 식별합니다.

### 0.1단계: 저장소 탐색

```bash
# Understand project structure
ls -la
find . -name "*.py" | head -30
find . -name "*.md" -o -name "*.txt" | xargs grep -l -i "result\|conclusion\|finding"
```

다음 항목을 확인하세요:
- `README.md` — 프로젝트 개요와 주장
- `results/`, `outputs/`, `experiments/` — 기존 발견
- `configs/` — 실험 설정
- `.bib` 파일 — 기존 인용
- 초안 문서 또는 메모

### 0.2단계: 작업 공간 구성

일관된 작업 공간 구조를 구축하세요:

```
workspace/
  paper/               # LaTeX source, figures, compiled PDFs
  experiments/         # Experiment runner scripts
  code/                # Core method implementation
  results/             # Raw experiment results (auto-generated)
  tasks/               # Task/benchmark definitions
  human_eval/          # Human evaluation materials (if needed)
```

### 0.3단계: 버전 관리 설정

```bash
git init  # if not already
git remote add origin <repo-url>
git checkout -b paper-draft  # or main
```

**Git 규율**: 완료된 모든 실험 배치는 설명적인 메시지로 커밋합니다. 예:
```
Add Monte Carlo constrained results (5 runs, Sonnet 4.6, policy memo task)
Add Haiku baseline comparison: autoreason vs refinement baselines at cheap model tier
```

### 0.4단계: 기여 식별

무엇이든 작성하기 전에 다음을 명확히 표현하세요:
- **무엇**: 이 논문이 기여하는 단 하나의 것은 무엇인가요?
- **왜**: 이를 뒷받침하는 증거는 무엇인가요?
- **그래서 무엇**: 독자가 왜 관심을 가져야 하나요?

> 과학자에게 다음과 같이 제안하세요. "제가 이해한 바에 따르면 주요 기여는 다음과 같습니다: [한 문장]. 핵심 결과는 [Y]를 보여줍니다. 이 프레이밍을 원하시나요?"

### 0.5단계: TODO 목록 만들기

`todo` 도구를 사용해 구조화된 프로젝트 계획을 만드세요:

```
Research Paper TODO:
- [ ] Define one-sentence contribution
- [ ] Literature review (related work + baselines)
- [ ] Design core experiments
- [ ] Run experiments
- [ ] Analyze results
- [ ] Write first draft
- [ ] Self-review (simulate reviewers)
- [ ] Revise based on review
- [ ] Submission prep
```

프로젝트가 진행되는 동안 이를 업데이트하세요. 세션 간 지속되는 상태로 활용됩니다.

### 0.6단계: 컴퓨팅 예산 추정

실험을 실행하기 전에 총 비용과 시간을 추정하세요:

```
Compute Budget Checklist:
- [ ] API costs: (model price per token) × (estimated tokens per run) × (number of runs)
- [ ] GPU hours: (time per experiment) × (number of experiments) × (number of seeds)
- [ ] Human evaluation costs: (annotators) × (hours) × (hourly rate)
- [ ] Total budget ceiling and contingency (add 30-50% for reruns)
```

실험을 실행하면서 실제 지출을 추적하세요:
```python
# Simple cost tracker pattern
import json, os
from datetime import datetime

COST_LOG = "results/cost_log.jsonl"

def log_cost(experiment: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }
    with open(COST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**예산이 빠듯할 때**: 전체 스윕을 시작하기 전에 파일럿 실험(시드 1~2개, 작업의 일부)을 실행하세요. 파이프라인 디버깅에는 더 저렴한 모델을 사용한 다음 최종 실행에서는 대상 모델로 전환하세요.

### 0.7단계: 여러 저자 간 조정

대부분의 논문에는 3~10명의 저자가 있습니다. 일찍부터 작업 흐름을 수립하세요:

| 작업 흐름 | 도구 | 사용 시점 |
|-------------|------|-----------|
| **Overleaf** | 브라우저 기반 | 여러 저자가 동시에 편집하거나 git 경험이 없을 때 |
| **Git + LaTeX** | 보조 파일용 `.gitignore`를 포함한 `git` | 기술 팀, 브랜치 기반 검토가 필요할 때 |
| **Overleaf + Git 동기화** | Overleaf premium | 두 방식의 장점을 모두 활용 — 실시간 협업과 버전 이력 |

**섹션 소유권**: 각 섹션에 주요 저자 한 명을 지정하세요. 다른 사람들은 직접 편집하지 않고 의견을 남깁니다. 이렇게 하면 병합 충돌과 스타일 불일치를 방지할 수 있습니다.

```
Author Coordination Checklist:
- [ ] Agree on section ownership (who writes what)
- [ ] Set up shared workspace (Overleaf or git repo)
- [ ] Establish notation conventions (before anyone writes)
- [ ] Schedule internal review rounds (not just at the end)
- [ ] Designate one person for final formatting pass
- [ ] Agree on figure style (colors, fonts, sizes) before creating figures
```

**미리 합의해야 할 LaTeX 규칙**:
- 일관된 방법 이름을 위한 `\method{}` 매크로
- 인용 스타일: `\citet{}`와 `\citep{}` 사용 방식
- 수학 표기법: 벡터는 소문자 굵게, 행렬은 대문자 굵게 등
- 영국식 철자와 미국식 철자 중 하나 선택

---

## 1단계: 문헌 검토

**목표**: 관련 연구를 찾고, 베이스라인을 식별하며, 인용을 수집합니다.

### 1.1단계: 시드 논문 식별

코드베이스에서 이미 참조된 논문부터 시작하세요:

```bash
# Via terminal:
grep -r "arxiv\|doi\|cite" --include="*.md" --include="*.bib" --include="*.py"
find . -name "*.bib"
```

### 1.2단계: 관련 연구 검색

구조화된 논문 탐색을 위해 `arxiv` 스킬을 **로드하세요**: `skill_view("arxiv")`. 이 스킬은 arXiv REST API 검색, Semantic Scholar 인용 그래프, 저자 프로필, BibTeX 생성을 제공합니다.

폭넓은 탐색에는 `web_search`를, 특정 논문을 가져올 때는 `web_extract`를 사용하세요:

```
# Via web_search:
web_search("[main technique] + [application domain] site:arxiv.org")
web_search("[baseline method] comparison ICML NeurIPS 2024")

# Via web_extract (for specific papers):
web_extract("https://arxiv.org/abs/2303.17651")
```

시도해 볼 추가 검색 쿼리:

```
Search queries:
- "[main technique] + [application domain]"
- "[baseline method] comparison"
- "[problem name] state-of-the-art"
- Author names from existing citations
```

**권장**: 실시간 학술 검색을 위해 **Exa MCP**를 설치하세요:
```bash
claude mcp add exa -- npx -y mcp-remote "https://mcp.exa.ai/mcp"
```

### 1.2b단계: 검색 심화

단 한 번의 질의만 수행하는 평면적 검색은 관련 연구를 놓치는 경우가 많습니다. 심층 연구 파이프라인에서 영감을 얻은 반복적인 폭 우선, 깊이 후속 패턴을 사용하세요:

```
Iterative Literature Search:

Round 1 (Breadth): 4-6 parallel queries covering different angles
  - "[method] + [domain]"
  - "[problem name] state-of-the-art 2024 2025"
  - "[baseline method] comparison"
  - "[alternative approach] vs [your approach]"
  → Collect papers, extract key concepts and terminology

Round 2 (Depth): Generate follow-up queries from Round 1 learnings
  - New terminology discovered in Round 1 papers
  - Papers cited by the most relevant Round 1 results
  - Contradictory findings that need investigation
  → Collect papers, identify remaining gaps

Round 3 (Targeted): Fill specific gaps
  - Missing baselines identified in Rounds 1-2
  - Concurrent work (last 6 months, same problem)
  - Key negative results or failed approaches
  → Stop when new queries return mostly papers you've already seen
```

**중단 시점**: 어떤 라운드에서든 반환된 논문의 80%를 초과하는 논문이 기존 수집 목록에 있다면 검색이 포화된 것입니다. 일반적으로 2~3라운드면 충분합니다. 서베이 논문은 4~5라운드를 예상하세요.

**에이전트 기반 워크플로의 경우**: `delegate_task`를 통해 각 라운드의 질의를 병렬로 위임하세요. 결과를 수집하고 중복을 제거한 다음, 통합된 학습 내용을 바탕으로 다음 라운드의 질의를 생성하세요.

### 1.3단계: 모든 인용 검증

**절대로 기억에 의존해 BibTeX를 생성하지 마세요. 항상 프로그래밍 방식으로 가져오세요.**

각 인용에 대해 다음의 필수 5단계 프로세스를 따르세요:

```
Citation Verification (MANDATORY per citation):
1. SEARCH → Query Semantic Scholar or Exa MCP with specific keywords
2. VERIFY → Confirm paper exists in 2+ sources (Semantic Scholar + arXiv/CrossRef)
3. RETRIEVE → Get BibTeX via DOI content negotiation (programmatically, not from memory)
4. VALIDATE → Confirm the claim you're citing actually appears in the paper
5. ADD → Add verified BibTeX to bibliography
If ANY step fails → mark as [CITATION NEEDED], inform scientist
```

```python
# Fetch BibTeX via DOI
import requests

def doi_to_bibtex(doi: str) -> str:
    response = requests.get(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/x-bibtex"}
    )
    response.raise_for_status()
    return response.text
```

인용을 검증할 수 없는 경우:

```latex
\cite{PLACEHOLDER_author2024_verify_this}  % TODO: Verify this citation exists
```

**과학자에게 항상 알리세요**: "검증이 필요한 인용 [X]개를 자리표시자로 표시했습니다."

전체 API 문서와 완전한 `CitationManager` 클래스는 [references/citation-workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/citation-workflow.md)를 참조하세요.

### 1.4단계: 관련 연구 구성

논문별이 아니라 방법론별로 논문을 묶으세요:

**좋은 예**: "한 연구 흐름은 X의 가정을 사용하는 반면 [refs], 우리는 Y의 가정을 사용한다. 그 이유는..."
**나쁜 예**: "Smith 등은 X를 도입했다. Jones 등은 Y를 도입했다. 우리는 둘을 결합한다."

---

## 2단계: 실험 설계

**목표**: 논문의 주장을 직접 뒷받침하는 실험을 설계합니다. 모든 실험은 구체적인 질문에 답해야 합니다.

### 2.1단계: 주장을 실험에 매핑

명시적인 매핑을 만드세요:

| 주장 | 실험 | 예상 증거 |
|-------|-----------|-------------------|
| "우리 방법이 기준선보다 우수하다" | 주요 비교(표 1) | 승률, 통계적 유의성 |
| "효과는 더 약한 모델에서 더 크다" | 모델 스케일링 연구 | 단조 증가 개선 곡선 |
| "수렴에는 범위 제약이 필요하다" | 제약 적용 대 비적용 | 수렴률 비교 |

**규칙**: 주장에 매핑되지 않는 실험은 실행하지 마세요.

### 2.2단계: 기준선 설계

강력한 기준선이 합격 논문과 탈락 논문을 가릅니다. 리뷰어는 "X와 비교했나요?"라고 질문할 것입니다.

표준 기준선 범주:
- **단순 기준선**: 가능한 가장 단순한 접근법
- **강력한 기준선**: 알려진 기존 방법 중 가장 우수한 방법
- **절제 기준선**: 구성 요소 하나를 제외한 당신의 방법
- **계산량 일치 기준선**: 동일한 계산 예산, 다른 할당

### 2.3단계: 평가 프로토콜 정의

무엇이든 실행하기 전에 다음을 지정하세요:
- **지표**: 무엇을 측정하는지, 방향 기호(높을수록/낮을수록 좋음)
- **집계**: 여러 실행/작업에 걸쳐 결과를 결합하는 방법
- **통계 검정**: 유의성을 입증할 검정
- **표본 크기**: 실행/문제/작업 수

### 2.4단계: 실험 스크립트 작성

성공적인 연구 파이프라인에서 다음 패턴을 따르세요:

**점진적 저장** — 충돌 복구를 위해 각 단계 후 결과를 저장합니다:
```python
# Save after each problem/task
result_path = f"results/{task}/{strategy}/result.json"
if os.path.exists(result_path):
    continue  # Skip already-completed work
# ... run experiment ...
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)
```

**아티팩트 보존** — 모든 중간 산출물을 저장합니다:
```
results/<experiment>/
  <task>/
    <strategy>/
      final_output.md          # Final result
      history.json             # Full trajectory
      pass_01/                 # Per-iteration artifacts
        version_a.md
        version_b.md
        critic.md
```

**관심사 분리** — 생성, 평가, 시각화를 분리해 유지합니다:
```
run_experiment.py              # Core experiment runner
run_baselines.py               # Baseline comparison
run_comparison_judge.py        # Blind evaluation
analyze_results.py             # Statistical analysis
make_charts.py                 # Visualization
```

전체 설계 패턴, cron 모니터링, 오류 복구는 [references/experiment-patterns.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/experiment-patterns.md)를 참조하세요.

### 2.5단계: 사람 평가 설계(해당하는 경우)

많은 NLP, HCI, 얼라인먼트 논문은 주요 또는 보완적 증거로 사람 평가를 요구합니다. 자동화된 실험을 실행하기 전에 이를 설계하세요. 사람 평가는 리드 타임이 더 긴 경우가 많습니다(IRB 승인, 평가자 모집).

**사람 평가가 필요한 경우:**
- 자동화된 지표로는 관심 대상을 포착할 수 없는 경우(유창성, 유용성, 안전성)
- 기여 내용이 사람을 대상으로 하는 품질에 관한 경우(가독성, 선호도, 신뢰)
- NLP 학회(ACL, EMNLP)의 리뷰어가 생성 과제에 사람 평가를 기대하는 경우

**핵심 설계 결정:**

| 결정 | 선택지 | 지침 |
|----------|---------|----------|
| **평가자 유형** | 전문가, 크라우드워커, 최종 사용자 | 주장을 입증하는 데 필요한 대상에 맞추세요 |
| **척도** | 리커트(1~5), 쌍대 비교, 순위 | LLM 출력에는 리커트보다 쌍대 비교가 더 신뢰할 만함 |
| **표본 크기** | 평가자별 및 전체 항목 수 | 검정력 분석 또는 최소 100개 항목, 3명 이상의 평가자 |
| **일치도 지표** | Cohen's kappa, Krippendorff's alpha, ICC | 2명 초과 평가자에는 Krippendorff's alpha를 사용하고 원시 일치도도 보고 |
| **플랫폼** | Prolific, MTurk, 내부 팀 | 품질은 Prolific, 규모는 MTurk, 도메인 전문성은 내부 팀 |

**주석 지침 체크리스트:**
```
- [ ] Clear task description with examples (good AND bad)
- [ ] Decision criteria for ambiguous cases
- [ ] At least 2 worked examples per category
- [ ] Attention checks / gold standard items (10-15% of total)
- [ ] Qualification task or screening round
- [ ] Estimated time per item and fair compensation (>= local minimum wage)
- [ ] IRB/ethics review if required by your institution
```

**보고 요구사항**(리뷰어는 다음을 모두 확인합니다):
- 평가자 수와 자격
- 구체적인 지표와 값으로 표시한 평가자 간 일치도
- 보상 세부 정보(금액, 예상 시급)
- 주석 인터페이스 설명 또는 스크린샷(부록)
- 총 주석 작업 시간

사람 평가 데이터의 통계 검정, 크라우드소싱 품질 관리 패턴, IRB 지침을 포함한 전체 가이드는 [references/human-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/human-evaluation.md)를 참조하세요.

---

## 3단계: 실험 실행 및 모니터링

**목표**: 실험을 안정적으로 실행하고, 진행 상황을 모니터링하며, 실패에서 복구합니다.

### 3.1단계: 실험 시작

장시간 실행되는 실험에는 `nohup`을 사용하세요:

```bash
nohup python run_experiment.py --config config.yaml > logs/experiment_01.log 2>&1 &
echo $!  # Record the PID
```

**병렬 실행**: 서로 독립적인 실험은 동시에 실행하되 API 속도 제한에 유의하세요. 동일한 API에서 4개 이상의 실험을 동시에 실행하면 각 실험이 느려집니다.

### 3.2단계: 모니터링 설정(Cron 패턴)

장시간 실행되는 실험에는 주기적인 상태 확인을 설정하세요. cron 프롬프트는 다음 템플릿을 따라야 합니다:

```
Monitor Prompt Template:
1. Check if process is still running: ps aux | grep <pattern>
2. Read last 30 lines of log: tail -30 <logfile>
3. Check for completed results: ls <result_dir>
4. If results exist, read and report: cat <result_file>
5. If all done, commit: git add -A && git commit -m "<descriptive message>" && git push
6. Report in structured format (tables with key metrics)
7. Answer the key analytical question for this experiment
```

**무음 모드**: 마지막 확인 이후 변경된 사항이 없으면 사용자 알림을 억제하기 위해 `[SILENT]`로 응답하세요. 새로운 내용이 있을 때만 보고하세요.

### 3.3단계: 실패 처리

일반적인 실패 유형과 복구 방법:

| 실패 | 감지 | 복구 |
|---------|----------|----------|
| API 속도 제한 / 크레딧 소진 | 로그의 402/429 오류 | 기다린 후 다시 실행(스크립트가 완료된 작업을 건너뜀) |
| 프로세스 충돌 | PID가 사라지고 결과가 불완전함 | 마지막 체크포인트부터 다시 실행 |
| 어려운 문제의 타임아웃 | 프로세스가 멈추고 로그 진행이 없음 | 종료하고 건너뛴 뒤 결과에 기록 |
| 잘못된 모델 ID | 모델 이름을 참조하는 오류 | ID를 수정하고 다시 실행 |

**핵심**: 스크립트는 항상 기존 결과를 확인하고 완료된 작업을 건너뛰어야 합니다. 이렇게 하면 재실행이 안전하고 효율적입니다.

### 3.4단계: 완료된 결과 커밋

각 실험 배치가 완료되면:

```bash
git add -A
git commit -m "Add <experiment name>: <key finding in 1 line>"
git push
```

### 3.5단계: 실험 저널 유지

Git 커밋은 발생한 일을 추적하지만 **탐색 트리** — 학습한 내용을 바탕으로 다음에 무엇을 시도할지에 대한 결정 — 는 추적하지 않습니다. 이 트리를 기록하는 구조화된 실험 저널을 유지하세요:

```json
// experiment_journal.jsonl — append one entry per experiment attempt
{
  "id": "exp_003",
  "parent": "exp_001",
  "timestamp": "2025-05-10T14:30:00Z",
  "hypothesis": "Adding scope constraints will fix convergence failure from exp_001",
  "plan": "Re-run autoreason with max_tokens=2000 and fixed structure template",
  "config": {"model": "haiku", "strategy": "autoreason", "max_tokens": 2000},
  "status": "completed",
  "result_path": "results/exp_003/",
  "key_metrics": {"win_rate": 0.85, "convergence_rounds": 3},
  "analysis": "Scope constraints fixed convergence. Win rate jumped from 0.42 to 0.85.",
  "next_steps": ["Try same constraints on Sonnet", "Test without structure template"],
  "figures": ["figures/exp003_convergence.pdf"]
}
```

**저널이 단순한 git보다 나은 이유**: Git은 파일 변경 사항을 추적합니다. 저널은 추론을 추적합니다. 즉, X를 시도한 이유, 무엇을 배웠는지, 그것이 다음 실험에 무엇을 의미하는지를 기록합니다. 논문을 작성할 때 이 트리는 방법론 섹션("X를 관찰했고, 이것이 Y에 대한 동기가 되었다")과 정직한 실패 보고에 매우 유용합니다.

**최선의 경로 선택**: 저널에 분기 트리(exp_001 → exp_002a, exp_002b, exp_003)가 나타나면 논문의 주장을 가장 잘 뒷받침하는 경로를 식별하세요. 막다른 분기는 절제 실험 또는 부정적 결과로 부록에 기록하세요.

**실험별 코드 스냅샷**: 각 실행 후 실험 스크립트를 복사하세요:
```bash
cp experiment.py results/exp_003/experiment_snapshot.py
```
이렇게 하면 이후 코드가 변경된 뒤에도 정확한 재현이 가능합니다.

---

## 4단계: 결과 분석

**목표**: 결과를 통해 발견 사항을 추출하고, 통계를 계산하며, 이야기를 파악합니다.
### 4.1단계: 결과 집계

분석 스크립트를 작성하여 다음을 수행합니다.
1. 배치의 모든 결과 파일을 로드합니다.
2. 작업별 및 전체 메트릭을 계산합니다.
3. 요약 표를 생성합니다.

```python
# Standard analysis pattern
import json, os
from pathlib import Path

results = {}
for result_file in Path("results/").rglob("result.json"):
    data = json.loads(result_file.read_text())
    strategy = result_file.parent.name
    task = result_file.parent.parent.name
    results.setdefault(strategy, {})[task] = data

# Compute aggregate metrics
for strategy, tasks in results.items():
    scores = [t["score"] for t in tasks.values()]
    print(f"{strategy}: mean={np.mean(scores):.1f}, std={np.std(scores):.1f}")
```

### 4.2단계: 통계적 유의성

항상 다음을 계산합니다.
- **오차 막대**: 표준편차 또는 표준오차를 사용하고, 어느 것인지 명시합니다.
- **신뢰 구간**: 핵심 결과에 대한 95% 신뢰 구간
- **쌍별 검정**: 두 방법을 비교하는 McNemar 검정
- **효과 크기**: Cohen의 d 또는 h를 사용한 실질적 유의성

McNemar 검정, 부트스트랩 신뢰 구간, Cohen의 h를 완전히 구현한 코드는 [references/experiment-patterns.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/experiment-patterns.md)를 참조하세요.

### 4.3단계: 스토리 파악

분석이 끝나면 다음 질문에 명시적으로 답합니다.
1. **가장 중요한 발견은 무엇인가?** 한 문장으로 기술합니다.
2. **무엇이 놀라웠는가?** 예상하지 못한 결과가 가장 좋은 논문을 만드는 경우가 많습니다.
3. **무엇이 실패했는가?** 실패한 실험은 가장 많은 정보를 줄 수 있습니다. 실패를 솔직하게 보고하면 논문이 더 강해집니다.
4. **어떤 후속 실험이 필요한가?** 결과는 새로운 질문을 제기하는 경우가 많습니다.

#### 부정적 결과 또는 영 결과 다루기

가설이 틀렸거나 결과가 불확실하다면 다음 세 가지 선택지가 있습니다.

| 상황 | 조치 | 적합한 학회/저널 |
|-----------|--------|-----------|
| 가설은 틀렸지만 **그 이유**가 유익함 | 왜 그런지 분석하는 방향으로 논문의 틀을 잡기 | NeurIPS, ICML (분석이 엄밀한 경우) |
| 방법이 기준선을 능가하지 못했지만 **새로운 사실을 드러냄** | 기여를 이해/분석으로 재구성하기 | ICLR (이해를 중시), 워크숍 논문 |
| 널리 알려진 주장에 대한 깔끔한 부정적 결과 | 논문화하기 — 이 분야에는 이런 결과가 알려져야 함 | NeurIPS Datasets & Benchmarks, TMLR, 워크숍 |
| 결과가 불확실하고 명확한 스토리가 없음 | 방향 전환 — 다른 실험을 하거나 틀을 다시 잡기 | 없는 스토리를 억지로 논문으로 만들지 않기 |

**부정적 결과 논문을 작성하는 방법:**
- 학계가 무엇을 믿고 있으며, 이를 검증하는 것이 왜 중요한지부터 제시합니다.
- 엄밀한 방법론을 설명합니다(빈틈이 없어야 하며, 심사자들은 더 엄격하게 검토합니다).
- 통계적 근거와 함께 영 결과를 명확히 제시합니다.
- 기대한 결과가 **왜** 나타나지 않았는지 분석합니다.
- 해당 분야에 미치는 영향을 논의합니다.

**부정적 결과를 명시적으로 환영하는 학회/저널**: NeurIPS (Datasets & Benchmarks 트랙), TMLR, ML Reproducibility Challenge, 주요 학회의 워크숍. 일부 워크숍은 특히 부정적 결과를 모집합니다.

### 4.4단계: 그림과 표 만들기

**그림**:
- 모든 플롯에 벡터 그래픽(PDF)을 사용합니다: `plt.savefig('fig.pdf')`
- 색각 이상을 고려한 팔레트(Okabe-Ito 또는 Paul Tol)를 사용합니다.
- 캡션만 읽어도 이해할 수 있도록 독립적인 캡션을 작성합니다.
- 그림 안에는 제목을 넣지 않습니다 — 캡션이 이 역할을 합니다.

**표**:
- `booktabs` LaTeX 패키지를 사용합니다.
- 메트릭별 최댓값을 굵게 표시합니다.
- 방향 기호(높을수록/낮을수록 좋음)를 포함합니다.
- 소수점 자릿수를 일관되게 유지합니다.

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

### 4.5단계: 실험을 더 할 것인가, 작성할 것인가?

| 상황 | 조치 |
|-----------|--------|
| 핵심 주장이 뒷받침되고 결과가 유의미함 | 5단계(작성)로 이동 |
| 결과가 불확실하고 데이터가 더 필요함 | 2단계(설계)로 돌아가기 |
| 예상하지 못한 발견이 새로운 방향을 시사함 | 2단계(설계)로 돌아가기 |
| 심사자가 요구할 만한 절제 실험 하나가 빠짐 | 해당 실험을 수행한 뒤 5단계로 이동 |
| 모든 실험을 마쳤지만 일부가 실패함 | 실패를 기록하고 5단계로 이동 |

### 4.6단계: 실험 로그 작성(논문 작성으로의 연결 고리)

논문 작성으로 넘어가기 전에 결과와 산문을 연결하는 구조화된 실험 로그를 만듭니다. 이는 실험과 논문 작성을 잇는 가장 중요한 연결 고리입니다. 이 로그가 없으면 작성 에이전트가 원시 결과 파일에서 스토리를 다시 도출해야 합니다.

**다음 구조로 `experiment_log.md`를 만듭니다.**

```markdown
# Experiment Log

## Contribution (one sentence)
[The paper's main claim]

## Experiments Run

### Experiment 1: [Name]
- **Claim tested**: [Which paper claim this supports]
- **Setup**: [Model, dataset, config, number of runs]
- **Key result**: [One sentence with the number]
- **Result files**: results/exp1/final_info.json
- **Figures generated**: figures/exp1_comparison.pdf
- **Surprising findings**: [Anything unexpected]

### Experiment 2: [Name]
...

## Figures
| Filename | Description | Which section it belongs in |
|----------|-------------|---------------------------|
| figures/main_comparison.pdf | Bar chart comparing all methods on benchmark X | Results, Figure 2 |
| figures/ablation.pdf | Ablation removing components A, B, C | Results, Figure 3 |
...

## Failed Experiments (document for honesty)
- [What was tried, why it failed, what it tells us]

## Open Questions
- [Anything the results raised that the paper should address]
```

**이것이 중요한 이유**: 논문을 초안 작성할 때 에이전트(또는 위임받은 하위 에이전트)는 LaTeX 템플릿과 함께 `experiment_log.md`를 로드하여 실제 결과에 근거한 초안을 만들 수 있습니다. 이 연결 고리가 없으면 작성 에이전트가 원시 JSON/CSV 파일을 파싱하고 스토리를 추론해야 하므로, 수치를 환각하거나 잘못 보고하는 일이 흔히 발생합니다.

**Git 규칙**: 이 로그를 해당 로그가 설명하는 결과와 함께 커밋합니다.

---

## 반복 개선: 전략 선택

이 파이프라인의 모든 출력(논문 초안, 실험 스크립트, 분석)은 반복적으로 개선할 수 있습니다. 오토리즌 연구는 각 개선 전략이 언제 효과적이고 언제 실패하는지에 대한 실증적 근거를 제공합니다. 다음 전략을 선택할 때 이 절을 활용하세요.

### 빠른 결정표

| 현재 상황 | 전략 | 이유 |
|---------------|----------|-----|
| 중간급 모델 + 제약된 작업 | **Autoreason** | 최적의 지점입니다. 생성-평가 격차가 가장 큽니다. 기준선이 약한 모델의 출력을 적극적으로 망칩니다. |
| 중간급 모델 + 개방형 작업 | 범위 제약을 추가한 **Autoreason** | 개선 공간을 제한할 수 있도록 고정된 사실, 구조 또는 결과물을 추가합니다. |
| 프런티어 모델 + 제약된 작업 | **Autoreason** | 프런티어에서도 제약된 작업의 3분의 2에서 승리합니다. |
| 프런티어 모델 + 비제약 작업 | **Critique-and-revise** 또는 **single pass** | Autoreason은 마지막 선택지입니다. 모델이 스스로 충분히 평가할 수 있습니다. |
| 구체적인 기술 작업(시스템 설계) | **Critique-and-revise** | 직접 찾고 수정하는 반복이 더 효율적입니다. |
| 템플릿 채우기 작업(정답 구조 하나) | **Single pass** 또는 **conservative** | 의사 결정 공간이 작습니다. 반복은 가치를 더하지 않습니다. |
| 테스트 사례가 있는 코드 | **Autoreason (code variant)** | 수정 전에 실패한 *이유*를 구조적으로 분석합니다. 복구율은 43% 대비 62%입니다. |
| 매우 약한 모델(Llama 8B급) | **Single pass** | 모델이 다양한 후보를 만들기에는 너무 약합니다. 생성 품질 향상에 투자합니다. |

### 생성-평가 격차

**핵심 통찰**: Autoreason의 가치는 모델의 생성 능력과 자체 평가 능력 사이의 격차에 달려 있습니다.

<!-- ascii-guard-ignore -->
```
Model Tier        │ Generation │ Self-Eval │ Gap    │ Autoreason Value
──────────────────┼────────────┼───────────┼────────┼─────────────────
Weak (Llama 8B)   │ Poor       │ Poor      │ Small  │ None — can't generate diverse candidates
Mid (Haiku 3.5)   │ Decent     │ Poor      │ LARGE  │ MAXIMUM — 42/42 perfect Borda
Mid (Gemini Flash)│ Decent     │ Moderate  │ Large  │ High — wins 2/3
Strong (Sonnet 4) │ Good       │ Decent    │ Medium │ Moderate — wins 3/5
Frontier (S4.6)   │ Excellent  │ Good      │ Small  │ Only with constraints
```
<!-- ascii-guard-ignore-end -->

이 격차는 일시적인 것이 아니라 구조적입니다. 비용이 내려가면 오늘의 프런티어가 내일의 중간급이 됩니다. 최적의 지점은 이동하지만 결코 사라지지 않습니다.

### Autoreason 반복(요약)

각 패스는 새로 생성되고 서로 격리된 에이전트로부터 세 개의 후보를 만듭니다.

1. **Critic** → 현재 A의 문제를 찾습니다(수정하지 않음).
2. **Author B** → 비평을 바탕으로 A를 수정합니다.
3. **Synthesizer** → A와 B를 병합합니다(라벨은 무작위화).
4. **Judge Panel** → 3명의 블라인드 CoT 심사자가 Borda count로 A, B, AB의 순위를 매깁니다.
5. **Convergence** → A가 연속 k=2회 승리하면 종료합니다.

**핵심 매개변수:**
- k=2 수렴(k=1은 너무 이르고, k=3은 너무 비싸며, 품질 향상이 없습니다).
- 항상 CoT 심사자를 사용합니다(수렴이 3배 빠름).
- 작성자는 temperature 0.8, 심사자는 0.3을 사용합니다.
- 보수적인 동률 처리: 동률이면 현재 항목이 승리합니다.
- 모든 역할은 공유된 맥락이 없는 새 에이전트입니다.

### 논문 초안에 적용하기

Autoreason으로 논문 자체를 개선할 때:
- **비평가에게 ground truth를 제공합니다**: 실제 실험 데이터, 결과 JSON, 통계 출력. 이것이 없으면 모델이 가짜 절제 연구와 가짜 신뢰 구간을 환각합니다.
- **작동하는 심사자를 최소 3명 사용합니다**: 심사자 파서가 고장 나면 잡음이 늘어나는 것이 아니라 평형 자체가 불가능해집니다.
- **수정 범위를 제한합니다**: "논문의 특정 약점을 해결하라"고 지시하고, "논문을 개선하라"고만 하지 않습니다.

### 실패 모드

| 실패 | 감지 | 수정 |
|---------|-----------|-----|
| 수렴하지 않음(A가 절대 승리하지 않음) | 20회 이상 패스에서 A의 승률이 &lt;15% | 작업에 범위 제약을 추가 |
| 합성 드리프트 | 단어 수가 제한 없이 증가 | 구조와 결과물을 제한 |
| 단일 패스보다 성능 저하 | 기준선의 점수가 반복된 출력보다 높음 | 단일 패스로 전환; 모델이 너무 약할 수 있음 |
| 과적합(코드) | 공개 테스트 통과율은 높고 비공개 테스트 통과율은 낮음 | 단순한 테스트 피드백이 아니라 구조화된 분석 사용 |
| 심사자 고장 | 파싱 실패로 패널이 3명 미만으로 감소 | 계속하기 전에 파서 수정 |

전체 프롬프트, Borda 점수 계산 세부 사항, 모델 선택 가이드, 범위 제약 설계 패턴, 연산 예산 참고 자료는 [references/autoreason-methodology.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/autoreason-methodology.md)를 참조하세요.

## 5단계: 논문 초안 작성

전체 초안 작성 절차(섹션별 순서, LaTeX 스캐폴딩, 그림/표 규칙, 초록 및 서론 공식, 관련 연구에서의 포지셔닝)는 `references/phase5-paper-drafting.md`에 있습니다. 이 단계에 도달하면 `read_file`로 해당 파일을 로드하세요.
이를 산문 수준의 스타일 규칙이 담긴 `references/writing-guide.md`와 함께 사용하세요.

## 6단계: 자체 검토 및 수정

**목표**: 제출 전에 심사 과정을 시뮬레이션합니다. 약점을 조기에 발견합니다.

### 6.1단계: 심사 시뮬레이션(앙상블 패턴)

여러 관점에서 심사를 생성합니다. 자동화된 연구 파이프라인(특히 SakanaAI의 AI-Scientist)에서 얻은 핵심 통찰은 메타 심사자와 함께하는 앙상블 심사가 단일 심사보다 훨씬 더 잘 보정된 피드백을 만든다는 것입니다.

**1단계: 독립적인 심사 N개 생성** (N=3-5)

서로 다른 모델이나 temperature 설정을 사용합니다. 각 심사자는 다른 심사 결과를 보지 않고 논문만 봅니다. **기본값은 부정적 편향**으로 설정합니다 — LLM은 평가에서 긍정 편향이 나타나는 것으로 잘 알려져 있습니다.

```
You are an expert reviewer for [VENUE]. You are critical and thorough.
If a paper has weaknesses or you are unsure about a claim, flag it clearly
and reflect that in your scores. Do not give the benefit of the doubt.

Review this paper according to the official reviewer guidelines. Evaluate:

1. Soundness (are claims well-supported? are baselines fair and strong?)
2. Clarity (is the paper well-written? could an expert reproduce it?)
3. Significance (does this matter to the community?)
4. Originality (new insights, not just incremental combination?)

Provide your review as structured JSON:
{
  "summary": "2-3 sentence summary",
  "strengths": ["strength 1", "strength 2", ...],
  "weaknesses": ["weakness 1 (most critical)", "weakness 2", ...],
  "questions": ["question for authors 1", ...],
  "missing_references": ["paper that should be cited", ...],
  "soundness": 1-4,
  "presentation": 1-4,
  "contribution": 1-4,
  "overall": 1-10,
  "confidence": 1-5
}
```

**2단계: 메타 심사(영역 의장 집계)**

모든 N개의 심사를 메타 심사자에게 전달합니다.

```
You are an Area Chair at [VENUE]. You have received [N] independent reviews
of a paper. Your job is to:

1. Identify consensus strengths and weaknesses across reviewers
2. Resolve disagreements by examining the paper directly
3. Produce a meta-review that represents the aggregate judgment
4. Use AVERAGED numerical scores across all reviews

Be conservative: if reviewers disagree on whether a weakness is serious,
treat it as serious until the authors address it.

Reviews:
[review_1]
[review_2]
...
```

**3단계: 성찰 반복** (선택 사항, 2-3회)

메타 심사를 본 후 각 심사자가 자신의 심사를 수정할 수 있습니다. 조기 종료 센티널을 사용합니다. 심사자가 "I am done"(변경 없음)이라고 응답하면 반복을 중지합니다.

**심사 모델 선택**: 논문을 더 저렴한 모델로 작성했더라도 심사는 사용 가능한 가장 강력한 모델로 수행하는 것이 좋습니다. 심사 모델은 작성 모델과 독립적으로 선택해야 합니다.

**퓨샷 보정**: 가능하다면 목표 학회/저널에 실제로 게재된 심사 결과 1~2개를 예시로 포함합니다. 이렇게 하면 점수 보정이 크게 향상됩니다. 예시 심사 결과는 [references/reviewer-guidelines.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/reviewer-guidelines.md)를 참조하세요.
### 6.1b단계: 시각적 검토 단계(VLM)

텍스트만 검토하면 그림의 품질, 레이아웃 문제, 시각적 일관성처럼 검토할 수 없는 문제가 많이 남습니다. 비전 기능을 지원하는 모델에 접근할 수 있다면, 컴파일된 연구 논문 PDF에 대해 별도의 **시각적 검토**를 실행하세요.

```
You are reviewing the visual presentation of this research paper PDF.
Check for:
1. Figure quality: Are plots readable? Labels legible? Colors distinguishable?
2. Figure-caption alignment: Does each caption accurately describe its figure?
3. Layout issues: Orphaned section headers, awkward page breaks, figures far from their references
4. Table formatting: Aligned columns, consistent decimal precision, bold for best results
5. Visual consistency: Same color scheme across all figures, consistent font sizes
6. Grayscale readability: Would the figures be understandable if printed in B&W?

For each issue, specify the page number and exact location.
```

이 검토는 텍스트 기반 검토로는 발견할 수 없는 문제를 포착합니다. 예를 들어 축 레이블을 읽기 어려운 그래프, 처음 언급된 위치에서 3페이지나 떨어져 배치된 그림, Figure 2와 Figure 5 사이의 일관되지 않은 색상 팔레트, 단 너비를 명백히 초과하는 표 등을 발견할 수 있습니다.

### 6.1c단계: 주장 검증 단계

모의 검토를 마친 뒤 별도의 검증 단계를 실행하세요. 이 단계에서는 검토자가 놓칠 수 있는 사실 오류를 포착합니다.

```
Claim Verification Protocol:
1. Extract every factual claim from the paper (numbers, comparisons, trends)
2. For each claim, trace it to the specific experiment/result that supports it
3. Verify the number in the paper matches the actual result file
4. Flag any claim without a traceable source as [VERIFY]
```

에이전트 기반 워크플로에서는 논문 텍스트와 원시 결과 파일만 전달받는 **새 하위 에이전트**에 검증을 위임하세요. 새로운 컨텍스트를 사용하면 확인 편향을 방지할 수 있습니다. 즉, 검증자는 결과가 어떠해야 하는지에 대한 기존 정보를 "기억하지" 않습니다.

### 6.2단계: 피드백 우선순위 지정

검토를 수집한 뒤 다음과 같이 분류하세요.

| 우선순위 | 조치 |
|----------|------|
| **치명적** (기술적 결함, 기준선 누락) | 반드시 수정합니다. 새로운 실험이 필요할 수 있으므로 → 2단계로 돌아갑니다. |
| **높음** (명확성 문제, 절제 실험 누락) | 이번 수정본에서 수정해야 합니다. |
| **중간** (사소한 문장 문제, 추가 실험) | 시간이 허락하면 수정합니다. |
| **낮음** (문체 선호, 주변적인 제안) | 향후 작업을 위해 기록합니다. |

### 6.3단계: 수정 주기

각 치명적/높은 우선순위 문제에 대해 다음을 수행합니다.

1. 영향을 받는 구체적인 섹션을 식별합니다.
2. 수정안을 작성합니다.
3. 수정 사항이 다른 주장을 훼손하지 않는지 확인합니다.
4. 논문을 업데이트합니다.
5. 검토자가 제기한 우려 사항을 기준으로 다시 확인합니다.

### 6.4단계: 반박문 작성

실제 검토에 응답할 때는 반박문이 수정 작업과는 별개의 기술이라는 점을 기억하세요.

**형식**: 항목별로 작성합니다. 각 검토자 우려 사항에 대해 다음과 같이 작성하세요.
```
> R1-W1: "The paper lacks comparison with Method X."

We thank the reviewer for this suggestion. We have added a comparison with 
Method X in Table 3 (revised). Our method outperforms X by 3.2pp on [metric] 
(p<0.05). We note that X requires 2x our compute budget.
```

**규칙**:
- 모든 우려 사항에 답변합니다. 검토자는 일부를 건너뛰었는지 알아차립니다.
- 가장 강력한 답변부터 제시합니다.
- 간결하고 직접적으로 작성합니다. 검토자는 수십 편의 반박문을 읽습니다.
- 반박 기간에 실험을 수행했다면 새로운 결과를 포함합니다.
- 약한 비판이라도 절대 방어적이거나 무시하는 태도를 보이지 않습니다.
- `latexdiff`를 사용해 변경 사항을 표시한 PDF를 생성합니다(전문 LaTeX 도구 섹션 참조).
- 일반적인 칭찬이 아니라 구체적이고 실행 가능한 피드백을 준 검토자에게 감사를 표합니다.

**하지 말아야 할 것**: 근거 없이 "정중히 동의하지 않습니다"라고만 쓰는 것. 설명 없이 "범위를 벗어납니다"라고 하는 것. 강점에만 답변하여 약점을 외면하는 것.

### 6.5단계: 논문 발전 과정 추적

주요 마일스톤마다 스냅샷을 저장하세요.
```
paper/
  paper.tex                    # Current working version
  paper_v1_first_draft.tex     # First complete draft
  paper_v2_post_review.tex     # After simulated review
  paper_v3_pre_submission.tex  # Final before submission
  paper_v4_camera_ready.tex    # Post-acceptance final
```

---

## 7단계: 제출 준비

**목표**: 최종 점검, 서식 지정 및 제출.

### 7.1단계: 학회 체크리스트

모든 학회에는 필수 체크리스트가 있습니다. 불완전한 체크리스트는 데스크 리젝으로 이어질 수 있으므로 주의 깊게 완료하세요.

[references/checklists.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/checklists.md)에서 다음을 확인하세요.
- NeurIPS 16개 항목 논문 체크리스트
- ICML 광범위한 영향 및 재현성
- ICLR LLM 공개 정책
- ACL 필수 한계점 섹션
- 제출 전 범용 체크리스트

### 7.2단계: 익명화 체크리스트

이중 블라인드 심사에서는 검토자가 논문 저자를 알 수 없어야 합니다. 다음 항목을 **모두** 확인하세요.

```
Anonymization Checklist:
- [ ] No author names or affiliations anywhere in the PDF
- [ ] No acknowledgments section (add after acceptance)
- [ ] Self-citations written in third person: "Smith et al. [1] showed..." not "We previously showed [1]..."
- [ ] No GitHub/GitLab URLs pointing to your personal repos
- [ ] Use Anonymous GitHub (https://anonymous.4open.science/) for code links
- [ ] No institutional logos or identifiers in figures
- [ ] No file metadata containing author names (check PDF properties)
- [ ] No "our previous work" or "in our earlier paper" phrasing
- [ ] Dataset names don't reveal institution (rename if needed)
- [ ] Supplementary materials don't contain identifying information
```

**흔한 실수**: 부록 코드에 보이는 Git 커밋 메시지, 기관 도구에서 생성된 워터마크 그림, 이전 초안에서 남겨 둔 감사의 글, 익명화 기간 전에 게시한 arXiv 프리프린트.

### 7.3단계: 서식 검증

```
Pre-Submission Format Check:
- [ ] Page limit respected (excluding references and appendix)
- [ ] All figures are vector (PDF) or high-res raster (600 DPI PNG)
- [ ] All figures readable in grayscale
- [ ] All tables use booktabs
- [ ] References compile correctly (no "?" in citations)
- [ ] No overfull hboxes in critical areas
- [ ] Appendix clearly labeled and separated
- [ ] Required sections present (limitations, broader impact, etc.)
```

### 7.4단계: 컴파일 전 검증

`pdflatex`를 실행하기 **전에** 다음 자동 검사를 실행하세요. 이 단계에서 오류를 발견하면 컴파일러 출력을 디버깅하는 것보다 빠르게 해결할 수 있습니다.

```bash
# 1. Lint with chktex (catches common LaTeX mistakes)
# Suppress noisy warnings: -n2 (sentence end), -n24 (parens), -n13 (intersentence), -n1 (command terminated)
chktex main.tex -q -n2 -n24 -n13 -n1

# 2. Verify all citations exist in .bib
# Extract \cite{...} from .tex, check each against .bib
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

# 3. Verify all referenced figures exist on disk
python3 -c "
import re, os
tex = open('main.tex').read()
figs = re.findall(r'\\\\includegraphics(?:\[.*?\])?{([^}]+)}', tex)
for fig in figs:
    if not os.path.exists(fig):
        print(f'WARNING: Figure file not found: {fig}')
"

# 4. Check for duplicate \label definitions
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

경고가 있으면 계속 진행하기 전에 모두 수정하세요. 에이전트 기반 워크플로에서는 chktex 출력을 에이전트에 전달하고 최소한의 수정만 하도록 지시하세요.

### 7.5단계: 최종 컴파일

```bash
# Clean build
rm -f *.aux *.bbl *.blg *.log *.out *.pdf
latexmk -pdf main.tex

# Or manual (triple pdflatex + bibtex for cross-references)
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Verify output exists and has content
ls -la main.pdf
```

**컴파일에 실패한 경우**: `.log` 파일에서 첫 번째 오류를 분석하세요. 일반적인 해결 방법은 다음과 같습니다.
- "Undefined control sequence" → 누락된 패키지 또는 명령 이름의 오타
- "Missing $ inserted" → 수학 기호를 수학 모드 밖에서 사용
- "File not found" → 잘못된 그림 경로 또는 누락된 `.sty` 파일
- "Citation undefined" → `.bib` 항목이 없거나 bibtex를 실행하지 않음

### 7.6단계: 학회별 요구 사항

| 학회 | 특별 요구 사항 |
|------|----------------|
| **NeurIPS** | 부록의 논문 체크리스트, 채택 시 일반 독자용 요약 |
| **ICML** | 광범위한 영향 진술(결론 뒤에 배치하며 분량 제한에 포함되지 않음) |
| **ICLR** | LLM 공개 필수, 상호 심사 동의 |
| **ACL** | 필수 한계점 섹션, 책임 있는 NLP 체크리스트 |
| **AAAI** | 엄격한 스타일 파일 — 어떠한 수정도 금지 |
| **COLM** | 언어 모델 커뮤니티를 대상으로 기여 내용을 구성 |

### 7.7단계: 학회 재제출 및 서식 변환

학회 템플릿 간에 변환할 때는 **LaTeX 프리앰블을 절대 복사하지 마세요**.

```bash
# 1. Start fresh with target template
cp -r templates/icml2026/ new_submission/

# 2. Copy ONLY content sections (not preamble)
#    - Abstract text, section content, figures, tables, bib entries

# 3. Adjust for page limits
# 4. Add venue-specific required sections
# 5. Update references
```

| 출발지 → 도착지 | 페이지 변경 | 주요 조정 사항 |
|----------------|------------|----------------|
| NeurIPS → ICML | 9 → 8 | 1페이지 축소, 광범위한 영향 추가 |
| ICML → ICLR | 8 → 9 | 실험 확장, LLM 공개 추가 |
| NeurIPS → ACL | 9 → 8 | NLP 관례에 맞게 재구성, 한계점 추가 |
| ICLR → AAAI | 9 → 7 | 상당한 축소, 엄격한 스타일 준수 |
| 모든 학회 → COLM | 다양 → 9 | 언어 모델 중심으로 재구성 |

페이지를 줄일 때는 증명을 부록으로 옮기고, 관련 연구를 간결하게 정리하며, 표를 결합하고, 서브피겨를 사용하세요. 페이지를 늘릴 때는 절제 실험을 추가하고, 한계점을 확장하며, 추가 기준선을 포함하고, 정성적 예시를 넣으세요.

**거절된 후**: 새 버전에서 검토자의 우려 사항을 해결하되, "변경 사항" 섹션을 포함하거나 이전 제출본을 언급하지 마세요(블라인드 심사).

### 7.8단계: 카메라 레디 준비(채택 후)

채택된 후에는 카메라 레디 버전을 준비하세요.

```
Camera-Ready Checklist:
- [ ] De-anonymize: add author names, affiliations, email addresses
- [ ] Add Acknowledgments section (funding, compute grants, helpful reviewers)
- [ ] Add public code/data URL (real GitHub, not anonymous)
- [ ] Address any mandatory revisions from meta-reviewer
- [ ] Switch template to camera-ready mode (if applicable — e.g., AAAI \anon → \camera)
- [ ] Add copyright notice if required by venue
- [ ] Update any "anonymous" placeholders in text
- [ ] Verify final PDF compiles cleanly
- [ ] Check page limit for camera-ready (sometimes differs from submission)
- [ ] Upload supplementary materials (code, data, appendix) to venue portal
```

### 7.9단계: arXiv 및 프리프린트 전략

arXiv에 게시하는 것은 ML 분야의 표준 관행이지만, 시기와 익명성에 관한 중요한 고려 사항이 있습니다.

**시기 결정 트리:**

| 상황 | 권장 사항 |
|------|----------|
| 이중 블라인드 학회(NeurIPS, ICML, ACL)에 제출 | arXiv에는 제출 마감일 **이후**에 게시하세요. 이전에 게시하면 기술적으로 익명성 정책을 위반할 수 있지만, 시행 방식은 학회마다 다릅니다. |
| ICLR에 제출 | ICLR은 제출 전에 arXiv에 게시하는 것을 명시적으로 허용합니다. 단, 제출본에는 저자 이름을 넣지 마세요. |
| 논문이 이미 arXiv에 있고 새 학회에 제출 | 대부분의 학회에서 허용됩니다. 심사 중에는 검토 내용을 반영한 변경 사항으로 arXiv 버전을 업데이트하지 **마세요**. |
| 우선권을 확립하려는 경우 | 다른 연구자에게 선점될 우려가 있다면 즉시 게시하세요. 단, 익명성을 포기해야 합니다. |

**arXiv 카테고리 선택** (ML/AI 논문):

| 카테고리 | 코드 | 적합한 분야 |
|----------|------|------------|
| Machine Learning | `cs.LG` | 일반적인 ML 방법 |
| Computation and Language | `cs.CL` | NLP, 언어 모델 |
| Artificial Intelligence | `cs.AI` | 추론, 계획, 에이전트 |
| Computer Vision | `cs.CV` | 비전 모델 |
| Information Retrieval | `cs.IR` | 검색, 추천 |

주 카테고리와 교차 등록 카테고리 1~2개를 선택하세요. 카테고리가 많을수록 노출은 늘어나지만, 실제로 관련성이 있는 경우에만 교차 등록하세요.

**버전 관리 전략:**
- **v1**: 최초 제출본(학회 제출본과 일치)
- **v2**: 채택 후 카메라 레디 수정본(초록에 "[학회]에 채택됨" 추가)
- 검토 기간 중에는 검토자 피드백에 대응한 변경 사항이 명확히 드러나는 v2를 게시하지 마세요.

```bash
# Check if your paper's title is already taken on arXiv
# (before choosing a title)
pip install arxiv
python -c "
import arxiv
results = list(arxiv.Search(query='ti:\"Your Exact Title\"', max_results=5).results())
print(f'Found {len(results)} matches')
for r in results: print(f'  {r.title} ({r.published.year})')
"
```
### 7.10단계: 연구 코드 패키징

깔끔하고 바로 실행할 수 있는 코드를 릴리스하면 인용과 심사자의 신뢰가 크게 높아집니다. 카메라 레디 제출물과 함께 코드를 패키징하세요.

**저장소 구조:**

```
your-method/
  README.md              # Setup, usage, reproduction instructions
  requirements.txt       # Or environment.yml for conda
  setup.py               # For pip-installable packages
  LICENSE                # MIT or Apache 2.0 recommended for research
  configs/               # Experiment configurations
  src/                   # Core method implementation
  scripts/               # Training, evaluation, analysis scripts
    train.py
    evaluate.py
    reproduce_table1.sh  # One script per main result
  data/                  # Small data or download scripts
    download_data.sh
  results/               # Expected outputs for verification
```

**연구 코드용 README 템플릿:**

```markdown
# [Paper Title]

Official implementation of "[Paper Title]" (Venue Year).

## Setup
[Exact commands to set up environment]

## Reproduction
To reproduce Table 1: `bash scripts/reproduce_table1.sh`
To reproduce Figure 2: `python scripts/make_figure2.py`

## Citation
[BibTeX entry]
```

**릴리스 전 체크리스트:**
```
- [ ] Code runs from a clean clone (test on fresh machine or Docker)
- [ ] All dependencies pinned to specific versions
- [ ] No hardcoded absolute paths
- [ ] No API keys, credentials, or personal data in repo
- [ ] README covers setup, reproduction, and citation
- [ ] LICENSE file present (MIT or Apache 2.0 for max reuse)
- [ ] Results are reproducible within expected variance
- [ ] .gitignore excludes data files, checkpoints, logs
```

**제출용 익명 코드** (채택 전):
```bash
# Use Anonymous GitHub for double-blind review
# https://anonymous.4open.science/
# Upload your repo → get an anonymous URL → put in paper
```

---

## 8단계: 채택 후 제출물

**목표**: 발표 자료와 커뮤니티 참여를 통해 채택된 논문의 영향력을 극대화합니다.

### 8.1단계: 학회 포스터

대부분의 학회는 포스터 세션을 운영합니다. 포스터 디자인 원칙은 다음과 같습니다.

| 요소 | 지침 |
|---------|-----------|
| **크기** | 학회 요구 사항을 확인하세요(일반적으로 24"x36" 또는 세로/가로형 A0) |
| **내용** | 제목, 저자, 한 문장으로 된 기여, 방법 그림, 핵심 결과 2~3개, 결론 |
| **흐름** | 왼쪽 위에서 오른쪽 아래로 이어지는 Z 패턴 또는 열 형식 |
| **텍스트** | 제목은 3m 거리에서, 본문은 1m 거리에서 읽을 수 있어야 합니다. 전체 문단은 쓰지 말고 글머리 기호만 사용하세요. |
| **그림** | 논문 그림을 더 높은 해상도로 재사용하세요. 핵심 결과를 확대하세요. |

**도구**: LaTeX (`beamerposter` package), PowerPoint/Keynote, Figma, Canva.

**제작**: 학회 2주 전 또는 그보다 일찍 주문하세요. 패브릭 포스터는 이동할 때 더 가볍습니다. 요즘은 가상/디지털 포스터를 지원하는 학회도 많습니다.

### 8.2단계: 학회 발표 / Spotlight

구두 발표나 spotlight 발표로 선정되었다면 다음과 같이 준비하세요.

| 발표 유형 | 시간 | 내용 |
|---------|----------|---------|
| **Spotlight** | 5분 | 문제, 접근법, 핵심 결과 하나. 정확히 5분에 맞도록 리허설하세요. |
| **구두 발표** | 15~20분 | 전체 이야기: 문제, 접근법, 핵심 결과, 절제 실험, 한계 |
| **워크숍 발표** | 10~15분 | 워크숍 청중에 맞게 조정하세요. 배경 설명이 더 필요할 수 있습니다. |

**슬라이드 디자인 규칙:**
- 슬라이드 하나에는 아이디어 하나만 담기
- 텍스트를 최소화하기 — 세부 사항은 말로 설명하고 화면에 모두 띄우지 않기
- 핵심 그림은 단계적으로 이해할 수 있도록 애니메이션 적용하기
- 마지막에 "takeaway" 슬라이드 포함하기(기여를 한 문장으로 정리)
- 예상 질문에 대비한 백업 슬라이드 준비하기

### 8.3단계: 블로그 게시물 / 소셜 미디어

이해하기 쉬운 요약은 영향력을 크게 높입니다.

- **Twitter/X 스레드**: 트윗 5~8개. 방법이 아니라 결과로 시작하세요. Figure 1과 핵심 결과 그림을 포함하세요.
- **블로그 게시물**: 800~1500단어. 심사자가 아니라 ML 실무자를 대상으로 작성하세요. 형식론은 생략하고 직관과 실용적 시사점을 강조하세요.
- **프로젝트 페이지**: 초록, 그림, 데모, 코드 링크, BibTeX가 포함된 HTML 페이지. GitHub Pages를 사용하세요.

**시점**: 논문이 proceedings 또는 arXiv 카메라 레디 버전으로 공개된 뒤 1~2일 이내에 게시하세요.

---

## 워크숍 논문 및 단편 논문

워크숍 논문과 단편 논문(예: ACL short papers, Findings papers)은 같은 파이프라인을 따르지만 제약과 기대 사항이 다릅니다.

### 워크숍 논문

| 속성 | 워크숍 | 주요 학회 |
|----------|----------|-----------------|
| **페이지 제한** | 일반적으로 4~6페이지 | 7~9페이지 |
| **심사 기준** | 완성도에 대한 기준이 더 낮음 | 완전하고 철저해야 함 |
| **심사 절차** | 대개 싱글 블라인드 또는 간소화된 심사 | 더블 블라인드, 엄격한 심사 |
| **가치 있게 평가되는 것** | 흥미로운 아이디어, 예비 결과, 관점 논문 | 강력한 베이스라인을 갖춘 완전한 실증적 이야기 |
| **arXiv** | 언제든 게시 가능 | 시점이 중요함(arXiv 전략 참고) |
| **기여 기준** | 새로운 방향, 흥미로운 부정적 결과, 진행 중인 작업 | 강력한 증거를 갖춘 유의미한 진전 |

**워크숍을 목표로 삼을 때:**
- 정식 논문을 쓰기 전에 피드백을 받고 싶은 초기 아이디어
- 8페이지 이상을 정당화하기 어려운 부정적 결과
- 시의성 있는 주제에 대한 관점 논문 또는 의견
- 재현 연구 또는 재현성 보고서

### ACL 단편 논문 및 Findings

ACL 학회에는 서로 다른 제출 유형이 있습니다.

| 유형 | 페이지 | 기대 사항 |
|------|-------|-----------------|
| **장편 논문** | 8 | 완전한 연구, 강력한 베이스라인, 절제 실험 |
| **단편 논문** | 4 | 증거로 뒷받침되는 집중된 기여: 명확한 한 가지 주장 |
| **Findings** | 8 | 주요 학회에 아깝게 선정되지 못한 탄탄한 연구 |

**단편 논문 전략**: 주장 하나를 고르고 철저하게 뒷받침하세요. 장편 논문을 4페이지로 억지로 압축하려 하지 말고, 더 집중적이고 다른 논문을 작성하세요.

---

## 실증적 ML을 넘어선 논문 유형

위의 주요 파이프라인은 실증적 ML 논문을 대상으로 합니다. 다른 논문 유형에는 서로 다른 구조와 증거 기준이 필요합니다. 각 유형에 대한 자세한 지침은 [references/paper-types.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/paper-types.md)를 참고하세요.

### 이론 논문

**구조**: 서론 → 예비 지식(정의, 표기법) → 주요 결과(정리) → 증명 개요 → 논의 → 전체 증명(부록)

**실증 논문과의 주요 차이:**
- 기여는 실험 수치가 아니라 정리, 바운드 또는 불가능성 결과입니다.
- 방법 섹션은 "예비 지식"과 "주요 결과"로 대체됩니다.
- 증명이 증거이며 실험이 아닙니다(이론의 예측을 실증적으로 검증하면 도움이 되지만 필수는 아닙니다).
- 본문에 증명 개요를 넣고 부록에 전체 증명을 넣는 것이 일반적입니다.
- 실험 섹션은 선택 사항이지만 이론적 예측을 검증한다면 논문을 강화할 수 있습니다.

**증명 작성 원칙:**
- 모든 가정을 명시적으로 포함해 정리를 형식적으로 서술하세요.
- 형식적 증명 전에 직관을 설명하세요("핵심 통찰은 ...입니다").
- 증명 개요는 0.5~1페이지 분량으로 핵심 아이디어를 전달해야 합니다.
- `\begin{proof}...\end{proof}` 환경을 사용하세요.
- 가정에 번호를 매기고 정리에서 참조하세요: "가정 1~3에 따르면 ..."

### 서베이 / 튜토리얼 논문

**구조**: 서론 → 분류 체계 / 구성 → 상세한 다룸 → 미해결 문제 → 결론

**주요 차이:**
- 기여는 새로운 방법이 아니라 구성, 종합, 미해결 문제의 식별입니다.
- 범위 내에서 포괄적이어야 합니다(심사자는 누락된 참고문헌을 확인합니다).
- 명확한 분류 체계 또는 구성 프레임워크가 필요합니다.
- 개별 논문이 제시하지 못하는 연구 간 연결에서 가치가 나옵니다.
- 적합한 학회/저널: TMLR(survey track), JMLR, Foundations and Trends in ML, ACM Computing Surveys

### 벤치마크 논문

**구조**: 서론 → 과제 정의 → 데이터셋 구축 → 베이스라인 평가 → 분석 → 의도된 사용 및 한계

**주요 차이:**
- 기여는 벤치마크 자체이며, 진정한 평가 공백을 메워야 합니다.
- 데이터셋 문서화는 선택 사항이 아니라 필수입니다(Datasheets, 5.11단계 참고).
- 벤치마크가 어려운 문제임을 입증해야 합니다(베이스라인이 포화 상태에 이르지 않아야 합니다).
- 벤치마크가 측정한다고 주장하는 것을 실제로 측정한다는 점을 입증해야 합니다(구성 타당도).
- 적합한 학회: NeurIPS Datasets & Benchmarks track, ACL(resource papers), LREC-COLING

### 관점 논문

**구조**: 서론 → 배경 → 논지 / 주장 → 뒷받침하는 증거 → 반론 → 시사점

**주요 차이:**
- 기여는 결과가 아니라 주장입니다.
- 반론을 진지하게 다뤄야 합니다.
- 증거는 실증적, 이론적 또는 논리적 분석일 수 있습니다.
- 적합한 학회/저널: ICML(position track), 워크숍, TMLR

---

## Hermes Agent 통합

이 스킬은 Hermes 에이전트를 위해 설계되었습니다. 전체 연구 생애주기에서 Hermes 도구, 위임, 일정 관리, 메모리를 사용합니다.

### 관련 스킬

특정 단계에서는 이 스킬을 다른 Hermes 스킬과 조합하세요.

| 스킬 | 사용 시점 | 로드 방법 |
|-------|-------------|-------------|
| **arxiv** | 1단계(문헌 검토): arXiv 검색, BibTeX 생성, Semantic Scholar를 통한 관련 논문 탐색 | `skill_view("arxiv")` |
| **subagent-driven-development** | 5단계(초안 작성): 2단계 검토(사양 준수 후 품질)를 통한 병렬 섹션 작성 | `skill_view("subagent-driven-development")` |
| **plan** | 0단계(설정): 실행 전 구조화된 계획 작성. `.hermes/plans/`에 기록 | `skill_view("plan")` |
| **qmd** | 1단계(문헌): 하이브리드 BM25+벡터 검색을 통한 로컬 지식 베이스(노트, 대화 기록, 문서) 검색 | 설치: `skill_manage("install", "qmd")` |
| **diagramming** | 4~5단계: Excalidraw 기반 그림 및 아키텍처 다이어그램 제작 | `skill_view("diagramming")` |
| **data-science** | 4단계(분석): 대화형 분석 및 시각화를 위한 Jupyter 라이브 커널 | `skill_view("data-science")` |

이 스킬은 `ml-paper-writing`을 대체합니다. `ml-paper-writing`의 모든 내용에 전체 실험/분석 파이프라인과 autoreason 방법론을 더한 것입니다.

### Hermes 도구 참고

| 도구 | 이 파이프라인에서의 사용 |
|------|----------------------|
| **`terminal`** | LaTeX 컴파일(`latexmk -pdf`), git 작업, 실험 실행(`nohup python run.py &`), 프로세스 확인 |
| **`process`** | 백그라운드 실험 관리: `process("start", ...)`, `process("poll", pid)`, `process("log", pid)`, `process("kill", pid)` |
| **`execute_code`** | 인용 검증, 통계 분석, 결과 집계에 Python 실행. RPC를 통해 도구에 접근할 수 있습니다. |
| **`read_file`** / **`write_file`** / **`patch`** | 논문 편집, 실험 스크립트, 결과 파일. 큰 `.tex` 파일의 특정 부분을 편집할 때는 `patch`를 사용하세요. |
| **`web_search`** | 문헌 탐색: `web_search("transformer attention mechanism 2024")` |
| **`web_extract`** | 논문 내용 가져오기, 인용 검증: `web_extract("https://arxiv.org/abs/2303.17651")` |
| **`delegate_task`** | **병렬 섹션 초안 작성** — 각 섹션에 대해 격리된 하위 에이전트를 생성합니다. 동시 인용 검증에도 사용합니다. |
| **`todo`** | 세션 간 주요 상태 추적. 단계가 전환될 때마다 업데이트합니다. |
| **`memory`** | 주요 결정 사항(기여 구성, 학회 선택, 심사자 피드백)을 저장합니다. |
| **`cronjob`** | 실험 모니터링, 마감일 카운트다운, 자동 arXiv 확인을 예약합니다. |
| **`clarify`** | 막혔을 때 사용자에게 구체적인 질문(학회 선택, 기여 구성)을 합니다. |
| **cron `deliver:`** | 실험이 완료되거나 초안이 준비되면 사용자에게 알립니다. 대화에 사용자가 없어도 알림을 보내려면 `deliver:` 대상이 있는 cron 작업으로 확인을 예약하세요(이제 에이전트에는 `send_message` 도구가 없으며, 외부 전달은 cron/`hermes send`가 처리합니다). |

### 도구 사용 패턴

**실험 모니터링** (가장 일반적인 경우):
```
terminal("ps aux | grep <pattern>")
→ terminal("tail -30 <logfile>")
→ terminal("ls results/")
→ execute_code("analyze results JSON, compute metrics")
→ terminal("git add -A && git commit -m '<descriptive message>' && git push")
→ (final response auto-delivers "Experiment complete: <summary>"; for unattended runs, schedule via cron with a deliver: target)
```

**병렬 섹션 초안 작성** (위임 사용):
```
delegate_task("Draft the Methods section based on these experiment scripts and configs. 
  Include: pseudocode, all hyperparameters, architectural details sufficient for 
  reproduction. Write in LaTeX using the neurips2025 template conventions.")

delegate_task("Draft the Related Work section. Use web_search and web_extract to 
  find papers. Verify every citation via Semantic Scholar. Group by methodology.")

delegate_task("Draft the Experiments section. Read all result files in results/. 
  State which claim each experiment supports. Include error bars and significance.")
```

각 delegate는 공유된 컨텍스트가 없는 **새 하위 에이전트**로 실행됩니다. 프롬프트에 필요한 모든 정보를 제공하세요. 결과를 수집하고 통합합니다.

**인용 검증** (`execute_code` 사용):
```python
# In execute_code:
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
### `memory` 및 `todo`를 사용한 상태 관리

**`memory` 도구** — 주요 결정 사항을 저장합니다(MEMORY.md에 약 2200자로 제한):

```
memory("add", "Paper: autoreason. Venue: NeurIPS 2025 (9 pages). 
  Contribution: structured refinement works when generation-evaluation gap is wide.
  Key results: Haiku 42/42, Sonnet 3/5, S4.6 constrained 2/3.
  Status: Phase 5 — drafting Methods section.")
```

주요 결정이나 단계 전환이 있을 때 memory를 업데이트합니다. 이 내용은 세션 간에 유지됩니다.

**`todo` 도구** — 세부 진행 상황을 추적합니다:

```
todo("add", "Design constrained task experiments for Sonnet 4.6")
todo("add", "Run Haiku baseline comparison")
todo("add", "Draft Methods section")
todo("update", id=3, status="in_progress")
todo("update", id=1, status="completed")
```

**세션 시작 프로토콜:**
```
1. todo("list")                           # Check current task list
2. memory("read")                         # Recall key decisions
3. terminal("git log --oneline -10")      # Check recent commits
4. terminal("ps aux | grep python")       # Check running experiments
5. terminal("ls results/ | tail -20")     # Check for new results
6. Report status to user, ask for direction
```

### cronjob을 사용한 모니터링

`cronjob`을 사용해 주기적으로 실험 상태를 확인합니다:

```
cronjob("create", {
  "schedule": "*/30 * * * *",  # Every 30 minutes
  "prompt": "Check experiment status:
    1. ps aux | grep run_experiment
    2. tail -30 logs/experiment_haiku.log
    3. ls results/haiku_baselines/
    4. If complete: read results, compute Borda scores, 
       git add -A && git commit -m 'Add Haiku results' && git push
    5. Report: table of results, key finding, next step
    6. If nothing changed: respond with [SILENT]"
})
```

**[SILENT] 프로토콜**: 마지막 확인 이후 변경 사항이 없으면 정확히 `[SILENT]`라고 응답합니다. 이렇게 하면 사용자에게 알림이 전송되지 않습니다. 사용자에게 알릴 만한 실제 변경 사항이 있을 때만 보고합니다.

**마감일 추적**:
```
cronjob("create", {
  "schedule": "0 9 * * *",  # Daily at 9am
  "prompt": "NeurIPS 2025 deadline: May 22. Today is {date}. 
    Days remaining: {compute}. 
    Check todo list — are we on track? 
    If <7 days: warn user about remaining tasks."
})
```

### 커뮤니케이션 패턴

**사용자에게 알릴 때** (직접/최종 응답 또는 무인 실행을 위한 cron `deliver:` 대상):
- 실험 배치가 완료되었을 때(결과 표 포함)
- 결정이 필요한 예기치 않은 결과나 실패가 발생했을 때
- 초안 섹션을 검토할 준비가 되었을 때
- 미완료 작업이 있는 상태에서 마감일이 다가올 때

**알리지 않을 때:**
- 실험이 아직 실행 중이고 새로운 결과가 없을 때 → `[SILENT]`
- 변경 사항이 없는 정기 모니터링일 때 → `[SILENT]`
- 사용자의 주의가 필요하지 않은 중간 단계일 때

**보고 형식** — 항상 구조화된 데이터를 포함합니다:
```
## Experiment: <name>
Status: Complete / Running / Failed

| Task | Method A | Method B | Method C |
|------|---------|---------|---------|
| Task 1 | 85.2 | 82.1 | **89.4** |

Key finding: <one sentence>
Next step: <what happens next>
```

### 사람의 입력이 필요한 결정 사항

정말로 작업이 막힌 경우에는 `clarify`를 사용해 구체적인 질문을 합니다:

| 결정 사항 | 질문할 시점 |
|-----------|------------|
| 목표 학회 | 논문을 시작하기 전(페이지 제한과 논지 구성에 영향을 줌) |
| 기여의 프레이밍 | 유효한 프레이밍이 여러 가지일 때 |
| 실험 우선순위 | 시간보다 할 일이 많을 때 |
| 제출 준비 상태 | 최종 제출 전에 |

**다음에 대해서는 묻지 마세요** (주도적으로 선택하고, 그 선택을 표시하세요):
- 단어 선택 및 섹션 순서
- 강조할 구체적인 결과
- 인용 완전성(찾은 자료로 초안을 작성하고 누락된 부분을 표시)

---

## 리뷰어 평가 기준

리뷰어가 무엇을 중점적으로 보는지 이해하면 초점을 맞추는 데 도움이 됩니다.

| 기준 | 확인하는 내용 |
|------|---------------|
| **품질** | 기술적 타당성, 충분히 뒷받침된 주장, 공정한 베이스라인 |
| **명확성** | 명확한 서술, 전문가가 재현할 수 있는지, 일관된 표기법 |
| **중요성** | 커뮤니티에 미치는 영향, 이해의 진전 |
| **독창성** | 새로운 통찰(새로운 방법이 반드시 필요한 것은 아님) |

**점수(NeurIPS 6점 척도):**
- 6: 강력한 채택 — 획기적이며 흠이 없음
- 5: 채택 — 기술적으로 탄탄하고 영향력이 큼
- 4: 경계선 채택 — 탄탄하지만 평가가 제한적임
- 3: 경계선 거절 — 약점이 강점보다 큼
- 2: 거절 — 기술적 결함이 있음
- 1: 강력한 거절 — 이미 알려진 결과이거나 윤리 문제가 있음

자세한 지침, 일반적인 우려 사항 및 반박 전략은 [references/reviewer-guidelines.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/reviewer-guidelines.md)를 참조하세요.

---

## 일반적인 문제와 해결책

| 문제 | 해결책 |
|------|--------|
| 초록이 너무 일반적임 | 어떤 ML 논문 앞에도 붙일 수 있는 첫 문장이라면 삭제합니다. 구체적인 기여로 시작하세요. |
| 서론이 1.5페이지를 초과함 | 배경을 Related Work로 나눕니다. 기여 항목을 앞부분에 배치하세요. |
| 실험에 명시적인 주장이 없음 | 각 실험 앞에 “이 실험은 [구체적인 주장]이 성립하는지를 검증한다...”를 추가합니다. |
| 리뷰어가 논문을 따라가기 어렵다고 느낌 | 안내 문구를 추가하고, 일관된 용어를 사용하며, 그림 캡션만으로도 이해할 수 있게 작성합니다. |
| 통계적 유의성이 누락됨 | 오차 막대, 실행 횟수, 통계 검정, 신뢰 구간을 추가합니다. |
| 실험 범위가 불필요하게 확대됨 | 모든 실험은 구체적인 주장과 연결되어야 합니다. 그렇지 않은 실험은 삭제하세요. |
| 논문이 거절되어 재제출해야 함 | Phase 7의 Conference Resubmission을 참조합니다. 리뷰를 언급하지 않고 리뷰어의 우려를 해결하세요. |
| broader impact statement가 누락됨 | Step 5.10을 참조합니다. 대부분의 학회에서 요구합니다. “부정적인 영향이 없음”은 거의 신뢰받지 못합니다. |
| 인간 평가가 약하다는 비판을 받음 | Step 2.5 및 [references/human-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/human-evaluation.md)를 참조합니다. 일치도 지표, 평가자 세부 정보, 보상 내용을 보고하세요. |
| 리뷰어가 재현성에 의문을 제기함 | 코드를 공개하고(Step 7.9), 모든 하이퍼파라미터를 문서화하며, 시드와 연산 자원 세부 정보를 포함합니다. |
| 이론 논문에 직관적 설명이 부족함 | 형식적 증명에 앞서 쉬운 언어로 설명하는 증명 개요를 추가합니다. [references/paper-types.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/paper-types.md)를 참조하세요. |
| 결과가 음성/귀무 결과임 | 음성 결과를 다루는 방법은 Phase 4.3을 참조합니다. 워크숍이나 TMLR을 고려하거나 분석 결과로 재구성하세요. |

---

## 참고 문서

| 문서 | 내용 |
|------|------|
| [references/writing-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/writing-guide.md) | Gopen & Swan의 7가지 원칙, Perez의 실전 팁, Lipton의 단어 선택, Steinhardt의 정확성, 그림 설계 |
| [references/citation-workflow.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/citation-workflow.md) | 인용 API, Python 코드, CitationManager 클래스, BibTeX 관리 |
| [references/checklists.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/checklists.md) | NeurIPS 16개 항목, ICML·ICLR·ACL 요구 사항, 범용 제출 전 체크리스트 |
| [references/reviewer-guidelines.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/reviewer-guidelines.md) | 평가 기준, 점수, 일반적인 우려 사항, 반박 템플릿 |
| [references/sources.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/sources.md) | 모든 작성 가이드, 학회 지침, API의 전체 참고문헌 |
| [references/experiment-patterns.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/experiment-patterns.md) | 실험 설계 패턴, 평가 프로토콜, 모니터링, 오류 복구 |
| [references/autoreason-methodology.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/autoreason-methodology.md) | Autoreason 루프, 전략 선택, 모델 가이드, 프롬프트, 범위 제약, Borda 점수 |
| [references/human-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/human-evaluation.md) | 인간 평가 설계, 주석 지침, 일치도 지표, 크라우드소싱 품질 관리, IRB 지침 |
| [references/paper-types.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/references/paper-types.md) | 이론 논문(증명 작성, 정리 구조), 서베이 논문, 벤치마크 논문, 포지션 페이퍼 |

### LaTeX 템플릿

`templates/`에는 **NeurIPS 2025**, **ICML 2026**, **ICLR 2026**, **ACL**, **AAAI 2026**, **COLM 2025**용 템플릿이 있습니다.

컴파일 방법은 [templates/README.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/research/research-paper-writing/templates/README.md)를 참조하세요.

### 주요 외부 자료

**작성 철학:**
- [Neel Nanda: ML 논문 작성법](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)
- [Sebastian Farquhar: ML 논문 작성법](https://sebastianfarquhar.com/on-research/2024/11/04/how_to_write_ml_papers/)
- [Gopen & Swan: 과학적 글쓰기의 과학](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf)
- [Lipton: 과학적 글쓰기를 위한 휴리스틱](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/)
- [Perez: 쉬운 논문 작성 팁](https://ethanperez.net/easy-paper-writing-tips/)

**API:** [Semantic Scholar](https://api.semanticscholar.org/api-docs/) | [CrossRef](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | [arXiv](https://info.arxiv.org/help/api/basics.html)

**학회:** [NeurIPS](https://neurips.cc/Conferences/2025/PaperInformation/StyleFiles) | [ICML](https://icml.cc/Conferences/2025/AuthorInstructions) | [ICLR](https://iclr.cc/Conferences/2026/AuthorGuide) | [ACL](https://github.com/acl-org/acl-style-files)
