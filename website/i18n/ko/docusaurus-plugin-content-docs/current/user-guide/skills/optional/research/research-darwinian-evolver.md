---
title: "Darwinian Evolver — Imbue의 진화 루프를 사용한 프롬프트/정규식/SQL/코드 진화"
sidebar_label: "Darwinian Evolver"
description: "Imbue의 진화 루프를 사용한 프롬프트/정규식/SQL/코드 진화"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Darwinian Evolver

Imbue의 진화 루프를 사용하여 프롬프트/정규식/SQL/코드를 진화시킵니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Optional — `hermes skills install official/research/darwinian-evolver` 명령으로 설치 |
| Path | `optional-skills/research/darwinian-evolver` |
| Version | `0.1.0` |
| Author | Bihruze (Asahi0x), Hermes Agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `evolution`, `optimization`, `prompt-engineering`, `research` |
| Related skills | [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv), [`jupyter-live-kernel`](/docs/user-guide/skills/bundled/data-science/data-science-jupyter-live-kernel) |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Darwinian Evolver

LLM 기반 진화적 탐색 루프인 Imbue의 [darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver)를 실행하여 적합도 함수(fitness function)에 맞게 **프롬프트, 정규식, SQL 쿼리 또는 작은 코드 조각**을 최적화합니다.

상태: 업스트림 도구 주변의 얇은 래퍼(wrapper)입니다. 이 스킬은 도구를 설치하고, 에이전트가 `Problem` 정의(유기체(organism) + 평가자(evaluator) + 변이자(mutator))를 작성하도록 안내하며, 업스트림 CLI나 작은 커스텀 파이썬 드라이버를 통해 루프를 구동합니다.

**라이선스:** 업스트림 도구는 **AGPL-3.0**입니다. 이 스킬은 오직 업스트림 CLI나 `subprocess`/`uv run` 호출을 통해서만 도구를 실행합니다(단순 집합). Hermes 내부에 업스트림 클래스를 임포트하지 **마십시오**.

## 사용 시기 (When to Use)

- 사용자가 "이 프롬프트를 최적화해 줘", "X를 위한 정규식을 진화시켜 줘", "이 코드/SQL을 자동 개선해 줘", "더 나은 지시문을 찾아 줘"라고 말할 때.
- 점수 평가기(정확한 일치, 정규식 통과율, 단위 테스트, LLM 심사자(judge), 런타임 지표)와 시작 후보(유기체)가 **있을 때**. 평가기가 없다면 멈추고 먼저 정의하세요 — 이것이 가장 어려운 부분입니다.
- 비용이 괜찮을 때: 일반적인 실행은 50~500번의 LLM 호출입니다. gpt-4o-mini에서는 몇 푼 수준이지만, Claude Sonnet에서는 몇 달러가 들 수 있습니다.

다음에 해당할 때는 사용하지 **마십시오**:
- 최적화 대상이 미분 가능할 때(경사 하강법 / DSPy 사용).
- 변형을 2~3가지만 시도해보면 될 때 — 그냥 손으로 작성하세요.
- 적합도 신호가 측정 가능한 기준 없이 전적으로 주관적일 때.

## 전제 조건 (Prerequisites)

- Python ≥3.11
- `git`, `uv` (또는 `pip`)
- 다음 중 하나: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, 또는 `OPENAI_API_KEY`

이 스킬은 OpenAI SDK를 통해 `OPENROUTER_API_KEY`를 사용하는 작은 `parrot_openrouter.py` 드라이버를 포함하고 있어, OpenRouter의 어떤 모델이든 작동합니다. 업스트림 CLI 자체는 Anthropic을 하드코딩하고 있으며 `ANTHROPIC_API_KEY`가 필요합니다.

## 설치 (Install - 1회성)

`terminal` 도구를 통해 실행:

```bash
mkdir -p ~/.hermes/cache/darwinian-evolver && cd ~/.hermes/cache/darwinian-evolver
[ -d darwinian_evolver ] || git clone --depth 1 https://github.com/imbue-ai/darwinian_evolver.git
cd darwinian_evolver && uv sync
```

확인:

```bash
cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver \
  && uv run darwinian_evolver --help | head -5
```

## 빠른 시작 — 내장된 Parrot 예제 (Quick Start — The Built-In Parrot Example)

간단한 스모크 테스트 (`ANTHROPIC_API_KEY` 필요):

```bash
cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver
uv run darwinian_evolver parrot \
  --num_iterations 2 \
  --num_parents_per_iteration 2 \
  --mutator_concurrency 2 --evaluator_concurrency 2 \
  --output_dir /tmp/parrot_demo
```

출력 결과:
- `/tmp/parrot_demo/snapshots/iteration_N.pkl` — 반복별 피클링(pickled)된 모집단
- `/tmp/parrot_demo/<jsonl>` — 반복별 JSON 로그 (종료 시 경로 출력됨)

브라우저에서 `~/.hermes/cache/darwinian-evolver/darwinian_evolver/darwinian_evolver/lineage_visualizer.html`을 열고 JSON 로그를 로드하여 진화 트리를 확인하세요.

## 빠른 시작 — OpenRouter 드라이버 (Anthropic 키 불필요)

스킬에는 `scripts/parrot_openrouter.py`가 포함되어 있습니다 — 같은 parrot 문제이지만, LLM 호출이 OpenRouter를 통과하므로 모든 제공자(provider)에서 작동합니다.

```bash
# 스킬이 설치된 위치에서:
SKILL_DIR=~/.hermes/skills/research/darwinian-evolver
DE_DIR=~/.hermes/cache/darwinian-evolver/darwinian_evolver

cd "$DE_DIR" && \
  EVOLVER_MODEL='openai/gpt-4o-mini' \
  uv run --with openai python "$SKILL_DIR/scripts/parrot_openrouter.py" \
    --num_iterations 3 --num_parents_per_iteration 2 \
    --output_dir /tmp/parrot_or
```

`scripts/show_snapshot.py`로 결과를 검사하세요:

```bash
uv run --with openai python "$SKILL_DIR/scripts/show_snapshot.py" \
  /tmp/parrot_or/snapshots/iteration_3.pkl
```

예상 결과: 점수별로 랭크된 7개의 진화된 프롬프트 템플릿이 출력되며, 최고 점수는 0.6–0.8 부근에 도달합니다 (초기 시드 `Say {{ phrase }}`는 0.000 점이었습니다).

## 사용자 정의 문제 정의하기 (Defining a Custom Problem)

스킬에는 `templates/custom_problem_template.py`가 포함되어 있습니다 — 복사, 수정, 실행하세요.
정의해야 할 세 가지 핵심 요소:

1. **`Organism`** — 진화 중인 아티팩트를 담는 Pydantic `BaseModel` 서브클래스 (`prompt_template: str`, `regex_pattern: str`, `sql_query: str`, `code_block: str` 등). 이를 실행하는 `run(*args)` 메서드를 추가하세요.

2. **`Evaluator`** — `.evaluate(organism) -> EvaluationResult(score=..., trainable_failure_cases=[...], holdout_failure_cases=[...], is_viable=True)`.
   - **`score`**는 `[0, 1]` 범위 내에 있습니다. 높을수록 좋습니다.
   - **`trainable_failure_cases`** — 변이자가 보는 실패 사례들입니다. LLM이 문제를 진단할 수 있도록 충분한 맥락(입력, 예상값, 실제값)을 포함하세요.
   - **`holdout_failure_cases`** — 변이자의 시야에서 제외된 실패 사례들입니다. 과적합(overfitting)을 감지하는 데 사용하세요.
   - **`is_viable=True`**로 설정하세요. 단, 유기체가 완전히 망가진 경우(오류 발생, None 반환 등)에는 예외입니다. 점수가 0이지만 실행 가능한 유기체는 괜찮습니다 — 부모 선택에서 가중치가 낮아질 뿐입니다.

3. **`Mutator`** — `.mutate(organism, failure_cases, learning_log_entries) -> list[Organism]`.
   일반적으로: 현재 유기체 + 실패 사례 + 수정 제안 요청을 포함하는 LLM 프롬프트를 만들고; LLM의 응답을 파싱하며; 새로운 `Organism`을 반환합니다. 파싱 실패 시 `[]`를 반환하세요 — 루프가 처리합니다.

그런 다음 `Problem(initial_organism, evaluator, [mutators])`을 `EvolveProblemLoop`에 연결하고 `loop.run(num_iterations=N)`을 반복(iterate)하는 드라이버 스크립트를 작성합니다 — 제공된 `scripts/parrot_openrouter.py`가 참고 모델입니다.

## 실제로 중요한 하이퍼파라미터 (Hyperparameters That Actually Matter)

| flag | default | when to change |
|---|---|---|
| `--num_iterations` | 5 | 평가자를 신뢰할 수 있게 되면 10–20으로 올림 |
| `--num_parents_per_iteration` | 4 | 저렴한 탐색을 위해 2로 낮춤 |
| `--mutator_concurrency` | 10 | 속도 제한(rate limits)을 피하기 위해 2–4로 낮춤 |
| `--evaluator_concurrency` | 10 | 동일함; 평가자도 LLM을 호출함 |
| `--batch_size` | 1 | 변이자가 여러 실패를 처리할 수 있게 되면 3–5로 올림 |
| `--verify_mutations` | off | 변이자의 낭비가 심할 때 켬 (Imbue에 따르면 후반부 실행에서 비용을 10배 이상 절약함) |
| `--midpoint_score` | `p75` | 점수들이 몰려 있지 않은 한 건드리지 않음 |
| `--sharpness` | 10 | 건드리지 않음 |

## 주의사항 (Pitfalls)

1. **`초기 유기체는 실행 가능해야 함(Initial organism must be viable)`** — 0점짜리 시드라 하더라도 `EvaluationResult`에서 `is_viable=True`로 설정하세요. 루프는 실행 불가능한 유기체를 거부합니다. 이는 루프가 진화의 시작점으로 삼을 것이 없다는 뜻이기 때문입니다.
2. **제공자의 콘텐츠 필터가 실행을 중단시킴.** Azure 기반 OpenRouter 모델들은 "이전 지시사항 무시"와 같은 문구를 HTTP 400 에러와 함께 거부합니다. LLM 호출을 `try/except`로 감싸고 `f"<LLM_ERROR: {e}>"`를 반환하세요 — 진화기(evolver)는 해당 유기체에 0점을 주고 다음으로 넘어갑니다.
3. **`loop.run()`은 제너레이터(generator)임** — 이를 호출하는 것만으로는 반복(iterate)하기 전까지 아무것도 실행되지 않습니다. `for snap in loop.run(num_iterations=N):` 형태로 사용하세요.
4. **스냅샷은 중첩된 피클(pickles)임.** `iteration_N.pkl`은 `population_snapshot`(이 또한 피클링된 바이트)을 가진 딕셔너리를 포함합니다. 역피클링(unpickle)하려면 피클링될 때와 동일한 점표기법 경로(dotted path)로 `Organism` 클래스를 임포트할 수 있어야 합니다.
5. **동시성 기본값이 매우 공격적임.** 10/10 설정은 대부분의 제공자에서 속도 제한(rate limits)에 걸립니다. 2/2로 시작하세요.
6. **CLI가 Anthropic으로 하드코딩됨.** `uv run darwinian_evolver <problem>`은 `ANTHROPIC_API_KEY`를 찾고 Claude Sonnet을 사용합니다. 다른 제공자를 사용하려면 `parrot_openrouter.py`와 같은 드라이버를 작성하세요.
7. **AGPL.** Hermes 코어 내부에서 절대 `from darwinian_evolver import ...`를 사용하지 마세요. `~/.hermes/skills/...` 아래의 사용자 정의 드라이버 스크립트는 사용자 측 코드이므로 괜찮습니다.
8. **PyPI 패키지 없음.** `pip install darwinian-evolver`는 엉뚱한 것을 다운로드합니다. 항상 GitHub 저장소에서 설치하세요.

## 검증 (Verification)

설치 및 parrot 실행 후, 아래 명령이 종료 코드(exit code) 0을 반환하면 충분합니다:

```bash
DE_DIR=~/.hermes/cache/darwinian-evolver/darwinian_evolver
ls "$DE_DIR/darwinian_evolver/lineage_visualizer.html" >/dev/null && \
cd "$DE_DIR" && uv run darwinian_evolver --help >/dev/null && \
echo "darwinian-evolver: OK"
```

## 참고 자료 (References)

- [Imbue research post](https://imbue.com/research/2026-02-27-darwinian-evolver/)
- [ARC-AGI-2 results](https://imbue.com/research/2026-02-27-arc-agi-2-evolution/)
- [imbue-ai/darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver) (AGPL-3.0)
- [Darwin Gödel Machines](https://arxiv.org/abs/2505.22954)
- [PromptBreeder](https://arxiv.org/abs/2309.16797)
