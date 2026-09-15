---
title: "Darwinian Evolver — Imbue의 진화 루프로 프롬프트/정규식/SQL/코드 진화시키기"
sidebar_label: "Darwinian Evolver"
description: "Imbue의 진화 루프로 프롬프트/정규식/SQL/코드 진화시키기"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Darwinian Evolver

Imbue의 진화 루프로 프롬프트/정규식/SQL/코드를 진화시킵니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/research/darwinian-evolver`로 설치 |
| 경로 | `optional-skills/research/darwinian-evolver` |
| 버전 | `0.1.0` |
| 작성자 | Bihruze (Asahi0x), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `evolution`, `optimization`, `prompt-engineering`, `research` |
| 관련 스킬 | [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv), [`jupyter-notebook`](/docs/user-guide/skills/optional/data-science/data-science-jupyter-notebook) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Darwinian Evolver

Imbue의 [darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver)를 실행하여 **프롬프트, 정규식, SQL 쿼리 또는 작은 코드 조각**을 적합도 함수에 맞게 최적화하는 **LLM 기반 진화 검색 루프**를 수행합니다.

상태: 업스트림 도구를 얇게 감싼 래퍼입니다. 이 스킬은 도구를 설치하고, `Problem` 정의(생물체 + 평가기 + 변이기)를 작성하는 과정을 안내하며, 업스트림 CLI 또는 작은 사용자 정의 Python 드라이버를 통해 루프를 실행합니다.

**라이선스:** 업스트림 도구는 **AGPL-3.0**입니다. 이 스킬은 업스트림 CLI 또는 `subprocess`/`uv run` 호출(단순 집합)에 의해서만 이를 실행합니다. 업스트림 클래스를 Hermes 자체로 가져오지 마세요.

## 사용 시점

- 사용자가 "이 프롬프트를 최적화해 줘", "X를 위한 정규식을 진화시켜 줘", "이 코드/SQL을 자동으로 개선해 줘", "더 나은 지침을 찾아 줘"라고 말할 때.
- 평가 점수(정확히 일치하는지, 정규식 통과율, 단위 테스트, LLM 평가자, 런타임 지표)와 시작 후보(생물체)가 모두 있을 때. 평가 점수가 없다면 먼저 정의해야 합니다 — 이것이 어려운 부분입니다.
- 비용을 감당할 수 있을 때: 일반적인 실행은 LLM 호출 50–500회입니다. gpt-4o-mini에서는 몇 센트, Claude Sonnet에서는 몇 달러 정도입니다.

다음과 같은 경우에는 사용하지 마세요.

- 최적화 대상이 미분 가능한 경우(경사 하강법 / DSPy를 사용하세요).
- 변형 2–3개만 시도하면 되는 경우 — 직접 작성하세요.
- 적합도 신호가 측정 가능한 기준 없이 순전히 주관적인 경우.

## 사전 요구 사항

- Python ≥3.11
- `git`, `uv`(또는 `pip`)
- 다음 중 하나: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

이 스킬에는 OpenAI SDK를 통해 `OPENROUTER_API_KEY`를 사용하는 작은 `parrot_openrouter.py` 드라이버가 포함되어 있으므로 OpenRouter의 어떤 모델이든 사용할 수 있습니다. 업스트림 CLI 자체는 Anthropic을 하드코딩하며 `ANTHROPIC_API_KEY`가 필요합니다.

## 설치(최초 한 번)

`terminal` 도구를 통해 실행합니다.

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

## 빠른 시작 — 내장 Parrot 예제

작은 스모크 테스트(`ANTHROPIC_API_KEY` 필요):

```bash
cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver
uv run darwinian_evolver parrot \
  --num_iterations 2 \
  --num_parents_per_iteration 2 \
  --mutator_concurrency 2 --evaluator_concurrency 2 \
  --output_dir /tmp/parrot_demo
```

출력:
- `/tmp/parrot_demo/snapshots/iteration_N.pkl` — 반복별 모집단을 저장한 pickle
- `/tmp/parrot_demo/<jsonl>` — 반복별 JSON 로그(마지막에 경로가 출력됨)

브라우저에서 `~/.hermes/cache/darwinian-evolver/darwinian_evolver/darwinian_evolver/lineage_visualizer.html`을 열고 JSON 로그를 로드하면 진화 트리를 볼 수 있습니다.

## 빠른 시작 — OpenRouter 드라이버(Anthropic 키 없음)

이 스킬에는 `scripts/parrot_openrouter.py`가 포함되어 있습니다. 동일한 parrot 문제를 사용하지만 LLM 호출은 OpenRouter를 거치므로 어떤 프로바이더든 사용할 수 있습니다.

```bash
# From wherever the skill is installed:
SKILL_DIR=~/.hermes/skills/research/darwinian-evolver
DE_DIR=~/.hermes/cache/darwinian-evolver/darwinian_evolver

cd "$DE_DIR" && \
  EVOLVER_MODEL='openai/gpt-4o-mini' \
  uv run --with openai python "$SKILL_DIR/scripts/parrot_openrouter.py" \
    --num_iterations 3 --num_parents_per_iteration 2 \
    --output_dir /tmp/parrot_or
```

결과를 `scripts/show_snapshot.py`로 확인합니다.

```bash
uv run --with openai python "$SKILL_DIR/scripts/show_snapshot.py" \
  /tmp/parrot_or/snapshots/iteration_3.pkl
```

예상 출력: 점수순으로 정렬된 진화된 프롬프트 템플릿 7개. 최고 점수는 약 0.6–0.8이며, 시드 `Say {{ phrase }}`의 점수는 0.000입니다.

## 사용자 정의 문제 정의

이 스킬에는 `templates/custom_problem_template.py`가 포함되어 있습니다 — 복사하고, 수정한 뒤 실행하세요. 다음 세 가지를 정의해야 합니다.

1. **`Organism`** — 진화시킬 산출물(`prompt_template: str`, `regex_pattern: str`, `sql_query: str`, `code_block: str` 등)을 담는 Pydantic `BaseModel` 서브클래스입니다. 이를 실행하는 `run(*args)` 메서드를 추가하세요.

2. **`Evaluator`** — `.evaluate(organism) -> EvaluationResult(score=..., trainable_failure_cases=[...], holdout_failure_cases=[...], is_viable=True)`입니다.
   - **`score`**는 `[0, 1]` 범위입니다. 높을수록 좋습니다.
   - **`trainable_failure_cases`** — 변이기가 보는 항목입니다. LLM이 진단할 수 있도록 충분한 맥락(입력, 기대값, 실제값)을 포함하세요.
   - **`holdout_failure_cases`** — 변이기 시야에서 제외됩니다. 과적합을 감지하는 데 사용하세요.
   - 생물체가 완전히 망가진 경우(예외 발생, `None` 반환 등)를 제외하면 **`is_viable=True`**입니다. 실행 가능한 생물체의 점수가 0이어도 괜찮습니다 — 부모 선택에서 가중치만 낮아집니다.

3. **`Mutator`** — `.mutate(organism, failure_cases, learning_log_entries) -> list[Organism]`입니다. 일반적으로 현재 생물체와 실패 사례, 수정안 제안 요청을 포함한 LLM 프롬프트를 만들고, LLM 응답을 파싱하여 새 `Organism`을 반환합니다. 파싱에 실패하면 `[]`을 반환하세요 — 루프가 처리합니다.

그런 다음 `Problem(initial_organism, evaluator, [mutators])`를 `EvolveProblemLoop`에 연결하고 `loop.run(num_iterations=N)`을 반복하는 드라이버 스크립트를 작성합니다 — 포함된 `scripts/parrot_openrouter.py`가 참고 자료입니다.

## 실제로 중요한 하이퍼파라미터

| 플래그 | 기본값 | 변경 시점 |
|---|---|---|
| `--num_iterations` | 5 | 평가기를 신뢰하게 되면 10–20으로 늘리세요 |
| `--num_parents_per_iteration` | 4 | 저렴한 탐색을 위해 2로 낮추세요 |
| `--mutator_concurrency` | 10 | 속도 제한을 피하려면 2–4로 낮추세요 |
| `--evaluator_concurrency` | 10 | 동일합니다. 평가기도 LLM을 호출합니다 |
| `--batch_size` | 1 | 변이기가 여러 실패를 처리하게 되면 3–5로 올리세요 |
| `--verify_mutations` | off | 변이기가 낭비적일 때 켜세요(Imbue에 따르면 이후 실행에서 비용을 10배 이상 절약) |
| `--midpoint_score` | `p75` | 점수가 한곳에 몰리지 않는 한 그대로 두세요 |
| `--sharpness` | 10 | 그대로 두세요 |

## 주의 사항

1. **`Initial organism must be viable`** — 점수가 0인 시드라도 `EvaluationResult`에서 `is_viable=True`로 설정하세요. 루프는 진화시킬 대상이 없다는 뜻인 실행 불가능한 생물체를 거부합니다.
2. **프로바이더 콘텐츠 필터가 실행을 중단할 수 있습니다.** Azure 기반 OpenRouter 모델은 HTTP 400으로 "ignore previous instructions"와 같은 문구를 거부합니다. LLM 호출을 `try/except`로 감싸고 `f"<LLM_ERROR: {e}>"`를 반환하세요 — 진화기는 해당 생물체의 점수를 0으로 매기고 계속 진행합니다.
3. **`loop.run()`은 제너레이터입니다** — 호출만 해서는 아무것도 실행되지 않고 반복해야 합니다. `for snap in loop.run(num_iterations=N):`을 사용하세요.
4. **스냅샷은 중첩된 pickle입니다.** `iteration_N.pkl`에는 `population_snapshot`(추가로 pickle된 바이트)이 있는 딕셔너리가 들어 있습니다. 언피클하려면 피클이 생성된 동일한 점 표기 경로에서 `Organism` 클래스를 import할 수 있어야 합니다.
5. **동시성 기본값이 공격적입니다.** 10/10은 대부분의 프로바이더에서 속도 제한에 걸립니다. 2/2로 시작하세요.
6. **CLI는 Anthropic으로 하드코딩되어 있습니다.** `uv run darwinian_evolver <problem>`은 `ANTHROPIC_API_KEY`를 사용하고 Claude Sonnet을 호출합니다. 다른 프로바이더를 사용하려면 `parrot_openrouter.py`와 같은 드라이버를 작성하세요.
7. **AGPL.** Hermes 코어 안에서 `from darwinian_evolver import ...`를 사용하지 마세요. `~/.hermes/skills/...` 아래의 사용자 정의 드라이버 스크립트는 사용자 측에서 실행되므로 괜찮습니다.
8. **PyPI 패키지가 없습니다.** `pip install darwinian-evolver`를 실행하면 잘못된 패키지가 설치됩니다. 항상 GitHub 저장소에서 설치하세요.

## 확인

설치 후 parrot을 실행하고 다음 명령의 종료 코드가 0이면 충분합니다.

```bash
DE_DIR=~/.hermes/cache/darwinian-evolver/darwinian_evolver
ls "$DE_DIR/darwinian_evolver/lineage_visualizer.html" >/dev/null && \
cd "$DE_DIR" && uv run darwinian_evolver --help >/dev/null && \
echo "darwinian-evolver: OK"
```

## 참고 자료

- [Imbue 연구 게시물](https://imbue.com/research/2026-02-27-darwinian-evolver/)
- [ARC-AGI-2 결과](https://imbue.com/research/2026-02-27-arc-agi-2-evolution/)
- [imbue-ai/darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver) (AGPL-3.0)
- [Darwin Gödel Machines](https://arxiv.org/abs/2505.22954)
- [PromptBreeder](https://arxiv.org/abs/2309.16797)
