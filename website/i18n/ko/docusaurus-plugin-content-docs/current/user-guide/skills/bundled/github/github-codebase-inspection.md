---
title: "Codebase Inspection — pygount로 코드베이스 검사: LOC, 언어, 비율"
sidebar_label: "Codebase Inspection"
description: "pygount로 코드베이스 검사: LOC, 언어, 비율"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Codebase Inspection

pygount로 코드베이스 검사: LOC, 언어, 비율.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/github/codebase-inspection` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `LOC`, `Code Analysis`, `pygount`, `Codebase`, `Metrics`, `Repository` |
| Related skills | [`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 것입니다.
:::

# pygount를 이용한 코드베이스 검사 (Codebase Inspection with pygount)

`pygount`를 사용하여 코드 라인, 언어 분류, 파일 수 및 코드 대 주석 비율에 대해 리포지토리를 분석합니다.

## 언제 사용해야 하나

- 사용자가 LOC(코드 라인 수)를 요청할 때
- 사용자가 리포지토리의 언어 분류를 알고 싶어할 때
- 사용자가 코드베이스의 크기나 구성에 대해 물어볼 때
- 사용자가 코드 대 주석 비율을 원할 때
- "이 리포지토리의 크기가 얼마나 되나요?"와 같은 일반적인 질문

## 전제 조건

```bash
pip install --break-system-packages pygount 2>/dev/null || pip install pygount
```

## 1. 기본 요약 (가장 일반적임)

파일 수, 코드 라인 및 주석 라인에 대한 전체 언어 분류를 가져옵니다:

```bash
cd /path/to/repo
pygount --format=summary \
  --folders-to-skip=".git,node_modules,venv,.venv,__pycache__,.cache,dist,build,.next,.tox,.eggs,*.egg-info" \
  .
```

**중요:** 종속성/빌드 디렉터리를 제외하려면 항상 `--folders-to-skip`을 사용하세요. 그렇지 않으면 pygount가 이를 크롤링하여 시간이 매우 오래 걸리거나 멈출 수 있습니다.

## 2. 일반적인 폴더 제외

프로젝트 유형에 따라 조정합니다:

```bash
# Python 프로젝트
--folders-to-skip=".git,venv,.venv,__pycache__,.cache,dist,build,.tox,.eggs,.mypy_cache"

# JavaScript/TypeScript 프로젝트
--folders-to-skip=".git,node_modules,dist,build,.next,.cache,.turbo,coverage"

# 일반적인 용도
--folders-to-skip=".git,node_modules,venv,.venv,__pycache__,.cache,dist,build,.next,.tox,vendor,third_party"
```

## 3. 특정 언어로 필터링

```bash
# Python 파일만 계산
pygount --suffix=py --format=summary .

# Python과 YAML만 계산
pygount --suffix=py,yaml,yml --format=summary .
```

## 4. 파일별 세부 출력

```bash
# 기본 형식은 파일별 분석을 보여줍니다
pygount --folders-to-skip=".git,node_modules,venv" .

# 코드 라인에 따라 정렬 (sort를 파이프로 연결)
pygount --folders-to-skip=".git,node_modules,venv" . | sort -t$'\t' -k1 -nr | head -20
```

## 5. 출력 형식

```bash
# 요약 표 (기본 권장 사항)
pygount --format=summary .

# 프로그래밍 방식으로 사용하기 위한 JSON 출력
pygount --format=json .

# 파이프 친화적: 언어, 파일 수, 코드, 문서, 빈칸, 문자열
pygount --format=summary . 2>/dev/null
```

## 6. 결과 해석

요약 테이블 열:
- **Language** — 감지된 프로그래밍 언어
- **Files** — 해당 언어의 파일 수
- **Code** — 실제 코드 라인 (실행/선언)
- **Comment** — 주석 또는 문서인 라인
- **%** — 전체 대비 비율

특별한 의사 언어(pseudo-languages):
- `__empty__` — 빈 파일
- `__binary__` — 바이너리 파일 (이미지, 컴파일된 파일 등)
- `__generated__` — 자동 생성된 파일 (휴리스틱하게 감지됨)
- `__duplicate__` — 내용이 동일한 파일
- `__unknown__` — 인식할 수 없는 파일 유형

## 주의 사항 (Pitfalls)

1. **항상 .git, node_modules, venv 제외** — `--folders-to-skip`이 없으면 pygount는 모든 것을 크롤링하며, 대규모 종속성 트리에서는 수 분이 걸리거나 중단될 수 있습니다.
2. **Markdown에서 코드 라인이 0으로 표시됨** — pygount는 모든 Markdown 콘텐츠를 코드가 아닌 주석으로 분류합니다. 이는 예상된 동작입니다.
3. **JSON 파일에서 낮은 코드 수가 표시됨** — pygount는 JSON 라인을 보수적으로 계산할 수 있습니다. 정확한 JSON 라인 수를 얻으려면 `wc -l`을 직접 사용하세요.
4. **대규모 모노레포(monorepos)** — 매우 큰 리포지토리의 경우, 모든 것을 스캔하는 대신 `--suffix`를 사용하여 특정 언어를 타겟팅하는 것을 고려하세요.
