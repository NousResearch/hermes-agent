---
title: "Codebase Inspection — pygount로 코드베이스 검사: LOC, 언어, 비율"
sidebar_label: "Codebase Inspection"
description: "pygount로 코드베이스 검사: LOC, 언어, 비율"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Codebase Inspection

pygount로 코드베이스를 검사합니다: LOC, 언어, 비율.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치) |
| 경로 | `skills/github/codebase-inspection` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `LOC`, `Code Analysis`, `pygount`, `Codebase`, `Metrics`, `Repository` |
| 관련 스킬 | [`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# pygount를 사용한 코드베이스 검사

`pygount`를 사용해 저장소의 코드 줄 수, 언어별 구성, 파일 수, 코드와 주석의 비율을 분석합니다.

## 사용 시점

- 사용자가 LOC(코드 줄 수)를 요청하는 경우
- 저장소의 언어별 구성을 알고 싶어 하는 경우
- 코드베이스의 규모나 구성을 묻는 경우
- 코드와 주석의 비율을 알고 싶어 하는 경우
- 일반적인 "이 저장소는 얼마나 큰가"라는 질문

## 사전 요구 사항

```bash
pip install --break-system-packages pygount 2>/dev/null || pip install pygount
```

## 1. 기본 요약(가장 일반적인 사용)

파일 수, 코드 줄 수, 주석 줄 수가 포함된 전체 언어별 구성을 가져옵니다.

```bash
cd /path/to/repo
pygount --format=summary \
  --folders-to-skip=".git,node_modules,venv,.venv,__pycache__,.cache,dist,build,.next,.tox,.eggs,*.egg-info" \
  .
```

**중요:** 항상 `--folders-to-skip`을 사용해 의존성/빌드 디렉터리를 제외합니다. 그렇지 않으면 pygount가 해당 디렉터리를 모두 탐색하느라 매우 오래 걸리거나 멈출 수 있습니다.

## 2. 일반적인 폴더 제외

프로젝트 유형에 따라 조정합니다.

```bash
# Python projects
--folders-to-skip=".git,venv,.venv,__pycache__,.cache,dist,build,.tox,.eggs,.mypy_cache"

# JavaScript/TypeScript projects
--folders-to-skip=".git,node_modules,dist,build,.next,.cache,.turbo,coverage"

# General catch-all
--folders-to-skip=".git,node_modules,venv,.venv,__pycache__,.cache,dist,build,.next,.tox,vendor,third_party"
```

## 3. 특정 언어로 필터링

```bash
# Only count Python files
pygount --suffix=py --format=summary .

# Only count Python and YAML
pygount --suffix=py,yaml,yml --format=summary .
```

## 4. 파일별 상세 출력

```bash
# Default format shows per-file breakdown
pygount --folders-to-skip=".git,node_modules,venv" .

# Sort by code lines (pipe through sort)
pygount --folders-to-skip=".git,node_modules,venv" . | sort -t$'\t' -k1 -nr | head -20
```

## 5. 출력 형식

```bash
# Summary table (default recommendation)
pygount --format=summary .

# JSON output for programmatic use
pygount --format=json .

# Pipe-friendly: Language, file count, code, docs, empty, string
pygount --format=summary . 2>/dev/null
```

## 6. 결과 해석

요약 표의 열:
- **Language** — 감지된 프로그래밍 언어
- **Files** — 해당 언어의 파일 수
- **Code** — 실제 코드 줄 수(실행 가능/선언적 코드)
- **Comment** — 주석 또는 문서 줄 수
- **%** — 전체에서 차지하는 비율

특수 의사 언어:
- `__empty__` — 빈 파일
- `__binary__` — 바이너리 파일(이미지, 컴파일된 파일 등)
- `__generated__` — 자동 생성 파일(휴리스틱으로 감지)
- `__duplicate__` — 내용이 동일한 파일
- `__unknown__` — 인식되지 않은 파일 형식

## 주의 사항

1. **항상 .git, node_modules, venv를 제외** — `--folders-to-skip` 없이 실행하면 pygount가 모든 항목을 탐색하므로 대규모 의존성 트리에서 몇 분이 걸리거나 멈출 수 있습니다.
2. **Markdown에서 코드 줄 수가 0으로 표시됨** — pygount는 모든 Markdown 콘텐츠를 코드가 아닌 주석으로 분류합니다. 이는 정상적인 동작입니다.
3. **JSON 파일의 코드 줄 수가 낮게 표시됨** — pygount는 JSON 줄 수를 보수적으로 계산할 수 있습니다. 정확한 JSON 줄 수를 세려면 `wc -l`을 직접 사용합니다.
4. **대규모 모노레포** — 매우 큰 저장소에서는 전체를 검사하는 대신 `--suffix`를 사용해 특정 언어를 대상으로 하는 것을 고려합니다.
