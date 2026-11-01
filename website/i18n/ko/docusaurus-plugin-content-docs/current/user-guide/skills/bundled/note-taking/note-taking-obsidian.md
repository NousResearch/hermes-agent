---
title: "Obsidian — Obsidian 보관함의 노트 읽기, 검색, 생성, 편집"
sidebar_label: "Obsidian"
description: "Obsidian 보관함의 노트를 읽고, 검색하고, 생성하고, 편집"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Obsidian

Obsidian 보관함의 노트를 읽고, 검색하고, 생성하고, 편집합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들 제공 (기본 설치) |
| 경로 | `skills/note-taking/obsidian` |
| 버전 | `1.0.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Obsidian`, `Notes`, `Markdown`, `Vault` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 불러오는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 확인하는 내용입니다.
:::

# Obsidian 보관함

파일 시스템 우선 Obsidian 보관함 작업에 이 스킬을 사용합니다. 노트 읽기, 노트 나열, 노트 파일 검색, 노트 생성, 콘텐츠 추가, 위키링크 추가를 지원합니다.

## 보관함 경로

파일 도구를 호출하기 전에 알고 있거나 확인된 보관함 경로를 사용합니다.

문서화된 보관함 경로 규칙은 예를 들어 `${HERMES_HOME:-~/.hermes}/.env`에서 가져오는 `OBSIDIAN_VAULT_PATH` 환경 변수입니다. 설정되지 않은 경우 `~/Documents/Obsidian Vault`를 사용합니다.

파일 도구는 셸 변수를 확장하지 않습니다. `read_file`, `write_file`, `patch` 또는 `search_files`에 `$OBSIDIAN_VAULT_PATH`가 포함된 경로를 전달하지 말고, 먼저 구체적인 절대 경로로 확인한 뒤 전달하세요. 보관함 경로에는 공백이 포함될 수 있으므로 파일 도구를 선호해야 하는 또 다른 이유가 됩니다.

보관함 경로를 알 수 없다면 `OBSIDIAN_VAULT_PATH`를 확인하거나 대체 경로가 존재하는지 검사하기 위해 `terminal`을 사용해도 됩니다. 경로를 확인한 뒤에는 파일 도구로 전환하세요.

## 노트 읽기

확인된 절대 경로를 사용해 노트에 `read_file`을 사용합니다. 줄 번호와 페이지 구분을 제공하므로 `cat`보다 이 방법을 선호하세요.

## 노트 나열

확인된 보관함 경로에 `target: "files"`를 사용해 `search_files`로 노트를 나열합니다. `find` 또는 `ls`보다 이 방법을 선호하세요.

- 모든 마크다운 노트를 나열하려면 보관함 경로 아래에 `pattern: "*.md"`를 사용합니다.
- 하위 폴더를 나열하려면 해당 폴더의 절대 경로 아래에서 검색합니다.

## 검색

파일 이름과 콘텐츠 검색 모두에 `search_files`를 사용합니다. `grep`, `find` 또는 `ls`보다 이 방법을 선호하세요.

- 파일 이름 검색에는 `target: "files"`와 파일 이름 `pattern`을 함께 사용합니다.
- 노트 콘텐츠 검색에는 `target: "content"`와 콘텐츠 정규식으로 `pattern`을 사용하고, 마크다운 노트로 제한하려면 `file_glob: "*.md"`를 지정합니다.

## 노트 생성

전체 마크다운 콘텐츠와 함께 확인된 절대 경로에 `write_file`을 사용합니다. 셸 heredoc이나 `echo`보다 이 방법을 선호하세요. 셸 인용 문제를 피하고 구조화된 결과를 반환하기 때문입니다.

## 노트에 추가

어색하지 않다면 네이티브 파일 도구 워크플로를 선호합니다:

- `read_file`로 대상 노트를 읽습니다.
- 기존 제목 뒤에 콘텐츠를 추가하거나 알려진 마지막 블록 앞에 콘텐츠를 추가하는 등 안정적인 문맥이 있다면 추가 지점을 고정한 `patch`를 사용합니다.
- 단순히 안정적인 문맥 없이 추가하는 경우에는 전체 노트를 다시 작성하는 것이 더 명확하다면 `write_file`을 사용합니다.

추가 지점을 고정한 `patch`에서는 기준 문맥을 새 콘텐츠와 함께 기준 문맥+새 콘텐츠로 교체합니다.

간단한 추가이고 안정적인 문맥이 없다면 가장 명확하고 안전한 방법으로 `terminal`을 사용해도 됩니다.

## 지정된 부분 편집

현재 콘텐츠에 안정적인 문맥이 있다면 집중된 노트 변경에는 `patch`를 사용합니다. 셸을 통한 텍스트 재작성보다 이 방법을 선호하세요.

## 위키링크

Obsidian은 `[[Note Name]]` 구문으로 노트를 연결합니다. 노트를 생성할 때 이 구문을 사용해 관련 노트에 연결하세요.
