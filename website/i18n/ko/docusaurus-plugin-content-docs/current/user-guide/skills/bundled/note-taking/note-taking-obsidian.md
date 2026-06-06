---
title: "Obsidian — Obsidian 보관소(vault)의 노트 읽기, 검색, 생성 및 편집"
sidebar_label: "Obsidian"
description: "Obsidian 보관소(vault)의 노트 읽기, 검색, 생성 및 편집"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Obsidian

Obsidian 보관소(vault)의 노트 읽기, 검색, 생성 및 편집.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/note-taking/obsidian` |
| Platforms | linux, macos, windows |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 것입니다.
:::

# Obsidian Vault

이 스킬은 노트 읽기, 노트 나열, 노트 파일 검색, 노트 생성, 콘텐츠 추가 및 위키링크(wikilink) 추가와 같이 파일 시스템 우선의 Obsidian 보관소(vault) 작업에 사용합니다.

## 보관소 경로 (Vault path)

파일 도구를 호출하기 전에 알려진 또는 확인된 보관소 경로를 사용하세요.

문서화된 보관소 경로 규칙은 `~/.hermes/.env`와 같은 곳의 `OBSIDIAN_VAULT_PATH` 환경 변수입니다. 만약 설정되지 않은 경우, `~/Documents/Obsidian Vault`를 사용하세요.

파일 도구는 셸 변수를 확장하지 않습니다. `read_file`, `write_file`, `patch` 또는 `search_files`에 `$OBSIDIAN_VAULT_PATH`가 포함된 경로를 전달하지 마세요. 보관소 경로를 먼저 확인하고 구체적인 절대 경로를 전달하세요. 보관소 경로에는 공백이 포함될 수 있으며, 이는 셸 명령어보다 파일 도구를 선호해야 하는 또 다른 이유입니다.

보관소 경로를 알 수 없는 경우, `terminal`을 사용하여 `OBSIDIAN_VAULT_PATH`를 확인하거나 폴백 경로가 존재하는지 확인하는 것은 허용됩니다. 경로를 알게 되면, 다시 파일 도구로 전환하세요.

## 노트 읽기

노트의 확인된 절대 경로와 함께 `read_file`을 사용하세요. 이는 줄 번호와 페이지네이션을 제공하므로 `cat`보다 이 방법을 선호합니다.

## 노트 나열

`target: "files"` 및 확인된 보관소 경로와 함께 `search_files`를 사용하세요. `find`나 `ls`보다 이 방법을 선호합니다.

- 모든 마크다운 노트를 나열하려면 보관소 경로 아래에서 `pattern: "*.md"`를 사용하세요.
- 하위 폴더를 나열하려면 해당 하위 폴더의 절대 경로 아래에서 검색하세요.

## 검색

파일 이름과 내용 검색 모두에 `search_files`를 사용하세요. `grep`, `find` 또는 `ls`보다 이 방법을 선호합니다.

- 파일 이름의 경우 `target: "files"` 및 파일 이름 `pattern`과 함께 `search_files`를 사용하세요.
- 노트 내용의 경우 `target: "content"`, 내용 정규식을 `pattern`으로 사용하고, 일치 항목을 마크다운 노트로 제한하려면 `file_glob: "*.md"`와 함께 `search_files`를 사용하세요.

## 노트 생성

확인된 절대 경로와 전체 마크다운 내용과 함께 `write_file`을 사용하세요. 이는 셸 따옴표 문제를 피하고 구조화된 결과를 반환하므로 셸 heredoc이나 `echo`보다 이 방법을 선호합니다.

## 노트에 추가하기

어색하지 않다면 네이티브 파일 도구 워크플로우를 선호하세요:

- `read_file`로 타겟 노트를 읽습니다.
- 기존 제목 뒤에 섹션을 추가하거나 알려진 후행 블록 앞에 추가하는 등 안정적인 컨텍스트가 있을 때 앵커(anchored) 추가를 위해 `patch`를 사용하세요.
- 취약한 패치를 구성하는 것보다 전체 노트를 다시 작성하는 것이 더 명확한 경우 `write_file`을 사용하세요.

`patch`를 사용한 앵커(anchored) 추가의 경우, 앵커를 앵커와 새로운 내용으로 교체하세요.

안정적인 컨텍스트가 없는 단순한 추가의 경우, 그것이 가장 명확하고 안전한 옵션이라면 `terminal`을 사용하는 것이 허용됩니다.

## 타겟 편집

현재 내용이 안정적인 컨텍스트를 제공하는 경우 집중적인 노트 변경을 위해 `patch`를 사용하세요. 셸 텍스트 다시 작성보다 이 방법을 선호합니다.

## 위키링크 (Wikilinks)

Obsidian은 `[[Note Name]]` 구문으로 노트를 연결합니다. 노트를 생성할 때 이것을 사용하여 관련 콘텐츠를 연결하세요.
