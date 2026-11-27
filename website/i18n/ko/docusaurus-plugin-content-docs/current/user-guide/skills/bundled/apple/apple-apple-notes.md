---
title: "Apple Notes — memo CLI로 Apple Notes 관리: 생성, 검색, 편집"
sidebar_label: "Apple Notes"
description: "memo CLI로 Apple Notes 관리: 생성, 검색, 편집"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Apple Notes

memo CLI를 통해 Apple Notes를 생성, 검색, 편집합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들 포함 (기본 설치) |
| 경로 | `skills/apple/apple-notes` |
| 버전 | `1.0.1` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | macos |
| 태그 | `Notes`, `Apple`, `macOS`, `note-taking` |
| 관련 스킬 | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# Apple Notes

`memo`를 사용해 터미널에서 직접 Apple Notes를 관리합니다. 노트는 iCloud를 통해 모든 Apple 기기에서 동기화됩니다.

## 사전 요구 사항

- **Notes.app이 설치된 macOS**
- 설치: `brew tap antoniorodr/memo && brew install antoniorodr/memo/memo`
- 메시지가 표시되면 Notes.app에 자동화 접근 권한 부여 (시스템 설정 → 개인정보 보호 및 보안 → 자동화)

## 사용 시점

- 사용자가 Apple Notes 생성, 보기 또는 검색을 요청할 때
- 여러 기기에서 접근할 수 있도록 Notes.app에 정보 저장
- 노트를 폴더로 정리
- 노트를 Markdown/HTML로 내보내기

## 사용하지 않을 때

- Obsidian 볼트 관리 → `obsidian` 스킬 사용
- Bear Notes → 별도 앱 (여기서는 지원되지 않음)
- 에이전트 전용 간단한 노트 → 대신 `memory` 도구 사용

## 빠른 참조

### 노트 보기

```bash
memo notes                        # List all notes
memo notes -f "Folder Name"       # Filter by folder
memo notes -s "query"             # Search notes (fuzzy)
```

### 노트 생성

```bash
memo notes -a                     # Add a note (opens your $EDITOR)
memo notes -a -f "Folder Name"    # Add a note into a specific folder
```

`-a`/`--add`는 단독 플래그이며, 노트를 작성할 수 있도록 사용자의 `$EDITOR`를 엽니다. 제목 인수를 받지 않습니다. 폴더를 지정하려면 `-f/--folder`를 사용합니다. 먼저 `$EDITOR`를 설정하세요 (예: `export EDITOR=vim`).

### 노트 편집

```bash
memo notes -e                     # Interactive selection to edit
```

### 노트 삭제

```bash
memo notes -d                     # Interactive selection to delete
```

### 노트 이동

```bash
memo notes -m                     # Move note to folder (interactive)
```

### 노트 내보내기

```bash
memo notes -ex                    # Export to HTML/Markdown
```

## 제한 사항

- 이미지나 첨부 파일이 포함된 노트는 편집할 수 없음
- 대화형 프롬프트에는 터미널 접근이 필요함 (필요하면 pty=true 사용)
- macOS 전용 — Apple Notes.app 필요

## 규칙

1. 사용자가 여러 기기에서 동기화하기를 원하면 Apple Notes 우선 사용
2. 동기화할 필요가 없는 에이전트 내부 노트에는 `memory` 도구 사용
3. Markdown 기반 지식 관리에는 `obsidian` 스킬 사용
