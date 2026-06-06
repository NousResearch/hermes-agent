---
title: "Apple Notes — memo CLI를 통한 Apple Notes 관리: 생성, 검색, 편집"
sidebar_label: "Apple Notes"
description: "memo CLI를 통한 Apple Notes 관리: 생성, 검색, 편집"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트를 통해 해당 스킬의 SKILL.md에서 자동 생성됩니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Apple Notes

memo CLI를 통한 Apple Notes 관리: 생성, 검색, 편집.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| 출처 (Source) | 번들 (기본적으로 설치됨) |
| 경로 (Path) | `skills/apple/apple-notes` |
| 버전 (Version) | `1.0.0` |
| 작성자 (Author) | Hermes Agent |
| 라이선스 (License) | MIT |
| 플랫폼 (Platforms) | macos |
| 태그 (Tags) | `Notes`, `Apple`, `macOS`, `note-taking` |
| 관련 스킬 (Related skills) | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것이 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Apple Notes

터미널에서 직접 Apple Notes를 관리하기 위해 `memo`를 사용합니다. 메모는 iCloud를 통해 모든 Apple 기기 간에 동기화됩니다.

## 전제 조건 (Prerequisites)

- Notes.app이 있는 **macOS**
- 설치: `brew tap antoniorodr/memo && brew install antoniorodr/memo/memo`
- 메시지가 표시되면 Notes.app에 대한 자동화(Automation) 권한 부여 (시스템 설정 → 개인정보 보호 및 보안 → 자동화)

## 사용 시기 (When to Use)

- 사용자가 Apple Notes를 생성, 보기 또는 검색하도록 요청할 때
- 여러 기기에서 액세스할 수 있도록 정보를 Notes.app에 저장할 때
- 폴더로 메모를 구성할 때
- 메모를 Markdown/HTML로 내보낼 때

## 사용하지 말아야 할 때 (When NOT to Use)

- Obsidian 금고(vault) 관리 → `obsidian` 스킬 사용
- Bear Notes → 별도의 앱 (여기서는 지원되지 않음)
- 빠른 에이전트 전용 메모 → 대신 `memory` 도구 사용

## 빠른 참조 (Quick Reference)

### 메모 보기 (View Notes)

```bash
memo notes                        # 모든 메모 목록
memo notes -f "Folder Name"       # 폴더별 필터링
memo notes -s "query"             # 메모 검색 (퍼지 검색)
```

### 메모 만들기 (Create Notes)

```bash
memo notes -a                     # 대화형 편집기
memo notes -a "Note Title"        # 제목과 함께 빠른 추가
```

### 메모 편집 (Edit Notes)

```bash
memo notes -e                     # 편집할 대화형 선택
```

### 메모 삭제 (Delete Notes)

```bash
memo notes -d                     # 삭제할 대화형 선택
```

### 메모 이동 (Move Notes)

```bash
memo notes -m                     # 폴더로 메모 이동 (대화형)
```

### 메모 내보내기 (Export Notes)

```bash
memo notes -ex                    # HTML/Markdown으로 내보내기
```

## 제한 사항 (Limitations)

- 이미지나 첨부 파일이 포함된 메모는 편집할 수 없습니다.
- 대화형 프롬프트는 터미널 액세스가 필요합니다 (필요한 경우 pty=true 사용).
- macOS 전용 — Apple Notes.app이 필요합니다.

## 규칙 (Rules)

1. 사용자가 기기 간(iPhone/iPad/Mac) 동기화를 원할 때 Apple Notes를 선호하십시오.
2. 동기화할 필요가 없는 에이전트 내부 메모의 경우 `memory` 도구를 사용하십시오.
3. Markdown 네이티브 지식 관리를 위해 `obsidian` 스킬을 사용하십시오.
