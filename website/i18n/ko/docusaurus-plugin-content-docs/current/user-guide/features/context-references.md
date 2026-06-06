---
sidebar_position: 9
sidebar_label: "컨텍스트 참조 (Context References)"
title: "컨텍스트 참조 (Context References)"
description: "파일, 폴더, git diff 및 URL을 메시지에 직접 첨부하기 위한 인라인 @ 구문"
---

# 컨텍스트 참조 (Context References)

`@` 뒤에 참조를 입력하여 콘텐츠를 메시지에 직접 주입하세요. Hermes는 참조를 인라인으로 확장하고 `--- Attached Context ---` 섹션 아래에 콘텐츠를 추가합니다.

## 지원되는 참조

| 구문 (Syntax) | 설명 |
|--------|-------------|
| `@file:path/to/file.py` | 파일 내용을 주입 |
| `@file:path/to/file.py:10-25` | 특정 줄 범위 주입 (1-인덱스, 포함) |
| `@folder:path/to/dir` | 파일 메타데이터가 포함된 디렉터리 트리 목록 주입 |
| `@diff` | `git diff` 주입 (스테이징되지 않은 작업 트리 변경 사항) |
| `@staged` | `git diff --staged` 주입 (스테이징된 변경 사항) |
| `@git:5` | 패치가 포함된 마지막 N개의 커밋 주입 (최대 10개) |
| `@url:https://example.com` | 웹 페이지 콘텐츠를 가져와서 주입 |

## 사용 예시

```text
@file:src/main.py 를 검토하고 개선 사항을 제안해 줘

무엇이 변경되었어? @diff

@file:old_config.yaml 과 @file:new_config.yaml 을 비교해 줘

@folder:src/components 에는 무엇이 있어?

이 기사를 요약해 줘 @url:https://arxiv.org/abs/2301.00001
```

하나의 메시지에 여러 참조를 사용할 수 있습니다:

```text
@file:main.py 를 확인하고, @file:test.py 도 확인해 줘.
```

참조 값 뒤에 오는 구두점(`.`, `,`, `;`, `!`, `?`)은 자동으로 제거됩니다.

## CLI 탭 자동 완성

대화형 CLI에서 `@`를 입력하면 자동 완성이 트리거됩니다:

- `@`는 모든 참조 유형(`@diff`, `@staged`, `@file:`, `@folder:`, `@git:`, `@url:`)을 표시합니다.
- `@file:` 및 `@folder:`는 파일 크기 메타데이터와 함께 파일 시스템 경로 자동 완성을 트리거합니다.
- 단독 `@` 뒤에 텍스트의 일부를 입력하면 현재 디렉터리에서 일치하는 파일과 폴더를 표시합니다.

## 줄 범위 (Line Ranges)

`@file:` 참조는 콘텐츠를 정밀하게 주입하기 위해 줄 범위를 지원합니다:

```text
@file:src/main.py:42        # 42번째 줄 단일
@file:src/main.py:10-25     # 10번째 줄부터 25번째 줄까지 (포함)
```

줄은 1-인덱스(1-indexed)입니다. 잘못된 범위는 조용히 무시됩니다 (전체 파일이 반환됨).

## 크기 제한 (Size Limits)

모델의 컨텍스트 윈도우를 압도하는 것을 방지하기 위해 컨텍스트 참조가 제한됩니다:

| 임계값 (Threshold) | 값 | 동작 |
|-----------|-------|----------|
| 소프트 리밋 (Soft limit) | 컨텍스트 길이의 25% | 경고가 추가되고, 확장이 계속 진행됨 |
| 하드 리밋 (Hard limit) | 컨텍스트 길이의 50% | 확장이 거부되고, 원본 메시지가 변경 없이 반환됨 |
| 폴더 항목 | 최대 200개 파일 | 초과 항목은 `- ...` 로 대체됨 |
| Git 커밋 | 최대 10개 | `@git:N`은 [1, 10] 범위로 제한됨 |

## 보안 (Security)

### 민감한 경로 차단

자격 증명 노출을 방지하기 위해 다음 경로들은 `@file:` 참조가 항상 차단됩니다:

- SSH 키 및 구성: `~/.ssh/id_rsa`, `~/.ssh/id_ed25519`, `~/.ssh/authorized_keys`, `~/.ssh/config`
- 쉘 프로필: `~/.bashrc`, `~/.zshrc`, `~/.profile`, `~/.bash_profile`, `~/.zprofile`
- 자격 증명 파일: `~/.netrc`, `~/.pgpass`, `~/.npmrc`, `~/.pypirc`
- Hermes 환경 변수: `$HERMES_HOME/.env`

다음 디렉터리들은 (내부의 어떤 파일이든) 완전히 차단됩니다:
- `~/.ssh/`, `~/.aws/`, `~/.gnupg/`, `~/.kube/`, `$HERMES_HOME/skills/.hub/`

### 경로 탐색 (Path Traversal) 보호

모든 경로는 작업 디렉터리를 기준으로 해결(resolved)됩니다. 허용된 작업 공간 루트를 벗어나는 참조는 거부됩니다.

### 바이너리 파일 감지

바이너리 파일은 MIME 타입과 널 바이트(null-byte) 스캔을 통해 감지됩니다. 알려진 텍스트 확장자(`.py`, `.md`, `.json`, `.yaml`, `.toml`, `.js`, `.ts` 등)는 MIME 기반 감지를 우회합니다. 바이너리 파일은 경고와 함께 거부됩니다.

## 플랫폼 가용성

컨텍스트 참조는 기본적으로 **CLI 기능**입니다. 대화형 CLI에서 작동하며 `@`가 탭 자동 완성을 트리거하고 메시지가 에이전트로 전송되기 전에 참조가 확장됩니다.

**메시징 플랫폼**(Telegram, Discord 등)에서는 게이트웨이가 `@` 구문을 확장하지 않으며 — 메시지가 있는 그대로 전달됩니다. 에이전트 자체는 여전히 `read_file`, `search_files`, `web_extract` 도구를 통해 파일을 참조할 수 있습니다.

## 컨텍스트 압축과의 상호 작용

대화 컨텍스트가 압축될 때, 확장된 참조 콘텐츠는 압축 요약에 포함됩니다. 이는 다음을 의미합니다:

- `@file:`을 통해 주입된 큰 파일 콘텐츠는 컨텍스트 사용량에 기여합니다.
- 대화가 나중에 압축되면 파일 콘텐츠가 요약됩니다 (원문 그대로 보존되지 않음).
- 매우 큰 파일의 경우, 줄 범위(`@file:main.py:100-200`)를 사용하여 관련 섹션만 주입하는 것을 고려하세요.

## 일반적인 패턴

```text
# 코드 검토 워크플로우
@diff 를 검토하고 보안 문제가 있는지 확인해 줘

# 컨텍스트를 활용한 디버깅
이 테스트가 실패하고 있어. 여기 테스트 코드가 있고 @file:tests/test_auth.py
여기 구현 코드가 있어 @file:src/auth.py:50-80

# 프로젝트 탐색
이 프로젝트는 무엇을 하는 프로젝트야? @folder:src @file:README.md

# 리서치
@url:https://arxiv.org/abs/2301.00001 와
@url:https://arxiv.org/abs/2301.00002 의 접근 방식을 비교해 줘
```

## 에러 처리

잘못된 참조는 실패하는 대신 인라인 경고를 생성합니다:

| 조건 | 동작 |
|-----------|----------|
| 파일을 찾을 수 없음 | 경고: "file not found" |
| 바이너리 파일 | 경고: "binary files are not supported" |
| 폴더를 찾을 수 없음 | 경고: "folder not found" |
| Git 명령어 실패 | git stderr를 포함한 경고 |
| URL이 콘텐츠를 반환하지 않음 | 경고: "no content extracted" |
| 민감한 경로 | 경고: "path is a sensitive credential file" |
| 경로가 작업 공간을 벗어남 | 경고: "path is outside the allowed workspace" |
