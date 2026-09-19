---
sidebar_position: 9
sidebar_label: "컨텍스트 참조"
title: "컨텍스트 참조"
description: "파일, 폴더, git diff 및 URL을 메시지에 직접 첨부하는 인라인 @ 구문"
---

# 컨텍스트 참조

`@` 뒤에 참조를 입력하면 콘텐츠가 메시지에 직접 삽입됩니다. Hermes는 참조를 인라인으로 확장하고 `--- Attached Context ---` 섹션 아래에 콘텐츠를 덧붙입니다.

## 지원되는 참조

| 구문 | 설명 |
|--------|-------------|
| `@file:path/to/file.py` | 파일 내용 삽입 |
| `@file:path/to/file.py:10-25` | 특정 줄 범위 삽입(1부터 시작, 양 끝 포함) |
| `@folder:path/to/dir` | 파일 메타데이터가 포함된 디렉터리 트리 목록 삽입 |
| `@diff` | `git diff` 삽입(스테이징되지 않은 작업 트리 변경 사항) |
| `@staged` | `git diff --staged` 삽입(스테이징된 변경 사항) |
| `@git:5` | 마지막 N개 커밋을 패치와 함께 삽입(최대 10개) |
| `@url:https://example.com` | 웹페이지 콘텐츠를 가져와 삽입 |

## 사용 예시

```text
Review @file:src/main.py and suggest improvements

What changed? @diff

Compare @file:old_config.yaml and @file:new_config.yaml

What's in @folder:src/components?

Summarize this article @url:https://arxiv.org/abs/2301.00001
```

하나의 메시지에서 여러 참조를 사용할 수 있습니다.

```text
Check @file:main.py, and also @file:test.py.
```

후행 구두점(`,`, `.`, `;`, `!`, `?`)은 참조 값에서 자동으로 제거됩니다.

## CLI 탭 자동 완성

대화형 CLI에서 `@`를 입력하면 자동 완성이 실행됩니다.

- `@`는 모든 참조 유형(`@diff`, `@staged`, `@file:`, `@folder:`, `@git:`, `@url:`)을 표시합니다
- `@file:` 및 `@folder:`는 파일 크기 메타데이터가 포함된 파일 시스템 경로 완성을 실행합니다
- 부분 텍스트가 뒤따르는 단독 `@`는 현재 디렉터리에서 일치하는 파일과 폴더를 표시합니다

## 줄 범위

정확한 콘텐츠 삽입을 위해 `@file:` 참조는 줄 범위를 지원합니다.

```text
@file:src/main.py:42        # Single line 42
@file:src/main.py:10-25     # Lines 10 through 25 (inclusive)
```

줄 번호는 1부터 시작합니다. 잘못된 범위는 조용히 무시되고 전체 파일이 반환됩니다.

## 크기 제한

컨텍스트 참조에는 모델의 컨텍스트 창이 넘치지 않도록 제한이 적용됩니다.

| 기준 | 값 | 동작 |
|-----------|-------|----------|
| 소프트 제한 | 컨텍스트 길이의 25% | 경고를 덧붙이고 확장을 진행합니다 |
| 하드 제한 | 컨텍스트 길이의 50% | 확장을 거부하고 변경하지 않은 원래 메시지를 반환합니다 |
| 폴더 항목 | 최대 200개 파일 | 초과 항목을 `- ...`로 대체합니다 |
| Git 커밋 | 최대 10개 | `@git:N`을 [1, 10] 범위로 제한합니다 |

## 보안

### 민감한 경로 차단

자격 증명이 노출되는 것을 방지하기 위해 다음 경로는 `@file:` 참조에서 항상 차단됩니다.

- SSH 키 및 설정: `~/.ssh/id_rsa`, `~/.ssh/id_ed25519`, `~/.ssh/authorized_keys`, `~/.ssh/config`
- 셸 프로필: `~/.bashrc`, `~/.zshrc`, `~/.profile`, `~/.bash_profile`, `~/.zprofile`
- 자격 증명 파일: `~/.netrc`, `~/.pgpass`, `~/.npmrc`, `~/.pypirc`
- Hermes 환경: `$HERMES_HOME/.env`

다음 디렉터리는 내부의 모든 파일이 완전히 차단됩니다.
- `~/.ssh/`, `~/.aws/`, `~/.gnupg/`, `~/.kube/`, `$HERMES_HOME/skills/.hub/`

### 경로 탐색 방지

모든 경로는 작업 디렉터리를 기준으로 확인됩니다. 허용된 작업 공간 루트 외부로 확인되는 참조는 거부됩니다.

### 바이너리 파일 감지

바이너리 파일은 MIME 유형과 널 바이트 스캔으로 감지됩니다. 알려진 텍스트 확장자(`.py`, `.md`, `.json`, `.yaml`, `.toml`, `.js`, `.ts` 등)는 MIME 기반 감지를 우회합니다. 바이너리 파일은 경고와 함께 거부됩니다.

## 플랫폼 지원

컨텍스트 참조는 주로 **CLI 기능**입니다. `@`가 탭 자동 완성을 실행하고 메시지가 에이전트로 전송되기 전에 참조가 확장되는 대화형 CLI에서 작동합니다.

**메시징 플랫폼**(Telegram, Discord 등)에서는 게이트웨이가 `@` 구문을 확장하지 않으며 메시지가 있는 그대로 전달됩니다. 그래도 에이전트 자체는 `read_file`, `search_files`, `web_extract` 도구를 통해 파일을 참조할 수 있습니다.

## 컨텍스트 압축과의 상호 작용

대화 컨텍스트가 압축될 때 확장된 참조 콘텐츠가 압축 요약에 포함됩니다. 이는 다음을 의미합니다.

- `@file:`로 삽입된 대용량 파일 내용은 컨텍스트 사용량에 포함됩니다
- 이후 대화가 압축되면 파일 내용은 요약되며(그대로 보존되지 않음)
- 매우 큰 파일의 경우 줄 범위(`@file:main.py:100-200`)를 사용해 필요한 부분만 삽입하는 것을 고려하세요

## 일반적인 패턴

```text
# Code review workflow
Review @diff and check for security issues

# Debug with context
This test is failing. Here's the test @file:tests/test_auth.py
and the implementation @file:src/auth.py:50-80

# Project exploration
What does this project do? @folder:src @file:README.md

# Research
Compare the approaches in @url:https://arxiv.org/abs/2301.00001
and @url:https://arxiv.org/abs/2301.00002
```

## 오류 처리

잘못된 참조는 실패를 일으키는 대신 인라인 경고를 생성합니다.

| 조건 | 동작 |
|-----------|----------|
| 파일을 찾을 수 없음 | 경고: "file not found" |
| 바이너리 파일 | 경고: "binary files are not supported" |
| 폴더를 찾을 수 없음 | 경고: "folder not found" |
| Git 명령 실패 | Git stderr가 포함된 경고 |
| URL이 콘텐츠를 반환하지 않음 | 경고: "no content extracted" |
| 민감한 경로 | 경고: "path is a sensitive credential file" |
| 작업 공간 외부 경로 | 경고: "path is outside the allowed workspace" |
