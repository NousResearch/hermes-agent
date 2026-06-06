---
sidebar_position: 3
sidebar_label: "Git 워크트리"
title: "Git 워크트리"
description: "git 워크트리와 격리된 체크아웃을 사용하여 동일한 저장소에서 여러 Hermes 에이전트를 안전하게 실행하기"
---

# Git 워크트리

Hermes Agent는 규모가 크고 오랫동안 유지되는 저장소에서 자주 사용됩니다. 다음과 같은 경우에 유용합니다:

- 동일한 프로젝트에서 **여러 에이전트를 병렬로 실행**하고자 할 때, 또는
- 실험적인 리팩토링을 메인 브랜치와 격리된 상태로 유지하고자 할 때

이럴 때 Git **워크트리**는 전체 저장소를 복제하지 않고도 각 에이전트에게 고유한 체크아웃을 제공하는 가장 안전한 방법입니다.

이 페이지에서는 Hermes와 워크트리를 결합하여 각 세션이 깨끗하고 격리된 작업 디렉토리를 갖도록 설정하는 방법을 보여줍니다.

## Hermes와 워크트리를 함께 사용하는 이유

Hermes는 **현재 작업 디렉토리**를 프로젝트 루트로 간주합니다:

- CLI: `hermes` 또는 `hermes chat`을 실행하는 디렉토리
- 메시징 게이트웨이: `~/.hermes/config.yaml`의 `terminal.cwd`로 설정된 디렉토리

**동일한 체크아웃**에서 여러 에이전트를 실행할 경우 각 에이전트의 변경 사항이 서로 충돌을 일으킬 수 있습니다:

- 한 에이전트가 다른 에이전트가 사용 중인 파일을 삭제하거나 덮어쓸 수 있습니다.
- 어떤 변경 사항이 어느 실험에 속하는지 파악하기가 더 어려워집니다.

워크트리를 사용하면 각 에이전트는 다음을 갖게 됩니다:

- **고유한 브랜치와 작업 디렉토리**
- `/rollback`을 위한 **독립적인 Checkpoint Manager 기록**

참고: [체크포인트 및 /rollback](./checkpoints-and-rollback.md).

## 빠른 시작: 워크트리 생성하기

메인 저장소(`.git/`이 있는 디렉토리)에서 기능 브랜치를 위한 새 워크트리를 생성합니다:

```bash
# 메인 저장소 루트에서
cd /path/to/your/repo

# ../repo-feature 에 새 브랜치와 워크트리 생성
git worktree add ../repo-feature feature/hermes-experiment
```

이렇게 하면 다음이 생성됩니다:

- 새 디렉토리: `../repo-feature`
- 해당 디렉토리에 체크아웃된 새 브랜치: `feature/hermes-experiment`

이제 새로운 워크트리 디렉토리로 이동(`cd`)하여 거기서 Hermes를 실행할 수 있습니다:

```bash
cd ../repo-feature

# 워크트리에서 Hermes 시작
hermes
```

Hermes는 다음과 같이 작동합니다:

- `../repo-feature`를 프로젝트 루트로 인식합니다.
- 컨텍스트 파일, 코드 수정, 도구 실행 등에 이 디렉토리를 사용합니다.
- 이 워크트리로 범위가 지정된 **별도의 체크포인트 기록**을 `/rollback`에 사용합니다.

## 여러 에이전트를 병렬로 실행하기

고유한 브랜치를 가진 여러 워크트리를 생성할 수 있습니다:

```bash
cd /path/to/your/repo

git worktree add ../repo-experiment-a feature/hermes-a
git worktree add ../repo-experiment-b feature/hermes-b
```

각각 별도의 터미널에서 실행합니다:

```bash
# 터미널 1
cd ../repo-experiment-a
hermes

# 터미널 2
cd ../repo-experiment-b
hermes
```

각 Hermes 프로세스는:

- 자체 브랜치에서 작업합니다 (`feature/hermes-a` vs `feature/hermes-b`).
- (워크트리 경로에서 파생된) 서로 다른 섀도우 저장소 해시 아래에 체크포인트를 기록합니다.
- 서로 영향을 주지 않고 독립적으로 `/rollback`을 사용할 수 있습니다.

이 방법은 특히 다음과 같은 경우에 유용합니다:

- 일괄(batch) 리팩토링을 실행할 때
- 동일한 작업에 대해 다양한 접근 방식을 시도할 때
- 동일한 업스트림 저장소에 대해 CLI와 게이트웨이 세션을 함께 페어링할 때

## 워크트리를 안전하게 정리하기

실험이 끝난 후:

1. 작업 결과를 유지할지 폐기할지 결정합니다.
2. 유지하려면:
   - 평소와 같이 이 브랜치를 메인 브랜치에 병합합니다.
3. 워크트리를 제거합니다:

```bash
cd /path/to/your/repo

# 워크트리 디렉토리 및 그 참조 제거
git worktree remove ../repo-feature
```

참고:

- 커밋되지 않은 변경 사항이 있는 워크트리는 강제로 명령을 내리지 않는 한 `git worktree remove`를 통해 제거할 수 없습니다.
- 워크트리를 제거한다고 해서 브랜치가 자동으로 삭제되는 것은 **아닙니다**. 일반적인 `git branch` 명령을 사용하여 브랜치를 삭제하거나 유지할 수 있습니다.
- `~/.hermes/checkpoints/` 밑에 위치한 Hermes 체크포인트 데이터는 워크트리 제거 시 함께 자동으로 지워지진 않지만 보통 크기가 매우 작습니다.

## 모범 사례

- **하나의 Hermes 실험당 하나의 워크트리 사용**
  - 실질적인 변경 작업을 위해 전용 브랜치/워크트리를 생성하세요.
  - 이렇게 하면 diff가 집중되고 PR(Pull Request)이 작아져 리뷰하기 쉬워집니다.
- **실험의 이름을 따서 브랜치 이름 지정하기**
  - 예: `feature/hermes-checkpoints-docs`, `feature/hermes-refactor-tests`.
- **자주 커밋하기**
  - 중요도 높은 마일스톤은 git 커밋을 사용하세요.
  - 이 과정 중 발생하는 툴 중심(tool-driven) 수정본은 안전망인 [체크포인트와 `/rollback`](./checkpoints-and-rollback.md)에 의존하는 것이 좋습니다.
- **워크트리 사용 시 기본 저장소 루트에서 Hermes를 실행하지 않기**
  - 각 에이전트의 범위가 명확해지도록 가급적이면 워크트리 디렉토리 내에서 실행하세요.

## `hermes -w` 사용하기 (자동 워크트리 모드)

Hermes에는 고유한 브랜치를 가진 **일회용 git 워크트리를 자동으로 생성**하는 내장 `-w` 플래그가 있습니다. 워크트리를 수동으로 설정할 필요 없이, 해당 저장소로 이동(`cd`)하여 다음 명령을 실행하면 됩니다:

```bash
cd /path/to/your/repo
hermes -w
```

Hermes는 다음을 수행합니다:

- 저장소 내 `.worktrees/` 아래에 임시 워크트리를 생성합니다.
- 격리된 브랜치(예: `hermes/hermes-<hash>`)를 체크아웃합니다.
- 해당 워크트리 내에서 전체 CLI 세션을 실행합니다.

이것은 워크트리를 격리시키는 가장 쉬운 방법입니다. 이 명령어를 단일 질의와 결합할 수도 있습니다:

```bash
hermes -w -q "Fix issue #123"
```

여러 터미널을 열어 각각 `hermes -w` 명령을 수행함으로써 병렬로 에이전트를 실행할 수 있으며, 이 경우 모든 실행이 고유한 워크트리와 브랜치를 자동으로 갖게 됩니다.

## 요약

- **git 워크트리**를 사용하여 각 Hermes 세션에 깨끗하고 독립된 체크아웃을 제공합니다.
- **브랜치**를 활용하여 실험 기록의 하이레벨(high-level) 변화를 파악합니다.
- 각 워크트리 내부의 실수를 되돌릴 때 **체크포인트 + `/rollback`**을 사용합니다.

이러한 조합을 통해 다음과 같은 이점을 얻을 수 있습니다:

- 서로 다른 에이전트와 실험이 충돌하지 않는다는 강력한 보장
- 잘못된 수정으로부터 쉽게 복구 가능한 빠른 반복(iteration) 주기
- 깔끔하고 리뷰하기 쉬운 Pull Request
