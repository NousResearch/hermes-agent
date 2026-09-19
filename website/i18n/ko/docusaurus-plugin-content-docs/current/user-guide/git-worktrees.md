---
sidebar_position: 3
sidebar_label: "Git 워크트리"
title: "Git 워크트리"
description: "git 워크트리와 격리된 체크아웃을 사용해 동일한 저장소에서 여러 Hermes 에이전트를 안전하게 실행합니다"
---

# Git 워크트리

Hermes Agent는 규모가 크고 오랫동안 유지되는 저장소에서 자주 사용됩니다. 다음과 같은 경우:

- 동일한 프로젝트에서 **여러 에이전트를 병렬로 실행**하거나
- 실험적인 리팩터링을 기본 브랜치와 격리해 유지하려는 경우,

Git **워크트리**는 전체 저장소를 복제하지 않고 각 에이전트에 자체 체크아웃을 제공하는 가장 안전한 방법입니다.

이 페이지에서는 워크트리와 Hermes를 결합해 각 세션이 깨끗하고 격리된 작업 디렉터리를 갖도록 하는 방법을 설명합니다.

## Hermes에서 워크트리를 사용하는 이유

Hermes는 **현재 작업 디렉터리**를 프로젝트 루트로 취급합니다.

- CLI: `hermes` 또는 `hermes chat`을 실행하는 디렉터리
- 메시징 게이트웨이: `~/.hermes/config.yaml`의 `terminal.cwd`에 설정된 디렉터리

동일한 체크아웃에서 여러 에이전트를 실행하면 서로의 변경 사항이 간섭할 수 있습니다.

- 한 에이전트가 다른 에이전트가 사용 중인 파일을 삭제하거나 다시 작성할 수 있습니다.
- 어떤 변경 사항이 어떤 실험에 속하는지 파악하기 어려워집니다.

워크트리를 사용하면 각 에이전트가 다음을 갖습니다.

- **자체 브랜치와 작업 디렉터리**
- `/rollback`을 위한 **자체 Checkpoint Manager 기록**

참조: [체크포인트와 /rollback](./checkpoints-and-rollback.md).

## 빠른 시작: 워크트리 만들기

기능 브랜치를 위한 새 워크트리를 만들려면 기본 저장소(`.git/` 포함)에서 다음을 실행합니다.

```bash
# From the main repo root
cd /path/to/your/repo

# Create a new branch and worktree in ../repo-feature
git worktree add ../repo-feature feature/hermes-experiment
```

다음이 생성됩니다.

- 새 디렉터리: `../repo-feature`
- 해당 디렉터리에서 체크아웃된 새 브랜치: `feature/hermes-experiment`

이제 새 워크트리로 이동해 그곳에서 Hermes를 실행할 수 있습니다.

```bash
cd ../repo-feature

# Start Hermes in the worktree
hermes
```

Hermes는 다음과 같이 동작합니다.

- `../repo-feature`를 프로젝트 루트로 인식합니다.
- 컨텍스트 파일, 코드 편집, 도구에 해당 디렉터리를 사용합니다.
- 이 워크트리에 한정된 **별도의 체크포인트 기록**을 `/rollback`에 사용합니다.

## 여러 에이전트를 병렬로 실행하기

각각 자체 브랜치를 가진 워크트리를 여러 개 만들 수 있습니다.

```bash
cd /path/to/your/repo

git worktree add ../repo-experiment-a feature/hermes-a
git worktree add ../repo-experiment-b feature/hermes-b
```

별도의 터미널에서 다음을 실행합니다.

```bash
# Terminal 1
cd ../repo-experiment-a
hermes

# Terminal 2
cd ../repo-experiment-b
hermes
```

각 Hermes 프로세스는 다음과 같습니다.

- 자체 브랜치(`feature/hermes-a`와 `feature/hermes-b`)에서 작업합니다.
- 서로 다른 shadow repo 해시(워크트리 경로에서 파생됨)에 체크포인트를 기록합니다.
- 다른 워크트리에 영향을 주지 않고 `/rollback`을 독립적으로 사용할 수 있습니다.

다음과 같은 경우에 특히 유용합니다.

- 일괄 리팩터링을 실행할 때
- 동일한 업스트림 저장소에 대해 여러 접근 방식을 시도할 때
- 동일한 업스트림 저장소에 CLI 세션과 게이트웨이 세션을 함께 사용할 때

## 워크트리를 안전하게 정리하기

실험이 끝나면 다음을 수행합니다.

1. 작업을 유지할지 폐기할지 결정합니다.
2. 유지하려면:
   - 평소처럼 브랜치를 기본 브랜치에 병합합니다.
3. 워크트리를 제거합니다.

```bash
cd /path/to/your/repo

# Remove the worktree directory and its reference
git worktree remove ../repo-feature
```

참고:

- 커밋되지 않은 변경 사항이 있는 워크트리는 강제하지 않는 한 `git worktree remove`로 제거할 수 없습니다.
- 워크트리를 제거해도 브랜치는 자동으로 삭제되지 않습니다. 일반적인 `git branch` 명령으로 브랜치를 삭제하거나 유지할 수 있습니다.
- 워크트리를 제거해도 `~/.hermes/checkpoints/`의 Hermes 체크포인트 데이터는 자동으로 정리되지 않지만, 대개 매우 작습니다.

## 모범 사례

- **Hermes 실험당 워크트리 하나**
  - 규모가 큰 변경마다 전용 브랜치/워크트리를 만듭니다.
  - 이렇게 하면 변경 사항이 집중되고 PR을 작고 검토하기 쉽게 유지할 수 있습니다.
- **실험 이름으로 브랜치 이름 지정**
  - 예: `feature/hermes-checkpoints-docs`, `feature/hermes-refactor-tests`.
- **자주 커밋하기**
  - 높은 수준의 마일스톤마다 Git 커밋을 사용합니다.
  - 그 사이 도구로 변경한 내용을 보호하려면 [체크포인트와 /rollback](./checkpoints-and-rollback.md)을 안전망으로 사용합니다.
- **워크트리를 사용할 때 bare 저장소 루트에서 Hermes를 실행하지 않기**
  - 각 에이전트의 범위를 명확히 할 수 있도록 워크트리 디렉터리에서 실행하는 것을 권장합니다.

## `hermes -w` 사용하기(자동 워크트리 모드)

Hermes에는 자체 브랜치가 있는 임시 Git 워크트리를 자동으로 생성하는 내장 `-w` 플래그가 있습니다. 워크트리를 수동으로 설정할 필요 없이 저장소로 이동해 다음을 실행하면 됩니다.

```bash
cd /path/to/your/repo
hermes -w
```

Hermes는 다음을 수행합니다.

- 저장소 내부의 `.worktrees/` 아래에 임시 워크트리를 만듭니다.
- 격리된 브랜치(예: `hermes/hermes-<hash>`)를 체크아웃합니다.
- 전체 CLI 세션을 해당 워크트리 안에서 실행합니다.

단일 쿼리와 함께 사용할 수도 있습니다.

```bash
hermes -w -z "Fix issue #123"
```

병렬 에이전트의 경우 여러 터미널을 열고 각 터미널에서 `hermes -w`를 실행하면 됩니다. 모든 호출은 자동으로 자체 워크트리를 생성합니다.

## 모두 연결하기

- Git **워크트리**를 사용해 각 Hermes 세션에 자체적으로 깨끗한 작업 공간을 제공합니다.
- **브랜치**를 사용해 실험의 높은 수준의 기록을 보존합니다.
- **체크포인트 + `/rollback`**을 사용해 각 워크트리 안에서 실수를 복구합니다.

이 조합은 다음을 제공합니다.

- 서로 다른 에이전트가 서로의 작업을 방해하지 않는다는 강력한 보장
- 잘못된 변경 사항에서 빠르게 복구할 수 있는 반복 작업
- 깔끔하고 검토하기 쉬운 PR

## 워크트리에서 UI 화면 개발하기

TypeScript 화면(`ui-tui/`, `apps/desktop/`)에는 각각 `node_modules`가 필요하므로, 새 워크트리마다 `npm ci`를 실행하면 모든 브랜치에 설치 파일이 중복됩니다. 여러 워크트리에서 TUI 또는 데스크톱 앱을 수정하는 경우, 하나의 설치를 심볼릭 링크로 공유하는 `htui` / `hgui` 도우미에 대해서는 [워크트리에서 TUI 및 데스크톱 사용하기](../developer-guide/worktree-ui-dev.md)를 참조하세요.
