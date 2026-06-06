---
sidebar_position: 10
title: 프로필 관리 (Profile Management)
description: Hermes 프로필의 생성, 전환, 공유 및 구성 격리.
---

# 프로필 (Profiles)

프로필을 사용하면 동일한 컴퓨터에서 각각 자체 구성, 세션 데이터, 스킬 및 `HERMES_HOME` 디렉터리를 갖는 여러 개의 독립적인 Hermes 에이전트를 실행할 수 있습니다. 

## 프로필이란?

기본적으로(기본 프로필), Hermes의 모든 것은 `~/.hermes` 디렉터리에 저장됩니다. 그러나 "dev", "work", "research"와 같은 다른 이름의 프로필을 생성하면 `~/.hermes-dev`, `~/.hermes-work` 등에 분리된 환경이 생성됩니다. 

각 프로필은 다음을 고유하게 갖습니다:
- **`config.yaml`**: 모델, 표시 옵션, 도구 설정.
- **`.env`**: API 키 (각 프로필이 동일한 키를 가져야 하는 것은 아님).
- **`MEMORY.md` & `USER.md`**: 내장된 기억과 정체성 파일.
- **`sessions/`**: 과거 대화 내역.
- **`skills/`**: 설치된 스킬 세트 (한 프로필의 스킬이 다른 프로필에 노출되지 않음).

## CLI 플래그: `--profile`

Hermes의 모든 명령어에는 프로필을 지정하는 `-p` 또는 `--profile` 플래그를 사용할 수 있습니다. 이를 통해 활성 프로필을 전환하지 않고도 특정 프로필에 대해 일회성 명령어(`chat`, `gateway`, `update` 등)를 실행할 수 있습니다.

```bash
hermes -p work chat -q "What's the status of the Jira ticket?"
hermes -p dev gateway start
hermes -p research config show
```

이 플래그가 제공되지 않으면 Hermes는 현재 고정된(sticky) 활성 프로필을 사용합니다(기본값은 `default`).

---

## 명령어 참조 (Command Reference)

모든 프로필 관리는 `hermes profile` 명령어를 통해 수행됩니다.

### `hermes profile list`

사용 가능한 모든 프로필을 나열하고 어떤 프로필이 활성화되어 있는지 표시합니다.

### `hermes profile use <name>`

기본 프로필을 지정한 프로필로 변경합니다. 이 프로필은 이후에 `-p`를 생략할 때 고정된 기본값으로 유지됩니다. 내부적으로 이 명령어는 `~/.hermes/active_profile` 심볼릭 링크를 업데이트합니다.

### `hermes profile create <name>`

처음부터 완전히 새로운 프로필을 생성합니다. 기본 프로필 생성 시와 마찬가지로 새로운 `.env` 프롬프트와 초기 설정 단계를 밟게 됩니다.

#### 옵션:

- `--clone`: **현재 활성 프로필**의 `config.yaml`, `.env`, `SOUL.md`를 새 프로필로 복사합니다. 대화나 기억은 복사되지 않으므로 키와 프롬프트를 공유하는 새로운 인스턴스를 시작할 때 유용합니다.
- `--clone-all`: 과거 세션, 기억 파일(`MEMORY.md`), 스킬을 포함한 모든 데이터를 현재 활성 프로필에서 복사합니다.
- `--clone-from <source>`: 현재 활성 프로필 대신 지정된 프로필에서 복사합니다 (예: `hermes profile create dev --clone-all --clone-from work`).
- `--no-alias`: `create`는 일반적으로 셸 시작 스크립트에 바로가기(예: `hermes-dev`)를 생성하도록 안내합니다. 이 옵션은 해당 마법사를 조용히 건너뜁니다.

### `hermes profile delete <name>`

프로필과 해당 홈 디렉터리(`~/.hermes-<name>`)를 완전히 삭제합니다. 이 작업은 되돌릴 수 없습니다.

### `hermes profile alias <name>`

해당 프로필을 래핑하는 편리한 셸 별칭 스크립트(예: `h-work`)를 관리합니다. 이렇게 하면 활성 프로필을 변경하거나 `-p work`를 입력하지 않고도 해당 프로필과 상호작용할 수 있습니다. 예를 들어, `h-work chat`은 `hermes -p work chat`과 같습니다.

- `--name <ALIAS>`: 자동 생성된 기본값(보통 `hermes-<name>`) 대신 특정 별칭 이름을 지정합니다.
- `--remove`: 기존에 생성된 별칭 스크립트를 삭제합니다.

### `hermes profile rename <old> <new>`

프로필의 이름을 변경하고 홈 디렉터리를 새 위치로 이동시킵니다. 별칭이 생성되어 있었다면 새 이름을 가리키도록 업데이트해야 할 수 있습니다.

### `hermes profile export <name>`

전체 프로필 디렉터리를 백업용 또는 공유용 단일 `.tar.gz` 아카이브로 압축합니다.

- `-o, --output <FILE>`: 아카이브를 저장할 위치를 지정합니다 (기본값은 현재 디렉터리의 `<name>-profile.tar.gz`).

### `hermes profile import <archive>`

내보낸 프로필 아카이브를 복원합니다.

- `--name <NAME>`: 가져올 프로필의 이름을 지정합니다 (기본값은 파일 이름에서 추출됨). 이 이름의 프로필이 이미 존재하면 파일이 덮어쓰입니다.

### `hermes profile install <source>`

주목할 만한 배포판 프로필 — 원격 Git 리포지토리 또는 커스텀 스킬, 프롬프트, 도구를 갖춘 선별된 설정으로 Hermes가 사전 구성된 로컬 폴더 — 을 설치합니다. 

```bash
hermes profile install github.com/user/coding-agent --alias
```

- `--name <NAME>`: 프로필을 저장할 로컬 이름 (기본값은 리포지토리 이름).
- `--alias`: 설치 중에 별칭을 설정하도록 안내합니다.
- `--force`: 덮어쓰기를 허용합니다.

자세한 내용은 배포판 문서를 참조하세요 (추가 예정).

### `hermes profile update <name>`

`install` 명령어로 리포지토리에서 설치된 프로필의 경우, 최신 변경 사항을 가져옵니다(pull). 사용자 데이터(`.env`, 세션, 기억 등)는 보존하면서 코어 설정 파일(명시적인 `--force-config`가 사용되지 않는 한 병합됨) 및 스킬을 업데이트합니다.

---

## 팁 & 트릭

1. **테스트를 위한 격리:** 코어 설정, 스킬 덮어쓰기, 위험한 권한을 테스트할 때는 기본 설정을 망가뜨리지 않도록 `--clone`을 사용하여 임시 프로필을 만드세요. 테스트가 완료되면 프로필을 삭제하세요.
2. **별칭을 사용한 게이트웨이 여러 개 실행:** 별칭을 생성하면 여러 게이트웨이를 동시에 백그라운드에서 쉽게 실행할 수 있습니다. 한 터미널에서는 `h-work gateway start`를 실행하고 다른 터미널에서는 `h-personal gateway start`를 실행할 수 있습니다. 각 봇은 고유한 PID 파일을 추적하며 포트 충돌(해당되는 경우)을 알아서 처리합니다.
3. **독립된 백업:** 프로필 내보내기를 사용하면 여러 시스템에 걸쳐 개별 에이전트를 안전하게 이동시킬 수 있습니다.
