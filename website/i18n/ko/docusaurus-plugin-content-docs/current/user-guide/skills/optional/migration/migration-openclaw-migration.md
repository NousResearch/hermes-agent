---
title: "Openclaw Migration — 사용자의 OpenClaw 커스터마이징 기록을 Hermes Agent로 마이그레이션"
sidebar_label: "Openclaw Migration"
description: "사용자의 OpenClaw 커스터마이징 기록을 Hermes Agent로 마이그레이션"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Openclaw Migration

사용자의 OpenClaw 커스터마이징 흔적을 Hermes Agent로 마이그레이션합니다. Hermes와 호환되는 메모, SOUL.md, 명령어 허용 목록, 사용자 스킬, 선택된 작업 공간 에셋을 ~/.openclaw에서 가져오고, 마이그레이션할 수 없는 항목과 그 이유를 정확하게 보고합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/migration/openclaw-migration` 명령어로 설치 |
| 경로 | `optional-skills/migration/openclaw-migration` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent (Nous Research) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Migration`, `OpenClaw`, `Hermes`, `Memory`, `Persona`, `Import` |
| 관련 스킬 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# OpenClaw -> Hermes 마이그레이션 (Migration)

사용자가 수동 정리를 최소화하면서 OpenClaw 설정을 Hermes Agent로 이동하려는 경우 이 스킬을 사용합니다.

## CLI 명령어

빠르고 대화형이 아닌 마이그레이션을 위해서는 내장된 CLI 명령어를 사용하세요:

```bash
hermes claw migrate              # 전체 대화형 마이그레이션
hermes claw migrate --dry-run    # 마이그레이션될 내용 미리보기
hermes claw migrate --preset user-data   # 비밀 정보(secrets) 없이 마이그레이션
hermes claw migrate --overwrite  # 기존의 충돌하는 항목 덮어쓰기
hermes claw migrate --source /custom/path/.openclaw  # 사용자 지정 소스
```

CLI 명령어는 아래에 설명된 것과 동일한 마이그레이션 스크립트를 실행합니다. 테스트 런(dry-run) 미리보기와 항목별 충돌 해결을 통한 대화형, 가이드 기반 마이그레이션을 원할 때 에이전트를 통해 이 스킬을 사용하세요.

**최초 설정:** `hermes setup` 마법사는 자동으로 `~/.openclaw`를 감지하고 구성이 시작되기 전에 마이그레이션을 제안합니다.

## 이 스킬의 기능

이 스킬은 `scripts/openclaw_to_hermes.py`를 사용하여 다음 작업을 수행합니다:

- `SOUL.md`를 Hermes 홈 디렉토리에 `SOUL.md`로 가져옵니다.
- OpenClaw `MEMORY.md` 및 `USER.md`를 Hermes 메모리 항목으로 변환합니다.
- OpenClaw 명령어 승인 패턴을 Hermes `command_allowlist`에 병합합니다.
- `TELEGRAM_ALLOWED_USERS`와 같은 Hermes 호환 메시징 설정을 마이그레이션하고, OpenClaw 작업 공간 설정을 Hermes 작업 디렉토리 구성에 매핑합니다.
- OpenClaw 스킬을 `~/.hermes/skills/openclaw-imports/`에 복사합니다.
- 선택적으로 OpenClaw 작업 공간 지시 파일을 선택한 Hermes 작업 공간으로 복사합니다.
- `workspace/tts/`와 같은 호환되는 작업 공간 에셋을 `~/.hermes/tts/`에 미러링합니다.
- 직접적인 Hermes 목적지가 없는 기밀(non-secret) 문서를 보관합니다.
- 마이그레이션된 항목, 충돌, 건너뛴 항목, 원인을 나열하는 구조화된 보고서를 생성합니다.

## 경로 확인

도우미 스크립트는 이 스킬 디렉토리 내에 있습니다:

- `scripts/openclaw_to_hermes.py`

이 스킬이 스킬 허브에서 설치된 경우 일반적인 위치는 다음과 같습니다:

- `~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py`

`~/.hermes/skills/openclaw-migration/...`와 같이 더 짧은 경로를 임의로 추측하지 마세요.

도우미를 실행하기 전에:

1. `~/.hermes/skills/migration/openclaw-migration/` 아래에 설치된 경로를 선호합니다.
2. 해당 경로가 실패하면 설치된 스킬 디렉토리를 검사하고 설치된 `SKILL.md`를 기준으로 스크립트 경로를 찾으세요.
3. 설치된 위치가 없거나 스킬을 수동으로 이동한 경우에만 예비책(fallback)으로 `find`를 사용하세요.
4. 터미널 도구를 호출할 때 `workdir: "~"`를 전달하지 마세요. 사용자의 홈 디렉토리와 같은 절대 디렉토리를 사용하거나 `workdir`를 완전히 생략하세요.

`--migrate-secrets`와 함께 실행하면, Hermes와 호환되는 소규모의 허용된 기밀 정보(secrets) 세트도 가져옵니다 (현재 다음과 같습니다):

- `TELEGRAM_BOT_TOKEN`

## 기본 워크플로우

1. 테스트 런(dry-run)으로 먼저 검사합니다.
2. 마이그레이션할 수 있는 항목, 없는 항목, 보관될 항목의 간단한 요약을 제시합니다.
3. `clarify` 도구를 사용할 수 있는 경우, 사용자에게 자유 형식 산문형 답변을 요구하는 대신 이를 통해 사용자의 결정을 받으세요.
4. 테스트 런에서 가져온 스킬 디렉토리 충돌이 발견되면 실행 전에 해당 문제를 어떻게 처리할지 묻습니다.
5. 실행 전 사용자에게 두 가지 지원되는 마이그레이션 모드 중 하나를 선택하도록 요청합니다.
6. 사용자가 작업 공간 지시 파일을 가져오길 원할 때만 대상 작업 공간 경로를 묻습니다.
7. 일치하는 프리셋과 플래그를 사용하여 마이그레이션을 실행합니다.
8. 결과를 요약합니다, 특히:
   - 마이그레이션된 항목
   - 수동 검토를 위해 보관된 항목
   - 건너뛴 항목 및 그 이유

## 사용자 상호작용 프로토콜

Hermes CLI는 대화형 프롬프트를 위한 `clarify` 도구를 지원하지만 다음과 같은 제한이 있습니다:

- 한 번에 한 가지 선택만 가능
- 최대 4개의 사전 정의된 선택지
- 자동 `기타(Other)` 자유 텍스트 옵션

단일 프롬프트에서 진정한 다중 선택 체크박스는 **지원하지 않습니다**.

모든 `clarify` 호출에 대해:

- 항상 비어있지 않은 `question`을 포함하세요.
- 실제 선택할 수 있는 프롬프트에만 `choices`를 포함하세요.
- `choices`는 2~4개의 일반 문자열 옵션으로 유지하세요.
- `...`와 같은 자리 표시자 또는 잘린 옵션을 내보내지 마세요.
- 선택지에 여분의 공백을 넣어 늘리거나 장식하지 마세요.
- 질문에 `여기에 디렉토리를 입력하세요`, 빈 줄 채우기, 또는 `_____`와 같은 가짜 폼 필드를 포함하지 마세요.
- 개방형 경로 질문의 경우 일반 문장으로만 질문하세요; 사용자는 패널 아래의 일반 CLI 프롬프트에 입력합니다.

`clarify` 호출에서 오류가 반환되면 오류 텍스트를 검사하고 페이로드를 수정한 다음 올바른 `question`과 정리된 선택지로 다시 한 번 시도하세요.

`clarify`를 사용할 수 있고 테스트 런(dry-run)에서 사용자 결정이 필요한 사항이 발견되면 **다음 액션은 반드시 `clarify` 도구 호출**이어야 합니다.
다지와 같은 일반 어시스턴트 메시지로 턴을 끝내지 마세요:

- "선택 사항을 제시하겠습니다"
- "어떻게 하시겠습니까?"
- "여기에 옵션이 있습니다"

사용자 결정이 필요한 경우 더 긴 문장을 생성하기 전에 `clarify`를 통해 수집하세요.
해결되지 않은 결정이 여러 개 남아 있는 경우, 그 사이에 설명 어시스턴트 메시지를 삽입하지 마세요. 하나의 `clarify` 응답을 받으면 일반적으로 다음 액션은 다음으로 필요한 `clarify` 호출이어야 합니다.

테스트 런이 다음과 같이 보고할 때마다 항목을 해결되지 않은 결정으로 취급하여 `workspace-agents`에 대해 물어보아야 합니다:

- `kind="workspace-agents"`
- `status="skipped"`
- `No workspace target was provided` 가 포함된 이유

이 경우 실행하기 전에 작업 공간 지시에 대해 반드시 질문해야 합니다. 조용히 이를 건너뛰는 것으로 간주하지 마세요.

이러한 제한 때문에 다음과 같이 단순화된 결정 흐름을 사용하세요:

1. `SOUL.md` 충돌의 경우 다음과 같은 선택지와 함께 `clarify`를 사용하세요:
   - `keep existing` (기존 유지)
   - `overwrite with backup` (백업 후 덮어쓰기)
   - `review first` (먼저 검토)
2. 테스트 런에서 `status="conflict"`인 `kind="skill"` 항목이 하나 이상 표시되면 다음과 같은 선택지와 함께 `clarify`를 사용하세요:
   - `keep existing skills` (기존 스킬 유지)
   - `overwrite conflicting skills with backup` (충돌하는 스킬을 백업 후 덮어쓰기)
   - `import conflicting skills under renamed folders` (충돌하는 스킬을 이름이 변경된 폴더에 가져오기)
3. 작업 공간 지시 사항에 대해 다음과 같은 선택지와 함께 `clarify`를 사용하세요:
   - `skip workspace instructions` (작업 공간 지시 건너뛰기)
   - `copy to a workspace path` (작업 공간 경로로 복사)
   - `decide later` (나중에 결정)
4. 사용자가 작업 공간 지시를 복사하도록 선택하면 **절대 경로**를 요청하는 후속 개방형 `clarify` 질문을 하세요.
5. 사용자가 `skip workspace instructions` 또는 `decide later`를 선택하면 `--workspace-target` 없이 진행합니다.
5. 마이그레이션 모드의 경우 다음 3가지 선택지와 함께 `clarify`를 사용하세요:
   - `user-data only` (사용자 데이터만)
   - `full compatible migration` (전체 호환 마이그레이션)
   - `cancel` (취소)
6. `user-data only` 의미: 사용자 데이터 및 호환 구성은 마이그레이션하지만, 허용 목록의 비밀 정보(secrets)는 가져오지 **않습니다**.
7. `full compatible migration` 의미: 허용된 비밀 정보(있는 경우)를 포함하여 동일한 호환 사용자 데이터를 마이그레이션합니다.
8. `clarify`를 사용할 수 없는 경우 일반 텍스트로 동일한 질문을 하되, 답변을 `user-data only`, `full compatible migration`, 또는 `cancel`로 제한하세요.

실행 게이트 (Execution gate):

- `No workspace target was provided`로 인한 `workspace-agents` 건너뛰기 상태가 해결되지 않은 동안 실행하지 마세요.
- 이를 해결할 수 있는 유일하고 유효한 방법은 다음과 같습니다:
  - 사용자가 명시적으로 `skip workspace instructions`을 선택
  - 사용자가 명시적으로 `decide later`를 선택
  - 사용자가 `copy to a workspace path`를 선택한 후 작업 공간 경로를 제공
- 테스트 런에 작업 공간 목표가 없는 것 자체가 실행을 허가하는 것은 아닙니다.
- 필수 `clarify` 결정이 미해결 상태인 동안에는 실행하지 마세요.

기본 패턴으로 이러한 정확한 `clarify` 페이로드 모양을 사용하세요:

- `{"question":"Your existing SOUL.md conflicts with the imported one. What should I do?","choices":["keep existing","overwrite with backup","review first"]}`
- `{"question":"One or more imported OpenClaw skills already exist in Hermes. How should I handle those skill conflicts?","choices":["keep existing skills","overwrite conflicting skills with backup","import conflicting skills under renamed folders"]}`
- `{"question":"Choose migration mode: migrate only user data, or run the full compatible migration including allowlisted secrets?","choices":["user-data only","full compatible migration","cancel"]}`
- `{"question":"Do you want to copy the OpenClaw workspace instructions file into a Hermes workspace?","choices":["skip workspace instructions","copy to a workspace path","decide later"]}`
- `{"question":"Please provide an absolute path where the workspace instructions should be copied."}`

## 의사결정과 명령어(Decision-to-command) 매핑

사용자 결정을 명령어 플래그에 정확히 매핑하세요:

- 사용자가 `SOUL.md`에 대해 `keep existing`을 선택하면 `--overwrite`를 추가하지 **않습니다**.
- 사용자가 `overwrite with backup`을 선택하면 `--overwrite`를 추가합니다.
- 사용자가 `review first`를 선택하면 실행 전에 중지하고 관련 파일을 검토합니다.
- 사용자가 `keep existing skills`를 선택하면 `--skill-conflict skip`을 추가합니다.
- 사용자가 `overwrite conflicting skills with backup`을 선택하면 `--skill-conflict overwrite`를 추가합니다.
- 사용자가 `import conflicting skills under renamed folders`를 선택하면 `--skill-conflict rename`을 추가합니다.
- 사용자가 `user-data only`를 선택하면 `--preset user-data`로 실행하고 `--migrate-secrets`를 추가하지 **않습니다**.
- 사용자가 `full compatible migration`을 선택하면 `--preset full --migrate-secrets`로 실행합니다.
- 사용자가 절대 작업 공간 경로를 명시적으로 제공한 경우에만 `--workspace-target`을 추가하세요.
- 사용자가 `skip workspace instructions` 또는 `decide later`를 선택하면 `--workspace-target`을 추가하지 마세요.

실행 전 계획된 정확한 명령어를 평문으로 다시 언급하고 사용자의 선택과 일치하는지 확인하세요.

## 실행 후 보고 규칙

실행 후에는 스크립트의 JSON 출력을 진실(source of truth)로 취급하세요.

1. 모든 집계 기준은 `report.summary`를 기반으로 합니다.
2. 상태(`status`)가 정확히 `migrated`인 항목만 "Successfully Migrated" 항목 아래에 나열하세요.
3. 보고서에 해당 항목이 `migrated`로 표시되지 않는 한, 충돌이 해결되었다고 주장하지 마세요.
4. `kind="soul"`인 항목의 상태가 `status="migrated"`인 경우가 아니라면 `SOUL.md`를 덮어썼다고 말하지 마세요.
5. `report.summary.conflict > 0`인 경우 성공을 조용히 암시하는 대신 충돌 섹션을 포함하세요.
6. 개수와 나열된 항목이 일치하지 않는 경우, 응답하기 전에 보고서와 일치하도록 목록을 수정하세요.
7. 사용자가 `report.json`, `summary.md`, 백업, 보관된 파일을 검사할 수 있도록 보고서의 `output_dir` 경로를 가능한 경우 포함하세요.
8. 메모리 또는 사용자 프로필 초과의 경우 보고서가 보관 경로를 명시적으로 보여주지 않는 한, 항목이 보관되었다고 말하지 마세요. `details.overflow_file`이 존재하는 경우 전체 초과 목록이 그곳으로 내보내졌다고 언급하세요.
9. 이름이 변경된 폴더로 스킬을 가져온 경우 최종 목적지를 보고하고 `details.renamed_from`을 언급하세요.
10. `report.skill_conflict_mode`가 존재하면 선택된 스킬 가져오기 충돌 정책의 기준으로 사용하세요.
11. 항목이 `status="skipped"`인 경우 이를 덮어썼거나 백업했거나, 마이그레이션했거나 해결했다고 설명하지 마세요.
12. `Target already matches source`라는 이유로 `kind="soul"`의 상태가 `status="skipped"`인 경우, 변경되지 않은 상태로 두었다고 말하고 백업에 대해 언급하지 마세요.
13. 이름이 변경된 가져온 스킬의 `details.backup`이 비어 있는 경우, 기존 Hermes 스킬의 이름이 바뀌었거나 백업되었다고 암시하지 마세요. 단지 가져온 사본이 새 목적지에 배치되었다고 말하고 원래 위치에 그대로 있는 기존 폴더를 `details.renamed_from`으로 참조하세요.

## 마이그레이션 프리셋

일반적인 사용에는 이 두 가지 프리셋을 선호합니다:

- `user-data`
- `full`

`user-data`에 포함되는 것:

- `soul`
- `workspace-agents`
- `memory`
- `user-profile`
- `messaging-settings`
- `command-allowlist`
- `skills`
- `tts-assets`
- `archive`

`full`은 `user-data`의 모든 내용을 포함하며 추가적으로 다음을 포함합니다:

- `secret-settings`

도우미 스크립트는 여전히 범주 수준의 `--include` / `--exclude`를 지원하지만, 이를 기본 UX라기보다는 고급 예비책으로 취급하세요.

## 명령어

전체 검색 기능이 포함된 테스트 런(Dry run):

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py
```

터미널 도구를 사용할 때는 다음과 같은 절대 호출 패턴을 선호합니다:

```json
{"command":"python3 /home/USER/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py","workdir":"/home/USER"}
```

user-data 프리셋을 사용한 테스트 런:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --preset user-data
```

user-data 마이그레이션 실행:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset user-data --skill-conflict skip
```

전체 호환 마이그레이션 실행:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset full --migrate-secrets --skill-conflict skip
```

작업 공간 지시를 포함하여 실행:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset user-data --skill-conflict rename --workspace-target "/absolute/workspace/path"
```

기본적으로 작업 공간 대상으로 `$PWD` 또는 홈 디렉토리를 사용하지 마세요. 먼저 명시적인 작업 공간 경로를 요청하세요.

## 중요 규칙

1. 사용자가 즉시 진행하라고 명시적으로 말하지 않는 한 작성하기 전에 테스트 런(dry-run)을 실행합니다.
2. 기본적으로 비밀 정보(secrets)는 마이그레이션하지 마세요. 사용자가 명시적으로 비밀 정보 마이그레이션을 요청하지 않는 한, 토큰, 인증 blob, 디바이스 자격 증명 및 원시 게이트웨이 구성은 Hermes 외부 상태로 두어야 합니다.
3. 사용자가 명시적으로 원하지 않는 한, 내용이 있는 대상 Hermes를 조용히 덮어쓰지 마세요. 덮어쓰기가 활성화된 경우 도우미 스크립트가 백업을 보존합니다.
4. 항상 사용자에게 건너뛴 항목에 대한 보고서를 제공하세요. 해당 보고서는 선택 사항이 아니라 마이그레이션의 일부입니다.
5. 기본 작업 공간(`workspace.default/`)보다 기본 OpenClaw 작업 공간(`~/.openclaw/workspace/`)을 선호하세요. 기본 파일이 없을 때만 예비책으로 기본 작업 공간을 사용하세요.
6. 비밀 정보 마이그레이션 모드에서도 비어있는 Hermes 목적지가 있는 비밀 정보만 마이그레이션하세요. 지원되지 않는 인증 blob은 여전히 건너뛴 것으로 보고해야 합니다.
7. 테스트 런에서 대규모 자산 복사, 충돌하는 `SOUL.md`, 또는 한도를 초과한 메모리 항목을 보여주는 경우 실행 전에 별도로 언급하세요.
8. 사용자가 확실하지 않을 때는 `user-data only`를 기본으로 설정합니다.
9. 사용자가 대상 작업 공간 경로를 명시적으로 제공한 경우에만 `workspace-agents`를 포함하세요.
10. 범주 수준 `--include` / `--exclude`를 일반적인 흐름이 아닌 고급 탈출 해치(escape hatch)로 취급하세요.
11. `clarify`를 사용할 수 있는 경우, 테스트 런 요약 후 "무엇을 하시겠습니까?"와 같은 모호한 말로 끝맺지 마세요. 대신 구조화된 후속 프롬프트를 사용하세요.
12. 실제 선택 프롬프트가 작동할 수 있을 때 개방형 `clarify` 프롬프트를 사용하지 마세요. 먼저 선택 가능한 옵션을 선호하고, 절대 경로나 파일 검토 요청의 경우에만 자유 텍스트를 사용하세요.
13. 테스트 런 이후 미해결 결정이 남아 있다면 요약 후 절대 멈추지 마세요. 가장 우선순위가 높은 차단 결정을 내리기 위해 즉시 `clarify`를 사용하세요.
14. 후속 질문의 우선순위:
    - `SOUL.md` 충돌
    - 가져온 스킬 충돌
    - 마이그레이션 모드
    - 작업 공간 지시 목적지
15. 같은 메시지 안에서 나중에 선택지를 제시하겠다고 약속하지 마세요. 실제로 `clarify`를 호출하여 제시하세요.
16. 마이그레이션 모드 답변 후 `workspace-agents` 문제가 아직 해결되지 않았는지 명시적으로 확인하세요. 만약 남아있다면 다음 작업은 반드시 작업 공간 지시 `clarify` 호출이어야 합니다.
17. 어느 `clarify` 답변 이후 다른 필요한 결정이 여전히 남아있을 경우 방금 결정된 사항에 대해 설명하지 마세요. 즉시 다음 필요한 질문을 물어보세요.

## 예상 결과

성공적으로 실행한 후 사용자는 다음을 얻게 됩니다:

- 가져온 Hermes 페르소나(persona) 상태
- 변환된 OpenClaw 지식으로 채워진 Hermes 메모리 파일
- `~/.hermes/skills/openclaw-imports/` 아래에서 사용 가능한 OpenClaw 스킬
- 충돌, 누락 또는 지원되지 않는 데이터가 표시된 마이그레이션 보고서
