---
title: "Openclaw 마이그레이션 — OpenClaw 설정(메모리, 스킬)을 Hermes로 가져오기"
sidebar_label: "Openclaw 마이그레이션"
description: "OpenClaw 설정(메모리, 스킬)을 Hermes로 가져오기"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Openclaw 마이그레이션

OpenClaw 설정(메모리, 스킬)을 Hermes로 가져옵니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/migration/openclaw-migration`으로 설치 |
| 경로 | `optional-skills/migration/openclaw-migration` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent (Nous Research) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Migration`, `OpenClaw`, `Hermes`, `Memory`, `Persona`, `Import` |
| 관련 스킬 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보는 내용입니다.
:::

# OpenClaw -> Hermes 마이그레이션

사용자가 OpenClaw 설정을 최소한의 수동 정리만 거쳐 Hermes Agent로 옮기려 할 때 이 스킬을 사용합니다.

## CLI 명령

빠른 비대화형 마이그레이션에는 내장 CLI 명령을 사용합니다.

```bash
hermes claw migrate              # Full interactive migration
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
hermes claw migrate --source /custom/path/.openclaw  # Custom source
```

CLI 명령은 아래 설명된 것과 동일한 마이그레이션 스크립트를 실행합니다. 드라이 런 미리 보기와 항목별 충돌 해결을 포함한 대화형 안내 마이그레이션이 필요하면 에이전트를 통해 이 스킬을 사용합니다.

**최초 설정:** `hermes setup` 마법사는 `~/.openclaw`를 자동으로 감지하고 설정을 시작하기 전에 마이그레이션을 제안합니다.

## 이 스킬이 하는 일

이 스킬은 `scripts/openclaw_to_hermes.py`를 사용해 다음을 수행합니다.

- `SOUL.md`를 Hermes 홈 디렉터리에 `SOUL.md`로 가져옵니다.
- OpenClaw의 `MEMORY.md`와 `USER.md`를 Hermes 메모리 항목으로 변환합니다.
- OpenClaw 명령 승인 패턴을 Hermes의 `command_allowlist`에 병합합니다.
- `TELEGRAM_ALLOWED_USERS` 같은 Hermes 호환 메시징 설정을 마이그레이션하고, OpenClaw 작업 공간 설정을 Hermes 작업 디렉터리 구성에 매핑합니다.
- OpenClaw 스킬을 `~/.hermes/skills/openclaw-imports/`로 복사합니다.
- 필요에 따라 OpenClaw 작업 공간 지침 파일을 선택한 Hermes 작업 공간으로 복사합니다.
- `workspace/tts/` 같은 호환 작업 공간 자산을 `~/.hermes/tts/`에 미러링합니다.
- Hermes에 직접 대응하는 대상이 없는 비시크릿 문서를 보관합니다.
- 마이그레이션된 항목, 충돌, 건너뛴 항목 및 그 이유를 나열하는 구조화된 보고서를 생성합니다.

## 경로 확인

도우미 스크립트는 이 스킬 디렉터리에 있습니다.

- `scripts/openclaw_to_hermes.py`

Skills Hub에서 이 스킬을 설치한 경우 일반적인 위치는 다음과 같습니다.

- `~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py`

`~/.hermes/skills/openclaw-migration/...`처럼 더 짧은 경로를 추측하지 마세요.

도우미를 실행하기 전에 다음을 따르세요.

1. `~/.hermes/skills/migration/openclaw-migration/` 아래의 설치 경로를 우선 사용합니다.
2. 해당 경로가 실패하면 설치된 스킬 디렉터리를 확인하고, 설치된 `SKILL.md`를 기준으로 스크립트의 상대 경로를 확인합니다.
3. 설치 위치가 없거나 스킬이 수동으로 이동된 경우에만 최후의 수단으로 `find`를 사용합니다.
4. 터미널 도구를 호출할 때 `workdir: "~"`을 전달하지 마세요. 사용자의 홈 디렉터리 같은 절대 경로를 사용하거나 `workdir`을 생략합니다.

`--migrate-secrets`를 사용하면 현재 다음과 같이 허용 목록에 있는 소수의 Hermes 호환 시크릿도 가져옵니다.

- `TELEGRAM_BOT_TOKEN`

## 기본 워크플로

1. 드라이 런으로 먼저 검사합니다.
2. 마이그레이션할 수 있는 항목, 마이그레이션할 수 없는 항목, 보관될 항목을 간단히 요약해 제시합니다.
3. `clarify` 도구를 사용할 수 있다면 자유 형식의 답변을 요청하는 대신 사용자 결정을 받는 데 사용합니다.
4. 드라이 런에서 가져온 스킬 디렉터리 충돌이 발견되면 실행하기 전에 처리 방법을 묻습니다.
5. 실행하기 전에 사용자가 지원되는 두 가지 마이그레이션 모드 중 하나를 선택하도록 합니다.
6. 작업 공간 지침 파일을 가져오려는 경우에만 대상 작업 공간 경로를 묻습니다.
7. 일치하는 프리셋과 플래그로 마이그레이션을 실행합니다.
8. 특히 다음 결과를 요약합니다.
   - 마이그레이션된 항목
   - 수동 검토를 위해 보관된 항목
   - 건너뛴 항목과 그 이유

## 사용자 상호작용 프로토콜

Hermes CLI는 대화형 프롬프트에 `clarify` 도구를 지원하지만 다음과 같이 제한됩니다.

- 한 번에 하나의 선택
- 미리 정의된 선택지 최대 4개
- 자동으로 제공되는 `Other` 자유 입력 옵션

한 번의 프롬프트에서 진정한 다중 선택 체크박스를 지원하지는 않습니다.

모든 `clarify` 호출에서는 다음을 따릅니다.

- 항상 비어 있지 않은 `question`을 포함합니다.
- 실제로 선택 가능한 프롬프트에만 `choices`를 포함합니다.
- `choices`는 평범한 문자열 선택지 2~4개로 유지합니다.
- `...` 같은 자리 표시자나 잘린 선택지를 절대 출력하지 않습니다.
- 선택지에 불필요한 공백을 추가하거나 꾸미지 않습니다.
- 질문에 `enter directory here`, 입력용 빈 줄, `_____` 같은 밑줄 등 가짜 양식 필드를 절대 넣지 않습니다.
- 자유 형식 경로 질문은 평범한 문장만 묻습니다. 사용자는 패널 아래의 일반 CLI 프롬프트에 입력합니다.

`clarify` 호출에서 오류가 반환되면 오류 텍스트를 확인하고, 유효한 `question`과 깔끔한 선택지로 페이로드를 수정한 뒤 한 번 재시도합니다.

`clarify`를 사용할 수 있고 드라이 런에서 사용자의 결정이 필요한 경우, **다음 작업은 반드시 `clarify` 도구 호출이어야 합니다.**
다음과 같은 일반 어시스턴트 메시지로 턴을 끝내지 마세요.

- "선택지를 제시하겠습니다"
- "어떻게 하시겠어요?"
- "선택지는 다음과 같습니다"

사용자 결정이 필요하면 먼저 일반 문장을 출력하지 말고 `clarify`를 통해 결정을 받습니다.
해결되지 않은 결정이 여러 개라면 그 사이에 설명용 어시스턴트 메시지를 넣지 마세요. 한 번의 `clarify` 응답을 받은 뒤 다음 작업은 대개 다음에 필요한 `clarify` 호출이어야 합니다.

`workspace-agents`는 드라이 런에서 다음을 보고할 때 해결되지 않은 결정으로 취급합니다.

- `kind="workspace-agents"`
- `status="skipped"`
- `No workspace target was provided`를 포함하는 이유

이 경우 작업 공간 지침에 대해 반드시 물어야 합니다. 이를 건너뛰기로 조용히 처리하지 마세요.

이 제한 때문에 다음과 같은 간소화된 결정 흐름을 사용합니다.

1. `SOUL.md` 충돌에는 다음과 같은 선택지로 `clarify`를 사용합니다.
   - `keep existing`
   - `overwrite with backup`
   - `review first`
2. 드라이 런에서 `status="conflict"`인 `kind="skill"` 항목이 하나 이상 표시되면 다음 선택지로 `clarify`를 사용합니다.
   - `keep existing skills`
   - `overwrite conflicting skills with backup`
   - `import conflicting skills under renamed folders`
3. 작업 공간 지침에는 다음 선택지로 `clarify`를 사용합니다.
   - `skip workspace instructions`
   - `copy to a workspace path`
   - `decide later`
4. 사용자가 복사를 선택하면 절대 경로를 요청하는 후속 자유 형식 `clarify` 질문을 합니다.
5. 사용자가 `skip workspace instructions` 또는 `decide later`를 선택하면 `--workspace-target` 없이 진행합니다.
5. 마이그레이션 모드에는 다음 3가지 선택지로 `clarify`를 사용합니다.
   - `user-data only`
   - `full compatible migration`
   - `cancel`
6. `user-data only`는 허용 목록에 있는 시크릿을 가져오지 않고 사용자 데이터와 호환 가능한 설정만 마이그레이션한다는 뜻입니다.
7. `full compatible migration`은 동일한 호환 사용자 데이터와 함께 존재하는 허용 목록의 시크릿도 마이그레이션한다는 뜻입니다.
8. `clarify`를 사용할 수 없다면 일반 텍스트로 같은 질문을 하되, 답변은 `user-data only`, `full compatible migration`, `cancel` 중 하나로 제한합니다.

실행 게이트:

- `No workspace target was provided`로 인해 발생한 `workspace-agents` 건너뛰기가 해결되지 않은 동안에는 실행하지 않습니다.
- 이를 해결하는 유효한 방법은 다음뿐입니다.
  - 사용자가 `skip workspace instructions`를 명시적으로 선택합니다.
  - 사용자가 `decide later`를 명시적으로 선택합니다.
  - 사용자가 `copy to a workspace path`를 선택한 뒤 작업 공간 경로를 제공합니다.
- 드라이 런에 작업 공간 대상이 없다는 사실만으로 실행을 허용한 것으로 간주하지 않습니다.
- 필요한 `clarify` 결정이 해결되지 않은 동안에는 실행하지 않습니다.

기본 패턴으로 다음의 정확한 `clarify` 페이로드 형태를 사용합니다.

- `{"question":"Your existing SOUL.md conflicts with the imported one. What should I do?","choices":["keep existing","overwrite with backup","review first"]}`
- `{"question":"One or more imported OpenClaw skills already exist in Hermes. How should I handle those skill conflicts?","choices":["keep existing skills","overwrite conflicting skills with backup","import conflicting skills under renamed folders"]}`
- `{"question":"Choose migration mode: migrate only user data, or run the full compatible migration including allowlisted secrets?","choices":["user-data only","full compatible migration","cancel"]}`
- `{"question":"Do you want to copy the OpenClaw workspace instructions file into a Hermes workspace?","choices":["skip workspace instructions","copy to a workspace path","decide later"]}`
- `{"question":"Please provide an absolute path where the workspace instructions should be copied."}`

## 결정 사항과 명령 매핑

사용자 결정을 정확히 다음 명령 플래그에 매핑합니다.

- 사용자가 `SOUL.md`에 대해 `keep existing`을 선택하면 `--overwrite`를 추가하지 않습니다.
- 사용자가 `overwrite with backup`을 선택하면 `--overwrite`를 추가합니다.
- 사용자가 `review first`를 선택하면 실행 전에 멈추고 관련 파일을 검토합니다.
- 사용자가 `keep existing skills`를 선택하면 `--skill-conflict skip`을 추가합니다.
- 사용자가 `overwrite conflicting skills with backup`을 선택하면 `--skill-conflict overwrite`를 추가합니다.
- 사용자가 `import conflicting skills under renamed folders`를 선택하면 `--skill-conflict rename`을 추가합니다.
- 사용자가 `user-data only`를 선택하면 `--preset user-data`로 실행하고 `--migrate-secrets`는 추가하지 않습니다.
- 사용자가 `full compatible migration`을 선택하면 `--preset full --migrate-secrets`로 실행합니다.
- 사용자가 명시적으로 절대 작업 공간 경로를 제공한 경우에만 `--workspace-target`을 추가합니다.
- 사용자가 `skip workspace instructions` 또는 `decide later`를 선택하면 `--workspace-target`을 추가하지 않습니다.

실행하기 전에 정확한 명령 계획을 평범한 언어로 다시 말하고, 사용자의 선택과 일치하는지 확인합니다.

## 실행 후 보고 규칙

실행 후에는 스크립트의 JSON 출력을 사실의 기준으로 취급합니다.

1. 모든 수치는 `report.summary`를 기준으로 합니다.
2. `status`가 정확히 `migrated`인 경우에만 "성공적으로 마이그레이션됨" 아래에 항목을 나열합니다.
3. 보고서에서 해당 항목이 `migrated`로 표시되지 않았다면 충돌이 해결되었다고 주장하지 않습니다.
4. `kind="soul"`인 보고서 항목의 `status`가 `migrated`일 때만 `SOUL.md`가 덮어써졌다고 말합니다.
5. `report.summary.conflict > 0`이면 성공을 암시하지 말고 충돌 섹션을 포함합니다.
6. 수치와 나열된 항목이 일치하지 않으면 응답하기 전에 보고서에 맞게 목록을 수정합니다.
7. 보고서에 `output_dir` 경로가 있으면 포함하여 사용자가 `report.json`, `summary.md`, 백업 및 보관 파일을 확인할 수 있도록 합니다.
8. 메모리 또는 사용자 프로필이 넘친 경우 보고서에 보관 경로가 명시되어 있을 때만 항목이 보관되었다고 말합니다. `details.overflow_file`이 있으면 전체 초과 목록이 해당 경로로 내보내졌다고 말합니다.
9. 가져온 스킬이 이름이 변경된 폴더에 배치된 경우 최종 대상 경로를 보고하고 `details.renamed_from`을 언급합니다.
10. `report.skill_conflict_mode`가 있으면 선택한 가져온 스킬 충돌 정책의 기준으로 사용합니다.
11. 항목의 `status="skipped"`를 덮어쓰기, 백업, 마이그레이션 또는 해결로 설명하지 않습니다.
12. `kind="soul"`의 `status="skipped"` 이유가 `Target already matches source`라면 변경하지 않고 그대로 두었다고 말하며 백업은 언급하지 않습니다.
13. 이름이 변경된 가져온 스킬에서 `details.backup`이 비어 있으면 기존 Hermes 스킬의 이름이 변경되었거나 백업되었다고 암시하지 않습니다. 가져온 사본이 새 대상에 배치되었다고만 말하고, 그대로 남아 있는 기존 폴더를 `details.renamed_from`으로 참조합니다.
## 마이그레이션 프리셋

일반적으로 다음 두 프리셋을 사용하세요.

- `user-data`
- `full`

`user-data`에는 다음 항목이 포함됩니다.

- `soul`
- `workspace-agents`
- `memory`
- `user-profile`
- `messaging-settings`
- `command-allowlist`
- `skills`
- `tts-assets`
- `archive`

`full`에는 `user-data`의 모든 항목과 다음 항목이 포함됩니다.

- `secret-settings`

도우미 스크립트는 여전히 카테고리 수준의 `--include` / `--exclude`를 지원하지만, 기본 UX가 아닌 고급 대체 수단으로 취급하세요.

## 명령어

전체 검색으로 시험 실행:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py
```

터미널 도구를 사용할 때는 다음과 같이 절대 경로를 사용하는 방식을 권장합니다.

```json
{"command":"python3 /home/USER/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py","workdir":"/home/USER"}
```

사용자 데이터 프리셋으로 시험 실행:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --preset user-data
```

사용자 데이터 마이그레이션 실행:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset user-data --skill-conflict skip
```

호환 가능한 전체 마이그레이션 실행:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset full --migrate-secrets --skill-conflict skip
```

워크스페이스 지침을 포함하여 실행:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset user-data --skill-conflict rename --workspace-target "/absolute/workspace/path"
```

기본적으로 `$PWD`나 홈 디렉터리를 워크스페이스 대상으로 사용하지 마세요. 먼저 명시적인 워크스페이스 경로를 요청하세요.

## 중요 규칙

1. 사용자가 즉시 진행하라고 명시하지 않았다면, 파일을 쓰기 전에 시험 실행을 수행하세요.
2. 기본적으로 시크릿을 마이그레이션하지 마세요. 토큰, 인증 블롭, 디바이스 자격 증명 및 원시 게이트웨이 구성은 사용자가 명시적으로 시크릿 마이그레이션을 요청하지 않는 한 Hermes 외부에 유지해야 합니다.
3. 사용자가 명시적으로 원하지 않는 한 비어 있지 않은 Hermes 대상에 조용히 덮어쓰지 마세요. 덮어쓰기가 활성화되면 도우미 스크립트가 백업을 보존합니다.
4. 항상 건너뛴 항목 보고서를 사용자에게 제공하세요. 해당 보고서는 마이그레이션의 일부이며 선택 사항이 아닙니다.
5. 기본 OpenClaw 워크스페이스(`~/.openclaw/workspace/`)를 `workspace.default/`보다 우선하세요. 기본 워크스페이스의 파일이 없는 경우에만 대체 수단으로 사용하세요.
6. 시크릿 마이그레이션 모드에서도 대상 Hermes 경로가 깨끗한 경우에만 시크릿을 마이그레이션하세요. 지원되지 않는 인증 블롭은 여전히 건너뛴 항목으로 보고해야 합니다.
7. 시험 실행에서 대규모 에셋 복사, 충돌하는 `SOUL.md` 또는 메모리 항목 초과가 표시되면 실행 전에 이를 별도로 알리세요.
8. 사용자가 확신하지 못하는 경우 기본값은 `user-data only`로 하세요.
9. 사용자가 대상 워크스페이스 경로를 명시적으로 제공한 경우에만 `workspace-agents`를 포함하세요.
10. 카테고리 수준의 `--include` / `--exclude`는 일반적인 흐름이 아닌 고급 우회 수단으로 취급하세요.
11. `clarify`를 사용할 수 있다면 시험 실행 요약을 막연한 “어떻게 하시겠어요?”로 끝내지 마세요. 구조화된 후속 프롬프트를 사용하세요.
12. 실제 선택지로 충분한 경우 개방형 `clarify` 프롬프트를 사용하지 마세요. 선택 가능한 항목을 먼저 제시하고, 절대 경로 또는 파일 검토 요청에만 자유 입력을 사용하세요.
13. 시험 실행 후 해결되지 않은 결정이 남아 있다면 요약만 하고 멈추지 마세요. 가장 우선순위가 높은 차단 결정에 대해 즉시 `clarify`를 호출하세요.
14. 후속 질문의 우선순위는 다음과 같습니다.
    - `SOUL.md` 충돌
    - 가져온 스킬 충돌
    - 마이그레이션 모드
    - 워크스페이스 지침 대상
15. 나중에 선택지를 제시하겠다고 약속하지 마세요. 실제로 `clarify`를 호출하여 제시하세요.
16. 마이그레이션 모드에 대한 답변을 받은 후에도 `workspace-agents`가 아직 미결정 상태인지 명시적으로 확인하세요. 미결정 상태라면 다음 작업은 반드시 워크스페이스 지침에 대한 `clarify` 호출이어야 합니다.
17. `clarify` 답변을 받은 후에도 필요한 결정이 남아 있다면, 방금 결정된 내용을 설명하지 말고 즉시 다음 필수 질문을 하세요.

## 예상 결과

성공적으로 실행되면 사용자는 다음을 갖게 됩니다.

- Hermes 페르소나 상태를 가져온 결과
- 변환된 OpenClaw 지식으로 채워진 Hermes 메모리 파일
- `~/.hermes/skills/openclaw-imports/`에 있는 OpenClaw 스킬
- 충돌, 누락 또는 지원되지 않는 데이터를 보여 주는 마이그레이션 보고서
