---
title: "Oss Forensics — GitHub 공급망 포렌식: 복구, IOC, 보고서 작성"
sidebar_label: "Oss Forensics"
description: "GitHub 공급망 포렌식: 복구, IOC, 보고서 작성"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Oss Forensics

GitHub 공급망 포렌식: 복구, IOC, 보고서 작성.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/security/oss-forensics`로 설치 |
| 경로 | `optional-skills/security/oss-forensics` |
| 버전 | `1.0.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Security`, `Forensics`, `GitHub`, `Supply-Chain` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# OSS 보안 포렌식 스킬

오픈 소스 공급망 공격을 조사하기 위한 7단계 멀티 에이전트 조사 프레임워크입니다.
RAPTOR의 포렌식 시스템에서 적용했습니다. GitHub Archive, Wayback Machine, GitHub API,
로컬 git 분석, IOC 추출, 증거에 기반한 가설 수립 및 검증,
최종 포렌식 보고서 생성을 다룹니다.

---

## ⚠️ 환각 방지 가드레일

모든 조사 단계 전에 다음 내용을 읽으세요. 이를 위반하면 보고서가 무효화됩니다.

1. **증거 우선 규칙**: 모든 보고서, 가설 또는 요약의 모든 주장은 최소 하나의 증거 ID(`EV-XXXX`)를 반드시 인용해야 합니다. 인용이 없는 주장은 금지됩니다.
2. **각자의 영역을 지키세요**: 각 하위 에이전트(조사자)는 단일 데이터 소스를 담당합니다. 소스를 섞지 마세요. GH Archive 조사자는 GitHub API를 조회하지 않으며, 그 반대도 마찬가지입니다. 역할 경계는 엄격합니다.
3. **사실과 가설 분리**: 검증되지 않은 모든 추론에는 `[HYPOTHESIS]`를 표시하세요. 원본 소스에 대조해 검증된 문장만 사실로 표현할 수 있습니다.
4. **증거 조작 금지**: 가설 검증자는 가설을 승인하기 전에 인용된 모든 증거 ID가 증거 저장소에 실제로 존재하는지 기계적으로 확인해야 합니다.
5. **반증에는 증거가 필요합니다**: 구체적이고 증거에 기반한 반론 없이는 가설을 기각할 수 없습니다. "증거를 찾지 못했다"는 반증에 충분하지 않으며, 가설을 미결정 상태로 만들 뿐입니다.
6. **SHA/URL 이중 검증**: 증거로 인용된 모든 커밋 SHA, URL 또는 외부 식별자는 검증된 것으로 표시하기 전에 최소 두 개의 소스에서 독립적으로 확인해야 합니다.
7. **의심스러운 코드 규칙**: 조사 대상 저장소에서 발견한 코드를 로컬에서 절대 실행하지 마세요. 정적으로만 분석하거나, 샌드박스 환경에서 `execute_code`를 사용하세요.
8. **비밀 정보 삭제**: 조사 중 발견한 모든 API 키, 토큰 또는 자격 증명은 최종 보고서에서 삭제해야 합니다. 내부적으로만 기록하세요.

---

## 예시 시나리오

- **시나리오 A: 의존성 혼동**: 악성 패키지 `internal-lib-v2`가 내부 패키지보다 높은 버전으로 NPM에 업로드됩니다. 조사자는 이 패키지가 처음 발견된 시점을 추적하고, 대상 저장소의 `package.json`을 이 버전으로 업데이트한 PushEvent가 있었는지 확인해야 합니다.
- **시나리오 B: 관리자 탈취**: 장기간 활동이 없던 기여자의 계정이 백도어가 삽입된 `.github/workflows/build.yml`을 푸시하는 데 사용됩니다. 조사자는 오랜 비활동 기간 이후 또는 새로운 IP/위치에서 이 사용자가 발생시킨 PushEvent를 찾습니다(BigQuery로 감지할 수 있는 경우).
- **시나리오 C: 강제 푸시 은폐**: 개발자가 실수로 프로덕션 비밀 정보를 커밋한 뒤 "수정"을 위해 강제 푸시합니다. 조사자는 `git fsck`와 GH Archive를 사용해 원래 커밋 SHA를 복구하고 무엇이 유출되었는지 확인합니다.

---

> **경로 규칙**: 이 스킬 전체에서 `SKILL_DIR`은 이 스킬의 설치 디렉터리(이 `SKILL.md`가 들어 있는 폴더)의 루트를 가리킵니다. 스킬이 로드될 때 `SKILL_DIR`을 실제 경로로 확인하세요 — 예: `~/.hermes/skills/security/oss-forensics/` 또는 `optional-skills/`에 해당하는 경로. 모든 스크립트와 템플릿 참조는 이 경로를 기준으로 합니다.

## 0단계: 초기화

1. 조사 작업 디렉터리를 생성합니다.
   ```bash
   mkdir investigation_$(echo "REPO_NAME" | tr '/' '_')
   cd investigation_$(echo "REPO_NAME" | tr '/' '_')
   ```
2. 증거 저장소를 초기화합니다.
   ```bash
   python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list
   ```
3. 포렌식 보고서 템플릿을 복사합니다.
   ```bash
   cp SKILL_DIR/templates/forensic-report.md ./investigation-report.md
   ```
4. 발견한 침해 지표를 추적할 `iocs.md` 파일을 생성합니다.
5. 조사 시작 시간, 대상 저장소, 명시된 조사 목표를 기록합니다.

---

## 1단계: 프롬프트 파싱 및 IOC 추출

**목표**: 사용자의 요청에서 구조화된 모든 조사 대상을 추출합니다.

**작업**:
- 사용자 프롬프트를 파싱하고 다음을 추출합니다.
  - 대상 저장소(`owner/repo`)
  - 대상 행위자(GitHub 핸들, 이메일 주소)
  - 관심 시간 범위(커밋 날짜 범위, PR 타임스탬프)
  - 제공된 침해 지표: 커밋 SHA, 파일 경로, 패키지 이름, IP 주소, 도메인, API 키/토큰, 악성 URL
  - 연결된 공급업체 보안 보고서 또는 블로그 게시물

**도구**: 추론만 사용하거나, 큰 텍스트 블록에서 정규식을 추출할 때 `execute_code`를 사용합니다.

**출력**: 추출한 IOC를 `iocs.md`에 기록합니다. 각 IOC에는 다음이 포함되어야 합니다.
- 유형(다음 중 하나: COMMIT_SHA, FILE_PATH, API_KEY, SECRET, IP_ADDRESS, DOMAIN, PACKAGE_NAME, ACTOR_USERNAME, MALICIOUS_URL, OTHER)
- 값
- 출처(사용자 제공, 추론)

**참조**: IOC 분류 체계는 [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md)를 참조하세요.

## 2단계: 병렬 증거 수집

`delegate_task`(배치 모드, 최대 3개 동시 실행)를 사용해 최대 5명의 전문 조사 하위 에이전트를 생성합니다. 각 조사자는 단일 데이터 소스만 담당하며 소스를 섞어서는 안 됩니다.

> **오케스트레이터 참고**: 각 위임 작업의 `context` 필드에 1단계의 IOC 목록과 조사 시간 범위를 전달합니다.

---

### 조사자 1: 로컬 Git 조사자

**역할 경계**: 로컬 GIT 저장소만 조회합니다. 외부 API를 호출하지 마세요.

**작업**:
```bash
# Clone repository
git clone https://github.com/OWNER/REPO.git target_repo && cd target_repo

# Full commit log with stats
git log --all --full-history --stat --format="%H|%ae|%an|%ai|%s" > ../git_log.txt

# Detect force-push evidence (orphaned/dangling commits)
git fsck --lost-found --unreachable 2>&1 | grep commit > ../dangling_commits.txt

# Check reflog for rewritten history
git reflog --all > ../reflog.txt

# List ALL branches including deleted remote refs
git branch -a -v > ../branches.txt

# Find suspicious large binary additions
git log --all --diff-filter=A --name-only --format="%H %ai" -- "*.so" "*.dll" "*.exe" "*.bin" > ../binary_additions.txt

# Check for GPG signature anomalies
git log --show-signature --format="%H %ai %aN" > ../signature_check.txt 2>&1
```

**수집할 증거**(`python3 SKILL_DIR/scripts/evidence-store.py add`로 추가):
- 각 dangling 커밋 SHA → 유형: `git`
- 강제 푸시 증거(기록이 재작성되었음을 보여 주는 reflog) → 유형: `git`
- 검증된 기여자의 서명되지 않은 커밋 → 유형: `git`
- 의심스러운 바이너리 파일 추가 → 유형: `git`

**참조**: 강제 푸시된 커밋에 접근하는 방법은 [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md)를 참조하세요.

---

### 조사자 2: GitHub API 조사자

**역할 경계**: GITHUB REST API만 조회합니다. 로컬에서 git 명령을 실행하지 마세요.

**작업**:
```bash
# Commits (paginated)
curl -s "https://api.github.com/repos/OWNER/REPO/commits?per_page=100" > api_commits.json

# Pull Requests including closed/deleted
curl -s "https://api.github.com/repos/OWNER/REPO/pulls?state=all&per_page=100" > api_prs.json

# Issues
curl -s "https://api.github.com/repos/OWNER/REPO/issues?state=all&per_page=100" > api_issues.json

# Contributors and collaborator changes
curl -s "https://api.github.com/repos/OWNER/REPO/contributors" > api_contributors.json

# Repository events (last 300)
curl -s "https://api.github.com/repos/OWNER/REPO/events?per_page=100" > api_events.json

# Check specific suspicious commit SHA details
curl -s "https://api.github.com/repos/OWNER/REPO/git/commits/SHA" > commit_detail.json

# Releases
curl -s "https://api.github.com/repos/OWNER/REPO/releases?per_page=100" > api_releases.json

# Check if a specific commit exists (force-pushed commits may 404 on commits/ but succeed on git/commits/)
curl -s "https://api.github.com/repos/OWNER/REPO/commits/SHA" | jq .sha
```

**상호 참조 대상**(불일치를 증거로 표시):
- 아카이브에는 PR이 있지만 API에는 없음 → 삭제의 증거
- 아카이브 이벤트에는 기여자가 있지만 contributors 목록에는 없음 → 권한 철회의 증거
- 아카이브 PushEvent에는 커밋이 있지만 API 커밋 목록에는 없음 → 강제 푸시/삭제의 증거

**참조**: GH 이벤트 유형은 [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md)를 참조하세요.

---

### 조사자 3: Wayback Machine 조사자

**역할 경계**: WAYBACK MACHINE CDX API만 조회합니다. GitHub API를 사용하지 마세요.

**목표**: 삭제된 GitHub 페이지(README, 이슈, PR, 릴리스, 위키 페이지)를 복구합니다.

**작업**:
```bash
# Search for archived snapshots of the repo main page
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO&output=json&limit=100&from=YYYYMMDD&to=YYYYMMDD" > wayback_main.json

# Search for a specific deleted issue
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/issues/NUM&output=json&limit=50" > wayback_issue_NUM.json

# Search for a specific deleted PR
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/pull/NUM&output=json&limit=50" > wayback_pr_NUM.json

# Fetch the best snapshot of a page
# Use the Wayback Machine URL: https://web.archive.org/web/TIMESTAMP/ORIGINAL_URL
# Example: https://web.archive.org/web/20240101000000*/github.com/OWNER/REPO

# Advanced: Search for deleted releases/tags
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/releases/tag/*&output=json" > wayback_tags.json

# Advanced: Search for historical wiki changes
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/wiki/*&output=json" > wayback_wiki.json
```

**수집할 증거**:
- 내용이 포함된 삭제된 이슈/PR의 보관 스냅샷
- 변경 사항이 드러나는 과거 README 버전
- 아카이브에는 있지만 현재 GitHub 상태에는 없는 콘텐츠의 증거

**참조**: CDX API 매개변수는 [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md)를 참조하세요.

---

### 조사자 4: GH Archive / BigQuery 조사자

**역할 경계**: BIGQUERY만 사용해 GITHUB ARCHIVE를 조회합니다. 이는 모든 공개 GitHub 이벤트의 변조 방지 기록입니다.

> **사전 요구 사항**: BigQuery 액세스 권한이 있는 Google Cloud 자격 증명이 필요합니다(`gcloud auth application-default login`). 사용할 수 없다면 이 조사자를 건너뛰고 보고서에 기록하세요.

**비용 최적화 규칙**(필수):
1. 모든 쿼리 전에 항상 `--dry_run`을 실행해 비용을 추정합니다.
2. `_TABLE_SUFFIX`를 사용해 날짜 범위를 필터링하고 스캔되는 데이터를 최소화합니다.
3. 필요한 열만 SELECT합니다.
4. 집계하는 경우를 제외하고 LIMIT을 추가합니다.

```bash
# Template: safe BigQuery query for PushEvents to OWNER/REPO
bq query --use_legacy_sql=false --dry_run "
SELECT created_at, actor.login, payload.commits, payload.before, payload.head,
       payload.size, payload.distinct_size
FROM \`githubarchive.month.*\`
WHERE _TABLE_SUFFIX BETWEEN 'YYYYMM' AND 'YYYYMM'
  AND type = 'PushEvent'
  AND repo.name = 'OWNER/REPO'
LIMIT 1000
"
# If cost is acceptable, re-run without --dry_run

# Detect force-pushes: zero-distinct_size PushEvents mean commits were force-erased
# payload.distinct_size = 0 AND payload.size > 0 → force push indicator

# Check for deleted branch events
bq query --use_legacy_sql=false "
SELECT created_at, actor.login, payload.ref, payload.ref_type
FROM \`githubarchive.month.*\`
WHERE _TABLE_SUFFIX BETWEEN 'YYYYMM' AND 'YYYYMM'
  AND type = 'DeleteEvent'
  AND repo.name = 'OWNER/REPO'
LIMIT 200
"
```

**수집할 증거**:
- 강제 푸시 이벤트(payload.size > 0, payload.distinct_size = 0)
- 브랜치/태그의 DeleteEvent
- 의심스러운 CI/CD 자동화를 위한 WorkflowRunEvent
- git 로그의 "공백"에 앞서 발생한 PushEvent(재작성의 증거)

**참조**: 12개 이벤트 유형과 쿼리 패턴 전체는 [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md)를 참조하세요.

---

### 조사자 5: IOC 보강 조사자

**역할 경계**: 1단계의 기존 IOC를 수동적인 공개 소스만 사용해 보강합니다. 대상 저장소의 코드를 실행하지 마세요.

**작업**:
- 각 커밋 SHA에 대해 직접 GitHub URL(`github.com/OWNER/REPO/commit/SHA.patch`)을 통한 복구를 시도합니다.
- 각 도메인/IP에 대해 공개 WHOIS 서비스의 `web_extract`를 사용해 수동 DNS와 WHOIS 기록을 확인합니다.
- 각 패키지 이름에 대해 npm/PyPI에서 일치하는 악성 패키지 보고서를 확인합니다.
- 각 행위자 사용자 이름에 대해 GitHub 프로필, 기여 기록, 계정 생성 시점을 확인합니다.
- 3가지 방법으로 강제 푸시된 커밋을 복구합니다([recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md) 참조).

---

## 3단계: 증거 통합

모든 조사자가 완료한 후 다음을 수행합니다.

1. `python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list`를 실행해 수집한 모든 증거를 확인합니다.
2. 각 증거의 `content_sha256` 해시가 원본 소스와 일치하는지 확인합니다.
3. 다음 기준으로 증거를 그룹화합니다.
   - **타임라인**: 타임스탬프가 있는 모든 증거를 시간순으로 정렬
   - **행위자**: GitHub 핸들 또는 이메일별로 그룹화
   - **IOC**: 증거를 관련 IOC에 연결
4. **불일치**를 식별합니다. 한 소스에는 있지만 다른 소스에는 없는 항목(삭제의 주요 지표)입니다.
5. 증거에 `[VERIFIED]`(2개 이상의 독립적인 소스에서 확인됨) 또는 `[UNVERIFIED]`(단일 소스만 확인됨)를 표시합니다.

---

## 4단계: 가설 수립

가설은 다음을 충족해야 합니다.
- 구체적인 주장을 제시합니다(예: "행위자 X가 DATE에 BRANCH에 강제 푸시하여 커밋 SHA를 삭제했다").
- 이를 뒷받침하는 증거 ID를 최소 2개 인용합니다(`EV-XXXX`, `EV-YYYY`).
- 이를 반증할 증거가 무엇인지 식별합니다.
- 검증될 때까지 `[HYPOTHESIS]`로 표시합니다.

**일반적인 가설 템플릿**([investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md) 참조):
- 관리자 침해: 탈취 후 정당한 계정을 사용해 악성 코드를 삽입
- 의존성 혼동: 설치를 가로채기 위한 패키지 이름 선점
- CI/CD 주입: 빌드 중 코드를 실행하도록 악성 워크플로 변경
- 타이포스쿼팅: 오타를 내는 사용자를 노리는 거의 동일한 패키지 이름
- 자격 증명 유출: 토큰/키를 실수로 커밋한 후 강제 푸시로 삭제

각 가설에 대해 `delegate_task` 하위 에이전트를 생성하여 확인 전에 반증 증거를 찾도록 합니다.

---

## 5단계: 가설 검증

검증 하위 에이전트는 반드시 기계적으로 다음을 확인해야 합니다.

1. 각 가설에서 인용된 모든 증거 ID를 추출합니다.
2. 각 ID가 `evidence.json`에 존재하는지 확인합니다(누락된 ID가 하나라도 있으면 가설이 조작되었을 가능성이 있으므로 즉시 실패 처리하고 가설을 거부합니다).
3. `[VERIFIED]`로 표시된 각 증거가 2개 이상의 소스에서 확인되었는지 검증합니다.
4. 논리적 일관성을 확인합니다. 증거가 보여 주는 타임라인이 가설을 뒷받침합니까?
5. 대안적 설명을 확인합니다. 같은 증거 패턴이 무해한 원인으로도 발생할 수 있습니까?

**출력**:
- `VALIDATED`: 인용된 모든 증거가 존재하고, 검증되었으며, 논리적으로 일관되고, 그럴듯한 대안 설명이 없습니다.
- `INCONCLUSIVE`: 증거가 가설을 뒷받침하지만 대안 설명이 존재하거나 증거가 불충분합니다.
- `REJECTED`: 증거 ID가 누락되었거나, 검증되지 않은 증거를 사실로 인용했거나, 논리적 불일치가 발견되었습니다.

거부된 가설은 4단계로 되돌려 수정합니다(최대 3회 반복).

---

## 6단계: 최종 보고서 생성

[forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md)의 템플릿을 사용해 `investigation-report.md`를 작성합니다.

**필수 섹션**:
- 요약: 신뢰도 수준과 함께 한 문단으로 판정(침해됨 / 정상 / 미결정)
- 타임라인: 증거 인용을 포함한 모든 주요 이벤트의 시간순 재구성
- 검증된 가설: 각 가설의 상태와 이를 뒷받침하는 증거 ID
- 증거 레지스트리: 출처, 유형, 검증 상태가 포함된 모든 `EV-XXXX` 항목의 표
- IOC 목록: 추출하고 보강한 모든 침해 지표
- 증거 보관 연속성: 증거를 어떻게, 어떤 소스에서, 어떤 타임스탬프에 수집했는지
- 권고 사항: 침해가 감지된 경우 즉시 수행할 완화 조치 및 모니터링 권고

**보고서 규칙**:
- 모든 사실 주장은 최소 하나의 `[EV-XXXX]` 인용을 포함해야 합니다.
- 요약에는 신뢰도 수준(높음 / 중간 / 낮음)을 명시해야 합니다.
- 모든 비밀 정보/자격 증명은 `[REDACTED]`로 삭제해야 합니다.

---

## 7단계: 완료

1. 최종 증거 개수를 확인합니다: `python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list`
2. 전체 조사 디렉터리를 보관합니다.
3. 침해가 확인된 경우:
   - 즉시 완화 조치를 나열합니다(자격 증명 교체, 의존성 해시 고정, 영향을 받은 사용자에게 알림).
   - 영향을 받은 버전/패키지를 식별합니다.
   - 공개 패키지인 경우 공개 의무를 기록하고 패키지 레지스트리와 조율합니다.
4. 최종 `investigation-report.md`를 사용자에게 제시합니다.

---

## 윤리적 사용 지침

이 스킬은 방어적 보안 조사, 즉 오픈 소스 소프트웨어를 공급망 공격으로부터 보호하기 위한 목적으로 설계되었습니다. 다음 용도로 사용해서는 안 됩니다.

- **기여자 또는 관리자 괴롭힘/스토킹**
- **신상 털기** — 악의적인 목적을 위해 GitHub 활동을 실제 신원과 연결하는 행위
- **경쟁 정보 수집** — 승인 없이 독점 또는 내부 저장소를 조사하는 행위
- **허위 비난** — 검증된 증거 없이 조사 결과를 공개하는 행위(위의 환각 방지 가드레일 참조)

조사는 **최소한의 침해** 원칙에 따라 수행해야 합니다. 가설을 검증하거나 반박하는 데 필요한 증거만 수집하세요. 결과를 공개할 때는 책임 있는 공개 관행을 따르고, 공개 전에 영향을 받은 관리자와 조율하세요.

조사에서 실제 침해가 드러나면 조정된 취약점 공개 절차를 따릅니다.
1. 먼저 저장소 관리자에게 비공개로 알립니다.
2. 문제를 해결할 합리적인 시간을 제공합니다(일반적으로 90일).
3. 게시된 패키지가 영향을 받은 경우 패키지 레지스트리(npm, PyPI 등)와 조율합니다.
4. 적절하다면 CVE를 등록합니다.

---

## API 속도 제한

GitHub REST API는 속도 제한을 적용하므로 적절히 관리하지 않으면 대규모 조사가 중단됩니다.

**인증된 요청**: 시간당 5,000회(`GITHUB_TOKEN` 환경 변수 또는 `gh` CLI 인증 필요)
**인증되지 않은 요청**: 시간당 60회(조사에 사용할 수 없음)

**권장 사항**:
- 항상 인증합니다: `export GITHUB_TOKEN=ghp_...`를 사용하거나 `gh` CLI를 사용합니다(자동 인증).
- 조건부 요청(`If-None-Match` / `If-Modified-Since` 헤더)을 사용해 변경되지 않은 데이터를 다시 가져오느라 할당량을 소비하지 않도록 합니다.
- 페이지가 매겨진 엔드포인트는 모든 페이지를 순서대로 가져옵니다. 동일한 엔드포인트에 대해 병렬화하지 마세요.
- `X-RateLimit-Remaining` 헤더를 확인합니다. 100 미만이면 `X-RateLimit-Reset` 타임스탬프까지 일시 중지합니다.
- BigQuery에는 자체 할당량(무료 등급에서 하루 10 TiB)이 있으므로 항상 먼저 dry-run을 실행합니다.
- Wayback Machine CDX API에는 공식적인 속도 제한이 없지만 예의를 지켜 초당 최대 1~2건을 요청합니다.

조사 중 속도 제한에 걸리면 부분 결과를 증거 저장소에 기록하고 보고서에 제한 사항을 명시합니다.

---

## 참고 자료

- [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md) — BigQuery 쿼리, CDX API, 12개 이벤트 유형
- [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md) — IOC 분류 체계, 증거 소스 유형, 관찰 유형
- [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md) — 삭제된 커밋, PR, 이슈 복구
- [investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md) — 공격 유형별 사전 작성 가설 템플릿
- [evidence-store.py](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/scripts/evidence-store.py) — JSON 증거 저장소를 관리하는 CLI 도구
- [forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md) — 구조화된 보고서 템플릿
