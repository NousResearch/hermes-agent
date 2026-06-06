---
title: "Oss Forensics — GitHub 저장소를 위한 공급망 조사, 증거 복구 및 포렌식 분석"
sidebar_label: "Oss Forensics"
description: "GitHub 저장소를 위한 공급망 조사, 증거 복구 및 포렌식 분석"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Oss Forensics

GitHub 저장소를 위한 공급망 조사, 증거 복구 및 포렌식 분석입니다.
삭제된 커밋 복구, 강제 푸시(force-push) 감지, IOC 추출, 다중 소스 증거 수집, 가설 설정/검증 및 구조화된 포렌식 보고를 다룹니다.
RAPTOR의 1800줄 이상 되는 OSS 포렌식 시스템에서 영감을 받았습니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/security/oss-forensics`로 설치 |
| 경로 | `optional-skills/security/oss-forensics` |
| 플랫폼 | linux, macos, windows |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# OSS 보안 포렌식 스킬

오픈소스 공급망 공격 조사를 위한 7단계 다중 에이전트 조사 프레임워크입니다.
RAPTOR의 포렌식 시스템을 응용했습니다. GitHub Archive, Wayback Machine, GitHub API, 로컬 git 분석, IOC 추출, 증거 기반 가설 설정 및 검증, 최종 포렌식 보고서 생성을 포함합니다.

---

## ⚠️ 환각 방지 가드레일 (Anti-Hallucination Guardrails)

모든 조사 단계 전에 이것들을 읽으십시오. 이를 위반하면 보고서가 무효화됩니다.

1. **증거 우선 원칙**: 보고서, 가설 또는 요약의 모든 주장은 최소한 하나의 증거 ID(`EV-XXXX`)를 인용해야 합니다. 인용 없는 주장은 금지됩니다.
2. **자신의 역할에만 충실할 것 (STAY IN YOUR LANE)**: 각 하위 에이전트(조사관)는 단일 데이터 소스만을 다룹니다. 소스를 섞지 마십시오. GH Archive 조사관은 GitHub API를 쿼리하지 않으며 그 반대도 마찬가지입니다. 역할 경계는 엄격합니다.
3. **사실과 가설의 분리**: 확인되지 않은 모든 추론은 `[HYPOTHESIS]`(가설)로 표시하십시오. 원본 출처를 통해 검증된 진술만 사실로 기술할 수 있습니다.
4. **증거 조작 금지**: 가설 검증자는 가설을 수용하기 전에 인용된 모든 증거 ID가 실제로 증거 저장소에 존재하는지 기계적으로 확인해야 합니다.
5. **반증은 증거 필수**: 구체적이고 증거에 기반한 반론 없이 가설을 묵살할 수 없습니다. "증거가 발견되지 않음"은 반증에 충분하지 않으며, 가설을 결정 불능 상태로 만들 뿐입니다.
6. **SHA/URL 이중 검증**: 증거로 인용되는 커밋 SHA, URL 또는 외부 식별자는 검증됨으로 표시되기 전에 최소 2개의 출처에서 독립적으로 확인되어야 합니다.
7. **의심스러운 코드 규칙**: 조사 대상 저장소 안에서 발견된 코드를 로컬 환경에서 실행하지 마십시오. 정적으로만 분석하거나 샌드박스 환경에서 `execute_code`를 사용하십시오.
8. **비밀 정보 검열**: 조사 중에 발견된 API 키, 토큰 또는 자격 증명은 최종 보고서에서 검열(Redact)되어야 합니다. 내부 로깅 용도로만 남겨두십시오.

---

## 예시 시나리오

- **시나리오 A: 의존성 혼동 (Dependency Confusion)**: 악의적인 패키지 `internal-lib-v2`가 내부 버전보다 높은 버전으로 NPM에 업로드되었습니다. 조사관은 이 패키지가 언제 처음 목격되었는지와 대상 저장소의 PushEvent 중 `package.json`을 이 버전으로 업데이트한 기록이 있는지 추적해야 합니다.
- **시나리오 B: 메인테이너 계정 탈취 (Maintainer Takeover)**: 오랜 기간 기여해 온 사람의 계정이 백도어가 포함된 `.github/workflows/build.yml`을 푸시하는 데 사용되었습니다. 조사관은 오랜 휴면 기간 이후에 발생한 PushEvent나 새로운 IP/위치에서 발생한 기록(BigQuery를 통해 탐지 가능한 경우)을 찾아야 합니다.
- **시나리오 C: 강제 푸시를 통한 은폐 (Force-Push Hide)**: 개발자가 실수로 운영 환경의 비밀 정보를 커밋한 뒤 이를 "수정"하기 위해 강제 푸시를 수행했습니다. 조사관은 `git fsck`와 GH Archive를 사용해 원본 커밋 SHA를 복구하고 무엇이 유출되었는지 확인합니다.

---

> **경로 규칙**: 이 스킬 전체에서 `SKILL_DIR`은 이 스킬의 설치 디렉터리 루트(이 `SKILL.md`가 포함된 폴더)를 뜻합니다. 스킬이 로드되면 `SKILL_DIR`을 실제 경로로(예: `~/.hermes/skills/security/oss-forensics/` 또는 그에 상응하는 `optional-skills/` 등) 해석하십시오. 스크립트와 템플릿 참조는 모두 이 디렉터리를 기준으로 합니다.

## 0단계: 초기화

1. 조사 작업 디렉터리를 생성합니다:
   ```bash
   mkdir investigation_$(echo "REPO_NAME" | tr '/' '_')
   cd investigation_$(echo "REPO_NAME" | tr '/' '_')
   ```
2. 증거 저장소를 초기화합니다:
   ```bash
   python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list
   ```
3. 포렌식 보고서 템플릿을 복사합니다:
   ```bash
   cp SKILL_DIR/templates/forensic-report.md ./investigation-report.md
   ```
4. 발견되는 침해 지표(Indicators of Compromise)를 추적할 `iocs.md` 파일을 생성합니다.
5. 조사가 시작된 시간, 대상 저장소 및 명시된 조사 목표를 기록합니다.

---

## 1단계: 프롬프트 파싱 및 IOC 추출

**목표**: 사용자 요청에서 조사 목표를 구조화하여 모두 추출합니다.

**작업**:
- 사용자 프롬프트를 파싱하고 다음을 추출합니다:
  - 대상 저장소 (`owner/repo`)
  - 대상 행위자 (GitHub 핸들, 이메일 주소)
  - 관심 시간대 (커밋 날짜 범위, PR 타임스탬프)
  - 제공된 침해 지표 (Indicators of Compromise): 커밋 SHA, 파일 경로, 패키지 이름, IP 주소, 도메인, API 키/토큰, 악성 URL
  - 참조된 벤더 보안 보고서나 블로그 포스트

**도구**: 논리적 추론만 사용하거나, 긴 텍스트에서 정규식 추출이 필요할 때 `execute_code`를 사용합니다.

**출력**: 추출된 IOC로 `iocs.md`를 채웁니다. 각 IOC는 다음을 포함해야 합니다:
- 유형(Type) (목록: COMMIT_SHA, FILE_PATH, API_KEY, SECRET, IP_ADDRESS, DOMAIN, PACKAGE_NAME, ACTOR_USERNAME, MALICIOUS_URL, OTHER)
- 값(Value)
- 출처(Source) (사용자 제공, 추론됨 등)

**참조**: IOC 분류 체계는 [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md)를 확인하십시오.

---

## 2단계: 병렬 증거 수집

`delegate_task`를 사용하여 최대 5명의 전문가 조사관 하위 에이전트를 생성합니다 (일괄 모드, 최대 3개 동시 실행). 각 조사관은 **단일 데이터 소스**를 가지며 다른 소스를 섞어서는 안 됩니다.

> **오케스트레이터 참고**: 1단계에서 추출된 IOC 목록과 조사 시간대를 각 위임된 작업의 `context` 필드에 전달하십시오.

---

### 조사관 1: 로컬 Git 조사관

**역할 경계**: 당신은 로컬 GIT 저장소만 쿼리합니다. 외부 API를 호출하지 마십시오.

**작업**:
```bash
# 저장소 클론
git clone https://github.com/OWNER/REPO.git target_repo && cd target_repo

# 통계를 포함한 전체 커밋 로그
git log --all --full-history --stat --format="%H|%ae|%an|%ai|%s" > ../git_log.txt

# 강제 푸시 증거 감지 (고아/연결이 끊긴 커밋)
git fsck --lost-found --unreachable 2>&1 | grep commit > ../dangling_commits.txt

# 조작된 이력을 위해 reflog 확인
git reflog --all > ../reflog.txt

# 삭제된 원격 참조를 포함한 모든 브랜치 나열
git branch -a -v > ../branches.txt

# 의심스러운 대용량 바이너리 추가 찾기
git log --all --diff-filter=A --name-only --format="%H %ai" -- "*.so" "*.dll" "*.exe" "*.bin" > ../binary_additions.txt

# GPG 서명 이상 징후 확인
git log --show-signature --format="%H %ai %aN" > ../signature_check.txt 2>&1
```

**수집할 증거** (`python3 SKILL_DIR/scripts/evidence-store.py add`를 통해 추가):
- 매 연결 끊긴 커밋 SHA → 유형: `git`
- 강제 푸시 증거(이력 변경을 보여주는 reflog) → 유형: `git`
- 검증된 기여자의 서명되지 않은 커밋 → 유형: `git`
- 의심스러운 바이너리 파일 추가 → 유형: `git`

**참조**: 강제 푸시된 커밋에 접근하는 방법은 [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md)를 확인하십시오.

---

### 조사관 2: GitHub API 조사관

**역할 경계**: 당신은 GITHUB REST API만 쿼리합니다. 로컬에서 git 명령을 실행하지 마십시오.

**작업**:
```bash
# 커밋 (페이지 매기기 적용됨)
curl -s "https://api.github.com/repos/OWNER/REPO/commits?per_page=100" > api_commits.json

# 닫히거나 삭제된 항목을 포함한 Pull Request
curl -s "https://api.github.com/repos/OWNER/REPO/pulls?state=all&per_page=100" > api_prs.json

# 이슈
curl -s "https://api.github.com/repos/OWNER/REPO/issues?state=all&per_page=100" > api_issues.json

# 기여자 및 협업자 변경 사항
curl -s "https://api.github.com/repos/OWNER/REPO/contributors" > api_contributors.json

# 저장소 이벤트 (최근 300개)
curl -s "https://api.github.com/repos/OWNER/REPO/events?per_page=100" > api_events.json

# 특정 의심스러운 커밋 SHA 세부 정보 확인
curl -s "https://api.github.com/repos/OWNER/REPO/git/commits/SHA" > commit_detail.json

# 릴리스
curl -s "https://api.github.com/repos/OWNER/REPO/releases?per_page=100" > api_releases.json

# 특정 커밋 존재 여부 확인 (강제 푸시된 커밋은 commits/ 에서는 404가 나지만 git/commits/ 에서는 성공할 수 있음)
curl -s "https://api.github.com/repos/OWNER/REPO/commits/SHA" | jq .sha
```

**대상 교차 검증** (불일치 발생 시 증거로 표시):
- Archive에는 PR이 존재하지만 API에는 없는 경우 → 삭제 증거
- 이벤트 기록엔 기여자가 있으나 contributors 목록에는 없는 경우 → 권한 회수 증거
- PushEvent에는 커밋이 있으나 API 커밋 목록에는 없는 경우 → 강제 푸시/삭제 증거

**참조**: GH 이벤트 유형은 [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md)를 참조하십시오.

---

### 조사관 3: Wayback Machine 조사관

**역할 경계**: 당신은 WAYBACK MACHINE CDX API만 쿼리합니다. GitHub API를 사용하지 마십시오.

**목표**: 삭제된 GitHub 페이지(README, 이슈, PR, 릴리스, 위키 페이지) 복구.

**작업**:
```bash
# 저장소 메인 페이지의 아카이브 스냅샷 검색
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO&output=json&limit=100&from=YYYYMMDD&to=YYYYMMDD" > wayback_main.json

# 특정 삭제된 이슈 검색
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/issues/NUM&output=json&limit=50" > wayback_issue_NUM.json

# 특정 삭제된 PR 검색
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/pull/NUM&output=json&limit=50" > wayback_pr_NUM.json

# 가장 적합한 스냅샷 가져오기
# Wayback Machine URL 사용: https://web.archive.org/web/TIMESTAMP/ORIGINAL_URL
# 예시: https://web.archive.org/web/20240101000000*/github.com/OWNER/REPO

# 고급: 삭제된 릴리스/태그 검색
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/releases/tag/*&output=json" > wayback_tags.json

# 고급: 위키의 과거 변경 내역 검색
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/wiki/*&output=json" > wayback_wiki.json
```

**수집할 증거**:
- 내용이 포함된 삭제된 이슈/PR의 보존된 스냅샷
- 변경 사항을 보여주는 과거 README 버전
- 아카이브에는 존재하나 현재 GitHub 상태에서는 누락된 콘텐츠에 대한 증거

**참조**: CDX API 파라미터는 [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md)를 참조하십시오.

---

### 조사관 4: GH Archive / BigQuery 조사관

**역할 경계**: 당신은 BIGQUERY를 통한 GITHUB ARCHIVE만 쿼리합니다. 이는 모든 공개 GitHub 이벤트에 대한 변조 방지 기록입니다.

> **사전 요구 사항**: BigQuery 접근 권한이 있는 Google Cloud 자격 증명이 필요합니다(`gcloud auth application-default login`). 만약 사용할 수 없다면 이 조사관을 건너뛰고 보고서에 기재하십시오.

**비용 최적화 규칙** (필수 사항):
1. 비용을 추정하기 위해 항상 모든 쿼리 전에 `--dry_run`을 먼저 실행하십시오.
2. 날짜 범위로 필터링하고 스캔되는 데이터를 최소화하기 위해 `_TABLE_SUFFIX`를 사용하십시오.
3. 필요한 열(columns)만 SELECT 하십시오.
4. 집계(aggregating)가 아니라면 항상 LIMIT을 추가하십시오.

```bash
# 템플릿: OWNER/REPO 에 대한 PushEvent를 조회하는 안전한 BigQuery 쿼리
bq query --use_legacy_sql=false --dry_run "
SELECT created_at, actor.login, payload.commits, payload.before, payload.head,
       payload.size, payload.distinct_size
FROM \`githubarchive.month.*\`
WHERE _TABLE_SUFFIX BETWEEN 'YYYYMM' AND 'YYYYMM'
  AND type = 'PushEvent'
  AND repo.name = 'OWNER/REPO'
LIMIT 1000
"
# 비용이 허용 가능한 수준이면 --dry_run 없이 다시 실행

# 강제 푸시 감지: distinct_size 가 0인 PushEvent는 커밋들이 강제로 삭제되었음을 의미함
# payload.distinct_size = 0 AND payload.size > 0 → 강제 푸시 징후

# 브랜치 삭제 이벤트 확인
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
- 강제 푸시 이벤트 (payload.size > 0, payload.distinct_size = 0)
- 브랜치/태그에 대한 DeleteEvent
- 의심스러운 CI/CD 자동화를 보여주는 WorkflowRunEvent
- git 로그 상의 "공백(gap)" 직전에 발생한 PushEvent (기록 변조의 증거)

**참조**: 전체 12개 이벤트 유형 및 쿼리 패턴은 [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md)를 참조하십시오.

---

### 조사관 5: IOC 심층 분석 조사관 (Enrichment Investigator)

**역할 경계**: 당신은 1단계에서 얻은 기존의 IOC들을 수동적 공개 출처(passive public sources)만을 활용해 심층 분석(enrich)합니다. 대상 저장소의 어떤 코드도 실행해서는 안 됩니다.

**작업**:
- 각 커밋 SHA에 대해: 직접적인 GitHub URL을 통해 복구 시도 (`github.com/OWNER/REPO/commit/SHA.patch`)
- 각 도메인/IP에 대해: 패시브 DNS, WHOIS 기록 확인 (공개 WHOIS 서비스에 `web_extract` 사용)
- 각 패키지 이름에 대해: 일치하는 악성 패키지 보고서가 npm/PyPI에 있는지 확인
- 각 행위자 사용자 이름에 대해: GitHub 프로필, 기여 이력, 계정 생성일 확인
- 3가지 방법을 사용해 강제 푸시된 커밋 복구 시도 ([recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md) 참조)

---

## 3단계: 증거 취합

모든 조사관의 작업이 완료된 후:

1. `python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list`를 실행하여 수집된 모든 증거를 봅니다.
2. 각 증거에 대해 `content_sha256` 해시가 원본과 일치하는지 검증합니다.
3. 다음 기준으로 증거를 그룹화합니다:
   - **타임라인**: 타임스탬프가 있는 모든 증거를 시간순 정렬
   - **행위자**: GitHub 핸들이나 이메일별 그룹화
   - **IOC**: 관련된 IOC에 증거 연결
4. **불일치** 확인: 한 소스에는 존재하지만 다른 곳에서는 누락된 항목 파악 (주요 삭제 지표).
5. 증거에 대해 `[VERIFIED]` (독립적인 2개 이상의 소스에서 확인됨) 또는 `[UNVERIFIED]` (단일 소스에서만 나옴)로 표시합니다.

---

## 4단계: 가설 설정

가설은 다음을 충족해야 합니다:
- 구체적인 주장을 명시할 것 (예: "행위자 X가 날짜 Y에 BRANCH로 강제 푸시하여 커밋 SHA를 지움")
- 주장을 뒷받침하는 최소 2개의 증거 ID 인용 (`EV-XXXX`, `EV-YYYY`)
- 이 가설을 반증할 수 있는 증거의 조건 식별
- 검증될 때까지 `[HYPOTHESIS]` 태그 부착

**일반적인 가설 템플릿** ([investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md) 참조):
- 메인테이너 계정 탈취: 합법적인 계정이 탈취당한 후 악성 코드를 주입하는 데 사용됨
- 의존성 혼동(Dependency Confusion): 설치를 가로채기 위해 패키지 이름을 선점(squatting)
- CI/CD 인젝션: 빌드 중 코드를 실행하기 위한 악성 워크플로 변경
- 타이포스쿼팅: 철자 오류를 노려 매우 유사한 패키지명 사용
- 자격 증명 유출: 토큰/키가 실수로 커밋된 후 이를 지우기 위해 강제 푸시

각 가설에 대해, 검증 완료 전 가설을 반증하는 증거를 찾기 위해 `delegate_task` 하위 에이전트를 생성합니다.

---

## 5단계: 가설 검증

검증자 하위 에이전트는 다음을 기계적으로 확인해야 합니다:

1. 각 가설에 대해 인용된 모든 증거 ID를 추출합니다.
2. 각 ID가 `evidence.json`에 존재하는지 확인합니다 (하나라도 누락된 ID가 있다면 즉시 실패 → 해당 가설은 조작되었을 가능성으로 기각).
3. `[VERIFIED]` 처리된 각 증거가 2개 이상의 출처에서 확인되었는지 검증합니다.
4. 논리적 일관성 확인: 증거들이 묘사하는 타임라인이 가설을 뒷받침하는가?
5. 대안 설명(Alternative explanations) 확인: 동일한 증거 패턴이 정상적인 이유로도 발생할 수 있는가?

**출력**:
- `VALIDATED` (검증됨): 모든 증거가 인용되었고, 검증되었으며, 논리적으로 일관되고, 그럴싸한 대안 설명이 없음.
- `INCONCLUSIVE` (결정 불가): 증거가 가설을 뒷받침하지만 대안 설명이 존재하거나 증거가 불충분함.
- `REJECTED` (기각됨): 누락된 증거 ID, 사실로 인용된 미검증 증거, 논리적 불일치 발견.

기각된 가설은 세부 수정을 위해 다시 4단계로 돌아갑니다 (최대 3회 반복).

---

## 6단계: 최종 보고서 작성

[forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md)의 템플릿을 사용하여 `investigation-report.md`를 채웁니다.

**필수 섹션**:
- 요약(Executive Summary): 한 문단으로 평결(침해됨 / 안전함 / 결정 불가) 및 확신 수준 명시
- 타임라인: 주요 사건들의 시간순 재구성과 증거 인용
- 검증된 가설: 상태 및 뒷받침하는 증거 ID 포함
- 증거 레지스트리: 출처, 유형, 검증 상태가 포함된 모든 `EV-XXXX` 항목 테이블
- IOC 목록: 추출되고 분석된 모든 침해 지표
- 보관 사슬(Chain of Custody): 어떤 증거가, 언제, 어디서 수집되었는지 기록
- 권장 사항: 침해가 확인된 경우 즉각적인 완화 조치 제안; 모니터링 권장 사항

**보고서 규칙**:
- 사실을 기반으로 한 모든 주장에는 최소 하나 이상의 `[EV-XXXX]` 인용이 있어야 합니다
- 요약(Executive Summary)에서는 반드시 확신 수준(높음 / 보통 / 낮음)을 명시해야 합니다
- 모든 비밀/자격 증명은 `[REDACTED]`(검열됨)로 가려져야 합니다

---

## 7단계: 완료

1. 최종 증거 개수 확인 실행: `python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list`
2. 전체 조사 디렉터리를 아카이브합니다.
3. 침해가 확인된 경우:
   - 즉각적 완화 조치 목록 작성 (자격 증명 교체, 의존성 해시 고정, 영향받은 사용자에게 알림)
   - 영향을 받은 버전/패키지 식별
   - 공개 의무 기록 (공개 패키지인 경우: 패키지 레지스트리와 협력)
4. 최종 `investigation-report.md`를 사용자에게 제시합니다.

---

## 윤리적 사용 지침

이 스킬은 오픈소스 소프트웨어를 공급망 공격으로부터 보호하기 위한 **방어적 보안 조사** 목적으로 설계되었습니다. 다음 용도로 절대 사용하지 마십시오:

- 기여자나 메인테이너에 대한 **괴롭힘 또는 스토킹**
- 악의적인 목적으로 GitHub 활동과 실제 신원을 연관 짓는 **신상 털기(Doxing)**
- 권한 없는 독점/내부 저장소에 대한 **경쟁 정보 수집(Competitive intelligence)**
- **허위 고발** — 검증된 증거 없이 조사 결과를 게시 (환각 방지 가드레일 참조)

조사는 **최소 침해** 원칙 하에 수행되어야 합니다: 가설을 입증하거나 반증하는 데 필요한 증거만 수집하십시오. 결과를 게시할 때, 책임감 있는 취약점 공개 관행에 따르고 대중에 공개하기 전 관련된 메인테이너들과 조율하십시오.

조사 결과 실제 침해 사실이 밝혀진다면, 조율된 취약점 공개(coordinated vulnerability disclosure) 프로세스를 따르십시오:
1. 저장소 메인테이너에게 먼저 비공개로 알림
2. 문제 해결을 위한 합당한 시간(일반적으로 90일) 부여
3. 패키지가 발행된 경우 패키지 레지스트리(npm, PyPI 등)와 조율
4. 필요시 CVE 등록

---

## API 속도 제한 (Rate Limiting)

GitHub REST API는 관리가 되지 않을 경우 대규모 조사를 중단시킬 수 있는 속도 제한(Rate limits)을 적용합니다.

**인증된 요청**: 5,000건/시간 (`GITHUB_TOKEN` 환경변수 또는 `gh` CLI 인증 필요)
**인증되지 않은 요청**: 60건/시간 (조사용으로는 부적합)

**모범 사례**:
- 항상 인증할 것: `export GITHUB_TOKEN=ghp_...`를 하거나 `gh` CLI(자동 인증)를 사용
- 변경되지 않은 데이터를 불러와 할당량을 소모하지 않도록 조건부 요청(`If-None-Match` / `If-Modified-Since` 헤더) 사용
- 페이지네이션을 지원하는 엔드포인트의 경우 순차적으로 페이지를 가져올 것 — 동일한 엔드포인트에 대해 병렬 요청 자제
- `X-RateLimit-Remaining` 헤더를 확인할 것; 만약 100 미만으로 떨어졌다면 `X-RateLimit-Reset` 타임스탬프가 될 때까지 멈출 것
- BigQuery에는 자체적인 쿼터(무료 티어: 하루 10 TiB)가 존재함 — 항상 dry-run 먼저 수행
- Wayback Machine CDX API: 공식적인 제한은 없으나, 에티켓(초당 1~2회 요청 최대) 준수

조사 도중 속도 제한에 걸렸다면, 부분적인 결과를 증거 저장소에 기록하고 보고서에 해당 한계를 기재하십시오.

---

## 참조 자료

- [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md) — BigQuery 쿼리, CDX API, 12 이벤트 유형
- [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md) — IOC 분류, 증거 출처 유형, 관찰 유형
- [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md) — 삭제된 커밋, PR, 이슈 복구 방법
- [investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md) — 공격 유형별 기본 내장 가설 템플릿
- [evidence-store.py](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/scripts/evidence-store.py) — 증거 JSON 저장소 관리를 위한 CLI 도구
- [forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md) — 구조화된 보고서 템플릿
