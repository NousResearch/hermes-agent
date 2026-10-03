---
title: "GitHub Issue To Pr — GitHub 이슈를 정직한 CI 상태와 함께 검증된 PR로 연결하기"
sidebar_label: "GitHub Issue To Pr"
description: "GitHub 이슈를 정직한 CI 상태와 함께 검증된 PR로 연결하기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# GitHub Issue To Pr

GitHub 이슈를 정직한 CI 상태와 함께 검증된 PR로 연결합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨 (기본 설치) |
| 경로 | `skills/github/github-issue-to-pr` |
| 버전 | `0.1.0` |
| 작성자 | Ben Barclay (benbarclay), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `GitHub`, `Issues`, `Coding`, `Pull-Requests`, `CI` |
| 관련 스킬 | [`github-issues`](/docs/user-guide/skills/bundled/github/github-github-issues), [`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow), [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging), [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보게 되는 지침입니다.
:::

# GitHub 이슈를 풀 리퀘스트로

GitHub 이슈를 테스트하고 검증한 PR로 전환합니다. 이 스킬은 전제 검증, 중복 작업 점검, 클래스 수준의 수정, 정직한 CI 보고를 포함한 엔드투엔드 규율을 담당합니다. GitHub 및 개발 관련 자매 스킬은 각자의 세부 작업을 담당합니다.

## 사용 시점

- "이슈 #123을 수정하고 PR을 열어 줘."
- "이 GitHub 기능 요청을 구현해 줘."
- "이 버그를 이슈에서 CI 통과 상태까지 처리해 줘."

다음에는 사용하지 않습니다: 기존 PR 검토 또는 변경 요청 없이 코드 질문에 답하는 경우.

## 절차

### 1. 현재 이슈 읽기 — 본문과 전체 스레드

`gh issue view <N> --comments`를 실행하려면 `terminal`을 사용합니다. 본문은 이슈를 등록할 당시의 스냅샷이고, 최신 댓글에는 이미 병합된 부분 수정, 새로운 근본 원인 분석, 유지 관리자의 결정, 또는 작업을 바꾸는 질문이 담겨 있습니다. 또한 저장소 지침(`AGENTS.md`, 기여 문서)을 `read_file`로 읽습니다. 현재 요청된 동작, 제외 범위, 답변되지 않은 스레드 질문을 파악하면 완료입니다.

### 2. 기존 작업 및 중복 작업 점검

작업을 작성하기 전에 `gh pr list --search "#<N>" --state all`을 실행하고, 증상의 키워드나 동의어를 두 가지 이상 사용해 `gh pr list --search "<subsystem> <symptom>" --state open`을 실행합니다. 인기 있는 이슈에는 여러 독립적인 수정이 올라오는 경우가 많으므로, 중복 작업을 만들면 기존 작업과 기여 기록을 낭비하게 됩니다. 최근 커밋에서 이미 수정되었는지도 확인합니다: `git log --oneline -20 -- <relevant files>`. 열려 있는 모든 PR과 이슈를 다루는 최근 커밋을 파악했거나, 그런 항목이 없음을 확인하면 완료입니다.

### 3. 현재 코드와 설계 의도를 기준으로 전제 검증

`search_files`와 `read_file`을 사용해 보고된 경로를 추적하면서, 현재 기본 브랜치에서 실패하는 테스트나 픽스처로 버그를 재현하거나 누락된 동작을 입증합니다. 그런 다음 두 번째 질문을 확인합니다. 이 "버그"가 실제로 의도된 설계인가? 이슈가 변경하려는 코드에서 `git log -p -S "<symbol>"`을 실행하고 원래 커밋의 의도를 읽습니다. 누락된 연결이나 제한은 기능인 경우가 많습니다. 현재 코드에서 근본 원인 또는 기능 공백이 입증되었고, 변경이 의도된 설계와 충돌하지 않음을 확인하면 완료입니다.

### 4. 수용 기준과 위험 정의

수용 기준, 인터페이스, 마이그레이션/상태 변경, 호환성, 보안/개인정보 보호, 출시, 롤백을 나열합니다. 각 기준을 테스트 또는 명시적인 검증에 연결합니다. 검토 가능한 유한한 계약이 마련되면 완료입니다.

### 5. 가장 작지만 완전한 변경 구현 — 그리고 클래스 전체 수정

격리된 브랜치 또는 worktree에서 작업하고, 버그 클래스에 해당하면 `systematic-debugging` 또는 `test-driven-development`를 로드합니다. 먼저 회귀 테스트를 추가한 다음 구현합니다. 수정이 완료되면 동일한 버그 형태가 있는 자매 호출 지점을 `search_files`로 찾아 이 PR에서 클래스 전체를 수정합니다. 알려진 자매 지점이 계속 고장 난 채 남는 불완전한 수정은 수정하지 않는 것보다 나쁩니다. 변경된 모든 줄은 이슈와 직접 연결되어야 하며, 별도의 정리 작업을 끼워 넣지 않습니다. 대상 테스트가 통과하고 원래 실패가 더 이상 재현되지 않으며, 자매 지점이 수정되었거나 명시적으로 제외되면 완료입니다.

### 6. 회귀 테스트가 실제로 회귀를 잡는지 입증 (사보타주 실행)

테스트 대상 정확한 함수의 이전 동작을 임시로 복원하고 새 테스트를 실행해 실패하는지 확인합니다. 그런 다음 수정 사항을 복원하고 통과하는지 확인합니다. 수정 전후 모두 통과하는 회귀 테스트는 아무것도 입증하지 못합니다. 테스트가 수정 전 코드에서 명확히 실패함을 확인하면 완료입니다.

### 7. 저장소 품질 검사를 실행한 후 즉시 PR 열기

영향을 받는 영역에서 포매터, 린트, 타입 검사, 저장소의 표준 테스트 진입점을 실행하고, diff에 `requesting-code-review`를 사용합니다. 그런 다음 즉시 push하고 PR을 엽니다. PR이 CI를 실행시키며 CI 지연이 가장 오래 걸리는 단계이므로, 완료된 작업을 붙잡고 기다리지 않습니다. PR 작업 절차에는 `github-pr-workflow`를 로드합니다. 규칙에 맞는 브랜치/커밋, 문제·접근 방식·테스트·위험·제외 사항을 담은 이슈 연결 본문을 사용합니다. PR을 다시 읽고 head SHA, base, 제목, 파일을 검증합니다. 의도한 diff와 함께 PR이 존재하고 CI가 실행 중이면 완료입니다.

### 8. CI를 정직하게 관리하고 마무리

`gh pr checks` / `gh run view --log-failed`를 통해 현재 검사와 실패 로그를 확인합니다. diff로 인해 발생한 실패와 기존 기준선 또는 인프라 실패를 구분합니다. 확실하지 않으면 기본 브랜치에서 재현하고, 실제 인프라 일시 오류인 경우에만 한 번 재실행합니다. 해당 상태에 대한 실시간 근거 없이 "통과했다", "병합됐다", "출시됐다"고 말하지 않습니다. PR이 반영되면 PR 링크와 한 줄 설명을 이슈에 댓글로 남겨 신고자가 추적 가능한 해결 경로를 확인할 수 있도록 합니다. CI 상태, 남은 차단 요소, 이슈 스레드가 모두 실제 상태를 반영하면 완료입니다.

## 주의할 점

- 이슈 댓글을 읽고, 중복 PR을 점검하고, 현재 코드를 읽기 전에 코딩하는 것.
- 원래 커밋에서 의도된 설계임을 보여 주는 동작을 "수정"하는 것.
- 한 호출 지점의 증상만 수정하고 자매 지점에는 같은 버그를 남겨 두는 것.
- 수정 없이도 통과하는 회귀 테스트를 배포하는 것.
- 테스트를 실행하지 않았거나 관련 없는 포매팅 변경이 섞인 PR을 여는 것.
- PR이 존재한다는 이유만으로 이슈가 해결되었다고 주장하는 것.

## 검증

- [ ] 전체 이슈 스레드를 읽고 최신 댓글의 상태를 계획에 반영함.
- [ ] 이슈 번호와 키워드 변형 두 가지를 사용해 중복 PR 점검을 수행함.
- [ ] 현재 코드에서 전제를 재현하고 git 이력을 통해 설계 의도를 확인함.
- [ ] 수정 없이 회귀 테스트가 실패함을 입증함.
- [ ] 자매 호출 지점을 수정했거나 명시적으로 제외함.
- [ ] 변경된 모든 줄이 이슈와 직접 연결됨.
- [ ] CI 상태를 실시간 근거로만 보고하고 PR 링크를 이슈에 댓글로 남김.
