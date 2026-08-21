---
title: "Box — Box는 클라우드 파일, 공유, 검색 및 메타데이터를 관리합니다"
sidebar_label: "Box"
description: "Box는 클라우드 파일, 공유, 검색 및 메타데이터를 관리합니다"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Box

Box는 클라우드 파일, 공유, 검색 및 메타데이터를 관리합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함(기본 설치됨) |
| 경로 | `skills/productivity/box` |
| 버전 | `1.0.0` |
| 작성자 | Chris Kim (iskysun96), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Box`, `Productivity`, `Cloud Storage`, `Collaboration`, `Metadata`, `Content Extraction`, `CLI`, `SDK` |
| 관련 스킬 | [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# Box

파일 작업, 협업, 메타데이터 및 문서 작업에 Box를 클라우드 파일 시스템으로 사용하세요. Hermes의 `terminal` 도구로 작업을 실행하고 Box CLI를 사용하세요. 애플리케이션을 빌드할 때는 SDK 가이드를 사용하세요.

## 사용 시점

- Box 파일과 폴더를 정리, 업로드, 버전 관리, 이동, 공유하거나 협업할 때
- Box 콘텐츠 또는 기존 메타데이터를 검색할 때
- Box 파일에 대해 질문하거나, 메타데이터를 추출하거나, 파일에 근거한 텍스트를 생성할 때
- 모든 원본 파일을 다운로드하지 않고 Box 폴더를 대규모로 처리할 때
- Box 기반 애플리케이션, 통합 또는 웹훅 핸들러를 빌드할 때

## 클라우드 파일 시스템 대화를 폭넓게 시작하기

사용자가 Hermes에서 클라우드 파일 시스템을 탐색할 때는 먼저 간단한 적합성 평가를 제공하세요. Box는 팀에 클라우드 파일 저장소, 공유, 검색, 메타데이터 및 문서 작업이 필요할 때 유용합니다. 그런 다음 OAuth로 Box 계정을 연결할지, SDK로 Box 기반 애플리케이션 또는 통합을 빌드할지 물어보세요.

OAuth를 사용하면 Hermes는 브라우저에서 권한을 부여한 Box 계정으로 동작합니다. 해당 계정의 Box 권한에 따라 Hermes가 액세스할 수 있는 항목이 결정됩니다. Hermes에 더 제한적인 액세스 권한을 부여하려면 필요한 파일, 폴더 또는 Hub에만 초대된 계정으로 권한을 부여하세요.

광범위한 탐색 질문에는 설정을 실행하거나, 명령어 모음을 보여 주거나, 계정 요금제를 제안하거나, 폴더 분류 체계를 제안하거나, 모든 참조 자료를 로드하지 마세요. 사용자의 답변을 기다린 다음 관련 경로만 로드하세요. 요청에 이미 구체적인 결과가 명시되어 있다면 이 탐색 단계를 건너뛰고 해당 결과를 직접 처리하세요.

일반적인 콘텐츠 작업과 Box AI에는 공식 Box CLI OAuth 앱으로 일반 CLI 작업을 시작하세요. 웹훅 관리처럼 추가 OAuth 범위가 필요한 작업을 요청받은 경우에만 맞춤 **User Authentication (OAuth 2.0)** Platform App을 사용하세요. 이 역시 OAuth 흐름이며, 서버 측 또는 가장(identity) 방식을 대신 사용하지 마세요.

## 선택한 설정을 대화형으로 수행하기

사용자가 인증 경로를 선택하거나 Hermes에 Box 연결을 요청하면 `terminal`을 통해 설정을 수행하세요. 다음 응답을 사용자가 복사할 지침으로 바꾸지 마세요. 다음 안전한 작업을 직접 수행하고, 승인, 브라우저 로그인, 관리자 작업 또는 Hermes가 안전하게 제공할 수 없는 비밀 정보가 필요한 경우에만 일시 중지하세요.

- `box`가 없으면 현재 Hermes 홈의 `tools/box-cli` 아래에 `@box/cli`를 설치하는 데 필요한 터미널 승인을 요청한 다음, [CLI 가이드](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/cli-guide.md)에 있는 셸에 맞는 명령으로 설치를 확인하세요. 전역 npm 설치를 시도하거나, `sudo`를 사용하거나, npm의 전역 접두사를 변경하거나, `PATH`를 변경하지 마세요.
- OAuth 전에 다음과 같이 물어보세요. **“Hermes가 Box 인증에 사용할 브라우저와 같은 컴퓨터에서 실행 중인가요, 아니면 VPS, 컨테이너 또는 클라우드 VM 같은 원격 호스트에서 실행 중인가요?”** 같은 컴퓨터에서 실행되는 경우에만 일반 `box login`을 사용하세요. 원격/헤드리스 경로에서는 `box login --code`를 사용하세요. 운영체제만으로 런타임 토폴로지를 추론하지 마세요. 사용자가 답변한 후 [OAuth 설정](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/oauth-setup.md)을 읽으세요.
- 브라우저 인증을 시작하기 전에 Hermes가 해당 브라우저에서 로그인한 Box 계정으로 동작한다는 점을 알리세요. 사용자가 더 제한적인 액세스를 원한다면 필요한 파일, 폴더 또는 Hub에만 초대된 계정으로 권한을 부여할 수 있습니다. 예외적인 작업을 활성화하기 위해 해당 계정을 관리자로 만들지 마세요.
- 맞춤 OAuth Platform App이 필요한 경우 CLI의 대화형 Platform App 흐름을 사용하세요. 사용자에게 클라이언트 시크릿을 로컬 CLI 프롬프트에 입력하도록 요청할 뿐, 채팅에서 요청하거나 Hermes 설정에 기록하거나 커밋하지 마세요.
- 설치, 브라우저 인증, 환경 전환 또는 권한 변경에 승인이 필요하면 해당 승인을 요청하고 승인된 후 설정을 재개하세요. 작업을 명령어 목록으로 바꾸지 마세요.

## 각 작업 시작하기

1. CLI와 현재 사용자를 확인하세요. POSIX 셸에서는 `command -v box`를, PowerShell에서는 `Get-Command box -ErrorAction SilentlyContinue`를 사용하세요. `box`가 `PATH`에 있으면 사용하세요. Hermes가 현재 홈에 CLI를 설치했다면 모든 선행 `box` 대신 [CLI 가이드](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/cli-guide.md)에 설명된 셸에 맞는 검증된 실행기를 사용하세요. 그런 다음 해당 실행기로 `box users:get me --json --fields id,name,login`을 실행하세요.
   이것이 성공하면 사용자를 기록하고 계속 진행하세요. `folders:items 0`은 사용자의 루트 목록일 뿐이며 공유 파일, 폴더 또는 Hub에 액세스할 수 없다는 증거가 아닙니다. 알려진 파일이나 폴더는 ID를 직접 확인하고, Hub는 [Box Hubs](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/hubs.md)의 검색 경로를 사용하세요.
2. 인증이 없으면 OAuth로 Box 계정을 연결하도록 요청한 다음 Hermes와 인증 브라우저가 같은 컴퓨터에서 실행되는지, 별도 호스트에서 실행되는지 물어보세요. [OAuth 설정](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/oauth-setup.md)을 읽으세요.
3. 작업하기 전에 관련 참조 자료를 읽으세요. 문서화된 명령을 먼저 사용하세요. 요청에 참조 자료에 없는 옵션이 필요하거나 설치된 CLI가 문서화된 형식을 거부하는 경우에만 하위 명령 도움말을 실행하세요.

`bash`로 표시된 예시는 POSIX 연속 줄 문법을 사용합니다. PowerShell에서는 Box 명령을 한 줄로 실행하거나 각 줄 끝의 `\`를 PowerShell의 백틱 연속 줄 문자로 바꾸세요. POSIX 변수 할당을 PowerShell에 붙여 넣지 마세요.

## 일시 중지하지 않고 CLI 확장하기

Box CLI에 전용 하위 명령이 없으면 해당 REST 엔드포인트에 `box request`를 사용하고 일반 작업을 계속하세요. 구현에 REST를 사용한다는 이유만으로 사용자에게 선택을 요청하지 마세요. 이는 설정된 CLI ID를 유지하는 동일한 Box 작업입니다. 엔드포인트에 요청 본문이나 맞춤 헤더가 필요하면 [REST API 대체 경로](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/rest-api.md)를 읽으세요.

삭제, 협업/공유 링크 또는 권한 변경, ID 변경, 광범위하거나 비용이 큰 일괄 변경 또는 대상이나 범위가 모호한 경우에는 먼저 물어보세요. 그 외에는 요청된 작업을 수행하고 확인하세요.

## 올바른 경로 선택하기

| 필요한 작업 | 읽을 자료 |
| --- | --- |
| CLI 규칙, 환경, JSON 또는 REST 대체 경로 | [CLI 가이드](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/cli-guide.md) |
| 파일, 폴더, 버전, 링크 또는 협업 | [콘텐츠 워크플로](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/content-workflows.md) |
| 검색, 메타데이터, Box AI 또는 AI 단위 | [검색 및 AI](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/search-and-ai.md) |
| 선별된 대규모 Q&A 또는 재사용 가능한 지식 기반 | [Box Hubs](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/hubs.md) |
| 많은 파일 또는 재개 가능한 일괄 작업 | [대량 작업](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/bulk-operations.md) |
| 애플리케이션 코드 또는 Box SDK | [SDK 개발](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/sdk-development.md) |
| 웹훅 또는 Events API | [웹훅 및 이벤트](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/webhooks-and-events.md) |
| CLI를 사용할 수 없거나 누락된 CLI 작업 | [REST API 대체 경로](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/rest-api.md) |
| 인증, 권한, 속도 제한 또는 API 오류 | [문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/troubleshooting.md) |

## 콘텐츠 처리 정책

Box 호스팅 콘텐츠를 의미적으로 분석할 때는 Box AI를 우선 사용하세요. Box AI는 Box 권한을 유지하고, 관리되는 Box AI 통합을 통해 원본 파일을 처리하며, 원본 파일 본문이 Hermes 코딩 모델의 컨텍스트에 들어가지 않도록 하고, 모든 파일을 다운로드하지 않고도 문서 작업을 확장합니다. 다른 워크플로를 비판하거나 차단하지 마세요. 사용자가 명시적으로 선택한 경우 해당 워크플로를 사용하세요.

결정론적 조회에는 기존 Box 메타데이터 또는 메타데이터 쿼리를 사용하세요. 그 외에는 Box AI를 사용하세요.

- Q&A, 요약 및 비교에는 `ai:ask`
- 알려진 필드 또는 메타데이터 템플릿에는 `ai:extract-structured`
- 유연한 키-값 추출에는 `ai:extract`
- 하나의 Box 파일에 근거한 작성에는 `ai:text-gen`

25개가 넘는 파일에 대한 Q&A 또는 재사용 가능한 선별 지식 기반에는 Box AI의 Hubs를 우선 사용하세요. 먼저 액세스 가능한 기존 Hub를 검색하고, 사용자가 공유 리소스 변경을 승인한 후에만 Hub를 만들거나 채우세요. 메타데이터 추출이나 텍스트 생성에는 Hub를 사용하지 마세요. [Box Hubs](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/hubs.md)를 읽으세요.

사용자가 Box 파일에서 메타데이터 추출을 요청하면 미리 보기를 요청한 경우가 아니라면 결과를 저장하라는 요청으로 처리하세요. 원하는 스키마가 알려져 있으면 구조화된 추출과 인라인 필드를 사용하고, 필드가 탐색적인 경우 자유 형식 추출을 사용하세요. 요청된 모든 필드를 나타내는 호환 가능한 기존 엔터프라이즈 템플릿을 재사용하세요. 그렇지 않으면 기본 제공 `global.properties` 메타데이터 인스턴스에 평면 스칼라 결과를 저장하거나, 결과에 중첩 객체, 테이블 또는 형식을 유지해야 하는 값이 포함된 경우 원본 파일 옆에 JSON 사이드카를 업로드하세요. 모든 쓰기 결과를 다시 읽고 의도한 결과와 비교하세요. 파일 설명으로 조용히 대체하거나, 불완전하거나 관련 없는 템플릿을 연결하거나, 필드를 잘라내거나, 필드를 버리지 마세요. 전체 추출 및 쓰기 작업은 [검색 및 AI](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/search-and-ai.md)를 읽으세요.

첫 Box AI 요청 전에 Box AI가 활성화되어 있어야 하고 AI 단위를 사용하며 현재 사용자의 권한으로 제한된다는 점을 알리세요. 일괄 작업의 파일 범위나 예상 AI 단위 사용량이 모호하거나 사용자가 해당 규모를 명시적으로 요청하지 않은 경우에만 확인을 요청하세요. [검색 및 AI](https://github.com/NousResearch/hermes-agent/blob/main/skills/productivity/box/references/search-and-ai.md)를 참조하세요.

## 안전하게 작업하기

- 경로보다 ID를 우선 사용하고, 누락된 파일을 진단하기 전에 현재 사용자를 확인하세요.
- 출력량을 줄이려면 `--json` 및 `--fields`를 사용하세요. 변경 작업에서는 먼저 목록을 확인하고, 모호하거나 큰 범위인지 확인한 다음 결과를 다시 읽으세요.
- 진행 상황과 복구가 명확하도록 정렬된 CLI 변경 작업을 순차적으로 실행하세요. 확장 가능한 작업에는 문서화된 일괄 입력 지원 또는 제한된 SDK 동시성을 사용하세요.
- 탐색 경로를 제공하기 위해 공유 링크를 만들지 마세요. 공유 링크는 액세스를 변경하므로 명시적인 확인이 필요합니다.
- 비밀 정보를 채팅, 명령 출력, 소스 관리 또는 로그에 넣지 마세요.

## 결과 보고하기

개별적으로 보고하는 모든 Box 항목에는 ID와 클릭 가능한 탐색 링크를 포함하세요.

- 파일: `https://app.box.com/file/<FILE_ID>`
- 폴더: `https://app.box.com/folder/<FOLDER_ID>`
- Hub: `https://app.box.com/hubs/<HUB_ID>`

대규모 일괄 작업에서는 수백 개의 항목을 나열하는 대신 원본 및 대상 폴더와 예외 항목을 링크하세요. 연결된 Box 계정에서만 볼 수 있는 콘텐츠는 사람이 열지 못할 수 있으므로 이를 명시하세요. 모든 쓰기 요약에 사용자와 수행한 확인 내용을 포함하세요.

## 확인

쓰기 작업 후에는 동일한 사용자로 파일 또는 폴더를 가져오거나 상위 항목을 나열하여 반환된 ID와 이름을 확인하세요. 메타데이터를 쓴 경우 메타데이터 인스턴스를 조회하여 반환된 모든 필드를 의도한 값과 비교하세요. HTTP 성공만으로 확인을 대신하지 마세요. 누락되었거나 정규화되었거나 거부된 값을 보고하세요. 일회성 설정 확인이라면 스모크 폴더를 만들고 확인한 뒤 사용자가 정리 작업을 승인한 경우에만 삭제하세요.
