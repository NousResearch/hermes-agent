---
title: "Microsoft Graph 애플리케이션 등록 (Register a Microsoft Graph Application)"
description: "Teams 회의 파이프라인을 구동하는 앱 등록을 생성하기 위한 Azure 포털 연습 가이드"
---

# Microsoft Graph 애플리케이션 등록 (Register a Microsoft Graph Application)

Teams 회의 파이프라인은 Microsoft Graph에서 회의 스크립트(transcripts), 녹화본 및 관련 아티팩트(artifacts)를 읽어옵니다. 이때 사용자 로그인이나 회의별 대화형 동의 없이 **앱 전용(app-only)** (데몬) 인증을 사용합니다. 이를 위해서는 관리자가 동의한 애플리케이션 권한을 갖춘 Azure AD 애플리케이션 등록이 필요합니다.

이 가이드에서는 다음 과정을 다룹니다:

1. 앱 등록 생성
2. 클라이언트 암호 생성
3. 파이프라인이 필요로 하는 Graph API 권한 부여
4. 해당 권한에 대한 관리자 동의 부여
5. (선택 사항) 애플리케이션 액세스 정책을 사용하여 앱 범위를 특정 사용자로 제한하기

이 과정을 완료하려면 **테넌트 관리자 권한**(또는 대신 동의를 부여해 줄 관리자)이 필요합니다. 수집한 값들은 북마크해 두세요 — 마지막에 이 값들은 `~/.hermes/.env` 파일에 들어가게 됩니다.

## 사전 요구 사항 (Prerequisites)

- 회의 스크립트 및 녹화본을 생성하는 Teams Premium 또는 Teams 라이선스가 포함된 Microsoft 365 테넌트
- [entra.microsoft.com](https://entra.microsoft.com)의 Azure 포털에 대한 관리자 액세스 권한
- Graph 변경 알림을 받기 위한 공개적으로 접속 가능한 HTTPS 엔드포인트 (이후 웹훅 리스너 설정 단계에서 세팅함)

## 1단계: 앱 등록 생성 (Create the App Registration)

1. 테넌트 관리자 자격으로 [entra.microsoft.com](https://entra.microsoft.com)에 로그인합니다.
2. **항목(Identity) → 애플리케이션(Applications) → 앱 등록(App registrations)** 으로 이동합니다.
3. **새 등록(New registration)** 을 클릭합니다.
4. 다음 정보를 입력합니다:
   - **이름(Name):** `Hermes Teams Meeting Pipeline` (또는 알아볼 수 있는 이름).
   - **지원되는 계정 유형(Supported account types):** *이 조직 디렉터리의 계정만 (단일 테넌트)*.
   - **리디렉션 URI(Redirect URI):** 비워 둡니다 — 앱 전용 인증에는 리디렉션 URI가 필요하지 않습니다.
5. **등록(Register)** 을 클릭합니다.

앱의 개요(Overview) 페이지로 이동하게 됩니다. 다음 두 값을 복사하세요:

- **애플리케이션(클라이언트) ID (Application (client) ID)** → `MSGRAPH_CLIENT_ID`
- **디렉터리(테넌트) ID (Directory (tenant) ID)** → `MSGRAPH_TENANT_ID`

## 2단계: 클라이언트 암호 생성 (Create a Client Secret)

1. 왼쪽 탐색 메뉴에서 **인증서 및 암호(Certificates & secrets)** 를 엽니다.
2. **새 클라이언트 암호(New client secret)** 를 클릭합니다.
3. **설명(Description):** `hermes-graph-secret`. **만료(Expires):** 조직의 교체 정책에 맞는 값을 선택합니다 (일반적으로 6-24개월).
4. **추가(Add)** 를 클릭합니다.
5. **값(Value)** 열의 내용을 즉시 복사하세요 — 이 값은 한 번만 표시됩니다. 이 값이 `MSGRAPH_CLIENT_SECRET`입니다.

> **암호 ID(Secret ID)** 열은 암호가 아닙니다. 반드시 **값(Value)** 열을 사용해야 합니다.

## 3단계: Graph API 권한 부여 (Grant Graph API Permissions)

파이프라인은 최소한의 필수 애플리케이션 권한 집합만 사용합니다. 필요한 권한만 추가하세요; 각 권한은 앱이 테넌트 전체에서 읽을 수 있는 범위를 넓히게 됩니다.

1. 왼쪽 탐색 메뉴에서 **API 권한(API permissions)** 을 엽니다.
2. **권한 추가(Add a permission) → Microsoft Graph → 애플리케이션 권한(Application permissions)** 을 클릭합니다.
3. 아래 표를 참고하여 파이프라인이 수행하고자 하는 작업에 맞는 권한을 추가합니다.
4. 권한을 추가한 후, **`<테넌트 이름>에 대한 관리자 동의 부여(Grant admin consent for <your tenant>)`** 를 클릭합니다. 모든 권한의 상태 열이 녹색 체크 표시로 바뀌어야 합니다.

### 스크립트 기반 요약 필수 권한 (Required for transcript-first summaries)

| 권한 (Permission) | 앱이 수행할 수 있는 작업 |
|------------|--------------------------|
| `OnlineMeetings.Read.All` | Teams 온라인 회의 메타데이터(주제, 참석자, 참여 URL) 읽기. |
| `OnlineMeetingTranscript.Read.All` | Teams에서 생성한 회의 스크립트 읽기. |

### 녹화본 대체 수단용 필수 권한 (Required for recording fallback)
(스크립트를 사용할 수 없을 때 사용)

| 권한 (Permission) | 앱이 수행할 수 있는 작업 |
|------------|--------------------------|
| `OnlineMeetingRecording.Read.All` | 오프라인 STT 처리를 위해 Teams 회의 녹화본 다운로드. |
| `CallRecords.Read.All` | 참여 URL만 알려진 경우 통화 기록에서 회의 정보 파악. |

### 아웃바운드 요약 전달 필수 권한 (Required for outbound summary delivery)
(Graph 모드 전용)

`platforms.teams.extra.delivery_mode`가 `graph`인 경우, 파이프라인은 Graph API를 통해 Teams 채널이나 채팅에 요약을 게시합니다. 대신 `incoming_webhook` 전달 모드를 사용하는 경우 이 부분은 건너뛰세요.

| 권한 (Permission) | 앱이 수행할 수 있는 작업 |
|------------|--------------------------|
| `ChannelMessage.Send` | 앱을 대신하여 Teams 채널에 메시지 게시. |
| `Chat.ReadWrite.All` | 1:1 및 그룹 채팅에 메시지 게시 (`chat_id`를 전달 대상(target)으로 설정한 경우에만). |

### 권장하지 않음 (Not recommended)

- `OnlineMeetings.ReadWrite.All` / `.All`이 없는 `Chat.ReadWrite` — 파이프라인이 필요로 하는 범위를 초과합니다.
- 위임된 권한 (Delegated permissions) — 파이프라인은 앱 전용(클라이언트 자격 증명) 흐름을 사용하므로 사용자 로그인 없이는 위임된 권한이 작동하지 않습니다.

## 4단계: (권장) 애플리케이션 액세스 정책을 사용하여 앱 범위 지정 (Scope the App with an Application Access Policy)

기본적으로 `OnlineMeetings.Read.All`과 같은 애플리케이션 권한은 앱에 테넌트 내의 **모든** 회의에 대한 액세스 권한을 부여합니다. 파트너 데모나 개발용 테넌트의 경우 괜찮지만 프로덕션 환경의 경우 대부분 앱이 읽을 수 있는 회의 사용자를 제한하려고 할 것입니다.

Microsoft는 이를 위해 정확하게 Teams용 **애플리케이션 액세스 정책(Application Access Policies)** 을 제공합니다. 이 정책은 PowerShell로만 적용할 수 있는 기능이며 포털 UI가 없습니다.

MicrosoftTeams 모듈이 설치되고 연결(`Connect-MicrosoftTeams`)된 관리자 PowerShell에서 실행하세요:

```powershell
# Hermes 앱의 범위를 제한하는 정책 생성
New-CsApplicationAccessPolicy `
  -Identity "Hermes-Meeting-Pipeline-Policy" `
  -AppIds "<MSGRAPH_CLIENT_ID>" `
  -Description "Restrict Hermes meeting pipeline to allow-listed users"

# 파이프라인이 회의 내용을 읽을 수 있도록 특정 사용자에게 정책 부여
Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "alice@example.com"

Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "bob@example.com"
```

부여 후 정책이 반영되기까지 최대 30분이 걸릴 수 있습니다. 다음 명령어로 확인하세요:

```powershell
Test-CsApplicationAccessPolicy -Identity "alice@example.com" -AppId "<MSGRAPH_CLIENT_ID>"
```

이 정책이 없으면 **모든** 사용자의 회의 내용을 읽을 수 있게 됩니다 — 이것이 해당 권한이 기술적으로 부여하는 방식입니다. 프로덕션 테넌트에서는 이 단계를 건너뛰지 마세요.

## 5단계: 환경 변수 파일에 자격 증명 쓰기 (Write the Credentials to Your Env File)

수집한 세 가지 값을 `~/.hermes/.env` 파일에 넣습니다:

```bash
MSGRAPH_TENANT_ID=<directory-tenant-id>
MSGRAPH_CLIENT_ID=<application-client-id>
MSGRAPH_CLIENT_SECRET=<client-secret-value>
```

사용자 본인만 시크릿을 읽을 수 있도록 파일 권한을 설정합니다:

```bash
chmod 600 ~/.hermes/.env
```

## 6단계: 토큰 흐름 확인 (Verify the Token Flow)

Hermes는 Graph 인증의 간단한 정상 작동 테스트(smoke-test)를 제공합니다. Hermes가 설치된 경로에서 실행하세요:

```python
python -c "
import asyncio
from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
provider = MicrosoftGraphTokenProvider.from_env()
token = asyncio.run(provider.get_access_token())
print('Token acquired, length:', len(token))
print(provider.inspect_token_health())
"
```

성공적으로 실행되면 긴 토큰 문자열과 함께 `cached: True` 및 3600에 가까운 `expires_in_seconds` 값을 보여주는 상태 딕셔너리가 출력됩니다. 실패 시 Azure 오류 코드와 함께 `MicrosoftGraphTokenError`가 발생합니다 — 가장 일반적인 오류들은 다음과 같습니다:

| Azure 오류 | 의미 | 해결책 |
|-------------|---------|-----|
| `AADSTS7000215: Invalid client secret` | 시크릿 값이 일치하지 않거나 만료됨. | 2단계에서 새 시크릿을 생성하고 `.env`를 업데이트하세요. |
| `AADSTS700016: Application not found` | 잘못된 `MSGRAPH_CLIENT_ID` 또는 잘못된 테넌트. | 1단계의 값들이 같은 앱에서 온 것인지 다시 확인하세요. |
| `AADSTS90002: Tenant not found` | `MSGRAPH_TENANT_ID` 오타. | 앱 개요 페이지에서 디렉터리(테넌트) ID를 다시 복사하세요. |
| 호출 시점에 `insufficient_claims` 발생 (토큰 발급 시점이 아님) | 토큰은 발급되었으나 Graph에서 401/403을 반환함. | 3단계의 관리자 동의를 건너뛰었거나, 권한을 추가한 후 다시 동의하지 않았습니다. API 권한으로 돌아가서 **관리자 동의 부여**를 다시 클릭하세요. |

## 클라이언트 암호 교체 (Rotating the Client Secret)

Azure 클라이언트 암호에는 하드 만료 기한이 있습니다. 기존 암호가 만료되기 전에 다음 절차를 따르세요:

1. 첫 번째 시크릿을 삭제하지 말고 2단계처럼 두 번째 클라이언트 시크릿을 생성합니다.
2. `~/.hermes/.env`의 `MSGRAPH_CLIENT_SECRET`을 새 값으로 업데이트합니다.
3. 새 시크릿이 적용되도록 게이트웨이를 다시 시작합니다: `hermes gateway restart`.
4. 위의 스모크 테스트 코드를 통해 작동을 확인합니다.
5. Azure 포털에서 기존(오래된) 시크릿을 삭제합니다.

## 다음 단계 (Next Steps)

자격 증명 확인이 정상적으로 완료되면 다음 단계로 진행하세요:

- **웹훅 리스너 설정** — Graph 변경 알림을 수신하는 `msgraph_webhook` 게이트웨이 플랫폼을 구축합니다.
- **파이프라인 설정** — Teams 회의 파이프라인 런타임과 운영자 CLI를 설정합니다.
- **아웃바운드 전달** — 요약본을 Teams 채널이나 채팅으로 다시 전송하도록 연동합니다.

이러한 문서 페이지들은 해당하는 런타임 추가 PR과 함께 위치합니다. 이번 인증 설정 가이드는 독립적인 선행 조건이므로 미리 완료해두어도 안전합니다.
