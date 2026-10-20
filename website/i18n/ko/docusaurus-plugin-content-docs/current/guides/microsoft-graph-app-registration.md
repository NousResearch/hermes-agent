---
title: "Microsoft Graph 애플리케이션 등록"
description: "Teams 회의 파이프라인을 구동하는 앱 등록을 생성하기 위한 Azure 포털 안내"
---

# Microsoft Graph 애플리케이션 등록

Teams 회의 파이프라인은 **앱 전용**(데몬) 인증을 사용하여 Microsoft Graph에서 회의 트랜스크립트, 녹화 및 관련 아티팩트를 읽습니다. 사용자 로그인이나 회의마다의 대화형 동의가 필요하지 않습니다. 이를 위해서는 관리자가 동의한 애플리케이션 권한이 포함된 Azure AD 애플리케이션 등록이 필요합니다.

이 가이드에서는 다음을 안내합니다.

1. 앱 등록 생성
2. 클라이언트 시크릿 생성
3. 파이프라인에 필요한 Graph API 권한 부여
4. 해당 권한에 대한 관리자 동의
5. (선택 사항) Application Access Policy를 사용하여 특정 사용자로 앱 범위 제한

이를 완료하려면 **테넌트 관리자 권한**(또는 대신 동의를 부여해 줄 관리자)이 필요합니다. 수집하는 값을 기록해 두세요. 마지막에 `~/.hermes/.env`에 입력합니다.

## 사전 요구 사항

- Teams 트랜스크립트와 녹화를 생성하는 Teams Premium 또는 Teams 라이선스가 있는 Microsoft 365 테넌트
- [entra.microsoft.com](https://entra.microsoft.com)의 Azure 포털 관리자 액세스
- Graph 변경 알림을 위한 공개 접근 가능한 HTTPS 엔드포인트(웹훅 리스너 단계에서 나중에 설정)

## 1단계: 앱 등록 생성

1. 테넌트 관리자로 [entra.microsoft.com](https://entra.microsoft.com)에 로그인합니다.
2. **Identity → Applications → App registrations**로 이동합니다.
3. **New registration**을 클릭합니다.
4. 다음을 입력합니다.
   - **Name:** `Hermes Teams Meeting Pipeline`(또는 알아보기 쉬운 이름)
   - **Supported account types:** *Accounts in this organizational directory only (Single tenant)*
   - **Redirect URI:** 비워 둡니다. 앱 전용 인증에는 필요하지 않습니다.
5. **Register**를 클릭합니다.

앱의 개요 페이지로 이동합니다. 다음 두 값을 복사하세요.

- **Application (client) ID** → `MSGRAPH_CLIENT_ID`
- **Directory (tenant) ID** → `MSGRAPH_TENANT_ID`

## 2단계: 클라이언트 시크릿 생성

1. 왼쪽 탐색 메뉴에서 **Certificates & secrets**를 엽니다.
2. **New client secret**를 클릭합니다.
3. **Description:** `hermes-graph-secret`. **Expires:** 시크릿 교체 정책에 맞는 값을 선택합니다(일반적으로 6~24개월).
4. **Add**를 클릭합니다.
5. **Value** 열을 즉시 복사합니다. 한 번만 표시됩니다. 이 값이 `MSGRAPH_CLIENT_SECRET`입니다.

> **Secret ID** 열은 시크릿이 아닙니다. **Value** 열의 값이 필요합니다.

## 3단계: Graph API 권한 부여

파이프라인은 애플리케이션 권한의 최소 필요 집합을 사용합니다. 필요한 것만 추가하세요. 권한을 추가할 때마다 앱이 테넌트 전체에서 읽을 수 있는 범위가 넓어집니다.

1. 왼쪽 탐색 메뉴에서 **API permissions**를 엽니다.
2. **Add a permission** → **Microsoft Graph** → **Application permissions**를 클릭합니다.
3. 파이프라인에서 수행하려는 작업에 맞는 아래 표의 권한을 추가합니다.
4. 추가한 후 **Grant admin consent for `<your tenant>`**를 클릭합니다. 모든 권한의 Status 열이 녹색 체크 표시로 바뀌어야 합니다.

### 트랜스크립트 우선 요약에 필요

| 권한 | 앱이 수행할 수 있는 작업 |
|------------|--------------------------|
| `OnlineMeetings.Read.All` | Teams 온라인 회의 메타데이터(제목, 참가자, 참가 URL)를 읽습니다. |
| `OnlineMeetingTranscript.Read.All` | Teams가 생성한 회의 트랜스크립트를 읽습니다. |

### 녹화 폴백에 필요(트랜스크립트를 사용할 수 없는 경우)

| 권한 | 앱이 수행할 수 있는 작업 |
|------------|--------------------------|
| `OnlineMeetingRecording.Read.All` | 오프라인 STT 처리를 위해 Teams 회의 녹화를 다운로드합니다. |
| `CallRecords.Read.All` | 참가 URL만 알고 있을 때 통화 기록에서 회의를 확인합니다. |

### 아웃바운드 요약 전달에 필요(Graph 모드만 해당)

`platforms.teams.extra.delivery_mode`가 `graph`이면 파이프라인은 Graph API를 통해 Teams 채널 또는 채팅에 요약을 게시합니다. `incoming_webhook` 전달 모드를 사용하는 경우에는 이 권한을 건너뛰세요.

| 권한 | 앱이 수행할 수 있는 작업 |
|------------|--------------------------|
| `ChannelMessage.Send` | 앱을 대신하여 Teams 채널에 메시지를 게시합니다. |
| `Chat.ReadWrite.All` | 1:1 및 그룹 채팅에 메시지를 게시합니다(전달 대상으로 `chat_id`를 설정한 경우에만). |

### 권장하지 않음

- `OnlineMeetings.ReadWrite.All` / `.All`이 없는 `Chat.ReadWrite` — 파이프라인에 필요한 범위보다 넓습니다.
- 위임된 권한 — 파이프라인은 앱 전용(client-credentials) 흐름을 사용하므로, 위임된 권한은 사용자 로그인 없이는 작동하지 않습니다.

## 4단계: (권장) Application Access Policy로 앱 범위 제한

기본적으로 `OnlineMeetings.Read.All`과 같은 애플리케이션 권한은 앱에 테넌트의 **모든** 회의에 대한 액세스 권한을 부여합니다. 파트너 데모와 개발 테넌트에서는 괜찮지만, 프로덕션에서는 앱이 읽을 수 있는 사용자 회의를 제한하는 것이 거의 확실히 좋습니다.

Microsoft는 이를 위해 Teams 전용 **Application Access Policies**를 제공합니다. 이 정책은 PowerShell 전용 기능이며 포털 UI는 없습니다.

MicrosoftTeams 모듈이 설치되고 연결된(`Connect-MicrosoftTeams`) 관리자 PowerShell에서 다음을 실행합니다.

```powershell
# Create a policy scoped to the Hermes app
New-CsApplicationAccessPolicy `
  -Identity "Hermes-Meeting-Pipeline-Policy" `
  -AppIds "<MSGRAPH_CLIENT_ID>" `
  -Description "Restrict Hermes meeting pipeline to allow-listed users"

# Grant the policy to specific users whose meetings the pipeline may read
Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "alice@example.com"

Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "bob@example.com"
```

부여 후 전파까지 최대 30분이 걸릴 수 있습니다. 다음으로 확인하세요.

```powershell
Test-CsApplicationAccessPolicy -Identity "alice@example.com" -AppId "<MSGRAPH_CLIENT_ID>"
```

이 정책이 없으면 **모든** 사용자의 회의를 읽을 수 있습니다. 기술적으로 권한이 그렇게 부여하기 때문입니다. 프로덕션 테넌트에서는 이 단계를 건너뛰지 마세요.

## 5단계: 환경 파일에 자격 증명 기록

수집한 세 가지 값을 `~/.hermes/.env`에 입력합니다.

```bash
MSGRAPH_TENANT_ID=<directory-tenant-id>
MSGRAPH_CLIENT_ID=<application-client-id>
MSGRAPH_CLIENT_SECRET=<client-secret-value>
```

시크릿을 본인만 읽을 수 있도록 파일 권한을 설정합니다.

```bash
chmod 600 ~/.hermes/.env
```

## 6단계: 토큰 흐름 검증

Hermes에는 Graph 인증 스모크 테스트가 포함되어 있습니다. Hermes 설치 환경에서 실행하세요.

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

성공하면 긴 토큰 문자열과 `cached: True`, 그리고 3600에 가까운 `expires_in_seconds` 값을 보여 주는 상태 dict가 출력됩니다. 실패하면 Azure 오류 코드가 포함된 `MicrosoftGraphTokenError`가 발생합니다. 가장 흔한 오류는 다음과 같습니다.

| Azure 오류 | 의미 | 해결 방법 |
|-------------|---------|-----|
| `AADSTS7000215: Invalid client secret` | 시크릿 값이 일치하지 않거나 만료되었습니다. | 2단계에서 새 시크릿을 생성하고 `.env`를 업데이트합니다. |
| `AADSTS700016: Application not found` | `MSGRAPH_CLIENT_ID`가 잘못되었거나 테넌트가 잘못되었습니다. | 1단계의 값이 같은 앱에서 나온 것인지 다시 확인합니다. |
| `AADSTS90002: Tenant not found` | `MSGRAPH_TENANT_ID`에 오타가 있습니다. | 앱 개요에서 Directory (tenant) ID를 다시 복사합니다. |
| 호출 시점의 `insufficient_claims`(토큰 시점이 아님) | 토큰은 발급되지만 Graph가 401/403을 반환합니다. | 3단계의 관리자 동의를 건너뛰었거나 권한을 추가하고 다시 동의하지 않았습니다. API permissions로 돌아가 **Grant admin consent**를 다시 클릭합니다. |

## 클라이언트 시크릿 교체

Azure 클라이언트 시크릿에는 만료 기한이 있습니다. 시크릿이 만료되기 전에 다음을 수행하세요.

1. 첫 번째 시크릿을 삭제하지 말고 2단계에서 두 번째 클라이언트 시크릿을 생성합니다.
2. 새 값으로 `~/.hermes/.env`의 `MSGRAPH_CLIENT_SECRET`을 업데이트합니다.
3. 새 시크릿을 읽도록 gateway를 다시 시작합니다: `hermes gateway restart`.
4. 위의 스모크 테스트로 확인합니다.
5. Azure 포털에서 기존 시크릿을 삭제합니다.

## 다음 단계

자격 증명이 문제없이 검증되면 다음을 진행하세요.

- **웹훅 리스너 설정** — Graph 변경 알림을 수신하는 `msgraph_webhook` gateway 플랫폼을 구동합니다.
- **파이프라인 설정** — Teams 회의 파이프라인 런타임과 운영자 CLI를 설정합니다.
- **아웃바운드 전달** — 요약을 Teams 채널 또는 채팅으로 다시 전달하도록 연결합니다.

해당 런타임을 추가하는 PR과 함께 이 페이지들이 추가됩니다. 이 자격 증명 설정은 독립적인 사전 요구 사항이므로 미리 완료해도 안전합니다.
