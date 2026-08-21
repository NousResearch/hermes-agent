# Bitwarden Secrets Manager

평문으로 `~/.hermes/.env`에 저장하는 대신 프로세스 시작 시 [Bitwarden Secrets Manager](https://bitwarden.com/products/secrets-manager/)에서 API 키를 가져옵니다. 하나의 부트스트랩 시크릿(머신 계정 액세스 토큰)이 프로바이더별 N개 키를 대체하며, 자격 증명 교체는 Bitwarden 웹 앱에서 한 번만 변경하면 됩니다.

## 작동 방식

1. Bitwarden Secrets Manager에서 **머신 계정**을 만들고, 프로젝트에 대한 읽기 액세스 권한을 부여한 다음 **액세스 토큰**을 생성합니다.
2. Hermes는 해당 토큰 하나를 `~/.hermes/.env`에 `BWS_ACCESS_TOKEN`으로 저장합니다.
3. `hermes`(또는 게이트웨이, cron 작업)가 시작될 때마다 `~/.hermes/.env`가 로드된 후 Hermes가 `bws secret list <project_id>`를 호출하고 반환된 키를 `os.environ`에 설정합니다.
4. 기본적으로 Hermes는 이미 환경에 있는 값을 **덮어쓰므로**, Bitwarden이 단일 기준이 됩니다. 웹 앱에서 키를 한 번 교체하면 다음 시작 시 모든 Hermes 프로세스가 이를 가져옵니다. 대신 `.env`가 우선하도록 하려면 config에서 `override_existing: false`로 변경합니다.

`bws` 바이너리는 처음 사용할 때 `~/.hermes/bin/`에 자동으로 다운로드됩니다. `apt`, `brew`, `sudo`가 필요하지 않습니다.

## 머신 계정을 사용하는 이유 (그리고 2FA 프롬프트가 없는 이유)

Bitwarden Secrets Manager는 비대화형 워크로드를 위해 설계되었습니다. 머신 계정은 사람의 개입이 없으므로 2FA를 적용할 수 없습니다. 액세스 토큰이 자격 증명입니다. 토큰을 가진 사람은 머신 계정이 액세스할 수 있는 모든 시크릿을 읽을 수 있으므로, 이를 고가치 bearer 토큰처럼 취급하세요. `.env`에 저장하고(`config.yaml`이 아님), 유출되었다면 Bitwarden 웹 앱에서 폐기하고 새로 생성하세요.

머신 계정은 일반적인 2FA가 적용되는 *웹 앱에서* 설정합니다. 그 후 토큰은 자율적으로 작동합니다.

## 설정

### 1. 머신 계정과 액세스 토큰 생성

[Bitwarden 웹 앱](https://vault.bitwarden.com)(EU 계정은 [vault.bitwarden.eu](https://vault.bitwarden.eu))에서:

1. 제품 전환기에서 **Secrets Manager**로 전환합니다.
2. **프로젝트**를 만들거나 선택합니다(예: "Hermes keys").
3. 프로바이더 키를 시크릿으로 추가합니다. 시크릿 **Name**이 환경 변수 이름이 됩니다. `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` 등을 사용하세요.
4. **Machine accounts → New machine account → My Hermes machine** → **Projects** 탭 → 프로젝트에 Read 액세스 권한을 부여합니다.
5. **Access tokens** 탭 → **Create access token** → 만료일을 **Never**로 설정(또는 날짜 선택) → 토큰을 복사합니다(`0.`으로 시작). Bitwarden은 토큰을 다시 조회할 수 없으므로 복사본을 보관하세요.

Secrets Manager는 제한 사항과 함께 Bitwarden 무료 요금제에 포함되어 있으므로, 시험 사용에 유료 플랜이 필요하지 않습니다.

### 2. 마법사 실행

```bash
hermes secrets bitwarden setup
```

다음 작업을 수행합니다.

1. `bws v2.0.0`을 `~/.hermes/bin/bws`에 다운로드하고 검증합니다.
2. 액세스 토큰을 입력하라는 메시지를 표시합니다(입력 내용은 숨겨짐). 토큰은 `~/.hermes/.env`에 `BWS_ACCESS_TOKEN`으로 저장됩니다.
3. 머신 계정이 속한 Bitwarden 리전을 묻습니다. **US Cloud**, **EU Cloud**, **self-hosted / custom URL** 중에서 선택합니다. 값은 `config.yaml`의 `secrets.bitwarden.server_url`에 저장되고 `BWS_SERVER_URL`로 `bws`에 전달됩니다.
4. 머신 계정이 볼 수 있는 프로젝트를 나열합니다. 하나를 선택합니다. 선택한 값은 `config.yaml`의 `secrets.bitwarden.project_id`에 저장됩니다.
5. 프로젝트의 시크릿을 테스트로 가져오고 어떤 env vars가 확인될지 보여줍니다.
6. `secrets.bitwarden.enabled: true`로 변경합니다.

플래그를 사용한 비대화형 설정도 지원합니다.

```bash
hermes secrets bitwarden setup \
  --access-token "$BWS_ACCESS_TOKEN" \
  --server-url https://vault.bitwarden.eu \
  --project-id <project-uuid>
```

### 3. 확인

```bash
hermes secrets bitwarden status
```

이제부터 모든 `hermes` 호출이 시작 시 최신 시크릿을 가져옵니다. 프로세스에서 시크릿이 처음 적용될 때 stderr에 한 줄 요약이 표시됩니다.

## CLI

| 명령 | 기능 |
|---|---|
| `hermes secrets bitwarden setup` | 대화형 마법사(바이너리 설치, 토큰 입력, 프로젝트 선택, 가져오기 테스트) |
| `hermes secrets bitwarden status` | config + 바이너리 버전 + 토큰 존재 여부/검증 결과 표시 |
| `hermes secrets bitwarden token` | 액세스 토큰 교체: Bitwarden에 새 토큰을 검증한 후 `.env`에 저장 |
| `hermes secrets bitwarden sync` | 드라이 런: 지금 시크릿을 가져오고 적용될 내용을 표시 |
| `hermes secrets bitwarden sync --apply` | 가져온 시크릿을 현재 셸 환경으로 내보냄 |
| `hermes secrets bitwarden install` | 고정된 `bws` 바이너리만 다운로드(인증 불필요) |
| `hermes secrets bitwarden disable` | `enabled: false`로 변경하고 토큰 + 프로젝트 ID는 유지 |

## 만료되었거나 폐기된 토큰 교체

머신 계정 토큰이 만료되거나, 폐기되거나, 계정이 삭제되면 시작 시 다음이 표시됩니다.

```
Bitwarden Secrets Manager: Bitwarden rejected the machine-account access token (BWS_ACCESS_TOKEN) — it was likely revoked, expired, or belongs to another region.  (...)
Bitwarden Secrets Manager: → Run `hermes secrets bitwarden token` to paste a fresh access token ...
```

전체 마법사를 다시 실행하지 않고 수정할 수 있습니다.

```bash
hermes secrets bitwarden token                     # masked prompt
hermes secrets bitwarden token --access-token 0.…  # non-interactive
```

이 명령은 무엇이든 쓰기 **전에** 새 토큰으로 Bitwarden을 조회합니다. 거부된 토큰은 현재 `.env`를 변경하지 않습니다. 성공하면 토큰을 저장하고 가져오기 캐시를 지우며, 구성된 프로젝트가 새 머신 계정에 표시되지 않으면 경고합니다.

## 구성

`~/.hermes/config.yaml`의 기본값:

```yaml
secrets:
  bitwarden:
    enabled: false
    access_token_env: BWS_ACCESS_TOKEN
    project_id: ""
    server_url: ""
    cache_ttl_seconds: 300
    encrypted_cache:
      enabled: false
      max_stale_seconds: 0
    override_existing: true
    auto_install: true
```

| 키 | 기본값 | 기능 |
|---|---|---|
| `enabled` | `false` | 마스터 스위치입니다. false이면 Bitwarden에 절대 연결하지 않습니다. |
| `access_token_env` | `BWS_ACCESS_TOKEN` | 부트스트랩 토큰을 담는 env var 이름입니다. 다른 용도로 `BWS_ACCESS_TOKEN`을 이미 사용 중이면 변경하세요. |
| `project_id` | `""` | 동기화할 프로젝트의 UUID입니다. |
| `server_url` | `""` | Bitwarden 리전 또는 self-hosted 엔드포인트입니다. 비어 있으면 `bws` 기본값(US Cloud, `https://vault.bitwarden.com`)을 사용합니다. EU Cloud는 `https://vault.bitwarden.eu`, self-hosted는 자체 URL을 설정하세요. `bws` 서브프로세스에 `BWS_SERVER_URL`로 전달됩니다. |
| `cache_ttl_seconds` | `300` | 프로세스 내 또는 디스크 가져오기 결과를 재사용하는 시간입니다. fresh-cache 재사용을 비활성화하려면 `0`으로 설정하세요. |
| `encrypted_cache.enabled` | `false` | 마지막으로 성공한 가져오기 결과를 `~/.hermes/cache/bws_cache.enc.json`의 AES-GCM 암호화 캐시에 저장합니다. |
| `encrypted_cache.max_stale_seconds` | `0` | 암호화 캐시가 활성화된 경우 네트워크/타임아웃 오류가 발생했을 때만 해당 캐시를 이 기간까지 사용합니다. 인증 오류에서는 오래된 시크릿을 사용하지 않습니다. 암호화된 쓰기가 성공하면 기존 평문 `cache/bws_cache.json`을 삭제합니다. |
| `override_existing` | `true` | true이면 Bitwarden 값이 env에 이미 있는 값을 덮어씁니다(웹 앱에서의 교체가 실제로 적용됨). 로컬에서 `.env` / 셸 export가 우선하도록 하려면 `false`로 변경하세요. |
| `auto_install` | `true` | true이면 처음 사용할 때 `bws`를 `~/.hermes/bin/`에 자동으로 다운로드합니다. |

## 오류 모드

Bitwarden 때문에 Hermes 시작이 차단되는 일은 없습니다. 문제가 발생하면 stderr에 한 줄 경고가 표시되고, Hermes는 `.env`에 이미 있던 자격 증명을 사용해 계속 실행됩니다.

| 증상 | 원인 | 해결 방법 |
|---|---|---|
| `BWS_ACCESS_TOKEN is not set` | config에서 활성화했지만 `.env`에서 토큰이 삭제됨 | `hermes secrets bitwarden setup`을 다시 실행 |
| `Bitwarden rejected the machine-account access token … invalid_client` | 토큰이 폐기되었거나 만료되었거나 머신 계정이 삭제됨 — 또는 토큰이 다른 리전에 속함(예: EU 토큰으로 US identity endpoint에 연결) | `hermes secrets bitwarden token`을 실행해 새 토큰을 입력합니다. 리전이 일치하지 않으면 setup을 다시 실행하고 EU/self-hosted를 선택하거나 `secrets.bitwarden.server_url`을 설정하세요. |
| `bws exited 1: invalid access token` | 토큰이 폐기되었거나 잘못됨 | 새 토큰으로 `hermes secrets bitwarden token`을 실행 |
| `bws timed out` | 네트워크가 차단되었거나 Bitwarden API가 느림 | `api.bitwarden.com`(또는 `server_url`)에 연결할 수 있는지 확인 |
| `bws binary not available` | `auto_install: false`이고 `bws`가 PATH에 없음 | [github.com/bitwarden/sdk-sm/releases](https://github.com/bitwarden/sdk-sm/releases)에서 수동으로 설치하거나 `auto_install`을 다시 켬 |
| `Checksum mismatch` | 다운로드가 손상되었거나 변조됨 | 다시 실행하면 재시도합니다. 계속되면 이슈를 등록하세요. |

이제 시작 경고에 실패를 정확히 해결하는 명령을 알려주는 `→` remediation line이 포함됩니다.

## 보안 참고 사항

- 부트스트랩 토큰(`BWS_ACCESS_TOKEN`) 자체가 민감한 정보입니다. 이 토큰을 가진 사람은 머신 계정이 액세스할 수 있는 모든 시크릿을 읽을 수 있습니다. 다른 API 키와 동일하게 취급하세요.
- `override_existing: true`인 경우에도 Hermes는 Bitwarden이 부트스트랩 토큰 자체를 덮어쓰지 못하게 합니다. 프로젝트 내부에 `BWS_ACCESS_TOKEN`을 시크릿으로 저장하면 적용할 때 조용히 건너뜁니다.
- `bws` 바이너리 다운로드는 동일한 GitHub 릴리스에서 공개된 SHA-256 체크섬과 대조하여 검증됩니다. 불일치하면 설치가 중단됩니다.
- 이 문서 작성 시점의 고정 버전(`bws v2.0.0`)은 이 저장소에 대한 PR을 통해 업데이트됩니다. 업스트림 릴리스 형태가 바뀔 수 있으므로 Hermes는 `bws`를 "latest"로 자동 업그레이드하지 않습니다.

## 이 기능을 사용하지 말아야 할 때

- `~/.hermes/.env`로 충분한 **단일 머신 개인 설정**. 자격 증명 하나를 다른 자격 증명으로 바꾸고 시작 시 네트워크 의존성을 추가하게 됩니다.
- `api.bitwarden.com`에 연결할 수 없는 **에어 갭 환경**.
- 기존 시크릿 주입 메커니즘(GitHub Actions secrets, Vault 등)이 이미 설정된 **CI/CD**. 두 경로를 함께 사용하지 말고 하나를 선택하세요.

이 기능이 특히 유용한 경우는 여러 머신으로 구성된 fleet, 공유 개발 박스, 게이트웨이 VPS 또는 여러 Hermes 설치 환경에서 중앙 집중식 교체와 폐기를 원하는 모든 설정입니다.
