# Bitwarden Secrets Manager (Bitwarden 비밀 관리자)

API 키를 `~/.hermes/.env` 내부에 평문으로 저장하는 대신 프로세스 시작 시 [Bitwarden Secrets Manager](https://bitwarden.com/products/secrets-manager/)에서 가져오세요. 단일 부트스트랩 비밀키(시스템 계정 액세스 토큰) 하나가 여러 제공자의 키를 대체하며, 자격 증명을 교체할 때는 Bitwarden 웹 앱에서 한 번만 변경하면 됩니다.

## 작동 방식 (How it works)

1. Bitwarden Secrets Manager에서 **시스템 계정(machine account)** 을 생성하고 프로젝트에 대한 읽기 권한을 부여한 다음, **액세스 토큰(access token)** 을 생성합니다.
2. Hermes는 그 단일 토큰을 `~/.hermes/.env`에 `BWS_ACCESS_TOKEN`으로 저장합니다.
3. `hermes`(또는 게이트웨이, 크론 작업)가 시작될 때마다 `~/.hermes/.env`가 로드된 후 Hermes는 `bws secret list <project_id>`를 호출하여 반환된 키를 `os.environ`에 설정합니다.
4. 기본적으로 Hermes는 환경 변수에 이미 있는 값을 **덮어쓰기** 때문에 Bitwarden이 단일 진실 공급원(source of truth)이 됩니다 — 웹 앱에서 키를 한 번 교체하면 다음에 시작되는 모든 Hermes 프로세스가 새 키를 가져옵니다. `.env` 값을 우선시하려면 설정에서 `override_existing: false`로 변경하세요.

`bws` 바이너리는 처음 사용할 때 `~/.hermes/bin/` 폴더에 자동 다운로드됩니다 — `apt`, `brew`, `sudo` 등이 필요하지 않습니다.

## 시스템 계정을 사용하는 이유 (그리고 2FA 프롬프트가 없는 이유)

Bitwarden Secrets Manager는 비대화형 워크로드를 위해 설계되었습니다. 시스템 계정은 작업 과정에 사람이 개입하지 않으므로 2FA 게이트를 설정할 수 없습니다. 액세스 토큰 자체가 자격 증명입니다. 이 토큰을 가진 사람은 누구나 해당 시스템 계정이 접근할 수 있는 모든 비밀키를 읽을 수 있으므로 가치가 높은 전달자 토큰(bearer token)처럼 취급해야 합니다 — `config.yaml`이 아닌 `.env`에 저장하고, 만약 유출될 경우 Bitwarden 웹 앱에서 폐기 및 재생성하세요.

일반적인 2FA가 적용되는 *웹 앱에서* 시스템 계정을 설정합니다. 그 이후에는 토큰이 자율적으로 작동합니다.

## 설정 (Setup)

### 1. 시스템 계정 및 액세스 토큰 생성

[Bitwarden 웹 앱](https://vault.bitwarden.com) (EU 계정의 경우 [vault.bitwarden.eu](https://vault.bitwarden.eu))에서:

1. 제품 전환기에서 **Secrets Manager**로 전환합니다.
2. **프로젝트(Project)** 를 생성하거나 선택합니다 (예: "Hermes keys").
3. 제공자 키를 비밀키로 추가합니다. 비밀키의 **이름(Name)** 이 환경 변수 이름이 됩니다 — `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` 등을 사용하세요.
4. **Machine accounts → New machine account → My Hermes machine** → **Projects** 탭 → 해당 프로젝트에 대해 읽기(Read) 권한을 부여합니다.
5. **Access tokens** 탭 → **Create access token** → 만료 기한 **Never**(또는 날짜 선택) → 토큰을 복사합니다 (`0.`으로 시작함). Bitwarden에서 이 토큰을 다시 확인할 수 없으므로 복사본을 보관하세요.

Secrets Manager는 한도가 있는 Bitwarden 무료 티어에 포함되어 있으므로 이 기능을 시도해 보기 위해 유료 플랜이 필요하지 않습니다.

### 2. 마법사 실행

```bash
hermes secrets bitwarden setup
```

다음 작업이 수행됩니다:

1. `bws v2.0.0`을 `~/.hermes/bin/bws`에 다운로드하고 검증합니다.
2. 액세스 토큰을 입력하라는 프롬프트를 표시합니다 (입력 내용은 숨겨짐). `~/.hermes/.env`에 `BWS_ACCESS_TOKEN`으로 저장됩니다.
3. 시스템 계정이 속한 Bitwarden 리전을 묻습니다 — **US Cloud**, **EU Cloud**, 또는 **self-hosted / custom URL**. `config.yaml`의 `secrets.bitwarden.server_url`에 저장되며 `bws`에는 `BWS_SERVER_URL`로 전달됩니다.
4. 시스템 계정이 볼 수 있는 프로젝트 목록을 표시합니다; 하나를 선택하세요. `config.yaml`의 `secrets.bitwarden.project_id`에 저장됩니다.
5. 테스트로 프로젝트의 비밀키를 가져와서 어떤 환경 변수들이 결정되는지 보여줍니다.
6. `secrets.bitwarden.enabled: true`로 설정합니다.

플래그를 사용한 비대화형 설정도 지원됩니다:

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

이제부터 모든 `hermes` 호출은 시작 시 최신 비밀키를 가져옵니다. 프로세스에서 처음 비밀키가 적용될 때 stderr에 한 줄 요약이 표시됩니다.

## CLI 명령어

| 명령어 | 수행하는 작업 |
|---|---|
| `hermes secrets bitwarden setup` | 대화형 마법사 (바이너리 설치, 토큰 요청, 프로젝트 선택, 가져오기 테스트) |
| `hermes secrets bitwarden status` | 구성 + 바이너리 버전 + 토큰 존재 여부 표시 |
| `hermes secrets bitwarden sync` | 예행 연습(Dry-run): 지금 비밀키를 가져오고 무엇이 적용될지 표시 |
| `hermes secrets bitwarden sync --apply` | 가져와서 현재 쉘의 환경에 내보내기 (export) |
| `hermes secrets bitwarden install` | 고정된 `bws` 바이너리만 다운로드 (인증 불필요) |
| `hermes secrets bitwarden disable` | `enabled: false` 설정; 토큰 + 프로젝트 ID는 유지됨 |

## 구성 (Configuration)

`~/.hermes/config.yaml`의 기본값:

```yaml
secrets:
  bitwarden:
    enabled: false
    access_token_env: BWS_ACCESS_TOKEN
    project_id: ""
    server_url: ""
    cache_ttl_seconds: 300
    override_existing: true
    auto_install: true
```

| 키 | 기본값 | 수행하는 작업 |
|---|---|---|
| `enabled` | `false` | 마스터 스위치입니다. false일 경우 Bitwarden에 접근하지 않습니다. |
| `access_token_env` | `BWS_ACCESS_TOKEN` | 부트스트랩 토큰을 담을 환경 변수 이름입니다. 다른 용도로 이미 `BWS_ACCESS_TOKEN`을 사용 중이라면 이 값을 변경하세요. |
| `project_id` | `""` | 동기화할 프로젝트의 UUID입니다. |
| `server_url` | `""` | Bitwarden 리전 또는 자체 호스팅 엔드포인트입니다. 비워두면 `bws` 기본값(US Cloud, `https://vault.bitwarden.com`)이 사용됩니다. EU Cloud의 경우 `https://vault.bitwarden.eu`로, 자체 호스팅의 경우 해당 URL로 설정하세요. 하위 프로세스 `bws`에 `BWS_SERVER_URL`로 전달됩니다. |
| `cache_ttl_seconds` | `300` | 프로세스 내부의 가져오기 결과 재사용 기간입니다. 캐싱을 비활성화하려면 `0`으로 설정하세요. 캐시는 프로세스 단위로 적용되며 새로운 `hermes` 호출은 새로 시작됩니다. |
| `override_existing` | `true` | true일 경우, 환경에 이미 존재하는 모든 값을 Bitwarden 값으로 덮어씁니다 (따라서 웹 앱에서의 교체가 실제로 효력을 발생합니다). 로컬의 `.env` / 쉘 내보내기 값을 우선시하려면 `false`로 설정하세요. |
| `auto_install` | `true` | true일 경우, 처음 사용 시 `~/.hermes/bin/`에 `bws`가 자동 다운로드됩니다. |

## 실패 모드 (Failure modes)

Bitwarden은 Hermes의 시작을 결코 차단하지 않습니다. 문제가 생기면 stderr에 한 줄 경고가 표시되고 Hermes는 `.env`에 이미 있는 자격 증명을 사용하여 계속 실행됩니다:

| 증상 | 원인 | 해결 방법 |
|---|---|---|
| `BWS_ACCESS_TOKEN is not set` | 설정에서는 활성화되었지만 `.env`에서 토큰이 지워짐 | `hermes secrets bitwarden setup` 다시 실행 |
| `bws exited 1: invalid access token` | 토큰이 취소되었거나 잘못됨 | 새 토큰을 생성하고 setup 다시 실행 |
| `[400 Bad Request] {"error":"invalid_client"}` | `bws`가 호출하는 리전과 다른 리전용 토큰임 (예: 미국 인증 엔드포인트를 호출하는 EU 토큰) | setup을 다시 실행하여 올바른 리전을 선택하거나, `secrets.bitwarden.server_url`을 `https://vault.bitwarden.eu` (또는 자체 호스팅 URL)로 설정 |
| `bws timed out` | 네트워크가 차단되었거나 Bitwarden API 응답 지연 | `api.bitwarden.com` (또는 `server_url`) 연결 상태 확인 |
| `bws binary not available` | `auto_install: false`이고 PATH에 `bws`가 없음 | [github.com/bitwarden/sdk-sm/releases](https://github.com/bitwarden/sdk-sm/releases)에서 수동으로 설치하거나 `auto_install`을 다시 켬 |
| `Checksum mismatch` | 다운로드가 손상되었거나 변조됨 | 재실행하면 재시도함; 지속될 경우 이슈 접수 바람 |

## 보안 참고 사항 (Security notes)

- 부트스트랩 토큰(`BWS_ACCESS_TOKEN`) 자체는 민감한 정보입니다 — 이를 소지한 누구나 시스템 계정이 접근할 수 있는 모든 비밀키를 읽을 수 있습니다. 다른 API 키와 동일하게 취급해야 합니다.
- Hermes는 `override_existing: true`이더라도 Bitwarden이 부트스트랩 토큰 자체를 덮어쓰는 것을 거부합니다. 프로젝트 내부에 `BWS_ACCESS_TOKEN`을 비밀키로 저장하더라도 적용 과정에서 조용히 건너뜁니다.
- 다운로드된 `bws` 바이너리는 동일한 GitHub 릴리스에 게시된 SHA-256 체크섬으로 검증됩니다. 일치하지 않으면 설치가 중단됩니다.
- 고정된 버전(이 글 작성 시점에는 `bws v2.0.0`)은 이 저장소의 PR을 통해 업데이트됩니다 — 업스트림의 릴리스 형태가 바뀔 수 있기 때문에 Hermes는 `bws`를 최신("latest")으로 자동 업그레이드하지 않습니다.

## 사용하지 않아야 할 경우 (When NOT to use this)

- `~/.hermes/.env`로도 충분한 **단일 기기 개인용 환경**. 자격 증명을 다른 것으로 대체할 뿐이며 시작 시 네트워크 의존성만 추가됩니다.
- `api.bitwarden.com`에 접근할 수 없는 **에어갭(Air-gapped) 환경**.
- 기존의 비밀키 주입 메커니즘(GitHub Actions 비밀키, Vault 등)이 이미 설정된 **CI/CD** 환경 — 두 가지 방법이 아닌 하나의 경로만 선택하세요.

이 기능은 여러 기기로 구성된 플릿(fleet), 공유 개발 서버, 게이트웨이 VPS 등 다수의 Hermes 설치 환경 전반에서 비밀키 교체 및 폐기를 중앙에서 집중 관리하려는 환경에 매우 적합합니다.
