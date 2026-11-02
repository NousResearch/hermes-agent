# 1Password

프로세스 시작 시 [1Password](https://1password.com/)에서 공급자 API 키를 확인하므로, 키를 `~/.hermes/.env`에 평문으로 저장할 필요가 없습니다. 키는 1Password 항목으로 관리하고 `op://vault/item/field`를 통해 참조합니다. 자격 증명을 교체할 때는 1Password에서 한 번만 변경하면 됩니다.

## 작동 방식

1. 공식 [1Password CLI](https://developer.1password.com/docs/cli/get-started/)(`op`)를 설치하고 인증합니다 — 헤드리스 서버에서는 **서비스 계정 토큰**을, 노트북에서는 **대화형/데스크톱 세션**을 사용합니다.
2. `~/.hermes/config.yaml`에서 환경 변수 이름을 `op://` 참조에 매핑합니다.
3. `hermes`(또는 gateway나 cron 작업)가 시작될 때마다 `~/.hermes/.env`를 읽은 후 Hermes가 각 참조에 대해 `op read`를 실행하고, 확인된 값을 `os.environ`에 설정합니다.
4. 기본적으로 Hermes는 환경에 이미 있는 값을 **덮어쓰므로** 1Password가 기준값이 됩니다 — 자격 증명을 한 번 교체하면 다음 시작 시 모든 Hermes 프로세스가 새 값을 가져옵니다. `.env`를 우선하려면 `override_existing: false`로 변경하세요.

Hermes는 사용자를 대신해 인증하지 않으며 `op`를 다운로드하지도 않습니다. 이미 설치하고 신뢰한 CLI를 셸에서 실행할 뿐입니다. `op`가 없거나, 세션이 잠겨 있거나, 참조가 잘못된 경우 Hermes는 한 줄 경고를 출력하고 `.env`에 이미 있던 자격 증명을 사용해 계속합니다 — 시작을 차단하지 않습니다.

## 인증

`op`는 비대화형 환경에 적합한 두 가지 모드를 지원하며, Hermes는 둘 다 사용할 수 있습니다.

- **서비스 계정**(서버/CI에 권장): 1Password에서 서비스 계정을 만들고 관련 vault에 대한 읽기 권한을 부여한 다음, 토큰을 `~/.hermes/.env`의 `OP_SERVICE_ACCOUNT_TOKEN`으로 export합니다. 토큰은 자격 증명이므로 다른 bearer token과 동일하게 취급하세요.
- **데스크톱 / 대화형 세션**(노트북): `op signin`을 실행하거나 1Password 앱에서 CLI 통합을 활성화합니다. Hermes는 `OP_SESSION_*` 변수를 `op` 자식 프로세스에 전달합니다. 1Password 캐시 키에는 이러한 세션 변수가 포함되므로 다른 계정에 로그인해도 이전 ID로 캐시된 값이 제공되지 않습니다.

## 부트스트랩 토큰

**서비스 계정 토큰**으로 인증할 때 이 토큰 자체가 Hermes가 `op://` 참조를 확인하기 전에 필요한 부트스트랩 자격 증명입니다. 비밀을 확인하는 모든 프로세스의 `os.environ`에 토큰이 있어야 합니다 — cron 작업(`kanban.dispatch_in_gateway: false`), 하위 프로세스, CLI 실행, macOS launchd 에이전트, Docker 컨테이너를 포함하며 대화형 gateway만 해당하는 것이 아닙니다. 토큰을 사용할 수 있게 하는 방법은 우선순위 순으로 세 가지입니다.

1. **`~/.hermes/.env`에 저장(권장).** `hermes secrets onepassword setup --token <token>`은 Bitwarden의 `BWS_ACCESS_TOKEN`과 동일한 방식으로 토큰을 `~/.hermes/.env`에 기록합니다. `load_hermes_dotenv()`는 항상 `.env`를 읽으므로 별도 설정 없이 어디서나 토큰을 사용할 수 있습니다. 가장 간단하고 안정적인 방법입니다.

2. **`~/.hermes/.op.env`에 저장(gitignored).** 서비스 계정 토큰을 `.env`와 분리하고 싶다면 — 예를 들어 `.env`는 비공개 dotfiles 저장소에 넣되 토큰은 버전 관리에서 제외하고 싶다면 — `~/.hermes/.op.env`에 저장하세요.

   ```bash
   echo 'OP_SERVICE_ACCOUNT_TOKEN=ops_...' > ~/.hermes/.op.env
   chmod 600 ~/.hermes/.op.env
   ```

   Hermes는 시작 시 `.env` **다음에** `.op.env`를 자동으로 읽으며, 환경에 이미 있는 토큰은 **절대** 덮어쓰지 않습니다. `.op.env`는 gitignored이므로 토큰이 커밋된 파일에 들어가지 않습니다.

3. **systemd `EnvironmentFile`을 통해 제공(Linux gateway).** gateway를 systemd로 실행한다면 토큰을 서비스 환경에 직접 주입할 수 있습니다.

   ```ini
   [Service]
   EnvironmentFile=-/home/youruser/.hermes/.op.env
   ```

   이렇게 주입한 토큰이 우선합니다 — Hermes는 `OP_SERVICE_ACCOUNT_TOKEN`이 이미 설정된 것을 감지하고 `.op.env` 로드를 완전히 건너뜁니다.

토큰이 대화형 셸(`op signin`, `.bashrc`의 `OP_SESSION_*` export 등)을 통해서만 접근 가능하다면 cron 작업이나 새로 생성된 하위 프로세스에 **상속되지 않습니다**. 이런 컨텍스트에서는 경고를 기록하고 `.env`에 이미 있던 자격 증명으로 대체합니다. 비대화형 작업에는 위 세 가지 방법 중 하나를 사용하세요.

## 설정

### 1. `op` 설치 및 로그인

[1Password CLI 시작 가이드](https://developer.1password.com/docs/cli/get-started/)를 따르세요. 작동하는지 확인합니다.

```bash
op whoami
```

### 2. 통합 활성화

```bash
hermes secrets onepassword setup
```

이 명령은 `op`가 `PATH`에 있는지 확인하고(또는 `--binary-path` 사용), 계정/토큰 설정을 기록하고, 활성 세션을 확인한 다음 `secrets.onepassword.enabled: true`로 변경합니다. 비대화형 플래그:

```bash
hermes secrets onepassword setup \
  --account my.1password.com \
  --token-env OP_SERVICE_ACCOUNT_TOKEN \
  --token "$OP_SERVICE_ACCOUNT_TOKEN"
```

### 3. 자격 증명 매핑

참조 형식은 `op://<vault>/<item>/<field>`입니다.

```bash
hermes secrets onepassword set OPENAI_API_KEY    "op://Private/OpenAI/api key"
hermes secrets onepassword set ANTHROPIC_API_KEY "op://Private/Anthropic/credential"
```

### 4. 미리 보기 및 확인

```bash
hermes secrets onepassword sync     # dry-run: resolve now, show what would apply
hermes secrets onepassword status   # config + binary + references + auth
```

이제부터 모든 `hermes` 호출은 시작 시 참조를 확인합니다. 한 프로세스에서 비밀이 처음 적용되면 stderr에 한 줄 요약이 표시됩니다.

## CLI

| 명령 | 기능 |
|---|---|
| `hermes secrets onepassword setup` | `op`를 확인하고, 계정 / 토큰 환경 변수를 설정하고, 활성화 |
| `hermes secrets onepassword status` | 설정, 바이너리, 인증, 구성된 참조를 표시 |
| `hermes secrets onepassword token` | 서비스 계정 토큰을 교체: `op whoami`로 확인한 후 `.env`에 저장 |
| `hermes secrets onepassword set ENV_VAR "op://…"` | 환경 변수를 참조에 매핑(공백 제거 후 저장 및 확인) |
| `hermes secrets onepassword remove ENV_VAR` | 매핑 삭제 |
| `hermes secrets onepassword sync` | 드라이런: 지금 참조를 확인하고 적용될 내용을 표시 |
| `hermes secrets onepassword sync --apply` | 확인한 값을 현재 셸의 환경으로 export |
| `hermes secrets onepassword disable` | `enabled: false`로 변경; 매핑은 유지 |

`op`와 `1password`는 `onepassword`의 별칭으로 사용할 수 있습니다.

## 구성

`~/.hermes/config.yaml`의 기본값:

```yaml
secrets:
  onepassword:
    enabled: false
    env:
      OPENAI_API_KEY: "op://Private/OpenAI/api key"
      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"
    account: ""
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
    binary_path: ""
    cache_ttl_seconds: 300
    override_existing: true
```

| 키 | 기본값 | 기능 |
|---|---|---|
| `enabled` | `false` | 마스터 스위치. false이면 `op`를 호출하지 않습니다. |
| `env` | `{}` | 환경 변수 이름 → `op://vault/item/field` 참조 매핑. 이름이 유효한 환경 변수 이름이 아니거나 값이 `op://` 참조가 아닌 항목은 경고와 함께 건너뜁니다. |
| `account` | `""` | `op read --account`로 전달할 계정 약칭 / 로그인 주소. 비어 있으면 `op`의 기본 계정을 사용합니다. |
| `service_account_token_env` | `OP_SERVICE_ACCOUNT_TOKEN` | Hermes가 서비스 계정 토큰을 읽을 환경 변수. 값은 `op`가 요구하는 이름인 `OP_SERVICE_ACCOUNT_TOKEN`으로 `op` 자식 프로세스에 export됩니다. 데스크톱/대화형 세션을 사용하려면 변수를 설정하지 마세요. |
| `binary_path` | `""` | `op`의 절대 경로. 설정하면 있는 그대로 사용하며 `PATH`를 **확인하지 않습니다** — `PATH`에서 어떤 `op`가 먼저 발견되든 신뢰하지 않도록 고정할 때 사용하세요. |
| `cache_ttl_seconds` | `300` | 확인된 값을 재사용하는 기간(프로세스 내 및 디스크). `0`으로 설정하면 **두** 캐시 계층을 비활성화하며 디스크에 값을 전혀 기록하지 않습니다. |
| `override_existing` | `true` | true이면 확인된 값이 환경에 이미 있는 값을 덮어씁니다(따라서 교체가 적용됨). `.env` / 셸 export를 우선하려면 `false`로 변경하세요. 이 경우 `op`를 호출하기 **전에** 해당 참조를 건너뜁니다. |

## 실패 모드

1Password는 Hermes의 시작을 절대 차단하지 않습니다. 문제가 발생하면 stderr에 한 줄 경고가 표시되고 Hermes는 계속 실행됩니다.

| 증상 | 원인 | 해결 방법 |
|---|---|---|
| `the op CLI was not found on PATH` | `op`가 설치되지 않았거나 `PATH`에 없음 | CLI를 설치하거나 `secrets.onepassword.binary_path`를 설정 |
| `op read failed for 'op://…': …` | 세션 잠김, 토큰 만료, 또는 vault 접근 권한 없음 | `op signin`을 실행하거나 `hermes secrets onepassword token`으로 서비스 계정 토큰을 교체하거나 서비스 계정에 접근 권한 부여 |
| `op read returned an empty value for 'op://…'` | 참조한 필드는 존재하지만 비어 있음 | 1Password에서 항목/필드를 수정(빈 값은 절대 적용되지 않으며 기존 환경 변수가 그대로 유지됨) |
| `… is not an op:// secret reference` | 매핑 값이 `op://` 참조가 아님 | 올바른 `op://vault/item/field` 형식으로 다시 설정 |
| `op read timed out` | 네트워크 차단 또는 1Password 응답 지연 | 연결 상태 / 데스크톱 앱 통합을 확인 |

이제 시작 경고에는 실패를 해결하는 정확한 명령을 알려 주는 `→` 해결 방법 줄이 포함됩니다.

## 캐싱

성공적으로 완전히 가져온 값은 프로세스 내와 `<hermes_home>/cache/op_cache.json`의 디스크에 캐시됩니다(원자적으로 기록되며 모드는 `0600`). 따라서 짧은 간격으로 실행되는 `hermes` 호출이 참조마다 매번 `op`를 다시 셸에서 실행하지 않아도 됩니다. 캐시는 다음과 같습니다.

- 확인된 비밀 **값**만 저장합니다 — 서비스 계정 토큰이나 원시 인증 자료는 저장하지 않습니다(인증 정보는 캐시 키에 fingerprint로 반영됨).
- 토큰, 계정, `OP_SESSION_*` 변수 또는 참조 집합이 변경되면 무효화됩니다.
- 참조별 오류가 하나라도 발생한 가져오기는 기록하지 않으므로 일시적인 인증 실패가 TTL 동안 고정되지 않습니다.
- `cache_ttl_seconds: 0`이면 읽기와 쓰기 모두 완전히 비활성화됩니다.

## 보안 참고 사항

- 1Password 서비스 계정 토큰은 해당 계정이 접근할 수 있는 모든 비밀을 읽을 수 있습니다. `~/.hermes/.env`에 저장하고(`config.yaml`에는 저장하지 않음), 유출되면 1Password에서 폐기하고 새로 생성하세요.
- `override_existing: true`인 경우에도 Hermes는 확인된 값이 토큰 환경 변수 자체를 덮어쓰지 못하게 합니다.
- `op` 자식 프로세스에는 전체 `os.environ`의 복사본이 아니라 최소한의 허용 목록 환경(auth/session 변수 + `PATH`/`HOME`)만 전달되므로, dotenv 이후의 공급자 자격 증명이 모두 자식 프로세스에 상속되지는 않습니다.
- 참조는 `op://`로 시작하는지 확인되며, 참조는 `--` 옵션 종료자 뒤에 전달되므로 조작된 값이 `op` 플래그로 해석될 수 없습니다.

## 사용하지 말아야 할 때

- `~/.hermes/.env`로 충분한 **단일 머신 개인 설정**.
- 1Password에 연결할 수 없는 **에어갭 환경**.
- 기존 비밀 주입 메커니즘이 이미 연결된 **CI/CD** — 두 경로가 아니라 하나를 선택하세요.

이 기능은 여러 머신으로 구성된 fleet, 공유 개발 머신, gateway VPS, 또는 여러 Hermes 설치 환경에서 중앙 집중식 교체와 폐기를 원할 때 적합합니다.
