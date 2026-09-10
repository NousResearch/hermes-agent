---
sidebar_position: 9
title: "Secret Source 플러그인"
description: "Hermes Agent용 시크릿 관리자 백엔드 플러그인을 빌드하는 방법"
---

# Secret Source 플러그인 빌드

Secret source는 외부 시크릿 관리자(볼트, 비밀번호 관리자, OS 키 저장소, 사용자 지정 스크립트)에서 프로세스 시작 시 환경 변수로 프로바이더 자격 증명을 가져옵니다. `~/.hermes/.env`가 로드된 후, Hermes가 자격 증명을 읽기 전에 실행됩니다. Bitwarden, 1Password, 일반 명령 헬퍼 소스는 트리 내에 포함되어 제공되며, **그 외 모든 백엔드는 플러그인입니다**. 이 가이드에서는 플러그인을 빌드하는 방법을 다룹니다.

:::tip
번들 세트가 의도적으로 닫혀 있는 것은 [메모리 프로바이더](/developer-guide/memory-provider-plugin)와 동일한 정책입니다. `agent/secret_sources/`에 새 볼트 백엔드를 추가하는 PR은 이 가이드를 안내하며 종료됩니다. 백엔드를 독립 플러그인 저장소로 게시하고 Nous Research Discord(`#plugins-skills-and-skins`)에서 공유하세요.
:::

## 첫 프로세스 부트스트랩 시점

`load_hermes_dotenv()`는 플러그인이 등록되기 **전에** 임포트 시점에 실행되는 경우가 많습니다. 이후 Hermes는 **활성화된** 플러그인 secret source가 구성되어 있으면 플러그인 검색 후 시크릿을 다시 가져옵니다. 활성화 여부는 소스의 `is_enabled(cfg)` 계약을 사용합니다. 표준 형식은 `secrets.<name>.enabled: true`이며, 사용자 지정 활성화도 계속 지원됩니다. 이를 통해 "Bitwarden을 내 볼트로 교체"하는 첫 프로세스 공백(#64177)이 해소됩니다.

- 다시 가져오기는 멱등적이며 fail-open입니다(시작을 차단하지 않음).
- 소스는 오케스트레이터를 통해서만 환경 변수를 제공하며, 소스 자체의 구성이 허용하는 범위를 넘어 다른 플러그인의 시크릿이나 사용자의 전체 시크릿 저장소를 덤프하는 **플러그인 API는 없습니다**.
- 로드 후 `os.environ`을 읽는 것은 모든 동일 프로세스 내 코드에서 가능하지만, 신뢰 경계는 여전히 "활성화된 플러그인은 에이전트 권한으로 실행된다"입니다.

## 프레임워크가 소유하는 것과 사용자가 소유하는 것

오케스트레이터(`agent.secret_sources.registry.apply_all`)는 보안 및 우선순위와 관련된 모든 것을 소유하므로 백엔드가 이를 잘못 처리할 수 없습니다.

| 프레임워크가 소유 | 사용자가 소유 |
|---|---|
| 소스 순서, 매핑된 값과 벌크 값의 우선순위 | 백엔드에서 값 가져오기 |
| 최초 주장 우선 충돌 처리 + 경고 | 참조 형식 검증 |
| `override_existing` 의미(소스 간에는 절대 전달되지 않음) | CLI/SDK/API와 통신 |
| 보호된 부트스트랩 토큰 | 어떤 환경 변수가 부트스트랩 토큰인지 선언 |
| 소스별 wall-clock 타임아웃 | `fetch()`를 충분히 빠르게 유지 |
| 변수별 출처 + `(from X)` 라벨 | 사람이 읽을 수 있는 `label` |
| `os.environ` 쓰기 | 없음 — 환경을 직접 건드리지 않음 |

## 디렉터리 구조

```
~/.hermes/plugins/my-vault/
├── plugin.yaml      # name, description
└── __init__.py      # SecretSource subclass + register(ctx)
```

## SecretSource ABC

`agent.secret_sources.base.SecretSource`를 구현합니다. 필수 메서드는 하나입니다.

```python
from pathlib import Path

from agent.secret_sources.base import (
    ErrorKind,
    FetchResult,
    SecretSource,
    run_secret_cli,
)


class MyVaultSource(SecretSource):
    name = "myvault"          # config section key: secrets.myvault
    label = "My Vault"        # used in startup lines + provenance labels
    shape = "mapped"          # "mapped" (explicit VAR→ref map) or "bulk" (project dump)
    scheme = "mv"             # optional: unique URI scheme you own (mv://...)

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        """Resolve secrets. MUST NOT raise. MUST NOT prompt."""
        result = FetchResult()
        token = os.environ.get("MYVAULT_TOKEN", "").strip()
        if not token:
            result.error = "secrets.myvault.enabled is true but MYVAULT_TOKEN is not set."
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        try:
            proc = run_secret_cli(
                ["myvault-cli", "export", "--json"],
                allow_env=["MYVAULT_TOKEN"],   # ONLY your auth vars — never full os.environ
                timeout=30,
            )
        except RuntimeError as exc:           # spawn failure / timeout
            result.error = str(exc)
            result.error_kind = ErrorKind.BINARY_MISSING
            return result

        if proc.returncode != 0:
            result.error = f"myvault-cli exited {proc.returncode}: {proc.stderr[:200]}"
            result.error_kind = ErrorKind.AUTH_FAILED
            return result

        result.secrets = parse_your_output(proc.stdout)  # {ENV_VAR: value}
        return result

    def protected_env_vars(self, cfg: dict):
        # Your bootstrap token — no source (including yours) may ever overwrite it.
        return frozenset({"MYVAULT_TOKEN"})
```

### 계약 규칙(권고가 아닌 강제 사항)

- **`fetch()`는 절대 예외를 발생시키지 않습니다.** 오류는 `result.error` + `result.error_kind`에 기록합니다. 예외를 발생시키는 fetch는 오케스트레이터가 포착하여 `INTERNAL`로 보고합니다. 이는 계약 위반이며 기능이 아닙니다.
- **`fetch()`는 절대 프롬프트를 표시하지 않습니다.** 시작은 TTY가 아닌 컨텍스트(게이트웨이, cron, Docker)에서 실행됩니다. `run_secret_cli()`는 stdin을 닫으므로 프롬프트를 표시하는 헬퍼는 빠르게 실패합니다. 대화형 인증은 시작 경로가 아닌 CLI 설정 플로에 속합니다.
- **예산 내에서 동기적으로 실행합니다.** 오케스트레이터는 wall-clock 타임아웃을 적용합니다(기본값 120초, `secrets.<name>.timeout_seconds`로 사용자가 조정 가능). 이를 초과하면 `TIMEOUT`을 보고하고 결과를 폐기합니다.
- **가져오는 것은 사용자의 역할이고, 적용하는 것은 오케스트레이터의 역할입니다.** 기여할 매핑을 반환하기만 합니다. 직접 `os.environ`에 쓰지 마세요. 우선순위, 충돌 감지, 출처를 우회하게 됩니다.
- **API 버전 관리.** `SecretSource.api_version`은 현재 `SECRET_SOURCE_API_VERSION`으로 기본 설정됩니다. 레지스트리는 다른 버전으로 빌드된 소스를 시작 시 충돌시키는 대신 경고와 함께 건너뜁니다.

### `shape` 선택

- `mapped` — 사용자가 구성에서 환경 변수 이름을 참조에 명시적으로 연결합니다(1Password의 `env:` 맵과 유사). 의도가 가장 분명합니다. 충돌하는 변수에서는 매핑된 주장이 벌크 주장보다 우선합니다.
- `bulk` — 프로젝트/폴더의 시크릿 전체를 암묵적으로 주입합니다(Bitwarden BSM과 유사). 매핑된 소스에 양보합니다.

### 선택적 훅

| 메서드 | 기본값 | 다음 경우에 재정의 |
|---|---|---|
| `is_enabled(cfg)` | `cfg.get("enabled")` | 사용자 지정 활성화 로직 |
| `override_existing(cfg)` | `cfg.get("override_existing", False)` | 다른 기본값을 원할 때(두 번들 소스는 로테이션을 위해 기본값이 `True`) |
| `protected_env_vars(cfg)` | 비어 있음 | 부트스트랩 토큰이 있을 때(거의 확실히 있음) |
| `fetch_timeout_seconds(cfg)` | 120초 | 백엔드에 다른 예산이 필요할 때 |
| `config_schema()` | `{}` | 설정 화면을 위해 구성 키를 선언할 때 |
| `remediation(kind, cfg)` | 일반적인 `ErrorKind`별 힌트 | 실패 경고가 자체 수정 명령을 가리키기를 원할 때(예: 번들 소스는 `AUTH_FAILED`에 대해 `Run hermes secrets <name> token…`을 반환함). 순수한 kind→string 매핑이어야 합니다. I/O를 수행하지 않고 절대 예외를 발생시키지 않습니다. 힌트를 숨기려면 `""`을 반환합니다. |

## 서브프로세스 안전성: `run_secret_cli()` 사용

백엔드가 CLI를 셸로 실행해야 한다면 `subprocess.run`을 직접 사용하는 대신 공유 헬퍼를 사용하세요. 그러면 감사된 보안 태세를 그대로 얻을 수 있습니다. argv만 사용(`shell=True` 없음), 최소 허용 목록의 자식 환경(소스가 실행될 때 `os.environ`에는 Hermes가 알고 있는 모든 자격 증명이 들어 있으므로 이를 자식 프로세스에 절대 전달하지 않음), `NO_COLOR` + ANSI가 제거된 stderr, 닫힌 stdin, 타임아웃 시 깔끔한 `RuntimeError`가 제공됩니다. 사용자가 제공한 참조 문자열은 `--` 종료자 뒤의 argv로 전달하여 플래그로 해석되지 않도록 하세요.

## 등록

```python
# __init__.py
def register(ctx):
    ctx.register_secret_source(MyVaultSource())
```

다음의 경우 등록이 거부됩니다(로그 경고만 기록되며 충돌하지 않음): `SecretSource` 인스턴스가 아님, 이름이 유효하지 않거나 중복됨, 다른 소스가 `scheme`을 소유함, `api_version`이 잘못됨, `shape`이 `mapped`/`bulk` 외의 값임.

:::note 타이밍
플러그인 검색은 최초 `load_hermes_dotenv()` 호출보다 늦게 시작됩니다. 검색 직후 Hermes는 활성화된 플러그인 secret source를 다시 가져오므로(`reset_secret_source_cache()` + `load_hermes_dotenv()`), 검색을 수행한 프로세스가 이를 실제로 반영합니다 — 위의 [첫 프로세스 부트스트랩 시점](#first-process-bootstrap-timing)을 참고하세요(#64177). 다시 가져오기는 fail-open이며 활성화된 플러그인 소스가 없으면 건너뜁니다. 플러그인 모듈의 임포트 또는 `register(ctx)` 중 `os.environ`을 읽는 코드는 다시 가져오기 전에 실행되므로 동일 소스가 제공하는 자격 증명에 의존할 수 없습니다. 자격 증명이 필요한 작업은 `fetch()` 안에 두세요. 게이트웨이, cron, 서브에이전트 프로세스도 동일한 검색/다시 가져오기 시퀀스를 수행합니다.
:::

## 사용자는 다른 소스와 동일한 방식으로 구성합니다

```yaml
secrets:
  sources: [myvault, bitwarden]   # optional ordering
  myvault:
    enabled: true
    # ... your config_schema keys
```

다중 소스 우선순위, 충돌 경고, `(from My Vault)` 출처 라벨은 모두 자동으로 작동합니다 — 우선순위 단계는 [사용자용 시크릿 문서](/user-guide/secrets/)를 참고하세요.

## 적합성 키트로 검증

Hermes 저장소의 적합성 키트(`tests/secret_sources/conformance.py`)를 플러그인 테스트에서 상속하세요.

```python
import pytest
from tests.secret_sources.conformance import SecretSourceConformance

class TestMyVaultConformance(SecretSourceConformance):
    @pytest.fixture
    def source(self):
        return MyVaultSource()
```

다른 사람에게 문제를 일으키는 규칙을 검사합니다. 잘못된 구성에서도 예외를 발생시키지 않는지, 기계 판독 가능한 오류 종류인지, 기본적으로 비활성화되어 있는지, 타임아웃이 양수인지, 보호된 변수 이름이 유효한지, 전체 `apply_all()` 왕복이 가능한지 확인합니다. 적합성 검사가 통과하는 것이 백엔드가 계약을 준수한다고 말할 수 있는 리뷰 기준입니다.

## ErrorKind 참고

| 종류 | 의미 |
|---|---|
| `NOT_CONFIGURED` | 활성화되었지만 토큰/프로젝트/맵이 없음 |
| `BINARY_MISSING` | 헬퍼 CLI를 찾을 수 없거나 실행할 수 없음 |
| `AUTH_FAILED` / `AUTH_EXPIRED` | 잘못되었거나 만료된 자격 증명 |
| `REF_INVALID` | 시크릿 참조가 검증에 실패함 |
| `NETWORK` | 전송 계층 오류 |
| `EMPTY_VALUE` | 백엔드가 참조에 대해 아무것도 반환하지 않음 — 올바른 자격 증명 위에 `""`을 적용하지 않음 |
| `TIMEOUT` | 가져오기가 예산을 초과함 |
| `INTERNAL` | 그 외 모든 것(버그, 예상하지 못한 형태) |
