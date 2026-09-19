# 명령 도우미 시크릿 소스

시작할 때 직접 도우미 명령을 실행해 자격 증명을 확인합니다 — CLI가 있는 모든 시크릿 저장소를 사용할 수 있습니다. 예를 들어 `keepassxc-cli`, `secret-tool` (GNOME Keyring), `pass`, `gpg`, Vaultwarden CLI 또는 tmpfs 환경 파일을 출력하는 스크립트가 있습니다. 도우미는 stdout에 `KEY=VALUE` 줄을 출력하며, Hermes는 [Bitwarden](./bitwarden) 및 [1Password](./onepassword)와 동일한 오케스트레이터를 통해 이를 적용하므로 여러 소스를 원하는 조합으로 동시에 활성화할 수 있습니다.

## 작동 방식

1. `config.yaml`에서 도우미 명령을 구성합니다 (`.env`에는 구성하지 않음 — 명령은 구성이고 `.env`는 값을 보관합니다).
2. 시작 시 `.env`를 로드한 후 Hermes가 `/bin/sh -c`를 통해 도우미를 한 번 실행하고 stdout을 dotenv 블롭으로 파싱합니다.
3. 파싱된 키는 표준 우선순위 계층을 따릅니다: `override_existing: true`가 아니면 `.env`/셸이 우선하고, 충돌하는 변수에서는 매핑된 소스가 이 대량 소스보다 우선하며, 최초 클레임이 승리합니다.

```yaml
secrets:
  command:
    enabled: true
    command: "cat /run/user/1000/hermes-secrets.env"
    # or any vault CLI that dumps KEY=VALUE lines:
    # command: "pass show hermes/env"
    # command: "secret-tool lookup service hermes-env"
```

## 구성

| 키 | 기본값 | 기능 |
|---|---|---|
| `enabled` | `false` | 마스터 스위치입니다. |
| `command` | `""` | `/bin/sh -c`를 통해 실행되는 도우미입니다. stdout에 `KEY=VALUE` 줄을 출력해야 합니다. |
| `helper_timeout_seconds` | `3` | 한 번의 도우미 실행에 적용되는 하드 타임아웃입니다. 의도적으로 짧게 설정되어 있으므로 도우미는 빠르고 비대화형이어야 합니다 (잠금 해제 프롬프트, 터치/PIN 없음). |
| `override_existing` | `false` | 도우미 값이 `.env`/셸 값을 덮어씁니다. 로컬 도우미는 중앙 집중식 교체 권한이 아니므로 Bitwarden/1Password와 달리 기본적으로 꺼져 있습니다. |

## 보안 모델

- 도우미 명령 문자열은 **사용자가 구성하는 값**이며, 사용자가 관리하는 `.env` 파일과 동일한 신뢰 수준을 가집니다.
- 출력은 1MiB로 엄격히 제한됩니다. 폭주하는 도우미가 시작을 멈추게 할 수 없습니다 (타임아웃 시 프로세스 그룹 종료).
- 도우미의 **stderr는 버려집니다** — vault CLI 진단 정보에 시크릿이 포함될 수 있으므로 Hermes 출력에 절대 전달되지 않습니다. 실패 시 명령 문자열이 아니라 구조화된 필드만 기록됩니다 (종료 코드 / 시그널 / errno).
- 공백만 포함된 값은 "값 없음"으로 처리됩니다 — 자리 표시자 항목이 Authorization 헤더로 전달되지 않습니다.
- POSIX 전용 (`/bin/sh` 필요)입니다. Windows에서는 소스가 구성되지 않은 것으로 보고하고 시작을 계속합니다.

## 실패 모드

시작은 절대 차단되지 않습니다. 오류는 한 줄과 `→` 해결 방법 힌트를 함께 출력합니다:

| 증상 | 원인 | 해결 방법 |
|---|---|---|
| `secrets.command.command is empty` | 명령 없이 활성화됨 | config.yaml에서 `secrets.command.command` 설정 |
| `helper command failed` | 0이 아닌 종료, 타임아웃, 생성 실패 | 셸에서 도우미를 수동으로 실행해 실제 오류 확인 (Hermes가 의도적으로 stderr를 버림) |
| `helper output was not a KEY=VALUE map` | 도우미가 단일 값이나 잘못된 내용을 출력함 | 도우미가 dotenv 형식의 줄을 출력하도록 수정 |

## 플러그인과 비교해 언제 사용할지

명령 소스는 번들 통합 기능이 없는 vault를 위한 우회 수단입니다. 복잡한 CLI 작업을 긴 스크립트로 감싸게 된다면 대신 적절한 [시크릿 소스 플러그인](/developer-guide/secret-source-plugin)을 고려하세요 — 플러그인은 캐싱, 출처 레이블, 타입이 지정된 구성을 제공합니다.
