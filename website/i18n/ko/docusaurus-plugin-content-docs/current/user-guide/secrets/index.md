# 시크릿

Hermes는 `~/.hermes/.env`에 저장하는 대신 프로세스 시작 시 외부 시크릿 관리자에서 API 키를 가져올 수 있습니다. 시크릿 관리자의 부트스트랩 토큰은 `.env`에 보관하고, 그 외 모든 제공자 키(OpenAI, Anthropic, OpenRouter 등)는 관리자에 보관하여 중앙에서 교체할 수 있습니다.

지원 항목:

- [Bitwarden Secrets Manager](./bitwarden) — `bws` CLI를 지연 설치하며, 무료 티어로도 작동합니다.
- [1Password](./onepassword) — 공식 `op` CLI를 통한 `op://` 참조, 서비스 계정 또는 데스크톱 세션 인증.
- [Command helper](./command) — 사용자 구성 헬퍼가 `KEY=VALUE` 라인을 출력하는 모든 CLI 볼트(`keepassxc-cli`, `secret-tool`, `pass`, 사용자 지정 스크립트).

## 여러 소스를 동시에 사용하기

둘 이상의 시크릿 소스를 동시에 활성화할 수 있습니다. 예를 들어 개인 볼트 플러그인과 팀 Bitwarden 프로젝트를 함께 사용할 수 있습니다. 소스는 환경 변수별로 결정적인 우선순위 사다리에 따라 조합됩니다.

1. **기본적으로 사용자의 `.env` / 셸이 우선합니다.** 소스 자체에 `override_existing: true`가 설정된 경우에만 기존 값을 대체합니다(Bitwarden은 중앙 교체가 작동하도록 기본값이 true입니다).
2. **매핑된 소스가 대량 소스보다 우선합니다.** 환경 변수를 참조에 명시적으로 연결하는(`env:` 맵) 소스는 전체 프로젝트의 시크릿을 암묵적으로 주입하는 소스보다 목록 순서와 관계없이 우선합니다.
3. **첫 번째 소스가 우선합니다.** 동일한 형태 안에서는 선택 사항인 `secrets.sources` 목록의 순서(또는 등록 순서)가 결정합니다. 이미 확보된 변수에 대한 이후 요청은 건너뛰며, 이때 조용히 처리하지 않고 시작 경고를 표시합니다.

`override_existing`가 한 소스에서 이미 확보한 변수를 다른 소스가 덮어쓰도록 허용하는 일은 없으며, 어떤 소스도 다른 소스의 부트스트랩 토큰(예: `BWS_ACCESS_TOKEN`)을 덮어쓸 수 없습니다.

```yaml
secrets:
  sources: [bitwarden]     # optional explicit ordering
  bitwarden:
    enabled: true
    project_id: "..."
```

소스가 주입한 모든 자격 증명에는 출처가 표시됩니다. 설정 흐름과 `hermes model`은 감지된 키 옆에 `(from Bitwarden)`을 표시하므로 값의 출처를 항상 알 수 있습니다.

## 프로필과 공유 볼트

두 가지 오케스트레이터 수준 설정으로 하나의 공유 볼트를 [프로필](../profiles) 간에 안전하게 사용할 수 있습니다.

- **`secrets.preserve_existing`** — 기존 `.env` / 셸 값을 항상 우선할 환경 변수 이름 목록입니다. `override_existing: true`인 소스보다도 우선합니다. 다른 모든 항목은 중앙에서 교체하면서 프로필별 플랫폼 시크릿(예: 프로필마다 의도적으로 다른 `FEISHU_APP_SECRET`)에 사용합니다.

  ```yaml
  secrets:
    preserve_existing: [FEISHU_APP_SECRET, TELEGRAM_BOT_TOKEN]
  ```

- **프로필 별칭**(기본적으로 활성화, 비활성화하려면 `secrets.profile_alias: false`) — Hermes가 이름이 지정된 프로필에서 실행될 때, `FOO_<PROFILE>`이라는 볼트 시크릿도(자격 증명 형태의 접미사만 허용: `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_KEY`, `*_PASSWORD`) 표준 `FOO`를 채웁니다. 공유 프로젝트에 `TELEGRAM_BOT_TOKEN_MILLA`를 저장하면 `milla` 프로필의 어댑터(고정 이름 `TELEGRAM_BOT_TOKEN`을 읽음)가 올바른 값을 자동으로 받습니다. 볼트가 표준 이름으로 직접 제공하는 변수는 별칭보다 항상 우선합니다.

두 설정은 모든 소스(번들 및 플러그인)에 적용됩니다. 백엔드가 아니라 오케스트레이터에 속하기 때문입니다.

## 자체 백엔드 추가

서드파티 시크릿 관리자는 코어 PR이 아니라 독립형 플러그인으로 제공합니다. 백엔드는 `agent.secret_sources.base.SecretSource`를 상속하고(필수 메서드 하나: `fetch(cfg, home_path) -> FetchResult`), 플러그인의 `register(ctx)`에서 `ctx.register_secret_source(MySource())`를 통해 등록합니다. 오케스트레이터가 우선순위, 충돌 처리, 시간 제한, 출처 정보를 관리하며, 소스는 가져오기만 담당합니다. 계약 규칙, 서브프로세스 안전 헬퍼, 적합성 키트를 포함한 전체 가이드: [시크릿 소스 플러그인 구축](/developer-guide/secret-source-plugin).

번들 세트는 의도적으로 제한되어 있습니다(메모리 제공자와 동일한 정책). Bitwarden과 1Password는 트리 내에 포함됩니다. 그 외의 모든 항목(Infisical, Proton Pass, HashiCorp Vault, AWS Secrets Manager, OS 키 저장소)은 플러그인 저장소에 속합니다. Nous Research Discord(`#plugins-skills-and-skins`)에서 공유하세요.
