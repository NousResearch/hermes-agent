---
sidebar_position: 14
title: "Egress 프록시 내부 구조"
description: "iron-proxy egress 방화벽이 Hermes와 통합되는 방식 — 모듈 구성, 수명 주기, 보안 불변식 및 확장 지점"
---

# Egress 프록시 내부 구조

이 페이지에서는 기여자 및 플러그인 작성자의 관점에서 egress 자격 증명 주입 방화벽(`hermes egress` / iron-proxy)의 아키텍처를 다룹니다. 최종 사용자용 설정 및 사용 문서는 [Egress 프록시](../user-guide/egress/iron-proxy.md)에 있습니다.

위협 모델과 상위 수준의 설계는 사용자 페이지에 요약되어 있습니다. 이 페이지에서는 이것이 *어떻게* 연결되어 있는지, 보안과 관련된 코드가 어디에 있는지, 그리고 이를 수정할 때 보존해야 하는 불변식이 무엇인지 설명합니다.

## 모듈 구성

```text
agent/proxy_sources/iron_proxy.py     Core: binary install, CA gen, config build,
                                       subprocess lifecycle, mappings I/O, PID/nonce
                                       defense.  Pure-function surface where possible.

hermes_cli/proxy_cli.py               Wizard + slash command handlers.
                                       `hermes egress {install,setup,start,stop,
                                       status,disable,config}`.  Wires the
                                       core module into argparse.

hermes_cli/main.py:_dispatch_egress   Top-level subparser dispatcher.
                                       dest='egress_command' (intentionally
                                       disjoint from the inbound OAuth
                                       `hermes proxy` subparser, which uses
                                       dest='proxy_command').

hermes_cli/config.py: proxy schema    The `proxy:` block in DEFAULT_CONFIG.
                                       Adding a knob means: add it here, add a
                                       wizard prompt or `setdefault` in
                                       proxy_cli.cmd_setup, and document it
                                       in the user-guide page.

tools/environments/docker.py
  _egress_proxy_args_for_docker()     Builds the volume_args / env_overrides /
                                       host_args triple that the Docker backend
                                       injects when `proxy.enabled: true`.

  DockerEnvironment.__init__          Docker-side merge logic: collision
                                       detection against critical egress vars,
                                       NODE_OPTIONS append-merge via the
                                       _HERMES_EGRESS_NODE_OPTIONS_APPEND
                                       sentinel, enforce_on_docker precedence.

tests/test_iron_proxy.py              Hermetic tests (~70).  Binary install
                                       path, config build, mappings I/O,
                                       subprocess lifecycle, docker arg builder,
                                       deny CIDR defaults, bind policy, CA
                                       TOCTOU, ensure_audit_log behaviour, etc.

tests/test_iron_proxy_cli.py          CLI handler unit tests (~20).  Argparse
                                       wiring, fail-loud paths, BWS refresh
                                       wire-up, dest='egress_command'
                                       regression guard.

tests/test_iron_proxy_e2e.py          Live E2E (gated on HERMES_RUN_E2E=1).
                                       Real iron-proxy binary, real curl,
                                       end-to-end token swap verified.
```

## 수명 주기

```text
hermes egress install
  -> agent.proxy_sources.iron_proxy.install_iron_proxy(force=...)
       Downloads pinned tarball + checksums.txt from GitHub Releases.
       SHA-256 verification before extraction.
       tarfile.extract(..., filter="data") on Python 3.12+ (PEP 706);
         falls back to plain extract on older Python with member-name
         sanitisation via _pick_tar_member.
       Stage into ~/.hermes/bin/.iron-proxy_XXXX, chmod 755, os.replace
         to ~/.hermes/bin/iron-proxy (atomic).
       _VERSION_CACHE.pop(target) so a forced reinstall re-probes
         --version on next call.

hermes egress setup [--from-bitwarden | --no-bitwarden] [--rotate-tokens]
  -> proxy_cli.cmd_setup
       Step 1. find_iron_proxy(install_if_missing=False) -> install if absent.
       Step 2. ensure_ca_cert()
                 Run openssl genrsa + req via subprocess.
                 Write CA key via os.open(O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW, 0o600)
                   + os.replace.  Never exists on disk under default umask.
                 Write CA cert with 0o644 (public).
       Step 3. discover_provider_mappings() or pull names from BWS via
                 fetch_bitwarden_secrets() when --from-bitwarden.
                 merge_mappings(existing=load_mappings(), discovered,
                                rotate=args.rotate_tokens) preserves prior
                 tokens unless --rotate-tokens is passed.
                 discover_uncovered_providers() and surface warnings.
       Step 4. ensure_audit_log(audit_log_path)   # raises on OSError
               build_proxy_config(...) with defaults applied at the call site
                 (deny CIDRs default, bind policy from _default_http_listen).
               write_proxy_config(cfg)            # atomic via .tmp + os.replace, 0o600
               write_mappings(mappings)           # atomic, 0o600
       Step 5. proxy_cfg["enabled"] = True; credential_source preservation logic
               (do NOT silently downgrade bitwarden -> env on re-run);
               save_config(cfg).

hermes egress start
  -> proxy_cli.cmd_start
       Pre-checks (refuse-start path):
         - credential_source=bitwarden? -> pre-validate access_token_env + project_id
       -> iron_proxy.start_proxy(
            refresh_secrets_from_bitwarden=...,
            bitwarden_config=...,
          )
            existing=_read_pid(); if alive, idempotent return.
            _build_proxy_subprocess_env(...):  ALLOWLIST + mapped real_env_names,
              strip HTTPS_PROXY/etc. to avoid recursion, optional BWS refresh
              (raises on missing values unless allow_env_fallback=true).
            Plant nonce: _proxy_nonce = sha256(urandom(16)); env[NONCE_ENV] = ...
            Open log_path via O_NOFOLLOW + 0o600 + st_uid check.
            Popen with stdin=DEVNULL, stdout=log_fd, stderr=STDOUT,
              start_new_session=True (POSIX).
            Close parent's log_fd in finally.
            _write_pidfile_safely(pidfile, proc.pid)
              O_EXCL + O_NOFOLLOW + uid check + persisted nonce sidecar.
              FileExistsError -> discriminate live vs stale, retry once if stale.
            Install SIGINT/SIGTERM handlers (main-thread only).
            Poll loop (do-while shape):
              while True:
                if proc.poll() is not None: tail log + unlink pidfile + raise
                if _port_listening(probe_host, tunnel_port): break  # probe_host = configured bind host
                if time.time() >= deadline: break  (do-while: checked AFTER first probe)
                time.sleep(0.1)
            If not listening at exit: _kill_and_wait(proc) + unlink pidfile + raise.

hermes egress stop
  -> iron_proxy.stop_proxy
       _read_pid + _pid_alive guard.
       starttime_before = _pid_proc_starttime(pid)   # Linux only; None elsewhere
       os.kill(pid, SIGTERM)
       Wait up to 5s for graceful exit.
       After grace: re-check starttime + _pid_alive.
         If recycled (starttime drift OR _pid_alive False), DO NOT SIGKILL.
         Otherwise os.kill(pid, _KILL_SIGNAL).
       _cleanup_state_files: unlink pidfile + nonce sibling.
```

## 보안 불변식

이것들은 하중을 지탱하는 속성입니다. 모듈을 수정한다면 반드시 보존해야 합니다. 회귀 테스트가 있는 경우 테스트 이름을 함께 표시했습니다.

### 파일 시스템 권한

| 경로 | 모드 | 테스트 |
|---|---|---|
| `~/.hermes/proxy/` (dir) | `0o700` | `test_proxy_state_dir_is_0o700` |
| `ca.key` | `0o600` | `test_ca_key_created_with_0o600` |
| `ca.crt` | `0o644` | (암묵적; `ensure_ca_cert`의 chmod 호출) |
| `proxy.yaml` | `0o600` | (원자적 이름 변경 후 `write_proxy_config`에서 chmod) |
| `mappings.json` | `0o600` | (원자적 이름 변경 후 `write_mappings`에서 chmod) |
| `iron-proxy.pid` | `0o600` | (`_write_pidfile_safely`의 `os.open(..., 0o600)` 모드) |
| `iron-proxy.nonce` | `0o600` | (`_write_pidfile_safely`의 `os.open(..., 0o600)` 모드) |
| `audit.log` | `0o600` | `test_ensure_audit_log_creates_with_0o600` |
| `iron-proxy.log` | `0o600` | (`os.open(..., 0o600)` + `fchmod`) |

모든 쓰기 경로는 `os.open(O_WRONLY | O_CREAT | O_NOFOLLOW, 0o600)` + `os.fstat().st_uid` 검사를 사용합니다. 기본 umask가 적용되는 창을 노출하므로 `shutil.copy2` + `os.chmod`는 금지됩니다.

### 서브프로세스 환경 최소화

`_build_proxy_subprocess_env`는 `os.environ.copy()`를 사용하면 안 됩니다. 허용 목록은 `_PROXY_SUBPROCESS_ENV_ALLOWLIST`(PATH, HOME, locale 등)와 `load_mappings()`가 참조하는 환경 변수 이름에 더해진 값입니다. 그 밖의 모든 값은 호스트에 남습니다.

회귀 테스트: `test_subprocess_env_strips_unrelated_secrets`, `test_subprocess_env_strips_proxy_recursion_vars`, `test_subprocess_env_keeps_infrastructure_vars`.

### 바인드 정책

`_default_http_listen`은 단일 요소 목록을 반환합니다. Linux에서는 Docker 브리지 게이트웨이 IP를 사용합니다(컨테이너는 `host.docker.internal:host-gateway`를 통해 프록시에 도달하며, 이는 브리지 게이트웨이로 해석됨 — 그 환경에서 loopback 바인드는 컨테이너 내부에서 도달할 수 없음). macOS/Windows Docker Desktop에서는 loopback을 사용합니다(VPNkit이 `host.docker.internal`을 호스트로 라우팅). 감지 가능한 docker0 브리지가 없는 Linux에서는 경고와 함께 loopback으로 대체합니다. `0.0.0.0`도, `:PORT`도 절대 사용하지 않습니다(INADDR_ANY).

`_detect_docker_bridge_ip`는 `ipaddress.IPv4Address`를 통해 검증하고 `is_unspecified` / `is_loopback` / `is_multicast` / `is_reserved` / `is_link_local` / `is_global`을 거부합니다. PATH에 있는 악성 `ip` shim은 `0.0.0.0`을 주입할 수 없습니다.

**v0.39 스키마 제약 및 리스너 역할(바이너리에 대해 실시간 검증됨):** 바이너리의 `config.Proxy` 구조체에는 단수형 리스너 필드만 있으며 `http_listens`(복수형) 목록은 없습니다. `tunnel_listen`은 CONNECT + MITM 리스너입니다(`HTTPS_PROXY` 트래픽이 도달하는 곳). `http_listen`은 절대 형식의 일반 HTTP 전달만 처리합니다(여기에 CONNECT를 보내면 일반 요청으로 upstream에 중계되어 400이 반환됨). 따라서 `build_proxy_config`는 플랫폼 바인드 호스트에서 `tunnel_port`에 `tunnel_listen`을, `tunnel_port + 1`에 `http_listen`을 바인드합니다. Docker 백엔드는 `HTTPS_PROXY`를 `tunnel_port`로, `HTTP_PROXY`를 `tunnel_port + 1`로 설정합니다.

liveness probe(`start_proxy` 폴링 루프, `get_status`)는 `_read_http_listen_from_config()`를 통해 구성된 바인드 호스트를 읽고 해당 호스트를 probe합니다 — 하드코딩된 loopback probe는 브리지에 바인드된 데몬을 비정상으로 보고합니다.

회귀 테스트: `test_default_bind_is_loopback_not_zero_zero`(INADDR_ANY가 없고 렌더링된 yaml에 `http_listens`가 없음을 검증), `test_default_bind_uses_docker_bridge_on_linux`, `test_default_bind_falls_back_to_loopback_without_bridge`, `test_default_bind_is_loopback_on_macos`, `test_detect_docker_bridge_ip_rejects_dangerous`(8개 공격 입력에 대해 parametrized).

### 메트릭 포트 충돌

iron-proxy v0.39에서 `metrics.listen`의 기본값은 `:9090`이며 Hermes의 기본 `tunnel_port: 9090`과 동일한 포트입니다. `build_proxy_config`는 `metrics.listen: 127.0.0.1:0`을 명시적으로 고정해야 합니다. 이렇게 하면 운영자가 선택한 `tunnel_port`와 관계없이 메트릭 바인딩이 절대 충돌할 수 없는 loopback 임시 포트를 사용하게 됩니다.

회귀 테스트: `test_metrics_listener_pinned_to_loopback_ephemeral`.

### 기본 deny CIDR

`_DEFAULT_UPSTREAM_DENY_CIDRS`는 loopback(v4 + v6), link-local(IMDS 169.254.169.254 및 IPv4-mapped-v6 형식 포함), RFC1918, IPv6 ULA, CGNAT, RFC2544 benchmark 범위를 포함합니다. `build_proxy_config(..., upstream_deny_cidrs=None)`은 기본값을 내보내야 하며, 명시적인 빈 목록만 이 기능을 해제합니다.

회귀 테스트: `test_default_deny_cidrs_present_when_unspecified`, `test_default_deny_includes_ipv4_mapped_v6`.

### 감사 로그 fail-loud

`ensure_audit_log`는 모든 `OSError`에서 `RuntimeError`를 발생시킵니다. 고정된 v0.39에서 데몬은 이 파일을 절대 쓰지 않습니다(`log.audit_path` 필드가 없음). 따라서 `cmd_setup`은 실패를 WARNING으로 처리하고(버전이 올라가기 전에는 파일이 핵심 동작을 담당하지 않음) 성공 행을 "reserved"로 한정합니다. 고정 버전이 `log.audit_path`를 지원하는 버전으로 이동하면 다시 검토해야 합니다. 사전 생성은 첫 바이트부터 0o600을 보장하는 핵심 동작이 되며 마법사는 다시 fail-loud해야 합니다.

**v0.39 스키마 제약:** `log.audit_path`는 iron-proxy v0.39의 `config.Log` 구조체 필드가 아니므로 `build_proxy_config`는 `audit_log` kwarg를 받지만 렌더링된 yaml에는 출력하지 않습니다. v0.39에서 요청별 레코드는 데몬 수준 이벤트와 함께 `iron-proxy.log`에 기록됩니다. `audit.log` 파일은 고정 버전이 별도 스트림을 지원하는 버전으로 올라갈 때 개인정보 보호 계약이 유지되도록 `O_NOFOLLOW` 및 `0o600`으로 여전히 사전 생성됩니다.

회귀 테스트: `test_ensure_audit_log_raises_on_immutable_parent`, `test_audit_log_kwarg_does_not_inject_audit_path_v039`.

### Bitwarden 모드 fail-loud

`credential_source: bitwarden`이고 `proxy.allow_env_fallback: false`(기본값)인 경우:

- access token 환경 변수 누락 -> `cmd_start`가 거부.
- `project_id` 누락 -> `cmd_start`가 거부.
- `bws secret list`가 하나 이상의 매핑된 provider에 대해 값을 반환하지 않음 -> `_build_proxy_subprocess_env`가 발생.

BW 모드에서 호스트 환경으로 대체하면 BW 경로가 해결하려는 바로 그 오래된 값 문제를 다시 도입하게 됩니다.

회귀 테스트: `test_cmd_start_refuses_when_bitwarden_token_missing`(CLI 계층); `_build_proxy_subprocess_env`의 strict-mode 검증(데몬 계층).

### docker_env 충돌 감지

`enforce_on_docker: true`일 때 egress를 제어하는 변수(HTTPS_PROXY, SSL_CERT_FILE, NODE_EXTRA_CA_CERTS 등) 또는 매핑된 `real_env_name`(OPENROUTER_API_KEY 등)에 대한 `docker_env` 재정의가 있으면 컨테이너가 시작되기 전에 `RuntimeError`가 발생합니다.

회귀 테스트: `test_docker_env_collision_with_proxy_raises_when_enforce`.

### PID 재사용 방어

`_pid_alive`는 `argv[0]` basename 일치를 신뢰하기 전에 프로세스 내 `_proxy_nonce`(동일 프로세스의 경우) 또는 디스크의 `iron-proxy.nonce`(CLI 간의 경우)를 반드시 확인해야 합니다. `stop_proxy`는 SIGKILL 전에 `/proc/<pid>/stat` starttime을 다시 확인하고 starttime drift가 있으면 신호를 억제해야 합니다.

회귀 테스트: `test_stop_proxy_suppresses_sigkill_on_pid_recycle`, `test_pid_proc_starttime_parses_comm_with_parens`, `test_persisted_nonce_roundtrip`.

### 재설정 시 토큰 보존

`merge_mappings(existing, discovered, rotate=False)`는 겹치는 provider에 대해 이전 토큰을 반환해야 합니다. `hermes egress setup`을 다시 실행해도 실행 중인 sandbox가 조용히 401을 반환하게 해서는 안 됩니다. `--rotate-tokens`가 명시적인 선택입니다.

회귀 테스트: `test_merge_mappings_preserves_existing_tokens`, `test_merge_mappings_rotate_mints_fresh_tokens`.

### `credential_source` 보존

명시적인 `--no-bitwarden` 플래그 없이 `cmd_setup`은 `credential_source: bitwarden`을 `env`로 낮추면 안 됩니다. `hermes egress setup`(플래그 없음)을 실행하면 이전에 구성된 값을 보존합니다.

CLI 테스트의 `cmd_setup` 흐름을 통해 테스트됩니다(`--from-bitwarden` 후 일반 `setup` 재실행 시 Bitwarden 보존 경로가 실행됨).

## 확장 지점

### 새로운 bearer-token provider 추가

`iron_proxy.py`의 `_BEARER_PROVIDERS`는 환경 변수 이름 -> upstream 호스트 튜플을 매핑합니다. 항목을 추가하면 `discover_provider_mappings()`에서 검색할 수 있으며, 환경 변수가 존재할 때 마법사가 자동으로 토큰을 발급합니다.

```python
_BEARER_PROVIDERS: Dict[str, Tuple[str, ...]] = {
    ...,
    "MY_PROVIDER_API_KEY": ("api.myprovider.com",),
}
```

또한 `_DEFAULT_ALLOWED_HOSTS`를 업데이트하여 기본적으로 프록시가 upstream을 허용하도록 합니다. 확인하려면 `test_discover_provider_mappings_*`를 실행합니다.

### 새로운 header-token provider 추가(x-api-key 계열)

provider가 고정된 NON-Authorization 헤더(Anthropic의 `x-api-key`, Azure의 `api-key`, Gemini의 `x-goog-api-key` 등)를 사용해 인증한다면 `_HEADER_AUTH_PROVIDERS`에 추가합니다 — iron-proxy의 `secrets.replace.match_headers`는 임의의 헤더 이름을 대상으로 하므로, 이런 provider는 우선 교체되는 provider입니다.

```python
_HEADER_AUTH_PROVIDERS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    ...,
    "MY_PROVIDER_API_KEY": {
        "hosts": ("api.myprovider.com",),
        "match_headers": ("x-my-auth-header", "Authorization"),
        "aliases": (),
    },
}
```

`aliases`는 동일한 자격 증명의 서로 교환 가능한 환경 변수 이름(예: `GEMINI_API_KEY`의 `GOOGLE_API_KEY`)에만 사용합니다. 별칭 이름은 하나의 매핑으로 합쳐집니다. 같은 호스트에 `require: true` 규칙이 두 개 있으면 서로의 요청을 거부하기 때문입니다. `_DEFAULT_ALLOWED_HOSTS`도 업데이트합니다.

### 새로운 서명 인증 provider 추가(미지원)

provider가 SigV4 / SDK가 발급하는 OAuth / 요청 서명을 사용한다면 정적 헤더 교체로는 처리할 수 없습니다. 해당 환경 변수를 `_NON_BEARER_PROVIDERS`에 추가하여 마법사와 `hermes egress status`가 이를 경고하도록 합니다.

```python
_NON_BEARER_PROVIDERS: Tuple[str, ...] = (
    ...,
    "MY_SIGNED_PROVIDER_ACCESS_KEY",
)
```

### Docker가 아닌 백엔드에 iron-proxy 연결

`_egress_proxy_args_for_docker`는 Docker 전용입니다. 유사한 연결이 필요한 백엔드는 다음을 수행하는 자체 대응 로직이 필요합니다.

1. `load_config().get("proxy", {})`를 읽고 `enabled`가 false이면 빈 인자를 반환합니다.
2. `iron_proxy.get_status()`를 호출하고 `configured` / `pid` / `listening` / `ca_cert_path` 실패 경로에서 `enforce` 의미를 표면화합니다.
3. `iron_proxy.load_mappings()`를 호출하고 비어 있으면서 `enforce_on_docker: true`이면 마운트를 거부합니다.
4. 7개 환경 변수(HTTPS_PROXY, NO_PROXY, REQUESTS_CA_BUNDLE, SSL_CERT_FILE, CURL_CA_BUNDLE, NODE_EXTRA_CA_CERTS, HERMES_EGRESS_PROXY)와 매핑별 `HERMES_PROXY_TOKEN_<NAME>` 변수를 설정합니다.
5. 런타임이 신뢰할 경로(일반적으로 `/etc/ssl/certs/hermes-egress-ca.crt`)에 CA 인증서를 sandbox로 배포합니다.
6. 사용자의 백엔드별 환경 설정과의 충돌 감지를 구현합니다.

Docker 구현은 약 150줄입니다. Modal / Daytona / SSH에도 비슷한 분량이 필요할 것으로 예상합니다.

### 요청별 감사 이벤트 구독

현재 고정된 v0.39에서 iron-proxy는 `~/.hermes/proxy/iron-proxy.log`에 줄 단위 JSON을 기록합니다(데몬 및 요청별 레코드가 결합됨; 사용자 가이드의 "Logging on iron-proxy v0.39" 참조). 플러그인 또는 외부 감시자는 이 파일을 tail하며 허용 목록 거부, 비밀 값 교체 또는 upstream 오류에 반응할 수 있습니다. 고정 버전이 `log.audit_path`를 지원하는 버전으로 올라가면 요청별 스트림은 `audit.log`로 이동하며, 해당 경로를 감시하도록 연결된 감시자는 운영자 조치 없이 활성화됩니다. 스키마는 [docs.iron.sh/audit](https://docs.iron.sh/audit)에 문서화되어 있습니다(link).

## 테스트

```bash
# Hermetic suite (no network, no real binary)
scripts/run_tests.sh tests/test_iron_proxy.py tests/test_iron_proxy_cli.py

# Live E2E (real binary, real curl, real CONNECT tunnel)
HERMES_RUN_E2E=1 scripts/run_tests.sh tests/test_iron_proxy_e2e.py

# Live PTY smoke against `hermes egress`
HERMES_HOME=/tmp/hermes-egress-test python3 -m hermes_cli.main egress --help
HERMES_HOME=/tmp/hermes-egress-test python3 -m hermes_cli.main egress setup --help
```

CLI는 argparse를 사용하므로 `--help`는 새 플래그가 올바르게 등록되었는지 확인하는 좋은 첫 번째 점검입니다.

## 함께 보기

- 사용자용 설정 및 문제 해결: [Egress 프록시](https://hermes-agent.nousresearch.com/docs/user-guide/egress/iron-proxy)
- Docker 백엔드 내부 구조: [Docker](https://hermes-agent.nousresearch.com/docs/user-guide/docker)
- Bitwarden Secrets Manager 통합: [`hermes secrets bitwarden`](https://hermes-agent.nousresearch.com/docs/user-guide/secrets/bitwarden)
- CLI 명령어 참조: [`hermes egress`](https://hermes-agent.nousresearch.com/docs/reference/cli-commands#hermes-egress)
- sandbox에 주입되는 환경 변수: [Egress 프록시(sandbox-injected)](https://hermes-agent.nousresearch.com/docs/reference/environment-variables#egress-proxy-sandbox-injected)
