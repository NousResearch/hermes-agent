---
sidebar_position: 14
title: "Egress proxy internals"
description: "How the iron-proxy egress firewall integrates with Hermes — module layout, lifecycle, security invariants, and extension points"
---

# Egress proxy internals

This page covers the architecture of the egress credential-injection firewall (`hermes egress` / iron-proxy) from a contributor / plugin author's perspective. End-user setup + usage docs live at [Egress proxy](../user-guide/egress/iron-proxy.md).

The threat model and high-level design are summarised on the user page; this page is about *how* it's wired, where the security-relevant code lives, and what invariants you have to preserve if you touch it.

## Module layout

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

## Lifecycle

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

## Security invariants

These are the load-bearing properties.  If you touch the module, you must preserve them.  Where there's a regression test, it's named.

### Filesystem perms

| Path | Mode | Test |
|---|---|---|
| `~/.hermes/proxy/` (dir) | `0o700` | `test_proxy_state_dir_is_0o700` |
| `ca.key` | `0o600` | `test_ca_key_created_with_0o600` |
| `ca.crt` | `0o644` | (implicit; chmod call in `ensure_ca_cert`) |
| `proxy.yaml` | `0o600` | (chmod after atomic rename in `write_proxy_config`) |
| `mappings.json` | `0o600` | (chmod after atomic rename in `write_mappings`) |
| `iron-proxy.pid` | `0o600` | (`os.open(..., 0o600)` mode in `_write_pidfile_safely`) |
| `iron-proxy.nonce` | `0o600` | (`os.open(..., 0o600)` mode in `_write_pidfile_safely`) |
| `audit.log` | `0o600` | `test_ensure_audit_log_creates_with_0o600` |
| `iron-proxy.log` | `0o600` | (`os.open(..., 0o600)` + `fchmod`) |

All write paths use `os.open(O_WRONLY | O_CREAT | O_NOFOLLOW, 0o600)` + `os.fstat().st_uid` check.  `shutil.copy2` + `os.chmod` is forbidden because it leaks a default-umask window.

### Subprocess env minimisation

`_build_proxy_subprocess_env` MUST NOT use `os.environ.copy()`.  The allowlist is `_PROXY_SUBPROCESS_ENV_ALLOWLIST` (PATH, HOME, locale, etc.) plus the env names referenced by `load_mappings()`.  Everything else stays on the host.

Regression: `test_subprocess_env_strips_unrelated_secrets`, `test_subprocess_env_strips_proxy_recursion_vars`, `test_subprocess_env_keeps_infrastructure_vars`.

### Bind policy

`_default_http_listen` returns a single-element list: on Linux the docker bridge gateway IP (containers reach the proxy via `host.docker.internal:host-gateway`, which resolves to the bridge gateway — a loopback bind is unreachable from inside containers there); on macOS/Windows Docker Desktop, loopback (VPNkit routes `host.docker.internal` to the host).  Linux without a detectable docker0 bridge falls back to loopback with a warning.  Never `0.0.0.0`, never `:PORT` (INADDR_ANY).

`_detect_docker_bridge_ip` validates via `ipaddress.IPv4Address` and rejects `is_unspecified` / `is_loopback` / `is_multicast` / `is_reserved` / `is_link_local` / `is_global`.  A hostile `ip` shim on PATH cannot inject `0.0.0.0`.

**v0.39 schema constraint and listener roles (verified live against the binary):** the binary's `config.Proxy` struct has only singular listener fields — there is no `http_listens` (plural) list.  `tunnel_listen` is the CONNECT + MITM listener (what `HTTPS_PROXY` traffic hits); `http_listen` only handles absolute-form plain-HTTP forwards (a CONNECT sent to it is relayed upstream as a regular request and 400s).  `build_proxy_config` therefore binds `tunnel_listen` on `tunnel_port` and `http_listen` on `tunnel_port + 1`, both on the platform bind host.  The Docker backend sets `HTTPS_PROXY` to `tunnel_port` and `HTTP_PROXY` to `tunnel_port + 1`.

The liveness probes (`start_proxy` poll loop, `get_status`) read the configured bind host via `_read_http_listen_from_config()` and probe THAT host — a hardcoded loopback probe would report a healthy bridge-bound daemon as dead.

Regression: `test_default_bind_is_loopback_not_zero_zero` (asserts no INADDR_ANY AND that `http_listens` is NOT in the rendered yaml), `test_default_bind_uses_docker_bridge_on_linux`, `test_default_bind_falls_back_to_loopback_without_bridge`, `test_default_bind_is_loopback_on_macos`, `test_detect_docker_bridge_ip_rejects_dangerous` (parametrized over 8 attack inputs).

### Metrics port collision

`metrics.listen` defaults to `:9090` in iron-proxy v0.39 — the SAME port as Hermes's default `tunnel_port: 9090`.  `build_proxy_config` MUST explicitly pin `metrics.listen: 127.0.0.1:0` so the metrics binding gets an ephemeral loopback port that can never collide with the proxy listener regardless of operator-chosen `tunnel_port`.

Regression: `test_metrics_listener_pinned_to_loopback_ephemeral`.

### Default deny CIDRs

`_DEFAULT_UPSTREAM_DENY_CIDRS` covers loopback (v4 + v6), link-local (incl. IMDS at 169.254.169.254 and the IPv4-mapped-v6 form), RFC1918, IPv6 ULA, CGNAT, and the RFC2544 benchmark range.  `build_proxy_config(..., upstream_deny_cidrs=None)` MUST emit the default; only an explicit empty list opts out.

Regression: `test_default_deny_cidrs_present_when_unspecified`, `test_default_deny_includes_ipv4_mapped_v6`.

### Audit log fail-loud

`ensure_audit_log` raises `RuntimeError` on any `OSError`.  On the pinned v0.39 the daemon never writes this file (no `log.audit_path` field), so `cmd_setup` treats the failure as a WARNING (the file is non-load-bearing until the version bump) and qualifies the success line as "reserved".  When the pin moves to a version with `log.audit_path`, revisit: the pre-create becomes load-bearing for the 0o600-from-first-byte guarantee and the wizard should fail loud again.

**v0.39 schema constraint:** `log.audit_path` is NOT a field in iron-proxy v0.39's `config.Log` struct, so `build_proxy_config` accepts the `audit_log` kwarg but does NOT emit it into the rendered yaml.  Per-request records on v0.39 land in `iron-proxy.log` alongside daemon-level events.  The `audit.log` file is still pre-created at `0o600` with `O_NOFOLLOW` so the privacy contract holds when the pinned version is bumped to one that supports the separate stream.

Regression: `test_ensure_audit_log_raises_on_immutable_parent`, `test_audit_log_kwarg_does_not_inject_audit_path_v039`.

### Bitwarden mode fail-loud

When `credential_source: bitwarden` AND `proxy.allow_env_fallback: false` (default):
- Missing access token env var -> `cmd_start` refuses.
- Missing `project_id` -> `cmd_start` refuses.
- `bws secret list` returns no values for one or more mapped providers -> `_build_proxy_subprocess_env` raises.

Falling back to host env in BW mode reintroduces exactly the staleness bug the BW path is meant to defeat.

Regression: `test_cmd_start_refuses_when_bitwarden_token_missing` (CLI layer); strict-mode assertions in `_build_proxy_subprocess_env` (daemon layer).

### docker_env collision detection

When `enforce_on_docker: true`, `docker_env` overrides on any of the egress-controlling vars (HTTPS_PROXY, SSL_CERT_FILE, NODE_EXTRA_CA_CERTS, etc.) OR any mapped `real_env_name` (OPENROUTER_API_KEY, etc.) raises `RuntimeError` BEFORE the container starts.

Regression: `test_docker_env_collision_with_proxy_raises_when_enforce`.

### PID recycling defense

`_pid_alive` MUST consult either the in-process `_proxy_nonce` (same-process case) OR the on-disk `iron-proxy.nonce` (cross-CLI case) before trusting an `argv[0]` basename match.  `stop_proxy` MUST re-check `/proc/<pid>/stat` starttime before SIGKILL and suppress the signal on starttime drift.

Regression: `test_stop_proxy_suppresses_sigkill_on_pid_recycle`, `test_pid_proc_starttime_parses_comm_with_parens`, `test_persisted_nonce_roundtrip`.

### Token preservation on re-setup

`merge_mappings(existing, discovered, rotate=False)` MUST return prior tokens for providers that overlap.  Re-running `hermes egress setup` cannot silently 401 running sandboxes.  `--rotate-tokens` is the explicit opt-in.

Regression: `test_merge_mappings_preserves_existing_tokens`, `test_merge_mappings_rotate_mints_fresh_tokens`.

### `credential_source` preservation

`cmd_setup` MUST NOT downgrade `credential_source: bitwarden` to `env` on re-run without an explicit `--no-bitwarden` flag.  Running `hermes egress setup` (no flag) preserves whatever was previously configured.

Tested via the `cmd_setup` flow in CLI tests (the bitwarden-preservation path is exercised when `--from-bitwarden` is followed by a plain `setup` re-run).

## Extension points

### Adding a new bearer-token provider

`_BEARER_PROVIDERS` in `iron_proxy.py` maps env var name -> tuple of upstream hosts.  Adding an entry makes it discoverable by `discover_provider_mappings()`; the wizard mints a token for it automatically when the env var is present.

```python
_BEARER_PROVIDERS: Dict[str, Tuple[str, ...]] = {
    ...,
    "MY_PROVIDER_API_KEY": ("api.myprovider.com",),
}
```

Also update `_DEFAULT_ALLOWED_HOSTS` so the proxy allows the upstream by default.  Run `test_discover_provider_mappings_*` to confirm.

### Adding a new header-token provider (x-api-key family)

If the provider authenticates with a static NON-Authorization header (like Anthropic's `x-api-key`, Azure's `api-key`, or Gemini's `x-goog-api-key`), add it to `_HEADER_AUTH_PROVIDERS` — iron-proxy's `secrets.replace.match_headers` targets arbitrary header names, so these are first-class swapped providers:

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

Use `aliases` ONLY for interchangeable env-var names of the *same* credential (e.g. `GOOGLE_API_KEY` for `GEMINI_API_KEY`) — aliased names collapse into a single mapping, because two `require: true` rules on the same host reject each other's requests. Also update `_DEFAULT_ALLOWED_HOSTS`.

### Adding a new signature-auth provider (uncovered)

If the provider uses SigV4 / SDK-minted OAuth / request signatures, a static header swap cannot cover it.  Add the env var to `_NON_BEARER_PROVIDERS` so the wizard and `hermes egress status` warn about it:

```python
_NON_BEARER_PROVIDERS: Tuple[str, ...] = (
    ...,
    "MY_SIGNED_PROVIDER_ACCESS_KEY",
)
```

### Wiring iron-proxy into a non-Docker backend

`_egress_proxy_args_for_docker` is Docker-specific.  Backends that want similar wiring need their own analogue that:

1. Reads `load_config().get("proxy", {})`; returns empty args if `enabled` is false.
2. Calls `iron_proxy.get_status()`; surfaces `enforce` semantics on `configured` / `pid` / `listening` / `ca_cert_path` failure paths.
3. Calls `iron_proxy.load_mappings()`; refuses to mount if empty AND `enforce_on_docker: true`.
4. Sets the seven env vars (HTTPS_PROXY, NO_PROXY, REQUESTS_CA_BUNDLE, SSL_CERT_FILE, CURL_CA_BUNDLE, NODE_EXTRA_CA_CERTS, HERMES_EGRESS_PROXY) and the per-mapping `HERMES_PROXY_TOKEN_<NAME>` vars.
5. Distributes the CA cert into the sandbox at a path the runtime will trust (typically `/etc/ssl/certs/hermes-egress-ca.crt`).
6. Implements collision detection against the user's backend-specific env config.

The Docker implementation is ~150 lines; expect similar volume for Modal / Daytona / SSH.

### Subscribing to per-request audit events

iron-proxy writes line-delimited JSON to `~/.hermes/proxy/iron-proxy.log` on the currently pinned v0.39 (daemon + per-request records combined; see "Logging on iron-proxy v0.39" in the user guide).  A plugin / external watcher can tail that file and react to allowlist denials, secret swaps, or upstream errors.  When the pinned version is bumped to one that supports `log.audit_path`, the per-request stream moves to `audit.log` and watchers wired to that path go live without operator action.  The schema is documented at [docs.iron.sh/audit](https://docs.iron.sh/audit) (link).

## Testing

```bash
# Hermetic suite (no network, no real binary)
scripts/run_tests.sh tests/test_iron_proxy.py tests/test_iron_proxy_cli.py

# Live E2E (real binary, real curl, real CONNECT tunnel)
HERMES_RUN_E2E=1 scripts/run_tests.sh tests/test_iron_proxy_e2e.py

# Live PTY smoke against `hermes egress`
HERMES_HOME=/tmp/hermes-egress-test python3 -m hermes_cli.main egress --help
HERMES_HOME=/tmp/hermes-egress-test python3 -m hermes_cli.main egress setup --help
```

The CLI uses argparse, so `--help` is a good first probe for "did my new flag register correctly".

## See also

- User-facing setup + troubleshooting: [Egress proxy](https://hermes-agent.nousresearch.com/docs/user-guide/egress/iron-proxy)
- Docker backend internals: [Docker](https://hermes-agent.nousresearch.com/docs/user-guide/docker)
- Bitwarden Secrets Manager integration: [`hermes secrets bitwarden`](https://hermes-agent.nousresearch.com/docs/user-guide/secrets/bitwarden)
- CLI command reference: [`hermes egress`](https://hermes-agent.nousresearch.com/docs/reference/cli-commands#hermes-egress)
- Sandbox-injected environment variables: [Egress proxy (sandbox-injected)](https://hermes-agent.nousresearch.com/docs/reference/environment-variables#egress-proxy-sandbox-injected)
