---
title: "Hermes S6 컨테이너 감독 — Hermes Docker 이미지에서 s6 서비스 수정 또는 디버깅"
sidebar_label: "Hermes S6 컨테이너 감독"
description: "Hermes Docker 이미지에서 s6 서비스 수정 또는 디버깅"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes S6 컨테이너 감독

Hermes Docker 이미지에서 s6 서비스를 수정하거나 디버깅합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Optional — `hermes skills install official/devops/hermes-s6-container-supervision`으로 설치 |
| 경로 | `optional-skills/devops/hermes-s6-container-supervision` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux |
| 태그 | `docker`, `s6`, `supervision`, `gateway`, `profiles` |
| 관련 스킬 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 활성화될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 지침으로 보는 내용입니다.
:::

# Hermes s6-overlay 컨테이너 감독

## 이 스킬을 사용하는 경우

다음 작업을 수행할 때 이 스킬을 로드하세요:
- Hermes Docker 이미지에 정적 서비스를 추가하거나 제거할 때(대시보드처럼 모든 컨테이너 시작 시 감독되어야 하는 서비스)
- 프로필별 게이트웨이가 시작되지 않거나, 재시작되지 않거나, `docker restart` 후에도 유지되지 않는 이유를 진단할 때
- 컨테이너의 CMD가 `/opt/hermes/docker/main-wrapper.sh`인 이유와 앞에 대시가 붙은 인수가 사용자의 프로그램에 전달되는 방식을 이해할 때
- `cont-init.d` 부트 스크립트(UID 재매핑, 볼륨 시드, 프로필 조정)를 수정할 때
- 프로필별 게이트웨이의 렌더링된 실행 스크립트를 변경할 때(4단계)

Hermes Agent를 실행하면서 Docker를 사용하려는 것뿐이라면 `website/docs/user-guide/docker.md`를 대신 참고하세요.

## 아키텍처 한눈에 보기

<!-- ascii-guard-ignore -->
```
/init                                  ← PID 1 (s6-overlay v3.2.3.0)
├── cont-init.d                        ← oneshot setup, runs as root
│   ├── 01-hermes-setup                ← docker/stage2-hook.sh
│   │   ├── UID/GID remap
│   │   ├── chown /opt/data
│   │   ├── chown /opt/data/profiles (every boot)
│   │   ├── seed .env / config.yaml / SOUL.md
│   │   └── skills_sync.py
│   └── 02-reconcile-profiles          ← hermes_cli.container_boot
│       ├── chown /run/service (hermes-writable for runtime register)
│       └── walk $HERMES_HOME/profiles/<name>/gateway_state.json
│           → recreate /run/service/gateway-<name>/
│           → auto-start only those with prior_state == "running"
│
├── s6-rc.d (static services, in /etc/s6-overlay/s6-rc.d/)
│   ├── main-hermes/run                ← exec sleep infinity (no-op slot)
│   └── dashboard/run                  ← if HERMES_DASHBOARD=1, runs `hermes dashboard`
│
├── /run/service (s6-svscan watches; tmpfs)
│   ├── gateway-coder/                 ← runtime-registered per-profile
│   │   ├── type        ("longrun")
│   │   ├── run         ("#!/command/with-contenv sh ... exec s6-setuidgid hermes hermes -p coder gateway run")
│   │   ├── down        (marker — present means "registered but don't auto-start")
│   │   └── log/run     (s6-log → $HERMES_HOME/logs/gateways/coder/current)
│   └── ...
│
└── CMD ("main program")               ← /opt/hermes/docker/main-wrapper.sh
    └── routes user args: bare exec | hermes subcommand | hermes (no args)
        — exec'd by /init with stdin/stdout/stderr inherited (TTY for --tui)
```
<!-- ascii-guard-ignore-end -->

## 주요 파일

| 경로 | 역할 |
|---|---|
| `Dockerfile` | s6-overlay 설치 + cont-init.d 연결 + `ENTRYPOINT ["/init", "/opt/hermes/docker/main-wrapper.sh"]` |
| `docker/stage2-hook.sh` | "기존 엔트리포인트 로직" — UID 재매핑, chown, 시드, 스킬 동기화. cont-init.d/01-hermes-setup으로 실행됩니다. |
| `docker/cont-init.d/02-reconcile-profiles` | 매 부트마다 `hermes_cli.container_boot`을 호출하여 영구 볼륨에서 프로필 게이트웨이 슬롯을 복원합니다. |
| `docker/main-wrapper.sh` | 컨테이너의 CMD. 사용자 인수를 라우팅하고 `s6-setuidgid`를 통해 hermes로 전환한 뒤 선택한 프로그램을 exec합니다. |
| `docker/s6-rc.d/main-hermes/run` | 동작이 없는 `sleep infinity` — s6-rc 사용자 번들이 유효하도록 슬롯이 존재합니다. 메인 hermes는 감독 서비스가 아니라 CMD로 실행됩니다. |
| `docker/s6-rc.d/dashboard/run` | 조건부 서비스 — `HERMES_DASHBOARD`가 참이 아니면 `exec sleep infinity`를 실행합니다. |
| `docker/entrypoint.sh` | 스테이지 2 훅을 `exec`하는 하위 호환성 심. 기존 엔트리포인트 경로를 하드코딩한 외부 스크립트도 계속 작동합니다. |
| `hermes_cli/service_manager.py` | `S6ServiceManager`: `register_profile_gateway`, `unregister_profile_gateway`, `start/stop/restart/is_running`, `list_profile_gateways`. |
| `hermes_cli/container_boot.py` | `reconcile_profile_gateways()` — 영구 프로필을 순회하고 s6 슬롯을 재생성하며 `container-boot.log`를 기록합니다. |
| `hermes_cli/gateway.py::_dispatch_via_service_manager_if_s6` | 컨테이너에서 실행 중일 때 `hermes gateway start/stop/restart`를 가로채 s6로 라우팅합니다. |

## 아키텍처 B를 사용하는 이유(CMD를 메인 프로그램으로 사용하며 s6가 감독하지 않음)

원래 계획(v1–v3)은 메인 hermes를 감독되는 s6-rc 서비스로 실행하는 것이었습니다. 하지만 실제 s6-overlay v3 메커니즘 두 가지가 이를 막았습니다.

1. **cont-init.d 스크립트는 CMD 인수를 받지 않습니다** — 따라서 스테이지 2 훅은 `docker run <image> chat -q "hi"`를 분석하여 서비스 `run` 스크립트가 사용할 `HERMES_ARGS`를 설정할 수 없습니다.
2. **`/run/s6/basedir/bin/halt`는 `/run/s6-linux-init-container-results/exitcode`에 기록된 종료 코드를 전파하지 않습니다.** 컨테이너는 항상 143(SIGTERM)으로 종료됩니다. s6 작성자인 skarnet도 [이슈 #477](https://github.com/just-containers/s6-overlay/issues/477)에서 이를 확인했습니다. _"컨테이너를 종료하려면 CMD를 종료시키거나, CMD가 없다면 원하는 컨테이너 종료 코드를 기록한 다음 halt를 호출해야 합니다."_

따라서 s6-overlay 네이티브 CMD 패턴인 `ENTRYPOINT ["/init", "/opt/hermes/docker/main-wrapper.sh"]`을 사용합니다. /init은 사용자 인수 앞에 래퍼를 자동으로 붙입니다. 즉 `docker run <image> --version`은 `/init main-wrapper.sh --version`이 되며, `--version`은 /init의 POSIX 셸에 의해 가로채지지 않습니다. 래퍼는 `s6-setuidgid`를 통해 hermes로 전환한 다음 선택한 프로그램을 exec합니다. 프로그램의 종료 코드가 컨테이너 종료 코드가 되므로, s6 이전 tini 계약과 정확히 일치합니다.

절충점: 메인 hermes는 s6 아래에서 감독되지 않습니다. 이는 s6 이전 이미지에서 tini로 실행될 때의 동작과 정확히 일치합니다. 대시보드 감독만이 **새로** 보장되는 기능이며, `/run/service/` 아래의 프로필별 게이트웨이는 완전한 감독을 받습니다.

## 빠른 레시피

### 실행 중인 컨테이너에서 s6가 PID 1인지 확인

```sh
docker exec <c> sh -c 'cat /proc/1/comm; readlink /proc/1/exe'
# Expect: s6-svscan or init / /package/admin/s6/.../s6-svscan
```

### 프로필 게이트웨이 서비스 검사

```sh
# /command/ isn't on docker-exec PATH — use absolute path
docker exec <c> /command/s6-svstat /run/service/gateway-<name>
# "up (pid …) … seconds"            → running
# "down (exitcode N) … seconds, normally up, want up, …" → s6 wants it up but the process keeps exiting (crash loop)
# "down … normally up, ready …"     → user stopped it
```

### 서비스를 수동으로 up/down

```sh
docker exec <c> /command/s6-svc -u /run/service/gateway-<name>   # up
docker exec <c> /command/s6-svc -d /run/service/gateway-<name>   # down
docker exec <c> /command/s6-svc -t /run/service/gateway-<name>   # SIGTERM (restart)
```

### cont-init 조정자 로그 확인

```sh
docker exec <c> tail -n 50 /opt/data/logs/container-boot.log
# 2026-05-21T06:18:05+0000 profile=coder prior_state=running action=started
# 2026-05-21T06:18:05+0000 profile=writer prior_state=stopped action=registered
```

### 새 정적 서비스 추가

1. `longrun\n`을 포함하는 `docker/s6-rc.d/<name>/type`과 `docker/s6-rc.d/<name>/run`을 생성합니다(`#!/command/with-contenv sh` + `# shellcheck shell=sh` 사용).
2. run의 맨 위에서 `s6-setuidgid hermes`를 통해 hermes로 전환합니다(특별히 root가 필요한 경우는 제외).
3. 기본 번들이 끝날 때까지 대기하도록 빈 `docker/s6-rc.d/<name>/dependencies.d/base`를 생성합니다.
4. 사용자 번들에 참여하도록 빈 `docker/s6-rc.d/user/contents.d/<name>`을 생성합니다.
5. Dockerfile의 `COPY docker/s6-rc.d/`가 자동으로 이를 포함하므로 다른 변경은 필요하지 않습니다.

### 프로필별 게이트웨이 실행 명령 변경

`hermes_cli/service_manager.py`의 `S6ServiceManager._render_run_script`를 편집합니다. 부트 조정 중 `hermes_cli/container_boot.py::_register_service`도 이 함수를 호출하므로, 이 함수가 단일 진실 공급원입니다. `tests/hermes_cli/test_service_manager.py::test_s6_register_creates_service_dir_and_triggers_scan`의 해당 어설션도 업데이트합니다.

### Docker 테스트 하네스 실행

```sh
docker build -t hermes-agent-harness:latest .
HERMES_TEST_IMAGE=hermes-agent-harness:latest scripts/run_tests.sh tests/docker/ -v
# Expect 19 passed, 0 xfailed against the s6 image
```

하네스는 `tests/docker/`에 있으며 Docker를 사용할 수 없으면 건너뜁니다. 테스트별 제한 시간은 180초로 늘어나 있습니다(`tests/docker/conftest.py` 참조).

## 일반적인 함정

### `docker exec`에서 "command not found"

`/command/`(s6-overlay가 바이너리를 배치하는 위치)는 감독 트리에서 생성한 프로세스(서비스, cont-init.d, main-wrapper.sh)에 대해서만 PATH에 포함됩니다. `docker exec <c> s6-svstat …`는 "command not found"로 실패하므로 항상 절대 경로 `/command/s6-svstat`을 사용하세요. Dockerfile이 `/opt/hermes/.venv/bin`을 런타임 `ENV PATH`에 추가하므로 `hermes` 바이너리는 작동합니다.

### 프로필 디렉터리 소유권

cont-init 조정자는 hermes로 실행됩니다(`02-reconcile-profiles`의 `s6-setuidgid hermes`). 프로필 디렉터리가 root 소유가 되면(예: `docker exec <c> hermes profile create …`가 기본적으로 root로 실행된 경우) 조정자가 SOUL.md를 읽지 못하고 `PermissionError`로 실패합니다. 완화 방법: `stage2-hook.sh`는 **매** 부트마다 멱등적으로 `$HERMES_HOME/profiles`를 hermes 소유로 chown합니다. 해당 블록을 제거하지 마세요.

### `docker exec`로 작성한 파일은 root 소유가 됨

`docker exec`는 기본적으로 root로 실행됩니다. `--user hermes`를 전달하거나 다음 재부팅 때 스테이지 2 chown 순회에 맡기세요. `$HERMES_HOME/profiles/<name>/` 아래에 root로 수동 파일을 작성하지 마세요. 다음 조정 단계에서 정리되기는 하지만 진행 중인 작업에서 권한 오류가 발생할 수 있습니다.

### 서비스 슬롯은 존재하지만 s6-svstat이 "s6-supervise not running"이라고 표시함

서비스 디렉터리는 tmpfs에 있으며 컨테이너 재시작 시 지워집니다. cont-init 조정자가 아직 실행되지 않았거나(`docker restart` 후 잠시 기다리세요) 조정자가 실패했을 수 있습니다. `docker logs <c> | grep '02-reconcile'`를 확인하세요.

### 게이트웨이가 시작되자마자 종료됨(svstat에서 `down (exitcode 1)`)

프로필에 모델 또는 인증 설정이 없을 가능성이 가장 큽니다. 서비스 슬롯은 올바르며 게이트웨이 자체가 구성되지 않은 것입니다. 먼저 `hermes -p <profile> setup`을 실행하세요. s6 감독자는 계속 재시작하는데, 이것이 의도된 동작입니다(구성을 수정하면 다음 시도가 성공하고 계속 실행됩니다).

### 조정자가 프로필을 건너뜀

조정자는 **SOUL.md의 존재**를 "실제 프로필" 표시로 사용합니다. `hermes profile create`는 항상 이를 시드합니다. 프로필 디렉터리에 SOUL.md가 없으면(불필요한 디렉터리, 불완전한 복원, 백업 진행 중) 조정자는 의도적으로 건너뜁니다. 다시 참여시키려면 `SOUL.md`를 추가하세요(비어 있어도 됩니다).

### "도와주세요, 컨테이너가 143으로 종료됩니다!"

무언가 `s6-svscanctl -t` 또는 `/run/s6/basedir/bin/halt`를 호출하는지 확인하세요. 둘 다 /init이 3단계 종료를 시작하게 하지만 원하는 종료 코드가 아닌 143(SIGTERM)을 반환합니다. 이는 A에서 B로 전환한 2단계 아키텍처 변경의 결과입니다. 실제 종료 코드로 컨테이너를 종료하려면 CMD(main-wrapper.sh)가 정상적으로 종료되도록 해야 하며, finish 스크립트에서 종료를 제어하려고 **하지 마세요**.

## 관련 스킬

- `hermes-agent-dev`: 일반적인 hermes-agent 코드베이스 탐색
- `hermes-tool-quirks`: 특정 Hermes 도구 우회 방법(sed/grep 등) — s6 스택과 Hermes 내장 도구의 상호작용을 디버깅할 때 로드
