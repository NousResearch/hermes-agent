---
title: "Hermes S6 컨테이너 감독"
sidebar_label: "Hermes S6 컨테이너 감독"
description: "Hermes Agent Docker 이미지 내부의 s6-overlay 감독 트리를 수정, 디버깅 또는 확장하세요 — 새로운 서비스 추가, 프로필 게이트웨이 디버깅, 아키텍처 B 메인 프로그램 패턴 이해 등."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes S6 컨테이너 감독

Hermes Agent Docker 이미지 내부의 s6-overlay 감독(supervision) 트리를 수정, 디버그 또는 확장합니다 — 새로운 서비스 추가, 프로필 게이트웨이 디버깅, 아키텍처 B(Architecture B) 메인 프로그램 패턴 이해 등을 포함합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/devops/hermes-s6-container-supervision`으로 설치 |
| 경로 | `optional-skills/devops/hermes-s6-container-supervision` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux |
| 태그 | `docker`, `s6`, `supervision`, `gateway`, `profiles` |
| 관련 스킬 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), `hermes-agent-dev` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Hermes s6-overlay 컨테이너 감독

## 언제 이 스킬을 사용해야 하나요?

다음과 같은 작업을 할 때 이 스킬을 로드하세요:
- Hermes Docker 이미지에 정적 서비스(대시보드와 같이 컨테이너가 시작될 때마다 감독되어야 하는 항목)를 추가하거나 제거할 때
- 프로필별 게이트웨이가 시작, 재시작 또는 `docker restart` 후 유지되지 않는 이유를 진단할 때
- 컨테이너의 CMD가 `/opt/hermes/docker/main-wrapper.sh`인 이유와 선행 대시(leading-dash) 인수가 사용자의 프로그램에 어떻게 전달되는지 이해할 때
- `cont-init.d` 부팅 스크립트 수정 시 (UID 리맵, 볼륨 시딩, 프로필 조정)
- 프로필별 게이트웨이(Phase 4)를 위한 렌더링된 실행 스크립트를 변경할 때

단순히 Hermes Agent를 실행하고 Docker를 사용하려는 경우에는 `website/docs/user-guide/docker.md`를 대신 참고하세요.

## 아키텍처 한눈에 보기

<!-- ascii-guard-ignore -->
```
/init                                  ← PID 1 (s6-overlay v3.2.3.0)
├── cont-init.d                        ← 원샷 설정, root로 실행
│   ├── 01-hermes-setup                ← docker/stage2-hook.sh
│   │   ├── UID/GID 리맵
│   │   ├── chown /opt/data
│   │   ├── chown /opt/data/profiles (매 부팅마다)
│   │   ├── seed .env / config.yaml / SOUL.md
│   │   └── skills_sync.py
│   └── 02-reconcile-profiles          ← hermes_cli.container_boot
│       ├── chown /run/service (런타임 등록을 위해 hermes 쓰기 권한 부여)
│       └── $HERMES_HOME/profiles/<name>/gateway_state.json 탐색
│           → /run/service/gateway-<name>/ 재생성
│           → prior_state == "running" 인 것만 자동 시작
│
├── s6-rc.d (정적 서비스, /etc/s6-overlay/s6-rc.d/ 내에 위치)
│   ├── main-hermes/run                ← exec sleep infinity (아무 동작도 하지 않는 슬롯)
│   └── dashboard/run                  ← HERMES_DASHBOARD=1 인 경우, `hermes dashboard` 실행
│
├── /run/service (s6-svscan이 감시함; tmpfs)
│   ├── gateway-coder/                 ← 런타임에 등록된 프로필별
│   │   ├── type        ("longrun")
│   │   ├── run         ("#!/command/with-contenv sh ... exec s6-setuidgid hermes hermes -p coder gateway run")
│   │   ├── down        (마커 — 존재 시 "등록되었으나 자동 시작은 하지 않음"을 의미)
│   │   └── log/run     (s6-log → $HERMES_HOME/logs/gateways/coder/current)
│   └── ...
│
└── CMD ("메인 프로그램")               ← /opt/hermes/docker/main-wrapper.sh
    └── 사용자 인수를 라우팅함: bare exec | hermes 하위 명령어 | hermes (인수 없음)
        — stdin/stdout/stderr를 상속받은 /init에 의해 실행됨 (--tui의 경우 TTY 지원)
```
<!-- ascii-guard-ignore-end -->

## 주요 파일

| 경로 | 역할 |
|---|---|
| `Dockerfile` | s6-overlay 설치 + cont-init.d 연결 + `ENTRYPOINT ["/init", "/opt/hermes/docker/main-wrapper.sh"]` |
| `docker/stage2-hook.sh` | "이전 진입점(entrypoint) 로직" — UID 리맵, chown, seed, 스킬 동기화. cont-init.d/01-hermes-setup으로 실행됨. |
| `docker/cont-init.d/02-reconcile-profiles` | 영구 볼륨에서 프로필 게이트웨이 슬롯을 복원하기 위해 매 부팅 시 `hermes_cli.container_boot`를 호출. |
| `docker/main-wrapper.sh` | 컨테이너의 CMD. 사용자 인수를 라우팅하고, `s6-setuidgid`를 통해 hermes 권한으로 전환한 다음, 선택한 프로그램을 실행(exec). |
| `docker/s6-rc.d/main-hermes/run` | 아무 동작도 하지 않는(No-op) `sleep infinity` — s6-rc 사용자 번들이 유효하도록 슬롯이 존재함; 메인 hermes는 감독되는 서비스가 아니라 CMD로 실행됨. |
| `docker/s6-rc.d/dashboard/run` | 조건부 서비스 — `HERMES_DASHBOARD`가 참값(truthy)이 아니면 `exec sleep infinity`. |
| `docker/entrypoint.sh` | stage2 훅을 `exec`하는 하위 호환용 래퍼(shim). 이전 엔트리포인트 경로를 하드 코딩한 외부 스크립트도 계속 동작하게 함. |
| `hermes_cli/service_manager.py` | `S6ServiceManager`: `register_profile_gateway`, `unregister_profile_gateway`, `start/stop/restart/is_running`, `list_profile_gateways`. |
| `hermes_cli/container_boot.py` | `reconcile_profile_gateways()` — 영구 프로필을 순회하며 s6 슬롯을 재생성하고 `container-boot.log`를 출력. |
| `hermes_cli/gateway.py::_dispatch_via_service_manager_if_s6` | 컨테이너 내에서 실행될 때 `hermes gateway start/stop/restart`를 가로채서 s6로 라우팅. |

## 왜 아키텍처 B인가 (s6의 감독 없이 CMD를 메인 프로그램으로)

원래 계획(v1–v3)에서는 메인 hermes를 s6-rc의 감독을 받는 서비스로 실행할 예정이었습니다. 하지만 실제 s6-overlay v3의 두 가지 메커니즘으로 인해 차질이 생겼습니다:

1. **cont-init.d 스크립트는 CMD 인수를 받지 못함** — 따라서 stage2 훅이 `docker run <image> chat -q "hi"`를 구문 분석하여 서비스의 `run` 스크립트가 사용할 `HERMES_ARGS`를 설정할 수 없습니다.
2. **`/run/s6/basedir/bin/halt`는 종료 코드(exit code)를 전달하지 않음** — `/run/s6-linux-init-container-results/exitcode`에 기록된 종료 코드가 무시됩니다. 컨테이너는 항상 143(SIGTERM) 상태로 종료됩니다. 이는 [이슈 #477](https://github.com/just-containers/s6-overlay/issues/477)에서 s6 작성자(skarnet)에 의해 확인되었습니다: _"컨테이너 종료 시 원하는 코드를 얻으려면 CMD가 종료되도록 하거나, CMD가 없는 경우 원하는 컨테이너 종료 코드를 직접 쓴 후 halt를 호출해야 합니다"_.

그래서 s6-overlay 고유의 CMD 패턴을 사용합니다: `ENTRYPOINT ["/init", "/opt/hermes/docker/main-wrapper.sh"]`. /init은 사용자 인수의 맨 앞에 자동으로 래퍼를 붙입니다 — 즉, `docker run <image> --version`은 `/init main-wrapper.sh --version`이 되고, `--version`은 /init의 POSIX 셸에 의해 가로채지지 않습니다. 래퍼는 `s6-setuidgid`를 통해 hermes 사용자로 전환한 다음, 선택한 프로그램을 실행(exec)합니다. 이때 해당 프로그램의 종료 코드가 컨테이너의 종료 코드가 되어 s6 이전의 tini 시절과 완벽히 동일하게 동작합니다.

트레이드오프: 메인 hermes는 s6 하에서 감독받지 않습니다. 이는 tini를 사용할 때(s6 도입 이전 이미지)의 동작과 정확히 일치합니다. 대시보드 감독만이 유일한 **새로운** 보장 사항이며 — `/run/service/` 아래의 프로필별 게이트웨이들은 완전한 감독(supervision)을 받습니다.

## 빠른 레시피

### 실행 중인 컨테이너에서 s6가 PID 1인지 확인하기

```sh
docker exec <c> sh -c 'cat /proc/1/comm; readlink /proc/1/exe'
# 예상 결과: s6-svscan 또는 init / /package/admin/s6/.../s6-svscan
```

### 프로필 게이트웨이 서비스 검사하기

```sh
# /command/는 docker-exec PATH에 없으므로 절대 경로를 사용
docker exec <c> /command/s6-svstat /run/service/gateway-<name>
# "up (pid …) … seconds"            → 실행 중
# "down (exitcode N) … seconds, normally up, want up, …" → s6는 실행 상태(up)를 원하지만 프로세스가 계속 종료됨 (크래시 루프)
# "down … normally up, ready …"     → 사용자가 중지함
```

### 서비스를 수동으로 켜거나 끄기

```sh
docker exec <c> /command/s6-svc -u /run/service/gateway-<name>   # 시작 (up)
docker exec <c> /command/s6-svc -d /run/service/gateway-<name>   # 중지 (down)
docker exec <c> /command/s6-svc -t /run/service/gateway-<name>   # SIGTERM (재시작)
```

### cont-init 리컨사일러 로그 보기

```sh
docker exec <c> tail -n 50 /opt/data/logs/container-boot.log
# 2026-05-21T06:18:05+0000 profile=coder prior_state=running action=started
# 2026-05-21T06:18:05+0000 profile=writer prior_state=stopped action=registered
```

### 새로운 정적 서비스 추가하기

1. `longrun\n` 이 포함된 `docker/s6-rc.d/<name>/type`과 `docker/s6-rc.d/<name>/run` 을 만듭니다. (`#!/command/with-contenv sh` + `# shellcheck shell=sh` 를 사용하세요.)
2. run 스크립트 상단에서 특별히 root 권한이 필요한 경우가 아니라면 `s6-setuidgid hermes`를 통해 hermes 권한으로 강등합니다.
3. 기본 번들을 대기하도록 비어 있는 `docker/s6-rc.d/<name>/dependencies.d/base` 파일을 만듭니다.
4. 사용자 번들에 추가되도록 비어 있는 `docker/s6-rc.d/user/contents.d/<name>` 파일을 만듭니다.
5. Dockerfile의 `COPY docker/s6-rc.d/` 명령을 통해 자동으로 처리되므로 다른 변경 사항은 필요 없습니다.

### 프로필별 게이트웨이 실행 명령 변경하기

`hermes_cli/service_manager.py`에 있는 `S6ServiceManager._render_run_script`를 편집하세요. 부팅 복원 중 `hermes_cli/container_boot.py::_register_service`에 의해서도 이 함수가 호출되므로, 단일 진실 공급원(Single Source of Truth)입니다. 변경 시 `tests/hermes_cli/test_service_manager.py::test_s6_register_creates_service_dir_and_triggers_scan` 에 있는 해당 어설션(assertion)도 같이 업데이트해야 합니다.

### docker 테스트 하네스 실행하기

```sh
docker build -t hermes-agent-harness:latest .
HERMES_TEST_IMAGE=hermes-agent-harness:latest scripts/run_tests.sh tests/docker/ -v
# 예상 결과: s6 이미지에 대해 19 passed, 0 xfailed
```

테스트 하네스는 `tests/docker/` 디렉터리에 있으며, Docker를 사용할 수 없을 때는 테스트를 건너뜁니다. 테스트별 타임아웃은 180초로 늘려져 있습니다(`tests/docker/conftest.py` 참조).

## 일반적인 함정 (Common Pitfalls)

### `docker exec` 실행 시 "command not found"

`/command/` (s6-overlay가 자신의 바이너리를 두는 곳)는 감독 트리(서비스, cont-init.d, main-wrapper.sh)에 의해 생성된 프로세스의 PATH에만 포함됩니다. `docker exec <c> s6-svstat …`는 "command not found"로 실패하므로 항상 절대 경로인 `/command/s6-svstat`를 사용해야 합니다. `hermes` 바이너리의 경우는 Dockerfile에서 런타임 `ENV PATH`에 `/opt/hermes/.venv/bin`을 추가하기 때문에 작동합니다.

### 프로필 디렉터리 소유권

cont-init 리컨사일러는 hermes 사용자 권한으로 동작합니다 (`02-reconcile-profiles`에서 `s6-setuidgid hermes` 실행). 만약 프로필 디렉터리의 소유권이 root로 되어 있다면 (예: `docker exec <c> hermes profile create …`가 기본적으로 root로 실행되어), 리컨사일러가 SOUL.md를 읽지 못해 `PermissionError`가 발생합니다. 완화 방법: `stage2-hook.sh`가 **매번** 부팅할 때마다 `$HERMES_HOME/profiles`의 소유권을 멱등성 있게 hermes로 변경합니다(chown). 이 블록을 지우지 마세요.

### `docker exec`으로 작성된 파일은 root 소유

`docker exec`의 기본 사용자는 root입니다. `--user hermes`를 전달하거나 다음 재부팅 시에 수행될 stage2의 일괄 chown에 맡겨야 합니다. `$HERMES_HOME/profiles/<name>/` 아래의 파일들을 수동으로 root 권한으로 기록하지 마세요 — 다음 리컨사일 패스에서 수정되기는 하지만, 실행 중인 작업에서 권한 오류가 발생할 수 있습니다.

### 서비스 슬롯은 존재하지만 s6-svstat이 "s6-supervise not running"이라고 말할 때

서비스 디렉터리는 tmpfs(메모리 기반 파일 시스템) 상에 있으며 컨테이너 재시작 시 초기화됩니다. 리컨사일러(cont-init)가 아직 실행되지 않았거나(`docker restart` 후 잠시 기다려보세요), 실패했을 가능성이 있습니다. `docker logs <c> | grep '02-reconcile'`로 확인하세요.

### 게이트웨이가 시작된 후 즉시 종료될 때 (svstat에서 `down (exitcode 1)`)

가장 큰 이유는 프로필에 모델이나 인증 정보가 구성되지 않은 경우입니다. 서비스 슬롯은 올바르지만 게이트웨이 자체의 설정이 누락된 것입니다. 먼저 `hermes -p <profile> setup`을 실행하세요. s6 감독자는 계속해서 해당 서비스를 재시작할 텐데, 이는 올바른 동작입니다 (설정을 수정하면 다음 시도에 성공하여 유지됩니다).

### 리컨사일러가 프로필을 건너뛰었을 때

리컨사일러는 "진짜 프로필" 마커로서 **`SOUL.md`의 존재 여부**를 확인합니다. `hermes profile create`는 항상 이 파일을 생성(seed)합니다. 프로필 디렉터리에 SOUL.md가 없다면 (잘못 생성된 디렉터리, 부분 복원, 백업 중 등), 리컨사일러는 고의로 이를 건너뜁니다. 다시 포함되기를 원한다면 비어 있더라도 `SOUL.md`를 추가하세요.

### "도와주세요, 컨테이너가 143 코드로 종료됩니다!"

누군가가 `s6-svscanctl -t`나 `/run/s6/basedir/bin/halt`를 호출하고 있는지 확인하세요 — 둘 다 /init이 단계 3 종료를 시작하게 하지만 원하는 종료 코드가 아닌 143(SIGTERM)을 반환하게 합니다. 이것이 Phase 2 아키텍처가 A에서 B로 전환된 핵심 이유입니다. 진짜 종료 코드를 반환하며 컨테이너가 종료되게 하려면 CMD(main-wrapper.sh)가 정상적으로 종료되게 내버려 두어야 합니다. 절대 finish 스크립트에서 종료를 통제하려 하지 마세요.

## 관련 스킬

- `hermes-agent-dev`: 전반적인 hermes-agent 코드베이스 탐색
- `hermes-tool-quirks`: 특정 Hermes 도구 관련 해결 방법(sed/grep/등) — s6 스택과 hermes 내장 도구 간의 상호작용을 디버깅할 때 로드하세요.
