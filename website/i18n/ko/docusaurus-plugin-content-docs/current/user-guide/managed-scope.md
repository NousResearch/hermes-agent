---
sidebar_position: 3
title: "관리 범위"
description: "시스템 수준의 관리 디렉터리를 통한 관리자 고정 및 사용자 불변 구성과 시크릿"
---

# 관리 범위

**관리 범위**를 사용하면 관리자가 표준(비루트) 사용자가 **재정의할 수 없는** 구성 및 시크릿의 기준선을 배포할 수 있습니다. IT에서 시스템의 모든 사용자에 대해 모델 공급자, 공유 API 기본 URL 또는 `security.redact_secrets: true` 등을 고정해야 하는 플릿/조직 배포를 위한 기능입니다.

관리 범위가 있으면 관리 범위가 지정한 정확한 키에 대해 해당 값이 사용자의 `~/.hermes/config.yaml`, `~/.hermes/.env`, 심지어 셸 환경보다도 우선합니다. 그 외의 모든 항목은 완전히 사용자 제어로 유지됩니다.

:::note 패키지 관리자 잠금 설치와 다름
패키지 관리자가 관리하는 설치(declarative-distro / formula)는 모든 구성 변경을 차단하고 패키지 관리자를 사용하도록 안내합니다. 관리 범위는 별도의 메커니즘입니다. 전체 구성을 잠그는 대신 키별로 *특정 불변 값*을 주입합니다. 두 메커니즘은 서로 독립적이며 함께 사용할 수 있습니다.
:::

## 위치

관리 범위는 기본적으로 `/etc/hermes`인 시스템 수준 디렉터리에서 읽습니다.

```text
/etc/hermes/
├── config.yaml     # managed config layer (wins over ~/.hermes/config.yaml)
└── .env            # managed env layer (wins over ~/.hermes/.env + shell)
```

디렉터리와 파일의 소유자는 `root`입니다(디렉터리 모드 `0755`, 파일 `0644`). 모두 읽을 수 있지만 관리자만 쓸 수 있습니다. **이 파일 시스템 권한이 적용 메커니즘**이므로 표준 사용자는 관리 파일을 읽을 수는 있어도 편집할 수 없습니다.

두 파일 모두 선택 사항입니다. 관리 디렉터리나 파일이 없으면 단순히 "관리 범위 없음"을 의미하며, 기능이 없는 경우와 정확히 동일하게 구성이 결정됩니다.

### 디렉터리 위치 변경

`HERMES_MANAGED_DIR` 환경 변수로 위치를 변경할 수 있습니다(컨테이너 또는 `/etc`가 아닌 배포 환경용). 이는 `HERMES_HOME`과 같은 배포/부트스트랩 경로 설정이며, 관리 파일을 소유한 동일한 관리자가 설정합니다. Hermes는 이 값을 어떤 `.env`에도 **절대 저장하지 않습니다**.

```bash
# Point managed scope at a custom directory (set by IT / the deployment, not the user)
export HERMES_MANAGED_DIR=/opt/org/hermes-policy
```

:::warning
`HERMES_MANAGED_DIR`를 설정할 수 있는 사용자는 자신이 제어하는 디렉터리로 관리 범위를 바꿔 이를 무력화할 수 있습니다. 실제 배포에서는 이 변수를 관리자가 고정해야 하며(예: 서비스 유닛/컨테이너 이미지에 내장), 사용자가 설정할 수 있도록 두어서는 안 됩니다. `hermes doctor`는 *확정된* 관리 디렉터리를 보고하므로 리디렉션을 확인할 수 있습니다.
:::

## 우선순위

관리 계층이 지정한 키에 대해 순서는 다음과 같습니다(높은 순서가 우선).

| 계층 | config.yaml | .env |
|---|---|---|
| 1 | `/etc/hermes/config.yaml` (관리) | `/etc/hermes/.env` (관리) |
| 2 | `~/.hermes/config.yaml` (사용자) | `~/.hermes/.env` (사용자) |
| 3 | 기본 제공 기본값 | 기존 셸 환경 |

병합은 **리프 수준**에서 수행됩니다. `model.default`를 고정해도 나머지 `model.*`이 고정되지는 않습니다. 다음과 같은 관리 `config.yaml`은:

```yaml
model:
  default: org/standard-model
```

모든 사용자에게 `model.default`를 강제하지만, `model.fallback`(및 다른 모든 키)은 사용자가 제어하도록 둡니다.

:::note 우선순위 참고
관리 범위는 고정한 키에 대해 셸 환경보다도 의도적으로 우선합니다. 그렇지 않으면 "관리"라고 할 수 없기 때문입니다. 이는 일반적으로 "환경 변수가 config.yaml보다 우선한다"는 규칙을 뒤집는 유일한 경우이며, 관리 계층이 지정한 특정 키에만 적용됩니다.
:::

## 관리되는 항목 확인

```bash
hermes config        # shows a header naming the managed source + the pinned keys
hermes doctor        # reports the resolved managed dir + pinned key counts
```

관리되는 값을 변경하려고 하면 Hermes는 거부하고 출처를 표시합니다.

```text
$ hermes config set model.default my/model
Cannot set 'model.default': it is managed by your administrator
(/etc/hermes/config.yaml) and cannot be changed.
```

관리되는 시크릿도 마찬가지입니다. 관리되는 `.env`에 고정된 환경 키에 대해 `hermes config set`/설정 마법사는 사용자 값을 기록하지 않습니다.

## 관리 범위 설정(관리자용)

```bash
sudo mkdir -p /etc/hermes

# Pin some config values for every user on this machine
sudo tee /etc/hermes/config.yaml >/dev/null <<'YAML'
model:
  provider: nous
security:
  redact_secrets: true
YAML

# Optionally pin a shared, non-sensitive env value
sudo tee /etc/hermes/.env >/dev/null <<'ENV'
OPENAI_API_BASE=https://inference.example.com/v1
ENV

sudo chmod 0755 /etc/hermes
sudo chmod 0644 /etc/hermes/config.yaml /etc/hermes/.env
```

변경 사항은 다음 Hermes 시작 시 적용됩니다(형식이 잘못된 관리 파일은 크게 기록되고 무시됩니다. 시작을 차단하지는 않지만, 관리자는 정책이 적용되는지 `hermes doctor`로 확인해야 합니다).

## 보안 모델 및 제한 사항(v1)

- **적용은 파일 시스템 권한으로만 이루어집니다.** 사용자가 관리 디렉터리에 대한 쓰기 권한을 갖거나 Hermes를 `root`로 실행하면 관리 범위는 권고 사항에 불과합니다.
- **관리되는 `.env`는 모든 사용자가 읽을 수 있습니다**(`0644`). 따라서 로컬 사용자는 누구나 이 파일을 통해 전달된 시크릿을 읽을 수 있습니다. 고도로 민감한 시크릿보다는 공유된 비민감 값(조직 API 기본 URL, 기능 기본값)에 사용하세요.
- **에이전트 자체 도구는 관리되는 *env* 값에 대해 강제로 차단되지 않습니다.** 관리되는 환경 변수는 시작 시 적용되지만, 에이전트가 자체 하위 프로세스 셸 안에서 다른 값을 설정하는 것을 막지는 않습니다. v1은 일반 사용자를 대상으로 한 관리 편의성 경계이지, 탈출할 수 없는 샌드박스가 아닙니다.

다음 항목은 v1에서 의도적으로 **범위에 포함되지 않으며** 추후 추가될 수 있습니다.

- 에이전트 자체가 탈출할 수 없는 강력한 경계
- macOS 및 Windows의 네이티브 관리 위치(v1은 Linux/POSIX 우선)
- 계층화된 정책을 위한 드롭인 조각 디렉터리(`managed.d/`)
- 서명된/무결성이 확인된 관리 파일
- 원격/디바이스 관리(MDM) 전달
- 관리 시크릿에 대한 더 엄격한(그룹 범위) 권한
