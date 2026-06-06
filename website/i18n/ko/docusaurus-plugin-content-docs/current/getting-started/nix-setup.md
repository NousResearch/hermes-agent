---
sidebar_position: 3
title: "Nix & NixOS 설정"
description: "Nix를 사용하여 Hermes Agent 설치 및 배포 — 빠른 `nix run`부터 컨테이너 모드를 지원하는 완전 선언적 NixOS 모듈까지"
---

# Nix & NixOS 설정

Hermes Agent는 세 가지 통합 레벨의 Nix flake를 제공합니다:

| 레벨 | 대상 | 제공되는 기능 |
|-------|-------------|--------------|
| **`nix run` / `nix profile install`** | 모든 Nix 사용자 (macOS, Linux) | 모든 의존성이 포함된 사전 빌드 바이너리 — 이후 표준 CLI 워크플로우 사용 |
| **NixOS 모듈 (네이티브)** | NixOS 서버 배포 | 선언적 설정, 강화된 systemd 서비스, 관리되는 비밀값(secrets) |
| **NixOS 모듈 (컨테이너)** | 자체 수정이 필요한 에이전트 | 위 기능에 더해 에이전트가 `apt`/`pip`/`npm install`을 수행할 수 있는 지속성 Ubuntu 컨테이너 제공 |

:::info 표준 설치와의 차이점
`curl | bash` 설치 프로그램은 Python, Node 및 의존성을 직접 관리합니다. Nix flake는 이 모든 것을 대체합니다. 모든 Python 의존성은 [uv2nix](https://github.com/pyproject-nix/uv2nix)에 의해 빌드된 Nix derivation이며, 런타임 도구(Node.js, git, ripgrep, ffmpeg)는 바이너리의 PATH에 래핑되어 있습니다. 런타임 pip나 venv 활성화, `npm install`은 존재하지 않습니다.

**NixOS 미사용자**의 경우, 설치 단계만 달라집니다. 그 이후의 과정(`hermes setup`, `hermes gateway install`, 설정 수정 등)은 표준 설치와 동일하게 작동합니다.

**NixOS 모듈 사용자**의 경우, 전체 생명주기가 달라집니다. 설정은 `configuration.nix`에 위치하고, 비밀값(secrets)은 sops-nix/agenix를 통해 관리되며, 서비스는 systemd 유닛으로 동작하고 CLI 설정 명령은 차단됩니다. 다른 NixOS 서비스를 관리하는 것과 동일한 방식으로 hermes를 관리합니다.
:::

## 요구 사항

- **Flake가 활성화된 Nix** — [Determinate Nix](https://install.determinate.systems) 권장 (기본적으로 flake 활성화됨)
- 사용할 서비스의 **API 키** (최소한 OpenRouter 또는 Anthropic 키 필요)

---

## 빠른 시작 (모든 Nix 사용자)

클론할 필요가 없습니다. Nix가 모든 것을 가져오고 빌드하며 실행합니다:

```bash
# 직접 실행 (첫 사용 시 빌드, 이후 캐시됨)
nix run github:NousResearch/hermes-agent -- setup
nix run github:NousResearch/hermes-agent -- chat

# 또는 영구 설치
nix profile install github:NousResearch/hermes-agent
hermes setup
hermes chat
```

`nix profile install`을 실행하면 `hermes`, `hermes-agent`, `hermes-acp`가 PATH에 추가됩니다. 여기서부터 워크플로우는 [표준 설치](./installation.md)와 동일합니다. `hermes setup`을 통해 프로바이더 선택 과정을 진행하고, `hermes gateway install`로 launchd (macOS) 또는 systemd 사용자 서비스를 설정하며, 설정 파일은 `~/.hermes/`에 저장됩니다.

:::warning 메시징 플랫폼 (Discord, Telegram, Slack)
기본 패키지에는 메시징 플랫폼 라이브러리가 포함되어 있지 않습니다. 이 라이브러리들은 요청 시 설치(on-demand) 방식으로 변경되었으나, Nix의 읽기 전용 환경에서는 작동할 수 없습니다. 에이전트를 Discord, Telegram, 또는 Slack에 연결하려는 경우 `messaging` 변형(variant)을 설치하십시오:

```bash
nix profile install github:NousResearch/hermes-agent#messaging
```

모든 선택적 추가 기능(음성, 모든 프로바이더, 모든 플랫폼)을 설치하려면 다음을 실행하십시오:

```bash
nix profile install github:NousResearch/hermes-agent#full
```

`full` 변형은 클로저(closure) 크기를 약 700MB 늘립니다. 메시징 플랫폼만 필요한 경우 `#messaging`은 약 33MB만 추가합니다.
:::

<details>
<summary><strong>로컬 클론에서 빌드하기</strong></summary>

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
nix build
./result/bin/hermes setup
```

</details>

---

## NixOS 모듈

Flake는 `nixosModules.default`를 내보냅니다. 이는 사용자 생성, 디렉터리, 설정 생성, 비밀값, 문서 및 서비스 생명주기를 선언적으로 관리하는 전체 NixOS 서비스 모듈입니다.

:::note
이 모듈은 NixOS가 필요합니다. NixOS가 아닌 시스템(macOS, 다른 Linux 배포판)에서는 `nix profile install` 및 위의 표준 CLI 워크플로우를 사용하십시오.
:::

### Flake 입력 추가하기

```nix
# /etc/nixos/flake.nix (또는 시스템 flake)
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    hermes-agent.url = "github:NousResearch/hermes-agent";
  };

  outputs = { nixpkgs, hermes-agent, ... }: {
    nixosConfigurations.your-host = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        hermes-agent.nixosModules.default
        ./configuration.nix
      ];
    };
  };
}
```

### 최소 설정

```nix
# configuration.nix
{ config, ... }: {
  services.hermes-agent = {
    enable = true;
    settings.model.default = "anthropic/claude-sonnet-4";
    environmentFiles = [ config.sops.secrets."hermes-env".path ];
    addToSystemPackages = true;
  };
}
```

이것이 전부입니다. `nixos-rebuild switch`를 실행하면 `hermes` 사용자가 생성되고, `config.yaml`이 생성되며, 비밀값이 연결되고, 게이트웨이가 시작됩니다. 게이트웨이는 에이전트를 메시징 플랫폼(Telegram, Discord 등)에 연결하고 수신 메시지를 대기하는 장기 실행 서비스입니다.

:::warning 비밀값(Secrets)이 필요합니다
위의 `environmentFiles` 라인은 [sops-nix](https://github.com/Mic92/sops-nix) 또는 [agenix](https://github.com/ryantm/agenix)가 설정되어 있다고 가정합니다. 이 파일에는 최소 하나 이상의 LLM 프로바이더 키가 포함되어 있어야 합니다(예: `OPENROUTER_API_KEY=sk-or-...`). 전체 설정은 [비밀값 관리](#비밀값-관리)를 참조하십시오. 아직 비밀값 관리 도구가 없다면 임시방편으로 일반 파일을 사용할 수 있습니다. 단, 이 파일은 타인이 읽을 수 없도록 권한을 설정해야 합니다:

```bash
echo "OPENROUTER_API_KEY=sk-or-your-key" | sudo install -m 0600 -o hermes /dev/stdin /var/lib/hermes/env
```

```nix
services.hermes-agent.environmentFiles = [ "/var/lib/hermes/env" ];
```
:::

:::tip addToSystemPackages
`addToSystemPackages = true` 설정은 두 가지 작업을 수행합니다: 시스템 PATH에 `hermes` CLI를 추가하고, 시스템 전역에 `HERMES_HOME`을 설정하여 대화형 CLI가 게이트웨이 서비스와 상태(세션, 스킬, 크론)를 공유하도록 합니다. 이 설정이 없으면 셸에서 `hermes`를 실행할 때 별도의 `~/.hermes/` 디렉터리가 생성됩니다.
:::

### 컨테이너 인식 CLI

:::info
`container.enable = true` 및 `addToSystemPackages = true`가 설정되면, 호스트에서의 **모든** `hermes` 명령은 자동으로 관리되는 컨테이너로 라우팅됩니다. 즉, 사용자의 대화형 CLI 세션이 게이트웨이 서비스와 동일한 환경에서 실행되며, 컨테이너 내부에 설치된 모든 패키지와 도구에 접근할 수 있게 됩니다.

- 라우팅은 투명하게 이루어집니다: `hermes chat`, `hermes sessions list`, `hermes version` 등은 내부적으로 컨테이너 안에서 실행됩니다.
- 모든 CLI 플래그는 있는 그대로 전달됩니다.
- 컨테이너가 실행 중이 아닐 경우, CLI는 잠시 재시도(대화형 사용 시 스피너와 함께 5초, 스크립트의 경우 백그라운드에서 10초)한 후 오류 메세지를 표시하며 실패합니다. 다른 환경으로 폴백하지 않습니다.
- hermes 코드베이스를 직접 작업하는 개발자의 경우, `HERMES_DEV=1`을 설정하여 컨테이너 라우팅을 우회하고 로컬 코드를 직접 실행할 수 있습니다.

호스트 CLI와 컨테이너가 세션, 설정 및 메모리를 공유할 수 있도록 서비스 상태 디렉터리를 가리키는 `~/.hermes` 심볼릭 링크를 생성하려면 `container.hostUsers`를 설정하십시오:

```nix
services.hermes-agent = {
  container.enable = true;
  container.hostUsers = [ "your-username" ];
  addToSystemPackages = true;
};
```

`hostUsers`에 명시된 사용자는 파일 권한 접근을 위해 자동으로 `hermes` 그룹에 추가됩니다.

**Podman 사용자:** NixOS 서비스는 컨테이너를 root 권한으로 실행합니다. Docker 사용자는 `docker` 그룹 소켓을 통해 접근할 수 있지만, Podman의 rootful 컨테이너는 sudo 권한이 필요합니다. 사용하는 컨테이너 런타임에 대해 패스워드 없는 sudo 권한을 부여하십시오:

```nix
security.sudo.extraRules = [{
  users = [ "your-username" ];
  commands = [{
    command = "/run/current-system/sw/bin/podman";
    options = [ "NOPASSWD" ];
  }];
}];
```

CLI는 sudo가 필요한 시점을 자동으로 감지하고 투명하게 적용합니다. 이 설정이 없으면 `sudo hermes chat`을 수동으로 실행해야 합니다.
:::

### 작동 확인

`nixos-rebuild switch` 실행 후, 서비스가 정상 작동하는지 확인합니다:

```bash
# 서비스 상태 확인
systemctl status hermes-agent

# 로그 확인 (종료하려면 Ctrl+C)
journalctl -u hermes-agent -f

# addToSystemPackages가 true인 경우 CLI 테스트
hermes version
hermes config       # 생성된 설정을 보여줍니다
```

### 배포 모드 선택

이 모듈은 `container.enable`에 의해 제어되는 두 가지 모드를 지원합니다:

| | **네이티브** (기본값) | **컨테이너** |
|---|---|---|
| 실행 방식 | 호스트에서 격리된 systemd 서비스로 실행 | `/nix/store`가 바인드 마운트된 지속성 Ubuntu 컨테이너로 실행 |
| 보안 | `NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp` | 컨테이너 격리, 내부에서 비특권 사용자로 실행 |
| 에이전트 패키지 자체 설치 가능 여부 | 불가능 — Nix가 제공하는 PATH의 도구만 사용 가능 | 가능 — `apt`, `pip`, `npm` 설치가 재시작 후에도 유지됨 |
| 설정 범위 | 동일 | 동일 |
| 선택 기준 | 표준 배포, 최대 보안, 재현성 필요 시 | 에이전트에 런타임 패키지 설치, 변경 가능한 환경, 실험적 도구가 필요할 때 |

컨테이너 모드를 활성화하려면 한 라인만 추가하면 됩니다:

```nix
{
  services.hermes-agent = {
    enable = true;
    container.enable = true;
    # ... 나머지 설정은 동일합니다
  };
}
```

:::info
컨테이너 모드는 `mkDefault`를 통해 `virtualisation.docker.enable`을 자동 활성화합니다. 대신 Podman을 사용하는 경우 `container.backend = "podman"` 및 `virtualisation.docker.enable = false`를 설정하십시오.
:::

---

## 설정

### 선언적 설정

`settings` 옵션은 `config.yaml`로 렌더링될 임의의 속성 세트(attrset)를 받습니다. 여러 모듈 정의에 걸쳐 깊은 병합(deep merge)을 지원하므로(`lib.recursiveUpdate` 사용), 여러 파일로 설정을 나눌 수 있습니다:

```nix
# base.nix
services.hermes-agent.settings = {
  model.default = "anthropic/claude-sonnet-4";
  toolsets = [ "all" ];
  terminal = { backend = "local"; timeout = 180; };
};

# personality.nix
services.hermes-agent.settings = {
  display = { compact = false; personality = "kawaii"; };
  memory = { memory_enabled = true; user_profile_enabled = true; };
};
```

두 설정은 평가 시점에 깊은 병합이 이루어집니다. Nix로 선언된 키는 디스크에 있는 기존 `config.yaml`의 키보다 항상 우선하지만, **Nix가 건드리지 않은 사용자 추가 키는 보존됩니다**. 즉, 에이전트나 수동 편집으로 `skills.disabled` 또는 `streaming.enabled`와 같은 키를 추가해도 `nixos-rebuild switch` 실행 시 유지됩니다.

:::note 모델 명명 규칙
`settings.model.default`에는 사용하는 프로바이더가 기대하는 모델 식별자를 입력합니다. [OpenRouter](https://openrouter.ai)(기본값)의 경우 `"anthropic/claude-sonnet-4"` 또는 `"google/gemini-3-flash"`와 같이 지정합니다. 프로바이더(Anthropic, OpenAI)를 직접 이용하는 경우에는 `settings.model.base_url`이 해당 API를 가리키도록 설정하고 프로바이더 고유의 모델 ID(예: `"claude-sonnet-4-20250514"`)를 사용하십시오. `base_url`이 설정되지 않은 경우 Hermes는 기본적으로 OpenRouter를 사용합니다.
:::

:::tip 사용 가능한 설정 키 찾기
`nix build .#configKeys && cat result`를 실행하여 Python의 `DEFAULT_CONFIG`에서 추출된 모든 설정 키 목록을 확인할 수 있습니다. 기존의 `config.yaml` 내용을 `settings` 속성 세트에 그대로 복사할 수 있으며, 구조는 1:1로 매핑됩니다.
:::

<details>
<summary><strong>전체 예시: 자주 수정되는 모든 설정</strong></summary>

```nix
{ config, ... }: {
  services.hermes-agent = {
    enable = true;
    container.enable = true;

    # ── 모델 ──────────────────────────────────────────────────────────
    settings = {
      model = {
        base_url = "https://openrouter.ai/api/v1";
        default = "anthropic/claude-opus-4.6";
      };
      toolsets = [ "all" ];
      max_turns = 100;
      terminal = { backend = "local"; cwd = "."; timeout = 180; };
      compression = {
        enabled = true;
        threshold = 0.85;
        summary_model = "google/gemini-3-flash-preview";
      };
      memory = { memory_enabled = true; user_profile_enabled = true; };
      display = { compact = false; personality = "kawaii"; };
      agent = { max_turns = 60; verbose = false; };
    };

    # ── 비밀값 ────────────────────────────────────────────────────────
    environmentFiles = [ config.sops.secrets."hermes-env".path ];

    # ── 문서 ──────────────────────────────────────────────────────
    documents = {
      "USER.md" = ./documents/USER.md;
    };

    # ── MCP 서버 ────────────────────────────────────────────────────
    mcpServers.filesystem = {
      command = "npx";
      args = [ "-y" "@modelcontextprotocol/server-filesystem" "/data/workspace" ];
    };

    # ── 컨테이너 옵션 ──────────────────────────────────────────────
    container = {
      image = "ubuntu:24.04";
      backend = "docker";
      hostUsers = [ "your-username" ];
      extraVolumes = [ "/home/user/projects:/projects:rw" ];
      extraOptions = [ "--gpus" "all" ];
    };

    # ── 서비스 튜닝 ─────────────────────────────────────────────────
    addToSystemPackages = true;
    extraArgs = [ "--verbose" ];
    restart = "always";
    restartSec = 5;
  };
}
```

</details>

### 비상 탈출구: 자체 설정 파일 직접 제공하기

`config.yaml`을 Nix 외부에서 완전히 따로 관리하고 싶다면 `configFile`을 사용하십시오:

```nix
services.hermes-agent.configFile = /etc/hermes/config.yaml;
```

이 방식은 `settings`를 완전히 무시하며 병합이나 생성을 하지 않습니다. 활성화할 때마다 해당 파일이 `$HERMES_HOME/config.yaml`로 그대로 복사됩니다.

### 커스텀 치트시트

Nix 사용자들이 가장 자주 사용자 정의하려는 항목에 대한 빠른 참조 정보입니다:

| 작업하고자 하는 내용 | 옵션 | 예시 |
|---|---|---|
| LLM 모델 변경 | `settings.model.default` | `"anthropic/claude-sonnet-4"` |
| 다른 프로바이더 엔드포인트 사용 | `settings.model.base_url` | `"https://openrouter.ai/api/v1"` |
| API 키 추가 | `environmentFiles` | `[ config.sops.secrets."hermes-env".path ]` |
| 에이전트 페르소나 설정 | `${services.hermes-agent.stateDir}/.hermes/SOUL.md` | 파일을 직접 관리 |
| MCP 도구 서버 추가 | `mcpServers.<이름>` | [MCP 서버](#mcp-서버) 참조 |
| Discord/Telegram/Slack 활성화 | `extraDependencyGroups` | `[ "messaging" ]` |
| 호스트 디렉터리를 컨테이너에 마운트 | `container.extraVolumes` | `[ "/data:/data:rw" ]` |
| 컨테이너에 GPU 접근 권한 부여 | `container.extraOptions` | `[ "--gpus" "all" ]` |
| Docker 대신 Podman 사용 | `container.backend` | `"podman"` |
| 호스트 CLI와 컨테이너 간 상태 공유 | `container.hostUsers` | `[ "sidbin" ]` |
| 에이전트가 추가 도구를 사용할 수 있도록 제공 | `extraPackages` | `[ pkgs.pandoc pkgs.imagemagick ]` |
| 커스텀 베이스 이미지 사용 | `container.image` | `"ubuntu:24.04"` |
| hermes 패키지 오버라이드 | `package` | `inputs.hermes-agent.packages.${system}.default.override { ... }` |
| 상태 디렉터리 변경 | `stateDir` | `"/opt/hermes"` |
| 에이전트의 작업 디렉터리 설정 | `workingDirectory` | `"/home/user/projects"` |

---

## 비밀값 관리

:::danger API 키를 `settings` 또는 `environment`에 직접 입력하지 마십시오
Nix 표현식의 값들은 누구나 읽을 수 있는 `/nix/store` 경로에 저장됩니다. 비밀값은 항상 비밀값 관리 도구와 `environmentFiles`를 함께 사용하십시오.
:::

`environment`(비밀이 아닌 변수) 및 `environmentFiles`(비밀값 파일)는 활성화 시점(`nixos-rebuild switch`)에 `$HERMES_HOME/.env`로 병합됩니다. Hermes는 시작할 때마다 이 파일을 읽으므로 `systemctl restart hermes-agent` 실행 시 변경 사항이 적용됩니다. 컨테이너를 다시 생성할 필요는 없습니다.

### sops-nix

```nix
{
  sops = {
    defaultSopsFile = ./secrets/hermes.yaml;
    age.keyFile = "/home/user/.config/sops/age/keys.txt";
    secrets."hermes-env" = { format = "yaml"; };
  };

  services.hermes-agent.environmentFiles = [
    config.sops.secrets."hermes-env".path
  ];
}
```

비밀값 파일은 키-값 쌍을 포함합니다:

```yaml
# secrets/hermes.yaml (sops로 암호화됨)
hermes-env: |
    OPENROUTER_API_KEY=sk-or-...
    TELEGRAM_BOT_TOKEN=123456:ABC...
    ANTHROPIC_API_KEY=sk-ant-...
```

### agenix

```nix
{
  age.secrets.hermes-env.file = ./secrets/hermes-env.age;

  services.hermes-agent.environmentFiles = [
    config.age.secrets.hermes-env.path
  ];
}
```

### OAuth / 인증 시딩 (Auth Seeding)

Discord 등 OAuth가 필요한 플랫폼의 경우 `authFile`을 사용하여 첫 배포 시 인증 정보를 주입할 수 있습니다:

```nix
{
  services.hermes-agent = {
    authFile = config.sops.secrets."hermes/auth.json".path;
    # authFileForceOverwrite = true;  # 활성화할 때마다 덮어쓰기
  };
}
```

이 파일은 `auth.json`이 아직 존재하지 않는 경우에만 복사됩니다(`authFileForceOverwrite = true`가 아닌 한). 런타임 중에 업데이트된 OAuth 토큰은 상태 디렉터리에 기록되어 리빌드 후에도 유지됩니다.

---

## 문서

`documents` 옵션은 파일을 에이전트의 작업 디렉터리(에이전트가 작업 공간으로 읽는 `workingDirectory`)에 설치합니다. Hermes는 관례에 따라 특정 파일명을 참조합니다:

- **`USER.md`** — 에이전트가 대화하는 사용자에 관한 컨텍스트 파일.
- 여기에 추가로 배치하는 모든 파일은 에이전트가 작업 공간 파일로 직접 볼 수 있습니다.

에이전트 자체의 페르소나 파일은 별도입니다. Hermes는 주요 페르소나 정보를 `$HERMES_HOME/SOUL.md`에서 읽어옵니다. NixOS 모듈의 경우 해당 경로는 `${services.hermes-agent.stateDir}/.hermes/SOUL.md`입니다. `documents`에 `SOUL.md`를 지정하면 작업 공간 파일로만 생성될 뿐 에이전트의 핵심 페르소나 파일을 대체하지는 않습니다.

```nix
{
  services.hermes-agent.documents = {
    "USER.md" = ./documents/USER.md;  # Nix 스토어에서 복사되는 경로 참조
  };
}
```

값은 인라인 문자열 또는 파일 경로가 될 수 있습니다. 파일은 `nixos-rebuild switch`가 실행될 때마다 재설치됩니다.

---

## MCP 서버

`mcpServers` 옵션은 [MCP (Model Context Protocol)](https://modelcontextprotocol.io) 서버를 선언적으로 설정합니다. 각 서버는 **stdio** (로컬 명령어) 또는 **HTTP** (원격 URL) 전송 방식을 사용합니다.

### Stdio 전송 방식 (로컬 서버)

```nix
{
  services.hermes-agent.mcpServers = {
    filesystem = {
      command = "npx";
      args = [ "-y" "@modelcontextprotocol/server-filesystem" "/data/workspace" ];
    };
    github = {
      command = "npx";
      args = [ "-y" "@modelcontextprotocol/server-github" ];
      env.GITHUB_PERSONAL_ACCESS_TOKEN = "\${GITHUB_TOKEN}"; # .env 파일에서 해석됨
    };
  };
}
```

:::tip
`env` 값에 들어있는 환경 변수는 런타임에 `$HERMES_HOME/.env`에서 해석됩니다. 비밀값 전달을 위해서는 `environmentFiles`를 이용하십시오. Nix 설정에 토큰을 직접 적으면 절대 안 됩니다.
:::

### HTTP 전송 방식 (원격 서버)

```nix
{
  services.hermes-agent.mcpServers.remote-api = {
    url = "https://mcp.example.com/v1/mcp";
    headers.Authorization = "Bearer \${MCP_REMOTE_API_KEY}";
    timeout = 180;
  };
}
```

### OAuth를 지원하는 HTTP 전송 방식

OAuth 2.1을 사용하는 서버인 경우 `auth = "oauth"`로 설정하십시오. Hermes는 메타데이터 탐색, 동적 클라이언트 등록, 토큰 교환 및 자동 갱신을 포함한 PKCE 전체 흐름을 지원합니다.

```nix
{
  services.hermes-agent.mcpServers.my-oauth-server = {
    url = "https://mcp.example.com/mcp";
    auth = "oauth";
  };
}
```

토큰은 `$HERMES_HOME/mcp-tokens/<서버-이름>.json`에 저장되어 재시작 및 리빌드 시에도 유지됩니다.

<details>
<summary><strong>헤드리스 서버에서의 초기 OAuth 인증</strong></summary>

최초 OAuth 인증 시에는 브라우저 기반 동의 절차가 필요합니다. 헤드리스 환경의 경우 Hermes는 브라우저를 직접 띄우지 못하므로 대신 표준 출력(stdout) 및 로그에 인증 URL을 출력합니다.

**방법 A: 대화형 부트스트랩** — `docker exec` (컨테이너 모드) 또는 `sudo -u hermes` (네이티브 모드) 명령을 사용하여 1회 실행합니다:

```bash
# 컨테이너 모드
docker exec -it hermes-agent \
  hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth

# 네이티브 모드
sudo -u hermes HERMES_HOME=/var/lib/hermes/.hermes \
  hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth
```

컨테이너 모드는 `--network=host`를 사용하므로 호스트 브라우저에서 `127.0.0.1`로 띄워진 OAuth 콜백 리스너에 바로 접근할 수 있습니다.

**방법 B: 토큰 미리 심기** — 로컬 PC에서 인증 흐름을 완료한 다음 서버로 토큰을 복사합니다:

```bash
hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth
scp ~/.hermes/mcp-tokens/my-oauth-server{,.client}.json \
    server:/var/lib/hermes/.hermes/mcp-tokens/
# 복사 후 소유자 및 권한 설정 필수: chown hermes:hermes, chmod 0600
```

</details>

### 샘플링 (서버 유도 LLM 요청)

일부 MCP 서버는 에이전트에게 LLM 텍스트 완성을 역으로 요청할 수 있습니다:

```nix
{
  services.hermes-agent.mcpServers.analysis = {
    command = "npx";
    args = [ "-y" "analysis-server" ];
    sampling = {
      enabled = true;
      model = "google/gemini-3-flash";
      max_tokens_cap = 4096;
      timeout = 30;
      max_rpm = 10;
    };
  };
}
```

---

## 관리형 모드 (Managed Mode)

Hermes가 NixOS 모듈을 통해 실행되는 경우, 디스크 설정과 선언적 설정 사이의 불일치를 막기 위해 아래 CLI 명령어가 **차단**되며, 대신 `configuration.nix`를 편집하도록 안내하는 오류 메시지가 출력됩니다:

| 차단된 명령어 | 이유 |
|---|---|
| `hermes setup` | 설정이 선언적으로 관리되므로 Nix 설정의 `settings`를 수정해야 함 |
| `hermes config edit` | 설정 파일이 Nix `settings`로부터 자동 생성됨 |
| `hermes config set <key> <value>` | 설정 파일이 Nix `settings`로부터 자동 생성됨 |
| `hermes gateway install` | systemd 서비스가 NixOS에 의해 제어 및 관리됨 |
| `hermes gateway uninstall` | systemd 서비스가 NixOS에 의해 제어 및 관리됨 |

이를 위해 두 가지 신호로 관리형 모드 여부를 감지합니다:

1. **`HERMES_MANAGED=true`** 환경 변수 — systemd 서비스가 설정하며, 게이트웨이 프로세스가 감지합니다.
2. **`HERMES_HOME` 내부의 `.managed` 마커 파일** — 활성화 스크립트에 의해 생성되며, 대화형 셸에서 감지합니다 (예: `docker exec` 상태에서 `hermes config set ...` 실행 시 차단됨).

설정을 변경하려면 Nix 설정을 수정한 후 `sudo nixos-rebuild switch`를 실행하십시오.

---

## 컨테이너 아키텍처

:::info
이 섹션은 `container.enable = true`를 사용하는 경우에만 해당됩니다. 네이티브 모드로 배포하는 경우에는 이 섹션을 생략하십시오.
:::

컨테이너 모드가 활성화되면 hermes는 지속성을 가지는 Ubuntu 컨테이너 내부에서 작동하며, 호스트에 빌드된 Nix 바이너리를 읽기 전용으로 바인드 마운트하여 사용합니다:

```
호스트 (Host)                          컨테이너 (Container)
────                                    ─────────
/nix/store/...-hermes-agent-0.1.0  ──►  /nix/store/... (읽기 전용)
~/.hermes -> /var/lib/hermes/.hermes       (심볼릭 링크 브릿지, hostUsers 기준)
/var/lib/hermes/                    ──►  /data/          (읽기/쓰기 가능)
  ├── current-package -> /nix/store/...    (심볼릭 링크, 리빌드 시마다 업데이트)
  ├── .gc-root -> /nix/store/...           (nix-collect-garbage로 인한 삭제 방지)
  ├── .container-identity                  (sha256 해시, 변경 시 컨테이너 재생성 트리거)
  ├── .hermes/                             (HERMES_HOME)
  │   ├── .env                             (environment + environmentFiles 병합 결과)
  │   ├── config.yaml                      (Nix에서 생성, 활성화 시점에 병합됨)
  │   ├── .managed                         (마커 파일)
  │   ├── .container-mode                  (라우팅 메타데이터: 백엔드, 실행 사용자 등)
  │   ├── state.db, sessions/, memories/   (런타임 상태 데이터)
  │   └── mcp-tokens/                      (MCP 서버용 OAuth 토큰)
  ├── home/                                ──►  /home/hermes    (읽기/쓰기 가능)
  └── workspace/                           (에이전트 작업 공간 디렉터리)
      ├── SOUL.md                          (documents 옵션에서 가져옴)
      └── (에이전트가 생성한 파일들)

컨테이너 쓰기 가능 레이어 (apt/pip/npm):    /usr, /usr/local, /tmp
```

Nix로 빌드된 바이너리는 `/nix/store` 경로가 마운트되어 있기 때문에 Ubuntu 컨테이너 내부에서도 올바르게 실행됩니다. 자체 인터프리터와 모든 의존성을 직접 가져오므로 컨테이너의 시스템 라이브러리에 의존하지 않습니다. 컨테이너의 진입점(entrypoint)은 `current-package` 심볼릭 링크를 해석하여 가리킵니다: `/data/current-package/bin/hermes gateway run --replace`. `nixos-rebuild switch`를 실행하면 이 심볼릭 링크만 업데이트되므로 컨테이너가 중단 없이 계속 작동합니다.

### 구성 요소별 변경 시 유지 방식

| 상황 | 컨테이너 재생성 여부 | `/data` (상태) | `/home/hermes` | 쓰기 가능 레이어 (`apt`/`pip`/`npm`) |
|---|---|---|---|---|
| `systemctl restart hermes-agent` | 재생성 없음 | 유지됨 | 유지됨 | 유지됨 |
| `nixos-rebuild switch` (코드 변경) | 재생성 없음 (심볼릭 링크만 변경) | 유지됨 | 유지됨 | 유지됨 |
| 호스트 재부팅 | 재생성 없음 | 유지됨 | 유지됨 | 유지됨 |
| `nix-collect-garbage` | 재생성 없음 (GC root 보호) | 유지됨 | 유지됨 | 유지됨 |
| 이미지 변경 (`container.image`) | **재생성됨** | 유지됨 | 유지됨 | **삭제됨** |
| 볼륨/옵션 변경 | **재생성됨** | 유지됨 | 유지됨 | **삭제됨** |
| `environment`/`environmentFiles` 변경 | 재생성 없음 | 유지됨 | 유지됨 | 유지됨 |

컨테이너는 **아이덴티티 해시(identity hash)**가 변경될 때만 새로 생성됩니다. 이 해시는 스키마 버전, 이미지명, `extraVolumes`, `extraOptions`, 진입점 스크립트의 변경 사항을 추적합니다. 환경 변수, 설정값, 문서 데이터, 혹은 hermes 패키지 자체의 변경은 컨테이너 재생성을 트리거하지 않습니다.

:::warning 쓰기 가능 레이어 삭제 주의
아이덴티티 해시가 변경되는 경우(이미지 업그레이드, 볼륨 추가, 컨테이너 옵션 수정 등), 기존 컨테이너는 삭제되고 `container.image`로부터 새롭게 컨테이너가 생성됩니다. 이에 따라 쓰기 가능 레이어에 `apt install`, `pip install`, `npm install` 등으로 설치했던 모든 패키지는 삭제됩니다. 바인드 마운트된 `/data` 및 `/home/hermes` 디렉터리의 데이터는 안전하게 보존됩니다.

에이전트에 특정 외부 패키지가 꼭 필요한 경우, 커스텀 이미지로 직접 빌드하여 지정하거나(`container.image = "my-registry/hermes-base:latest"`), 에이전트의 `SOUL.md` 내부 스크립트에서 런타임에 설치 흐름을 제어하도록 구성하는 것을 권장합니다.
:::

### GC Root 보호

`preStart` 스크립트는 현재 실행 중인 hermes 패키지를 가리키는 GC root를 `${stateDir}/.gc-root`에 생성합니다. 이를 통해 `nix-collect-garbage`를 수행할 때 현재 서비스 작동 중인 바이너리가 임의로 지워지는 것을 방지합니다. 만약 이 파일에 문제가 생겨도 서비스를 재시작하면 자동으로 다시 생성됩니다.

---

## 플러그인

NixOS 모듈은 명령형 방식의 `hermes plugins install`을 사용하지 않고, 선언적 플러그인 설치를 지원합니다.

### 디렉터리 플러그인 (`extraPlugins`)

`plugin.yaml` 및 `__init__.py`가 포함된 소스 트리 구조의 플러그인인 경우(예: [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm)):

```nix
services.hermes-agent.extraPlugins = [
  (pkgs.fetchFromGitHub {
    owner = "stephenschoettler";
    repo = "hermes-lcm";
    rev = "v0.7.0";
    hash = "sha256-...";
  })
];
```

이 플러그인들은 활성화 시점에 `$HERMES_HOME/plugins/` 디렉터리에 심볼릭 링크로 연결됩니다. Hermes는 일반 디렉터리 스캔을 통해 이 플러그인들을 찾아냅니다. 리스트에서 플러그인을 제거하고 `nixos-rebuild switch`를 실행하면 해당 심볼릭 링크도 제거됩니다.

### 엔트리 포인트 플러그인 (`extraPythonPackages`)

`[project.entry-points."hermes_agent.plugins"]`를 통해 등록되는 pip 패키지 형태의 플러그인인 경우(예: [rtk-hermes](https://github.com/ogallotti/rtk-hermes)):

```nix
services.hermes-agent.extraPythonPackages = [
  (pkgs.python312Packages.buildPythonPackage {
    pname = "rtk-hermes";
    version = "1.0.0";
    src = pkgs.fetchFromGitHub {
      owner = "ogallotti";
      repo = "rtk-hermes";
      rev = "v1.0.0";
      hash = "sha256-...";
    };
    format = "pyproject";
    build-system = [ pkgs.python312Packages.setuptools ];
  })
];
```

패키지의 `site-packages` 경로는 hermes 래퍼 내부에서 PYTHONPATH 환경 변수에 추가됩니다. 세션이 시작될 때 `importlib.metadata`가 엔트리 포인트를 탐색하여 찾아냅니다.

### 선택적 의존성 그룹 (`extraDependencyGroups`)

hermes-agent의 `pyproject.toml`에 선언된 선택적 추가 기능(extras)의 경우, 빌드 시 격리된 venv에 이들을 포함시키기 위해 `extraDependencyGroups`를 사용할 수 있습니다. Nix에서는 읽기 전용 스토어에 런타임 설치가 불가능하므로, 기본 `[all]` 세트에 포함되지 않은 기능을 사용하려면 이 설정이 필수적입니다.

```nix
# Discord, Telegram, Slack 활성화
services.hermes-agent.extraDependencyGroups = [ "messaging" ];
```

```nix
# 메모리 프로바이더 활성화
services.hermes-agent = {
  extraDependencyGroups = [ "hindsight" ];
  settings.memory.provider = "hindsight";
};
```

이 설정은 핵심 의존성과 함께 uv에 의해 해석되므로 PYTHONPATH를 따로 조작할 필요가 없으며, 라이브러리 간 충돌 위험이 없습니다. 사용 가능한 그룹은 다음과 같습니다:

| 그룹 | 제공 기능 |
|-------|-----------------|
| `messaging` | Discord, Telegram, Slack |
| `matrix` | Matrix/Element (암호화가 지원되는 mautrix; Linux 전용) |
| `dingtalk` | DingTalk |
| `feishu` | Feishu/Lark |
| `voice` | 로컬 음성 인식 (faster-whisper) |
| `edge-tts` | Edge TTS 프로바이더 |
| `tts-premium` | ElevenLabs TTS |
| `anthropic` | Anthropic 순수 SDK (OpenRouter를 사용할 때는 불필요) |
| `bedrock` | AWS Bedrock (boto3) |
| `azure-identity` | Azure Entra ID 인증 |
| `honcho` | Honcho 메모리 프로바이더 |
| `hindsight` | Hindsight 메모리 프로바이더 |
| `modal` | Modal 터미널 백엔드 |
| `daytona` | Daytona 터미널 백엔드 |
| `exa` | Exa 웹 검색 |
| `firecrawl` | Firecrawl 웹 검색 |
| `fal` | FAL 이미지 생성 |

개별 추가 설정 대신 미리 빌드된 `#messaging` 또는 `#full` flake 패키지를 바로 사용할 수도 있습니다 ([빠른 시작](#빠른-시작-모든-nix-사용자) 참조).

**상황별 사용법 요약:**

| 요구 사항 | 옵션 |
|------|--------|
| pyproject.toml에 선언된 선택적 추가 기능(extra)을 활성화할 때 | `extraDependencyGroups` |
| pyproject.toml에 없는 외부 Python 플러그인을 추가할 때 | `extraPythonPackages` |
| 시스템 바이너리(pandoc, jq 등)를 추가할 때 | `extraPackages` |
| 디렉터리 기반의 플러그인 소스 트리를 추가할 때 | `extraPlugins` |

### 동시 사용 예시

서드 파티 Python 의존성을 가지는 디렉터리 플러그인의 경우, 다음과 같이 여러 옵션을 동시에 설정해야 합니다:

```nix
services.hermes-agent = {
  extraPlugins = [ my-plugin-src ];          # 플러그인 소스 경로
  extraPythonPackages = [ pkgs.python312Packages.redis ];  # 필요한 Python 라이브러리 의존성
  extraPackages = [ pkgs.redis ];            # 필요한 시스템 바이너리
};
```

### 오버레이(Overlay) 사용법

외부 Flake에서 패키지를 직접 오버라이드할 수도 있습니다:

```nix
{
  inputs.hermes-agent.url = "github:NousResearch/hermes-agent";
  outputs = { hermes-agent, nixpkgs, ... }: {
    nixpkgs.overlays = [ hermes-agent.overlays.default ];
    # 사용 예시:
    #   pkgs.hermes-agent.override { extraPythonPackages = [...]; }
    #   pkgs.hermes-agent.override { extraDependencyGroups = [ "hindsight" ]; }
  };
}
```

### 플러그인 활성화 설정

설치된 플러그인은 `config.yaml`에서도 활성화해 주어야 합니다. 선언적 settings 옵션에 추가하십시오:

```nix
services.hermes-agent.settings.plugins.enabled = [
  "hermes-lcm"
  "rtk-rewrite"
];
```

:::note
빌드 시점 충돌 검사 프로세스를 통해 플러그인 패키지가 hermes의 코어 의존성을 덮어씌우는 것을 미연에 방지합니다. 격리된 venv에 이미 존재하는 패키지를 플러그인이 중복 제공하는 경우, `nixos-rebuild` 시 명확한 에러 메시지와 함께 빌드가 실패합니다.
:::

---

## 개발

### 개발 환경 셸 (Dev Shell)

Flake는 Python 3.12, uv, Node.js 및 기타 실행에 필요한 모든 런타임 도구가 포함된 개발 환경 셸을 제공합니다:

```bash
cd hermes-agent
nix develop

# 셸 환경 정보:
#   - Python 3.12 + uv (진입 시 .venv 환경에 의존성이 자동 설치됨)
#   - Node.js 22, ripgrep, git, openssh, ffmpeg가 PATH에 제공됨
#   - 스탬프 파일 최적화 적용: 의존성이 변경되지 않은 경우 재진입이 거의 즉시 이루어짐

hermes setup
hermes chat
```

### direnv (권장)

기본 포함된 `.envrc`를 사용하면 디렉터리 이동 시 개발 셸이 자동으로 활성화됩니다:

```bash
cd hermes-agent
direnv allow    # 1회 승인 필요
# 이후 진입 시 스탬프 파일 검사를 거쳐 의존성 설치 단계를 건너뛰고 바로 진입합니다.
```

### Flake 검사 (Flake Checks)

Flake 패키지는 로컬 환경 및 CI 단계에서 실행되는 빌드 검증 작업을 포함하고 있습니다:

```bash
# 모든 검사 수행
nix flake check

# 개별 검사 항목 직접 실행
nix build .#checks.x86_64-linux.package-contents   # 바이너리 파일 존재 및 버전 작동 확인
nix build .#checks.x86_64-linux.entry-points-sync  # pyproject.toml ↔ Nix 패키지 동기화 검증
nix build .#checks.x86_64-linux.cli-commands        # gateway/config 서브커맨드 작동 확인
nix build .#checks.x86_64-linux.managed-guard       # HERMES_MANAGED 상태에서 수정 동작 차단 검증
nix build .#checks.x86_64-linux.bundled-skills      # 패키지 내 스킬 목록 포함 여부 검증
nix build .#checks.x86_64-linux.config-roundtrip    # 병합 스크립트의 7가지 시나리오 검증: 최초 설치, Nix 오버라이드, 사용자 설정 보존, 복합 병합, MCP 추가 병합, 깊은 병합, 멱등성 검증
```

<details>
<summary><strong>각 검사 항목 설명</strong></summary>

| 검사 | 상세 내용 |
|---|---|
| `package-contents` | `hermes` 및 `hermes-agent` 바이너리 검사 및 `hermes version` 작동 여부 확인 |
| `entry-points-sync` | `pyproject.toml` 내 `[project.scripts]`의 모든 항목이 Nix 패키지에 래핑된 바이너리로 준비되어 있는지 검증 |
| `cli-commands` | `hermes --help`에서 `gateway` 및 `config` 서브커맨드가 정상적으로 노출되는지 검증 |
| `managed-guard` | `HERMES_MANAGED=true hermes config set ...` 실행 시 지정된 NixOS용 에러가 출력되는지 검증 |
| `bundled-skills` | 스킬 디렉터리의 존재 여부, SKILL.md의 유무, 래퍼의 `HERMES_BUNDLED_SKILLS` 변수 설정 검증 |
| `config-roundtrip` | 설정 병합 검증: 신규 설치, Nix 설정 덮어쓰기, 사용자 지정 키 보호, 다중 병합, MCP 추가 병합, 다중 중첩 병합, 동일한 값 리빌드 테스트 |

</details>

---

## 옵션 레퍼런스

### 핵심 설정 (Core)

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `enable` | `bool` | `false` | hermes-agent 서비스 활성화 여부 |
| `package` | `package` | `hermes-agent` | 사용할 hermes-agent 패키지 지정 |
| `user` | `str` | `"hermes"` | 서비스를 작동시킬 시스템 사용자명 |
| `group` | `str` | `"hermes"` | 서비스를 작동시킬 시스템 그룹명 |
| `createUser` | `bool` | `true` | 사용자 및 그룹의 자동 생성 여부 |
| `stateDir` | `str` | `"/var/lib/hermes"` | 상태 저장 디렉터리 (`HERMES_HOME`이 여기에 포함됨) |
| `workingDirectory` | `str` | `"${stateDir}/workspace"` | 에이전트의 작업 공간 디렉터리 경로 |
| `addToSystemPackages` | `bool` | `false` | 시스템 PATH에 `hermes` CLI를 등록하고 `HERMES_HOME` 환경 변수를 시스템 전역으로 설정 |

### 설정 관리 (Configuration)

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `settings` | `attrs` (깊은 병합) | `{}` | `config.yaml`로 변환될 선언적 속성값. 중첩된 구조를 지원하며 여러 정의가 선언된 경우 `lib.recursiveUpdate`를 거쳐 병합됩니다. |
| `configFile` | `null` 또는 `path` | `null` | 외부의 `config.yaml` 파일 경로. 설정 시 위의 `settings` 명세를 완전히 무시하고 지정된 파일 내용을 사용합니다. |

### 비밀값 및 환경 변수 (Secrets & Environment)

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `environmentFiles` | `listOf str` | `[]` | 비밀값을 담은 환경 변수 파일 목록. 서비스 구동 시 `$HERMES_HOME/.env`로 병합됩니다. |
| `environment` | `attrsOf str` | `{}` | 공개 환경 변수 속성값. **Nix 스토어 경로에 그대로 노출되므로** 민감한 API 키나 비밀값은 여기에 적지 마십시오. |
| `authFile` | `null` 또는 `path` | `null` | OAuth 초기 자격 증명 템플릿. 최초 배포 시 1회만 타겟 경로로 복사됩니다. |
| `authFileForceOverwrite` | `bool` | `false` | 리빌드 시점에 항상 `authFile` 값으로 `auth.json` 파일을 덮어쓸지 여부 |

### 문서 (Documents)

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `documents` | `attrsOf (either str path)` | `{}` | 에이전트 작업 공간에 기본 제공할 파일 목록. 키 이름은 파일명, 값은 텍스트 내용 혹은 로컬 파일 경로입니다. 활성화 시 `workingDirectory`에 복사됩니다. |

### MCP 서버

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `mcpServers` | `attrsOf submodule` | `{}` | MCP 서버 설정 명세. 내부 설정의 `settings.mcp_servers` 항목에 자동 병합됩니다. |
| `mcpServers.<이름>.command` | `null` 또는 `str` | `null` | 실행할 로컬 명령어 (stdio 전송 방식 사용 시 설정) |
| `mcpServers.<이름>.args` | `listOf str` | `[]` | 명령행 인자 목록 |
| `mcpServers.<이름>.env` | `attrsOf str` | `{}` | 해당 MCP 서버 프로세스에 추가할 환경 변수 |
| `mcpServers.<이름>.url` | `null` 또는 `str` | `null` | 원격 MCP 서버 주소 (HTTP/StreamableHTTP 전송 방식 사용 시 설정) |
| `mcpServers.<이름>.headers` | `attrsOf str` | `{}` | HTTP 요청에 포함할 헤더 명세 (예: `Authorization`) |
| `mcpServers.<이름>.auth` | `null` 또는 `"oauth"` | `null` | 인증 처리 방식 지정. `"oauth"` 입력 시 OAuth 2.1 PKCE 처리 절차가 활성화됩니다. |
| `mcpServers.<이름>.enabled` | `bool` | `true` | 이 서버의 활성화 여부 |
| `mcpServers.<이름>.timeout` | `null` 또는 `int` | `null` | 도구 호출 제한 시간 (초 단위, 기본값: 120) |
| `mcpServers.<이름>.connect_timeout` | `null` 또는 `int` | `null` | 연결 재시도 제한 시간 (초 단위, 기본값: 60) |
| `mcpServers.<이름>.tools` | `null` 또는 `submodule` | `null` | 도구 필터링 규칙 설정 (`include`/`exclude` 목록) |
| `mcpServers.<이름>.sampling` | `null` 또는 `submodule` | `null` | 서버가 에이전트에게 역으로 요청하는 LLM 호출에 대한 제한 옵션 |

### 서비스 상세 제어 (Service Behavior)

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `extraArgs` | `listOf str` | `[]` | `hermes gateway` 명령 실행 시 추가로 덧붙일 인수 목록 |
| `extraPackages` | `listOf package` | `[]` | 에이전트 환경에서 접근 가능한 시스템 패키지 목록. `hermes` 전용 프로필에 패키지가 함께 추가되어 터미널, 스킬, 크론 프로세스가 즉시 참조할 수 있습니다. |
| `extraPlugins` | `listOf package` | `[]` | `$HERMES_HOME/plugins/` 디렉터리에 링크할 플러그인 소스 패키지 목록. 하위에 `plugin.yaml` 파일이 준비되어 있어야 합니다. |
| `extraPythonPackages` | `listOf package` | `[]` | 플러그인 엔트리 포인트 스캔이 가능하도록 PYTHONPATH에 포함할 추가 Python 패키지 목록. `python312Packages` 계열을 사용해야 합니다. |
| `extraDependencyGroups` | `listOf str` | `[]` | 빌드 단계에서 hermes 고유 가상환경(venv)에 내장할 `pyproject.toml` 상의 추가 패키지 그룹 목록 (예: `["hindsight"]`). uv를 거치므로 충돌이 방지됩니다. |
| `restart` | `str` | `"always"` | systemd의 `Restart=` 재시작 정책 값 |
| `restartSec` | `int` | `5` | systemd의 `RestartSec=` 재시작 대기 시간 (초 단위) |

### 컨테이너 모드 설정 (Container)

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `container.enable` | `bool` | `false` | OCI 컨테이너 격리 모드 활성화 여부 |
| `container.backend` | `enum ["docker" "podman"]` | `"docker"` | 사용할 컨테이너 런타임 종류 |
| `container.image` | `str` | `"ubuntu:24.04"` | 컨테이너 구동에 사용할 베이스 이미지 (실행 시 다운로드됨) |
| `container.extraVolumes` | `listOf str` | `[]` | 컨테이너에 연동할 추가 볼륨 마운트 목록 (`host:container:mode`) |
| `container.extraOptions` | `listOf str` | `[]` | 컨테이너 생성(`docker create`) 시점에 추가할 원시 옵션 목록 |
| `container.hostUsers` | `listOf str` | `[]` | 대화형 셸을 사용하는 계정명 목록. 지정 시 각 계정 하위에 `~/.hermes` 심볼릭 링크가 생성되며, `hermes` 파일 공유를 위한 그룹에 자동 가입 처리됩니다. |

---

## 디렉터리 구조

### 네이티브 모드 (Native Mode)

```
/var/lib/hermes/                     # stateDir (소유자 hermes:hermes, 권한 0750)
├── .hermes/                         # HERMES_HOME
│   ├── config.yaml                  # Nix에서 관리 및 병합되어 생성된 파일 (리빌드 시 자동 갱신)
│   ├── .managed                     # 마커 파일: CLI를 통한 설정 임의 조작 방지용
│   ├── .env                         # environment 및 environmentFiles가 병합된 환경 파일
│   ├── auth.json                    # OAuth 인증 관련 파일 (최초 주입 후 에이전트가 자체 관리)
│   ├── gateway.pid
│   ├── state.db
│   ├── mcp-tokens/                  # MCP 서버용 OAuth 토큰 저장 경로
│   ├── sessions/
│   ├── memories/
│   ├── skills/
│   ├── cron/
│   └── logs/
├── home/                            # 에이전트의 HOME 디렉터리
└── workspace/                       # 에이전트의 실질적 작업 공간 디렉터리
    ├── SOUL.md                      # documents 설정을 통해 준비된 파일
    └── (에이전트가 자체 생성한 파일 목록)
```

### 컨테이너 모드 (Container Mode)

컨테이너 안으로 호스트의 해당 물리 경로들이 다음과 같이 마운트됩니다:

| 컨테이너 내 경로 | 호스트 측 실제 경로 | 접근 모드 | 참고 정보 |
|---|---|---|---|
| `/nix/store` | `/nix/store` | `ro` (읽기 전용) | Hermes 실행 바이너리 및 Nix 의존 패키지 전체 |
| `/data` | `/var/lib/hermes` | `rw` (읽고 쓰기) | 서비스의 설정값, 데이터 상태, 에이전트 작업 공간 디렉터리 |
| `/home/hermes` | `${stateDir}/home` | `rw` (읽고 쓰기) | 에이전트 홈 환경 유지 — `pip install --user`, 도구 캐시 저장 경로 |
| `/usr`, `/usr/local`, `/tmp` | (컨테이너 내 쓰기 레이어) | `rw` (읽고 쓰기) | `apt`/`pip`/`npm` 패키지 설치 공간 — 컨테이너가 켜져 있는 동안 유지되며, 이미지 재생성 시 초기화됨 |

---

## 업데이트

```bash
# flake 입력 정보 갱신 (flake.nix 파일이 있는 경로에서 실행해야 합니다)
cd /etc/nixos && nix flake update hermes-agent

# 시스템 리빌드 및 전환
sudo nixos-rebuild switch
```

컨테이너 모드의 경우, 호스트에서 `current-package` 심볼릭 링크가 갱신되며, 에이전트 서비스가 재시작될 때 새로운 바이너리를 자동으로 불러옵니다. 이 과정에서 컨테이너 자체가 삭제 및 재생성되지 않으므로 임의로 설치했던 패키지들도 유실되지 않고 유지됩니다.

---

## 트러블슈팅

:::tip Podman 사용자
아래의 모든 `docker` 명령어들은 `podman` 환경에서도 명칭만 변경하여 완전히 동일하게 사용하실 수 있습니다. `container.backend = "podman"`으로 설정하셨다면 상황에 맞게 명령어를 바꿔서 실행해 주십시오.
:::

### 서비스 로그 확인

```bash
# 네이티브 및 컨테이너 모드 모두 동일한 systemd 유닛을 사용합니다
journalctl -u hermes-agent -f

# 컨테이너 모드의 경우 다음 명령으로도 직접 조회가 가능합니다
docker logs -f hermes-agent
```

### 컨테이너 상태 점검

```bash
systemctl status hermes-agent
docker ps -a --filter name=hermes-agent
docker inspect hermes-agent --format='{{.State.Status}}'
docker exec -it hermes-agent bash
docker exec hermes-agent readlink /data/current-package
docker exec hermes-agent cat /data/.container-identity
```

### 컨테이너 강제 초기화 및 재생성

쓰기 가능 레이어를 깨끗한 우분투 상태로 완전히 되돌리고자 하는 경우:

```bash
sudo systemctl stop hermes-agent
docker rm -f hermes-agent
sudo rm /var/lib/hermes/.container-identity
sudo systemctl start hermes-agent
```

### 비밀값 로드 여부 검증

에이전트는 시작되었으나 LLM 프로바이더 연동에 계속 실패하는 경우, `.env` 설정에 환경 변수값들이 정상적으로 로드되었는지 확인하십시오:

```bash
# 네이티브 모드
sudo -u hermes cat /var/lib/hermes/.hermes/.env

# 컨테이너 모드
docker exec hermes-agent cat /data/.hermes/.env
```

### GC Root 검증

```bash
nix-store --query --roots $(docker exec hermes-agent readlink /data/current-package)
```

### 자주 발생하는 문제들

| 증상 | 원인 | 조치 방법 |
|---|---|---|
| `Cannot save configuration: managed by NixOS` | CLI 가드 동작 중 | `configuration.nix` 파일을 직접 수정한 뒤 `nixos-rebuild switch` 명령을 실행하십시오. |
| `No adapter available for discord` (또는 telegram/slack) | 격리된 Nix venv 환경에 메시징 의존 패키지 누락 | `#messaging` 변형 패키지를 설치하십시오 (`nix profile install ...#messaging`). NixOS 모듈의 경우에는 `extraDependencyGroups = [ "messaging" ]` 설정을 추가하십시오. 자세한 원인 파악을 위해 `journalctl -u hermes-agent` 출력 로그의 `FeatureUnavailable` 또는 `requirements not met` 문구를 살펴보십시오. |
| 예기치 않은 컨테이너 재생성 발생 | `extraVolumes`, `extraOptions` 혹은 `image` 설정의 변경 | 설정이 변경되면 컨테이너가 다시 빌드되며 쓰기 가능 레이어가 초기화되는 것이 정상적인 동작입니다. 필요한 패키지를 컨테이너 기동 후 재설치하거나 직접 커스텀 이미지를 제작하여 빌드하십시오. |
| `hermes version`에 예전 버전 정보가 출력됨 | 컨테이너가 리빌드 시점에 재시작되지 않음 | `systemctl restart hermes-agent` 명령을 통해 재시작을 유도하십시오. |
| `/var/lib/hermes` 경로 접근 시 권한 오류 (Permission denied) | 상태 디렉터리의 권한이 `0750 hermes:hermes`로 차단됨 | `docker exec` 방식을 사용하거나 `sudo -u hermes` 권한을 대여하여 실행하십시오. |
| `nix-collect-garbage` 작동 이후 hermes가 삭제됨 | GC root 유실됨 | 서비스를 재시작하십시오 (preStart 단계에서 자동으로 GC root를 복구합니다). |
| `no container with name or ID "hermes-agent"` (Podman 사용 시) | 일반 사용자가 실행하여 rootful 컨테이너 정보를 조회하지 못함 | podman 명령어 호출 시 sudo 권한을 대여할 수 있도록 무암호 설정을 등록해 주십시오 ([컨테이너 모드](#컨테이너-인식-cli) 설명 섹션 참고). |
| `unable to find user hermes` | 컨테이너 부팅 단계 진행 중 (엔트리 포인트 프로세스가 아직 사용자를 미생성함) | 몇 초 후 다시 재시도하십시오 (CLI가 내부적으로 자동 재시도를 처리합니다). |
| `extraPackages`에 추가한 도구를 터미널 내부에서 찾을 수 없음 | per-user 프로필이 갱신되지 않음 | `nixos-rebuild switch` 실행을 완료한 뒤 서비스를 반드시 다시 기동해 주십시오: `nixos-rebuild switch && systemctl restart hermes-agent` |
