---
sidebar_position: 3
title: "Nix 및 NixOS 설정"
description: "Nix를 사용해 Hermes Agent 설치 및 배포하기 — 간단한 `nix run`부터 완전한 선언적 NixOS 모듈과 컨테이너 모드까지"
---

# Nix 및 NixOS 설정

:::warning Tier 2 플랫폼
Nix와 NixOS는 [Tier 2 플랫폼](./platform-support.md#tier-2)입니다. 여기서 설명하는 flake와 NixOS 모듈은 최선의 노력으로만 유지 관리됩니다. `main`에 커밋되면 언제든 이 패키지가 동작하지 않을 수 있습니다.

지원되는 설정을 사용하려면 표준 [설치](./installation.md) 경로인 Docker 또는 FHS 환경 중 하나를 이용하세요.
:::

Hermes Agent는 Nix flake와 NixOS 모듈을 제공합니다.

| 수준 | 대상 | 제공되는 것 |
|-------|-------------|--------------|
| **`nix run` / `nix profile install`** | 모든 Nix 사용자(macOS, Linux) | 모든 의존성이 포함된 사전 빌드 바이너리 — 이후 표준 CLI 워크플로 사용 |
| **NixOS 모듈(네이티브)** | NixOS 서버 배포 | 선언적 설정, 보안이 강화된 systemd 서비스, 관리되는 시크릿 |
| **NixOS 모듈(컨테이너)** | 자체 수정이 필요한 에이전트 | 위의 모든 기능과 함께, 에이전트가 `apt`/`pip`/`npm install`을 실행할 수 있는 영구 Ubuntu 컨테이너 |

:::info 표준 설치와 다른 점
`curl | bash` 설치 프로그램은 Python, Node 및 의존성을 직접 관리합니다. Nix flake는 이 모든 작업을 대체합니다. 모든 Python 의존성은 [uv2nix](https://github.com/pyproject-nix/uv2nix)가 빌드한 Nix derivation이며, 런타임 도구(Node.js, git, ripgrep, ffmpeg)는 바이너리의 PATH에 포함되도록 래핑됩니다. 런타임 pip도, venv 활성화도, `npm install`도 필요하지 않습니다.

**NixOS가 아닌 사용자**에게는 설치 단계만 달라집니다. 그 이후의 모든 과정(`hermes setup`, `hermes gateway install`, 설정 편집)은 표준 설치와 동일하게 작동합니다.

**NixOS 모듈 사용자**에게는 전체 수명 주기가 달라집니다. 설정은 `configuration.nix`에 저장하고, 시크릿은 sops-nix/agenix를 통해 전달하며, 서비스는 systemd 유닛이고, CLI 설정 명령은 차단됩니다. 다른 NixOS 서비스와 동일한 방식으로 hermes를 관리합니다.
:::

## 사전 요구 사항

- **flakes가 활성화된 Nix** — [Determinate Nix](https://install.determinate.systems) 권장(flakes가 기본으로 활성화됨)
- **사용하려는 서비스의 API 키**(최소 하나의 OpenRouter 또는 Anthropic 키)

---

## 빠른 시작(모든 Nix 사용자)

클론할 필요가 없습니다. Nix가 모든 것을 가져오고, 빌드하고, 실행합니다.

```bash
# Run the desktop app
nix run github:NousResearch/hermes-agent#desktop

# Or install persistently
nix profile install github:NousResearch/hermes-agent#desktop

# run the tui
nix run github:NousResearch/hermes-agent -- setup
nix run github:NousResearch/hermes-agent -- --tui

# or install it in your profile
nix profile install github:NousResearch/hermes-agent
hermes setup
hermes --tui
```

`nix profile install` 후에는 `hermes`, `hermes-agent`, `hermes-acp`가 PATH에 추가됩니다. 이제부터의 워크플로는 [표준 설치](./installation.md)와 동일합니다. `hermes setup`은 공급자 선택 과정을 안내하고, `hermes gateway install`은 launchd(macOS) 또는 systemd 사용자 서비스를 설정하며, 설정은 `~/.hermes/`에 저장됩니다.

:::warning 메시징 플랫폼(Discord, Telegram, Slack)
기본 패키지에는 hermes-agent에 필요할 수 있는 모든 라이브러리가 포함되어 있습니다. 더 작은 변형을 원한다면 다른 flake 출력을 확인하세요.

`default` 패키지는 closure에 약 700MB를 추가합니다. 메시징 플랫폼만 필요한 경우 `#messaging`을 사용하면 약 33MB만 추가됩니다.

:::

<details>
<summary><strong>로컬 클론에서 실행하기</strong></summary>

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
nix develop
hermes setup
```

</details>

---

## NixOS 모듈

flake는 `nixosModules.default`를 내보냅니다. 이는 사용자 생성, 디렉터리, 설정 생성, 시크릿, 문서 및 서비스 수명 주기를 선언적으로 관리하는 완전한 NixOS 서비스 모듈입니다.

:::note
이 모듈은 NixOS가 필요합니다. NixOS가 아닌 시스템(macOS, 기타 Linux 배포판)에서는 `nix profile install`과 위의 표준 CLI 워크플로를 사용하세요.
:::

### Flake 입력 추가

```nix
# /etc/nixos/flake.nix (or your system flake)
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

이것으로 충분합니다. `nixos-rebuild switch`가 `hermes` 사용자를 생성하고, `config.yaml`을 만들고, 시크릿을 연결하고, 게이트웨이를 시작합니다. 게이트웨이는 에이전트를 메시징 플랫폼(Telegram, Discord 등)에 연결하고 수신 메시지를 기다리는 장기 실행 서비스입니다.

:::warning 시크릿이 필요합니다
위의 `environmentFiles` 줄은 [sops-nix](https://github.com/Mic92/sops-nix) 또는 [agenix](https://github.com/ryantm/agenix)가 설정되어 있다고 가정합니다. 파일에는 최소 하나의 LLM 공급자 키(예: `OPENROUTER_API_KEY=sk-or-...`)가 있어야 합니다. 전체 설정은 [시크릿 관리](#secrets-management)를 참고하세요. 아직 시크릿 관리자가 없다면 일반 파일을 시작점으로 사용할 수 있습니다. 단, 다른 사용자가 읽을 수 없도록 하세요.

```bash
echo "OPENROUTER_API_KEY=sk-or-your-key" | sudo install -m 0600 -o hermes /dev/stdin /var/lib/hermes/env
```

```nix
services.hermes-agent.environmentFiles = [ "/var/lib/hermes/env" ];
```
:::

:::tip addToSystemPackages
`addToSystemPackages = true`로 설정하면 두 가지 작업이 수행됩니다. `hermes` CLI를 시스템 PATH에 추가하고, 시스템 전체에 `HERMES_HOME`을 설정하여 대화형 CLI가 게이트웨이 서비스와 상태(세션, 스킬, cron)를 공유하게 합니다. 이 설정이 없으면 셸에서 `hermes`를 실행할 때 별도의 `~/.hermes/` 디렉터리가 생성됩니다.
:::

### 컨테이너 인식 CLI

:::info
`container.enable = true`와 `addToSystemPackages = true`가 설정되면 호스트에서 실행하는 **모든** `hermes` 명령이 자동으로 관리되는 컨테이너로 라우팅됩니다. 즉, 대화형 CLI 세션이 게이트웨이 서비스와 동일한 환경에서 실행되며, 컨테이너에 설치된 모든 패키지와 도구에 접근할 수 있습니다.

- 라우팅은 투명하게 이루어집니다. `hermes chat`, `hermes sessions list`, `hermes version` 등은 모두 내부적으로 컨테이너에서 실행됩니다.
- 모든 CLI 플래그는 그대로 전달됩니다.
- 컨테이너가 실행 중이 아니면 CLI가 잠시 재시도합니다(대화형 사용은 스피너와 함께 5초, 스크립트는 조용히 10초). 이후 명확한 오류와 함께 실패하며, 조용히 폴백하지 않습니다.
- hermes 코드베이스에서 작업하는 개발자는 `HERMES_DEV=1`을 설정하여 컨테이너 라우팅을 우회하고 로컬 체크아웃을 직접 실행할 수 있습니다.

`container.hostUsers`를 설정하면 서비스 상태 디렉터리에 대한 `~/.hermes` 심볼릭 링크가 생성되어 호스트 CLI와 컨테이너가 세션, 설정 및 메모리를 공유합니다.

```nix
services.hermes-agent = {
  container.enable = true;
  container.hostUsers = [ "your-username" ];
  addToSystemPackages = true;
};
```

`hostUsers`에 등록된 사용자는 파일 권한에 접근할 수 있도록 자동으로 `hermes` 그룹에 추가됩니다.

**Podman 사용자:** NixOS 서비스는 컨테이너를 root로 실행합니다. Docker 사용자는 `docker` 그룹 소켓을 통해 접근하지만, Podman의 rootful 컨테이너에는 sudo가 필요합니다. 컨테이너 런타임에 대해 비밀번호 없는 sudo를 허용하세요.

```nix
security.sudo.extraRules = [{
  users = [ "your-username" ];
  commands = [{
    command = "/run/current-system/sw/bin/podman";
    options = [ "NOPASSWD" ];
  }];
}];
```

CLI는 sudo가 필요한 경우를 자동으로 감지하고 투명하게 사용합니다. 이 설정이 없으면 직접 `sudo hermes chat`을 실행해야 합니다.
:::

### 작동 여부 확인

`nixos-rebuild switch` 후 서비스가 실행 중인지 확인하세요.

```bash
# Check service status
systemctl status hermes-agent

# Watch logs (Ctrl+C to stop)
journalctl -u hermes-agent -f

# If addToSystemPackages is true, test the CLI
hermes version
hermes config       # shows the generated config
```

### 배포 모드 선택

이 모듈은 `container.enable`로 제어되는 두 가지 모드를 지원합니다.

| | **네이티브**(기본값) | **컨테이너** |
|---|---|---|
| 실행 방식 | 호스트에서 보안이 강화된 systemd 서비스 | `/nix/store`가 바인드 마운트된 영구 Ubuntu 컨테이너 |
| 보안 | `NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp` | 컨테이너 격리, 내부에서 권한이 없는 사용자로 실행 |
| 에이전트의 패키지 자체 설치 | 불가 — Nix가 제공하는 PATH의 도구만 사용 | 가능 — `apt`, `pip`, `npm` 설치가 재시작 후에도 유지됨 |
| 설정 범위 | 동일 | 동일 |
| 선택 시점 | 표준 배포, 최대 보안, 재현성 | 에이전트에 런타임 패키지 설치, 변경 가능한 환경, 실험적 도구가 필요한 경우 |

컨테이너 모드를 활성화하려면 한 줄을 추가하세요.

```nix
{
  services.hermes-agent = {
    enable = true;
    container.enable = true;
    # ... rest of config is identical
  };
}
```

:::info
컨테이너 모드는 `mkDefault`를 통해 `virtualisation.docker.enable`을 자동으로 활성화합니다. 대신 Podman을 사용한다면 `container.backend = "podman"`으로 설정하고 `virtualisation.docker.enable = false`로 지정하세요.
:::

---

## 설정

### 선언적 설정

`settings` 옵션은 `config.yaml`로 렌더링되는 임의의 attrset을 받습니다. 여러 모듈 정의에 걸쳐 깊은 병합을 지원하므로(`lib.recursiveUpdate` 사용), 여러 파일로 설정을 나눌 수 있습니다.

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

두 설정은 평가 시점에 깊게 병합됩니다. Nix로 선언한 키는 디스크에 이미 존재하는 `config.yaml`의 키보다 항상 우선하지만, **Nix가 건드리지 않는 사용자 추가 키는 보존됩니다**. 따라서 에이전트 또는 수동 편집으로 `skills.disabled`나 `streaming.enabled` 같은 키를 추가해도 `nixos-rebuild switch` 후에 유지됩니다.

:::note 모델 이름 지정
`settings.model.default`에는 공급자가 요구하는 모델 식별자를 사용합니다. 기본 공급자인 [OpenRouter](https://openrouter.ai)의 경우 `"anthropic/claude-sonnet-4"` 또는 `"google/gemini-3-flash"`와 같은 형식입니다. 공급자를 직접 사용하는 경우(Anthropic, OpenAI) `settings.model.base_url`을 해당 API를 가리키도록 설정하고 네이티브 모델 ID(예: `"claude-sonnet-4-20250514"`)를 사용하세요. `base_url`을 설정하지 않으면 Hermes는 OpenRouter를 기본값으로 사용합니다.
:::

:::tip 사용 가능한 설정 키 확인
`nix build .#configKeys && cat result`를 실행하면 Python의 `DEFAULT_CONFIG`에서 추출한 모든 말단 설정 키를 확인할 수 있습니다. 기존 `config.yaml`을 `settings` attrset에 붙여 넣을 수 있으며, 구조는 1:1로 대응합니다.
:::

<details>
<summary><strong>전체 예시: 자주 사용자 지정하는 모든 설정</strong></summary>

```nix
{ config, ... }: {
  services.hermes-agent = {
    enable = true;
    container.enable = true;

    # ── Model ──────────────────────────────────────────────────────────
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

    # ── Secrets ────────────────────────────────────────────────────────
    environmentFiles = [ config.sops.secrets."hermes-env".path ];

    # ── Documents ──────────────────────────────────────────────────────
    documents = {
      "USER.md" = ./documents/USER.md;
    };

    # ── MCP Servers ────────────────────────────────────────────────────
    mcpServers.filesystem = {
      command = "npx";
      args = [ "-y" "@modelcontextprotocol/server-filesystem" "/data/workspace" ];
    };

    # ── Container options ──────────────────────────────────────────────
    container = {
      image = "ubuntu:24.04";
      backend = "docker";
      hostUsers = [ "your-username" ];
      extraVolumes = [ "/home/user/projects:/projects:rw" ];
      extraOptions = [ "--gpus" "all" ];
    };

    # ── Service tuning ─────────────────────────────────────────────────
    addToSystemPackages = true;
    extraArgs = [ "--verbose" ];
    restart = "always";
    restartSec = 5;
  };
}
```

</details>

### 다른 설정 관리 방식: 직접 config.yaml 관리

Nix 외부에서 `config.yaml`을 직접 관리하고 싶다면 `configFile`을 사용하세요:

```nix
services.hermes-agent.configFile = /etc/hermes/config.yaml;
```

이 옵션은 `settings`를 완전히 우회합니다 — 병합이나 생성이 일어나지 않습니다. 파일은 활성화할 때마다 있는 그대로 `$HERMES_HOME/config.yaml`에 복사됩니다.

### 사용자 지정 빠른 참조

Nix 사용자가 가장 자주 변경하는 항목을 한눈에 확인할 수 있습니다:

| 변경하려는 항목 | 옵션 | 예시 |
|---|---|---|
| LLM 모델 변경 | `settings.model.default` | `"anthropic/claude-sonnet-4"` |
| 다른 provider 엔드포인트 사용 | `settings.model.base_url` | `"https://openrouter.ai/api/v1"` |
| API 키 추가 | `environmentFiles` | `[ config.sops.secrets."hermes-env".path ]` |
| 에이전트에 개성 부여 | `${services.hermes-agent.stateDir}/.hermes/SOUL.md` | 파일을 직접 관리 |
| MCP 도구 서버 추가 | `mcpServers.<name>` | [MCP 서버](#mcp-servers) 참고 |
| Discord/Telegram/Slack 활성화 | `extraDependencyGroups` | `[ "messaging" ]` |
| 호스트 디렉터리를 컨테이너에 마운트 | `container.extraVolumes` | `[ "/data:/data:rw" ]` |
| GPU 액세스를 컨테이너에 전달 | `container.extraOptions` | `[ "--gpus" "all" ]` |
| Docker 대신 Podman 사용 | `container.backend` | `"podman"` |
| 호스트 CLI와 컨테이너 간 상태 공유 | `container.hostUsers` | `[ "sidbin" ]` |
| 에이전트에서 추가 도구 사용 가능하게 설정 | `extraPackages` | `[ pkgs.pandoc pkgs.imagemagick ]` |
| 사용자 지정 base image 사용 | `container.image` | `"ubuntu:24.04"` |
| hermes 패키지 재정의 | `package` | `inputs.hermes-agent.packages.${system}.default.override { ... }` |
| 상태 디렉터리 변경 | `stateDir` | `"/opt/hermes"` |
| 에이전트의 작업 디렉터리 설정 | `workingDirectory` | `"/home/user/projects"` |

---

## 시크릿 관리

:::danger API 키를 `settings` 또는 `environment`에 절대 넣지 마세요
Nix 표현식의 값은 `/nix/store`에 저장되어 누구나 읽을 수 있습니다. 시크릿 관리자를 사용하는 `environmentFiles`를 항상 사용하세요.
:::

`environment`(시크릿이 아닌 변수)와 `environmentFiles`(시크릿 파일)는 활성화 시점(`nixos-rebuild switch`)에 `$HERMES_HOME/.env`로 병합됩니다. Hermes는 시작할 때마다 이 파일을 읽으므로, `systemctl restart hermes-agent`만 실행하면 컨테이너를 다시 만들지 않고도 변경 사항이 적용됩니다.

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

시크릿 파일에는 키-값 쌍이 들어 있습니다:

```yaml
# secrets/hermes.yaml (encrypted with sops)
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

### OAuth / 인증 정보 시드

OAuth가 필요한 플랫폼(예: Discord)의 경우 `authFile`을 사용해 최초 배포 시 인증 정보를 미리 넣으세요:

```nix
{
  services.hermes-agent = {
    authFile = config.sops.secrets."hermes/auth.json".path;
    # authFileForceOverwrite = true;  # overwrite on every activation
  };
}
```

`auth.json`이 아직 없을 때만 파일이 복사됩니다(`authFileForceOverwrite = true`인 경우에는 예외). 실행 중 OAuth 토큰 갱신은 상태 디렉터리에 기록되며 rebuild를 실행해도 유지됩니다.

---

## 문서

`documents` 옵션은 에이전트의 작업 디렉터리(`workingDirectory`, 에이전트가 workspace로 읽는 위치)에 파일을 설치합니다. Hermes는 관례에 따라 다음과 같은 특정 파일 이름을 찾습니다:

- **`USER.md`** — 에이전트가 상호작용하는 사용자에 대한 컨텍스트입니다.
- 여기에 넣은 다른 파일은 모두 에이전트가 workspace 파일로 볼 수 있습니다.

에이전트 identity 파일은 별개입니다. Hermes는 `$HERMES_HOME/SOUL.md`를 주된 `SOUL.md`로 로드하며, NixOS 모듈에서는 `${services.hermes-agent.stateDir}/.hermes/SOUL.md`에 해당합니다. `documents`에 `SOUL.md`를 넣으면 workspace 파일만 생성되며, 주된 persona 파일을 대체하지 않습니다.

```nix
{
  services.hermes-agent.documents = {
    "USER.md" = ./documents/USER.md;  # path reference, copied from Nix store
  };
}
```

값은 인라인 문자열이나 경로 참조로 지정할 수 있습니다. 파일은 모든 `nixos-rebuild switch` 실행 시 설치됩니다.

---

## MCP 서버

`mcpServers` 옵션은 [MCP (Model Context Protocol)](https://modelcontextprotocol.io) 서버를 선언적으로 구성합니다. 각 서버는 **stdio**(로컬 명령) 또는 **HTTP**(원격 URL) 전송을 사용합니다.

### Stdio 전송(로컬 서버)

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
      env.GITHUB_PERSONAL_ACCESS_TOKEN = "\${GITHUB_TOKEN}"; # resolved from .env
    };
  };
}
```

:::tip
`env` 값의 환경 변수는 런타임에 `$HERMES_HOME/.env`에서 확인됩니다. 시크릿을 주입하려면 `environmentFiles`를 사용하세요 — 토큰을 Nix config에 직접 넣지 마세요.
:::

### HTTP 전송(원격 서버)

```nix
{
  services.hermes-agent.mcpServers.remote-api = {
    url = "https://mcp.example.com/v1/mcp";
    headers.Authorization = "Bearer \${MCP_REMOTE_API_KEY}";
    timeout = 180;
  };
}
```

### OAuth를 사용하는 HTTP 전송

OAuth 2.1을 사용하는 서버에는 `auth = "oauth"`를 설정하세요. Hermes는 메타데이터 탐색, 동적 클라이언트 등록, 토큰 교환, 자동 갱신을 포함한 전체 PKCE 흐름을 구현합니다.

```nix
{
  services.hermes-agent.mcpServers.my-oauth-server = {
    url = "https://mcp.example.com/mcp";
    auth = "oauth";
  };
}
```

토큰은 `$HERMES_HOME/mcp-tokens/<server-name>.json`에 저장되며 재시작과 rebuild 후에도 유지됩니다.

<details>
<summary><strong>헤드리스 서버에서 최초 OAuth 인증</strong></summary>

최초 OAuth 인증에는 브라우저 기반 동의 흐름이 필요합니다. 헤드리스 배포에서는 Hermes가 브라우저를 여는 대신 인증 URL을 stdout/logs에 출력합니다.

**옵션 A: 대화형 부트스트랩** — `docker exec`(컨테이너) 또는 `sudo -u hermes`(네이티브)를 통해 한 번 흐름을 실행하세요:

```bash
# Container mode
docker exec -it hermes-agent \
  hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth

# Native mode
sudo -u hermes HERMES_HOME=/var/lib/hermes/.hermes \
  hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth
```

컨테이너는 `--network=host`를 사용하므로 `127.0.0.1`의 OAuth 콜백 리스너에 호스트 브라우저에서 접근할 수 있습니다.

**옵션 B: 토큰 미리 넣기** — 워크스테이션에서 흐름을 완료한 다음 토큰을 복사하세요:

```bash
hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth
scp ~/.hermes/mcp-tokens/my-oauth-server{,.client}.json \
    server:/var/lib/hermes/.hermes/mcp-tokens/
# Ensure: chown hermes:hermes, chmod 0600
```

</details>

### 샘플링(서버가 시작하는 LLM 요청)

일부 MCP 서버는 에이전트에 LLM completion을 요청할 수 있습니다:

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

## 관리 모드

hermes가 NixOS 모듈을 통해 실행될 때 다음 CLI 명령은 `configuration.nix`를 가리키는 설명 오류와 함께 **차단됩니다**:

| 차단된 명령 | 이유 |
|---|---|
| `hermes setup` | 설정이 선언적이므로 Nix config의 `settings`를 편집해야 합니다 |
| `hermes config edit` | 설정이 `settings`에서 생성됩니다 |
| `hermes config set <key> <value>` | 설정이 `settings`에서 생성됩니다 |
| `hermes gateway install` | systemd 서비스가 NixOS에서 관리됩니다 |
| `hermes gateway uninstall` | systemd 서비스가 NixOS에서 관리됩니다 |

이는 Nix에서 선언한 내용과 디스크에 있는 내용이 서로 달라지는 것을 방지합니다. 감지에는 다음 두 신호가 사용됩니다:

1. **`HERMES_MANAGED=true` 환경 변수** — systemd 서비스가 설정하며 gateway 프로세스에서 확인할 수 있습니다.
2. **`HERMES_HOME`의 `.managed` 마커 파일** — activation script가 설정하며 대화형 셸에서도 확인할 수 있습니다(예: `docker exec -it hermes-agent hermes config set ...`도 차단됩니다).

설정을 변경하려면 Nix config를 편집한 뒤 `sudo nixos-rebuild switch`를 실행하세요.

---

## 컨테이너 아키텍처

:::info
이 섹션은 `container.enable = true`를 사용하는 경우에만 해당합니다. 네이티브 모드 배포에서는 건너뛰세요.
:::

컨테이너 모드가 활성화되면 hermes는 영구 Ubuntu 컨테이너 내부에서 실행되며, Nix로 빌드한 바이너리는 호스트에서 읽기 전용으로 bind mount됩니다:

```
Host                                    Container
────                                    ─────────
/nix/store/...-hermes-agent-0.1.0  ──►  /nix/store/... (ro)
~/.hermes -> /var/lib/hermes/.hermes       (symlink bridge, per hostUsers)
/var/lib/hermes/                    ──►  /data/          (rw)
  ├── current-package -> /nix/store/...    (symlink, updated each rebuild)
  ├── .gc-root -> /nix/store/...           (prevents nix-collect-garbage)
  ├── .container-identity                  (sha256 hash, triggers recreation)
  ├── .hermes/                             (HERMES_HOME)
  │   ├── .env                             (merged from environment + environmentFiles)
  │   ├── config.yaml                      (Nix-generated, deep-merged by activation)
  │   ├── .managed                         (marker file)
  │   ├── .container-mode                  (routing metadata: backend, exec_user, etc.)
  │   ├── state.db, sessions/, memories/   (runtime state)
  │   └── mcp-tokens/                      (OAuth tokens for MCP servers)
  ├── home/                                ──►  /home/hermes    (rw)
  └── workspace/                           (agent working directory)
      ├── SOUL.md                          (from documents option)
      └── (agent-created files)

Container writable layer (apt/pip/npm):   /usr, /usr/local, /tmp
```

Nix로 빌드한 바이너리는 `/nix/store`가 bind mount되기 때문에 Ubuntu 컨테이너 내부에서 작동합니다 — 자체 인터프리터와 모든 종속성을 함께 가져오므로 컨테이너의 시스템 라이브러리에 의존하지 않습니다. 컨테이너 진입점은 `current-package` symlink를 통해 확인됩니다: `/data/current-package/bin/hermes gateway run --replace`. `nixos-rebuild switch`를 실행하면 symlink만 업데이트되고 컨테이너는 계속 실행됩니다.

### 어떤 상황에서 무엇이 유지되는가

| 이벤트 | 컨테이너 재생성 여부 | `/data` (상태) | `/home/hermes` | 쓰기 가능 레이어(`apt`/`pip`/`npm`) |
|---|---|---|---|---|
| `systemctl restart hermes-agent` | 아니요 | 유지됨 | 유지됨 | 유지됨 |
| `nixos-rebuild switch` (코드 변경) | 아니요 (symlink 업데이트) | 유지됨 | 유지됨 | 유지됨 |
| 호스트 재부팅 | 아니요 | 유지됨 | 유지됨 | 유지됨 |
| `nix-collect-garbage` | 아니요 (GC root) | 유지됨 | 유지됨 | 유지됨 |
| 이미지 변경(`container.image`) | **예** | 유지됨 | 유지됨 | **손실됨** |
| 볼륨/옵션 변경 | **예** | 유지됨 | 유지됨 | **손실됨** |
| `environment`/`environmentFiles` 변경 | 아니요 | 유지됨 | 유지됨 | 유지됨 |

컨테이너는 **identity hash**가 변경될 때만 재생성됩니다. 이 해시에는 schema version, image, `extraVolumes`, `extraOptions`, entrypoint script가 포함됩니다. 환경 변수, settings, documents 또는 hermes package의 변경은 재생성을 유발하지 않습니다.

:::warning 쓰기 가능 레이어 손실
identity hash가 변경되면(이미지 업그레이드, 새 볼륨, 새 컨테이너 옵션) 컨테이너가 삭제되고 `container.image`를 새로 pull하여 재생성됩니다. 쓰기 가능 레이어에 있는 `apt install`, `pip install`, `npm install` 패키지는 모두 손실됩니다. `/data`와 `/home/hermes`의 상태는 유지됩니다(이 경로들은 bind mount입니다).

에이전트가 특정 패키지에 의존한다면 사용자 지정 이미지(`container.image = "my-registry/hermes-base:latest"`)에 패키지를 포함하거나, 에이전트의 SOUL.md에서 설치를 스크립트로 실행하는 방법을 고려하세요.
:::
### GC Root 보호

`preStart` 스크립트는 `${stateDir}/.gc-root`에 현재 hermes 패키지를 가리키는 GC 루트를 생성합니다. 이를 통해 실행 중인 바이너리가 `nix-collect-garbage`에 의해 제거되지 않도록 합니다. GC 루트가 손상되면 서비스를 다시 시작할 때 다시 생성됩니다.

---

## 플러그인

NixOS 모듈은 선언적 플러그인 설치를 지원하므로 명령형 `hermes plugins install`을 사용할 필요가 없습니다.

### 디렉터리 플러그인(`extraPlugins`)

`plugin.yaml`과 `__init__.py`가 있는 소스 트리로만 구성된 플러그인(예: [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm))의 경우:

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

활성화 시 플러그인이 `$HERMES_HOME/plugins/`에 심볼릭 링크됩니다. Hermes는 일반 디렉터리 검색을 통해 플러그인을 탐색합니다. 목록에서 플러그인을 제거하고 `nixos-rebuild switch`를 실행하면 심볼릭 링크가 제거됩니다.

### 엔트리 포인트 플러그인(`extraPythonPackages`)

`[project.entry-points."hermes_agent.plugins"]`를 통해 등록되는 pip 패키지 플러그인(예: [rtk-hermes](https://github.com/ogallotti/rtk-hermes))의 경우:

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

패키지의 `site-packages`는 hermes 래퍼의 PYTHONPATH에 추가됩니다. `importlib.metadata`가 세션 시작 시 엔트리 포인트를 탐색합니다.

### 선택적 의존성 그룹(`extraDependencyGroups`)

hermes-agent의 `pyproject.toml`에 선언된 선택적 extra를 포함하려면 `extraDependencyGroups`를 사용해 빌드 시 봉인된 venv에 포함하세요. 기본 `[all]` 세트에 포함되지 않은 모든 extra에 필요합니다. Nix에서는 읽기 전용 스토어에 런타임 설치를 할 수 없기 때문입니다.

```nix
# Enable Discord, Telegram, Slack
services.hermes-agent.extraDependencyGroups = [ "messaging" ];
```

```nix
# Enable a memory provider
services.hermes-agent = {
  extraDependencyGroups = [ "hindsight" ];
  settings.memory.provider = "hindsight";
};
```

이는 핵심 의존성과 함께 uv로 해결되므로 PYTHONPATH 패치나 충돌 위험이 없습니다. 사용 가능한 그룹은 다음과 같습니다.

| 그룹 | 활성화되는 기능 |
|------|-----------------|
| `messaging` | Discord, Telegram, Slack |
| `matrix` | Matrix/Element(암호화를 사용하는 mautrix; Linux 전용) |
| `dingtalk` | DingTalk |
| `feishu` | Feishu/Lark |
| `voice` | 로컬 음성-텍스트 변환(faster-whisper) |
| `edge-tts` | Edge TTS 제공자 |
| `tts-premium` | ElevenLabs TTS |
| `anthropic` | 기본 Anthropic SDK(OpenRouter를 사용하는 경우 불필요) |
| `bedrock` | AWS Bedrock(boto3) |
| `azure-identity` | Azure Entra ID 인증 |
| `honcho` | Honcho 메모리 제공자 |
| `hindsight` | Hindsight 메모리 제공자 |
| `modal` | Modal 터미널 백엔드 |
| `daytona` | Daytona 터미널 백엔드 |
| `exa` | Exa 웹 검색 |
| `firecrawl` | Firecrawl 웹 검색 |
| `fal` | FAL 이미지 생성 |

또는 extra별 설정 대신 미리 빌드된 `#messaging` 또는 `#full` flake 패키지를 사용하세요([빠른 시작](#quick-start-any-nix-user) 참조).

**어떤 것을 사용할지:**

| 필요 사항 | 옵션 |
|------|--------|
| pyproject.toml 선택적 extra 활성화 | `extraDependencyGroups` |
| pyproject.toml에 없는 외부 Python 플러그인 추가 | `extraPythonPackages` |
| 시스템 바이너리 추가(pandoc, jq 등) | `extraPackages` |
| 디렉터리 기반 플러그인 소스 트리 추가 | `extraPlugins` |

### 두 옵션 함께 사용하기

서드파티 Python 의존성이 필요한 디렉터리 플러그인에는 두 옵션이 모두 필요합니다.

```nix
services.hermes-agent = {
  extraPlugins = [ my-plugin-src ];          # plugin source
  extraPythonPackages = [ pkgs.python312Packages.redis ];  # its Python dep
  extraPackages = [ pkgs.redis ];            # system binary it needs
};
```

### 오버레이 사용하기

외부 flakes는 패키지를 직접 재정의할 수 있습니다.

```nix
{
  inputs.hermes-agent.url = "github:NousResearch/hermes-agent";
  outputs = { hermes-agent, nixpkgs, ... }: {
    nixpkgs.overlays = [ hermes-agent.overlays.default ];
    # Then:
    #   pkgs.hermes-agent.override { extraPythonPackages = [...]; }
    #   pkgs.hermes-agent.override { extraDependencyGroups = [ "hindsight" ]; }
  };
}
```

### 플러그인 구성

플러그인은 여전히 `config.yaml`에서 활성화해야 합니다. 선언적 설정을 통해 다음과 같이 추가하세요.

```nix
services.hermes-agent.settings.plugins.enabled = [
  "hermes-lcm"
  "rtk-rewrite"
];
```

:::note
빌드 시 충돌 검사를 통해 플러그인 패키지가 hermes 핵심 의존성을 가리지 않도록 합니다. 플러그인이 봉인된 venv에 이미 있는 패키지를 제공하면 `nixos-rebuild`가 명확한 오류와 함께 실패합니다.
:::

---

## 개발

### 개발 셸

flake는 Python 3.12, uv, Node.js 및 모든 런타임 도구가 포함된 개발 셸을 제공합니다.

```bash
cd hermes-agent
nix develop

# Shell provides:
#   - Python 3.12 + uv (deps installed into .venv on first entry)
#   - Node.js 26, ripgrep, git, openssh, ffmpeg on PATH
#   - Stamp-file optimization: re-entry is near-instant if deps haven't changed

hermes setup
hermes chat
```

### direnv(권장)

포함된 `.envrc`는 개발 셸을 자동으로 활성화합니다.

```bash
cd hermes-agent
direnv allow    # one-time
# Subsequent entries are near-instant (stamp file skips dep install)
```

### Flake 검사

flake에는 CI와 로컬에서 실행되는 빌드 시 검증이 포함되어 있습니다.

```bash
# Run all checks
nix flake check

# Individual checks
nix build .#checks.x86_64-linux.package-contents   # binaries exist + version
nix build .#checks.x86_64-linux.entry-points-sync  # pyproject.toml ↔ Nix package sync
nix build .#checks.x86_64-linux.cli-commands        # gateway/config subcommands
nix build .#checks.x86_64-linux.managed-guard       # HERMES_MANAGED blocks mutation
nix build .#checks.x86_64-linux.bundled-skills      # skills present in package
nix build .#checks.x86_64-linux.config-roundtrip    # merge script preserves user keys
```

<details>
<summary><strong>각 검사가 확인하는 내용</strong></summary>

| 검사 | 테스트 내용 |
|---|---|
| `package-contents` | `hermes`와 `hermes-agent` 바이너리가 존재하고 `hermes version`이 실행되는지 확인 |
| `entry-points-sync` | `pyproject.toml`의 모든 `[project.scripts]` 항목에 Nix 패키지의 래핑된 바이너리가 있는지 확인 |
| `cli-commands` | `hermes --help`가 `gateway` 및 `config` 하위 명령을 노출하는지 확인 |
| `managed-guard` | `HERMES_MANAGED=true hermes config set ...`이 NixOS 오류를 출력하는지 확인 |
| `bundled-skills` | skills 디렉터리가 존재하고 SKILL.md 파일을 포함하며 래퍼에 `HERMES_BUNDLED_SKILLS`가 설정되는지 확인 |
| `config-roundtrip` | 7가지 병합 시나리오를 확인: 새로 설치, Nix 재정의, 사용자 키 보존, 혼합 병합, MCP 추가 병합, 중첩된 깊은 병합, 멱등성 |

</details>

---

## 옵션 참조

### 핵심

| 옵션 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `enable` | `bool` | `false` | hermes-agent 서비스 활성화 |
| `package` | `package` | `hermes-agent` | 사용할 hermes-agent 패키지 |
| `user` | `str` | `"hermes"` | 시스템 사용자 |
| `group` | `str` | `"hermes"` | 시스템 그룹 |
| `createUser` | `bool` | `true` | 사용자/그룹 자동 생성 |
| `stateDir` | `str` | `"/var/lib/hermes"` | 상태 디렉터리(`HERMES_HOME`의 상위 디렉터리) |
| `workingDirectory` | `str` | `"${stateDir}/workspace"` | 에이전트 작업 디렉터리 |
| `addToSystemPackages` | `bool` | `false` | 시스템 PATH에 `hermes` CLI를 추가하고 시스템 전체에 `HERMES_HOME` 설정 |

### 구성

| 옵션 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `settings` | `attrs`(깊은 병합) | `{}` | `config.yaml`로 렌더링되는 선언적 구성. 임의의 중첩을 지원하며 여러 정의는 `lib.recursiveUpdate`로 병합 |
| `configFile` | `null` 또는 `path` | `null` | 기존 `config.yaml`의 경로. 설정된 경우 `settings` 전체를 재정의 |

### 비밀 및 환경

| 옵션 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `environmentFiles` | `listOf str` | `[]` | 비밀이 포함된 env 파일의 경로. 활성화 시 `$HERMES_HOME/.env`에 병합 |
| `environment` | `attrsOf str` | `{}` | 비밀이 아닌 환경 변수. **Nix 스토어에 노출됨** — 여기에 비밀을 넣지 마세요 |
| `authFile` | `null` 또는 `path` | `null` | OAuth 자격 증명 시드. 최초 배포 시에만 복사 |
| `authFileForceOverwrite` | `bool` | `false` | 활성화할 때마다 `authFile`에서 `auth.json`을 덮어씀 |

### 문서

| 옵션 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `documents` | `attrsOf (either str path)` | `{}` | 작업 공간 파일. 키는 파일 이름이고 값은 인라인 문자열 또는 경로입니다. 활성화 시 `workingDirectory`에 설치 |

### MCP 서버

| 옵션 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `mcpServers` | `attrsOf submodule` | `{}` | MCP 서버 정의. `settings.mcp_servers`에 병합 |
| `mcpServers.<name>.command` | `null` 또는 `str` | `null` | 서버 명령(stdio 전송) |
| `mcpServers.<name>.args` | `listOf str` | `[]` | 명령 인수 |
| `mcpServers.<name>.env` | `attrsOf str` | `{}` | 서버 프로세스의 환경 변수 |
| `mcpServers.<name>.url` | `null` 또는 `str` | `null` | 서버 엔드포인트 URL(HTTP/StreamableHTTP 전송) |
| `mcpServers.<name>.headers` | `attrsOf str` | `{}` | HTTP 헤더(예: `Authorization`) |
| `mcpServers.<name>.auth` | `null` 또는 `"oauth"` | `null` | 인증 방식. `"oauth"`는 OAuth 2.1 PKCE를 활성화 |
| `mcpServers.<name>.enabled` | `bool` | `true` | 이 서버 활성화 또는 비활성화 |
| `mcpServers.<name>.timeout` | `null` 또는 `int` | `null` | 도구 호출 제한 시간(초, 기본값: 120) |
| `mcpServers.<name>.connect_timeout` | `null` 또는 `int` | `null` | 연결 제한 시간(초, 기본값: 60) |
| `mcpServers.<name>.tools` | `null` 또는 `submodule` | `null` | 도구 필터링(`include`/`exclude` 목록) |
| `mcpServers.<name>.sampling` | `null` 또는 `submodule` | `null` | 서버가 시작한 LLM 요청의 샘플링 구성 |

### 서비스 동작

| 옵션 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `extraArgs` | `listOf str` | `[]` | `hermes gateway`의 추가 인수 |
| `extraPackages` | `listOf package` | `[]` | 에이전트가 사용할 수 있는 추가 패키지. hermes 사용자의 사용자별 프로필에 추가되므로 터미널 명령, 스킬, cron 작업에서 모두 확인 가능 |
| `extraPlugins` | `listOf package` | `[]` | `$HERMES_HOME/plugins/`에 심볼릭 링크할 디렉터리 플러그인 패키지. 각각 `plugin.yaml`을 포함해야 함 |
| `extraPythonPackages` | `listOf package` | `[]` | 엔트리 포인트 플러그인 탐색을 위해 PYTHONPATH에 추가되는 Python 패키지. `python312Packages`로 빌드 |
| `extraDependencyGroups` | `listOf str` | `[]` | 봉인된 venv에 포함할 pyproject.toml 선택적 extra(예: `["hindsight"]`). uv로 해결되며 충돌 없음 |
| `restart` | `str` | `"always"` | systemd `Restart=` 정책 |
| `restartSec` | `int` | `5` | systemd `RestartSec=` 값 |

### 컨테이너

| 옵션 | 유형 | 기본값 | 설명 |
|---|---|---|---|
| `container.enable` | `bool` | `false` | OCI 컨테이너 모드 활성화 |
| `container.backend` | `enum ["docker" "podman"]` | `"docker"` | 컨테이너 런타임 |
| `container.image` | `str` | `"ubuntu:24.04"` | 기본 이미지(런타임에 가져옴) |
| `container.extraVolumes` | `listOf str` | `[]` | 추가 볼륨 마운트(`host:container:mode`) |
| `container.extraOptions` | `listOf str` | `[]` | `docker create`에 전달되는 추가 인수 |
| `container.hostUsers` | `listOf str` | `[]` | 서비스 `stateDir`로 연결되는 `~/.hermes` 심볼릭 링크를 받고 `hermes` 그룹에 자동 추가되는 대화형 사용자 |

---

## 디렉터리 구성

### 네이티브 모드

```
/var/lib/hermes/                     # stateDir (owned by hermes:hermes, 0750)
├── .hermes/                         # HERMES_HOME
│   ├── config.yaml                  # Nix-generated (deep-merged each rebuild)
│   ├── .managed                     # Marker: CLI config mutation blocked
│   ├── .env                         # Merged from environment + environmentFiles
│   ├── auth.json                    # OAuth credentials (seeded, then self-managed)
│   ├── gateway.pid
│   ├── state.db
│   ├── mcp-tokens/                  # OAuth tokens for MCP servers
│   ├── sessions/
│   ├── memories/
│   ├── skills/
│   ├── cron/
│   └── logs/
├── home/                            # Agent HOME
└── workspace/                       # Agent working directory
    ├── SOUL.md                      # From documents option
    └── (agent-created files)
```
### 컨테이너 모드

동일한 레이아웃이며, 컨테이너에 마운트됩니다:

| 컨테이너 경로 | 호스트 경로 | 모드 | 참고 |
|---|---|---|---|
| `/nix/store` | `/nix/store` | `ro` | Hermes 바이너리 + 모든 Nix 종속성 |
| `/data` | `/var/lib/hermes` | `rw` | 모든 상태, 구성, 작업 공간 |
| `/home/hermes` | `${stateDir}/home` | `rw` | 영구 에이전트 홈 — `pip install --user`, 도구 캐시 |
| `/usr`, `/usr/local`, `/tmp` | (쓰기 가능한 레이어) | `rw` | `apt`/`pip`/`npm` 설치 — 재생성 시 사라짐 |

---

## 업데이트

```bash
# Update the flake input (run from the directory containing flake.nix)
cd /etc/nixos && nix flake update hermes-agent

# Rebuild
sudo nixos-rebuild switch
```

컨테이너 모드에서는 `current-package` 심볼릭 링크가 업데이트되고, 에이전트는 재시작할 때 새 바이너리를 사용합니다. 컨테이너를 재생성할 필요가 없으며, 설치된 패키지도 손실되지 않습니다.

## 문제 해결

:::tip Podman 사용자
아래의 모든 `docker` 명령은 `podman`에서도 동일하게 작동합니다. `container.backend = "podman"`을 설정했다면 그에 맞게 바꿔 사용하세요.
:::

### 서비스 로그

```bash
# Both modes use the same systemd unit
journalctl -u hermes-agent -f

# Container mode: also available directly
docker logs -f hermes-agent
```

### 컨테이너 검사

```bash
systemctl status hermes-agent
docker ps -a --filter name=hermes-agent
docker inspect hermes-agent --format='{{.State.Status}}'
docker exec -it hermes-agent bash
docker exec hermes-agent readlink /data/current-package
docker exec hermes-agent cat /data/.container-identity
```

### 컨테이너 강제 재생성

쓰기 가능한 레이어를 초기화해야 하는 경우(새 Ubuntu):

```bash
sudo systemctl stop hermes-agent
docker rm -f hermes-agent
sudo rm /var/lib/hermes/.container-identity
sudo systemctl start hermes-agent
```

### 시크릿이 로드되었는지 확인

에이전트가 시작되지만 LLM 제공자에 인증할 수 없다면 `.env` 파일이 올바르게 병합되었는지 확인하세요:

```bash
# Native mode
sudo -u hermes cat /var/lib/hermes/.hermes/.env

# Container mode
docker exec hermes-agent cat /data/.hermes/.env
```

### GC 루트 확인

```bash
nix-store --query --roots $(docker exec hermes-agent readlink /data/current-package)
```

### 일반적인 문제

| 증상 | 원인 | 해결 방법 |
|---|---|---|
| `Cannot save configuration: managed by NixOS` | CLI 가드가 활성화됨 | `configuration.nix`를 편집하고 `nixos-rebuild switch`를 실행하세요 |
| `No adapter available for discord` (또는 telegram/slack) | 봉인된 Nix venv에 메시징 종속성이 없음 | `#messaging` 변형을 설치하세요: `nix profile install ...#messaging`. NixOS 모듈의 경우: `extraDependencyGroups = [ "messaging" ]`. 근본 원인의 `FeatureUnavailable` 또는 `requirements not met`을 확인하려면 `journalctl -u hermes-agent`를 확인하세요. |
| 컨테이너가 예기치 않게 재생성됨 | `extraVolumes`, `extraOptions` 또는 `image`가 변경됨 | 예상된 동작입니다 — 쓰기 가능한 레이어가 초기화됩니다. 패키지를 다시 설치하거나 사용자 지정 이미지를 사용하세요 |
| `hermes version`에 이전 버전이 표시됨 | 컨테이너가 재시작되지 않음 | `systemctl restart hermes-agent` |
| `/var/lib/hermes`에 대한 권한 거부 | 상태 디렉터리가 `0750 hermes:hermes`임 | `docker exec` 또는 `sudo -u hermes`를 사용하세요 |
| `nix-collect-garbage`가 hermes를 삭제함 | GC 루트가 없음 | 서비스를 재시작하세요(`preStart`가 GC 루트를 다시 생성합니다) |
| `no container with name or ID "hermes-agent"` (Podman) | 일반 사용자에게 Podman rootful 컨테이너가 표시되지 않음 | Podman에 비밀번호 없는 sudo를 추가하세요([컨테이너 모드](#container-mode) 섹션 참고) |
| `unable to find user hermes` | 컨테이너가 아직 시작 중임(entrypoint가 아직 사용자를 생성하지 않음) | 몇 초 기다린 후 다시 시도하세요 — CLI가 자동으로 재시도합니다 |
| `extraPackages`로 추가한 도구를 터미널에서 찾을 수 없음 | 사용자별 프로필을 업데이트하려면 `nixos-rebuild switch`가 필요함 | 다시 빌드하고 재시작하세요: `nixos-rebuild switch && systemctl restart hermes-agent` |
