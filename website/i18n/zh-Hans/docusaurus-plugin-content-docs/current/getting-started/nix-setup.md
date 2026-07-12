---
sidebar_position: 3
title: "Nix & NixOS 安装配置"
description: "使用 Nix 安装和部署 Hermes Agent——从快速 `nix run` 到完全声明式的 NixOS 模块（含容器模式）"
---

# Nix & NixOS 安装配置

Hermes Agent 提供了一个 Nix flake，支持三个层级的集成：

| 层级 | 适用对象 | 提供内容 |
|-------|-------------|--------------|
| **`nix run` / `nix profile install`** | 任意 Nix 用户（macOS、Linux） | 包含所有依赖的预构建二进制文件——然后使用标准 CLI 工作流 |
| **NixOS 模块（原生）** | NixOS 服务器部署 | 声明式配置、加固的 systemd 服务、托管密钥 |
| **NixOS 模块（容器）** | 需要自我修改能力的 Agent | 以上所有功能，加上一个持久化 Ubuntu 容器，Agent 可在其中执行 `apt`/`pip`/`npm install` |

:::info 与标准安装的区别
`curl | bash` 安装程序自行管理 Python、Node 及依赖项。Nix flake 替代了所有这些——每个 Python 依赖都是由 [uv2nix](https://github.com/pyproject-nix/uv2nix) 构建的 Nix derivation，运行时工具（Node.js、git、ripgrep、ffmpeg）已封装进二进制文件的 PATH 中。不需要运行时 pip，不需要激活 venv，不需要 `npm install`。

**对于非 NixOS 用户**，这只影响安装步骤。之后的操作（`hermes setup`、`hermes gateway install`、编辑配置）与标准安装完全相同。

**对于 NixOS 模块用户**，整个生命周期有所不同：配置存放在 `configuration.nix` 中，密钥通过 sops-nix/agenix 管理，服务是一个 systemd 单元，CLI 配置命令被屏蔽。管理 hermes 的方式与管理其他 NixOS 服务相同。
:::

## 前提条件

- **已启用 flakes 的 Nix** — 推荐使用 [Determinate Nix](https://install.determinate.systems)（默认启用 flakes）
- **API 密钥**，用于你想使用的服务（至少需要一个 OpenRouter 或 Anthropic 密钥）

---

## 快速开始（任意 Nix 用户）

无需克隆仓库。Nix 会自动获取、构建并运行所有内容：

```bash
# 直接运行（首次使用时构建，之后使用缓存）
nix run github:NousResearch/hermes-agent -- setup
nix run github:NousResearch/hermes-agent -- chat

# 或持久化安装
nix profile install github:NousResearch/hermes-agent
hermes setup
hermes chat
```

执行 `nix profile install` 后，`hermes`、`hermes-agent` 和 `hermes-acp` 将出现在你的 PATH 中。之后的工作流与[标准安装](./installation.md)完全相同——`hermes setup` 引导你完成提供商选择，`hermes gateway install` 设置 launchd（macOS）或 systemd 用户服务，配置存放在 `~/.hermes/`。

<details>
<summary><strong>从本地克隆构建</strong></summary>

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
nix build
./result/bin/hermes setup
```

</details>

---

## NixOS 模块

该 flake 导出 `nixosModules.default`——一个完整的 NixOS 服务模块，以声明式方式管理用户创建、目录、配置生成、密钥、文档和服务生命周期。

:::note
此模块需要 NixOS。对于非 NixOS 系统（macOS、其他 Linux 发行版），请使用 `nix profile install` 和上述标准 CLI 工作流。
:::

### 添加 Flake 输入

```nix
# /etc/nixos/flake.nix（或你的系统 flake）
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

### 最小化配置

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

就这些。`nixos-rebuild switch` 会创建 `hermes` 用户、生成 `config.yaml`、连接密钥并启动 gateway——这是一个长期运行的服务，将 Agent 连接到消息平台（Telegram、Discord 等）并监听传入消息。

:::warning 密钥是必需的
上面的 `environmentFiles` 行假设你已配置 [sops-nix](https://github.com/Mic92/sops-nix) 或 [agenix](https://github.com/ryantm/agenix)。该文件至少应包含一个 LLM 提供商密钥（例如 `OPENROUTER_API_KEY=sk-or-...`）。完整设置请参阅[密钥管理](#secrets-management)。如果你还没有密钥管理器，可以先使用普通文件——只需确保它不是全局可读的：

```bash
echo "OPENROUTER_API_KEY=sk-or-your-key" | sudo install -m 0600 -o hermes /dev/stdin /var/lib/hermes/env
```

```nix
services.hermes-agent.environmentFiles = [ "/var/lib/hermes/env" ];
```
:::

:::tip addToSystemPackages
设置 `addToSystemPackages = true` 有两个作用：将 `hermes` CLI 添加到系统 PATH，**并**在系统范围内设置 `HERMES_HOME`，使交互式 CLI 与 gateway 服务共享状态（会话、技能、cron）。不设置此项时，在 shell 中运行 `hermes` 会创建独立的 `~/.hermes/` 目录。
:::

### 容器感知 CLI

:::info
当 `container.enable = true` 且 `addToSystemPackages = true` 时，主机上的**所有** `hermes` 命令都会自动路由到托管容器中执行。这意味着你的交互式 CLI 会话在与 gateway 服务相同的环境中运行——可以访问所有容器内安装的包和工具。

- 路由是透明的：`hermes chat`、`hermes sessions list`、`hermes version` 等命令都会在底层 exec 进容器
- 所有 CLI 参数原样转发
- 如果容器未运行，CLI 会短暂重试（交互式使用时显示 5 秒 spinner，脚本中静默等待 10 秒），然后以明确的错误退出——不会静默回退
- 对于在 hermes 代码库上工作的开发者，设置 `HERMES_DEV=1` 可绕过容器路由，直接运行本地检出版本

设置 `container.hostUsers` 可创建 `~/.hermes` 到服务状态目录的符号链接，使主机 CLI 和容器共享会话、配置和记忆：

```nix
services.hermes-agent = {
  container.enable = true;
  container.hostUsers = [ "your-username" ];
  addToSystemPackages = true;
};
```

`hostUsers` 中列出的用户会自动加入 `hermes` 组以获得文件权限访问。

**Podman 用户：** NixOS 服务以 root 身份运行容器。Docker 用户通过 `docker` 组 socket 获得访问权限，但 Podman 的 rootful 容器需要 sudo。为你的容器运行时授予免密 sudo：

```nix
security.sudo.extraRules = [{
  users = [ "your-username" ];
  commands = [{
    command = "/run/current-system/sw/bin/podman";
    options = [ "NOPASSWD" ];
  }];
}];
```

CLI 会自动检测何时需要 sudo 并透明地使用它。没有此配置，你需要手动运行 `sudo hermes chat`。
:::

### 验证运行状态

执行 `nixos-rebuild switch` 后，检查服务是否正在运行：

```bash
# 检查服务状态
systemctl status hermes-agent

# 查看日志（Ctrl+C 停止）
journalctl -u hermes-agent -f

# 如果 addToSystemPackages 为 true，测试 CLI
hermes version
hermes config       # 显示生成的配置
```

### 选择部署模式

模块支持两种模式，由 `container.enable` 控制：

| | **原生**（默认） | **容器** |
|---|---|---|
| 运行方式 | 主机上加固的 systemd 服务 | 持久化 Ubuntu 容器，`/nix/store` 以只读方式绑定挂载 |
| 安全性 | `NoNewPrivileges`、`ProtectSystem=strict`、`PrivateTmp` | 容器隔离，内部以非特权用户运行 |
| Agent 可自行安装包 | 否——仅限 Nix 提供的 PATH 上的工具 | 是——`apt`、`pip`、`npm` 安装的包在重启后持久保留 |
| 配置界面 | 相同 | 相同 |
| 适用场景 | 标准部署、最高安全性、可重现性 | Agent 需要运行时安装包、可变环境、实验性工具 |

启用容器模式只需添加一行：

```nix
{
  services.hermes-agent = {
    enable = true;
    container.enable = true;
    # ... 其余配置相同
  };
}
```

:::info
容器模式通过 `mkDefault` 自动启用 `virtualisation.docker.enable`。如果你使用 Podman，请设置 `container.backend = "podman"` 并将 `virtualisation.docker.enable` 设为 `false`。
:::

---

## 配置

### 声明式设置

`settings` 选项接受任意 attrset，并将其渲染为 `config.yaml`。它支持跨多个模块定义的深度合并（通过 `lib.recursiveUpdate`），因此你可以将配置拆分到多个文件中：

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

两者在求值时深度合并。Nix 声明的键始终优先于磁盘上现有 `config.yaml` 中的键，但 **Nix 未涉及的用户添加键会被保留**。这意味着如果 Agent 或手动编辑添加了 `skills.disabled` 或 `streaming.enabled` 等键，它们在 `nixos-rebuild switch` 后仍会保留。

:::note 模型命名
`settings.model.default` 使用你的提供商所期望的模型标识符。使用 [OpenRouter](https://openrouter.ai)（默认）时，格式如 `"anthropic/claude-sonnet-4"` 或 `"google/gemini-3-flash"`。如果直接使用提供商（Anthropic、OpenAI），请将 `settings.model.base_url` 指向其 API，并使用其原生模型 ID（例如 `"claude-sonnet-4-20250514"`）。未设置 `base_url` 时，Hermes 默认使用 OpenRouter。
:::

:::tip 查找可用配置键
运行 `nix build .#configKeys && cat result` 可查看从 Python `DEFAULT_CONFIG` 中提取的所有叶配置键。你可以将现有的 `config.yaml` 粘贴到 `settings` attrset 中——结构是 1:1 对应的。
:::

<details>
<summary><strong>完整示例：所有常用自定义设置</strong></summary>

```nix
{ config, ... }: {
  services.hermes-agent = {
    enable = true;
    container.enable = true;

    # ── 模型 ──────────────────────────────────────────────────────────
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

    # ── 密钥 ────────────────────────────────────────────────────────
    environmentFiles = [ config.sops.secrets."hermes-env".path ];

    # ── 文档 ──────────────────────────────────────────────────────────
    documents = {
      "USER.md" = ./documents/USER.md;
    };

    # ── MCP 服务器 ────────────────────────────────────────────────────
    mcpServers.filesystem = {
      command = "npx";
      args = [ "-y" "@modelcontextprotocol/server-filesystem" "/data/workspace" ];
    };

    # ── 容器选项 ──────────────────────────────────────────────────────
    container = {
      image = "ubuntu:24.04";
      backend = "docker";
      hostUsers = [ "your-username" ];
      extraVolumes = [ "/home/user/projects:/projects:rw" ];
      extraOptions = [ "--gpus" "all" ];
    };

    # ── 服务调优 ─────────────────────────────────────────────────────
    addToSystemPackages = true;
    extraArgs = [ "--verbose" ];
    restart = "always";
    restartSec = 5;
  };
}
```

</details>

### 逃生舱：自带配置文件

如果你希望完全在 Nix 之外管理 `config.yaml`，请使用 `configFile`：

```nix
services.hermes-agent.configFile = /etc/hermes/config.yaml;
```

这会完全绕过 `settings`——不合并，不生成。每次激活时，该文件会原样复制到 `$HERMES_HOME/config.yaml`。

### 自定义速查表

Nix 用户最常见自定义需求的快速参考：

| 我想要... | 选项 | 示例 |
|---|---|---|
| 更改 LLM 模型 | `settings.model.default` | `"anthropic/claude-sonnet-4"` |
| 使用不同的提供商端点 | `settings.model.base_url` | `"https://openrouter.ai/api/v1"` |
| 添加 API 密钥 | `environmentFiles` | `[ config.sops.secrets."hermes-env".path ]` |
| 给 Agent 设置个性 | `${services.hermes-agent.stateDir}/.hermes/SOUL.md` | 直接管理该文件 |
| 添加 MCP 工具服务器 | `mcpServers.<name>` | 参见 [MCP 服务器](#mcp-servers) |
| 将主机目录挂载到容器 | `container.extraVolumes` | `[ "/data:/data:rw" ]` |
| 为容器传入 GPU 访问 | `container.extraOptions` | `[ "--gpus" "all" ]` |
| 使用 Podman 替代 Docker | `container.backend` | `"podman"` |
| 在主机 CLI 和容器间共享状态 | `container.hostUsers` | `[ "sidbin" ]` |
| 为 Agent 提供额外工具 | `extraPackages` | `[ pkgs.pandoc pkgs.imagemagick ]` |
| 使用自定义基础镜像 | `container.image` | `"ubuntu:24.04"` |
| 覆盖 hermes 包 | `package` | `inputs.hermes-agent.packages.${system}.default.override { ... }` |
| 更改状态目录 | `stateDir` | `"/opt/hermes"` |
| 设置 Agent 的工作目录 | `workingDirectory` | `"/home/user/projects"` |

---

## 密钥管理

:::danger 切勿将 API 密钥放入 `settings` 或 `environment`
Nix 表达式中的值会进入 `/nix/store`，该目录是全局可读的。请始终使用带有密钥管理器的 `environmentFiles`。
:::

`environment`（非密钥变量）和 `environmentFiles`（密钥文件）在激活时（`nixos-rebuild switch`）都会合并到 `$HERMES_HOME/.env` 中。Hermes 在每次启动时读取此文件，因此更改在 `systemctl restart hermes-agent` 后生效——无需重建容器。

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

密钥文件包含键值对：

```yaml
# secrets/hermes.yaml（使用 sops 加密）
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

### OAuth / 认证预置

对于需要 OAuth 的平台（例如 Discord），使用 `authFile` 在首次部署时预置凭据：

```nix
{
  services.hermes-agent = {
    authFile = config.sops.secrets."hermes/auth.json".path;
    # authFileForceOverwrite = true;  # 每次激活时强制覆盖
  };
}
```

仅当 `auth.json` 不存在时才复制该文件（除非 `authFileForceOverwrite = true`）。运行时 OAuth token 刷新会写入状态目录，并在重建后保留。

---

## 文档

`documents` 选项将文件安装到 Agent 的工作目录（即 `workingDirectory`，Agent 将其作为工作区读取）。Hermes 按约定查找特定文件名：

- **`USER.md`** — 关于 Agent 正在交互的用户的上下文信息。
- 你放置在此处的任何其他文件对 Agent 都可见，作为工作区文件。

Agent 身份文件是独立的：Hermes 从 `$HERMES_HOME/SOUL.md` 加载其主要 `SOUL.md`，在 NixOS 模块中对应 `${services.hermes-agent.stateDir}/.hermes/SOUL.md`。将 `SOUL.md` 放入 `documents` 只会创建一个工作区文件，不会替换主角色文件。

```nix
{
  services.hermes-agent.documents = {
    "USER.md" = ./documents/USER.md;  # 路径引用，从 Nix store 复制
  };
}
```

值可以是内联字符串或路径引用。文件在每次 `nixos-rebuild switch` 时安装。

---

## MCP 服务器

`mcpServers` 选项以声明式方式配置 [MCP（Model Context Protocol，模型上下文协议）](https://modelcontextprotocol.io)服务器。每个服务器使用 **stdio**（本地命令）或 **HTTP**（远程 URL）传输方式。

### stdio 传输（本地服务器）

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
      env.GITHUB_PERSONAL_ACCESS_TOKEN = "\${GITHUB_TOKEN}"; # 从 .env 解析
    };
  };
}
```

:::tip
`env` 值中的环境变量在运行时从 `$HERMES_HOME/.env` 解析。使用 `environmentFiles` 注入密钥——切勿将 token 直接放入 Nix 配置。
:::

### HTTP 传输（远程服务器）

```nix
{
  services.hermes-agent.mcpServers.remote-api = {
    url = "https://mcp.example.com/v1/mcp";
    headers.Authorization = "Bearer \${MCP_REMOTE_API_KEY}";
    timeout = 180;
  };
}
```

### 带 OAuth 的 HTTP 传输

对于使用 OAuth 2.1 的服务器，设置 `auth = "oauth"`。Hermes 实现了完整的 PKCE 流程——元数据发现、动态客户端注册、token 交换和自动刷新。

```nix
{
  services.hermes-agent.mcpServers.my-oauth-server = {
    url = "https://mcp.example.com/mcp";
    auth = "oauth";
  };
}
```

Token 存储在 `$HERMES_HOME/mcp-tokens/<server-name>.json` 中，在重启和重建后持久保留。

<details>
<summary><strong>无头服务器上的初始 OAuth 授权</strong></summary>

首次 OAuth 授权需要基于浏览器的同意流程。在无头部署中，Hermes 将授权 URL 打印到 stdout/日志，而不是打开浏览器。

**方案 A：交互式引导** — 通过 `docker exec`（容器）或 `sudo -u hermes`（原生）运行一次流程：

```bash
# 容器模式
docker exec -it hermes-agent \
  hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth

# 原生模式
sudo -u hermes HERMES_HOME=/var/lib/hermes/.hermes \
  hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth
```

容器使用 `--network=host`，因此 `127.0.0.1` 上的 OAuth 回调监听器可从主机浏览器访问。

**方案 B：预置 token** — 在工作站上完成流程，然后复制 token：

```bash
hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth
scp ~/.hermes/mcp-tokens/my-oauth-server{,.client}.json \
    server:/var/lib/hermes/.hermes/mcp-tokens/
# 确保：chown hermes:hermes，chmod 0600
```

</details>

### Sampling（服务器发起的 LLM 请求）

部分 MCP 服务器可以向 Agent 请求 LLM 补全：

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

## 托管模式

当 hermes 通过 NixOS 模块运行时，以下 CLI 命令会被**屏蔽**，并显示指向 `configuration.nix` 的描述性错误：

| 被屏蔽的命令 | 原因 |
|---|---|
| `hermes setup` | 配置是声明式的——请在 Nix 配置中编辑 `settings` |
| `hermes config edit` | 配置由 `settings` 生成 |
| `hermes config set <key> <value>` | 配置由 `settings` 生成 |
| `hermes gateway install` | systemd 服务由 NixOS 管理 |
| `hermes gateway uninstall` | systemd 服务由 NixOS 管理 |

这可以防止 Nix 声明的内容与磁盘上实际内容之间产生漂移。检测使用两个信号：

1. **`HERMES_MANAGED=true`** 环境变量——由 systemd 服务设置，对 gateway 进程可见
2. **`.managed` 标记文件**，位于 `HERMES_HOME` 中——由激活脚本设置，对交互式 shell 可见（例如 `docker exec -it hermes-agent hermes config set ...` 也会被屏蔽）

要更改配置，请编辑你的 Nix 配置并运行 `sudo nixos-rebuild switch`。

---

## 容器架构

:::info
本节仅在使用 `container.enable = true` 时相关。原生模式部署可跳过。
:::

启用容器模式后，hermes 在持久化 Ubuntu 容器内运行，Nix 构建的二进制文件以只读方式从主机绑定挂载：

```
主机                                    容器
────                                    ─────────
/nix/store/...-hermes-agent-0.1.0  ──►  /nix/store/... (ro)
~/.hermes -> /var/lib/hermes/.hermes       （符号链接桥接，按 hostUsers）
/var/lib/hermes/                    ──►  /data/          (rw)
  ├── current-package -> /nix/store/...    （符号链接，每次重建更新）
  ├── .gc-root -> /nix/store/...           （防止 nix-collect-garbage）
  ├── .container-identity                  （sha256 哈希，触发重建）
  ├── .hermes/                             （HERMES_HOME）
  │   ├── .env                             （从 environment + environmentFiles 合并）
  │   ├── config.yaml                      （Nix 生成，激活时深度合并）
  │   ├── .managed                         （标记文件）
  │   ├── .container-mode                  （路由元数据：backend、exec_user 等）
  │   ├── state.db, sessions/, memories/   （运行时状态）
  │   └── mcp-tokens/                      （MCP 服务器的 OAuth token）
  ├── home/                                ──►  /home/hermes    (rw)
  └── workspace/                           （MESSAGING_CWD）
      ├── SOUL.md                          （来自 documents 选项）
      └── （Agent 创建的文件）

容器可写层（apt/pip/npm）：   /usr, /usr/local, /tmp
```

Nix 构建的二进制文件能在 Ubuntu 容器内运行，是因为 `/nix/store` 被绑定挂载——它携带自己的解释器和所有依赖，不依赖容器的系统库。容器入口点通过 `current-package` 符号链接解析：`/data/current-package/bin/hermes gateway run --replace`。执行 `nixos-rebuild switch` 时，只更新符号链接——容器继续运行。

### 各事件的持久性

| 事件 | 容器重建？ | `/data`（状态） | `/home/hermes` | 可写层（`apt`/`pip`/`npm`） |
|---|---|---|---|---|
| `systemctl restart hermes-agent` | 否 | 保留 | 保留 | 保留 |
| `nixos-rebuild switch`（代码变更） | 否（更新符号链接） | 保留 | 保留 | 保留 |
| 主机重启 | 否 | 保留 | 保留 | 保留 |
| `nix-collect-garbage` | 否（GC root） | 保留 | 保留 | 保留 |
| 镜像变更（`container.image`） | **是** | 保留 | 保留 | **丢失** |
| 卷/选项变更 | **是** | 保留 | 保留 | **丢失** |
| `environment`/`environmentFiles` 变更 | 否 | 保留 | 保留 | 保留 |

仅当容器的**身份哈希**发生变化时才会重建容器。哈希涵盖：schema 版本、镜像、`extraVolumes`、`extraOptions` 和入口点脚本。环境变量、settings、文档或 hermes 包本身的变更**不会**触发重建。

:::warning 可写层丢失
当身份哈希发生变化（镜像升级、新卷、新容器选项）时，容器会被销毁并从 `container.image` 的全新拉取重建。可写层中通过 `apt install`、`pip install` 或 `npm install` 安装的包将丢失。`/data` 和 `/home/hermes` 中的状态会保留（这些是绑定挂载）。

如果 Agent 依赖特定包，考虑将其烘焙到自定义镜像中（`container.image = "my-registry/hermes-base:latest"`），或在 Agent 的 SOUL.md 中编写安装脚本。
:::

### GC Root 保护

`preStart` 脚本在 `${stateDir}/.gc-root` 创建一个指向当前 hermes 包的 GC root。这可以防止 `nix-collect-garbage` 删除正在运行的二进制文件。如果 GC root 损坏，重启服务会重新创建它。

---

## 插件

NixOS 模块支持声明式插件安装——无需命令式的 `hermes plugins install`。

### 目录插件（`extraPlugins`）

对于只包含 `plugin.yaml` + `__init__.py` 的源码树插件（例如 [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm)）：

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

插件在激活时以符号链接方式安装到 `$HERMES_HOME/plugins/`。Hermes 通过其正常的目录扫描发现它们。从列表中移除插件并运行 `nixos-rebuild switch` 会删除符号链接。

### 入口点插件（`extraPythonPackages`）

对于通过 `[project.entry-points."hermes_agent.plugins"]` 注册的 pip 打包插件（例如 [rtk-hermes](https://github.com/ogallotti/rtk-hermes)）：

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

该包的 `site-packages` 会添加到 hermes wrapper 的 PYTHONPATH 中。`importlib.metadata` 在会话启动时发现入口点。

### 可选依赖组（`extraDependencyGroups`）

对于已在 hermes-agent 的 `pyproject.toml` 中声明的可选 extras（例如 `hindsight` 或 `honcho` 等记忆提供商），使用 `extraDependencyGroups` 在构建时将其包含到封闭的 venv 中：

```nix
services.hermes-agent = {
  extraDependencyGroups = [ "hindsight" ];
  settings.memory.provider = "hindsight";
};
```

这由 uv 与核心依赖在单次解析中完成——不需要 PYTHONPATH 补丁，没有冲突风险。可用的组与 `pyproject.toml` 中 `[project.optional-dependencies]` 的键对应（例如 `"hindsight"`、`"honcho"`、`"voice"`、`"matrix"`、`"mistral"`、`"bedrock"`）。

**何时使用哪个：**

| 需求 | 选项 |
|------|--------|
| 启用 pyproject.toml 可选 extra | `extraDependencyGroups` |
| 添加不在 pyproject.toml 中的外部 Python 插件 | `extraPythonPackages` |
| 添加系统二进制文件（pandoc、jq 等） | `extraPackages` |
| 添加基于目录的插件源码树 | `extraPlugins` |

### 组合使用

带有第三方 Python 依赖的目录插件需要同时使用两个选项：

```nix
services.hermes-agent = {
  extraPlugins = [ my-plugin-src ];          # 插件源码
  extraPythonPackages = [ pkgs.python312Packages.redis ];  # 其 Python 依赖
  extraPackages = [ pkgs.redis ];            # 其需要的系统二进制文件
};
```

### 使用 Overlay

外部 flake 可以直接覆盖包：

```nix
{
  inputs.hermes-agent.url = "github:NousResearch/hermes-agent";
  outputs = { hermes-agent, nixpkgs, ... }: {
    nixpkgs.overlays = [ hermes-agent.overlays.default ];
    # 然后：
    #   pkgs.hermes-agent.override { extraPythonPackages = [...]; }
    #   pkgs.hermes-agent.override { extraDependencyGroups = [ "hindsight" ]; }
  };
}
```

### 插件配置

插件仍需在 `config.yaml` 中启用。通过声明式 settings 添加：

```nix
services.hermes-agent.settings.plugins.enabled = [
  "hermes-lcm"
  "rtk-rewrite"
];
```

:::note
构建时冲突检查可防止插件包覆盖核心 hermes 依赖。如果插件提供了封闭 venv 中已有的包，`nixos-rebuild` 会以明确的错误失败。
:::

---

## 开发

### 开发 Shell

该 flake 提供了一个包含 Python 3.12、uv、Node.js 和所有运行时工具的开发 shell：

```bash
cd hermes-agent
nix develop

# Shell 提供：
#   - Python 3.12 + uv（首次进入时将依赖安装到 .venv）
#   - Node.js 22、ripgrep、git、openssh、ffmpeg 在 PATH 上
#   - 戳记文件优化：依赖未变更时重新进入几乎即时

hermes setup
hermes chat
```

### direnv（推荐）

包含的 `.envrc` 会自动激活开发 shell：

```bash
cd hermes-agent
direnv allow    # 仅需一次
# 后续进入几乎即时（戳记文件跳过依赖安装）
```

### Flake 检查

该 flake 包含在 CI 和本地运行的构建时验证：

```bash
# 运行所有检查
nix flake check

# 单独检查
nix build .#checks.x86_64-linux.package-contents   # 二进制文件存在 + 版本
nix build .#checks.x86_64-linux.entry-points-sync  # pyproject.toml ↔ Nix 包同步
nix build .#checks.x86_64-linux.cli-commands        # gateway/config 子命令
nix build .#checks.x86_64-linux.managed-guard       # HERMES_MANAGED 屏蔽变更操作
nix build .#checks.x86_64-linux.bundled-skills      # 包中存在 skills
nix build .#checks.x86_64-linux.config-roundtrip    # 合并脚本保留用户键
```

<details>
<summary><strong>每项检查的验证内容</strong></summary>

| 检查 | 测试内容 |
|---|---|
| `package-contents` | `hermes` 和 `hermes-agent` 二进制文件存在且 `hermes version` 可运行 |
| `entry-points-sync` | `pyproject.toml` 中 `[project.scripts]` 的每个条目在 Nix 包中都有对应的封装二进制文件 |
| `cli-commands` | `hermes --help` 暴露 `gateway` 和 `config` 子命令 |
| `managed-guard` | `HERMES_MANAGED=true hermes config set ...` 打印 NixOS 错误 |
| `bundled-skills` | skills 目录存在，包含 SKILL.md 文件，wrapper 中设置了 `HERMES_BUNDLED_SKILLS` |
| `config-roundtrip` | 7 种合并场景：全新安装、Nix 覆盖、用户键保留、混合合并、MCP 累加合并、嵌套深度合并、幂等性 |

</details>

---

## 选项参考

### 核心

| 选项 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `enable` | `bool` | `false` | 启用 hermes-agent 服务 |
| `package` | `package` | `hermes-agent` | 使用的 hermes-agent 包 |
| `user` | `str` | `"hermes"` | 系统用户 |
| `group` | `str` | `"hermes"` | 系统组 |
| `createUser` | `bool` | `true` | 自动创建用户/组 |
| `stateDir` | `str` | `"/var/lib/hermes"` | 状态目录（`HERMES_HOME` 的父目录） |
| `workingDirectory` | `str` | `"${stateDir}/workspace"` | Agent 工作目录（`MESSAGING_CWD`） |
| `addToSystemPackages` | `bool` | `false` | 将 `hermes` CLI 添加到系统 PATH 并在系统范围内设置 `HERMES_HOME` |

### 配置

| 选项 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `settings` | `attrs`（深度合并） | `{}` | 声明式配置，渲染为 `config.yaml`。支持任意嵌套；多个定义通过 `lib.recursiveUpdate` 合并 |
| `configFile` | `null` 或 `path` | `null` | 现有 `config.yaml` 的路径。设置后完全覆盖 `settings` |

### 密钥与环境

| 选项 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `environmentFiles` | `listOf str` | `[]` | 包含密钥的 env 文件路径。激活时合并到 `$HERMES_HOME/.env` |
| `environment` | `attrsOf str` | `{}` | 非密钥环境变量。**在 Nix store 中可见**——请勿在此放置密钥 |
| `authFile` | `null` 或 `path` | `null` | OAuth 凭据预置文件。仅在首次部署时复制 |
| `authFileForceOverwrite` | `bool` | `false` | 每次激活时始终从 `authFile` 覆盖 `auth.json` |

### 文档

| 选项 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `documents` | `attrsOf (either str path)` | `{}` | 工作区文件。键为文件名，值为内联字符串或路径。激活时安装到 `workingDirectory` |

### MCP 服务器

| 选项 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `mcpServers` | `attrsOf submodule` | `{}` | MCP 服务器定义，合并到 `settings.mcp_servers` |
| `mcpServers.<name>.command` | `null` 或 `str` | `null` | 服务器命令（stdio 传输） |
| `mcpServers.<name>.args` | `listOf str` | `[]` | 命令参数 |
| `mcpServers.<name>.env` | `attrsOf str` | `{}` | 服务器进程的环境变量 |
| `mcpServers.<name>.url` | `null` 或 `str` | `null` | 服务器端点 URL（HTTP/StreamableHTTP 传输） |
| `mcpServers.<name>.headers` | `attrsOf str` | `{}` | HTTP 头，例如 `Authorization` |
| `mcpServers.<name>.auth` | `null` 或 `"oauth"` | `null` | 认证方式。`"oauth"` 启用 OAuth 2.1 PKCE |
| `mcpServers.<name>.enabled` | `bool` | `true` | 启用或禁用此服务器 |
| `mcpServers.<name>.timeout` | `null` 或 `int` | `null` | 工具调用超时（秒，默认：120） |
| `mcpServers.<name>.connect_timeout` | `null` 或 `int` | `null` | 连接超时（秒，默认：60） |
| `mcpServers.<name>.tools` | `null` 或 `submodule` | `null` | 工具过滤（`include`/`exclude` 列表） |
| `mcpServers.<name>.sampling` | `null` 或 `submodule` | `null` | 服务器发起 LLM 请求的 sampling 配置 |

### 服务行为

| 选项 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `extraArgs` | `listOf str` | `[]` | `hermes gateway` 的额外参数 |
| `extraPackages` | `listOf package` | `[]` | Agent 可用的额外包。添加到 hermes 用户的每用户 profile，终端命令、skills 和 cron 任务均可见 |
| `extraPlugins` | `listOf package` | `[]` | 以符号链接方式安装到 `$HERMES_HOME/plugins/` 的目录插件包。每个包必须包含 `plugin.yaml` |
| `extraPythonPackages` | `listOf package` | `[]` | 添加到 PYTHONPATH 用于入口点插件发现的 Python 包。使用 `python312Packages` 构建 |
| `extraDependencyGroups` | `listOf str` | `[]` | 包含到封闭 venv 中的 pyproject.toml 可选 extras（例如 `["hindsight"]`）。由 uv 解析——无冲突 |
| `restart` | `str` | `"always"` | systemd `Restart=` 策略 |
| `restartSec` | `int` | `5` | systemd `RestartSec=` 值 |

### 容器

| 选项 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `container.enable` | `bool` | `false` | 启用 OCI 容器模式 |
| `container.backend` | `enum ["docker" "podman"]` | `"docker"` | 容器运行时 |
| `container.image` | `str` | `"ubuntu:24.04"` | 基础镜像（运行时拉取） |
| `container.extraVolumes` | `listOf str` | `[]` | 额外卷挂载（`host:container:mode`） |
| `container.extraOptions` | `listOf str` | `[]` | 传递给 `docker create` 的额外参数 |
| `container.hostUsers` | `listOf str` | `[]` | 获得 `~/.hermes` 符号链接（指向服务 stateDir）的交互式用户，自动加入 `hermes` 组 |

---

## 目录结构

### 原生模式

```
/var/lib/hermes/                     # stateDir（归 hermes:hermes 所有，权限 0750）
├── .hermes/                         # HERMES_HOME
│   ├── config.yaml                  # Nix 生成（每次重建深度合并）
│   ├── .managed                     # 标记：CLI 配置变更被屏蔽
│   ├── .env                         # 从 environment + environmentFiles 合并
│   ├── auth.json                    # OAuth 凭据（预置后自我管理）
│   ├── gateway.pid
│   ├── state.db
│   ├── mcp-tokens/                  # MCP 服务器的 OAuth token
│   ├── sessions/
│   ├── memories/
│   ├── skills/
│   ├── cron/
│   └── logs/
├── home/                            # Agent HOME
└── workspace/                       # MESSAGING_CWD
    ├── SOUL.md                      # 来自 documents 选项
    └── （Agent 创建的文件）
```

### 容器模式

相同的布局，挂载到容器中：

| 容器路径 | 主机路径 | 模式 | 说明 |
|---|---|---|---|
| `/nix/store` | `/nix/store` | `ro` | Hermes 二进制文件 + 所有 Nix 依赖 |
| `/data` | `/var/lib/hermes` | `rw` | 所有状态、配置、工作区 |
| `/home/hermes` | `${stateDir}/home` | `rw` | 持久化 Agent home——`pip install --user`、工具缓存 |
| `/usr`、`/usr/local`、`/tmp` | （可写层） | `rw` | `apt`/`pip`/`npm` 安装——重启后持久，重建后丢失 |

---

## 更新

```bash
# 更新 flake 输入（在包含 flake.nix 的目录中运行）
cd /etc/nixos && nix flake update hermes-agent

# 重建
sudo nixos-rebuild switch
```

在容器模式下，`current-package` 符号链接会更新，Agent 在重启时获取新的二进制文件。不会重建容器，不会丢失已安装的包。

---

## 故障排查

:::tip Podman 用户
以下所有 `docker` 命令在 `podman` 中同样适用。如果你设置了 `container.backend = "podman"`，请相应替换。
:::

### 服务日志

```bash
# 两种模式使用相同的 systemd 单元
journalctl -u hermes-agent -f

# 容器模式：也可直接查看
docker logs -f hermes-agent
```

### 容器检查

```bash
systemctl status hermes-agent
docker ps -a --filter name=hermes-agent
docker inspect hermes-agent --format='{{.State.Status}}'
docker exec -it hermes-agent bash
docker exec hermes-agent readlink /data/current-package
docker exec hermes-agent cat /data/.container-identity
```

### 强制重建容器

如果需要重置可写层（全新 Ubuntu）：

```bash
sudo systemctl stop hermes-agent
docker rm -f hermes-agent
sudo rm /var/lib/hermes/.container-identity
sudo systemctl start hermes-agent
```

### 验证密钥已加载

如果 Agent 启动但无法向 LLM 提供商认证，检查 `.env` 文件是否正确合并：

```bash
# 原生模式
sudo -u hermes cat /var/lib/hermes/.hermes/.env

# 容器模式
docker exec hermes-agent cat /data/.hermes/.env
```

### GC Root 验证

```bash
nix-store --query --roots $(docker exec hermes-agent readlink /data/current-package)
```

### 常见问题

| 现象 | 原因 | 解决方法 |
|---|---|---|
| `Cannot save configuration: managed by NixOS` | CLI 守卫已激活 | 编辑 `configuration.nix` 并执行 `nixos-rebuild switch` |
| 容器意外重建 | `extraVolumes`、`extraOptions` 或 `image` 发生变更 | 预期行为——可写层重置。重新安装包或使用自定义镜像 |
| `hermes version` 显示旧版本 | 容器未重启 | `systemctl restart hermes-agent` |
| `/var/lib/hermes` 权限拒绝 | 状态目录为 `0750 hermes:hermes` | 使用 `docker exec` 或 `sudo -u hermes` |
| `nix-collect-garbage` 删除了 hermes | GC root 缺失 | 重启服务（preStart 会重新创建 GC root） |
| `no container with name or ID "hermes-agent"`（Podman） | Podman rootful 容器对普通用户不可见 | 为 podman 添加免密 sudo（参见[容器模式](#container-mode)章节） |
| `unable to find user hermes` | 容器仍在启动中（入口点尚未创建用户） | 等待几秒后重试——CLI 会自动重试 |
| 通过 `extraPackages` 添加的工具在终端中找不到 | 需要 `nixos-rebuild switch` 更新每用户 profile | 重建并重启：`nixos-rebuild switch && systemctl restart hermes-agent` |