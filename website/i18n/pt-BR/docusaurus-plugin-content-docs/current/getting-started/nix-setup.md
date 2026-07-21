---
sidebar_position: 3
title: "Nix & NixOS Setup"
description: "Instale e implante o Hermes Agent com Nix — desde `nix run` rápido até módulo NixOS totalmente declarativo com modo container"
---

# Nix & NixOS Setup

:::warning Plataforma Tier 2
Nix e NixOS são [plataformas Tier 2](./platform-support.md#tier-2). O flake e módulo NixOS documentados aqui são mantidos apenas como melhor esforço. Commits no `main` podem quebrar estes pacotes a qualquer momento.

Para uma configuração suportada, use um dos caminhos de [instalação](./installation.md) padrão — Docker ou ambiente FHS.
:::

O Hermes Agent oferece um flake Nix e um módulo NixOS.

| Nível          | Para quem é                                          | O que você obtém                                                              |
|----------------|------------------------------------------------------|-------------------------------------------------------------------------------|
| **`nix run` / `nix profile install`** | Qualquer usuário Nix (macOS, Linux) | Binário pré-construído com todas as dependências — depois use o fluxo CLI padrão |
| **Módulo NixOS (nativo)** | Implantações em servidor NixOS             | Configuração declarativa, serviço systemd com hardening, segredos gerenciados |
| **Módulo NixOS (container)** | Agentes que precisam de auto-modificação | Tudo acima, mais um container Ubuntu persistente onde o agente pode `apt`/`pip`/`npm install` |

:::info O que é diferente da instalação padrão
O instalador `curl | bash` gerencia Python, Node e dependências por conta própria. O flake Nix substitui tudo isso — toda dependência Python é uma derivação Nix construída por [uv2nix](https://github.com/pyproject-nix/uv2nix), e ferramentas de runtime (Node.js, git, ripgrep, ffmpeg) são incluídas no PATH do binário. Não há pip em runtime, nenhuma ativação de venv, nenhum `npm install`.

**Para usuários não-NixOS**, isto apenas muda a etapa de instalação. Tudo depois (`hermes setup`, `hermes gateway install`, edição de config) funciona de forma idêntica à instalação padrão.

**Para usuários do módulo NixOS**, todo o ciclo de vida é diferente: a configuração vive em `configuration.nix`, segredos vão através de sops-nix/agenix, o serviço é uma unit systemd, e comandos de configuração do CLI são bloqueados. Você gerencia o Hermes da mesma forma que gerencia qualquer outro serviço NixOS.
:::

## Pré-requisitos

- **Nix com flakes ativados** — [Determinate Nix](https://install.determinate.systems) recomendado (ativa flakes por padrão)
- **Chaves de API** para os serviços que você deseja usar (no mínimo: uma chave OpenRouter ou Anthropic)

---

## Início Rápido (Qualquer Usuário Nix)

Nenhum clone necessário. O Nix busca, constrói e executa tudo:

```bash
# Executar o aplicativo desktop
nix run github:NousResearch/hermes-agent#desktop

# Ou instalar persistentemente
nix profile install github:NousResearch/hermes-agent#desktop

# Executar o tui
nix run github:NousResearch/hermes-agent -- setup
nix run github:NousResearch/hermes-agent -- --tui

# Ou instalar em seu profile
nix profile install github:NousResearch/hermes-agent
hermes setup
hermes --tui
```

Após `nix profile install`, `hermes`, `hermes-agent` e `hermes-acp` estão no seu PATH. A partir daqui, o fluxo de trabalho é idêntico à [instalação padrão](./installation.md) — `hermes setup` guia você pela seleção de provider, `hermes gateway install` configura um serviço launchd (macOS) ou systemd user service, e a configuração vive em `~/.hermes/`.

:::warning Plataformas de Mensagem (Discord, Telegram, Slack)
O pacote padrão inclui TODAS as bibliotecas que o hermes-agent pode precisar. Se você quiser uma variante menor, verifique as outras saídas do flake.

O pacote `default` adiciona ~700 MB ao closure. Se você só precisa de plataformas de mensagem, `#messaging` adiciona apenas ~33 MB.
:::

<details>
<summary><strong>Executando a partir de um clone local</strong></summary>

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
nix develop
hermes setup
```

</details>

---

## Módulo NixOS

O flake exporta `nixosModules.default` — um módulo de serviço NixOS completo que gerencia declarativamente criação de usuário, diretórios, geração de configuração, segredos, documentos e ciclo de vida do serviço.

:::note
Este módulo requer NixOS. Para sistemas não-NixOS (macOS, outras distribuições Linux), use `nix profile install` e o fluxo CLI padrão acima.
:::

### Adicione a Entrada do Flake

```nix
# /etc/nixos/flake.nix (ou seu flake de sistema)
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

### Configuração Mínima

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

Pronto. `nixos-rebuild switch` cria o usuário `hermes`, gera `config.yaml`, conecta segredos e inicia o gateway — um serviço de longa duração que conecta o agente a plataformas de mensagem (Telegram, Discord, etc.) e escuta mensagens recebidas.

:::warning Segredos são obrigatórios
A linha `environmentFiles` acima assume que você tem [sops-nix](https://github.com/Mic92/sops-nix) ou [agenix](https://github.com/ryantm/agenix) configurado. O arquivo deve conter pelo menos uma chave de provider LLM (ex.: `OPENROUTER_API_KEY=sk-or-...`). Veja [Gerenciamento de Segredos](#secrets-management) para configuração completa. Se você ainda não tem um gerenciador de segredos, pode usar um arquivo simples como ponto de partida — apenas garanta que não seja legível mundialmente:

```bash
echo "OPENROUTER_API_KEY=«redacted:sk-…»" | sudo install -m 0600 -o hermes /dev/stdin /var/lib/hermes/env
```

```nix
services.hermes-agent.environmentFiles = [ "/var/lib/hermes/env" ];
```
:::

:::tip addToSystemPackages
Definir `addToSystemPackages = true` faz duas coisas: coloca o CLI `hermes` no PATH do seu sistema **e** define `HERMES_HOME` em todo o sistema para que o CLI interativo compartilhe estado (sessões, skills, cron) com o serviço gateway. Sem isso, executar `hermes` em seu shell cria um diretório `~/.hermes/` separado.
:::

### CLI Ciente de Container

:::info
Quando `container.enable = true` e `addToSystemPackages = true`, **todo** comando `hermes` no host roteia automaticamente para dentro do container gerenciado. Isso significa que sua sessão CLI interativa executa dentro do mesmo ambiente que o serviço gateway.
:::

### Verifique se Funciona

```bash
# Verificar status do serviço
systemctl status hermes-agent

# Ver logs (Ctrl+C para parar)
journalctl -u hermes-agent -f

# Se addToSystemPackages é true, teste o CLI
hermes version
hermes config
```

### Escolhendo um Modo de Implantação

|                       | **Nativo** (padrão)                         | **Container**                                                    |
|-----------------------|---------------------------------------------|------------------------------------------------------------------|
| Como executa          | Serviço systemd com hardening no host       | Container Ubuntu persistente com `/nix/store` bind-mount         |
| Segurança             | `NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp` | Isolamento de container, executa como usuário sem privilégios   |
| Agente pode instalar pacotes | Não — apenas ferramentas no PATH fornecido pelo Nix | Sim — instalações `apt`, `pip`, `npm` persistem entre reinícios |
| Superfície de config  | Mesma                                        | Mesma                                                             |
| Quando escolher       | Implantações padrão, máxima segurança, reprodutibilidade | Agente precisa de instalação de pacotes em runtime, ambiente mutável |

Para ativar o modo container, adicione uma linha:

```nix
{
  services.hermes-agent = {
    enable = true;
    container.enable = true;
    # ... resto da config é idêntico
  };
}
```

---

## Configuração

### Configurações Declarativas

A opção `settings` aceita um attrset arbitrário que é renderizado como `config.yaml`. Suporta merge profundo entre múltiplas definições de módulo:

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

(Continua com seções de gerenciamento de segredos, documentos, servidores MCP — consulte o original em inglês para valores YAML/Nix exatos, pois blocos de código não são traduzidos.)

---

## Gerenciamento de Segredos

:::danger Nunca coloque chaves de API em `settings` ou `environment`
Valores em expressões Nix acabam em `/nix/store`, que é legível mundialmente. Sempre use `environmentFiles` com um gerenciador de segredos.
:::

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

### agenix

```nix
{
  age.secrets.hermes-env.file = ./secrets/hermes-env.age;

  services.hermes-agent.environmentFiles = [
    config.age.secrets.hermes-env.path
  ];
}
```

---

## Documentos

A opção `documents` instala arquivos no diretório de trabalho do agente.

```nix
{
  services.hermes-agent.documents = {
    "USER.md" = ./documents/USER.md;
  };
}
```

---

## Servidores MCP

A opção `mcpServers` configura declarativamente servidores [MCP (Model Context Protocol)](https://modelcontextprotocol.io).

### Transporte Stdio (Servidores Locais)

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
      env.GITHUB_PERSONAL_ACCESS_TOKEN = "\${GITHUB_TOKEN}";
    };
  };
}
```

### Transporte HTTP (Servidores Remotos)

```nix
{
  services.hermes-agent.mcpServers.remote-api = {
    url = "https://mcp.example.com/v1/mcp";
    headers.Authorization = "Bearer \${MCP_REMOTE_API_KEY}";
    timeout = 180;
  };
}
```

### Transporte HTTP com OAuth

```nix
{
  services.hermes-agent.mcpServers.my-oauth-server = {
    url = "https://mcp.example.com/mcp";
    auth = "oauth";
  };
}
```

(As seções restantes do nix-setup.md — autorização OAuth inicial em servidores headless e informações detalhadas de configuração — estão disponíveis no original em inglês.)
