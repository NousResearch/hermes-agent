---
sidebar_position: 3
title: "Configuração com Nix e NixOS"
description: "Instale e faça deploy do Hermes Agent com Nix — do `nix run` rápido ao módulo NixOS totalmente declarativo com modo container"
---

# Configuração com Nix e NixOS

:::warning Plataforma Tier 2
Nix e NixOS são [plataformas Tier 2](./platform-support.md#tier-2). O flake e o módulo NixOS documentados aqui são mantidos apenas com o melhor esforço possível. Commits em `main` podem quebrar esses pacotes a qualquer momento.

Para uma configuração com suporte oficial, use um dos caminhos padrão de [instalação](./installation.md) — Docker ou um ambiente FHS.
:::

O Hermes Agent inclui um flake Nix e um módulo NixOS.

| Nível | Para quem | O que você obtém |
|-------|-------------|--------------|
| **`nix run` / `nix profile install`** | Qualquer usuário Nix (macOS, Linux) | Binário pré-compilado com todas as deps — depois use o fluxo padrão do CLI |
| **Módulo NixOS (nativo)** | Deployments em servidores NixOS | Config declarativa, serviço systemd hardened, secrets gerenciados |
| **Módulo NixOS (container)** | Agents que precisam de auto-modificação | Tudo acima, mais um container Ubuntu persistente onde o agent pode fazer `apt`/`pip`/`npm install` |

:::info O que muda em relação à instalação padrão
O instalador `curl | bash` gerencia Python, Node e dependências por conta própria. O flake Nix substitui tudo isso — cada dependência Python é uma derivação Nix construída pelo [uv2nix](https://github.com/pyproject-nix/uv2nix), e ferramentas de runtime (Node.js, git, ripgrep, ffmpeg) são incluídas no PATH do binário. Não há pip em runtime, nem ativação de venv, nem `npm install`.

**Para usuários que não usam NixOS**, isso muda apenas a etapa de instalação. Tudo depois (`hermes setup`, `hermes gateway install`, edição de config) funciona de forma idêntica à instalação padrão.

**Para usuários do módulo NixOS**, todo o ciclo de vida é diferente: a configuração fica em `configuration.nix`, secrets passam por sops-nix/agenix, o serviço é uma unit systemd, e comandos de config do CLI são bloqueados. Você gerencia o hermes da mesma forma que qualquer outro serviço NixOS.
:::

## Pré-requisitos

- **Nix com flakes habilitados** — [Determinate Nix](https://install.determinate.systems) recomendado (habilita flakes por padrão)
- **API keys** para os serviços que você quer usar (no mínimo: uma chave OpenRouter ou Anthropic)

---

## Início rápido (qualquer usuário Nix) {#quick-start-any-nix-user}

Não precisa clonar. O Nix busca, constrói e executa tudo:

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

Depois de `nix profile install`, `hermes`, `hermes-agent` e `hermes-acp` ficam no seu PATH. A partir daí, o fluxo é idêntico à [instalação padrão](./installation.md) — `hermes setup` guia você na escolha do provider, `hermes gateway install` configura um serviço launchd (macOS) ou systemd de usuário, e a config fica em `~/.hermes/`.

:::warning Plataformas de messaging (Discord, Telegram, Slack)
O pacote padrão inclui TODAS as bibliotecas que o hermes-agent pode precisar. Se quiser uma variante menor, confira os outros outputs do flake.

O pacote `default` adiciona ~700 MB ao closure. Se você só precisa de plataformas de messaging, `#messaging` adiciona apenas ~33 MB.

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

O flake exporta `nixosModules.default` — um módulo de serviço NixOS completo que gerencia de forma declarativa criação de usuário, diretórios, geração de config, secrets, documentos e ciclo de vida do serviço.

:::note
Este módulo exige NixOS. Para sistemas que não são NixOS (macOS, outras distros Linux), use `nix profile install` e o fluxo padrão do CLI acima.
:::

### Adicionar o input do flake

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

### Configuração mínima

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

Pronto. `nixos-rebuild switch` cria o usuário `hermes`, gera `config.yaml`, conecta os secrets e inicia o gateway — um serviço de longa duração que conecta o agent às plataformas de messaging (Telegram, Discord, etc.) e escuta mensagens recebidas.

:::warning Secrets são obrigatórios
A linha `environmentFiles` acima assume que você tem [sops-nix](https://github.com/Mic92/sops-nix) ou [agenix](https://github.com/ryantm/agenix) configurado. O arquivo deve conter pelo menos uma chave de provider LLM (ex.: `OPENROUTER_API_KEY=sk-or-...`). Veja [Gerenciamento de secrets](#secrets-management) para a configuração completa. Se você ainda não tem um gerenciador de secrets, pode usar um arquivo simples como ponto de partida — só garanta que não seja legível por todos:

```bash
echo "OPENROUTER_API_KEY=sk-or-your-key" | sudo install -m 0600 -o hermes /dev/stdin /var/lib/hermes/env
```

```nix
services.hermes-agent.environmentFiles = [ "/var/lib/hermes/env" ];
```
:::

:::tip addToSystemPackages
Definir `addToSystemPackages = true` faz duas coisas: coloca o CLI `hermes` no PATH do sistema **e** define `HERMES_HOME` em todo o sistema, para que o CLI interativo compartilhe estado (sessões, skills, cron) com o serviço gateway. Sem isso, executar `hermes` no seu shell cria um diretório `~/.hermes/` separado.
:::

### CLI com suporte a container

:::info
Quando `container.enable = true` e `addToSystemPackages = true`, **todo** comando `hermes` no host é roteado automaticamente para o container gerenciado. Isso significa que sua sessão interativa do CLI roda dentro do mesmo ambiente do serviço gateway — com acesso a todos os pacotes e ferramentas instalados no container.

- O roteamento é transparente: `hermes chat`, `hermes sessions list`, `hermes version`, etc. executam no container por baixo dos panos
- Todas as flags do CLI são repassadas como estão
- Se o container não estiver rodando, o CLI tenta novamente por um instante (5s com spinner para uso interativo, 10s em silêncio para scripts) e então falha com um erro claro — sem fallback silencioso
- Para desenvolvedores trabalhando no codebase do hermes, defina `HERMES_DEV=1` para ignorar o roteamento do container e executar o checkout local diretamente

Defina `container.hostUsers` para criar um symlink `~/.hermes` para o diretório de estado do serviço, para que o CLI no host e o container compartilhem sessões, config e memórias:

```nix
services.hermes-agent = {
  container.enable = true;
  container.hostUsers = [ "your-username" ];
  addToSystemPackages = true;
};
```

Usuários listados em `hostUsers` são adicionados automaticamente ao grupo `hermes` para acesso às permissões de arquivo.

**Usuários de Podman:** O serviço NixOS executa o container como root. Usuários de Docker obtêm acesso via o socket do grupo `docker`, mas containers rootful do Podman exigem sudo. Conceda sudo sem senha para o seu runtime de container:

```nix
security.sudo.extraRules = [{
  users = [ "your-username" ];
  commands = [{
    command = "/run/current-system/sw/bin/podman";
    options = [ "NOPASSWD" ];
  }];
}];
```

O CLI detecta automaticamente quando sudo é necessário e o usa de forma transparente. Sem isso, você precisará executar `sudo hermes chat` manualmente.
:::

### Verificar se está funcionando

Depois de `nixos-rebuild switch`, confira se o serviço está rodando:

```bash
# Check service status
systemctl status hermes-agent

# Watch logs (Ctrl+C to stop)
journalctl -u hermes-agent -f

# If addToSystemPackages is true, test the CLI
hermes version
hermes config       # shows the generated config
```

### Escolhendo um modo de deployment {#container-mode}

O módulo suporta dois modos, controlados por `container.enable`:

| | **Nativo** (padrão) | **Container** |
|---|---|---|
| Como roda | Serviço systemd hardened no host | Container Ubuntu persistente com `/nix/store` bind-mounted |
| Segurança | `NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp` | Isolamento por container, roda como usuário sem privilégios dentro |
| Agent pode auto-instalar pacotes | Não — apenas ferramentas no PATH fornecido pelo Nix | Sim — instalações com `apt`, `pip`, `npm` persistem entre reinícios |
| Superfície de config | Igual | Igual |
| Quando escolher | Deployments padrão, máxima segurança, reprodutibilidade | Agent precisa de instalação de pacotes em runtime, ambiente mutável, ferramentas experimentais |

Para habilitar o modo container, adicione uma linha:

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
O modo container habilita automaticamente `virtualisation.docker.enable` via `mkDefault`. Se você usa Podman, defina `container.backend = "podman"` e `virtualisation.docker.enable = false`.
:::

---

## Configuração

### Settings declarativas

A opção `settings` aceita um attrset arbitrário que é renderizado como `config.yaml`. Ela suporta deep merge entre múltiplas definições de módulo (via `lib.recursiveUpdate`), então você pode dividir a config entre arquivos:

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

Ambos são deep-merged no tempo de avaliação. Chaves declaradas no Nix sempre prevalecem sobre chaves em um `config.yaml` existente no disco, mas **chaves adicionadas pelo usuário que o Nix não toca são preservadas**. Isso significa que, se o agent ou uma edição manual adicionar chaves como `skills.disabled` ou `streaming.enabled`, elas sobrevivem a `nixos-rebuild switch`.

:::note Nomenclatura de modelos
`settings.model.default` usa o identificador de modelo que o seu provider espera. Com [OpenRouter](https://openrouter.ai) (o padrão), ficam assim: `"anthropic/claude-sonnet-4"` ou `"google/gemini-3-flash"`. Se você usa um provider diretamente (Anthropic, OpenAI), defina `settings.model.base_url` apontando para a API deles e use os IDs nativos de modelo (ex.: `"claude-sonnet-4-20250514"`). Quando nenhum `base_url` está definido, o Hermes usa OpenRouter por padrão.
:::

:::tip Descobrindo chaves de config disponíveis
Execute `nix build .#configKeys && cat result` para ver cada chave folha de config extraída do `DEFAULT_CONFIG` do Python. Você pode colar seu `config.yaml` existente no attrset `settings` — a estrutura mapeia 1:1.
:::

<details>
<summary><strong>Exemplo completo: todas as settings comumente personalizadas</strong></summary>

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

### Escape hatch: traga sua própria config

Se preferir gerenciar `config.yaml` totalmente fora do Nix, use `configFile`:

```nix
services.hermes-agent.configFile = /etc/hermes/config.yaml;
```

Isso ignora `settings` por completo — sem merge, sem geração. O arquivo é copiado como está para `$HERMES_HOME/config.yaml` a cada ativação.

### Cheatsheet de personalização

Referência rápida para o que usuários Nix mais costumam personalizar:

| Eu quero... | Opção | Exemplo |
|---|---|---|
| Mudar o modelo LLM | `settings.model.default` | `"anthropic/claude-sonnet-4"` |
| Usar um endpoint de provider diferente | `settings.model.base_url` | `"https://openrouter.ai/api/v1"` |
| Adicionar API keys | `environmentFiles` | `[ config.sops.secrets."hermes-env".path ]` |
| Dar personalidade ao agent | `${services.hermes-agent.stateDir}/.hermes/SOUL.md` | gerencie o arquivo diretamente |
| Adicionar servidores MCP | `mcpServers.<name>` | Veja [Servidores MCP](#mcp-servers) |
| Habilitar Discord/Telegram/Slack | `extraDependencyGroups` | `[ "messaging" ]` |
| Montar diretórios do host no container | `container.extraVolumes` | `[ "/data:/data:rw" ]` |
| Passar acesso a GPU para o container | `container.extraOptions` | `[ "--gpus" "all" ]` |
| Usar Podman em vez de Docker | `container.backend` | `"podman"` |
| Compartilhar estado entre CLI no host e container | `container.hostUsers` | `[ "sidbin" ]` |
| Disponibilizar ferramentas extras para o agent | `extraPackages` | `[ pkgs.pandoc pkgs.imagemagick ]` |
| Usar uma imagem base customizada | `container.image` | `"ubuntu:24.04"` |
| Sobrescrever o pacote hermes | `package` | `inputs.hermes-agent.packages.${system}.default.override { ... }` |
| Mudar o diretório de estado | `stateDir` | `"/opt/hermes"` |
| Definir o diretório de trabalho do agent | `workingDirectory` | `"/home/user/projects"` |

---

## Gerenciamento de secrets {#secrets-management}

:::danger Nunca coloque API keys em `settings` ou `environment`
Valores em expressões Nix acabam em `/nix/store`, que é legível por todos. Sempre use `environmentFiles` com um gerenciador de secrets.
:::

Tanto `environment` (vars não secretas) quanto `environmentFiles` (arquivos secretos) são merged em `$HERMES_HOME/.env` no momento da ativação (`nixos-rebuild switch`). O Hermes lê esse arquivo a cada startup, então mudanças entram em vigor com `systemctl restart hermes-agent` — sem recriar o container.

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

O arquivo de secrets contém pares chave-valor:

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

### OAuth / seeding de auth

Para plataformas que exigem OAuth (ex.: Discord), use `authFile` para fazer seed das credenciais no primeiro deploy:

```nix
{
  services.hermes-agent = {
    authFile = config.sops.secrets."hermes/auth.json".path;
    # authFileForceOverwrite = true;  # overwrite on every activation
  };
}
```

O arquivo só é copiado se `auth.json` ainda não existir (a menos que `authFileForceOverwrite = true`). Refreshes de token OAuth em runtime são gravados no diretório de estado e preservados entre rebuilds.

---

## Documentos

A opção `documents` instala arquivos no diretório de trabalho do agent (`workingDirectory`, que o agent lê como workspace). O Hermes procura nomes de arquivo específicos por convenção:

- **`USER.md`** — contexto sobre o usuário com quem o agent interage.
- Quaisquer outros arquivos que você colocar aqui ficam visíveis para o agent como arquivos do workspace.

O arquivo de identidade do agent é separado: o Hermes carrega seu `SOUL.md` principal de `$HERMES_HOME/SOUL.md`, que no módulo NixOS é `${services.hermes-agent.stateDir}/.hermes/SOUL.md`. Colocar `SOUL.md` em `documents` apenas cria um arquivo no workspace e não substitui o arquivo principal de persona.

```nix
{
  services.hermes-agent.documents = {
    "USER.md" = ./documents/USER.md;  # path reference, copied from Nix store
  };
}
```

Valores podem ser strings inline ou referências de path. Arquivos são instalados a cada `nixos-rebuild switch`.

---

## Servidores MCP {#mcp-servers}

A opção `mcpServers` configura de forma declarativa servidores [MCP (Model Context Protocol)](https://modelcontextprotocol.io). Cada servidor usa transporte **stdio** (comando local) ou **HTTP** (URL remota).

### Transporte stdio (servidores locais)

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
Variáveis de ambiente em valores `env` são resolvidas a partir de `$HERMES_HOME/.env` em runtime. Use `environmentFiles` para injetar secrets — nunca coloque tokens diretamente na config Nix.
:::

### Transporte HTTP (servidores remotos)

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

Defina `auth = "oauth"` para servidores que usam OAuth 2.1. O Hermes implementa o fluxo PKCE completo — descoberta de metadata, registro dinâmico de client, troca de token e refresh automático.

```nix
{
  services.hermes-agent.mcpServers.my-oauth-server = {
    url = "https://mcp.example.com/mcp";
    auth = "oauth";
  };
}
```

Tokens ficam armazenados em `$HERMES_HOME/mcp-tokens/<server-name>.json` e persistem entre reinícios e rebuilds.

<details>
<summary><strong>Autorização OAuth inicial em servidores headless</strong></summary>

A primeira autorização OAuth exige um fluxo de consentimento no browser. Em um deployment headless, o Hermes imprime a URL de autorização em stdout/logs em vez de abrir um browser.

**Opção A: Bootstrap interativo** — execute o fluxo uma vez via `docker exec` (container) ou `sudo -u hermes` (nativo):

```bash
# Container mode
docker exec -it hermes-agent \
  hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth

# Native mode
sudo -u hermes HERMES_HOME=/var/lib/hermes/.hermes \
  hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth
```

O container usa `--network=host`, então o listener de callback OAuth em `127.0.0.1` fica acessível a partir do browser no host.

**Opção B: Pre-seed de tokens** — complete o fluxo em uma workstation e copie os tokens:

```bash
hermes mcp add my-oauth-server --url https://mcp.example.com/mcp --auth oauth
scp ~/.hermes/mcp-tokens/my-oauth-server{,.client}.json \
    server:/var/lib/hermes/.hermes/mcp-tokens/
# Ensure: chown hermes:hermes, chmod 0600
```

</details>

### Sampling (requisições LLM iniciadas pelo servidor)

Alguns servidores MCP podem solicitar completions LLM do agent:

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

## Modo gerenciado

Quando o hermes roda via o módulo NixOS, os seguintes comandos CLI são **bloqueados** com um erro descritivo apontando para `configuration.nix`:

| Comando bloqueado | Motivo |
|---|---|
| `hermes setup` | A config é declarativa — edite `settings` na sua config Nix |
| `hermes config edit` | A config é gerada a partir de `settings` |
| `hermes config set <key> <value>` | A config é gerada a partir de `settings` |
| `hermes gateway install` | O serviço systemd é gerenciado pelo NixOS |
| `hermes gateway uninstall` | O serviço systemd é gerenciado pelo NixOS |

Isso evita drift entre o que o Nix declara e o que está no disco. A detecção usa dois sinais:

1. **Variável de ambiente `HERMES_MANAGED=true`** — definida pelo serviço systemd, visível ao processo gateway
2. **Arquivo marcador `.managed`** em `HERMES_HOME` — definido pelo script de ativação, visível a shells interativos (ex.: `docker exec -it hermes-agent hermes config set ...` também é bloqueado)

Para mudar a configuração, edite sua config Nix e execute `sudo nixos-rebuild switch`.

---

## Arquitetura do container

:::info
Esta seção só é relevante se você usa `container.enable = true`. Pule para deployments em modo nativo.
:::

Quando o modo container está habilitado, o hermes roda dentro de um container Ubuntu persistente com o binário construído pelo Nix bind-mounted read-only a partir do host:

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

O binário construído pelo Nix funciona dentro do container Ubuntu porque `/nix/store` é bind-mounted — ele traz seu próprio interpretador e todas as dependências, então não há dependência das bibliotecas de sistema do container. O entrypoint do container resolve via um symlink `current-package`: `/data/current-package/bin/hermes gateway run --replace`. Em `nixos-rebuild switch`, apenas o symlink é atualizado — o container continua rodando.

### O que persiste em cada evento

| Evento | Container recriado? | `/data` (estado) | `/home/hermes` | Camada gravável (`apt`/`pip`/`npm`) |
|---|---|---|---|---|
| `systemctl restart hermes-agent` | Não | Persiste | Persiste | Persiste |
| `nixos-rebuild switch` (mudança de código) | Não (symlink atualizado) | Persiste | Persiste | Persiste |
| Reinício do host | Não | Persiste | Persiste | Persiste |
| `nix-collect-garbage` | Não (GC root) | Persiste | Persiste | Persiste |
| Mudança de imagem (`container.image`) | **Sim** | Persiste | Persiste | **Perdido** |
| Mudança de volume/opções | **Sim** | Persiste | Persiste | **Perdido** |
| Mudança em `environment`/`environmentFiles` | Não | Persiste | Persiste | Persiste |

O container só é recriado quando seu **hash de identidade** muda. O hash cobre: versão do schema, imagem, `extraVolumes`, `extraOptions` e o script de entrypoint. Mudanças em variáveis de ambiente, settings, documentos ou no próprio pacote hermes **não** disparam recriação.

:::warning Perda da camada gravável
Quando o hash de identidade muda (upgrade de imagem, novos volumes, novas opções de container), o container é destruído e recriado a partir de um pull novo de `container.image`. Quaisquer pacotes de `apt install`, `pip install` ou `npm install` na camada gravável são perdidos. O estado em `/data` e `/home/hermes` é preservado (são bind mounts).

Se o agent depende de pacotes específicos, considere incluí-los em uma imagem customizada (`container.image = "my-registry/hermes-base:latest"`) ou scriptar a instalação no SOUL.md do agent.
:::

### Proteção com GC root

O script `preStart` cria um GC root em `${stateDir}/.gc-root` apontando para o pacote hermes atual. Isso impede que `nix-collect-garbage` remova o binário em execução. Se o GC root quebrar de alguma forma, reiniciar o serviço o recria.

---

## Plugins

O módulo NixOS suporta instalação declarativa de plugins — sem precisar de `hermes plugins install` imperativo.

### Plugins de diretório (`extraPlugins`)

Para plugins que são apenas uma árvore de código com `plugin.yaml` + `__init__.py` (ex.: [hermes-lcm](https://github.com/stephenschoettler/hermes-lcm)):

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

Plugins são symlinkados em `$HERMES_HOME/plugins/` no momento da ativação. O Hermes os descobre via seu scan normal de diretórios. Remover um plugin da lista e executar `nixos-rebuild switch` remove o symlink.

### Plugins por entry point (`extraPythonPackages`)

Para plugins empacotados via pip que registram via `[project.entry-points."hermes_agent.plugins"]` (ex.: [rtk-hermes](https://github.com/ogallotti/rtk-hermes)):

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

O `site-packages` do pacote é adicionado ao PYTHONPATH no wrapper do hermes. `importlib.metadata` descobre o entry point no início da sessão.

### Grupos de dependências opcionais (`extraDependencyGroups`)

Para extras opcionais declarados no `pyproject.toml` do hermes-agent, use `extraDependencyGroups` para incluí-los no venv sealed no build time. Isso é necessário para qualquer extra que não esteja no conjunto `[all]` padrão — no Nix, instalação em runtime no store read-only não é possível.

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

Isso é resolvido pelo uv junto com as dependências core — sem patch de PYTHONPATH, sem risco de colisão. Grupos disponíveis:

| Grupo | O que habilita |
|-------|-----------------|
| `messaging` | Discord, Telegram, Slack |
| `matrix` | Matrix/Element (mautrix com criptografia; apenas Linux) |
| `dingtalk` | DingTalk |
| `feishu` | Feishu/Lark |
| `voice` | Speech-to-text local (faster-whisper) |
| `edge-tts` | Provider Edge TTS |
| `tts-premium` | ElevenLabs TTS |
| `anthropic` | SDK nativo Anthropic (não necessário via OpenRouter) |
| `bedrock` | AWS Bedrock (boto3) |
| `azure-identity` | Auth Azure Entra ID |
| `honcho` | Provider de memória Honcho |
| `hindsight` | Provider de memória Hindsight |
| `modal` | Backend de terminal Modal |
| `daytona` | Backend de terminal Daytona |
| `exa` | Busca web Exa |
| `firecrawl` | Busca web Firecrawl |
| `fal` | Geração de imagem FAL |

Ou use os pacotes flake pré-construídos `#messaging` ou `#full` em vez de config por extra (veja [Início rápido](#quick-start-any-nix-user)).

**Quando usar qual:**

| Necessidade | Opção |
|------|--------|
| Habilitar um extra opcional do pyproject.toml | `extraDependencyGroups` |
| Adicionar um plugin Python externo fora do pyproject.toml | `extraPythonPackages` |
| Adicionar um binário de sistema (pandoc, jq, etc.) | `extraPackages` |
| Adicionar uma árvore de código de plugin baseada em diretório | `extraPlugins` |

### Combinando ambos

Um plugin de diretório com dependências Python de terceiros precisa das duas opções:

```nix
services.hermes-agent = {
  extraPlugins = [ my-plugin-src ];          # plugin source
  extraPythonPackages = [ pkgs.python312Packages.redis ];  # its Python dep
  extraPackages = [ pkgs.redis ];            # system binary it needs
};
```

### Usando o overlay

Flakes externos podem sobrescrever o pacote diretamente:

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

### Configuração de plugins

Plugins ainda precisam ser habilitados em `config.yaml`. Adicione-os via settings declarativas:

```nix
services.hermes-agent.settings.plugins.enabled = [
  "hermes-lcm"
  "rtk-rewrite"
];
```

:::note
Uma verificação de colisão em build time impede que pacotes de plugin sombreiem dependências core do hermes. Se um plugin fornece um pacote que já está no venv sealed, `nixos-rebuild` falha com um erro claro.
:::

---

## Desenvolvimento

### Dev shell

O flake fornece um shell de desenvolvimento com Python 3.12, uv, Node.js e todas as ferramentas de runtime:

```bash
cd hermes-agent
nix develop

# Shell provides:
#   - Python 3.12 + uv (deps installed into .venv on first entry)
#   - Node.js 22, ripgrep, git, openssh, ffmpeg on PATH
#   - Stamp-file optimization: re-entry is near-instant if deps haven't changed

hermes setup
hermes chat
```

### direnv (recomendado)

O `.envrc` incluído ativa o dev shell automaticamente:

```bash
cd hermes-agent
direnv allow    # one-time
# Subsequent entries are near-instant (stamp file skips dep install)
```

### Flake checks

O flake inclui verificação em build time que roda no CI e localmente:

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
<summary><strong>O que cada check verifica</strong></summary>

| Check | O que testa |
|---|---|
| `package-contents` | Binários `hermes` e `hermes-agent` existem e `hermes version` executa |
| `entry-points-sync` | Cada entry em `[project.scripts]` no `pyproject.toml` tem um binário wrapped no pacote Nix |
| `cli-commands` | `hermes --help` expõe subcomandos `gateway` e `config` |
| `managed-guard` | `HERMES_MANAGED=true hermes config set ...` imprime o erro NixOS |
| `bundled-skills` | Diretório de skills existe, contém arquivos SKILL.md, `HERMES_BUNDLED_SKILLS` está definido no wrapper |
| `config-roundtrip` | 7 cenários de merge: instalação nova, override Nix, preservação de chaves do usuário, merge misto, merge aditivo MCP, deep merge aninhado, idempotência |

</details>

---

## Referência de opções

### Core

| Option | Type | Default | Description |
|---|---|---|---|
| `enable` | `bool` | `false` | Habilita o serviço hermes-agent |
| `package` | `package` | `hermes-agent` | Pacote hermes-agent a usar |
| `user` | `str` | `"hermes"` | Usuário de sistema |
| `group` | `str` | `"hermes"` | Grupo de sistema |
| `createUser` | `bool` | `true` | Cria usuário/grupo automaticamente |
| `stateDir` | `str` | `"/var/lib/hermes"` | Diretório de estado (pai de `HERMES_HOME`) |
| `workingDirectory` | `str` | `"${stateDir}/workspace"` | Diretório de trabalho do agent |
| `addToSystemPackages` | `bool` | `false` | Adiciona o CLI `hermes` ao PATH do sistema e define `HERMES_HOME` em todo o sistema |

### Configuration

| Option | Type | Default | Description |
|---|---|---|---|
| `settings` | `attrs` (deep-merged) | `{}` | Config declarativa renderizada como `config.yaml`. Suporta aninhamento arbitrário; múltiplas definições são merged via `lib.recursiveUpdate` |
| `configFile` | `null` or `path` | `null` | Path para um `config.yaml` existente. Substitui `settings` por completo se definido |

### Secrets & Environment

| Option | Type | Default | Description |
|---|---|---|---|
| `environmentFiles` | `listOf str` | `[]` | Paths para arquivos env com secrets. Merged em `$HERMES_HOME/.env` no momento da ativação |
| `environment` | `attrsOf str` | `{}` | Vars de ambiente não secretas. **Visível no Nix store** — não coloque secrets aqui |
| `authFile` | `null` or `path` | `null` | Seed de credenciais OAuth. Copiado apenas no primeiro deploy |
| `authFileForceOverwrite` | `bool` | `false` | Sempre sobrescreve `auth.json` a partir de `authFile` na ativação |

### Documents

| Option | Type | Default | Description |
|---|---|---|---|
| `documents` | `attrsOf (either str path)` | `{}` | Arquivos do workspace. Chaves são nomes de arquivo, valores são strings inline ou paths. Instalados em `workingDirectory` na ativação |

### MCP Servers

| Option | Type | Default | Description |
|---|---|---|---|
| `mcpServers` | `attrsOf submodule` | `{}` | Definições de servidores MCP, merged em `settings.mcp_servers` |
| `mcpServers.<name>.command` | `null` or `str` | `null` | Comando do servidor (transporte stdio) |
| `mcpServers.<name>.args` | `listOf str` | `[]` | Argumentos do comando |
| `mcpServers.<name>.env` | `attrsOf str` | `{}` | Variáveis de ambiente para o processo do servidor |
| `mcpServers.<name>.url` | `null` or `str` | `null` | URL do endpoint do servidor (transporte HTTP/StreamableHTTP) |
| `mcpServers.<name>.headers` | `attrsOf str` | `{}` | Headers HTTP, ex.: `Authorization` |
| `mcpServers.<name>.auth` | `null` or `"oauth"` | `null` | Método de autenticação. `"oauth"` habilita OAuth 2.1 PKCE |
| `mcpServers.<name>.enabled` | `bool` | `true` | Habilita ou desabilita este servidor |
| `mcpServers.<name>.timeout` | `null` or `int` | `null` | Timeout de chamada de tool em segundos (padrão: 120) |
| `mcpServers.<name>.connect_timeout` | `null` or `int` | `null` | Timeout de conexão em segundos (padrão: 60) |
| `mcpServers.<name>.tools` | `null` or `submodule` | `null` | Filtragem de tools (listas `include`/`exclude`) |
| `mcpServers.<name>.sampling` | `null` or `submodule` | `null` | Config de sampling para requisições LLM iniciadas pelo servidor |

### Service Behavior

| Option | Type | Default | Description |
|---|---|---|---|
| `extraArgs` | `listOf str` | `[]` | Args extras para `hermes gateway` |
| `extraPackages` | `listOf package` | `[]` | Pacotes extras disponíveis para o agent. Adicionados ao profile per-user do usuário hermes para que comandos de terminal, skills e cron jobs os vejam |
| `extraPlugins` | `listOf package` | `[]` | Pacotes de plugin de diretório para symlink em `$HERMES_HOME/plugins/`. Cada um deve conter `plugin.yaml` |
| `extraPythonPackages` | `listOf package` | `[]` | Pacotes Python adicionados ao PYTHONPATH para descoberta de plugins por entry point. Construa com `python312Packages` |
| `extraDependencyGroups` | `listOf str` | `[]` | Extras opcionais do pyproject.toml para incluir no venv sealed (ex.: `["hindsight"]`). Resolvido pelo uv — sem colisões |
| `restart` | `str` | `"always"` | Política systemd `Restart=` |
| `restartSec` | `int` | `5` | Valor systemd `RestartSec=` |

### Container

| Option | Type | Default | Description |
|---|---|---|---|
| `container.enable` | `bool` | `false` | Habilita modo container OCI |
| `container.backend` | `enum ["docker" "podman"]` | `"docker"` | Runtime de container |
| `container.image` | `str` | `"ubuntu:24.04"` | Imagem base (puxada em runtime) |
| `container.extraVolumes` | `listOf str` | `[]` | Mounts de volume extras (`host:container:mode`) |
| `container.extraOptions` | `listOf str` | `[]` | Args extras passados para `docker create` |
| `container.hostUsers` | `listOf str` | `[]` | Usuários interativos que recebem um symlink `~/.hermes` para o stateDir do serviço e são adicionados automaticamente ao grupo `hermes` |

---

## Layout de diretórios

### Modo nativo

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

### Modo container

Mesmo layout, montado no container:

| Container path | Host path | Mode | Notes |
|---|---|---|---|
| `/nix/store` | `/nix/store` | `ro` | Binário Hermes + todas as deps Nix |
| `/data` | `/var/lib/hermes` | `rw` | Todo estado, config, workspace |
| `/home/hermes` | `${stateDir}/home` | `rw` | Home persistente do agent — `pip install --user`, caches de tools |
| `/usr`, `/usr/local`, `/tmp` | (writable layer) | `rw` | Instalações `apt`/`pip`/`npm` — persistem entre reinícios, perdidas na recriação |

---

## Atualização

```bash
# Update the flake input (run from the directory containing flake.nix)
cd /etc/nixos && nix flake update hermes-agent

# Rebuild
sudo nixos-rebuild switch
```

No modo container, o symlink `current-package` é atualizado e o agent pega o novo binário no restart. Sem recriação de container, sem perda de pacotes instalados.

---

## Solução de problemas

:::tip Usuários de Podman
Todos os comandos `docker` abaixo funcionam da mesma forma com `podman`. Substitua conforme necessário se você definiu `container.backend = "podman"`.
:::

### Logs do serviço

```bash
# Both modes use the same systemd unit
journalctl -u hermes-agent -f

# Container mode: also available directly
docker logs -f hermes-agent
```

### Inspeção do container

```bash
systemctl status hermes-agent
docker ps -a --filter name=hermes-agent
docker inspect hermes-agent --format='{{.State.Status}}'
docker exec -it hermes-agent bash
docker exec hermes-agent readlink /data/current-package
docker exec hermes-agent cat /data/.container-identity
```

### Forçar recriação do container

Se precisar resetar a camada gravável (Ubuntu limpo):

```bash
sudo systemctl stop hermes-agent
docker rm -f hermes-agent
sudo rm /var/lib/hermes/.container-identity
sudo systemctl start hermes-agent
```

### Verificar se os secrets foram carregados

Se o agent inicia mas não consegue autenticar com o provider LLM, confira se o arquivo `.env` foi merged corretamente:

```bash
# Native mode
sudo -u hermes cat /var/lib/hermes/.hermes/.env

# Container mode
docker exec hermes-agent cat /data/.hermes/.env
```

### Verificação do GC root

```bash
nix-store --query --roots $(docker exec hermes-agent readlink /data/current-package)
```

### Problemas comuns

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot save configuration: managed by NixOS` | Guards do CLI ativos | Edite `configuration.nix` e execute `nixos-rebuild switch` |
| `No adapter available for discord` (ou telegram/slack) | Deps de messaging ausentes no venv Nix sealed | Instale a variante `#messaging`: `nix profile install ...#messaging`. Para o módulo NixOS: `extraDependencyGroups = [ "messaging" ]`. Confira `journalctl -u hermes-agent` por `FeatureUnavailable` ou `requirements not met` para o erro subjacente. |
| Container recriado inesperadamente | `extraVolumes`, `extraOptions` ou `image` mudaram | Esperado — a camada gravável reseta. Reinstale pacotes ou use uma imagem customizada |
| `hermes version` mostra versão antiga | Container não reiniciado | `systemctl restart hermes-agent` |
| Permission denied em `/var/lib/hermes` | Diretório de estado é `0750 hermes:hermes` | Use `docker exec` ou `sudo -u hermes` |
| `nix-collect-garbage` removeu o hermes | GC root ausente | Reinicie o serviço (preStart recria o GC root) |
| `no container with name or ID "hermes-agent"` (Podman) | Container rootful do Podman invisível para usuário comum | Adicione sudo sem senha para podman (veja a seção [Modo container](#container-mode)) |
| `unable to find user hermes` | Container ainda iniciando (entrypoint ainda não criou o usuário) | Aguarde alguns segundos e tente de novo — o CLI tenta novamente automaticamente |
| Tool adicionada via `extraPackages` não encontrada no terminal | Exige `nixos-rebuild switch` para atualizar o profile per-user | Rebuild e restart: `nixos-rebuild switch && systemctl restart hermes-agent` |
