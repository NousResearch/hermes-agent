---
sidebar_position: 2
title: "Profiles"
description: "Execute múltiplos agentes Hermes independentes na mesma máquina — cada um com config, API keys, memória, sessões e gateway próprios."
---

# Profiles: Running Multiple Agents {#profiles-running-multiple-agents}

Execute múltiplos agentes Hermes independentes na mesma máquina — cada um com config, API keys, memória, sessões, skills e estado de gateway próprios.

## What are profiles? {#what-are-profiles}

Um profile é um diretório home Hermes separado. Cada profile recebe seu próprio diretório contendo seu próprio `config.yaml`, `.env`, `SOUL.md`, memórias, sessões, skills, cron jobs e state database. Profiles permitem executar agentes separados para propósitos diferentes — um assistente de código, um bot pessoal, um agente de pesquisa — sem misturar o estado do Hermes.

Quando você cria um profile, ele automaticamente se torna seu próprio comando. Crie um profile chamado `coder` e você imediatamente tem `coder chat`, `coder setup`, `coder gateway start`, etc.

## Quick start {#quick-start}

```bash
hermes profile create coder       # cria profile + alias de comando "coder"
coder setup                       # configura API keys e model
coder chat                        # inicia o chat
```

É isso. `coder` agora é seu próprio profile Hermes com config, memória e estado próprios.

## Creating a profile {#creating-a-profile}

:::tip
Setup mais rápido: execute `hermes setup --portal` dentro do novo profile para conectar models + tools de uma vez. Veja [Nous Portal](/integrations/nous-portal).
:::

### Blank profile {#blank-profile}

```bash
hermes profile create mybot
```

Cria um profile novo com skills bundled semeadas. Execute `mybot setup` para configurar API keys, model e tokens de gateway.

Se você planeja usar este profile como worker kanban (ou quer que o orchestrator kanban roteie trabalho para ele), passe `--description "<role>"` na criação para o orchestrator saber no que ele é bom:

```bash
hermes profile create researcher --description "Reads source code and external docs, writes findings."
```

Você também pode definir ou auto-gerar a descrição depois com `hermes profile describe` — veja o [guia Kanban](./features/kanban#auto-vs-manual-orchestration) para o modelo de roteamento completo.

### Clone config only (`--clone`) {#clone-config-only-clone}

```bash
hermes profile create work --clone
```

Copia o `config.yaml`, `.env`, `SOUL.md` e skills do seu profile atual para o novo profile. Mesmas API keys, model e capacidades, mas sessões e memória novas. Edite `~/.hermes/profiles/work/.env` para API keys diferentes, ou `~/.hermes/profiles/work/SOUL.md` para uma personalidade diferente.

### Clone everything (`--clone-all`) {#clone-everything-clone-all}

```bash
hermes profile create backup --clone-all
```

Copia **tudo** — config, API keys, personalidade, todas as memórias, skills, cron jobs, plugins. Um snapshot funcional completo. Histórico por profile é excluído (histórico de sessão, `state.db`, `backups/`, `state-snapshots/`, `checkpoints/`) — estes pertencem ao profile de origem e podem chegar a dezenas de GB. Para backup completo incluindo histórico, use `hermes profile export` ou `hermes backup`.

### Clone from a specific profile {#clone-from-a-specific-profile}

```bash
hermes profile create work --clone-from coder
```

`--clone-from <source>` seleciona o profile de origem diretamente e implica um clone de config/skills/SOUL. Combine com `--clone-all` quando quiser uma cópia completa desse profile de origem:

```bash
hermes profile create work-backup --clone-from coder --clone-all
```

:::tip Honcho memory + profiles
Quando Honcho está habilitado, operações de clone criam automaticamente um AI peer dedicado para o novo profile enquanto compartilham o mesmo user workspace. Cada profile constrói suas próprias observações e identidade. Veja [Honcho — Multi-agent / Profiles](./features/memory-providers.md#honcho) para detalhes.
:::

## Using profiles {#using-profiles}

### Command aliases {#command-aliases}

Todo profile recebe automaticamente um alias de comando em `~/.local/bin/<name>`:

```bash
coder chat                    # chat com o agente coder
coder setup                   # configura as settings do coder
coder gateway start           # inicia o gateway do coder
coder doctor                  # verifica a saúde do coder
coder skills list             # lista as skills do coder
coder config set model.default anthropic/claude-sonnet-4
```

O alias funciona com todo subcomando hermes — é apenas `hermes -p <name>` por baixo dos panos.

### The `-p` flag {#the-p-flag}

Você também pode direcionar um profile explicitamente com qualquer comando:

```bash
hermes -p coder chat
hermes --profile=coder doctor
hermes chat -p coder -q "hello"    # funciona em qualquer posição
```

### Sticky default (`hermes profile use`) {#sticky-default-hermes-profile-use}

```bash
hermes profile use coder
hermes chat                   # agora direciona para coder
hermes tools                  # configura as tools do coder
hermes profile use default    # voltar
```

Define um padrão para comandos `hermes` simples direcionarem aquele profile. Como `kubectl config use-context`.

### Knowing where you are {#knowing-where-you-are}

A CLI sempre mostra qual profile está ativo:

- **Prompt**: `coder ❯` em vez de `❯`
- **Banner**: Mostra `Profile: coder` na inicialização
- **`hermes profile`**: Mostra nome do profile atual, path, model, status do gateway

## Profiles vs workspaces vs sandboxing {#profiles-vs-workspaces-vs-sandboxing}

Profiles são frequentemente confundidos com workspaces ou sandboxes, mas são coisas diferentes:

- Um **profile** dá ao Hermes seu próprio diretório de estado: `config.yaml`, `.env`, `SOUL.md`, sessões, memória, logs, cron jobs e estado de gateway.
- Um **workspace** ou **diretório de trabalho** é onde os comandos de terminal começam. Isso é controlado separadamente por `terminal.cwd`.
- Um **sandbox** é o que limita acesso ao filesystem. Profiles **não** fazem sandbox do agente.

No backend de terminal `local` padrão, o agente ainda tem o mesmo acesso ao filesystem da sua conta de usuário. Um profile não impede que ele acesse pastas fora do diretório do profile.

Se quiser que um profile inicie em uma pasta de projeto específica, defina um `terminal.cwd` absoluto explícito no `config.yaml` daquele profile:

```yaml
terminal:
  backend: local
  cwd: /absolute/path/to/project
```

Usar `cwd: "."` no backend local significa "o diretório de onde o Hermes foi lançado", não "o diretório do profile".

Também note:

- `SOUL.md` pode guiar o model, mas não impõe um limite de workspace.
- Mudanças em `SOUL.md` têm efeito limpo em uma sessão nova. Sessões existentes podem ainda estar usando o estado antigo do prompt.
- Perguntar ao model "em qual diretório você está?" não é um teste confiável de isolamento. Se precisar de um diretório inicial previsível para tools, defina `terminal.cwd` explicitamente.

## Running gateways {#running-gateways}

Cada profile executa seu próprio gateway como processo separado com seu próprio bot token:

```bash
coder gateway start           # inicia o gateway do coder
assistant gateway start       # inicia o gateway do assistant (processo separado)
```

### Different bot tokens {#different-bot-tokens}

Cada profile tem seu próprio arquivo `.env`. Configure um bot token Telegram/Discord/Slack diferente em cada:

```bash
# Editar tokens do coder
nano ~/.hermes/profiles/coder/.env

# Editar tokens do assistant
nano ~/.hermes/profiles/assistant/.env
```

### Safety: token locks {#safety-token-locks}

Se dois profiles usarem acidentalmente o mesmo bot token, o segundo gateway será bloqueado com um erro claro nomeando o profile conflitante. Suportado para Telegram, Discord, Slack, WhatsApp e Signal.

### Persistent services {#persistent-services}

```bash
coder gateway install         # cria serviço systemd/launchd hermes-gateway-coder
assistant gateway install     # cria serviço hermes-gateway-assistant
```

Cada profile recebe seu próprio nome de serviço. Eles executam independentemente.

:::note Dentro da imagem Docker oficial
Gateways por profile são supervisionados pelo [s6-overlay](https://github.com/just-containers/s6-overlay) (PID 1 no container), então `hermes profile create <name>` registra automaticamente um slot de serviço s6 em `/run/service/gateway-<name>/`. `hermes -p <name> gateway start/stop/restart` despacha para `s6-svc` em vez de gerar um processo bare — crashes são auto-reiniciados e `docker restart` preserva o conjunto de gateways que estava executando. Veja [Supervisão de gateway por profile](/user-guide/docker#per-profile-gateway-supervision) para detalhes.
:::

## Configuring profiles {#configuring-profiles}

Cada profile tem seu próprio:

- **`config.yaml`** — model, provider, toolsets, todas as settings
- **`.env`** — API keys, bot tokens
- **`SOUL.md`** — personalidade e instruções

```bash
coder config set model.default anthropic/claude-sonnet-4
echo "You are a focused coding assistant." > ~/.hermes/profiles/coder/SOUL.md
```

Se quiser que este profile trabalhe em um projeto específico por padrão, também defina seu próprio `terminal.cwd`:

```bash
coder config set terminal.cwd /absolute/path/to/project
```

### From the dashboard {#from-the-dashboard}

O [web dashboard](features/web-dashboard.md#managing-multiple-profiles)
é uma superfície em nível de máquina que pode gerenciar config, API
keys, skills, MCPs e model de **qualquer** profile via o seletor de profile
na sidebar — sem dashboard por profile necessário. `coder dashboard` roteia
para o dashboard da máquina com o profile `coder` pré-selecionado. A aba Chat
do dashboard também segue o seletor, gerando uma conversa sob o home
do profile selecionado.

Nota: "Set as active" na página Profiles do dashboard é o padrão sticky
para **execuções futuras de CLI/gateway** (mesmo que `hermes profile use`) —
para editar um profile a partir do dashboard, use o seletor.

## Updating {#updating}

`hermes update` puxa o código uma vez (compartilhado) e sincroniza novas skills bundled para **todos** os profiles automaticamente:

```bash
hermes update
# → Code updated (12 commits)
# → Skills synced: default (up to date), coder (+2 new), assistant (+2 new)
```

Skills modificadas pelo usuário nunca são sobrescritas.

## Managing profiles {#managing-profiles}

```bash
hermes profile list           # mostra todos os profiles com status
hermes profile show coder     # info detalhada de um profile
hermes profile rename coder dev-bot   # renomeia (atualiza alias + serviço)
hermes profile export coder   # exporta para coder.tar.gz
hermes profile import coder.tar.gz   # importa de archive
```

## Deleting a profile {#deleting-a-profile}

```bash
hermes profile delete coder
```

Isso para o gateway, remove o serviço systemd/launchd, remove o alias de comando e exclui todos os dados do profile. Você será solicitado a digitar o nome do profile para confirmar.

Use `--yes` para pular confirmação: `hermes profile delete coder --yes`

:::note
Você não pode excluir o profile padrão (`~/.hermes`). Para remover tudo, use `hermes uninstall`.
:::

## Tab completion {#tab-completion}

```bash
# Bash
eval "$(hermes completion bash)"

# Zsh
eval "$(hermes completion zsh)"
```

Adicione a linha ao seu `~/.bashrc` ou `~/.zshrc` para completion persistente. Completa nomes de profile após `-p`, subcomandos de profile e comandos de top-level.

## How it works {#how-it-works}

Profiles usam a variável de ambiente `HERMES_HOME`. Quando você executa `coder chat`, o wrapper script define `HERMES_HOME=~/.hermes/profiles/coder` antes de lançar o hermes. Como 119+ arquivos no codebase resolvem paths via `get_hermes_home()`, o estado Hermes escopa automaticamente para o diretório do profile — config, sessões, memória, skills, state database, PID do gateway, logs e cron jobs.

Isso é separado do diretório de trabalho do terminal. A execução de tools começa de `terminal.cwd` (ou o diretório de launch quando `cwd: "."` no backend local), não automaticamente de `HERMES_HOME`.

Em instalações no host, subprocessos de tool mantêm o `HOME` real do seu usuário OS por padrão para
credenciais CLI existentes sob `~` continuarem funcionando entre profiles. Dados do profile são
isolados por `HERMES_HOME`, não por mudar `HOME`. Backends de container ainda usam
`{HERMES_HOME}/home` para estado persistente de tool, e usuários de host que precisam de config
de tool estrita por profile podem optar com `terminal.home_mode: profile`.

Isso significa duas coisas fáceis de confundir:

- `HERMES_HOME` é o limite do profile. Controla config Hermes, `.env`,
  memória, sessões, skills, logs, cron jobs, estado de gateway e outros dados
  Hermes.
- `HOME` é o home do sistema operacional/usuário que CLIs externas esperam. Em instalações
  no host, o Hermes o mantém como o home real do usuário por padrão para tools como
  `git`, `ssh`, `gh`, `az`, `npm`, Claude Code e Codex encontrarem as mesmas
  credenciais que usam no seu shell normal.

O tradeoff é que profiles no host compartilham estado CLI em nível de usuário normal por padrão.
Se precisar de identidades CLI separadas por profile, defina `terminal.home_mode:
profile` no `config.yaml` daquele profile. Nesse modo o Hermes lança subprocessos
de tool com `HOME={HERMES_HOME}/home`; você então precisa inicializar ou linkar
`~/.ssh`, `~/.gitconfig`, `~/.config/gh`, auth de cloud CLI,
auth Claude/Codex, estado npm e arquivos similares dentro desse home do profile.

O Hermes também expõe `HERMES_REAL_HOME` para subprocessos para scripts ainda encontrarem
o home real da conta quando `home_mode: profile` está ativo.

O profile padrão é simplesmente o próprio `~/.hermes`. Nenhuma migração necessária — instalações existentes funcionam de forma idêntica.

## Sharing profiles as distributions {#sharing-profiles-as-distributions}

Um profile que você construiu em uma máquina pode ser empacotado como um **repositório git** e instalado com um comando em outra máquina — sua própria workstation, o laptop de um colega ou o ambiente de um usuário da comunidade. O pacote compartilhado inclui SOUL, config, skills, cron jobs e conexões MCP. Credenciais, memórias e sessões permanecem por máquina.

```bash
# Instalar um agente completo de um repo git
hermes profile install github.com/you/research-bot --alias

# Atualizar depois quando o autor lançar uma nova versão (mantém suas memórias + .env)
hermes profile update research-bot
```

Veja **[Profile Distributions: Share a Whole Agent](./profile-distributions.md)** para o guia completo — autoria, publicação, semântica de update, modelo de segurança e casos de uso.
