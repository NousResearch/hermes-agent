---
sidebar_position: 15
title: "Web Dashboard"
description: "Painel de administração no browser para gerenciar configuração, API keys, servidores MCP, pairing de mensagens, webhooks, gateway, memória, credenciais, sessões, logs, analytics, cron jobs e skills"
---

# Web Dashboard {#web-dashboard}

O web dashboard é uma UI no browser para gerenciar sua instalação Hermes Agent. Em vez de editar arquivos YAML ou rodar comandos CLI, você pode configurar settings, gerenciar API keys e monitorar sessões numa interface web limpa.

:::tip
Auth em hosted mode usa Nous Portal OAuth; se também quer que o dashboard fale com backend real, `hermes setup --portal` configura model e tool gateway. Veja [Nous Portal](/integrations/nous-portal).
:::

## Início rápido {#quick-start}

```bash
hermes dashboard
```

Isso inicia um servidor web local e abre `http://127.0.0.1:9119` no seu browser. O dashboard roda inteiramente na sua máquina — nenhum dado sai do localhost.

### Opções {#options}

| Flag | Padrão | Descrição |
|------|---------|-------------|
| `--port` | `9119` | Porta em que o servidor web roda |
| `--host` | `127.0.0.1` | Endereço de bind |
| `--no-open` | — | Não abrir o browser automaticamente |
| `--insecure` | off | Permite bind em hosts não-localhost (**PERIGOSO** — expõe API keys na rede; combine com firewall e auth forte) |
| `--isolated` | off | Quando lançado de um profile nomeado (`worker dashboard`), roda um servidor dedicado por profile em vez de rotear para o dashboard da máquina |

```bash
# Custom port
hermes dashboard --port 8080

# Bind to all interfaces (use with caution on shared networks)
hermes dashboard --host 0.0.0.0

# Start without opening browser
hermes dashboard --no-open
```

## Gerenciando múltiplos profiles {#managing-multiple-profiles}

O dashboard é uma superfície de gerenciamento em **nível de máquina**: um servidor gerencia
cada [profile](../profiles.md) na máquina. Um seletor de profile na
barra lateral (visível sempre que existe mais de um profile) decide qual
profile as páginas de gerenciamento leem e escrevem — Config, API Keys, Skills,
MCP, Models e a aba Chat seguem todos ele. Enquanto um profile diferente do
próprio dashboard estiver selecionado, um banner âmbar nomeia o profile gerenciado
para que o alvo de escrita nunca seja ambíguo.

A seleção fica na URL (`?profile=<name>`), então deep links como
`http://127.0.0.1:9119/skills?profile=worker` abrem com o seletor
pré-selecionado e sobrevivem ao refresh.

Lançar o dashboard a partir de um alias de profile roteia para o dashboard da
máquina em vez de iniciar um segundo servidor:

```bash
worker dashboard
# → already running: opens the browser at ?profile=worker
# → not running:     starts the machine dashboard with "worker" preselected
```

Passe `--isolated` para optar por fora e rodar um servidor dedicado escopado àquele
profile (comportamento pré-unificação — útil se você expõe deliberadamente
dashboards de profiles diferentes com auth diferente).

A aba **Chat** também segue o seletor: um chat escopado gera seu filho PTY
com o `HERMES_HOME` do profile selecionado, então a conversa roda
com o model, skills, memória e histórico de sessão daquele profile. Trocar
de profile inicia uma sessão de terminal nova.

O que permanece por profile e *não* é absorvido pelo seletor: processos de
gateway (gerencie com `hermes -p <name> gateway …`), o banco de sessões de
cada profile e schedulers cron (a página Cron já agrega
entre profiles com seu próprio filtro).

## Pré-requisitos {#prerequisites}

A instalação padrão do `hermes-agent` não inclui a stack HTTP nem o helper PTY — são extras opcionais. O **web dashboard** precisa de FastAPI e Uvicorn (extra `web`). A aba **Chat** também precisa de `ptyprocess` para gerar o TUI embutido atrás de um pseudo-terminal (extra `pty` em POSIX). Instale ambos com:

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[web,pty]"
```

O extra `web` puxa FastAPI/Uvicorn; `pty` puxa `ptyprocess` (POSIX) ou `pywinpty` (Windows nativo — note que o TUI embutido em si ainda exige WSL). `cd ~/.hermes/hermes-agent && uv pip install -e ".[all]"` inclui ambos os extras e é o caminho mais fácil se você também quer messaging/voice/etc.

Quando você roda `hermes dashboard` sem as dependências, ele informa o que instalar. Se o frontend ainda não foi buildado e `npm` está disponível, ele builda automaticamente no primeiro launch.

A aba Chat faz parte de todo launch de `hermes dashboard` — o painel de chat no browser (rodando o TUI sobre PTY/WebSocket) está sempre disponível, sem flag extra.

## Páginas {#pages}

### Status {#status}

A landing page mostra uma visão ao vivo da sua instalação:

- **Versão do agent** e data de release
- **Status do gateway** — running/stopped, PID, plataformas conectadas e seu estado
- **Sessões ativas** — contagem de sessões ativas nos últimos 5 minutos
- **Sessões recentes** — lista das 20 sessões mais recentes com model, contagem de mensagens, uso de tokens e preview da conversa

A página de status atualiza automaticamente a cada 5 segundos.

### Chat {#chat}

A aba **Chat** embute o TUI completo do Hermes (a mesma interface que você obtém com `hermes --tui`) diretamente no browser. Tudo que você pode fazer no TUI de terminal — slash commands, seletor de model, cards de tool call, streaming markdown, prompts clarify/sudo/approval, theming de skin — funciona de forma idêntica aqui, porque o dashboard roda o binário real do TUI e renderiza sua saída ANSI via [xterm.js](https://xtermjs.org/) com o renderer WebGL para layout de células pixel-perfect.

**Como funciona:**

- `/api/pty` abre um WebSocket autenticado com o token de sessão do dashboard
- O servidor gera `hermes --tui` atrás de um pseudo-terminal POSIX
- Teclas viajam ao PTY; saída ANSI volta ao browser em stream
- O renderer WebGL do xterm.js pinta cada célula em uma grade de pixels inteiros; mouse tracking (SGR 1006), caracteres largos (Unicode 11) e glifos de box-drawing renderizam nativamente
- Redimensionar a janela do browser redimensiona o TUI via o addon `@xterm/addon-fit`

**Retomar uma sessão existente:** na aba **Sessions**, clique no ícone play (▶) ao lado de qualquer sessão. Isso vai para `/chat?resume=<id>` e lança o TUI com `--resume`, carregando o histórico completo.

**Seletor de sessão (trilho direito):** a aba Chat traz sua própria lista de conversas estilo ChatGPT em um trilho fino à direita do terminal, para você trocar conversas sem sair da página. O trilho empilha o seletor de model no topo e a lista de sessões logo abaixo; o terminal ocupa a maior parte da tela. A lista mostra suas sessões mais recentes do profile ativo — título (fallback para preview de mensagem), tempo relativo da última atividade, contagem de mensagens e canal de origem para sessões não-CLI. Clique em qualquer linha para retomá-la no lugar (o terminal respawna com o histórico daquela conversa); a sessão ativa fica destacada. **New chat** inicia sessão nova, e um controle de refresh repuxa a lista. O trilho é read-only para troca — delete, rename, export e limpeza em massa continuam na aba **Sessions**. Em telas estreitas ele vira um painel slide-over.

**Pré-requisitos:**

- Node.js (mesmo requisito que `hermes --tui`; o bundle TUI é buildado no primeiro launch)
- `ptyprocess` — instalado pelo extra `pty` (`cd ~/.hermes/hermes-agent && uv pip install -e ".[web,pty]"`, ou `[all]` cobre ambos)
- Kernel POSIX (Linux, macOS ou WSL2). O painel de terminal `/chat` especificamente precisa de um PTY POSIX — Python Windows nativo não tem equivalente, então numa instalação Windows nativa o resto do dashboard (sessões, jobs, métricas, editor de config) funciona, mas a aba `/chat` mostra um banner pedindo WSL2 para esse recurso.

Feche a aba do browser e o PTY é encerrado limpo no servidor. Reabrir gera sessão nova.

Para apontar o [Hermes Desktop](#connecting-hermes-desktop-to-a-remote-backend) a um dashboard rodando em outra máquina em vez do backend bundled próprio, veja a seção de backend remoto abaixo.

### Conectando o Hermes Desktop a um backend remoto {#connecting-hermes-desktop-to-a-remote-backend}

O Hermes Desktop normalmente lança seu próprio backend local, mas também pode anexar a um dashboard rodando em máquina remota (VM, homelab, etc.) via **Settings → Gateway → Remote gateway**. Esta é a fonte mais comum de relatos de "Desktop diz que o backend está pronto mas o chat nunca funciona", porque a checagem de prontidão do Desktop verifica menos do que a conexão de chat ao vivo realmente precisa.

:::info Pré-requisito: um `hermes dashboard` deve estar rodando no host remoto
O "backend remoto" ao qual o Desktop conecta **é** um processo `hermes dashboard` rodando na máquina remota — o mesmo servidor que esta página documenta. Ele precisa estar up e alcançável antes que qualquer passo abaixo importe; o Desktop anexa a ele, não o inicia por você. Mantenha rodando sob `systemd`/`tmux`/etc. para sobreviver a logout e reboots. O **gateway** (Telegram/Discord/Slack/etc.) é um processo long-running *separado* — inicie independentemente se você depende de canais de messaging; não é o que o app desktop conecta.
:::

A sonda "remote backend is ready" do Desktop só atinge `GET /api/status`, que é endpoint público — responde assim que *qualquer* dashboard está rodando no host. A conexão de chat ao vivo é um **WebSocket separado** para `/api/ws` (e `/api/pty`), e esse socket passa por duas checagens a mais que a sonda de status nunca toca:

1. **Você precisa estar autenticado.** Quando o dashboard está bound a endereço não-loopback, ele aciona o auth gate. Proteja com username e password (o [provedor username/password](#usernamepassword-provider-no-oauth-idp) bundled); o Desktop faz sign in uma vez e reutiliza a sessão resultante para o WebSocket via ticket de uso único. Sem provedor configurado, um dashboard não-loopback **falha fechado na inicialização**.
2. **O host de bind deve permitir o client e bater com o header Host.** Um bind loopback (`127.0.0.1`) só aceita clients loopback, então uma máquina remota é rejeitada na camada de socket independente de credenciais. Faça bind a endereço não-loopback (`--host 0.0.0.0`) para o peer-IP guard deixar o client remoto passar. A Remote URL que você digita no Desktop deve alcançar o dashboard pelo mesmo host ao qual ele fez bind — o guard de DNS-rebinding exige que o header Host bata.

#### Configuração do dashboard remoto {#remote-dashboard-setup}

Defina username e password, então rode o dashboard bound a endereço alcançável. Para um serviço `systemd`:

```ini
[Service]
EnvironmentFile=%h/.hermes/.env
ExecStart=/path/to/venv/bin/python -m hermes_cli.main dashboard \
    --host 0.0.0.0 --port 9119 --no-open
```

com `~/.hermes/.env` contendo:

```bash
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
HERMES_DASHBOARD_BASIC_AUTH_SECRET=<32+ random bytes; openssl rand -base64 32>
```

Então no Desktop digite a **Remote URL** (ex.: `http://VM_IP:9119`) e **Sign in** com aquele username e password. Veja a seção [provedor username/password](#usernamepassword-provider-no-oauth-idp) para a superfície completa de configuração.

:::tip Verifique que o gate está ligado antes de tentar o Desktop de novo
De qualquer máquina, confira que o dashboard anuncia o provedor username/password:

```bash
curl -s http://VM_IP:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["basic"]
```

- `auth_required: true` e `"basic"` na lista de providers → o fluxo **Sign in** do Desktop funcionará.
- `auth_required: false` → o bind é loopback, ou o gate não acionou. Faça bind a endereço não-loopback.
- `auth_required: true` mas sem provider `"basic"` → as env vars username/password não foram carregadas. Corrija isso primeiro.
:::

Se `/api/status` mostra o gate ligado com provider `"basic"` e o Desktop *ainda* falha ao conectar após sign in, o problema passou do setup básico — pegue um `desktop.log` fresco (Settings → Gateway → Open logs) mais os logs do dashboard na mesma janela de retry e procure o close code de `/api/ws` (4403 = chat WS rejeitado pelo request guard, ex. Host/peer mismatch; 4401 = o ticket WS não autenticou).

### Config {#config}

Um editor baseado em formulário para `config.yaml`. Todos os 150+ campos de configuração são auto-descobertos de `DEFAULT_CONFIG` e organizados em categorias com abas:

![Config admin page — section filters on the left, auto-discovered fields on the right](/img/dashboard/admin-config.png)


- **model** — model padrão, provider, base URL, settings de reasoning
- **terminal** — backend (local/docker/ssh/modal), timeout, preferências de shell
- **display** — skin, progresso de ferramentas, exibição de resume, settings do spinner
- **agent** — max iterations, gateway timeout, service tier
- **delegation** — limites de subagent, reasoning effort
- **memory** — seleção de provider, settings de injeção de contexto
- **approvals** — modo de aprovação de comandos perigosos (smart/manual/off)
- E mais — cada seção de config.yaml tem campos de formulário correspondentes

Campos com valores válidos conhecidos (terminal backend, skin, approval mode, etc.) renderizam como dropdowns. Booleanos renderizam como toggles. Todo o resto é input de texto.

**Ações:**

- **Save** — grava mudanças em `config.yaml` imediatamente
- **Reset to defaults** — reverte todos os campos aos valores padrão (não salva até você clicar Save)
- **Export** — baixa a config atual como JSON
- **Import** — envia arquivo JSON de config para substituir os valores atuais

:::tip
Mudanças de config entram em vigor na próxima sessão do agent ou restart do gateway. O web dashboard edita o mesmo arquivo `config.yaml` que `hermes config set` e o gateway leem.
:::

### API Keys {#api-keys}

Gerencie o arquivo `.env` onde API keys e credenciais são armazenadas. Keys são agrupadas por categoria:

- **LLM Providers** — OpenRouter, Anthropic, OpenAI, DeepSeek, etc.
- **Tool API Keys** — Browserbase, Firecrawl, Tavily, ElevenLabs, etc.
- **Messaging Platforms** — tokens de bot Telegram, Discord, Slack, etc.
- **Agent Settings** — env vars não-secretas como `API_SERVER_ENABLED`

Cada key mostra:
- Se está definida atualmente (com preview redacted do valor)
- Descrição do que serve
- Link para a página de signup/key do provider
- Campo de input para definir ou atualizar o valor
- Botão delete para remover

Keys avançadas/raramente usadas ficam ocultas por padrão atrás de um toggle.

### Sessions {#sessions}

Navegue e inspecione todas as sessões do agent. Cada linha mostra título da sessão, ícone da plataforma de origem (CLI, Telegram, Discord, Slack, cron), nome do model, contagem de mensagens, contagem de tool calls e há quanto tempo esteve ativa. Sessões ao vivo são marcadas com badge pulsante.

- **Search** — busca full-text em todo o conteúdo de mensagens via FTS5. Resultados mostram snippets destacados e auto-scroll até a primeira mensagem correspondente quando expandido.
- **Stats** — barra resumo mostra total de sessões, quantas estão ativas no store, contagem arquivada, total de mensagens e breakdown por origem.
- **Expand** — clique numa sessão para carregar o histórico completo de mensagens. Mensagens são coloridas por role (user, assistant, system, tool) e renderizadas como Markdown com syntax highlighting.
- **Tool calls** — mensagens assistant com tool calls mostram blocos colapsáveis com nome da função e argumentos JSON.
- **Rename** — defina ou limpe o título da sessão inline (ícone de lápis).
- **Export** — baixe uma sessão (metadata + histórico completo de mensagens) como JSON (ícone de download).
- **Prune** — o botão "Prune old sessions" no header apaga sessões encerradas com mais de N dias.
- **Delete** — remova uma sessão e seu histórico de mensagens com o ícone de lixeira.

![Sessions admin page — stats bar, prune, and per-row rename / export / delete](/img/dashboard/admin-sessions.png)

### Logs {#logs}

Veja arquivos de log do agent, gateway e errors com filtragem e tail ao vivo.

- **File** — alterne entre arquivos de log `agent`, `errors` e `gateway`
- **Level** — filtre por nível de log: ALL, DEBUG, INFO, WARNING ou ERROR
- **Component** — filtre por componente de origem: all, gateway, agent, tools, cli ou cron
- **Lines** — escolha quantas linhas exibir (50, 100, 200 ou 500)
- **Auto-refresh** — toggle de tail ao vivo que faz poll de novas linhas de log a cada 5 segundos
- **Color-coded** — linhas de log são coloridas por severidade (vermelho para errors, amarelo para warnings, dim para debug)

### Analytics {#analytics}

Analytics de uso e custo computados do histórico de sessões. Selecione um período (7, 30 ou 90 dias) para ver:

- **Summary cards** — total de tokens (input/output), percentual de cache hit, custo total estimado ou real, e contagem total de sessões com média diária
- **Daily token chart** — gráfico de barras empilhadas mostrando uso de tokens input e output por dia, com tooltips hover mostrando breakdowns e custo
- **Daily breakdown table** — data, contagem de sessões, tokens input, tokens output, taxa de cache hit e custo para cada dia
- **Per-model breakdown** — tabela mostrando cada model usado, sua contagem de sessões, uso de tokens e custo estimado

### Cron {#cron}

Crie e gerencie cron jobs agendados que rodam prompts do agent em schedule recorrente.

- **Create** — preencha nome (opcional), prompt, expressão cron (ex.: `0 9 * * *`) e alvo de delivery (local, Telegram, Discord, Slack ou email)
- **Job list** — cada job mostra nome, preview do prompt, expressão de schedule, badge de estado (enabled/paused/error), alvo de delivery, hora da última execução e próxima execução
- **Pause / Resume** — alterne um job entre estados ativo e pausado
- **Edit** — abra modal pré-preenchido para mudar prompt, schedule, nome ou alvo de delivery do job
- **Trigger now** — execute um job imediatamente fora do schedule normal
- **Delete** — remova permanentemente um cron job

### Profiles {#profiles}

Crie e gerencie [profiles](../profiles.md) — instâncias Hermes isoladas com config, skills e sessões próprias.

- **Profile cards** — cada um mostra model/provider, contagem de skills, estado do gateway, descrição e badges (active, default, alias)
- **Create** — nome + clone-from-default / clone-everything / no-bundled-skills opcionais, descrição e model; a página dedicada Profile Builder (`/profiles/new`) oferece o fluxo completo (model, MCPs, skills)
- **Manage skills & tools** — pula para a página Skills escopada àquele profile (define o seletor de profile da sidebar)
- **Set as active** — inverte o default sticky que **execuções futuras de CLI/gateway** pegam (igual a `hermes profile use`). Isso *não* muda o que o dashboard gerencia — isso é trabalho do seletor de profile
- **Edit model / description / SOUL** — editores inline escrevendo naquele profile
- **Rename / Delete** — apenas profiles nomeados

### Skills {#skills}

Navegue, busque e alterne skills e toolsets instalados, e instale novos do hub. Skills são carregadas de `~/.hermes/skills/` e agrupadas por categoria.

- **Search** — filtre skills e toolsets instalados por nome, descrição ou categoria
- **Category filter** — clique nas pills de categoria para estreitar a lista (ex.: MLOps, MCP, Red Teaming, AI)
- **Toggle** — habilite ou desabilite skills individuais com um switch. Mudanças entram em vigor na próxima sessão.
- **Toolsets** — view separada mostra toolsets built-in (operações de arquivo, web browsing, etc.) com status active/inactive, requisitos de setup e lista de ferramentas incluídas
- **Browse hub** — terceira view busca o skill hub em todas as fontes (igual a `hermes skills search`), instala qualquer resultado por identificador com log de install ao vivo, e oferece botão "Update all" para atualizar skills instaladas.

![Skills admin page — the Browse hub view: search, install, and update](/img/dashboard/admin-skills-hub.png)

### MCP {#mcp}

Gerencie servidores [MCP](/integrations/mcp) sem o CLI. O mesmo bloco `mcp_servers`
em `config.yaml` que `hermes mcp` lê.

**Seus servidores MCP:**

- **Add** — registre um servidor HTTP/SSE (URL) ou stdio (command + args), com variáveis de ambiente `KEY=VALUE` opcionais para servidores stdio
- **Enable / disable** — alterne um servidor ligado ou desligado sem deletá-lo. Um servidor desabilitado permanece na config para você reabilitar depois. Entra em vigor no próximo restart do gateway.
- **Test** — conecte a um servidor, liste suas ferramentas e desconecte — verifica a conexão antes do agent depender dela
- **Remove** — delete um servidor da config
- Valores de env com formato de secret são redacted na view de lista

**Catalog:** navegue os servidores MCP aprovados pela Nous (catálogo bundled `optional-mcps/`)
e instale qualquer um com um clique. Entradas que precisam de API keys
pedem inline; os valores vão para `.env`. Este é o mesmo catálogo que
`hermes mcp catalog` / `hermes mcp install` usam.

![MCP admin page — your servers with enable/disable toggles, plus the install catalog](/img/dashboard/admin-mcp.png)

### Webhooks {#webhooks}

Gerencie [webhook subscriptions](/user-guide/messaging/webhooks) dinâmicas. A
plataforma webhook deve estar habilitada nas settings de messaging primeiro; a página mostra uma
dica quando não está.

- **Create** — nome, descrição, filtro de eventos, alvo de delivery, modo direct-delivery opcional e prompt do agent. Na criação a página exibe a route URL e o HMAC secret de uso único para copiar.
- **Enable / disable** — alterne uma subscription ligada ou desligada. Routes desabilitadas permanecem no arquivo de subscriptions mas o gateway rejeita seus eventos incoming (403). O gateway hot-reload o arquivo, então a mudança entra em vigor no próximo evento — sem restart.
- **List** — cada subscription mostra URL, eventos e alvo de delivery
- **Delete** — remova uma subscription

![Webhooks admin page — subscriptions with enable/disable toggles](/img/dashboard/admin-webhooks.png)

### Pairing {#pairing}

Aprove e revogue usuários de messaging sem o CLI — como um admin remoto
onboarda usuários Telegram/Discord/etc. num gateway paired. Paridade total com
`hermes pairing`.

- **Pending requests** — cada um mostra plataforma, code, user e idade, com botão Approve
- **Approved users** — cada um mostra plataforma e user, com botão Revoke
- **Clear pending** — descarte todos os pairing codes pendentes

![Pairing admin page](/img/dashboard/admin-pairing.png)

### Channels {#channels}

Conecte o Hermes a qualquer plataforma de messaging pelo browser — paridade total com
`hermes setup gateway`. A página lista todo canal suportado (Telegram,
Discord, Slack, Matrix, Mattermost, WhatsApp, Signal, BlueBubbles/iMessage,
Email, SMS/Twilio, DingTalk, Feishu/Lark, WeCom, WeChat, QQ Bot, Yuanbao, mais
API server e webhook endpoints) com status de conexão ao vivo.

- **Configure** — abra formulário por plataforma com exatamente os campos que aquele canal precisa (bot token, app token, server URL, allowlist, etc.). Secrets renderizam como password inputs e são armazenados redacted; deixar campo em branco mantém o valor existente. Campos obrigatórios são marcados e validados. Link "Setup guide" aponta para a doc de credenciais da plataforma.
- **Enable / disable** — alterne um canal ligado ou desligado. A credencial permanece no disco; só o estado active muda.
- **Test** — verifique se o canal está configurado, habilitado e reportando conexão ao vivo do gateway.
- **Restart gateway** — credenciais são escritas em `~/.hermes/.env` e a flag enabled em `config.yaml`; o gateway conecta cada canal habilitado no próximo restart, que você pode disparar direto da página.

![Channels admin page — every messaging platform with status, enable toggles, and per-platform setup forms](/img/dashboard/admin-channels.png)

### System {#system}

Painel de administração consolidado para operações em toda a instalação:

- **Host** — stats de sistema ao vivo: OS / kernel, arquitetura, hostname, versões Python e Hermes, contagem de cores CPU + utilização, memória, uso de disco do Hermes home, uptime e load average. (CPU/memória/disco vêm de `psutil` quando instalado; campos de identidade são sempre mostrados.) A versão Hermes mostra **badge de status de update** (up to date / N commits behind) e botão **Check for updates**. Quando há update disponível numa instalação git ou pip, botão **Update now** abre diálogo de confirmação — mostrando quantos commits você puxará — antes de rodar `hermes update` em background. Em instalações Docker/Nix/Homebrew o dashboard não pode aplicar o update in place, então mostra o comando out-of-band correto.
- **Nous Portal** — status de login, provider de inferência ativo e tabela de roteamento do Tool Gateway (quais ferramentas rodam via Portal vs. localmente), com link para gerenciar sua assinatura. Espelho read-only de `hermes portal`.
- **Skill curator** — status da manutenção de skills em background (active / paused, interval, last run) com pause/resume e botão run-now. Espelha `hermes curator`.
- **Gateway** — inicie, pare e reinicie o messaging gateway, com status ao vivo (running/stopped, PID, state)
- **Memory** — escolha o memory provider externo (ou só built-in) e resete os stores built-in `MEMORY.md` / `USER.md`
- **Credential pool** — adicione e remova as API keys rotativas pelas quais o agent faz round-robin (por provider). Keys são redacted na lista; o valor bruto só chega ao agent.
- **Operations** — rode `doctor`, security audit, crie backup, restaure de arquivo de backup, atualize skills, mostre breakdown do tamanho do system prompt, gere support dump ou migre config de settings aposentadas. Cada um gera ação em background cujo log ao vivo faz stream na página.
- **Checkpoints** — veja o tamanho do shadow store de `/rollback` e faça prune
- **Shell hooks** — liste hooks configurados com status de consent + executable, **create** um hook (event, command, matcher, timeout, com grant de consent opt-in) e remova um. Hooks rodam comandos arbitrários, então o formulário de criação traz aviso de segurança e o hook só dispara após consent concedido.

![System admin page — host stats and Nous Portal status](/img/dashboard/admin-system-top.png)

![System admin page — skill curator, gateway, memory, and credential pool](/img/dashboard/admin-system-curator.png)

![System admin page — operations, checkpoints, and shell hooks](/img/dashboard/admin-system-ops.png)

Criando um shell hook (note o checkbox de consent e o aviso de run-arbitrary-commands):

![New shell hook modal](/img/dashboard/admin-hook-create.png)

:::warning Segurança
O web dashboard lê e escreve seu arquivo `.env`, que contém API keys e secrets. Ele faz bind em `127.0.0.1` por padrão — acessível só da sua máquina local. Se você fizer bind em `0.0.0.0`, qualquer um na sua rede pode ver e modificar suas credenciais. O dashboard não tem autenticação própria.
:::

## Comando slash `/reload` {#reload-slash-command}

O PR do dashboard também adiciona um comando slash `/reload` ao CLI interativo. Depois de mudar API keys via web dashboard (ou editando `.env` diretamente), use `/reload` numa sessão CLI ativa para pegar as mudanças sem reiniciar:

```
You → /reload
  Reloaded .env (3 var(s) updated)
```

Isso re-lê `~/.hermes/.env` no ambiente do processo em execução. Útil quando você adicionou uma nova provider key via dashboard e quer usá-la imediatamente.

## REST API {#rest-api}

O web dashboard expõe uma REST API que o frontend consome. Você também pode chamar esses endpoints diretamente para automação:

:::tip Endpoints escopados por profile
As famílias de endpoints de gerenciamento — `/api/config`, `/api/env`, `/api/skills`,
`/api/tools/toolsets`, `/api/mcp` e `/api/model/{info,options,auxiliary,set}` —
aceitam parâmetro de query opcional `?profile=<name>` (ou `"profile"` no
JSON body para writes) que escopa read/write ao `HERMES_HOME` daquele profile.
Omitido = profile próprio do dashboard. Nomes de profile desconhecidos
retornam `404`. O WebSocket `/api/pty` aceita o mesmo parâmetro para gerar
chat sob o profile selecionado.
:::

### GET /api/status {#get-apistatus}

Retorna versão do agent, status do gateway, estados das plataformas e contagem de sessões ativas.

### GET /api/sessions {#get-apisessions}

Retorna as 20 sessões mais recentes com metadata (model, contagem de tokens, timestamps, preview).

### GET /api/config {#get-apiconfig}

Retorna o conteúdo atual de `config.yaml` como JSON.

### GET /api/config/defaults {#get-apiconfigdefaults}

Retorna os valores padrão de configuração.

### GET /api/config/schema {#get-apiconfigschema}

Retorna um schema descrevendo cada campo de config — type, description, category e select options quando aplicável. O frontend usa isso para renderizar o widget de input correto para cada campo.

### PUT /api/config {#put-apiconfig}

Salva nova configuração. Body: `{"config": {...}}`.

### GET /api/env {#get-apienv}

Retorna todas as variáveis de ambiente conhecidas com status set/unset, valores redacted, descriptions e categories.

### PUT /api/env {#put-apienv}

Define variável de ambiente. Body: `{"key": "VAR_NAME", "value": "secret"}`.

### DELETE /api/env {#delete-apienv}

Remove variável de ambiente. Body: `{"key": "VAR_NAME"}`.

### GET /api/sessions/\{session_id\} {#get-apisessionssession_id}

Retorna metadata de uma sessão.

### GET /api/sessions/\{session_id\}/messages {#get-apisessionssession_idmessages}

Retorna o histórico completo de mensagens de uma sessão, incluindo tool calls e timestamps.

### GET /api/sessions/search {#get-apisessionssearch}

Busca full-text no conteúdo de mensagens. Parâmetro de query: `q`. Retorna IDs de sessão correspondentes com snippets destacados.

### DELETE /api/sessions/\{session_id\} {#delete-apisessionssession_id}

Deleta uma sessão e seu histórico de mensagens.

### GET /api/logs {#get-apilogs}

Retorna linhas de log. Parâmetros de query: `file` (agent/errors/gateway), `lines` (count), `level`, `component`.

### GET /api/analytics/usage {#get-apianalyticsusage}

Retorna uso de tokens, custo e analytics de sessão. Parâmetro de query: `days` (default 30). Resposta inclui breakdowns diários e agregados por model.

### GET /api/cron/jobs {#get-apicronjobs}

Retorna todos os cron jobs configurados com state, schedule e histórico de execução.

### POST /api/cron/jobs {#post-apicronjobs}

Cria novo cron job. Body: `{"prompt": "...", "schedule": "0 9 * * *", "name": "...", "deliver": "local"}`.

### POST /api/cron/jobs/\{job_id\}/pause {#post-apicronjobsjob_idpause}

Pausa um cron job.

### POST /api/cron/jobs/\{job_id\}/resume {#post-apicronjobsjob_idresume}

Retoma um cron job pausado.

### POST /api/cron/jobs/\{job_id\}/trigger {#post-apicronjobsjob_idtrigger}

Dispara imediatamente um cron job fora do schedule.

### DELETE /api/cron/jobs/\{job_id\} {#delete-apicronjobsjob_id}

Deleta um cron job.

### GET /api/skills {#get-apiskills}

Retorna todas as skills com name, description, category e status enabled.

### PUT /api/skills/toggle {#put-apiskillstoggle}

Habilita ou desabilita uma skill. Body: `{"name": "skill-name", "enabled": true}`.

### GET /api/tools/toolsets {#get-apitoolstoolsets}

Retorna todos os toolsets com label, description, lista de tools e status active/configured.

### Admin endpoints {#admin-endpoints}

Estes alimentam as páginas MCP, Channels, Webhooks, Pairing e System. Todos ficam atrás do
mesmo auth gate que o resto de `/api/`.

| Método e caminho | Propósito |
|---------------|---------|
| `GET /api/mcp/servers` | Lista servidores MCP configurados (valores env redacted) |
| `POST /api/mcp/servers` | Adiciona servidor. Body: `{name, url?, command?, args?, env?, auth?}` |
| `POST /api/mcp/servers/{name}/test` | Conecta, lista tools, desconecta |
| `PUT /api/mcp/servers/{name}/enabled` | Habilita / desabilita servidor |
| `DELETE /api/mcp/servers/{name}` | Remove servidor |
| `GET /api/mcp/catalog` | Navega catálogo MCP aprovado pela Nous |
| `POST /api/mcp/catalog/install` | Instala entrada do catálogo (com env obrigatório) |
| `GET /api/messaging/platforms` | Lista todo canal de messaging com status + campos de setup por plataforma |
| `PUT /api/messaging/platforms/{id}` | Configura canal. Body: `{enabled?, env?, clear_env?}` (env escreve em `.env`, enabled em `config.yaml`) |
| `POST /api/messaging/platforms/{id}/test` | Reporta se canal está configurado, habilitado e conectado |
| `GET /api/pairing` | Lista usuários de messaging pending + approved |
| `POST /api/pairing/approve` | Aprova code. Body: `{platform, code}` |
| `POST /api/pairing/revoke` | Revoga user. Body: `{platform, user_id}` |
| `POST /api/pairing/clear-pending` | Descarta todos os codes pending |
| `GET /api/webhooks` | Lista subscriptions + status platform-enabled |
| `POST /api/webhooks` | Cria subscription (retorna secret de uso único) |
| `DELETE /api/webhooks/{name}` | Remove subscription |
| `GET /api/credentials/pool` | Lista keys de rotação pooled (redacted) |
| `POST /api/credentials/pool` | Adiciona key. Body: `{provider, api_key, label?}` |
| `DELETE /api/credentials/pool/{provider}/{index}` | Remove key (índice 1-based) |
| `GET /api/memory` | Provider ativo + providers disponíveis + tamanhos de arquivos built-in |
| `PUT /api/memory/provider` | Seleciona provider (vazio = só built-in) |
| `POST /api/memory/reset` | Reseta memória built-in. Body: `{target: all\|memory\|user}` |
| `POST /api/gateway/start` · `/stop` · `/restart` | Ciclo de vida do gateway (backgrounded) |
| `POST /api/ops/doctor` · `/security-audit` · `/backup` · `/import` | Diagnóstico e manutenção (backgrounded; tail via `/api/actions/{name}/status`) |
| `GET /api/ops/hooks` | Shell hooks configurados + status allowlist |
| `GET /api/ops/checkpoints` · `POST .../prune` | Inspeciona / faz prune do store `/rollback` |
| `POST /api/ops/hooks` · `DELETE /api/ops/hooks` | Cria / remove shell hook (consent-gated) |
| `GET /api/system/stats` | Stats do host — OS, CPU, memória, disco, uptime |
| `GET /api/hermes/update/check` | Reporta disponibilidade de update (commits behind, install method) sem aplicar. Para installs git/pip que estão behind, também retorna lista `commits` (`sha`, `summary`, `author`, `at`) do que mudou. `?force=1` quebra cache de 6h |
| `GET /api/curator` · `PUT .../paused` · `POST .../run` | Status skill-curator + pause/resume + run |
| `GET /api/portal` | Auth Nous Portal + roteamento Tool Gateway (read-only) |
| `POST /api/ops/prompt-size` · `/dump` · `/config-migrate` | Diagnóstico (backgrounded) |
| `PUT /api/webhooks/{name}/enabled` | Habilita / desabilita webhook route |
| `POST /api/skills/hub/install` · `/uninstall` · `/update` | Ações skills hub (backgrounded) |
| `GET /api/skills/hub/search` | Busca skill hub em todas as fontes |
| `GET /api/sessions/stats` | Estatísticas do session store |
| `PATCH /api/sessions/{id}` | Renomeia / arquiva sessão |
| `GET /api/sessions/{id}/export` | Exporta sessão (metadata + messages) como JSON |
| `POST /api/sessions/prune` | Deleta sessões encerradas com mais de N dias |
| `PUT /api/cron/jobs/{id}` | Edita prompt / schedule / name / deliver de cron job |

## Autenticação (modo gated) {#authentication-gated-mode}

Quando o dashboard está bound a endereço público ou não-loopback — qualquer coisa além de `127.0.0.1` / `localhost` — o Hermes Agent aciona um auth gate. Toda requisição deve carregar cookie de sessão verificado ou é redirecionada à página de login. Três providers vêm na caixa:

- **[Username/password](#usernamepassword-provider-no-oauth-idp)** — o caminho mais simples para colocar auth num dashboard self-hosted / on-prem / homelab. Sem identity provider externo. **Use apenas numa rede confiável ou atrás de VPN — não para exposição na internet pública.**
- **[OAuth (Nous Portal)](#default-provider-nous-research)** — para deploys hosted e qualquer dashboard alcançável pela internet pública, e o caminho recomendado para [conexão remota do Hermes Desktop](#connecting-hermes-desktop-to-a-remote-backend). Todo login é verificado contra sua conta Nous, então este é o provider adequado para uso internet-facing.
- **[OIDC self-hosted](#self-hosted-oidc-provider)** — para trazer seu próprio identity provider via OpenID Connect padrão (Keycloak, Auth0, Okta, Google, GitHub via ponte OIDC, etc.). Sem Nous Portal envolvido; adequado para exposição internet pública quando fronteado por servidor OIDC conformante.

Dashboards de operador bound a loopback não são afetados — sem auth, sem página de login.

### Quando o gate aciona {#when-the-gate-engages}

| Flags | Auth gate | Caso de uso |
|-------|-----------|----------|
| `hermes dashboard` (default — bind em `127.0.0.1`) | OFF | Desenvolvimento local |
| `hermes dashboard --host 0.0.0.0` | **ON** | Remoto / produção — proteja com provedor username/password ou OAuth |

O gate está ligado se e somente se:

1. O host de bind não é `127.0.0.1`, `::1`, `localhost` ou `0.0.0.0` E
2. A flag `--insecure` **não** está definida.

:::danger `--insecure` desabilita auth completamente
`--insecure` pula o gate e serve dashboard não autenticado que lê/escreve seu `.env` (API keys, secrets) e pode rodar comandos do agent. **Não use para conexão remota.** Para expor o dashboard a outra máquina, configure o [provedor username/password](#usernamepassword-provider-no-oauth-idp) (ou OAuth) e deixe `--insecure` desligado. A flag existe só como escape hatch de último recurso numa rede single-host totalmente confiável e com firewall.
:::

### Semântica fail-closed {#fail-closed-semantics}

Se o gate acionaria mas **nenhum** `DashboardAuthProvider` está registrado (sem plugin Nous, sem plugin custom), `hermes dashboard` recusa bind com mensagem de erro explícita. Não há fallback "default-deny mas aceita tudo" — um dashboard gated mal configurado nunca inicia.

Quando você roda `hermes dashboard --host 0.0.0.0` **interativamente** (terminal real) e nenhum provider está configurado ainda, o Hermes não só falha — oferece configurar na hora: escolha **username & password** (escreve `dashboard.basic_auth` em `config.yaml` e você está rodando em segundos) ou **OAuth** (aponta para `hermes dashboard register`). Chamadores não-interativos — Docker/s6, CI, runs piped — pulam o prompt e batem no erro fail-closed acima, então deploy unattended nunca inicia sem auth.

### Provedor padrão: Nous Research {#default-provider-nous-research}

O plugin bundled `plugins/dashboard_auth/nous` está **sempre instalado** e auto-carregado. Auto-registra um `DashboardAuthProvider` nomeado `nous` quando um client ID está configurado.

Como todo login é verificado contra Nous Portal e protegido pela sua conta Nous, **o provider Nous é o adequado para expor um dashboard à internet pública.**

#### Registrando um dashboard {#registering-a-dashboard}

Para usar o provider Nous você precisa de OAuth client ID (formato `agent:{id}`). Há duas formas de obter um:

- **CLI — `hermes dashboard register`.** Rode no host onde o dashboard vive. Resolve seu login Nous existente (rode `hermes setup` primeiro se não estiver logado), registra client OAuth self-hosted no Portal e escreve `HERMES_DASHBOARD_OAUTH_CLIENT_ID` em `~/.hermes/.env` por você. Flags opcionais: `--name` (label legível, senão auto-gerado) e `--redirect-uri` (URL de callback HTTPS pública para host internet-facing).

  ```bash
  hermes dashboard register
  # ✓ Registered dashboard "swift_falcon"
  # …writes HERMES_DASHBOARD_OAUTH_CLIENT_ID to ~/.hermes/.env
  ```

- **GUI — página Local Dashboards.** Abra [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards) no Nous Portal para registrar, nomear, gerenciar e revogar dashboards self-hosted pelo browser. Copie o client ID `agent:{id}` resultante para `HERMES_DASHBOARD_OAUTH_CLIENT_ID` (env) ou `dashboard.oauth.client_id` (config.yaml). Também é onde você revoga dashboard registrado via CLI.

#### Configuração {#configuration}

O plugin lê de duas superfícies, com variável de ambiente vencendo quando definida não-vazia:

**`config.yaml`** — superfície canônica:

```yaml
dashboard:
  oauth:
    client_id: agent:01HXYZ…             # required to engage the gate
```

**Variáveis de ambiente** — overrides de operador:

| Env var | Substitui | Formato | Provisionado por |
|---------|-----------|--------|----------------|
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | `dashboard.oauth.client_id` | `agent:{instance_id}` | `hermes dashboard register` |

Pela convenção Hermes Agent (`~/.hermes/.env` é só para API keys / secrets), **`config.yaml` é o lugar recomendado para definir esses valores** para dev local, on-prem e qualquer deploy que você controla diretamente. O caminho de variável de ambiente existe para plataformas de hosting injetarem secrets por deploy com `client_id`s sem ninguém editar `config.yaml` dentro da imagem — esse é seu propósito principal.

Valores de ambiente vazios são tratados como unset, então secret de plataforma provisionado mas não populado não pode shadow acidentalmente entrada válida de `config.yaml`.

Se nenhuma fonte fornece client_id, o plugin reporta o motivo específico e o erro fail-closed de bind do dashboard diz exatamente o que corrigir:

```
Refusing to bind dashboard to 0.0.0.0 — the OAuth auth gate engages on
non-loopback binds, but no auth providers are registered.

Bundled providers reported these issues:
  • nous: HERMES_DASHBOARD_OAUTH_CLIENT_ID is not set (and
    dashboard.oauth.client_id in config.yaml is empty). The Nous Portal
    provisions this env var (shape 'agent:{instance_id}') when it
    deploys a Hermes Agent instance — set it to your provisioned
    client id (either as an env var or under dashboard.oauth.client_id
    in config.yaml), or pass --insecure to skip the OAuth gate entirely.

Or pass --insecure to skip the auth gate (NOT recommended on untrusted
networks).
```

#### Exemplo prático: Nous Research {#worked-example-nous-research}

De uma instalação Hermes logada a um dashboard gated pela Nous em três passos.

**1. Faça login e registre o dashboard.** `hermes dashboard register` usa seu login Nous existente para provisionar client OAuth e escreve `HERMES_DASHBOARD_OAUTH_CLIENT_ID` em `~/.hermes/.env` por você:

```bash
hermes setup            # if you're not already logged into Nous Portal
hermes dashboard register
# ✓ Registered dashboard "swift_falcon"
# …writes HERMES_DASHBOARD_OAUTH_CLIENT_ID to ~/.hermes/.env
```

**2. Rode o dashboard num endereço alcançável.** Bind não-loopback sem `--insecure` aciona o OAuth gate, e o `client_id` recém-escrito ativa o provider `nous`:

```bash
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

**3. Faça login.** Abra `http://<host>:9119/`, você será redirecionado a `/login`. Clique **Sign in with Nous Research** → autentique no Portal → volte ao dashboard autenticado. Verifique o gate de qualquer máquina:

```bash
curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["nous"]
```

`GET /api/auth/me` então retorna a sessão verificada (`provider: nous`). Para host internet-facing, registre com `--redirect-uri https://hermes.example.com/auth/callback` e defina `HERMES_DASHBOARD_PUBLIC_URL` para o callback OAuth resolver à sua URL pública (veja [Public URL override](#public-url-override)).

### Provedor username/password (sem OAuth IDP) {#usernamepassword-provider-no-oauth-idp}

Se você não quer configurar identity provider OAuth — deploy self-hosted "só colocar senha no meu dashboard" — o plugin bundled `plugins/dashboard_auth/basic` registra um `DashboardAuthProvider` nomeado `basic` que autentica com **username e password** em vez de redirect OAuth.

Ele encaixa no mesmo gate que o provider OAuth: o gate aciona em bind não-loopback sem `--insecure`, a página de login renderiza formulário de credenciais para este provider (em vez de botão "Log in with X"), e tudo downstream do login — cookies de sessão, refresh transparente, tickets WS, logout, audit log — é idêntico ao caminho OAuth. Sessões são tokens stateless assinados HMAC que o provider emite, então **não há database nem IDP externo**. Hash de password usa `scrypt` da stdlib (sem dependência third-party).

:::warning Use apenas em redes confiáveis — não na internet pública
O provider username/password é destinado a dashboards self-hosted / on-prem / homelab numa **rede confiável**, ou alcançável só via **VPN**. Protege credencial compartilhada única sem identity provider externo, MFA ou contas por usuário, então **não é adequado para expor dashboard diretamente à internet pública**. Para dashboard internet-facing, use o [provider Nous Research](#default-provider-nous-research) (ou seu próprio [OIDC self-hosted](#self-hosted-oidc-provider) / [OAuth custom](#custom-providers)).
:::

#### Configuração {#configuration-1}

Como o provider Nous, lê de `config.yaml` (canônico) com variáveis de ambiente vencendo quando definidas não-vazias. Ativa só quando `username` mais `password_hash` (preferido) ou `password` estão configurados — senão é no-op, então usuários OAuth e operadores loopback/`--insecure` não são afetados.

**`config.yaml`:**

```yaml
dashboard:
  basic_auth:
    username: admin
    # Preferred — no plaintext at rest. Compute with:
    #   python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"
    password_hash: "scrypt$16384$8$1$…$…"
    # ...or a plaintext password (hashed in-memory at load; less safe at rest):
    # password: "s3cret"
    secret: "<32+ random bytes, base64 or hex>"  # token-signing key
    session_ttl_seconds: 43200                    # optional; access-token lifetime (default 12h)
```

**Overrides de ambiente:**

| Env var | Substitui | Notas |
|---------|-----------|-------|
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | `dashboard.basic_auth.username` | obrigatório para ativar |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` | `dashboard.basic_auth.password_hash` | preferido (sem plaintext at rest) |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | `dashboard.basic_auth.password` | plaintext; **vence `password_hash` da config** para você rotacionar via env |
| `HERMES_DASHBOARD_BASIC_AUTH_SECRET` | `dashboard.basic_auth.secret` | chave de assinatura de token |
| `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS` | `dashboard.basic_auth.session_ttl_seconds` | lifetime do access token |

:::caution Defina `secret` explícito para sessões estáveis
Quando `secret` está vazio, chave de assinatura aleatória por processo é gerada. Isso é ok para processo único, mas significa que **toda sessão é invalidada no restart** e sessões **não abrangem múltiplos workers**. Defina `secret` explícito para deploys multi-worker / que sobrevivem restart.
:::

O endpoint `/auth/password-login` é rate-limited por IP de client (default 10 tentativas/minuto → HTTP 429) e retorna `401 Invalid credentials` genérico único tanto para users desconhecidos quanto passwords errados, então não pode ser usado como oráculo de enumeração de username.

#### Exemplo prático: username/password {#worked-example-usernamepassword}

Do zero a dashboard gated por password numa rede confiável em três passos.

**1. Defina credenciais em `~/.hermes/.env`.** Faça hash da password para não haver plaintext at rest, e defina signing secret estável para sessões sobreviverem restarts:

```bash
# Compute a scrypt hash of your chosen password:
HASH=$(python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('choose-a-strong-password'))")

cat >> ~/.hermes/.env <<EOF
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH=$HASH
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env
```

**2. Rode o dashboard num endereço alcançável.** Bind não-loopback sem `--insecure` aciona o gate, e username + hash ativam o provider `basic`:

```bash
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

**3. Faça login.** Abra `http://<host>:9119/`, você será redirecionado a `/login` — **formulário de credenciais** (não botão "Sign in with X"). Digite `admin` / sua password → chegue ao dashboard autenticado. Verifique o gate de qualquer máquina:

```bash
curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["basic"]
```

`GET /api/auth/me` então retorna a sessão verificada (`provider: basic`). Mantenha atrás de VPN — veja o aviso acima; para host público use [Nous Research](#default-provider-nous-research) ou [OIDC self-hosted](#self-hosted-oidc-provider).

#### Escrevendo seu próprio password provider {#writing-your-own-password-provider}

`basic` é só uma implementação de um extension point. Qualquer plugin pode registrar password provider: defina `supports_password = True` na sua subclasse `DashboardAuthProvider` e implemente `complete_password_login(*, username, password) -> Session` (raise `InvalidCredentialsError` na rejeição, `ProviderError` se backing store estiver down). Os métodos OAuth `start_login` / `complete_login` podem ficar como stubs `NotImplementedError` para provider pure-password. Este é o caminho para LDAP-bind, database de credenciais ou qualquer esquema de auth non-redirect — o framework cuida do formulário, route, cookies e refresh por você.

### Provedor OIDC self-hosted {#self-hosted-oidc-provider}

Se você roda seu próprio identity provider, o plugin bundled `plugins/dashboard_auth/self_hosted` autentica o dashboard contra ele usando **OpenID Connect padrão** — sem código por IDP, sem Nous Portal envolvido. É verificado contra e funciona com qualquer servidor OIDC conformante:

> **Authentik · Keycloak · Zitadel · Authelia · Auth0 · Okta · Google · …**

Como o provider Nous, auto-carrega e só se registra quando configurado, então é no-op para dashboards loopback / `--insecure`.

#### Configuração {#configuration-2}

Configure **issuer** e **client_id** (client PKCE público — sem client secret). O plugin busca `authorization_endpoint`, `token_endpoint` e `jwks_uri` do IDP em `{issuer}/.well-known/openid-configuration`, então você nunca hardcode URLs de endpoint.

**`config.yaml`** — superfície canônica:

```yaml
dashboard:
  oauth:
    provider: self-hosted
    self_hosted:
      issuer: https://auth.example.com/application/o/hermes/   # required
      client_id: hermes-dashboard                              # required
      scopes: "openid profile email"                           # optional (this is the default)
```

**Variáveis de ambiente** — overrides de operador (env vence `config.yaml` quando definido não-vazio; valor vazio é tratado como unset):

| Env var | Substitui | Notas |
|---------|-----------|-------|
| `HERMES_DASHBOARD_OIDC_ISSUER` | `dashboard.oauth.self_hosted.issuer` | URL issuer OIDC — obrigatório |
| `HERMES_DASHBOARD_OIDC_CLIENT_ID` | `dashboard.oauth.self_hosted.client_id` | Public client id — obrigatório |
| `HERMES_DASHBOARD_OIDC_SCOPES` | `dashboard.oauth.self_hosted.scopes` | Default `openid profile email` |

No seu IDP, registre application/client **público** com grant authorization-code + PKCE (S256) e adicione callback do dashboard como redirect URI permitido. O callback é `<dashboard public URL>/auth/callback` (veja [Public URL override](#public-url-override) para como o dashboard deriva URL pública atrás de proxy).

#### O que verifica {#what-it-verifies}

O provider verifica **ID token** OpenID Connect (RS256/ES256) contra `jwks_uri` descoberto, com claims `iss` e `aud` pinned ao `issuer` e `client_id` configurados. Claims OIDC padrão mapeiam para sessão do dashboard:

| Campo da sessão | Claim(s) |
|---------------|----------|
| `user_id` | `sub` (required) |
| `email` | `email` |
| `display_name` | `name` → `preferred_username` → `nickname` → `email` |
| `org_id` | `org_id` / `organization`, senão `groups` joined |

O ID token é o que estabelece identidade — access token é tratado como opaco (spec OIDC não exige JWT). URLs de endpoint devem ser HTTPS (loopback `http://` permitido para IDPs local-dev), e `issuer` anunciado no discovery document deve bater com o configurado (diferença de trailing slash tolerada). Refresh tokens, quando IDP emite, são usados para re-auth silenciosa via grant `refresh_token` padrão; logout chama `revocation_endpoint` RFC 7009 do IDP quando anunciado.

> **Confidential clients** (com `client_secret`) ainda não são suportados — configure client público + PKCE, escolha típica para dashboard browser-facing.

#### Exemplo prático: Keycloak {#worked-example-keycloak}

[Keycloak](https://www.keycloak.org/) é um dos servidores OIDC self-hosted mais fáceis de subir para teste local — roda como container único em dev mode (DB in-memory) e expõe discovery OIDC textbook. Este walkthrough leva do zero a login funcional no dashboard em poucos minutos.

**1. Rode Keycloak com realm pré-configurado.** Salve este export de realm como `realm-hermes.json` — define realm `hermes`, **client PKCE público** (`hermes-dashboard`) e user de teste, todos importados no boot sem cliques na admin UI:

```json
{
  "realm": "hermes",
  "enabled": true,
  "clients": [
    {
      "clientId": "hermes-dashboard",
      "name": "Hermes Agent Dashboard",
      "enabled": true,
      "publicClient": true,
      "standardFlowEnabled": true,
      "protocol": "openid-connect",
      "redirectUris": ["http://localhost:9119/auth/callback"],
      "webOrigins": ["http://localhost:9119"],
      "attributes": { "pkce.code.challenge.method": "S256" }
    }
  ],
  "users": [
    {
      "username": "testuser",
      "enabled": true,
      "emailVerified": true,
      "email": "testuser@example.com",
      "firstName": "Test",
      "lastName": "User",
      "credentials": [
        { "type": "password", "value": "testpassword", "temporary": false }
      ]
    }
  ]
}
```

Inicie (Keycloak 26+), montando esse arquivo no diretório de import:

```bash
docker run --rm -p 8080:8080 \
  -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
  -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin \
  -v "$PWD/realm-hermes.json:/opt/keycloak/data/import/realm-hermes.json:ro" \
  quay.io/keycloak/keycloak:26.0 \
  start-dev --import-realm
```

Quando estiver up, o realm anuncia discovery OIDC padrão em
`http://localhost:8080/realms/hermes/.well-known/openid-configuration` (issuer
`http://localhost:8080/realms/hermes`). Admin console em
`http://localhost:8080/` (`admin` / `admin`).

**2. Aponte o dashboard para ele.** O plugin self-hosted permite issuer loopback `http://` (HTTPS obrigatório para issuer não-loopback), então Keycloak local funciona as-is:

```bash
export HERMES_DASHBOARD_OIDC_ISSUER="http://localhost:8080/realms/hermes"
export HERMES_DASHBOARD_OIDC_CLIENT_ID="hermes-dashboard"
export HERMES_DASHBOARD_PUBLIC_URL="http://localhost:9119"
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

`HERMES_DASHBOARD_PUBLIC_URL` diz ao dashboard que OAuth callback é
`http://localhost:9119/auth/callback` — redirect URI que o realm registrou
acima. Bind em `0.0.0.0` (bind não-loopback) sem `--insecure` é o que
aciona o OAuth gate.

**3. Faça login.** Abra `http://localhost:9119/`, você será redirecionado a `/login`. Clique **Sign in with Self-Hosted OIDC** → autentique no Keycloak como `testuser` / `testpassword` → volte ao dashboard autenticado. Sidebar mostra `Logged in as Test User via self-hosted`, e `GET /api/auth/me` retorna sessão verificada (`provider: self-hosted`, `email: testuser@example.com`).

> Se você fizer bind ou navegar em host/porta diferente, adicione
> `…/auth/callback` daquela origin em **Valid redirect URIs** do client na
> admin console Keycloak (Clients → hermes-dashboard → Settings). O mesmo padrão funciona
> para Authentik, Zitadel, Authelia e outros servidores OIDC — só URL do issuer
> e UI de registro de client diferem.

### Public URL override {#public-url-override}

Por padrão, o dashboard reconstrói OAuth callback URL da requisição — `X-Forwarded-Host` + `X-Forwarded-Proto` + `X-Forwarded-Prefix` (quando uvicorn está configurado com `proxy_headers=True`, que `start_server` habilita sob o gate). Funciona out of the box atrás de reverse proxy que define os três headers corretamente.

Para deploys atrás de reverse proxies que não encaminham esses headers de forma confiável (setups nginx manuais, ingresses on-prem, deploys custom-domain com cadeias proxy parciais), defina `dashboard.public_url` (ou `HERMES_DASHBOARD_PUBLIC_URL`) para a **URL pública completa** pela qual o dashboard é alcançado:

```yaml
dashboard:
  public_url: "https://dashboard.example.com/hermes"
```

Quando definido, OAuth callback URL vira `<public_url>/auth/callback` verbatim — `X-Forwarded-Prefix` é ignorado nesse code path porque operador declarou explicitamente a URL pública. Isso é intencional: empilhar prefix em cima double-prefixaria o caso comum onde prefix já está baked em `public_url`.

Mesma precedência das outras settings do dashboard — env vence `config.yaml`:

| Superfície | Caminho de override | Quando usar |
|---------|---------------|-------------|
| `dashboard.public_url` em `config.yaml` | `HERMES_DASHBOARD_PUBLIC_URL` | Dev local / on-prem (canônico) |
| Env var `HERMES_DASHBOARD_PUBLIC_URL` | — | Secrets plataforma hosting / CI |
| (unset) | — | Default — reconstruir de headers `X-Forwarded-*` |

Validação rejeita valores sem scheme `http://` / `https://`, sem host, ou contendo quote / angle / whitespace / control characters. Valor malformado cai silenciosamente para reconstrução por header para login flow continuar funcionando em vez de despachar user para URL hostil.

> **Nota:** `public_url` override só OAuth callback URL. Flag cookie `Secure` ainda é controlada por `request.url.scheme` (X-Forwarded-Proto sob proxy_headers), então `public_url` `http://` num deploy público TLS-terminated produz cookies non-Secure. Footgun de operador — combine `public_url` com terminação TLS adequada upstream.

### OAuth flow {#oauth-flow}

O provider implementa [Nous Portal OAuth contract v1](https://github.com/NousResearch/nous-account-service/blob/main/docs/agent-dashboard-oauth-contract.md) — authorization-code grant com PKCE (S256):

1. User atinge `/` sem cookie de sessão → gate redireciona a `/login`.
2. Página de login mostra botão "Continue with Nous Research" → `/auth/login?provider=nous`.
3. Servidor guarda state PKCE em cookie curto, redireciona user a `https://portal.nousresearch.com/oauth/authorize?…`.
4. User autentica no Portal, aterrisa em `/auth/callback?code=…&state=…`.
5. Servidor troca code por access token em `POST /api/oauth/token`, verifica assinatura JWT contra JWKS do Portal (`/.well-known/jwks.json`) e define cookie `hermes_session_at`.
6. User é redirecionado a `/` (ou path deep-link original via parâmetro `next=`).

Access tokens têm TTL de 15 minutos. **Não há refresh token no contract v1** — quando token expira, fetch wrapper da SPA detecta envelope 401 e full-page-navigate de volta a `/login` para re-executar flow.

### Cookies definidos {#cookies-set}

| Nome | Lifetime | Notas |
|------|----------|-------|
| `hermes_session_at` | Token TTL (15 min) | HttpOnly, SameSite=Lax, Secure-when-HTTPS |
| `hermes_session_pkce` | 10 min | HttpOnly; guarda PKCE verifier + provider hint durante round trip |
| `hermes_session_rt` | unused in v1 | Reservado forward-compat; não escrito quando `refresh_token` vazio |

Os três são `Path=/` e `SameSite=Lax`. Flag `Secure` é definida quando dashboard é alcançado via HTTPS (detectado via request URL scheme — honra `X-Forwarded-Proto` de terminador TLS upstream sob `proxy_headers=True`).

### Logout {#logout}

Widget da sidebar mostra `Logged in as <user_id…> via nous` com ícone logout. Clicar POSTa `/auth/logout`, que limpa todos cookies dashboard-auth e redireciona de volta a `/login`.

### Audit log {#audit-log}

Todo login start, success, failure e session-verify failure é escrito como linha JSON em `$HERMES_HOME/logs/dashboard-auth.log`. Campos sensíveis (`access_token`, `refresh_token`, `code`, `code_verifier`, `state`, header `Authorization`) são redacted antes de log.

### Custom providers {#custom-providers}

Para plugar provider OAuth não-Nous (ex.: Google, GitHub, OIDC custom), crie plugin que registra `DashboardAuthProvider`:

```python
# ~/.hermes/plugins/dashboard-auth-myidp/__init__.py
from hermes_cli.dashboard_auth import DashboardAuthProvider, Session, LoginStart

class MyIdPProvider(DashboardAuthProvider):
    name = "myidp"
    display_name = "My Identity Provider"

    def start_login(self, *, redirect_uri): ...
    def complete_login(self, *, code, state, code_verifier, redirect_uri): ...
    def verify_session(self, *, access_token): ...
    def refresh_session(self, *, refresh_token): ...
    def revoke_session(self, *, refresh_token): ...

def register(ctx):
    ctx.register_dashboard_auth_provider(MyIdPProvider())
```

Página de login lista todos providers registrados; múltiplos providers podem ser empilhados e user escolhe um em `/login`.

### Auth não-interativa (bearer token) {#non-interactive-bearer-token-auth}

Junto ao login humano interativo (cookies de sessão + refresh), ABC `DashboardAuthProvider` suporta capacidade **não-interativa, service-to-service** via `supports_token = True` + `verify_token(token=...)`. Quando provider opta in, `Authorization: Bearer <token>` inbound é verificado e, em success, `TokenPrincipal` é anexado à requisição (`request.state.token_principal`) para endpoints que provider marca token-authable — sem cookie, sem redirect, sem refresh.

Primeiro consumer bundled é provider **drain** (`plugins/dashboard_auth/drain`): `nous-account-service` provisiona secret por agent via `HERMES_DASHBOARD_DRAIN_SECRET`, e provider verifica bearer tokens inbound contra ele com compare constant-time, registrando `/api/gateway/drain` como token-authable. **Falha fechado** — secret fraco/curto (< 256 bits) é rejeitado no registro e endpoint fica desabilitado; no-op quando env var unset. Knobs comportamentais (`scope`, `min_secret_chars`) vivem sob `dashboard.drain_auth` em `config.yaml`.

Providers custom podem implementar `supports_token`/`verify_token` da mesma forma para expor endpoints machine-authable próprios.

### Verificando que o gate está ligado {#verifying-the-gate-is-on}

```bash
# Quick env-var path.
HERMES_DASHBOARD_OAUTH_CLIENT_ID=agent:test \
  hermes dashboard --host 0.0.0.0

# Or the equivalent via config.yaml (recommended for local dev / on-prem):
#
#   dashboard:
#     oauth:
#       client_id: agent:test
#
# then just:
hermes dashboard --host 0.0.0.0

# Hit /api/status to see the gate state:
curl -s http://127.0.0.1:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["nous"]
```

StatusPage React do dashboard mostra os mesmos campos sob "Web server". AuthWidget na sidebar exibe identidade atual depois que você fez sign in.

## Conectando o Hermes Desktop a um backend remoto {#connecting-hermes-desktop-to-a-remote-backend}

O Hermes Desktop pode dirigir backend Hermes rodando em outra máquina (VPS, home server, Mini atrás de Tailscale). No app isso fica em **Settings → Gateway → Remote gateway**, que pede **Remote URL** e forma de **Sign in**. (Para o app desktop em si — install, settings, chat — veja a página [Hermes Desktop](/user-guide/desktop).)

Você protege o dashboard remoto com um dos auth providers bundled, e o app desktop faz sign in contra qualquer um que o backend anunciar. Para backend alcançável além da sua máquina — VPS, host público, qualquer coisa internet-facing — o provider recomendado é **OAuth (Nous Portal)** (registre com [`hermes dashboard register`](#registering-a-dashboard) e sign in com *Sign in with Nous Research*). O [provedor username/password](#usernamepassword-provider-no-oauth-idp) bundled é a opção mais rápida quando backend está numa LAN confiável ou alcançável só via VPN, mas **não é adequado para exposição direta na internet pública**. Bind do dashboard a endereço não-loopback aciona auth gate; depois de sign in, Desktop reutiliza sessão para chat WebSocket automaticamente — não há token para copiar ou colar.

A receita abaixo usa caminho username/password porque é o mais rápido numa rede confiável; para caminho OAuth veja [Provedor padrão: Nous Research](#default-provider-nous-research).

### No backend (máquina remota) {#on-the-backend-the-remote-machine}

```bash
# 1. Set the dashboard login credentials in ~/.hermes/.env (secrets file, 0600).
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
# Recommended: a stable signing secret so sessions survive restarts.
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env

# 2. Run the dashboard bound to a reachable address. The non-loopback bind
#    engages the auth gate; the username/password provider handles login.
hermes dashboard --no-open --host 0.0.0.0 --port 9119
```

Prefere sem plaintext at rest? Use `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` com hash scrypt — veja [Provedor username/password](#usernamepassword-provider-no-oauth-idp) para superfície completa.

Se você roda dashboard como serviço systemd, `~/.hermes/.env` é picked up automaticamente quando unit tem `EnvironmentFile=%h/.hermes/.env`, então credenciais estão no ambiente no boot.

:::warning
O dashboard lê e escreve seu `.env` (API keys, secrets) e pode rodar comandos do agent. Setup **username/password** mostrado aqui é para rede confiável — nunca exponha dashboard protegido por password diretamente à internet aberta. Coloque atrás de VPN. [Tailscale](https://tailscale.com/) é opção limpa: bind ao tailscale IP da máquina (`--host <tailscale-ip>`) e use `http://<tailscale-ip>:9119` como Remote URL. Só devices na sua tailnet alcançam. Para alcançar backend pela internet pública, use provider **OAuth (Nous Portal)**.
:::

### No Hermes Desktop {#in-hermes-desktop}

**Settings → Gateway → Remote gateway:**

- **Remote URL** — `http://<backend-host>:9119` (prefixos de path como `/hermes` são suportados se você frontear com reverse proxy)
- **Sign in** — app detecta gateway username/password e mostra botão **Sign in**; clique e digite credenciais do passo 1
- **Save and reconnect** — troca shell desktop para backend remoto

Sessão refresh automaticamente e sobrevive restarts quando `HERMES_DASHBOARD_BASIC_AUTH_SECRET` está definido no backend.

### Override por variável de ambiente {#environment-variable-override}

Em vez da setting in-app, você pode apontar desktop a backend com env var antes de lançar. Quando `HERMES_DESKTOP_REMOTE_URL` está definida, ela override URL in-app salva (painel Gateway settings mostra badge "env override" e desabilita edição); você ainda **Sign in** com username e password do painel.

| Env var | Valor |
|---------|-------|
| `HERMES_DESKTOP_REMOTE_URL` | `http://<backend-host>:9119` |

### Troubleshooting {#troubleshooting}

- **"Remote gateway incomplete"** — você não digitou remote URL.
- **Sign-in falha com 401 / "Invalid credentials"** — username ou password não batem com `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` do backend. Backend retorna mesmo erro genérico para user desconhecido e password errada, então confira ambos. Confirme gate com `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'` — deve reportar `true` e incluir `"basic"`.
- **Sem botão "Sign in" — pede session token** — provider username/password não está ativo (`/api/status` não listará `"basic"`). Garanta que username e password (ou password hash) estão definidos e processo dashboard os carregou.
- **Signed out a cada restart** — defina `HERMES_DASHBOARD_BASIC_AUTH_SECRET` valor estável; senão signing key é regenerada por boot.
- **Connection refused / times out** — backend fez bind em `127.0.0.1` (default) em vez de endereço alcançável, ou firewall/VPN bloqueia porta. Bind em `0.0.0.0` ou tailscale IP e abra porta para rede confiável.

## CORS {#cors}

O servidor web restringe CORS a origins localhost apenas:

- `http://localhost:9119` / `http://127.0.0.1:9119` (production)
- `http://localhost:3000` / `http://127.0.0.1:3000`
- `http://localhost:5173` / `http://127.0.0.1:5173` (Vite dev server)

Se você roda servidor em porta custom, aquela origin é adicionada automaticamente.

## Desenvolvimento {#development}

Se você contribui para frontend do web dashboard:

```bash
# Terminal 1: start the backend API
hermes dashboard --no-open

# Terminal 2: start the Vite dev server with HMR
cd web/
npm install
npm run dev
```

Vite dev server em `http://localhost:5173` faz proxy de requisições `/api` para backend FastAPI em `http://127.0.0.1:9119`.

Frontend é buildado com React 19, TypeScript, Tailwind CSS v4 e componentes estilo shadcn/ui. Production builds output em `hermes_cli/web_dist/` que servidor FastAPI serve como SPA estática.

## Build automático no update {#automatic-build-on-update}

Quando você roda `hermes update`, frontend web é rebuildado automaticamente se `npm` está disponível. Isso mantém dashboard sincronizado com updates de código. Se `npm` não está instalado, update pula build do frontend e `hermes dashboard` buildará no primeiro launch.

## Temas e plugins {#themes--plugins}

O dashboard vem com oito temas built-in e pode ser estendido com temas definidos pelo usuário, abas de plugin e rotas API backend — tudo drop-in, sem clone de repo.

**Troque temas ao vivo** na barra do header — clique ícone de paleta ao lado do seletor de idioma. Seleção persiste em `config.yaml` sob `dashboard.theme` e é restaurada no page load.

**Mude fonte independentemente** do mesmo picker — seção **Font** abaixo da lista de temas override fonte UI do tema ativo. Escolha persiste entre trocas de tema (`config.yaml` → `dashboard.font`); escolha **Theme default** para limpar e voltar à fonte do tema ativo.

Temas built-in:

| Tema | Característica |
|-------|-----------|
| **Hermes Teal** (`default`) | Teal escuro + cream, system fonts, espaçamento confortável |
| **Hermes Teal (Large)** (`default-large`) | Igual ao default com texto 18px e espaçamento mais amplo |
| **Nous Blue** (`nous-blue`) | Acentos azuis Nous-branded com espaçamento arejado |
| **Midnight** (`midnight`) | Azul-violeta profundo, Inter + JetBrains Mono |
| **Ember** (`ember`) | Crimson quente + bronze, Spectral serif + IBM Plex Mono |
| **Mono** (`mono`) | Grayscale, IBM Plex, compacto |
| **Cyberpunk** (`cyberpunk`) | Verde neon sobre preto, Share Tech Mono |
| **Rosé** (`rose`) | Rosa + ivory, Fraunces serif, espaçoso |

Para buildar seu tema, adicionar aba de plugin, injetar em shell slots ou expor REST endpoints específicos de plugin, veja **[Extending the Dashboard](./extending-the-dashboard)** — o guia completo cobre:

- Theme YAML schema — palette, typography, layout, assets, componentStyles, colorOverrides, customCSS
- Layout variants — `standard`, `cockpit`, `tiled`
- Plugin manifest, SDK, shell slots, page-scoped slots (injete widgets em páginas built-in sem override), backend FastAPI routes
- Walkthrough completo combinado theme-plus-plugin (demo Strike Freedom cockpit)
- Discovery, reload e troubleshooting
