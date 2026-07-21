---
sidebar_position: 3
title: "Discord"
description: "Configure o Hermes Agent como bot do Discord"
---

# Configuração do Discord {#discord-setup}

O Hermes Agent integra-se ao Discord como bot, permitindo que você converse com seu assistente de IA por mensagens diretas ou canais do servidor. O bot recebe suas mensagens, processa-as pelo pipeline do Hermes Agent (incluindo uso de ferramentas, memória e raciocínio) e responde em tempo real. Suporta texto, mensagens de voz, anexos de arquivo e comandos slash.

Antes da configuração, eis o que a maioria das pessoas quer saber: como o Hermes se comporta depois de entrar no seu servidor.

## Como o Hermes se comporta {#how-hermes-behaves}

| Context | Behavior |
|---------|----------|
| **DMs** | O Hermes responde a toda mensagem. Não é necessário `@mention`. Cada DM tem sua própria sessão. |
| **Canais do servidor** | Por padrão, o Hermes só responde quando você o `@mention`. Se você publicar em um canal sem mencioná-lo, o Hermes ignora a mensagem. |
| **Canais de resposta livre** | Você pode tornar canais específicos livres de menção com `DISCORD_FREE_RESPONSE_CHANNELS`, ou desabilitar menções globalmente com `DISCORD_REQUIRE_MENTION=false`. Mensagens nesses canais são respondidas inline — a criação automática de threads é ignorada para manter o canal como um chat leve. |
| **Threads** | O Hermes responde na mesma thread. As regras de menção ainda se aplicam, a menos que essa thread ou seu canal pai esteja configurado como resposta livre. Threads permanecem isoladas do canal pai para o histórico de sessão. |
| **Canais compartilhados com vários usuários** | Por padrão, o Hermes isola o histórico de sessão por usuário dentro do canal, por segurança e clareza. Duas pessoas conversando no mesmo canal não compartilham uma transcrição, a menos que você desabilite isso explicitamente. |
| **Mensagens mencionando outros usuários** | Quando `DISCORD_IGNORE_NO_MENTION` é `true` (o padrão), o Hermes permanece em silêncio se uma mensagem @menciona outros usuários, mas **não** menciona o bot. Isso impede que o bot entre em conversas direcionadas a outras pessoas. Defina como `false` se quiser que o bot responda a todas as mensagens, independentemente de quem é mencionado. Isso se aplica apenas em canais do servidor, não em DMs. |

:::tip
Se você quer um canal normal de ajuda do bot, onde as pessoas possam falar com o Hermes sem marcá-lo toda vez, adicione esse canal a `DISCORD_FREE_RESPONSE_CHANNELS`.
:::

### Modelo de gateway do Discord {#discord-gateway-model}

O Hermes no Discord não é um webhook que responde de forma stateless. Ele roda pelo gateway de mensagens completo, o que significa que cada mensagem recebida passa por:

1. autorização (`DISCORD_ALLOWED_USERS`)
2. verificações de menção / resposta livre
3. busca de sessão
4. carregamento da transcrição da sessão
5. execução normal do agente Hermes, incluindo ferramentas, memória e comandos slash
6. entrega da resposta de volta ao Discord

Isso importa porque o comportamento em um servidor movimentado depende tanto do roteamento do Discord quanto da política de sessão do Hermes.

### Modelo de sessão no Discord {#session-model-in-discord}

Por padrão:

- cada DM recebe sua própria sessão
- cada thread do servidor recebe seu próprio namespace de sessão
- cada usuário em um canal compartilhado recebe sua própria sessão dentro desse canal

Então, se Alice e Bob conversam com o Hermes em `#research`, o Hermes trata isso como conversas separadas por padrão, mesmo usando o mesmo canal visível do Discord.

Isso é controlado por `config.yaml`:

```yaml
group_sessions_per_user: true
```

Defina como `false` somente se quiser explicitamente uma conversa compartilhada para toda a sala:

```yaml
group_sessions_per_user: false
```

Sessões compartilhadas podem ser úteis para uma sala colaborativa, mas também significam:

- usuários compartilham crescimento de contexto e custos de tokens
- uma tarefa longa e pesada em ferramentas de uma pessoa pode inflar o contexto de todos os outros
- uma execução em andamento de uma pessoa pode interromper o follow-up de outra pessoa na mesma sala

### Interrupções e concorrência {#interrupts-and-concurrency}

O Hermes rastreia agentes em execução pela chave de sessão.

Com o padrão `group_sessions_per_user: true`:

- Alice interrompendo sua própria requisição em andamento afeta apenas a sessão de Alice naquele canal
- Bob pode continuar conversando no mesmo canal sem herdar o histórico de Alice ou interromper a execução de Alice

Com `group_sessions_per_user: false`:

- toda a sala compartilha um slot de agente em execução para aquele canal/thread
- mensagens de follow-up de pessoas diferentes podem interromper ou entrar na fila umas das outras

Este guia percorre o processo completo de configuração — desde a criação do bot no Portal de Desenvolvedores do Discord até o envio da sua primeira mensagem.

### Saúde do WebSocket do gateway {#gateway-websocket-health}

O REST do Discord e o WebSocket do Gateway são transportes separados. Uma resposta REST bem-sucedida (incluindo `fetch_user()` retornando HTTP 200) não prova que o bot ainda consegue receber eventos do Gateway. O Hermes, portanto, combina o estado ready, o estado de fechamento do cliente/socket, a abertura do socket, a idade do ACK de heartbeat e a latência finita de heartbeat.

Após o número configurado de amostras consecutivas não saudáveis, o adaptador emite um evento fatal retryable. O watcher de reconexão existente do gateway cria um adaptador novo; o adaptador do Discord não inicia um segundo loop de reconexão ilimitado.

Configure os limites não secretos em `config.yaml`:

```yaml
discord:
  websocket_liveness_interval_seconds: 15
  websocket_liveness_failure_threshold: 2
  websocket_heartbeat_ack_max_age_seconds: 60
  websocket_max_latency_seconds: 30
```

Os nomes antigos `liveness_interval_seconds` e `liveness_failure_threshold` permanecem apenas como aliases de compatibilidade; eles não significam mais sondagem REST.

## Passo 1: Crie uma aplicação Discord {#step-1-create-a-discord-application}

1. Acesse o [Discord Developer Portal](https://discord.com/developers/applications) e entre com sua conta Discord.
2. Clique em **New Application** no canto superior direito.
3. Digite um nome para sua aplicação (ex.: "Hermes Agent") e aceite os Termos de Serviço para Desenvolvedores.
4. Clique em **Create**.

Você chegará à página **General Information**. Anote o **Application ID** — você precisará dele depois para montar a URL de convite.

## Passo 2: Crie o bot {#step-2-create-the-bot}

1. Na barra lateral esquerda, clique em **Bot**.
2. O Discord cria automaticamente um usuário bot para sua aplicação. Você verá o nome de usuário do bot, que pode personalizar.
3. Em **Authorization Flow**:
   - Defina **Public Bot** como **ON** — necessário para usar o link de convite fornecido pelo Discord (recomendado). Isso permite que a aba Installation gere uma URL de autorização padrão.
   - Deixe **Require OAuth2 Code Grant** como **OFF**.

:::tip
Você pode definir avatar e banner personalizados para seu bot nesta página. É isso que os usuários verão no Discord.
:::

:::info[Alternativa de bot privado]
Se preferir manter seu bot privado (Public Bot = OFF), você **deve** usar o método **Manual URL** no Passo 5 em vez da aba Installation. O link fornecido pelo Discord exige que Public Bot esteja habilitado.
:::

## Passo 3: Habilite os Privileged Gateway Intents {#step-3-enable-privileged-gateway-intents}

Esta é a etapa mais crítica de toda a configuração. Sem os intents corretos habilitados, seu bot conectará ao Discord, mas **não conseguirá ler o conteúdo das mensagens**.

Na página **Bot**, role até **Privileged Gateway Intents**. Você verá três toggles:

| Intent | Purpose | Required? |
|--------|---------|-----------|
| **Presence Intent** | Ver status online/offline do usuário | Opcional |
| **Server Members Intent** | Acessar a lista de membros, resolver nomes de usuário | **Obrigatório** |
| **Message Content Intent** | Ler o conteúdo de texto das mensagens | **Obrigatório** |

**Habilite Server Members Intent e Message Content Intent** alternando-os para **ON**.

- Sem **Message Content Intent**, seu bot recebe eventos de mensagem, mas o texto da mensagem fica vazio — o bot literalmente não consegue ver o que você digitou.
- Sem **Server Members Intent**, o bot não consegue resolver nomes de usuário para a lista de usuários permitidos e pode falhar ao identificar quem está enviando mensagens.

:::warning[Esta é a razão #1 pela qual bots Discord não funcionam]
Se seu bot está online, mas nunca responde a mensagens, o **Message Content Intent** quase certamente está desabilitado. Volte ao [Developer Portal](https://discord.com/developers/applications), selecione sua aplicação → Bot → Privileged Gateway Intents e certifique-se de que **Message Content Intent** está alternado para ON. Clique em **Save Changes**.
:::

**Sobre a contagem de servidores:**
- Se seu bot está em **menos de 100 servidores**, você pode simplesmente alternar intents livremente.
- Se seu bot está em **100 ou mais servidores**, o Discord exige que você envie uma solicitação de verificação para usar privileged intents. Para uso pessoal, isso não é uma preocupação.

Clique em **Save Changes** na parte inferior da página.

## Passo 4: Obtenha o token do bot {#step-4-get-the-bot-token}

O token do bot é a credencial que o Hermes Agent usa para entrar como seu bot. Ainda na página **Bot**:

1. Na seção **Token**, clique em **Reset Token**.
2. Se você tiver autenticação de dois fatores habilitada na sua conta Discord, digite seu código 2FA.
3. O Discord exibirá seu novo token. **Copie-o imediatamente.**

:::warning[Token exibido apenas uma vez]
O token é exibido apenas uma vez. Se você perdê-lo, precisará resetá-lo e gerar um novo. Nunca compartilhe seu token publicamente nem faça commit no Git — qualquer pessoa com esse token tem controle total do seu bot.
:::

Guarde o token em um lugar seguro (um gerenciador de senhas, por exemplo). Você precisará dele no Passo 8.

## Passo 5: Gere a URL de convite {#step-5-generate-the-invite-url}

Você precisa de uma URL OAuth2 para convidar o bot para seu servidor. Há duas formas de fazer isso:

### Opção A: Usando a aba Installation (recomendado) {#option-a-using-the-installation-tab-recommended}

:::note[Requer Public Bot]
Este método exige que **Public Bot** esteja definido como **ON** no Passo 2. Se você definiu Public Bot como OFF, use o método Manual URL abaixo.
:::

1. Na barra lateral esquerda, clique em **Installation**.
2. Em **Installation Contexts**, habilite **Guild Install**.
3. Para **Install Link**, selecione **Discord Provided Link**.
4. Em **Default Install Settings** para Guild Install:
   - **Scopes**: selecione `bot` e `applications.commands`
   - **Permissions**: selecione as permissões listadas abaixo.

### Opção B: URL manual {#option-b-manual-url}

Você pode construir a URL de convite diretamente neste formato:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=274878286912
```

Substitua `YOUR_APP_ID` pelo Application ID do Passo 1.

### Permissões obrigatórias {#required-permissions}

Estas são as permissões mínimas que seu bot precisa:

- **View Channels** — ver os canais aos quais tem acesso
- **Send Messages** — responder às suas mensagens
- **Embed Links** — formatar respostas ricas
- **Attach Files** — enviar imagens, áudio e saídas de arquivo
- **Read Message History** — manter o contexto da conversa

### Permissões adicionais recomendadas {#recommended-additional-permissions}

- **Send Messages in Threads** — responder em conversas de thread
- **Add Reactions** — reagir a mensagens para confirmação

### Inteiros de permissão {#permission-integers}

| Level | Permissions Integer | What's Included |
|-------|---------------------|-----------------|
| Minimal | `117760` | View Channels, Send Messages, Read Message History, Attach Files |
| Recommended | `274878286912` | Tudo acima mais Embed Links, Send Messages in Threads, Add Reactions |

## Passo 6: Convide para seu servidor {#step-6-invite-to-your-server}

1. Abra a URL de convite no navegador (da aba Installation ou da URL manual que você construiu).
2. No dropdown **Add to Server**, selecione seu servidor.
3. Clique em **Continue**, depois **Authorize**.
4. Complete o CAPTCHA se solicitado.

:::info
Você precisa da permissão **Manage Server** no servidor Discord para convidar um bot. Se não vir seu servidor no dropdown, peça a um admin do servidor para usar o link de convite.
:::

Após autorizar, o bot aparecerá na lista de membros do seu servidor (mostrará offline até você iniciar o gateway do Hermes).

## Passo 7: Encontre seu Discord User ID {#step-7-find-your-discord-user-id}

O Hermes Agent usa seu Discord User ID para controlar quem pode interagir com o bot. Para encontrá-lo:

1. Abra o Discord (desktop ou web app).
2. Vá em **Settings** → **Advanced** → alterne **Developer Mode** para **ON**.
3. Feche as configurações.
4. Clique com o botão direito no seu próprio nome de usuário (em uma mensagem, na lista de membros ou no seu perfil) → **Copy User ID**.

Seu User ID é um número longo como `284102345871466496`.

:::tip
O Developer Mode também permite copiar **Channel IDs** e **Server IDs** da mesma forma — clique com o botão direito no nome do canal ou servidor e selecione Copy ID. Você precisará de um Channel ID se quiser definir um canal home manualmente.
:::

## Passo 8: Configure o Hermes Agent {#step-8-configure-hermes-agent}

### Opção A: Setup interativo (recomendado) {#option-a-interactive-setup-recommended}

Execute o comando de setup guiado:

```bash
hermes gateway setup
```

Selecione **Discord** quando solicitado, depois cole seu token do bot e user ID quando pedido.

### Opção B: Configuração manual {#option-b-manual-configuration}

Adicione o seguinte ao seu arquivo `~/.hermes/.env`:

```bash
# Required
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=284102345871466496

# Multiple allowed users (comma-separated)
# DISCORD_ALLOWED_USERS=284102345871466496,198765432109876543
```

Depois inicie o gateway:

```bash
hermes gateway
```

O bot deve ficar online no Discord em alguns segundos. Envie uma mensagem — seja uma DM ou em um canal que ele possa ver — para testar.

:::tip
Você pode executar `hermes gateway` em background ou como serviço systemd para operação persistente. Veja a documentação de deployment para detalhes.
:::

## Referência de configuração {#configuration-reference}

O comportamento do Discord é controlado por dois arquivos: **`~/.hermes/.env`** para credenciais e toggles em nível de env, e **`~/.hermes/config.yaml`** para configurações estruturadas. Variáveis de ambiente sempre têm precedência sobre valores do config.yaml quando ambos estão definidos.

### Variáveis de ambiente (`.env`) {#environment-variables-env}

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DISCORD_BOT_TOKEN` | **Sim** | — | Token do bot do [Discord Developer Portal](https://discord.com/developers/applications). |
| `DISCORD_ALLOWED_USERS` | Condicional | — | IDs de usuário Discord separados por vírgula permitidos a interagir com o bot. Sem isso **ou** `DISCORD_ALLOWED_ROLES`, o gateway nega todos os usuários, a menos que `DISCORD_ALLOW_ALL_USERS=true`, `GATEWAY_ALLOW_ALL_USERS=true` ou `DISCORD_ALLOWED_CHANNELS` delimite explicitamente o acesso à guild. |
| `DISCORD_ALLOWED_ROLES` | Não | — | IDs de cargo Discord separados por vírgula. Qualquer membro com um desses cargos é autorizado — semântica OR com `DISCORD_ALLOWED_USERS`. Habilita automaticamente o **Server Members Intent** na conexão. Útil quando equipes de moderação mudam: novos mods ganham acesso assim que o cargo é concedido, sem push de config. |
| `DISCORD_ALLOW_ALL_USERS` | Não | `false` | Opt-in explícito para permitir todo usuário Discord que consiga alcançar o bot. Restaura o comportamento aberto pré-0.18 apenas para Discord; use somente em guilds confiáveis/privadas ou desenvolvimento. |
| `GATEWAY_ALLOW_ALL_USERS` | Não | `false` | Opt-in global allow-all para toda plataforma do gateway. Prefira o `DISCORD_ALLOW_ALL_USERS` específico da plataforma, a menos que queira intencionalmente todas as plataformas conectadas abertas. |
| `DISCORD_HOME_CHANNEL` | Não | — | ID do canal onde o bot envia mensagens proativas (saída de cron, lembretes, notificações). |
| `DISCORD_HOME_CHANNEL_NAME` | Não | `"Home"` | Nome de exibição do canal home em logs e saída de status. |
| `DISCORD_COMMAND_SYNC_POLICY` | Não | `"safe"` | Controla a sincronização de startup de comandos slash nativos. `"safe"` faz diff dos comandos globais existentes e atualiza apenas o que mudou, recriando comandos quando mudanças de metadados do Discord não podem ser aplicadas via patch. `"bulk"` preserva o comportamento antigo de `tree.sync()`. `"off"` pula a sincronização de startup inteiramente. |
| `DISCORD_REQUIRE_MENTION` | Não | `true` | Quando `true`, o bot só responde em canais do servidor quando `@mentioned`. Defina como `false` para responder a todas as mensagens em todo canal. |
| `DISCORD_THREAD_REQUIRE_MENTION` | Não | `false` | Quando `true`, o atalho de menção dentro da thread é desabilitado — threads são gated da mesma forma que canais, exigindo `@mention` mesmo depois que o bot já participou. Use quando vários bots compartilham uma thread e você quer que cada um dispare apenas com `@mention` explícita. |
| `DISCORD_FREE_RESPONSE_CHANNELS` | Não | — | IDs de canal separados por vírgula onde o bot responde sem exigir `@mention`, mesmo quando `DISCORD_REQUIRE_MENTION` é `true`. |
| `DISCORD_IGNORE_NO_MENTION` | Não | `true` | Quando `true`, o bot permanece em silêncio se uma mensagem `@mentions` outros usuários, mas **não** menciona o bot. Impede que o bot entre em conversas direcionadas a outras pessoas. Aplica-se apenas em canais do servidor, não em DMs. |
| `DISCORD_AUTO_THREAD` | Não | `true` | Quando `true`, cria automaticamente uma nova thread para cada `@mention` em um canal de texto, isolando cada conversa (similar ao comportamento do Slack). Mensagens já dentro de threads ou DMs não são afetadas. |
| `DISCORD_ALLOW_BOTS` | Não | `"none"` | Controla como o bot lida com mensagens de outros bots Discord. `"none"` — ignora todos os outros bots. `"mentions"` — aceita apenas mensagens de bot que `@mention` o Hermes. `"all"` — aceita todas as mensagens de bot. |
| `DISCORD_REACTIONS` | Não | `true` | Quando `true`, o bot adiciona reações emoji às mensagens durante o processamento (👀 ao iniciar, ✅ em sucesso, ❌ em erro). Defina como `false` para desabilitar reações inteiramente. |
| `DISCORD_IGNORED_CHANNELS` | Não | — | IDs de canal separados por vírgula onde o bot **nunca** responde, mesmo quando `@mentioned`. Tem prioridade sobre todas as outras configurações de canal. |
| `DISCORD_ALLOWED_CHANNELS` | Não | — | IDs de canal separados por vírgula. Quando definido, o bot **só** responde nesses canais (mais DMs se permitido). Substitui `discord.allowed_channels` do `config.yaml`. Combine com `DISCORD_IGNORED_CHANNELS` para expressar regras allow/deny. |
| `DISCORD_NO_THREAD_CHANNELS` | Não | — | IDs de canal separados por vírgula onde o bot responde diretamente no canal em vez de criar uma thread. Relevante apenas quando `DISCORD_AUTO_THREAD` é `true`. |
| `DISCORD_HISTORY_BACKFILL` | Não | `true` | Quando `true`, prepende scrollback recente do canal (desde a última resposta do bot) à mensagem do usuário quando o bot é mencionado. Recupera contexto que o bot perderia com `require_mention`. Ignorado em DMs e canais de resposta livre. Defina como `false` para desabilitar. |
| `DISCORD_HISTORY_BACKFILL_LIMIT` | Não | `50` | Número máximo de mensagens a escanear para trás ao montar o bloco de backfill. Na prática, o scan geralmente para antes — na última mensagem do próprio bot no canal. |
| `DISCORD_REPLY_TO_MODE` | Não | `"first"` | Controla o comportamento de reply-reference: `"off"` — nunca responde à mensagem original, `"first"` — reply-reference apenas no primeiro chunk de mensagem (padrão), `"all"` — reply-reference em todo chunk. |
| `DISCORD_ALLOW_MENTION_EVERYONE` | Não | `false` | Quando `false` (padrão), o bot não pode pingar `@everyone` ou `@here`, mesmo se sua resposta contiver esses tokens. Defina como `true` para optar de volta. Veja [Controle de menções](#mention-control) abaixo. |
| `DISCORD_ALLOW_MENTION_ROLES` | Não | `false` | Quando `false` (padrão), o bot não pode pingar menções `@role`. Defina como `true` para permitir. |
| `DISCORD_ALLOW_MENTION_USERS` | Não | `true` | Quando `true` (padrão), o bot pode pingar usuários individuais por ID. |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | Não | `true` | Quando `true` (padrão), responder a uma mensagem pinga o autor original. |
| `DISCORD_PROXY` | Não | — | URL de proxy para conexões Discord (HTTP, WebSocket, REST). Substitui `HTTPS_PROXY`/`ALL_PROXY`. Suporta esquemas `http://`, `https://` e `socks5://`. |
| `DISCORD_ALLOW_ANY_ATTACHMENT` | Não | `false` | Quando `true`, o bot aceita anexos de qualquer tipo de arquivo (não apenas a allowlist embutida PDF/text/zip/office). Tipos desconhecidos são cacheados em disco e expostos ao agente como caminho local com MIME `application/octet-stream` para inspeção com `terminal` / `read_file` / `ffprobe` / etc. |
| `DISCORD_MAX_ATTACHMENT_BYTES` | Não | `33554432` | Bytes máximos por anexo que o gateway baixará e cacheará. Padrão 32 MiB. Defina como `0` para sem limite (anexos ficam em memória enquanto são escritos, então ilimitado tem custo real de memória). |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | Não | `0.6` | Janela de graça que o adaptador espera antes de flush de um chunk de texto enfileirado. Útil para suavizar saída streamed. |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | Não | `2.0` | Atraso entre chunks divididos quando uma única mensagem excede o limite de tamanho do Discord. |

:::warning Conversa bot-a-bot não é suportada
`DISCORD_ALLOW_BOTS` existe para aceitar entrada de um bot confiável específico (ex.: um bot relay ou webhook), não para deixar dois perfis Hermes conversarem entre si. O padrão, `"none"`, ignora todos os outros bots e é a configuração segura.

Conectar vários perfis Hermes para responderem uns aos outros em um canal compartilhado — definindo `"mentions"` ou `"all"` em vários perfis — é uma topologia não suportada. O Discord auto-`@mentions` o autor respondido em toda resposta, então com `"mentions"` dois bots satisfarão o gate de menção um do outro indefinidamente e entrarão em ack-loop. Não há circuit breaker para isso porque a configuração suportada é simplesmente deixar `DISCORD_ALLOW_BOTS` em `"none"`. Se precisar aceitar um bot específico, delimite a aceitação de forma estreita e nunca para outro agente que responde automaticamente.
:::

### Arquivo de config (`config.yaml`) {#config-file-configyaml}

A seção `discord` em `~/.hermes/config.yaml` espelha as env vars acima. Configurações do config.yaml são aplicadas como padrões — se a env var equivalente já estiver definida, a env var vence.

```yaml
# Discord-specific settings
discord:
  require_mention: true           # Require @mention in server channels
  thread_require_mention: false   # If true, require @mention in threads too (multi-bot threads)
  free_response_channels: ""      # Comma-separated channel IDs (or YAML list)
  auto_thread: true               # Auto-create threads on @mention
  reactions: true                 # Add emoji reactions during processing
  ignored_channels: []            # Channel IDs where bot never responds
  no_thread_channels: []          # Channel IDs where bot responds without threading
  history_backfill: true          # Prepend recent channel scrollback on mention (default: true)
  history_backfill_limit: 50      # Max messages to scan backwards (default: 50)
  missed_message_backfill:        # Replay messages missed while disconnected (opt-in)
    enabled: false
    channels: []                  # Empty uses free_response_channels
    window_seconds: 21600         # Look back at most 6 hours
    limit: 100                    # Global scan cap per reconnect
    max_dispatches: 10            # Recovery dispatch cap per reconnect
  channel_prompts: {}             # Per-channel ephemeral system prompts
  allow_mentions:                 # What the bot is allowed to ping (safe defaults)
    everyone: false               # @everyone / @here pings (default: false)
    roles: false                  # @role pings (default: false)
    users: true                   # @user pings (default: true)
    replied_user: true            # reply-reference pings the author (default: true)

# Session isolation (applies to all gateway platforms, not just Discord)
group_sessions_per_user: true     # Isolate sessions per user in shared channels
```

#### `discord.require_mention` {#discordrequire_mention}

**Type:** boolean — **Default:** `true`

Quando habilitado, o bot só responde em canais do servidor quando diretamente `@mentioned`. DMs sempre recebem resposta, independentemente desta configuração.

#### `discord.thread_require_mention` {#discordthread_require_mention}

**Type:** boolean — **Default:** `false`

Por padrão, depois que o bot participou de uma thread (auto-criada em `@mention` ou respondeu uma vez), ele continua respondendo a toda mensagem subsequente naquela thread sem precisar ser `@mentioned` de novo. Esse é o padrão certo para conversas one-on-one.

Em **threads multi-bot** onde usuários endereçam um bot por turno, esse padrão vira uma armadilha — todo outro bot na thread também dispara em toda mensagem, queimando créditos e spammando o canal. Defina `thread_require_mention: true` para desabilitar o atalho dentro da thread e fazer gate de threads da mesma forma que canais. `@mentions` explícitas ainda funcionam como antes.

```yaml
discord:
  require_mention: true
  thread_require_mention: true    # multi-bot setup
```

#### `discord.free_response_channels` {#discordfree_response_channels}

**Type:** string or list — **Default:** `""`

IDs de canal onde o bot responde a todas as mensagens sem precisar de `@mention`. Aceita string separada por vírgula ou lista YAML:

```yaml
# String format
discord:
  free_response_channels: "1234567890,9876543210"

# List format
discord:
  free_response_channels:
    - 1234567890
    - 9876543210
```

Se o canal pai de uma thread está nesta lista, a thread também fica livre de menção.

Canais de resposta livre também **pulam auto-threading** — o bot responde inline em vez de criar uma nova thread por mensagem. Isso mantém o canal utilizável como superfície de chat leve. Se quiser comportamento de threading, não liste o canal como resposta livre (use o fluxo normal de `@mention`).

#### `discord.auto_thread` {#discordauto_thread}

**Type:** boolean — **Default:** `true`

Quando habilitado, cada `@mention` em um canal de texto regular cria automaticamente uma nova thread para a conversa. Isso mantém o canal principal limpo e dá a cada conversa seu próprio histórico de sessão isolado. Depois que uma thread é criada, mensagens subsequentes naquela thread não exigem `@mention` — o bot sabe que já está participando. Defina [`thread_require_mention`](#discordthread_require_mention) como `true` para desabilitar esse atalho dentro da thread em setups multi-bot.

Mensagens enviadas em threads existentes ou DMs não são afetadas por esta configuração. Canais listados em `discord.free_response_channels` ou `discord.no_thread_channels` também ignoram auto-threading e recebem respostas inline.

#### `discord.reactions` {#discordreactions}

**Type:** boolean — **Default:** `true`

Controla se o bot adiciona reações emoji às mensagens como feedback visual:
- 👀 adicionado quando o bot começa a processar sua mensagem
- ✅ adicionado quando a resposta é entregue com sucesso
- ❌ adicionado se ocorrer um erro durante o processamento

Desabilite se achar as reações distrativas ou se o cargo do bot não tiver a permissão **Add Reactions**.

#### `discord.ignored_channels` {#discordignored_channels}

**Type:** string or list — **Default:** `[]`

IDs de canal onde o bot **nunca** responde, mesmo quando diretamente `@mentioned`. Isso tem a maior prioridade — se um canal está nesta lista, o bot ignora silenciosamente todas as mensagens lá, independentemente de `require_mention`, `free_response_channels` ou qualquer outra configuração.

```yaml
# String format
discord:
  ignored_channels: "1234567890,9876543210"

# List format
discord:
  ignored_channels:
    - 1234567890
    - 9876543210
```

Se o canal pai de uma thread está nesta lista, mensagens naquela thread também são ignoradas.

#### `discord.no_thread_channels` {#discordno_thread_channels}

**Type:** string or list — **Default:** `[]`

IDs de canal onde o bot responde diretamente no canal em vez de auto-criar uma thread. Só tem efeito quando `auto_thread` é `true` (o padrão). Nesses canais, o bot responde inline como uma mensagem normal em vez de spawnar uma nova thread.

```yaml
discord:
  no_thread_channels:
    - 1234567890  # Bot responds inline here
```

Útil para canais dedicados à interação com o bot, onde threads adicionariam ruído desnecessário.

#### `discord.channel_prompts` {#discordchannel_prompts}

**Type:** mapping — **Default:** `{}`

Prompts de sistema efêmeros por canal injetados a cada turno no canal ou thread Discord correspondente, sem serem persistidos no histórico da transcrição.

```yaml
discord:
  channel_prompts:
    "1234567890": |
      This channel is for research tasks. Prefer deep comparisons,
      citations, and concise synthesis.
    "9876543210": |
      This forum is for therapy-style support. Be warm, grounded,
      and non-judgmental.
```

Comportamento:
- Correspondências exatas de ID de thread/canal vencem.
- Se uma mensagem chega dentro de uma thread ou post de fórum e essa thread não tem entrada explícita, o Hermes faz fallback para o ID do canal/fórum pai.
- Prompts são aplicados efemeramente em runtime, então alterá-los afeta turnos futuros imediatamente sem reescrever o histórico de sessão passado.

#### `discord.history_backfill` {#discordhistory_backfill}

**Type:** boolean — **Default:** `true`

Quando habilitado, o bot recupera mensagens de canal perdidas a cada `@mention`. Com `require_mention: true`, o bot só processa mensagens que o marcam diretamente — todo o resto no canal é invisível para a transcrição da sessão. O history backfill escaneia para trás pelo histórico recente do canal quando acionado, coletando mensagens entre a última resposta do bot e a menção atual, e as inclui como contexto.

Comportamento por superfície:

- **Canais do servidor** (com `require_mention: true`): backfill escaneia o canal desde a última resposta do bot. Útil quando outros participantes postaram enquanto o bot não estava sendo endereçado.
- **Threads**: backfill escaneia apenas a thread — `channel.history()` do Discord em uma thread retorna apenas mensagens daquela thread, não do canal pai. Esse é o escopo certo porque threads são geralmente conversas autocontidas.
- **DMs**: ignorado. Toda mensagem de DM aciona o bot, então a transcrição da sessão já está completa — não há gap de menção a preencher.
- **Canais de resposta livre** e **threads auto-criadas pelo bot**: ignorados pelo mesmo motivo — sem gate de menção significa sem gap.

Sessões por usuário (`group_sessions_per_user: true`, o padrão) também se beneficiam: a sessão de um usuário está faltando o contexto postado por outros participantes do canal e as próprias mensagens do usuário de antes de marcar o bot. O backfill preenche ambos os gaps.

```yaml
discord:
  history_backfill: true   # default
```

Para desligar:

```yaml
discord:
  history_backfill: false
```

> **Note:** Mensagens que chegam *enquanto* o bot está processando (entre um trigger e sua resposta) não são capturadas. Isso é uma simplificação aceita — o usuário pode reenviar ou marcar de novo.

#### `discord.history_backfill_limit` {#discordhistory_backfill_limit}

**Type:** integer — **Default:** `50`

Número máximo de mensagens a escanear para trás ao recuperar contexto do canal. Na prática, o scan geralmente para muito antes — na última mensagem do próprio bot no canal, que é o limite natural entre turnos. Este limite é um cap de segurança para cold starts e gaps longos onde não existe mensagem anterior do bot no histórico recente.

```yaml
discord:
  history_backfill: true
  history_backfill_limit: 50
```

#### `discord.missed_message_backfill` {#discordmissed_message_backfill}

**Type:** object — **Default:** disabled

A janela de resume do WebSocket do Discord pode expirar durante um restart ou queda de rede. Mensagens enviadas durante esse gap não são entregues como eventos live do gateway. Quando esta opção está habilitada, o Hermes escaneia um conjunto delimitado de históricos de canal e thread configurados depois que o Discord reconecta, depois envia mensagens ainda não tratadas pelo mesmo caminho de autorização, menção, canal, deduplicação e dispatch que eventos live.

```yaml
discord:
  missed_message_backfill:
    enabled: true
    channels: ["123456789012345678"]
    window_seconds: 3600
    limit: 100
    max_dispatches: 10
```

Se `channels` estiver vazio, o Hermes usa `discord.free_response_channels`. Defina como `"*"` somente quando o bot deve inspecionar todo canal de texto de servidor alcançável. O ledger de recuperação é armazenado por perfil em `gateway/discord_message_recovery.db`, impedindo que uma mensagem respondida com sucesso seja replayada depois de um restart posterior.

#### `group_sessions_per_user` {#groupsessions_per_user}

**Type:** boolean — **Default:** `true`

Esta é uma configuração global do gateway (não específica do Discord) que controla se usuários no mesmo canal recebem históricos de sessão isolados.

Quando `true`: Alice e Bob conversando em `#research` têm cada um sua conversa separada com o Hermes. Quando `false`: todo o canal compartilha uma transcrição de conversa e um slot de agente em execução.

```yaml
group_sessions_per_user: true
```

Veja a seção [Modelo de sessão](#session-model-in-discord) acima para as implicações completas de cada modo.

#### `display.tool_progress` {#displaytool_progress}

**Type:** string — **Default:** `"all"` — **Values:** `off`, `new`, `all`, `verbose`

Controla se o bot envia mensagens de progresso no chat durante o processamento (ex.: "Reading file...", "Running terminal command..."). Esta é uma configuração global do gateway que se aplica a todas as plataformas.

```yaml
display:
  tool_progress: "all"    # off | new | all | verbose
```

- `off` — sem mensagens de progresso
- `new` — mostra apenas a primeira chamada de ferramenta por turno
- `all` — mostra todas as chamadas de ferramenta (truncadas a 40 caracteres em mensagens do gateway)
- `verbose` — mostra detalhes completos das chamadas de ferramenta (pode produzir mensagens longas)

#### `display.tool_progress_command` {#displaytool_progress_command}

**Type:** boolean — **Default:** `false`

Quando habilitado, torna o comando slash `/verbose` disponível no gateway, permitindo alternar modos de progresso de ferramentas (`off → new → all → verbose → off`) sem editar config.yaml.

```yaml
display:
  tool_progress_command: true
```

## Controle de acesso a comandos slash {#slash-command-access-control}

Por padrão, todo usuário permitido pode executar todo comando slash. Para dividir sua allowlist em **admins** (acesso completo a comandos slash) e **usuários regulares** (apenas comandos que você habilitar explicitamente), adicione `allow_admin_from` e `user_allowed_commands` ao bloco `extra` da plataforma Discord:

```yaml
gateway:
  platforms:
    discord:
      extra:
        # Existing user allowlist (unchanged)
        allow_from:
          - "123456789012345678"  # admin user ID
          - "999888777666555444"  # regular user ID

        # NEW — admins get all slash commands (built-in + plugin)
        allow_admin_from:
          - "123456789012345678"

        # NEW — non-admin allowed users can only run these slash commands.
        # /help and /whoami are always allowed so users can see their access.
        user_allowed_commands:
          - status
          - model
          - history

        # Optional: separate admin / command lists for server channels
        group_allow_admin_from:
          - "123456789012345678"
        group_user_allowed_commands:
          - status
```

**Comportamento:**

- Um usuário em `allow_admin_from` para um escopo (DM ou canal do servidor) pode executar **todo** comando slash registrado — embutido E registrado por plugin — pelo registro de comandos live.
- Um usuário que não está em `allow_admin_from` só pode executar comandos listados em `user_allowed_commands`, mais o piso sempre permitido: `/help` e `/whoami`.
- Chat simples (mensagens não-slash) não é afetado. Usuários não-admin ainda podem falar com o agente normalmente; só não podem acionar comandos arbitrários.
- **Compatibilidade retroativa:** se `allow_admin_from` não estiver definido para um escopo, o gate de comandos slash fica desabilitado para aquele escopo. Instalações existentes continuam funcionando sem mudanças.
- Status de admin em DM não implica status de admin em canal do servidor. Cada escopo tem sua própria lista de admin.

Use `/whoami` para ver o escopo ativo, seu tier (admin / user / unrestricted) e quais comandos slash você pode executar.

## Seletor interativo de modelo {#interactive-model-picker}

Envie `/model` sem argumentos em um canal Discord para abrir um seletor de modelo baseado em dropdown:

1. **Seleção de provider** — um dropdown Select mostrando providers disponíveis (até 25).
2. **Seleção de modelo** — um segundo dropdown com modelos do provider escolhido (até 25).

O seletor expira após 120 segundos. Apenas usuários autorizados (aqueles em `DISCORD_ALLOWED_USERS`) podem interagir com ele. Se souber o nome do modelo, digite `/model <name>` diretamente.

## Comandos slash nativos para skills {#native-slash-commands-for-skills}

O Hermes registra automaticamente skills instaladas como **Discord Application Commands nativos**. Isso significa que skills aparecem no menu autocomplete `/` do Discord junto com comandos embutidos.

- Cada skill vira um comando slash Discord (ex.: `/code-review`, `/ascii-art`)
- Skills aceitam um parâmetro string `args` opcional
- O Discord tem limite de 100 application commands por bot — se você tiver mais skills do que slots disponíveis, skills extras são ignoradas com aviso nos logs
- Skills são registradas durante o startup do bot junto com comandos embutidos como `/model`, `/reset` e `/background`

Nenhuma configuração extra é necessária — qualquer skill instalada via `hermes skills install` é automaticamente registrada como comando slash Discord no próximo restart do gateway.

### Desabilitando registro de comandos slash {#disabling-slash-command-registration}

Se você executa vários gateways Hermes contra a mesma aplicação Discord (ex.: staging + production), apenas um deles deve possuir o registro global de comandos slash — caso contrário, o último startup vence e os registros ficam oscilando. Desligue o registro slash no gateway "follower":

```yaml
gateway:
  platforms:
    discord:
      extra:
        slash_commands: false   # default: true
```

Deixar isso em `true` no gateway "primary" mantém o comportamento normal — comandos globais do menu `/` para embutidos e skills instaladas.

## Enviando mídia (tags inline `MEDIA:`) {#sending-media-inline-media-tags}

O adaptador Discord suporta uploads nativos de arquivo para todo tipo comum de mídia via tags inline `MEDIA:/path/to/file` emitidas na resposta do agente — o adaptador remove a tag e faz auto-upload do arquivo:

| Type | How it's delivered |
|------|--------------------|
| Images (PNG/JPG/WebP) | Anexo de imagem nativo do Discord com preview inline |
| Animated GIFs | `send_animation` faz upload como `animation.gif` para o Discord reproduzir inline (não como thumbnail estático) |
| Video (MP4/MOV) | `send_video` — player de vídeo nativo |
| Audio / Voice | `send_voice` — mensagem de voz nativa quando possível, anexo de arquivo caso contrário |
| Documents (PDF/ZIP/docx/etc.) | `send_document` — anexo nativo com botão de download |

O limite de tamanho por upload do Discord depende do tier de boost do servidor (25 MB grátis, até 500 MB). Se o Hermes receber HTTP 413, o adaptador faz fallback para um link apontando ao caminho do cache local em vez de falhar silenciosamente.

## Recebendo tipos arbitrários de arquivo {#receiving-arbitrary-file-types}

Qualquer tipo de arquivo que um usuário enviar é aceito. Autorização para enviar mensagem ao agente é o gate — não a extensão do arquivo. Todo upload é baixado, cacheado em `~/.hermes/cache/documents/` e exposto ao agente como evento de mensagem tipado `DOCUMENT` para inspeção com `terminal` (`ffprobe`, `unzip`, `file`, `strings`, etc.) ou `read_file`.

- Tipos conhecidos (PDF, docx/xlsx/pptx, zip, imagens/áudio/vídeo, etc.) mantêm seu MIME preciso.
- Tipos desconhecidos fazem fallback para o content type reportado do upload, ou `application/octet-stream` quando nenhum é fornecido.
- Arquivos pequenos decodificáveis em UTF-8 (texto, código, config, HTML, CSS, JSON, YAML, ...) têm seu conteúdo auto-injetado no prompt até 100 KiB. Arquivos binários que não podem ser decodificados são expostos apenas como nota de contexto apontando caminho (auto-traduzida para terminais sandboxed Docker/Modal via `to_agent_visible_cache_path`), para não estourar a janela de contexto.

O único limite inbound é o cap de tamanho por arquivo (padrão 32 MiB):

```yaml
discord:
  # Optional — raise/disable the per-file size cap. Default is 32 MiB.
  # The whole file is held in memory while being cached, so unlimited
  # uploads carry a real memory cost.
  max_attachment_bytes: 33554432   # bytes; 0 = unlimited
```

Env var equivalente: `DISCORD_MAX_ATTACHMENT_BYTES=33554432` (ou `0` para sem cap).

A flag legada `discord.allow_any_attachment` agora é no-op — qualquer tipo de arquivo é sempre aceito — e é mantida apenas para configs existentes não darem erro.

:::warning Custo de memória do ilimitado
Desabilitar o cap de tamanho (`max_attachment_bytes: 0`) significa que um usuário pode soltar um arquivo multi-GB no bot e o gateway vai bufferizá-lo diligentemente pela memória enquanto cacheia em disco. Defina isso apenas em instalações confiáveis de usuário único. Para bots compartilhados, mantenha o padrão de 32 MiB ou aumente conservadoramente.
:::

## Prompts interativos (clarify) {#interactive-prompts-clarify}

Quando o agente chama a ferramenta `clarify` — para perguntar qual abordagem você prefere, obter feedback pós-tarefa ou verificar antes de uma decisão não trivial — o Discord renderiza a pergunta com **um botão por escolha**:

> Which framework should I use for the dashboard?
>
> [1. Next.js] [2. Remix] [3. Astro] [Other (type answer)]

Clique em um botão numerado para responder, ou clique em **Other** para digitar uma resposta livre (a próxima mensagem que você enviar naquele canal vira a resposta). Chamadas `clarify` abertas (sem escolhas preset) pulam os botões e apenas capturam sua próxima mensagem.

Os botões se desabilitam depois que uma escolha é feita, para cliques duplicados não resolverem o prompt duas vezes. Configure o timeout de resposta via `agent.clarify_timeout` em `~/.hermes/config.yaml` (padrão `600` segundos). Se você não responder dentro do timeout, o agente desbloqueia com uma mensagem sentinela e se adapta em vez de travar.

## Canal home {#home-channel}

Você pode designar um "canal home" onde o bot envia mensagens proativas (como saída de cron job, lembretes e notificações). Há duas formas de definir:

### Usando o comando slash {#using-the-slash-command}

Digite `/sethome` em qualquer canal Discord onde o bot está presente. Esse canal vira o canal home.

### Configuração manual {#manual-configuration}

Adicione isto ao seu `~/.hermes/.env`:

```bash
DISCORD_HOME_CHANNEL=123456789012345678
DISCORD_HOME_CHANNEL_NAME="#bot-updates"
```

Substitua o ID pelo channel ID real (clique com botão direito → Copy Channel ID com Developer Mode ligado).

## Mensagens de voz {#voice-messages}

O Hermes Agent suporta mensagens de voz do Discord:

- **Mensagens de voz recebidas** são automaticamente transcritas usando o provider STT configurado: `faster-whisper` local (sem key), Groq Whisper (`GROQ_API_KEY`) ou OpenAI Whisper (`VOICE_TOOLS_OPENAI_KEY`).
- **Text-to-speech**: Use `/voice tts` para o bot enviar respostas em áudio falado junto com respostas de texto.
- **Canais de voz Discord**: O Hermes também pode entrar em um canal de voz, ouvir usuários falando e responder no canal.

Para o guia completo de setup e operação, veja:
- [Voice Mode](/user-guide/features/voice-mode)
- [Use Voice Mode with Hermes](/guides/use-voice-mode-with-hermes)

### Efeitos de áudio em canal de voz (ambient + acks verbais) {#voice-channel-audio-effects-ambient--verbal-acks}

Quando o bot está em um canal de voz, você pode dar uma sensação mais conversacional: um ack verbal curto ("let me look into that") antes de começar a trabalhar, e uma cama ambient sutil de "thinking" que toca por baixo enquanto ferramentas rodam — a fala faz duck do ambient para baixo e o aumenta de volta quando termina, similar ao modo de voz Grok.

O discord.py reproduz apenas um stream de áudio por conexão, então o Hermes instala um mixer de software no stream de saída que soma um loop ambient, acks e respostas TTS naquele único stream — eles se sobrepõem em vez de se cortarem.

Isso está **desligado por padrão**. Habilite em `config.yaml`:

```yaml
discord:
  voice_fx:
    enabled: true          # master switch
    ambient_enabled: true  # idle "thinking" bed while tools run
    ambient_path: ""       # custom loop file (any audio format); "" = built-in synthesised pad
    ambient_gain: 0.18     # idle bed loudness (0.0–1.0)
    duck_gain: 0.06        # ambient loudness while the bot is speaking
    speech_gain: 1.0       # TTS / acknowledgement loudness
    ack_enabled: true      # speak a short phrase before the first tool call of a turn
    ack_phrases:           # picked at random; set to [] to disable the spoken ack
      - "Let me look into that."
      - "One moment."
      - "Checking on that now."
```

Notas:
- O ack dispara no máximo uma vez por turno, apenas quando o bot está em um canal de voz e o mixer está ativo. Usa seu provider TTS configurado.
- `ambient_path` aceita qualquer arquivo que o `ffmpeg` consiga decodificar; é loopado seamlessmente. Deixe vazio para usar o pad sintetizado embutido (sem asset necessário).
- Todas as configurações ficam em `config.yaml` (não `.env`) — são comportamentais, não segredos.
- Quando `voice_fx.enabled` é `false`, a reprodução de voz usa o caminho one-shot original e nada muda.


## Canais de fórum {#forum-channels}

Canais de fórum do Discord (tipo 15) não aceitam mensagens diretas — todo post em um fórum deve ser uma thread. O Hermes auto-detecta canais de fórum e cria um novo post de thread sempre que precisa enviar lá, então respostas de texto, TTS, imagens, mensagens de voz e anexos de arquivo funcionam sem tratamento especial do agente.

- **Nome da thread** é derivado da primeira linha da mensagem (prefixo de heading markdown removido, limitado a 100 chars). Quando a mensagem é apenas anexo, o nome do arquivo é usado como fallback do nome da thread.
- **Anexos** vão junto na mensagem inicial da nova thread — sem passo de upload separado, sem envios parciais.
- **Uma chamada, uma thread**: cada envio ao fórum cria uma nova thread. Envios sucessivos ao mesmo fórum, portanto, produzem threads separadas.
- **Detecção é em três camadas**: primeiro o cache do diretório de canais, depois um cache de probe local ao processo, e por último um probe live `GET /channels/{id}` (cujo resultado é memoizado pela vida do processo).

Atualizar o diretório (`/channels refresh` em plataformas que expõem isso, ou restart do gateway) popula o cache com fóruns criados depois que o bot iniciou.

## Solução de problemas {#troubleshooting}

### Bot online, mas não responde a mensagens {#bot-is-online-but-not-responding-to-messages}

**Causa**: Ou o Message Content Intent está desabilitado, ou a autenticação Discord está falhando fechada porque nenhuma política de acesso está configurada.

**Solução**:

1. Vá ao [Developer Portal](https://discord.com/developers/applications) → seu app → Bot → Privileged Gateway Intents → habilite **Message Content Intent** → Save Changes.
2. Verifique que pelo menos uma política de acesso Discord está configurada:

   ```bash
   # recommended: allow specific users
   DISCORD_ALLOWED_USERS=284102345871466496

   # or allow a trusted guild/dev bot to behave like pre-0.18 Discord
   DISCORD_ALLOW_ALL_USERS=true
   ```

3. Reinicie o gateway:

   ```bash
   hermes gateway restart
   ```

Se o log do gateway diz que o Discord está conectado e checks da REST API funcionam, mas toda mensagem inbound fica silenciosa, procure este aviso em `~/.hermes/logs/gateway.log`:

```text
No Discord access policy configured; inbound Discord messages will be denied by default.
```

O Hermes 0.18 falha fechado intencionalmente em adaptadores externamente alcançáveis. Um bot Discord sem `DISCORD_ALLOWED_USERS`, sem `DISCORD_ALLOWED_ROLES`, sem `DISCORD_ALLOWED_CHANNELS` e sem flag allow-all explícita conectará com sucesso, mas negará usuários inbound antes do tratamento normal de mensagens.

### Erro "Disallowed Intents" no startup {#disallowed-intents-error-on-startup}

**Causa**: Seu código solicita intents que não estão habilitados no Developer Portal.

**Solução**: Habilite os três Privileged Gateway Intents (Presence, Server Members, Message Content) nas configurações do Bot, depois reinicie.

### Bot não vê mensagens em um canal específico {#bot-cant-see-messages-in-a-specific-channel}

**Causa**: O cargo do bot não tem permissão para ver aquele canal.

**Solução**: No Discord, vá em configurações do canal → Permissions → adicione o cargo do bot com **View Channel** e **Read Message History** habilitados.

### Erros 403 Forbidden {#403-forbidden-errors}

**Causa**: O bot está faltando permissões obrigatórias.

**Solução**: Re-convide o bot com as permissões corretas usando a URL do Passo 5, ou ajuste manualmente as permissões do cargo do bot em Server Settings → Roles.

### Bot offline {#bot-is-offline}

**Causa**: O gateway Hermes não está rodando, ou o token está incorreto.

**Solução**: Verifique se `hermes gateway` está rodando. Confirme `DISCORD_BOT_TOKEN` no seu arquivo `.env`. Se você resetou o token recentemente, atualize-o.

### "User not allowed" / Bot ignora você {#user-not-allowed--bot-ignores-you}

**Causa**: Seu User ID não está em `DISCORD_ALLOWED_USERS`.

**Solução**: Adicione seu User ID a `DISCORD_ALLOWED_USERS` em `~/.hermes/.env` e reinicie o gateway.

### Pessoas no mesmo canal compartilhando contexto inesperadamente {#people-in-the-same-channel-are-sharing-context-unexpectedly}

**Causa**: `group_sessions_per_user` está desabilitado, ou a plataforma não consegue fornecer user ID para as mensagens naquele contexto.

**Solução**: Defina isto em `~/.hermes/config.yaml` e reinicie o gateway:

```yaml
group_sessions_per_user: true
```

Se você quer intencionalmente uma conversa compartilhada de sala, deixe desligado — apenas espere histórico de transcrição compartilhado e comportamento de interrupção compartilhado.

## Segurança {#security}

:::warning
Sempre defina `DISCORD_ALLOWED_USERS` (ou `DISCORD_ALLOWED_ROLES`) para restringir quem pode interagir com o bot. Sem um ou outro, o gateway nega todos os usuários por padrão como medida de segurança. Autorize apenas pessoas em quem confia — usuários autorizados têm acesso completo às capacidades do agente, incluindo uso de ferramentas e acesso ao sistema.
:::

### Controle de acesso baseado em cargos {#role-based-access-control}

Para servidores onde o acesso é gerenciado por cargos em vez de listas individuais de usuários (equipes de moderação, suporte, ferramentas internas), use `DISCORD_ALLOWED_ROLES` — uma lista separada por vírgula de IDs de cargo. Qualquer membro com um desses cargos é autorizado.

```bash
# ~/.hermes/.env — works alongside or instead of DISCORD_ALLOWED_USERS
DISCORD_ALLOWED_ROLES=987654321098765432,876543210987654321
```

Semântica:

- **OR com allowlist de usuários.** Um usuário é autorizado se seu ID está em `DISCORD_ALLOWED_USERS` **ou** ele tem qualquer cargo em `DISCORD_ALLOWED_ROLES`.
- **Server Members Intent auto-habilitado.** Quando `DISCORD_ALLOWED_ROLES` está definido, o bot habilita o Members intent na conexão — necessário para o Discord enviar informação de cargo com registros de membro.
- **IDs de cargo, não nomes.** Pegue no Discord: **User Settings → Advanced → Developer Mode ON**, depois clique com botão direito em qualquer cargo → **Copy Role ID**.
- **Fallback em DM.** Em DMs, a verificação de cargo escaneia guilds mútuas; um usuário com cargo permitido em qualquer servidor compartilhado é autorizado em DMs também.

Este é o padrão preferido quando a equipe de moderação muda — novos moderadores ganham acesso no momento em que o cargo é concedido, sem editar `.env` ou reiniciar o gateway.

### Controle de menções {#mention-control}

Por padrão, o Hermes impede que o bot pingue `@everyone`, `@here` e menções de cargo, mesmo se sua resposta contiver esses tokens. Isso evita que um prompt mal formulado ou conteúdo ecoado do usuário spame um servidor inteiro. Pings individuais `@user` e pings de reply-reference (o chipzinho "replying to…") permanecem habilitados para conversa normal continuar funcionando.

Você pode relaxar esses padrões via env vars ou `config.yaml`:

```yaml
# ~/.hermes/config.yaml
discord:
  allow_mentions:
    everyone: false      # allow the bot to ping @everyone / @here
    roles: false         # allow the bot to ping @role mentions
    users: true          # allow the bot to ping individual @users
    replied_user: true   # ping the author when replying to their message
```

```bash
# ~/.hermes/.env — env vars win over config.yaml
DISCORD_ALLOW_MENTION_EVERYONE=false
DISCORD_ALLOW_MENTION_ROLES=false
DISCORD_ALLOW_MENTION_USERS=true
DISCORD_ALLOW_MENTION_REPLIED_USER=true
```

:::tip
Deixe `everyone` e `roles` em `false`, a menos que saiba exatamente por que precisa deles. É muito fácil um LLM produzir a string `@everyone` dentro de uma resposta de aparência normal; sem essa proteção, isso notificaria todo membro do seu servidor.
:::

Para mais informações sobre proteger seu deployment do Hermes Agent, veja o [Guia de Segurança](../security.md).

