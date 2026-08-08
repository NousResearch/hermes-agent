---
sidebar_position: 4
title: "Slack"
description: "Configure o Hermes Agent como um bot do Slack usando o Socket Mode"
---

# Configuração do Slack

Conecte o Hermes Agent ao Slack como um bot usando o Socket Mode. O Socket Mode usa WebSockets em vez de
endpoints HTTP públicos, então sua instância do Hermes não precisa estar publicamente acessível — funciona
atrás de firewalls, no seu laptop, ou em um servidor privado.

:::warning Aplicativos Clássicos do Slack Descontinuados
Aplicativos clássicos do Slack (usando a API RTM) foram **totalmente descontinuados em março de 2025**. O Hermes usa o
SDK Bolt moderno com Socket Mode. Se você tiver um aplicativo clássico antigo, precisará criar um novo seguindo
as etapas abaixo.
:::

## Visão Geral {#overview}

| Componente | Valor |
|-----------|-------|
| **Biblioteca** | `slack-bolt` / `slack_sdk` para Python (Socket Mode) |
| **Conexão** | WebSocket — nenhuma URL pública necessária |
| **Tokens de autenticação necessários** | Bot Token (`xoxb-`) + App-Level Token (`xapp-`) |
| **Identificação de usuário** | Slack Member IDs (ex.: `U01ABC2DEF3`) |

---

## Etapa 1: Criar um Aplicativo Slack {#step-1-create-a-slack-app}

O caminho mais rápido é colar um manifest que o Hermes gera para você. Ele
declara todo comando de barra nativo (`/btw`, `/stop`, `/model`, …),
todo escopo OAuth necessário, toda assinatura de eventos, e habilita o Socket
Mode — tudo de uma vez.

### Opção A: A partir de um manifest gerado pelo Hermes (recomendado) {#option-a-from-a-hermes-generated-manifest-recommended}

1. Gere o manifest. Novos aplicativos Slack devem usar a visualização Agent:
   ```bash
   hermes slack manifest --agent-view --write
   ```
   Isso grava `~/.hermes/slack-manifest.json` e exibe instruções para colar. Aplicativos
   existentes que ainda usam a visualização legada Assistant do Slack
   podem omitir `--agent-view` até estarem prontos para migrar.

   Para preencher a descrição longa do aplicativo no Slack a partir de um arquivo de texto UTF-8 ou
   Markdown existente, adicione `--long-description-file`:

   ```bash
   hermes slack manifest --agent-view \
     --long-description-file AGENTS.md --write
   ```

   O conteúdo do arquivo é preservado exatamente dentro do intervalo de 175–4.000 caracteres
   do Slack. Use `--long-description "..."` para texto inline; as opções inline
   e de arquivo são mutuamente exclusivas e não podem ser combinadas com
   `--slashes-only`.
2. Acesse [https://api.slack.com/apps](https://api.slack.com/apps) →
   **Create New App** → **From an app manifest**
3. Escolha seu workspace, cole o conteúdo JSON, revise, clique em **Next**
   → **Create**
4. Pule para a **Etapa 6: Instalar o Aplicativo no Workspace**. O manifest
   já cuidou dos escopos, eventos e comandos de barra para você.

### Opção B: Do zero (manual) {#option-b-from-scratch-manual}

1. Acesse [https://api.slack.com/apps](https://api.slack.com/apps)
2. Clique em **Create New App**
3. Escolha **From scratch**
4. Digite um nome para o aplicativo (ex.: "Hermes Agent") e selecione seu workspace
5. Clique em **Create App**

Você chegará à página **Basic Information** do aplicativo. Continue com
as Etapas 2–6 abaixo.

---

## Etapa 2: Configurar os Escopos do Bot Token {#step-2-configure-bot-token-scopes}

Navegue até **Features → OAuth & Permissions** na barra lateral. Role até **Scopes → Bot Token Scopes** e adicione os seguintes:

| Escopo | Finalidade |
|-------|---------|
| `chat:write` | Enviar mensagens como o bot |
| `app_mentions:read` | Detectar quando é @mencionado em canais |
| `channels:history` | Ler mensagens em canais públicos em que o bot está |
| `channels:read` | Listar e obter informações sobre canais públicos |
| `groups:history` | Ler mensagens em canais privados para os quais o bot foi convidado |
| `im:history` | Ler o histórico de mensagens diretas |
| `im:read` | Ver informações básicas de DMs |
| `im:write` | Abrir e gerenciar DMs |
| `mpim:history` | Ler o histórico de mensagens diretas em grupo (DM multipessoa) |
| `mpim:read` | Ver informações básicas de DMs em grupo |
| `users:read` | Consultar informações de usuários |
| `files:read` | Ler e baixar arquivos anexados, incluindo notas de voz/áudio |
| `files:write` | Fazer upload de arquivos (imagens, áudio, documentos) |

:::caution Escopos ausentes = funcionalidades ausentes
Sem `channels:history` e `groups:history`, o bot **não receberá mensagens em canais** —
funcionará apenas em DMs. Sem `files:read`, o Hermes consegue conversar, mas **não consegue ler de forma confiável os anexos enviados pelo usuário**.
Esses são os escopos mais comumente esquecidos.
:::

**Escopos opcionais:**

| Escopo | Finalidade |
|-------|---------|
| `groups:read` | Listar e obter informações sobre canais privados |
| `assistant:write` | Renderiza a linha de status de "em andamento" ("is thinking…") ao lado do nome do bot enquanto ele processa uma mensagem. Sem esse escopo, a chamada `assistant.threads.setStatus` falha silenciosamente e o Slack exibe seus próprios placeholders genéricos rotativos ("Finding answers…", "Reviewing findings…", …) — o Hermes nunca controla o texto. Necessário para que `typing_status_text` tenha algum efeito visível. |

---

## Etapa 3: Habilitar o Socket Mode {#step-3-enable-socket-mode}

O Socket Mode permite que o bot se conecte via WebSocket em vez de exigir uma URL pública.

1. Na barra lateral, vá para **Settings → Socket Mode**
2. Ative **Enable Socket Mode**
3. Você será solicitado a criar um **App-Level Token**:
   - Dê um nome como `hermes-socket` (o nome não importa)
   - Adicione o escopo **`connections:write`**
   - Clique em **Generate**
4. **Copie o token** — ele começa com `xapp-`. Esse é o seu `SLACK_APP_TOKEN`

:::tip
Você sempre pode encontrar ou regenerar tokens de nível de aplicativo em **Settings → Basic Information → App-Level Tokens**.
:::

---

## Etapa 4: Assinar Eventos {#step-4-subscribe-to-events}

Esta etapa é fundamental — ela controla quais mensagens o bot pode ver.


1. Na barra lateral, vá para **Features → Event Subscriptions**
2. Ative **Enable Events**
3. Expanda **Subscribe to bot events** e adicione:

| Evento | Obrigatório? | Finalidade |
|-------|-----------|---------|
| `message.im` | **Sim** | O bot recebe mensagens diretas |
| `message.mpim` | **Sim** | O bot recebe mensagens em **DMs em grupo** (multipessoa) às quais foi adicionado |
| `message.channels` | **Sim** | O bot recebe mensagens em canais **públicos** aos quais foi adicionado |
| `message.groups` | **Recomendado** | O bot recebe mensagens em canais **privados** para os quais foi convidado |
| `app_mention` | **Sim** | Evita erros do SDK Bolt quando o bot é @mencionado |

4. Clique em **Save Changes** no final da página

:::danger A ausência de assinaturas de eventos é o problema de configuração nº 1
Se o bot funciona em DMs, mas **não em canais**, você quase certamente esqueceu de adicionar
`message.channels` (para canais públicos) e/ou `message.groups` (para canais privados).
Sem esses eventos, o Slack simplesmente nunca entrega mensagens de canal ao bot.
:::


---

## Etapa 5: Habilitar a Aba de Mensagens {#step-5-enable-the-messages-tab}

Esta etapa habilita mensagens diretas para o bot. Sem ela, os usuários veem **"Sending messages to this app has been turned off"** ao tentar enviar uma DM ao bot.

1. Na barra lateral, vá para **Features → App Home**
2. Role até **Show Tabs**
3. Ative **Messages Tab**
4. Marque **"Allow users to send Slash commands and messages from the messages tab"**

:::danger Sem esta etapa, as DMs ficam completamente bloqueadas
Mesmo com todos os escopos e assinaturas de eventos corretos, o Slack não permitirá que os usuários enviem mensagens diretas ao bot, a menos que a Aba de Mensagens esteja habilitada. Esse é um requisito da plataforma Slack, não um problema de configuração do Hermes.
:::

---

## Etapa 6: Instalar o Aplicativo no Workspace {#step-6-install-app-to-workspace}

1. Na barra lateral, vá para **Settings → Install App**
2. Clique em **Install to Workspace**
3. Revise as permissões e clique em **Allow**
4. Após a autorização, você verá um **Bot User OAuth Token** começando com `xoxb-`
5. **Copie esse token** — esse é o seu `SLACK_BOT_TOKEN`

:::tip
Se você alterar escopos ou assinaturas de eventos posteriormente, **precisará reinstalar o aplicativo** para que as alterações tenham
efeito. A página Install App mostrará um banner solicitando que você faça isso.
:::

---

## Etapa 7: Encontrar IDs de Usuário para a Lista de Permissões {#step-7-find-user-ids-for-the-allowlist}

O Hermes usa **Member IDs** do Slack (não nomes de usuário ou nomes de exibição) para a lista de permissões.

Para encontrar um Member ID:

1. No Slack, clique no nome ou avatar do usuário
2. Clique em **View full profile**
3. Clique no botão **⋮** (mais)
4. Selecione **Copy member ID**

Os Member IDs se parecem com `U01ABC2DEF3`. Você precisa, no mínimo, do seu próprio Member ID.

---

## Etapa 8: Configurar o Hermes {#step-8-configure-hermes}

Adicione o seguinte ao seu arquivo `~/.hermes/.env`:

```bash
# Required
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SLACK_ALLOWED_USERS=U01ABC2DEF3              # Comma-separated Member IDs

# Optional
SLACK_HOME_CHANNEL=C01234567890              # Default channel for cron/scheduled messages
SLACK_HOME_CHANNEL_NAME=general              # Human-readable name for the home channel (optional)
```

Ou execute a configuração interativa:

```bash
hermes gateway setup    # Select Slack when prompted
```

Depois inicie o gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

:::tip Segurança do reasoning-effort do Codex
Para canais de peer-agent no Slack baseados no Codex, prefira `agent.reasoning_effort: high` ou menor. `xhigh`
pode gastar o turno inteiro em raciocínio oculto e nunca produzir texto visível do assistente; o Hermes agora
suprime esses avisos de turno incompleto da thread e mantém os diagnósticos nos logs do gateway.
:::

---

## Etapa 9: Convidar o Bot para Canais {#step-9-invite-the-bot-to-channels}

Depois de iniciar o gateway, você precisa **convidar o bot** para qualquer canal em que deseja que ele responda:

```
/invite @Hermes Agent
```

O bot **não** entrará automaticamente em canais. Você precisa convidá-lo individualmente para cada canal.

---

## Comandos de Barra {#slash-commands}

Todo comando do Hermes (`/btw`, `/stop`, `/new`, `/model`, `/help`, ...)
é um comando de barra nativo do Slack — exatamente da mesma forma que funcionam no Telegram
e no Discord. Digite `/` no Slack e o seletor de autocompletar lista todo
comando do Hermes com sua descrição.

Nos bastidores: o Hermes vem com um manifest de aplicativo Slack gerado (veja a
Etapa 1, Opção A) que declara todo comando em
[`COMMAND_REGISTRY`](https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/commands.py)
como um comando de barra. No Socket Mode, o Slack roteia o evento de comando
através do WebSocket independentemente do campo `url` do manifest.

### Experiência de mensagens do Agent {#agent-messaging-experience}

Novos aplicativos Slack usam a experiência de mensagens **Agent** do Slack. Aplicativos Hermes
Assistant existentes podem migrar regenerando o manifest com `--agent-view`:

```bash
hermes slack manifest --agent-view --write
```

Atualize o manifest em **Features → App Manifest** e depois reinstale o aplicativo se
o Slack solicitar. A visualização Agent não pode ser revertida para a visualização Assistant, e os usuários podem precisar
atualizar o Slack com hard-refresh após a mudança. O manifest Agent gerado assina
`message.im`, `app_home_opened` e `app_context_changed`, para que o Hermes possa
identificar uma DM na Aba de Mensagens e receber o contexto ativo do usuário no Slack em um
turno. O Hermes fornece esse contexto apenas como um rótulo; ele não lê o histórico do
canal visualizado.

### Atualizando comandos de barra após atualizações {#refreshing-slash-commands-after-updates}

Quando o Hermes adiciona novos comandos (por exemplo, após `hermes update`), regenere
o manifest e atualize seu aplicativo Slack:

```bash
hermes slack manifest --write
```

Depois, no Slack:
1. Abra [https://api.slack.com/apps](https://api.slack.com/apps) →
   seu aplicativo Hermes
2. **Features → App Manifest → Edit**
3. Cole o novo conteúdo de `~/.hermes/slack-manifest.json`
4. **Save**. O Slack solicitará a reinstalação do aplicativo se os escopos ou comandos de barra
   tiverem mudado.

### O `/hermes <subcommand>` legado ainda funciona {#legacy-hermes-subcommand-still-works}

Para compatibilidade retroativa com manifests mais antigos, você ainda pode digitar
`/hermes btw run the tests` — o Hermes o roteia da mesma forma que `/btw
run the tests`. Perguntas em texto livre também funcionam: `/hermes what's the
weather?` é tratado como uma mensagem comum.

### Usando comandos dentro de threads (o prefixo `!cmd`) {#using-commands-inside-threads-the-cmd-prefix}

O próprio Slack bloqueia comandos de barra nativos dentro de respostas em thread — tente
`/queue` em uma thread e o Slack responde com *"/queue is not supported
in threads. Sorry!"*. Não há nenhuma configuração do lado do aplicativo que os reative;
o Slack simplesmente nunca os entrega ao Hermes.

Como solução alternativa, o Hermes reconhece um `!` inicial como um prefixo de
comando alternativo que funciona em threads (e em qualquer outro lugar). Digite
`!queue`, `!stop`, `!model gpt-5.4`, etc. como uma resposta normal em thread —
o Hermes trata isso de forma idêntica à forma com barra e responde na mesma thread.

Apenas o primeiro token é verificado em relação à lista de comandos conhecidos, então
mensagens casuais como `!nice work` passam para o agente sem alterações.
A forma com `!` também funciona atrás de uma menção (`@Hermes !stop`) e com
espaço em branco inicial — ambas são despachadas como comandos em threads.

Prompts de aprovação (aprovação de comando perigoso / `execute_code`) normalmente
são renderizados como botões interativos. Quando os botões não podem ser entregues e
o Hermes recorre a um prompt de texto, o prompt instrui você a responder
com `!approve` / `!deny` — a forma que funciona dentro de threads.

### Respostas de comandos de barra são efêmeras {#slash-replies-are-ephemeral}

As respostas a um comando de barra nativo (por exemplo, `/status`, `/help`) são entregues de forma
**efêmera** — "Only visible to you" — de modo que a saída do comando nunca lota o
canal. O placeholder "Running /cmd…" é substituído pela resposta real; respostas
longas são divididas em mensagens efêmeras de acompanhamento. O Slack limita o fluxo de respostas
a 5 postagens, então saídas extremamente longas são encerradas com um aviso explícito de
truncamento em vez de serem silenciosamente descartadas. Se o caminho efêmero
principal falhar, o Hermes tenta novamente via um segundo caminho de API efêmero — uma resposta de comando de barra nunca é
postada publicamente no canal como alternativa. (Comandos digitados como mensagens comuns —
`!cmd` em threads, `@Hermes /cmd` — respondem como mensagens visíveis normais.)

### Prompts de clarificação (botões de um toque) {#clarify-prompts-one-tap-buttons}

Quando o agente precisa fazer uma pergunta de múltipla escolha (a ferramenta `clarify`),
o Slack a renderiza como **botões do Block Kit** — um toque por opção, mais um botão
"✏️ Other…" que muda para o modo de texto livre (sua próxima mensagem digitada
se torna a resposta). Após um toque, a mensagem é atualizada no lugar para mostrar quem
respondeu e o que foi escolhido; cliques adicionais no mesmo prompt são ignorados.
Cliques em botões respeitam a mesma autorização de usuário que as mensagens, e prompts
expirados (reinício do gateway, tempo esgotado) instruem você a perguntar novamente em vez de simplesmente
ignorar o clique silenciosamente. Perguntas de clarificação abertas são renderizadas como uma pergunta simples e
aceitam sua próxima resposta digitada. Nenhuma configuração é necessária — isso funciona independentemente
da configuração `rich_blocks`.

### Avançado: emitir apenas o array de comandos de barra {#advanced-emit-only-the-slash-commands-array}

Se você mantém seu manifest do Slack manualmente e só quer a lista de
comandos de barra:

```bash
hermes slack manifest --slashes-only > /tmp/slashes.json
```

Cole esse array na chave `features.slash_commands` do seu manifest
existente.

---

## Como o Bot Responde {#how-the-bot-responds}

Entendendo como o Hermes se comporta em diferentes contextos:

| Contexto | Comportamento |
|---------|----------|
| **DMs** | O bot responde a toda mensagem — nenhuma @menção é necessária |
| **Canais** | O bot **só responde quando @mencionado** (ex.: `@Hermes Agent what time is it?`). Em canais, o Hermes responde em uma thread anexada a essa mensagem. |
| **Threads** | Se você @mencionar o Hermes dentro de uma thread existente, ele responde na mesma thread. Depois que o bot tiver uma sessão ativa em uma thread, **respostas subsequentes nessa thread não exigem @menção** — o bot acompanha a conversa naturalmente. |

:::tip
Em canais, sempre @mencione o bot para iniciar uma conversa. Depois que o bot estiver ativo em uma thread, você pode responder nessa thread sem mencioná-lo. Fora de threads, mensagens sem @menção são ignoradas para evitar ruído em canais movimentados.
:::

---

## Opções de Configuração {#configuration-options}

Além das variáveis de ambiente obrigatórias da Etapa 8, você pode personalizar o comportamento do bot do Slack através de `~/.hermes/config.yaml`.

### Comportamento de Thread e Resposta {#thread-reply-behavior}

```yaml
platforms:
  slack:
    # Controls how multi-part responses are threaded
    # "off"   — never thread replies to the original message
    # "first" — first chunk threads to user's message (default)
    # "all"   — all chunks thread to user's message
    reply_to_mode: "first"

    extra:
      # Whether to reply in a thread (default: true).
      # When false, channel messages get direct channel replies instead
      # of threads. Messages inside existing threads still reply in-thread.
      reply_in_thread: true

      # Also post thread replies to the main channel
      # (Slack's "Also send to channel" feature).
      # Only the first chunk of the first reply is broadcast.
      reply_broadcast: false

      # Render agent messages as Slack Block Kit blocks (default: false).
      # When true, the final agent message is sent with structured blocks —
      # section headers, dividers, true nested lists (via rich_text), and
      # native Block Kit tables — instead of flat mrkdwn text. A plain-text
      # fallback is always sent alongside for notifications/accessibility.
      # Tables exceeding Slack's limits (100 rows / 20 cols / 10k chars)
      # gracefully fall back to aligned monospace.
      rich_blocks: false

      # Append Slack-native feedback controls to final Block Kit replies.
      # Requires rich_blocks: true. Default: false.
      feedback_buttons: false

      # Suggested prompts pinned at the top of Agent view's Messages tab.
      # Either a list of {title, message} rows, or a titled object:
      # {title: "Start here", prompts: [{title: "Plan", message: "..."}]}
      suggested_prompts: []

      # Title Agent/Assistant DM threads from the first user message.
      # Default: true. Set false to leave Slack's default thread titles.
      assistant_thread_titles: true

      # Accept messages posted by other Slack bots (default: "none").
      # "none" ignores bots, "mentions" accepts a bot message only when
      # that message itself @mentions Hermes, and "all" accepts every
      # other bot. Hermes always ignores its own bot user to prevent
      # self-echoes.
      allow_bots: "none"

      # Continuable-cron delivery surface (default: "thread").
      # "in_channel" delivers a continuable cron job FLAT into the channel
      # (no dedicated thread); pair with reply_in_thread: false (and
      # require_mention: false) so a plain reply continues the job.
      # See the cron guide → "Flat, in-channel continuation".
      cron_continuable_surface: thread
```

| Chave | Padrão | Descrição |
|-----|---------|-------------|
| `platforms.slack.reply_to_mode` | `"first"` | Modo de threading para mensagens de múltiplas partes: `"off"`, `"first"`, ou `"all"` |
| `platforms.slack.extra.reply_in_thread` | `true` | Quando `false`, mensagens de canal recebem respostas diretas em vez de threads. Mensagens dentro de threads existentes continuam respondendo na thread. |
| `platforms.slack.extra.reply_broadcast` | `false` | Quando `true`, respostas em thread também são postadas no canal principal. Apenas o primeiro trecho é transmitido. |
| `platforms.slack.extra.rich_blocks` | `false` | Quando `true`, mensagens do agente são renderizadas como blocos do [Block Kit](https://docs.slack.dev/block-kit/) (cabeçalhos, divisores, listas aninhadas reais e tabelas nativas). Um fallback em texto simples é sempre enviado junto. Tabelas que excedem os limites do Slack recorrem a texto monoespaçado alinhado. Nenhuma reinstalação do aplicativo é necessária — é uma alteração apenas do lado do envio. |
| `platforms.slack.extra.feedback_buttons` | `false` | Quando `true` junto com `rich_blocks`, adiciona controles de feedback nativos do Slack às respostas finais. |
| `platforms.slack.extra.suggested_prompts` | `[]` | Até quatro prompts `{title, message}` para os pontos de entrada de DM do Agent/Assistant; aceita uma lista ou `{title, prompts}`. |
| `platforms.slack.extra.assistant_thread_titles` | `true` | Quando `true`, nomeia as threads de DM do Agent/Assistant a partir da primeira mensagem do usuário. |
| `platforms.slack.extra.allow_bots` | `"none"` | Controla mensagens de outros bots do Slack: `"none"` os ignora, `"mentions"` aceita uma mensagem de bot apenas quando **essa própria mensagem** @menciona o Hermes, e `"all"` aceita todas elas. Use `"mentions"` para o modo mais seguro de colaboração bot a bot. Veja [Aceitando mensagens de outros bots](#accepting-messages-from-other-bots-allow_bots). |
| `platforms.slack.extra.cron_continuable_surface` | `"thread"` | Superfície de entrega para [jobs de cron continuáveis](../features/cron.md#flat-in-channel-continuation-slack). `"thread"` abre uma thread dedicada por entrega (padrão); `"in_channel"` entrega de forma direta na linha do tempo do canal. Combine `in_channel` com `reply_in_thread: false` (e `require_mention: false`) para que uma resposta simples no canal continue o job. |

A variável de ambiente equivalente é `SLACK_ALLOW_BOTS=none|mentions|all`.
Quando ambas são definidas, `platforms.slack.extra.allow_bots` tem precedência. Evite
`all` quando bots pares podem responder uns aos outros sem uma menção explícita, porque
suas próprias políticas de resposta ainda podem criar loops.

### Linha de Status de Trabalho em Andamento {#working-state-status-line}

Enquanto o agente processa uma mensagem, o Slack exibe uma linha de status ao lado do nome
do bot na thread. Por padrão, o Hermes a define como `is thinking...`; personalize-a
com `typing_status_text` — por exemplo, um assistente gatinho chamado Ada:

```yaml
platforms:
  slack:
    # Custom working-state status line (default: "is thinking...").
    typing_status_text: "is pouncing… 🐾"
```

| Chave | Padrão | Descrição |
|-----|---------|-------------|
| `platforms.slack.typing_status_text` | `"is thinking..."` | Texto da linha de status de trabalho em andamento exibida enquanto o agente processa uma mensagem. Requer o escopo `assistant:write` — sem ele, a chamada de status falha silenciosamente e o Slack renderiza seu próprio placeholder genérico, independentemente do que estiver configurado aqui. Defina `typing_indicator: false` para desabilitar completamente a linha de status. |

:::note Onde o status é renderizado
O status personalizado aparece no **rodapé abaixo do compositor de respostas** ("*NomeDoBot* is thinking…"), não embutido na lista de mensagens. As linhas embutidas "Generating response…" / "Finding answers…" que o Slack exibe na área de mensagens enquanto um aplicativo de IA trabalha são **indicadores rotativos próprios do Slack** — `assistant.threads.setStatus` não controla esses, e ambos podem aparecer ao mesmo tempo.
:::

A mesma chave personaliza o marcador visível de trabalho em andamento do Google Chat
(`platforms.google_chat.typing_status_text`, padrão `"Hermes is thinking…"`) —
observe que no Google Chat é uma mensagem realmente postada que é corrigida na
resposta, não um status efêmero.

### Status ao Vivo (por ferramenta) {#live-status-per-tool}

Por padrão, a linha de status é atualizada **ao vivo enquanto o agente trabalha**: em vez de um
`is thinking...` estático, ela mostra o que o agente está fazendo no momento — `is
running pytest tests/…`, `is reading docs/api.md…`, `is searching the web for
slack api limits…`. Entre as chamadas de ferramentas, ela reverte para o texto estático. Isso
aproveita o ciclo de atualização de status existente, então não faz chamadas adicionais à API do Slack,
e funciona mesmo com `tool_progress: off` (o padrão do Slack) — ao contrário das
bolhas de progresso, a linha de status é efêmera e não deixa nada para trás no
canal.

Controle isso com `display.live_status` (global ou por plataforma):

```yaml
display:
  platforms:
    slack:
      # full = verb + argument ("is running pytest…")   [default]
      # verb = verb only ("is running…") — hides commands/paths,
      #        useful in shared or customer-facing channels
      # off  = static text (typing_status_text or "is thinking...")
      live_status: full
```

| Chave | Padrão | Descrição |
|-----|---------|-------------|
| `display.live_status` | `"full"` | Linha de status ao vivo por ferramenta. `full` mostra verbo + prévia do argumento; `verb` mostra apenas o verbo (mantém caminhos de arquivo e comandos fora de canais compartilhados); `off` restaura o texto estático. Requer o escopo `assistant:write`, assim como a linha de status estática. |

### Isolamento de Sessão {#session-isolation}

```yaml
# Global setting — applies to Slack and all other platforms
group_sessions_per_user: true
```

Quando `true` (o padrão), cada usuário em um canal compartilhado obtém sua própria sessão de conversa isolada. Duas pessoas conversando com o Hermes em `#general` terão históricos e contextos separados.

Defina como `false` se você quiser um modo colaborativo em que todo o canal compartilhe uma única sessão de conversa. Esteja ciente de que isso significa que os usuários compartilham o crescimento de contexto e os custos de tokens, e o `/reset` de um usuário limpa a sessão para todos.

### Comportamento de Menção e Acionamento {#mention-trigger-behavior}

```yaml
slack:
  # Require @mention in channels (this is the default behavior;
  # the Slack adapter enforces @mention gating in channels regardless,
  # but you can set this explicitly for consistency with other platforms)
  require_mention: true

  # Prevent thread auto-engagement: only reply to channel messages that
  # contain an explicit @mention. With this OFF (default), Slack can
  # "auto-engage" — remembering past mentions in a thread and following
  # up on bot-message replies, and resuming active sessions without a
  # fresh mention. With strict_mention ON, every new channel message
  # must @mention the bot before Hermes will respond.
  strict_mention: false

  # Ignore messages addressed to another user: when a channel or thread
  # message *opens* by @mentioning someone other than the bot (e.g.
  # "@rasha can you take this?"), stay silent unless the bot is also
  # mentioned. Only a *leading* mention counts as "addressed to" — a
  # message that references someone mid-sentence ("loop in @rasha")
  # still reaches the bot. Overrides free_response_channels and thread
  # auto-engagement. Opt-in; default off. Env: SLACK_IGNORE_OTHER_USER_MENTIONS.
  ignore_other_user_mentions: false

  # Require an explicit @mention for THREAD replies, while leaving
  # top-level channel messages governed by require_mention /
  # free_response_channels. Narrower than strict_mention: use it when a
  # free-response bot should not join every follow-up in busy threads.
  # Opt-in; default off. Env: SLACK_THREAD_REQUIRE_MENTION.
  thread_require_mention: false

  # Per-channel force-mention override — the opposite direction of
  # free_response_channels. Channels listed here ALWAYS require an
  # explicit @mention, even when require_mention is false globally.
  # Ongoing conversations still auto-follow (mentioned threads, active
  # sessions, bot-authored threads). Comma-separated IDs or a list.
  # Env: SLACK_REQUIRE_MENTION_CHANNELS.
  require_mention_channels: ""

  # Custom mention patterns that trigger the bot
  # (in addition to the default @mention detection)
  mention_patterns:
    - "hey hermes"
    - "hermes,"

  # Text prepended to every outgoing message
  reply_prefix: ""
```

:::tip Quando usar `strict_mention`
Defina isso como `true` em workspaces movimentados onde o comportamento padrão do Slack de "o bot lembra desta thread" surpreende os usuários — por exemplo, uma longa thread de suporte técnico em que o bot ajudou no início e você prefere que ele fique em silêncio a menos que seja explicitamente chamado novamente. DMs e sessões interativas ativas não são afetadas.
:::

:::tip Quando usar `ignore_other_user_mentions`
Defina isso como `true` quando o bot acompanha threads movimentadas (via auto-engajamento de thread ou `free_response_channels`) e se intromete em mensagens que humanos endereçam uns aos outros. É uma ferramenta mais específica que `strict_mention`: acompanhamentos simples em uma thread engajada ainda recebem respostas; apenas mensagens que começam @mencionando outra pessoa são ignoradas. **DMs 1:1 não são afetadas**; DMs em grupo (MPIMs) e canais aplicam isso da mesma forma, seguindo a política de superfície compartilhada abaixo. Tokens de transmissão (`@here`, `@channel`) e referências de canal endereçam a sala, não uma pessoa, então nunca são ignorados.
:::

:::info
O Slack suporta ambos os padrões: por padrão, é necessário @mencionar para iniciar uma conversa, mas você pode excluir canais específicos disso via `SLACK_FREE_RESPONSE_CHANNELS` (IDs de canal separados por vírgula) ou `slack.free_response_channels` em `config.yaml`. Depois que o bot tiver uma sessão ativa em uma thread, respostas subsequentes na thread não exigem menção. Em **DMs 1:1**, o bot sempre responde sem precisar de menção.
:::

:::caution DMs em grupo (MPIMs) são superfícies compartilhadas, não DMs 1:1
Uma **mensagem direta 1:1** é uma conversa privada com uma pessoa, então é isenta de menção. Uma **DM em grupo (MPIM / DM multipessoa)** é uma *superfície compartilhada* — várias pessoas podem ver e acionar o bot — então ela obedece aos mesmos controles do operador que um canal: `require_mention`, `strict_mention`, `free_response_channels` e `allowed_channels` se aplicam, e o bot só adiciona reações `:eyes:`/`:white_check_mark:` quando é realmente `@mencionado`. Para permitir que o bot responda livremente em uma DM em grupo específica, adicione o ID desse canal (começa com `G`) a `free_response_channels`.
:::

#### Qual opção de menção eu quero? {#which-mention-option-do-i-want}

As opções de restrição se combinam — cada uma responde a uma pergunta diferente:

| Opção | Pergunta que responde | Padrão | Escopo |
|--------|--------------------|---------|-------|
| `require_mention` | **Mensagens de canal de nível superior** precisam de @menção? | `true` | Todos os canais |
| `free_response_channels` | Quais canais são isentos de `require_mention`? | nenhum | Canais listados |
| `require_mention_channels` | Quais canais SEMPRE precisam de @menção, mesmo quando `require_mention` é `false` ou o canal é de resposta livre? Prevalece sobre ambos. | nenhum | Canais listados |
| `thread_require_mention` | **Respostas em thread** precisam de @menção, mesmo quando mensagens de nível superior não precisam? Threads mencionadas não são lembradas. | `false` | Apenas threads |
| `strict_mention` | **Toda** mensagem de canal (nível superior e thread) precisa de uma @menção nova? Desabilita todo auto-acompanhamento: memória de thread mencionada, acompanhamentos de resposta do bot, retomada de sessão ativa. | `false` | Todos os canais + threads |
| `ignore_other_user_mentions` | Uma mensagem que **começa @mencionando outra pessoa** (`@rasha can you take this?`) deve ser ignorada? Sobrepõe resposta livre e auto-acompanhamento de thread; referências no meio da frase ainda alcançam o bot. | `false` | Canais + DMs em grupo |

Regras práticas: `strict_mention` é o martelo mais amplo; `thread_require_mention` silencia threads movimentadas sem afetar a restrição de nível superior; `require_mention_channels` volta a restringir canais individuais em um bot de resposta livre; `ignore_other_user_mentions` só ignora mensagens explicitamente endereçadas a outra pessoa. DMs 1:1 sempre respondem e não são afetadas por nenhuma dessas.

### Aceitando mensagens de outros bots (`allow_bots`) {#accepting-messages-from-other-bots-allow_bots}

Por padrão, o Hermes ignora toda mensagem criada por outro bot ou aplicativo do Slack (incluindo postagens do Workflow Builder). Para workspaces multiagente — várias instâncias do Hermes ou bots pares colaborando em um canal — habilite com `allow_bots`:

```yaml
platforms:
  slack:
    extra:
      # "none" (default) — ignore all bot/app-authored messages
      # "mentions"       — accept a bot message only when THAT message
      #                    @mentions this bot
      # "all"            — accept every bot message (except the bot's own)
      allow_bots: mentions
```

Equivalente de ambiente: `SLACK_ALLOW_BOTS=none|mentions|all` (a chave de configuração prevalece quando ambas são definidas). Valores desconhecidos são tratados como `none`.

Como o modo `mentions` restringe:

- Uma mensagem de bot par é aceita **somente quando a própria mensagem contém uma `@menção` atual a este bot** — em seu texto ou em seus blocos do Block Kit. O histórico da thread não conta: um bot ter sido mencionado anteriormente na thread, respostas às próprias mensagens do bot, e sessões de thread ativas **não** admitem mensagens posteriores de bots pares sem menção. Isso é deliberado — é o que quebra loops de confirmação/status entre agentes.
- Mensagens humanas não são afetadas; a restrição de menção normal se aplica a elas.
- O Hermes sempre ignora suas próprias mensagens, em todos os modos, para evitar loops de auto-eco.

`mentions` é o modo recomendado para colaboração bot a bot: cada agente precisa convocar explicitamente o outro a cada turno. Evite `all`, a menos que a política de resposta de cada bot par seja segura contra loops — dois bots que respondem a tudo vão responder um ao outro para sempre. A detecção cobre mensagens de bot rotuladas (`bot_id`, `subtype: bot_message`), eventos originados de aplicativos, e *usuários* bot não rotulados (verificados via `users.info`), então agentes Hermes pares são filtrados de forma consistente entre workspaces.

Para implantações estritas com múltiplos bots, combine com `require_mention: true` e `strict_mention: true` — veja o perfil de verificação rápida abaixo.

### Gatilhos de Reação (`reaction_triggers`) {#reaction-triggers-reaction_triggers}

Por padrão, reações de emoji são reconhecidas e descartadas — um 👍 em uma mensagem do bot não faz nada. Defina `slack.reaction_triggers` para rotear reações para o loop do agente (requer o escopo `reactions:read` mais as assinaturas de eventos de bot `reaction_added`/`reaction_removed` no seu manifest do aplicativo Slack — regenere com `hermes slack manifest`):

```yaml
slack:
  # Opt-in. false/absent (default) = reactions are acked and dropped.
  # true = any reaction ON THE BOT'S OWN MESSAGES routes to the agent.
  reaction_triggers: true
  # Or an explicit emoji allowlist — only these names route, and they may
  # target ANY message (emoji-handoff workflows, e.g. :task: to capture):
  # reaction_triggers: [white_check_mark, thumbsup, task]
  # Optional handoff target: respond in this channel (top-level) or thread
  # (C123:<thread_ts>) instead of the reacted-to message's thread.
  # reaction_trigger_target: C0123456789
```

Equivalentes de ambiente: `SLACK_REACTION_TRIGGERS` (`true`/`all` ou uma lista separada por vírgula) e `SLACK_REACTION_TRIGGER_TARGET`.

Comportamento:

- A reação chega como um turno de agente normal com o texto
  `reaction:added:👍` / `reaction:removed:👍` (nomes comuns do Slack são
  traduzidos para unicode; nomes desconhecidos passam como estão, ex.:
  `reaction:added:custom-emoji`), encadeada sob a mensagem que recebeu a reação, para que o agente veja o que foi reagido e o
  turno chegue na mesma sessão que uma resposta chegaria.
- Quem reagiu se torna o usuário da mensagem, então **a autorização de usuário e a restrição de
  `allowed_channels` se aplicam exatamente como para mensagens digitadas** — a reação de um
  usuário aleatório não pode acionar o agente em nenhum lugar onde sua mensagem não pudesse.
- Com `reaction_triggers: true`, apenas reações nas **próprias** mensagens do bot
  são roteadas (fluxos de aprovação/confirmação). Com uma lista de permissões explícita de
  emojis, os emojis listados são roteados a partir de qualquer mensagem.
- As próprias reações de ciclo de vida do bot (`:eyes:` etc.) nunca retornam.
- Independentemente dessa opção, toda reação humana dispara os
  [hooks de gateway](../features/hooks.md#available-events) `reaction:added`/`reaction:removed`
  para observadores que não precisam de turnos do agente.

### Verificação Rápida de Peer-Agent {#peer-agent-smoke-check}

Para implantações do Slack com múltiplos bots que dependem de menções estritas por turno, mantenha o seguinte perfil:

```yaml
slack:
  require_mention: true
  strict_mention: true
  allow_bots: mentions
  allowed_channels: ""
```

Após alterações de configuração do gateway, deploys, ou reinicializações, execute este alvo de verificação sintético:

```bash
uv run --frozen pytest -q tests/gateway/test_slack_peer_agent_smoke.py -o addopts=''
```

Este alvo usa apenas eventos sintéticos do Slack em processo. Ele não envia mensagens reais do Slack e não requer tokens de bot reais por padrão.

Categorias de falha:

- `config:` `test_peer_agent_smoke_preflight_contract` detectou uma incompatibilidade de perfil (`require_mention`, `strict_mention`, `allow_bots`, ou `allowed_channels`).
- `platform_connectivity:` o adaptador/cliente não foi inicializado, então a verificação de roteamento ainda não é um sinal confiável.
- `bot_identity:` o adaptador nunca resolveu o ID de usuário do bot, então as verificações de menção da mensagem atual não podem funcionar.
- `routing_logic:` o adaptador do Slack regrediu em um dos invariantes de peer-agent (roteamento de menção humana, ignorar bot par, admitir menção explícita de par, ou supressão passiva de confirmação/status/erro).

Se este alvo passar, mas um workspace real ainda estiver roteando mensagens incorretamente, investigue a conectividade do token/workspace do Slack e o estado de implantação em tempo de execução fora da própria lógica de roteamento.

### Lista de permissões de canal (`allowed_channels`) {#channel-allowlist-allowed_channels}

Restringe o bot a um conjunto fixo de canais do Slack — útil quando o bot é convidado para muitos canais, mas deve responder apenas em alguns. Quando definido, mensagens de canais que NÃO estão nessa lista são **silenciosamente ignoradas**, mesmo se o bot for `@mencionado`.

**DMs 1:1 são isentas** desse filtro, então usuários autorizados sempre podem alcançar o bot em uma mensagem direta. **DMs em grupo (MPIMs) não são isentas** — assim como os canais, uma MPIM precisa estar na lista de permissões (seu ID começa com `G`) ou suas mensagens são descartadas.

```yaml
slack:
  allowed_channels:
    - "C0123456789"   # #ops
    - "C0987654321"   # #incident-response
```

Ou via variável de ambiente (separada por vírgulas):

```bash
SLACK_ALLOWED_CHANNELS="C0123456789,C0987654321"
```

Comportamento:

- Vazio / não definido → sem restrição (totalmente compatível com versões anteriores).
- Não vazio → o ID do canal precisa estar na lista, ou a mensagem é descartada antes de qualquer outra restrição (exigência de menção, `free_response_channels`, etc.) ser executada.
- IDs de canal do Slack começam com `C` (público), `G` (privado), ou `D` (DM). Consulte-os através do painel "Open channel details" → "About" da interface do Slack, ou via a API.

Veja também: [divisão de comandos de barra admin/usuário](../../reference/slash-commands.md#permissions-and-adminuser-split).

### Tratamento de Usuário Não Autorizado {#unauthorized-user-handling}

```yaml
slack:
  # What happens when an unauthorized user (not in SLACK_ALLOWED_USERS) DMs the bot
  # "pair"   — prompt them for a pairing code (default)
  # "ignore" — silently drop the message
  unauthorized_dm_behavior: "pair"
```

Você também pode definir isso globalmente para todas as plataformas:

```yaml
unauthorized_dm_behavior: "pair"
```

A configuração específica da plataforma sob `slack:` tem precedência sobre a configuração global.

### Transcrição de Voz {#voice-transcription}

```yaml
# Global setting — enable/disable automatic transcription of incoming voice messages
stt_enabled: true
```

Quando `true` (o padrão), mensagens de áudio recebidas são automaticamente transcritas usando o provedor de STT configurado antes de serem processadas pelo agente.

### Exemplo Completo {#full-example}

```yaml
# Global gateway settings
group_sessions_per_user: true
unauthorized_dm_behavior: "pair"
stt_enabled: true

# Slack-specific settings
slack:
  require_mention: true
  unauthorized_dm_behavior: "pair"

# Platform config
platforms:
  slack:
    reply_to_mode: "first"
    extra:
      reply_in_thread: true
      reply_broadcast: false
```

---


## Canal Principal {#home-channel}

Defina `SLACK_HOME_CHANNEL` para um ID de canal onde o Hermes entregará mensagens agendadas,
resultados de jobs de cron, e outras notificações proativas. Para encontrar um ID de canal:

1. Clique com o botão direito no nome do canal no Slack
2. Clique em **View channel details**
3. Role até o final — o ID do Canal é exibido ali

```bash
SLACK_HOME_CHANNEL=C01234567890
```

Certifique-se de que o bot foi **convidado para o canal** (`/invite @Hermes Agent`).

### Direcionamento de entrega do cron {#cron-delivery-targeting}

Jobs de cron (veja o [guia de cron](../features/cron.md#delivery-options)) podem direcionar para o Slack de três formas:

| Valor de `deliver:` | Onde é entregue |
|------------------|----------------|
| `slack` | O canal principal (`SLACK_HOME_CHANNEL`) |
| `slack:C0123456789` | Um canal específico por ID |
| `slack:U0123456789` | A **DM** desse usuário — o ID de usuário simples é resolvido automaticamente para uma conversa de DM (requer o escopo `im:write`) |

A entrega funciona mesmo quando o processo de cron não está colocado junto com o gateway — o Hermes recorre a um remetente independente de Web API usando `SLACK_BOT_TOKEN`. Anexos `MEDIA:` na saída do cron são enviados como compartilhamentos de arquivo nativos do Slack para o mesmo destino.

### Enviando mensagens e mídia (`send_message`) {#sending-messages-and-media-send_message}

A ferramenta `send_message` do agente aceita os mesmos formatos de destino: um ID de canal (`C…`/`G…`), uma conversa de DM (`D…`), ou um ID de usuário simples (`U…`/`W…`), que é resolvido para a DM do usuário em todos os caminhos de envio — texto, mídia e prompts interativos igualmente. Anexos `MEDIA:<path>` (imagens, PDFs, documentos) são enviados como compartilhamentos de arquivo nativos; quando uma mensagem curta acompanha um único anexo, ela vai como a legenda do arquivo em vez de uma mensagem separada. Arquivos ausentes são reportados por arquivo como avisos em vez de falhar todo o envio.

---

## Suporte a Múltiplos Workspaces {#multi-workspace-support}

O Hermes pode se conectar a **múltiplos workspaces do Slack** simultaneamente usando uma única instância de gateway. Cada workspace é autenticado independentemente com seu próprio ID de usuário de bot.

### Configuração {#configuration}

Forneça múltiplos bot tokens como uma **lista separada por vírgulas** em `SLACK_BOT_TOKEN`:

```bash
# Multiple bot tokens — one per workspace
SLACK_BOT_TOKEN=xoxb-workspace1-token,xoxb-workspace2-token,xoxb-workspace3-token

# A single app-level token is still used for Socket Mode
SLACK_APP_TOKEN=xapp-your-app-token
```

Ou em `~/.hermes/config.yaml`:

```yaml
platforms:
  slack:
    token: "xoxb-workspace1-token,xoxb-workspace2-token"
```

### Arquivo de Token OAuth {#oauth-token-file}

Além dos tokens no ambiente ou na configuração, o Hermes também carrega tokens de um **arquivo de token OAuth** em:

```
~/.hermes/slack_tokens.json
```

Esse arquivo é um objeto JSON que mapeia IDs de equipe para entradas de token:

```json
{
  "T01ABC2DEF3": {
    "token": "xoxb-workspace-token-here",
    "team_name": "My Workspace"
  }
}
```

Os tokens desse arquivo são mesclados com quaisquer tokens especificados via `SLACK_BOT_TOKEN`. Tokens duplicados são automaticamente removidos.

### Como funciona {#how-it-works}

- O **primeiro token** na lista é o token primário, usado para a conexão do Socket Mode (AsyncApp).
- Cada token é autenticado via `auth.test` na inicialização. O gateway mapeia cada `team_id` para seu próprio `WebClient` e `bot_user_id`.
- Quando uma mensagem chega, o Hermes usa o cliente específico do workspace correto para responder.
- O `bot_user_id` primário (do primeiro token) é usado para compatibilidade retroativa com funcionalidades que esperam uma única identidade de bot.

---

## Mensagens de Voz {#voice-messages}

O Hermes suporta voz no Slack:

- **Recebendo:** Mensagens de voz/áudio são automaticamente transcritas usando o provedor de STT configurado: `faster-whisper` local, Groq Whisper (`GROQ_API_KEY`), ou OpenAI Whisper (`VOICE_TOOLS_OPENAI_KEY`)
- **Enviando:** Respostas de TTS são enviadas como anexos de arquivo de áudio

---

## Prompts por Canal {#per-channel-prompts}

Atribua prompts de sistema efêmeros a canais específicos do Slack. O prompt é injetado em tempo de execução a cada turno — nunca persistido no histórico da transcrição — então as alterações entram em vigor imediatamente.

```yaml
slack:
  channel_prompts:
    "C01RESEARCH": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "C02ENGINEERING": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

As chaves são IDs de canal do Slack (encontre-os via detalhes do canal → "About" → role até o final). Todas as mensagens no canal correspondente recebem o prompt injetado como uma instrução de sistema efêmera.

## Vinculações de Skill por Canal {#per-channel-skill-bindings}

Carregue automaticamente uma skill sempre que uma nova sessão começar em um canal ou DM específico. Diferente dos prompts por canal (que são injetados a cada turno), as vinculações de skill injetam o conteúdo da skill como uma mensagem do usuário no **início da sessão** — ela se torna parte do histórico da conversa e não precisa ser recarregada em turnos subsequentes.

Isso é ideal para DMs ou canais com um propósito dedicado (flashcards, um bot de perguntas e respostas específico de domínio, um canal de triagem de suporte, etc.) onde você não quer que o próprio seletor de skills do modelo decida se deve carregar a cada resposta curta.

```yaml
slack:
  channel_skill_bindings:
    # DM channel — always runs in "german-flashcards" mode
    - id: "D0ATH9TQ0G6"
      skills:
        - german-flashcards
    # Research channel — preload multiple skills in order
    - id: "C01RESEARCH"
      skills:
        - arxiv
        - writing-plans
    # Short form: single skill as a string
    - id: "C02SUPPORT"
      skill: hubspot-on-demand
```

Notas:
- A vinculação corresponde pelo ID do canal. Para mensagens em thread em um canal vinculado, a thread herda a vinculação do canal pai.
- A skill é carregada apenas no início da sessão (nova sessão ou após reinicialização automática). Se você alterar a vinculação, execute `/new` ou aguarde a sessão reiniciar automaticamente para que tenha efeito.
- Combine com `channel_prompts` para tom/restrições por canal além das instruções da skill.

## Solução de Problemas {#troubleshooting}

| Problema | Solução |
|---------|----------|
| O bot não responde a DMs | Verifique se `message.im` está em suas assinaturas de eventos e se o aplicativo foi reinstalado |
| O bot funciona em DMs, mas não em canais | **Problema mais comum.** Adicione `message.channels` e `message.groups` às assinaturas de eventos, reinstale o aplicativo, e convide o bot para o canal com `/invite @Hermes Agent` |
| O bot não responde a @menções em canais | 1) Verifique se o evento `message.channels` está assinado. 2) O bot precisa ser convidado para o canal. 3) Certifique-se de que o escopo `channels:history` foi adicionado. 4) Reinstale o aplicativo após alterações de escopo/evento |
| O bot ignora mensagens em canais privados | Adicione tanto a assinatura de evento `message.groups` quanto o escopo `groups:history`, depois reinstale o aplicativo e `/invite` o bot |
| O bot não responde em DMs em grupo (DMs multipessoa) | Adicione a assinatura de evento `message.mpim` e o escopo `mpim:history` (mais `mpim:read`), depois **reinstale** o aplicativo. Sem `message.mpim`, o Slack nunca entrega mensagens de DM em grupo ao bot — mesmo que as DMs 1:1 funcionem. |
| "Sending messages to this app has been turned off" em DMs | Habilite a **Aba de Mensagens** nas configurações do App Home (veja a Etapa 5) |
| Erros de "not_authed" ou "invalid_auth" | Regenere seu Bot Token e App Token, atualize o `.env` |
| O bot responde, mas não consegue postar em um canal | Convide o bot para o canal com `/invite @Hermes Agent` |
| O bot consegue conversar, mas não consegue ler imagens/arquivos enviados | Adicione `files:read`, depois **reinstale** o aplicativo. O Hermes agora exibe diagnósticos de acesso a anexos no chat quando o Slack retorna falhas de escopo/autenticação/permissão. |
| Erro `missing_scope` | Adicione o escopo necessário em OAuth & Permissions, depois **reinstale** o aplicativo |
| Desconexões frequentes do socket | Verifique sua rede; o Bolt reconecta automaticamente, mas conexões instáveis causam atrasos |
| Escopos/eventos alterados, mas nada mudou | Você **precisa reinstalar** o aplicativo no seu workspace após qualquer alteração de escopo ou assinatura de evento |

### Checklist Rápida {#quick-checklist}

Se o bot não estiver funcionando em canais, verifique **todos** os itens a seguir:

1. ✅ O evento `message.channels` está assinado (para canais públicos)
2. ✅ O evento `message.groups` está assinado (para canais privados)
3. ✅ O evento `app_mention` está assinado
4. ✅ O escopo `channels:history` foi adicionado (para canais públicos)
5. ✅ O escopo `groups:history` foi adicionado (para canais privados)
6. ✅ O aplicativo foi **reinstalado** após adicionar escopos/eventos
7. ✅ O bot foi **convidado** para o canal (`/invite @Hermes Agent`)
8. ✅ Você está **@mencionando** o bot na sua mensagem

---

## Segurança {#security}

:::warning
**Sempre defina `SLACK_ALLOWED_USERS`** com os Member IDs dos usuários autorizados. Sem essa configuração,
o gateway **negará todas as mensagens** por padrão como medida de segurança. Nunca compartilhe seus tokens de bot —
trate-os como senhas.
:::

- Os tokens devem ser armazenados em `~/.hermes/.env` (permissões de arquivo `600`)
- Gire os tokens periodicamente através das configurações do aplicativo Slack
- Audite quem tem acesso ao seu diretório de configuração do Hermes
- O Socket Mode significa que nenhum endpoint público é exposto — uma superfície de ataque a menos
