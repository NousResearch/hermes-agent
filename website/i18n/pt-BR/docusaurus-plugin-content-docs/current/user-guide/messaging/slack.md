---
sidebar_position: 4
title: "Slack"
description: "Configure o Hermes Agent como bot do Slack usando Socket Mode"
---

# Configuração do Slack {#slack-setup}

Conecte o Hermes Agent ao Slack como bot usando Socket Mode. O Socket Mode usa WebSockets em vez de
endpoints HTTP públicos, então sua instância do Hermes não precisa ser acessível publicamente — funciona
atrás de firewalls, no seu laptop ou em um servidor privado.

:::warning Apps clássicos do Slack descontinuados
Apps clássicos do Slack (que usavam a RTM API) foram **totalmente descontinuados em março de 2025**. O Hermes usa o
Bolt SDK moderno com Socket Mode. Se você tem um app clássico antigo, deve criar um novo seguindo
as etapas abaixo.
:::

## Visão geral {#overview}

| Component | Value |
|-----------|-------|
| **Library** | `slack-bolt` / `slack_sdk` for Python (Socket Mode) |
| **Connection** | WebSocket — no public URL required |
| **Auth tokens needed** | Bot Token (`xoxb-`) + App-Level Token (`xapp-`) |
| **User identification** | Slack Member IDs (e.g., `U01ABC2DEF3`) |

---

## Passo 1: Crie um app Slack {#step-1-create-a-slack-app}

O caminho mais rápido é colar um manifest que o Hermes gera para você. Ele
declara todo comando slash embutido (`/btw`, `/stop`, `/model`, …),
todo escopo OAuth necessário, toda assinatura de evento e habilita o Socket
Mode — tudo de uma vez.

### Opção A: A partir de um manifest gerado pelo Hermes (recomendado) {#option-a-from-a-hermes-generated-manifest-recommended}

1. Gere o manifest. Novos apps Slack devem usar a visualização Agent:
   ```bash
   hermes slack manifest --agent-view --write
   ```
   Isso grava `~/.hermes/slack-manifest.json` e imprime instruções para
   colar. Apps existentes que ainda usam a visualização Assistant legada do Slack
   podem omitir `--agent-view` até estarem prontos para migrar.
2. Acesse [https://api.slack.com/apps](https://api.slack.com/apps) →
   **Create New App** → **From an app manifest**
3. Escolha seu workspace, cole o conteúdo JSON, revise, clique **Next**
   → **Create**
4. Pule para **Passo 6: Instale o app no workspace**. O manifest
   cuidou dos escopos, eventos e comandos slash para você.

### Opção B: Do zero (manual) {#option-b-from-scratch-manual}

1. Acesse [https://api.slack.com/apps](https://api.slack.com/apps)
2. Clique **Create New App**
3. Escolha **From scratch**
4. Digite um nome para o app (ex.: "Hermes Agent") e selecione seu workspace
5. Clique **Create App**

Você chegará na página **Basic Information** do app. Continue com
os Passos 2–6 abaixo.

---

## Passo 2: Configure os escopos do Bot Token {#step-2-configure-bot-token-scopes}

Navegue até **Features → OAuth & Permissions** na barra lateral. Role até **Scopes → Bot Token Scopes** e adicione o seguinte:

| Scope | Purpose |
|-------|---------|
| `chat:write` | Enviar mensagens como o bot |
| `app_mentions:read` | Detectar quando @mencionado em canais |
| `channels:history` | Ler mensagens em canais públicos em que o bot está |
| `channels:read` | Listar e obter informações sobre canais públicos |
| `groups:history` | Ler mensagens em canais privados aos quais o bot foi convidado |
| `im:history` | Ler histórico de mensagens diretas |
| `im:read` | Ver informações básicas de DM |
| `im:write` | Abrir e gerenciar DMs |
| `mpim:history` | Ler histórico de mensagens diretas em grupo (DM multi-pessoa) |
| `mpim:read` | Ver informações básicas de DM em grupo |
| `users:read` | Consultar informações de usuários |
| `files:read` | Ler e baixar arquivos anexados, incluindo notas de voz/áudio |
| `files:write` | Enviar arquivos (imagens, áudio, documentos) |

:::caution Escopos ausentes = recursos ausentes
Sem `channels:history` e `groups:history`, o bot **não receberá mensagens em canais** —
funcionará apenas em DMs. Sem `files:read`, o Hermes pode conversar, mas **não consegue ler de forma confiável anexos enviados pelo usuário**.
Esses são os escopos mais comumente esquecidos.
:::

**Escopos opcionais:**

| Scope | Purpose |
|-------|---------|
| `groups:read` | Listar e obter informações sobre canais privados |
| `assistant:write` | Renderizar a linha de status de estado de trabalho ("is thinking…") ao lado do nome do bot enquanto processa uma mensagem. Sem este escopo, a chamada `assistant.threads.setStatus` falha silenciosamente e o Slack exibe seus próprios placeholders genéricos rotativos ("Finding answers…", "Reviewing findings…", …) — o Hermes nunca controla o texto. Necessário para `typing_status_text` ter qualquer efeito visível. |

---

## Passo 3: Habilite o Socket Mode {#step-3-enable-socket-mode}

O Socket Mode permite que o bot se conecte via WebSocket em vez de exigir uma URL pública.

1. Na barra lateral, acesse **Settings → Socket Mode**
2. Ative **Enable Socket Mode** para ON
3. Você será solicitado a criar um **App-Level Token**:
   - Dê um nome como `hermes-socket` (o nome não importa)
   - Adicione o escopo **`connections:write`**
   - Clique **Generate**
4. **Copie o token** — ele começa com `xapp-`. Este é o seu `SLACK_APP_TOKEN`

:::tip
Você sempre pode encontrar ou regenerar tokens de nível de app em **Settings → Basic Information → App-Level Tokens**.
:::

---

## Passo 4: Assine eventos {#step-4-subscribe-to-events}

Esta etapa é crítica — ela controla quais mensagens o bot pode ver.


1. Na barra lateral, acesse **Features → Event Subscriptions**
2. Ative **Enable Events** para ON
3. Expanda **Subscribe to bot events** e adicione:

| Event | Required? | Purpose |
|-------|-----------|---------|
| `message.im` | **Yes** | O bot recebe mensagens diretas |
| `message.mpim` | **Yes** | O bot recebe mensagens em **DMs em grupo** (DMs multi-pessoa) das quais foi adicionado |
| `message.channels` | **Yes** | O bot recebe mensagens em canais **públicos** aos quais foi adicionado |
| `message.groups` | **Recommended** | O bot recebe mensagens em canais **privados** aos quais foi convidado |
| `app_mention` | **Yes** | Evita erros do Bolt SDK quando o bot é @mencionado |

4. Clique **Save Changes** na parte inferior da página

:::danger Assinaturas de evento ausentes são o problema nº 1 de configuração
Se o bot funciona em DMs, mas **não em canais**, você quase certamente esqueceu de adicionar
`message.channels` (para canais públicos) e/ou `message.groups` (para canais privados).
Sem esses eventos, o Slack simplesmente nunca entrega mensagens de canal ao bot.
:::


---

## Passo 5: Habilite a aba Messages {#step-5-enable-the-messages-tab}

Esta etapa habilita mensagens diretas para o bot. Sem ela, os usuários veem **"Sending messages to this app has been turned off"** ao tentar enviar DM ao bot.

1. Na barra lateral, acesse **Features → App Home**
2. Role até **Show Tabs**
3. Ative **Messages Tab** para ON
4. Marque **"Allow users to send Slash commands and messages from the messages tab"**

:::danger Sem esta etapa, DMs ficam completamente bloqueadas
Mesmo com todos os escopos e assinaturas de evento corretos, o Slack não permitirá que usuários enviem mensagens diretas ao bot a menos que a aba Messages esteja habilitada. Isso é um requisito da plataforma Slack, não um problema de configuração do Hermes.
:::

---

## Passo 6: Instale o app no workspace {#step-6-install-app-to-workspace}

1. Na barra lateral, acesse **Settings → Install App**
2. Clique **Install to Workspace**
3. Revise as permissões e clique **Allow**
4. Após a autorização, você verá um **Bot User OAuth Token** começando com `xoxb-`
5. **Copie este token** — este é o seu `SLACK_BOT_TOKEN`

:::tip
Se você alterar escopos ou assinaturas de evento depois, **deve reinstalar o app** para que as alterações
entrem em vigor. A página Install App exibirá um banner solicitando que você faça isso.
:::

---

## Passo 7: Encontre IDs de usuário para a allowlist {#step-7-find-user-ids-for-the-allowlist}

O Hermes usa **Member IDs** do Slack (não nomes de usuário ou nomes de exibição) para a allowlist.

Para encontrar um Member ID:

1. No Slack, clique no nome ou avatar do usuário
2. Clique **View full profile**
3. Clique no botão **⋮** (more)
4. Selecione **Copy member ID**

Member IDs têm o formato `U01ABC2DEF3`. Você precisa, no mínimo, do seu próprio Member ID.

---

## Passo 8: Configure o Hermes {#step-8-configure-hermes}

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
Para canais Slack peer-agent com backend Codex, prefira `agent.reasoning_effort: high` ou inferior. `xhigh`
pode gastar todo o turno em raciocínio oculto e nunca produzir texto visível do assistente; o Hermes agora
suprime esses avisos de turno incompleto da thread e mantém os diagnósticos nos logs do gateway.
:::

---

## Passo 9: Convide o bot para os canais {#step-9-invite-the-bot-to-channels}

Após iniciar o gateway, você precisa **convidar o bot** para qualquer canal em que queira que ele responda:

```
/invite @Hermes Agent
```

O bot **não** entra automaticamente em canais. Você deve convidá-lo para cada canal individualmente.

---

## Comandos slash {#slash-commands}

Todo comando Hermes (`/btw`, `/stop`, `/new`, `/model`, `/help`, ...)
é um comando slash nativo do Slack — exatamente como funcionam no Telegram
e no Discord. Digite `/` no Slack e o seletor de autocompletar lista todo
comando Hermes com sua descrição.

Por baixo dos panos: o Hermes inclui um manifest de app Slack gerado (veja
Passo 1, Opção A) que declara todo comando em
[`COMMAND_REGISTRY`](https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/commands.py)
como comando slash. No Socket Mode, o Slack roteia o evento de comando
pelo WebSocket independentemente do campo `url` do manifest.

### Experiência de mensagens Agent {#agent-messaging-experience}

Novos apps Slack usam a experiência de mensagens **Agent** do Slack. Apps Hermes
Assistant existentes podem migrar regenerando o manifest com `--agent-view`:

```bash
hermes slack manifest --agent-view --write
```

Atualize o manifest em **Features → App Manifest** e reinstale o app se
o Slack solicitar. A visualização Agent não pode ser revertida para Assistant, e os usuários podem precisar
atualizar o Slack com hard-refresh após a mudança. O manifest Agent gerado assina
`message.im`, `app_home_opened` e `app_context_changed`, para que o Hermes possa
identificar um DM da aba Messages e receber o contexto Slack ativo do usuário a cada
turno. O Hermes apenas fornece esse contexto como rótulo; não lê o histórico do canal
visualizado.

### Atualizar comandos slash após updates {#refreshing-slash-commands-after-updates}

Quando o Hermes adiciona novos comandos (ex.: após `hermes update`), regenere
o manifest e atualize seu app Slack:

```bash
hermes slack manifest --write
```

Depois no Slack:
1. Abra [https://api.slack.com/apps](https://api.slack.com/apps) →
   seu app Hermes
2. **Features → App Manifest → Edit**
3. Cole o novo conteúdo de `~/.hermes/slack-manifest.json`
4. **Save**. O Slack solicitará reinstalar o app se escopos ou comandos slash
   tiverem mudado.

### O legado `/hermes <subcommand>` ainda funciona {#legacy-hermes-subcommand-still-works}

Por compatibilidade com manifests antigos, você ainda pode digitar
`/hermes btw run the tests` — o Hermes roteia da mesma forma que `/btw
run the tests`. Perguntas em texto livre também funcionam: `/hermes what's the
weather?` é tratado como mensagem normal.

### Usar comandos dentro de threads (o prefixo `!cmd`) {#using-commands-inside-threads-the-cmd-prefix}

O próprio Slack bloqueia comandos slash nativos dentro de respostas em thread — tente
`/queue` em uma thread e o Slack responde com *"/queue is not supported
in threads. Sorry!"* Não há configuração no app que os reabilite;
o Slack nunca os entrega ao Hermes.

Como alternativa, o Hermes reconhece um `!` inicial como prefixo alternativo
de comando que funciona em threads (e em qualquer outro lugar). Digite
`!queue`, `!stop`, `!model gpt-5.4`, etc. como resposta normal em thread —
o Hermes trata de forma idêntica ao formato slash e responde na mesma
thread.

Apenas o primeiro token é verificado contra a lista de comandos conhecidos, então
mensagens casuais como `!nice work` passam para o agente inalteradas.

Prompts de aprovação (comando perigoso / aprovação de `execute_code`) normalmente
são renderizados como botões interativos. Quando os botões não podem ser entregues e
o Hermes recorre a um prompt de texto, o prompt instrui você a responder
com `!approve` / `!deny` — a forma que funciona dentro de threads.

### Avançado: emitir apenas o array slash-commands {#advanced-emit-only-the-slash-commands-array}

Se você mantém seu manifest Slack manualmente e só quer a lista de
comandos slash:

```bash
hermes slack manifest --slashes-only > /tmp/slashes.json
```

Cole esse array na chave `features.slash_commands` do seu
manifest existente.

---

## Como o bot responde {#how-the-bot-responds}

Entenda como o Hermes se comporta em diferentes contextos:

| Context | Behavior |
|---------|----------|
| **DMs** | O bot responde a toda mensagem — sem @mention necessária |
| **Channels** | O bot **só responde quando @mencionado** (ex.: `@Hermes Agent what time is it?`). Em canais, o Hermes responde em uma thread anexada a essa mensagem. |
| **Threads** | Se você @mencionar o Hermes dentro de uma thread existente, ele responde na mesma thread. Quando o bot tem uma sessão ativa em uma thread, **respostas subsequentes nessa thread não exigem @mention** — o bot acompanha a conversa naturalmente. |

:::tip
Em canais, sempre @mencione o bot para iniciar uma conversa. Quando o bot estiver ativo em uma thread, você pode responder nessa thread sem mencioná-lo. Fora de threads, mensagens sem @mention são ignoradas para evitar ruído em canais movimentados.
:::

---

## Opções de configuração {#configuration-options}

Além das variáveis de ambiente obrigatórias do Passo 8, você pode personalizar o comportamento do bot Slack via `~/.hermes/config.yaml`.

### Comportamento de thread e resposta {#thread--reply-behavior}

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

      # Continuable-cron delivery surface (default: "thread").
      # "in_channel" delivers a continuable cron job FLAT into the channel
      # (no dedicated thread); pair with reply_in_thread: false (and
      # require_mention: false) so a plain reply continues the job.
      # See the cron guide → "Flat, in-channel continuation".
      cron_continuable_surface: thread
```

| Key | Default | Description |
|-----|---------|-------------|
| `platforms.slack.reply_to_mode` | `"first"` | Modo de threading para mensagens multipartes: `"off"`, `"first"` ou `"all"` |
| `platforms.slack.extra.reply_in_thread` | `true` | Quando `false`, mensagens de canal recebem respostas diretas em vez de threads. Mensagens dentro de threads existentes ainda respondem na thread. |
| `platforms.slack.extra.reply_broadcast` | `false` | Quando `true`, respostas em thread também são postadas no canal principal. Apenas o primeiro chunk é transmitido. |
| `platforms.slack.extra.rich_blocks` | `false` | Quando `true`, mensagens do agente são renderizadas como blocos [Block Kit](https://docs.slack.dev/block-kit/) (cabeçalhos, divisores, listas aninhadas reais e tabelas nativas). Um fallback em texto simples é sempre enviado. Tabelas acima dos limites do Slack recorrem a monospace alinhado. Não exige reinstalar o app — é apenas uma mudança no envio. |
| `platforms.slack.extra.feedback_buttons` | `false` | Quando `true` com `rich_blocks`, adiciona controles de feedback nativos do Slack às respostas finais. |
| `platforms.slack.extra.suggested_prompts` | `[]` | Até quatro prompts `{title, message}` para pontos de entrada de DM Agent/Assistant; aceita lista ou `{title, prompts}`. |
| `platforms.slack.extra.assistant_thread_titles` | `true` | Quando `true`, nomeia threads de DM Agent/Assistant a partir da primeira mensagem do usuário. |
| `platforms.slack.extra.cron_continuable_surface` | `"thread"` | Superfície de entrega para [continuable cron jobs](../features/cron.md#flat-in-channel-continuation-slack). `"thread"` abre uma thread dedicada por entrega (padrão); `"in_channel"` entrega de forma plana na timeline do canal. Combine `in_channel` com `reply_in_thread: false` (e `require_mention: false`) para que uma resposta simples no canal continue o job. |

### Linha de status de estado de trabalho {#working-state-status-line}

Enquanto o agente processa uma mensagem, o Slack exibe uma linha de status ao lado do nome do bot
na thread. Por padrão, o Hermes define como `is thinking...`; personalize com
`typing_status_text` — ex.: um assistente gatinho chamado Ada:

```yaml
platforms:
  slack:
    # Custom working-state status line (default: "is thinking...").
    typing_status_text: "is pouncing… 🐾"
```

| Key | Default | Description |
|-----|---------|-------------|
| `platforms.slack.typing_status_text` | `"is thinking..."` | Texto da linha de status de estado de trabalho exibida enquanto o agente processa uma mensagem. Requer o escopo `assistant:write` — sem ele, a chamada de status falha silenciosamente e o Slack renderiza seu próprio placeholder genérico, independentemente desta configuração. Defina `typing_indicator: false` para desabilitar a linha de status por completo. |

:::note Onde o status é renderizado
O status personalizado aparece no **rodapé abaixo do compositor de resposta** ("*BotName* is thinking…"), não inline na lista de mensagens. As linhas inline "Generating response…" / "Finding answers…" que o Slack exibe na área de mensagens enquanto um app de IA trabalha são **indicadores rotativos próprios do Slack** — `assistant.threads.setStatus` não controla esses, e ambos podem aparecer ao mesmo tempo.
:::

A mesma chave personaliza a mensagem de marcador de estado de trabalho visível do Google Chat
(`platforms.google_chat.typing_status_text`, padrão `"Hermes is thinking…"`) —
observe que no Google Chat é uma mensagem real postada que é atualizada na
resposta, não um status efêmero.

### Status ao vivo (por ferramenta) {#live-status-per-tool}

Por padrão, a linha de status atualiza **ao vivo enquanto o agente trabalha**: em vez de um
`is thinking...` estático, mostra o que o agente está fazendo agora — `is
running pytest tests/…`, `is reading docs/api.md…`, `is searching the web for
slack api limits…`. Entre chamadas de ferramenta, volta ao texto estático. Isso
usa a cadência existente de atualização de status, então não faz chamadas adicionais à API do Slack,
e funciona mesmo com `tool_progress: off` (padrão do Slack) — diferente de
bolhas de progresso, a linha de status é efêmera e não deixa nada no
canal.

Controle com `display.live_status` (global ou por plataforma):

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

| Key | Default | Description |
|-----|---------|-------------|
| `display.live_status` | `"full"` | Linha de status ao vivo por ferramenta. `full` mostra verbo + preview do argumento; `verb` mostra apenas o verbo (mantém caminhos de arquivo e comandos fora de canais compartilhados); `off` restaura o texto estático. Requer o escopo `assistant:write`, igual à linha de status estática. |

### Isolamento de sessão {#session-isolation}

```yaml
# Global setting — applies to Slack and all other platforms
group_sessions_per_user: true
```

Quando `true` (o padrão), cada usuário em um canal compartilhado tem sua própria sessão de conversa isolada. Duas pessoas conversando com o Hermes em `#general` terão históricos e contextos separados.

Defina como `false` se quiser um modo colaborativo em que o canal inteiro compartilha uma sessão de conversa. Esteja ciente de que os usuários compartilham crescimento de contexto e custos de tokens, e um `/reset` de um usuário limpa a sessão para todos.

### Comportamento de menção e gatilho {#mention--trigger-behavior}

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

  # Custom mention patterns that trigger the bot
  # (in addition to the default @mention detection)
  mention_patterns:
    - "hey hermes"
    - "hermes,"

  # Text prepended to every outgoing message
  reply_prefix: ""
```

:::tip Quando usar `strict_mention`
Defina como `true` em workspaces movimentados onde o comportamento padrão do Slack de "o bot lembra desta thread" surpreende usuários — por exemplo, uma thread longa de suporte técnico em que o bot ajudou no início e você prefere que fique em silêncio a menos que seja explicitamente mencionado de novo. DMs e sessões interativas ativas não são afetadas.
:::

:::info
O Slack suporta ambos os padrões: `@mention` necessária para iniciar uma conversa por padrão, mas você pode isentar canais específicos via `SLACK_FREE_RESPONSE_CHANNELS` (IDs de canal separados por vírgula) ou `slack.free_response_channels` em `config.yaml`. Quando o bot tem uma sessão ativa em uma thread, respostas subsequentes na thread não exigem menção. Em **DMs 1:1**, o bot sempre responde sem precisar de menção.
:::

:::caution DMs em grupo (MPIMs) são superfícies compartilhadas, não DMs 1:1
Uma **mensagem direta 1:1** é uma conversa privada com uma pessoa, então está isenta de menção. Um **DM em grupo (MPIM / multi-person DM)** é uma *superfície compartilhada* — várias pessoas podem ver e acionar o bot — então obedece os mesmos controles de operador que um canal: `require_mention`, `strict_mention`, `free_response_channels` e `allowed_channels` se aplicam, e o bot só adiciona reações `:eyes:`/`:white_check_mark:` quando é de fato `@mentioned`. Para permitir que o bot responda livremente em um DM em grupo específico, adicione o ID do canal (começa com `G`) a `free_response_channels`.
:::

### Allowlist de canais (`allowed_channels`) {#channel-allowlist-allowed_channels}

Restringe o bot a um conjunto fixo de canais Slack — útil quando o bot é convidado para muitos canais, mas deve responder apenas em alguns. Quando definido, mensagens de canais que NÃO estão nesta lista são **silenciosamente ignoradas**, mesmo se o bot for `@mentioned`.

**DMs 1:1 estão isentas** deste filtro, então usuários autorizados sempre podem alcançar o bot em mensagem direta. **DMs em grupo (MPIMs) não estão isentos** — como canais, um MPIM deve estar na allowlist (seu ID começa com `G`) ou suas mensagens são descartadas.

```yaml
slack:
  allowed_channels:
    - "C0123456789"   # #ops
    - "C0987654321"   # #incident-response
```

Ou via variável de ambiente (separados por vírgula):

```bash
SLACK_ALLOWED_CHANNELS="C0123456789,C0987654321"
```

Comportamento:

- Vazio / não definido → sem restrição (totalmente retrocompatível).
- Não vazio → o ID do canal deve estar na lista, ou a mensagem é descartada antes de qualquer outra regra (exigência de menção, `free_response_channels`, etc.) ser aplicada.
- IDs de canal Slack começam com `C` (público), `G` (privado) ou `D` (DM). Consulte via "Open channel details" → painel "About" na UI do Slack, ou via API.

Veja também: [admin/user slash command split](../../reference/slash-commands.md#permissions-and-adminuser-split).

### Tratamento de usuários não autorizados {#unauthorized-user-handling}

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

A configuração específica da plataforma em `slack:` tem precedência sobre a configuração global.

### Transcrição de voz {#voice-transcription}

```yaml
# Global setting — enable/disable automatic transcription of incoming voice messages
stt_enabled: true
```

Quando `true` (o padrão), mensagens de áudio recebidas são automaticamente transcritas usando o provedor STT configurado antes de serem processadas pelo agente.

### Exemplo completo {#full-example}

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


## Canal home {#home-channel}

Defina `SLACK_HOME_CHANNEL` como um ID de canal onde o Hermes entregará mensagens agendadas,
resultados de jobs cron e outras notificações proativas. Para encontrar um ID de canal:

1. Clique com o botão direito no nome do canal no Slack
2. Clique **View channel details**
3. Role até o final — o Channel ID é exibido lá

```bash
SLACK_HOME_CHANNEL=C01234567890
```

Certifique-se de que o bot foi **convidado para o canal** (`/invite @Hermes Agent`).

---

## Suporte a múltiplos workspaces {#multi-workspace-support}

O Hermes pode se conectar a **múltiplos workspaces Slack** simultaneamente usando uma única instância de gateway. Cada workspace é autenticado independentemente com seu próprio bot user ID.

### Configuração {#configuration}

Forneça múltiplos bot tokens como **lista separada por vírgulas** em `SLACK_BOT_TOKEN`:

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

### Arquivo de token OAuth {#oauth-token-file}

Além de tokens no ambiente ou config, o Hermes também carrega tokens de um **arquivo de token OAuth** em:

```
~/.hermes/slack_tokens.json
```

Este arquivo é um objeto JSON mapeando team IDs para entradas de token:

```json
{
  "T01ABC2DEF3": {
    "token": "xoxb-workspace-token-here",
    "team_name": "My Workspace"
  }
}
```

Tokens deste arquivo são mesclados com quaisquer tokens especificados via `SLACK_BOT_TOKEN`. Tokens duplicados são automaticamente deduplicados.

### Como funciona {#how-it-works}

- O **primeiro token** da lista é o token primário, usado para a conexão Socket Mode (AsyncApp).
- Cada token é autenticado via `auth.test` na inicialização. O gateway mapeia cada `team_id` para seu próprio `WebClient` e `bot_user_id`.
- Quando uma mensagem chega, o Hermes usa o client específico do workspace correto para responder.
- O `bot_user_id` primário (do primeiro token) é usado para compatibilidade com recursos que esperam uma única identidade de bot.

---

## Mensagens de voz {#voice-messages}

O Hermes suporta voz no Slack:

- **Incoming:** Mensagens de voz/áudio são automaticamente transcritas usando o provedor STT configurado: `faster-whisper` local, Groq Whisper (`GROQ_API_KEY`) ou OpenAI Whisper (`VOICE_TOOLS_OPENAI_KEY`)
- **Outgoing:** Respostas TTS são enviadas como anexos de arquivo de áudio

---

## Prompts por canal {#per-channel-prompts}

Atribua prompts de sistema efêmeros a canais Slack específicos. O prompt é injetado em tempo de execução a cada turno — nunca persistido no histórico da transcrição — então alterações entram em vigor imediatamente.

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

As chaves são IDs de canal Slack (encontre-os via detalhes do canal → "About" → role até o final). Todas as mensagens no canal correspondente recebem o prompt injetado como instrução de sistema efêmera.

## Vínculos de skill por canal {#per-channel-skill-bindings}

Carregue automaticamente uma skill sempre que uma nova sessão iniciar em um canal ou DM específico. Diferente de prompts por canal (injetados a cada turno), vínculos de skill injetam o conteúdo da skill como mensagem de usuário no **início da sessão** — passa a fazer parte do histórico da conversa e não precisa ser recarregada nos turnos subsequentes.

Isso é ideal para DMs ou canais com propósito dedicado (flashcards, bot de Q&A específico de domínio, canal de triagem de suporte, etc.) em que você não quer que o seletor de skill do próprio modelo decida se carrega a cada resposta curta.

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
- O vínculo corresponde por ID de canal. Para mensagens em thread em um canal vinculado, a thread herda o vínculo do canal pai.
- A skill é carregada apenas no início da sessão (nova sessão ou após auto-reset). Se você alterar o vínculo, execute `/new` ou aguarde a sessão fazer auto-reset para entrar em vigor.
- Combine com `channel_prompts` para tom/restrições por canal além das instruções da skill.

## Solução de problemas {#troubleshooting}

| Problem | Solution |
|---------|----------|
| O bot não responde a DMs | Verifique se `message.im` está nas assinaturas de evento e se o app foi reinstalado |
| O bot funciona em DMs, mas não em canais | **Problema mais comum.** Adicione `message.channels` e `message.groups` às assinaturas de evento, reinstale o app e convide o bot para o canal com `/invite @Hermes Agent` |
| O bot não responde a @mentions em canais | 1) Verifique se o evento `message.channels` está assinado. 2) O bot deve ser convidado para o canal. 3) Certifique-se de que o escopo `channels:history` foi adicionado. 4) Reinstale o app após alterações de escopo/evento |
| O bot ignora mensagens em canais privados | Adicione a assinatura de evento `message.groups` e o escopo `groups:history`, depois reinstale o app e `/invite` o bot |
| O bot não responde em DMs em grupo (multi-person DMs) | Adicione a assinatura de evento `message.mpim` e o escopo `mpim:history` (mais `mpim:read`), depois **reinstale** o app. Sem `message.mpim`, o Slack nunca entrega mensagens de DM em grupo ao bot — mesmo que DMs 1:1 funcionem. |
| "Sending messages to this app has been turned off" em DMs | Habilite a **Messages Tab** nas configurações de App Home (veja Passo 5) |
| Erros "not_authed" ou "invalid_auth" | Regenere seu Bot Token e App Token, atualize `.env` |
| O bot responde, mas não consegue postar em um canal | Convide o bot para o canal com `/invite @Hermes Agent` |
| O bot conversa, mas não consegue ler imagens/arquivos enviados | Adicione `files:read`, depois **reinstale** o app. O Hermes agora exibe diagnósticos de acesso a anexos no chat quando o Slack retorna falhas de escopo/auth/permissão. |
| Erro `missing_scope` | Adicione o escopo necessário em OAuth & Permissions, depois **reinstale** o app |
| Socket desconecta com frequência | Verifique sua rede; o Bolt reconecta automaticamente, mas conexões instáveis causam lag |
| Alterou escopos/eventos, mas nada mudou | Você **deve reinstalar** o app no workspace após qualquer alteração de escopo ou assinatura de evento |

### Checklist rápido {#quick-checklist}

Se o bot não funciona em canais, verifique **todos** os itens a seguir:

1. ✅ O evento `message.channels` está assinado (para canais públicos)
2. ✅ O evento `message.groups` está assinado (para canais privados)
3. ✅ O evento `app_mention` está assinado
4. ✅ O escopo `channels:history` foi adicionado (para canais públicos)
5. ✅ O escopo `groups:history` foi adicionado (para canais privados)
6. ✅ O app foi **reinstalado** após adicionar escopos/eventos
7. ✅ O bot foi **convidado** para o canal (`/invite @Hermes Agent`)
8. ✅ Você está **@mencionando** o bot na sua mensagem

---

## Segurança {#security}

:::warning
**Sempre defina `SLACK_ALLOWED_USERS`** com os Member IDs de usuários autorizados. Sem esta configuração,
o gateway **negará todas as mensagens** por padrão como medida de segurança. Nunca compartilhe seus bot tokens —
trate-os como senhas.
:::

- Tokens devem ser armazenados em `~/.hermes/.env` (permissões de arquivo `600`)
- Rotacione tokens periodicamente via configurações do app Slack
- Audite quem tem acesso ao diretório de config do Hermes
- Socket Mode significa que nenhum endpoint público é exposto — uma superfície de ataque a menos
