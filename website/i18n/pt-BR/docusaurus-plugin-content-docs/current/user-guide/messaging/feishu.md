---
sidebar_position: 11
title: "Feishu / Lark"
description: "Configure o Hermes Agent como bot Feishu ou Lark"
---

# Feishu / Lark Setup

O Hermes Agent integra-se ao Feishu e Lark como bot completo. Depois de conectado, você pode conversar com o agente em mensagens diretas ou chats de grupo, receber resultados de cron jobs em um chat home e enviar texto, imagens, áudio e anexos de arquivo pelo fluxo normal do gateway.

A integração suporta ambos os modos de conexão:

- `websocket` — recomendado; o Hermes abre a conexão de saída e você não precisa de endpoint webhook público
- `webhook` — útil quando você quer que Feishu/Lark enviem eventos ao seu gateway via HTTP

## Como o Hermes se comporta {#how-hermes-behaves}

| Context | Behavior |
|---------|----------|
| Direct messages | O Hermes responde a toda mensagem. |
| Group chats | O Hermes responde apenas quando o bot é @mencionado no chat. |
| Shared group chats | Por padrão, o histórico de sessão é isolado por usuário dentro de um chat compartilhado. |

Esse comportamento de chat compartilhado é controlado por `config.yaml`:

```yaml
group_sessions_per_user: true
```

Defina como `false` somente se quiser explicitamente uma conversa compartilhada por chat.

## Passo 1: Crie um app Feishu / Lark {#step-1-create-a-feishu--lark-app}

### Recomendado: Scan-to-Create (um comando)

```bash
hermes gateway setup
```

Selecione **Feishu / Lark** e escaneie o QR code com o app móvel Feishu ou Lark. O Hermes criará automaticamente um aplicativo bot com as permissões corretas e salvará as credenciais.

### Alternativa: Configuração manual

Se scan-to-create não estiver disponível, o assistente recorre à entrada manual:

1. Abra o console de desenvolvedor Feishu ou Lark:
   - Feishu: [https://open.feishu.cn/](https://open.feishu.cn/)
   - Lark: [https://open.larksuite.com/](https://open.larksuite.com/)
2. Crie um novo app.
3. Em **Credentials & Basic Info**, copie o **App ID** e **App Secret**.
4. Habilite a capacidade **Bot** para o app.
5. Execute `hermes gateway setup`, selecione **Feishu / Lark** e insira as credenciais quando solicitado.

:::warning
Mantenha o App Secret privado. Qualquer pessoa com ele pode se passar pelo seu app.
:::

### Configure permissões

No console de desenvolvedor Feishu, vá em **Permission Management** e adicione os seguintes scopes. Você pode importá-los em massa na página de permissões.

**Permissões obrigatórias:**

| Scope | Purpose |
|-------|---------|
| `im:message` | Receber e ler mensagens |
| `im:message:send_as_bot` | Enviar mensagens como o bot |
| `im:resource` | Acessar imagens, arquivos e áudio enviados por usuários |
| `im:chat` | Acessar metadados de chat/grupo |
| `im:chat:readonly` | Ler lista de chats e membros |

**Permissões recomendadas (para funcionalidade completa):**

| Scope | Purpose |
|-------|---------|
| `im:message.reactions:readonly` | Receber eventos de reação emoji |
| `admin:app.info:readonly` | Auto-detectar identidade do bot para gating de @mention |
| `contact:user.id:readonly` | Resolver IDs de usuário para correspondência de allowlist |

### Configure eventos

Em **Events and Callbacks**:

1. Defina o modo de conexão como **Long Connection (WebSocket)** (recomendado) ou configure uma URL webhook
2. Na seção **Event Configuration**, inscreva-se em:
   - `im.message.receive_v1` — obrigatório para receber mensagens

### Publique o app

Depois de configurar permissões e eventos, vá em **Version Management** e publique uma nova versão do app. As permissões não terão efeito até que uma versão seja publicada e aprovada (para apps empresariais, isso pode exigir aprovação de admin).

## Passo 2: Escolha um modo de conexão {#step-2-choose-a-connection-mode}

### Recomendado: Modo WebSocket

Use o modo WebSocket quando o Hermes roda no seu laptop, workstation ou servidor privado. Nenhuma URL pública é necessária. O SDK oficial Lark abre e mantém uma conexão WebSocket de saída persistente com reconexão automática.

```bash
FEISHU_CONNECTION_MODE=websocket
```

**Requisitos:** O pacote Python `websockets` deve estar instalado. O SDK trata ciclo de vida da conexão, heartbeats e auto-reconexão internamente.

**Como funciona:** O adaptador executa o cliente WebSocket do SDK Lark em uma thread executor em background. Eventos de entrada (mensagens, reações, ações de card) são despachados para o loop asyncio principal. Na desconexão, o SDK tentará reconectar automaticamente.

### Opcional: Modo webhook

Use o modo webhook somente quando você já roda o Hermes atrás de um endpoint HTTP acessível.

```bash
FEISHU_CONNECTION_MODE=webhook
```

No modo webhook, o Hermes inicia um servidor HTTP (via `aiohttp`) e serve um endpoint Feishu em:

```text
/feishu/webhook
```

**Requisitos:** O pacote Python `aiohttp` deve estar instalado.

Você pode personalizar o endereço de bind e o caminho do servidor webhook:

```bash
FEISHU_WEBHOOK_HOST=127.0.0.1   # default: 127.0.0.1
FEISHU_WEBHOOK_PORT=8765         # default: 8765
FEISHU_WEBHOOK_PATH=/feishu/webhook  # default: /feishu/webhook
```

Quando o Feishu envia um desafio de verificação de URL (`type: url_verification`), o webhook responde automaticamente para que você possa completar a configuração de inscrição no console de desenvolvedor Feishu. A resposta ao desafio é gated em `FEISHU_VERIFICATION_TOKEN` quando definido — requisições de desafio com token ausente ou incompatível são rejeitadas para que um remoto não autenticado não possa provar controle de endpoint ecoando dados de desafio controlados por atacante.

## Passo 3: Configure o Hermes {#step-3-configure-hermes}

### Opção A: Configuração interativa

```bash
hermes gateway setup
```

Selecione **Feishu / Lark** e preencha as instruções.

### Opção B: Configuração manual

Adicione o seguinte a `~/.hermes/.env`:

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=secret_xxx
FEISHU_DOMAIN=feishu
FEISHU_CONNECTION_MODE=websocket

# Optional but strongly recommended
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
FEISHU_HOME_CHANNEL=oc_xxx
```

`FEISHU_DOMAIN` aceita:

- `feishu` para Feishu China
- `lark` para Lark internacional

## Passo 4: Inicie o gateway {#step-4-start-the-gateway}

```bash
hermes gateway
```

Depois envie mensagem ao bot pelo Feishu/Lark para confirmar que a conexão está ativa.

## Chat home {#home-chat}

Use `/set-home` em um chat Feishu/Lark para marcá-lo como canal home para resultados de cron jobs e notificações cross-platform.

Você também pode pré-configurá-lo:

```bash
FEISHU_HOME_CHANNEL=oc_xxx
```

## Segurança {#security}

### Allowlist de usuários

Para uso em produção, defina uma allowlist de Feishu Open IDs:

```bash
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
```

Se deixar a allowlist vazia, qualquer pessoa que consiga alcançar o bot pode usá-lo. Em chats de grupo, a allowlist é verificada contra o open_id do remetente antes da mensagem ser processada.

### Chave de criptografia webhook

Ao rodar em modo webhook, defina uma chave de criptografia para habilitar verificação de assinatura de payloads webhook de entrada:

```bash
FEISHU_ENCRYPT_KEY=your-encrypt-key
```

Essa chave está na seção **Event Subscriptions** da configuração do seu app Feishu. Quando definida, o adaptador verifica cada requisição webhook usando o algoritmo de assinatura:

```
SHA256(timestamp + nonce + encrypt_key + body)
```

O hash computado é comparado contra o header `x-lark-signature` usando comparação timing-safe. Requisições com assinaturas inválidas ou ausentes são rejeitadas com HTTP 401.

:::tip
No modo WebSocket, a verificação de assinatura é tratada pelo próprio SDK, então `FEISHU_ENCRYPT_KEY` é opcional. No modo webhook, é fortemente recomendado para produção.
:::

### Verification Token

Uma camada adicional de autenticação que verifica o campo `token` dentro de payloads webhook:

```bash
FEISHU_VERIFICATION_TOKEN=your-verification-token
```

Esse token também está na seção **Event Subscriptions** do seu app Feishu. Quando definido, todo payload webhook de entrada deve conter um `token` correspondente em seu objeto `header`. Tokens incompatíveis são rejeitados com HTTP 401.

Tanto `FEISHU_ENCRYPT_KEY` quanto `FEISHU_VERIFICATION_TOKEN` podem ser usados juntos para defesa em profundidade.

## Política de mensagens de grupo {#group-message-policy}

A variável de ambiente `FEISHU_GROUP_POLICY` controla se e como o Hermes responde em chats de grupo:

```bash
FEISHU_GROUP_POLICY=allowlist   # default
```

| Value | Behavior |
|-------|----------|
| `open` | O Hermes responde a @menções de qualquer usuário em qualquer grupo. |
| `allowlist` | O Hermes responde apenas a @menções de usuários listados em `FEISHU_ALLOWED_USERS`. |
| `disabled` | O Hermes ignora todas as mensagens de grupo inteiramente. |

Em todos os modos, o bot deve ser explicitamente @mencionado (ou @all) no grupo antes da mensagem ser processada. Mensagens diretas sempre contornam esse gate.

Defina `FEISHU_REQUIRE_MENTION=false` para deixar o Hermes ler todo tráfego de grupo sem exigir @mention:

```bash
FEISHU_REQUIRE_MENTION=false
```

Para controle por chat, defina `require_mention` em uma entrada `group_rules` — veja [Controle de acesso por grupo](#per-group-access-control) abaixo.

### Identidade do bot

O Hermes auto-detecta o `open_id` e nome de exibição do bot na inicialização. Você só precisa defini-los manualmente quando a auto-detecção não consegue alcançar a API Feishu, ou quando seu app usa user IDs com escopo de tenant:

```bash
FEISHU_BOT_OPEN_ID=ou_xxx     # only when auto-detection fails
FEISHU_BOT_USER_ID=xxx        # required if your app uses sender_id_type=user_id
FEISHU_BOT_NAME=MyBot         # only when auto-detection fails
```

## Mensagens bot-a-bot {#bot-to-bot-messaging}

Por padrão o Hermes ignora mensagens enviadas por outros bots. Habilite mensagens bot-a-bot quando quiser que o Hermes participe de orquestração A2A ou receba notificações de outros bots no mesmo grupo.

```bash
FEISHU_ALLOW_BOTS=mentions   # default: none
```

| Value | Behavior |
|-------|----------|
| `none` | Ignora todas as mensagens de outros bots (padrão). |
| `mentions` | Aceita apenas quando o bot parceiro @menciona o Hermes. |
| `all` | Aceita toda mensagem de bot parceiro. |

Também configurável como `feishu.allow_bots` em `config.yaml` (env prevalece quando ambos estão definidos).

Bots parceiros não precisam ser adicionados a `FEISHU_ALLOWED_USERS` — essa allowlist se aplica apenas a remetentes humanos.

Conceda o scope `application:bot.basic_info:read` para exibir nomes de bots parceiros; sem ele, bots parceiros ainda roteiam corretamente mas aparecem como seu `open_id`.

## Ações de card interativo {#interactive-card-actions}

Quando usuários clicam botões ou interagem com cards interativos enviados pelo bot, o adaptador roteia isso como eventos sintéticos de comando `/card`:

- Cliques de botão viram: `/card button {"key": "value", ...}`
- O payload `value` da ação da definição do card é incluído como JSON.
- Ações de card são deduplicadas com janela de 15 minutos para evitar processamento duplo.

Prompts de update conduzidos pelo gateway usam um card Feishu nativo `Yes` / `No` em vez de recorrer a respostas em texto simples. Quando `hermes update --gateway` precisa de confirmação, o adaptador registra a resposta selecionada no arquivo `.update_response` do Hermes e substitui o card inline com um estado resolvido.

Eventos de ação de card são despachados com `MessageType.COMMAND`, então fluem pelo pipeline normal de processamento de comandos.

É assim que **aprovação de comando** funciona — quando o agente precisa rodar um comando perigoso, envia um card interativo com botões Allow Once / Session / Always / Deny. O usuário clica um botão, e o callback de ação do card entrega a decisão de aprovação de volta ao agente.

### Configuração obrigatória do app Feishu {#required-feishu-app-configuration}

Cards interativos requerem **três** etapas de configuração no Feishu Developer Console. Faltar qualquer uma causa erro **200340** quando usuários clicam botões de card.

1. **Inscreva-se no evento de ação de card:**
   Em **Event Subscriptions**, adicione `card.action.trigger` aos seus eventos inscritos.

2. **Habilite a capacidade Interactive Card:**
   Em **App Features > Bot**, certifique-se de que o toggle **Interactive Card** está habilitado. Isso informa ao Feishu que seu app pode receber callbacks de ação de card.

3. **Configure a Card Request URL (somente modo webhook):**
   Em **App Features > Bot > Message Card Request URL**, defina a URL para o mesmo endpoint do seu event webhook (ex.: `https://your-server:8765/feishu/webhook`). No modo WebSocket isso é tratado automaticamente pelo SDK.

:::warning
Sem todas as três etapas, o Feishu *enviará* com sucesso cards interativos (enviar requer apenas permissão `im:message:send`), mas clicar qualquer botão retornará erro 200340. O card parece funcionar — o erro só aparece quando um usuário interage com ele.
:::

## Resposta inteligente a comentários de documento {#document-comment-intelligent-reply}

Além de chat, o adaptador também pode responder a `@`-menções deixadas em **documentos Feishu/Lark**. Quando um usuário comenta em um documento (seleção de texto local ou comentário de documento inteiro) e @-menciona o bot, o Hermes lê o documento mais o thread de comentários ao redor e publica uma resposta LLM inline no thread.

Alimentado pelo evento `drive.notice.comment_add_v1`, o handler:

- Busca conteúdo do documento e timeline de comentários em paralelo (20 mensagens para threads de documento inteiro, 12 para threads de seleção local).
- Executa o agente com os toolsets `feishu_doc` + `feishu_drive` escopados a essa única sessão de comentário.
- Fragmenta respostas em 4000 chars e as publica de volta como respostas encadeadas.
- Armazena em cache sessões por documento por 1 hora com cap de 50 mensagens para que comentários de follow-up no mesmo doc mantenham contexto.

### Controle de acesso em 3 níveis

Respostas a comentários de documento são **somente concessão explícita** — não há modo allow-all implícito. Permissões resolvem nesta ordem (primeira correspondência vence, por campo):

1. **Doc exato** — regra escopada a um token de documento específico.
2. **Wildcard** — regra que corresponde a um padrão de docs.
3. **Top-level** — regra padrão para o workspace.

Duas políticas estão disponíveis por regra:

- **`allowlist`** — lista estática de usuários / tenants.
- **`pairing`** — lista estática ∪ store aprovado em runtime. Útil para rollouts onde moderadores podem conceder acesso ao vivo.

Regras ficam em `~/.hermes/feishu_comment_rules.json` (concessões pairing em `~/.hermes/feishu_comment_pairing.json`) com hot-reload em cache mtime — edições têm efeito no próximo evento de comentário sem reiniciar o gateway.

CLI:

```bash
# Inspect current rules and pairing state
python -m gateway.platforms.feishu_comment_rules status

# Simulate an access check for a specific doc + user
python -m gateway.platforms.feishu_comment_rules check <fileType:fileToken> <user_open_id>

# Manage pairing grants at runtime
python -m gateway.platforms.feishu_comment_rules pairing list
python -m gateway.platforms.feishu_comment_rules pairing add <user_open_id>
python -m gateway.platforms.feishu_comment_rules pairing remove <user_open_id>
```

### Configuração obrigatória do app Feishu

Além das permissões de chat/card já concedidas, adicione o evento de comentário drive:

- Inscreva-se em `drive.notice.comment_add_v1` em **Event Subscriptions**.
- Conceda os scopes `docs:doc:readonly` e `drive:drive:readonly` para que o handler possa ler conteúdo do documento.

## Eventos de convite para reunião {#meeting-invitation-events}

Você pode convidar o bot Hermes Feishu/Lark para uma videoconferência da mesma forma que convida um participante humano. Quando o bot recebe o evento de convite para reunião, o Hermes pode automaticamente iniciar um turno de agente que tenta entrar na reunião.

Alimentado pelo evento `vc.bot.meeting_invited_v1`, o fluxo é:

- Um usuário convida o bot para uma videoconferência Feishu/Lark.
- Feishu/Lark envia ao Hermes o evento de convite para reunião.
- O Hermes extrai o convidante, tópico da reunião e número da reunião.
- Se o convidante for autorizado pela allowlist normal do gateway ou política de pairing, o agente recebe o número da reunião e tenta entrar automaticamente.
- Se o convite estiver malformado, ou o agente não conseguir entrar, o Hermes descarta o evento ou responde ao convidante com uma explicação concisa.

Convites malformados que não incluem tanto um convidante quanto um `meeting_no` são ignorados.

### Configuração obrigatória do app Feishu

Além das permissões de chat/card já concedidas, adicione o evento de convite para videoconferência:

- Inscreva-se em `vc.bot.meeting_invited_v1` em **Event Subscriptions**.
- Habilite o scope de permissão Video Conferencing solicitado pelo console de desenvolvedor Feishu/Lark para esse evento.
- Mantenha `im:message` e `im:message:send_as_bot` habilitados para que o Hermes possa responder ao convidante.
- Certifique-se de que a allowlist de usuários do gateway ou política de pairing autoriza o convidante. Convites para reunião não contornam verificações normais de acesso do gateway.

## Suporte a mídia {#media-support}

### Entrada (recebimento)

O adaptador recebe e armazena em cache os seguintes tipos de mídia de usuários:

| Type | Extensions | How it's processed |
|------|-----------|-------------------|
| **Images** | .jpg, .jpeg, .png, .gif, .webp, .bmp | Baixadas via API Feishu e armazenadas em cache localmente |
| **Audio** | .ogg, .mp3, .wav, .m4a, .aac, .flac, .opus, .webm | Baixado e armazenado em cache; arquivos de texto pequenos são auto-extraídos |
| **Video** | .mp4, .mov, .avi, .mkv, .webm, .m4v, .3gp | Baixado e armazenado em cache como documentos |
| **Files** | .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx, and more | Baixado e armazenado em cache como documentos |

Mídia de mensagens rich-text (post), incluindo imagens inline e anexos de arquivo, também é extraída e armazenada em cache.

Para documentos pequenos baseados em texto (.txt, .md), o conteúdo do arquivo é automaticamente injetado no texto da mensagem para que o agente possa lê-lo diretamente sem precisar de ferramentas.

### Saída (envio)

| Method | What it sends |
|--------|--------------|
| `send` | Mensagens de texto ou post rich (auto-detectadas com base em conteúdo markdown) |
| `send_image` / `send_image_file` | Faz upload de imagem ao Feishu, depois envia como bolha de imagem nativa (com legenda opcional) |
| `send_document` | Faz upload de arquivo à API Feishu, depois envia como anexo de arquivo |
| `send_voice` | Faz upload de arquivo de áudio como anexo de arquivo Feishu |
| `send_video` | Faz upload de vídeo e envia como mensagem de mídia nativa |
| `send_animation` | GIFs são rebaixados para anexos de arquivo (Feishu não tem bolha nativa de GIF) |

Roteamento de upload de arquivo é automático com base na extensão:

- `.ogg`, `.opus` → enviado como áudio `opus`
- `.mp4`, `.mov`, `.avi`, `.m4v` → enviado como mídia `mp4`
- `.pdf`, `.doc(x)`, `.xls(x)`, `.ppt(x)` → enviado com seu tipo de documento
- Todo o resto → enviado como arquivo stream genérico

## Renderização Markdown e fallback Post {#markdown-rendering-and-post-fallback}

Quando texto de saída contém formatação markdown (cabeçalhos, negrito, listas, blocos de código, links, etc.), o adaptador envia automaticamente como mensagem **post** Feishu com tag `md` embutida em vez de texto simples. Isso habilita renderização rica no cliente Feishu.

Se a API Feishu rejeitar o payload post (ex.: por construtos markdown não suportados), o adaptador recorre automaticamente a envio como texto simples com markdown removido. Esse fallback em duas etapas garante que mensagens sejam sempre entregues.

Mensagens de texto simples (nenhum markdown detectado) são enviadas como o tipo de mensagem simples `text`.

## Reações de status de processamento {#processing-status-reactions}

Enquanto o agente trabalha, o bot exibe uma reação `Typing` na sua mensagem. É removida quando a resposta chega, ou substituída por `CrossMark` se o processamento falhou.

Defina `FEISHU_REACTIONS=false` para desligar.

## Proteção contra burst e batching {#burst-protection-and-batching}

O adaptador inclui debouncing para bursts rápidos de mensagens para evitar sobrecarregar o agente:

### Batching de texto

Quando um usuário envia várias mensagens de texto em rápida sucessão, elas são mescladas em um único evento antes de serem despachadas:

| Setting | Env Var | Default |
|---------|---------|---------|
| Quiet period | `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | 0.6s |
| Max messages per batch | `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | 8 |
| Max characters per batch | `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | 4000 |

### Batching de mídia

Vários anexos de mídia enviados em rápida sucessão (ex.: arrastar várias imagens) são mesclados em um único evento:

| Setting | Env Var | Default |
|---------|---------|---------|
| Quiet period | `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | 0.8s |

### Serialização por chat

Mensagens dentro do mesmo chat são processadas serialmente (uma de cada vez) para manter coerência de conversa. Cada chat tem seu próprio lock, então mensagens em chats diferentes são processadas concorrentemente.

## Rate limiting (modo webhook) {#rate-limiting-webhook-mode}

No modo webhook, o adaptador impõe rate limiting por IP para proteger contra abuso:

- **Window:** janela deslizante de 60 segundos
- **Limit:** 120 requisições por janela por tripla (app_id, path, IP)
- **Tracking cap:** Até 4096 chaves únicas rastreadas (evita crescimento ilimitado de memória)

Requisições que excedem o limite recebem HTTP 429 (Too Many Requests).

### Rastreamento de anomalias webhook

O adaptador rastreia respostas de erro consecutivas por endereço IP. Após 25 erros consecutivos do mesmo IP dentro de uma janela de 6 horas, um aviso é registrado. Isso ajuda a detectar clientes mal configurados ou tentativas de probing.

Proteções adicionais de webhook:
- **Body size limit:** 1 MB máximo
- **Body read timeout:** 30 segundos
- **Content-Type enforcement:** Apenas `application/json` é aceito

## Ajuste WebSocket {#websocket-tuning}

Ao usar modo `websocket`, você pode personalizar comportamento de reconexão e ping:

```yaml
platforms:
  feishu:
    extra:
      ws_reconnect_interval: 120   # Seconds between reconnect attempts (default: 120)
      ws_ping_interval: 30         # Seconds between WebSocket pings (optional; SDK default if unset)
```

| Setting | Config key | Default | Description |
|---------|-----------|---------|-------------|
| Reconnect interval | `ws_reconnect_interval` | 120s | Quanto tempo esperar entre tentativas de reconexão |
| Ping interval | `ws_ping_interval` | _(SDK default)_ | Frequência de pings keepalive WebSocket |

## Controle de acesso por grupo {#per-group-access-control}

Além da `FEISHU_GROUP_POLICY` global, você pode definir regras detalhadas por chat de grupo usando `group_rules` em config.yaml:

```yaml
platforms:
  feishu:
    extra:
      default_group_policy: "open"     # Default for groups not in group_rules
      admins:                          # Users who can manage bot settings
        - "ou_admin_open_id"
      group_rules:
        "oc_group_chat_id_1":
          policy: "allowlist"          # open | allowlist | blacklist | admin_only | disabled
          allowlist:
            - "ou_user_open_id_1"
            - "ou_user_open_id_2"
        "oc_group_chat_id_2":
          policy: "admin_only"
        "oc_group_chat_id_3":
          policy: "blacklist"
          blacklist:
            - "ou_blocked_user"
        "oc_free_chat":
          policy: "open"
          require_mention: false       # overrides FEISHU_REQUIRE_MENTION for this chat
```

| Policy | Description |
|--------|-------------|
| `open` | Qualquer pessoa no grupo pode usar o bot |
| `allowlist` | Apenas usuários na `allowlist` do grupo podem usar o bot |
| `blacklist` | Todos exceto usuários na `blacklist` do grupo podem usar o bot |
| `admin_only` | Apenas usuários na lista global `admins` podem usar o bot neste grupo |
| `disabled` | Bot ignora todas as mensagens neste grupo |

Defina `require_mention: false` em uma entrada `group_rules` para pular o requisito de @-mention para aquele chat específico. Quando omitido, o chat herda o valor global `FEISHU_REQUIRE_MENTION`.

Grupos não listados em `group_rules` recorrem a `default_group_policy` (padrão para o valor de `FEISHU_GROUP_POLICY`).

## Deduplicação {#deduplication}

Mensagens de entrada são deduplicadas usando IDs de mensagem com TTL de 24 horas. O estado dedup persiste entre reinicializações em `~/.hermes/feishu_seen_message_ids.json`.

| Setting | Env Var | Default |
|---------|---------|---------|
| Cache size | `HERMES_FEISHU_DEDUP_CACHE_SIZE` | 2048 entries |

## Todas as variáveis de ambiente {#all-environment-variables}

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEISHU_APP_ID` | ✅ | — | Feishu/Lark App ID |
| `FEISHU_APP_SECRET` | ✅ | — | Feishu/Lark App Secret |
| `FEISHU_DOMAIN` | — | `feishu` | `feishu` (China) ou `lark` (internacional) |
| `FEISHU_CONNECTION_MODE` | — | `websocket` | `websocket` ou `webhook` |
| `FEISHU_ALLOWED_USERS` | — | _(empty)_ | Lista open_id separada por vírgula para allowlist de usuários |
| `FEISHU_ALLOW_BOTS` | — | `none` | Aceitar mensagens de outros bots: `none`, `mentions`, ou `all` |
| `FEISHU_REQUIRE_MENTION` | — | `true` | Se mensagens de grupo devem @mencionar o bot |
| `FEISHU_HOME_CHANNEL` | — | — | ID de chat para saída de cron/notificação |
| `FEISHU_ENCRYPT_KEY` | — | _(empty)_ | Chave de criptografia para verificação de assinatura webhook |
| `FEISHU_VERIFICATION_TOKEN` | — | _(empty)_ | Verification token para auth de payload webhook |
| `FEISHU_GROUP_POLICY` | — | `allowlist` | Política de mensagens de grupo: `open`, `allowlist`, `disabled` |
| `FEISHU_BOT_OPEN_ID` | — | _(empty)_ | open_id do bot (para detecção de @mention) |
| `FEISHU_BOT_USER_ID` | — | _(empty)_ | user_id do bot (para detecção de @mention) |
| `FEISHU_BOT_NAME` | — | _(empty)_ | Nome de exibição do bot (para detecção de @mention) |
| `FEISHU_WEBHOOK_HOST` | — | `127.0.0.1` | Endereço de bind do servidor webhook |
| `FEISHU_WEBHOOK_PORT` | — | `8765` | Porta do servidor webhook |
| `FEISHU_WEBHOOK_PATH` | — | `/feishu/webhook` | Caminho do endpoint webhook |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | — | `2048` | Máximo de IDs de mensagem deduplicados a rastrear |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | — | `0.6` | Período quiet de debounce de burst de texto |
| `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | — | `8` | Máximo de mensagens mescladas por batch de texto |
| `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | — | `4000` | Máximo de caracteres mesclados por batch de texto |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | — | `0.8` | Período quiet de debounce de burst de mídia |

Configurações WebSocket e ACL por grupo são configuradas via `config.yaml` em `platforms.feishu.extra` (veja [Ajuste WebSocket](#websocket-tuning) e [Controle de acesso por grupo](#per-group-access-control) acima).

## Solução de problemas {#troubleshooting}

| Problem | Fix |
|---------|-----|
| `lark-oapi not installed` | Instale o SDK: `pip install lark-oapi` |
| `websockets not installed; websocket mode unavailable` | Instale websockets: `pip install websockets` |
| `aiohttp not installed; webhook mode unavailable` | Instale aiohttp: `pip install aiohttp` |
| `FEISHU_APP_ID or FEISHU_APP_SECRET not set` | Defina ambas as env vars ou configure via `hermes gateway setup` |
| `Another local Hermes gateway is already using this Feishu app_id` | Apenas uma instância Hermes pode usar o mesmo app_id por vez. Pare o outro gateway primeiro. |
| Bot doesn't respond in groups | Certifique-se de que o bot é @mencionado, verifique `FEISHU_GROUP_POLICY` e confirme que o remetente está em `FEISHU_ALLOWED_USERS` se a política for `allowlist` |
| `Webhook rejected: invalid verification token` | Certifique-se de que `FEISHU_VERIFICATION_TOKEN` corresponde ao token na config Event Subscriptions do seu app Feishu |
| `Webhook rejected: invalid signature` | Certifique-se de que `FEISHU_ENCRYPT_KEY` corresponde à chave de criptografia na config do seu app Feishu |
| Post messages show as plain text | A API Feishu rejeitou o payload post; esse é comportamento normal de fallback. Verifique logs para detalhes. |
| Images/files not received by bot | Conceda scopes de permissão `im:message` e `im:resource` ao seu app Feishu |
| Bot identity not auto-detected | Geralmente um problema transitório de rede ao alcançar o endpoint bot info do Feishu. Defina `FEISHU_BOT_OPEN_ID` e `FEISHU_BOT_NAME` manualmente como workaround. |
| Peer bot messages still ignored after enabling `FEISHU_ALLOW_BOTS` | O Hermes ainda não consegue se identificar — defina `FEISHU_BOT_OPEN_ID` (e `FEISHU_BOT_USER_ID` se seu app usa `sender_id_type=user_id`). |
| Peer bots show as `ou_xxxxxx` instead of by name | Conceda o scope `application:bot.basic_info:read`. |
| Error 200340 when clicking approval buttons | Habilite a capacidade **Interactive Card** e configure **Card Request URL** no Feishu Developer Console. Veja [Configuração obrigatória do app Feishu](#required-feishu-app-configuration) acima. |
| `Webhook rate limit exceeded` | Mais de 120 requisições/minuto do mesmo IP. Isso geralmente é misconfiguração ou loop. |

## Toolset {#toolset}

Feishu / Lark usa o preset de plataforma `hermes-feishu`, que inclui as mesmas ferramentas core que Telegram e outras plataformas de mensagens baseadas em gateway.
