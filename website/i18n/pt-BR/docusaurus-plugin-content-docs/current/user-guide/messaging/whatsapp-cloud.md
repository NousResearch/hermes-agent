---
sidebar_position: 6
title: "WhatsApp Business (Cloud API)"
description: "Configure o Hermes Agent como bot do WhatsApp via a Cloud API oficial da Meta"
---

# Configuração WhatsApp Business Cloud API {#whatsapp-business-cloud-api-setup}

O Hermes pode se conectar ao WhatsApp pela **oficial** WhatsApp Business Cloud API da Meta. Este é o caminho de produção: sem subprocesso Node.js da ponte, sem QR codes, sem risco de banimento de conta.

Em troca:

- Você precisa de uma **conta Meta Business** (não WhatsApp pessoal).
- O bot opera em um número de telefone comercial dedicado, não no seu número pessoal.
- O gateway Hermes precisa de uma **URL HTTPS pública** para a Meta entregar mensagens recebidas via webhook.
- Respostas mais de 24 horas após a última mensagem do usuário exigem um **template** pré-aprovado (regra da "janela de atendimento ao cliente" da Meta, não limite do Hermes).

Se essas restrições não funcionam para seu caso, a [integração ponte Baileys](./whatsapp.md) é a alternativa — conta pessoal, sem URL pública, mas não oficial e sujeita a banimento.

:::tip Qual devo usar?
- **Cloud API (este guia)** — bot comercial real, quer estabilidade, aceita verificação Meta + burocracia de templates
- **[Ponte Baileys](./whatsapp.md)** — projetos pessoais, demos rápidas, setups de usuário único, disposto a arriscar a conta do telefone do bot
:::

---

## Início rápido {#quick-start}

```bash
hermes whatsapp-cloud
```

O assistente percorre cada credencial, valida cada uma ao colar (captura a armadilha #1 de configuração — colar um número de telefone no campo Phone Number ID), e imprime instruções exatas para as partes que precisam acontecer fora do assistente (iniciar cloudflared, configurar o painel de webhook da Meta).

O restante desta página é a referência manual.

---

## Pré-requisitos {#prerequisites}

1. **Uma conta Meta Business**. Crie uma em [business.facebook.com](https://business.facebook.com/).
2. **Um app Meta com WhatsApp habilitado**. Veja "Creating the Meta app" abaixo.
3. **Uma forma de expor uma porta local à internet pública** com HTTPS. Cloudflare Tunnel (`cloudflared`) é recomendado — grátis, sem port forwarding, sem domínio. ngrok, seu próprio domínio com reverse proxy + TLS, ou um VPS com o gateway ligado diretamente a um IP público também funcionam.
4. **Opcional mas recomendado**: ffmpeg no `PATH` para mensagens de voz de saída renderizarem como bolhas nativas de nota de voz WhatsApp (forma de onda verde) em vez de anexos MP3. O Hermes degrada graciosamente se ausente.

---

## Criando o app Meta {#creating-the-meta-app}

1. Acesse [developers.facebook.com/apps](https://developers.facebook.com/apps) → **Create App**.
2. Escolha o caso de uso: **"Connect with customers through WhatsApp"** → **Next**.
3. Escolha ou crie um business portfolio. Revise os requisitos de publicação. Confirme → **Create app**.
4. Após a criação você chega em **Customize use case → Connect on WhatsApp → Quickstart**. Clique **Start using the API** → agora está na página **API Setup**.
5. Garanta que uma WhatsApp Business Account (WABA) está vinculada. Se criou um portfolio novo no passo 3, uma foi auto-criada. Verifique na página API Setup.

Você precisará destes valores do painel — o assistente solicita nesta ordem:

| Value | Where in dashboard | Field shape | Notes |
|---|---|---|---|
| **Phone Number ID** | App Dashboard → WhatsApp → API Setup → below the "From" dropdown | Numeric, 15-17 digits | **NOT** the phone number itself. The #1 setup mistake is pasting the actual phone number here. |
| **Access Token** | App Dashboard → WhatsApp → API Setup → "Generate access token" | Starts with `EAA`, 100+ chars | Temp tokens last 24h — see "Permanent token" below for production. |
| **App Secret** | App Dashboard → Settings → Basic → click "Show" next to App secret | 32-character lowercase hex | Used to verify incoming webhook signatures.  Without it, inbound delivery is refused with 503. |
| **App ID** (optional) | App Dashboard → Settings → Basic | Numeric, 15-16 digits | Not required for messaging, useful for analytics. |
| **WABA ID** (optional) | App Dashboard → WhatsApp → API Setup → near the top | Numeric, 15+ digits | Not required for messaging, useful for analytics. |

---

## Token permanente (produção) {#permanent-token-production}

Tokens de acesso temporários expiram após **24 horas**, o que significa que um token gerado hoje para de funcionar amanhã. Para deployments de produção use um **token permanente de System User**:

1. Acesse [business.facebook.com/latest/settings](https://business.facebook.com/latest/settings) → **System users** (barra lateral esquerda).
2. **Add** → nome (ex.: `hermes-bot`) → role: **Admin**.
3. Selecione o novo usuário → **Assign Assets**:
   - Selecione seu app → toggle **Manage app** under Full control.
   - Selecione sua conta WhatsApp → toggle **Manage WhatsApp Business Accounts** under Full control.
   - Clique **Assign assets**.
4. **Generate token** com estas permissões:
   - `business_management`
   - `whatsapp_business_messaging`
   - `whatsapp_business_management`
5. Defina **token expiration: Never**.
6. Copie o token → atualize `WHATSAPP_CLOUD_ACCESS_TOKEN` em `~/.hermes/.env` → reinicie o gateway.

Tokens de System User não expiram a menos que você revogue explicitamente.

---

## Expondo o Hermes à internet {#exposing-hermes-to-the-internet}

A Cloud API entrega mensagens recebidas por HTTPS POST na URL do seu webhook — isso significa que o gateway Hermes precisa ser alcançável pelos servidores da Meta. Três formas comuns:

### Cloudflare Tunnel (recomendado) {#cloudflare-tunnel-recommended}

Grátis, sem port forwarding, funciona em Windows / macOS / Linux. Roda como processo separado junto ao gateway.

**Install:**

```bash
# Windows
winget install Cloudflare.cloudflared

# macOS
brew install cloudflared

# Linux
# Download the binary from https://github.com/cloudflare/cloudflared/releases
```

**Run a quick tunnel** (no Cloudflare account needed — gives you a `https://<random>.trycloudflare.com` URL):

```bash
cloudflared tunnel --url http://localhost:8090
```

Anote a URL impressa — é o que você dará à Meta.

:::warning Quick tunnels rotate
A URL do quick-tunnel grátis muda a cada reinício do `cloudflared`. Para URL estável, faça login com `cloudflared tunnel login` e crie um tunnel nomeado. Contas Cloudflare grátis têm tunnels nomeados ilimitados — veja a [documentação Cloudflare](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/) para o fluxo de tunnel nomeado.
:::

### ngrok {#ngrok}

```bash
ngrok http 8090
```

O tier grátis mostra URL diferente a cada reinício. Tier pago dá subdomínio estável.

### Seu próprio domínio + reverse proxy {#your-own-domain--reverse-proxy}

Se já tem servidor com cert TLS (Caddy, nginx, etc.), aponte uma rota para `localhost:8090`. Opção mais estável para produção, mas exige infraestrutura existente.

---

## Configurando o webhook no lado da Meta {#configuring-the-webhook-on-metas-side}

Com o tunnel rodando:

1. Anote a URL pública impressa pelo tunnel — digamos `https://abc123.trycloudflare.com`.
2. Gere um **Verify Token** — o assistente faz isso com `secrets.token_urlsafe(32)`; se configurar manualmente, execute:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   Salve como `WHATSAPP_CLOUD_VERIFY_TOKEN` em `~/.hermes/.env`.
3. Inicie o gateway Hermes: `hermes gateway`.
4. No Meta App Dashboard → **WhatsApp → Configuration** (ou **Use cases → Customize → Configuration** dependendo da versão da UI) → clique **Edit** na seção Webhook.
5. Preencha:
   - **Callback URL**: `https://abc123.trycloudflare.com/whatsapp/webhook`
   - **Verify Token**: a string do passo 2 (deve corresponder exatamente)
6. Clique **Verify and save**. A Meta acessa sua URL com GET request, o gateway ecoa o challenge de volta, e a Meta marca o webhook como verificado.
7. Em **Webhook fields**, clique **Manage** → inscreva-se no campo **messages**. Isso diz à Meta para de fato entregar mensagens recebidas ao seu webhook.

**Para verificar o loop manualmente** (de um terceiro terminal):

```bash
TUNNEL="https://abc123.trycloudflare.com"
VERIFY="<your verify token>"

# Should print HTTP 200 with body "hello"
curl -i "$TUNNEL/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=$VERIFY&hub.challenge=hello"

# Health endpoint — should show verify_token_configured: true and app_secret_configured: true
curl "$TUNNEL/health"
```

---

## Whitelist de destinatários (lado Meta) {#recipient-whitelist-meta-side}

Em modo de desenvolvimento (antes do app passar por App Review), a Meta restringe quais números seu bot pode contatar:

1. App Dashboard → WhatsApp → API Setup → dropdown **To**.
2. Clique **Manage phone number list**.
3. Adicione os números que quer contatar (seu, da equipe, testers amigáveis). A Meta envia a cada um um código de verificação de 6 dígitos via SMS ou WhatsApp.

Até 5 números em modo dev. App Review remove esse limite.

---

## Allowlist (lado Hermes) {#allowlist-hermes-side}

Além da whitelist de destinatários da Meta, o Hermes tem sua própria allowlist por plataforma que controla **quais mensagens recebidas o agente processa**. Adicione em `~/.hermes/.env`:

```bash
# Comma-separated phone numbers, country code, no '+' / spaces / dashes
WHATSAPP_CLOUD_ALLOWED_USERS=15551234567,15557654321

# Or allow everyone (only safe in combination with Meta's recipient whitelist)
# WHATSAPP_CLOUD_ALLOW_ALL_USERS=true
```

O assistente define isso no passo 6. Sem allowlist, **toda mensagem recebida é negada** — isso é intencional, para o bot não ser invocado por números aleatórios se a whitelist de destinatários for relaxada.

---

## Polindo o perfil WhatsApp do seu bot {#polishing-your-bots-whatsapp-profile}

O WhatsApp exibe **nome e foto de perfil** do seu bot no cabeçalho do chat e na lista de contatos. Isso não pode ser definido via Cloud API — fica no Business Manager da Meta.

Quando o bot estiver funcionando, acesse **[business.facebook.com/wa/manage/phone-numbers](https://business.facebook.com/wa/manage/phone-numbers/)**, clique no seu número de telefone, e você encontrará:

| What | Where | Notes |
|---|---|---|
| **Display name** | Top of the phone-number page | Changes go through Meta's name-review process (~24–48 hours). |
| **Profile picture** | Top of the phone-number page | Square image, ≥640×640px recommended. Updates immediately. |
| **About / description / website / email / hours / category** | "Edit profile" button | These appear in the info pane when a user taps the bot's name. Cosmetic. |
| **Verified badge** (green checkmark) | Business Manager → Security Center → Start Verification | Requires Meta's separate business verification process. |

O assistente `hermes whatsapp-cloud` imprime esses links no final da configuração. Nada disso é necessário para o bot funcionar — é polimento puro de como seu bot aparece aos usuários.

---

## Referência de configuração {#configuration-reference}

Todas as configurações ficam em `~/.hermes/.env`. Valores obrigatórios em **negrito**.

| Variable | Default | Description |
|---|---|---|
| **`WHATSAPP_CLOUD_PHONE_NUMBER_ID`** | — | The 15-17 digit ID from API Setup.  **Not** the phone number. |
| **`WHATSAPP_CLOUD_ACCESS_TOKEN`** | — | Meta access token (starts with `EAA`).  Temp 24h or System User permanent. |
| **`WHATSAPP_CLOUD_APP_SECRET`** | — | 32-char hex from Settings → Basic.  Without it, inbound is refused with 503. |
| **`WHATSAPP_CLOUD_VERIFY_TOKEN`** | — | Shared secret for the GET handshake.  Auto-generated by the wizard. |
| **`WHATSAPP_CLOUD_ALLOWED_USERS`** | — | Comma-separated wa_ids allowed to message the bot. |
| `WHATSAPP_CLOUD_ALLOW_ALL_USERS` | `false` | Set to `true` to bypass the allowlist. |
| `WHATSAPP_CLOUD_APP_ID` | — | Optional, for future analytics integration. |
| `WHATSAPP_CLOUD_WABA_ID` | — | Optional, for future analytics integration. |
| `WHATSAPP_CLOUD_WEBHOOK_HOST` | `0.0.0.0` | Interface the webhook server binds to. |
| `WHATSAPP_CLOUD_WEBHOOK_PORT` | `8090` | Port the webhook server binds to.  Must match the port your tunnel forwards. |
| `WHATSAPP_CLOUD_WEBHOOK_PATH` | `/whatsapp/webhook` | URL path Meta posts to. |
| `WHATSAPP_CLOUD_API_VERSION` | `v20.0` | Meta Graph API version. Only override if a newer version is recommended in Meta's docs. |
| `WHATSAPP_CLOUD_HOME_CHANNEL` | — | wa_id to use as the bot's home channel (for cron jobs etc). |

Você pode ter **ambos** os adaptadores Baileys (`whatsapp`) e Cloud (`whatsapp_cloud`) habilitados simultaneamente, com números de telefone diferentes.

---

## Recursos {#features}

### Entrada {#inbound}

- **Mensagens de texto** — passadas diretamente ao agente.
- **Imagens** — baixadas automaticamente e anexadas à entrada do agente. Modelos com visão nativa (Claude, GPT-4o, Gemini, etc.) leem a imagem diretamente; modelos sem visão recebem descrição de texto auto-gerada.
- **Notas de voz** — baixadas como `.ogg`, transcritas via seu provedor STT configurado (faster-whisper local, OpenAI/Nous, Groq, etc.), depois entregues ao agente como texto.
- **Documentos** — baixados automaticamente. Arquivos pequenos legíveis como texto (`.txt`, `.md`, `.json`, `.py`, `.csv`, etc.) até 100KB são inlined na entrada do agente para leitura sem chamada de ferramenta. Arquivos maiores ficam em cache local para outras ferramentas do agente.
- **Toques em botões** — quando o usuário toca um botão que o bot enviou antes (escolha clarify, aprovação de comando, confirmação de slash command), o toque é roteado diretamente ao handler certo. Toques obsoletos caem de volta a ser tratados como entrada de texto regular.
- **Contexto de resposta** — quando o usuário responde a uma mensagem anterior do bot, o agente vê a mensagem original como contexto.

### Saída {#outbound}

- **Texto** — markdown é convertido automaticamente para a sintaxe flavor do WhatsApp (`**bold**` → `*bold*`, `~~strike~~` → `~strike~`, headers → bold, `[link](url)` → `link (url)`). Mensagens longas divididas em 4096 chars por fragmento.
- **Imagens** — imagens geradas pelo agente e arquivos de imagem locais suportados, entregues como anexos nativos de foto.
- **Mensagens de voz** — saída text-to-speech é convertida via ffmpeg na bolha nativa de nota de voz WhatsApp (forma de onda verde). Sem ffmpeg instalado, cai para anexo MP3. Veja "Voice messages" abaixo.
- **Vídeo / documentos** — ambos suportados, enviados como anexos nativos.

### UX interativa {#interactive-ux}

Quando o agente invoca qualquer desses fluxos, o Hermes usa mensagens interativas nativas do WhatsApp — botões toque-para-responder em vez de prompts "responda com o número":

- **Ferramenta `clarify`** — perguntas multi-escolha renderizam como botões de resposta rápida (1–3 escolhas) ou folha de lista toque-para-abrir (4+ escolhas). Escolher "✏️ Other" deixa o usuário digitar resposta livre que o agente recebe como resolução.
- **Aprovações de comando perigoso** — quando terminal/execução de código do agente encontra comando gated, o usuário vê botões `✅ Approve` / `❌ Deny` em vez de precisar digitar `/approve` ou `/deny`.
- **Confirmações de slash command** — comandos privilegiados como `/reload-mcp` mostram botões `✅ Approve Once` / `🔒 Always` / `❌ Cancel`.

Todos os prompts interativos degradam graciosamente para texto simples se os botões falharem ao renderizar (ex.: em clientes WhatsApp legados).

### Confirmações de leitura e indicador de digitação {#read-receipts-and-typing-indicator}

O Hermes confirma mensagens recebidas imediatamente:

- Sua mensagem mostra **checkmarks duplos azuis** assim que o gateway a recebe.
- O nome do bot no seu chat WhatsApp mostra **"typing…"** enquanto o agente prepara a resposta.
- O indicador de digitação some automaticamente quando a primeira mensagem de resposta do bot chega.

Isso deixa claro quando o bot viu sua mensagem versus quando ainda está trabalhando na resposta.

### Mensagens de voz {#voice-messages}

O WhatsApp distingue entre "nota de voz" (bolha de forma de onda verde) e anexo de arquivo de áudio genérico. A diferença é puramente codec: notas de voz precisam ser `audio/ogg` com codificação `opus`.

O TTS do Hermes produz MP3. Dois caminhos:

- **Com ffmpeg no PATH** (recomendado) — TTS de saída é convertido e chega como nota de voz adequada. Instale:
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: gerenciador de pacotes
- **Sem ffmpeg** — TTS de saída chega como anexo MP3. Reproduz normalmente, só não parece nota de voz. Um aviso único dispara no log do gateway para você saber.

Você pode verificar se o gateway encontrou ffmpeg via o endpoint health:

```bash
curl http://localhost:8090/health
# look for "ffmpeg_present": true
```

---

## Limitações conhecidas {#known-limitations}

### Janela de conversa de 24 horas {#24-hour-conversation-window}

A Meta só permite **mensagens livres** dentro de uma janela de 24 horas após a última mensagem recebida do usuário. Fora dessa janela, a única coisa que a API da Meta aceita é um **message template** pré-aprovado.

**O que isso significa na prática:**

- Chat reativo (usuário envia DM → bot responde em 24h → usuário responde → ...) funciona para sempre. Cobre >95% do uso normal de bot.
- **Cron jobs que entregam no WhatsApp** após gap > 24h falham com Graph error code `131047` ("Re-engagement message").
- **Resultados async de `delegate_task` de longa duração** que levam mais de 24h falham da mesma forma.
- **Assinantes de webhook** que roteiam eventos externos para WhatsApp falham quando o usuário não enviou DM ao bot recentemente.

O Hermes avisa o agente sobre essa janela no system prompt, para o modelo saber mencioná-la ao agendar mensagens atrasadas.

Suporte a message templates (workaround para envios fora da janela) ainda não está implementado no Hermes. Se precisar, [abra uma issue](https://github.com/NousResearch/hermes-agent/issues) — está planejado mas aguardando sinal claro de demanda.

### Chats em grupo {#group-chats}

A Cloud API tem suporte limitado a grupos (capability-tier gated pela Meta). O adaptador `whatsapp_cloud` do Hermes atualmente trata **apenas mensagens diretas** na v1. Se precisar de grupos, use a ponte Baileys.

### Limite de taxa de saída {#outbound-rate-limit}

O throughput padrão da Meta é **80 mensagens/segundo por número de telefone comercial**, com upgrades disponíveis. O Hermes atualmente não impõe isso client-side — envios de volume extremamente alto podem atingir o limite da Meta.

---

## Solução de problemas {#troubleshooting}

### Verificação de setup falha ("URL couldn't be validated") no painel Meta {#setup-verification-fails-url-couldnt-be-validated-in-meta-dashboard}

Quase sempre um destes:

- **URL do tunnel errada ou obsoleta** — quick tunnels cloudflared rotacionam. Obtenha URL nova e atualize tanto `.env` quanto o painel da Meta.
- **Verify token não corresponde** — o token em `WHATSAPP_CLOUD_VERIFY_TOKEN` de `~/.hermes/.env` deve corresponder exatamente ao que você digitou no painel da Meta. Execute o probe curl acima para confirmar que o handshake de verify do gateway funciona localmente primeiro.
- **Gateway não está rodando** — verifique se `hermes gateway` está ativo.
- **App Secret não definido** — sem ele, o Hermes recusa POSTs recebidos com 503. A Meta interpreta isso como "can't validate."

### `graph error 100`: Object with ID '...' does not exist {#graph-error-100-object-with-id--does-not-exist}

Você colou seu número de telefone (10-11 dígitos) em `WHATSAPP_CLOUD_PHONE_NUMBER_ID` em vez do Phone Number ID (ID interno de 15-17 dígitos da Meta). Reconfira a página API Setup — o Phone Number ID aparece *abaixo* do dropdown "From".

O assistente captura isso com um validador agora, mas vale saber se configurar manualmente.

### `graph error 190`: Authentication Error {#graph-error-190-authentication-error}

Seu access token é inválido. Subcodes:

- `subcode 463` — token expirado. Tokens temp duram 24h. Regenere, ou mude para token permanente de System User (veja acima).
- `subcode 467` — token invalidado (revogado ou senha alterada).
- Outros 190 — token não tinha as permissões necessárias ao ser gerado. Garanta que as três (`business_management`, `whatsapp_business_messaging`, `whatsapp_business_management`) foram selecionadas.

### `graph error 131047`: Re-engagement message {#graph-error-131047-re-engagement-message}

A janela de conversa de 24 horas expirou (veja "Known limitations"). Ou:

- Peça ao usuário para enviar DM ao bot primeiro para reabrir a janela.
- Aguarde suporte a templates chegar no Hermes.

### Mensagem recebida: `media metadata fetch failed (status=401)` {#inbound-message-media-metadata-fetch-failed-status401}

Mesmas causas raiz 401 que outbound (`graph error 190`) — o access token é inválido ou expirado. Corrija o token.

### Respostas do bot aparecem como JSON bruto / vazamento de tool-call {#bot-replies-appear-as-raw-json--tool-call-leakage}

Causa comum: o toolset configurado para `whatsapp_cloud` está faltando as ferramentas que o agente quer chamar. Verifique `hermes tools list` e confirme que a plataforma usa `hermes-whatsapp` (toolset padrão do adaptador Cloud, mesmo da Baileys).

Se o modelo emite texto com forma de tool-call em vez de chamada estruturada, geralmente significa que o toolset estava efetivamente vazio. Veja `hermes_cli/platforms.py` para o mapeamento plataforma → toolset padrão.

### STT (transcrição de nota de voz) retorna vazio / "could not transcribe" {#stt-voice-note-transcription-returns-empty--could-not-transcribe}

O padrão `stt.provider: local` exige `pip install faster-whisper`. Se você é assinante Nous, pode rotear STT pelo gateway de áudio gerenciado da Meta:

```bash
hermes config set stt.provider openai
hermes config set stt.use_gateway true
hermes gateway restart
```

Isso usa seu access token Nous Portal em vez de precisar de chave OpenAI separada.

---

## Notas de segurança {#security-notes}

- **Trate o App Secret como senha** — quem tiver pode forjar payloads de webhook que o Hermes aceitará como autênticos.
- **O verify token é segredo compartilhado** — vazamentos são de menor impacto (pior caso alguém re-inscreve o webhook da Meta em outra URL deles), mas ainda evite commitá-lo.
- **O access token é a identidade do seu bot** — tokens de System User equivalem a API keys de longa duração. Rotacione imediatamente se um deployment for comprometido.
- **O endpoint webhook aceita apenas requisições assinadas quando `WHATSAPP_CLOUD_APP_SECRET` está definido** — mantenha definido mesmo em desenvolvimento. Sem ele, o gateway recusa entrega recebida com HTTP 503.
- **O endpoint `/health` não é autenticado** — é seguro expor porque só reporta booleanos de presença de config, não os valores. Mas se preferir não expor, restrinja acesso na camada reverse proxy / tunnel.

---

## Comparação com a ponte Baileys {#comparison-to-the-baileys-bridge}

| | Baileys (`hermes whatsapp`) | Cloud API (`hermes whatsapp-cloud`) |
|---|---|---|
| Account type | Personal | Business |
| Setup | QR code scan | Meta app + WABA + token |
| Dependencies | Node.js + npm | Pure Python (httpx + aiohttp) |
| Process | Managed Node subprocess | aiohttp webhook server |
| Public URL needed? | No | Yes |
| Account ban risk | Yes (unofficial API) | No (officially supported) |
| Inbound | Polling Node bridge | Webhook POST from Meta |
| Outbound | Local bridge → Baileys | HTTPS to graph.facebook.com |
| Groups | Full support | DMs only (v1) |
| 24h window | No restriction | Hard rule — templates required after |
| Voice notes (out) | Native | Native with ffmpeg, MP3 fallback otherwise |
| Read receipts | No | Yes (blue double-checkmarks) |
| Typing indicator | No | Yes (auto-dismisses on response) |
| Interactive buttons | Text fallback only | Native (clarify, approval, slash-confirm) |
| Production use | Risky (Meta can ban) | Designed for it |

A maioria dos usuários do Hermes para projetos pessoais prefere Baileys. A maioria dos bots voltados ao cliente prefere Cloud API.

---

## Veja também {#see-also}

- [Documentação oficial WhatsApp Business Cloud API da Meta](https://developers.facebook.com/documentation/business-messaging/whatsapp/) — referência autoritativa da plataforma subjacente, preços, App Review e limites de taxa do lado Meta.
- [Configuração WhatsApp (ponte Baileys)](whatsapp.md) — integração alternativa para projetos pessoais.
- [Visão geral das plataformas de mensagens](index.md) — todas as integrações de mensagens de relance.
