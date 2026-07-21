---
sidebar_position: 12
title: "Google Chat"
description: "Configure o Hermes Agent como bot do Google Chat usando Cloud Pub/Sub"
---

# Google Chat Setup

Conecte o Hermes Agent ao Google Chat como bot. A integração usa assinaturas pull Cloud Pub/Sub para eventos de entrada e a Chat REST API para mensagens de saída. Ergonomia equivalente ao Slack Socket Mode ou long-polling do Telegram: seu processo Hermes não precisa de URL pública, túnel ou certificado TLS. Ele conecta, autentica e escuta em uma assinatura — da mesma forma que um bot Telegram escuta em um token.

> Execute `hermes gateway setup` e escolha **Google Chat** para um walk-through guiado.

:::note Workspace edition
O Google Chat faz parte do Google Workspace. Você pode usar esta integração com um Workspace pessoal (`@yourdomain.com` registrado pelo Google) ou um Workspace corporativo onde você tem direitos de Admin para publicar um app. Contas só Gmail não podem hospedar apps Chat.
:::

## Overview {#overview}

| Component | Value |
|-----------|-------|
| **Libraries** | `google-cloud-pubsub`, `google-api-python-client`, `google-auth` |
| **Inbound transport** | Assinatura pull Cloud Pub/Sub (sem endpoint público) |
| **Outbound transport** | Chat REST API (`chat.googleapis.com`) |
| **Authentication** | Service Account JSON com `roles/pubsub.subscriber` na assinatura |
| **User identification** | Resource names Chat (`users/{id}`) + e-mail |

---

## Step 1: Create or pick a GCP project {#step-1-create-or-pick-a-gcp-project}

Você precisa de um projeto Google Cloud para hospedar o tópico Pub/Sub. Se não tiver um, crie em [console.cloud.google.com](https://console.cloud.google.com) — contas pessoais têm free tier que cobre facilmente o tráfego de um bot.

Anote o project ID (ex.: `my-chat-bot-123`). Você vai usá-lo em cada passo seguinte.

---

## Step 2: Enable two APIs {#step-2-enable-two-apis}

No console, vá em **APIs & Services → Library** e habilite:

- **Google Chat API**
- **Cloud Pub/Sub API**

Ambas são gratuitas para os volumes que um bot pessoal gera.

---

## Step 3: Create a Service Account {#step-3-create-a-service-account}

**IAM & Admin → Service Accounts → Create Service Account.**

- Name: `hermes-chat-bot`
- Pule o passo "Grant this service account access to project". IAM na assinatura específica é tudo que você precisa — **NÃO** conceda roles Pub/Sub em nível de projeto.

Após criar, abra a SA, vá em **Keys → Add Key → Create new key → JSON** e baixe o arquivo. Salve onde só o Hermes possa ler (ex.: `~/.hermes/google-chat-sa.json`, `chmod 600`).

:::caution There is NO "Chat Bot Caller" role
Um erro comum é buscar uma role IAM específica de Chat e concedê-la em nível de projeto. Essa role não existe. A autoridade de bot Chat vem de estar instalado em um space, não de IAM. Tudo que sua SA precisa é Pub/Sub subscriber na assinatura que você cria no próximo passo.
:::

---

## Step 4: Create the Pub/Sub topic and subscription {#step-4-create-the-pubsub-topic-and-subscription}

**Pub/Sub → Topics → Create topic.**

- Topic ID: `hermes-chat-events`
- Deixe os padrões para o resto.

Após criar, a página de detalhes do tópico tem a aba **Subscriptions**. Crie uma:

- Subscription ID: `hermes-chat-events-sub`
- Delivery type: **Pull**
- Message retention: **7 days** (para backlog sobreviver a reinício do hermes)
- Deixe o resto padrão.

---

## Step 5: IAM binding on the topic (critical) {#step-5-iam-binding-on-the-topic-critical}

No **tópico** (não na assinatura), adicione um principal IAM:

- Principal: `chat-api-push@system.gserviceaccount.com`
- Role: `Pub/Sub Publisher`

Sem isso, o Google Chat não consegue publicar eventos no seu tópico e seu bot nunca receberá nada.

---

## Step 6: IAM binding on the subscription {#step-6-iam-binding-on-the-subscription}

Na **assinatura**, adicione sua própria Service Account como principal:

- Principal: `hermes-chat-bot@<your-project>.iam.gserviceaccount.com`
- Role: `Pub/Sub Subscriber`

Também conceda `Pub/Sub Viewer` na mesma assinatura — o Hermes chama `subscription.get()` na inicialização como verificação de reachability.

---

## Step 7: Configure the Chat app {#step-7-configure-the-chat-app}

Vá em **APIs & Services → Google Chat API → Configuration**.

- **App name**: o que quiser que usuários vejam ("Hermes" é razoável).
- **Avatar URL**: qualquer PNG público (Google tem alguns padrões).
- **Description**: frase curta no diretório de apps.
- **Functionality**: habilite **Receive 1:1 messages** e **Join spaces and group conversations**.
- **Connection settings**: selecione **Cloud Pub/Sub**, informe o nome do tópico `projects/<your-project>/topics/hermes-chat-events`.
- **Visibility**: restrinja ao seu workspace (ou usuários específicos) — não publique para todos enquanto testa.

Salve.

---

## Step 8: Install the bot in a test space {#step-8-install-the-bot-in-a-test-space}

Abra o Google Chat no navegador. Inicie uma DM com seu app buscando o nome no menu **+ New Chat**. Na primeira mensagem, o Google envia um evento `ADDED_TO_SPACE` que o Hermes usa para cachear o próprio `users/{id}` do bot para filtragem de mensagens próprias.

---

## Step 9: Configure Hermes {#step-9-configure-hermes}

Adicione a seção Google Chat em `~/.hermes/.env`:

```bash
# Obrigatório
GOOGLE_CHAT_PROJECT_ID=my-chat-bot-123
GOOGLE_CHAT_SUBSCRIPTION_NAME=projects/my-chat-bot-123/subscriptions/hermes-chat-events-sub
GOOGLE_CHAT_SERVICE_ACCOUNT_JSON=/home/you/.hermes/google-chat-sa.json

# Autorização — cole os e-mails de quem pode falar com o bot
GOOGLE_CHAT_ALLOWED_USERS=you@yourdomain.com,coworker@yourdomain.com

# Opcional
GOOGLE_CHAT_HOME_CHANNEL=spaces/AAAA...         # destino padrão de entrega para cron jobs
GOOGLE_CHAT_MAX_MESSAGES=1                      # Pub/Sub FlowControl; 1 serializa comandos por sessão
GOOGLE_CHAT_MAX_BYTES=16777216                  # 16 MiB — limite de bytes de mensagem in-flight
```

O project ID também faz fallback para `GOOGLE_CLOUD_PROJECT`, e o caminho da SA para `GOOGLE_APPLICATION_CREDENTIALS` — use a convenção que preferir.

Instale as dependências que o adapter Google Chat precisa (nenhum extra Hermes está publicado atualmente — instale diretamente):

```bash
pip install google-cloud-pubsub google-api-python-client google-auth google-auth-oauthlib
```

Inicie o gateway:

```bash
hermes gateway
```

Você deve ver uma linha de log como:

```
[GoogleChat] Connected; project=my-chat-bot-123, subscription=<redacted>,
             bot_user_id=users/XXXX, flow_control(msgs=1, bytes=16777216)
```

Envie "hola" na DM de teste. O bot publica um marcador "Hermes is thinking…", depois edita a mesma mensagem in place com a resposta real — sem tombstones de "message deleted".

### Customizing the working-state marker

O texto do marcador é configurável via `typing_status_text` em `~/.hermes/config.yaml` — ex.: um assistente gatinho chamado Ada:

```yaml
platforms:
  google_chat:
    # Texto customizado do marcador de estado de trabalho (padrão: "Hermes is thinking…").
    typing_status_text: "is pouncing… 🐾"
```

Diferente da linha de status efêmera do Slack, esta é uma **mensagem real publicada** que é editada in place com a resposta — então o que você definir aparece brevemente no chat como mensagem normal. Defina `typing_indicator: false` para desabilitar o marcador.

---

## Formatting and capabilities {#formatting-and-capabilities}

O Google Chat renderiza um subconjunto limitado de markdown:

| Supported | Not supported |
|-----------|---------------|
| `*bold*`, `_italic_`, `~strike~`, `` `code` `` | Headings, lists |
| Inline images via URL | Interactive Card v2 buttons (v1 deste gateway) |
| Anexos de arquivo nativos (após `/setup-files` — veja Step 10) | Voice notes / circular video notes nativos |

O system prompt do agente inclui uma dica específica Google Chat para que ele conheça esses limites e evite formatação que não renderiza.

Limite de tamanho de mensagem: 4000 caracteres por mensagem. Respostas longas do agente são divididas automaticamente em várias mensagens.

Suporte a threads: quando um usuário responde dentro de uma thread, o Hermes detecta `thread.name` e publica a resposta na mesma thread, então cada thread recebe uma sessão Hermes separada.

---

## Step 10: Native attachment delivery (optional) {#step-10-native-attachment-delivery-optional}

Out of the box o bot pode postar texto, imagens inline via URL e cards de download para áudio/vídeo/documentos. Para entregar **anexos nativos** Chat — o mesmo widget de arquivo quando um humano arrasta-e-solta — cada usuário autoriza o bot uma vez via fluxo OAuth por usuário.

### Why a separate flow

O endpoint `media.upload` do Google Chat rejeita hard auth de service account:

> This method doesn't support app authentication with a service account.
> Authenticate with a user account.

Não há role IAM ou scope que corrija isso. O endpoint só aceita credenciais de usuário. Então o bot precisa agir *como um usuário* sempre que faz upload — especificamente, como o usuário que pediu o arquivo.

### One-time setup (per profile)

1. Vá em **APIs & Services → Credentials** no mesmo projeto GCP.
2. **Create credentials → OAuth client ID → Desktop app**.
3. Baixe o JSON. Mova para o host que roda o Hermes.
4. Registre o client no Hermes (execute no perfil que quer escopar):

```bash
# Perfil padrão:
python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json

# Um perfil nomeado recebe registro separado:
hermes -p <profile> python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json
```

Isso grava o client secret no Hermes home do perfil ativo (ex.: `~/.hermes/google_chat_user_client_secret.json` para o perfil padrão). O client secret é **escopado por perfil, não compartilhado entre perfis** — cada perfil registra o seu. Isso é intencional: perfis são fronteiras de auth isoladas, então dois perfis podem apontar para apps/contas Google OAuth diferentes. Registre uma vez por perfil que precisa de entrega de anexos Google Chat.

### Per-user authorization (in chat)

Cada usuário executa o fluxo uma vez, na própria DM com o bot:

1. Enviam `/setup-files` ao bot. Ele responde com status e o próximo passo.
2. Enviam `/setup-files start`. O bot responde com uma URL OAuth.
3. Abrem a URL, clicam **Allow** e veem o navegador falhar ao carregar `http://localhost:1/?...&code=...`. Essa falha é esperada — o auth code está na barra de URL.
4. Copiam a URL falha (ou só o valor `code=...`) e colam no chat como `/setup-files <PASTED_URL>`. O bot troca por refresh token.

O token vai para `~/.hermes/google_chat_user_tokens/<sanitized_email>.json`. Pedidos subsequentes de arquivo na DM daquele usuário usam *seu* token, então o bot faz upload como eles e a mensagem cai no space deles.

Para revogar depois: `/setup-files revoke` apaga só o token daquele usuário. Tokens de outros usuários ficam intactos.

### Scope

O fluxo solicita exatamente um scope: `chat.messages.create`. Cobre tanto `media.upload` quanto o `messages.create` que referencia o `attachmentDataRef` enviado. Sem Drive, sem scopes Chat mais amplos — least-privilege de propósito.

### Multi-user behavior

Quando quem pediu ainda não tem token por usuário, o bot faz fallback para token legacy de usuário único em `~/.hermes/google_chat_user_token.json` (se existir de instalação pré-multi-user). Quando nenhum está disponível, o bot publica aviso claro pedindo para executar `/setup-files`.

Revogar de um usuário limpa só o slot dele. 401/403 de um token evicta só o cache daquele usuário. Usuários não atrapalham uns aos outros.

---

## Troubleshooting {#troubleshooting}

**Bot fica silencioso após enviar "hola."**

1. Verifique se a assinatura Pub/Sub tem mensagens não entregues no console. Se sim, o Hermes não está autenticado — confira `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` e se a SA está listada como `Pub/Sub Subscriber` na assinatura.
2. Se a assinatura tem zero mensagens, o Google Chat não está publicando. Confira o IAM binding no **tópico**: `chat-api-push@system.gserviceaccount.com` deve ter `Pub/Sub Publisher`.
3. Verifique logs de `hermes gateway` por `[GoogleChat] Connected`. Se vir `[GoogleChat] Config validation failed`, a mensagem de erro indica qual env var corrigir.

**Bot responde mas aparece mensagem de erro em vez da resposta do agente.**

Verifique logs por `[GoogleChat] Pub/Sub stream died` — se repetir, suas credenciais SA podem ter sido rotacionadas ou a assinatura deletada. Após 10 tentativas o adapter se marca fatal.

**"403 Forbidden" em toda mensagem outbound.**

O bot foi removido do space, ou você revogou no console Chat API. Reinstale no space (o próximo evento `ADDED_TO_SPACE` reabilitará mensagens automaticamente).

**Muitos avisos "Rate limit hit".**

As quotas padrão da Chat API permitem 60 mensagens por space por minuto. Se seu agente produz respostas longas em streaming acima disso, o adapter retenta com backoff exponencial — mas você ainda verá latência visível. Considere respostas concisas ou aumente a quota no console GCP.

**Bot continua postando aviso "/setup-files" em vez de arquivos.**

Quem pediu não tem token OAuth por usuário e não há fallback legacy. Execute `/setup-files` na DM deles e siga o Step 10. Após a troca completar, o próximo pedido de arquivo faz upload nativamente sem reiniciar o gateway.

**`/setup-files start` diz "No client credentials stored."**

O setup one-time não foi feito *para este perfil* (o client secret é escopado por perfil, então registro em um perfil não é visto por outro). No terminal, execute no perfil que o gateway usa:

```bash
# Perfil padrão:
python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json

# Perfil nomeado:
hermes -p <profile> python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json
```

Depois envie `/setup-files start` novamente.

**`/setup-files <PASTED_URL>` diz "Token exchange failed."**

O auth code é single-use e de curta duração (tipicamente alguns minutos). Envie `/setup-files start` para URL fresca e tente de novo.

---

## Security notes {#security-notes}

- **Service Account scope**: o adapter solicita scopes `chat.bot` e `pubsub`. IAM deve ser a enforcement real — conceda à SA o mínimo (`roles/pubsub.subscriber` + `roles/pubsub.viewer` na assinatura), não roles Pub/Sub em nível de projeto ou org.
- **Attachment download protection**: o Hermes só anexa o bearer token SA a URLs cujo host corresponde a uma allowlist curta de domínios Google (`googleapis.com`, `drive.google.com`, `lh[3-6].googleusercontent.com` e alguns outros). Qualquer outro host é rejeitado antes da requisição HTTP, para proteger contra SSRF onde um evento craftado redirecionaria o bearer token ao metadata service GCE.
- **Redaction**: e-mails de Service Account, caminhos de assinatura e tópicos são removidos da saída de log por `agent/redact.py`. O dump de envelope debug (`GOOGLE_CHAT_DEBUG_RAW=1`) passa pelo mesmo filtro de redação e loga em nível DEBUG.
- **Compliance**: se planeja conectar este bot a um workspace regulado (qualquer coisa com política de residência de dados ou governança de IA), obtenha aprovação antes da primeira instalação.
- **User OAuth scope**: o fluxo de anexo por usuário solicita *apenas* `chat.messages.create` — o mínimo que cobre `media.upload` mais o `messages.create` de follow-up. Tokens persistem como JSON plain em `~/.hermes/google_chat_user_tokens/<sanitized_email>.json` (permissões de filesystem são a proteção — mesmo modelo do arquivo de chave SA). Cada token pertence a exatamente um usuário; revoke é escopado a esse usuário.
