---
sidebar_position: 12
title: "Google Chat"
description: "Configure o Hermes Agent como um bot do Google Chat usando o Cloud Pub/Sub"
---

# Configuração do Google Chat

Conecte o Hermes Agent ao Google Chat como um bot. A integração usa assinaturas
pull do Cloud Pub/Sub para eventos de entrada e a API REST do Chat para mensagens
de saída. Ergonomia equivalente ao Slack Socket Mode ou ao long-polling do
Telegram: seu processo Hermes não precisa de uma URL pública, de um túnel ou de
um certificado TLS. Ele se conecta, se autentica e escuta em uma assinatura — da
mesma forma que um bot do Telegram escuta em um token.

> Execute `hermes gateway setup` e escolha **Google Chat** para um passo a passo guiado.

:::note Edição Workspace
O Google Chat faz parte do Google Workspace. Você pode usar esta integração com
um Workspace pessoal (`@seudominio.com` registrado através do Google) ou um
Workspace de trabalho onde você tenha direitos de Admin para publicar um app.
Contas somente Gmail não podem hospedar apps do Chat.
:::

## Visão geral {#overview}

| Componente | Valor |
|-----------|-------|
| **Bibliotecas** | `google-cloud-pubsub`, `google-api-python-client`, `google-auth` |
| **Transporte de entrada** | Assinatura pull do Cloud Pub/Sub (sem endpoint público) |
| **Transporte de saída** | API REST do Chat (`chat.googleapis.com`) |
| **Autenticação** | JSON de Service Account com `roles/pubsub.subscriber` na assinatura |
| **Identificação de usuário** | Nomes de recurso do Chat (`users/{id}`) + e-mail |

---

## Passo 1: crie ou escolha um projeto GCP {#step-1-create-or-pick-a-gcp-project}

Você precisa de um projeto do Google Cloud para hospedar o tópico do Pub/Sub.
Se ainda não tiver um, crie-o em
[console.cloud.google.com](https://console.cloud.google.com) — contas pessoais
recebem um nível gratuito que cobre facilmente o tráfego de um bot.

Anote o ID do projeto (por exemplo, `my-chat-bot-123`). Você o usará em todos
os passos seguintes.

---

## Passo 2: ative duas APIs {#step-2-enable-two-apis}

No console, vá em **APIs & Services → Library** e ative:

- **Google Chat API**
- **Cloud Pub/Sub API**

Ambas são gratuitas para os volumes que um bot pessoal gera.

---

## Passo 3: crie uma Service Account {#step-3-create-a-service-account}

**IAM & Admin → Service Accounts → Create Service Account.**

- Nome: `hermes-chat-bot`
- Pule a etapa "Grant this service account access to project". O IAM na
  assinatura específica é tudo o que você precisa — **NÃO** conceda papéis de
  Pub/Sub em nível de projeto.

Após a criação, abra a SA, vá em **Keys → Add Key → Create new key → JSON** e
baixe o arquivo. Salve-o em um local que só o Hermes consiga ler (por exemplo,
`~/.hermes/google-chat-sa.json`, `chmod 600`).

:::caution NÃO existe um papel "Chat Bot Caller"
Um erro comum é procurar por um papel de IAM específico do Chat e concedê-lo em
nível de projeto. Esse papel não existe. A autoridade de bot do Chat vem de
estar instalado em um espaço, não do IAM. Tudo que sua SA precisa é ser
assinante do Pub/Sub na assinatura que você cria no próximo passo.
:::

---

## Passo 4: crie o tópico e a assinatura do Pub/Sub {#step-4-create-the-pubsub-topic-and-subscription}

**Pub/Sub → Topics → Create topic.**

- ID do tópico: `hermes-chat-events`
- Deixe os padrões para o restante.

Após a criação, a página de detalhes do tópico tem uma aba **Subscriptions**.
Crie uma:

- ID da assinatura: `hermes-chat-events-sub`
- Tipo de entrega: **Pull**
- Retenção de mensagens: **7 dias** (para que o backlog sobreviva a um reinício
  do hermes)
- Deixe o restante no padrão.

---

## Passo 5: vínculo de IAM no tópico (crítico) {#step-5-iam-binding-on-the-topic-critical}

No **tópico** (não na assinatura), adicione um principal de IAM:

- Principal: `chat-api-push@system.gserviceaccount.com`
- Papel: `Pub/Sub Publisher`

Sem isso, o Google Chat não consegue publicar eventos no seu tópico e seu bot
nunca receberá nada.

---

## Passo 6: vínculo de IAM na assinatura {#step-6-iam-binding-on-the-subscription}

Na **assinatura**, adicione sua própria Service Account como principal:

- Principal: `hermes-chat-bot@<your-project>.iam.gserviceaccount.com`
- Papel: `Pub/Sub Subscriber`

Conceda também `Pub/Sub Viewer` na mesma assinatura — o Hermes chama
`subscription.get()` na inicialização como verificação de alcançabilidade.

---

## Passo 7: configure o app do Chat {#step-7-configure-the-chat-app}

Vá em **APIs & Services → Google Chat API → Configuration**.

- **Nome do app**: o que você quiser que os usuários vejam ("Hermes" é razoável).
- **URL do avatar**: qualquer PNG público (o Google tem alguns padrões).
- **Descrição**: uma frase curta exibida no diretório de apps.
- **Funcionalidade**: ative **Receive 1:1 messages** e **Join spaces and group
  conversations**.
- **Configurações de conexão**: selecione **Cloud Pub/Sub**, digite o nome do
  tópico `projects/<your-project>/topics/hermes-chat-events`.
- **Visibilidade**: restrinja ao seu workspace (ou usuários específicos) — não
  publique para todos enquanto estiver testando.

Salve.

---

## Passo 8: instale o bot em um espaço de teste {#step-8-install-the-bot-in-a-test-space}

Abra o Google Chat em um navegador. Inicie uma DM com seu app procurando pelo
nome dele no menu **+ New Chat**. Na primeira vez que você mandar uma mensagem
para ele, o Google envia um evento `ADDED_TO_SPACE` que o Hermes usa para
armazenar em cache o próprio `users/{id}` do bot, para filtragem de
autonomensagens.

---

## Passo 9: configure o Hermes {#step-9-configure-hermes}

Adicione a seção do Google Chat em `~/.hermes/.env`:

```bash
# Required
GOOGLE_CHAT_PROJECT_ID=my-chat-bot-123
GOOGLE_CHAT_SUBSCRIPTION_NAME=projects/my-chat-bot-123/subscriptions/hermes-chat-events-sub
GOOGLE_CHAT_SERVICE_ACCOUNT_JSON=/home/you/.hermes/google-chat-sa.json

# Authorization — paste the emails of people allowed to talk to the bot
GOOGLE_CHAT_ALLOWED_USERS=you@yourdomain.com,coworker@yourdomain.com

# Optional
GOOGLE_CHAT_HOME_CHANNEL=spaces/AAAA...         # default delivery destination for cron jobs
GOOGLE_CHAT_MAX_MESSAGES=1                      # Pub/Sub FlowControl; 1 serializes commands per session
GOOGLE_CHAT_MAX_BYTES=16777216                  # 16 MiB — cap on in-flight message bytes
```

O ID do projeto também recai sobre `GOOGLE_CLOUD_PROJECT`, e o caminho da SA
recai sobre `GOOGLE_APPLICATION_CREDENTIALS` — use a convenção que preferir.

Instale as dependências do adaptador do Google Chat através do instalador
mantido para isso. Ele aplica os mesmos pisos de segurança fixados usados
pelas verificações em tempo de execução:

```bash
python -m plugins.platforms.google_chat.oauth --install-deps
```

Inicie o gateway:

```bash
hermes gateway
```

Você deve ver uma linha de log como esta:

```
[GoogleChat] Connected; project=my-chat-bot-123, subscription=<redacted>,
             bot_user_id=users/XXXX, flow_control(msgs=1, bytes=16777216)
```

Envie "hola" na DM de teste. O bot posta um marcador "Hermes is thinking…" e
depois edita essa mesma mensagem no lugar com a resposta real — sem lápides de
"mensagem apagada".

### Personalizando o marcador de estado de trabalho {#customizing-the-working-state-marker}

O texto do marcador é configurável via `typing_status_text` em
`~/.hermes/config.yaml` — por exemplo, um assistente gatinho chamado Ada:

```yaml
platforms:
  google_chat:
    # Custom working-state marker text (default: "Hermes is thinking…").
    typing_status_text: "is pouncing… 🐾"
```

Diferente da linha de status efêmera do Slack, isso é uma **mensagem
realmente postada** que é editada no lugar com a resposta — então o que você
definir aqui aparece brevemente no chat como uma mensagem normal. Defina
`typing_indicator: false` para desativar o marcador por completo.

---

## Formatação e capacidades {#formatting-and-capabilities}

O Google Chat renderiza um subconjunto limitado de markdown:

| Suportado | Não suportado |
|-----------|---------------|
| `*bold*`, `_italic_`, `~strike~`, `` `code` `` | Títulos, listas |
| Imagens inline via URL | Botões de Interactive Card v2 (v1 deste gateway) |
| Anexos de arquivo nativos (após `/setup-files` — veja o Passo 10) | Notas de voz nativas / vídeos circulares |

O prompt de sistema do agente inclui uma dica específica do Google Chat para
que ele conheça esses limites e evite formatação que não vai renderizar.

Limite de tamanho de mensagem: 4000 caracteres por mensagem. Respostas mais
longas do agente são automaticamente divididas em várias mensagens.

Suporte a tópicos (threads): quando um usuário responde dentro de um tópico, o
Hermes detecta o `thread.name` e posta sua resposta no mesmo tópico, de modo
que cada tópico recebe uma sessão Hermes separada.

### Perguntas de esclarecimento como cards interativos {#clarify-questions-as-interactive-cards}

Quando o agente faz uma pergunta de esclarecimento de múltipla escolha, o
adaptador a renderiza como um **Card v2** nativo com um botão por opção mais
um botão **"Other / type answer"**, em vez de uma simples lista numerada em
texto. Clicar em um botão responde a pergunta diretamente (os eventos
`CARD_CLICKED` encaminham a escolha de volta para a sessão em espera). Se o
card falhar ao ser enviado, ou se a pergunta não tiver opções fixas, o
adaptador recorre ao esclarecimento padrão em texto. Nenhuma configuração é
necessária.

---

## Passo 10: entrega nativa de anexos (opcional) {#step-10-native-attachment-delivery-optional}

Por padrão o bot já consegue postar texto, imagens inline via URL e cards de
download para áudio/vídeo/documentos. Para entregar anexos **nativos** do
Chat — o mesmo widget de arquivo que você obtém quando um humano arrasta e
solta um arquivo — cada usuário autoriza o bot uma vez através de um fluxo
OAuth por usuário.

### Por que um fluxo separado {#why-a-separate-flow}

O endpoint `media.upload` do Google Chat rejeita categoricamente a
autenticação por service account:

> This method doesn't support app authentication with a service account.
> Authenticate with a user account.

Não há papel de IAM ou escopo que corrija isso. O endpoint só aceita
credenciais de usuário. Então o bot precisa agir *como um usuário* sempre que
enviar um arquivo — especificamente, como o usuário que pediu o arquivo.

### Configuração única (por perfil) {#one-time-setup-per-profile}

1. Vá em **APIs & Services → Credentials** no mesmo projeto GCP.
2. **Create credentials → OAuth client ID → Desktop app**.
3. Baixe o JSON. Mova-o para o host que executa o Hermes.
4. Registre o cliente no Hermes (execute sob o perfil ao qual você quer
   restringi-lo):

```bash
# Default profile:
python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json

# A named profile gets its own separate registration:
hermes -p <profile> python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json
```

Isso grava o client secret no diretório home do Hermes do perfil ativo (por
exemplo, `~/.hermes/google_chat_user_client_secret.json` para o perfil
padrão). O client secret é **restrito ao perfil, não compartilhado entre
perfis** — cada perfil registra o seu próprio. Isso é deliberado: perfis são
fronteiras de autenticação isoladas, então dois perfis podem apontar para
apps/contas OAuth do Google diferentes. Registre-o uma vez por perfil que
precise de entrega de anexos do Google Chat.

### Autorização por usuário (no chat) {#per-user-authorization-in-chat}

Cada usuário executa o fluxo uma vez, em sua própria DM com o bot:

1. Ele envia `/setup-files` ao bot. O bot responde com o status e o próximo
   passo.
2. Ele envia `/setup-files start`. O bot responde com uma URL OAuth.
3. Ele abre a URL, clica em **Allow** e observa o navegador falhar ao carregar
   `http://localhost:1/?...&code=...`. Essa falha é esperada — o código de
   autenticação está na barra de URL.
4. Ele copia a URL que falhou (ou apenas o valor de `code=...`) e cola de
   volta no chat como `/setup-files <PASTED_URL>`. O bot troca isso por um
   refresh token.

O token fica salvo em
`~/.hermes/google_chat_user_tokens/<sanitized_email>.json`. Solicitações de
arquivo subsequentes na DM daquele usuário usam *o token dele*, então o bot
faz o upload como ele e a mensagem chega no espaço dele.

Para revogar depois: `/setup-files revoke` apaga apenas o token daquele
usuário. Os tokens dos outros usuários permanecem intactos.

### Escopo {#scope}

O fluxo solicita exatamente um escopo: `chat.messages.create`. Isso cobre
tanto `media.upload` quanto o `messages.create` que referencia o
`attachmentDataRef` enviado. Sem Drive, sem escopos mais amplos do Chat —
isso é privilégio mínimo de propósito.

### Comportamento multiusuário {#multi-user-behavior}

Quando quem pediu ainda não tem um token por usuário, o bot recorre a um
token legado de usuário único em `~/.hermes/google_chat_user_token.json` (se
presente, de uma instalação pré-multiusuário). Quando nenhum dos dois está
disponível, o bot posta um aviso claro em texto pedindo para quem solicitou
executar `/setup-files`.

Um usuário que revoga limpa apenas o próprio slot. Um 401/403 do token de um
usuário remove do cache apenas o daquele usuário. Os usuários não interferem
uns nos outros.

---

## Solução de problemas {#troubleshooting}

**O bot fica em silêncio depois de enviar "hola".**

1. Verifique se a assinatura do Pub/Sub tem mensagens não entregues no
   console. Se tiver, o Hermes não está autenticado — verifique
   `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` e se a SA está listada como
   `Pub/Sub Subscriber` na assinatura.
2. Se a assinatura tiver zero mensagens, o Google Chat não está publicando.
   Verifique novamente o vínculo de IAM no **tópico**:
   `chat-api-push@system.gserviceaccount.com` precisa ter `Pub/Sub Publisher`.
3. Verifique os logs de `hermes gateway` por `[GoogleChat] Connected`. Se você
   ver `[GoogleChat] Config validation failed`, a mensagem de erro diz qual
   variável de ambiente corrigir.

**O bot responde, mas aparece uma mensagem de erro em vez da resposta do
agente.**

Verifique os logs por `[GoogleChat] Pub/Sub stream died` — se isso se repetir,
suas credenciais de SA podem ter sido rotacionadas ou a assinatura pode ter
sido excluída. Após 10 tentativas o adaptador se marca como fatal.

**"403 Forbidden" em toda mensagem de saída.**

O bot foi removido do espaço, ou você o revogou no console da Chat API.
Reinstale-o no espaço (o próximo evento `ADDED_TO_SPACE` reativará o envio de
mensagens automaticamente).

**Muitos avisos de "Rate limit hit".**

As cotas padrão da Chat API permitem 60 mensagens por espaço por minuto. Se o
seu agente produzir respostas de streaming longas que ultrapassem isso, o
adaptador tenta novamente com backoff exponencial — mas você ainda verá
latência visível para o usuário. Considere respostas mais concisas ou
aumentar a cota no console do GCP.

**O bot continua postando o aviso de "/setup-files" em vez de arquivos.**

Quem pediu não tem um token OAuth por usuário e não há fallback legado.
Execute `/setup-files` na DM dele e siga o Passo 10. Depois que a troca for
concluída, a próxima solicitação de arquivo faz upload nativo sem precisar
reiniciar o gateway.

**`/setup-files start` diz "No client credentials stored."**

A configuração única não foi feita *para este perfil* (o client secret é
restrito ao perfil, então um registro feito sob um perfil não é visto por
outro). A partir de um terminal, execute-o sob o perfil que o gateway usa:

```bash
# Default profile:
python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json

# Named profile:
hermes -p <profile> python -m plugins.platforms.google_chat.oauth \
    --client-secret /path/to/client_secret.json
```

Depois envie `/setup-files start` novamente.

**`/setup-files <PASTED_URL>` diz "Token exchange failed."**

O código de autenticação é de uso único e de curta duração (tipicamente
alguns minutos). Envie `/setup-files start` para obter uma URL nova e tente
de novo.

---

## Notas de segurança {#security-notes}

- **Escopo da Service Account**: o adaptador solicita os escopos `chat.bot` e
  `pubsub`. O IAM deve ser a aplicação real — conceda à sua SA o mínimo
  (`roles/pubsub.subscriber` + `roles/pubsub.viewer` na assinatura), não
  papéis de Pub/Sub em nível de projeto ou de organização.
- **Proteção de download de anexos**: o Hermes só anexa o token bearer da SA a
  URLs cujo host corresponda a uma lista curta de domínios pertencentes ao
  Google (`googleapis.com`, `drive.google.com`, `lh[3-6].googleusercontent.com`,
  e alguns outros). Qualquer outro host é rejeitado antes da requisição HTTP,
  para proteger contra cenários de SSRF em que um evento manipulado poderia
  redirecionar o token bearer para o serviço de metadados do GCE.
- **Redação**: e-mails de Service Account, caminhos de assinatura e caminhos
  de tópico são removidos da saída de log por `agent/redact.py`. O dump de
  envelope de depuração (`GOOGLE_CHAT_DEBUG_RAW=1`) passa pelo mesmo filtro
  de redação e registra em nível DEBUG.
- **Conformidade**: se você planeja conectar este bot a um workspace
  regulamentado (qualquer coisa com política de residência de dados ou
  governança de IA), obtenha essa aprovação antes da primeira instalação.
- **Escopo OAuth do usuário**: o fluxo de anexos por usuário solicita
  *apenas* `chat.messages.create` — o mínimo que cobre `media.upload` mais o
  `messages.create` subsequente. Os tokens são persistidos como JSON simples
  em `~/.hermes/google_chat_user_tokens/<sanitized_email>.json` (as
  permissões do sistema de arquivos são a proteção — o mesmo modelo do
  arquivo de chave da SA). Cada token pertence a exatamente um usuário; a
  revogação é restrita a esse usuário.
