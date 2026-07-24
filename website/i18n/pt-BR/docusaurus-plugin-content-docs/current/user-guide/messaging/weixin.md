---
sidebar_position: 15
title: "Weixin (WeChat)"
description: "Conecte o Hermes Agent a contas pessoais WeChat via iLink Bot API"
---

# Weixin (WeChat)

Conecte o Hermes ao [WeChat](https://weixin.qq.com/) (微信), a plataforma de mensagens pessoais da Tencent. O adaptador usa a **iLink Bot API** da Tencent para contas pessoais WeChat — isso é distinto do WeCom (Enterprise WeChat). Mensagens são entregues via long-polling, então nenhum endpoint público ou webhook é necessário.

:::info
Este adaptador é para **contas pessoais WeChat** (微信). Se você precisa de WeChat empresarial/corporativo, veja o [adaptador WeCom](./wecom.md).
:::

:::warning iLink bot identity — ordinary WeChat groups may not work
O login por QR conecta o Hermes a uma **identidade de bot iLink** (ex.: `a5ace6fd482e@im.bot`), **não** a uma conta pessoal WeChat totalmente scriptável. Consequências:

- A identidade de bot iLink geralmente **não pode ser convidada para grupos WeChat comuns** da forma que um contato normal pode.
- O iLink tipicamente **não entrega eventos de grupos WeChat comuns** (incluindo `@`-menções da conta pessoal usada para login QR) ao gateway para a maioria dos tipos de conta bot.
- `@`-mencionar a conta pessoal WeChat usada para escanear o QR code **não** é o mesmo que `@`-mencionar o bot iLink — o bot é uma identidade separada.
- As configurações `WEIXIN_GROUP_POLICY` / `WEIXIN_GROUP_ALLOWED_USERS` abaixo só têm efeito quando o iLink realmente retorna eventos de grupo para seu tipo de conta. Se não retornar, mensagens de grupo nunca chegarão ao Hermes independentemente da política.

Na prática, a maioria dos deployments só consegue DMs para o bot iLink funcionando de forma confiável. Se a entrega em grupos não funcionar após a configuração, a limitação está no lado iLink, não no Hermes. O gateway registra um `WARNING` na inicialização sempre que `WEIXIN_GROUP_POLICY` estiver definido como qualquer coisa diferente de `disabled`.
:::

## Pré-requisitos {#prerequisites}

- Uma conta pessoal WeChat
- Pacotes Python: `aiohttp` e `cryptography`
- A renderização de QR no terminal está incluída quando o Hermes é instalado com o extra `messaging`

Instale as dependências necessárias:

```bash
pip install aiohttp cryptography
# Optional: for terminal QR code display
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"
```

## Configuração {#setup}

### 1. Execute o assistente de setup

A forma mais fácil de conectar sua conta WeChat é pelo setup interativo:

```bash
hermes gateway setup
```

Selecione **Weixin** quando solicitado. O assistente irá:

1. Solicitar um QR code da iLink Bot API
2. Exibir o QR code no seu terminal (ou fornecer uma URL)
3. Aguardar você escanear o QR code com o app móvel WeChat
4. Pedir que você confirme o login no telefone
5. Salvar automaticamente as credenciais da conta em `~/.hermes/weixin/accounts/`

Depois de confirmado, você verá uma mensagem como:

```
微信连接成功，account_id=your-account-id
```

O assistente armazena `account_id`, `token` e `base_url` para que você não precise configurá-los manualmente.

### 2. Configure variáveis de ambiente

Após o login QR inicial, defina no mínimo o account ID em `~/.hermes/.env`:

```bash
WEIXIN_ACCOUNT_ID=your-account-id

# Optional: override the token (normally auto-saved from QR login)
# WEIXIN_TOKEN=your-bot-token

# Optional: restrict access
WEIXIN_DM_POLICY=open
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2

# Optional: restore legacy multiline splitting behavior
# WEIXIN_SPLIT_MULTILINE_MESSAGES=true

# Optional: home channel for cron/notifications
WEIXIN_HOME_CHANNEL=chat_id
WEIXIN_HOME_CHANNEL_NAME=Home
```

### 3. Inicie o gateway

```bash
hermes gateway
```

O adaptador restaurará credenciais salvas, conectará à API iLink e começará o long-polling de mensagens.

## Recursos {#features}

- **Transporte long-poll** — nenhum endpoint público, webhook ou WebSocket necessário
- **Login por QR code** — setup scan-to-connect via `hermes gateway setup`
- **Mensagens DM** — políticas de acesso configuráveis; mensagens de grupo dependem do iLink realmente entregar eventos de grupo para a identidade conectada (frequentemente não é o caso para contas bot iLink — veja o aviso acima)
- **Suporte a mídia** — imagens, vídeo, arquivos e mensagens de voz
- **CDN criptografado AES-128-ECB** — criptografia/descriptografia automática para todas as transferências de mídia
- **Persistência de context token** — continuidade de resposta em disco entre reinicializações
- **Formatação Markdown** — preserva Markdown, incluindo cabeçalhos, tabelas e blocos de código, para que clientes WeChat que suportam Markdown possam renderizá-lo nativamente
- **Chunking inteligente de mensagens** — mensagens permanecem em uma única bolha quando abaixo do limite; apenas payloads oversized dividem em limites lógicos
- **Indicadores de digitação** — exibe status "typing…" no cliente WeChat enquanto o agente processa
- **Proteção SSRF** — URLs de mídia de saída são validadas antes do download
- **Deduplicação de mensagens** — janela deslizante de 5 minutos evita processamento duplo
- **Retry automático com backoff** — recupera de erros transitórios de API

## Opções de configuração {#configuration-options}

Defina estes em `config.yaml` em `platforms.weixin.extra`:

| Key | Default | Description |
|-----|---------|-------------|
| `account_id` | — | iLink Bot account ID (obrigatório) |
| `token` | — | iLink Bot token (obrigatório, salvo automaticamente do login QR) |
| `base_url` | `https://ilinkai.weixin.qq.com` | URL base da API iLink |
| `cdn_base_url` | `https://novac2c.cdn.weixin.qq.com/c2c` | URL base CDN para transferência de mídia |
| `dm_policy` | `open` | Acesso a DM: `open`, `allowlist`, `disabled`, `pairing` |
| `group_policy` | `disabled` | Acesso a grupo: `open`, `allowlist`, `disabled` |
| `allow_from` | `[]` | IDs de usuário permitidos para DMs (quando dm_policy=allowlist) |
| `group_allow_from` | `[]` | IDs de grupo permitidos (quando group_policy=allowlist) |
| `split_multiline_messages` | `false` | Quando `true`, divide respostas multilinha em várias mensagens de chat (comportamento legado). Quando `false`, mantém respostas multilinha como uma mensagem, a menos que excedam o limite de tamanho. |
| `text_batch_delay_seconds` | `3.0` | Período quiet (segundos) antes de um burst bufferizado de mensagens de texto rápidas ser liberado como uma solicitação combinada. O iLink entrega mensagens individualmente, então esse debounce evita uma invocação de agente por fragmento. Defina `0` para despachar cada mensagem imediatamente. |
| `text_batch_split_delay_seconds` | `5.0` | Delay de flush estendido usado quando o fragmento mais recente está perto do limiar de split (mensagens longas que o iLink pode ter fragmentado). |

## Políticas de acesso {#access-policies}

### Política de DM

Controla quem pode enviar mensagens diretas ao bot:

| Value | Behavior |
|-------|----------|
| `open` | Qualquer pessoa pode enviar DM ao bot (padrão) |
| `allowlist` | Apenas IDs de usuário em `allow_from` podem enviar DM |
| `disabled` | Todas as DMs são ignoradas |
| `pairing` | Modo pairing (para configuração inicial) |

```bash
WEIXIN_DM_POLICY=allowlist
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2
```

`WEIXIN_ALLOWED_USERS` é um **filtro de entrada**, não um sistema de convite. O login QR
conecta uma identidade de bot iLink ao Hermes. Outras pessoas não escaneiam o
QR code do Hermes com suas próprias contas; elas devem enviar mensagem ao bot iLink
conectado/contato pelo WeChat, e o Hermes processará a DM somente se o ID de usuário Weixin
do remetente estiver presente em `WEIXIN_ALLOWED_USERS`.

Um fluxo prático de configuração é:

1. Faça pairing do Hermes uma vez com `hermes gateway setup` e anote a conta de bot iLink
   conectada.
2. Peça a cada usuário permitido que envie uma mensagem direta a esse bot/contato.
3. Leia o sender/user ID dos logs do gateway ou do payload do evento de entrada.
4. Adicione esses IDs a `WEIXIN_ALLOWED_USERS` e reinicie o gateway.

Se apenas a conta que escaneou o QR code consegue falar com o Hermes, verifique se os
outros usuários estão enviando mensagem à identidade de bot iLink em si, não à conta pessoal WeChat
que realizou o login QR. O bot iLink é uma identidade separada, e
o roteamento de contato/grupo WeChat comum pode ser limitado pelo comportamento iLink da Tencent.

### Política de grupo

Controla em quais grupos o bot responde **quando o iLink entrega eventos de grupo para a identidade conectada**. Para identidades de bot iLink de login QR (ex.: `...@im.bot`), eventos de grupo tipicamente não são entregues de todo, então essa política pode não ter efeito — veja o aviso de limitação de bot iLink no topo da página.

| Value | Behavior |
|-------|----------|
| `open` | Bot responde em todos os grupos (se eventos forem entregues) |
| `allowlist` | Bot responde apenas em IDs de grupo listados em `group_allow_from` (se eventos forem entregues) |
| `disabled` | Todas as mensagens de grupo são ignoradas (padrão) |

```bash
WEIXIN_GROUP_POLICY=allowlist
# NOTE: this is a comma-separated list of group chat IDs, NOT member user IDs,
# despite the variable name containing "USERS". Keep this in mind when configuring.
WEIXIN_GROUP_ALLOWED_USERS=group_id_1,group_id_2
```

:::note
A política de grupo padrão é `disabled` para Weixin (diferente do WeCom, onde o padrão é `open`). Isso é intencional — contas pessoais WeChat podem estar em muitos grupos, e identidades de bot iLink tipicamente não conseguem receber mensagens de grupos WeChat comuns de todo. O gateway registra um `WARNING` na inicialização se você definir `WEIXIN_GROUP_POLICY` como qualquer coisa diferente de `disabled`.
:::

## Suporte a mídia {#media-support}

### Entrada (recebimento)

O adaptador recebe anexos de mídia de usuários, baixa-os do CDN WeChat, descriptografa-os e os armazena em cache localmente para processamento pelo agente:

| Type | How it's handled |
|------|-----------------| 
| **Images** | Baixadas, descriptografadas com AES e armazenadas em cache como JPEG. |
| **Video** | Baixado, descriptografado com AES e armazenado em cache como MP4. |
| **Files** | Baixados, descriptografados com AES e armazenados em cache. Nome de arquivo original preservado. |
| **Voice** | Se uma transcrição de texto estiver disponível, é extraída como texto. Caso contrário, o áudio (formato SILK) é baixado e armazenado em cache. |

**Mensagens citadas:** Mídia de mensagens citadas (respondidas) também é extraída, para que o agente tenha contexto sobre o que o usuário está respondendo.

### CDN criptografado AES-128-ECB

Arquivos de mídia WeChat são transferidos por um CDN criptografado. O adaptador trata isso transparentemente:

- **Entrada:** Mídia criptografada é baixada do CDN usando URLs `encrypted_query_param`, depois descriptografada com AES-128-ECB usando a chave por arquivo fornecida no payload da mensagem.
- **Saída:** Arquivos são criptografados localmente com uma chave AES-128-ECB aleatória, enviados ao CDN, e a referência criptografada é incluída na mensagem de saída.
- A chave AES tem 16 bytes (128-bit). Chaves podem chegar como base64 bruto ou codificadas em hex — o adaptador trata ambos os formatos.
- Isso requer o pacote Python `cryptography`.

Nenhuma configuração é necessária — criptografia e descriptografia acontecem automaticamente.

### Saída (envio)

| Method | What it sends |
|--------|--------------|
| `send` | Mensagens de texto com formatação Markdown | 
| `send_image` / `send_image_file` | Mensagens de imagem nativas (via upload CDN) |
| `send_document` | Anexos de arquivo (via upload CDN) |
| `send_video` | Mensagens de vídeo (via upload CDN) |

Toda mídia de saída passa pelo fluxo de upload CDN criptografado:

1. Gere uma chave AES-128 aleatória
2. Criptografe o arquivo com AES-128-ECB + padding PKCS#7
3. Solicite uma URL de upload da API iLink (`getuploadurl`)
4. Envie o ciphertext ao CDN
5. Envie a mensagem com a referência de mídia criptografada

## Persistência de context token {#context-token-persistence}

A iLink Bot API requer que um `context_token` seja ecoado de volta com cada mensagem de saída para um peer dado. O adaptador mantém um store de context token em disco:

- Tokens são salvos por account+peer em `~/.hermes/weixin/accounts/<account_id>.context-tokens.json`
- Na inicialização, tokens previamente salvos são restaurados
- Toda mensagem de entrada atualiza o token armazenado para esse remetente
- Mensagens de saída incluem automaticamente o context token mais recente

Isso garante continuidade de resposta mesmo após reinicializações do gateway.

## Formatação Markdown {#markdown-formatting}

Clientes WeChat conectados pela iLink Bot API podem renderizar Markdown diretamente, então o adaptador preserva Markdown em vez de reescrevê-lo:

- **Headers** permanecem como cabeçalhos Markdown (`#`, `##`, ...)
- **Tables** permanecem como tabelas Markdown
- **Code fences** permanecem como blocos de código fenced
- **Linhas em branco excessivas** são colapsadas para duplas quebras de linha fora de blocos de código fenced

## Chunking de mensagens {#message-chunking}

Mensagens são entregues como uma única mensagem de chat sempre que couberem no limite da plataforma. Apenas payloads oversized são divididos para entrega:

- Comprimento máximo de mensagem: **4000 caracteres**
- Mensagens abaixo do limite permanecem intactas mesmo quando contêm vários parágrafos ou quebras de linha
- Mensagens oversized dividem em limites lógicos (parágrafos, linhas em branco, code fences)
- Code fences são mantidos intactos sempre que possível (nunca divididos no meio do bloco, a menos que o fence em si exceda o limite)
- Blocos individuais oversized recorrem à lógica de truncamento do adaptador base
- Um delay inter-chunk de 0,3 s evita drops de rate limit do WeChat quando vários chunks são enviados

## Indicadores de digitação {#typing-indicators}

O adaptador exibe status de digitação no cliente WeChat:

1. Quando uma mensagem chega, o adaptador busca um `typing_ticket` via API `getconfig`
2. Typing tickets são armazenados em cache por 10 minutos por usuário
3. `send_typing` envia um sinal de início de digitação; `stop_typing` envia um sinal de parada de digitação
4. O gateway dispara automaticamente indicadores de digitação enquanto o agente processa uma mensagem

## Conexão long-poll {#long-poll-connection}

O adaptador usa HTTP long-polling (não WebSocket) para receber mensagens:

### Como funciona

1. **Connect:** Valida credenciais e inicia o loop de poll
2. **Poll:** Chama `getupdates` com timeout de 35 segundos; o servidor segura a requisição até mensagens chegarem ou o timeout expirar
3. **Dispatch:** Mensagens de entrada são despachadas concorrentemente via `asyncio.create_task`
4. **Sync buffer:** Um cursor de sync persistente (`get_updates_buf`) é salvo em disco para que o adaptador retome da posição correta após reinicializações

### Comportamento de retry

Em erros de API, o adaptador usa uma estratégia simples de retry:

| Condition | Behavior |
|-----------|----------|
| Transient error (1st–2nd) | Retry after 2 seconds |
| Repeated errors (3+) | Back off for 30 seconds, then reset counter |
| Session expired (`errcode=-14`) | Pause for 10 minutes (re-login may be needed) |
| Timeout | Immediately re-poll (normal long-poll behavior) |

### Deduplicação

Mensagens de entrada são deduplicadas usando IDs de mensagem com janela de 5 minutos. Isso evita processamento duplo durante instabilidades de rede ou respostas de poll sobrepostas.

### Token lock

Apenas uma instância de gateway Weixin pode usar um token dado por vez. O adaptador adquire um lock com escopo na inicialização e o libera no shutdown. Se outro gateway já estiver usando o mesmo token, a inicialização falha com uma mensagem de erro informativa.

## Todas as variáveis de ambiente {#all-environment-variables}

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WEIXIN_ACCOUNT_ID` | ✅ | — | iLink Bot account ID (do login QR) |
| `WEIXIN_TOKEN` | ✅ | — | iLink Bot token (salvo automaticamente do login QR) |
| `WEIXIN_BASE_URL` | — | `https://ilinkai.weixin.qq.com` | URL base da API iLink |
| `WEIXIN_CDN_BASE_URL` | — | `https://novac2c.cdn.weixin.qq.com/c2c` | URL base CDN para transferência de mídia |
| `WEIXIN_DM_POLICY` | — | `open` | Política de acesso a DM: `open`, `allowlist`, `disabled`, `pairing` |
| `WEIXIN_GROUP_POLICY` | — | `disabled` | Política de acesso a grupo: `open`, `allowlist`, `disabled` |
| `WEIXIN_ALLOWED_USERS` | — | _(empty)_ | IDs de usuário separados por vírgula para allowlist de DM |
| `WEIXIN_GROUP_ALLOWED_USERS` | — | _(empty)_ | **IDs de chat de grupo** separados por vírgula (não IDs de usuário membro) para allowlist de grupo. O nome da variável é legado — espera IDs de grupo, não IDs de usuário. |
| `WEIXIN_HOME_CHANNEL` | — | — | ID de chat para saída de cron/notificação |
| `WEIXIN_HOME_CHANNEL_NAME` | — | `Home` | Nome de exibição do canal home |
| `WEIXIN_ALLOW_ALL_USERS` | — | — | Flag de nível gateway para permitir todos os usuários (usada pelo assistente de setup) |

## Solução de problemas {#troubleshooting}

| Problem | Fix |
|---------|-----|
| `Weixin startup failed: aiohttp and cryptography are required` | Instale ambos: `pip install aiohttp cryptography` |
| `Weixin startup failed: WEIXIN_TOKEN is required` | Execute `hermes gateway setup` para completar login QR, ou defina `WEIXIN_TOKEN` manualmente |
| `Weixin startup failed: WEIXIN_ACCOUNT_ID is required` | Defina `WEIXIN_ACCOUNT_ID` no seu `.env` ou execute `hermes gateway setup` |
| `Another local Hermes gateway is already using this Weixin token` | Pare a outra instância do gateway primeiro — apenas um poller por token é permitido |
| Session expired (`errcode=-14`) | Sua sessão de login expirou. Execute novamente `hermes gateway setup` para escanear um novo QR code |
| QR code expired during setup | O QR atualiza automaticamente até 3 vezes. Se continuar expirando, verifique sua conexão de rede |
| Bot doesn't respond to DMs | Verifique `WEIXIN_DM_POLICY` — se definido como `allowlist`, o remetente deve estar em `WEIXIN_ALLOWED_USERS` |
| Bot ignores group messages | Política de grupo padrão é `disabled`. Defina `WEIXIN_GROUP_POLICY=open` ou `allowlist` — mas note que identidades de bot iLink de login QR (`...@im.bot`) tipicamente não conseguem receber mensagens de grupos WeChat comuns de todo. Se os logs do gateway não mostrarem eventos brutos de entrada para mensagens de grupo, a limitação está no lado iLink, não no Hermes. |
| Media download/upload fails | Certifique-se de que `cryptography` está instalado. Verifique acesso de rede a `novac2c.cdn.weixin.qq.com` |
| `Blocked unsafe URL (SSRF protection)` | A URL de mídia de saída aponta para endereço privado/interno. Apenas URLs públicas são permitidas |
| Voice messages show as text | Se o WeChat fornece transcrição, o adaptador usa o texto. Esse é o comportamento esperado |
| Messages appear duplicated | O adaptador deduplica por ID de mensagem. Se vir duplicatas, verifique se várias instâncias de gateway estão rodando |
| `iLink POST ... HTTP 4xx/5xx` | Erro de API do serviço iLink. Verifique validade do token e conectividade de rede |
| Terminal QR code doesn't render | Reinstale com o extra messaging: `cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"`. Alternativamente, abra a URL impressa acima do QR |
