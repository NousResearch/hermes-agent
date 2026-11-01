---
sidebar_position: 10
title: "DingTalk"
description: "Configure o Hermes Agent como chatbot DingTalk"
---

# DingTalk Setup

O Hermes Agent integra-se ao DingTalk (钉钉) como chatbot, permitindo que você converse com seu assistente de IA por mensagens diretas ou chats de grupo. O bot conecta via Stream Mode do DingTalk — uma conexão WebSocket de longa duração que não requer URL pública ou servidor webhook — e responde usando mensagens formatadas em markdown pela API session webhook do DingTalk.

Antes da configuração, aqui está o que a maioria das pessoas quer saber: como o Hermes se comporta depois de estar no seu workspace DingTalk.

## Como o Hermes se comporta {#how-hermes-behaves}

| Context | Behavior |
|---------|----------|
| **DMs (chat 1:1)** | O Hermes responde a toda mensagem. Nenhuma `@mention` necessária. Cada DM tem sua própria sessão. |
| **Chats de grupo** | O Hermes responde quando você o `@mention`. Sem menção, o Hermes ignora a mensagem. |
| **Grupos compartilhados com vários usuários** | Por padrão, o Hermes isola o histórico de sessão por usuário dentro do grupo. Duas pessoas conversando no mesmo grupo não compartilham uma transcrição, a menos que você desabilite isso explicitamente. |

### Modelo de sessão no DingTalk {#session-model-in-dingtalk}

Por padrão:

- cada DM recebe sua própria sessão
- cada usuário em um chat de grupo compartilhado recebe sua própria sessão dentro desse grupo

Isso é controlado por `config.yaml`:

```yaml
group_sessions_per_user: true
```

Defina como `false` somente se quiser explicitamente uma conversa compartilhada para todo o grupo:

```yaml
group_sessions_per_user: false
```

Este guia percorre o processo completo de configuração — desde a criação do seu bot DingTalk até o envio da primeira mensagem.

## Pré-requisitos {#prerequisites}

Instale os pacotes Python necessários:

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[dingtalk]"
```

Ou individualmente:

```bash
pip install dingtalk-stream httpx alibabacloud-dingtalk
```

- `dingtalk-stream` — SDK oficial do DingTalk para Stream Mode (mensagens em tempo real baseadas em WebSocket)
- `httpx` — cliente HTTP assíncrono usado para enviar respostas via session webhooks
- `alibabacloud-dingtalk` — SDK DingTalk OpenAPI para AI Cards, reações emoji e downloads de mídia

## Passo 1: Crie um app DingTalk {#step-1-create-a-dingtalk-app}

1. Acesse o [DingTalk Developer Console](https://open-dev.dingtalk.com/).
2. Faça login com sua conta de administrador DingTalk.
3. Clique em **Application Development** → **Custom Apps** → **Create App via H5 Micro-App** (ou **Robot**, dependendo da versão do seu console).
4. Preencha:
   - **App Name**: ex., `Hermes Agent`
   - **Description**: opcional
5. Após criar, navegue até **Credentials & Basic Info** para encontrar seu **Client ID** (AppKey) e **Client Secret** (AppSecret). Copie ambos.

:::warning[Credentials shown only once]
O Client Secret é exibido apenas uma vez quando você cria o app. Se perdê-lo, precisará regenerá-lo. Nunca compartilhe essas credenciais publicamente ou as commite no Git.
:::

## Passo 2: Habilite a capacidade Robot {#step-2-enable-the-robot-capability}

1. Na página de configurações do seu app, vá em **Add Capability** → **Robot**.
2. Habilite a capacidade robot.
3. Em **Message Reception Mode**, selecione **Stream Mode** (recomendado — nenhuma URL pública necessária).

:::tip
Stream Mode é a configuração recomendada. Usa uma conexão WebSocket de longa duração iniciada da sua máquina, então você não precisa de IP público, nome de domínio ou endpoint webhook. Funciona atrás de NAT, firewalls e em máquinas locais.
:::

## Passo 3: Encontre seu DingTalk User ID {#step-3-find-your-dingtalk-user-id}

O Hermes Agent usa seu DingTalk User ID para controlar quem pode interagir com o bot. DingTalk User IDs são strings alfanuméricas definidas pelo administrador da sua organização.

Para encontrar o seu:

1. Peça ao administrador da organização DingTalk — User IDs são configurados no console de administração DingTalk em **Contacts** → **Members**.
2. Alternativamente, o bot registra o `sender_id` de cada mensagem recebida. Inicie o gateway, envie uma mensagem ao bot e verifique os logs para seu ID.

## Passo 4: Configure o Hermes Agent {#step-4-configure-hermes-agent}

### Opção A: Configuração interativa (recomendada)

Execute o comando de setup guiado:

```bash
hermes gateway setup
```

Selecione **DingTalk** quando solicitado. O assistente de configuração pode autorizar por um de dois caminhos:

- **Fluxo de dispositivo por QR code (recomendado).** Escaneie o QR impresso no terminal com o app móvel DingTalk — seu Client ID e Client Secret são retornados automaticamente e gravados em `~/.hermes/.env`. Nenhuma ida ao developer console necessária.
- **Colar manualmente.** Se você já tem credenciais (ou escanear QR não é conveniente), cole seu Client ID, Client Secret e IDs de usuário permitidos quando solicitado.

:::note openClaw branding disclosure
Como o `verification_uri_complete` do DingTalk está hardcoded para a identidade openClaw na camada de API, o QR atualmente autoriza sob uma string de origem `openClaw` até Alibaba / DingTalk-Real-AI registrar um template específico Hermes server-side. Isso é puramente como o DingTalk apresenta a tela de consentimento — o bot que você cria é totalmente seu e privado ao seu tenant.
:::

### Opção B: Configuração manual

Adicione o seguinte ao seu arquivo `~/.hermes/.env`:

```bash
# Required
DINGTALK_CLIENT_ID=your-app-key
DINGTALK_CLIENT_SECRET=your-app-secret

# Security: restrict who can interact with the bot
DINGTALK_ALLOWED_USERS=user-id-1

# Multiple allowed users (comma-separated)
# DINGTALK_ALLOWED_USERS=user-id-1,user-id-2

# Optional: group-chat gating (mirrors Slack/Telegram/Discord/WhatsApp)
# DINGTALK_REQUIRE_MENTION=true
# DINGTALK_FREE_RESPONSE_CHATS=cidABC==,cidDEF==
# DINGTALK_MENTION_PATTERNS=^小马
# DINGTALK_HOME_CHANNEL=cidXXXX==
# DINGTALK_ALLOW_ALL_USERS=true
```

Configurações opcionais de comportamento em `~/.hermes/config.yaml`:

```yaml
group_sessions_per_user: true

gateway:
  platforms:
    dingtalk:
      extra:
        # Require @mention in groups before the bot replies (parity with Slack/Telegram/Discord).
        # DMs ignore this — the bot always replies in 1:1 chats.
        require_mention: true

        # Per-platform allowlist. When set, only these DingTalk user IDs can interact with the bot
        # (same semantics as DINGTALK_ALLOWED_USERS, but scoped here instead of in .env).
        allowed_users:
          - user-id-1
          - user-id-2
```

- `group_sessions_per_user: true` mantém o contexto de cada participante isolado dentro de chats de grupo compartilhados
- `require_mention: true` impede que o bot responda a toda mensagem de grupo — só responde quando alguém o @-menciona
- `allowed_users` em `dingtalk.extra` é uma alternativa a `DINGTALK_ALLOWED_USERS`; defina um ou outro (se ambos estiverem definidos, apenas usuários presentes em ambas as listas são autorizados)

### Inicie o gateway

Depois de configurado, inicie o gateway DingTalk:

```bash
hermes gateway
```

O bot deve conectar ao Stream Mode do DingTalk em alguns segundos. Envie uma mensagem — DM ou em um grupo onde foi adicionado — para testar.

:::tip
Você pode executar `hermes gateway` em background ou como serviço systemd para operação persistente. Veja a documentação de deployment para detalhes.
:::

## Recursos {#features}

### AI Cards

O Hermes pode responder usando DingTalk AI Cards em vez de mensagens markdown simples. Cards oferecem exibição mais rica e estruturada e suportam atualizações streaming conforme o agente gera sua resposta.

Para habilitar AI Cards, configure um card template ID em `config.yaml`:

```yaml
platforms:
  dingtalk:
    enabled: true
    extra:
      card_template_id: "your-card-template-id"
```

Você encontra seu card template ID no DingTalk Developer Console nas configurações AI Card do seu app. Quando AI Cards estão habilitados, todas as respostas são enviadas como cards com atualizações de texto streaming.

### Reações emoji

O Hermes adiciona automaticamente reações emoji às suas mensagens para mostrar status de processamento:

- 🤔Thinking — adicionada quando o bot começa a processar sua mensagem
- 🥳Done — adicionada quando a resposta está completa (substitui a reação Thinking)

Essas reações funcionam em DMs e chats de grupo.

### Configurações de exibição

Você pode personalizar o comportamento de exibição do DingTalk independentemente de outras plataformas:

```yaml
display:
  platforms:
    dingtalk:
      show_reasoning: false   # Show model reasoning/thinking in replies
      streaming: true         # Enable streaming responses (works with AI Cards)
      tool_progress: all      # Show tool execution progress (all/new/off)
      interim_assistant_messages: true  # Show intermediate commentary messages
```

Para desabilitar progresso de ferramentas e mensagens intermediárias para uma experiência mais limpa:

```yaml
display:
  platforms:
    dingtalk:
      tool_progress: off
      interim_assistant_messages: false
```

## Solução de problemas {#troubleshooting}

### Bot não responde a mensagens

**Causa**: A capacidade robot não está habilitada, ou `DINGTALK_ALLOWED_USERS` não inclui seu User ID.

**Correção**: Verifique se a capacidade robot está habilitada nas configurações do app e se Stream Mode está selecionado. Confirme que seu User ID está em `DINGTALK_ALLOWED_USERS`. Reinicie o gateway.

### Erro "dingtalk-stream not installed"

**Causa**: O pacote Python `dingtalk-stream` não está instalado.

**Correção**: Instale-o:

```bash
pip install dingtalk-stream httpx
```

### "DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET required"

**Causa**: As credenciais não estão definidas no seu ambiente ou arquivo `.env`.

**Correção**: Verifique se `DINGTALK_CLIENT_ID` e `DINGTALK_CLIENT_SECRET` estão definidos corretamente em `~/.hermes/.env`. O Client ID é seu AppKey, e o Client Secret é seu AppSecret do DingTalk Developer Console.

### Stream desconecta / loops de reconexão

**Causa**: Instabilidade de rede, manutenção da plataforma DingTalk ou problemas de credenciais.

**Correção**: O adaptador reconecta automaticamente com backoff exponencial (2s → 5s → 10s → 30s → 60s). Verifique se suas credenciais são válidas e se seu app não foi desativado. Confirme que sua rede permite conexões WebSocket de saída.

### Bot está offline

**Causa**: O gateway Hermes não está rodando, ou falhou ao conectar.

**Correção**: Verifique se `hermes gateway` está rodando. Olhe a saída do terminal em busca de mensagens de erro. Problemas comuns: credenciais erradas, app desativado, `dingtalk-stream` ou `httpx` não instalados.

### "No session_webhook available"

**Causa**: O bot tentou responder mas não tem URL de session webhook. Isso tipicamente acontece se o webhook expirou ou o bot foi reiniciado entre receber a mensagem e enviar a resposta.

**Correção**: Envie uma nova mensagem ao bot — cada mensagem recebida fornece um session webhook novo para respostas. Essa é uma limitação normal do DingTalk; o bot só pode responder a mensagens que recebeu recentemente.

## Segurança {#security}

:::warning
Sempre defina `DINGTALK_ALLOWED_USERS` para restringir quem pode interagir com o bot. Sem isso, o gateway nega todos os usuários por padrão como medida de segurança. Adicione apenas User IDs de pessoas em quem você confia — usuários autorizados têm acesso completo às capacidades do agente, incluindo uso de ferramentas e acesso ao sistema.
:::

Para mais informações sobre proteger seu deployment Hermes Agent, veja o [Security Guide](../security.md).

## Notas {#notes}

- **Stream Mode**: Nenhuma URL pública, nome de domínio ou servidor webhook necessário. A conexão é iniciada da sua máquina via WebSocket, então funciona atrás de NAT e firewalls.
- **AI Cards**: Opcionalmente responda com AI Cards ricos em vez de markdown simples. Configure via `card_template_id`.
- **Reações emoji**: Reações automáticas 🤔Thinking/🥳Done para status de processamento.
- **Respostas Markdown**: Respostas são formatadas no formato markdown do DingTalk para exibição rich text.
- **Suporte a mídia**: Imagens e arquivos em mensagens recebidas são resolvidos automaticamente e podem ser processados por ferramentas de visão.
- **Deduplicação de mensagens**: O adaptador deduplica mensagens com janela de 5 minutos para evitar processar a mesma mensagem duas vezes.
- **Reconexão automática**: Se a conexão stream cair, o adaptador reconecta automaticamente com backoff exponencial.
- **Limite de tamanho de mensagem**: Respostas são limitadas a 20.000 caracteres por mensagem. Respostas mais longas são truncadas.
