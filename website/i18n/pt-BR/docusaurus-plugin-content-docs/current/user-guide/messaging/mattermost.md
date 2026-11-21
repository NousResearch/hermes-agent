---
sidebar_position: 8
title: "Mattermost"
description: "Configure o Hermes Agent como bot do Mattermost"
---

# Mattermost Setup

O Hermes Agent integra com o Mattermost como bot, permitindo que você converse com seu assistente de IA por mensagens diretas ou canais de equipe. O Mattermost é uma alternativa open source self-hosted ao Slack — você roda na sua própria infraestrutura, mantendo controle total dos seus dados. O bot conecta via REST API (v4) e WebSocket do Mattermost para eventos em tempo real, processa mensagens pelo pipeline Hermes Agent (incluindo uso de ferramentas, memória e raciocínio) e responde em tempo real. Suporta texto, anexos de arquivo, imagens e slash commands.

Nenhuma biblioteca externa Mattermost é necessária — o adapter usa `aiohttp`, que já é dependência do Hermes.

Antes da configuração, aqui está o que a maioria quer saber: como o Hermes se comporta depois de estar na sua instância Mattermost.

## How Hermes Behaves {#how-hermes-behaves}

| Context | Behavior |
|---------|----------|
| **DMs** | O Hermes responde a toda mensagem. Sem `@mention` necessária. Cada DM tem sua própria sessão. |
| **Canais públicos/privados** | O Hermes responde quando você o `@menciona`. Sem menção, o Hermes ignora a mensagem. |
| **Threads** | Se `MATTERMOST_REPLY_MODE=thread`, o Hermes responde em uma thread sob sua mensagem. O contexto da thread fica isolado do canal pai. |
| **Canais compartilhados com vários usuários** | Por padrão, o Hermes isola o histórico de sessão por usuário dentro do canal. Duas pessoas conversando no mesmo canal não compartilham uma transcrição, a menos que você desabilite isso explicitamente. |

:::tip
Se quiser que o Hermes responda como conversas em thread (aninhadas sob sua mensagem original), defina `MATTERMOST_REPLY_MODE=thread`. O padrão é `off`, que envia mensagens planas no canal.
:::

### Session Model in Mattermost

Por padrão:

- cada DM recebe sua própria sessão
- cada thread recebe seu próprio namespace de sessão
- cada usuário em um canal compartilhado recebe sua própria sessão dentro desse canal

Isso é controlado por `config.yaml`:

```yaml
group_sessions_per_user: true
```

Defina como `false` somente se quiser explicitamente uma conversa compartilhada para o canal inteiro:

```yaml
group_sessions_per_user: false
```

Sessões compartilhadas podem ser úteis para um canal colaborativo, mas também significam:

- usuários compartilham crescimento de contexto e custos de tokens
- uma tarefa longa e pesada em ferramentas de uma pessoa pode inflar o contexto de todos
- uma execução em andamento de uma pessoa pode interromper o follow-up de outra no mesmo canal

Este guia percorre o processo completo de configuração — da criação do bot no Mattermost ao envio da primeira mensagem.

## Step 1: Enable Bot Accounts {#step-1-enable-bot-accounts}

Contas de bot devem estar habilitadas no seu servidor Mattermost antes de criar uma.

1. Faça login no Mattermost como **System Admin**.
2. Vá em **System Console** → **Integrations** → **Bot Accounts**.
3. Defina **Enable Bot Account Creation** como **true**.
4. Clique em **Save**.

:::info
Se você não tem acesso System Admin, peça ao administrador Mattermost para habilitar contas de bot e criar uma para você.
:::

## Step 2: Create a Bot Account {#step-2-create-a-bot-account}

1. No Mattermost, clique no menu **☰** (canto superior esquerdo) → **Integrations** → **Bot Accounts**.
2. Clique em **Add Bot Account**.
3. Preencha os detalhes:
   - **Username**: ex.: `hermes`
   - **Display Name**: ex.: `Hermes Agent`
   - **Description**: opcional
   - **Role**: `Member` é suficiente
4. Clique em **Create Bot Account**.
5. O Mattermost exibirá o **bot token**. **Copie imediatamente.**

:::warning[Token shown only once]
O bot token só é exibido uma vez quando você cria a conta de bot. Se perder, precisará regenerá-lo nas configurações da conta de bot. Nunca compartilhe seu token publicamente nem faça commit no Git — quem tiver este token tem controle total do bot.
:::

Guarde o token em local seguro (gerenciador de senhas, por exemplo). Você vai precisar dele no Passo 5.

:::tip
Você também pode usar um **personal access token** em vez de conta de bot. Vá em **Profile** → **Security** → **Personal Access Tokens** → **Create Token**. Isso é útil se quiser que o Hermes poste como seu próprio usuário em vez de um usuário bot separado.
:::

## Step 3: Add the Bot to Channels {#step-3-add-the-bot-to-channels}

O bot precisa ser membro de qualquer canal onde você quer que responda:

1. Abra o canal desejado.
2. Clique no nome do canal → **Add Members**.
3. Busque o username do bot (ex.: `hermes`) e adicione.

Para DMs, basta abrir mensagem direta com o bot — ele poderá responder imediatamente.

## Step 4: Find Your Mattermost User ID {#step-4-find-your-mattermost-user-id}

O Hermes Agent usa seu Mattermost User ID para controlar quem pode interagir com o bot. Para encontrá-lo:

1. Clique no seu **avatar** (canto superior esquerdo) → **Profile**.
2. Seu User ID aparece no diálogo de perfil — clique para copiar.

Seu User ID é uma string alfanumérica de 26 caracteres como `3uo8dkh1p7g1mfk49ear5fzs5c`.

:::warning
Seu User ID **não** é seu username. O username é o que aparece após `@` (ex.: `@alice`). O User ID é um identificador alfanumérico longo que o Mattermost usa internamente.
:::

**Alternativa**: você também pode obter seu User ID via API:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-mattermost-server/api/v4/users/me | jq .id
```

:::tip
Para obter um **Channel ID**: clique no nome do canal → **View Info**. O Channel ID aparece no painel de informações. Você precisará dele se quiser definir um canal home manualmente.
:::

## Step 5: Configure Hermes Agent {#step-5-configure-hermes-agent}

### Option A: Interactive Setup (Recommended)

Execute o comando de setup guiado:

```bash
hermes gateway setup
```

Selecione **Mattermost** quando solicitado, depois cole a URL do servidor, bot token e user ID quando pedido.

### Option B: Manual Configuration

Adicione o seguinte em `~/.hermes/.env`:

```bash
# Obrigatório
MATTERMOST_URL=https://mm.example.com
MATTERMOST_TOKEN=***
MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c

# Vários usuários permitidos (separados por vírgula)
# MATTERMOST_ALLOWED_USERS=3uo8dkh1p7g1mfk49ear5fzs5c,8fk2jd9s0a7bncm1xqw4tp6r3e

# Opcional: modo de resposta (thread ou off, padrão: off)
# MATTERMOST_REPLY_MODE=thread

# Opcional: responder sem @mention (padrão: true = exigir menção)
# MATTERMOST_REQUIRE_MENTION=false

# Opcional: canais onde o bot responde sem @mention (IDs de canal separados por vírgula)
# MATTERMOST_FREE_RESPONSE_CHANNELS=channel_id_1,channel_id_2
```

Configurações opcionais de comportamento em `~/.hermes/config.yaml`:

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true` mantém o contexto de cada participante isolado dentro de canais e threads compartilhados

### Start the Gateway

Após configurar, inicie o gateway Mattermost:

```bash
hermes gateway
```

O bot deve conectar ao seu servidor Mattermost em alguns segundos. Envie uma mensagem — DM ou canal onde foi adicionado — para testar.

:::tip
Você pode rodar `hermes gateway` em background ou como serviço systemd para operação persistente. Veja a documentação de deployment para detalhes.
:::

## Home Channel {#home-channel}

Você pode designar um "canal home" onde o bot envia mensagens proativas (como saída de cron jobs, lembretes e notificações). Há duas formas:

### Using the Slash Command

Digite `/sethome` em qualquer canal Mattermost onde o bot está presente. Esse canal vira o canal home.

### Manual Configuration

Adicione em `~/.hermes/.env`:

```bash
MATTERMOST_HOME_CHANNEL=abc123def456ghi789jkl012mn
```

Substitua o ID pelo channel ID real (clique no nome do canal → View Info → copie o ID).

## Reply Mode {#reply-mode}

A configuração `MATTERMOST_REPLY_MODE` controla como o Hermes publica respostas:

| Mode | Behavior |
|------|----------|
| `off` (padrão) | O Hermes publica mensagens planas no canal, como um usuário normal. |
| `thread` | O Hermes responde em uma thread sob sua mensagem original. Mantém canais limpos quando há muito vai-e-vem. |

Defina em `~/.hermes/.env`:

```bash
MATTERMOST_REPLY_MODE=thread
```

## Mention Behavior {#mention-behavior}

Por padrão, o bot só responde em canais quando `@mencionado`. Você pode alterar isso:

| Variable | Default | Description |
|----------|---------|-------------|
| `MATTERMOST_REQUIRE_MENTION` | `true` | Defina `false` para responder a todas as mensagens em canais (DMs sempre funcionam). |
| `MATTERMOST_FREE_RESPONSE_CHANNELS` | _(nenhum)_ | IDs de canal separados por vírgula onde o bot responde sem `@mention`, mesmo com require_mention true. |

Para encontrar um channel ID no Mattermost: abra o canal, clique no cabeçalho do nome do canal e procure o ID na URL ou detalhes do canal.

Quando o bot é `@mencionado`, a menção é removida automaticamente da mensagem antes do processamento.

## Channel allowlist (`allowed_channels`) {#channel-allowlist-allowed_channels}

Restringe o bot a um conjunto fixo de canais Mattermost. Quando definido, o bot **só** responde em canais cujo ID está na lista — mensagens de qualquer outro canal são ignoradas silenciosamente, mesmo se o bot for `@mencionado`.

**DMs são isentas** deste filtro, então usuários autorizados sempre podem alcançar o bot em mensagem direta.

```yaml
mattermost:
  allowed_channels:
    - "abc123def456ghi789jkl012mno"   # #ops
    - "xyz987uvw654rst321opq098nml"   # #incident-response
```

Ou via variável de ambiente (separados por vírgula):

```bash
MATTERMOST_ALLOWED_CHANNELS="abc123def456ghi789jkl012mno,xyz987uvw654rst321opq098nml"
```

Comportamento:

- Vazio / não definido → sem restrição (totalmente retrocompatível).
- Não vazio → o channel ID deve estar na lista, ou a mensagem é descartada antes de qualquer outro gate (exigência de menção, `MATTERMOST_FREE_RESPONSE_CHANNELS`, etc.).
- Encontre um channel ID via UI Mattermost → cabeçalho do canal → "View Info", ou leia da URL do canal.

Veja também: [admin/user slash command split](../../reference/slash-commands.md#permissions-and-adminuser-split).

## Troubleshooting {#troubleshooting}

### Bot is not responding to messages

**Cause**: O bot não é membro do canal, ou `MATTERMOST_ALLOWED_USERS` não inclui seu User ID.

**Fix**: Adicione o bot ao canal (nome do canal → Add Members → busque o bot). Verifique se seu User ID está em `MATTERMOST_ALLOWED_USERS`. Reinicie o gateway.

### 403 Forbidden errors

**Cause**: O bot token é inválido, ou o bot não tem permissão para postar no canal.

**Fix**: Verifique se `MATTERMOST_TOKEN` em `.env` está correto. Confirme que a conta de bot não foi desativada. Verifique se o bot foi adicionado ao canal. Se usar personal access token, garanta que sua conta tem as permissões necessárias.

### WebSocket disconnects / reconnection loops

**Cause**: Instabilidade de rede, reinícios do servidor Mattermost ou problemas de firewall/proxy com conexões WebSocket.

**Fix**: O adapter reconecta automaticamente com backoff exponencial (2s → 60s). Verifique a config WebSocket do servidor — reverse proxies (nginx, Apache) precisam de headers de upgrade WebSocket. Confirme que nenhum firewall bloqueia conexões WebSocket no servidor Mattermost.

Para nginx, garanta que sua config inclui:

```nginx
location /api/v4/websocket {
    proxy_pass http://mattermost-backend;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 600s;
}
```

### "Failed to authenticate" on startup

**Cause**: O token ou URL do servidor está incorreto.

**Fix**: Verifique se `MATTERMOST_URL` aponta para seu servidor Mattermost (inclua `https://`, sem barra final). Confira se `MATTERMOST_TOKEN` é válido — teste com curl:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/api/v4/users/me
```

Se retornar info do usuário bot, o token é válido. Se retornar erro, regenere o token.

### Bot is offline

**Cause**: O gateway Hermes não está rodando, ou falhou ao conectar.

**Fix**: Verifique se `hermes gateway` está rodando. Veja a saída do terminal em busca de erros. Problemas comuns: URL errada, token expirado, servidor Mattermost inacessível.

### "User not allowed" / Bot ignores you

**Cause**: Seu User ID não está em `MATTERMOST_ALLOWED_USERS`.

**Fix**: Adicione seu User ID em `MATTERMOST_ALLOWED_USERS` em `~/.hermes/.env` e reinicie o gateway. Lembre-se: o User ID é uma string alfanumérica de 26 caracteres, não seu `@username`.

## Per-Channel Prompts {#per-channel-prompts}

Atribua system prompts efêmeros a canais Mattermost específicos. O prompt é injetado em runtime a cada turno — nunca persistido no histórico da transcrição — então mudanças entram em vigor imediatamente.

```yaml
mattermost:
  channel_prompts:
    "channel_id_abc123": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "channel_id_def456": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

As chaves são Mattermost channel IDs (encontre na URL do canal ou via API). Todas as mensagens no canal correspondente recebem o prompt injetado como instrução de sistema efêmera.

## Security {#security}

:::warning
Sempre defina `MATTERMOST_ALLOWED_USERS` para restringir quem pode interagir com o bot. Sem isso, o gateway nega todos os usuários por padrão como medida de segurança. Adicione apenas User IDs de pessoas em quem confia — usuários autorizados têm acesso total às capacidades do agente, incluindo uso de ferramentas e acesso ao sistema.
:::

Para mais informações sobre proteger seu deployment Hermes Agent, veja o [Security Guide](../security.md).

## Notes {#notes}

- **Self-hosted friendly**: Funciona com qualquer instância Mattermost self-hosted. Não requer conta Mattermost Cloud nem assinatura.
- **Sem dependências extras**: O adapter usa `aiohttp` para HTTP e WebSocket, já incluído com o Hermes Agent.
- **Compatível com Team Edition**: Funciona com Mattermost Team Edition (grátis) e Enterprise Edition.
