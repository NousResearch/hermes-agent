---
sidebar_position: 9
title: "Matrix"
description: "Configure o Hermes Agent como bot Matrix"
---

# Configuração do Matrix {#matrix-setup}

O Hermes Agent integra com Matrix, o protocolo de mensagens aberto e federado. O Matrix permite que você rode seu próprio homeserver ou use um público como matrix.org — de qualquer forma, você mantém controle das suas comunicações. O bot conecta via SDK Python `mautrix`, processa mensagens pelo pipeline do Hermes Agent (incluindo uso de ferramentas, memória e raciocínio) e responde em tempo real. Suporta texto, anexos de arquivo, imagens, áudio, vídeo e criptografia ponta a ponta opcional (E2EE).

O Hermes funciona com qualquer homeserver Matrix — Synapse, Conduit, Dendrite ou matrix.org.

Antes da configuração, aqui está o que a maioria das pessoas quer saber: como o Hermes se comporta depois de conectado.

## Como o Hermes se comporta {#how-hermes-behaves}

| Contexto | Comportamento |
|---------|----------|
| **DMs** | O Hermes responde a toda mensagem. Não é necessário `@mention`. Cada DM tem sua própria sessão. Defina `MATRIX_DM_MENTION_THREADS=true` para iniciar uma thread quando o bot for `@mentioned` em um DM. |
| **Salas** | Por padrão, o Hermes exige um `@mention` para responder. Defina `MATRIX_REQUIRE_MENTION=false` ou adicione IDs de sala em `MATRIX_FREE_RESPONSE_ROOMS` para salas de resposta livre. Convites para salas são aceitos automaticamente. |
| **Threads** | O Hermes suporta threads do Matrix (MSC3440). Se você responder em uma thread, o Hermes mantém o contexto da thread isolado da timeline principal da sala. Threads em que o bot já participou não exigem menção. |
| **Auto-threading** | Por padrão, o Hermes cria automaticamente uma thread para cada mensagem à qual responde em uma sala. Isso mantém conversas isoladas. Defina `MATRIX_AUTO_THREAD=false` para desativar. Defina `MATRIX_DM_AUTO_THREAD=true` (padrão false) para também criar threads automaticamente para mensagens de DM — isso é distinto de `MATRIX_DM_MENTION_THREADS`, que só inicia uma thread quando o bot é `@mentioned` em um DM. |
| **Comandos** | O Hermes aceita `/commands` normais quando seu cliente Matrix os envia. Se seu cliente reserva `/` para comandos locais, use `!commands` em vez disso; o Hermes normaliza aliases `!command` conhecidos para `/command`. |
| **Controles interativos** | Aprovação de comandos perigosos e seleção via `/model` podem usar reações do Matrix. Reações de aprovação podem ser limitadas ao usuário que solicitou a ação. |
| **Thinking e atividade de ferramentas** | O Matrix usa painéis editáveis de thinking/atividade de ferramentas em threads quando o progresso do gateway está habilitado, para que atualizações não inundem a timeline principal da sala. |
| **Salas compartilhadas com vários usuários** | Por padrão, o Hermes isola o histórico de sessão por usuário dentro da sala. Duas pessoas conversando na mesma sala não compartilham uma transcrição, a menos que você desative isso explicitamente. |

:::tip
O bot entra automaticamente em salas quando convidado. Basta convidar o usuário Matrix do bot para qualquer sala e ele entrará e começará a responder.
:::

## Matriz de capacidades {#capability-matrix}

Esta tabela é respaldada pela declaração de capacidades do adaptador Matrix e pela cobertura de testes do Matrix. E2EE é baseado em modo porque implantações escolhem se salas criptografadas ficam desabilitadas, oportunistas ou obrigatórias.

| Capacidade | Matrix |
|------------|--------|
| text | yes |
| threads | yes |
| reactions | yes |
| approvals | yes |
| model picker | yes |
| thinking panes | yes |
| images | yes |
| multiple images | yes |
| files | yes |
| voice/audio | yes |
| video | yes |
| E2EE | off / optional / required |
| diagnostics | yes |

### Modelo de sessão no Matrix

Por padrão:

- cada DM recebe sua própria sessão
- cada thread recebe seu próprio namespace de sessão
- cada usuário em uma sala compartilhada recebe sua própria sessão dentro dessa sala

Isso é controlado por `config.yaml`:

```yaml
group_sessions_per_user: true
```

Defina como `false` somente se você quiser explicitamente uma conversa compartilhada para a sala inteira:

```yaml
group_sessions_per_user: false
```

Sessões compartilhadas podem ser úteis para uma sala colaborativa, mas também significam:

- usuários compartilham crescimento de contexto e custos de tokens
- uma tarefa longa e pesada em ferramentas de uma pessoa pode inflar o contexto de todos os outros
- uma execução em andamento de uma pessoa pode interromper o follow-up de outra pessoa na mesma sala

### Configuração de menções e threads

Você pode configurar o comportamento de menções e auto-threading via variáveis de ambiente ou `config.yaml`:

```yaml
matrix:
  require_mention: true           # Require @mention in rooms (default: true)
  allowed_users:                  # Matrix users allowed to trigger agent turns
    - "@alice:matrix.org"
  allowed_rooms:                  # Matrix rooms allowed to trigger agent turns
    - "!abc123:matrix.org"
  free_response_rooms:            # Rooms exempt from mention requirement
    - "!abc123:matrix.org"
  ignore_user_patterns:           # Bridge/appservice ghost users to ignore
    - "^@telegram_"
    - "^@whatsapp_"
  process_notices: false          # Ignore m.notice by default
  session_scope: room             # auto|room|thread; room is recommended for project rooms
  auto_thread: true               # Auto-create threads for responses (default: true)
  dm_mention_threads: false       # Create thread when @mentioned in DM (default: false)
  max_message_length: 16000       # Outbound chunk size in chars (default: 16000, max: 65535)
```

Ou via variáveis de ambiente:

```bash
MATRIX_REQUIRE_MENTION=true
MATRIX_ALLOWED_USERS=@alice:matrix.org
MATRIX_ALLOWED_ROOMS=!abc123:matrix.org
MATRIX_FREE_RESPONSE_ROOMS=!abc123:matrix.org,!def456:matrix.org
MATRIX_IGNORE_USER_PATTERNS='^@telegram_,^@whatsapp_'
MATRIX_PROCESS_NOTICES=false
MATRIX_SESSION_SCOPE=room       # recommended for stable project-room context
MATRIX_AUTO_THREAD=true
MATRIX_DM_MENTION_THREADS=false
MATRIX_REACTIONS=true          # default: true — emoji reactions during processing
MATRIX_ALLOW_ROOM_MENTIONS=false
```

:::tip Desativando reações
`MATRIX_REACTIONS=false` desativa as reações emoji do ciclo de vida de processamento (👀/✅/❌) que o bot publica em mensagens recebidas. Útil em salas onde eventos de reação são barulhentos ou não são suportados por todos os clientes participantes.
:::

:::tip Menções para a sala inteira
O Hermes envia menções estruturadas de usuários Matrix para IDs Matrix explícitos como `@alice:example.org`. Notificações `@room` para a sala inteira ficam desabilitadas por padrão; defina `MATRIX_ALLOW_ROOM_MENTIONS=true` somente em salas onde o bot pode notificar todos.
:::

:::note
Se você está atualizando de uma versão que não tinha `MATRIX_REQUIRE_MENTION`, o bot respondia anteriormente a todas as mensagens em salas. Para preservar esse comportamento, defina `MATRIX_REQUIRE_MENTION=false`.
:::

### Isolamento de salas de projeto

Se você usa o mesmo bot Matrix em várias salas de projeto, configure sessões estáveis com escopo de sala:

```bash
MATRIX_SESSION_SCOPE=room
MATRIX_AUTO_THREAD=false
```

`MATRIX_SESSION_SCOPE` aceita:

| Escopo | Comportamento |
|-------|----------|
| `auto` | Padrão retrocompatível. O comportamento existente de `MATRIX_AUTO_THREAD` controla threads sintéticas. |
| `room` | Mensagens de sala sem thread permanecem em uma sessão estável de sala. Threads Matrix reais ainda usam sua raiz de thread. |
| `thread` | Mensagens de sala sem thread sintetizam uma thread/sessão a partir do ID do evento que as disparou. |

O Hermes agora inclui o nome atual da sala Matrix, ID da sala, tópico, ID da mensagem e uma nota de limite de sala Matrix no prompt do agente. `/status` também mostra o escopo atual de sala/sessão Matrix, e `/resume` não retomará silenciosamente uma sessão nomeada de outra sala Matrix, a menos que você use explicitamente `/resume --cross-room <session name>`.

`MATRIX_SESSION_SCOPE=room` controla a faixa sala/thread. A configuração existente `group_sessions_per_user` ainda controla se usuários dentro dessa sala compartilham a faixa. Com `group_sessions_per_user: true` (padrão), Alice e Bob recebem sessões separadas do Projeto B. Com `group_sessions_per_user: false`, a sala tem uma transcrição compartilhada do Projeto B.

Este guia percorre o processo completo de configuração — desde a criação da conta do bot até o envio da sua primeira mensagem.

## Passo 1: Crie uma conta de bot {#step-1-create-a-bot-account}

Você precisa de uma conta de usuário Matrix para o bot. Há várias formas de fazer isso:

### Opção A: Registre no seu homeserver (recomendado)

Se você roda seu próprio homeserver (Synapse, Conduit, Dendrite):

1. Use a API de admin ou ferramenta de registro para criar um novo usuário:

```bash
# Synapse example
register_new_matrix_user -c /etc/synapse/homeserver.yaml http://localhost:8008
```

2. Escolha um nome de usuário como `hermes` — o ID completo do usuário será `@hermes:your-server.org`.

### Opção B: Use matrix.org ou outro homeserver público

1. Acesse [Element Web](https://app.element.io) e crie uma nova conta.
2. Escolha um nome de usuário para seu bot (ex.: `hermes-bot`).

### Opção C: Use sua própria conta

Você também pode rodar o Hermes como seu próprio usuário. Isso significa que o bot publica como você — útil para assistentes pessoais.

## Passo 2: Obtenha um access token {#step-2-get-an-access-token}

O Hermes precisa de um access token para autenticar com o homeserver. Você tem duas opções:

### Opção A: Access token (recomendado)

A forma mais confiável de obter um token:

**Via Element:**
1. Faça login no [Element](https://app.element.io) com a conta do bot.
2. Vá em **Settings** → **Help & About**.
3. Role para baixo e expanda **Advanced** — o access token é exibido ali.
4. **Copie imediatamente.**

**Via a API:**

```bash
curl -X POST https://your-server/_matrix/client/v3/login \
  -H "Content-Type: application/json" \
  -d '{
    "type": "m.login.password",
    "user": "@hermes:your-server.org",
    "password": "your-password"
  }'
```

A resposta inclui um campo `access_token` — copie-o.

:::warning[Mantenha seu access token seguro]
O access token dá acesso total à conta Matrix do bot. Nunca compartilhe publicamente nem faça commit no Git. Se comprometido, revogue fazendo logout de todas as sessões desse usuário.
:::

### Opção B: Login com senha

Em vez de fornecer um access token, você pode dar ao Hermes o user ID e a senha do bot. O Hermes fará login automaticamente na inicialização. Isso é mais simples, mas significa que a senha fica armazenada no seu arquivo `.env`.

```bash
MATRIX_USER_ID=@hermes:your-server.org
MATRIX_PASSWORD=your-password
```

## Passo 3: Encontre seu Matrix User ID {#step-3-find-your-matrix-user-id}

O Hermes Agent usa seu Matrix User ID para controlar quem pode interagir com o bot. Matrix User IDs seguem o formato `@username:server`.

Para encontrar o seu:

1. Abra o [Element](https://app.element.io) (ou seu cliente Matrix preferido).
2. Clique no seu avatar → **Settings**.
3. Seu User ID é exibido no topo do perfil (ex.: `@alice:matrix.org`).

:::tip
Matrix User IDs sempre começam com `@` e contêm um `:` seguido do nome do servidor. Por exemplo: `@alice:matrix.org`, `@bob:your-server.com`.
:::

## Passo 4: Configure o Hermes Agent {#step-4-configure-hermes-agent}

### Opção A: Configuração interativa (recomendado)

Execute o comando de configuração guiada:

```bash
hermes gateway setup
```

Selecione **Matrix** quando solicitado, depois forneça a URL do homeserver, access token (ou user ID + senha) e IDs de usuários permitidos quando perguntado.

### Opção B: Configuração manual

Adicione o seguinte ao seu arquivo `~/.hermes/.env`:

**Usando access token:**

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_ACCESS_TOKEN=***

# Optional: user ID (auto-detected from token if omitted)
# MATRIX_USER_ID=@hermes:matrix.example.org

# Security: restrict who can interact with the bot
MATRIX_ALLOWED_USERS=@alice:matrix.example.org

# Optional: restrict which rooms can trigger the bot
MATRIX_ALLOWED_ROOMS=!abc123:matrix.example.org

# Multiple allowed users (comma-separated)
# MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
```

**Usando login com senha:**

```bash
# Required
MATRIX_HOMESERVER=https://matrix.example.org
MATRIX_USER_ID=@hermes:matrix.example.org
MATRIX_PASSWORD=***

# Security
MATRIX_ALLOWED_USERS=@alice:matrix.example.org
```

## Endurecimento para implantação privada {#private-deployment-hardening}

Para implantações Matrix privadas, defina listas de permissão de usuários e salas. Se `MATRIX_ALLOWED_USERS` não estiver definido, qualquer remetente que consiga alcançar o bot em uma sala em que ele entrou pode disparar um turno do agente. Se `MATRIX_ALLOWED_ROOMS` não estiver definido, qualquer sala em que o bot entrar pode disparar um turno do agente. Uma implantação restrita deve definir ambos:

```bash
MATRIX_ALLOWED_USERS=@alice:matrix.example.org,@bob:matrix.example.org
MATRIX_ALLOWED_ROOMS=!ops:matrix.example.org,!dmroom:matrix.example.org
```

Implantações de bridge e appservice precisam de proteção extra contra loops. O Hermes sempre ignora seus próprios eventos, usuários no estilo appservice Matrix cujo localpart começa com `_`, IDs de evento duplicados, eventos antigos de startup, eventos de substituição de edição e eventos `m.notice` por padrão. Adicione padrões de ghost de bridge específicos da implantação quando sua bridge usa uma convenção de nomenclatura diferente:

```bash
MATRIX_IGNORE_USER_PATTERNS='^@telegram_,^@slack_,^@whatsapp_'
```

Habilite notices somente quando um fluxo humano confiável realmente envia `m.notice`:

```bash
MATRIX_PROCESS_NOTICES=true
```

Notificações de sala inteira de saída ficam desabilitadas por padrão. Mantenha `MATRIX_ALLOW_ROOM_MENTIONS=false`, a menos que o bot tenha permissão explícita para acordar a sala inteira com `@room`.

Diagnósticos e payloads de debug redigem access tokens Matrix, chaves de recuperação, identificadores de dispositivo e corpos de mensagem. Downloads de mídia são limitados a URIs de conteúdo Matrix `mxc://` e rejeitados quando excedem `MATRIX_MAX_MEDIA_BYTES`. Trate salas federadas e homeservers não confiáveis como entrada não confiável: mantenha listas de permissão de salas restritas, prefira DMs ou salas privadas para trabalho pesado em ferramentas e evite autorizar ghosts de bridge ou puppets de appservice como usuários permitidos.

Configurações opcionais de comportamento em `~/.hermes/config.yaml`:

```yaml
group_sessions_per_user: true
```

- `group_sessions_per_user: true` mantém o contexto de cada participante isolado dentro de salas compartilhadas

### Inicie o gateway

Depois de configurado, inicie o gateway Matrix:

```bash
hermes gateway
```

O bot deve conectar ao seu homeserver e começar a sincronizar em alguns segundos. Envie uma mensagem — seja um DM ou em uma sala em que ele entrou — para testar.

:::tip
Você pode rodar `hermes gateway` em segundo plano ou como serviço systemd para operação persistente. Veja a documentação de implantação para detalhes.
:::

## Criptografia ponta a ponta (E2EE) {#end-to-end-encryption-e2ee}

O Hermes suporta criptografia ponta a ponta do Matrix, para que você possa conversar com seu bot em salas criptografadas.

### Requisitos

E2EE exige a biblioteca `mautrix` com extras de criptografia e a biblioteca C `libolm`:

```bash
# Install mautrix with E2EE support
pip install 'mautrix[encryption]'

# Or install with hermes extras
cd ~/.hermes/hermes-agent && uv pip install -e ".[matrix]"
```

Você também precisa de `libolm` instalado no seu sistema:

```bash
# Debian/Ubuntu
sudo apt install libolm-dev

# macOS
brew install libolm

# Fedora
sudo dnf install libolm-devel
```

### Habilite E2EE

Adicione ao seu `~/.hermes/.env`:

```bash
MATRIX_E2EE_MODE=required
```

`MATRIX_E2EE_MODE` aceita:

| Modo | Comportamento |
|------|----------|
| `off` | Do not initialize Matrix E2EE. |
| `optional` | Try E2EE when dependencies are available, but keep unencrypted rooms working if crypto cannot initialize. |
| `required` | Fail closed if E2EE dependencies or crypto setup are not available. |

O modo optional pode recuar para operação sem E2EE quando a configuração de crypto não está disponível. O modo required falha fechado em vez de rebaixar silenciosamente.

Por retrocompatibilidade, `MATRIX_ENCRYPTION=true` ainda habilita o comportamento E2EE required.

Quando E2EE está habilitado, o Hermes:

- Armazena chaves de criptografia em `~/.hermes/platforms/matrix/store/` (instalações legadas: `~/.hermes/matrix/store/`)
- Faz upload de chaves de dispositivo na primeira conexão
- Descriptografa mensagens recebidas e criptografa mensagens enviadas automaticamente
- Entra automaticamente em salas criptografadas quando convidado

### Ferramentas e controles Matrix

Em conversas Matrix, o Hermes expõe ferramentas específicas do Matrix ao agente:

- `matrix_send_reaction`
- `matrix_redact_message`
- `matrix_create_room`
- `matrix_invite_user`
- `matrix_fetch_history`
- `matrix_set_presence`

Essas ferramentas têm escopo de contextos Matrix e não estão disponíveis em toolsets que não sejam Matrix. Ferramentas no estilo admin ficam desabilitadas por padrão: redação exige `MATRIX_TOOLS_ALLOW_REDACTION=true`, convites exigem `MATRIX_TOOLS_ALLOW_INVITES=true` e criação de sala exige `MATRIX_TOOLS_ALLOW_ROOM_CREATE=true`. Criação de sala pública também exige `MATRIX_ALLOW_PUBLIC_ROOMS=true`.
Ferramentas Matrix são limitadas à sala Matrix atual por padrão. Alvos explícitos entre salas exigem `MATRIX_TOOLS_ALLOW_CROSS_ROOM=true`; redação e ações entre salas semelhantes a convite exigem adicionalmente `MATRIX_TOOLS_ALLOW_CROSS_ROOM_DESTRUCTIVE=true`. Se `MATRIX_ALLOWED_ROOMS` estiver definido, ferramentas Matrix só podem mirar nessas salas.

Controles de reação usam:

- ✅ aprovar uma vez
- ♾️ aprovar sempre
- ❌ negar
- reações numéricas para escolhas de `/model`

Defina `MATRIX_APPROVAL_REQUIRE_SENDER=false` se você quiser intencionalmente que qualquer usuário Matrix autorizado na sala opere um prompt de aprovação/seletor de modelo. O padrão é vinculado ao solicitante quando o Hermes sabe quem solicitou a ação.

### Limites de mídia

O Hermes faz upload e download de imagens, arquivos, áudio e vídeo Matrix pelas APIs de mídia do Matrix. Várias imagens geradas são enviadas como um lote lógico ordenado, preservando legendas e contexto de thread ao longo do lote.

Por padrão, mídia Matrix acima de 100 MB é rejeitada antes de upload/download. Sobrescreva com:

```bash
MATRIX_MAX_MEDIA_BYTES=104857600
```

Mídia recebida deve usar URIs de conteúdo Matrix `mxc://`. O Hermes rejeita URLs de mídia HTTP(S) arbitrárias em eventos Matrix para evitar transformar uma sala federada em um downloader irrestrito.

## Testes de integração Synapse {#synapse-integration-tests}

O Hermes inclui um harness Synapse opt-in para validação local:

```bash
docker compose -f tests/e2e/matrix_synapse_gateway/docker-compose.yml up -d
HERMES_MATRIX_SYNAPSE_INTEGRATION=1 \
  scripts/run_tests.sh -m "integration and matrix_synapse" \
  tests/e2e/matrix_synapse_gateway/test_gateway.py
docker compose -f tests/e2e/matrix_synapse_gateway/docker-compose.yml down -v
```

O harness cria usuários temporários via registro shared-secret do Synapse e cobre envio/recebimento em sala privada, convite/entrada em sala nomeada, upload/download de mídia, entrega de resposta do bot e filtragem de eventos antigos de startup. Cobertura smoke de E2EE é marcada separadamente com `matrix_e2ee` para permanecer opt-in em máquinas de desenvolvimento.

### Verificação de cross-signing (recomendado)

Se sua conta Matrix tem cross-signing habilitado (o padrão no Element), defina a chave de recuperação para que o bot possa auto-assinar seu dispositivo na inicialização. Sem isso, outros clientes Matrix podem recusar compartilhar sessões de criptografia com o bot após uma rotação de chave de dispositivo.

```bash
MATRIX_RECOVERY_KEY=EsT... your recovery key here
```

**Onde encontrar:** No Element, vá em **Settings** → **Security & Privacy** → **Encryption** → sua chave de recuperação (também chamada de "Security Key"). Esta é a chave que você foi solicitado a salvar quando configurou cross-signing pela primeira vez.

A cada inicialização, se `MATRIX_RECOVERY_KEY` estiver definido, o Hermes importa chaves de cross-signing do armazenamento seguro de segredos do homeserver e assina o dispositivo atual. Isso é idempotente e seguro de deixar habilitado permanentemente.

Se o Hermes inicializar uma nova chave de recuperação Matrix, ele nunca registra a chave bruta. Defina `MATRIX_RECOVERY_KEY_OUTPUT_FILE=/secure/path/matrix-recovery-key.txt` antes da inicialização para escrever uma chave gerada uma vez com modo de arquivo `0600`; o arquivo não é sobrescrito se já existir.

:::warning[Excluindo o crypto store]
Se você excluir `~/.hermes/platforms/matrix/store/crypto.db`, o bot perde sua identidade de criptografia. Simplesmente reiniciar com o mesmo device ID **não** recuperará completamente — o homeserver ainda mantém one-time keys assinadas com a identity key antiga, e peers não conseguem estabelecer novas sessões Olm.

O Hermes detecta essa condição na inicialização e recusa habilitar E2EE, registrando: `device XXXX has stale one-time keys on the server signed with a previous identity key`.

**Recuperação mais fácil: gere um novo access token** (que obtém um device ID novo sem histórico de chaves obsoletas). Veja a seção "Upgrading from a previous version with E2EE" abaixo. Este é o caminho mais confiável e evita tocar no banco de dados do homeserver.

**Recuperação manual** (avançado — mantém o mesmo device ID):

1. Pare o Synapse e exclua o dispositivo antigo do banco de dados:
   ```bash
   sudo systemctl stop matrix-synapse
   sudo sqlite3 /var/lib/matrix-synapse/homeserver.db "
     DELETE FROM e2e_device_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_one_time_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM e2e_fallback_keys_json WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
     DELETE FROM devices WHERE device_id = 'DEVICE_ID' AND user_id = '@hermes:your-server';
   "
   sudo systemctl start matrix-synapse
   ```
   Ou via a API admin do Synapse (note o user ID codificado na URL):
   ```bash
   curl -X DELETE -H "Authorization: Bearer ADMIN_TOKEN" \
     'https://your-server/_synapse/admin/v2/users/%40hermes%3Ayour-server/devices/DEVICE_ID'
   ```
   Nota: excluir um dispositivo via API admin também pode invalidar o access token associado. Você pode precisar gerar um novo token depois.

2. Exclua o crypto store local e reinicie o Hermes:
   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db*
   # restart hermes
   ```

Outros clientes Matrix (Element, matrix-commander) podem cachear as chaves antigas do dispositivo. Após a recuperação, digite `/discardsession` no Element para forçar uma nova sessão de criptografia com o bot.
:::

:::info
Se `mautrix[encryption]` não estiver instalado ou `libolm` estiver ausente, o bot recua automaticamente para um cliente simples (não criptografado). Você verá um aviso nos logs.
:::

## Sala home {#home-room}

Você pode designar uma "home room" onde o bot envia mensagens proativas (como saída de cron jobs, lembretes e notificações). Há duas formas de definir:

### Usando o comando slash

Digite `/sethome` em qualquer sala Matrix onde o bot esteja presente. Essa sala se torna a home room.
Se seu cliente Matrix intercepta comandos slash, digite `!sethome` em vez disso.

### Configuração manual

Adicione isto ao seu `~/.hermes/.env`:

```bash
MATRIX_HOME_ROOM=!abc123def456:matrix.example.org
```

## Lista de permissão de salas (`allowed_rooms`) {#room-allowlist-allowed_rooms}

Restrinja o bot a um conjunto fixo de salas Matrix. Quando definido, o bot **somente** responde em salas cujo ID aparece na lista — mensagens de qualquer outra sala são ignoradas silenciosamente, mesmo se o bot for mencionado.

**DMs (salas de chat direto) são isentos** deste filtro, para que usuários autorizados possam sempre alcançar o bot individualmente.

```yaml
matrix:
  allowed_rooms:
    - "!abc123def456:matrix.example.org"
    - "!opsroom789:matrix.example.org"
```

Ou via variável de ambiente (separada por vírgulas):

```bash
MATRIX_ALLOWED_ROOMS="!abc123def456:matrix.example.org,!opsroom789:matrix.example.org"
```

Comportamento:

- Vazio / não definido → sem restrição (padrão).
- Não vazio → o ID da sala deve estar na lista. A verificação roda **antes** de qualquer outra regra (exigência de menção, lista de permissão de remetentes, etc.).
- Use o **ID interno** da sala (`!abc...:server`), não seu alias (`#room:server`). Você encontra o ID interno de uma sala no Element via Sala → Settings → Advanced.

Veja também: [admin/user slash command split](../../reference/slash-commands.md#permissions-and-adminuser-split).


:::tip
Para encontrar um Room ID: no Element, vá na sala → **Settings** → **Advanced** → o **Internal room ID** é exibido ali (começa com `!`).
:::

## Comandos no Matrix {#commands-in-matrix}

O Hermes suporta os mesmos comandos de gateway no Matrix que suporta em outras plataformas de mensagens, incluindo `/commands`, `/model`, `/stop`, `/queue`, `/steer`, `/goal`, `/subgoal`, `/background`, `/bg`, `/btw`, `/tasks` e `/yolo`.

Alguns clientes Matrix reservam `/` inicial para comandos locais do cliente e podem não enviar comandos slash desconhecidos para a sala. Nesse caso, use `!` como alias seguro para Matrix:

```text
!commands
!model
!model gpt-5.5 --provider openrouter
!queue continue with the next task
!stop
```

O Hermes só normaliza `!command` quando o comando é conhecido pelo gateway, um comando de plugin registrado ou um comando de skill instalado. Exclamações comuns como `!important` permanecem mensagens de chat normais.

## Solução de problemas {#troubleshooting}

### O bot não responde a mensagens

**Causa**: O bot não entrou na sala, `MATRIX_ALLOWED_USERS` não inclui seu User ID, `MATRIX_ALLOWED_ROOMS` não inclui a sala, ou uma mensagem de sala não mencionou o bot.

**Correção**: Convide o bot para a sala — ele entra automaticamente ao ser convidado. Verifique se seu User ID está em `MATRIX_ALLOWED_USERS` (use o formato completo `@user:server`) e se o ID da sala está em `MATRIX_ALLOWED_ROOMS` se essa lista de permissão estiver configurada. Em salas, mencione o bot ou adicione a sala a `MATRIX_FREE_RESPONSE_ROOMS`. Reinicie o gateway.

### O bot entra em salas, mas descarta silenciosamente toda mensagem (clock skew)

**Causa**: O relógio do sistema do host está adiantado em relação ao horário real. O adaptador Matrix aplica um filtro de graça de startup de 5 segundos (`event_ts < startup_ts - 5`) para ignorar eventos replicados da sincronização inicial. Quando o relógio de parede está adiantado, todo evento recebido parece "mais antigo que o startup" e é descartado antes de chegar ao handler de mensagens — o bot parece conectado, mas nunca responde. Veja [#12614](https://github.com/NousResearch/hermes-agent/issues/12614).

**Sintoma**: O log do gateway mostra `Matrix: dropped N live events as 'too old' more than 30s after startup`.

**Correção**: Sincronize o relógio do host com NTP e reinicie o bot:

```bash
# Debian/Ubuntu
sudo timedatectl set-ntp true
timedatectl status   # confirm "System clock synchronized: yes"

# macOS
sudo sntp -sS time.apple.com
```

### "Failed to authenticate" / "whoami failed" na inicialização

**Causa**: O access token ou a URL do homeserver está incorreto.

**Correção**: Verifique se `MATRIX_HOMESERVER` aponta para seu homeserver (inclua `https://`, sem barra final). Confira se `MATRIX_ACCESS_TOKEN` é válido — teste com curl:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-server/_matrix/client/v3/account/whoami
```

Se isso retornar suas informações de usuário, o token é válido. Se retornar erro, gere um novo token.

### Erro "mautrix not installed"

**Causa**: O pacote Python `mautrix` não está instalado.

**Correção**: Instale-o:

```bash
pip install 'mautrix[encryption]'
```

Ou com extras do Hermes:

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[matrix]"
```

### Erros de criptografia / "could not decrypt event"

**Causa**: Chaves de criptografia ausentes, `libolm` não instalado ou o dispositivo do bot não é confiável.

**Correção**:
1. Verifique se `libolm` está instalado no seu sistema (veja a seção E2EE acima).
2. Certifique-se de que `MATRIX_ENCRYPTION=true` está definido no seu `.env`.
3. No seu cliente Matrix (Element), vá ao perfil do bot -> Sessions -> verifique/confie no dispositivo do bot.
4. Se o bot acabou de entrar em uma sala criptografada, ele só consegue descriptografar mensagens enviadas *depois* que entrou. Mensagens mais antigas são inacessíveis.

### Atualizando de uma versão anterior com E2EE

:::tip
Se você também excluiu manualmente `crypto.db`, veja o aviso "Deleting the crypto store" na seção E2EE acima — há passos adicionais para limpar one-time keys obsoletas do homeserver.
:::

Se você usou anteriormente o Hermes com `MATRIX_ENCRYPTION=true` e está atualizando para uma versão que usa o novo crypto store baseado em SQLite, a identidade de criptografia do bot mudou. Seu cliente Matrix (Element) pode cachear as chaves antigas do dispositivo e recusar compartilhar sessões de criptografia com o bot.

**Sintomas**: O bot conecta e mostra "E2EE enabled" nos logs, mas todas as mensagens mostram "could not decrypt event" e o bot nunca responde.

**O que está acontecendo**: O estado de criptografia antigo (do backend anterior `matrix-nio` ou `mautrix` baseado em serialização) é incompatível com o novo crypto store SQLite. O bot cria uma identidade de criptografia nova, mas seu cliente Matrix ainda tem as chaves antigas em cache e não compartilhará a sessão de criptografia da sala com um dispositivo cujas chaves mudaram. Isso é um recurso de segurança do Matrix — clientes tratam identity keys alteradas para o mesmo dispositivo como suspeitas.

**Correção** (migração única):

1. **Gere um novo access token** para obter um device ID novo. A forma mais simples:

   ```bash
   curl -X POST https://your-server/_matrix/client/v3/login \
     -H "Content-Type: application/json" \
     -d '{
       "type": "m.login.password",
       "identifier": {"type": "m.id.user", "user": "@hermes:your-server.org"},
       "password": "***",
       "initial_device_display_name": "Hermes Agent"
     }'
   ```

   Copie o novo `access_token` e atualize `MATRIX_ACCESS_TOKEN` em `~/.hermes/.env`.

2. **Exclua o estado de criptografia antigo**:

   ```bash
   rm -f ~/.hermes/platforms/matrix/store/crypto.db
   rm -f ~/.hermes/platforms/matrix/store/crypto_store.*
   ```

3. **Defina sua chave de recuperação** (se você usa cross-signing — a maioria dos usuários Element usa). Adicione em `~/.hermes/.env`:

   ```bash
   MATRIX_RECOVERY_KEY=EsT... your recovery key here
   ```

   Isso permite que o bot auto-assine com chaves de cross-signing na inicialização, para que o Element confie no novo dispositivo imediatamente. Sem isso, o Element pode ver o novo dispositivo como não verificado e recusar compartilhar sessões de criptografia. Encontre sua chave de recuperação no Element em **Settings** → **Security & Privacy** → **Encryption**.

4. **Force seu cliente Matrix a rotacionar a sessão de criptografia**. No Element, abra a sala de DM com o bot e digite `/discardsession`. Isso força o Element a criar uma nova sessão de criptografia e compartilhá-la com o novo dispositivo do bot.

5. **Reinicie o gateway**:

   ```bash
   hermes gateway run
   ```

   Se `MATRIX_RECOVERY_KEY` estiver definido, você deve ver `Matrix: cross-signing verified via recovery key` nos logs.

6. **Envie uma nova mensagem**. O bot deve descriptografar e responder normalmente.

:::note
Após a migração, mensagens enviadas *antes* da atualização não podem ser descriptografadas — as chaves de criptografia antigas se foram. Isso afeta apenas a transição; novas mensagens funcionam normalmente.
:::

:::tip
**Novas instalações não são afetadas.** Esta migração só é necessária se você tinha uma configuração E2EE funcionando com uma versão anterior do Hermes e está atualizando.

**Por que um novo access token?** Cada access token Matrix está vinculado a um device ID específico. Reutilizar o mesmo device ID com novas chaves de criptografia faz outros clientes Matrix desconfiarem do dispositivo (eles veem identity keys alteradas como uma possível violação de segurança). Um novo access token obtém um device ID novo sem histórico de chaves obsoletas, então outros clientes confiam nele imediatamente.
:::

## Modo proxy (E2EE no macOS) {#proxy-mode-e2ee-on-macos}

E2EE do Matrix exige `libolm`, que não compila no macOS ARM64 (Apple Silicon). O extra `hermes-agent[matrix]` é restrito a Linux. Se você está no macOS, o modo proxy permite rodar E2EE em um container Docker em uma VM Linux enquanto o agente real roda nativamente no macOS com acesso total aos seus arquivos locais, memória e skills.

### Como funciona

```
macOS (Host):
  └─ hermes gateway
       ├─ api_server adapter ← listens on 0.0.0.0:8642
       ├─ AIAgent ← single source of truth
       ├─ Sessions, memory, skills
       └─ Local file access (Obsidian, projects, etc.)

Linux VM (Docker):
  └─ hermes gateway (proxy mode)
       ├─ Matrix adapter ← E2EE decryption/encryption
       └─ HTTP forward → macOS:8642/v1/chat/completions
           (no LLM API keys, no agent, no inference)
```

O container Docker só trata do protocolo Matrix + E2EE. Quando uma mensagem chega, ele a descriptografa e encaminha o texto para o host via uma requisição HTTP padrão. O host roda o agente, chama ferramentas, gera uma resposta e a transmite de volta. O container criptografa e envia a resposta ao Matrix. Todas as sessões são unificadas — CLI, Matrix, Telegram e qualquer outra plataforma compartilham a mesma memória e histórico de conversa.

### Passo 1: Configure o host (macOS)

Habilite o API server para que o host aceite requisições recebidas do container Docker.

Adicione em `~/.hermes/.env`:

```bash
API_SERVER_ENABLED=true
API_SERVER_KEY=your-secret-key-here
API_SERVER_HOST=0.0.0.0
```

- `API_SERVER_HOST=0.0.0.0` faz bind em todas as interfaces para que o container Docker possa alcançá-lo.
- `API_SERVER_KEY` é obrigatório para bind não-loopback. Escolha uma string aleatória forte.
- O API server roda na porta 8642 por padrão (altere com `API_SERVER_PORT` se necessário).

Inicie o gateway:

```bash
hermes gateway
```

Você deve ver o API server iniciar junto com quaisquer outras plataformas que tenha configurado. Verifique se é acessível a partir da VM:

```bash
# From the Linux VM
curl http://<mac-ip>:8642/health
```

### Passo 2: Configure o container Docker (VM Linux)

O container precisa de credenciais Matrix e da URL do proxy. Ele NÃO precisa de chaves de API de LLM.

**`docker-compose.yml`:**

```yaml
services:
  hermes-matrix:
    build: .
    environment:
      # Matrix credentials
      MATRIX_HOMESERVER: "https://matrix.example.org"
      MATRIX_ACCESS_TOKEN: "syt_..."
      MATRIX_ALLOWED_USERS: "@you:matrix.example.org"
      MATRIX_ENCRYPTION: "true"
      MATRIX_DEVICE_ID: "HERMES_BOT"

      # Proxy mode — forward to host agent
      GATEWAY_PROXY_URL: "http://192.168.1.100:8642"
      GATEWAY_PROXY_KEY: "your-secret-key-here"
    volumes:
      - ./matrix-store:/root/.hermes/platforms/matrix/store
```

**`Dockerfile`:**

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libolm-dev && rm -rf /var/lib/apt/lists/*
RUN cd ~/.hermes/hermes-agent && uv pip install -e ".[matrix]"

CMD ["hermes", "gateway"]
```

Esse é o container inteiro. Sem chaves de API para OpenRouter, Anthropic ou qualquer provedor de inferência.

### Passo 3: Inicie ambos

1. Inicie o gateway do host primeiro:
   ```bash
   hermes gateway
   ```

2. Inicie o container Docker:
   ```bash
   docker compose up -d
   ```

3. Envie uma mensagem em uma sala Matrix criptografada. O container descriptografa, encaminha para o host e transmite a resposta de volta.

### Referência de configuração

O modo proxy é configurado no **lado do container** (o gateway fino):

| Configuração | Descrição |
|---------|-------------|
| `GATEWAY_PROXY_URL` | URL do API server remoto do Hermes (ex.: `http://192.168.1.100:8642`) |
| `GATEWAY_PROXY_KEY` | Bearer token para autenticação (deve corresponder a `API_SERVER_KEY` no host) |
| `gateway.proxy_url` | Igual a `GATEWAY_PROXY_URL`, mas em `config.yaml` |

O lado do host precisa de:

| Configuração | Descrição |
|---------|-------------|
| `API_SERVER_ENABLED` | Defina como `true` |
| `API_SERVER_KEY` | Bearer token (compartilhado com o container) |
| `API_SERVER_HOST` | Defina como `0.0.0.0` para acesso em rede |
| `API_SERVER_PORT` | Número da porta (padrão: `8642`) |

### Funciona para qualquer plataforma

O modo proxy não se limita ao Matrix. Qualquer adaptador de plataforma pode usá-lo — defina `GATEWAY_PROXY_URL` em qualquer instância de gateway e ela encaminhará para o agente remoto em vez de rodar um localmente. Isso é útil para implantações em que o adaptador de plataforma precisa rodar em um ambiente diferente do agente (isolamento de rede, requisitos de E2EE, restrições de recursos).

:::tip
A continuidade de sessão é mantida via o header `X-Hermes-Session-Id`. O API server do host rastreia sessões por esse ID, então conversas persistem entre mensagens como aconteceria com um agente local.
:::

:::note
**Limitações (v1):** Mensagens de progresso de ferramentas do agente remoto não são retransmitidas — o usuário vê apenas a resposta final transmitida, não chamadas individuais de ferramentas. Prompts de aprovação de comandos perigosos são tratados no lado do host, não retransmitidos ao usuário Matrix. Isso pode ser endereçado em atualizações futuras.
:::

### O bot conecta e envia, mas ignora mensagens recebidas

**Causa**: Handlers de eventos Matrix só disparam quando payloads de sync são despachados pela maquinaria `handle_sync()` do mautrix. Um poll `client.sync()` bruto que nunca chama `handle_sync()` pode deixar o adaptador conectado (envio funciona) enquanto mensagens recebidas nunca chegam a `_on_room_message`.

**Correção**: O Hermes usa um loop de sync explícito que chama `client.handle_sync()` tanto no sync inicial quanto em toda resposta de sync incremental. Isso corresponde ao diagnóstico na issue upstream #7914 e no PR fechado #37807, mas mantém as tarefas de manutenção em background do Hermes (rastreamento de salas entrantes, tratamento de convites, compartilhamento de chaves E2EE) em vez de delegar o ciclo de vida completo a `client.start()`. Se mensagens recebidas ainda falharem após reiniciar o gateway, verifique se os handlers estão registrados antes do primeiro sync e confira os logs por `sync event dispatch error`.

### Problemas de sync / bot fica para trás

**Causa**: Execuções longas de ferramentas podem atrasar o loop de sync, ou o homeserver está lento.

**Correção**: O loop de sync tenta novamente automaticamente a cada 5 segundos em erro. Verifique os logs do Hermes por avisos relacionados a sync. Se o bot consistentemente fica para trás, garanta que seu homeserver tenha recursos adequados.

### O bot está offline

**Causa**: O gateway Hermes não está rodando, ou falhou ao conectar.

**Correção**: Verifique se `hermes gateway` está rodando. Olhe a saída do terminal por mensagens de erro. Problemas comuns: URL de homeserver errada, access token expirado, homeserver inacessível.

### "User not allowed" / O bot ignora você

**Causa**: Seu User ID não está em `MATRIX_ALLOWED_USERS`.

**Correção**: Adicione seu User ID a `MATRIX_ALLOWED_USERS` em `~/.hermes/.env` e reinicie o gateway. Use o formato completo `@user:server`.

### O bot ignora uma sala inteira

**Causa**: `MATRIX_ALLOWED_ROOMS` está definido e o ID da sala atual não está listado, ou a sala exige menção e a mensagem não mencionou o bot.

**Correção**: Adicione o ID da sala a `MATRIX_ALLOWED_ROOMS`, ou remova a lista de permissão de salas se for uma implantação pessoal. Para encontrar um Room ID no Element, abra as configurações da sala e confira **Advanced**.

### Mensagens de bridge entram em loop ou ecoam

**Causa**: Um puppet de bridge/appservice está retransmitindo a saída do bot de volta como uma nova mensagem de usuário, ou uma bridge usa IDs de ghost user não padrão.

**Correção**: Mantenha ghosts de bridge fora de `MATRIX_ALLOWED_USERS`, adicione uma entrada correspondente em `MATRIX_IGNORE_USER_PATTERNS` e deixe `MATRIX_PROCESS_NOTICES=false`, a menos que notices façam parte de um fluxo confiável.

## Segurança {#security}

:::warning
Sempre defina `MATRIX_ALLOWED_USERS` e, para implantações compartilhadas/privadas, `MATRIX_ALLOWED_ROOMS`. Sem eles, qualquer pessoa que consiga enviar mensagem ao bot em uma sala em que ele entrou pode disparar o agente. Autorize somente pessoas e salas em que você confia — usuários autorizados têm acesso total às capacidades do agente, incluindo uso de ferramentas e acesso ao sistema.
:::

Para mais informações sobre proteger sua implantação do Hermes Agent, veja o [Security Guide](../security.md).

## Notas {#notes}

- **Qualquer homeserver**: Funciona com Synapse, Conduit, Dendrite, matrix.org ou qualquer homeserver Matrix compatível com a especificação. Nenhum software de homeserver específico é necessário.
- **Federação**: Se você está em um homeserver federado, o bot pode se comunicar com usuários de outros servidores — basta adicionar seus IDs completos `@user:server` a `MATRIX_ALLOWED_USERS`.
- **Auto-join**: O bot aceita automaticamente convites para salas e entra. Começa a responder imediatamente após entrar.
- **Suporte a mídia**: O Hermes pode enviar e receber imagens, áudio, vídeo e anexos de arquivo. Mídia é enviada ao seu homeserver usando a API de repositório de conteúdo do Matrix.
- **Mensagens de voz nativas (MSC3245)**: O adaptador Matrix marca automaticamente mensagens de voz enviadas com a flag `org.matrix.msc3245.voice`. Isso significa que respostas TTS e áudio de voz são renderizados como **bolhas de voz nativas** no Element e em outros clientes que suportam MSC3245, em vez de anexos genéricos de arquivo de áudio. Mensagens de voz recebidas com a flag MSC3245 também são identificadas corretamente e encaminhadas para transcrição speech-to-text. Nenhuma configuração é necessária — isso funciona automaticamente.
