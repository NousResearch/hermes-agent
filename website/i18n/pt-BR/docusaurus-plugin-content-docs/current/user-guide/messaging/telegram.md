---
sidebar_position: 1
title: "Telegram"
description: "Configure o Hermes Agent como bot do Telegram"
---

# Configuração do Telegram {#telegram-setup}

O Hermes Agent se integra ao Telegram como um bot conversacional completo. Depois de conectado, você pode conversar com seu agente em qualquer dispositivo, enviar mensagens de voz que são transcritas automaticamente, receber resultados de tarefas agendadas e usar o agente em chats de grupo. A integração é construída sobre [python-telegram-bot](https://python-telegram-bot.org/) e suporta texto, voz, imagens e anexos de arquivos.

## Passo 1: Crie um bot via BotFather {#step-1-create-a-bot-via-botfather}

Todo bot do Telegram precisa de um token de API emitido pelo [@BotFather](https://t.me/BotFather), a ferramenta oficial de gerenciamento de bots do Telegram.

1. Abra o Telegram e busque por **@BotFather**, ou visite [t.me/BotFather](https://t.me/BotFather)
2. Envie `/newbot`
3. Escolha um **nome de exibição** (ex.: "Hermes Agent") — pode ser qualquer coisa
4. Escolha um **username** — deve ser único e terminar em `bot` (ex.: `my_hermes_bot`)
5. O BotFather responde com seu **token de API**. Ele se parece com isto:

```
123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

:::warning
Mantenha o token do bot em segredo. Qualquer pessoa com esse token pode controlar seu bot. Se vazar, revogue imediatamente via `/revoke` no BotFather.
:::

## Passo 2: Personalize seu bot (Opcional) {#step-2-customize-your-bot-optional}

Esses comandos do BotFather melhoram a experiência do usuário. Envie mensagem para @BotFather e use:

| Command | Purpose |
|---------|---------|
| `/setdescription` | O texto "O que este bot pode fazer?" exibido antes do usuário iniciar a conversa |
| `/setabouttext` | Texto curto na página de perfil do bot |
| `/setuserpic` | Envie um avatar para seu bot |
| `/setcommands` | Define o menu de comandos (o botão `/` no chat) |
| `/setprivacy` | Controla se o bot vê todas as mensagens de grupo (veja o Passo 3) |

:::tip
Para `/setcommands`, um conjunto inicial útil:

```
help - Show help information
new - Start a new conversation
sethome - Set this chat as the home channel
```
:::

### Indicador de status Online/Offline (Opcional) {#onlineoffline-status-indicator-optional}

Bots do Telegram não têm um ponto real de presença online/offline — aquele ponto verde é um
recurso de *conta de usuário*, não algo que a Bot API expõe para bots. A superfície mais
próxima é a **descrição curta** do bot (a linha exibida abaixo do nome no perfil do bot).

Ative `status_indicator` e o Hermes define essa descrição curta como **Online**
quando o gateway conecta e **Offline** em um encerramento limpo:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        status_indicator: true
        # Optional custom strings (defaults: "Online" / "Offline"):
        status_online: "🟢 Online"
        status_offline: "🔴 Offline"
```

Observações:

- A descrição curta é **global** para o bot (visível a todos os usuários), não
  por chat. Os usuários a veem na página de perfil do bot, não como um badge ao vivo dentro
  de um chat aberto.
- Apenas um encerramento **limpo** do gateway (`/stop`, `disconnect`) grava "Offline".
  Uma falha abrupta mantém o último status conhecido — a limitação inerente de um
  indicador baseado em texto de perfil.
- Desativado por padrão, pois altera o perfil global do bot.

### Prioridade e limite do menu de comandos (Opcional) {#command-menu-priority-and-cap-optional}

O Hermes registra seu menu de comandos automaticamente quando o gateway do Telegram inicia. O menu é construído a partir do registro central de slash commands mais comandos elegíveis de plugins/skills, e então limitado para que o Telegram aceite o payload de forma confiável. O limite padrão é 60 comandos — suficiente para manter visíveis todos os comandos integrados mais comandos comuns de skills.

Se você tem comandos locais ou de plugin que devem permanecer visíveis no seletor `/` do Telegram, priorize-os em `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      command_menu:
        max_commands: 60
        priority_mode: prepend  # prepend | append | replace
        priority:
          - my_plugin_command
```

`priority_mode` controla como sua lista se combina com a lista de prioridade integrada do Hermes:

- `prepend`: coloca seus comandos primeiro, depois os padrões do Hermes
- `append`: mantém os padrões do Hermes primeiro, depois seus comandos
- `replace`: usa apenas sua lista para ordenação de prioridade

O Telegram permite até 100 BotCommands, mas payloads grandes de comandos podem falhar. O Hermes usa 60 por padrão por confiabilidade e limita valores configurados a `1..100`; use `/commands` para a lista completa de comandos.

## Passo 3: Modo de privacidade (Crítico para grupos) {#step-3-privacy-mode-critical-for-groups}

Bots do Telegram têm um **modo de privacidade** que vem **ativado por padrão**. Esta é a fonte de confusão mais comum ao usar bots em grupos.

**Com o modo de privacidade LIGADO**, seu bot só pode ver:
- Mensagens que começam com um comando `/`
- Respostas diretamente às mensagens do próprio bot
- Mensagens de serviço (entrada/saída de membros, mensagens fixadas, etc.)
- Mensagens em canais onde o bot é admin

**Com o modo de privacidade DESLIGADO**, o bot recebe todas as mensagens do grupo.

### Como desativar o modo de privacidade {#how-to-disable-privacy-mode}

1. Envie mensagem para **@BotFather**
2. Envie `/mybots`
3. Selecione seu bot
4. Vá em **Bot Settings → Group Privacy → Turn off**

:::warning
**Você deve remover e adicionar novamente o bot a qualquer grupo** após alterar a configuração de privacidade. O Telegram armazena em cache o estado de privacidade quando um bot entra em um grupo, e ele não será atualizado até que o bot seja removido e readicionado.
:::

:::tip
Uma alternativa a desativar o modo de privacidade: promova o bot a **admin do grupo**. Bots admin sempre recebem todas as mensagens independentemente da configuração de privacidade, e isso evita precisar alternar o modo de privacidade global.
:::

### Observar conversas de grupo sem responder automaticamente {#observe-group-chatter-without-auto-replying}

Para comportamento de grupo estilo OpenClaw/Yuanbao, configure o Telegram para que o bot possa **ver** mensagens ordinárias de grupo, mas só **responda** quando acionado diretamente:

```yaml
telegram:
  allowed_chats:
    - "-1001234567890"
  group_allowed_chats:
    - "-1001234567890"
  require_mention: true
  observe_unmentioned_group_messages: true
```

Com este modo ativado, mensagens de grupo sem menção de chats/tópicos explicitamente na allowlist são anexadas à transcrição da sessão compartilhada de chat/tópico como contexto observado, mas não disparam o agente. `allowed_chats` controla onde o bot responde; `group_allowed_chats` autoriza a sessão de grupo compartilhada usada para contexto observado, então use os mesmos IDs de chat para este modo. Uma menção posterior `@botname`, resposta ao bot ou padrão de menção configurado nesse mesmo chat/tópico na allowlist pode usar esse contexto observado. A mensagem acionada também é marcada com `[nickname|user_id]` e recebe um prompt de segurança por turno para que o modelo trate linhas observadas anteriores como contexto, não como instruções endereçadas ao bot.

Variável de ambiente equivalente:

```bash
TELEGRAM_ALLOWED_CHATS=-1001234567890
TELEGRAM_GROUP_ALLOWED_CHATS=-1001234567890
TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES=true
```

Isso exige que o Telegram entregue mensagens ordinárias de grupo ao gateway, então desative o modo de privacidade do BotFather ou promova o bot a admin do grupo conforme descrito acima.

## Passo 4: Encontre seu ID de usuário {#step-4-find-your-user-id}

O Hermes Agent usa IDs numéricos de usuário do Telegram para controlar o acesso. Seu ID de usuário **não** é seu username — é um número como `123456789`.

**Método 1 (recomendado):** Envie mensagem para [@userinfobot](https://t.me/userinfobot) — ele responde instantaneamente com seu ID de usuário.

**Método 2:** Envie mensagem para [@get_id_bot](https://t.me/get_id_bot) — outra opção confiável.

Guarde este número; você precisará dele no próximo passo.

## Passo 5: Configure o Hermes {#step-5-configure-hermes}

### Opção A: Configuração interativa (Recomendado) {#option-a-interactive-setup-recommended}

```bash
hermes gateway setup
```

Selecione **Telegram** quando solicitado. O assistente pede seu token de bot e IDs de usuários permitidos, e então grava a configuração para você.

### Opção B: Configuração manual {#option-b-manual-configuration}

Adicione o seguinte em `~/.hermes/.env`:

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789    # Comma-separated for multiple users
```

### Inicie o gateway {#start-the-gateway}

```bash
hermes gateway
```

O bot deve ficar online em segundos. Envie uma mensagem a ele no Telegram para verificar.

## Enviando arquivos gerados de terminais com Docker {#sending-generated-files-from-docker-backed-terminals}

Se seu backend de terminal é `docker`, lembre-se de que anexos do Telegram são
enviados pelo **processo do gateway**, não de dentro do container. Isso significa que o
caminho final `MEDIA:/...` deve ser legível no host onde o gateway está
em execução.

Armadilha comum:

- o agente grava um arquivo dentro do Docker em `/workspace/report.txt`
- o modelo emite `MEDIA:/workspace/report.txt`
- a entrega no Telegram falha porque `/workspace/report.txt` só existe dentro do
  container, não no host

Padrão recomendado:

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/.hermes/cache/documents:/output"
```

Então:

- grave arquivos dentro do Docker em `/output/...`
- emita o caminho **visível no host** em `MEDIA:`, por exemplo:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`

Se você já tem uma seção `docker_volumes:`, adicione o novo mount à mesma
lista. Chaves YAML duplicadas sobrescrevem silenciosamente as anteriores.

### Extensões de arquivo `MEDIA:` suportadas {#supported-media-file-extensions}

O gateway extrai tags `MEDIA:/path/to/file` das respostas do agente e envia o arquivo referenciado como anexo nativo da plataforma. Extensões suportadas em todas as plataformas do gateway:

| Category | Extensions |
|---|---|
| Images | `png`, `jpg`, `jpeg`, `gif`, `webp`, `bmp`, `tiff`, `svg` |
| Audio | `mp3`, `wav`, `ogg`, `m4a`, `opus`, `flac`, `aac` |
| Video | `mp4`, `mov`, `webm`, `mkv`, `avi` |
| **Documents** | `pdf`, `txt`, `md`, `csv`, `json`, `xml`, `html`, `yaml`, `yml`, `log` |
| **Office** | `docx`, `xlsx`, `pptx`, `odt`, `ods`, `odp` |
| **Archives** | `zip`, `rar`, `7z`, `tar`, `gz`, `bz2` |
| **Books / packages** | `epub`, `apk`, `ipa` |

Qualquer item desta lista é entregue como anexo nativo em plataformas que suportam (Telegram, Discord, Signal, Slack, WhatsApp, Feishu, Matrix, etc.); em plataformas sem suporte nativo, cai para um link ou indicador em texto simples. As categorias em **negrito** foram adicionadas nas últimas versões — se você dependia do modelo dizendo `here is the file: /path/to/report.docx`, troque para `MEDIA:/path/to/report.docx` para entrega nativa.

## Modo webhook {#webhook-mode}

Por padrão, o Hermes se conecta ao Telegram usando **long polling** — o gateway faz requisições de saída aos servidores do Telegram para buscar novas atualizações. Isso funciona bem para implantações locais e sempre ativas.

Para **implantações em nuvem** (Fly.io, Railway, Render, etc.), o **modo webhook** é mais econômico. Essas plataformas podem acordar automaticamente máquinas suspensas com tráfego HTTP de entrada, mas não com conexões de saída. Como o polling é de saída, um bot em polling nunca pode dormir. O modo webhook inverte a direção — o Telegram envia atualizações para a URL HTTPS do seu bot, permitindo implantações que dormem quando ociosas.

| | Polling (default) | Webhook |
|---|---|---|
| Direction | Gateway → Telegram (outbound) | Telegram → Gateway (inbound) |
| Best for | Local, always-on servers | Cloud platforms with auto-wake |
| Setup | No extra config | Set `TELEGRAM_WEBHOOK_URL` |
| Idle cost | Machine must stay running | Machine can sleep between messages |

### Configuração {#configuration}

Adicione o seguinte em `~/.hermes/.env`:

```bash
TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
TELEGRAM_WEBHOOK_SECRET="$(openssl rand -hex 32)"  # required
# TELEGRAM_WEBHOOK_PORT=8443        # optional, default 8443
```

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_WEBHOOK_URL` | Yes | URL HTTPS pública onde o Telegram enviará atualizações. O caminho da URL é extraído automaticamente (ex.: `/telegram` do exemplo acima). |
| `TELEGRAM_WEBHOOK_SECRET` | **Yes** (when `TELEGRAM_WEBHOOK_URL` is set) | Token secreto que o Telegram ecoa em toda requisição de webhook para verificação. O gateway se recusa a iniciar sem ele — veja [GHSA-3vpc-7q5r-276h](https://github.com/NousResearch/hermes-agent/security/advisories/GHSA-3vpc-7q5r-276h). Gere com `openssl rand -hex 32`. |
| `TELEGRAM_WEBHOOK_PORT` | No | Porta local em que o servidor de webhook escuta (padrão: `8443`). |

Quando `TELEGRAM_WEBHOOK_URL` está definido, o gateway inicia um servidor HTTP de webhook em vez de polling. Quando não definido, o modo polling é usado — sem mudança de comportamento em relação a versões anteriores.

### Exemplo de implantação em nuvem (Fly.io) {#cloud-deployment-example-flyio}

1. Adicione as variáveis de ambiente aos secrets do seu app Fly.io:

```bash
fly secrets set TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
fly secrets set TELEGRAM_WEBHOOK_SECRET=$(openssl rand -hex 32)
```

2. Exponha a porta do webhook no seu `fly.toml`:

```toml
[[services]]
  internal_port = 8443
  protocol = "tcp"

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

3. Implante:

```bash
fly deploy
```

O log do gateway deve mostrar: `[telegram] Connected to Telegram (webhook mode)`.

## Suporte a proxy {#proxy-support}

Se a API do Telegram estiver bloqueada ou você precisar rotear tráfego por um proxy, defina uma URL de proxy específica do Telegram. Isso tem prioridade sobre as variáveis de ambiente genéricas `HTTPS_PROXY` / `HTTP_PROXY`.

**Opção 1: config.yaml (recomendado)**

```yaml
telegram:
  proxy_url: "socks5://127.0.0.1:1080"
```

**Opção 2: variável de ambiente**

```bash
TELEGRAM_PROXY=socks5://127.0.0.1:1080
```

Esquemas suportados: `http://`, `https://`, `socks5://`.

O proxy se aplica tanto à conexão principal do Telegram quanto ao transporte de IP de fallback. Se nenhum proxy específico do Telegram estiver definido, o gateway recorre a `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` (ou detecção automática de proxy do sistema macOS).

## Canal home {#home-channel}

Use o comando `/sethome` em qualquer chat do Telegram (DM ou grupo) para designá-lo como **canal home**. Tarefas agendadas (cron jobs) entregam seus resultados neste canal.

Você também pode definir manualmente em `~/.hermes/.env`:

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"
```

:::tip
IDs de chat de grupo são números negativos (ex.: `-1001234567890`). Seu ID de chat de DM pessoal é o mesmo que seu ID de usuário.
:::

### Entregas de cron no modo de tópicos {#cron-deliveries-in-topic-mode}

Se você tem o modo de tópicos ativado no DM do bot, mensagens de cron entregues ao chat raiz caem no lobby reservado ao sistema — responder lá não abre sessão e você vê o aviso "main chat is reserved for system commands". Crie um tópico de fórum dedicado (ex.: `Cron`) e defina:

```bash
TELEGRAM_CRON_THREAD_ID=<topic_thread_id>
```

`TELEGRAM_CRON_THREAD_ID` substitui `TELEGRAM_HOME_CHANNEL_THREAD_ID` apenas para entregas de cron. Respostas nesse tópico continuam a sessão existente do tópico.

## Mensagens de voz {#voice-messages}

### Voz recebida (Speech-to-Text) {#incoming-voice-speech-to-text}

Mensagens de voz que você envia no Telegram são transcritas automaticamente pelo provedor STT configurado do Hermes e injetadas como texto na conversa.

- `local` usa `faster-whisper` na máquina que executa o Hermes — nenhuma chave de API necessária
- `groq` usa Groq Whisper e requer `GROQ_API_KEY`
- `openai` usa OpenAI Whisper e requer `VOICE_TOOLS_OPENAI_KEY`

#### Ignorar STT: passar o arquivo de áudio bruto ao agente

Se você preferir que o **próprio agente** trate o áudio — para diarização, uma ferramenta de transcrição personalizada ou apenas arquivar a gravação — defina `stt.enabled: false` em `~/.hermes/config.yaml`:

```yaml
stt:
  enabled: false
```

Com STT desativado, o gateway ainda baixa o anexo de voz/áudio para o cache de áudio do Hermes, mas **não o transcreve**. O agente recebe a mensagem com um marcador como:

```
[The user sent a voice message: /home/<user>/.hermes/cache/audio/<hash>.ogg]
```

Suas ferramentas ou skills podem então ler esse caminho diretamente (ex.: repassá-lo a um pipeline local de diarização, um modelo de transcrição mais rico ou enviá-lo para armazenamento de longo prazo). A extensão do arquivo reflete o formato original entregue pelo Telegram (`.ogg` para notas de voz, `.mp3`/`.m4a`/etc. para anexos de áudio).

Isso combina naturalmente com a seção [servidor Bot API local](#large-files-20mb-via-local-bot-api-server) abaixo, que eleva o teto de 20MB do getFile do Telegram para 2GB — útil quando as gravações que você quer processar duram mais que alguns minutos.

### Voz enviada (Text-to-Speech) {#outgoing-voice-text-to-speech}

Quando o agente gera áudio via TTS, ele é entregue como **bolhas de voz** nativas do Telegram — o tipo redondo e reproduzível inline.

- **OpenAI e ElevenLabs** produzem Opus nativamente — nenhuma configuração extra necessária
- **Edge TTS** (o provedor gratuito padrão) gera MP3 e requer **ffmpeg** para converter para Opus:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

Sem ffmpeg, o áudio Edge TTS é enviado como arquivo de áudio regular (ainda reproduzível, mas usa o player retangular em vez de uma bolha de voz).

Configure o provedor TTS no seu `config.yaml` sob a chave `tts.provider`.

## Arquivos grandes (>20MB) via servidor Bot API local {#large-files-20mb-via-local-bot-api-server}

A Bot API **pública** do Telegram limita downloads `getFile` a **20 MB**, então qualquer nota de voz, arquivo de áudio, vídeo ou documento maior que isso é silenciosamente rejeitado pelo Hermes com uma resposta "too large". A forma documentada de contornar isso é executar um daemon **local** [telegram-bot-api](https://github.com/tdlib/telegram-bot-api) — o mesmo software de servidor que o Telegram usa, mas rodando na sua rede. Um servidor local eleva o teto de arquivos para **2 GB** e o Hermes eleva automaticamente seu próprio limite interno quando vê um `base_url` personalizado configurado.

Isso desbloqueia fluxos como:

- Enviar memos de voz longos (reuniões de 45 minutos, podcasts) ao bot
- Enviar vídeos grandes para processamento com ferramentas de visão
- Arquivar áudio bruto para pipelines offline como diarização, alinhamento ou dados de treinamento

### Passo 1: Obtenha credenciais da API do Telegram {#step-1-obtain-telegram-api-credentials}

O servidor local fala diretamente com a camada MTProto do Telegram (não a Bot API pública), então precisa de **credenciais MTProto**:

1. Visite [my.telegram.org/apps](https://my.telegram.org/apps) e entre com sua conta do Telegram.
2. Crie um novo aplicativo (qualquer nome e descrição curta serve).
3. Copie o `api_id` e `api_hash` — ambos são obrigatórios.

### Passo 2: Execute o servidor telegram-bot-api {#step-2-run-the-telegram-bot-api-server}

A imagem Docker mantida pela comunidade [`aiogram/telegram-bot-api`](https://hub.docker.com/r/aiogram/telegram-bot-api) é o caminho mais fácil. Um `docker-compose.yaml` mínimo (use o modo `--local` para habilitar os limites maiores):

```yaml
services:
  tg-bot-api:
    image: aiogram/telegram-bot-api:latest
    container_name: tg-bot-api
    restart: unless-stopped
    ports:
      - "127.0.0.1:8081:8081"   # bind to loopback only; see security note
    environment:
      TELEGRAM_API_ID: "12345"           # your api_id from Step 1
      TELEGRAM_API_HASH: "abcdef..."     # your api_hash from Step 1
      TELEGRAM_LOCAL: "1"                # enable --local mode (raises 20MB → 2GB)
    volumes:
      - ./tg-bot-api-data:/var/lib/telegram-bot-api
```

Suba o serviço:

```bash
docker compose up -d tg-bot-api
docker logs --tail 20 tg-bot-api
```

:::warning Security
O servidor Bot API local recebe seu token de bot no caminho da URL (ex. `/bot<TOKEN>/getMe`) **sem autenticação adicional**. Qualquer pessoa que alcançar a porta pode controlar totalmente seu bot — ler toda mensagem que ele pode ver, enviar mensagens como ele, etc. Vincule o container a `127.0.0.1` e/ou coloque um reverse proxy em uma rede privada. **Nunca exponha a porta 8081 à internet pública.**
:::

### Passo 3: Deslogue o bot da API pública (uma vez) {#step-3-log-the-bot-out-of-the-public-api-one-time}

Um bot só pode estar ativo em **um** servidor Bot API por vez. Se seu bot já estava rodando contra `api.telegram.org` (o que quase certamente estava), você deve deslogá-lo explicitamente lá antes que o servidor local o aceite:

```bash
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/logOut"
# expected response: {"ok":true,"result":true}
```

Este é um passo de migração único — você não repete a cada reinício. O Telegram entrega quaisquer mensagens recebidas após `logOut` pelo novo servidor.

Verifique se o servidor local consegue falar com o Telegram em nome do bot:

```bash
curl "http://127.0.0.1:8081/bot<YOUR_BOT_TOKEN>/getMe"
# expected response: {"ok":true,"result":{"id":...,"is_bot":true,...}}
```

### Passo 4: Aponte o Hermes para o servidor local {#step-4-point-hermes-at-the-local-server}

Adicione as URLs em `platforms.telegram.extra` em `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      base_url: "http://127.0.0.1:8081/bot"
      base_file_url: "http://127.0.0.1:8081/file/bot"
      local_mode: true        # see Step 5 below — only set this if the bot's data
                              # directory is readable by the Hermes process
```

:::caution Use `platforms.telegram.extra`, not `telegram.extra`
No momento, apenas a forma `platforms.<name>.extra` é deep-merged na config da plataforma. Chaves colocadas diretamente sob um bloco `telegram.extra` de nível superior são silenciosamente descartadas.
:::

Quando `base_url` está definido, o Hermes:

- Constrói o cliente python-telegram-bot contra o servidor local
- Eleva automaticamente seu limite interno de tamanho de documento/áudio de 20 MB → 2 GB
- Reporta o limite ativo na mensagem de erro "too large" (`Maximum: 2048 MB.`) para ficar óbvio em qual modo você está

Reinicie o gateway e procure uma linha de confirmação no log:

```bash
hermes gateway restart
grep -E "Using custom Telegram base_url|Using Telegram local_mode" ~/.hermes/logs/gateway.log | tail
```

### Passo 5: `local_mode` — acesso a arquivos em disco {#step-5-local_mode-file-access-on-disk}

O servidor local tem **duas formas** de entregar arquivos:

1. **Sem `--local`** (o padrão): arquivos são servidos via HTTP em `/file/bot<TOKEN>/<path>`, igual à Bot API pública. O teto de 20MB permanece. Útil apenas como correção de rede (ex. quando `api.telegram.org` está inacessível mas você pode self-host); não é o que você quer para elevar o tamanho.
2. **Com `--local`** (definido via `TELEGRAM_LOCAL=1` acima): arquivos são gravados no filesystem do servidor e a resposta `getFile` retorna um **caminho absoluto** em vez de uma URL HTTP. O teto de 20MB é removido. O Hermes deve então ler os bytes **do disco**, não via HTTP.

Para fazer o caminho de leitura em disco funcionar, defina `local_mode: true` na config acima **e** certifique-se de que o processo Hermes pode ler o caminho que o servidor retorna. Dois cenários:

- **Mesma máquina** — telegram-bot-api e Hermes rodam no mesmo host. Faça bind-mount do volume de dados em um diretório que o Hermes possa ler (ex.: `/var/lib/telegram-bot-api`) e garanta que a propriedade dos arquivos corresponda. O container reduz privilégios para seu usuário interno `telegram-bot-api` (uid varia por imagem); a correção mais simples é adicionar `user: "<UID>:<GID>"` ao serviço compose para que os arquivos sejam de um uid que o Hermes já executa.
- **Máquinas diferentes** — o servidor do bot roda em um host (ex.: NAS, VM separada) e o Hermes em outro. O diretório de dados do servidor deve ser compartilhado com a máquina Hermes no **mesmo caminho absoluto** que o servidor reporta (tipicamente `/var/lib/telegram-bot-api`). NFS funciona bem para isso; CIFS/SMB com remapeamento de mount `uid=` é mais amigável se você não quiser lidar com incompatibilidades de uid no nível do filesystem.

Se `local_mode: true` está definido mas o Hermes não consegue fazer `stat` no caminho de arquivo retornado (permissões ou mount errado), python-telegram-bot silenciosamente recorre a um `getFile` HTTP contra o servidor local — que no modo `--local` responde com `404 Not Found`. O sintoma aparece em `gateway.log` como:

```
[Telegram] Failed to cache voice: Not Found
telegram.error.InvalidToken: Not Found
```

Se você vir isso, a elevação do limite está funcionando mas o compartilhamento de arquivos não. Verifique `ls -la /var/lib/telegram-bot-api/<TOKEN>/voice/` do host Hermes como o usuário que executa o gateway, e confirme que um único arquivo é `cat`-ável sem erro de permissão.

### Passo 6: Teste {#step-6-test-it}

Envie ao bot uma nota de voz ou arquivo de áudio maior que 20 MB. Acompanhe o log do gateway:

```bash
tail -f ~/.hermes/logs/gateway.log | grep -iE "telegram|cache"
```

Você deve ver uma linha `[Telegram] Cached user voice at /home/<user>/.hermes/cache/audio/...` e **nenhuma** rejeição "too large". Combinado com `stt.enabled: false` (acima), o caminho para o arquivo de áudio original então chega na mensagem de entrada do agente para processamento posterior.

## Uso em chats de grupo {#group-chat-usage}

O Hermes Agent funciona em chats de grupo do Telegram com algumas considerações:

- O **modo de privacidade** determina quais mensagens o bot pode ver (veja [Passo 3](#step-3-privacy-mode-critical-for-groups))
- `TELEGRAM_ALLOWED_USERS` ainda se aplica — apenas usuários autorizados podem acionar o bot, mesmo em grupos
- Você pode impedir que o bot responda a conversas ordinárias de grupo com `telegram.require_mention: true`
- Com `telegram.require_mention: true`, mensagens de grupo são aceitas quando são:
  - respostas a uma das mensagens do bot
  - menções `@botusername`
  - `/command@botusername` (forma de comando do menu de bot do Telegram que inclui o nome do bot)
  - correspondências para uma das suas palavras de ativação regex configuradas em `telegram.mention_patterns`
- Em grupos com múltiplos bots Hermes, `telegram.exclusive_bot_mentions` mantém o roteamento determinístico. Quando uma mensagem menciona explicitamente um ou mais usernames de bot do Telegram, apenas os perfis de bot mencionados a processam; outros bots Hermes a ignoram antes que fallbacks de resposta e palavra de ativação executem. Isso está ativado por padrão.
- Use `telegram.ignored_threads` para manter o Hermes silencioso em tópicos específicos de fórum do Telegram, mesmo quando o grupo permitiria respostas livres ou respostas acionadas por menção
- Se `telegram.require_mention` for deixado unset ou false, o Hermes mantém o comportamento anterior de grupo aberto e responde a mensagens normais de grupo que consegue ver

### Múltiplos bots Hermes em um grupo {#multiple-hermes-bots-in-one-group}

Se você executa vários perfis Hermes no mesmo grupo do Telegram, crie um token de bot do Telegram por perfil e inicie um gateway por perfil. Não reutilize o mesmo token de bot em múltiplos gateways em execução; o Telegram rejeitará polling concorrente para o mesmo token.

Config de grupo recomendada:

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns: []
```

Com esta configuração, uma mensagem de grupo como `@research_bot @ops_bot summarize this` é processada apenas por `research_bot` e `ops_bot`. Outros bots Hermes no grupo permanecem silenciosos, mesmo se a mensagem for resposta a uma de suas mensagens anteriores ou corresponder a uma palavra de ativação compartilhada.

Defina `exclusive_bot_mentions: false` apenas para grupos legados onde menções explícitas não devem substituir gatilhos de resposta e palavra de ativação.

Para operar vários perfis, execute o comando gateway uma vez por perfil. Por exemplo:

```bash
# default profile
hermes gateway start
hermes gateway status
hermes gateway stop

# named profiles
hermes -p research gateway start
hermes -p research gateway status
hermes -p research gateway stop
```

Para uma frota pequena e fixa, use um loop shell ou script que chama `hermes gateway <action>` para o perfil padrão e `hermes -p <profile> gateway <action>` para cada perfil nomeado. Isso é mais confiável do que assumir que um único comando em nível de processo controla todo perfil nomeado em todo service manager.

### Solução de problemas: funciona em DMs mas não em grupos {#troubleshooting-works-in-dms-but-not-groups}

Se o bot responde em chat privado mas permanece silencioso em um grupo, verifique estes
gates em ordem:

1. **Entrega do Telegram:** desative o modo de privacidade do BotFather, promova o bot a
   admin ou mencione o bot diretamente. O Hermes não pode responder a mensagens de grupo
   que o Telegram nunca entrega ao bot.
2. **Reentrar após alterar privacidade:** remova o bot do grupo e adicione-o
   novamente após alterar configurações de privacidade do BotFather. O Telegram pode manter o
   comportamento de entrega antigo para memberships existentes.
3. **Autorização Hermes:** certifique-se de que o remetente está listado em
   `TELEGRAM_ALLOWED_USERS` ou `TELEGRAM_GROUP_ALLOWED_USERS`, ou permita o
   chat de grupo com `TELEGRAM_GROUP_ALLOWED_CHATS`.
4. **Filtros de menção:** se `telegram.require_mention: true` está definido, conversas
   ordinárias de grupo são ignoradas a menos que a mensagem seja um slash command, resposta ao
   bot, menção `@botusername` ou correspondência de `mention_patterns` configurado.
5. **Roteamento multi-bot:** se um grupo contém vários bots, certifique-se de que cada
   perfil Hermes usa um token de bot único e mantenha `exclusive_bot_mentions`
   ativado a menos que você queira intencionalmente o comportamento legado de gatilho compartilhado.

IDs de chat negativos são normais para grupos e supergrupos do Telegram. Se você usa
autorização por escopo de chat, coloque esses IDs em `TELEGRAM_GROUP_ALLOWED_CHATS`, não
na allowlist de usuário remetente.

### Exemplo de configuração de gatilho de grupo {#example-group-trigger-configuration}

Adicione isto em `~/.hermes/config.yaml`:

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns:
    - "^\\s*chompy\\b"
  ignored_threads:
    - 31
    - "42"
```

Este exemplo permite todos os gatilhos diretos usuais mais mensagens que começam com `chompy`, mesmo sem `@mention`.
Mensagens nos tópicos do Telegram `31` e `42` são sempre ignoradas antes que as verificações de menção e resposta livre executem.

### Notas sobre `mention_patterns` {#notes-on-mention_patterns}

- Padrões usam expressões regulares Python
- A correspondência é case-insensitive
- Padrões são verificados tanto em mensagens de texto quanto em legendas de mídia
- Padrões regex inválidos são ignorados com um aviso nos logs do gateway em vez de derrubar o bot
- Se você quer que um padrão corresponda apenas no início de uma mensagem, ancora-o com `^`

## Tópicos de chat privado (Bot API 9.4) {#private-chat-topics-bot-api-94}

A Bot API 9.4 do Telegram (fevereiro de 2026) introduziu **Tópicos de Chat Privado** — bots podem criar threads de tópicos estilo fórum diretamente em chats DM 1-a-1, sem precisar de supergrupo. Isso permite executar múltiplos workspaces isolados dentro do seu DM existente com o Hermes.

### Caso de uso {#use-case}

Se você trabalha em vários projetos de longa duração, tópicos mantêm seu contexto separado:

- **Tópico "Website"** — trabalhe no seu serviço web de produção
- **Tópico "Research"** — revisão de literatura e exploração de papers
- **Tópico "General"** — tarefas diversas e perguntas rápidas

Cada tópico recebe sua própria sessão de conversa, histórico e contexto — completamente isolado dos outros.

### Configuração {#configuration-1}

:::caution Prerequisites
Antes de adicionar tópicos à sua config, o usuário deve **ativar o modo Topics** no chat DM com o bot:

1. Abra seu chat privado com o bot Hermes no Telegram
2. Toque no nome do bot no topo para abrir as informações do chat
3. Ative **Topics** (o toggle para transformar o chat em fórum)

Sem isso, o Hermes registrará `The chat is not a forum` na inicialização e pulará a criação de tópicos. Esta é uma configuração do lado do cliente Telegram — o bot não pode ativá-la programaticamente.
:::

Adicione tópicos em `platforms.telegram.extra.dm_topics` em `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      dm_topics:
      - chat_id: 123456789        # Your Telegram user ID
        topics:
        - name: General
          icon_color: 7322096
        - name: Website
          icon_color: 9367192
        - name: Research
          icon_color: 16766590
          skill: arxiv              # Auto-load a skill in this topic
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Nome de exibição do tópico |
| `icon_color` | No | Código de cor do ícone do Telegram (inteiro) |
| `icon_custom_emoji_id` | No | ID de emoji personalizado para o ícone do tópico |
| `skill` | No | Skill para auto-carregar em novas sessões neste tópico |
| `thread_id` | No | Preenchido automaticamente após criação do tópico — não defina manualmente |

### Como funciona {#how-it-works}

1. Na inicialização do gateway, o Hermes chama `createForumTopic` para cada tópico que ainda não tem `thread_id`
2. O `thread_id` é salvo de volta em `config.yaml` automaticamente — reinícios subsequentes pulam a chamada de API
3. Cada tópico mapeia para uma chave de sessão isolada: `agent:main:telegram:dm:{chat_id}:{thread_id}`
4. Mensagens em cada tópico têm seu próprio histórico de conversa, flush de memória e janela de contexto

### Tratamento do DM raiz {#root-dm-handling}

Por padrão, mensagens enviadas ao DM raiz (fora de qualquer tópico) são processadas
normalmente. Defina `ignore_root_dm: true` para transformar o DM raiz em um lobby — mensagens
normais são silenciosamente ignoradas para usuários que têm tópicos DM configurados, enquanto
comandos de sistema (`/start`, `/help`, `/status`, etc.) ainda funcionam.

```yaml
platforms:
  telegram:
    extra:
      ignore_root_dm: true
      dm_topics:
        - chat_id: 123456789
          topics:
            - name: General
```

A verificação é **por chat**: apenas usuários com pelo menos uma entrada em `dm_topics`
terão seu DM raiz afetado. Usuários sem tópicos configurados não são
afetados.

### Vinculação de skill {#skill-binding}

Tópicos com um campo `skill` carregam automaticamente essa skill quando uma nova sessão inicia no tópico. Isso funciona exatamente como digitar `/skill-name` no início de uma conversa — o conteúdo da skill é injetado na primeira mensagem, e mensagens subsequentes o veem no histórico da conversa.

Por exemplo, um tópico com `skill: arxiv` terá a skill arxiv pré-carregada sempre que sua sessão for resetada (por timeout de inatividade, reset diário ou `/reset` manual).

:::tip
Tópicos criados fora da config (ex.: chamando manualmente a API do Telegram) são descobertos automaticamente quando uma mensagem de serviço `forum_topic_created` chega. Você também pode adicionar tópicos à config enquanto o gateway está rodando — eles serão capturados no próximo cache miss.
:::

## Modo DM multi-sessão (`/topic`) {#multi-session-dm-mode-topic}

Um DM multi-sessão estilo ChatGPT — um bot, muitas conversas paralelas. Diferente dos `extra.dm_topics` curados pelo operador acima, este modo é **dirigido pelo usuário**: sem config, sem nomes de tópicos pré-declarados. O usuário final ativa com `/topic`, então toca no botão **+** do Telegram para criar quantos tópicos quiser, cada um uma sessão Hermes totalmente independente.

### Subcomandos `/topic` {#topic-subcommands}

| Form | Context | Effect |
|------|---------|--------|
| `/topic` | Root DM, not yet enabled | Verifica capacidades do BotFather, ativa modo multi-sessão, cria tópico System fixado |
| `/topic` | Root DM, already enabled | Mostra status: sessões não vinculadas disponíveis para restaurar |
| `/topic` | Inside a topic | Mostra a vinculação de sessão do tópico atual |
| `/topic help` | Any | Uso inline |
| `/topic off` | Root DM | Desativa modo multi-sessão e limpa todas as vinculações de tópico deste chat |
| `/topic <session-id>` | Inside a topic | Restaura uma sessão Telegram anterior no tópico atual |

Apenas usuários autorizados (allowlist via `TELEGRAM_ALLOWED_USERS` / config de auth da plataforma) podem executar `/topic`. Um remetente não autorizado recebe uma recusa em vez de ativação.

### Tópicos DM vs modo DM multi-sessão {#dm-topics-vs-multi-session-dm-mode}

| | `extra.dm_topics` (config-driven) | `/topic` (user-driven) |
|---|---|---|
| Who activates it | Operador, em `config.yaml` | Usuário final, enviando `/topic` |
| Topic list | Conjunto fixo declarado na config | Usuário cria/exclui tópicos livremente |
| Topic names | Escolhidos pelo operador | Escolhidos pelo usuário; renomeados automaticamente para corresponder ao título da sessão Hermes |
| Root DM behavior | Chat normal (lobby se `ignore_root_dm: true`) | Torna-se um lobby de sistema (mensagens que não são comandos são rejeitadas) |
| Primary use case | Workspaces permanentes com vinculação opcional de skill | Sessões paralelas ad hoc |
| Persistence | `extra.dm_topics` na config | Tabelas SQLite `telegram_dm_topic_mode` + `telegram_dm_topic_bindings` |

Ambos os recursos podem coexistir no mesmo bot — você executaria `/topic` a partir do DM de um usuário, e `extra.dm_topics` continua gerenciando tópicos declarados pelo operador para outros chats.

### Pré-requisitos {#prerequisites}

No **@BotFather**, abra seu bot → **Bot Settings → Threads Settings**:

1. Ative **Threaded Mode** (habilita `has_topics_enabled`)
2. **Não** desative usuários criando tópicos (mantém `allows_users_to_create_topics` ligado)

Quando o usuário executa `/topic` pela primeira vez, o Hermes chama `getMe` para verificar ambas as flags. Se alguma estiver desligada, o Hermes envia uma captura de tela da página Threads Settings do BotFather e explica o que alternar — nenhuma ativação acontece até que os pré-requisitos sejam atendidos.

### Fluxo de ativação {#activation-flow}

Do DM raiz, envie:

```
/topic
```

O Hermes irá:

1. Verificar `getMe().has_topics_enabled` e `allows_users_to_create_topics`
2. Se ambos forem true, ativar modo de tópicos multi-sessão para este DM
3. Criar e fixar um tópico **System** para status/comandos (best-effort)
4. Responder com uma lista de sessões Telegram anteriores não vinculadas que o usuário pode restaurar

Após a ativação, o **DM raiz é um lobby**: prompts normais são rejeitados com orientação apontando para **All Messages**. Comandos de sistema (`/status`, `/sessions`, `/usage`, `/help`, etc.) ainda funcionam na raiz.

### Criando um novo tópico (fluxo do usuário final) {#creating-a-new-topic-end-user-flow}

1. Abra o DM do bot no Telegram
2. Toque em **All Messages** no topo da interface do bot, então envie qualquer mensagem
3. O Telegram cria um novo tópico para essa mensagem
4. O Hermes responde dentro desse tópico — o tópico agora é uma sessão independente

Cada tópico recebe seu próprio histórico de conversa, estado do modelo, execução de ferramentas e ID de sessão. A chave de isolamento é `agent:main:telegram:dm:{chat_id}:{thread_id}` — idêntica ao isolamento de tópicos DM dirigido por config.

### Tópicos renomeados automaticamente {#auto-renamed-topics}

Quando o Hermes gera um título de sessão para um tópico (via pipeline de auto-título, após a primeira troca), o próprio tópico do Telegram é renomeado para corresponder — ex.: "New Topic" vira "Database migration plan". A renomeação é best-effort: falhas são registradas mas não quebram a sessão.

Para desativar isso e manter seus nomes de tópico escolhidos manualmente intactos, defina:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_topic_auto_rename: true
```

Quando esta flag está ligada, o Hermes ainda gera um título de sessão interno (usado por `hermes sessions`, TUI, etc.) mas nunca edita o nome do tópico do Telegram. Útil quando você organiza tópicos manualmente sob Threaded Mode do BotFather e não quer que toda primeira resposta sobrescreva o título.

### `/new` dentro de um tópico {#new-inside-a-topic}

Reseta a sessão do tópico atual (novo ID de sessão, histórico fresco) sem tocar outros tópicos. O Hermes responde com um lembrete de que para trabalho paralelo, criar outro tópico (via **All Messages**) é geralmente o que você quer.

### Restaurando uma sessão anterior {#restoring-a-previous-session}

Dentro de um tópico, envie:

```
/topic <session-id>
```

Isso vincula o tópico atual a uma sessão Hermes existente em vez de começar do zero. Útil para continuar uma conversa que começou antes do modo de tópicos ser ativado. Restrições:

- A sessão alvo deve pertencer ao mesmo usuário do Telegram
- A sessão alvo não deve já estar vinculada a outro tópico

O Hermes confirma com o título da sessão e reproduz a última mensagem do assistente para contexto.

Para descobrir IDs de sessão, envie `/topic` (sem argumento) no DM raiz — o Hermes lista as sessões Telegram não vinculadas do usuário.

### `/topic` dentro de um tópico (sem argumento) {#topic-inside-a-topic-no-argument}

Mostra a vinculação do tópico atual: título da sessão, ID da sessão e dicas para `/new` vs criar outro tópico.

### Por baixo dos panos {#under-the-hood}

- A ativação persiste em `telegram_dm_topic_mode(chat_id, user_id, enabled, ...)` em `state.db`
- Cada vinculação de tópico persiste em `telegram_dm_topic_bindings(chat_id, thread_id, session_id, ...)` com `ON DELETE CASCADE` em `session_id` — podar uma sessão limpa automaticamente sua vinculação de tópico
- A migração SQLite do modo de tópicos é **opt-in**: executa na primeira chamada `/topic`, nunca na inicialização do gateway. Até um usuário executar `/topic` neste perfil, `state.db` permanece inalterado
- Cada mensagem DM de entrada consulta sua vinculação `(chat_id, thread_id)`. Se presente, a consulta roteia a mensagem para a sessão vinculada via `SessionStore.switch_session()` para que o mapeamento session-key-to-session-id permaneça consistente em disco
- `/new` dentro de um tópico reescreve a linha de vinculação para apontar ao novo ID de sessão, então a próxima mensagem permanece na sessão fresca
- Tópicos declarados em `extra.dm_topics` **nunca são renomeados automaticamente** — o nome escolhido pelo operador é preservado mesmo quando o modo multi-sessão está ativado
- Defina `extra.disable_topic_auto_rename: true` para desativar auto-renomeação para **todos** os tópicos no chat (tópicos ad hoc criados via Threaded Mode incluídos)
- O tópico General (fixado no topo) em um DM com fórum habilitado é tratado como o lobby raiz, independentemente de o Telegram entregar suas mensagens com `message_thread_id=1` ou sem thread_id
- Lembretes do lobby raiz são limitados a uma mensagem a cada 30 segundos por chat — um usuário que esquece que o modo de tópicos está ligado e digita dez prompts na raiz não receberá dez respostas
- Capturas de tela de configuração do BotFather são limitadas a um envio a cada 5 minutos por chat — tentativas repetidas de `/topic` enquanto Threads Settings ainda estiver desabilitado não reenviarão a mesma imagem
- `/background <prompt>` iniciado dentro de um tópico entrega seu resultado de volta ao mesmo tópico; sessões em background não disparam auto-renomeação do tópico proprietário
- `/topic` em si é controlado pela verificação de autorização de usuário do bot — DMs não autorizados recebem recusa em vez de ativação

### Desativando modo multi-sessão {#disabling-multi-session-mode}

Envie `/topic off` no DM raiz. O Hermes desliga a linha, limpa as vinculações `(thread_id → session_id)` do chat, e o DM raiz volta a ser um chat Hermes normal. Tópicos existentes no Telegram não são excluídos — eles apenas param de ser controlados como sessões independentes. Execute `/topic` novamente depois para reativar.

Se você precisar limpar manualmente (ex.: reset em massa em muitos chats), remova as linhas diretamente:

```bash
sqlite3 ~/.hermes/state.db \
  "UPDATE telegram_dm_topic_mode SET enabled = 0 WHERE chat_id = '<your_chat_id>'; \
   DELETE FROM telegram_dm_topic_bindings WHERE chat_id = '<your_chat_id>';"
```

### Fazendo downgrade do Hermes {#downgrading-hermes}

Se você fizer downgrade para uma versão Hermes anterior a `/topic`, o recurso simplesmente para de funcionar — as tabelas `telegram_dm_topic_mode` e `telegram_dm_topic_bindings` permanecem em `state.db` mas são ignoradas por código mais antigo. DMs voltam ao isolamento nativo por thread (cada `message_thread_id` ainda recebe sua própria sessão via `build_session_key`), então seus tópicos Telegram existentes continuam funcionando como sessões paralelas. O DM raiz deixa de ser um lobby — mensagens lá vão para o agente como antes. Re-upgrading reativa o modo multi-sessão exatamente onde estava.

## Vinculação de skill em tópicos de fórum de grupo {#group-forum-topic-skill-binding}

Supergrupos com **modo Topics** ativado (também chamado "forum topics") já recebem isolamento de sessão por tópico — cada `thread_id` mapeia para sua própria conversa. Mas você pode querer **auto-carregar uma skill** quando mensagens chegam em um tópico de grupo específico, assim como funciona a vinculação de skill em tópicos DM.

### Caso de uso {#use-case-1}

Um supergrupo de equipe com tópicos de fórum para diferentes fluxos de trabalho:

- Tópico **Engineering** → auto-carrega a skill `software-development`
- Tópico **Research** → auto-carrega a skill `arxiv`
- Tópico **General** → sem skill, assistente de propósito geral

### Configuração {#configuration-2}

Adicione vinculações de tópico em `platforms.telegram.extra.group_topics` em `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      group_topics:
      - chat_id: -1001234567890       # Supergroup ID
        topics:
        - name: Engineering
          thread_id: 5
          skill: software-development
        - name: Research
          thread_id: 12
          skill: arxiv
        - name: General
          thread_id: 1
          # No skill — general purpose
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `chat_id` | Yes | ID numérico do supergrupo (número negativo começando com `-100`) |
| `name` | No | Rótulo legível do tópico (apenas informativo) |
| `thread_id` | Yes | ID do tópico de fórum do Telegram — visível em links `t.me/c/<group_id>/<thread_id>` |
| `skill` | No | Skill para auto-carregar em novas sessões neste tópico |

### Como funciona {#how-it-works-1}

1. Quando uma mensagem chega em um tópico de grupo mapeado, o Hermes consulta `chat_id` e `thread_id` na config `group_topics`
2. Se uma entrada correspondente tem um campo `skill`, essa skill é auto-carregada para a sessão — idêntico à vinculação de skill em tópicos DM
3. Tópicos sem chave `skill` recebem apenas isolamento de sessão (comportamento existente, inalterado)
4. Valores `thread_id` ou `chat_id` não mapeados passam silenciosamente — sem erro, sem skill

### Diferenças dos tópicos DM {#differences-from-dm-topics}

| | DM Topics | Group Topics |
|---|---|---|
| Config key | `extra.dm_topics` | `extra.group_topics` |
| Topic creation | Hermes cria tópicos via API se `thread_id` estiver ausente | Admin cria tópicos na UI do Telegram |
| `thread_id` | Preenchido automaticamente após criação | Deve ser definido manualmente |
| `icon_color` / `icon_custom_emoji_id` | Suportado | Não aplicável (admin controla aparência) |
| Skill binding | ✓ | ✓ |
| Session isolation | ✓ | ✓ (já integrado para tópicos de fórum) |

:::tip
Para encontrar o `thread_id` de um tópico, abra o tópico no Telegram Web ou Desktop e olhe a URL: `https://t.me/c/1234567890/5` — o último número (`5`) é o `thread_id`. O `chat_id` para supergrupos é o ID do grupo prefixado com `-100` (ex.: grupo `1234567890` vira `-1001234567890`).
:::

## Recursos recentes da Bot API {#recent-bot-api-features}

- **Bot API 9.4 (Fev 2026):** Tópicos de Chat Privado — bots podem criar tópicos de fórum em chats DM 1-a-1 via `createForumTopic`. O Hermes usa isso para dois recursos distintos: [Tópicos de Chat Privado](#private-chat-topics-bot-api-94) curados pelo operador (dirigido por config, lista fixa de tópicos) e [Modo DM multi-sessão](#multi-session-dm-mode-topic) dirigido pelo usuário (ativado por `/topic`, tópicos ilimitados criados pelo usuário).
- **Política de privacidade:** O Telegram agora exige que bots tenham uma política de privacidade. Defina uma via BotFather com `/setprivacy_policy`, ou o Telegram pode gerar automaticamente um placeholder. Isso é particularmente importante se seu bot é público.
- **Bot API 9.5 (Mar 2026): Streaming nativo via `sendMessageDraft`.** O Hermes suporta a API nativa de streaming-draft do Telegram como transporte opt-in para chats privados. O padrão permanece o caminho legado `editMessageText` porque previews de draft podem colapsar e re-renderizar visivelmente em alguns clientes Telegram.

### Transporte de streaming (`gateway.streaming.transport`) {#streaming-transport-gatewaystreamingtransport}

Quando streaming está ativado (`gateway.streaming.enabled: true`), o Hermes escolhe um de quatro transportes:

| Value | Behaviour |
|---|---|
| `auto` (default) | Streaming draft nativo em chats suportados (atualmente DMs Telegram); caminho legado baseado em edit caso contrário. Recorre graciosamente se um frame de draft falhar. |
| `draft` | Força drafts nativos. Registra downgrade e recorre a edit se o chat não suporta drafts (ex. grupos/tópicos). |
| `edit` | Polling progressivo legado `editMessageText` para todo tipo de chat. |
| `off` | Desativa streaming completamente (apenas resposta final, sem atualizações progressivas). |

Em `~/.hermes/config.yaml`:

```yaml
gateway:
  streaming:
    enabled: true
    transport: auto    # auto | draft | edit | off
```

**O que você verá em DMs com `edit` (padrão)** — o gateway envia uma mensagem de preview normal e a atualiza progressivamente via `editMessageText`, evitando o efeito de colapso/rollback do preview de draft do Telegram.

**O que você verá em DMs com `auto` ou `draft`** — o Telegram mostra um preview de draft animado que atualiza token a token. Quando a resposta termina, é entregue como mensagem regular e o preview de draft limpa naturalmente no cliente. Drafts não têm message id, então a resposta final é o que permanece no histórico do chat.

**E quanto a grupos, supergrupos, tópicos de fórum?** O Telegram restringe `sendMessageDraft` a chats privados (DMs). O gateway recorre transparentemente ao caminho baseado em edit para todo o resto — mesma UX de antes.

**E se um frame de draft falhar?** Qualquer falha (erro de rede transitório, rejeição do servidor, instalação antiga de python-telegram-bot) reverte essa resposta ao caminho baseado em edit pelo resto do stream. A próxima resposta recebe uma nova tentativa.

## Renderização: mensagens ricas, tabelas e previews de link {#rendering-rich-messages-tables-and-link-previews}

**Mensagens ricas (Bot API 10.1).** Respostas finais que contêm construtos que o caminho legado MarkdownV2 degrada — tabelas, listas de tarefas, `<details>` colapsáveis e matemática em bloco — são enviadas com [`sendRichMessage`](https://core.telegram.org/bots/api#sendrichmessage) nativo do Telegram usando o **markdown bruto** do agente, então renderizam nativamente sem achatamento do lado do cliente. Durante streaming, a resposta final é entregue **editando o preview existente in-place** via parâmetro `rich_message` do `editMessageText` — sem segunda mensagem, sem delete, então não há flicker de entrega duplicada no fim de um turno. Em DMs o preview de streaming ao vivo também usa `sendRichMessageDraft`, então o draft animado corresponde à mensagem rica final. Respostas ordinárias (prosa simples, negrito/itálico, listas simples) permanecem no caminho MarkdownV2 para peso de fonte e espaçamento consistentes entre clientes.

O caminho rico é pulado automaticamente quando o conteúdo excede o limite de 32.768 caracteres de rich text, e qualquer rejeição do Telegram (endpoint não suportado em python-telegram-bot mais antigo, erro de parser, blocos/colunas oversized) **recorre transparentemente** ao caminho MarkdownV2 — sua mensagem nunca se perde. Erros transitórios/de rede *não* são reenviados silenciosamente (sem mensagem final duplicada).

**Fallback MarkdownV2.** Quando o caminho rico não está disponível para uma mensagem, o Hermes converte markdown para MarkdownV2. Como MarkdownV2 não tem sintaxe nativa de tabela, tabelas pipe são normalizadas:

- **Tabelas pequenas** são achatadas em **bullets de grupo de linha** — cada linha vira uma lista com bullets legível sob os cabeçalhos de coluna. Bom para 2–4 colunas e células curtas.
- **Tabelas maiores ou mais largas** recorrem a um **bloco de código fenced** com colunas alinhadas para que nada colapse.

Mensagens ricas são **opt-in**. O padrão permanece no caminho legado MarkdownV2 porque clientes Telegram atuais podem tornar mensagens ricas da Bot API difíceis de copiar como texto simples, o que é especialmente doloroso para snippets de comando e handoffs mobile. Para habilitar renderização nativa de tabelas/listas de tarefas/details/matemática:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        rich_messages: true
        rich_drafts: false
```

Esta configuração é para compatibilidade de renderização/cópia no cliente; o Hermes já recorre automaticamente quando o Telegram rejeita a chamada da API rica. `rich_drafts` controla o caminho experimental de preview de draft rico durante streaming de DM Telegram e permanece desligado por padrão porque Telegram Desktop/macOS pode sobrepor visualmente frames de draft rico até o chat redesenhar. Se você quer apenas o comportamento legado "sempre code-block" de tabelas mantendo mensagens ricas ativadas, desative normalização de tabelas definindo `telegram.pretty_tables: false` em `config.yaml` (padrão: `true`).

**Previews de link.** O Telegram gera automaticamente previews de link para URLs em mensagens de bot. Se você preferir suprimi-los (saída longa de `/tools`, resposta do agente que menciona dez links, etc.):

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_link_previews: true
```

Quando ativado, o Hermes anexa `LinkPreviewOptions(is_disabled=True)` do Telegram a toda mensagem de saída e recorre ao parâmetro legado `disable_web_page_preview` em versões mais antigas de python-telegram-bot.

## Allowlist de grupos {#group-allowlisting}

Grupos e chats de fórum do Telegram têm dois gates ortogonais que você pode configurar:

- **IDs de usuário remetente** (`group_allow_from` / `TELEGRAM_GROUP_ALLOWED_USERS`) — allowlist por escopo de remetente que se aplica apenas a mensagens de grupo/fórum. Use quando quer que usuários específicos possam invocar o bot em grupos sem adicioná-los a `TELEGRAM_ALLOWED_USERS` (o que também daria acesso a DM).
- **IDs de chat** (`group_allowed_chats` / `TELEGRAM_GROUP_ALLOWED_CHATS`) — allowlist por escopo de chat. Qualquer membro desses grupos/fóruns pode interagir com o bot. Útil para bots de equipe/suporte onde a própria membership do grupo é o sinal de acesso.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # Global access (DMs + groups). Users here can always invoke the bot.
        allow_from:
          - "123456789"
        # Sender IDs allowed in groups/forums only. Does NOT grant DM access.
        group_allow_from:
          - "987654321"
        # Entire groups/forums — any member is authorized.
        group_allowed_chats:
          - "-1001234567890"
```

Variáveis de ambiente equivalentes:

```bash
TELEGRAM_ALLOWED_USERS="123456789"
TELEGRAM_GROUP_ALLOWED_USERS="987654321"
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

Comportamento:

- `TELEGRAM_ALLOWED_USERS` cobre todos os tipos de chat (DMs, grupos, fóruns).
- `TELEGRAM_GROUP_ALLOWED_USERS` autoriza apenas os remetentes listados em grupos/fóruns. Eles ainda não podem enviar DM ao bot a menos que estejam listados em `TELEGRAM_ALLOWED_USERS`.
- Um chat em `TELEGRAM_GROUP_ALLOWED_CHATS` autoriza todo membro desse chat, independentemente do remetente.
- Use `*` em qualquer um destes para permitir qualquer remetente/chat.
- Isso se sobrepõe a gatilhos existentes de menção/padrão e a `group_topics` + `ignored_threads`.

### Migração de antes do PR #17686 {#migration-from-before-pr-17686}

Antes desta divisão, `TELEGRAM_GROUP_ALLOWED_USERS` era o único controle e usuários colocavam **IDs de chat** nele. Por compatibilidade retroativa, valores com formato de ID de chat (começando com `-`) em `TELEGRAM_GROUP_ALLOWED_USERS` ainda são honrados como IDs de chat e um aviso de depreciação é registrado uma vez. Migração:

```bash
# Old (still works, but deprecated)
TELEGRAM_GROUP_ALLOWED_USERS="-1001234567890"

# New
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

### Bypass de @mention para convidados (`guest_mode`) {#guest-mention-bypass-guest_mode}

Em uma config típica, `group_allowed_chats` é um gate rígido: mensagens de grupos fora da lista são silenciosamente descartadas, mesmo se um membro @mencionar explicitamente o bot. Esse é o padrão certo para bots de suporte/equipe.

Para configs mais casuais — chats de grupo de amigos onde você quer o bot **majoritariamente silencioso** mas **ocasionalmente disponível em ping explícito** — ative `guest_mode`:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        group_allowed_chats:
          - "-1001234567890"   # your main allowlisted group
        guest_mode: true       # non-allowlisted groups: allow on @mention only
```

Equivalente em env:

```bash
TELEGRAM_GUEST_MODE=true
```

Padrão: `false`.

Com `guest_mode: true`, uma mensagem de um grupo não allowlisted é processada **apenas** se @mencionar explicitamente o bot. A menção é exigida a cada turno — não há stickiness de sessão para interações de convidado, então o bot nunca se engaja automaticamente em um thread de grupo de amigos em que não foi pingado.

DMs e grupos allowlisted se comportam exatamente como antes.

## Controle de acesso a slash commands {#slash-command-access-control}

Por padrão, todo usuário permitido pode executar todo slash command. Para dividir sua allowlist em **admins** (acesso completo a slash commands) e **usuários regulares** (apenas comandos que você habilita explicitamente), adicione `allow_admin_from` e `user_allowed_commands` ao bloco `extra` da plataforma:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # Existing allowlists (unchanged)
        allow_from:
          - "123456789"     # admin
          - "555555555"     # regular user
          - "777777777"     # regular user

        # NEW — admins get all slash commands (built-in + plugin)
        allow_admin_from:
          - "123456789"

        # NEW — non-admin allowed users can only run these slash commands.
        # /help and /whoami are always allowed so users can see their access.
        user_allowed_commands:
          - status
          - model
          - history

        # Optional: separate admin/command lists for groups
        group_allow_admin_from:
          - "123456789"
        group_user_allowed_commands:
          - status
```

**Comportamento:**

- Um usuário listado em `allow_admin_from` para um escopo (DM ou grupo) pode executar **todo** slash command registrado — comandos integrados E registrados por plugin — através do registro ao vivo.
- Um usuário em `allow_from` mas **não** em `allow_admin_from` só pode executar comandos listados em `user_allowed_commands`, mais o piso sempre permitido: `/help` e `/whoami`.
- Chat simples (mensagens que não são slash) não é afetado. Usuários não-admin ainda podem conversar com o agente normalmente, apenas não podem acionar comandos arbitrários.
- **Compat retroativa:** se `allow_admin_from` não está definido para um escopo, o controle de slash commands está desativado para esse escopo. Instalações existentes continuam funcionando sem mudanças.
- Status de admin em DM não implica status de admin em grupo. Cada escopo tem sua própria lista de admin.
- Se apenas `group_allow_admin_from` está definido, o escopo DM permanece em modo irrestrito (compat retroativa).

Use `/whoami` para ver o escopo ativo, seu nível (admin / user / unrestricted) e quais slash commands você pode executar.

## Seletor interativo de modelo {#interactive-model-picker}

Quando você envia `/model` sem argumentos em um chat Telegram, o Hermes mostra um teclado inline interativo para trocar modelos:

1. **Seleção de provedor** — botões mostrando cada provedor disponível com contagem de modelos (ex.: "OpenAI (15)", "✓ Anthropic (12)" para o provedor atual).
2. **Seleção de modelo** — lista paginada de modelos com navegação **Prev**/**Next**, botão **Back** para voltar aos provedores e **Cancel**.

O modelo e provedor atuais são exibidos no topo. Toda navegação acontece editando a mesma mensagem in-place (sem poluir o chat).

:::tip
Se você sabe o nome exato do modelo, digite `/model <name>` diretamente para pular o seletor. Você também pode digitar `/model <name> --global` para persistir a mudança entre sessões.
:::

## IPs de fallback DNS-over-HTTPS {#dns-over-https-fallback-ips}

Em algumas redes restritas, `api.telegram.org` pode resolver para um IP inacessível. O adaptador Telegram inclui um mecanismo de **IP de fallback** que tenta novamente conexões transparentemente contra IPs alternativos preservando o hostname TLS e SNI corretos.

### Como funciona {#how-it-works-2}

1. Se `TELEGRAM_FALLBACK_IPS` está definido, esses IPs são usados diretamente.
2. Caso contrário, o adaptador consulta automaticamente **Google DNS** e **Cloudflare DNS** via DNS-over-HTTPS (DoH) para descobrir IPs alternativos para `api.telegram.org`.
3. IPs retornados por DoH que diferem do resultado DNS do sistema são usados como fallbacks.
4. Se DoH também estiver bloqueado, um IP seed hardcoded (`149.154.167.220`) é usado como último recurso.
5. Uma vez que um IP de fallback tem sucesso, ele se torna "sticky" — requisições subsequentes o usam diretamente sem tentar o caminho primário primeiro.

### Configuração {#configuration-3}

```bash
# Explicit fallback IPs (comma-separated)
TELEGRAM_FALLBACK_IPS=149.154.167.220,149.154.167.221
```

Ou em `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      fallback_ips:
        - "149.154.167.220"
```

:::tip
Você geralmente não precisa configurar isso manualmente. A auto-descoberta via DoH lida com a maioria dos cenários de rede restrita. A variável de ambiente `TELEGRAM_FALLBACK_IPS` só é necessária se DoH também estiver bloqueado na sua rede.
:::

## Suporte a proxy {#proxy-support-1}

Se sua rede exige um proxy HTTP para alcançar a internet (comum em ambientes corporativos), o adaptador Telegram lê automaticamente variáveis de ambiente de proxy padrão e roteia todas as conexões pelo proxy.

### Variáveis suportadas {#supported-variables}

O adaptador verifica estas variáveis de ambiente em ordem, usando a primeira que estiver definida:

1. `HTTPS_PROXY`
2. `HTTP_PROXY`
3. `ALL_PROXY`
4. `https_proxy` / `http_proxy` / `all_proxy` (variantes minúsculas)

### Configuração {#configuration-4}

Defina o proxy no seu ambiente antes de iniciar o gateway:

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
hermes gateway
```

Ou adicione em `~/.hermes/.env`:

```bash
HTTPS_PROXY=http://proxy.example.com:8080
```

O proxy se aplica tanto ao transporte primário quanto a todos os transportes de IP de fallback. Nenhuma configuração Hermes adicional é necessária — se a variável de ambiente estiver definida, é usada automaticamente.

:::note
Isso cobre a camada de transporte de fallback personalizada que o Hermes usa para conexões Telegram. O cliente `httpx` padrão usado em outros lugares já respeita variáveis de proxy nativamente.
:::

## Reações a mensagens {#message-reactions}

O bot pode adicionar reações emoji a mensagens como feedback visual de processamento:

- 👀 quando o bot começa a processar sua mensagem
- ✅ quando a resposta é entregue com sucesso
- ❌ se ocorrer um erro durante o processamento

Reações estão **desativadas por padrão**. Ative em `config.yaml`:

```yaml
telegram:
  reactions: true
```

Ou via variável de ambiente:

```bash
TELEGRAM_REACTIONS=true
```

:::note
Diferente do Discord (onde reações são aditivas), a Bot API do Telegram substitui todas as reações do bot em uma única chamada. A transição de 👀 para ✅/❌ acontece atomicamente — você não verá ambas ao mesmo tempo.
:::

:::tip
Se o bot não tem permissão para adicionar reações em um grupo, as chamadas de reação falham silenciosamente e o processamento de mensagens continua normalmente.
:::

## Prompts por canal {#per-channel-prompts}

Atribua prompts de sistema efêmeros a grupos Telegram específicos ou tópicos de fórum. O prompt é injetado em runtime a cada turno — nunca persistido no histórico da transcrição — então mudanças têm efeito imediato.

```yaml
telegram:
  channel_prompts:
    "-1001234567890": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "42":  |
      This topic is for creative writing feedback. Be warm and
      constructive.
```

As chaves são IDs de chat (grupos/supergrupos) ou IDs de tópico de fórum. Para grupos de fórum, prompts em nível de tópico substituem o prompt em nível de grupo:

- Mensagem no tópico `42` dentro do grupo `-1001234567890` → usa o prompt do tópico `42`
- Mensagem no tópico `99` (sem entrada explícita) → recorre ao prompt do grupo `-1001234567890`
- Mensagem em um grupo sem entrada → nenhum prompt de canal aplicado

Chaves YAML numéricas são automaticamente normalizadas para strings.

## Solução de problemas {#troubleshooting}

| Problem | Solution |
|---------|----------|
| Bot não responde de forma alguma | Verifique se `TELEGRAM_BOT_TOKEN` está correto. Confira os logs de `hermes gateway` por erros. |
| Bot responde com "unauthorized" | Seu ID de usuário não está em `TELEGRAM_ALLOWED_USERS`. Confira novamente com @userinfobot. |
| Bot ignora mensagens de grupo | Modo de privacidade provavelmente está ligado. Desative-o (Passo 3) ou torne o bot admin do grupo. **Lembre-se de remover e readicionar o bot após alterar a privacidade.** |
| Mensagens de voz não transcritas | Verifique se STT está disponível: instale `faster-whisper` para transcrição local, ou defina `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY` em `~/.hermes/.env`. |
| Respostas de voz são arquivos, não bolhas | Instale `ffmpeg` (necessário para conversão Opus do Edge TTS). |
| Token de bot revogado/inválido | Gere um novo token via `/revoke` depois `/newbot` ou `/token` no BotFather. Atualize seu arquivo `.env`. |
| Webhook não recebe atualizações | Verifique se `TELEGRAM_WEBHOOK_URL` é publicamente acessível (teste com `curl`). Certifique-se de que sua plataforma/reverse proxy roteia tráfego HTTPS de entrada da porta da URL para a porta local de escuta configurada por `TELEGRAM_WEBHOOK_PORT` (não precisam ser o mesmo número). Certifique-se de que SSL/TLS está ativo — o Telegram só envia para URLs HTTPS. Verifique regras de firewall. |

## Aprovação de exec {#exec-approval}

Quando o agente tenta executar um comando potencialmente perigoso, ele pede sua aprovação no chat:

> ⚠️ This command is potentially dangerous (recursive delete). Reply "yes" to approve.

Responda "yes"/"y" para aprovar ou "no"/"n" para negar.

## Prompts interativos (clarify) {#interactive-prompts-clarify}

Quando o agente chama a ferramenta `clarify` — para perguntar qual abordagem você prefere, obter feedback pós-tarefa ou verificar antes de uma decisão não trivial — o Telegram renderiza a pergunta com **botões de teclado inline**:

> ❓ Which framework should I use for the dashboard?
>
> [1. Next.js] [2. Remix] [3. Astro]
> [✏️ Other (type answer)]

Toque em um botão para responder, ou toque em **Other** para digitar uma resposta livre (a próxima mensagem que você enviar se torna a resposta). Chamadas `clarify` abertas (sem escolhas predefinidas) pulam os botões e apenas capturam sua próxima mensagem.

Configure o timeout de resposta via `agent.clarify_timeout` em `~/.hermes/config.yaml` (padrão `600` segundos). Se você não responder dentro do timeout, o agente desbloqueia com uma mensagem sentinela e se adapta em vez de travar.

## Volume de notificações push {#push-notification-volume}

O Telegram dispara uma notificação push a cada mensagem que o bot envia. Para turnos longos do agente que emitem bolhas de progresso de ferramenta, atualizações de streaming e callbacks de status, isso fica barulhento rapidamente. O adaptador Telegram tem dois modos de notificação:

| Mode | Behavior |
|------|----------|
| `important` (default) | Apenas **respostas finais**, **prompts de aprovação** e **confirmações de slash command** tocam. Progresso de ferramenta, chunks de streaming e mensagens de status são entregues com `disable_notification=true`. |
| `all` | Toda mensagem de saída dispara uma notificação push. Comportamento legado; opt-in se você genuinamente quer ouvir sobre toda chamada de ferramenta. |

Configure em `~/.hermes/config.yaml`:

```yaml
display:
  platforms:
    telegram:
      notifications: important   # or "all"
```

Override em env (útil para teste A/B rápido):

```bash
HERMES_TELEGRAM_NOTIFICATIONS=all
```

Valores desconhecidos registram um aviso e recorrem a `important`.

## Mensagens de status editadas in-place {#status-messages-edited-in-place}

O adaptador Telegram roteia callbacks recorrentes de status do agente (ex. "Compressing context…", "Calling tool…") através de `send_or_update_status()`, que mantém um cache `{(chat_id, status_key) → message_id}` e **edita a bolha existente** em emissões subsequentes em vez de anexar uma nova a cada vez. Valores `status_key` distintos recebem suas próprias mensagens; chats distintos nunca colidem. Se a edição falhar (ex. o usuário excluiu a mensagem, ou é mais antiga do que o Telegram permite para edições), a entrada do cache é descartada e a próxima emissão posta uma mensagem fresca e re-cacheia seu ID. Nenhuma config necessária — este é o comportamento padrão do Telegram. Outros adaptadores que não implementam `send_or_update_status` recorrem a `send()` simples inalterado.

## Fixar mensagem de usuário recebida durante turno do agente {#pin-incoming-user-message-during-agent-turn}

Quando um usuário envia uma mensagem que dispara um turno do agente, o adaptador Telegram fixa essa mensagem recebida pela duração do turno e a desfixa quando a resposta termina — um indicador visual leve de que o bot está trabalhando ativamente na mensagem em vez de ignorá-la. A fixação usa `disable_notification=true` para evitar pings extras. Nenhuma config necessária.

## Segurança {#security}

:::warning
Sempre defina `TELEGRAM_ALLOWED_USERS` para restringir quem pode interagir com seu bot. Sem isso, o gateway nega todos os usuários por padrão como medida de segurança.
:::

Nunca compartilhe seu token de bot publicamente. Se comprometido, revogue imediatamente via comando `/revoke` do BotFather.

Para mais detalhes, veja a [documentação de Segurança](/user-guide/security). Você também pode usar [pareamento de DM](/user-guide/messaging#dm-pairing-alternative-to-allowlists) para uma abordagem mais dinâmica de autorização de usuários.
