---
sidebar_position: 4
title: "Tutorial: Assistente de Equipe no Telegram"
description: "Guia passo a passo para configurar um bot do Telegram que toda a sua equipe pode usar para ajuda com código, pesquisa, administração de sistemas e muito mais"
---

# Configure um Assistente de Equipe no Telegram

Este tutorial mostra como configurar um bot do Telegram alimentado pelo Hermes Agent que vários membros da equipe podem usar. Ao final, sua equipe terá um assistente de IA compartilhado para pedir ajuda com código, pesquisa, administração de sistemas e qualquer outra coisa — protegido com autorização por usuário.

## O Que Vamos Construir {#what-were-building}

Um bot do Telegram que:

- **Qualquer membro autorizado da equipe** pode mandar mensagem direta pedindo ajuda — revisões de código, pesquisa, comandos de shell, depuração
- **Roda no seu servidor** com acesso total a ferramentas — terminal, edição de arquivos, pesquisa na web, execução de código
- **Sessões por usuário** — cada pessoa tem seu próprio contexto de conversa
- **Seguro por padrão** — apenas usuários aprovados podem interagir, com dois métodos de autorização
- **Tarefas agendadas** — reuniões diárias (standups), verificações de saúde e lembretes entregues a um canal da equipe

---

## Pré-requisitos {#prerequisites}

Antes de começar, certifique-se de que você tem:

- **Hermes Agent instalado** em um servidor ou VPS (não no seu notebook — o bot precisa continuar rodando). Siga o [guia de instalação](/getting-started/installation) se ainda não o fez.
- **Uma conta no Telegram** para você (o proprietário do bot)
- **Um provedor de LLM configurado** — no mínimo, uma chave de API para OpenAI, Anthropic ou outro provedor suportado em `~/.hermes/.env`

:::tip
Um VPS de US$ 5/mês já é suficiente para rodar o gateway. O próprio Hermes é leve — as chamadas de API do LLM são o que custa dinheiro, e elas acontecem remotamente.
:::

---

## Passo 1: Crie um Bot no Telegram {#step-1-create-a-telegram-bot}

Todo bot do Telegram começa com o **@BotFather** — o bot oficial do Telegram para criar bots.

1. **Abra o Telegram** e procure por `@BotFather`, ou acesse [t.me/BotFather](https://t.me/BotFather)

2. **Envie `/newbot`** — o BotFather vai perguntar duas coisas:
   - **Nome de exibição** — o que os usuários veem (ex.: `Team Hermes Assistant`)
   - **Nome de usuário** — deve terminar em `bot` (ex.: `myteam_hermes_bot`)

3. **Copie o token do bot** — o BotFather responde com algo como:
   ```
   Use this token to access the HTTP API:
   7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...
   ```
   Salve esse token — você vai precisar dele no próximo passo.

4. **Defina uma descrição** (opcional, mas recomendado):
   ```
   /setdescription
   ```
   Escolha seu bot e digite algo como:
   ```
   Team AI assistant powered by Hermes Agent. DM me for help with code, research, debugging, and more.
   ```

5. **Defina comandos do bot** (opcional — dá aos usuários um menu de comandos):
   ```
   /setcommands
   ```
   Escolha seu bot e cole:
   ```
   new - Start a fresh conversation
   model - Show or change the AI model
   status - Show session info
   help - Show available commands
   stop - Stop the current task
   ```

:::warning
Mantenha o token do seu bot em segredo. Qualquer pessoa com o token pode controlar o bot. Se ele for exposto, use `/revoke` no BotFather para gerar um novo.
:::

---

## Passo 2: Configure o Gateway {#step-2-configure-the-gateway}

Você tem duas opções: o assistente de configuração interativo (recomendado) ou a configuração manual.

### Opção A: Configuração Interativa (Recomendado) {#option-a-interactive-setup-recommended}

```bash
hermes gateway setup
```

Isso guia você por tudo com seleção por teclas de seta. Escolha **Telegram**, cole o token do seu bot e digite seu ID de usuário quando solicitado.

### Opção B: Configuração Manual {#option-b-manual-configuration}

Adicione estas linhas ao `~/.hermes/.env`:

```bash
# Telegram bot token from BotFather
TELEGRAM_BOT_TOKEN=7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...

# Your Telegram user ID (numeric)
TELEGRAM_ALLOWED_USERS=123456789
```

### Encontrando Seu ID de Usuário {#finding-your-user-id}

Seu ID de usuário do Telegram é um valor numérico (não seu nome de usuário). Para encontrá-lo:

1. Envie uma mensagem para [@userinfobot](https://t.me/userinfobot) no Telegram
2. Ele responde instantaneamente com seu ID de usuário numérico
3. Copie esse número em `TELEGRAM_ALLOWED_USERS`

:::info
IDs de usuário do Telegram são números permanentes como `123456789`. Eles são diferentes do seu `@username`, que pode mudar. Sempre use o ID numérico nas listas de permissão.
:::

---

## Passo 3: Inicie o Gateway {#step-3-start-the-gateway}

### Teste Rápido {#quick-test}

Rode o gateway em primeiro plano primeiro para garantir que tudo funciona:

```bash
hermes gateway
```

Você deve ver uma saída parecida com:

```
[Gateway] Starting Hermes Gateway...
[Gateway] Telegram adapter connected
[Gateway] Cron scheduler started (tick every 60s)
```

Abra o Telegram, encontre seu bot e envie uma mensagem. Se ele responder, está tudo funcionando. Pressione `Ctrl+C` para parar.

### Produção: Instale como um Serviço {#production-install-as-a-service}

Para uma implantação persistente que sobrevive a reinicializações:

```bash
hermes gateway install
sudo hermes gateway install --system   # Linux only: boot-time system service
```

Isso cria um serviço em segundo plano: um serviço de usuário **systemd** no Linux por padrão, um serviço **launchd** no macOS, ou um serviço de sistema Linux iniciado no boot se você passar `--system`.

```bash
# Linux — manage the default user service
hermes gateway start
hermes gateway stop
hermes gateway status

# View live logs
journalctl --user -u hermes-gateway -f

# Keep running after SSH logout
sudo loginctl enable-linger $USER

# Linux servers — explicit system-service commands
sudo hermes gateway start --system
sudo hermes gateway status --system
journalctl -u hermes-gateway -f
```

```bash
# macOS — manage the service
hermes gateway start
hermes gateway stop
tail -f ~/.hermes/logs/gateway.log
```

:::tip macOS PATH
O plist do launchd captura o PATH do seu shell no momento da instalação, para que os subprocessos do gateway encontrem ferramentas como Node.js e ffmpeg. Se você instalar novas ferramentas depois, execute novamente `hermes gateway install` para atualizar o plist.
:::

### Verifique se Está Rodando {#verify-its-running}

```bash
hermes gateway status
```

Depois envie uma mensagem de teste para o seu bot no Telegram. Você deve receber uma resposta em poucos segundos.

---

## Passo 4: Configure o Acesso da Equipe {#step-4-set-up-team-access}

Agora vamos dar acesso aos seus colegas de equipe. Existem duas abordagens.

### Abordagem A: Lista de Permissão Estática {#approach-a-static-allowlist}

Colete o ID de usuário do Telegram de cada membro da equipe (peça que enviem mensagem para [@userinfobot](https://t.me/userinfobot)) e adicione-os como uma lista separada por vírgulas:

```bash
# In ~/.hermes/.env
TELEGRAM_ALLOWED_USERS=123456789,987654321,555555555
```

Reinicie o gateway após as alterações:

```bash
hermes gateway stop && hermes gateway start
```

### Abordagem B: Pareamento por Mensagem Direta (Recomendado para Equipes) {#approach-b-dm-pairing-recommended-for-teams}

O pareamento por mensagem direta é mais flexível — você não precisa coletar IDs de usuário antecipadamente. Veja como funciona:

1. **O colega de equipe manda mensagem direta ao bot** — como ele não está na lista de permissão, o bot responde com um código de pareamento de uso único:
   ```
   🔐 Pairing code: XKGH5N7P
   Send this code to the bot owner for approval.
   ```

2. **O colega de equipe envia o código para você** (por qualquer canal — Slack, e-mail, pessoalmente)

3. **Você aprova no servidor:**
   ```bash
   hermes pairing approve telegram XKGH5N7P
   ```

4. **Ele está dentro** — o bot começa imediatamente a responder às mensagens dele

**Gerenciando usuários pareados:**

```bash
# See all pending and approved users
hermes pairing list

# Revoke someone's access
hermes pairing revoke telegram 987654321

# Clear expired pending codes
hermes pairing clear-pending
```

:::tip
O pareamento por mensagem direta é ideal para equipes porque você não precisa reiniciar o gateway ao adicionar novos usuários. As aprovações têm efeito imediato.
:::

### Considerações de Segurança {#security-considerations}

- **Nunca defina `GATEWAY_ALLOW_ALL_USERS=true`** em um bot com acesso a terminal — qualquer pessoa que encontre seu bot poderia executar comandos no seu servidor
- Os códigos de pareamento expiram após **1 hora** e usam aleatoriedade criptográfica
- A limitação de taxa previne ataques de força bruta: 1 requisição por usuário a cada 10 minutos, no máximo 3 códigos pendentes por plataforma
- Após 5 tentativas de aprovação falhas, a plataforma entra em um bloqueio de 1 hora
- Todos os dados de pareamento são armazenados com permissões `chmod 0600`

---

## Passo 5: Configure o Bot {#step-5-configure-the-bot}

### Defina um Canal Principal {#set-a-home-channel}

Um **canal principal** (home channel) é onde o bot entrega resultados de tarefas agendadas e mensagens proativas. Sem um, as tarefas agendadas não têm para onde enviar a saída.

**Opção 1:** Use o comando `/sethome` em qualquer grupo ou chat do Telegram onde o bot seja membro.

**Opção 2:** Defina manualmente em `~/.hermes/.env`:

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="Team Updates"
```

Para encontrar o ID de um canal, adicione o [@userinfobot](https://t.me/userinfobot) ao grupo — ele informará o chat ID do grupo.

### Configure a Exibição de Progresso das Ferramentas {#configure-tool-progress-display}

Controle quanto detalhe o bot mostra ao usar ferramentas. Em `~/.hermes/config.yaml`:

```yaml
display:
  tool_progress: new    # off | new | all | verbose
```

| Modo | O Que Você Vê |
|------|-------------|
| `off` | Apenas respostas limpas — sem atividade de ferramentas |
| `new` | Status breve para cada nova chamada de ferramenta (recomendado para mensagens) |
| `all` | Cada chamada de ferramenta com detalhes |
| `verbose` | Saída completa das ferramentas, incluindo resultados de comandos |

Os usuários também podem alterar isso por sessão com o comando `/verbose` no chat.

### Configure uma Personalidade com o SOUL.md {#set-up-a-personality-with-soulmd}

Personalize como o bot se comunica editando `~/.hermes/SOUL.md`:

Para um guia completo, veja [Usar SOUL.md com o Hermes](/guides/use-soul-with-hermes).

```markdown
# Soul
You are a helpful team assistant. Be concise and technical.
Use code blocks for any code. Skip pleasantries — the team
values directness. When debugging, always ask for error logs
before guessing at solutions.
```

### Adicione Contexto do Projeto {#add-project-context}

Se sua equipe trabalha em projetos específicos, crie arquivos de contexto para que o bot conheça sua stack:

```markdown
<!-- ~/.hermes/AGENTS.md -->
# Team Context
- We use Python 3.12 with FastAPI and SQLAlchemy
- Frontend is React with TypeScript
- CI/CD runs on GitHub Actions
- Production deploys to AWS ECS
- Always suggest writing tests for new code
```

:::info
Os arquivos de contexto são injetados no prompt de sistema de cada sessão. Mantenha-os concisos — cada caractere conta contra seu orçamento de tokens.
:::

---

## Passo 6: Configure Tarefas Agendadas {#step-6-set-up-scheduled-tasks}

Com o gateway em execução, você pode agendar tarefas recorrentes que entregam resultados ao canal da sua equipe.

### Resumo Diário de Standup {#daily-standup-summary}

Envie uma mensagem para o bot no Telegram:

```
Every weekday at 9am, check the GitHub repository at
github.com/myorg/myproject for:
1. Pull requests opened/merged in the last 24 hours
2. Issues created or closed
3. Any CI/CD failures on the main branch
Format as a brief standup-style summary.
```

O agente cria uma tarefa agendada (cron job) automaticamente e entrega os resultados no chat onde você perguntou (ou no canal principal).

### Verificação de Saúde do Servidor {#server-health-check}

```
Every 6 hours, check disk usage with 'df -h', memory with 'free -h',
and Docker container status with 'docker ps'. Report anything unusual —
partitions above 80%, containers that have restarted, or high memory usage.
```

### Gerenciando Tarefas Agendadas {#managing-scheduled-tasks}

```bash
# From the CLI
hermes cron list          # View all scheduled jobs
hermes cron status        # Check if scheduler is running

# From Telegram chat
/cron list                # View jobs
/cron remove <job_id>     # Remove a job
```

:::warning
Os prompts de tarefas agendadas rodam em sessões completamente novas, sem memória de conversas anteriores. Certifique-se de que cada prompt contenha **todo** o contexto de que o agente precisa — caminhos de arquivos, URLs, endereços de servidor e instruções claras.
:::

---

## Dicas de Produção {#production-tips}

### Use Docker para Segurança {#use-docker-for-safety}

Em um bot de equipe compartilhado, use o Docker como backend de terminal para que os comandos do agente rodem em um contêiner em vez de no seu host:

```bash
# In ~/.hermes/.env
TERMINAL_ENV=docker
TERMINAL_DOCKER_IMAGE=nikolaik/python-nodejs:python3.11-nodejs20
```

Ou em `~/.hermes/config.yaml`:

```yaml
terminal:
  backend: docker
  container_cpu: 1
  container_memory: 5120
  container_persistent: true
```

Dessa forma, mesmo que alguém peça ao bot para executar algo destrutivo, seu sistema host fica protegido.

### Monitore o Gateway {#monitor-the-gateway}

```bash
# Check if the gateway is running
hermes gateway status

# Watch live logs (Linux)
journalctl --user -u hermes-gateway -f

# Watch live logs (macOS)
tail -f ~/.hermes/logs/gateway.log
```

### Mantenha o Hermes Atualizado {#keep-hermes-updated}

No Telegram, envie `/update` para o bot — ele buscará a versão mais recente e reiniciará. Ou pelo servidor:

```bash
hermes update
hermes gateway stop && hermes gateway start
```

### Localizações dos Logs {#log-locations}

| O Quê | Localização |
|------|----------|
| Logs do gateway | `journalctl --user -u hermes-gateway` (Linux) ou `~/.hermes/logs/gateway.log` (macOS) |
| Saída de tarefas agendadas | `~/.hermes/cron/output/{job_id}/{timestamp}.md` |
| Definições de tarefas agendadas | `~/.hermes/cron/jobs.json` |
| Dados de pareamento | `~/.hermes/pairing/` |
| Histórico de sessões | `~/.hermes/sessions/` |

---

## Continuando {#going-further}

Você agora tem um assistente de equipe no Telegram funcionando. Aqui estão alguns próximos passos:

- **[Guia de Segurança](/user-guide/security)** — aprofunde-se em autorização, isolamento de contêineres e aprovação de comandos
- **[Gateway de Mensagens](/user-guide/messaging)** — referência completa sobre arquitetura do gateway, gerenciamento de sessões e comandos de chat
- **[Configuração do Telegram](/user-guide/messaging/telegram)** — detalhes específicos da plataforma, incluindo mensagens de voz e TTS
- **[Tarefas Agendadas](/user-guide/features/cron)** — agendamento avançado com opções de entrega e expressões cron
- **[Arquivos de Contexto](/user-guide/features/context-files)** — AGENTS.md, SOUL.md e .cursorrules para conhecimento do projeto
- **[Personalidade](/user-guide/features/personality)** — predefinições de personalidade integradas e definições de persona personalizadas
- **Adicione mais plataformas** — o mesmo gateway pode rodar simultaneamente [Discord](/user-guide/messaging/discord), [Slack](/user-guide/messaging/slack) e [WhatsApp](/user-guide/messaging/whatsapp)

---

*Dúvidas ou problemas? Abra uma issue no GitHub — contribuições são bem-vindas.*
