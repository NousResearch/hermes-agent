---
sidebar_position: 5
title: "Tarefas agendadas (Cron)"
description: "Agende tarefas automatizadas em linguagem natural, gerencie-as com uma única ferramenta cron e anexe uma ou mais skills"
---

# Tarefas agendadas (Cron)

Agende tarefas para rodar automaticamente com linguagem natural ou expressões cron. O Hermes expõe o gerenciamento de cron por meio de uma única ferramenta `cronjob` com operações no estilo action, em vez de ferramentas separadas de schedule/list/remove.

## O que o cron pode fazer agora {#what-cron-can-do-now}

Jobs cron podem:

- agendar tarefas únicas ou recorrentes
- pausar, retomar, editar, disparar e remover jobs
- anexar zero, uma ou várias skills a um job
- entregar resultados de volta ao chat de origem, arquivos locais ou destinos de plataforma configurados
- rodar em sessões de agente novas com a lista estática normal de ferramentas
- rodar em **modo no-agent** — um script em cronograma, com stdout entregue literalmente, zero envolvimento de LLM (veja a seção [modo no-agent](#no-agent-mode-script-only-jobs) abaixo)

Tudo isso está disponível para o próprio Hermes pela ferramenta `cronjob`, então você pode criar, pausar, editar e remover jobs pedindo em linguagem natural — sem CLI.

:::tip
Na criação, um job não fixado (um ao qual você não dá `provider`/`model` explícitos) segue o padrão global selecionado por `hermes model` — e o Hermes **captura um snapshot** desse provider e model no job. Se o padrão global mudar depois, o job **falha fechada**: pula a execução, não faz chamada de inferência e envia um alerta pedindo que você fixe provider/model explicitamente (`cronjob action=update job_id=… provider=… model=…`) para continuar. Isso impede que um job desacompanhado herde silenciosamente uma troca para provider/model pago e gaste dinheiro que você não pretendia (#44585). Para fazer um job acompanhar deliberadamente seu padrão global, fixe-o nos novos valores depois de alterá-los. `hermes setup --portal` é a opção de menor atrito para execuções desacompanhadas, já que o refresh OAuth é automático. Veja [Nous Portal](/integrations/nous-portal).
:::

:::warning
Sessões executadas por cron não podem criar recursivamente mais jobs cron. O Hermes desabilita ferramentas de gerenciamento de cron dentro de execuções cron para evitar loops de agendamento descontrolados.
:::

## Criando tarefas agendadas {#creating-scheduled-tasks}

### No chat com `/cron`

```bash
/cron add 30m "Remind me to check the build"
/cron add "every 2h" "Check server status"
/cron add "every 1h" "Summarize new feed items" --skill blogwatcher
/cron add "every 1h" "Use both skills and combine the result" --skill blogwatcher --skill maps
```

### Pela CLI standalone

```bash
hermes cron create "every 2h" "Check server status"
hermes cron create "every 1h" "Summarize new feed items" --skill blogwatcher
hermes cron create "every 1h" "Use both skills and combine the result" \
  --skill blogwatcher \
  --skill maps \
  --name "Skill combo"
```

### Por conversa natural

Peça ao Hermes normalmente:

```text
Every morning at 9am, check Hacker News for AI news and send me a summary on Telegram.
```

O Hermes usará internamente a ferramenta unificada `cronjob`.

## Jobs cron com skills {#skill-backed-cron-jobs}

Um job cron pode carregar uma ou mais skills antes de executar o prompt.

### Skill única

```python
cronjob(
    action="create",
    skill="blogwatcher",
    prompt="Check the configured feeds and summarize anything new.",
    schedule="0 9 * * *",
    name="Morning feeds",
)
```

### Várias skills

As skills são carregadas em ordem. O prompt vira a instrução da tarefa sobreposta a essas skills.

```python
cronjob(
    action="create",
    skills=["blogwatcher", "maps"],
    prompt="Look for new local events and interesting nearby places, then combine them into one short brief.",
    schedule="every 6h",
    name="Local brief",
)
```

Isso é útil quando você quer que um agente agendado herde fluxos reutilizáveis sem encher o prompt cron com o texto completo da skill.

## Executando um job dentro de um diretório de projeto {#running-a-job-inside-a-project-directory}

Jobs cron rodam por padrão desconectados de qualquer repositório — nenhum `AGENTS.md`, `CLAUDE.md` ou `.cursorrules` é carregado, e as ferramentas terminal / file / code-exec rodam a partir do diretório de trabalho em que o gateway iniciou. Passe `--workdir` (CLI) ou `workdir=` (chamada de ferramenta) para mudar isso:

```bash
# Standalone CLI (schedule and prompt are positional)
hermes cron create "every 1d at 09:00" \
  "Audit open PRs, summarize CI health, and post to #eng" \
  --workdir /home/me/projects/acme
```

```python
# From a chat, via the cronjob tool
cronjob(
    action="create",
    schedule="every 1d at 09:00",
    workdir="/home/me/projects/acme",
    prompt="Audit open PRs, summarize CI health, and post to #eng",
)
```

Quando `workdir` está definido:

- `AGENTS.md`, `CLAUDE.md` e `.cursorrules` desse diretório são injetados no system prompt (mesma ordem de descoberta da CLI interativa)
- `terminal`, `read_file`, `write_file`, `patch`, `search_files` e `execute_code` usam esse diretório como working directory
- O caminho deve ser um diretório absoluto que existe — caminhos relativos e diretórios inexistentes são rejeitados na criação / atualização
- Passe `--workdir ""` (ou `workdir=""` pela ferramenta) na edição para limpar e restaurar o comportamento antigo

:::note Serialização
Jobs com `workdir` rodam sequencialmente no tick do scheduler, não no pool paralelo. Isso é intencional: o worker cron aplica o workdir do job por meio de estado global de terminal do processo, então dois jobs com workdir rodando ao mesmo tempo corromperiam o cwd um do outro. Jobs sem workdir ainda rodam em paralelo como antes.
:::

## Editando jobs {#editing-jobs}

Você não precisa excluir e recriar jobs só para alterá-los.

:::tip Referência de job
O placeholder `<job_id>` abaixo (e em [Ações de ciclo de vida](#lifecycle-actions)) também aceita o nome do job (sem distinção de maiúsculas/minúsculas) — útil quando você lembra `morning-digest` mas não o ID hex. Um job ID exato tem precedência sobre correspondências por nome; se a referência não for um ID e um nome corresponder a mais de um job, o comando recusa e imprime os IDs candidatos para você desambiguar.
:::

### Chat

```bash
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Use the revised task"
/cron edit <job_id> --skill blogwatcher --skill maps
/cron edit <job_id> --remove-skill blogwatcher
/cron edit <job_id> --clear-skills
```

### CLI standalone

```bash
hermes cron edit <job_id> --schedule "every 4h"
hermes cron edit <job_id> --prompt "Use the revised task"
hermes cron edit <job_id> --skill blogwatcher --skill maps
hermes cron edit <job_id> --add-skill maps
hermes cron edit <job_id> --remove-skill blogwatcher
hermes cron edit <job_id> --clear-skills
```

Notas:

- `--skill` repetido substitui a lista de skills anexadas do job
- `--add-skill` acrescenta à lista existente sem substituí-la
- `--remove-skill` remove skills anexadas específicas
- `--clear-skills` remove todas as skills anexadas

## Ações de ciclo de vida {#lifecycle-actions}

Jobs cron agora têm um ciclo de vida mais completo do que apenas create/remove.

### Chat

```bash
/cron list
/cron pause <job_id>
/cron resume <job_id>
/cron run <job_id>
/cron remove <job_id>
```

### CLI standalone

```bash
hermes cron list
hermes cron pause <job_id_or_name>
hermes cron resume <job_id_or_name>
hermes cron run <job_id_or_name>
hermes cron remove <job_id_or_name>
hermes cron edit <job_id_or_name> [...flags]
hermes cron status
hermes cron tick
```

O que fazem:

- `pause` — mantém o job, mas para de agendá-lo
- `resume` — reabilita o job e calcula a próxima execução futura
- `run` — dispara o job no próximo tick do scheduler
- `remove` — exclui por completo
- `edit` — modifica schedule, prompt, delivery, etc.

**Busca por nome.** Todos os quatro verbos mutantes (`pause`, `resume`, `run`, `remove`, `edit`) mais a ferramenta `cronjob` do agente agora aceitam um **nome** de job (sem distinção de maiúsculas/minúsculas) no lugar do ID hex. O agente e a CLI preferem correspondência exata por ID se existir; correspondências ambíguas por nome (vários jobs com o mesmo nome) são recusadas com a lista completa de IDs candidatos para você escolher um explicitamente. Nomes não são únicos, então essa proteção é estrutural — impede mutar silenciosamente o job errado quando dois compartilham um nome.

## Como funciona {#how-it-works}

**A execução cron é tratada pelo daemon do gateway.** O gateway faz tick no scheduler a cada 60 segundos, executando jobs devidos em sessões de agente isoladas.

```bash
hermes gateway install     # Install as a user service
sudo hermes gateway install --system   # Linux: boot-time system service for servers
hermes gateway             # Or run in foreground

hermes cron list
hermes cron status
```

### Comportamento do scheduler do gateway

A cada tick o Hermes:

1. carrega jobs de `~/.hermes/cron/jobs.json`
2. verifica `next_run_at` contra a hora atual
3. inicia uma sessão `AIAgent` nova para cada job devido
4. opcionalmente injeta uma ou mais skills anexadas nessa sessão nova
5. executa o prompt até a conclusão
6. entrega a resposta final
7. atualiza metadados de execução e o próximo horário agendado

Um file lock em `~/.hermes/cron/.tick.lock` impede que ticks sobrepostos do scheduler executem duas vezes o mesmo lote de jobs.

### Histórico de execução

O Hermes registra cada tentativa cron reivindicada no
`~/.hermes/cron/executions.db` local ao perfil antes do dispatch do executor ou provider. Tentativas
passam por `claimed`, `running` e um estado terminal imutável:
`completed`, `failed` ou `unknown`. Após reinício, o Hermes marca uma tentativa abandonada como `unknown` somente quando o PID original e a impressão digital de início do processo provam
que seu dono se foi. Tentativas unknown são registros de auditoria e nunca
são reexecutadas automaticamente.

Inspecione tentativas recentes com `hermes cron runs [job-id] --limit 20` (alias:
`history`). O histórico terminal é limitado; tentativas ativas nunca são podadas. O
ledger está incluído em backups rápidos.

## Opções de entrega {#delivery-options}

Ao agendar jobs, você especifica para onde vai a saída:

| Opção | Descrição | Exemplo |
|--------|-------------|---------|
| `"origin"` | De volta ao local onde o job foi criado | Padrão em plataformas de mensagens |
| `"local"` | Salvar apenas em arquivos locais (`~/.hermes/cron/output/`) | Padrão na CLI |
| `"telegram"` | Canal home do Telegram | Usa `TELEGRAM_HOME_CHANNEL` |
| `"telegram:123456"` | Chat específico do Telegram por ID | Entrega direta |
| `"telegram:-100123:17585"` | Tópico específico do Telegram | Formato `chat_id:thread_id` |
| `"discord"` | Canal home do Discord | Usa `DISCORD_HOME_CHANNEL` |
| `"discord:#engineering"` | Canal específico do Discord | Por nome do canal |
| `"slack"` | Canal home do Slack | |
| `"whatsapp"` | Home do WhatsApp | |
| `"signal"` | Signal | |
| `"matrix"` | Sala home do Matrix | |
| `"mattermost"` | Canal home do Mattermost | |
| `"email"` | Email | |
| `"sms"` | SMS via Twilio | |
| `"homeassistant"` | Home Assistant | |
| `"dingtalk"` | DingTalk | |
| `"feishu"` | Feishu/Lark | |
| `"wecom"` | WeCom | |
| `"weixin"` | Weixin (WeChat) | |
| `"bluebubbles"` | BlueBubbles (iMessage) | |
| `"qqbot"` | QQ Bot (Tencent QQ) | |
| `"all"` | Distribuir para todo canal home conectado | Resolvido no momento do disparo |
| `"telegram,discord"` | Distribuir para um conjunto específico de canais | Lista separada por vírgula |
| `"origin,all"` | Entregar na origem **mais** todo outro canal conectado | Combine quaisquer tokens |

A resposta final do agente é entregue automaticamente ao destino `deliver:` configurado — o agente não envia mensagens por conta própria, então não há nada a chamar no prompt cron.

### Intenção de roteamento (`all`)

`all` permite enviar um job cron para todo canal de mensagens que você configurou, sem precisar enumerá-los por nome. É **resolvido no momento do disparo**, então um job criado antes de você configurar o Telegram passará a incluir o Telegram no próximo tick depois que você definir `TELEGRAM_HOME_CHANNEL`.

Semântica: `all` expande para toda plataforma com canal home configurado. Zero está ok; o job simplesmente não produz destinos de entrega e é registrado como falha de entrega upstream.

`all` compõe com destinos explícitos. `origin,all` entrega no chat de origem *mais* todo outro canal home conectado, deduplicando por `(platform, chat_id, thread_id)`.

### Tópico cron do Telegram (`TELEGRAM_CRON_THREAD_ID`)

Quando o modo de tópicos do Telegram está habilitado, o DM raiz fica reservado como lobby do sistema — respostas enviadas lá são rejeitadas com um lembrete de lobby e `reply_to_message_id` é descartado, então você não pode responder a uma mensagem cron que caiu no chat principal.

Aponte o cron para um tópico de fórum dedicado:

1. No Telegram, abra o DM do bot e crie um tópico chamado, por exemplo, `Cron`. Pressione longamente o cabeçalho do tópico → **Copy link**; o inteiro final é o `message_thread_id` do tópico.
2. Defina `TELEGRAM_CRON_THREAD_ID=<that id>` no seu `.env`.

Isso se aplica apenas a entregas cron. `TELEGRAM_HOME_CHANNEL_THREAD_ID` (usado em outros lugares, ex.: notificações de reinício) permanece inalterado. Destinos explícitos `deliver="telegram:chat_id:thread_id"` continuam prevalecendo sobre a env var. Respostas a mensagens cron agora chegam na sessão do tópico existente, então você pode agir sobre elas diretamente.

### Envelope da resposta

Por padrão, a saída cron entregue é envolvida com cabeçalho e rodapé para o destinatário saber que veio de uma tarefa agendada:

```
Cronjob Response: Morning feeds
-------------

<agent output here>

Note: The agent cannot see this message, and therefore cannot respond to it.
```

Para entregar a saída bruta do agente sem o envelope, defina `cron.wrap_response` como `false`:

```yaml
# ~/.hermes/config.yaml
cron:
  wrap_response: false
```

### Jobs continuáveis (responder a uma entrega cron)

Por padrão uma entrega cron é dispara e esquece: a mensagem é enviada, mas não
fica no histórico de conversa do chat, então se você responder a ela o agente
não tem registro do que disse. Defina um job como **continuable** e o brief entregue
vira uma conversa na qual você pode responder — o agente tem o brief em contexto
em vez de perguntar "o que é a Tarefa #2?".

Opt-in, **desligado por padrão**. Habilite globalmente na config, ou por job via
`attach_to_session` da ferramenta `cronjob`
(que sobrescreve a configuração global para aquele job):

```yaml
# ~/.hermes/config.yaml
cron:
  mirror_delivery: false   # set true to make cron deliveries continuable
```

O comportamento é **thread-preferred**, limitado ao chat de origem do job:

- **Plataformas com suporte a threads** (tópicos do Telegram, threads Discord/Slack): cada
  entrega abre sua própria thread dedicada e o brief é semeado na sessão dessa
  thread, então uma resposta na thread continua com contexto completo. Um
  job recorrente (ex.: um brief diário) abre uma thread nova por execução, mantendo isolada a
  discussão de follow-up de cada entrega.
- **Plataformas só-DM** (WhatsApp, Signal, SMS): não existem threads, então o brief
  é espelhado na sessão DM de origem — o próprio DM é a
  superfície de continuação.

Apenas o chat de origem é tocado: destinos fan-out / broadcast (`all`,
entregas explícitas em outros chats) nunca são tornados continuáveis. O espelho é
escrito como um turn de usuário rotulado (`[Cron delivery: <task name>]`), o que mantém
o histórico de conversa seguro para alternância em todos os providers de model.

#### Continuação plana no canal (Slack)

O comportamento thread-preferred acima cria uma thread dedicada a cada
entrega. Se você preferir que um job continuável caia **plano na timeline do canal
** — sem thread — defina a **superfície continuável** do Slack como `in_channel`:

```yaml
# ~/.hermes/config.yaml
slack:
  cron_continuable_surface: in_channel   # default: thread
  reply_in_thread: false                 # required pairing (see below)
  require_mention: false                 # so a plain reply continues the job
```

No modo `in_channel` o brief é entregue como mensagem comum de topo de canal
(nenhuma thread é aberta), e sua resposta continua o job via a
sessão compartilhada do canal. Três configurações trabalham juntas:

- **`cron_continuable_surface: in_channel`** — pula criação de thread na entrega.
- **`reply_in_thread: false`** (obrigatório) — faz o bot responder sua resposta
  *plana* no canal e associá-la à mesma sessão de canal inteiro em que o brief
  foi semeado. Sem isso a continuação ainda funciona, mas chega em uma
  thread (cai com segurança para continuação estilo thread, nunca uma resposta
  perdida — o gateway registra um aviso na inicialização para você detectar a incompatibilidade).
- **`require_mention: false`** (ou adicione o canal a `free_response_channels`)
  — para você responder com mensagem simples; caso contrário o bot só acorda quando você
  menciona com `@` em cada resposta.

Como a continuação é a sessão de **canal inteiro**, ela é compartilhada: outras
conversas no canal — e um segundo job continuável in-channel — entram na mesma
conversa contínua. Isso é inerente a "plano em um canal" e é o mesmo
tradeoff que usuários de `reply_in_thread: false` já aceitam; use a superfície padrão
`thread` quando quiser o follow-up de cada entrega isolado.

Isso é uma capacidade do Slack hoje. Outras plataformas aceitam a chave, mas caem para a superfície
`thread` (suas primitivas de continuação diferem); a escolha é
por plataforma, definida na config de cada plataforma. É uma flag de config do lado do gateway
— um `/restart` a aplica; não é necessário reinstalar o app Slack.

:::note DMs 1:1
`cron_continuable_surface` é uma configuração de **canal** — um DM 1:1 não tem
escolha thread-vs-timeline (o DM já é plano), então a chave
não tem efeito lá. O que governa se uma entrega cron em DM é continuável é o
knob separado e pré-existente **`slack.dm_top_level_threads_as_sessions`**:

- **`false`** — todos os DMs de topo compartilham uma sessão DM contínua, então um brief cron
  continuável e sua resposta caem na **mesma** sessão e o job continua em
  contexto. É o que você quer para cron continuável em DM.
- **`true`** (padrão) — cada mensagem DM de topo é sua própria sessão, então uma resposta
  a um brief entregue inicia uma sessão *nova* sem registro do brief.
  A continuação não funciona nesse modo (para cron ou qualquer outra entrega plana).

Então, para um job cron continuável entregue em DM 1:1, defina
`slack.dm_top_level_threads_as_sessions: false`. `cron_continuable_surface` não é
obrigatório (e é ignorado) para DMs.
:::

### Supressão silenciosa

Se a resposta final do agente contiver `[SILENT]`, a entrega é suprimida por completo. A saída ainda é salva localmente para auditoria (em `~/.hermes/cron/output/`), mas nenhuma mensagem é enviada ao destino de entrega.

Isso é útil para jobs de monitoramento que só devem reportar quando algo está errado:

```text
Check if nginx is running. If everything is healthy, respond with only [SILENT].
Otherwise, report the issue.
```

Jobs com falha sempre entregam independentemente do marcador `[SILENT]` — apenas execuções bem-sucedidas podem ser silenciadas. Para jobs de monitoramento silenciosos, peça ao agente para responder apenas com `[SILENT]` quando não houver nada a reportar.

## Timeout de script {#script-timeout}

Scripts pré-execução (anexados via parâmetro `script`) têm timeout padrão de 3600 segundos (1 hora). Isso limita **apenas o script** — jobs baseados em skill / dirigidos por LLM rodam em um orçamento de inatividade separado e não são limitados por esse valor. Se seus scripts precisarem de outro limite, você pode alterá-lo:

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 1800   # 30 minutes
```

Ou defina a variável de ambiente `HERMES_CRON_SCRIPT_TIMEOUT`. A ordem de resolução é: env var → config.yaml → padrão 3600s.

## Modo no-agent (jobs só de script) {#no-agent-mode-script-only-jobs}

Para jobs recorrentes que não precisam de raciocínio LLM — watchdogs clássicos, alertas de disco/memória, heartbeats, pings de CI — passe `no_agent=True` na criação. O scheduler executa seu script no cronograma e entrega o stdout diretamente, pulando o agente por completo:

```bash
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"
```

Semântica:

- stdout do script (trimmed) → entregue literalmente como mensagem.
- **stdout vazio → tick silencioso**, sem entrega. Esse é o padrão watchdog: "só falar quando algo estiver errado".
- exit não zero ou timeout → um alerta de erro é entregue, então um watchdog quebrado não pode falhar silenciosamente.
- `{"wakeAgent": false}` na última linha → tick silencioso (mesmo gate que jobs LLM usam).
- Sem tokens, sem model, sem fallback de provider — o job nunca toca a camada de inferência.

Arquivos `.sh` / `.bash` rodam sob `/bin/bash`; qualquer outro sob o interpretador Python atual (`sys.executable`). Scripts devem ficar em `~/.hermes/scripts/` (mesma regra de sandbox do gate de script pré-execução).

### O agente configura isso para você

O schema da ferramenta `cronjob` expõe `no_agent` diretamente ao Hermes, então você pode descrever um watchdog no chat e deixar o agente montar:

```text
Ping me on Telegram if RAM is over 85%, every 5 minutes.
```

O Hermes escreverá o script de verificação em `~/.hermes/scripts/` via `write_file`, depois chamará:

```python
cronjob(action="create", schedule="every 5m",
        script="memory-watchdog.sh", no_agent=True,
        deliver="telegram", name="memory-watchdog")
```

Ele escolhe `no_agent=True` automaticamente quando o conteúdo da mensagem é totalmente determinado pelo script (watchdogs, alertas de limiar, heartbeats). A mesma ferramenta também deixa o agente pausar, retomar, editar e remover jobs — então todo o ciclo de vida é guiado pelo chat sem ninguém tocar na CLI.

Veja o [guia Script-Only Cron Jobs](/guides/cron-script-only) para exemplos práticos.

## Encadeando jobs com `context_from` {#chaining-jobs-with-context_from}

Jobs cron rodam em sessões isoladas sem memória de execuções anteriores. Mas às vezes a saída de um job é exatamente o que o próximo precisa. O parâmetro `context_from` conecta isso automaticamente — o prompt do Job B recebe a saída mais recente do Job A prepended como contexto em runtime.

```python
# Job 1: Collect raw data
cronjob(
    action="create",
    prompt="Fetch the top 10 AI/ML stories from Hacker News. Save them to ~/.hermes/data/briefs/raw.md in markdown format with title, URL, and score.",
    schedule="0 7 * * *",
    name="AI News Collector",
)

# Job 2: Triage — receives Job 1's output as context
# Get Job 1's ID from: cronjob(action="list")
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/raw.md. Score each story 1–10 for engagement potential and novelty. Output the top 5 to ~/.hermes/data/briefs/ranked.md.",
    schedule="30 7 * * *",
    context_from="<job1_id>",
    name="AI News Triage",
)

# Job 3: Ship — receives Job 2's output as context
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/ranked.md. Write 3 tweet drafts (hook + body + hashtags). Deliver to telegram:7976161601.",
    schedule="0 8 * * *",
    context_from="<job2_id>",
    name="AI News Brief",
)
```

**Como funciona:**

- Quando o Job 2 dispara, o Hermes lê a saída mais recente do Job 1 em `~/.hermes/cron/output/{job1_id}/*.md`
- Essa saída é prepended ao prompt do Job 2 automaticamente
- O Job 2 não precisa hardcodar "read this file" — recebe o conteúdo como contexto
- A cadeia pode ter qualquer comprimento: Job 1 → Job 2 → Job 3 → ...

**O que `context_from` aceita:**

| Formato | Exemplo |
|--------|---------|
| ID de job único (string) | `context_from="a1b2c3d4"` |
| Vários IDs de job (lista) | `context_from=["job_a", "job_b"]` |

As saídas são concatenadas na ordem listada.

**Quando usar:**

- Pipelines multiestágio (collect → filter → format → deliver)
- Tarefas dependentes em que o trabalho do passo N depende da saída do passo N−1
- Padrões fan-out/fan-in em que um job agrega resultados de vários outros

## Recuperação de provider {#provider-recovery}

Jobs cron herdam seus fallback providers configurados e rotação de credential pool. Se a API key primária estiver rate-limited ou o provider retornar erro, o agente cron pode:

- **Fazer fallback para um provider alternativo** se você tiver `fallback_providers` (ou o legado `fallback_model`) configurado em `config.yaml`
- **Rotacionar para a próxima credencial** no seu [credential pool](/user-guide/configuration#credential-pool-strategies) para o mesmo provider

Isso significa que jobs cron que rodam com alta frequência ou em horários de pico são mais resilientes — uma única key rate-limited não derruba a execução inteira.

## Formatos de schedule {#schedule-formats}

A resposta final do agente é entregue automaticamente ao destino `deliver:` do job — o agente não dispara mensagens por conta própria, então o conteúdo voltado ao usuário vai simplesmente na resposta final. Para entregar em **destinos adicionais ou diferentes**, liste vários destinos `deliver:` no job cron (separados por vírgula, ex.: `deliver: "telegram,discord"`) em vez de fazer o agente enviá-los.

### Atrasos relativos (one-shot)

```text
30m     → Run once in 30 minutes
2h      → Run once in 2 hours
1d      → Run once in 1 day
```

### Intervalos (recorrentes)

```text
every 30m    → Every 30 minutes
every 2h     → Every 2 hours
every 1d     → Every day
```

### Expressões cron

```text
0 9 * * *       → Daily at 9:00 AM
0 9 * * 1-5     → Weekdays at 9:00 AM
0 */6 * * *     → Every 6 hours
30 8 1 * *      → First of every month at 8:30 AM
0 0 * * 0       → Every Sunday at midnight
```

### Timestamps ISO

```text
2026-03-15T09:00:00    → One-time at March 15, 2026 9:00 AM
```

## Comportamento de repetição {#repeat-behavior}

| Tipo de schedule | Repetição padrão | Comportamento |
|--------------|----------------|----------|
| One-shot (`30m`, timestamp) | 1 | Executa uma vez |
| Intervalo (`every 2h`) | forever | Executa até ser removido |
| Expressão cron | forever | Executa até ser removido |

Você pode sobrescrever:

```python
cronjob(
    action="create",
    prompt="...",
    schedule="every 2h",
    repeat=5,
)
```

## Gerenciando jobs programaticamente {#managing-jobs-programmatically}

A API voltada ao agente é uma ferramenta:

```python
cronjob(action="create", ...)
cronjob(action="list")
cronjob(action="update", job_id="...")
cronjob(action="pause", job_id="...")
cronjob(action="resume", job_id="...")
cronjob(action="run", job_id="...")
cronjob(action="remove", job_id="...")
```

Para `update`, passe `skills=[]` para remover todas as skills anexadas.

## Toolsets disponíveis para jobs cron {#toolsets-available-to-cron-jobs}

O cron executa cada job em uma sessão de agente nova sem plataforma de chat anexada. Por padrão o agente cron recebe **o toolset que você configurou para a plataforma `cron` em `hermes tools`** — não o padrão da CLI, não tudo sob o sol.

```bash
hermes tools
# → pick the "cron" platform in the curses UI
# → toggle toolsets on/off just like you would for Telegram/Discord/etc.
```

Controle mais apertado por job está disponível via o campo `enabled_toolsets` em `cronjob.create` (ou em um job existente via `cronjob.update`):

```text
cronjob(action="create", name="weekly-news-summary",
        schedule="every sunday 9am",
        enabled_toolsets=["web", "file"],      # just web + file, no terminal/browser/etc.
        prompt="Summarize this week's AI news: ...")
```

Quando `enabled_toolsets` está definido em um job ele prevalece; caso contrário a config da plataforma cron em `hermes tools` prevalece; caso contrário o Hermes cai para os padrões built-in. Isso importa para controle de custo: carregar `browser`, `delegation` em todo job minúsculo de "fetch news" incha o prompt de schema de ferramentas em cada chamada LLM.

### Pulando o agente por completo: `wakeAgent`

Se seu job cron anexa um script de pré-verificação (via `script=`), o script pode decidir em runtime se o Hermes deve invocar o agente. Emita uma linha final de stdout no formato:

```text
{"wakeAgent": false}
```

…e o cron pula a execução do agente por completo neste tick. Útil para polls frequentes (a cada 1–5 min) que só precisam acordar o LLM quando o estado realmente mudou — caso contrário você paga por turns de agente sem conteúdo repetidamente.

```python
# pre-check script
import json, sys
latest = fetch_latest_issue_count()
prev = read_state("issue_count")
if latest == prev:
    print(json.dumps({"wakeAgent": False}))   # skip this tick
    sys.exit(0)
write_state("issue_count", latest)
print(json.dumps({"wakeAgent": True, "context": {"new_issues": latest - prev}}))
```

Quando `wakeAgent` é omitido, o padrão é `true` (acordar o agente como de costume).

#### Receitas: gates baratos de pré-execução

O gate `wakeAgent` dá a você uma forma de $0 de decidir se um job agendado deve gastar tokens LLM. Três padrões cobrem a maioria dos casos.

**Gate de mudança de arquivo** — só executar quando um arquivo observado tiver conteúdo novo desde o último tick bem-sucedido. O scheduler registra `last_run_at` de cada job; compare contra o mtime do arquivo.

```bash
#!/bin/bash
# ~/.hermes/scripts/feed-changed.sh
FEED="$HOME/data/feed.json"
STATE="$HOME/.hermes/scripts/.feed-changed.last"
test -f "$FEED" || { echo '{"wakeAgent": false}'; exit 0; }
mtime=$(stat -c %Y "$FEED")
last=$(cat "$STATE" 2>/dev/null || echo 0)
if [ "$mtime" -le "$last" ]; then
  echo '{"wakeAgent": false}'
else
  echo "$mtime" > "$STATE"
  echo '{"wakeAgent": true}'
fi
```

```text
cronjob(action="create", name="process-feed",
        schedule="every 30m",
        script="feed-changed.sh",
        prompt="A new ~/data/feed.json has landed. Summarize what changed.")
```

**Gate de flag externa** — só executar quando outro processo sinalizou prontidão (ex.: um hook de deploy deixa um arquivo, um job CI define um valor no seu state store).

```bash
#!/bin/bash
# ~/.hermes/scripts/flag-ready.sh
if test -f /tmp/new-data-ready; then
  rm -f /tmp/new-data-ready
  echo '{"wakeAgent": true}'
else
  echo '{"wakeAgent": false}'
fi
```

```text
cronjob(action="create", name="nightly-analysis",
        schedule="0 9 * * *",
        script="flag-ready.sh",
        prompt="Run the nightly analysis over today's batch.")
```

**Gate de contagem SQL** — só executar quando houver linhas novas para processar no seu próprio banco. O script também pode passar a contagem ao agente via `context`, para o agente saber o volume sem reconsultar.

```python
#!/usr/bin/env python
# ~/.hermes/scripts/new-rows.py
import json, sqlite3
conn = sqlite3.connect("/home/me/data/app.db")
n = conn.execute(
    "SELECT COUNT(*) FROM messages WHERE ts > strftime('%s','now','-2 hours')"
).fetchone()[0]
if n < 1:
    print(json.dumps({"wakeAgent": False}))
else:
    print(json.dumps({"wakeAgent": True, "context": {"new_rows": n}}))
```

```text
cronjob(action="create", name="summarize-new-msgs",
        schedule="every 2h",
        script="new-rows.py",
        prompt="Summarize the new messages from the last 2 hours.")
```

O mesmo padrão funciona para qualquer fonte de dados que você consulte de um script — Postgres, uma API HTTP, seu próprio state store — sem embutir um avaliador SQL no subsistema cron.

:::tip
O próprio `~/.hermes/state.db` do Hermes é um schema interno que muda entre releases. Não consulte-o de um gate de pré-execução — aponte para seu próprio banco ou feed.
:::

Crédito: este conjunto de receitas foi motivado pela exploração de @iankar8 em [#2654](https://github.com/NousResearch/hermes-agent/pull/2654), que propôs triggers sql/file/command paralelos. O gate `script` + `wakeAgent` já cobre os três casos a $0, então o trabalho virou documentação.

### Encadeando jobs: `context_from`

Um job cron pode consumir a saída bem-sucedida mais recente de um ou mais outros jobs listando seus nomes (ou IDs) em `context_from`:

```text
cronjob(action="create", name="daily-digest",
        schedule="every day 7am",
        context_from=["ai-news-fetch", "github-prs-fetch"],
        prompt="Write the daily digest using the outputs above.")
```

As saídas completadas mais recentes dos jobs referenciados são injetadas acima do prompt como contexto para esta execução. Cada entrada upstream deve ser um job ID ou nome válido (veja `cronjob action="list"`). Nota: o encadeamento lê a saída *completada mais recente* — não espera jobs upstream rodando no mesmo tick.

## Armazenamento de jobs {#job-storage}

Jobs ficam em `~/.hermes/cron/jobs.json`. A saída das execuções é salva em `~/.hermes/cron/output/{job_id}/{timestamp}.md`.

:::tip
Peça ao agente para gerenciar jobs pela ferramenta `cronjob`, `hermes cron edit` ou `/cron` — não editando `jobs.json` diretamente. Edições diretas podem falhar silenciosamente quando [file write safety](../security.md#file-write-safety) bloqueia o caminho (por exemplo quando `HERMES_WRITE_SAFE_ROOT` está definido), e o rodapé do [file-mutation verifier](../configuration.md#file-mutation-verifier) é o sinal autoritativo de que nada foi salvo.
:::

Jobs podem armazenar `model` e `provider` como `null`. Quando esses campos são omitidos, o Hermes os resolve na hora da execução a partir da configuração global. Eles só aparecem no registro do job quando há override por job.

O armazenamento usa escritas atômicas de arquivo para que escritas interrompidas não deixem um arquivo de job parcialmente escrito.

## Prompts autocontidos ainda importam {#self-contained-prompts-still-matter}

:::warning Importante
Jobs cron rodam em uma sessão de agente completamente nova. O prompt deve conter tudo que o agente precisa que ainda não seja fornecido pelas skills anexadas.
:::

**RUIM:** `"Check on that server issue"`

**BOM:** `"SSH into server 192.168.1.100 as user 'deploy', check if nginx is running with 'systemctl status nginx', and verify https://example.com returns HTTP 200."`

## Segurança {#security}

Prompts de tarefas agendadas são escaneados em busca de padrões de prompt-injection e exfiltração de credenciais na criação e atualização. Prompts com truques Unicode invisíveis, tentativas de backdoor SSH ou payloads óbvios de exfiltração de segredos são bloqueados.
