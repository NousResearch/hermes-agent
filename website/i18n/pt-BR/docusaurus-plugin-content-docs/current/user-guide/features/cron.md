---
sidebar_position: 5
title: "Tarefas Agendadas (Cron)"
description: "Agende tarefas automatizadas com linguagem natural, gerencie com uma ferramenta cron e anexe uma ou mais skills"
---

# Tarefas Agendadas (Cron) {#scheduled-tasks-cron}

Agende tarefas para rodar automaticamente com linguagem natural ou expressões cron. O Hermes expõe gestão de cron por uma única ferramenta `cronjob` com operações estilo action em vez de ferramentas separadas schedule/list/remove.

## O que o cron pode fazer agora {#what-cron-can-do-now}

Jobs cron podem:

- agendar tarefas one-shot ou recorrentes
- pausar, retomar, editar, disparar e remover jobs
- anexar zero, uma ou múltiplas skills a um job
- entregar resultados de volta ao chat de origem, arquivos locais ou targets de plataforma configurados
- rodar em sessões de agente novas com a lista estática normal de ferramentas
- rodar em **modo no-agent** — um script num schedule, seu stdout entregue verbatim, zero envolvimento LLM (veja a seção [modo no-agent](#no-agent-mode-script-only-jobs) abaixo)

Tudo isso está disponível ao próprio Hermes pela ferramenta `cronjob`, então você pode criar, pausar, editar e remover jobs pedindo em linguagem natural — sem CLI.

:::tip
**Em qual modelo um job cron roda?** A resolução no fire time é: pin por job → `cron.model` em `config.yaml` → default global de `hermes model`.

- **Pin por job** — definido por *você* via dashboard, `hermes cron create/edit --model … --provider …`, ou editando `~/.hermes/cron/jobs.json`. Uma vez definido, permanece até você mudar. A ferramenta `cronjob` do agente não pode definir ou mudar modelos por job — pins de inferência são de propriedade do usuário.
- **`cron.model` / `cron.model_provider`** — default da frota cron: todo job não pinado roda neste modelo, independente do seu modelo de chat. Defina uma vez (`hermes config set cron.model <name>`) e trocar seu modelo de chat com `hermes model` ou `/model` nunca toca sua frota cron.
- **Default global** — só quando nenhum dos acima está definido um job segue `hermes model`. Neste caso o Hermes **snapshot** provider e modelo na criação, e se o default global mudar depois o job **falha fechado**: pula a execução, não faz chamada de inferência e alerta você **uma vez** — o job permanece pulado (e silencioso) nos ticks seguintes até você agir ou a config ser restaurada (#44585). Para jobs recorrentes ou de outro modo repetíveis, pinar o provider/modelo explicitamente (`hermes cron edit <job_id> --provider <provider> --model <model>`) para continuar. Um one-shot finito já consumido não pode ser atualizado; crie um novo one-shot futuro com provider e modelo explícitos. Isso impede job desassistido de herdar silenciosamente troca para provider/modelo pago. Definir `cron.model` (ou pin por job) é a forma deliberada de rotear gasto cron, e o drift guard não entra num eixo coberto por ele. Operadores que em vez disso querem jobs não pinados acompanharem o default global mutável podem [desabilitar o drift guard](#letting-unpinned-jobs-track-global-defaults).

`hermes setup --portal` é a opção de menor atrito para execuções desassistidas já que refresh OAuth é automático. Veja [Nous Portal](/integrations/nous-portal).
:::

:::tip
**Esforço de reasoning por job.** Um job pode fixar seu próprio nível de thinking, independente do pin de modelo: um de `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra`. Quando definido, sobrescreve tanto o `agent.reasoning_effort` global quanto os `agent.reasoning_overrides` por modelo nas runs daquele job (`none` desabilita thinking). Defina via `hermes cron create/edit --reasoning-effort high`; passe uma string vazia no edit para limpar o pin e seguir a config de novo. (Deliberadamente não é exposto na ferramenta `cronjob` do agente — configuração de modelo permanece decisão do usuário.) Níveis que um modelo não suporte são clamped ou omitidos pelo provider no request — fixar `xhigh` num modelo que limita em `high` roda em `high`. O pin não tem efeito em jobs `no_agent` (não há chamada LLM para ajustar). Use para rodar análises agendadas pesadas em `high` enquanto jobs recorrentes baratos rodam em `minimal`, sem tocar no default global.
:::

:::warning
Sessões rodadas por cron não podem criar recursivamente mais jobs cron. O Hermes desabilita ferramentas de gestão cron dentro de execuções cron para prevenir loops de agendamento descontrolados.
:::

## Criando tarefas agendadas {#creating-scheduled-tasks}

### No chat com `/cron` {#in-chat-with-cron}

```bash
/cron add 30m "Remind me to check the build"
/cron add "every 2h" "Check server status"
/cron add "every 1h" "Summarize new feed items" --skill blogwatcher
/cron add "every 1h" "Use both skills and combine the result" --skill blogwatcher --skill maps
```

### Pela CLI standalone {#from-the-standalone-cli}

```bash
hermes cron create "every 2h" "Check server status"
hermes cron create "every 1h" "Summarize new feed items" --skill blogwatcher
hermes cron create "every 1h" "Use both skills and combine the result" \
  --skill blogwatcher \
  --skill maps \
  --name "Skill combo"
```

### Por conversa natural {#through-natural-conversation}

Peça ao Hermes normalmente:

```text
Every morning at 9am, check Hacker News for AI news and send me a summary on Telegram.
```

O Hermes usará a ferramenta unificada `cronjob` internamente.

## Validação de configuração pré-dispatch {#pre-dispatch-configuration-validation}

Antes de construir qualquer maquinaria de agente para execução agendada, o scheduler
valida que a config do job pode de fato produzir execução bem-sucedida:

- a API key do provider resolve (pulada quando cadeia `fallback_providers` está
  configurada, já que o caminho fallback pode resgatar key primária ausente),
- skills anexadas estão prontas (sem env vars, comandos ou arquivos de credencial
  obrigatórios ausentes),
- targets de plataforma de entrega são conhecidos e têm credenciais de gateway configuradas
  (targets `local`/`origin` nunca são checados).

Quando a validação falha, `last_status` do job vira `blocked_config`, UM
alerta é entregue (não repetido a cada tick), e **nenhuma chamada LLM é
feita** — job mal configurado nunca gasta tokens. A próxima execução saudável limpa
o estado blocked para uma quebra futura de config alertar de novo.

Para desabilitar a validação e restaurar comportamento antigo (a execução prossegue e
falha durante execução):

```yaml
cron:
  preflight: false
```

Ou: `hermes config set cron.preflight false`

## Deixando jobs não pinados acompanhar defaults globais {#letting-unpinned-jobs-track-global-defaults}

O drift guard de modelo/provider está habilitado por padrão. Se seus jobs cron não pinados
devem deliberadamente seguir toda mudança global de modelo ou provider, desabilite
em `config.yaml`:

```yaml
cron:
  model_drift_guard: false
```

Ou use o comando de config:

```bash
hermes config set cron.model_drift_guard false
```

Isso desabilita tanto o bloqueio runtime quanto o aviso mostrado quando configurações globais
de inferência mudam. Snapshots existentes permanecem armazenados, então voltar a opção para
`true` re-habilita proteção sem recriar jobs.

:::warning
Com o guard desabilitado, jobs desassistidos não pinados herdam imediatamente defaults globais
alterados. Troca para provider ou modelo pago pode portanto gastar dinheiro
a cada execução agendada.
:::

## Jobs cron com skills {#skill-backed-cron-jobs}

Um job cron pode carregar uma ou mais skills antes de rodar o prompt.

### Skill única {#single-skill}

```python
cronjob(
    action="create",
    skill="blogwatcher",
    prompt="Check the configured feeds and summarize anything new.",
    schedule="0 9 * * *",
    name="Morning feeds",
)
```

### Múltiplas skills {#multiple-skills}

Skills são carregadas em ordem. O prompt vira a instrução de tarefa sobreposta a essas skills.

```python
cronjob(
    action="create",
    skills=["blogwatcher", "maps"],
    prompt="Look for new local events and interesting nearby places, then combine them into one short brief.",
    schedule="every 6h",
    name="Local brief",
)
```

Útil quando quer um agente agendado que herda workflows reutilizáveis sem encher o prompt cron com texto completo da skill.

## Rodando job dentro de diretório de projeto {#running-a-job-inside-a-project-directory}

Jobs cron rodam por padrão detached de qualquer repo — nenhum `AGENTS.md`, `CLAUDE.md` ou `.cursorrules` é carregado, e ferramentas terminal / file / code-exec rodam do working directory onde o gateway iniciou. Passe `--workdir` (CLI) ou `workdir=` (tool call) para mudar:

```bash
# CLI standalone (schedule e prompt são posicionais)
hermes cron create "every 1d at 09:00" \
  "Audit open PRs, summarize CI health, and post to #eng" \
  --workdir /home/me/projects/acme
```

```python
# Do chat, via ferramenta cronjob
cronjob(
    action="create",
    schedule="every 1d at 09:00",
    workdir="/home/me/projects/acme",
    prompt="Audit open PRs, summarize CI health, and post to #eng",
)
```

Quando `workdir` está definido:

- `AGENTS.md`, `CLAUDE.md` e `.cursorrules` desse diretório são injetados no system prompt (mesma ordem de descoberta que CLI interativa)
- `terminal`, `read_file`, `write_file`, `patch`, `search_files` e `execute_code` usam esse diretório como working directory
- O caminho deve ser diretório absoluto existente — caminhos relativos e diretórios ausentes são rejeitados em create / update
- Passe `--workdir ""` (ou `workdir=""` via ferramenta) no edit para limpar e restaurar comportamento antigo

:::note Serialização
Jobs com `workdir` rodam sequencialmente no tick do scheduler, não no pool paralelo. Isso é deliberado: o worker cron aplica o workdir do job via estado global de terminal do processo, então dois jobs workdir rodando ao mesmo tempo corromperiam o cwd um do outro. Jobs sem workdir ainda rodam em paralelo como antes.
:::

## Editando jobs {#editing-jobs}

Você não precisa deletar e recriar jobs só para mudá-los.

:::tip Referência de job
O placeholder `<job_id>` abaixo (e em [Ações de lifecycle](#lifecycle-actions)) também aceita o **nome** do job (case-insensitive) — útil quando lembra `morning-digest` mas não o ID hex. ID exato de job tem precedência sobre matches por nome; se a referência não for ID e um nome corresponder a mais de um job, o comando recusa e imprime IDs candidatos para desambiguar.
:::

### Chat {#chat}

```bash
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Use the revised task"
/cron edit <job_id> --skill blogwatcher --skill maps
/cron edit <job_id> --remove-skill blogwatcher
/cron edit <job_id> --clear-skills
```

### CLI standalone {#standalone-cli}

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
- `--add-skill` anexa à lista existente sem substituir
- `--remove-skill` remove skills anexadas específicas
- `--clear-skills` remove todas as skills anexadas

## Ações de lifecycle {#lifecycle-actions}

Jobs cron agora têm lifecycle mais completo que só create/remove.

### Chat {#chat-1}

```bash
/cron list
/cron pause <job_id>
/cron resume <job_id>
/cron run <job_id>
/cron remove <job_id>
```

### CLI standalone {#standalone-cli-1}

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

- `pause` — mantém o job mas para de agendar
- `resume` — re-habilita o job e calcula próxima execução futura
- `run` — dispara o job no próximo tick do scheduler
- `remove` — deleta por completo
- `edit` — modifica schedule, prompt, entrega, etc.

**Lookup por nome.** Todos os quatro verbos mutantes (`pause`, `resume`, `run`, `remove`, `edit`) mais a ferramenta `cronjob` do agente agora aceitam **nome** de job (case-insensitive) no lugar do ID hex. Agente e CLI preferem match exato de ID se existir; matches ambíguos por nome (múltiplos jobs com mesmo nome) são recusados com lista completa de IDs candidatos para escolher explicitamente. Nomes não são únicos, então este guard é load-bearing — impede mutar silenciosamente o job errado quando dois compartilham nome.

## Agendamento gerenciado pelo agente (jobs cron que gerenciam jobs cron) {#agent-managed-scheduling-cron-jobs-that-manage-cron-jobs}

Por padrão, agentes lançados *pelo* scheduler não podem usar a ferramenta `cronjob` —
um job agendado não pode criar, editar ou remover outros jobs. Opt-in via
`config.yaml`:

```yaml
cron:
  allow_agent_scheduling: true   # default: false
```

Quando habilitado, um agente agendado pode gerenciar a tabela cron como qualquer sessão
de chat: agendar one-shots de acompanhamento de dentro de trabalho agendado, ajustar
sua própria cadência, ou rodar um job "cron librarian" que reconcilia a tabela inteira
(listar, depois update/remove/create conforme necessário). Duas propriedades mantêm
isso coerente:

- **Uma tabela plana, de propriedade do usuário.** Jobs criados a partir de uma execução
  cron caem no mesmo `jobs.json` que qualquer outro job, sem ownership especial — você
  pode listar, editar ou removê-los exatamente como se os tivesse criado você mesmo.
- **Sem entrega pendurada.** Uma execução cron é efêmera, então `deliver: origin`
  de dentro de uma é resolvido **no momento da criação** para o target concreto do
  próprio job criador (`platform:chat_id[:thread_id]`, ou `local` se o job criador
  não entrega em lugar nenhum). Um job criado por um agente agendado nunca pode apontar
  sua saída para uma sessão que não existe mais. Targets explícitos
  (`local`, `all`, `telegram:<chat_id>`) são honrados verbatim.

Prefira prompts que atualizam jobs existentes (liste primeiro, depois atualize por ID)
em vez de prompts que criam jobs novos a cada execução.

## Como funciona {#how-it-works}

**Execução cron é tratada pelo daemon gateway.** O gateway faz tick no scheduler a cada 60 segundos, rodando jobs devidos em sessões de agente isoladas.

```bash
hermes gateway install     # Instalar como serviço de usuário
sudo hermes gateway install --system   # Linux: serviço de sistema boot-time para servidores
hermes gateway             # Ou rodar em foreground

hermes cron list
hermes cron status
```

### Comportamento do scheduler gateway {#gateway-scheduler-behavior}

A cada tick o Hermes:

1. carrega jobs de `~/.hermes/cron/jobs.json`
2. checa `next_run_at` contra hora atual
3. inicia sessão `AIAgent` nova para cada job devido
4. opcionalmente injeta uma ou mais skills anexadas nessa sessão nova
5. roda o prompt até completar
6. entrega a resposta final
7. atualiza metadados de execução e próximo horário agendado

File lock em `~/.hermes/cron/.tick.lock` impede ticks sobrepostos do scheduler de double-run do mesmo batch de jobs.

### Histórico de execução {#execution-history}

O Hermes registra cada tentativa cron reivindicada no
`~/.hermes/cron/executions.db` local ao profile antes de dispatch do executor ou provider. Tentativas
passam por `claimed`, `running`, e um estado terminal imutável:
`completed`, `failed`, ou `unknown`. Após restart, o Hermes marca tentativa abandonada
`unknown` só quando PID original e fingerprint de start do processo provam
que seu dono se foi. Tentativas unknown são registros de auditoria e nunca
são rerun automaticamente.

Inspecione tentativas recentes com `hermes cron runs [job-id] --limit 20` (alias:
`history`). Histórico terminal é limitado; tentativas ativas nunca são podadas. O
ledger está incluído em quick backups.

### Nudge de revisão por falhas repetidas {#repeated-failure-review-nudge}

Cada job rastreia um `failure_streak` — execuções consecutivas em que o agente falhou
(falhas de entrega não contam). Quando o streak de um job *recorrente* atinge o
limiar, a mensagem de falha entregue no chat ganha um nudge de revisão dizendo
que o job falhou N execuções seguidas e sugerindo que você conserte, pause
(`hermes cron pause <job>`), ou remova. Qualquer execução bem-sucedida zera o
streak, e `hermes cron list` mostra o streak ao lado da última execução de um job
falhando. Jobs one-shot nunca enviam nudge.

```yaml
cron:
  failure_nudge_threshold: 3   # default; 0 disables the nudge
```

## Opções de entrega {#delivery-options}

Ao agendar jobs, você especifica para onde vai a saída:

| Opção | Descrição | Exemplo |
|--------|-------------|---------|
| `"origin"` | De volta a onde o job foi criado | Padrão em plataformas de mensagens |
| `"local"` | Salvar só em arquivos locais (`~/.hermes/cron/output/`) | Padrão na CLI |
| `"telegram"` | Canal home Telegram | Usa `TELEGRAM_HOME_CHANNEL` |
| `"telegram:123456"` | Chat Telegram específico por ID | Entrega direta |
| `"telegram:-100123:17585"` | Tópico Telegram específico | Formato `chat_id:thread_id` |
| `"discord"` | Canal home Discord | Usa `DISCORD_HOME_CHANNEL` |
| `"discord:#engineering"` | Canal Discord específico | Por nome de canal |
| `"slack"` | Canal home Slack | |
| `"whatsapp"` | Home WhatsApp | |
| `"signal"` | Signal | |
| `"matrix"` | Sala home Matrix | |
| `"mattermost"` | Canal home Mattermost | |
| `"email"` | Email | |
| `"sms"` | SMS via Twilio | |
| `"homeassistant"` | Home Assistant | |
| `"dingtalk"` | DingTalk | |
| `"feishu"` | Feishu/Lark | |
| `"wecom"` | WeCom | |
| `"weixin"` | Weixin (WeChat) | |
| `"bluebubbles"` | BlueBubbles (iMessage) | |
| `"qqbot"` | QQ Bot (Tencent QQ) | |
| `"all"` | Fan-out para todo canal home conectado | Resolvido no fire time |
| `"telegram,discord"` | Fan-out para conjunto específico de canais | Lista separada por vírgula |
| `"origin,all"` | Entrega na origem **mais** todo outro canal conectado | Combine quaisquer tokens |

A resposta final do agente é entregue automaticamente ao target `deliver:` configurado — o agente não envia mensagens sozinho, então não há nada para chamar no prompt cron.

### Intenção de roteamento (`all`) {#routing-intent-all}

`all` permite enviar um job cron para todo canal de mensagens que você configurou, sem enumerar por nome. É **resolvido no fire time**, então job criado antes de você configurar Telegram pega Telegram no próximo tick após definir `TELEGRAM_HOME_CHANNEL`.

Semântica: `all` expande para toda plataforma com canal home configurado. Zero é ok; o job simplesmente não produz targets de entrega e é registrado como falha de entrega upstream.

`all` compõe com targets explícitos. `origin,all` entrega no chat de origem *mais* todo outro canal home conectado, deduplicando por `(platform, chat_id, thread_id)`.

### Tópico cron Telegram (`TELEGRAM_CRON_THREAD_ID`) {#telegram-cron-topic-telegram_cron_thread_id}

Com modo de tópico Telegram habilitado, o DM raiz fica reservado como lobby de sistema — respostas enviadas lá são rechaçadas com lembrete de lobby e `reply_to_message_id` é dropado, então você não pode responder a mensagem cron que caiu no chat principal.

Aponte cron para tópico de fórum dedicado:

1. No Telegram, abra o DM do bot e crie tópico ex. `Cron`. Long-press no header do tópico → **Copy link**; o inteiro final é o `message_thread_id` do tópico.
2. Defina `TELEGRAM_CRON_THREAD_ID=<that id>` no seu `.env`.

Isso aplica só a entregas cron. `TELEGRAM_HOME_CHANNEL_THREAD_ID` (usado em outros lugares, ex. notificações de restart) permanece inalterado. Targets explícitos `deliver="telegram:chat_id:thread_id"` continuam vencendo sobre a env var. Respostas a mensagens cron agora chegam na sessão de tópico existente, para agir diretamente.

### Envelope de resposta {#response-wrapping}

Por padrão, saída cron entregue vem envelopada com header e footer para o destinatário saber que veio de tarefa agendada:

```
Cronjob Response: Morning feeds
-------------

<agent output here>

Note: The agent cannot see this message, and therefore cannot respond to it.
```

Para entregar saída bruta do agente sem envelope, defina `cron.wrap_response` como `false`:

```yaml
# ~/.hermes/config.yaml
cron:
  wrap_response: false
```

### Jobs continuáveis (responder a entrega cron) {#continuable-jobs-reply-to-a-cron-delivery}

Por padrão entrega cron é fire-and-forget: a mensagem é enviada, mas não
vive no histórico de conversa do chat, então se você responder o agente
não tem registro do que disse. Defina job **continuable** e o brief entregue
vira conversa em que pode responder — o agente tem o brief no contexto
em vez de perguntar "what is Task #2?".

Opt-in, **desligado por padrão**. Habilite globalmente na config, ou por job via
`attach_to_session` da ferramenta `cronjob` (que sobrescreve config global para aquele job):

```yaml
# ~/.hermes/config.yaml
cron:
  mirror_delivery: false   # true para tornar entregas cron continuáveis
```

Comportamento é **thread-preferred**, escopado ao chat de origem do job:

- **Plataformas com thread** (tópicos Telegram, threads Discord/Slack): cada
  entrega abre thread dedicada própria e o brief é seeded na sessão dessa
  thread, então reply in-thread continua com contexto completo. Job recorrente
  (ex. brief diário) abre thread nova por execução, isolando discussão de follow-up de cada entrega.
- **Plataformas só DM** (WhatsApp, Signal, SMS): threads não existem, então o brief
  é espelhado na sessão DM de origem — o próprio DM é a superfície de continuação.

Só o chat de origem é tocado: targets fan-out / broadcast (`all`,
entregas explícitas em outros chats) nunca são tornados continuáveis. O espelho é
escrito como turno user rotulado (`[Cron delivery: <task name>]`), o que mantém
alternação do histórico de conversa segura em todos os providers de modelo.

#### Continuação flat in-channel (Slack) {#flat-in-channel-continuation-slack}

O comportamento thread-preferred acima cria thread dedicada a cada
entrega. Se preferir job continuável cair **flat na timeline do canal
** — sem thread — defina a **superfície continuável** Slack como `in_channel`:

```yaml
# ~/.hermes/config.yaml
slack:
  cron_continuable_surface: in_channel   # padrão: thread
  reply_in_thread: false                 # par obrigatório (veja abaixo)
  require_mention: false                 # para reply simples continuar o job
```

Em modo `in_channel` o brief é entregue como mensagem top-level comum do canal
(sem thread aberta), e sua resposta continua o job via sessão compartilhada do canal. Três configs trabalham juntas:

- **`cron_continuable_surface: in_channel`** — pula criação de thread na entrega.
- **`reply_in_thread: false`** (obrigatório) — faz o bot responder sua reply
  *flat* no canal e keyar na mesma sessão whole-channel onde o brief foi seeded.
  Sem isso continuação ainda funciona mas chega num thread (fallback seguro para continuação estilo thread, nunca reply dropada — gateway loga warning no startup para spotar mismatch).
- **`require_mention: false`** (ou adicione canal a `free_response_channels`)
  — para responder com mensagem simples; senão o bot só acorda quando você
  `@`-mentiona a cada reply.

Como continuação é sessão **whole-channel**, é compartilhada: outro
chatter no canal — e segundo job continuável in-channel — entram na mesma
conversa rolling. Isso é inerente a "flat num canal" e é o mesmo
tradeoff que usuários de `reply_in_thread: false` já aceitam; use superfície
`thread` padrão quando quiser follow-up de cada entrega isolado.

Isso é capacidade Slack hoje. Outras plataformas aceitam a chave mas fazem fallback
para superfície `thread` (primitivas de continuação diferem); a escolha é
por plataforma, definida sob config de cada plataforma. É flag de config gateway-side
— `/restart` pega; sem reinstall do app Slack.

:::note DMs 1:1
`cron_continuable_surface` é config de **canal** — DM 1:1 não tem
split thread-vs-timeline para escolher (o DM já é flat), então a chave
não tem efeito lá. O que governa se entrega cron em DM é continuável é
o knob separado e pré-existente **`slack.dm_top_level_threads_as_sessions`**:

- **`false`** — todos DMs top-level compartilham uma sessão DM rolling, então brief cron continuável e sua reply caem na **mesma** sessão e o job continua em
  contexto. É o que você quer para cron continuável em DM.
- **`true`** (padrão) — cada mensagem DM top-level é sessão própria, então reply
  a brief entregue inicia sessão *nova* sem registro do brief.
  Continuação não funciona neste modo (para cron ou qualquer outra entrega flat).

Então para job cron continuável entregue em DM 1:1, defina
`slack.dm_top_level_threads_as_sessions: false`. `cron_continuable_surface` não
é necessário (e é ignorado) para DMs.
:::

### Supressão silenciosa {#silent-suppression}

Se a resposta final do agente contém `[SILENT]`, entrega é suprimida por completo. Saída ainda é salva localmente para auditoria (em `~/.hermes/cron/output/`), mas nenhuma mensagem vai ao target de entrega.

Útil para jobs de monitoramento que só devem reportar quando algo está errado:

```text
Check if nginx is running. If everything is healthy, respond with only [SILENT].
Otherwise, report the issue.
```

Jobs falhos sempre entregam independente do marcador `[SILENT]` — só execuções bem-sucedidas podem ser silenciadas. Para jobs de monitoramento quietos, peça ao agente responder só com `[SILENT]` quando não houver nada a reportar.

## Timeout de script {#script-timeout}

Scripts pré-execução (anexados via parâmetro `script`) têm timeout padrão de 3600 segundos (1 hora). Isso limita **só o script** — jobs baseados em skill / LLM rodam em orçamento de inatividade separado e não são limitados por este valor. Se seus scripts precisam de limite diferente, mude:

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 1800   # 30 minutos
```

Ou defina env var `HERMES_CRON_SCRIPT_TIMEOUT`. Ordem de resolução: env var → config.yaml → default 3600s.

O cron também limita a limpeza pós-execução de sessão e recursos do agente. Isso acontece depois que o turno LLM retorna, então é separado do timeout de inatividade. O padrão é 10 segundos por operação de limpeza. Se um finalizador de storage ou client parar de retornar, o scheduler registra um erro, libera o guard in-flight do job e permite que execuções posteriores sejam despachadas em vez de pular aquele job para sempre.

```yaml
# ~/.hermes/config.yaml
cron:
  cleanup_timeout_seconds: 10
```

Defina `cleanup_timeout_seconds: 0` só para restaurar o comportamento legado de limpeza sem limite.

## Timeout de envio de mídia {#media-send-timeout}

Quando uma entrega cron inclui anexos de mídia (um PDF gerado, áudio TTS, um relatório exportado) enviados por um adapter de gateway live, cada upload de anexo é limitado por um timeout — 300 segundos por padrão. Arquivos grandes em uplinks lentos podem precisar de mais:

```yaml
# ~/.hermes/config.yaml
cron:
  media_send_timeout_seconds: 600   # 10 minutes per attachment
```

Ou defina a variável de ambiente `HERMES_CRON_MEDIA_SEND_TIMEOUT`. A ordem de resolução é: env var → config.yaml → default 300s. Um anexo que estoura o timeout é registrado no status da run do job como falha parcial de entrega (o texto ainda entrega).

## Modo no-agent (jobs só script) {#no-agent-mode-script-only-jobs}

Para jobs recorrentes que não precisam de raciocínio LLM — watchdogs clássicos, alertas de disco/memória, heartbeats, pings CI — passe `no_agent=True` na criação. O scheduler roda seu script no schedule e entrega stdout diretamente, pulando o agente por completo:

```bash
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"
```

Semântica:

- stdout do script (trimmed) → entregue verbatim como mensagem.
- **stdout vazio → tick silencioso**, sem entrega. Padrão watchdog: "só falar quando algo está errado".
- Exit não-zero ou timeout → alerta de erro é entregue, para watchdog quebrado não falhar silenciosamente.
- `{"wakeAgent": false}` na última linha → tick silencioso (mesmo gate que jobs LLM usam).
- Sem tokens, sem modelo, sem fallback de provider — job nunca toca camada de inferência.

Arquivos `.sh` / `.bash` rodam sob `bash` do `PATH` quando disponível, senão `/bin/bash` (importante no Windows Git Bash). Qualquer outra coisa roda sob interpretador Python atual (`sys.executable`). Scripts devem resolver dentro de `$HERMES_HOME/scripts/` — nomes relativos, caminhos absolutos e caminhos com prefixo `~` são aceitos quando o target resolvido permanece nesse diretório; caminhos que escapam são rejeitados. Env de subprocesso é sanitizado (`_sanitize_subprocess_env`): credenciais API de provider e outros segredos gerenciados pelo Hermes **não** são herdados por scripts cron.

### O agente configura isso para você {#the-agent-sets-these-up-for-you}

O schema da ferramenta `cronjob` expõe `no_agent` ao Hermes diretamente, então você descreve watchdog no chat e deixa o agente configurar:

```text
Ping me on Telegram if RAM is over 85%, every 5 minutes.
```

O Hermes escreverá o script de checagem em `~/.hermes/scripts/` via `write_file`, depois chama:

```python
cronjob(action="create", schedule="every 5m",
        script="memory-watchdog.sh", no_agent=True,
        deliver="telegram", name="memory-watchdog")
```

Ele escolhe `no_agent=True` automaticamente quando conteúdo da mensagem é totalmente determinado pelo script (watchdogs, alertas de threshold, heartbeats). A mesma ferramenta também deixa o agente pausar, retomar, editar e remover jobs — lifecycle inteiro guiado por chat sem ninguém tocar CLI.

Veja o [guia Script-Only Cron Jobs](/guides/cron-script-only) para exemplos trabalhados.

## Encadeando jobs com `context_from` {#chaining-jobs-with-context_from}

Jobs cron rodam em sessões isoladas sem memória de execuções anteriores. Mas às vezes saída de um job é exatamente o que o próximo precisa. Parâmetro `context_from` conecta isso automaticamente — prompt do Job B recebe saída mais recente do Job A prepended como contexto em runtime.

```python
# Job 1: Coletar dados brutos
cronjob(
    action="create",
    prompt="Fetch the top 10 AI/ML stories from Hacker News. Save them to ~/.hermes/data/briefs/raw.md in markdown format with title, URL, and score.",
    schedule="0 7 * * *",
    name="AI News Collector",
)

# Job 2: Triagem — recebe saída do Job 1 como contexto
# Pegue ID do Job 1 de: cronjob(action="list")
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/raw.md. Score each story 1–10 for engagement potential and novelty. Output the top 5 to ~/.hermes/data/briefs/ranked.md.",
    schedule="30 7 * * *",
    context_from="<job1_id>",
    name="AI News Triage",
)

# Job 3: Ship — recebe saída do Job 2 como contexto
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/ranked.md. Write 3 tweet drafts (hook + body + hashtags). Deliver to telegram:7976161601.",
    schedule="0 8 * * *",
    context_from="<job2_id>",
    name="AI News Brief",
)
```

**Como funciona:**

- Quando Job 2 dispara, Hermes lê saída mais recente do Job 1 de `~/.hermes/cron/output/{job1_id}/*.md`
- Essa saída é prepended ao prompt do Job 2 automaticamente
- Job 2 não precisa hardcodar "read this file" — recebe conteúdo como contexto
- Cadeia pode ter qualquer comprimento: Job 1 → Job 2 → Job 3 → ...

**O que `context_from` aceita:**

| Formato | Exemplo |
|--------|---------|
| ID de job único (string) | `context_from="a1b2c3d4"` |
| Múltiplos IDs (list) | `context_from=["job_a", "job_b"]` |

Saídas são concatenadas na ordem listada.

**Continuidade: carregar a saída da execução anterior**

Defina `continuity=true` e o job injeta *sua própria* saída mais recente em cada execução. Jobs recorrentes normalmente começam cada execução com amnésia — um scout de notícias re-reporta as mesmas histórias, um monitor re-alerta na mesma condição. Com continuidade ligada, o job acorda vendo o que reportou da última vez e pode deduplicar e continuar de onde parou:

```python
cronjob(
    action="create",
    prompt="Scan HN and arXiv for new agent-tooling papers. Report only items NOT already covered in your previous run's output.",
    schedule="every 6h",
    continuity=True,
    name="Agent Tooling Scout",
)
```

A primeira execução não tem saída anterior, então o prompt roda como está. Em execuções seguintes a saída anterior é prepended com framing de continuidade ("evite repetir o que já foi reportado"). Combina livremente com jobs upstream (`context_from=["<other_job_id>"]` mais `continuity=true`), e `continuity=false` no update desliga isso preservando outras entradas de `context_from`. Internamente o flag é armazenado como a entrada reservada `self` em `context_from`.

Pela CLI: `hermes cron create "every 6h" "Scan for news" --continuity`, e `hermes cron edit <job_id> --continuity` / `--no-continuity` para alternar em um job existente. O mesmo toggle aparece no editor cron do dashboard e no diálogo de routines do Bot Mode no desktop.

**Quando usar:**

- Pipelines multi-estágio (collect → filter → format → deliver)
- Tarefas dependentes onde trabalho do passo N depende de saída do passo N−1
- Padrões fan-out/fan-in onde um job agrega resultados de vários outros
- Scouts/monitores recorrentes que devem deduplicar contra o próprio relatório anterior (`continuity=true`)

## Recuperação de provider {#provider-recovery}

Jobs cron herdam fallback providers configurados e rotação de credential pool. Se API key primária está rate-limited ou provider retorna erro, agente cron pode:

- **Fazer fallback para provider alternativo** se tem `fallback_providers` (ou legacy `fallback_model`) em `config.yaml`
- **Rotacionar para próxima credencial** no seu [credential pool](/user-guide/configuration#credential-pool-strategies) para o mesmo provider

Jobs cron de alta frequência ou em horários de pico são mais resilientes — uma key rate-limited não falha a execução inteira.

## Fires agendados perdidos (`last_fire_error`) {#missed-scheduled-fires-last_fire_error}

Em deployments hosted (managed-cron), um fire agendado viaja do scheduler da plataforma pelo dashboard até o API server interno do gateway. Se esse hand-off final falha — o processo do gateway está down, ou seu listener do API server nunca iniciou — a run nunca começa, então não há registro de execução nem `last_status` para inspecionar. O formato típico: o job funciona toda vez que você dispara manualmente, mas nunca auto-dispara.

Esses misses são carimbados no registro do job como `last_fire_error` (timestamp + motivo) e aparecem em:

- ferramenta `cronjob` → `action: "list"` — o campo `last_fire_error`
- `hermes cron list` — uma linha vermelha `⚠ Missed scheduled fire:` sob o job
- A view de job do dashboard

O carimbo sempre reflete a saúde **atual** de auto-fire: é sobrescrito por misses mais novos e limpo automaticamente pela próxima run bem-sucedida. Se você o vir, o job e o schedule estão bem — o lado gateway do caminho de fire precisa de atenção (mais comumente, reinicie o gateway pelo supervisor para carregar o ambiente completo do profile: `hermes gateway restart`).

### Catch-up de misfire {#misfire-catch-up}

Quando um provider externo de scheduler está ativo (managed cron em deployments hosted), o gateway também roda um sweep de catch-up: um job cujo horário agendado passou sem fire entregue — e cuja janela de grace já passou — é claimed e rodado localmente, então uma outage no hand-off de fire custa minutos em vez do dia inteiro. O sweep é de-duplicado contra retries tardios do scheduler pela mesma claim de store usada em fires normais.

```yaml
cron:
  misfire_grace_minutes: 10   # wait this long for the scheduler's own retries
                              # before catching up locally; 0 disables catch-up
```

Deployments locais (ticker built-in) não precisam disso — o ticker já pega jobs atrasados no próximo tick.

## Formatos de schedule {#schedule-formats}

Resposta final do agente é entregue automaticamente ao target `deliver:` do job — agente não dispara mensagens sozinho, então conteúdo user-facing vai simplesmente na resposta final. Para entregar em **targets adicionais ou diferentes**, liste múltiplos targets `deliver:` no job cron (separados por vírgula, ex. `deliver: "telegram,discord"`) em vez de fazer agente enviar.

### Atrasos relativos (one-shot) {#relative-delays-one-shot}

```text
30m     → Rodar uma vez em 30 minutos
2h      → Rodar uma vez em 2 horas
1d      → Rodar uma vez em 1 dia
```

### Intervalos (recorrentes) {#intervals-recurring}

```text
every 30m    → A cada 30 minutos
every 2h     → A cada 2 horas
every 1d     → Todo dia
```

### Expressões cron {#cron-expressions}

```text
0 9 * * *       → Diariamente às 9:00
0 9 * * 1-5     → Dias úteis às 9:00
0 */6 * * *     → A cada 6 horas
30 8 1 * *      → Primeiro dia de todo mês às 8:30
0 0 * * 0       → Todo domingo à meia-noite
```

### Timestamps ISO {#iso-timestamps}

```text
2026-03-15T09:00:00    → One-time em 15 de março de 2026 9:00
```

## Comportamento de repetição {#repeat-behavior}

| Tipo de schedule | Repetição padrão | Comportamento |
|--------------|----------------|----------|
| One-shot (`30m`, timestamp) | 1 | Roda uma vez |
| Intervalo (`every 2h`) | forever | Roda até remover |
| Expressão cron | forever | Roda até remover |

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

API voltada ao agente é uma ferramenta:

```python
cronjob(action="create", ...)
cronjob(action="list")
cronjob(action="update", job_id="...")
cronjob(action="pause", job_id="...")
cronjob(action="resume", job_id="...")
cronjob(action="run", job_id="...")
cronjob(action="remove", job_id="...")
```

Para `update`, passe `skills=[]` para remover todas skills anexadas.

### Execuções manuais são assíncronas {#manual-runs-are-asynchronous}

`cronjob(action="run")` dispara job imediatamente **em background** (como
`delegate_task`): chamada da ferramenta retorna na hora com handle, e outcome do job —
sucesso/falha, target de entrega, próxima execução agendada e excerpt de saída
— reentra conversa como nova mensagem quando execução termina.
Agente (e você) podem continuar trabalhando enquanto isso, e job que já está
mid-run é recusado com "already running" em vez de double-fire.

Você também pode passar `prompt` com `action="run"` para injetar contexto transitório por execução:

```python
cronjob(action="run", job_id="...", prompt="CONTEXT: focus on the EU region today")
```

Contexto é anexado ao prompt armazenado do job sob header `## Run Context`
só para aquele fire — nunca persiste na definição do job,
e passa pelo mesmo scan de prompt-injection que prompts armazenados.

Runtimes que não recebem resultados detached (`hermes -z` one-shot, `hermes
cron run` da CLI, sessões filhas cron, workers Kanban) fazem fallback para
execução síncrona automaticamente.

## Toolsets disponíveis para jobs cron {#toolsets-available-to-cron-jobs}

Cron roda cada job em sessão de agente nova sem plataforma de chat anexada. Por padrão agente cron recebe **toolset que você configurou para plataforma `cron` em `hermes tools`** — não default CLI, não tudo sob o sol.

```bash
hermes tools
# → escolha plataforma "cron" na UI curses
# → toggle toolsets on/off como faria para Telegram/Discord/etc.
```

Controle mais fino por job via campo `enabled_toolsets` em `cronjob.create` (ou job existente via `cronjob.update`):

```text
cronjob(action="create", name="weekly-news-summary",
        schedule="every sunday 9am",
        enabled_toolsets=["web", "file"],      # só web + file, sem terminal/browser/etc.
        prompt="Summarize this week's AI news: ...")
```

Quando `enabled_toolsets` está no job vence; senão config cron-platform de `hermes tools` vence; senão Hermes faz fallback para defaults built-in. Importa para controle de custo: carregar `browser`, `delegation` em todo job minúsculo "fetch news" incha prompt de tool-schema a cada chamada LLM.

### Pulando agente por completo: `wakeAgent` {#skipping-the-agent-entirely-wakeagent}

Se job cron anexa script de pre-check (via `script=`), script pode decidir em runtime se Hermes deve invocar agente. Emita linha final de stdout da forma:

```text
{"wakeAgent": false}
```

…e cron pula execução do agente por completo neste tick. Útil para polls frequentes (a cada 1–5 min) que só precisam acordar LLM quando estado mudou de fato — senão você paga por turns de agente sem conteúdo repetidamente.

```python
# script de pre-check
import json, sys
latest = fetch_latest_issue_count()
prev = read_state("issue_count")
if latest == prev:
    print(json.dumps({"wakeAgent": False}))   # skip this tick
    sys.exit(0)
write_state("issue_count", latest)
print(json.dumps({"wakeAgent": True, "context": {"new_issues": latest - prev}}))
```

Quando `wakeAgent` é omitido, default é `true` (acordar agente como usual).

#### Receitas: gates baratos de pre-run {#recipes-cheap-pre-run-gates}

Gate `wakeAgent` dá forma de $0 de decidir se job agendado deve gastar tokens LLM. Três padrões cobrem maioria dos casos.

**Gate de mudança de arquivo** — rodar só quando arquivo observado tem conteúdo novo desde último tick bem-sucedido. Scheduler registra `last_run_at` de cada job; compare contra mtime do arquivo.

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

**Gate de flag externa** — rodar só quando outro processo sinalizou prontidão (ex. hook de deploy dropa arquivo, job CI define valor no state store).

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

**Gate de contagem SQL** — rodar só quando há linhas novas no seu banco. Script também pode passar contagem ao agente via `context`, para agente saber quanto está olhando sem re-consultar.

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

Mesmo padrão funciona para qualquer fonte de dados consultável de script — Postgres, API HTTP, state store próprio — sem embutir avaliador SQL no subsistema cron.

:::tip
`~/.hermes/state.db` do próprio Hermes é schema interno que muda entre releases. Não consulte de gate de pre-run — aponte para seu banco ou feed.
:::

Crédito: este conjunto de receitas foi motivado pela exploração de @iankar8 em [#2654](https://github.com/NousResearch/hermes-agent/pull/2654), que propôs triggers sql/file/command como mecanismo paralelo. Gate `script` + `wakeAgent` já cobre os três casos a $0, então o trabalho virou documentação.

### Encadeando jobs: `context_from` {#chaining-jobs-context_from}

Job cron pode consumir saída bem-sucedida mais recente de um ou mais outros jobs listando nomes (ou IDs) em `context_from`:

```text
cronjob(action="create", name="daily-digest",
        schedule="every day 7am",
        context_from=["ai-news-fetch", "github-prs-fetch"],
        prompt="Write the daily digest using the outputs above.")
```

Saídas completadas mais recentes dos jobs referenciados são injetadas acima do prompt como contexto para esta execução. Cada entrada upstream deve ser ID ou nome de job válido (veja `cronjob action="list"`). Nota: encadeamento lê saída *mais recente completada* — não espera jobs upstream rodando no mesmo tick.

## Armazenamento de jobs {#job-storage}

Jobs ficam em `~/.hermes/cron/jobs.json`. Saída de execuções vai para `~/.hermes/cron/output/{job_id}/{timestamp}.md`.

Definições de job são JSON simples no disco: sobrevivem `hermes update`, restarts de gateway e reboots. Job mid-run durante restart é marcado `unknown` no ledger de execução — não é rerun automaticamente, mas próximo tick agendado dispara normalmente. Veja [Histórico de execução](#execution-history) para detalhes.

:::tip
Peça ao agente gerenciar jobs pela ferramenta `cronjob`, `hermes cron edit` ou `/cron` — não patchando `jobs.json` diretamente. Edições diretas podem falhar silenciosamente quando [file write safety](../security.md#file-write-safety) bloqueia o caminho (ex. quando `HERMES_WRITE_SAFE_ROOT` está definido), e footer do [file-mutation verifier](../configuration.md#file-mutation-verifier) é sinal autoritativo de que nada foi salvo.
:::

Jobs podem armazenar `model` e `provider` como `null`. Quando campos omitidos, Hermes resolve em execution time da config global. Só aparecem no registro do job quando override por job está definido.

Armazenamento usa escritas atômicas de arquivo para escritas interrompidas não deixarem job file parcialmente escrito.

## Prompts autocontidos ainda importam {#self-contained-prompts-still-matter}

:::warning Importante
Jobs cron rodam em sessão de agente completamente nova. Prompt deve conter tudo que agente precisa que não já venha de skills anexadas.
:::

**RUIM:** `"Check on that server issue"`

**BOM:** `"SSH into server 192.168.1.100 as user 'deploy', check if nginx is running with 'systemctl status nginx', and verify https://example.com returns HTTP 200."`

## Segurança {#security}

Prompts de tarefas agendadas são escaneados por padrões de prompt-injection e exfiltração de credenciais em create e update. Prompts com truques Unicode invisíveis, tentativas de backdoor SSH ou payloads óbvios de exfiltração de segredos são bloqueados.
