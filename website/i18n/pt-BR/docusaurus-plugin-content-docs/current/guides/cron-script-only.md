---
sidebar_position: 13
title: "Tarefas Cron Somente com Script (Sem LLM)"
description: "Tarefas cron watchdog clássicas que ignoram completamente o LLM — um script é executado por agendamento e sua saída padrão (se houver) é entregue à sua plataforma de mensageria. Alertas de memória, alertas de disco, pings de CI, verificações de saúde periódicas."
---

# Tarefas Cron Somente com Script {#script-only-cron-jobs}

Às vezes você já sabe exatamente qual mensagem quer enviar. Você não precisa de um agente para raciocinar sobre isso — só precisa de um script executando em um timer, e sua saída (se houver) chegando ao Telegram / Discord / Slack / Signal.

O Hermes chama isso de **modo sem agente**. É o sistema de cron menos o LLM.

<!-- ascii-guard-ignore -->
```
   ┌──────────────────┐          ┌──────────────────┐
   │ scheduler tick   │  every   │ run script       │
   │ (every N minutes)│ ──────▶ │ (bash or python) │
   └──────────────────┘          └──────────────────┘
                                          │
                                          │ stdout
                                          ▼
                                 ┌──────────────────┐
                                 │ delivery router  │
                                 │ (telegram/disc…) │
                                 └──────────────────┘
```
<!-- ascii-guard-ignore-end -->

- **Sem chamada de LLM.** Zero tokens, zero loop de agente, zero gasto de modelo.
- **O script é a tarefa.** O script decide se deve alertar. Emite saída → a mensagem é enviada. Não emite nada → tick silencioso.
- **Bash ou Python.** Arquivos `.sh` / `.bash` são executados sob `/bin/bash`; qualquer outra extensão é executada sob o interpretador Python atual. Qualquer coisa em `~/.hermes/scripts/` é aceita.
- **Mesmo agendador.** Vive no `cronjob` junto com as tarefas de LLM — pausar, retomar, listar, logs e roteamento de entrega funcionam todos da mesma forma.

## Quando Usá-lo {#when-to-use-it}

Use o modo sem agente para:

- **Watchdogs de memória / disco / GPU.** Execute a cada 5 minutos, alerte apenas quando um limiar for violado.
- **Hooks de CI.** Deploy terminado → publique o SHA do commit. Build falhou → envie as últimas 100 linhas do log.
- **Métricas periódicas.** "Receita diária do Stripe às 9h" como uma simples chamada de API + formatação bonita.
- **Verificadores de eventos externos.** Verifique uma API, alerte em mudança de estado.
- **Heartbeats.** Faça ping em um dashboard a cada N minutos para provar que o host está vivo.

Use uma tarefa cron normal (orientada por LLM) quando você precisar que o agente **decida** o que dizer — resumir um documento longo, escolher itens interessantes de um feed, redigir uma mensagem amigável para humanos. O caminho sem agente é para casos em que a saída padrão do script já É a mensagem.

## Crie Uma a Partir do Chat {#create-one-from-chat}

O grande trunfo do modo sem agente é que o próprio agente pode configurar o watchdog para você — sem editor, sem shell, sem precisar lembrar flags da CLI. Você descreve o que quer, o Hermes escreve o script, o agenda e te avisa quando ele vai disparar.

### Exemplo de transcrição {#example-transcript}

> **Você:** me avise no telegram se a RAM passar de 85% a cada 5 minutos
>
> **Hermes:** *(escreve `~/.hermes/scripts/memory-watchdog.sh`, depois chama `cronjob(...)` com `no_agent=true`)*
>
> Configurado. Executa a cada 5 min, alerta no Telegram apenas quando a RAM está acima de 85%. Script: `memory-watchdog.sh`. ID da tarefa: `abc123`.

Por baixo dos panos, o agente faz duas chamadas de ferramenta:

```python
# 1. Write the check script
write_file(
    path="~/.hermes/scripts/memory-watchdog.sh",
    content='''#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  echo "RAM ${ram_pct}% on $(hostname)"
fi
# Empty stdout = silent tick; no message sent.
''',
)

# 2. Schedule it — no_agent=True skips the LLM on every tick
cronjob(
    action="create",
    schedule="every 5m",
    script="memory-watchdog.sh",
    no_agent=True,
    deliver="telegram",
    name="memory-watchdog",
)
```

A partir desse ponto, cada tick é gratuito: o agendador executa o script, envia sua saída padrão para o Telegram se não estiver vazia, e nunca toca em um modelo.

### O que o agente decide por você {#what-the-agent-decides-for-you}

Quando você formula uma solicitação como "me avise quando X" ou "a cada N minutos verifique Y e me diga se Z", a descrição da ferramenta `cronjob` do Hermes diz a ele para usar `no_agent=True` sempre que o conteúdo da mensagem for totalmente determinado pelo script. Ele recorre ao caminho normal orientado por LLM quando a solicitação exige raciocínio (*"resuma as novas issues"*, *"escolha as manchetes mais interessantes"*, *"redija um lembrete amigável"*).

Você não precisa especificar `--no-agent` você mesmo. Basta descrever o comportamento.

### Gerenciando watchdogs a partir do chat {#managing-watchdogs-from-chat}

O agente pode pausar, retomar, editar e remover tarefas da mesma forma que as cria:

> **Você:** pare o watchdog de memória por hoje à noite
>
> **Hermes:** *(chama `cronjob(action='pause', job_id='abc123')`)*
>
> Pausado. Retome com "ative de novo" ou via `hermes cron resume abc123`.

> **Você:** mude para a cada 15 minutos
>
> **Hermes:** *(chama `cronjob(action='update', job_id='abc123', schedule='every 15m')`)*

O ciclo de vida completo (criar / listar / atualizar / pausar / retomar / executar-agora / remover) está disponível para o agente sem que você precise aprender nenhum comando de CLI.

## Crie Uma a Partir da CLI {#create-one-from-the-cli}

Prefere o shell? O caminho da CLI dá a você o mesmo resultado com três comandos:

```bash
# 1. Write your script
cat > ~/.hermes/scripts/memory-watchdog.sh <<'EOF'
#!/usr/bin/env bash
# Alert when RAM usage is over 85%. Silent otherwise.
RAM_PCT=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$RAM_PCT" -ge 85 ]; then
  echo "⚠ RAM ${RAM_PCT}% on $(hostname)"
fi
# Empty stdout = silent run; no message sent.
EOF
chmod +x ~/.hermes/scripts/memory-watchdog.sh

# 2. Schedule it
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"

# 3. Verify
hermes cron list
hermes cron run <job_id>    # fire it once to test
```

É isso, no fim das contas. Sem prompt, sem skill, sem modelo.


## Como a Saída do Script Mapeia para a Entrega {#how-script-output-maps-to-delivery}

| Comportamento do script | Resultado |
|-----------------|--------|
| Exit 0, saída padrão não vazia | a saída padrão é entregue literalmente |
| Exit 0, saída padrão vazia | Tick silencioso — sem entrega |
| Exit 0, saída padrão contém `{"wakeAgent": false}` na última linha | Tick silencioso (portão compartilhado com tarefas de LLM) |
| Código de saída diferente de zero | Um alerta de erro é entregue (para que um watchdog quebrado não falhe silenciosamente) |
| Timeout do script | Um alerta de erro é entregue |

O comportamento de "silencioso quando vazio" é a chave do padrão clássico de watchdog: o script é livre para executar a cada minuto, mas o canal só vê uma mensagem quando algo realmente precisa de atenção.

## Regras dos Scripts {#script-rules}

Os scripts devem estar em `~/.hermes/scripts/`. Isso é aplicado tanto no momento de criação da tarefa quanto no momento de execução — caminhos absolutos, expansão de `~/` e padrões de travessia de caminho (`../`) são rejeitados. O mesmo diretório é compartilhado com o portão de script de pré-verificação usado pelas tarefas de LLM.

A escolha do interpretador é feita pela extensão do arquivo:

| Extensão | Interpretador |
|-----------|-------------|
| `.sh`, `.bash` | `/bin/bash` |
| qualquer outra | `sys.executable` (Python atual) |

Nós intencionalmente NÃO respeitamos shebangs `#!/...` — manter o interpretador definido de forma explícita e pequena reduz a superfície que o agendador confia.

## Sintaxe de Agendamento {#schedule-syntax}

Igual a todas as outras tarefas cron:

```bash
hermes cron create "every 5m"        # interval
hermes cron create "every 2h"
hermes cron create "0 9 * * *"       # standard cron: 9am daily
hermes cron create "30m"             # one-shot: run once in 30 minutes
```

Veja a [referência do recurso cron](/user-guide/features/cron) para a sintaxe completa.

## Destinos de Entrega {#delivery-targets}

`--deliver` aceita tudo o que o gateway conhece. Alguns formatos comuns:

```bash
--deliver telegram                       # platform home channel
--deliver telegram:-1001234567890        # specific chat
--deliver telegram:-1001234567890:17585  # specific Telegram forum topic
--deliver discord:#ops
--deliver slack:#engineering
--deliver signal:+15551234567
--deliver local                          # just save to ~/.hermes/cron/output/
```

Nenhum gateway em execução é necessário no momento da execução do script para plataformas de token de bot (Telegram, Discord, Slack, Signal, SMS, WhatsApp) — a ferramenta chama o endpoint REST de cada plataforma diretamente usando as credenciais já em `~/.hermes/.env` / `~/.hermes/config.yaml`.

## Edição e Ciclo de Vida {#editing-and-lifecycle}

```bash
hermes cron list                                    # see all jobs
hermes cron pause <job_id>                          # stop firing, keep definition
hermes cron resume <job_id>
hermes cron edit <job_id> --schedule "every 10m"    # adjust cadence
hermes cron edit <job_id> --agent                   # flip to LLM mode
hermes cron edit <job_id> --no-agent --script …     # flip back
hermes cron remove <job_id>                         # delete it
```

Tudo o que funciona em tarefas de LLM (pausar, retomar, disparo manual, alterações de destino de entrega) também funciona em tarefas sem agente.

## Exemplo Prático: Alerta de Espaço em Disco {#worked-example-disk-space-alert}

```bash
cat > ~/.hermes/scripts/disk-alert.sh <<'EOF'
#!/usr/bin/env bash
# Alert when / or /home is over 90% full.
THRESHOLD=90
df -h / /home 2>/dev/null | awk -v t="$THRESHOLD" '
  NR > 1 && $5+0 >= t {
    printf "⚠ Disk %s full on %s\n", $5, $6
  }
'
EOF
chmod +x ~/.hermes/scripts/disk-alert.sh

hermes cron create "*/15 * * * *" \
  --no-agent \
  --script disk-alert.sh \
  --deliver telegram \
  --name "disk-alert"
```

Silencioso quando ambos os sistemas de arquivos estão abaixo de 90%; dispara exatamente uma linha por sistema de arquivos acima do limiar quando um deles se enche.

## Comparação com Outros Padrões {#comparison-with-other-patterns}

| Abordagem | O que executa | Quando usar |
|----------|-----------|-------------|
| `cronjob --no-agent` (esta página) | Seu script no agendamento do Hermes | Watchdogs recorrentes / alertas / métricas que não precisam de raciocínio |
| `cronjob` (padrão, LLM) | Agente com script opcional de pré-verificação | Quando o conteúdo da mensagem exige raciocínio sobre dados |
| Cron do SO + `curl` para uma [inscrição de webhook](/user-guide/messaging/webhooks) | Seu script no agendamento do SO | Quando o Hermes pode estar indisponível (a própria coisa que você está monitorando) |

Para watchdogs críticos de saúde do sistema que precisam disparar *mesmo quando o gateway está fora do ar*, use o cron no nível do SO com um simples `curl` para uma inscrição de webhook do Hermes (ou qualquer endpoint de alerta externo) — esses são executados como processos independentes do SO e não dependem de o Hermes estar ativo. O agendador dentro do gateway é a escolha certa quando a coisa monitorada é externa.

## Relacionados {#related}

- [Automatize Qualquer Coisa com Cron](/guides/automate-with-cron) — padrões de cron orientados por LLM.
- [Referência de Tarefas Agendadas (Cron)](/user-guide/features/cron) — sintaxe completa de agendamento, ciclo de vida, roteamento de entrega.
- [Inscrições em Webhooks](/user-guide/messaging/webhooks) — pontos de entrada HTTP fire-and-forget para agendadores externos.
- [Internos do Gateway](/developer-guide/gateway-internals) — internos do roteador de entrega.
