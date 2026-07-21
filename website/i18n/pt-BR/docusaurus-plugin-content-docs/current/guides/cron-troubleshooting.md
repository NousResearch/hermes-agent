---
sidebar_position: 12
title: "Solução de Problemas do Cron"
description: "Diagnostique e corrija problemas comuns do cron do Hermes — tarefas que não disparam, falhas de entrega, erros de carregamento de skills e problemas de desempenho"
---

# Solução de Problemas do Cron {#cron-troubleshooting}

Quando uma tarefa cron não está se comportando como esperado, percorra estas verificações em ordem. A maioria dos problemas se encaixa em uma de quatro categorias: tempo, entrega, permissões ou carregamento de skills.

---

## Tarefas Não Disparam {#jobs-not-firing}

### Verificação 1: Confirme que a tarefa existe e está ativa {#check-1-verify-the-job-exists-and-is-active}

```bash
hermes cron list
```

Procure a tarefa e confirme que seu estado é `[active]` (não `[paused]` ou `[completed]`). Se mostrar `[completed]`, o contador de repetições pode estar esgotado — edite a tarefa para redefini-lo.

### Verificação 2: Confirme que o agendamento está correto {#check-2-confirm-the-schedule-is-correct}

Um agendamento mal formatado silenciosamente assume o padrão de disparo único ou é rejeitado completamente. Teste sua expressão:

| Sua expressão | Deve resultar em |
|----------------|-------------------|
| `0 9 * * *` | 9h00 todos os dias |
| `0 9 * * 1` | 9h00 toda segunda-feira |
| `every 2h` | A cada 2 horas a partir de agora |
| `30m` | 30 minutos a partir de agora |
| `2025-06-01T09:00:00` | 1º de junho de 2025 às 9h00 UTC |

Se a tarefa disparar uma vez e depois desaparecer da lista, é um agendamento de disparo único (`30m`, `1d`, ou um timestamp ISO) — comportamento esperado.

### Verificação 3: O gateway está em execução? {#check-3-is-the-gateway-running}

Tarefas cron são disparadas pela thread de ticker em segundo plano do gateway, que faz um tick a cada 60 segundos. Uma sessão de chat normal na CLI **não** dispara tarefas cron automaticamente.

Se você espera que as tarefas disparem automaticamente, você precisa de um gateway em execução (`hermes gateway` em primeiro plano, ou `hermes gateway start` para o serviço instalado). Para depuração pontual, você pode disparar manualmente um tick com `hermes cron tick`.

### Verificação 4: Verifique o relógio do sistema e o fuso horário {#check-4-check-the-system-clock-and-timezone}

As tarefas usam o fuso horário local. Se o relógio da sua máquina estiver errado ou em um fuso horário diferente do esperado, as tarefas dispararão nos horários errados. Verifique:

```bash
date
hermes cron list   # Compare next_run times with local time
```

---

## Falhas de Entrega {#delivery-failures}

### Verificação 1: Confirme que o destino de entrega está correto {#check-1-verify-the-deliver-target-is-correct}

Destinos de entrega diferenciam maiúsculas de minúsculas e exigem que a plataforma correta esteja configurada. Um destino mal configurado silenciosamente descarta a resposta.

| Destino | Requer |
|--------|----------|
| `telegram` | `TELEGRAM_BOT_TOKEN` em `~/.hermes/.env` |
| `discord` | `DISCORD_BOT_TOKEN` em `~/.hermes/.env` |
| `slack` | `SLACK_BOT_TOKEN` em `~/.hermes/.env` |
| `whatsapp` | Gateway do WhatsApp configurado |
| `signal` | Gateway do Signal configurado |
| `matrix` | Servidor Matrix configurado |
| `email` | SMTP configurado em `config.yaml` |
| `sms` | Provedor de SMS configurado |
| `local` | Acesso de escrita a `~/.hermes/cron/output/` |
| `origin` | Entrega no chat onde a tarefa foi criada |

Outras plataformas suportadas incluem `mattermost`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot` e `webhook`. Você também pode direcionar um chat específico com a sintaxe `platform:chat_id` (por exemplo, `telegram:-1001234567890`).

Se a entrega falhar, a tarefa ainda é executada — só não enviará a nenhum lugar. Verifique `hermes cron list` para o campo `last_error` atualizado (se disponível).

### Verificação 2: Verifique o uso de `[SILENT]` {#check-2-check-silent-usage}

Se sua tarefa cron não produz saída, a entrega é suprimida. Se a resposta do agente incluir o marcador de silêncio do cron `[SILENT]`, a entrega também é suprimida. Isso é intencional para tarefas de monitoramento — mas certifique-se de que seu prompt não esteja acidentalmente suprimindo tudo.

Use prompts como "responda apenas com [SILENT] se nada mudou." Evite pedir ao agente para incluir `[SILENT]` dentro de uma explicação mais longa, porque o cron trata esse marcador como um sinal de supressão.

### Verificação 3: Permissões de token da plataforma {#check-3-platform-token-permissions}

Cada bot de plataforma de mensageria precisa de permissões específicas para receber mensagens. Se a entrega falhar silenciosamente:

- **Telegram**: O bot deve ser administrador no grupo/canal de destino
- **Discord**: O bot deve ter permissão para enviar no canal de destino
- **Slack**: O bot deve ser adicionado ao workspace e ter o escopo `chat:write`

### Verificação 4: Empacotamento de resposta {#check-4-response-wrapping}

Por padrão, as respostas do cron são empacotadas com um cabeçalho e um rodapé (`cron.wrap_response: true` em `config.yaml`). Algumas plataformas ou integrações podem não lidar bem com isso. Para desabilitar:

```yaml
cron:
  wrap_response: false
```

---

## Falhas de Carregamento de Skills {#skill-loading-failures}

### Verificação 1: Confirme que as skills estão instaladas {#check-1-verify-skills-are-installed}

```bash
hermes skills list
```

Skills devem ser instaladas antes de poderem ser vinculadas a tarefas cron. Se uma skill estiver faltando, instale-a primeiro com `hermes skills install <skill-name>` ou via `/skills` na CLI.

### Verificação 2: Verifique o nome da skill vs. o nome da pasta da skill {#check-2-check-skill-name-vs-skill-folder-name}

Nomes de skills diferenciam maiúsculas de minúsculas e devem corresponder ao nome da pasta da skill instalada. Se sua tarefa especifica `ai-funding-report`, mas a pasta da skill é `ai-funding-daily-report`, confirme o nome exato em `hermes skills list`.

### Verificação 3: Skills que exigem ferramentas interativas {#check-3-skills-that-require-interactive-tools}

Tarefas cron são executadas com os toolsets `cronjob`, `messaging` e `clarify` desabilitados. Isso evita a criação recursiva de cron, o envio direto de mensagens (a entrega é feita pelo agendador) e prompts interativos. Se uma skill depende desses toolsets, ela não funcionará em um contexto de cron.

Verifique a documentação da skill para confirmar se ela funciona em modo não interativo (headless).

### Verificação 4: Ordenação de múltiplas skills {#check-4-multi-skill-ordering}

Ao usar várias skills, elas são carregadas em ordem. Se a Skill A depende de contexto da Skill B, certifique-se de que B seja carregada primeiro:

```bash
/cron add "0 9 * * *" "..." --skill context-skill --skill target-skill
```

Neste exemplo, `context-skill` é carregada antes de `target-skill`.

---

## Erros e Falhas de Tarefas {#job-errors-and-failures}

### Verificação 1: Revise a saída recente da tarefa {#check-1-review-recent-job-output}

Se uma tarefa foi executada e falhou, você pode ver o contexto do erro em:

1. O chat para onde a tarefa entrega (se a entrega for bem-sucedida)
2. `~/.hermes/logs/agent.log` para mensagens do agendador (ou `errors.log` para avisos)
3. Os metadados `last_run` da tarefa via `hermes cron list`

### Verificação 2: Padrões comuns de erro {#check-2-common-error-patterns}

**"No such file or directory" para scripts**
O caminho `script` deve ser um caminho absoluto (ou relativo ao diretório de configuração do Hermes). Verifique:
```bash
ls ~/.hermes/scripts/your-script.py   # Must exist
hermes cron edit <job_id> --script ~/.hermes/scripts/your-script.py
```

**"Skill not found" na execução da tarefa**
A skill deve estar instalada na máquina que executa o agendador. Se você mudar de máquina, as skills não sincronizam automaticamente — reinstale-as com `hermes skills install <skill-name>`.

**A tarefa é executada, mas não entrega nada**
Provavelmente um problema de destino de entrega (veja Falhas de Entrega acima), sem saída, ou uma resposta contendo o marcador de silêncio do cron `[SILENT]`.

**A tarefa fica presa ou expira**
O agendador usa um timeout baseado em inatividade (600s por padrão, configurável via a variável de ambiente `HERMES_CRON_TIMEOUT`, `0` para ilimitado). O agente pode executar por quanto tempo estiver ativamente chamando ferramentas — o temporizador só dispara após inatividade sustentada. Tarefas de longa duração devem usar scripts para lidar com a coleta de dados e entregar apenas o resultado.

### Verificação 3: Contenção de lock {#check-3-lock-contention}

O agendador usa bloqueio baseado em arquivo para evitar ticks sobrepostos. Se duas instâncias do gateway estiverem em execução (ou uma sessão de CLI conflitar com um gateway), as tarefas podem ser atrasadas ou puladas.

Encerre processos duplicados do gateway:
```bash
ps aux | grep hermes
# Kill duplicate processes, keep only one
```

### Verificação 4: Permissões em jobs.json {#check-4-permissions-on-jobsjson}

As tarefas são armazenadas em `~/.hermes/cron/jobs.json`. Se este arquivo não for legível/gravável pelo seu usuário, o agendador falhará silenciosamente:

```bash
ls -la ~/.hermes/cron/jobs.json
chmod 600 ~/.hermes/cron/jobs.json   # Your user should own it
```

---

## Problemas de Desempenho {#performance-issues}

### Inicialização lenta da tarefa {#slow-job-startup}

Cada tarefa cron cria uma nova sessão de AIAgent, o que pode envolver autenticação de provedor e carregamento de modelo. Para agendamentos sensíveis ao tempo, adicione um tempo de margem (por exemplo, `0 8 * * *` em vez de `0 9 * * *`).

### Muitas tarefas sobrepostas {#too-many-overlapping-jobs}

O agendador executa as tarefas sequencialmente dentro de cada tick. Se várias tarefas estão programadas para o mesmo horário, elas são executadas uma após a outra. Considere escalonar os agendamentos (por exemplo, `0 9 * * *` e `5 9 * * *` em vez de ambos em `0 9 * * *`) para evitar atrasos.

### Saída grande de script {#large-script-output}

Scripts que despejam megabytes de saída vão desacelerar o agente e podem atingir limites de tokens. Filtre/resuma no nível do script — emita apenas o que o agente precisa para raciocinar.

---

## Comandos de Diagnóstico {#diagnostic-commands}

```bash
hermes cron list                    # Show all jobs, states, next_run times
hermes cron run <job_id>            # Schedule for next tick (for testing)
hermes cron edit <job_id>           # Fix configuration issues
hermes logs                         # View recent Hermes logs
hermes skills list                  # Verify installed skills
```

---

## Obtendo Mais Ajuda {#getting-more-help}

Se você percorreu este guia e o problema persiste:

1. Execute a tarefa com `hermes cron run <job_id>` (dispara no próximo tick do gateway) e observe erros na saída do chat
2. Verifique `~/.hermes/logs/agent.log` para mensagens do agendador e `~/.hermes/logs/errors.log` para avisos
3. Abra uma issue em [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) com:
   - O ID e o agendamento da tarefa
   - O destino de entrega
   - O que você esperava vs. o que aconteceu
   - Mensagens de erro relevantes dos logs

---

*Para a referência completa do cron, veja [Automatize Qualquer Coisa com Cron](/guides/automate-with-cron) e [Tarefas Agendadas (Cron)](/user-guide/features/cron).*
