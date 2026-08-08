---
title: "Operar o Pipeline de Reuniões do Teams"
description: "Runbook, checklist de go-live e planilha de operador para o pipeline de reuniões do Microsoft Teams"
---

# Operar o Pipeline de Reuniões do Teams

Use este guia depois de já ter ativado o recurso em [Reuniões do Teams](/user-guide/messaging/teams-meetings).

Esta página cobre:
- fluxos de CLI do operador
- manutenção rotineira de inscrições
- triagem de falhas
- verificações de go-live
- planilha de lançamento

## Comandos Principais do Operador {#core-operator-commands}

### Valide o snapshot de configuração {#validate-the-config-snapshot}

```bash
hermes teams-pipeline validate
```

Use este primeiro após qualquer alteração de configuração.

### Inspecione a saúde do token {#inspect-token-health}

```bash
hermes teams-pipeline token-health
hermes teams-pipeline token-health --force-refresh
```

Use `--force-refresh` quando suspeitar de estado de autenticação obsoleto.

### Inspecione as inscrições {#inspect-subscriptions}

```bash
hermes teams-pipeline subscriptions
```

### Renove inscrições perto de expirar {#renew-near-expiry-subscriptions}

```bash
hermes teams-pipeline maintain-subscriptions
hermes teams-pipeline maintain-subscriptions --dry-run
```

### Automatizando a renovação de inscrições (OBRIGATÓRIO para produção) {#automating-subscription-renewal-required-for-production}

**As inscrições do Microsoft Graph expiram em no máximo 72 horas.** Se nada as renovar, as notificações de reunião silenciosamente parão depois de 3 dias e o pipeline vai parecer "quebrado". Este é o modo de falha operacional nº 1 para qualquer integração baseada no Graph.

Você DEVE executar o `maintain-subscriptions` em um cronograma. Escolha uma destas três opções:

#### Opção 1: Cron do Hermes (recomendado se você já roda o gateway do Hermes)

O Hermes vem com um agendador de tarefas (cron) integrado. O modo `--no-agent` roda um script como a tarefa (em vez de usar um LLM), e `--script` precisa apontar para um arquivo em `~/.hermes/scripts/`. Primeiro crie o script:

```bash
mkdir -p ~/.hermes/scripts
cat > ~/.hermes/scripts/maintain-teams-subscriptions.sh <<'EOF'
#!/usr/bin/env bash
exec hermes teams-pipeline maintain-subscriptions
EOF
chmod +x ~/.hermes/scripts/maintain-teams-subscriptions.sh
```

Depois registre uma tarefa agendada somente de script que roda a cada 12 horas (dá uma margem de 6x contra a janela de expiração de 72h):

```bash
hermes cron create "0 */12 * * *" \
  --name "teams-pipeline-maintain-subscriptions" \
  --no-agent \
  --script maintain-teams-subscriptions.sh \
  --deliver local
```

Verifique se foi registrada e inspecione o horário da próxima execução:

```bash
hermes cron list
hermes cron status        # scheduler status
```

#### Opção 2: timer do systemd (recomendado para implantações de produção em Linux)

Crie `/etc/systemd/system/hermes-teams-pipeline-maintain.service`:

```ini
[Unit]
Description=Hermes Teams pipeline subscription maintenance
After=network-online.target

[Service]
Type=oneshot
User=hermes
EnvironmentFile=/etc/hermes/env
ExecStart=/usr/local/bin/hermes teams-pipeline maintain-subscriptions
```

E `/etc/systemd/system/hermes-teams-pipeline-maintain.timer`:

```ini
[Unit]
Description=Run Hermes Teams pipeline subscription maintenance every 12 hours

[Timer]
OnBootSec=5min
OnUnitActiveSec=12h
Persistent=true

[Install]
WantedBy=timers.target
```

Ative:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-teams-pipeline-maintain.timer
systemctl list-timers hermes-teams-pipeline-maintain.timer
```

#### Opção 3: crontab simples

```cron
0 */12 * * * /usr/local/bin/hermes teams-pipeline maintain-subscriptions >> /var/log/hermes/teams-pipeline-maintain.log 2>&1
```

Certifique-se de que o ambiente do cron tenha as credenciais `MSGRAPH_*`. A solução mais simples: carregue (`source`) o `~/.hermes/.env` no início de um script wrapper que o crontab chama.

#### Verificando se a renovação está funcionando

Depois de configurar o cronograma, verifique a atividade de renovação após a primeira execução agendada:

```bash
hermes teams-pipeline subscriptions   # should show expirationDateTime advanced
hermes teams-pipeline maintain-subscriptions --dry-run   # should show "0 expiring soon" most of the time
```

Se você algum dia ver seu webhook do Graph misteriosamente "parar de funcionar" depois de exatamente ~72 horas, esta é a primeira coisa a verificar: a tarefa de renovação realmente rodou?

### Inspecione tarefas recentes {#inspect-recent-jobs}

```bash
hermes teams-pipeline list
hermes teams-pipeline list --status failed
hermes teams-pipeline show <job-id>
```

### Reproduza uma tarefa armazenada {#replay-a-stored-job}

```bash
hermes teams-pipeline run <job-id>
```

### Busca de teste (dry-run) de artefatos de reunião {#dry-run-meeting-artifact-fetches}

```bash
hermes teams-pipeline fetch --meeting-id <meeting-id>
hermes teams-pipeline fetch --join-web-url "<join-url>"
```

## Runbook de Rotina {#routine-runbook}

### Depois da primeira configuração {#after-first-setup}

Execute estes na ordem:

```bash
hermes teams-pipeline validate
hermes teams-pipeline token-health --force-refresh
hermes teams-pipeline subscriptions
```

Depois acione ou espere por um evento de reunião real e confirme:

```bash
hermes teams-pipeline list
hermes teams-pipeline show <job-id>
```

### Verificações diárias ou periódicas {#daily-or-periodic-checks}

- execute `hermes teams-pipeline maintain-subscriptions --dry-run`
- inspecione `hermes teams-pipeline list --status failed`
- verifique se o destino de entrega no Teams ainda é o chat ou canal correto

### Antes de alterar URLs de webhook ou destinos de entrega {#before-changing-webhook-urls-or-delivery-targets}

- atualize a URL pública de notificação ou a configuração do destino no Teams
- execute `hermes teams-pipeline validate`
- renove ou recrie as inscrições afetadas
- confirme que os novos eventos chegam ao destino esperado

## Triagem de Falhas {#failure-triage}

### Nenhuma tarefa está sendo criada {#no-jobs-are-being-created}

Verifique:
- `msgraph_webhook` está ativado
- a URL pública de notificação aponta para `/msgraph/webhook`
- o client state na inscrição corresponde a `MSGRAPH_WEBHOOK_CLIENT_STATE`
- as inscrições ainda existem remotamente e não estão expiradas

### Tarefas ficam em retentativa ou falham antes da sumarização {#jobs-stay-in-retry-or-fail-before-summarization}

Verifique:
- permissões e disponibilidade da transcrição
- permissões e disponibilidade do artefato de gravação
- disponibilidade do `ffmpeg`, se o fallback de gravação estiver ativado
- saúde do token do Graph

### Resumos são produzidos, mas não entregues ao Teams {#summaries-are-produced-but-not-delivered-to-teams}

Verifique:
- `platforms.teams.enabled: true`
- `delivery_mode`
- `incoming_webhook_url` para o modo webhook
- `chat_id` ou `team_id` mais `channel_id` para o modo Graph
- configuração de autenticação do Teams, se a postagem via Graph estiver sendo usada

### Reproduções duplicadas ou inesperadas {#duplicate-or-unexpected-replays}

Verifique:
- se você reproduziu manualmente uma tarefa com `hermes teams-pipeline run`
- se o registro de destino já existe para aquela reunião
- se você ativou intencionalmente um caminho de reenvio na sua configuração local

## Checklist de Go-Live {#go-live-checklist}

- [ ] As credenciais do Graph estão presentes e corretas
- [ ] `msgraph_webhook` está ativado e acessível pela internet pública
- [ ] `MSGRAPH_WEBHOOK_CLIENT_STATE` está definido e corresponde às inscrições
- [ ] a inscrição de transcrição está criada
- [ ] a inscrição de gravação está criada se o fallback de STT for necessário
- [ ] o `ffmpeg` está instalado se o fallback de gravação estiver ativado
- [ ] o destino de entrega de saída do Teams está configurado e verificado
- [ ] os destinos do Notion e do Linear estão configurados apenas se realmente necessários
- [ ] `hermes teams-pipeline validate` retorna um snapshot OK
- [ ] `hermes teams-pipeline token-health --force-refresh` é bem-sucedido
- [ ] **`maintain-subscriptions` está agendado** (cron do Hermes, timer do systemd ou crontab — veja [Automatizando a renovação de inscrições](#automating-subscription-renewal-required-for-production)). Sem isso, as inscrições do Graph expiram silenciosamente em até 72 horas.
- [ ] um evento de reunião real de ponta a ponta produziu uma tarefa armazenada
- [ ] pelo menos um resumo chegou ao destino de entrega pretendido

## Guia de Decisão do Modo de Entrega {#delivery-mode-decision-guide}

| Modo | Use quando | Compensação |
|------|----------|----------|
| `incoming_webhook` | você só precisa de postagem simples no Teams | configuração mais simples, menos controle |
| `graph` | você precisa de postagem em canal ou chat via Graph | mais controle, mais autenticação e configuração de destino |

## Planilha do Operador {#operator-worksheet}

Preencha isto antes do lançamento:

| Item | Valor |
|------|-------|
| URL pública de notificação | |
| ID do tenant do Graph | |
| ID do client do Graph | |
| Client state do webhook | |
| Inscrição de recurso de transcrição | |
| Inscrição de recurso de gravação | |
| Modo de entrega do Teams | |
| ID do chat ou team/canal do Teams | |
| ID do banco de dados do Notion | |
| ID do team do Linear | |
| Substituição do caminho de armazenamento, se houver | |
| Responsável pelas verificações diárias | |

## Planilha de Revisão de Alterações {#change-review-worksheet}

Use isto antes de alterar a implantação:

| Pergunta | Resposta |
|----------|--------|
| Estamos alterando a URL pública do webhook? | |
| Estamos rotacionando as credenciais do Graph? | |
| Estamos alterando o modo de entrega do Teams? | |
| Estamos movendo para um novo chat ou canal do Teams? | |
| As inscrições precisam ser recriadas ou renovadas? | |
| Precisamos de uma nova execução de verificação de ponta a ponta? | |

## Documentos Relacionados {#related-docs}

- [Configuração de Reuniões do Teams](/user-guide/messaging/teams-meetings)
- [Configuração do bot do Microsoft Teams](/user-guide/messaging/teams)
