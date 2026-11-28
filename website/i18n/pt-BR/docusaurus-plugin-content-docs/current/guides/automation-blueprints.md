---
sidebar_position: 15
title: "Blueprints de Automação"
description: "Blueprints de automação prontos para usar — tarefas agendadas, gatilhos de eventos do GitHub, webhooks de API e fluxos de trabalho com múltiplas skills"
---

# Blueprints de Automação {#automation-blueprints}

Blueprints de copiar e colar para padrões comuns de automação. Cada blueprint usa o [agendador cron](/user-guide/features/cron) integrado do Hermes para gatilhos baseados em tempo e a [plataforma de webhook](/user-guide/messaging/webhooks) para gatilhos orientados a eventos.

Todo blueprint funciona com **qualquer modelo** — não é preso a um único provedor.

Para blueprints parametrizados com formulários em vez de sintaxe cron, veja o [Catálogo de Blueprints de Automação](/reference/automation-blueprints-catalog).

:::tip Três Tipos de Gatilho
| Gatilho | Como | Ferramenta |
|---------|-----|------|
| **Agendamento** | Executa em uma cadência (a cada hora, todas as noites, semanalmente) | ferramenta `cronjob` ou comando de barra `/cron` |
| **Evento do GitHub** | Disparado por abertura de PR, pushes, issues, resultados de CI | Plataforma de webhook (`hermes webhook subscribe`) |
| **Chamada de API** | Serviço externo faz POST de JSON para seu endpoint | Plataforma de webhook (rotas em config.yaml ou `hermes webhook subscribe`) |

Todos os três suportam entrega para Telegram, Discord, Slack, SMS, e-mail, comentários no GitHub ou arquivos locais.
:::

---

## Fluxo de Trabalho de Desenvolvimento {#development-workflow}

### Triagem Noturna do Backlog {#nightly-backlog-triage}

Rotule, priorize e resuma novas issues todas as noites. Entrega um resumo para o canal da sua equipe.

**Gatilho:** Agendamento (noturno)

```bash
hermes cron create "0 2 * * *" \
  "You are a project manager triaging the NousResearch/hermes-agent GitHub repo.

1. Run: gh issue list --repo NousResearch/hermes-agent --state open --json number,title,labels,author,createdAt --limit 30
2. Identify issues opened in the last 24 hours
3. For each new issue:
   - Suggest a priority label (P0-critical, P1-high, P2-medium, P3-low)
   - Suggest a category label (bug, feature, docs, security)
   - Write a one-line triage note
4. Summarize: total open issues, new today, breakdown by priority

Format as a clean digest. If no new issues, respond with [SILENT]." \
  --name "Nightly backlog triage" \
  --deliver telegram
```

### Revisão Automática de Código de PR {#automatic-pr-code-review}

Revise cada pull request automaticamente quando ele for aberto. Publica um comentário de revisão diretamente no PR.

**Gatilho:** Webhook do GitHub

**Opção A — Inscrição dinâmica (CLI):**

```bash
hermes webhook subscribe github-pr-review \
  --events "pull_request" \
  --prompt "Review this pull request:
Repository: {repository.full_name}
PR #{pull_request.number}: {pull_request.title}
Author: {pull_request.user.login}
Action: {action}
Diff URL: {pull_request.diff_url}

Fetch the diff with: curl -sL {pull_request.diff_url}

Review for:
- Security issues (injection, auth bypass, secrets in code)
- Performance concerns (N+1 queries, unbounded loops, memory leaks)
- Code quality (naming, duplication, error handling)
- Missing tests for new behavior

Post a concise review. If the PR is a trivial docs/typo change, say so briefly." \
  --skill github-code-review \
  --deliver github_comment
```

**Opção B — Rota estática (config.yaml):**

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      secret: "your-global-secret"
      routes:
        github-pr-review:
          events: ["pull_request"]
          secret: "github-webhook-secret"
          prompt: |
            Review PR #{pull_request.number}: {pull_request.title}
            Repository: {repository.full_name}
            Author: {pull_request.user.login}
            Diff URL: {pull_request.diff_url}
            Review for security, performance, and code quality.
          skills: ["github-code-review"]
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{pull_request.number}"
```

Depois, no GitHub: **Settings → Webhooks → Add webhook** → Payload URL: `http://your-server:8644/webhooks/github-pr-review`, Content type: `application/json`, Secret: `github-webhook-secret`, Events: **Pull requests**.

### Detecção de Divergência na Documentação {#docs-drift-detection}

Verificação semanal de PRs mesclados para encontrar alterações de API que precisam de atualizações na documentação.

**Gatilho:** Agendamento (semanal)

```bash
hermes cron create "0 9 * * 1" \
  "Scan the NousResearch/hermes-agent repo for documentation drift.

1. Run: gh pr list --repo NousResearch/hermes-agent --state merged --json number,title,files,mergedAt --limit 30
2. Filter to PRs merged in the last 7 days
3. For each merged PR, check if it modified:
   - Tool schemas (tools/*.py) — may need docs/reference/tools-reference.md update
   - CLI commands (hermes_cli/commands.py, hermes_cli/main.py) — may need docs/reference/cli-commands.md update
   - Config options (hermes_cli/config.py) — may need docs/user-guide/configuration.md update
   - Environment variables — may need docs/reference/environment-variables.md update
4. Cross-reference: for each code change, check if the corresponding docs page was also updated in the same PR

Report any gaps where code changed but docs didn't. If everything is in sync, respond with [SILENT]." \
  --name "Docs drift detection" \
  --deliver telegram
```

### Auditoria de Segurança de Dependências {#dependency-security-audit}

Verificação diária por vulnerabilidades conhecidas nas dependências do projeto.

**Gatilho:** Agendamento (diário)

```bash
hermes cron create "0 6 * * *" \
  "Run a dependency security audit on the hermes-agent project.

1. cd ~/.hermes/hermes-agent && source .venv/bin/activate
2. Run: pip audit --format json 2>/dev/null || pip audit 2>&1
3. Run: npm audit --json 2>/dev/null (in website/ directory if it exists)
4. Check for any CVEs with CVSS score >= 7.0

If vulnerabilities found:
- List each one with package name, version, CVE ID, severity
- Check if an upgrade is available
- Note if it's a direct dependency or transitive

If no vulnerabilities, respond with [SILENT]." \
  --name "Dependency audit" \
  --deliver telegram
```

---

## DevOps e Monitoramento {#devops--monitoring}

### Verificação de Deploy {#deploy-verification}

Dispare testes de fumaça após cada deploy. Seu pipeline de CI/CD faz POST para o webhook quando um deploy é concluído.

**Gatilho:** Chamada de API (webhook)

```bash
hermes webhook subscribe deploy-verify \
  --events "deployment" \
  --prompt "A deployment just completed:
Service: {service}
Environment: {environment}
Version: {version}
Deployed by: {deployer}

Run these verification steps:
1. Check if the service is responding: curl -s -o /dev/null -w '%{http_code}' {health_url}
2. Search recent logs for errors: check the deployment payload for any error indicators
3. Verify the version matches: curl -s {health_url}/version

Report: deployment status (healthy/degraded/failed), response time, any errors found.
If healthy, keep it brief. If degraded or failed, provide detailed diagnostics." \
  --deliver telegram
```

Seu pipeline de CI/CD o dispara:

```bash
curl -X POST http://your-server:8644/webhooks/deploy-verify \
  -H "Content-Type: application/json" \
  -H "X-Hub-Signature-256: sha256=$(echo -n '{"service":"api","environment":"prod","version":"2.1.0","deployer":"ci","health_url":"https://api.example.com/health"}' | openssl dgst -sha256 -hmac 'your-secret' | cut -d' ' -f2)" \
  -d '{"service":"api","environment":"prod","version":"2.1.0","deployer":"ci","health_url":"https://api.example.com/health"}'
```

### Triagem de Alertas {#alert-triage}

Correlacione alertas de monitoramento com alterações recentes para elaborar uma resposta. Funciona com Datadog, PagerDuty, Grafana ou qualquer sistema de alertas que possa fazer POST de JSON.

**Gatilho:** Chamada de API (webhook)

```bash
hermes webhook subscribe alert-triage \
  --prompt "Monitoring alert received:
Alert: {alert.name}
Severity: {alert.severity}
Service: {alert.service}
Message: {alert.message}
Timestamp: {alert.timestamp}

Investigate:
1. Search the web for known issues with this error pattern
2. Check if this correlates with any recent deployments or config changes
3. Draft a triage summary with:
   - Likely root cause
   - Suggested first response steps
   - Escalation recommendation (P1-P4)

Be concise. This goes to the on-call channel." \
  --deliver slack
```

### Monitor de Uptime {#uptime-monitor}

Verifique endpoints a cada 30 minutos. Notifique apenas quando algo estiver fora do ar.

**Gatilho:** Agendamento (a cada 30 min)

```python title="~/.hermes/scripts/check-uptime.py"
import urllib.request, json, time

ENDPOINTS = [
    {"name": "API", "url": "https://api.example.com/health"},
    {"name": "Web", "url": "https://www.example.com"},
    {"name": "Docs", "url": "https://docs.example.com"},
]

results = []
for ep in ENDPOINTS:
    try:
        start = time.time()
        req = urllib.request.Request(ep["url"], headers={"User-Agent": "Hermes-Monitor/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        elapsed = round((time.time() - start) * 1000)
        results.append({"name": ep["name"], "status": resp.getcode(), "ms": elapsed})
    except Exception as e:
        results.append({"name": ep["name"], "status": "DOWN", "error": str(e)})

down = [r for r in results if r.get("status") == "DOWN" or (isinstance(r.get("status"), int) and r["status"] >= 500)]
if down:
    print("OUTAGE DETECTED")
    for r in down:
        print(f"  {r['name']}: {r.get('error', f'HTTP {r[\"status\"]}')} ")
    print(f"\nAll results: {json.dumps(results, indent=2)}")
else:
    print("NO_ISSUES")
```

```bash
hermes cron create "every 30m" \
  "If the script reports OUTAGE DETECTED, summarize which services are down and suggest likely causes. If NO_ISSUES, respond with [SILENT]." \
  --script ~/.hermes/scripts/check-uptime.py \
  --name "Uptime monitor" \
  --deliver telegram
```

---

## Pesquisa e Inteligência {#research--intelligence}

### Batedor de Repositórios Concorrentes {#competitive-repository-scout}

Monitore repositórios concorrentes por PRs, recursos e decisões arquiteturais interessantes.

**Gatilho:** Agendamento (diário)

```bash
hermes cron create "0 8 * * *" \
  "Scout these AI agent repositories for notable activity in the last 24 hours:

Repos to check:
- anthropics/claude-code
- openai/codex
- All-Hands-AI/OpenHands
- Aider-AI/aider

For each repo:
1. gh pr list --repo <repo> --state all --json number,title,author,createdAt,mergedAt --limit 15
2. gh issue list --repo <repo> --state open --json number,title,labels,createdAt --limit 10

Focus on:
- New features being developed
- Architectural changes
- Integration patterns we could learn from
- Security fixes that might affect us too

Skip routine dependency bumps and CI fixes. If nothing notable, respond with [SILENT].
If there are findings, organize by repo with brief analysis of each item." \
  --skill competitive-pr-scout \
  --name "Competitor scout" \
  --deliver telegram
```

### Resumo de Notícias de IA {#ai-news-digest}

Resumo semanal dos desenvolvimentos de IA/ML.

**Gatilho:** Agendamento (semanal)

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly AI news digest covering the past 7 days:

1. Search the web for major AI announcements, model releases, and research breakthroughs
2. Search for trending ML repositories on GitHub
3. Check arXiv for highly-cited papers on language models and agents

Structure:
## Headlines (3-5 major stories)
## Notable Papers (2-3 papers with one-sentence summaries)
## Open Source (interesting new repos or major releases)
## Industry Moves (funding, acquisitions, launches)

Keep each item to 1-2 sentences. Include links. Total under 600 words." \
  --name "Weekly AI digest" \
  --deliver telegram
```

### Resumo de Artigos com Notas {#paper-digest-with-notes}

Verificação diária no arXiv que salva resumos no seu sistema de notas.

**Gatilho:** Agendamento (diário)

```bash
hermes cron create "0 8 * * *" \
  "Search arXiv for the 3 most interesting papers on 'language model reasoning' OR 'tool-use agents' from the past day. For each paper, create an Obsidian note with the title, authors, abstract summary, key contribution, and potential relevance to Hermes Agent development." \
  --skill arxiv --skill obsidian \
  --name "Paper digest" \
  --deliver local
```

---

## Automações de Eventos do GitHub {#github-event-automations}

### Rotulagem Automática de Issues {#issue-auto-labeling}

Rotule e responda automaticamente a novas issues.

**Gatilho:** Webhook do GitHub

```bash
hermes webhook subscribe github-issues \
  --events "issues" \
  --prompt "New GitHub issue received:
Repository: {repository.full_name}
Issue #{issue.number}: {issue.title}
Author: {issue.user.login}
Action: {action}
Body: {issue.body}
Labels: {issue.labels}

If this is a new issue (action=opened):
1. Read the issue title and body carefully
2. Suggest appropriate labels (bug, feature, docs, security, question)
3. If it's a bug report, check if you can identify the affected component from the description
4. Post a helpful initial response acknowledging the issue

If this is a label or assignment change, respond with [SILENT]." \
  --deliver github_comment
```

### Análise de Falha de CI {#ci-failure-analysis}

Analise falhas de CI e publique diagnósticos no PR.

**Gatilho:** Webhook do GitHub

```yaml
# config.yaml route
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        ci-failure:
          events: ["check_run"]
          secret: "ci-secret"
          prompt: |
            CI check failed:
            Repository: {repository.full_name}
            Check: {check_run.name}
            Status: {check_run.conclusion}
            PR: #{check_run.pull_requests.0.number}
            Details URL: {check_run.details_url}

            If conclusion is "failure":
            1. Fetch the log from the details URL if accessible
            2. Identify the likely cause of failure
            3. Suggest a fix
            If conclusion is "success", respond with [SILENT].
          deliver: "github_comment"
          deliver_extra:
            repo: "{repository.full_name}"
            pr_number: "{check_run.pull_requests.0.number}"
```

### Portar Alterações Automaticamente Entre Repositórios {#auto-port-changes-across-repos}

Quando um PR é mesclado em um repositório, porte automaticamente a alteração equivalente para outro.

**Gatilho:** Webhook do GitHub

```bash
hermes webhook subscribe auto-port \
  --events "pull_request" \
  --prompt "PR merged in the source repository:
Repository: {repository.full_name}
PR #{pull_request.number}: {pull_request.title}
Author: {pull_request.user.login}
Action: {action}
Merge commit: {pull_request.merge_commit_sha}

If action is 'closed' and pull_request.merged is true:
1. Fetch the diff: curl -sL {pull_request.diff_url}
2. Analyze what changed
3. Determine if this change needs to be ported to the Go SDK equivalent
4. If yes, create a branch, apply the equivalent changes, and open a PR on the target repo
5. Reference the original PR in the new PR description

If action is not 'closed' or not merged, respond with [SILENT]." \
  --skill github-pr-workflow \
  --deliver log
```

---

## Operações de Negócio {#business-operations}

### Monitoramento de Pagamentos do Stripe {#stripe-payment-monitoring}

Acompanhe eventos de pagamento e receba resumos de falhas.

**Gatilho:** Chamada de API (webhook)

```bash
hermes webhook subscribe stripe-payments \
  --events "payment_intent.succeeded,payment_intent.payment_failed,charge.dispute.created" \
  --prompt "Stripe event received:
Event type: {type}
Amount: {data.object.amount} cents ({data.object.currency})
Customer: {data.object.customer}
Status: {data.object.status}

For payment_intent.payment_failed:
- Identify the failure reason from {data.object.last_payment_error}
- Suggest whether this is a transient issue (retry) or permanent (contact customer)

For charge.dispute.created:
- Flag as urgent
- Summarize the dispute details

For payment_intent.succeeded:
- Brief confirmation only

Keep responses concise for the ops channel." \
  --deliver slack
```

### Resumo Diário de Receita {#daily-revenue-summary}

Compile métricas-chave do negócio toda manhã.

**Gatilho:** Agendamento (diário)

```bash
hermes cron create "0 8 * * *" \
  "Generate a morning business metrics summary.

Search the web for:
1. Current Bitcoin and Ethereum prices
2. S&P 500 status (pre-market or previous close)
3. Any major tech/AI industry news from the last 12 hours

Format as a brief morning briefing, 3-4 bullet points max.
Deliver as a clean, scannable message." \
  --name "Morning briefing" \
  --deliver telegram
```

---

## Fluxos de Trabalho com Múltiplas Skills {#multi-skill-workflows}

### Pipeline de Auditoria de Segurança {#security-audit-pipeline}

Combine várias skills para uma revisão de segurança semanal abrangente.

**Gatilho:** Agendamento (semanal)

```bash
hermes cron create "0 3 * * 0" \
  "Run a comprehensive security audit of the hermes-agent codebase.

1. Check for dependency vulnerabilities (pip audit, npm audit)
2. Search the codebase for common security anti-patterns:
   - Hardcoded secrets or API keys
   - SQL injection vectors (string formatting in queries)
   - Path traversal risks (user input in file paths without validation)
   - Unsafe deserialization (pickle.loads, yaml.load without SafeLoader)
3. Review recent commits (last 7 days) for security-relevant changes
4. Check if any new environment variables were added without being documented

Write a security report with findings categorized by severity (Critical, High, Medium, Low).
If nothing found, report a clean bill of health." \
  --skill codebase-security-audit \
  --name "Weekly security audit" \
  --deliver telegram
```

### Pipeline de Conteúdo {#content-pipeline}

Pesquise, redija e prepare conteúdo em um agendamento.

**Gatilho:** Agendamento (semanal)

```bash
hermes cron create "0 10 * * 3" \
  "Research and draft a technical blog post outline about a trending topic in AI agents.

1. Search the web for the most discussed AI agent topics this week
2. Pick the most interesting one that's relevant to open-source AI agents
3. Create an outline with:
   - Hook/intro angle
   - 3-4 key sections
   - Technical depth appropriate for developers
   - Conclusion with actionable takeaway
4. Save the outline to ~/drafts/blog-$(date +%Y%m%d).md

Keep the outline to ~300 words. This is a starting point, not a finished post." \
  --name "Blog outline" \
  --deliver local
```

---

## Referência Rápida {#quick-reference}

### Sintaxe de Agendamento do Cron {#cron-schedule-syntax}

| Expressão | Significado |
|-----------|---------|
| `every 30m` | A cada 30 minutos |
| `every 2h` | A cada 2 horas |
| `0 2 * * *` | Diariamente às 2h00 |
| `0 9 * * 1` | Toda segunda-feira às 9h00 |
| `0 9 * * 1-5` | Dias de semana às 9h00 |
| `0 3 * * 0` | Todo domingo às 3h00 |
| `0 */6 * * *` | A cada 6 horas |

### Destinos de Entrega {#delivery-targets}

| Destino | Flag | Notas |
|--------|------|-------|
| Mesmo chat | `--deliver origin` | Padrão — entrega onde a tarefa foi criada |
| Arquivo local | `--deliver local` | Salva a saída, sem notificação |
| Telegram | `--deliver telegram` | Canal principal, ou `telegram:CHAT_ID` para um específico |
| Discord | `--deliver discord` | Canal principal, ou `discord:CHANNEL_ID` |
| Slack | `--deliver slack` | Canal principal |
| SMS | `--deliver sms:+15551234567` | Direto para um número de telefone |
| Thread específica | `--deliver telegram:-100123:456` | Tópico de fórum do Telegram |

### Variáveis de Template de Webhook {#webhook-template-variables}

| Variável | Descrição |
|----------|-------------|
| `{pull_request.title}` | Título do PR |
| `{issue.number}` | Número da issue |
| `{repository.full_name}` | `owner/repo` |
| `{action}` | Ação do evento (opened, closed, etc.) |
| `{__raw__}` | Payload JSON completo (truncado em 4000 caracteres) |
| `{sender.login}` | Usuário do GitHub que disparou o evento |

### O Padrão [SILENT] {#the-silent-pattern}

Quando a resposta de uma tarefa cron contém `[SILENT]`, a entrega é suprimida. Use isso para evitar spam de notificações em execuções silenciosas:

```
If nothing noteworthy happened, respond with [SILENT].
```

Isso significa que você só é notificado quando o agente tem algo a relatar.
