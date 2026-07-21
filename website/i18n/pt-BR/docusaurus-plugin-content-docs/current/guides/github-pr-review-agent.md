---
sidebar_position: 10
title: "Tutorial: Agente de Revisão de PRs do GitHub"
description: "Construa um revisor de código com IA automatizado que monitora seus repositórios, revisa pull requests e entrega feedback — sem intervenção manual"
---

# Tutorial: Construa um Agente de Revisão de PRs do GitHub

**O problema:** Sua equipe abre PRs mais rápido do que você consegue revisar. PRs ficam parados por dias esperando alguém olhar. Devs juniores mesclam bugs porque ninguém teve tempo de checar. Você passa suas manhãs correndo atrás de diffs em vez de construir.

**A solução:** Um agente de IA que observa seus repositórios o tempo todo, revisa cada novo PR em busca de bugs, problemas de segurança e qualidade de código, e te envia um resumo — assim você só gasta tempo nos PRs que realmente precisam de julgamento humano.

**O que você vai construir:**

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Cron Timer  ──▶  Hermes Agent  ──▶  GitHub API  ──▶  Review     │
│   (every 2h)       + gh CLI           (PR diffs)       delivery   │
│                    + skill                             (Telegram, │
│                    + memory                            Discord,   │
│                                                        local)     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

Este guia usa **tarefas agendadas (cron jobs)** para consultar por PRs em um cronograma — sem necessidade de servidor ou endpoint público. Funciona atrás de NAT e firewalls.

:::tip Quer revisões em tempo real?
Se você tem um endpoint público disponível, veja [Comentários Automatizados em PRs do GitHub via Webhooks](./webhook-github-pr-review.md) — o GitHub envia eventos ao Hermes instantaneamente quando PRs são abertos ou atualizados.
:::

---

## Pré-requisitos {#prerequisites}

- **Hermes Agent instalado** — veja o [guia de instalação](/getting-started/installation)
- **Gateway em execução** para as tarefas agendadas:
  ```bash
  hermes gateway install   # Install as a service
  # or
  hermes gateway           # Run in foreground
  ```
- **GitHub CLI (`gh`) instalado e autenticado:**
  ```bash
  # Install
  brew install gh        # macOS
  sudo apt install gh    # Ubuntu/Debian

  # Authenticate
  gh auth login
  ```
- **Mensagens configuradas** (opcional) — [Telegram](/user-guide/messaging/telegram) ou [Discord](/user-guide/messaging/discord)

:::tip Sem mensagens? Sem problema
Use `deliver: "local"` para salvar as revisões em `~/.hermes/cron/output/`. Ótimo para testar antes de configurar notificações.
:::

---

## Passo 1: Verifique a Configuração {#step-1-verify-the-setup}

Certifique-se de que o Hermes consegue acessar o GitHub. Inicie um chat:

```bash
hermes
```

Teste com um comando simples:

```
Run: gh pr list --repo NousResearch/hermes-agent --state open --limit 3
```

Você deve ver uma lista de PRs abertos. Se isso funcionar, você está pronto.

---

## Passo 2: Tente uma Revisão Manual {#step-2-try-a-manual-review}

Ainda no chat, peça ao Hermes para revisar um PR real:

```
Review this pull request. Read the diff, check for bugs, security issues,
and code quality. Be specific about line numbers and quote problematic code.

Run: gh pr diff 3888 --repo NousResearch/hermes-agent
```

O Hermes vai:
1. Executar `gh pr diff` para buscar as alterações de código
2. Ler o diff inteiro
3. Produzir uma revisão estruturada com descobertas específicas

Se você estiver satisfeito com a qualidade, é hora de automatizar.

---

## Passo 3: Crie uma Skill de Revisão {#step-3-create-a-review-skill}

Uma skill dá ao Hermes diretrizes de revisão consistentes que persistem entre sessões e execuções agendadas. Sem uma, a qualidade da revisão varia.

```bash
mkdir -p ~/.hermes/skills/code-review
```

Crie `~/.hermes/skills/code-review/SKILL.md`:

```markdown
---
name: code-review
description: Review pull requests for bugs, security issues, and code quality
---

# Code Review Guidelines

When reviewing a pull request:

## What to Check
1. **Bugs** — Logic errors, off-by-one, null/undefined handling
2. **Security** — Injection, auth bypass, secrets in code, SSRF
3. **Performance** — N+1 queries, unbounded loops, memory leaks
4. **Style** — Naming conventions, dead code, missing error handling
5. **Tests** — Are changes tested? Do tests cover edge cases?

## Output Format
For each finding:
- **File:Line** — exact location
- **Severity** — Critical / Warning / Suggestion
- **What's wrong** — one sentence
- **Fix** — how to fix it

## Rules
- Be specific. Quote the problematic code.
- Don't flag style nitpicks unless they affect readability.
- If the PR looks good, say so. Don't invent problems.
- End with: APPROVE / REQUEST_CHANGES / COMMENT
```

Verifique se ela foi carregada — inicie o `hermes` e você deve ver `code-review` na lista de skills durante a inicialização.

---

## Passo 4: Ensine Suas Convenções {#step-4-teach-it-your-conventions}

Isso é o que torna o revisor realmente útil. Inicie uma sessão e ensine ao Hermes os padrões da sua equipe:

```
Remember: In our backend repo, we use Python with FastAPI.
All endpoints must have type annotations and Pydantic models.
We don't allow raw SQL — only SQLAlchemy ORM.
Test files go in tests/ and must use pytest fixtures.
```

```
Remember: In our frontend repo, we use TypeScript with React.
No `any` types allowed. All components must have props interfaces.
We use React Query for data fetching, never useEffect for API calls.
```

Essas memórias persistem para sempre — o revisor vai aplicar suas convenções sem que você precise repeti-las a cada vez.

---

## Passo 5: Crie a Tarefa Agendada Automatizada {#step-5-create-the-automated-cron-job}

Agora vamos juntar tudo. Crie uma tarefa agendada que roda a cada 2 horas:

```bash
hermes cron create "0 */2 * * *" \
  "Check for new open PRs and review them.

Repos to monitor:
- myorg/backend-api
- myorg/frontend-app

Steps:
1. Run: gh pr list --repo REPO --state open --limit 5 --json number,title,author,createdAt
2. For each PR created or updated in the last 4 hours:
   - Run: gh pr diff NUMBER --repo REPO
   - Review the diff using the code-review guidelines
3. Format output as:

## PR Reviews — today

### [repo] #[number]: [title]
**Author:** [name] | **Verdict:** APPROVE/REQUEST_CHANGES/COMMENT
[findings]

If no new PRs found, say: No new PRs to review." \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

Verifique se está agendada:

```bash
hermes cron list
```

### Outros cronogramas úteis {#other-useful-schedules}

| Cronograma | Quando |
|----------|------|
| `0 */2 * * *` | A cada 2 horas |
| `0 9,13,17 * * 1-5` | Três vezes ao dia, apenas em dias úteis |
| `0 9 * * 1` | Resumo semanal na segunda-feira de manhã |
| `30m` | A cada 30 minutos (repositórios de alto tráfego) |

---

## Passo 6: Execute Sob Demanda {#step-6-run-it-on-demand}

Não quer esperar pelo cronograma? Acione manualmente:

```bash
hermes cron run pr-review
```

Ou de dentro de uma sessão de chat:

```
/cron run pr-review
```

---

## Continuando {#going-further}

### Poste Revisões Diretamente no GitHub {#post-reviews-directly-to-github}

Em vez de entregar no Telegram, faça o agente comentar diretamente no próprio PR:

Adicione isto ao seu prompt de tarefa agendada:

```
After reviewing, post your review:
- For issues: gh pr review NUMBER --repo REPO --comment --body "YOUR_REVIEW"
- For critical issues: gh pr review NUMBER --repo REPO --request-changes --body "YOUR_REVIEW"
- For clean PRs: gh pr review NUMBER --repo REPO --approve --body "Looks good"
```

:::caution
Certifique-se de que o `gh` tenha um token com escopo `repo`. As revisões são postadas como quem estiver autenticado no `gh`.
:::

### Painel Semanal de PRs {#weekly-pr-dashboard}

Crie uma visão geral de segunda-feira de manhã de todos os seus repositórios:

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly PR dashboard:
- myorg/backend-api
- myorg/frontend-app
- myorg/infra

For each repo show:
1. Open PR count and oldest PR age
2. PRs merged this week
3. Stale PRs (older than 5 days)
4. PRs with no reviewer assigned

Format as a clean summary." \
  --name "weekly-dashboard" \
  --deliver telegram
```

### Monitoramento Multi-Repositório {#multi-repo-monitoring}

Escale adicionando mais repositórios ao prompt. O agente os processa sequencialmente — sem necessidade de configuração extra.

---

## Solução de Problemas {#troubleshooting}

### "gh: command not found" {#gh-command-not-found}
O gateway roda em um ambiente mínimo. Certifique-se de que o `gh` esteja no PATH do sistema e reinicie o gateway.

### As revisões estão muito genéricas {#reviews-are-too-generic}
1. Adicione a skill `code-review` (Passo 3)
2. Ensine ao Hermes suas convenções via memória (Passo 4)
3. Quanto mais contexto ele tiver sobre sua stack, melhores as revisões

### A tarefa agendada não roda {#cron-job-doesnt-run}
```bash
hermes gateway status    # Is the gateway running?
hermes cron list         # Is the job enabled?
```

### Limites de taxa {#rate-limits}
O GitHub permite 5.000 requisições de API por hora para usuários autenticados. Cada revisão de PR usa de ~3 a 5 requisições (listagem + diff + comentários opcionais). Mesmo revisando 100 PRs/dia, você fica bem dentro dos limites.

---

## O Que Vem a Seguir? {#whats-next}

- **[Revisões de PR Baseadas em Webhook](./webhook-github-pr-review.md)** — receba revisões instantâneas quando PRs são abertos (requer um endpoint público)
- **[Bot de Resumo Diário](/guides/daily-briefing-bot)** — combine revisões de PR com seu resumo de notícias matinal
- **[Construa um Plugin](/developer-guide/plugins)** — encapsule a lógica de revisão em um plugin compartilhável
- **[Perfis](/user-guide/profiles)** — rode um perfil de revisor dedicado com sua própria memória e configuração
- **[Provedores de Fallback](/user-guide/features/fallback-providers)** — garanta que as revisões continuem rodando mesmo quando um provedor estiver fora do ar
