---
sidebar_position: 11
title: "Automatize Qualquer Coisa com Cron"
description: "Padrões reais de automação usando o cron do Hermes — monitoramento, relatórios, pipelines e fluxos de trabalho com múltiplas skills"
---

# Automatize Qualquer Coisa com Cron {#automate-anything-with-cron}

O [tutorial do bot de resumo diário](/guides/daily-briefing-bot) cobre o básico. Este guia vai além — cinco padrões reais de automação que você pode adaptar para seus próprios fluxos de trabalho.

Para a referência completa do recurso, veja [Tarefas Agendadas (Cron)](/user-guide/features/cron).

:::info Conceito Chave
Tarefas cron são executadas em sessões de agente novas, sem memória do seu chat atual. Os prompts precisam ser **completamente autocontidos** — inclua tudo o que o agente precisa saber.
:::

:::tip Não precisa do LLM? Você tem duas opções sem custo de tokens.
- **Watchdog recorrente** onde o script já produz a mensagem exata (alertas de memória, alertas de disco, heartbeats): use [tarefas cron somente com script](/guides/cron-script-only). Mesmo agendador, sem LLM. Você pode pedir ao Hermes para configurar uma para você no chat — a ferramenta `cronjob` sabe quando usar `no_agent=True` e escreve o script para você.
- **Disparo único a partir de um script que já está em execução** (etapa de CI, hook pós-commit, script de deploy, monitor agendado externamente): use o [`hermes send`](/guides/pipe-script-output) para enviar a saída padrão ou um arquivo diretamente ao Telegram / Discord / Slack / etc., sem configurar uma entrada de cron.
:::

---

## Padrão 1: Monitor de Alterações de Site {#pattern-1-website-change-monitor}

Observe uma URL por alterações e seja notificado apenas quando algo for diferente.

O parâmetro `script` é a arma secreta aqui. Um script Python é executado antes de cada execução, e sua saída padrão se torna contexto para o agente. O script lida com o trabalho mecânico (buscar, comparar); o agente lida com o raciocínio (essa alteração é interessante?).

Crie o script de monitoramento:

```bash
mkdir -p ~/.hermes/scripts
```

```python title="~/.hermes/scripts/watch-site.py"
import hashlib, json, os, urllib.request

URL = "https://example.com/pricing"
STATE_FILE = os.path.expanduser("~/.hermes/scripts/.watch-site-state.json")

# Fetch current content
req = urllib.request.Request(URL, headers={"User-Agent": "Hermes-Monitor/1.0"})
content = urllib.request.urlopen(req, timeout=30).read().decode()
current_hash = hashlib.sha256(content.encode()).hexdigest()

# Load previous state
prev_hash = None
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        prev_hash = json.load(f).get("hash")

# Save current state
with open(STATE_FILE, "w") as f:
    json.dump({"hash": current_hash, "url": URL}, f)

# Output for the agent
if prev_hash and prev_hash != current_hash:
    print(f"CHANGE DETECTED on {URL}")
    print(f"Previous hash: {prev_hash}")
    print(f"Current hash: {current_hash}")
    print(f"\nCurrent content (first 2000 chars):\n{content[:2000]}")
else:
    print("NO_CHANGE")
```

Configure a tarefa cron:

```bash
/cron add "every 1h" "If the script output says CHANGE DETECTED, summarize what changed on the page and why it might matter. If it says NO_CHANGE, respond with just [SILENT]." --script ~/.hermes/scripts/watch-site.py --name "Pricing monitor" --deliver telegram
```

:::tip O Truque do [SILENT]
Para tarefas cron de monitoramento, instrua o agente a responder apenas com `[SILENT]` quando nada mudar. A entrega do cron trata `[SILENT]` como o marcador de silêncio, então você só é notificado quando algo realmente acontece — sem spam nas horas tranquilas.
:::

---

## Padrão 2: Relatório Semanal {#pattern-2-weekly-report}

Compile informações de várias fontes em um resumo formatado. Isso é executado uma vez por semana e entrega no seu canal principal.

```bash
/cron add "0 9 * * 1" "Generate a weekly report covering:

1. Search the web for the top 5 AI news stories from the past week
2. Search GitHub for trending repositories in the 'machine-learning' topic
3. Check Hacker News for the most discussed AI/ML posts

Format as a clean summary with sections for each source. Include links.
Keep it under 500 words — highlight only what matters." --name "Weekly AI digest" --deliver telegram
```

Da CLI:

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly report covering the top AI news, trending ML GitHub repos, and most-discussed HN posts. Format with sections, include links, keep under 500 words." \
  --name "Weekly AI digest" \
  --deliver telegram
```

O `0 9 * * 1` é uma expressão cron padrão: 9h00 toda segunda-feira.

---

## Padrão 3: Observador de Repositório do GitHub {#pattern-3-github-repository-watcher}

Monitore um repositório em busca de novas issues, PRs ou releases.

```bash
/cron add "every 6h" "Check the GitHub repository NousResearch/hermes-agent for:
- New issues opened in the last 6 hours
- New PRs opened or merged in the last 6 hours
- Any new releases

Use the terminal to run gh commands:
  gh issue list --repo NousResearch/hermes-agent --state open --json number,title,author,createdAt --limit 10
  gh pr list --repo NousResearch/hermes-agent --state all --json number,title,author,createdAt,mergedAt --limit 10

Filter to only items from the last 6 hours. If nothing new, respond with [SILENT].
Otherwise, provide a concise summary of the activity." --name "Repo watcher" --deliver discord
```

:::warning Prompts Autocontidos
Observe como o prompt inclui os comandos `gh` exatos. O agente do cron não tem memória de execuções anteriores ou de suas preferências — detalhe tudo.
:::

---

## Padrão 4: Pipeline de Coleta de Dados {#pattern-4-data-collection-pipeline}

Colete dados em intervalos regulares, salve em arquivos e detecte tendências ao longo do tempo. Esse padrão combina um script (para coleta) com o agente (para análise).

```python title="~/.hermes/scripts/collect-prices.py"
import json, os, urllib.request
from datetime import datetime

DATA_DIR = os.path.expanduser("~/.hermes/data/prices")
os.makedirs(DATA_DIR, exist_ok=True)

# Fetch current data (example: crypto prices)
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
data = json.loads(urllib.request.urlopen(url, timeout=30).read())

# Append to history file
entry = {"timestamp": datetime.now().isoformat(), "prices": data}
history_file = os.path.join(DATA_DIR, "history.jsonl")
with open(history_file, "a") as f:
    f.write(json.dumps(entry) + "\n")

# Load recent history for analysis
lines = open(history_file).readlines()
recent = [json.loads(l) for l in lines[-24:]]  # Last 24 data points

# Output for the agent
print(f"Current: BTC=${data['bitcoin']['usd']}, ETH=${data['ethereum']['usd']}")
print(f"Data points collected: {len(lines)} total, showing last {len(recent)}")
print(f"\nRecent history:")
for r in recent[-6:]:
    print(f"  {r['timestamp']}: BTC=${r['prices']['bitcoin']['usd']}, ETH=${r['prices']['ethereum']['usd']}")
```

```bash
/cron add "every 1h" "Analyze the price data from the script output. Report:
1. Current prices
2. Trend direction over the last 6 data points (up/down/flat)
3. Any notable movements (>5% change)

If prices are flat and nothing notable, respond with [SILENT].
If there's a significant move, explain what happened." \
  --script ~/.hermes/scripts/collect-prices.py \
  --name "Price tracker" \
  --deliver telegram
```

O script faz a coleta mecânica; o agente adiciona a camada de raciocínio.

---

## Padrão 5: Fluxo de Trabalho com Múltiplas Skills {#pattern-5-multi-skill-workflow}

Encadeie skills para tarefas agendadas complexas. As skills são carregadas em ordem antes da execução do prompt.

```bash
# Use the arxiv skill to find papers, then the obsidian skill to save notes
/cron add "0 8 * * *" "Search arXiv for the 3 most interesting papers on 'language model reasoning' from the past day. For each paper, create an Obsidian note with the title, authors, abstract summary, and key contribution." \
  --skill arxiv \
  --skill obsidian \
  --name "Paper digest"
```

Diretamente pela ferramenta:

```python
cronjob(
    action="create",
    skills=["arxiv", "obsidian"],
    prompt="Search arXiv for papers on 'language model reasoning' from the past day. Save the top 3 as Obsidian notes.",
    schedule="0 8 * * *",
    name="Paper digest",
    deliver="local"
)
```

As skills são carregadas em ordem — `arxiv` primeiro (ensina ao agente como buscar artigos), depois `obsidian` (ensina como escrever notas). O prompt as conecta.

---

## Gerenciando Suas Tarefas {#managing-your-jobs}

```bash
# List all active jobs
/cron list

# Trigger a job immediately (for testing)
/cron run <job_id>

# Pause a job without deleting it
/cron pause <job_id>

# Edit a running job's schedule or prompt
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Updated task description"

# Add or remove skills from an existing job
/cron edit <job_id> --skill arxiv --skill obsidian
/cron edit <job_id> --clear-skills

# Remove a job permanently
/cron remove <job_id>
```

---

## Destinos de Entrega {#delivery-targets}

A flag `--deliver` controla para onde os resultados vão:

| Destino | Exemplo | Caso de uso |
|--------|---------|----------|
| `origin` | `--deliver origin` | Mesmo chat que criou a tarefa (padrão) |
| `local` | `--deliver local` | Salva apenas em um arquivo local |
| `telegram` | `--deliver telegram` | Seu canal principal do Telegram |
| `discord` | `--deliver discord` | Seu canal principal do Discord |
| `slack` | `--deliver slack` | Seu canal principal do Slack |
| Chat específico | `--deliver telegram:-1001234567890` | Um grupo específico do Telegram |
| Em thread | `--deliver telegram:-1001234567890:17585` | Uma thread de tópico específica do Telegram |

---

## Dicas {#tips}

**Torne os prompts autocontidos.** O agente em uma tarefa cron não tem memória das suas conversas. Inclua URLs, nomes de repositórios, preferências de formato e instruções de entrega diretamente no prompt.

**Use `[SILENT]` deliberadamente.** Para tarefas de monitoramento, inclua instruções como "se nada mudou, responda apenas com `[SILENT]`." Não peça ao agente para explicar o token em casos silenciosos — o cron trata `[SILENT]` como o marcador de supressão de entrega.

**Use scripts para coleta de dados.** O parâmetro `script` permite que um script Python lide com as partes tediosas (requisições HTTP, I/O de arquivos, rastreamento de estado). O agente vê apenas a saída padrão do script e aplica raciocínio sobre ela. Isso é mais barato e mais confiável do que fazer o agente buscar os dados sozinho.

**Teste com `/cron run`.** Antes de esperar o agendamento disparar, use `/cron run <job_id>` para executar imediatamente e verificar se a saída parece correta.

**Expressões de agendamento.** Formatos suportados: atrasos relativos (`30m`), intervalos (`every 2h`), expressões cron padrão (`0 9 * * *`) e timestamps ISO (`2025-06-15T09:00:00`). Linguagem natural como `daily at 9am` não é suportada — use `0 9 * * *` em vez disso.

---

*Para a referência completa do cron — todos os parâmetros, casos extremos e detalhes internos — veja [Tarefas Agendadas (Cron)](/user-guide/features/cron).*
