---
sidebar_position: 13
title: "Delegação e Trabalho Paralelo"
description: "Quando e como usar a delegação de subagentes — padrões para pesquisa paralela, revisão de código e trabalho com múltiplos arquivos"
---

# Delegação e Trabalho Paralelo {#delegation--parallel-work}

O Hermes pode gerar agentes filhos isolados para trabalhar em tarefas em paralelo. Cada subagente recebe sua própria conversa, sessão de terminal e conjunto de ferramentas. Apenas o resumo final retorna — as chamadas de ferramentas intermediárias nunca entram na sua janela de contexto.

Para a referência completa do recurso, veja [Delegação de Subagentes](/user-guide/features/delegation).

---

## Quando Delegar {#when-to-delegate}

**Bons candidatos para delegação:**
- Subtarefas que exigem muito raciocínio (depuração, revisão de código, síntese de pesquisa)
- Tarefas que inundariam seu contexto com dados intermediários
- Fluxos de trabalho paralelos independentes (pesquisar A e B simultaneamente)
- Tarefas de contexto novo em que você quer que o agente aborde sem viés

**Use outra coisa:**
- Uma única chamada de ferramenta → basta usar a ferramenta diretamente
- Trabalho mecânico de múltiplas etapas com lógica entre elas → `execute_code`
- Tarefas que exigem interação do usuário → subagentes não podem usar `clarify`
- Edições rápidas de arquivos → faça-as diretamente
- Trabalho duradouro de longa duração que precisa sobreviver ao fechamento da sessão ou a um reinício do processo → `cronjob` ou `terminal(background=True, notify_on_complete=True)`. A delegação de nível superior é assíncrona, mas ainda é local ao processo.

---

## Padrão: Pesquisa Paralela {#pattern-parallel-research}

Pesquise três tópicos simultaneamente e obtenha resumos estruturados de volta:

```
Research these three topics in parallel:
1. Current state of WebAssembly outside the browser
2. RISC-V server chip adoption in 2025
3. Practical quantum computing applications

Focus on recent developments and key players.
```

Por baixo dos panos, o Hermes usa:

```python
delegate_task(tasks=[
    {
        "goal": "Research WebAssembly outside the browser in 2025",
        "context": "Focus on: runtimes (Wasmtime, Wasmer), cloud/edge use cases, WASI progress"
    },
    {
        "goal": "Research RISC-V server chip adoption",
        "context": "Focus on: server chips shipping, cloud providers adopting, software ecosystem"
    },
    {
        "goal": "Research practical quantum computing applications",
        "context": "Focus on: error correction breakthroughs, real-world use cases, key companies"
    }
])
```

Todos os três são executados de forma concorrente. Cada subagente pesquisa na web de forma independente e retorna um resumo. O agente pai então sintetiza tudo em um resumo coerente.

---

## Padrão: Revisão de Código {#pattern-code-review}

Delegue uma revisão de segurança a um subagente de contexto novo que aborda o código sem preconceitos:

```
Review the authentication module at src/auth/ for security issues.
Check for SQL injection, JWT validation problems, password handling,
and session management. Fix anything you find and run the tests.
```

A chave é o campo `context` — ele deve incluir tudo o que o subagente precisa:

```python
delegate_task(
    goal="Review src/auth/ for security issues and fix any found",
    context="""Project at /home/user/webapp. Python 3.11, Flask, PyJWT, bcrypt.
    Auth files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py
    Test command: pytest tests/auth/ -v
    Focus on: SQL injection, JWT validation, password hashing, session management.
    Fix issues found and verify tests pass."""
)
```

:::warning O Problema do Contexto
Subagentes não sabem **absolutamente nada** sobre sua conversa. Eles começam completamente do zero. Se você delegar "corrija o bug que estávamos discutindo", o subagente não tem ideia de qual bug você quer dizer. Sempre passe caminhos de arquivo, mensagens de erro, estrutura do projeto e restrições explicitamente.
:::

---

## Padrão: Comparar Alternativas {#pattern-compare-alternatives}

Avalie várias abordagens para o mesmo problema em paralelo, depois escolha a melhor:

```
I need to add full-text search to our Django app. Evaluate three approaches
in parallel:
1. PostgreSQL tsvector (built-in)
2. Elasticsearch via django-elasticsearch-dsl
3. Meilisearch via meilisearch-python

For each: setup complexity, query capabilities, resource requirements,
and maintenance overhead. Compare them and recommend one.
```

Cada subagente pesquisa uma opção de forma independente. Como eles são isolados, não há contaminação cruzada — cada avaliação se sustenta por seus próprios méritos. O agente pai recebe os três resumos e faz a comparação.

---

## Padrão: Refatoração de Múltiplos Arquivos {#pattern-multi-file-refactoring}

Divida uma grande tarefa de refatoração entre subagentes paralelos, cada um lidando com uma parte diferente da base de código:

```python
delegate_task(tasks=[
    {
        "goal": "Refactor all API endpoint handlers to use the new response format",
        "context": """Project at /home/user/api-server.
        Files: src/handlers/users.py, src/handlers/auth.py, src/handlers/billing.py
        Old format: return {"data": result, "status": "ok"}
        New format: return APIResponse(data=result, status=200).to_dict()
        Import: from src.responses import APIResponse
        Run tests after: pytest tests/handlers/ -v"""
    },
    {
        "goal": "Update all client SDK methods to handle the new response format",
        "context": """Project at /home/user/api-server.
        Files: sdk/python/client.py, sdk/python/models.py
        Old parsing: result = response.json()["data"]
        New parsing: result = response.json()["data"] (same key, but add status code checking)
        Also update sdk/python/tests/test_client.py"""
    },
    {
        "goal": "Update API documentation to reflect the new response format",
        "context": """Project at /home/user/api-server.
        Docs at: docs/api/. Format: Markdown with code examples.
        Update all response examples from old format to new format.
        Add a 'Response Format' section to docs/api/overview.md explaining the schema."""
    }
])
```

:::tip
Cada subagente recebe sua própria sessão de terminal. Eles podem trabalhar no mesmo diretório do projeto sem pisar um no outro — desde que estejam editando arquivos diferentes. Se dois subagentes puderem tocar no mesmo arquivo, cuide desse arquivo você mesmo depois que o trabalho paralelo terminar.
:::

---

## Padrão: Coletar e Depois Analisar {#pattern-gather-then-analyze}

Use `execute_code` para coleta mecânica de dados, depois delegue a análise que exige muito raciocínio:

```python
# Step 1: Mechanical gathering (execute_code is better here — no reasoning needed)
execute_code("""
from hermes_tools import web_search, web_extract

results = []
for query in ["AI funding Q1 2026", "AI startup acquisitions 2026", "AI IPOs 2026"]:
    r = web_search(query, limit=5)
    for item in r["data"]["web"]:
        results.append({"title": item["title"], "url": item["url"], "desc": item["description"]})

# Extract full content from top 5 most relevant
urls = [r["url"] for r in results[:5]]
content = web_extract(urls)

# Save for the analysis step
import json
with open("/tmp/ai-funding-data.json", "w") as f:
    json.dump({"search_results": results, "extracted": content["results"]}, f)
print(f"Collected {len(results)} results, extracted {len(content['results'])} pages")
""")

# Step 2: Reasoning-heavy analysis (delegation is better here)
delegate_task(
    goal="Analyze AI funding data and write a market report",
    context="""Raw data at /tmp/ai-funding-data.json contains search results and
    extracted web pages about AI funding, acquisitions, and IPOs in Q1 2026.
    Write a structured market report: key deals, trends, notable players,
    and outlook. Focus on deals over $100M."""
)
```

Esse é frequentemente o padrão mais eficiente: `execute_code` lida com as 10 ou mais chamadas de ferramentas sequenciais de forma barata, e então um subagente realiza a única tarefa de raciocínio cara com um contexto limpo.

---

## Acesso Herdado a Ferramentas {#inherited-tool-access}

Subagentes herdam os toolsets habilitados do agente pai. `delegate_task` não aceita um parâmetro `toolsets` voltado ao modelo, então o trabalho delegado não pode conceder a si mesmo capacidades que o pai não tem. Configure as ferramentas do agente pai antes de iniciar a conversa quando uma tarefa delegada precisar de acesso à web, terminal, arquivos ou outros. O Hermes ainda remove ferramentas bloqueadas para filhos, como `clarify`, `memory` e `execute_code`.

---

## Restrições {#constraints}

- **3 tarefas paralelas por padrão**: os lotes têm como padrão 3 subagentes concorrentes (configurável via `delegation.max_concurrent_children` em config.yaml, sem teto rígido, apenas um piso de 1)
- **Delegação aninhada é opt-in**: subagentes leaf (padrão) não podem chamar `delegate_task`, `clarify`, `memory` ou `execute_code`. Subagentes orquestradores (`role="orchestrator"`) mantêm `delegate_task` para delegação adicional, mas apenas quando `delegation.max_spawn_depth` é elevado acima do padrão de 1 (piso 1, sem teto); os outros três permanecem bloqueados. Desabilite globalmente via `delegation.orchestrator_enabled: false`.

### Ajustando Concorrência e Profundidade {#tuning-concurrency-and-depth}

| Config | Padrão | Intervalo | Efeito |
|--------|---------|-------|--------|
| `max_concurrent_children` | 3 | >=1 | Tamanho do lote paralelo por chamada de `delegate_task` |
| `max_spawn_depth` | 1 | >=1 | Quantos níveis de delegação podem gerar mais delegações |

Exemplo: executando 30 workers paralelos com subagentes aninhados:

```yaml
delegation:
  max_concurrent_children: 30
  max_spawn_depth: 2
```

- **Terminais separados** — cada subagente recebe sua própria sessão de terminal com diretório de trabalho e estado separados
- **Sem histórico de conversa** — subagentes veem apenas o `goal` e o `context` que o agente pai passa ao chamar `delegate_task`
- **50 iterações por padrão** — defina `max_iterations` mais baixo para tarefas simples para economizar custo
- **Não duradoura** — a delegação de nível superior é executada em segundo plano e publica seu resultado depois, mas continua vinculada à sessão proprietária e ao processo do Hermes. O fechamento da sessão, `/stop`, `/new` ou um reinício de processo podem cancelar ou deixar órfão o trabalho em andamento. Use `cronjob` ou `terminal(background=True, notify_on_complete=True)` para trabalho que precisa sobreviver a esses limites.

---

## Dicas {#tips}

**Seja específico nos objetivos.** "Corrija o bug" é muito vago. "Corrija o TypeError em api/handlers.py linha 47, onde process_request() recebe None de parse_body()" dá ao subagente o suficiente para trabalhar.

**Inclua caminhos de arquivo.** Subagentes não conhecem a estrutura do seu projeto. Sempre inclua caminhos absolutos para os arquivos relevantes, a raiz do projeto e o comando de teste.

**Use delegação para isolamento de contexto.** Às vezes você quer uma perspectiva nova. Delegar te força a articular o problema claramente, e o subagente o aborda sem as suposições que se acumularam na sua conversa.

**Verifique os resultados.** Resumos de subagentes são apenas isso — resumos. Se um subagente diz "corrigi o bug e os testes passam", verifique executando os testes você mesmo ou lendo o diff.

---

*Para a referência completa de delegação — todos os parâmetros, integração com ACP e configuração avançada — veja [Delegação de Subagentes](/user-guide/features/delegation).*
