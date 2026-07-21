---
sidebar_position: 8
title: "Execução de código"
description: "Execução programática Python com acesso RPC a ferramentas — colapse fluxos multi-etapa em um único turno"
---

# Execução de código (chamada programática de ferramentas)

A ferramenta `execute_code` permite que o agente escreva scripts Python que chamam ferramentas do Hermes programaticamente, colapsando fluxos multi-etapa em um único turno de LLM. O script roda em um processo filho no host do agente, comunicando-se com o Hermes por RPC via Unix domain socket.

## Como funciona {#how-it-works}

1. O agente escreve um script Python usando `from hermes_tools import ...`
2. O Hermes gera um módulo stub `hermes_tools.py` com funções RPC
3. O Hermes abre um Unix domain socket e inicia uma thread listener RPC
4. O script roda em um processo filho — chamadas de ferramenta viajam pelo socket de volta ao Hermes
5. Apenas a saída `print()` do script volta ao LLM; resultados intermediários de ferramentas nunca entram na janela de contexto

```python
# The agent can write scripts like:
from hermes_tools import web_search, web_extract

results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # ... filter and process ...
print(summary)
```

**Ferramentas disponíveis dentro de scripts:** `web_search`, `web_extract`, `read_file`, `write_file`, `search_files`, `patch`, `terminal` (apenas foreground).

## Quando o agente usa isso {#when-the-agent-uses-this}

O agente usa `execute_code` quando há:

- **3+ chamadas de ferramenta** com lógica de processamento entre elas
- Filtragem em massa de dados ou ramificação condicional
- Loops sobre resultados

O benefício-chave: resultados intermediários de ferramentas nunca entram na janela de contexto — apenas a saída final de `print()` volta, reduzindo drasticamente o uso de tokens.

## Exemplos práticos {#practical-examples}

### Pipeline de processamento de dados {#data-processing-pipeline}

```python
from hermes_tools import search_files, read_file
import json

# Find all config files and extract database settings
matches = search_files("database", path=".", file_glob="*.yaml", limit=20)
configs = []
for match in matches.get("matches", []):
    content = read_file(match["path"])
    configs.append({"file": match["path"], "preview": content["content"][:200]})

print(json.dumps(configs, indent=2))
```

### Pesquisa web multi-etapa {#multi-step-web-research}

```python
from hermes_tools import web_search, web_extract
import json

# Search, extract, and summarize in one turn
results = web_search("Rust async runtime comparison 2025", limit=5)
summaries = []
for r in results["data"]["web"]:
    page = web_extract([r["url"]])
    for p in page.get("results", []):
        if p.get("content"):
            summaries.append({
                "title": r["title"],
                "url": r["url"],
                "excerpt": p["content"][:500]
            })

print(json.dumps(summaries, indent=2))
```

### Refatoração em massa de arquivos {#bulk-file-refactoring}

```python
from hermes_tools import search_files, read_file, patch

# Find all Python files using deprecated API and fix them
matches = search_files("old_api_call", path="src/", file_glob="*.py")
fixed = 0
for match in matches.get("matches", []):
    result = patch(
        path=match["path"],
        old_string="old_api_call(",
        new_string="new_api_call(",
        replace_all=True
    )
    if "error" not in str(result):
        fixed += 1

print(f"Fixed {fixed} files out of {len(matches.get('matches', []))} matches")
```

### Pipeline de build e teste {#build-and-test-pipeline}

```python
from hermes_tools import terminal, read_file
import json

# Run tests, parse results, and report
result = terminal("cd /project && python -m pytest --tb=short -q 2>&1", timeout=120)
output = result.get("output", "")

# Parse test output
passed = output.count(" passed")
failed = output.count(" failed")
errors = output.count(" error")

report = {
    "passed": passed,
    "failed": failed,
    "errors": errors,
    "exit_code": result.get("exit_code", -1),
    "summary": output[-500:] if len(output) > 500 else output
}

print(json.dumps(report, indent=2))
```

## Modo de execução {#execution-mode}

`execute_code` tem dois modos de execução controlados por `code_execution.mode` em `~/.hermes/config.yaml`:

| Modo | Diretório de trabalho | Interpretador Python |
|------|-------------------|--------------------|
| **`project`** (padrão) | O diretório de trabalho da sessão (igual a `terminal()`) | Python do `VIRTUAL_ENV` / `CONDA_PREFIX` ativo, fallback para o python do Hermes |
| `strict` | Um diretório de staging temporário isolado da árvore do projeto | `sys.executable` (python do Hermes) |

**Quando deixar em `project`:** você quer que `import pandas`, `from my_project import foo` ou caminhos relativos como `open(".env")` funcionem igual em `terminal()`. Isso é quase sempre o que você quer.

**Quando mudar para `strict`:** você precisa de máxima reprodutibilidade — quer o mesmo interpretador toda sessão independentemente de qual venv o usuário ativou, e quer scripts em quarentena da árvore do projeto (sem risco de ler arquivos do projeto acidentalmente por caminho relativo).

```yaml
# ~/.hermes/config.yaml
code_execution:
  mode: project   # ou "strict"
```

Comportamento de fallback em modo `project`: se `VIRTUAL_ENV` / `CONDA_PREFIX` estiver unset, quebrado ou apontar para Python anterior a 3.8, o resolvedor faz fallback limpo para `sys.executable` — nunca deixa o agente sem interpretador funcional.

Invariantes críticos de segurança são idênticos em ambos os modos:

- scrubbing de ambiente (chaves de API, tokens, credenciais removidos)
- whitelist de ferramentas (scripts não podem chamar `execute_code` recursivamente, `delegate_task` ou ferramentas MCP)
- limites de recursos (timeout, cap de stdout, cap de chamadas de ferramenta)

Trocar o modo altera onde os scripts rodam e qual interpretador os executa, não quais credenciais eles veem ou quais ferramentas podem chamar.

## Limites de recursos {#resource-limits}

| Recurso | Limite | Notas |
|----------|-------|-------|
| **Timeout** | 5 minutos (300s) | Script é morto com SIGTERM, depois SIGKILL após 5s de grace |
| **Stdout** | 50 KB | Saída truncada com aviso `[output truncated at 50KB]` |
| **Stderr** | 10 KB | Incluído na saída em exit não-zero para debug |
| **Chamadas de ferramenta** | 50 por execução | Erro retornado quando o limite é atingido |

Todos os limites são configuráveis via `config.yaml`:

```yaml
# In ~/.hermes/config.yaml
code_execution:
  mode: project      # project (padrão) | strict
  timeout: 300       # Máx. segundos por script (padrão: 300)
  max_tool_calls: 50 # Máx. chamadas de ferramenta por execução (padrão: 50)
```

## Como chamadas de ferramenta funcionam dentro de scripts {#how-tool-calls-work-inside-scripts}

Quando seu script chama uma função como `web_search("query")`:

1. A chamada é serializada para JSON e enviada por Unix domain socket ao processo pai
2. O pai despacha pelo handler padrão `handle_function_call`
3. O resultado é enviado de volta pelo socket
4. A função retorna o resultado parseado

Isso significa que chamadas de ferramenta dentro de scripts se comportam igual às chamadas normais — mesmos rate limits, mesmo tratamento de erro, mesmas capacidades. A única restrição é que `terminal()` é apenas foreground (sem parâmetros `background` ou `pty`).

## Tratamento de erros {#error-handling}

Quando um script falha, o agente recebe informação estruturada de erro:

- **Exit code não-zero**: stderr é incluído na saída para o agente ver o traceback completo
- **Timeout**: Script é morto e o agente vê `"Script timed out after 300s and was killed."`
- **Interrupção**: Se o usuário enviar nova mensagem durante a execução, o script é terminado e o agente vê `[execution interrupted — user sent a new message]`
- **Limite de chamadas de ferramenta**: Quando o limite de 50 é atingido, chamadas subsequentes retornam mensagem de erro

A resposta sempre inclui `status` (success/error/timeout/interrupted), `output`, `tool_calls_made` e `duration_seconds`.

## Segurança {#security}

:::danger Modelo de segurança
O processo filho roda com **ambiente mínimo**. Chaves de API, tokens e credenciais são removidos por padrão. O script acessa ferramentas exclusivamente via canal RPC — não pode ler segredos de variáveis de ambiente a menos que explicitamente permitido.
:::

Variáveis de ambiente contendo `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD` ou `AUTH` em seus nomes são excluídas. Apenas variáveis seguras do sistema (`PATH`, `HOME`, `LANG`, `SHELL`, `PYTHONPATH`, `VIRTUAL_ENV`, etc.) passam.

### Passthrough de variáveis de ambiente de skill {#skill-environment-variable-passthrough}

Quando uma skill declara `required_environment_variables` em seu frontmatter, essas variáveis são **automaticamente repassadas** tanto a `execute_code` quanto a processos filhos de `terminal` após a skill ser carregada. Isso permite que skills usem suas chaves de API declaradas sem enfraquecer a postura de segurança para código arbitrário.

Para casos fora de skill, você pode allowlist explicitamente variáveis em `config.yaml`:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

Veja o [guia de Security](/user-guide/security#environment-variable-passthrough) para detalhes completos.

### Variáveis `HERMES_*` no filho {#hermes-variables-in-the-child}

O processo filho recebe apenas um conjunto pequeno e fixo de variáveis operacionais `HERMES_*`
por nome exato:

- `HERMES_HOME`
- `HERMES_PROFILE`
- `HERMES_CONFIG`
- `HERMES_ENV`

(mais `HERMES_RPC_DIR` / `HERMES_RPC_SOCKET` / `TZ` / `HOME`, que o Hermes
injeta explicitamente para o canal RPC funcionar).

:::note Mudança de comportamento
Versões anteriores repassavam **qualquer** variável cujo nome começasse com `HERMES_`
ao filho. Esse prefixo amplo foi removido para hardening de segurança: podia vazar
configuração nomeada `HERMES_*` que não corresponde a substring de segredo
(por exemplo `HERMES_BASE_URL`, `HERMES_KANBAN_DB` ou um endpoint `HERMES_*_WEBHOOK`)
para código sandboxed arbitrário.

Se um script `execute_code` — ou um módulo de repo/plugin que ele importa na import time
— dependia de uma variável `HERMES_*` fora dos quatro nomes operacionais acima, agora
encontrará essa variável **unset** no filho. A remoção é intencional,
não um bug.
:::

**Contorno — opt-in explícito da variável.** Ambas as rotas repassam a
variável por `execute_code` *e* filhos de `terminal`, e nenhuma enfraquece
a garantia de remoção de segredos (credenciais de provider gerenciadas pelo Hermes nunca
podem ser re-permitidas assim):

1. **Por máquina, em `config.yaml`** — adicione o nome exato da variável ao
   allowlist de passthrough:

   ```yaml
   terminal:
     env_passthrough:
       - HERMES_KANBAN_DB
       - HERMES_BASE_URL
   ```

2. **Por skill, no frontmatter da skill** — declare para ser registrada
   automaticamente sempre que essa skill for carregada:

   ```yaml
   required_environment_variables:
     - HERMES_KANBAN_DB
   ```

**Diagnosticando.** Quando o filho remove uma ou mais variáveis `HERMES_*` não allowlisted,
o Hermes emite um log `debug` de uma linha nomeando-as e apontando para o
escape hatch `env_passthrough`. Rode com debug logging (`hermes logs --level
DEBUG`, ou confira `~/.hermes/logs/agent.log`) e procure
`execute_code: dropped N non-allowlisted HERMES_* var(s)` se um script se comportar
como se uma variável `HERMES_*` estivesse faltando.

O Hermes sempre escreve o script e o stub RPC `hermes_tools.py` auto-gerado em um diretório de staging temporário limpo após a execução. Em modo `strict` o script também *roda* lá; em modo `project` roda no diretório de trabalho da sessão (o diretório de staging permanece no `PYTHONPATH` para imports ainda resolverem). O processo filho roda em seu próprio process group para ser morto limpo em timeout ou interrupção.

## execute_code vs terminal {#execute_code-vs-terminal}

| Caso de uso | execute_code | terminal |
|----------|-------------|----------|
| Fluxos multi-etapa com chamadas de ferramenta entre elas | ✅ | ❌ |
| Comando shell simples | ❌ | ✅ |
| Filtrar/processar saídas grandes de ferramentas | ✅ | ❌ |
| Rodar build ou suíte de testes | ❌ | ✅ |
| Loop sobre resultados de busca | ✅ | ❌ |
| Processos interativos/background | ❌ | ✅ |
| Precisa de chaves de API no ambiente | ⚠️ Apenas via [passthrough](/user-guide/security#environment-variable-passthrough) | ✅ (a maioria passa) |

**Regra prática:** Use `execute_code` quando precisar chamar ferramentas Hermes programaticamente com lógica entre chamadas. Use `terminal` para rodar comandos shell, builds e processos.

## Suporte a plataforma {#platform-support}

A execução de código requer Unix domain sockets e está disponível apenas em **Linux e macOS**. É desabilitada automaticamente no Windows — o agente faz fallback para chamadas sequenciais normais de ferramentas.
